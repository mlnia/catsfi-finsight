"""Agent tools. Arguments are validated with Pydantic (the JSON schema sent to the LLM is generated
from the same model), outputs are compact JSON with citable `ref` ids, and failures are returned
as `{"error": ...}` so the model can self-correct instead of crashing the run."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from finsight.ingestion.pipeline import IngestionService
from finsight.ingestion.sec_client import SECClient
from finsight.llm.base import ToolSpec
from finsight.market import data as market_data
from finsight.market.fundamentals import SUPPORTED_METRICS, annual_series, fmt_value
from finsight.market.indicators import snapshot
from finsight.models import SearchFilters
from finsight.observability.metrics import TOOL_CALLS
from finsight.observability.tracing import span
from finsight.rag.service import RAGService


class SearchFilingsArgs(BaseModel):
    query: str = Field(description="Natural-language search query.")
    ticker: str | None = Field(None, description="Restrict to one ticker, e.g. AAPL.")
    form: Literal["10-K", "10-Q"] | None = None
    fiscal_year: int | None = None
    top_k: int = Field(5, ge=1, le=10)


class FinancialMetricsArgs(BaseModel):
    ticker: str
    metrics: list[str] = Field(description=f"Any of: {', '.join(SUPPORTED_METRICS)}")
    years: int = Field(3, ge=1, le=10)


class MarketSnapshotArgs(BaseModel):
    symbol: str = Field(description="Stock ticker (AAPL) or USDT-M crypto pair (BTCUSDT).")
    interval: Literal["1h", "1d", "1wk"] = "1d"


class IngestArgs(BaseModel):
    ticker: str
    form: Literal["10-K", "10-Q"] = "10-K"
    limit: int = Field(1, ge=1, le=4)


@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[[BaseModel], dict]
    requires_approval: bool = False

    @property
    def spec(self) -> ToolSpec:
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        return ToolSpec(self.name, self.description, schema)

    def __call__(self, raw_args: dict) -> dict:
        with span(f"tool.{self.name}", {"tool.args": json.dumps(raw_args)[:500]}) as sp:
            t0 = time.perf_counter()
            try:
                out = self.fn(self.args_model.model_validate(raw_args))
                TOOL_CALLS.labels(self.name, "ok").inc()
            except ValidationError as e:
                out = {"error": f"invalid arguments: {e.errors(include_url=False)}"}
                TOOL_CALLS.labels(self.name, "invalid_args").inc()
            except Exception as e:
                out = {"error": f"{type(e).__name__}: {e}"}
                TOOL_CALLS.labels(self.name, "error").inc()
            sp.set_attribute("tool.ms", round((time.perf_counter() - t0) * 1000, 1))
            sp.set_attribute("tool.error", "error" in out)
            return out


def build_tools(rag: RAGService, sec: SECClient | None, ingestion: IngestionService | None,
                fetch_closes: Callable = market_data.fetch_closes) -> dict[str, Tool]:
    def search_filings(a: SearchFilingsArgs) -> dict:
        f = SearchFilters(ticker=a.ticker, form=a.form, fiscal_year=a.fiscal_year)
        chunks, quarantined = rag.retrieve(a.query, f, top_k=a.top_k)
        return {
            "results": [{"ref": f"doc:{c.id[:8]}", "label": c.label(), "url": c.metadata.get("url"), "text": c.text[:1800]}
                        for c in chunks],
            "quarantined": len(quarantined),
        }

    def get_financial_metrics(a: FinancialMetricsArgs) -> dict:
        if sec is None:
            raise RuntimeError("SEC client unavailable")
        facts = sec.company_facts(a.ticker)
        out: dict[str, list] = {}
        for m in a.metrics:
            rows = annual_series(facts, m, a.years)
            out[m] = [{**r, "display": fmt_value(r["value"], r["unit"]), "ref": f"xbrl:{a.ticker.upper()}:{m}:FY{r['fy']}"}
                      for r in rows]
        return {"ticker": a.ticker.upper(), "company": facts.get("entityName"), "source": "SEC XBRL companyfacts",
                "metrics": out}

    def get_market_snapshot(a: MarketSnapshotArgs) -> dict:
        closes = fetch_closes(a.symbol, a.interval)
        snap = snapshot(closes)
        return {"symbol": a.symbol.upper(), "interval": a.interval, "bars": len(closes),
                "ref": f"mkt:{a.symbol.upper()}:{a.interval}", **snap}

    def ingest_company_filings(a: IngestArgs) -> dict:
        if ingestion is None:
            raise RuntimeError("ingestion unavailable")
        rep = ingestion.ingest_ticker(a.ticker, forms=(a.form,), limit=a.limit)
        return {"filings": [f"{f.ticker} {f.form} {f.accession} ({f.status})" for f in rep.filings], "chunks": rep.chunks}

    tools = [
        Tool("search_filings", "Hybrid semantic + keyword search over indexed SEC 10-K/10-Q filings. Returns passages with refs.",
             SearchFilingsArgs, search_filings),
        Tool("get_financial_metrics", "Audited annual figures from SEC XBRL company facts (most reliable source for numbers).",
             FinancialMetricsArgs, get_financial_metrics),
        Tool("get_market_snapshot", "CATsFi technical analysis: RSI(14), RSI divergences, SMA50 trend, efficiency ratio, "
             "high-density support/resistance zones.", MarketSnapshotArgs, get_market_snapshot),
        Tool("ingest_company_filings", "Download and index SEC filings for a company. REQUIRES HUMAN APPROVAL.",
             IngestArgs, ingest_company_filings, requires_approval=True),
    ]
    return {t.name: t for t in tools}


def evidence_from_tool(name: str, out: dict) -> dict[str, str]:
    """Flatten tool output into ref -> text, used to verify the agent's final answer."""
    ev: dict[str, str] = {}
    if name == "search_filings":
        for r in out.get("results", []):
            ev[r["ref"]] = r["text"]
    elif name == "get_financial_metrics":
        for m, rows in out.get("metrics", {}).items():
            for r in rows:
                ev[r["ref"]] = f"{out['ticker']} {m} FY{r['fy']} {r['display']} {r['value']} {r['period_end']}"
    elif name == "get_market_snapshot" and "ref" in out:
        ev[out["ref"]] = json.dumps(out) + " " + " ".join(_rounded(out))
    return ev


def _rounded(obj) -> list[str]:
    """Rounded renderings of every float so '230.12' / '230.1' in an answer match 230.1199951."""
    if isinstance(obj, float):
        return [f"{obj:.2f}", f"{obj:.1f}", f"{obj:.0f}"]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in _rounded(v)]
    if isinstance(obj, list):
        return [s for v in obj for s in _rounded(v)]
    return []
