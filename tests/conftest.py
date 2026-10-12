"""Shared fixtures: fully offline container (in-memory Qdrant, hashing embeddings, offline LLM,
fake SEC + market data) so the suite is fast, deterministic and runs in CI without secrets."""

from __future__ import annotations

import math
import os

import pytest

# Environment variables beat a developer's local .env -> the suite is hermetic.
os.environ.update({"EMBEDDING_PROVIDER": "hash", "RERANKER": "none", "LLM_PROVIDER": "offline", "API_KEYS": "",
                   "QDRANT_URL": "", "SEED_TICKERS": "", "OTEL_EXPORTER_OTLP_ENDPOINT": ""})

from finsight.config import Settings
from finsight.container import build_container
from finsight.eval.runner import DATASETS, FIXTURE_REF
from finsight.models import FilingRef

FIXTURE_HTML = (DATASETS / "nwrb_10k.html").read_text(encoding="utf-8")


def _fy(val, fy, start=True, form="10-K", fp="FY", filed=None):
    item = {"end": f"{fy}-12-31", "val": val, "fy": fy + 1, "fp": fp, "form": form, "filed": filed or f"{fy + 1}-02-20",
            "accn": f"0009999999-{fy % 100 + 1:02d}-000001"}
    if start:
        item["start"] = f"{fy}-01-01"
    return item


FACTS = {
    "entityName": "Northwind Robotics Inc.",
    "facts": {"us-gaap": {
        "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
            _fy(1_756_000_000, 2023), _fy(2_049_000_000, 2024), _fy(2_418_000_000, 2025),
            {**_fy(600_000_000, 2025), "start": "2025-10-01", "fp": "FY"},  # quarter inside the 10-K -> ignored
            _fy(500_000_000, 2025, fp="Q3", form="10-Q"),
        ]}},
        "NetIncomeLoss": {"units": {"USD": [_fy(121_000_000, 2023), _fy(198_000_000, 2024), _fy(286_000_000, 2025)]}},
        "EarningsPerShareDiluted": {"units": {"USD/shares": [_fy(1.01, 2023), _fy(1.62, 2024), _fy(2.31, 2025)]}},
    }},
}


class FakeSEC:
    def __init__(self):
        self.downloads = 0

    def lookup(self, ticker):
        if ticker.upper() not in {"NWRB", "ACME"}:
            raise KeyError(ticker)
        return 9999999, "Northwind Robotics Inc."

    def list_filings(self, ticker, forms=("10-K",), limit=1):
        self.lookup(ticker)
        if ticker.upper() == "NWRB":
            return [FIXTURE_REF]
        return [FilingRef(ticker="ACME", company="Acme Corp (synthetic)", cik=1, form="10-K", accession="0000000001-26-000001",
                          filing_date="2026-02-01", report_date="2025-12-31", primary_document="acme.htm",
                          url="https://example.com/acme.htm")]

    def fetch_document(self, ref):
        self.downloads += 1
        if ref.ticker == "ACME":
            return FIXTURE_HTML.replace("Northwind Robotics", "Acme Corp").replace("Kestrel Semiconductor", "Falcon Optics")
        return FIXTURE_HTML

    def company_facts(self, ticker):
        self.lookup(ticker)
        return FACTS


def synthetic_closes(n=250):
    # uptrend with oscillation -> deterministic RSI / pivots / zones
    return [100 + 0.15 * i + 4 * math.sin(i / 4) for i in range(n)]


@pytest.fixture
def settings(tmp_path):
    return Settings(env="test", data_dir=tmp_path, llm_provider="offline", embedding_provider="hash", reranker="none",
                    chunk_tokens=120, chunk_overlap_tokens=20, answer_cache_ttl_s=600, rate_limit_per_minute=1000)


@pytest.fixture
def fake_sec():
    return FakeSEC()


@pytest.fixture
def container(settings, fake_sec):
    return build_container(settings, sec=fake_sec, fetch_closes=lambda symbol, interval="1d": synthetic_closes())


@pytest.fixture
def indexed(container):
    container.ingestion.ingest_html(FIXTURE_HTML, FIXTURE_REF)
    return container
