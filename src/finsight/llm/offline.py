"""Deterministic, key-less LLM stand-in.

* RAG: extractive answering - picks the context sentences that best cover the question and
  cites them, or abstains (insufficient_context) when coverage is too low.
* Agent: keyword router that emits real tool calls (incl. the approval-gated ingest tool) and
  then summarises tool outputs with citation refs.

It lets the whole system (API, agent graph, guardrails, evals, CI) run end-to-end with no API
key, and gives evaluation runs a reproducible baseline to compare real models against.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter

from finsight.ingestion.chunker import count_tokens
from finsight.llm.base import LLMProvider, LLMResponse, ToolCall
from finsight.models import Usage
from finsight.retrieval.embeddings import tokenize

SOURCE_RE = re.compile(r'<source id="(?P<id>[^"]+)"[^>]*>\n?(?P<body>.*?)\n?</source>', re.S)
QUESTION_RE = re.compile(r"<question>\s*(.*?)\s*</question>", re.S)
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"“])|\n")

COMPANY_TICKERS = {
    "apple": "AAPL", "microsoft": "MSFT", "nvidia": "NVDA", "tesla": "TSLA", "amazon": "AMZN",
    "alphabet": "GOOGL", "google": "GOOGL", "meta": "META", "netflix": "NFLX", "intel": "INTC", "amd": "AMD",
    "bitcoin": "BTCUSDT", "ethereum": "ETHUSDT",
}
NOT_TICKERS = {"I", "A", "RSI", "EPS", "MD", "SEC", "US", "USD", "CEO", "CFO", "AI", "FY", "Q", "IT", "R", "D",
               "AND", "OR", "THE", "OF", "VS", "K", "TA", "S", "XBRL", "API", "GAAP", "ESG", "IPO", "EBITDA"}
METRIC_WORDS = {
    "revenue": "Revenues", "sales": "Revenues", "net income": "NetIncomeLoss", "profit": "NetIncomeLoss",
    "eps": "EarningsPerShareDiluted", "earnings per share": "EarningsPerShareDiluted",
    "operating income": "OperatingIncomeLoss", "gross profit": "GrossProfit", "r&d": "ResearchAndDevelopmentExpense",
    "research and development": "ResearchAndDevelopmentExpense", "cash": "CashAndCashEquivalentsAtCarryingValue",
    "assets": "Assets", "liabilities": "Liabilities", "operating cash flow": "NetCashProvidedByUsedInOperatingActivities",
}
MARKET_RE = re.compile(r"\b(rsi|technical|price action|support|resistance|divergence|momentum|chart|trend|stock price)\b", re.I)
INGEST_RE = re.compile(r"\b(ingest|index|load|add)\b.*\b(10-?k|10-?q|filings?|annual report)\b", re.I)


def _tickers(text: str) -> list[str]:
    found: list[str] = []
    low = text.lower()
    for name, t in COMPANY_TICKERS.items():
        if re.search(rf"\b{name}\b", low) and t not in found:
            found.append(t)
    for t in re.findall(r"\b(?:[A-Z0-9]{2,12}USDT|[A-Z]{1,5})\b", text):
        if t not in NOT_TICKERS and t not in found:
            found.append(t)
    return found


QUESTION_NOISE = {"explain", "describe", "tell", "show", "give", "list", "summarize", "summarise", "discuss", "company",
                  "compan", "please", "me"}


def extractive_answer(question: str, sources: list[tuple[str, str]], max_sents: int = 3) -> dict:
    tickers = {t.lower() for t in _tickers(question)}
    q = {t for t in tokenize(question) if t not in QUESTION_NOISE and t not in tickers}
    if not q or not sources:
        return {"answer": "", "citations": [], "confidence": "low", "insufficient_context": True}
    sents: list[tuple[str, str, int]] = []
    for rank, (sid, body) in enumerate(sources):
        for s in SENT_SPLIT.split(body):
            s = s.strip()
            if 25 <= len(s) <= 700 and not s.startswith("[") and s.count("|") < 12:
                sents.append((sid, s, rank))
    df = Counter(t for _, s, _ in sents for t in set(tokenize(s)))
    n = max(len(sents), 1)

    def score(s: str, rank: int) -> tuple[float, float]:
        toks = set(tokenize(s))
        hit = q & toks
        w = sum(math.log(1 + n / (1 + df[t])) for t in hit)
        return w * (1 - 0.04 * rank), len(hit) / len(q)

    scored = sorted(((score(s, r), sid, s) for sid, s, r in sents), key=lambda x: x[0][0], reverse=True)
    picked: list[tuple[str, str, float]] = []
    seen: list[str] = []
    for (w, cov), sid, s in scored:
        norm = " ".join(tokenize(s))
        if cov < 0.34 or w <= 0 or any(norm in x or x in norm for x in seen):  # overlap duplicates
            continue
        picked.append((sid, s, cov))
        seen.append(norm)
        if len(picked) >= max_sents:
            break
    if not picked:
        return {"answer": "", "citations": [], "confidence": "low", "insufficient_context": True}
    best = max(c for _, _, c in picked)
    return {
        "answer": " ".join(f"{s} [{sid}]" for sid, s, _ in picked),
        "citations": [{"chunk_id": sid, "quote": " ".join(s.split()[:30])} for sid, s, _ in picked],
        "confidence": "high" if best >= 0.75 else "medium" if best >= 0.5 else "low",
        "insufficient_context": False,
    }


class OfflineProvider(LLMProvider):
    name = "offline"
    model = "offline-extractive"

    def _chat(self, messages, tools, schema, temperature, max_tokens) -> LLMResponse:
        prompt_text = "\n".join(str(m.get("content") or "") for m in messages)
        usage_in = count_tokens(prompt_text)
        if schema is not None:
            user = next(m["content"] for m in reversed(messages) if m["role"] == "user")
            qm = QUESTION_RE.search(user)
            question = qm.group(1) if qm else user
            sources = [(m.group("id"), m.group("body")) for m in SOURCE_RE.finditer(user)]
            out = json.dumps(extractive_answer(question, sources))
            return LLMResponse(text=out, model=self.model, usage=Usage(input_tokens=usage_in, output_tokens=count_tokens(out)))
        if tools:
            return self._agent_step(messages, {t.name for t in tools}, usage_in)
        return LLMResponse(text="(offline mode) no answer", model=self.model, usage=Usage(input_tokens=usage_in))

    # ---- agent -------------------------------------------------------------------
    def _agent_step(self, messages: list[dict], tool_names: set[str], usage_in: int) -> LLMResponse:
        last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
        question = messages[last_user]["content"]
        tool_msgs = [m for m in messages[last_user + 1 :] if m["role"] == "tool"]
        if not tool_msgs:
            calls = self._route(question, tool_names)
            return LLMResponse(tool_calls=calls, model=self.model, usage=Usage(input_tokens=usage_in, output_tokens=40 * len(calls)))
        text = self._summarise(question, tool_msgs)
        return LLMResponse(text=text, model=self.model, usage=Usage(input_tokens=usage_in, output_tokens=count_tokens(text)))

    def _route(self, q: str, tool_names: set[str]) -> list[ToolCall]:
        tickers = _tickers(q) or [None]
        calls: list[ToolCall] = []
        cid = iter(range(1, 100))
        if INGEST_RE.search(q) and "ingest_company_filings" in tool_names and tickers[0]:
            form = "10-Q" if re.search(r"10-?q", q, re.I) else "10-K"
            return [ToolCall(f"call_{next(cid)}", "ingest_company_filings", {"ticker": tickers[0], "form": form, "limit": 1})]
        if MARKET_RE.search(q) and "get_market_snapshot" in tool_names and tickers[0]:
            for t in tickers[:2]:
                calls.append(ToolCall(f"call_{next(cid)}", "get_market_snapshot", {"symbol": t, "interval": "1d"}))
        low = q.lower()
        metrics = sorted({v for k, v in METRIC_WORDS.items() if k in low})
        if metrics and "get_financial_metrics" in tool_names and tickers[0]:
            for t in tickers[:3]:
                if not t.endswith("USDT"):
                    calls.append(ToolCall(f"call_{next(cid)}", "get_financial_metrics", {"ticker": t, "metrics": metrics, "years": 3}))
        wants_text = not calls or re.search(r"\b(why|risk|explain|describe|strategy|factor|discuss|guidance|driver)", low)
        if wants_text and "search_filings" in tool_names:
            args = {"query": q}
            if tickers[0] and not tickers[0].endswith("USDT") and len(tickers) == 1:
                args["ticker"] = tickers[0]
            calls.append(ToolCall(f"call_{next(cid)}", "search_filings", args))
        return calls

    def _summarise(self, question: str, tool_msgs: list[dict]) -> str:
        parts: list[str] = []
        for m in tool_msgs:
            try:
                data = json.loads(m["content"])
            except json.JSONDecodeError:
                parts.append(m["content"][:400])
                continue
            if isinstance(data, dict) and data.get("error"):
                parts.append(f"Tool `{m.get('name')}` failed: {data['error']}")
            elif m.get("name") == "search_filings":
                res = data.get("results", [])
                ext = extractive_answer(question, [(r["ref"], r["text"]) for r in res], max_sents=2)
                parts.append(ext["answer"] or "The indexed filings do not contain a clear answer to this question.")
            elif m.get("name") == "get_financial_metrics":
                for metric, rows in data.get("metrics", {}).items():
                    vals = ", ".join(f"FY{r['fy']}: {r['display']} [{r['ref']}]" for r in rows)
                    parts.append(f"{data['ticker']} {metric} — {vals}.")
            elif m.get("name") == "get_market_snapshot":
                d = data
                divs = ", ".join(x["type"] for x in d.get("divergences", [])) or "none"
                zones = "; ".join(f"{z['low']:.2f}–{z['high']:.2f}" for z in d.get("sr_zones", [])[:3]) or "none"
                parts.append(
                    f"{d['symbol']} ({d['interval']}): last close {d['last_close']:.2f}, RSI(14) {d['rsi']:.1f}, "
                    f"trend {d['trend']} (efficiency {d['efficiency_ratio']:.2f}), RSI divergences: {divs}, "
                    f"high-density S/R zones: {zones} [{d['ref']}]."
                )
            elif m.get("name") == "ingest_company_filings":
                parts.append(f"Ingested {data.get('chunks', 0)} chunks from {data.get('filings', [])}.")
            else:
                parts.append(json.dumps(data)[:400])
        return "\n\n".join(parts)

