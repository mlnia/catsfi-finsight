"""SEC EDGAR client: ticker -> CIK, filing index, primary documents and XBRL company facts.

Respects SEC fair-access rules (descriptive User-Agent, <= 10 req/s) and caches raw
responses on disk so re-ingestion and tests are cheap and reproducible.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

import httpx

from finsight.models import FilingRef

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"


class SECClient:
    def __init__(self, user_agent: str, cache_dir: Path, rps: float = 8.0, client: httpx.Client | None = None):
        if "@" not in user_agent:
            raise ValueError("SEC_USER_AGENT must contain a contact e-mail (SEC fair-access policy).")
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._min_interval = 1.0 / max(rps, 0.1)
        self._last = 0.0
        self._lock = threading.Lock()
        self._http = client or httpx.Client(
            headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=30.0,
            follow_redirects=True,
        )

    # ---- low level ------------------------------------------------------------
    def _throttle(self) -> None:
        with self._lock:
            wait = self._min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()

    def _get(self, url: str, *, ttl_s: int | None = None) -> bytes:
        key = hashlib.sha256(url.encode()).hexdigest()[:24]
        path = self.cache_dir / key
        if path.exists() and (ttl_s is None or time.time() - path.stat().st_mtime < ttl_s):
            return path.read_bytes()
        for attempt in range(4):
            self._throttle()
            resp = self._http.get(url)
            if resp.status_code in (429, 500, 502, 503) and attempt < 3:
                time.sleep(2**attempt)
                continue
            resp.raise_for_status()
            path.write_bytes(resp.content)
            return resp.content
        raise RuntimeError(f"unreachable: {url}")

    def _json(self, url: str, ttl_s: int | None = 86400) -> dict:
        return json.loads(self._get(url, ttl_s=ttl_s))

    # ---- public API -----------------------------------------------------------------
    def lookup(self, ticker: str) -> tuple[int, str]:
        data = self._json(TICKERS_URL, ttl_s=7 * 86400)
        t = ticker.upper().replace(".", "-")
        for row in data.values():
            if row["ticker"].upper() == t:
                return int(row["cik_str"]), row["title"]
        raise KeyError(f"Unknown ticker: {ticker}")

    def list_filings(self, ticker: str, forms: tuple[str, ...] = ("10-K",), limit: int = 1) -> list[FilingRef]:
        cik, company = self.lookup(ticker)
        sub = self._json(SUBMISSIONS_URL.format(cik=cik), ttl_s=6 * 3600)
        recent = sub["filings"]["recent"]
        out: list[FilingRef] = []
        for i, form in enumerate(recent["form"]):
            if form not in forms:
                continue
            acc = recent["accessionNumber"][i]
            doc = recent["primaryDocument"][i]
            out.append(
                FilingRef(
                    ticker=ticker.upper(),
                    company=company,
                    cik=cik,
                    form=form,
                    accession=acc,
                    filing_date=recent["filingDate"][i],
                    report_date=recent.get("reportDate", [None] * (i + 1))[i] or None,
                    primary_document=doc,
                    url=ARCHIVE_URL.format(cik=cik, acc=acc.replace("-", ""), doc=doc),
                )
            )
            if len(out) >= limit:
                break
        return out

    def fetch_document(self, ref: FilingRef) -> str:
        raw = self._get(ref.url, ttl_s=None)  # filings are immutable -> cache forever
        return raw.decode("utf-8", errors="replace")

    def company_facts(self, ticker: str) -> dict:
        cik, _ = self.lookup(ticker)
        return self._json(FACTS_URL.format(cik=cik), ttl_s=12 * 3600)
