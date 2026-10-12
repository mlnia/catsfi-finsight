"""Document ingestion pipeline: download -> parse -> section split -> chunk -> embed -> upsert.

Idempotent per filing (accession number): re-running is a no-op unless `force=True`, in which
case the old points are deleted first, so the index never holds stale duplicates.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from finsight.ingestion.chunker import chunk_filing
from finsight.ingestion.parser import PARSER_VERSION, parse_filing
from finsight.ingestion.sec_client import SECClient
from finsight.models import FilingRef
from finsight.observability.metrics import INGESTED_CHUNKS
from finsight.observability.tracing import log_event, span
from finsight.retrieval.store import VectorStore

log = logging.getLogger(__name__)


class FilingResult(BaseModel):
    ticker: str
    form: str
    accession: str
    fiscal_year: int | None
    sections: int = 0
    chunks: int = 0
    status: str = "indexed"  # indexed | skipped | failed
    error: str | None = None
    seconds: float = 0.0


class IngestReport(BaseModel):
    filings: list[FilingResult] = Field(default_factory=list)

    @property
    def chunks(self) -> int:
        return sum(f.chunks for f in self.filings)


class IngestionService:
    def __init__(self, store: VectorStore, sec: SECClient | None, chunk_tokens: int = 450, overlap_tokens: int = 60):
        self.store, self.sec = store, sec
        self.chunk_tokens, self.overlap_tokens = chunk_tokens, overlap_tokens
        self.on_change: list = []  # callbacks (e.g. clear answer cache)

    def ingest_ticker(self, ticker: str, forms: tuple[str, ...] = ("10-K",), limit: int = 1, force: bool = False) -> IngestReport:
        if self.sec is None:
            raise RuntimeError("SEC client not configured")
        report = IngestReport()
        with span("ingest.ticker", {"ticker": ticker, "forms": ",".join(forms), "limit": limit}):
            for ref in self.sec.list_filings(ticker, forms=forms, limit=limit):
                t0 = time.perf_counter()
                try:
                    if not force and self.store.has_document(ref.accession):
                        report.filings.append(FilingResult(ticker=ref.ticker, form=ref.form, accession=ref.accession,
                                                           fiscal_year=ref.fiscal_year, status="skipped"))
                        continue
                    with span("ingest.download", {"url": ref.url}):
                        html = self.sec.fetch_document(ref)
                    res = self.ingest_html(html, ref, force=force)
                except Exception as e:  # one bad filing must not abort the batch
                    log.exception("ingest failed")
                    res = FilingResult(ticker=ref.ticker, form=ref.form, accession=ref.accession,
                                       fiscal_year=ref.fiscal_year, status="failed", error=str(e))
                res.seconds = round(time.perf_counter() - t0, 2)
                report.filings.append(res)
        return report

    def ingest_html(self, html: str, ref: FilingRef, force: bool = False) -> FilingResult:
        if force:
            self.store.delete_document(ref.accession)
        with span("ingest.parse", {"accession": ref.accession}):
            sections = parse_filing(html, ref.form)
        meta = {
            "ticker": ref.ticker, "company": ref.company, "cik": ref.cik, "form": ref.form,
            "accession": ref.accession, "filing_date": ref.filing_date, "report_date": ref.report_date,
            "fiscal_year": ref.fiscal_year, "url": ref.url, "parser_version": PARSER_VERSION,
            "ingested_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        with span("ingest.chunk", {"sections": len(sections)}):
            chunks = chunk_filing(sections, meta, self.chunk_tokens, self.overlap_tokens)
        with span("ingest.embed_upsert", {"chunks": len(chunks)}):
            self.store.upsert(chunks)
        INGESTED_CHUNKS.labels(ref.form).inc(len(chunks))
        for cb in self.on_change:
            cb()
        log_event(log, "filing indexed", ticker=ref.ticker, form=ref.form, accession=ref.accession,
                  sections=len(sections), chunks=len(chunks))
        return FilingResult(ticker=ref.ticker, form=ref.form, accession=ref.accession, fiscal_year=ref.fiscal_year,
                            sections=len(sections), chunks=len(chunks))
