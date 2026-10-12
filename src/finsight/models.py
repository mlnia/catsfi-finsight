"""Domain models shared across ingestion, retrieval, generation and the API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class FilingRef(BaseModel):
    ticker: str
    company: str
    cik: int
    form: str
    accession: str
    filing_date: str
    report_date: str | None = None
    primary_document: str
    url: str

    @property
    def fiscal_year(self) -> int | None:
        d = self.report_date or self.filing_date
        return int(d[:4]) if d else None


class Section(BaseModel):
    key: str  # e.g. "item_7"
    title: str  # e.g. "Item 7. Management's Discussion and Analysis..."
    text: str


class Chunk(BaseModel):
    id: str
    text: str  # text with contextual header (what gets embedded)
    body: str  # raw chunk body (what gets quoted)
    metadata: dict[str, Any]


class RetrievedChunk(BaseModel):
    id: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)

    def label(self) -> str:
        m = self.metadata
        return f"{m.get('ticker', '?')} {m.get('form', '?')} FY{m.get('fiscal_year', '?')} · {m.get('section_title', '')}"


class SearchFilters(BaseModel):
    ticker: str | None = None
    form: str | None = None
    fiscal_year: int | None = None
    section: str | None = None


# ---- structured LLM output ------------------------------------------------------


class Citation(BaseModel):
    chunk_id: str = Field(description="ID of the context chunk that supports the claim, e.g. 'c3'.")
    quote: str = Field(description="Short verbatim quote (<= 30 words) copied from that chunk.")


class RAGAnswer(BaseModel):
    answer: str = Field(description="Concise answer in English. Cite chunk ids inline like [c1].")
    citations: list[Citation] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "medium"
    insufficient_context: bool = Field(
        default=False, description="True when the context does not contain the answer."
    )


# ---- API responses ----------------------------------------------------------------


class GuardrailReport(BaseModel):
    input_flags: list[str] = Field(default_factory=list)
    pii_masked: list[str] = Field(default_factory=list)
    quarantined_chunks: list[str] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)
    unsupported_numbers: list[str] = Field(default_factory=list)
    groundedness: float | None = None
    action: Literal["pass", "flagged", "blocked", "abstained"] = "pass"


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def __add__(self, o: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + o.input_tokens,
            output_tokens=self.output_tokens + o.output_tokens,
            cost_usd=round(self.cost_usd + o.cost_usd, 6),
        )


class SourceOut(BaseModel):
    chunk_id: str
    label: str
    url: str | None = None
    quote: str | None = None
    score: float | None = None


class QueryResponse(BaseModel):
    answer: str
    confidence: str
    sources: list[SourceOut]
    guardrails: GuardrailReport
    usage: Usage
    latency_ms: float
    prompt_version: str
    model: str
    cached: bool = False
    trace_id: str | None = None
