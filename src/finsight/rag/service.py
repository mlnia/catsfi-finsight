"""Grounded question answering over SEC filings.

question -> PII mask -> injection check -> cache -> hybrid retrieval (dense + BM25, RRF) -> rerank
         -> quarantine poisoned chunks -> versioned prompt -> structured LLM output (validated + repaired)
         -> citation / quote / numeric grounding checks -> policy (pass | flag | abstain) -> response
"""

from __future__ import annotations

import hashlib
import json
import logging
import time

from finsight.cache import TTLCache
from finsight.config import Settings
from finsight.guardrails.grounding import check_grounding
from finsight.guardrails.injection import detect_injection, sanitize_untrusted
from finsight.guardrails.pii import mask_pii
from finsight.llm.base import LLMProvider, StructuredOutputError
from finsight.models import (
    GuardrailReport,
    QueryResponse,
    RAGAnswer,
    RetrievedChunk,
    SearchFilters,
    SourceOut,
    Usage,
)
from finsight.observability.metrics import CACHE_HITS, GROUNDEDNESS, GUARDRAIL_EVENTS, RETRIEVAL_LATENCY
from finsight.observability.tracing import current_trace_id, log_event, span
from finsight.prompts.registry import get_prompt
from finsight.retrieval.reranker import Reranker
from finsight.retrieval.store import VectorStore

log = logging.getLogger(__name__)

ABSTAIN = "I could not find sufficient, verifiable support for this in the indexed filings."
BLOCKED = "This request was blocked by the input safety policy (possible prompt injection)."


def render_context(chunks: list[RetrievedChunk]) -> tuple[str, dict[str, RetrievedChunk]]:
    by_id: dict[str, RetrievedChunk] = {}
    blocks = []
    for i, c in enumerate(chunks, 1):
        sid = f"c{i}"
        by_id[sid] = c
        blocks.append(f'<source id="{sid}" label="{c.label()}">\n{c.text}\n</source>')
    return "\n\n".join(blocks), by_id


class RAGService:
    def __init__(self, store: VectorStore, reranker: Reranker, llm: LLMProvider, settings: Settings):
        self.store, self.reranker, self.llm, self.s = store, reranker, llm, settings
        self.cache = TTLCache(settings.answer_cache_size, settings.answer_cache_ttl_s)

    # ---- retrieval -------------------------------------------------------------------
    def retrieve(self, query: str, filters: SearchFilters | None = None, top_k: int | None = None,
                 mode: str = "hybrid") -> tuple[list[RetrievedChunk], list[str]]:
        top_k = top_k or self.s.top_k
        with span("retrieval.hybrid", {"mode": mode, "filters": filters.model_dump_json() if filters else None}) as sp:
            t0 = time.perf_counter()
            cands = self.store.search(query, filters, limit=self.s.rerank_candidates,
                                      candidates=self.s.retrieval_candidates, mode=mode)
            RETRIEVAL_LATENCY.labels("fusion").observe(time.perf_counter() - t0)
            sp.set_attribute("retrieval.candidates", len(cands))
        with span("retrieval.rerank", {"reranker": self.reranker.name}):
            t0 = time.perf_counter()
            ranked = self.reranker.rerank(query, cands, top_k + 4)
            RETRIEVAL_LATENCY.labels("rerank").observe(time.perf_counter() - t0)
        clean, quarantined = [], []
        for c in ranked:  # indirect prompt injection: documents are untrusted input
            text, removed = sanitize_untrusted(c.text, self.s.injection_threshold)
            if removed:
                quarantined.append(c.id)
                GUARDRAIL_EVENTS.labels("indirect_injection", "sanitize" if text else "quarantine").inc()
            if text:
                clean.append(c if not removed else c.model_copy(update={"text": text}))
        return clean[:top_k], quarantined

    # ---- answering -------------------------------------------------------------------
    def answer(self, question: str, filters: SearchFilters | None = None, prompt_ref: str | None = None,
               top_k: int | None = None, use_cache: bool = True, mode: str = "hybrid") -> QueryResponse:
        t0 = time.perf_counter()
        prompt = get_prompt(prompt_ref or self.s.rag_prompt)
        report = GuardrailReport()
        with span("rag.answer", {"prompt": prompt.ref, "prompt.sha": prompt.sha, "llm.model": self.llm.model}):
            q = question.strip()
            if self.s.mask_pii:
                q, report.pii_masked = mask_pii(q)
                for p in report.pii_masked:
                    GUARDRAIL_EVENTS.labels(f"pii_{p.lower()}", "mask").inc()

            inj = detect_injection(q)
            report.input_flags = inj.flags
            if inj.is_attack(self.s.injection_threshold):
                GUARDRAIL_EVENTS.labels("prompt_injection", self.s.guardrail_mode).inc()
                if self.s.guardrail_mode == "block":
                    report.action = "blocked"
                    return self._resp(BLOCKED, "low", [], report, Usage(), t0, prompt.ref)
                report.action = "flagged"

            key = hashlib.sha256(json.dumps([q.lower(), filters.model_dump() if filters else None, prompt.ref,
                                             top_k, mode, self.llm.model]).encode()).hexdigest()
            if use_cache and (hit := self.cache.get(key)) is not None:
                CACHE_HITS.labels("answer").inc()
                return hit.model_copy(update={"cached": True, "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                                              "trace_id": current_trace_id()})

            chunks, report.quarantined_chunks = self.retrieve(q, filters, top_k, mode)
            if not chunks:
                report.action = "abstained"
                return self._resp(ABSTAIN + " No relevant passages were retrieved.", "low", [], report, Usage(), t0, prompt.ref)

            context, by_id = render_context(chunks)
            messages = prompt.render(context=context, question=q)
            try:
                with span("rag.generate"):
                    ans, usage = self.llm.generate_structured(messages, RAGAnswer, temperature=self.s.llm_temperature,
                                                              max_tokens=self.s.llm_max_tokens)
            except StructuredOutputError as e:
                log.warning("structured output failed: %s", e)
                report.action = "abstained"
                return self._resp(ABSTAIN, "low", self._sources(chunks, {}), report, Usage(), t0, prompt.ref)

            with span("guardrails.grounding") as sp:
                g = check_grounding(ans.answer, {k: v.text for k, v in by_id.items()},
                                    [(c.chunk_id, c.quote) for c in ans.citations])
                report.groundedness = g.groundedness
                report.invalid_citations = g.invalid_citations + [f"quote:{c}" for c in g.bad_quotes]
                report.unsupported_numbers = g.unsupported_numbers
                sp.set_attribute("groundedness", g.groundedness)
            GROUNDEDNESS.observe(g.groundedness)

            text, confidence = ans.answer, ans.confidence
            if ans.insufficient_context:
                report.action = "abstained"
                text = ans.answer or ABSTAIN
                confidence = "low"
            elif not g.passes(self.s.min_groundedness) or g.invalid_citations:
                GUARDRAIL_EVENTS.labels("ungrounded_answer", self.s.guardrail_mode).inc()
                if self.s.guardrail_mode == "block":
                    report.action = "abstained"
                    text, confidence = ABSTAIN, "low"
                else:
                    report.action = "flagged" if report.action == "pass" else report.action
                    confidence = "low"
            if self.s.mask_pii:
                text, out_pii = mask_pii(text)
                report.pii_masked = sorted(set(report.pii_masked) | set(out_pii))

            quotes = {c.chunk_id: c.quote for c in ans.citations}
            cited = set(g.cited_ids)
            sources = self._sources([by_id[k] for k in by_id if k in cited] or chunks, quotes, by_id)
            resp = self._resp(text, confidence, sources, report, usage, t0, prompt.ref)
            log_event(log, "rag answer", prompt=prompt.ref, action=report.action, groundedness=g.groundedness,
                      chunks=len(chunks), cost_usd=usage.cost_usd, latency_ms=resp.latency_ms)
            if report.action == "pass":
                self.cache.set(key, resp)
            return resp

    def _sources(self, chunks: list[RetrievedChunk], quotes: dict[str, str],
                 by_id: dict[str, RetrievedChunk] | None = None) -> list[SourceOut]:
        rev = {v.id: k for k, v in (by_id or {}).items()}
        return [SourceOut(chunk_id=rev.get(c.id, c.id), label=c.label(), url=c.metadata.get("url"),
                          quote=quotes.get(rev.get(c.id, "")), score=round(c.score, 4)) for c in chunks]

    def _resp(self, text, confidence, sources, report, usage, t0, prompt_ref) -> QueryResponse:
        return QueryResponse(answer=text, confidence=confidence, sources=sources, guardrails=report, usage=usage,
                             latency_ms=round((time.perf_counter() - t0) * 1000, 1), prompt_version=prompt_ref,
                             model=self.llm.model, trace_id=current_trace_id())
