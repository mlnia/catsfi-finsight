"""Second-stage reranking of fused candidates."""

from __future__ import annotations

from typing import Protocol

from finsight.config import Settings
from finsight.models import RetrievedChunk


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]: ...


class NoopReranker:
    name = "none"

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        return chunks[:top_k]


class CrossEncoderReranker:
    """ms-marco MiniLM cross-encoder via fastembed (ONNX, CPU-friendly)."""

    def __init__(self, model: str):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._m = TextCrossEncoder(model_name=model)
        self.name = model

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not chunks:
            return []
        scores = list(self._m.rerank(query, [c.text for c in chunks]))
        ranked = sorted(zip(scores, chunks, strict=True), key=lambda x: x[0], reverse=True)[:top_k]
        return [c.model_copy(update={"score": float(s)}) for s, c in ranked]


def build_reranker(s: Settings) -> Reranker:
    return CrossEncoderReranker(s.reranker_model) if s.reranker == "cross-encoder" else NoopReranker()
