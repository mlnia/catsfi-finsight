"""Dense + sparse text encoders behind small interfaces.

Dense:  fastembed (local ONNX, BAAI/bge-small-en-v1.5), OpenAI / Azure OpenAI, or a deterministic
        hashing encoder used in tests / CI (no model download, no network).
Sparse: BM25-style term weights (TF saturation + length normalisation). Qdrant applies the IDF
        modifier server side, so sparse + dense fuse into hybrid search inside one query.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Protocol

import numpy as np

from finsight.config import Settings

_TOKEN = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")
STOPWORDS = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in", "is", "it", "its", "of", "on", "or", "that", "the", "this", "to", "was", "were", "will", "with", "our", "we", "us", "which", "what", "who", "how", "when", "where", "why", "do", "does", "did", "not", "no", "than", "then", "there", "their", "they", "these", "those", "been", "being", "into", "about", "over", "under", "such", "can", "could", "may", "might", "should", "would", "also", "other", "any", "all", "each"]
)


def tokenize(text: str) -> list[str]:
    toks = _TOKEN.findall(text.lower())
    return [_stem(t) for t in toks if t not in STOPWORDS and len(t) > 1]


def _stem(t: str) -> str:
    # tiny suffix stripper; good enough for financial English, zero dependencies
    for suf in ("ies", "es", "s", "ing", "ed"):
        if len(t) > len(suf) + 3 and t.endswith(suf) and not t[-len(suf) - 1].isdigit():
            return t[: -len(suf)] + ("y" if suf == "ies" else "")
    return t


class DenseEncoder(Protocol):
    dim: int
    name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class HashEncoder:
    """Feature-hashing bag of uni+bi-grams. Deterministic stand-in for tests and offline demos."""

    name = "hash-384"

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        toks = tokenize(text)
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:], strict=False)]
        v = np.zeros(self.dim, dtype=np.float32)
        for f in feats:
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        n = np.linalg.norm(v)
        return (v / n if n else v).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class FastEmbedEncoder:
    def __init__(self, model: str):
        from fastembed import TextEmbedding

        self._m = TextEmbedding(model_name=model)
        self.name = model
        self.dim = len(next(iter(self._m.embed(["probe"]))))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._m.embed(texts, batch_size=32)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._m.query_embed(text))).tolist()


class OpenAIEncoder:
    def __init__(self, client, model: str, dim: int | None = None):
        self._c, self.name = client, model
        self.dim = dim or (3072 if "large" in model else 1536)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), 128):
            resp = self._c.embeddings.create(model=self.name, input=texts[i : i + 128])
            out.extend(d.embedding for d in resp.data)
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def build_dense_encoder(s: Settings) -> DenseEncoder:
    if s.embedding_provider == "hash":
        return HashEncoder()
    if s.embedding_provider == "fastembed":
        return FastEmbedEncoder(s.embedding_model)
    if s.embedding_provider == "openai":
        from openai import OpenAI

        return OpenAIEncoder(OpenAI(api_key=s.openai_api_key, base_url=s.openai_base_url), s.embedding_model)
    if s.embedding_provider == "azure_openai":
        from openai import AzureOpenAI

        c = AzureOpenAI(api_key=s.azure_openai_api_key, azure_endpoint=s.azure_openai_endpoint, api_version=s.azure_openai_api_version)
        return OpenAIEncoder(c, s.embedding_model)
    raise ValueError(s.embedding_provider)


class BM25SparseEncoder:
    """Produces (indices, values) for Qdrant sparse vectors with the IDF modifier enabled."""

    def __init__(self, k1: float = 1.2, b: float = 0.75, avg_len: float = 220.0, buckets: int = 2**31 - 1):
        self.k1, self.b, self.avg_len, self.buckets = k1, b, avg_len, buckets

    def _idx(self, tok: str) -> int:
        return int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=4).digest(), "little") % self.buckets

    def encode_document(self, text: str) -> tuple[list[int], list[float]]:
        toks = tokenize(text)
        tf = Counter(toks)
        norm = self.k1 * (1 - self.b + self.b * len(toks) / self.avg_len)
        agg: dict[int, float] = {}
        for tok, f in tf.items():
            i = self._idx(tok)
            agg[i] = agg.get(i, 0.0) + f * (self.k1 + 1) / (f + norm)
        return list(agg), list(agg.values())

    def encode_query(self, text: str) -> tuple[list[int], list[float]]:
        idx = sorted({self._idx(t) for t in tokenize(text)})
        return idx, [1.0] * len(idx)


def cosine(a: list[float], b: list[float]) -> float:
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b, strict=False)) / (na * nb) if na and nb else 0.0
