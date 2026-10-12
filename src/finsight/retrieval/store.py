"""Qdrant vector store with named dense + sparse vectors and server-side RRF hybrid fusion.

Runs against a Qdrant server (docker / Qdrant Cloud) or in embedded mode (on-disk path or
in-memory) so the same code works for single-container deploys, local dev and tests.
"""

from __future__ import annotations

import warnings

from qdrant_client import QdrantClient, models

from finsight.config import Settings
from finsight.models import Chunk, RetrievedChunk, SearchFilters
from finsight.retrieval.embeddings import BM25SparseEncoder, DenseEncoder

DENSE, SPARSE = "dense", "bm25"
INDEXED_FIELDS = {
    "ticker": models.PayloadSchemaType.KEYWORD,
    "form": models.PayloadSchemaType.KEYWORD,
    "section": models.PayloadSchemaType.KEYWORD,
    "accession": models.PayloadSchemaType.KEYWORD,
    "fiscal_year": models.PayloadSchemaType.INTEGER,
}


def make_client(s: Settings) -> QdrantClient:
    if s.qdrant_url:
        return QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key, timeout=30)
    if s.qdrant_path == ":memory:" or (s.env == "test" and not s.qdrant_path):
        return QdrantClient(location=":memory:")
    return QdrantClient(path=s.qdrant_path or str(s.data_dir / "qdrant"))


class VectorStore:
    def __init__(self, client: QdrantClient, dense: DenseEncoder, sparse: BM25SparseEncoder, collection: str):
        self.client, self.dense, self.sparse, self.collection = client, dense, sparse, collection
        self.ensure_collection()

    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            return
        self.client.create_collection(
            self.collection,
            vectors_config={DENSE: models.VectorParams(size=self.dense.dim, distance=models.Distance.COSINE)},
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        with warnings.catch_warnings():  # embedded mode warns that payload indexes are no-ops
            warnings.simplefilter("ignore")
            for field, schema in INDEXED_FIELDS.items():
                self.client.create_payload_index(self.collection, field, schema)

    # ---- writes ------------------------------------------------------------------
    def upsert(self, chunks: list[Chunk], batch: int = 64) -> int:
        for i in range(0, len(chunks), batch):
            part = chunks[i : i + batch]
            dense = self.dense.embed_documents([c.text for c in part])
            points = []
            for c, dv in zip(part, dense, strict=True):
                si, sv = self.sparse.encode_document(c.text)
                points.append(
                    models.PointStruct(
                        id=c.id,
                        vector={DENSE: dv, SPARSE: models.SparseVector(indices=si, values=sv)},
                        payload={**c.metadata, "text": c.text, "body": c.body},
                    )
                )
            self.client.upsert(self.collection, points=points, wait=True)
        return len(chunks)

    def delete_document(self, accession: str) -> None:
        self.client.delete(self.collection, points_selector=models.FilterSelector(filter=_eq("accession", accession)))

    def has_document(self, accession: str) -> bool:
        return self.count(_eq("accession", accession)) > 0

    def count(self, flt: models.Filter | None = None) -> int:
        return self.client.count(self.collection, count_filter=flt, exact=True).count

    def documents(self) -> list[dict]:
        """Distinct indexed filings (for the UI / admin endpoints)."""
        seen: dict[str, dict] = {}
        offset = None
        while True:
            pts, offset = self.client.scroll(
                self.collection, limit=256, offset=offset, with_vectors=False,
                with_payload=["ticker", "company", "form", "fiscal_year", "filing_date", "accession", "url"],
            )
            for p in pts:
                acc = p.payload["accession"]
                d = seen.setdefault(acc, {**p.payload, "chunks": 0})
                d["chunks"] += 1
            if offset is None:
                break
        return sorted(seen.values(), key=lambda d: (d["ticker"], d.get("filing_date", "")), reverse=False)

    # ---- reads -------------------------------------------------------------------
    def search(self, query: str, filters: SearchFilters | None = None, limit: int = 20, candidates: int = 30,
               mode: str = "hybrid") -> list[RetrievedChunk]:
        flt = build_filter(filters)
        dq = self.dense.embed_query(query)
        si, sv = self.sparse.encode_query(query)
        sq = models.SparseVector(indices=si, values=sv)
        if mode == "dense":
            res = self.client.query_points(self.collection, query=dq, using=DENSE, query_filter=flt, limit=limit, with_payload=True)
        elif mode == "sparse":
            res = self.client.query_points(self.collection, query=sq, using=SPARSE, query_filter=flt, limit=limit, with_payload=True)
        else:
            res = self.client.query_points(
                self.collection,
                prefetch=[
                    models.Prefetch(query=dq, using=DENSE, filter=flt, limit=candidates),
                    models.Prefetch(query=sq, using=SPARSE, filter=flt, limit=candidates),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                with_payload=True,
            )
        out = []
        for p in res.points:
            payload = dict(p.payload or {})
            text = payload.pop("body", "") or payload.get("text", "")
            payload.pop("text", None)
            out.append(RetrievedChunk(id=str(p.id), text=text, score=float(p.score), metadata=payload))
        return out


def _eq(field: str, value) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(key=field, match=models.MatchValue(value=value))])


def build_filter(f: SearchFilters | None) -> models.Filter | None:
    if not f:
        return None
    must = []
    if f.ticker:
        must.append(models.FieldCondition(key="ticker", match=models.MatchValue(value=f.ticker.upper())))
    if f.form:
        must.append(models.FieldCondition(key="form", match=models.MatchValue(value=f.form.upper())))
    if f.fiscal_year:
        must.append(models.FieldCondition(key="fiscal_year", match=models.MatchValue(value=int(f.fiscal_year))))
    if f.section:
        must.append(models.FieldCondition(key="section", match=models.MatchValue(value=f.section)))
    return models.Filter(must=must) if must else None
