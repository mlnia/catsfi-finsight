"""Composition root: builds every component once from Settings (easy to override in tests)."""

from __future__ import annotations

from dataclasses import dataclass

from finsight.agent.graph import build_agent_graph
from finsight.agent.service import AgentService
from finsight.agent.tools import build_tools
from finsight.config import Settings
from finsight.ingestion.pipeline import IngestionService
from finsight.ingestion.sec_client import SECClient
from finsight.llm.base import LLMProvider
from finsight.llm.providers import build_llm
from finsight.rag.service import RAGService
from finsight.retrieval.embeddings import BM25SparseEncoder, build_dense_encoder
from finsight.retrieval.reranker import build_reranker
from finsight.retrieval.store import VectorStore, make_client


@dataclass
class Container:
    settings: Settings
    store: VectorStore
    llm: LLMProvider
    rag: RAGService
    sec: SECClient | None
    ingestion: IngestionService
    agent: AgentService


def build_container(s: Settings, *, llm: LLMProvider | None = None, sec: SECClient | None = None,
                    fetch_closes=None, qdrant_client=None) -> Container:
    store = VectorStore(qdrant_client or make_client(s), build_dense_encoder(s), BM25SparseEncoder(), s.collection)
    llm = llm or build_llm(s)
    rag = RAGService(store, build_reranker(s), llm, s)
    if sec is None and s.env != "test":
        sec = SECClient(s.sec_user_agent, s.data_dir / "sec_cache", rps=s.sec_rps)
    ingestion = IngestionService(store, sec, s.chunk_tokens, s.chunk_overlap_tokens)
    ingestion.on_change.append(rag.cache.clear)  # new documents invalidate cached answers
    kw = {"fetch_closes": fetch_closes} if fetch_closes else {}
    tools = build_tools(rag, sec, ingestion, **kw)
    agent = AgentService(build_agent_graph(llm, tools, s), s)
    return Container(s, store, llm, rag, sec, ingestion, agent)
