"""FastAPI application: RAG query, research agent (with HITL approvals), ingestion, ops endpoints."""

from __future__ import annotations

import logging
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from finsight import __version__
from finsight.agent.service import AgentResult
from finsight.api.security import Principal, RateLimiter, require
from finsight.config import Settings, get_settings
from finsight.container import Container, build_container
from finsight.ingestion.pipeline import IngestReport
from finsight.models import QueryResponse, SearchFilters
from finsight.observability.metrics import REQ_LATENCY
from finsight.observability.tracing import setup_logging, setup_tracing
from finsight.prompts.registry import list_prompts

log = logging.getLogger(__name__)
STATIC = Path(__file__).resolve().parent.parent / "static"


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    filters: SearchFilters | None = None
    prompt_version: str | None = Field(None, description="e.g. rag_answer@v1 for A/B comparison")
    top_k: int | None = Field(None, ge=1, le=12)
    use_cache: bool = True


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=1000)
    filters: SearchFilters | None = None
    top_k: int = Field(8, ge=1, le=20)
    mode: Literal["hybrid", "dense", "sparse"] = "hybrid"


class AgentRequest(BaseModel):
    message: str = Field(min_length=2, max_length=4000)
    thread_id: str | None = Field(None, pattern=r"^[A-Za-z0-9_-]{1,64}$")


class ApprovalRequest(BaseModel):
    approved: bool
    reason: str | None = Field(None, max_length=500)


class IngestRequest(BaseModel):
    tickers: list[str] = Field(min_length=1, max_length=10)
    forms: list[Literal["10-K", "10-Q"]] = ["10-K"]
    limit: int = Field(1, ge=1, le=4)
    force: bool = False


def _seed(c: Container) -> None:
    forms = tuple(f.strip() for f in c.settings.seed_forms.split(",") if f.strip())
    for t in (x.strip() for x in c.settings.seed_tickers.split(",") if x.strip()):
        try:
            rep = c.ingestion.ingest_ticker(t, forms, 1)
            log.info("seeded %s: %s chunks", t, rep.chunks)
        except Exception:
            log.exception("seeding %s failed", t)


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    s = settings or (container.settings if container else get_settings())
    setup_logging(s.log_level)
    setup_tracing(s.otel_service_name, s.otel_exporter_otlp_endpoint, s.otel_console)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.container = container or build_container(s)
        app.state.rate_limiter = RateLimiter(s.rate_limit_per_minute)
        log.info("FinSight ready (llm=%s/%s, embeddings=%s, reranker=%s)", s.llm_provider,
                 app.state.container.llm.model, s.embedding_provider, s.reranker)
        if s.seed_tickers and app.state.container.store.count() == 0:
            threading.Thread(target=_seed, args=(app.state.container,), daemon=True, name="seed").start()
        yield

    app = FastAPI(title="CATsFi FinSight", version=__version__, lifespan=lifespan,
                  description="Grounded RAG + agentic research over SEC filings, XBRL fundamentals and CATsFi technical analysis.")
    app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in s.cors_origins.split(",")],
                       allow_methods=["*"], allow_headers=["*"])
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app, excluded_urls="metrics,healthz,readyz")
    except Exception:  # instrumentation is optional
        pass

    @app.middleware("http")
    async def timing(request: Request, call_next):
        t0 = time.perf_counter()
        resp = await call_next(request)
        route = request.scope.get("route")
        REQ_LATENCY.labels(getattr(route, "path", "unmatched")).observe(time.perf_counter() - t0)
        resp.headers["X-Response-Time-ms"] = f"{(time.perf_counter() - t0) * 1000:.1f}"
        return resp

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        log.exception("unhandled error on %s", request.url.path)
        return JSONResponse({"detail": "internal error"}, status_code=500)

    def c(request: Request) -> Container:
        return request.app.state.container

    # ---- ops ----------------------------------------------------------------------
    @app.get("/healthz", include_in_schema=False)
    def healthz():
        return {"status": "ok"}

    @app.get("/readyz", include_in_schema=False)
    def readyz(request: Request):
        try:
            n = c(request).store.count()
        except Exception as e:
            raise HTTPException(503, f"vector store unavailable: {e}") from e
        return {"status": "ready", "chunks": n}

    @app.get("/metrics", include_in_schema=False)
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/", include_in_schema=False)
    def ui():
        return FileResponse(STATIC / "index.html")

    # ---- RAG ------------------------------------------------------------------------
    @app.post("/v1/query", response_model=QueryResponse, tags=["rag"])
    async def query(req: QueryRequest, request: Request, _: Principal = Depends(require("viewer"))):
        try:
            return await run_in_threadpool(c(request).rag.answer, req.question, req.filters, req.prompt_version,
                                           req.top_k, req.use_cache)
        except KeyError as e:
            raise HTTPException(400, str(e)) from e

    @app.post("/v1/search", tags=["rag"])
    async def search(req: SearchRequest, request: Request, _: Principal = Depends(require("viewer"))):
        chunks, quarantined = await run_in_threadpool(c(request).rag.retrieve, req.query, req.filters, req.top_k, req.mode)
        return {"results": [{"id": ch.id, "label": ch.label(), "score": ch.score, "text": ch.text, "metadata": ch.metadata}
                            for ch in chunks], "quarantined": quarantined}

    # ---- agent ------------------------------------------------------------------------
    @app.post("/v1/agent/chat", response_model=AgentResult, tags=["agent"])
    async def agent_chat(req: AgentRequest, request: Request, _: Principal = Depends(require("viewer"))):
        try:
            return await run_in_threadpool(c(request).agent.run, req.message, req.thread_id)
        except ValueError as e:
            raise HTTPException(409, str(e)) from e

    @app.get("/v1/agent/pending", tags=["agent"])
    def agent_pending(request: Request, _: Principal = Depends(require("admin"))):
        return c(request).agent.list_pending()

    @app.post("/v1/agent/{thread_id}/approval", response_model=AgentResult, tags=["agent"])
    async def agent_approve(thread_id: str, req: ApprovalRequest, request: Request,
                            p: Principal = Depends(require("admin"))):
        try:
            return await run_in_threadpool(c(request).agent.resume, thread_id, req.approved, p.key_id, req.reason)
        except KeyError as e:
            raise HTTPException(404, str(e)) from e

    # ---- ingestion / admin ----------------------------------------------------------
    @app.post("/v1/ingest", tags=["admin"])
    async def ingest(req: IngestRequest, request: Request, _: Principal = Depends(require("admin"))):
        reports: list[IngestReport] = []
        for t in req.tickers:
            reports.append(await run_in_threadpool(c(request).ingestion.ingest_ticker, t, tuple(req.forms), req.limit, req.force))
        filings = [f for r in reports for f in r.filings]
        return {"filings": [f.model_dump() for f in filings], "chunks": sum(f.chunks for f in filings)}

    @app.get("/v1/documents", tags=["admin"])
    def documents(request: Request, _: Principal = Depends(require("viewer"))):
        return c(request).store.documents()

    @app.get("/v1/prompts", tags=["admin"])
    def prompts(_: Principal = Depends(require("viewer"))):
        return list_prompts()

    @app.get("/v1/config", tags=["admin"])
    def config(request: Request, _: Principal = Depends(require("viewer"))):
        st = c(request).settings
        return {"llm_provider": st.llm_provider, "model": c(request).llm.model, "embedding_provider": st.embedding_provider,
                "reranker": st.reranker, "rag_prompt": st.rag_prompt, "agent_prompt": st.agent_prompt,
                "guardrail_mode": st.guardrail_mode, "auth": st.auth_enabled, "version": __version__}

    return app
