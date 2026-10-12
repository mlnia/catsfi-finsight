"""Prometheus metrics (scraped at /metrics, visualised by the bundled Grafana dashboard)."""

from prometheus_client import Counter, Gauge, Histogram

REQ_LATENCY = Histogram(
    "finsight_request_seconds", "End-to-end request latency", ["endpoint"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32),
)
LLM_LATENCY = Histogram("finsight_llm_seconds", "LLM call latency", ["provider", "op"],
                        buckets=(0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32))
LLM_TOKENS = Counter("finsight_llm_tokens_total", "LLM tokens", ["provider", "model", "direction"])
LLM_COST = Counter("finsight_llm_cost_usd_total", "Estimated LLM spend in USD", ["provider", "model"])
RETRIEVAL_LATENCY = Histogram("finsight_retrieval_seconds", "Retrieval (hybrid + rerank) latency", ["stage"],
                              buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2))
GUARDRAIL_EVENTS = Counter("finsight_guardrail_events_total", "Guardrail interventions", ["type", "action"])
GROUNDEDNESS = Histogram("finsight_groundedness", "Answer groundedness score",
                         buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
CACHE_HITS = Counter("finsight_cache_hits_total", "Answer cache hits", ["cache"])
INGESTED_CHUNKS = Counter("finsight_ingested_chunks_total", "Chunks written to the vector store", ["form"])
TOOL_CALLS = Counter("finsight_agent_tool_calls_total", "Agent tool invocations", ["tool", "status"])
PENDING_APPROVALS = Gauge("finsight_agent_pending_approvals", "Agent runs waiting for human approval")
