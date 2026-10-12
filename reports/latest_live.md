# FinSight evaluation — 2026-10-13T01:26:25

Dataset: **live** · LLM: `offline/offline-extractive` · embeddings: `BAAI/bge-small-en-v1.5` · reranker: `none`

| mode | prompt | hit@5 | mrr | keyword_recall | groundedness | citation_validity | abstention_accuracy | latency_p50_ms | latency_p95_ms | tokens | cost_usd | n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid | rag_answer@v2 | 1.0 | 0.8 | 0.9 | 1.0 | 1.0 | 0.9167 | 133.2 | 149.5 | 32162 | 0.0 | 12 |
| dense | rag_answer@v2 | 0.9 | 0.75 | 0.9 | 1.0 | 1.0 | 0.9167 | 49.3 | 57.5 | 31053 | 0.0 | 12 |
| sparse | rag_answer@v2 | 0.9 | 0.7333 | 0.9 | 1.0 | 1.0 | 0.9167 | 117.0 | 133.3 | 32236 | 0.0 | 12 |

**Prompt-injection detector** — precision 1.0, recall 1.0, F1 1.0 (n=20)

Setup: {"AAPL": 0, "MSFT": 0, "NVDA": 0}

Gate: **PASSED** []
