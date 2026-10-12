# FinSight evaluation — 2026-10-13T01:26:13

Dataset: **live** · LLM: `offline/offline-extractive` · embeddings: `BAAI/bge-small-en-v1.5` · reranker: `Xenova/ms-marco-MiniLM-L-6-v2`

| mode | prompt | hit@5 | mrr | keyword_recall | groundedness | citation_validity | abstention_accuracy | latency_p50_ms | latency_p95_ms | tokens | cost_usd | n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid | rag_answer@v2 | 1.0 | 0.9 | 0.9 | 1.0 | 1.0 | 0.9167 | 2411.5 | 2782.2 | 31807 | 0.0 | 12 |
| dense | rag_answer@v2 | 1.0 | 0.95 | 0.9 | 1.0 | 1.0 | 0.9167 | 2301.1 | 2466.6 | 31677 | 0.0 | 12 |
| sparse | rag_answer@v2 | 1.0 | 0.9 | 0.9 | 1.0 | 1.0 | 0.9167 | 2424.9 | 2540.7 | 31868 | 0.0 | 12 |

**Prompt-injection detector** — precision 1.0, recall 1.0, F1 1.0 (n=20)

Setup: {"AAPL": 0, "MSFT": 0, "NVDA": 0}

Gate: **PASSED** []
