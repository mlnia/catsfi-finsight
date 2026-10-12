# FinSight evaluation — 2026-10-13T01:23:13

Dataset: **fixture** · LLM: `offline/offline-extractive` · embeddings: `hash-384` · reranker: `none`

| mode | prompt | hit@5 | mrr | keyword_recall | groundedness | citation_validity | abstention_accuracy | latency_p50_ms | latency_p95_ms | tokens | cost_usd | n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid | rag_answer@v1 | 1.0 | 0.9583 | 1.0 | 1.0 | 1.0 | 1.0 | 8.7 | 14.4 | 10991 | 0.0 | 14 |
| hybrid | rag_answer@v2 | 1.0 | 0.9583 | 1.0 | 1.0 | 1.0 | 1.0 | 9.0 | 10.5 | 13609 | 0.0 | 14 |
| dense | rag_answer@v1 | 1.0 | 0.9167 | 1.0 | 1.0 | 1.0 | 1.0 | 4.5 | 5.3 | 10618 | 0.0 | 14 |
| dense | rag_answer@v2 | 1.0 | 0.9167 | 1.0 | 1.0 | 1.0 | 1.0 | 4.5 | 5.9 | 13236 | 0.0 | 14 |
| sparse | rag_answer@v1 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 4.4 | 5.3 | 9991 | 0.0 | 14 |
| sparse | rag_answer@v2 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 5.0 | 5.6 | 12422 | 0.0 | 14 |

**Prompt-injection detector** — precision 1.0, recall 1.0, F1 1.0 (n=20)

Setup: {"chunks": 11, "poisoned_chunks_sanitized": 1, "injection_leaked_to_context": false}

Gate: **PASSED** []
