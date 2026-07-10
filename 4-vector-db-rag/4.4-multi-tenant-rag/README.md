# 4.4 — Multi-Tenant RAG

Tenant-aware document isolation for RAG systems. Demonstrates the
difference between isolated (tenant-aware) and non-isolated (buggy)
retrieval, measuring recall, precision, and cross-tenant leakage.

## Quick Start

```python
from multi_tenant_rag.data import default_corpus
from multi_tenant_rag.tenant_store import TenantVectorStore
from multi_tenant_rag.retriever import EmbeddingClient, TenantRetriever
from multi_tenant_rag.pipeline import evaluate_isolation, evaluate_cross_tenant_leakage
from multi_tenant_rag.reporting import evaluation_to_markdown

# 1. Build corpus and store
corpus = default_corpus()  # 3 tenants × 8 docs × 3 queries

store = TenantVectorStore()
client = EmbeddingClient("http://localhost:8080")

for tenant in corpus.tenants:
    vecs = client.embed_many(tenant.documents)
    store.add_many(tenant.id, tenant.documents, vecs)

retriever = TenantRetriever(store=store, client=client)

# 2. Evaluate both modes
isolated = evaluate_isolation(retriever, corpus, top_k=5)
leaky = evaluate_cross_tenant_leakage(retriever, corpus, top_k=5)

# 3. Compare
print(evaluation_to_markdown(isolated, leaky))
```

## Architecture

```
multi_tenant_rag/
├── data.py           # 3 tenants: Acme Corp, GlobeBank, HealthPlus
├── tenant_store.py   # TenantVectorStore — partitioned by tenant ID
├── retriever.py      # EmbeddingClient, TenantRetriever
├── pipeline.py       # evaluate_isolation, evaluate_cross_tenant_leakage
└── reporting.py      # Markdown/JSON reports with comparison tables
```

## Key Concept: Tenant Isolation

```
# ✅ Isolated — tenant A only sees tenant A's docs
store.search("tenant_a", query_vec) -> [a1, a2, a3]

# ❌ Non-isolated — tenant A sees everyone's docs
store.search_all_tenants(query_vec) -> {a: [a1], b: [b1], c: [c1]}
```

## Tenants

| Tenant | Domain | Documents | Queries |
|--------|--------|-----------|---------|
| **Acme Corp** | Technology & engineering | 8 | 3 |
| **GlobeBank** | Finance & banking | 8 | 3 |
| **HealthPlus** | Healthcare | 8 | 3 |

## Metrics

| Metric | Description |
|--------|-------------|
| Recall | Fraction of relevant docs retrieved for the correct tenant |
| Precision | Fraction of retrieved docs that are relevant |
| **Leakage** | Documents returned that belong to OTHER tenants (should be 0) |

## Prerequisites

```bash
pip install numpy requests pytest

# Start llama.cpp with embeddings
llama-server --model your-model.gguf --embedding --cors * --port 8080
```

## Tests

```bash
cd 4-vector-db-rag/4.4-multi-tenant-rag
python -m pytest -v
```
