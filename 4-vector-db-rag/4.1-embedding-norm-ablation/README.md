# 4.1 — Embedding Norm Ablation

Ablation study measuring how L2 normalisation of embeddings affects
vector retrieval quality. Tests four treatments across similarity
functions and normalisation strategies and reports precision/recall/AP.

## Quick Start

```python
from embedding_norm_ablation.data import default_corpus
from embedding_norm_ablation.embeddings import EmbeddingClient
from embedding_norm_ablation.ablation import AblationConfig, run_ablation
from embedding_norm_ablation.reporting import report_to_markdown

# 1. Load the built-in test corpus (30 docs, 10 queries, relevance judgments)
corpus = default_corpus()

# 2. Point at a running llama.cpp server
client = EmbeddingClient(server_url="http://localhost:8080")

# 3. Run all four standard treatments
config = AblationConfig(corpus=corpus, top_k=5)
result = run_ablation(client, config)

# 4. View results
print(report_to_markdown(result))

# Compare two treatments
summary = result.summary()
norm = summary["norm-cosine"]
unnorm = summary["unnorm-cosine"]
print(f"Norm MAP:   {norm['mean_ap']:.4f}")
print(f"Unnorm MAP: {unnorm['mean_ap']:.4f}")
```

## Architecture

```
embedding_norm_ablation/
├── data.py         # Built-in corpus with relevance judgments
├── embeddings.py   # EmbeddingClient, L2 normalisation, norm stats
├── retrieval.py    # Similarity functions, retrieval, evaluation metrics
├── ablation.py     # Experiment runner, treatment configs
└── reporting.py    # Markdown/JSON report generation
```

## The Four Standard Treatments

| Treatment | Vectors | Similarity | Notes |
|---|---|---|---|
| `unnorm-cosine` | raw | cosine | Baseline |
| `norm-cosine` | L2-normalised | cosine | = dot product on unit vectors |
| `unnorm-dot` | raw | dot product | Magnitude-sensitive |
| `norm-dot` | L2-normalised | dot product | Equivalent to cosine |

Run a subset by passing `treatments`:

```python
from embedding_norm_ablation.ablation import TREATMENT_NORM_COSINE

result = run_ablation(client, config, treatments=[TREATMENT_NORM_COSINE])
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| P@1 | Precision at rank 1 |
| P@5 | Precision at rank 5 |
| R@5 | Recall at rank 5 |
| AP | Average precision (area under P@k curve) |

## Custom Corpus

```python
from embedding_norm_ablation.data import Corpus

corpus = Corpus(
    documents=["Doc A", "Doc B", "Doc C"],
    queries=["query 1", "query 2"],
    relevant_doc_ids={0: [0, 1], 1: [2]},
)
```

## Built-in Corpora

- `default_corpus()` — 30 documents, 10 queries, 5 topic areas
- `mini_corpus()` — 5 documents, 2 queries for quick smoke tests

## Prerequisites

```bash
pip install numpy requests pytest

# Start llama.cpp with embedding support
llama-server --model your-model.gguf --embedding --cors * --port 8080
```

## Tests

```bash
cd 4-vector-db-rag/4.1-embedding-norm-ablation
python -m pytest -v
```
