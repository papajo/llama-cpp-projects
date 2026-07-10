# 4.2 — Local Re-Rank RAG

A retrieve-then-rerank RAG pipeline: fast bi-encoder retrieval via
llama.cpp embeddings, followed by LLM-as-judge re-ranking for higher
accuracy. Includes a keyword-overlap baseline for comparison.

## Quick Start

```python
from local_rerank_rag.data import default_corpus
from local_rerank_rag.retriever import EmbeddingClient, Retriever, VectorStore
from local_rerank_rag.reranker import LlamaClient, Reranker
from local_rerank_rag.pipeline import evaluate_pipeline
from local_rerank_rag.reporting import evaluation_to_markdown

# 1. Build vector store from the built-in corpus
corpus = default_corpus()
client = EmbeddingClient("http://localhost:8080")

store = VectorStore()
doc_vecs = client.embed_many(corpus.documents)
store.add_many(corpus.documents, doc_vecs)

# 2. Create retriever and reranker
retriever = Retriever(store=store, client=client)
llm = LlamaClient("http://localhost:8080")
reranker = Reranker(client=llm)

# 3. Evaluate pipeline (retrieve 20, rerank top 5)
ev = evaluate_pipeline(
    retriever, reranker, corpus,
    retrieve_k=20, final_k=5,
    include_baseline=True,  # compare against keyword overlap
)
print(evaluation_to_markdown(ev))
```

## Architecture

```
local_rerank_rag/
├── data.py           # 25-doc / 10-query corpus with relevance judgments
├── retriever.py      # EmbeddingClient, VectorStore, Retriever, cosine sim
├── reranker.py       # LlamaClient, Reranker (LLM-as-judge), keyword overlap
├── pipeline.py       # run_pipeline, evaluate_pipeline
├── retrieval_metrics # MRR, recall@k
└── reporting.py      # Markdown/JSON evaluation reports
```

## Pipeline Flow

```
Query → Embed query → VectorStore search (top-k1)
    → LLM-as-judge relevance score for each candidate
    → Sort by relevance score → Return top-k2
```

## Reranker: LLM-as-Judge

The `Reranker` sends each (query, document) pair to llama.cpp with a
structured prompt asking for a relevance score (0–10) and rationale.
The default system prompt:

> *Evaluate how relevant a document is to a given query. Return JSON:
> `{"score": <0-10>, "rationale": "..."}`*

The reranker parses the LLM output, trying:
1. JSON parse (with or without `` ```json `` code fences)
2. Regex fallback (extract any integer 0–10)
3. Default to 0 if no number found

## Baseline: Keyword Overlap

`keyword_overlap_score(query, document)` is a simple token-overlap metric
that requires no LLM calls. Useful as a cheap baseline — pass
``include_baseline=True`` to ``evaluate_pipeline`` to compare.

## Metrics

| Metric | Description |
|--------|-------------|
| MRR | Mean Reciprocal Rank — how high the first relevant doc ranks |
| Recall@k | Fraction of relevant docs found in the top-k |

## Prerequisites

```bash
pip install numpy requests pytest

# Start llama.cpp with both embedding and chat endpoints
llama-server --model your-model.gguf --embedding --cors * --port 8080
```

## Tests

```bash
cd 4-vector-db-rag/4.2-local-rerank-rag
python -m pytest -v
```
