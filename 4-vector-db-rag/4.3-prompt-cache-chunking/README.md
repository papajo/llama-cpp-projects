# 4.3 — Prompt Cache Chunking

Compare how different document chunking strategies affect KV-cache
efficiency in a RAG prompt pipeline. Models the prompt cache as a
prefix-reuse cache and measures hit ratios across strategies.

## Quick Start

```python
from prompt_cache_chunking.data import default_corpus
from prompt_cache_chunking.experiment import run_experiment
from prompt_cache_chunking.reporting import experiment_to_markdown

# Run the full comparison across all default chunkers
corpus = default_corpus()
result = run_experiment(corpus)
print(experiment_to_markdown(result))

# See which strategy wins on cache hit ratio
summary = result.summary()
for name, metrics in summary.items():
    print(f"{name}: hit ratio={metrics['cache_hit_ratio']:.2%}, "
          f"{metrics['chunk_count']} chunks")
```

## Architecture

```
prompt_cache_chunking/
├── data.py         # 10 paragraph-length documents, 10 queries
├── chunkers.py     # 4 chunking strategies (6 variants)
├── cache_sim.py    # KV-cache prefix-reuse simulation
├── experiment.py   # Run multiple chunkers and compare
└── reporting.py    # Markdown/JSON reports
```

## Chunking Strategies

| Strategy | Description | Variants |
|---|---|---|
| `FixedSizeChunker` | Split by character count with configurable overlap | `200/20`, `100/10` |
| `SentenceChunker` | Group sentences into chunks with overlap | `3/1`, `2/0` |
| `ParagraphChunker` | Split at natural paragraph breaks | single |
| `RecursiveChunker` | Recursively split on sentence boundaries | `150 chars` |

## Cache Model

The simulator models a simple prefix cache:

1. Each RAG prompt = `system prompt + query + chunk_1 + chunk_2 + ...`
2. When the same `system_prompt|query` prefix appears across prompts,
   the tokens are served from cache (no recomputation)
3. When individual chunks are reused across queries, their prefix is
   also cached
4. **Cache hit ratio** = `cached_tokens / total_tokens`

Strategies that produce more reusable prefixes (fewer unique chunks,
more overlap between queries) score higher cache hit ratios.

## Custom Experiment

```python
from prompt_cache_chunking.chunkers import FixedSizeChunker, RecursiveChunker

result = run_experiment(
    corpus,
    chunkers=[
        FixedSizeChunker(chunk_size=300, overlap=30),
        RecursiveChunker(max_chars=200),
    ],
    top_k=5,
)
```

## Metrics

| Metric | Description |
|--------|-------------|
| Cache hit ratio | Fraction of tokens served from cache |
| Chunk count | Total chunks produced |
| Prompts per chunk (avg) | How many prompts reference each chunk |

## Tests

```bash
cd 4-vector-db-rag/4.3-prompt-cache-chunking
python -m pytest -v
```
