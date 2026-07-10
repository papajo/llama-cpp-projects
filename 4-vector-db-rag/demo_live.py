"""Live demo for Category 4 — Vector DB / RAG projects with real LLM."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _shared.integrations import RAGAdapter


def demo_embedding_norm():
    adapter = RAGAdapter()
    tag = "✅ LIVE" if adapter.connected else "⏸️ MOCK"
    print(f"\n  4.1 Embedding Norm Ablation [{tag}]")
    texts = [
        "The capital of France is Paris.",
        "Paris is the capital of France.",
        "Pizza is a popular Italian dish.",
    ]
    embs = adapter.embed(texts)
    for t, e in zip(texts, embs):
        norm = sum(v * v for v in e) ** 0.5
        print(f"     norm={norm:.3f}  |  {t}")


def demo_local_rerank_rag():
    adapter = RAGAdapter()
    tag = "✅ LIVE" if adapter.connected else "⏸️ MOCK"
    print(f"\n  4.2 Local Rerank RAG [{tag}]")
    context = [
        "Python is a high-level, interpreted programming language.",
        "Guido van Rossum created Python and released it in 1991.",
        "Python emphasizes code readability with significant indentation.",
    ]
    answer = adapter.generate("Who created Python?", context)
    print(f"     Q: Who created Python?")
    print(f"     A: {answer}")


def demo_prompt_cache_chunking():
    adapter = RAGAdapter()
    tag = "✅ LIVE" if adapter.connected else "⏸️ MOCK"
    print(f"\n  4.3 Prompt Cache Chunking [{tag}]")
    context = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with many layers.",
        "Transformers are a type of neural network architecture.",
    ]
    answer = adapter.generate("What is deep learning?", context)
    print(f"     Q: What is deep learning?")
    print(f"     A: {answer}")


def demo_multi_tenant_rag():
    adapter = RAGAdapter()
    tag = "✅ LIVE" if adapter.connected else "⏸️ MOCK"
    print(f"\n  4.4 Multi-Tenant RAG [{tag}]")
    emb = adapter.embed_query("tenant-specific document about sales data")
    print(f"     Embedding dims: {len(emb)}")
    context = ["Tenant A: quarterly revenue increased 15%."]
    answer = adapter.generate("What is the revenue trend?", context)
    print(f"     Q: What is the revenue trend?")
    print(f"     A: {answer}")


def demo():
    demo_embedding_norm()
    demo_local_rerank_rag()
    demo_prompt_cache_chunking()
    demo_multi_tenant_rag()


if __name__ == "__main__":
    demo()
