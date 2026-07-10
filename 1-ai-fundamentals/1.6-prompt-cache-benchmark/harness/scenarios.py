"""
Scenario templates for cache benchmarking.

Each scenario defines a set of prompt templates and a cache-access
pattern to simulate different real-world usage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class CacheScenario:
    """A benchmark scenario for testing prompt cache behaviour."""
    name: str
    description: str
    prompts: List[str]          # List of prompts to send sequentially
    shared_prefix: str = ""     # Common prefix shared by all prompts
    warmup: int = 1             # Warmup runs before measurement
    cache_type: str = "hybrid"  # Cache type to test
    expected_speedup: float = 0.0  # Expected speedup (filled by benchmark)


# ── Scenario generators ───────────────────────────────────────────

def _prefix(prefix: str, bodies: List[str]) -> List[str]:
    """Prepend a shared prefix to each body."""
    return [f"{prefix}{body}" for body in bodies]


def multi_turn_conversation(num_turns: int = 5) -> CacheScenario:
    """Simulate a multi-turn conversation where each turn includes prior context."""
    prefix = "<|begin_of_text|>"
    messages = [
        "What is the capital of France?",
        "Tell me more about its history.",
        "What are the best museums to visit?",
        "What about the food scene?",
        "How do I get there from London?",
        "What is the weather like in summer?",
        "Are there any day trips from the city?",
        "What is the nightlife like?",
        "What is the best time to visit?",
        "What are some nearby cities to explore?",
    ][:num_turns]

    # Build each prompt with accumulated context
    prompts = []
    context = ""
    for msg in messages:
        context += f"<|user|>\n{msg}\n<|assistant|>\n"
        prompts.append(f"{prefix}{context}")

    return CacheScenario(
        name=f"multi-turn-{num_turns}turns",
        description=f"Multi-turn conversation ({num_turns} turns), each including full history",
        prompts=prompts,
        shared_prefix=prefix,
        cache_type="hybrid",
    )


def repeated_prefix(template: str = "story", num_variations: int = 5) -> CacheScenario:
    """Same system prompt with different user inputs."""
    system = "You are a helpful assistant. Please respond concisely and accurately."
    questions = [
        "What is machine learning?",
        "Explain neural networks.",
        "What is natural language processing?",
        "Describe reinforcement learning.",
        "What is computer vision?",
        "How do transformers work?",
        "Explain backpropagation.",
        "What is a tensor?",
        "Describe supervised learning.",
        "What is unsupervised learning?",
    ][:num_variations]

    prompts = [
        f"<|begin_of_text|>\n<|system|>\n{system}\n<|user|>\n{q}\n<|assistant|>\n"
        for q in questions
    ]

    return CacheScenario(
        name=f"repeated-prefix-{num_variations}q",
        description=f"Shared system prompt with {num_variations} different queries",
        prompts=prompts,
        shared_prefix=f"<|begin_of_text|>\n<|system|>\n{system}",
        cache_type="hybrid",
    )


def long_prefix_short_query(num_pairs: int = 5) -> CacheScenario:
    """Long shared document prefix with short queries."""
    # Simulate a long document
    doc = " ".join(["The quick brown fox jumps over the lazy dog." for _ in range(50)])
    prefix = f"<|begin_of_text|>\nDocument: {doc}\n\n"

    queries = [
        "What animals are in the document?",
        "What color is the fox?",
        "What is the fox doing?",
        "Is the dog active or passive?",
        "How many animals are mentioned?",
        "What is the setting of this text?",
        "Write a summary of the document.",
        "What is the main theme?",
        "Count the adjectives used.",
        "Translate to French.",
    ][:num_pairs]

    prompts = [f"{prefix}Question: {q}\nAnswer:" for q in queries]

    return CacheScenario(
        name=f"long-prefix-{num_pairs}q",
        description=f"Long shared document ({len(doc)} chars) with {num_pairs} queries",
        prompts=prompts,
        shared_prefix=prefix,
        cache_type="hybrid",
    )


def batch_processing(batch_size: int = 4, num_batches: int = 3) -> CacheScenario:
    """Simulate batch processing of similar prompts."""
    templates = [
        "Classify this text: '{}'",
        "Translate to French: '{}'",
        "Summarize: '{}'",
    ]
    texts = [
        "The movie was fantastic and well-directed.",
        "Artificial intelligence is transforming healthcare.",
        "The stock market reached new heights today.",
        "A new study shows the benefits of meditation.",
        "The weather is beautiful in California.",
        "Python is a versatile programming language.",
    ]

    prompts = []
    for b in range(num_batches):
        for t in templates:
            for txt in texts[:batch_size]:
                prompts.append(
                    f"<|begin_of_text|>\n<|user|>\n{t.format(txt)}\n<|assistant|>\n"
                )

    return CacheScenario(
        name=f"batch-{batch_size}x{num_batches}",
        description=f"Batch processing ({batch_size} items × {num_batches} batches)",
        prompts=prompts,
        shared_prefix="<|begin_of_text|>",
        cache_type="hybrid",
    )


def code_edit_loop(num_edits: int = 5) -> CacheScenario:
    """Simulate code editing where the full file is always present."""
    file_header = "# File: calculator.py\n# A simple calculator implementation\n\n"
    code = (
        "def add(a, b): return a + b\n"
        "def sub(a, b): return a - b\n"
        "def mul(a, b): return a * b\n"
    )

    edits = [
        "Add a divide function",
        "Add error handling for division by zero",
        "Add a power function",
        "Add type hints to all functions",
        "Add a main function that runs the calculator REPL",
        "Add unit tests for all functions",
        "Add a square root function",
        "Add logging to all functions",
    ][:num_edits]

    prefix = f"<|begin_of_text|>\n{file_header}{code}\n\n"
    prompts = [f"{prefix}// Edit: {e}\ncode:" for e in edits]

    return CacheScenario(
        name=f"code-edit-{num_edits}",
        description=f"Code editing loop ({num_edits} edits) with full file context",
        prompts=prompts,
        shared_prefix=prefix,
        cache_type="hybrid",
    )


# ── All scenarios ─────────────────────────────────────────────────

def list_scenarios() -> List[Dict]:
    """Return metadata about all available scenarios."""
    scenarios = [
        multi_turn_conversation(3),
        multi_turn_conversation(5),
        multi_turn_conversation(10),
        repeated_prefix(num_variations=3),
        repeated_prefix(num_variations=5),
        repeated_prefix(num_variations=10),
        long_prefix_short_query(3),
        long_prefix_short_query(5),
        code_edit_loop(3),
        code_edit_loop(5),
        batch_processing(2, 2),
        batch_processing(4, 2),
    ]
    return [
        {
            "name": s.name,
            "description": s.description,
            "num_prompts": len(s.prompts),
            "avg_prompt_len": sum(len(p) for p in s.prompts) // max(len(s.prompts), 1),
            "cache_type": s.cache_type,
        }
        for s in scenarios
    ]


def get_scenario(name: str) -> Optional[CacheScenario]:
    """Get a scenario by name."""
    registry = {
        s.name: s
        for s in [
            multi_turn_conversation(3),
            multi_turn_conversation(5),
            multi_turn_conversation(10),
            repeated_prefix(num_variations=3),
            repeated_prefix(num_variations=5),
            repeated_prefix(num_variations=10),
            long_prefix_short_query(3),
            long_prefix_short_query(5),
            code_edit_loop(3),
            code_edit_loop(5),
            batch_processing(2, 2),
            batch_processing(4, 2),
        ]
    }
    return registry.get(name)
