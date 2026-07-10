#!/usr/bin/env python3
"""
End-to-end demo of ``RunnableRouter``.

This script assumes a llama.cpp router server is running at
http://127.0.0.1:8080 with at least two models registered,
e.g. via::

    llama-server \\
        --models-dir /path/to/ggufs \\
        --models-preset presets/coding-chat.ini \\
        --models-max 3 --models-autoload \\
        --sleep-idle-seconds 300 \\
        --host 127.0.0.1 --port 8080

Usage:
    python examples/demo_router.py [--base-url http://...]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

# Add the parent directory so we can import the package directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_llamacpp_router import (
    RunnableRouter,
    RouterModel,
    load_presets_ini,
    NoSuitableModelError,
    ModelColdStartError,
    RouterConnectionError,
)


def build_demo_router(base_url: str = "http://127.0.0.1:8080") -> RunnableRouter:
    """Build a ``RunnableRouter`` with synthetic demo models.

    Uses hardcoded models for illustration.  In production, load from
    a preset INI with ``router.register_presets(...)``.
    """
    router = RunnableRouter(
        base_url=base_url,
        default_routing_key="fast",
        max_cold_start_retries=2,
        cold_start_retry_delay=1.0,
        request_timeout=60.0,
    )

    router.register_model(RouterModel(
        alias="coder",
        tags=["code", "generation", "fast"],
        description="Qwen 2.5 7B — fast code generation",
    ))
    router.register_model(RouterModel(
        alias="chat",
        tags=["chat", "creative", "slow", "deep"],
        description="Llama 3.1 70B — creative writing (slower, heavier)",
    ))
    router.register_model(RouterModel(
        alias="reviewer",
        tags=["code", "review", "audit"],
        description="Code review specialist",
    ))

    return router


def demo_tag_routing(router: RunnableRouter):
    """Demonstrate tag-based routing."""
    print("\n" + "=" * 60)
    print("1. TAG-BASED ROUTING")
    print("=" * 60)

    test_cases = [
        ("code", "Write a Python function that computes fibonacci numbers"),
        ("creative", "Tell me a short story about a robot learning to paint"),
        ("review", "Find the bug in this code:\n\ndef add(a, b):\n    return a + b\n"),
    ]

    for routing_key, prompt in test_cases:
        print(f"\n  ── Routing key: {routing_key!r}")
        print(f"  Prompt: {prompt[:60]}...")
        try:
            response = router.invoke(
                [HumanMessage(content=prompt)],
                config={"routing_key": routing_key},
            )
            print(f"  ✓  Response ({len(response.content)} chars):")
            print(f"     {response.content[:200]}...")
        except NoSuitableModelError as e:
            print(f"  ✗  No model found: {e}")
        except RouterConnectionError as e:
            print(f"  ✗  Connection failed: {e}")
            print("     (Is the llama.cpp router server running?)")
            return
        except ModelColdStartError as e:
            print(f"  ⏳  Cold-start failed: {e}")


def demo_explicit_alias(router: RunnableRouter):
    """Demonstrate explicit model alias routing (bypasses tag lookup)."""
    print("\n" + "=" * 60)
    print("2. EXPLICIT MODEL ALIAS")
    print("=" * 60)

    prompt = "What are the key differences between Rust and Go?"
    print(f"\n  Prompt: {prompt}")

    try:
        response = router.invoke(
            [HumanMessage(content=prompt)],
            config={"model_alias": "coder"},
        )
        print(f"  ✓  Response from 'coder':")
        print(f"     {response.content[:200]}...")
    except RouterConnectionError as e:
        print(f"  ✗  Connection failed: {e}")
    except ModelColdStartError as e:
        print(f"  ⏳  Cold-start failed: {e}")


def demo_fallback_chain(router: RunnableRouter):
    """Demonstrate fallback chain when primary model is unavailable."""
    print("\n" + "=" * 60)
    print("3. FALLBACK CHAIN")
    print("=" * 60)
    print("  (Primary: nonexistent model → falls back to 'coder')")

    prompt = "Explain what a closure is in JavaScript"
    try:
        response = router.invoke(
            [HumanMessage(content=prompt)],
            config={
                "model_alias": "nonexistent-model",
                "fallback_aliases": ["coder", "chat"],
            },
        )
        print(f"  ✓  Response from fallback:")
        print(f"     {response.content[:200]}...")
    except NoSuitableModelError as e:
        print(f"  ✗  All models in chain failed: {e}")
    except RouterConnectionError as e:
        print(f"  ✗  Connection failed: {e}")


def demo_streaming(router: RunnableRouter):
    """Demonstrate streaming output."""
    print("\n" + "=" * 60)
    print("4. STREAMING")
    print("=" * 60)

    prompt = "Count from 1 to 5 with a short description of each number."
    print(f"\n  Prompt: {prompt}")
    print("  Response (streaming):")
    print("  ", end="", flush=True)

    try:
        collected: list[str] = []
        for chunk in router.stream(
            [HumanMessage(content=prompt)],
            config={"routing_key": "fast"},
        ):
            print(chunk.content, end="", flush=True)
            collected.append(chunk.content)
            time.sleep(0.02)  # slow down for visual effect
        print()
        print(f"  ✓  Total chars: {len(''.join(collected))}")
    except RouterConnectionError as e:
        print(f"\n  ✗  Connection failed: {e}")
    except ModelColdStartError as e:
        print(f"\n  ⏳  Cold-start failed: {e}")


def demo_load_from_presets(base_url: str):
    """Demonstrate loading models from an INI preset file."""
    print("\n" + "=" * 60)
    print("5. LOAD FROM PRESET INI")
    print("=" * 60)

    preset_path = Path(__file__).resolve().parent.parent / "presets" / "coding-chat.ini"
    if not preset_path.exists():
        print(f"  ✗  Preset file not found: {preset_path}")
        return

    router = RunnableRouter(base_url=base_url)
    presets = load_presets_ini(preset_path)
    router.register_presets(presets)

    print(f"\n  Loaded {len(presets)} models from {preset_path.name}:")
    for name, preset in presets.items():
        print(f"    • {name:20s} tags={preset.tags}")
    print(f"\n  TIP: Use router.register_presets() to bulk-load models")


def main():
    parser = argparse.ArgumentParser(
        description="RunnableRouter demo — requires a running llama.cpp router server"
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8080",
        help="llama.cpp router server URL (default: http://127.0.0.1:8080)",
    )
    parser.add_argument(
        "--skip-stream",
        action="store_true",
        help="Skip the streaming demo (saves time in CI)",
    )
    args = parser.parse_args()

    router = build_demo_router(args.base_url)

    print(f"🔌  Connecting to router at {args.base_url}")
    print(f"📦  Registered models: {list(router.registered_models)}")

    demo_tag_routing(router)
    demo_explicit_alias(router)
    demo_fallback_chain(router)

    if not args.skip_stream:
        demo_streaming(router)

    demo_load_from_presets(args.base_url)

    router.close()
    print("\n" + "=" * 60)
    print("✨  Demo complete.")


if __name__ == "__main__":
    main()
