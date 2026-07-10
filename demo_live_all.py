#!/usr/bin/env python3
"""Top-level live demo — runs every project with real LLM inference.

Usage:
    python3 demo_live_all.py              # Full demo
    python3 demo_live_all.py --quick       # One example per category
    python3 demo_live_all.py --category 2  # Only Category 2 (LangChain)
"""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

# Ensure _shared is importable
_PROJECTS_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(_PROJECTS_ROOT))

from _shared import live_projects_summary, get_client


def banner(title: str) -> None:
    sep = "─" * 60
    print(f"\n{sep}\n  {title}\n{sep}")


def run_module(module_path: str, category_name: str) -> None:
    """Import and run a demo_live module."""
    try:
        mod = importlib.import_module(module_path)
        if hasattr(mod, "demo"):
            banner(f"{category_name} — {mod.__name__}")
            t0 = time.time()
            mod.demo()
            print(f"  ⏱  {time.time() - t0:.1f}s")
    except ImportError as e:
        print(f"  ⚠️  {module_path}: {e}")
    except Exception as e:
        print(f"  ❌  {module_path}: {e}")


def main():
    quick = "--quick" in sys.argv
    cat_filter = None
    for arg in sys.argv[1:]:
        if arg.startswith("--category="):
            cat_filter = arg.split("=", 1)[1]

    print(live_projects_summary())
    client = get_client(prefer="ollama", auto_start=False)
    if not client.connected:
        print("\n⚠️  No LLM server detected. Start one with: ollama serve")
        print("Demos will use mock data.\n")
        if not quick:
            input("Press Enter to continue with mock data...")

    categories = [
        ("1-ai-fundamentals", "1-ai-fundamentals"),
        ("2-langchain", "2-langchain"),
        ("3-prompt-engineering", "3-prompt-engineering"),
        ("4-vector-db-rag", "4-vector-db-rag"),
        ("5-langgraph", "5-langgraph"),
        ("6-mcp", "6-mcp"),
        ("7-bonus", "7-bonus"),
    ]

    for cat_dir, cat_module in categories:
        if cat_filter and cat_dir != cat_filter:
            continue
        if quick:
            run_module(f"{cat_module}.demo_live", cat_dir)
        else:
            # Try each sub-project's demo_live
            projects_dir = _PROJECTS_ROOT / cat_dir
            if not projects_dir.is_dir():
                continue
            for proj_dir in sorted(projects_dir.iterdir()):
                if not proj_dir.is_dir() or proj_dir.name.startswith("."):
                    continue
                demo_file = proj_dir / "demo_live.py"
                if demo_file.exists():
                    mod_path = f"{cat_module}.{proj_dir.name}.demo_live"
                    run_module(mod_path, f"{cat_dir}/{proj_dir.name}")

    print("\n✅ Demo complete!")


if __name__ == "__main__":
    main()
