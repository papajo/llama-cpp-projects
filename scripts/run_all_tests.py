#!/usr/bin/env python3
"""Run every project's test suite and summarise the results.

Each project ships its own pyproject.toml, so pytest treats each one as its own
rootdir. This runs pytest once per project directory and aggregates.

Modes:
    --mode offline   (default) run only the in-process/mocked unit tests
    --mode live      run only tests marked `live`, against the real servers
    --mode both      run both passes

`source env.sh` first so PYTHONPATH and PYTEST_PLUGINS are set.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _python() -> Path:
    """The shared venv interpreter; worktrees have no .venv of their own."""
    env = os.environ.get("LLAMA_VENV")
    if env and (Path(env) / "bin" / "python").exists():
        return Path(env) / "bin" / "python"
    local = REPO / ".venv" / "bin" / "python"
    if local.exists():
        return local
    return Path(sys.executable)


PYTHON = _python()

CATEGORIES = {
    "core": ["1-ai-fundamentals", "3-prompt-engineering"],
    "rag": ["4-vector-db-rag", "2-langchain"],
    "graph": ["5-langgraph", "6-mcp"],
    "bonus": ["7-bonus"],
}


def find_projects(categories: list[str] | None) -> list[Path]:
    wanted: set[str] = set()
    if categories:
        for c in categories:
            wanted.update(CATEGORIES.get(c, [c]))
    projects = []
    for top in sorted(REPO.iterdir()):
        if not top.is_dir() or top.name.startswith((".", "_")):
            continue
        if wanted and top.name not in wanted:
            continue
        if top.name in {"scripts", "tools", "config", "tests"}:
            continue
        for proj in sorted(top.iterdir()):
            if proj.is_dir() and (proj / "tests").is_dir():
                projects.append(proj)
    return projects


def run_one(proj: Path, mode: str) -> dict:
    env = dict(os.environ)
    args = [str(PYTHON), "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider"]
    if mode == "live":
        env["LLM_LIVE"] = "1"
        args += ["-m", "live"]
    else:
        env.pop("LLM_LIVE", None)
        args += ["-m", "not live"]
    proc = subprocess.run(
        args, cwd=proj, env=env, capture_output=True, text=True, timeout=1800
    )
    tail = (proc.stdout or proc.stderr).strip().splitlines()
    return {
        "project": str(proj.relative_to(REPO)),
        "mode": mode,
        "returncode": proc.returncode,
        "summary": tail[-1] if tail else "",
        "output": proc.stdout + proc.stderr,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["offline", "live", "both"], default="offline")
    ap.add_argument("--category", action="append", help="core|rag|graph|bonus or a dir")
    ap.add_argument("--json", type=Path, help="write full results here")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    modes = ["offline", "live"] if args.mode == "both" else [args.mode]
    projects = find_projects(args.category)
    if not projects:
        print("no projects with a tests/ directory matched", file=sys.stderr)
        return 2

    results = []
    failed = 0
    for mode in modes:
        print(f"\n=== {mode.upper()} ===")
        for proj in projects:
            r = run_one(proj, mode)
            results.append(r)
            status = "PASS" if r["returncode"] == 0 else "FAIL"
            if r["returncode"] == 5:  # no tests collected
                status = "NONE"
            if status == "FAIL":
                failed += 1
            print(f"  [{status}] {r['project']}: {r['summary']}")
            if args.verbose and status == "FAIL":
                print(r["output"])

    if args.json:
        args.json.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.json}")

    print(f"\n{len(results)} runs, {failed} failing")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
