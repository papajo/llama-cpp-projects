"""
Prompt corpora for speculative decoding benchmarks.

Three categories representing different workload types:
  - code/    — code generation and completion tasks
  - prose/   — natural language generation (chat, summarisation, creative)
  - logs/    — repetitive / log-line generation (structured, templated)

Each category exposes a list of (name, prompt) tuples.
"""

from .code import CODE_PROMPTS
from .prose import PROSE_PROMPTS
from .logs import LOG_PROMPTS

# Label-to-prompt mapping for easy access
PROMPT_CATEGORIES = {
    "code": CODE_PROMPTS,
    "prose": PROSE_PROMPTS,
    "logs": LOG_PROMPTS,
}

ALL_PROMPTS = CODE_PROMPTS + PROSE_PROMPTS + LOG_PROMPTS


def get_category(name: str):
    """Get prompts for a specific category."""
    if name in PROMPT_CATEGORIES:
        return PROMPT_CATEGORIES[name]
    raise KeyError(f"Unknown category: {name}.  Options: {list(PROMPT_CATEGORIES.keys())}")


def list_categories():
    return list(PROMPT_CATEGORIES.keys())
