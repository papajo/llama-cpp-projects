"""
GBNF grammar utilities — parsing, validation, and explanation.

Provides tools to:
  - Parse GBNF grammar rules
  - Validate GBNF syntax
  - Explain what a grammar rule does in plain English
  - Highlight which token positions are constrained
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── Data models ───────────────────────────────────────────────────

@dataclass
class GrammarRule:
    """A single GBNF grammar rule."""
    name: str                    # rule name (e.g., "root", "array")
    definition: str              # raw definition body
    is_root: bool = False

    def summary(self) -> str:
        """Short human-readable description."""
        if self.is_root:
            return f"root ::= {self.definition[:60]}..."
        return f"{self.name} ::= {self.definition[:60]}..."


@dataclass
class GrammarMaskStep:
    """What the grammar allowed/masked at a single token position."""
    token_index: int
    token_text: str
    token_id: int
    logprob: float
    was_accepted: bool          # True = passed grammar filter
    masked_by_grammar: bool     # True = grammar rejected this token
    num_candidates: int         # total candidates at this step
    num_allowed: int            # candidates that passed grammar filter

    def as_dict(self) -> dict:
        return {
            "index": self.token_index,
            "text": self.token_text,
            "id": self.token_id,
            "logprob": self.logprob,
            "was_accepted": self.was_accepted,
            "masked_by_grammar": self.masked_by_grammar,
            "num_candidates": self.num_candidates,
            "num_allowed": self.num_allowed,
        }


@dataclass
class TokenTraceResult:
    """
    Full token-by-token trace of a constrained generation.
    
    Includes constrained and (optionally) unconstrained traces
    for comparison.
    """
    constrained_steps: List[GrammarMaskStep] = field(default_factory=list)
    unconstrained_steps: List[GrammarMaskStep] = field(default_factory=list)
    grammar_used: str = ""
    json_schema_used: str = ""
    full_text: str = ""
    total_tokens: int = 0
    tokens_masked: int = 0
    mask_rate: float = 0.0

    def as_dict(self) -> dict:
        return {
            "grammar": self.grammar_used,
            "json_schema": self.json_schema_used,
            "total_tokens": self.total_tokens,
            "tokens_masked": self.tokens_masked,
            "mask_rate": self.mask_rate,
            "full_text": self.full_text,
            "constrained": [s.as_dict() for s in self.constrained_steps],
            "unconstrained": [s.as_dict() for s in self.unconstrained_steps],
        }


# ── Grammar parser ────────────────────────────────────────────────

class GBNFParser:
    """Parse GBNF grammar text into structured rules."""
    
    @staticmethod
    def parse(grammar_text: str) -> List[GrammarRule]:
        """Parse raw GBNF text into a list of GrammarRule objects."""
        rules = []
        lines = grammar_text.split("\n")
        current_name = None
        current_def = []
        is_root = False

        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            # Check for rule definition: name ::= body
            match = re.match(r'^(\w+)\s*::=\s*(.*)', stripped)
            if match:
                # Save previous rule
                if current_name is not None:
                    rules.append(GrammarRule(
                        name=current_name,
                        definition=" ".join(current_def),
                        is_root=is_root,
                    ))
                current_name = match.group(1)
                current_def = [match.group(2)]
                is_root = (current_name == "root")
            elif current_name is not None:
                current_def.append(stripped)

        if current_name is not None:
            rules.append(GrammarRule(
                name=current_name,
                definition=" ".join(current_def),
                is_root=is_root,
            ))

        return rules

    @staticmethod
    def validate(grammar_text: str) -> List[str]:
        """
        Validate GBNF syntax. Returns a list of error messages.
        Empty list means valid.
        """
        errors = []
        if not grammar_text.strip():
            return ["Grammar is empty"]

        rules = GBNFParser.parse(grammar_text)
        if not rules:
            return ["No valid rules found"]

        # Check for root rule
        if not any(r.is_root for r in rules):
            errors.append("No 'root' rule defined (required)")

        # Check for common issues. A GBNF definition has three lexical
        # contexts - bare, inside a "string literal", and inside a
        # [character class] - and a delimiter only counts in the bare one.
        for rule in rules:
            depth_paren = depth_bracket = 0
            in_quote = in_class = escaped = False
            stray_close = False

            for ch in rule.definition:
                if escaped:
                    escaped = False
                    continue
                if ch == "\\":
                    escaped = True
                    continue

                if in_quote:
                    if ch == '"':
                        in_quote = False
                    continue
                if in_class:
                    if ch == "]":
                        in_class = False
                        depth_bracket -= 1
                    continue

                if ch == '"':
                    in_quote = True
                elif ch == "[":
                    in_class = True
                    depth_bracket += 1
                elif ch == "]":
                    stray_close = True
                elif ch == "(":
                    depth_paren += 1
                elif ch == ")":
                    depth_paren -= 1
                    if depth_paren < 0:
                        stray_close = True

            if in_quote:
                errors.append(
                    f"Rule '{rule.name}': unterminated string literal"
                )
            if depth_paren != 0 or stray_close:
                errors.append(
                    f"Rule '{rule.name}': mismatched parentheses"
                )
            if in_class or depth_bracket != 0:
                errors.append(
                    f"Rule '{rule.name}': unterminated character class"
                )

        return errors


# ── Grammar debugger ──────────────────────────────────────────────

class GrammarDebugger:
    """
    Analyse a constrained generation to understand grammar effects.
    
    Given the /completion response with n_probs, determine which
    candidate tokens were masked by the grammar at each step.
    """

    @staticmethod
    def analyse_completion(
        completion_data: dict,
        grammar_text: str = "",
        json_schema: str = "",
    ) -> TokenTraceResult:
        """
        Analyse a /completion response with n_probs for grammar effects.
        
        The completion_data should include:
          - content: the generated text
          - timings: timing info
          - n_probs (if requested): list of per-token probability data
        
        Each n_probs entry as llama-server sends it:
          { "id": token_id, "token": token_text, "bytes": [int, ...],
            "logprob": float,
            "top_logprobs": [{"id":..., "token":..., "bytes":...,
                              "logprob":...}, ...] }
        """
        content = completion_data.get("content", "")
        n_probs = completion_data.get("completion_probabilities", [])
        
        # For constrained decoding, we estimate which tokens were masked
        # by comparing the actual chosen token's logprob rank against
        # the full candidate list.
        constrained_steps: List[GrammarMaskStep] = []
        total_candidates = 0
        total_allowed = 0
        tokens_masked = 0

        for i, prob_entry in enumerate(n_probs):
            if isinstance(prob_entry, dict):
                top = prob_entry.get("top_logprobs", [])
                # llama-server names this "token" (matching the OpenAI
                # logprobs schema); "text" is only for older payloads.
                chosen_token = prob_entry.get("token", prob_entry.get("text", ""))
                chosen_id = prob_entry.get("id", -1)
                chosen_logprob = prob_entry.get("logprob", 0.0)
            else:
                continue

            # Estimate: if the chosen token has very low probability
            # compared to the top candidate, it may have been forced
            # by grammar masking.
            sorted_candidates = sorted(
                top, key=lambda x: x.get("logprob", -1e9), reverse=True
            ) if top else []

            num_candidates = len(sorted_candidates)
            num_allowed = num_candidates  # We can't know exactly without server internals
            was_masked = False

            # Heuristic: if the chosen token is not in the top 3 by logprob,
            # it was likely forced by grammar constraints
            if sorted_candidates:
                chosen_rank = next(
                    (j for j, c in enumerate(sorted_candidates)
                     if c.get("id") == chosen_id),
                    -1
                )
                if chosen_rank >= 3:
                    was_masked = True
                    tokens_masked += 1

            step = GrammarMaskStep(
                token_index=i,
                token_text=chosen_token,
                token_id=chosen_id,
                logprob=chosen_logprob,
                was_accepted=True,
                masked_by_grammar=was_masked,
                num_candidates=num_candidates,
                num_allowed=num_allowed,
            )
            constrained_steps.append(step)
            total_candidates += num_candidates
            total_allowed += num_allowed

        total_tokens = len(constrained_steps)
        mask_rate = tokens_masked / total_tokens if total_tokens > 0 else 0.0

        return TokenTraceResult(
            constrained_steps=constrained_steps,
            grammar_used=grammar_text,
            json_schema_used=json_schema,
            full_text=content,
            total_tokens=total_tokens,
            tokens_masked=tokens_masked,
            mask_rate=mask_rate,
        )
