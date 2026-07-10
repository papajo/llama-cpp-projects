# 3.5 — Prompt Optimizer Loop

A self-improving prompt optimization engine: define a prompt template
with variables, supply test cases, run the prompt against an LLM,
score the results, and automatically iterate toward a better prompt.

## Quick Start

```python
from prompt_optimizer.prompt import PromptTemplate, TestCase
from prompt_optimizer.evaluator import ExactMatchEvaluator
from prompt_optimizer.optimizer import OptimizationLoop, LlamaClient

# 1. Define a starting prompt
template = PromptTemplate("Translate {{text}} to French. Reply only with the translation.")

# 2. Create test cases (input → expected output)
test_cases = [
    TestCase(input_vars={"text": "Hello"}, expected="Bonjour"),
    TestCase(input_vars={"text": "Goodbye"}, expected="Au revoir"),
    TestCase(input_vars={"text": "Thank you"}, expected="Merci"),
    TestCase(input_vars={"text": "Please"}, expected="S'il vous plaît"),
]

# 3. Evaluate outputs (exact, substring, or LLM-as-judge)
evaluator = ExactMatchEvaluator()

# 4. Point at a running llama.cpp server
llm = LlamaClient(server_url="http://localhost:8080")

# 5. Run the optimization loop
loop = OptimizationLoop(
    template=template,
    test_cases=test_cases,
    evaluator=evaluator,
    llm_client=llm,
    min_score=0.9,
    max_iterations=10,
)
history = loop.run()

# 6. Inspect results
print(f"Best score: {loop.best_score:.2f}")
print(f"Best prompt:\n{loop.best_prompt}")

# 7. Generate a markdown report
from prompt_optimizer.reporter import build_report
report = build_report(history, initial_prompt=template.template)
report.write_markdown("optimization-report.md")
report.write_json("optimization-report.json")
```

## Architecture

```
prompt_optimizer/
├── prompt.py      # PromptTemplate ({{variables}}), TestCase
├── evaluator.py   # Scoring: exact match, contains, rubric (LLM-as-judge)
├── optimizer.py   # LlamaClient, OptimizationLoop, IterationSnapshot
└── reporter.py    # OptimizationReport, build_report, markdown/JSON output
```

## Components

### PromptTemplate (`prompt.py`)

A template with `{{variable}}` placeholders.

```python
pt = PromptTemplate("Summarize: {{text}}")
pt.render({"text": "Long article here..."})
# → "Summarize: Long article here..."
```

- `render(variables)` — substitute values, raises `PromptTemplateError` if variables missing
- `has_variable(name)` — check if a variable exists
- `from_file(path)` — load from a text file
- Extra variables in the dict are silently ignored

### Evaluator (`evaluator.py`)

| Evaluator | Description |
|-----------|-------------|
| `ExactMatchEvaluator` | Score 1.0 if actual == expected (with optional strip) |
| `ContainsEvaluator` | Score 1.0 if expected substring is in actual |
| `RubricEvaluator` | LLM-as-judge: asks the LLM to score 0–10 using a rubric |
| `CompositeEvaluator` | Runs multiple evaluators, combines via min/max/avg |

**RubricEvaluator** sends a judge prompt to the same llama.cpp server,
asking it to score the output on correctness/completeness/clarity.
The judge returns JSON: `{"score": <0-10>, "rationale": "..."}`.

```python
rubric = RubricEvaluator(server_url="http://localhost:8080")
score = rubric(actual_output, expected_output)
```

### Optimization Loop (`optimizer.py`)

```python
loop = OptimizationLoop(
    template=template,          # PromptTemplate
    test_cases=[...],            # list of TestCase
    evaluator=...,               # BaseEvaluator
    llm_client=LlamaClient(...), # LLM for generation + improvement
    min_score=0.9,               # stop when overall ≥ this
    max_iterations=10,           # hard cap
    plateau_window=3,            # stop if no improvement for N iterations
    meta_prompt=DEFAULT_META_PROMPT,  # custom improvement prompt template
    verbose=True,
)
history = loop.run()
```

Each iteration:

1. **Render** the prompt template with each test case's variables
2. **Generate** outputs via the LLM
3. **Evaluate** each output, producing a score per test case
4. **Average** scores → overall score
5. **If below target:** call the meta-prompt to suggest an improved prompt
6. **Check stop conditions:** target reached, max iterations, plateau

#### Meta-Prompt

The meta-prompt receives the current prompt, overall score, and
per-test failures, and suggests an improved prompt. Customize via:

```python
loop.meta_prompt = """
You are optimizing a prompt. Current version:
{current_prompt}
Score: {overall_score:.2f}

Failures:
{test_results}

Provide an improved prompt.
"""
```

Available placeholders: `current_prompt`, `overall_score`, `test_results`.

### Reporter (`reporter.py`)

- `build_report(history, initial_prompt)` — builds an `OptimizationReport`
- `report.to_markdown()` — full markdown report with score trajectory table and per-iteration details
- `report.to_json()` — JSON serialization
- `report.write_markdown(path)` / `report.write_json(path)` — write to files

## Prerequisites

- Python 3.10+
- A running llama.cpp server with an OpenAI-compatible endpoint:

```bash
llama-server --model your-model.gguf --cors * --port 8080
```

## Running Tests

```bash
cd 3-prompt-engineering/3.5-prompt-optimizer-loop
python -m pytest -v
```

## Example: Translation Optimizer

See [Quick Start](#quick-start) above for a complete translation-prompt
optimization workflow that iteratively improves a "Translate to French"
prompt until it correctly handles four test phrases.

## Stop Conditions

| Condition | Trigger |
|-----------|---------|
| ✅ Target reached | `overall_score >= min_score` |
| ⏱ Max iterations | `iteration > max_iterations` |
| 📉 Plateau | No score improvement for `plateau_window` consecutive iterations |
