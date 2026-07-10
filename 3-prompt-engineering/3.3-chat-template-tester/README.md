# Chat Template Tester

Test and compare how different LLM chat templates format the same
conversation. Render message sequences through ~10 popular templates
(Llama 3, ChatML, Mistral, Vicuna, Gemma, Phi-3, DeepSeek, Qwen 2.5,
Command R) and see the raw side-by-side output.

## Problem

Every model family uses a different chat template — special tokens, role
markers, BOS/EOS tokens, and structural rules all differ. A prompt that
works perfectly for one model can produce garbage for another. Without a
tool to visualise the formatted output, it's hard to debug:

- Is the BOS token in the right place?
- Are assistant turns ending with EOS?
- Does the template support system messages?
- What happens with multi-turn conversations?

## Solution

A library of faithful Jinja2 templates (sourced from official Hugging
Face `tokenizer_config.json` files), a formatter that renders message
sequences through any template, and a comparison report generator.

- **10 built-in templates** covering all major model families
- **Real Jinja2 rendering** — same engine llama.cpp uses
- **Side-by-side reports** — summary table, raw output, clean output,
  token estimates, special token counts, structural comparison
- **Custom templates** — register your own for testing
- **No server needed** — pure offline rendering

## Quick Start

```python
from chat_template_tester.tester import run_test
from chat_template_tester.comparator import format_comparison

# Define a conversation
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]

# Test specific templates
run = run_test(messages, template_names=["llama3", "chatml", "mistral"])

# Generate a markdown report
report = format_comparison(run, title="Llama 3 vs ChatML vs Mistral")
print(report)
```

## Available Templates

| Name | BOS | EOS | Style | Used By |
|------|-----|-----|-------|---------|
| `llama3` | `<\|begin_of_text\|>` | `<\|eot_id\|>` | Header-based roles | Meta Llama 3, 3.1, 3.2 |
| `llama2` | `<s>` | `</s>` | `[INST]` / `[/INST]` | Meta Llama 2 |
| `chatml` | — | `<\|im_end\|>` | `<\|im_start\|>role` | Qwen, many fine-tunes |
| `mistral` | `<s>` | `</s>` | `[INST]` (no system) | Mistral 7B, Mixtral |
| `vicuna` | — | `</s>` | `USER:` / `ASSISTANT:` | Vicuna 1.1 |
| `gemma` | `<bos>` | `<eos>` | `<start_of_turn>` | Google Gemma |
| `phi3` | — | `<\|end\|>` | `<\|user\|>` / `<\|assistant\|>` | Microsoft Phi-3 |
| `deepseek` | `｜begin▁of▁sentence｜` | `｜end▁of▁sentence｜` | `User:` / `Assistant:` | DeepSeek V2/V3 |
| `command-r` | `<BOS_TOKEN>` | — | Turn tokens with role tags | Cohere Command R |
| `qwen2.5` | — | `<\|im_end\|>` | ChatML with auto-system | Qwen 2.5 |

## Multi-Turn Example

```python
messages = [
    {"role": "system", "content": "Be concise."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"},
    {"role": "user", "content": "And 3+3?"},
]

run = run_test(messages, template_names=["chatml", "vicuna", "llama2"])
print(format_comparison(run, title="Multi-Turn Comparison"))
```

## Custom Templates

```python
from chat_template_tester.templates import TemplateInfo, registry

registry.register(TemplateInfo(
    name="my-custom",
    template="{% for msg in messages %}[{{ msg.role }}] {{ msg.content }}\n{% endfor %}",
    bos_token=None,
    eos_token="\n",
    description="Simple bracket-style template",
))

run = run_test(messages, custom_templates=[registry.get("my-custom")])
```

Or pass one-off templates directly:

```python
from chat_template_tester.tester import run_test
from chat_template_tester.templates import TemplateInfo

custom = TemplateInfo(
    name="inline",
    template="Q: {{ messages[-1]['content'] }}\nA:",
)
run = run_test(messages, custom_templates=[custom])
```

## Project Structure

```
3.3-chat-template-tester/
├── chat_template_tester/
│   ├── __init__.py
│   ├── templates.py     # TemplateInfo, TemplateRegistry, 10 built-in templates
│   ├── formatter.py     # ChatFormatter — Jinja2 rendering, estimate_tokens
│   ├── tester.py        # run_test(), TemplateTestResult, TestRun
│   └── comparator.py    # format_comparison() — markdown reports
├── tests/
│   ├── test_templates.py    # 15 tests (registry, builtins, structure)
│   ├── test_formatter.py    # 22 tests (ChatFormatter, estimates, special tokens)
│   ├── test_tester.py       # 15 tests (run_test, results, custom templates)
│   └── test_comparator.py   # 15 tests (report format, sections, multi-turn)
├── pyproject.toml
├── README.md
└── todo-plan.md
```

## API Reference

### TemplateRegistry
- `register(info)` — add a template
- `get(name)` — look up by name (raises `KeyError` if missing)
- `list()` — all templates sorted by name
- `name in registry` — membership check

### ChatFormatter
- `format(template, messages, add_generation_prompt=False)` — render
- `format_all(templates, messages, ...)` — render through multiple templates; errors are returned as strings prefixed with `<ERROR:`

### run_test(messages, ...)
- `template_names` — subset to test (`None` = all builtins)
- `add_generation_prompt` — append assistant turn starter (default: `True`)
- `custom_templates` — additional `TemplateInfo` objects
- Returns `TestRun` with `.results`, `.ok_results()`, `.error_results()`

### format_comparison(run, ...)
- `show_special` / `show_raw` — toggle detail sections
- `notes` — optional list of notes appended to the report
- Returns a markdown string

## Requirements

- Python 3.11+
- `jinja2>=3.1` (installed automatically)

## Running Tests

```bash
pip install -e ".[test]"
pytest -v
```

All 68 tests passing.
