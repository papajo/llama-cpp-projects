# 3.3 — Chat Template Tester: Todo & Plan

## Goal

Build a tool to test and compare how different chat templates format the
same conversation. Provides a registry of ~10 popular templates (Llama 3,
ChatML, Mistral, Vicuna, Gemma, Phi-3, DeepSeek, Qwen 2.5, Command R),
renders message sequences through each, and produces side-by-side
comparisons showing raw output, token estimates, and structural
differences.

## Module Architecture

```
3.3-chat-template-tester/
├── chat_template_tester/
│   ├── __init__.py
│   ├── templates.py     # TemplateRegistry, TemplateInfo, built-in templates
│   ├── formatter.py     # ChatFormatter — Jinja2 rendering with special tokens
│   ├── tester.py        # run_test — test messages across selected templates
│   └── comparator.py    # format_comparison — side-by-side markdown output
├── tests/
│   ├── test_templates.py
│   ├── test_formatter.py
│   ├── test_tester.py
│   └── test_comparator.py
├── pyproject.toml
├── README.md
└── todo-plan.md
```

## Tasks

| # | Task | Status |
|---|------|--------|
| 1 | Scaffold project (pyproject.toml, dirs, \_\_init\_\_.py) | ✅ |
| 2 | `templates.py` — TemplateInfo, TemplateRegistry, ~10 built-in templates | |
| 3 | `formatter.py` — ChatFormatter, Jinja2 env with custom globals (raise_exception), special token injection | |
| 4 | `tester.py` — run_test, collect formatted outputs | |
| 5 | `comparator.py` — side-by-side report with raw output, char count, est. tokens | |
| 6 | Tests for all 4 modules | |
| 7 | `pytest -v` — all tests passing | |
| 8 | README.md with quick-start, API ref, template list | |

## Key Design Decisions

- Use actual `jinja2` for rendering (faithful to llama.cpp's behavior)
- `raise_exception` added as Jinja2 global so Llama 2 / Mistral / Vicuna templates don't crash
- Templates stored as `TemplateInfo(name, template_str, bos, eos, description)` in a registry
- Formatter passes `messages`, `bos_token`, `eos_token`, `add_generation_prompt` to the template
- No tokenizer dependency — token estimates use ~4 chars/token heuristic
- Templates sourced from official model repos (huggingface tokenizer_config.json)
