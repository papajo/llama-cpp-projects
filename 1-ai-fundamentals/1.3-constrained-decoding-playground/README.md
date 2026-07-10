# Constrained Decoding Playground

Visualise how JSON Schema and GBNF grammar constraints affect **token-by-token** generation in llama.cpp.

## What it does

- **JSON Schema → GBNF conversion**: Paste any JSON Schema and see the auto-generated GBNF grammar
- **Token-level trace**: Generate with `n_probs` and see which candidate tokens were **accepted** vs **masked** by the grammar at each step
- **Streaming proxy**: Frontend streams tokens live from `llama-server` through the backend
- **Grammar debugger**: Validate GBNF syntax, parse rules, inspect definitions
- **Sample schemas**: Pre-built schemas (weather, extraction, function-calling, email) to experiment with

## Project structure

```
1.3-constrained-decoding-playground/
├── backend/
│   ├── __init__.py
│   ├── server.py              # FastAPI proxy (completion, trace, convert-schema, etc.)
│   ├── grammar.py             # GBNF parser, validator, debugger
│   └── schema_converter.py    # JSON Schema → GBNF converter
├── frontend/
│   ├── src/
│   │   ├── App.tsx            # Main app with tab navigation
│   │   ├── api.ts             # Backend API client
│   │   ├── styles.css         # All styling
│   │   └── components/
│   │       ├── Playground.tsx  # Main input/output panel
│   │       ├── TokenTrace.tsx  # Token-level trace viewer
│   │       └── SchemaViewer.tsx # Schema browser panel
│   ├── package.json
│   └── vite.config.ts          # Dev proxy → backend
├── sample-schemas/
│   ├── weather.json            # Weather API response schema
│   ├── extraction.json         # Entity extraction schema
│   ├── function-call.json      # Function calling schema
│   ├── email.json              # Email composition schema
│   ├── json-array.gbnf         # Raw GBNF: JSON array of strings
│   └── arithmetic.gbnf         # Raw GBNF: arithmetic expressions
├── tests/
│   └── test_grammar.py         # 9 tests (parsing, validation, conversion, trace)
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Quick start

### 1. Start llama-server

```bash
llama-server -m path/to/model.gguf --host 0.0.0.0 -ngl -1
```

### 2. Start the backend

```bash
pip install -r requirements.txt
python -m backend.server
# → Listening on http://127.0.0.1:8100
```

Optional environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_SERVER_URL` | `http://127.0.0.1:8080` | llama-server endpoint |
| `LLAMA_SERVER_BINARY` | `llama-server` | Path to llama-server binary |
| `LLAMA_MODEL_PATH` | `""` | Model path (for auto-launch) |
| `HOST` | `127.0.0.1` | Backend listen address |
| `PORT` | `8100` | Backend listen port |

### 3. Start the frontend

```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

Vite proxies `/api/*` to the backend automatically.

## How to use

### Playground tab

1. **Enter a prompt** — something that should produce structured output
2. **Paste a JSON Schema** (or pick one from the Schema Browser tab)
3. Click **Convert to GBNF** — translates JSON Schema → GBNF grammar
4. Set **n_probs** (> 0) to enable token-level tracing
5. Click **Generate** — output appears on the right

### Token Trace tab

After generating with `n_probs > 0`:

- **Stats bar**: total tokens, masked count, mask rate, grammar size
- **Inline view** (default): coloured tokens — green = accepted, red = masked by grammar
- **Cards view**: table with per-token logprob, candidate count, allowed count
- Hover any token to see detailed info

### Schema Browser tab

- Browse pre-built sample schemas
- JSON schemas are auto-converted to GBNF with "Use in Playground" button
- GBNF grammars available directly

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Backend + llama-server health |
| POST | `/api/completion` | Proxy to llama-server `/completion` |
| POST | `/api/convert-schema` | JSON Schema → GBNF grammar |
| POST | `/api/validate-grammar` | Validate GBNF syntax |
| POST | `/api/grammar-info` | Parse and explain GBNF rules |
| POST | `/api/trace` | Analyse completion for masking |
| GET | `/api/sample-schemas` | List sample schema files |
| GET | `/api/sample-schema/{name}` | Get schema file content |

## Tests

```bash
python tests/test_grammar.py
```

9 tests covering: grammar parsing, validation, JSON Schema conversion (simple, enum, nested, weather), and token trace analysis.

## Key learning outcomes

- How **constrained sampling** intercepts logits after the model computes them
- The **real cost** of grammar-constrained generation (tokens masked vs accepted)
- How JSON Schema translates to GBNF rules under the hood
- Why `n_probs` is essential for debugging silent schema/grammar failures
