# Source this before running demos or tests:  source env.sh
#
# Real llama-server layout (two separate servers, one model each):
#   chat  -> 127.0.0.1:8090   SmolLM2-360M-Instruct Q8_0   (n_ctx 2048)
#   embed -> 127.0.0.1:8081   nomic-embed-text-v1.5 Q8_0   (n_embd 768)
#
# Port 8080 is Open WebUI (a third-party app), NOT a llama.cpp server.
# Never point tests or demos at 8080 - it will not speak the OpenAI API.

export LLAMACPP_HOST=127.0.0.1
export LLAMACPP_PORT=8090              # chat server; overrides the 8080 default
export LLM_PREFER=llama.cpp            # else Ollama on 11434 wins the probe

# Canonical base URLs. Projects and tests read these instead of hardcoding ports.
export LLM_CHAT_BASE_URL=http://127.0.0.1:8090
export LLM_EMBED_BASE_URL=http://127.0.0.1:8081

# Model ids as the real servers report them via /v1/models
export LLM_CHAT_MODEL='HuggingFaceTB/SmolLM2-360M-Instruct-GGUF:Q8_0'
export LLM_EMBED_MODEL='nomic-ai/nomic-embed-text-v1.5-GGUF:Q8_0'

export LLM_MODELS_DIR=/home/pa-joshi/Models
export LD_LIBRARY_PATH=/home/pa-joshi/llama.cpp/build/bin:$LD_LIBRARY_PATH

# Set to 1 to enable the live integration tests (tests/live_*, -m live).
# Unset means the live layer is skipped and only the offline unit tests run.
export LLM_LIVE=${LLM_LIVE:-}

# Resolve to whichever checkout this env.sh lives in, so a git worktree uses
# its own _shared/ rather than the main repo's.
LLAMA_PROJECTS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LLAMA_PROJECTS_ROOT

# Repo root on the path so `_shared` is importable from any project, and the
# live fixtures load as a global pytest plugin regardless of each project's
# own pytest rootdir (every project ships its own pyproject.toml).
export PYTHONPATH=${LLAMA_PROJECTS_ROOT}${PYTHONPATH:+:$PYTHONPATH}
export PYTEST_PLUGINS=_shared.live_fixtures

# The shared venv lives in the main checkout; worktrees reuse it.
export LLAMA_VENV=/home/pa-joshi/llama-cpp-projects/.venv
export PATH=${LLAMA_VENV}/bin:$PATH

# Cross-encoder reranker (own server: --reranking forces pooling=rank, which
# would corrupt /v1/embeddings if enabled on the embeddings server).
export LLM_RERANK_BASE_URL=http://127.0.0.1:8082
export LLM_RERANK_MODEL='gpustack/bge-reranker-v2-m3-GGUF:Q8_0'
