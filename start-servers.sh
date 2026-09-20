#!/usr/bin/env bash
# Start the three llama-server instances this repo's live tests run against.
#
#   8090  chat        SmolLM2-360M-Instruct      --metrics
#   8081  embeddings  nomic-embed-text-v1.5      --embeddings --embd-normalize -1 --metrics
#   8082  reranker    bge-reranker-v2-m3         --reranking --pooling rank
#
# Three processes rather than one, because the flags conflict:
#   --reranking forces pooling to "rank", which corrupts /v1/embeddings on the
#   same server (vectors come back as denormals). Embeddings and reranking
#   therefore cannot share a process. See REAL_VS_MOCK.md section 1a.
#
#   --embd-normalize -1 returns raw, un-normalised vectors. The llama-server
#   default (2 = L2) makes every vector unit-norm, which renders project 4.1's
#   norm-vs-unnorm ablation degenerate - it would measure nothing.
#
# Port 8080 is Open WebUI on this machine, NOT llama.cpp. This script never
# touches it, and the live fixtures hard-fail on any 8080 URL.
#
# If the systemd user units are installed (llama-chat/embed/rerank.service)
# this script delegates to them, so systemd stays the single owner of each
# port. Without the units it launches the processes directly.
#
# Usage:
#   ./start-servers.sh            start whatever is not already running
#   ./start-servers.sh --restart  stop the ones this script manages, then start
#   ./start-servers.sh --stop     stop them and exit
#   ./start-servers.sh --status   report health and exit
#
# Models download on first run via -hf and are cached by llama.cpp afterwards.

set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

BIN=${LLAMA_SERVER_BIN:-/home/pa-joshi/llama.cpp/build/bin/llama-server}
LOG_DIR=${LLAMA_LOG_DIR:-/tmp/llama-servers}
HOST=${LLAMACPP_HOST:-127.0.0.1}
STARTUP_TIMEOUT=${LLAMA_STARTUP_TIMEOUT:-600}   # first run downloads a model

# name|port|logfile|args...
SERVERS=(
  "chat|8090|chat.log|-hf HuggingFaceTB/SmolLM2-360M-Instruct-GGUF:Q8_0 -c 6144 --parallel 3 -t 2 --metrics"
  "embed|8081|embed.log|-hf nomic-ai/nomic-embed-text-v1.5-GGUF:Q8_0 --embeddings --embd-normalize -1 --metrics -c 2048 -t 1"
  "rerank|8082|rerank.log|-hf gpustack/bge-reranker-v2-m3-GGUF:Q8_0 --reranking --pooling rank -c 2048 -t 2"
)

is_up() { curl -sf -m 2 "http://${HOST}:$1/health" >/dev/null 2>&1; }

# systemd user units (llama-chat/embed/rerank.service) are the preferred
# manager when installed: they restart on crash and start at boot. Delegate to
# them so the two mechanisms never both own a port. Falls back to launching
# directly when the units are absent.
unit_for() { echo "llama-$1.service"; }
has_unit() {
  command -v systemctl >/dev/null 2>&1 &&
    systemctl --user cat "$(unit_for "$1")" >/dev/null 2>&1
}

# Only ever match a llama-server bound to the port we manage, so an unrelated
# process (Open WebUI, someone's editor) is never signalled.
pid_on_port() {
  ss -ltnp 2>/dev/null | grep ":$1 " | grep -oP 'pid=\K[0-9]+' | head -1
}

stop_one() {
  local name=$1 port=$2 pid
  if has_unit "$name"; then
    systemctl --user stop "$(unit_for "$name")"
    echo "  ${name} (${port}): stopped via systemd"
    return 0
  fi
  pid=$(pid_on_port "$port") || true
  if [ -z "${pid:-}" ]; then
    echo "  ${name} (${port}): not running"
    return 0
  fi
  if ! tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null | grep -q 'llama-server'; then
    echo "  ${name} (${port}): occupied by a NON-llama-server process (pid ${pid}); leaving it alone" >&2
    return 0
  fi
  kill "$pid"
  local waited=0
  while is_up "$port" && [ "$waited" -lt 20 ]; do sleep 1; waited=$((waited + 1)); done
  if is_up "$port"; then
    echo "  ${name} (${port}): did not stop after ${waited}s" >&2
    return 1
  fi
  echo "  ${name} (${port}): stopped (was pid ${pid})"
}

start_one() {
  local name=$1 port=$2 logfile=$3 args=$4
  if is_up "$port"; then
    echo "  ${name} (${port}): already up"
    return 0
  fi
  local pid
  pid=$(pid_on_port "$port") || true
  if [ -n "${pid:-}" ]; then
    echo "  ${name} (${port}): port held by pid ${pid} but /health does not answer; refusing to start" >&2
    return 1
  fi

  if has_unit "$name"; then
    systemctl --user start "$(unit_for "$name")"
    local waited=0
    while [ "$waited" -lt "$STARTUP_TIMEOUT" ]; do
      is_up "$port" && { echo "  ${name} (${port}): up via systemd"; return 0; }
      sleep 2
      waited=$((waited + 2))
    done
    echo "  ${name} (${port}): systemd unit did not become healthy in ${STARTUP_TIMEOUT}s" >&2
    echo "  journalctl --user -u $(unit_for "$name") -n 20" >&2
    return 1
  fi

  mkdir -p "$LOG_DIR"
  local log="${LOG_DIR}/${logfile}"
  # shellcheck disable=SC2086  # args are intentionally word-split
  setsid nohup "$BIN" $args --host "$HOST" --port "$port" \
    > "$log" 2>&1 < /dev/null &

  local waited=0
  while [ "$waited" -lt "$STARTUP_TIMEOUT" ]; do
    if is_up "$port"; then
      echo "  ${name} (${port}): up  [${log}]"
      return 0
    fi
    # Surface a dead process immediately instead of waiting out the timeout.
    if ! pgrep -f "port ${port}" >/dev/null 2>&1 && [ "$waited" -gt 3 ]; then
      echo "  ${name} (${port}): process exited during startup; see ${log}" >&2
      tail -5 "$log" >&2 || true
      return 1
    fi
    sleep 2
    waited=$((waited + 2))
  done
  echo "  ${name} (${port}): failed to become healthy in ${STARTUP_TIMEOUT}s; see ${log}" >&2
  return 1
}

status() {
  local rc=0
  for entry in "${SERVERS[@]}"; do
    IFS='|' read -r name port _ _ <<<"$entry"
    if is_up "$port"; then
      local model
      model=$(curl -sf -m 3 "http://${HOST}:${port}/v1/models" 2>/dev/null \
              | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || echo "?")
      printf "  %-7s %-5s up    %s\n" "$name" "$port" "$model"
    else
      printf "  %-7s %-5s DOWN\n" "$name" "$port"
      rc=1
    fi
  done
  return $rc
}

case "${1:-start}" in
  --status)
    echo "llama-server status:"
    status
    ;;
  --stop)
    echo "Stopping llama-servers:"
    for entry in "${SERVERS[@]}"; do
      IFS='|' read -r name port _ _ <<<"$entry"
      stop_one "$name" "$port"
    done
    ;;
  --restart)
    echo "Stopping llama-servers:"
    for entry in "${SERVERS[@]}"; do
      IFS='|' read -r name port _ _ <<<"$entry"
      stop_one "$name" "$port"
    done
    echo "Starting llama-servers:"
    for entry in "${SERVERS[@]}"; do
      IFS='|' read -r name port logfile args <<<"$entry"
      start_one "$name" "$port" "$logfile" "$args"
    done
    echo
    echo "Ready. Run the live suite with:"
    echo "  source env.sh && python scripts/run_all_tests.py --mode live"
    ;;
  start)
    if [ ! -x "$BIN" ]; then
      echo "llama-server not found at ${BIN}" >&2
      echo "Set LLAMA_SERVER_BIN to its path." >&2
      exit 1
    fi
    echo "Starting llama-servers (logs in ${LOG_DIR}):"
    failed=0
    for entry in "${SERVERS[@]}"; do
      IFS='|' read -r name port logfile args <<<"$entry"
      start_one "$name" "$port" "$logfile" "$args" || failed=1
    done
    if [ "$failed" -ne 0 ]; then
      echo "One or more servers failed to start." >&2
      exit 1
    fi
    echo
    echo "Ready. Run the live suite with:"
    echo "  source env.sh && python scripts/run_all_tests.py --mode live"
    ;;
  *)
    echo "usage: $0 [--status|--stop|--restart]" >&2
    exit 2
    ;;
esac
