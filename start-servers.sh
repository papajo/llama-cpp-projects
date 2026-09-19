#!/usr/bin/env bash
# Start the llama.cpp router serving both local GGUFs on one port.
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh
BIN=/home/pa-joshi/llama.cpp/build/bin/llama-server
if curl -sf -m 3 "http://${LLAMACPP_HOST}:${LLAMACPP_PORT}/health" >/dev/null 2>&1; then
  echo "router already up on ${LLAMACPP_PORT}"; exit 0
fi
setsid nohup "$BIN" \
  --models-preset "$(pwd)/config/router.preset.ini" \
  --port "${LLAMACPP_PORT}" --models-max 2 \
  > /tmp/llama-router.log 2>&1 < /dev/null &
for _ in $(seq 1 60); do
  sleep 1
  curl -sf -m 2 "http://${LLAMACPP_HOST}:${LLAMACPP_PORT}/health" >/dev/null 2>&1 && { echo "router up on ${LLAMACPP_PORT}"; exit 0; }
done
echo "router failed to start; see /tmp/llama-router.log" >&2; exit 1
