#!/bin/bash
# Open the llama-server launcher dashboard in your default browser
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-launcher}"

case "$MODE" in
  launcher)
    echo "🦙 Opening llama-server launcher..."
    open "$SCRIPT_DIR/llama-server-launcher.html"
    ;;
  interactive)
    echo "🦙 Starting interactive dashboard server..."
    echo "   Open http://127.0.0.1:3000 in your browser"
    cd "$SCRIPT_DIR" && python dashboard/server.py
    ;;
  both)
    echo "🦙 Opening launcher + starting interactive dashboard..."
    open "$SCRIPT_DIR/llama-server-launcher.html"
    cd "$SCRIPT_DIR" && python dashboard/server.py
    ;;
  *)
    echo "Usage: $0 [launcher|interactive|both]"
    echo "  launcher     — open static HTML launcher (default)"
    echo "  interactive  — start the interactive FastAPI dashboard on port 3000"
    echo "  both         — do both"
    exit 1
    ;;
esac
