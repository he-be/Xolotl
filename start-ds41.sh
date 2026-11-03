#!/bin/bash
# llama-swap cmd for Xolotl DeepSeek-V4.1: CONFIG PORT.  As /root/strata/start.sh: wait until the previous engine
# (Strata or ds41) has exited and released its pinned RAM / VRAM, then start the server.  The serve is this worktree's
# (a second worktree such as /root/xolotl-1103 runs its own serve_ds41.py); the python is its .venv, else
# DS41_VENV (default /root/xolotl/.venv, the first worktree's).
C=$1; P=$2
for i in $(seq 1 240); do pgrep -x strata >/dev/null || pgrep -x ds41_generate >/dev/null || break; sleep 0.5; done
D=$(cd "$(dirname "$0")" && pwd)
PY=$D/.venv/bin/python
[ -x "$PY" ] || PY=${DS41_VENV:-/root/xolotl/.venv}/bin/python
cd "$D"
exec "$PY" "$D/tools/v41/serve/serve_ds41.py" --engine strata --config "$C" --host 127.0.0.1 --port "$P"
