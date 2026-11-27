#!/usr/bin/env bash
# Run Python with the InferEvo root on PYTHONPATH and the engine's locked
# environment (the interpreter that executes the RoofLang bridge).
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PY="${INFEREVO_PYTHON:-$ROOT/engines/simthesizer/.venv/bin/python}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$PY" "$@"
