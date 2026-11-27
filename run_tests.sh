#!/usr/bin/env bash
# InferEvo tests (engine environment via ie.sh).
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec ./ie.sh -m unittest discover -s tests -t . -v "$@"
