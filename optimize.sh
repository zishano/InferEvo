#!/usr/bin/env bash
# Agent interface: autonomous search for the best deployment (Tier-1).
# Given GPUs, hardware, ISL/OSL and SLO, the agent searches P:D split, attention
# TP, prefill budget, per-side layout PP1xEP16 / PP2xEP8 / PP4xEP4 and decode
# batch, where validated Tier-1 surrogates exist (instances of 16 GPUs) and prints the best config, the tok/s/GPU vs tok/s/user Pareto set,
# the ledger and a ./simulate.sh command.
#
#   ./optimize.sh --gpus 128 --isl 8192 --osl 1024              # 16 evaluations (~5 min)
#   ./optimize.sh --gpus 64 --isl 2000 --osl 500 --ttft-slo 2   # other workload / SLO
#   ./optimize.sh --gpus 128 --isl 8192 --osl 1024 --grid       # exhaustive, for checking
#   ./optimize.sh --help
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$root"
exec ./ie.sh tools/optimize.py "$@"
