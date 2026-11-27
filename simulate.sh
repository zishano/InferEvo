#!/usr/bin/env bash
# Manual interface: one simulation, every parallel parameter explicit and checked.
#
# Per instance: GPUs = PP x (attention TP x DP) = PP x (MoE EP x ETP).
# --tp/--ep/--etp/--pp set both sides; --p-*/--d-* set one side.
# ETP/PP and EP != 16 need --tier 2 (no Tier-1 surrogate yet).
#
#   ./simulate.sh --p 7 --d 1 --tp 4 --ep 16 --p-budget 2048 --rate 2.1     # 112 + 16 = 128 GPUs
#   ./simulate.sh --p 2 --d 2 --p-tp 8 --p-dp 2 --d-tp 8 --d-dp 2 --rate .5  # 64 GPUs, TP8 x DP2
#   ./simulate.sh --p 6 --d 2 --capacity                                     # 128 GPUs: max SLO rate
#   ./simulate.sh --p 1 --d 1 --d-ep 32 --requests 1 --tier 2                # 16 + 32 GPUs, Tier-2 only
#   ./simulate.sh --p 1 --d 1 --ep 8 --pp 2 --requests 1 --tier 2            # 2 stages x 8 GPUs per instance
#   ./simulate.sh --p 1 --d 1 --d-ep 8 --d-etp 2 --requests 1 --tier 2       # decode experts over 2 GPUs
#   ./simulate.sh --help                                                     # all options
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$root"
exec ./ie.sh tools/simulate.py "$@"
