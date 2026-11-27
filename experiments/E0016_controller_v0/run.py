"""E0016: controller v0 vs exhaustive grid (H200 TP4/EP16, 64 GPUs).

Workloads: realistic trace x4 (prefill-heavy saturation) and x1. Design
space: P:D in {1:3, 2:2, 3:1} (16-GPU instances). The diagnosis-driven
controller starts from 1:3 (decode-heavy, deliberately wrong); the grid
evaluates every split. Reports evaluations used and archive quality.

    ./ie.sh experiments/E0016_controller_v0/run.py
"""
import json
from pathlib import Path
from adapters import workloads
from adapters.topologies import Deployment
from agent.controller import Controller, search

here = Path(__file__).parent
out = {}
for label, mult in (("realistic-x4", 4), ("realistic-x1", 1)):
    rows = workloads.scaled(mult)
    ctl = Controller(here / f"ledger-{label}.jsonl", rows, label)
    archive = search(ctl, Deployment(prefill_instances=1, decode_instances=3))
    grid = Controller(here / f"grid-{label}.jsonl", rows, label)
    for p, d in ((1, 3), (2, 2), (3, 1)):
        grid.evaluate(Deployment(prefill_instances=p, decode_instances=d), "grid")
    summary = lambda c: [{"key": e["key"], "rationale": e["rationale"], **e["objectives"],
                          "bottlenecks": [b["class"] for b in e["bottlenecks"]]} for e in c.seen.values()]
    out[label] = {"controller_evaluations": len(ctl.seen), "controller_path": summary(ctl),
                  "controller_archive": [e["key"] for e in archive],
                  "grid": summary(grid), "grid_archive": [e["key"] for e in grid.archive]}
    print(label)
    for e in summary(ctl):
        print("  ctl ", e["key"], "%.2f tok/s/GPU TTFT p99 %.0f ms TPOT p50 %.2f ms" % (e["tokens_per_s_gpu"], e["ttft_p99_ms"], e["tpot_p50_ms"]), e["bottlenecks"], "|", e["rationale"][:60])
    for e in summary(grid):
        print("  grid", e["key"], "%.2f tok/s/GPU TTFT p99 %.0f ms TPOT p50 %.2f ms" % (e["tokens_per_s_gpu"], e["ttft_p99_ms"], e["tpot_p50_ms"]), e["bottlenecks"])
    print("  archives: ctl", out[label]["controller_archive"], "grid", out[label]["grid_archive"])
(here / "result.json").write_text(json.dumps(out, indent=1))
