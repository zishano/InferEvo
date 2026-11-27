"""E0018: controller (diagnosis + Pareto-neighbour exploration) vs full grid
over P:D x prefill budget (H200 TP4/EP16, 64 GPUs; 9 deployments).

    ./ie.sh experiments/E0018_controller_budget/run.py
"""
import json
from pathlib import Path
from adapters import workloads
from adapters.topologies import Deployment
from agent.controller import Controller, search, dominates

here = Path(__file__).parent
out = {}
for f in here.glob("*.jsonl"):
    f.unlink()                      # ledgers are regenerated (deterministic)
for label, mult in (("realistic-x4", 4), ("realistic-x1", 1)):
    rows = workloads.scaled(mult)
    grid = Controller(here / f"grid-{label}.jsonl", rows, label)
    for p, d in ((1, 3), (2, 2), (3, 1)):
        for b in (128, 512, 2048):
            grid.evaluate(Deployment(prefill_instances=p, decode_instances=d, prefill_budget=b), "grid")
    ctl = Controller(here / f"ledger-{label}.jsonl", rows, label)
    search(ctl, Deployment(prefill_instances=1, decode_instances=3), budget=6)
    evaluated = [e for e in ctl.seen.values() if "objectives" in e]
    grid_front = {e["key"] for e in grid.archive}
    found = {e["key"] for e in ctl.archive}
    # a controller archive member is "true" if no grid point dominates it
    true_members = {e["key"] for e in ctl.archive
                    if not any(dominates(g["objectives"], e["objectives"]) for g in grid.archive)}
    out[label] = {"grid_front": sorted(grid_front), "controller_archive": sorted(found),
                  "controller_evaluations": len(evaluated), "grid_evaluations": 9,
                  "front_recall": len(true_members & grid_front) / len(grid_front),
                  "path": [{"key": e["key"], "rationale": e["rationale"], **e["objectives"],
                            "bottlenecks": [b["class"] for b in e["bottlenecks"]]} for e in evaluated],
                  "grid": [{"key": e["key"], **e["objectives"],
                            "bottlenecks": [b["class"] for b in e["bottlenecks"]]} for e in grid.seen.values()]}
    print(label, "| grid front", sorted(grid_front))
    for e in out[label]["grid"]:
        print("  grid %-36s %6.2f tok/s/GPU TTFT p99 %7.0f ms TPOT %5.2f ms %s" % (e["key"], e["tokens_per_s_gpu"], e["ttft_p99_ms"], e["tpot_p50_ms"], e["bottlenecks"]))
    for e in out[label]["path"]:
        print("  ctl  %-36s %6.2f | %s" % (e["key"], e["tokens_per_s_gpu"], e["rationale"][:70]))
    print("  controller evals", len(evaluated), "front recall %.2f" % out[label]["front_recall"])
(here / "result.json").write_text(json.dumps(out, indent=1))
