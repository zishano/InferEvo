"""E0023: scale dimension — 32/64/128 x H200 (2/4/8 instances, EP16) at fixed
per-instance offered load (tiled realistic trace: copies = instances,
Poisson rate = 10.88 req/s per instance = E0018 x4 per-instance load).
Full grid per scale over P:D x budget x TP (equal P/D) + controller (D21,
10 evals) from P1:D(n-1) TP4 b128. Tier-1 screening (decode: E0022 for TP4).

    ./ie.sh experiments/E0023_scale/run.py
"""
import json
from pathlib import Path
from adapters import workloads
from adapters.topologies import Deployment
from agent.controller import Controller, search, search_quality

here = Path(__file__).parent
for f in here.glob("*.jsonl"):
    f.unlink()
PER_INSTANCE_RATE = 10.88
out = {}
for n in (2, 4, 8):
    rows = workloads.tiled(n, PER_INSTANCE_RATE * n, seed=n)
    space = [Deployment(prefill_instances=p, decode_instances=n - p, prefill_budget=b, prefill_tp=t, decode_tp=t)
             for p in range(1, n) for b in (128, 512, 2048) for t in (4, 8)]
    grid = Controller(here / f"grid-{n}.jsonl", rows, f"tiled-{n}")
    for d in space:
        grid.evaluate(d, "grid")
    ref = list(grid.seen.values())
    best = max(ref, key=lambda e: e["objectives"]["goodput_tokens_per_s_gpu"])
    ctl = Controller(here / f"rule-{n}.jsonl", rows, f"tiled-{n}")
    keys = {d.key for d in space}
    search(ctl, Deployment(prefill_instances=1, decode_instances=n - 1), budget=10, allowed=lambda d: d.key in keys)
    found = [e for e in ctl.seen.values() if "objectives" in e]
    q = search_quality(found, ref) | {"goodput_best_found": best["key"] in {e["key"] for e in found}}
    top = sorted(ref, key=lambda e: -e["objectives"]["goodput_tokens_per_s_gpu"])[:5]
    out[n] = {"gpus": 16 * n, "best": best["key"], "best_objectives": best["objectives"],
              "top5": [(e["key"], e["objectives"]) for e in top], "search": q, "grid_size": len(space),
              "grid": {e["key"]: e["objectives"] for e in ref}}
    print(f"{16*n} GPUs ({len(space)} pts): best {best['key']}")
    for k, o in out[n]["top5"]:
        print("   %-36s good %6.2f tok %6.2f TTFTp99 %6.0f TPOT %5.2f" % (k, o["goodput_tokens_per_s_gpu"],
              o["tokens_per_s_gpu"], o["ttft_p99_ms"], o["tpot_p50_ms"]))
    print("   search:", q, flush=True)
(here / "result.json").write_text(json.dumps(out, indent=1))
