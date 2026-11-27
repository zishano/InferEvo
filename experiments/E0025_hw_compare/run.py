"""E0025: H200 vs B300 at 64 GPUs (TP4/EP16; B300 TP8 surrogates not built),
P:D x prefill budget, two workloads (bursty realistic x4; sustained tiled
Poisson 4 copies at 43.5 req/s). Best SLO goodput per hardware. Tier-1.

    ./ie.sh experiments/E0025_hw_compare/run.py
"""
import json
from pathlib import Path
from adapters import workloads
from adapters.topologies import Deployment
from agent.controller import Controller

here = Path(__file__).parent
for f in here.glob("*.jsonl"):
    f.unlink()
out = {}
for label, rows in (("realistic-x4", workloads.scaled(4)), ("tiled-4", workloads.tiled(4, 43.52, seed=4))):
    for hw in ("h200", "b300"):
        c = Controller(here / f"{hw}-{label}.jsonl", rows, label)
        for p in (1, 2, 3):
            for b in (128, 512, 2048):
                c.evaluate(Deployment(hardware=hw, prefill_instances=p, decode_instances=4 - p, prefill_budget=b), "grid")
        best = max(c.seen.values(), key=lambda e: e["objectives"]["goodput_tokens_per_s_gpu"])
        o = best["objectives"]
        out[f"{label}/{hw}"] = {"best": best["key"], **o, "bottlenecks": [b["class"] for b in best["bottlenecks"]],
                                "grid": {e["key"]: e["objectives"] for e in c.seen.values()}}
        print("%-13s %s best %-36s good %6.2f tok %6.2f TTFTp99 %6.0f TPOT %5.2f %s" % (label, hw, best["key"],
              o["goodput_tokens_per_s_gpu"], o["tokens_per_s_gpu"], o["ttft_p99_ms"], o["tpot_p50_ms"],
              out[f"{label}/{hw}"]["bottlenecks"]), flush=True)
(here / "result.json").write_text(json.dumps(out, indent=1))
