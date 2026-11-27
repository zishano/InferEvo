"""E0019: diagnosis-driven policy vs random search vs full grid (Tier-1).

Space: 64xH200 TP4/EP16, P:D in {1:3,2:2,3:1} x prefill budget {128,512,2048}.
Budget: 6 evaluations; random search over 5 seeds. SLO (assumption):
TTFT <= 2 s, TPOT <= 50 ms. Metrics: per-objective best found, hypervolume
ratio vs grid, goodput-best found.

    ./ie.sh experiments/E0019_search_policies/run.py
"""
import json, random, statistics
from pathlib import Path
from adapters import workloads
from adapters.topologies import Deployment
from agent.controller import Controller, search, search_quality

here = Path(__file__).parent
for f in here.glob("*.jsonl"):
    f.unlink()
SPACE = [Deployment(prefill_instances=p, decode_instances=4 - p, prefill_budget=b)
         for p in (1, 2, 3) for b in (128, 512, 2048)]
out = {}
for label, mult in (("x4", 4), ("x1", 1)):
    rows = workloads.scaled(mult)
    grid = Controller(here / f"grid-{label}.jsonl", rows, label)
    for d in SPACE:
        grid.evaluate(d, "grid")
    ref = list(grid.seen.values())
    goodput_best = max(ref, key=lambda e: e["objectives"]["goodput_tokens_per_s_gpu"])["key"]
    ctl = Controller(here / f"rule-{label}.jsonl", rows, label)
    search(ctl, Deployment(prefill_instances=1, decode_instances=3), budget=6,
           allowed=lambda d: d.key in {x.key for x in SPACE})
    found = [e for e in ctl.seen.values() if "objectives" in e]
    rule = search_quality(found, ref) | {"goodput_best_found": goodput_best in {e["key"] for e in found}}
    rand = []
    for seed in range(5):
        pick = random.Random(seed).sample(range(len(SPACE)), 6)
        sel = [grid.seen[SPACE[i].key] for i in pick]   # same Tier-1 results, no re-run
        rand.append(search_quality(sel, ref) | {"goodput_best_found": goodput_best in {e["key"] for e in sel}})
    out[label] = {"rule": rule, "random": rand, "goodput_best": goodput_best,
                  "grid": {e["key"]: e["objectives"] for e in ref}}
    rh = statistics.fmean(r["hypervolume_ratio"] for r in rand)
    rg = sum(r["goodput_best_found"] for r in rand) / len(rand)
    print(f"{label}: goodput-best {goodput_best}")
    print(f"  rule  : best {rule['best_found']} HV {rule['hypervolume_ratio']:.3f} goodput-best {rule['goodput_best_found']}")
    print(f"  random: HV mean {rh:.3f} goodput-best rate {rg:.2f} best-throughput rate "
          f"{sum(r['best_found']['tokens_per_s_gpu'] for r in rand)/len(rand):.2f}")
    for k, o in out[label]["grid"].items():
        print(f"    {k:38s} goodput {o['goodput_tokens_per_s_gpu']:6.2f} attain {o['slo_attainment']:.2f}")
(here / "result.json").write_text(json.dumps(out, indent=1))
