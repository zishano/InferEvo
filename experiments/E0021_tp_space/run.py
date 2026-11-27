"""E0021: larger design space — P:D x prefill budget x attention TP (equal P/D: engine
requires equal TP for MLA+KDA state handoff) (18 deployments, 64xH200, EP16). Grid vs diagnosis policy (10 evals) vs random
(10 evals, 5 seeds), x4 workload. Tier-1 screening.

    ./ie.sh experiments/E0021_tp_space/run.py
"""
import json, random, statistics
from pathlib import Path
from adapters import workloads
from adapters.topologies import Deployment
from agent.controller import Controller, search, search_quality

here = Path(__file__).parent
for f in here.glob("*.jsonl"):
    f.unlink()
SPACE = [Deployment(prefill_instances=p, decode_instances=4 - p, prefill_budget=b, prefill_tp=t, decode_tp=t)
         for p in (1, 2, 3) for b in (128, 512, 2048) for t in (4, 8)]
rows = workloads.scaled(4)
grid = Controller(here / "grid.jsonl", rows, "x4")
for d in SPACE:
    grid.evaluate(d, "grid")
ref = list(grid.seen.values())
gp = max(ref, key=lambda e: e["objectives"]["goodput_tokens_per_s_gpu"])
ctl = Controller(here / "rule.jsonl", rows, "x4")
search(ctl, Deployment(prefill_instances=1, decode_instances=3), budget=10)
found = [e for e in ctl.seen.values() if "objectives" in e]
rule = search_quality(found, ref) | {"goodput_best_found": gp["key"] in {e["key"] for e in found}}
rand = []
for seed in range(5):
    sel = [grid.seen[SPACE[i].key] for i in random.Random(seed).sample(range(len(SPACE)), 10)]
    rand.append(search_quality(sel, ref) | {"goodput_best_found": gp["key"] in {e["key"] for e in sel}})
for e in sorted(ref, key=lambda e: -e["objectives"]["goodput_tokens_per_s_gpu"])[:8]:
    o = e["objectives"]
    print("%-36s tok %6.2f good %6.2f TTFTp99 %6.0f TPOT %5.2f %s" % (e["key"], o["tokens_per_s_gpu"],
          o["goodput_tokens_per_s_gpu"], o["ttft_p99_ms"], o["tpot_p50_ms"], [b["class"] for b in e["bottlenecks"]]))
print("rule  :", rule)
print("random: HV %.3f goodput-best rate %.2f" % (statistics.fmean(r["hypervolume_ratio"] for r in rand),
      statistics.fmean(r["goodput_best_found"] for r in rand)))
for e in found:
    print("  ctl %-36s good %6.2f | %s" % (e["key"], e["objectives"]["goodput_tokens_per_s_gpu"], e["rationale"][:60]))
(here / "result.json").write_text(json.dumps({"rule": rule, "random": rand, "goodput_best": gp["key"],
    "grid": {e["key"]: e["objectives"] for e in ref}, "path": [(e["key"], e["rationale"]) for e in found]}, indent=1))
