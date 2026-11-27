"""E0024: joint prefill surrogate across token budgets (B300 TP4/EP16).
Same method as E0017; budget-128 rows: E0014 b300-tp4 is graph-bound based,
so budget 128 is sampled from the engine here too.

Engine (Tier-2) samples under each budget's launcher config (P1+D1):
  A) k requests, total <= budget, arriving together -> one round (TTFT).
  B) one prompt of L > budget tokens -> back-to-back budget-sized chunks.
One model is fitted on all rounds
(features describe round composition, not the budget); held-out error is
reported per budget.

    ./ie.sh experiments/E0024_b300_budgets/run.py
"""
import json, random, statistics, subprocess, tempfile
from dataclasses import asdict
from pathlib import Path
from adapters import tier1
from adapters.cost_model import cost_model_version
from adapters.engines import ROOT, SIMTHESIZER, source_identity
from adapters.topologies import Deployment, tier2_config, CACHE

here = Path(__file__).parent
E = ROOT / "experiments"
BIN = SIMTHESIZER / "target" / "release" / "simthesizer"
cache = here / "samples.jsonl"
done = {(r["budget"], r["index"]): r for r in map(json.loads, cache.read_text().splitlines())} if cache.exists() else {}
log = lambda *a: print(*a, flush=True)


def plans(budget, seed):
    rng = random.Random(seed)
    out = []
    for _ in range(28):
        k = rng.randint(1, 8)
        cuts = sorted(rng.sample(range(1, budget), k - 1)) if k > 1 else []
        total = rng.randint(k, budget)
        sizes = [max(1, round((b - a) * total / budget)) for a, b in zip([0, *cuts], [*cuts, budget])]
        out.append(("A", rng.sample(range(1000), k), sizes))
    for _ in range(8):
        out.append(("B", [rng.randrange(1000)], [rng.randint(budget + 1, 16384)]))
    return out


for budget, seed in ((128, 30), (512, 31), (2048, 32)):
    dep = Deployment(hardware="b300", prefill_budget=budget)
    tier2_config(dep)
    config = CACHE / f"{dep.key}.toml"
    for idx, (kind, ids, sizes) in enumerate(plans(budget, seed)):
        if (budget, idx) in done:
            continue
        with tempfile.TemporaryDirectory(prefix="e0024-") as tmp:
            wl = Path(tmp) / "w.jsonl"
            wl.write_text("".join(json.dumps({"request_id": i, "arrival_time": 1.0, "initial": [0],
                "sub_requests": [{"kind": "llm", "input_tokens": q, "output_tokens": 2, "next": []}]}) + "\n"
                for i, q in zip(ids, sizes)))
            subprocess.run([str(BIN), "--config", str(config), "--workload", str(wl),
                            "--output-jsonl", str(Path(tmp) / "r.jsonl")], check=True, capture_output=True)
            t = {json.loads(l)["request_id"]: json.loads(l)["sub_requests"][0]["token_latency"][0]
                 for l in (Path(tmp) / "r.jsonl").read_text().splitlines()}
        if kind == "A":
            rounds, target = [[(i, q, 0) for i, q in zip(ids, sizes)]], statistics.fmean(t.values())
        else:
            (i,), (length,) = ids, sizes
            rounds, target = [[(i, min(budget, length - d), d)] for d in range(0, length, budget)], t[i]
        row = {"budget": budget, "index": idx, "kind": kind, "rounds": rounds, "engine_s": target}
        with cache.open("a") as f:
            f.write(json.dumps(row) + "\n")
        done[(budget, idx)] = row
        log(f"budget {budget} sample {idx} {kind}: {target*1e3:.1f} ms")

samples = list(done.values())
key = {"hardware": "b300", "tp": 4, "ep": 16, "groups": 4, "stage": "p", "efficiency": {},
       "cost_model_version": cost_model_version(), "target": "Tier-2 engine TTFT/rounds, budgets 128/512/2048"}
is_test = lambda s: s["index"] % 5 == 0
model = tier1.fit_round_sums([s["rounds"] for s in samples if not is_test(s)],
                             [s["engine_s"] for s in samples if not is_test(s)], key)
per_budget = {}
for b in (128, 512, 2048):
    err = [abs(sum(model.step_time(r) for r in s["rounds"]) - s["engine_s"]) / s["engine_s"]
           for s in samples if s["budget"] == b and is_test(s)]
    per_budget[b] = {"n_test": len(err), "median": statistics.median(err), "max": max(err)}
log(json.dumps(per_budget, indent=1))
(here / "result.json").write_text(json.dumps({"experiment": "E0024", "prefill": asdict(model),
    "heldout_by_budget": per_budget, "source_identity": source_identity()}, indent=1))
