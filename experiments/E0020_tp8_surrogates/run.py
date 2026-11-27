"""E0020: Tier-1 surrogates for H200 attention TP8 (EP16).

Decode on graph bounds (kv <= 32k, up to 32 requests); prefill jointly over
budgets 128/512/2048 on engine samples (E0017 method) under each budget's
TP8 launcher config. Held-out errors reported per budget.

    ./ie.sh experiments/E0020_tp8_surrogates/run.py
"""
import json, random, statistics, subprocess, tempfile
from dataclasses import asdict
from pathlib import Path
from adapters import tier1
from adapters.cost_model import cost_model_version
from adapters.engines import SIMTHESIZER, source_identity
from adapters.topologies import CACHE, Deployment, tier2_config

here = Path(__file__).parent
BIN = SIMTHESIZER / "target" / "release" / "simthesizer"
TP = 8
log = lambda *a: print(*a, flush=True)
dec_path = here / "decode_model.json"
if dec_path.exists():
    dec = tier1.Tier1Model(**json.loads(dec_path.read_text()))
else:
    dec = tier1.fit("h200", TP, 16, "d", samples=120, seed=3, max_requests=32, max_kv=32768)
    dec_path.write_text(json.dumps(asdict(dec)))
log("decode", dec.train_median_rel_err, dec.train_p90_rel_err)

cache = here / "samples.jsonl"
done = {(r["budget"], r["index"]): r for r in map(json.loads, cache.read_text().splitlines())} if cache.exists() else {}


def plans(budget, seed):
    rng = random.Random(seed)
    out = []
    for _ in range(24):
        k = rng.randint(1, 8)
        cuts = sorted(rng.sample(range(1, budget), k - 1)) if k > 1 else []
        total = rng.randint(k, budget)
        out.append(("A", rng.sample(range(1000), k),
                    [max(1, round((b - a) * total / budget)) for a, b in zip([0, *cuts], [*cuts, budget])]))
    for _ in range(6):
        out.append(("B", [rng.randrange(1000)], [rng.randint(budget + 1, 16384)]))
    return out


for budget, seed in ((128, 21), (512, 22), (2048, 23)):
    dep = Deployment(prefill_tp=TP, decode_tp=TP, prefill_budget=budget)
    tier2_config(dep)
    for idx, (kind, ids, sizes) in enumerate(plans(budget, seed)):
        if (budget, idx) in done:
            continue
        with tempfile.TemporaryDirectory(prefix="e0020-") as tmp:
            wl = Path(tmp) / "w.jsonl"
            wl.write_text("".join(json.dumps({"request_id": i, "arrival_time": 1.0, "initial": [0],
                "sub_requests": [{"kind": "llm", "input_tokens": q, "output_tokens": 2, "next": []}]}) + "\n"
                for i, q in zip(ids, sizes)))
            subprocess.run([str(BIN), "--config", str(CACHE / f"{dep.key}.toml"), "--workload", str(wl),
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
        log(f"tp8 budget {budget} sample {idx} {kind}: {target*1e3:.1f} ms")
samples = list(done.values())
key = {"hardware": "h200", "tp": TP, "ep": 16, "groups": 16 // TP, "stage": "p", "efficiency": {},
       "cost_model_version": cost_model_version(), "target": "Tier-2 engine TTFT/rounds, budgets 128/512/2048"}
is_test = lambda s: s["index"] % 5 == 0
pre = tier1.fit_round_sums([s["rounds"] for s in samples if not is_test(s)],
                           [s["engine_s"] for s in samples if not is_test(s)], key)
per = {}
for b in (128, 512, 2048):
    err = [abs(sum(pre.step_time(r) for r in s["rounds"]) - s["engine_s"]) / s["engine_s"]
           for s in samples if s["budget"] == b and is_test(s)]
    per[b] = {"n_test": len(err), "median": statistics.median(err), "max": max(err)}
log(json.dumps(per, indent=1))
(here / "result.json").write_text(json.dumps({"experiment": "E0020", "decode": asdict(dec), "prefill": asdict(pre),
    "heldout_by_budget": per, "source_identity": source_identity()}, indent=1))
