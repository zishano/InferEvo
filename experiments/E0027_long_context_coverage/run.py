"""E0027: long-context coverage of the H200 Tier-1 surrogates (TP4, TP8).

Prefill: Tier-2 engine samples of single prompts 16k-40k tokens at budgets
128/512/2048 (Tier-2 runs in parallel, temp dirs removed). Report the
current models' extrapolation error on them, then refit jointly with all
earlier samples (E0012/E0015/E0017 for TP4, E0020 for TP8); held-out = every
5th sample.
Decode: graph-bound anchors with KV up to 48k (exported graphs, exact
bounds) + the loaded Tier-2 rounds (E0022 TP4 / E0026 TP8). Report old
model error at long KV vs bound, and the refit's error.

    ./ie.sh experiments/E0027_long_context_coverage/run.py
"""
import json, random, statistics, subprocess, tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
import numpy as np
from adapters import tier1
from adapters.engines import SIMTHESIZER, source_identity
from adapters.topologies import CACHE, Deployment, tier2_config
from validation.decode_rounds import reconstruct_multi

here = Path(__file__).parent
E = here.parent
BIN = SIMTHESIZER / "target" / "release" / "simthesizer"
log = lambda *a: print(*a, flush=True)
cache = here / "prefill_samples.jsonl"
done = {(r["tp"], r["budget"], r["index"]): r for r in map(json.loads, cache.read_text().splitlines())} if cache.exists() else {}

jobs = []
for tp in (4, 8):
    for budget in (128, 512, 2048):
        dep = Deployment(prefill_tp=tp, decode_tp=tp, prefill_budget=budget)
        tier2_config(dep)
        rng = random.Random(100 * tp + budget)
        for idx in range(6):
            length = rng.randint(16384, 40960)
            if (tp, budget, idx) not in done:
                jobs.append((tp, budget, idx, length, CACHE / f"{dep.key}.toml"))


def engine(job):
    tp, budget, idx, length, config = job
    with tempfile.TemporaryDirectory(prefix="e0027-") as tmp:
        wl = Path(tmp) / "w.jsonl"
        wl.write_text(json.dumps({"request_id": idx, "arrival_time": 1.0, "initial": [0], "sub_requests": [
            {"kind": "llm", "input_tokens": length, "output_tokens": 2, "next": []}]}) + "\n")
        subprocess.run([str(BIN), "--config", str(config), "--workload", str(wl),
                        "--output-jsonl", str(Path(tmp) / "r.jsonl")], check=True, capture_output=True)
        ttft = json.loads((Path(tmp) / "r.jsonl").read_text())["sub_requests"][0]["token_latency"][0]
    return {"tp": tp, "budget": budget, "index": idx, "kind": "B", "engine_s": ttft,
            "rounds": [[(idx, min(budget, length - d), d)] for d in range(0, length, budget)]}


with ThreadPoolExecutor(12) as pool:
    for row in pool.map(engine, jobs):
        with cache.open("a") as f:
            f.write(json.dumps(row) + "\n")
        done[(row["tp"], row["budget"], row["index"])] = row
        log(f"tp{row['tp']} b{row['budget']} #{row['index']} len {sum(r[0][1] for r in row['rounds'])}: {row['engine_s']:.2f} s")

load = lambda p: [json.loads(l) for l in (E / p).read_text().splitlines()]
earlier = {4: [dict(r, budget=r.get("budget", 128)) for p in ("E0012_tier1_prefill_engine/samples.jsonl",
                                                                "E0015_agent_tier1/long_prefill_samples.jsonl")
               for r in load(p)] + load("E0017_prefill_budgets/samples.jsonl"),
           8: load("E0020_tp8_surrogates/samples.jsonl")}
old_pre = {4: json.loads((E / "E0017_prefill_budgets/result.json").read_text())["prefill"],
           8: json.loads((E / "E0020_tp8_surrogates/result.json").read_text())["prefill"]}
old_dec = {4: json.loads((E / "E0022_decode_loaded_refit/decode_model.json").read_text()),
           8: json.loads((E / "E0026_tp8_decode_refit/decode_model.json").read_text())}
loaded_runs = {4: ["h200-p1x4-d3x4-ep16-b128-32-m32-32", "h200-p1x4-d3x4-ep16-b2048-32-m32-32"],
               8: ["h200-p1x8-d3x8-ep16-b2048-32-m32-32"]}
rel = lambda m, rounds, y: (sum(m.step_time(r) for r in rounds) - y) / y
out = {"experiment": "E0027", "source_identity": source_identity()}
for tp in (4, 8):
    long = [r for k, r in done.items() if k[0] == tp]
    old = tier1.Tier1Model(**old_pre[tp])
    extrap = [rel(old, s["rounds"], s["engine_s"]) for s in long]
    samples = earlier[tp] + long
    test = lambda i, s: i % 5 == 0
    new = tier1.fit_round_sums([s["rounds"] for i, s in enumerate(samples) if not test(i, s)],
                               [s["engine_s"] for i, s in enumerate(samples) if not test(i, s)],
                               dict(old.key, target="Tier-2 engine TTFT/rounds, budgets 128/512/2048, prompts to 40k"))
    held = [abs(rel(new, s["rounds"], s["engine_s"])) for i, s in enumerate(samples) if test(i, s)]
    held_long = [abs(rel(new, s["rounds"], s["engine_s"])) for i, s in enumerate(samples) if test(i, s) and s in long]
    in_long = [abs(rel(new, s["rounds"], s["engine_s"])) for s in long]
    # decode
    od = tier1.Tier1Model(**old_dec[tp])
    template = tier1._template("h200", tp, 16, "d", None)
    anchors = tier1.sample_batches("d", 120, 50 + tp, 32, 49152, 512)
    anchor_y = [tier1.bound_of(template, b, "h200") for b in anchors]
    long_idx = [i for i, b in enumerate(anchors) if max(kv for _, _, kv in b) > 16384]
    dec_old_long = [(od.step_time(anchors[i]) - anchor_y[i]) / anchor_y[i] for i in long_idx]
    loaded = []
    for run in loaded_runs[tp]:
        for rs in reconstruct_multi(E / f"E0018_controller_budget/tier2/{run}.jsonl").values():
            loaded += [([(i, 1, kv) for i, kv in zip(r.requests, r.kv_lens)], r.duration_s)
                       for r in rs if r.duration_s is not None and not r.first_decode]
    tr = [i for i in range(len(anchors)) if i % 5]
    nd = tier1.fit_samples([b for b, _ in loaded] + [anchors[i] for i in tr], [y for _, y in loaded] + [anchor_y[i] for i in tr],
                           dict(od.key, target="loaded Tier-2 decode rounds + graph-bound anchors (KV to 48k)"))
    dec_anchor_test = [abs(nd.step_time(anchors[i]) - anchor_y[i]) / anchor_y[i] for i in range(len(anchors)) if i % 5 == 0]
    dec_loaded = [abs(nd.step_time(b) - y) / y for b, y in loaded]
    out[f"tp{tp}"] = {
        "prefill_old_on_long": {"median_signed": statistics.median(extrap), "max_abs": max(map(abs, extrap)), "n": len(extrap)},
        "prefill_new": {"heldout_median": statistics.median(held), "heldout_max": max(held),
                        "long_in_sample_median": statistics.median(in_long), "long_max": max(in_long),
                        "long_heldout": held_long},
        "decode_old_long_kv_vs_bound": {"median_signed": statistics.median(dec_old_long), "max_abs": max(map(abs, dec_old_long))},
        "decode_new": {"anchor_heldout_median": statistics.median(dec_anchor_test), "anchor_heldout_max": max(dec_anchor_test),
                       "loaded_median": statistics.median(dec_loaded)},
        "prefill": asdict(new), "decode": asdict(nd)}
    log(f"TP{tp}", json.dumps({k: v for k, v in out[f'tp{tp}'].items() if k not in ('prefill', 'decode')}, indent=1))
(here / "result.json").write_text(json.dumps(out, indent=1))
