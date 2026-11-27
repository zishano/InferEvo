"""E0034: Tier-1 surrogates for H200 PP2 x EP8, attention TP4 (16 GPUs/instance).

Prefill: Tier-2 engine samples (E0017 design) under budgets 128/512/2048:
  A) k requests, total <= budget, one round (mean TTFT);
  B) one prompt > budget -> sequential chunks (a request never overlaps itself).
Decode: exported-graph bounds (validation/round_bounds) of random batches,
<= 32 requests, KV to 48k. The model predicts a whole round (both stages and
the boundary send); the step runner splits it by layer share (47/46).
Validation: Tier-1 vs Tier-2 on one loaded P1:D1 workload, PP2xEP8 and the
PP1xEP16 baseline (existing surrogates) for scale.

    ./ie.sh experiments/E0034_pp2_ep8_surrogates/run.py
"""
import json, random, statistics, subprocess, tempfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from adapters import steady, tier1, tier1_engine as te, workloads
from adapters.cost_model import cost_model_version
from adapters.engines import SIMTHESIZER, source_identity
from adapters.topologies import CACHE, Deployment, tier2_config

here = Path(__file__).parent
BIN = SIMTHESIZER / "target" / "release" / "simthesizer"
TP, EP, PP = 4, 8, 2
KEY = {"hardware": "h200", "tp": TP, "ep": EP, "pp": PP, "groups": EP // TP, "efficiency": {},
       "cost_model_version": cost_model_version()}
log = lambda *a: print(*a, flush=True)
cache = here / "prefill_samples.jsonl"


def plans(budget, seed):
    rng = random.Random(seed)
    out = []
    for _ in range(12):
        k = rng.randint(1, 8)
        cuts = sorted(rng.sample(range(1, budget), k - 1)) if k > 1 else []
        total = rng.randint(k, budget)
        sizes = [max(1, round((b - a) * total / budget)) for a, b in zip([0, *cuts], [*cuts, budget])]
        out.append(("A", rng.sample(range(1000), k), sizes))
    for _ in range(6):
        out.append(("B", [rng.randrange(1000)], [round(budget * (40960 / budget) ** rng.random()) + 1]))
    return out


def engine(job):
    budget, idx, kind, ids, sizes, config = job
    with tempfile.TemporaryDirectory(prefix="e0034-") as tmp:
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
    return {"budget": budget, "index": idx, "kind": kind, "rounds": rounds, "engine_s": target}


def anchor(batch):
    template = tier1._template("h200", TP, EP, "d", None, pp=PP)
    return tier1.bound_of(template, batch, "h200")


def validate(dep, models, tier):
    """Loaded P1:D1 run; returns steady metrics."""
    cfg = tier2_config(dep) if tier == 2 else te.tier1_config(tier2_config(dep), models)
    with tempfile.TemporaryDirectory(prefix="e0034v-") as tmp:
        tmp = Path(tmp)
        te.write_config(cfg, tmp / "c.toml")
        wl = workloads.write(workloads.fixed(24, 3.0, 2048, 32, seed=34), tmp / "w.jsonl")
        out = here / "tier2" / f"{dep.key}.jsonl"
        if tier == 2 and not out.exists():
            subprocess.run([str(BIN), "--config", str(tmp / "c.toml"), "--workload", str(wl),
                            "--output-jsonl", str(tmp / "r.jsonl")], check=True, capture_output=True)
            out.parent.mkdir(exist_ok=True)
            out.write_text((tmp / "r.jsonl").read_text())
        if tier == 1:
            te.run(tmp / "c.toml", wl, tmp / "r.jsonl")
            out = tmp / "r.jsonl"
        return steady.metrics(out, dep.gpus, {"ttft_s": 5.0, "tpot_s": 0.05}, warm=0, cool=0)


if __name__ == "__main__":
    done = {(r["budget"], r["index"]): r for r in map(json.loads, cache.read_text().splitlines())} if cache.exists() else {}
    jobs = []
    for budget, seed in ((128, 40), (512, 41), (2048, 42)):
        dep = Deployment(prefill_ep=EP, decode_ep=EP, prefill_pp=PP, decode_pp=PP, prefill_budget=budget)
        tier2_config(dep)
        for idx, (kind, ids, sizes) in enumerate(plans(budget, seed)):
            if (budget, idx) not in done:
                jobs.append((budget, idx, kind, ids, sizes, CACHE / f"{dep.key}.toml"))
    pp_dep = Deployment(prefill_ep=EP, decode_ep=EP, prefill_pp=PP, decode_pp=PP, prefill_budget=2048)
    base_dep = Deployment(prefill_budget=2048)
    tier2_config(base_dep)
    with ThreadPoolExecutor(len(jobs) + 2) as pool:
        loaded = {d.key: pool.submit(validate, d, None, 2) for d in (pp_dep, base_dep)}
        for row in pool.map(engine, jobs):
            with cache.open("a") as f:
                f.write(json.dumps(row) + "\n")
            done[(row["budget"], row["index"])] = row
            log(f"b{row['budget']} #{row['index']} {row['kind']}: {row['engine_s'] * 1e3:.1f} ms")
        samples = sorted(done.values(), key=lambda r: (r["budget"], r["index"]))
        test = lambda s: s["index"] % 5 == 0
        pre = tier1.fit_round_sums([s["rounds"] for s in samples if not test(s)],
                                   [s["engine_s"] for s in samples if not test(s)],
                                   dict(KEY, stage="p", target="Tier-2 engine TTFT/rounds, budgets 128/512/2048, prompts to 40k"))
        rel = lambda s: (sum(pre.step_time(r) for r in s["rounds"]) - s["engine_s"]) / s["engine_s"]
        held = {b: [abs(rel(s)) for s in samples if test(s) and s["budget"] == b] for b in (128, 512, 2048)}
        batches = tier1.sample_batches("d", 120, 340, 32, 49152, 512)
        with ProcessPoolExecutor(24) as procs:
            ys = list(procs.map(anchor, batches))
        train = [i for i in range(len(batches)) if i % 5]
        dec = tier1.fit_samples([batches[i] for i in train], [ys[i] for i in train],
                                dict(KEY, stage="d", target="exported-graph bounds, <=32 requests, KV to 48k"))
        dec_held = [abs(dec.step_time(batches[i]) - ys[i]) / ys[i] for i in range(len(batches)) if i % 5 == 0]
        log("prefill held-out median/max per budget:",
            {b: (round(statistics.median(v), 4), round(max(v), 4)) for b, v in held.items()})
        log("decode held-out median/max:", round(statistics.median(dec_held), 4), round(max(dec_held), 4))
        from agent.controller import models_for
        compare = {}
        for d, models in ((pp_dep, {"p": pre, "d": dec}), (base_dep, models_for(base_dep))):
            t2, t1 = loaded[d.key].result(), validate(d, models, 1)
            compare[d.key] = {
                f"tier{n}": {"ttft_p50_ms": m["ttft_ms"]["p50"], "ttft_p99_ms": m["ttft_ms"]["p99"],
                             "tpot_p50_ms": m["tpot_ms"]["p50"], "tok_s_gpu": m["tokens_per_s_gpu"]}
                for n, m in ((1, t1), (2, t2))}
            log(d.key, json.dumps(compare[d.key]))
    (here / "result.json").write_text(json.dumps({
        "experiment": "E0034", "prefill": asdict(pre), "decode": asdict(dec),
        "prefill_heldout": {b: {"median": statistics.median(v), "max": max(v)} for b, v in held.items()},
        "decode_heldout": {"median": statistics.median(dec_held), "max": max(dec_held)},
        "validation_tier1_vs_tier2": compare, "source_identity": source_identity()}, indent=1))
