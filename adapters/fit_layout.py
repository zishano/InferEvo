"""Fit and load-calibrate Tier-1 surrogates for one side layout (E0034/E0035 method).

Layout = (hardware, attention TP, MoE ranks per stage, PP), P1:D1 with the same
layout on both sides.
  prefill: Tier-2 engine samples under budgets 128/512/2048 (k requests in one
           round -> mean TTFT; one long prompt -> sequential chunks);
  decode:  exported-graph bounds of random batches (<= max_requests, KV to 48k);
  calibration: one multiplicative scale per side fitted on loaded run A
           (TPOT p50 -> decode, TTFT p50 -> prefill), reported on held-out run B.
All Tier-2 outputs are kept under the experiment directory.
"""

from __future__ import annotations

import json
import multiprocessing
import random
import statistics
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

from adapters import steady, tier1, tier1_engine as te, workloads
from adapters.cost_model import cost_model_version
from adapters.engines import SIMTHESIZER
from adapters.topologies import CACHE, Deployment, tier2_config

BIN = SIMTHESIZER / "target" / "release" / "simthesizer"
BUDGETS = (128, 512, 2048)
SLO = {"ttft_s": 5.0, "tpot_s": 0.05}
RUN_A = (24, 3.0, 2048, 32, 34)      # requests, rate, ISL, OSL, seed
RUN_B = (24, 1.5, 4096, 64, 35)


def layout_dep(hw, tp, ep, pp, budget=2048, decode_batch=32, options=None, decode_options=None):
    """options / decode_options: per-side lowering options (mechanism variables)."""
    return Deployment(hardware=hw, prefill_tp=tp, decode_tp=tp, prefill_ep=ep, decode_ep=ep,
                      prefill_pp=pp, decode_pp=pp, prefill_budget=budget,
                      decode_budget=decode_batch, decode_max_batch=decode_batch,
                      **{f"prefill_{k}": v for k, v in (options or {}).items()},
                      **{f"decode_{k}": v for k, v in (decode_options or {}).items()})


def plans(budget, seed, short=12, long=6):
    rng = random.Random(seed)
    out = []
    for _ in range(short):
        k = rng.randint(1, 8)
        cuts = sorted(rng.sample(range(1, budget), k - 1)) if k > 1 else []
        total = rng.randint(k, budget)
        sizes = [max(1, round((b - a) * total / budget)) for a, b in zip([0, *cuts], [*cuts, budget])]
        out.append(("A", rng.sample(range(1000), k), sizes))
    for _ in range(long):
        out.append(("B", [rng.randrange(1000)], [round(budget * (40960 / budget) ** rng.random()) + 1]))
    return out


def _engine(job):
    budget, idx, kind, ids, sizes, config = job
    with tempfile.TemporaryDirectory(prefix="fit-") as tmp:
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


def _anchor(args):
    """Graph bound of one decode batch; None if it cannot fit in HBM (never admitted)."""
    hw, tp, ep, pp, batch, options = args
    try:
        return tier1.bound_of(tier1._template(hw, tp, ep, "d", None, pp=pp, options=options), batch, hw)
    except ValueError as exc:
        if "OOM" in str(exc) or "exceed" in str(exc):
            return None
        raise


def _loaded(dep, run, out, models=None):
    n, rate, isl, osl, seed = run
    with tempfile.TemporaryDirectory(prefix="fitv-") as tmp:
        tmp = Path(tmp)
        wl = workloads.write(workloads.fixed(n, rate, isl, osl, seed=seed), tmp / "w.jsonl")
        if models is None:
            if not out.exists():
                te.write_config(tier2_config(dep), tmp / "c.toml")
                subprocess.run([str(BIN), "--config", str(tmp / "c.toml"), "--workload", str(wl),
                                "--output-jsonl", str(tmp / "r.jsonl")], check=True, capture_output=True)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text((tmp / "r.jsonl").read_text())
            path = out
        else:
            te.write_config(te.tier1_config(tier2_config(dep), models), tmp / "c.toml")
            te.run(tmp / "c.toml", wl, tmp / "r.jsonl")
            path = tmp / "r.jsonl"
        m = steady.metrics(path, dep.gpus, SLO, warm=0, cool=0)
    return {"ttft_p50": m["ttft_ms"]["p50"], "ttft_p99": m["ttft_ms"]["p99"], "tpot_p50": m["tpot_ms"]["p50"]}


def _scaled(m, k):
    return replace(m, coef=[c * k for c in m.coef])


def _solve(target, value, lo=0.7, hi=1.6, steps=22):
    for _ in range(steps):           # value(k) increases with k
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if value(mid) < target else (lo, mid)
    return (lo + hi) / 2


def fit(here: Path, hw: str, tp: int, ep: int, pp: int, *, decode_requests: int = 32,
        prefill: dict | None = None, threads: int = 40, runs=(RUN_A, RUN_B), log=print,
        options: dict | None = None, decode_options: dict | None = None) -> dict:
    """Fit (or reuse a fixed `prefill` model, then only decode is calibrated)
    and calibrate; writes here/result.json. Returns it. `options` /
    `decode_options` = per-side lowering options (mechanism variables)."""
    options, decode_options = options or {}, decode_options or {}
    run_a, run_b = runs
    key = {"hardware": hw, "tp": tp, "ep": ep, "pp": pp, "groups": ep // tp, "efficiency": {},
           "cost_model_version": cost_model_version()}
    dep = layout_dep(hw, tp, ep, pp, decode_batch=decode_requests, options=options, decode_options=decode_options)
    cache = here / "prefill_samples.jsonl"
    done = {(r["budget"], r["index"]): r for r in map(json.loads, cache.read_text().splitlines())} \
        if cache.exists() else {}
    jobs = []
    if prefill is None:
        (here / "configs").mkdir(parents=True, exist_ok=True)
        for budget, seed in zip(BUDGETS, (40, 41, 42)):
            d = layout_dep(hw, tp, ep, pp, budget, options=options)
            path = here / "configs" / f"{d.key}.toml"      # per-side lowering applied
            te.write_config(tier2_config(d), path)
            jobs += [(budget, i, *p, path) for i, p in enumerate(plans(budget, seed))
                     if (budget, i) not in done]
    tier2_config(dep)
    # Decode anchors first: forking (fork: forkserver/spawn would re-import the
    # caller's __main__) while Tier-2 threads run can deadlock the children.
    batches = tier1.sample_batches("d", 120, 340 + tp + ep, decode_requests, 49152, 512)
    with ProcessPoolExecutor(24, mp_context=multiprocessing.get_context("fork")) as procs:
        ys = list(procs.map(_anchor, [(hw, tp, ep, pp, b, decode_options) for b in batches]))
    batches, ys = zip(*[(b, y) for b, y in zip(batches, ys) if y is not None])
    with ThreadPoolExecutor(max(2, min(threads, len(jobs) + 2))) as pool:
        loaded = {r: pool.submit(_loaded, dep, run, here / "tier2" / f"{r}-{dep.key}.jsonl")
                  for r, run in (("A", run_a), ("B", run_b))}
        for row in pool.map(_engine, jobs):
            with cache.open("a") as f:
                f.write(json.dumps(row) + "\n")
            done[(row["budget"], row["index"])] = row
            log(f"  {hw} TP{tp} EP{ep} PP{pp} b{row['budget']} #{row['index']} {row['kind']}: {row['engine_s'] * 1e3:.1f} ms")
        result = {"layout": key, "decode_anchors": len(ys)}
        if prefill is None:
            samples = sorted(done.values(), key=lambda r: (r["budget"], r["index"]))
            test = lambda s: s["index"] % 5 == 0
            pre = tier1.fit_round_sums([s["rounds"] for s in samples if not test(s)],
                                       [s["engine_s"] for s in samples if not test(s)],
                                       dict(key, stage="p", target="Tier-2 engine TTFT/rounds, budgets 128/512/2048, prompts to 40k",
                                            **({"options": options} if options else {})))
            rel = lambda s: abs(sum(pre.step_time(r) for r in s["rounds"]) - s["engine_s"]) / s["engine_s"]
            result["prefill_heldout"] = {b: {"median": statistics.median(v), "max": max(v)} for b in BUDGETS
                                         for v in [[rel(s) for s in samples if test(s) and s["budget"] == b]]}
        else:
            pre = tier1.Tier1Model(**prefill)
        train = [i for i in range(len(batches)) if i % 5]
        dec = tier1.fit_samples([batches[i] for i in train], [ys[i] for i in train],
                                dict(key, stage="d", max_requests=decode_requests,
                                     target=f"exported-graph bounds, <= {decode_requests} requests, KV to 48k",
                                     **({"options": decode_options} if decode_options else {})))
        held = [abs(dec.step_time(batches[i]) - ys[i]) / ys[i] for i in range(len(batches)) if i % 5 == 0]
        result["decode_heldout"] = {"median": statistics.median(held), "max": max(held)}
        tier2 = {r: f.result() for r, f in loaded.items()}
    kp = kd = 1.0
    for _ in range(2 if prefill is None else 1):
        kd = _solve(tier2["A"]["tpot_p50"], lambda k: _loaded(dep, run_a, None, {"p": _scaled(pre, kp), "d": _scaled(dec, k)})["tpot_p50"])
        if prefill is None:
            kp = _solve(tier2["A"]["ttft_p50"], lambda k: _loaded(dep, run_a, None, {"p": _scaled(pre, k), "d": _scaled(dec, kd)})["ttft_p50"])
    cal = {"p": _scaled(pre, kp), "d": _scaled(dec, kd)}
    for m, k in ((cal["p"], kp), (cal["d"], kd)):
        m.key = dict(m.key, target=m.key["target"] + f"; x{k:.4f} load calibration (run A)")
    runs = {}
    for r, run in (("A", run_a), ("B", run_b)):
        t1 = _loaded(dep, run, None, {"p": pre, "d": dec})
        t1c = _loaded(dep, run, None, cal)
        runs[r] = {"tier2": tier2[r], "rel_err": {k: t1[k] / tier2[r][k] - 1 for k in t1},
                   "rel_err_calibrated": {k: t1c[k] / tier2[r][k] - 1 for k in t1c}}
    result.update({"scales": {"prefill": kp, "decode": kd}, "runs": runs,
                   "prefill": asdict(cal["p"]), "decode": asdict(cal["d"])})
    (here / "result.json").write_text(json.dumps(result, indent=1))
    log(f"{hw} TP{tp} EP{ep} PP{pp}: scales p x{kp:.3f} d x{kd:.3f}; held-out B calibrated "
        + json.dumps({k: round(v, 3) for k, v in runs["B"]["rel_err_calibrated"].items()}))
    return result
