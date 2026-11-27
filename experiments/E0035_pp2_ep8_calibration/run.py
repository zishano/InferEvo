"""E0035: load calibration of the E0034 PP2xEP8 TP4 surrogates, held-out check.

E0034 validation (run A: 24 x 2048/32 at 3 req/s) showed Tier-1 optimistic
under load (TTFT p50 -10 %, TPOT p50 -8 %): bound-fitted decode and no
contention (P->D handoff vs prefill on the same NICs, overlapping rounds).
Fit one multiplicative scale per side on run A (decode to TPOT p50, then
prefill to TTFT p50, two passes), then test on run B (24 x 4096/64 at
1.5 req/s, Tier-2, not used for fitting).

    ./ie.sh experiments/E0035_pp2_ep8_calibration/run.py
"""
import json, subprocess, tempfile
from dataclasses import asdict, replace
from pathlib import Path

from adapters import steady, tier1, tier1_engine as te, workloads
from adapters.engines import SIMTHESIZER, source_identity
from adapters.topologies import Deployment, tier2_config

here = Path(__file__).parent
E = here.parent
BIN = SIMTHESIZER / "target" / "release" / "simthesizer"
DEP = Deployment(prefill_ep=8, decode_ep=8, prefill_pp=2, decode_pp=2, prefill_budget=2048)
SLO = {"ttft_s": 5.0, "tpot_s": 0.05}
RUNS = {"A": (24, 3.0, 2048, 32, 34, E / "E0034_pp2_ep8_surrogates/tier2" / f"{DEP.key}.jsonl"),
        "B": (24, 1.5, 4096, 64, 35, here / "tier2" / f"{DEP.key}.jsonl")}
fit = json.loads((E / "E0034_pp2_ep8_surrogates/result.json").read_text())
base = {"p": tier1.Tier1Model(**fit["prefill"]), "d": tier1.Tier1Model(**fit["decode"])}
scaled = lambda m, k: replace(m, coef=[c * k for c in m.coef])


def metrics(run, models=None):
    n, rate, isl, osl, seed, out = RUNS[run]
    with tempfile.TemporaryDirectory(prefix="e0035-") as tmp:
        tmp = Path(tmp)
        wl = workloads.write(workloads.fixed(n, rate, isl, osl, seed=seed), tmp / "w.jsonl")
        if models is None:
            if not out.exists():
                te.write_config(tier2_config(DEP), tmp / "c.toml")
                subprocess.run([str(BIN), "--config", str(tmp / "c.toml"), "--workload", str(wl),
                                "--output-jsonl", str(tmp / "r.jsonl")], check=True, capture_output=True)
                out.parent.mkdir(exist_ok=True)
                out.write_text((tmp / "r.jsonl").read_text())
            m = steady.metrics(out, DEP.gpus, SLO, warm=0, cool=0)
        else:
            te.write_config(te.tier1_config(tier2_config(DEP), models), tmp / "c.toml")
            te.run(tmp / "c.toml", wl, tmp / "r.jsonl")
            m = steady.metrics(tmp / "r.jsonl", DEP.gpus, SLO, warm=0, cool=0)
    return {"ttft_p50": m["ttft_ms"]["p50"], "ttft_p99": m["ttft_ms"]["p99"], "tpot_p50": m["tpot_ms"]["p50"]}


def solve(target, value, lo=0.8, hi=1.5, steps=24):
    for _ in range(steps):           # value(k) increases with k
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if value(mid) < target else (lo, mid)
    return (lo + hi) / 2


if __name__ == "__main__":
    tier2 = {r: metrics(r) for r in RUNS}
    kp = kd = 1.0
    for _ in range(2):
        kd = solve(tier2["A"]["tpot_p50"], lambda k: metrics("A", {"p": scaled(base["p"], kp), "d": scaled(base["d"], k)})["tpot_p50"])
        kp = solve(tier2["A"]["ttft_p50"], lambda k: metrics("A", {"p": scaled(base["p"], k), "d": scaled(base["d"], kd)})["ttft_p50"])
    cal = {"p": scaled(base["p"], kp), "d": scaled(base["d"], kd)}
    for m, k, side in ((cal["p"], kp, "prefill"), (cal["d"], kd, "decode")):
        m.key = dict(m.key, target=m.key["target"] + f"; x{k:.4f} load calibration (E0035 run A)")
    rows = {}
    for r in RUNS:
        t1, t1c = metrics(r, base), metrics(r, cal)
        rows[r] = {"tier2": tier2[r], "tier1": t1, "tier1_calibrated": t1c,
                   "rel_err": {k: t1[k] / tier2[r][k] - 1 for k in t1},
                   "rel_err_calibrated": {k: t1c[k] / tier2[r][k] - 1 for k in t1c}}
        print(r, json.dumps({k: {m: round(v, 3) for m, v in rows[r][k].items()} for k in ("rel_err", "rel_err_calibrated")}))
    print(f"scales: prefill x{kp:.4f}, decode x{kd:.4f}")
    (here / "result.json").write_text(json.dumps({
        "experiment": "E0035", "scales": {"prefill": kp, "decode": kd}, "runs": rows,
        "prefill": asdict(cal["p"]), "decode": asdict(cal["d"]), "source_identity": source_identity()}, indent=1))
