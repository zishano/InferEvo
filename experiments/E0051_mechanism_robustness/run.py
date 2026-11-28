"""E0051: robustness of the promoted mechanisms' gains to cost-model
assumptions (D11 ROBUST: the claim holds across the sensitivity band).

Each mechanism's incremental gain along the path the optimum uses, Tier-2:
  M-MOECHUNK     prefill PP4xEP4: k8 vs k1 (leader dispatch, bf16)
  M-RANKDISPATCH prefill PP4xEP4: ranks vs leader (k8, bf16);
                 prefill PP1xEP16: ranks vs leader (k1) — where it is network-bound
  M-FP8LINEAR    prefill PP4xEP4: fp8 vs bf16 (k8, ranks);
                 decode EP16 TP4: fp8 vs bf16 at a fixed round size of 64
under cost assumptions (validation/cost_profiles.json + network/launch sweeps):
  peak (default), h200_dense_v1 (E0006 fit), h200_pessimistic_v1 (lower band),
  network x0.5, network x0.25 (uncalibrated), 10 us per kernel, compute x0.5.
Prefill: P1:D1 TP4 budget 2048, 8 x 8192 burst, makespan. Decode: 64 x (64 in,
64 out) at once, TPOT p50 (peak concurrency must be 64 in both variants).
A claim is ROBUST if its gain is > 0 under every assumption.

    ./ie.sh experiments/E0051_mechanism_robustness/run.py [b300]

`b300` (Phase 6, mechanism transfer): same claims on B300 (native FP4/FP8,
288 GB, 8 TB/s) under peak, b300_transfer_v1 and the network/launch/compute
sweeps; outputs under tier2/b300/ and result-b300-*.json.
"""

import hashlib
import json
import statistics
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from adapters import steady, tier1_engine as te, workloads
from adapters.cost_model import cost_model_version
from adapters.engines import ROOT, SIMTHESIZER, source_identity
from adapters.topologies import Deployment, tier2_config

HW = sys.argv[1] if len(sys.argv) > 1 else "h200"
HERE = Path(__file__).parent
OUT = HERE / "tier2" / ("" if HW == "h200" else HW)
PROFILES = json.loads((ROOT / "validation" / "cost_profiles.json").read_text())
CALIBRATED = {"h200": ("h200_dense_v1", "h200_pessimistic_v1"), "b300": ("b300_transfer_v1",)}[HW]
ASSUMPTIONS = {
    "peak": {},
    **{name: PROFILES[name]["efficiency"] for name in CALIBRATED},
    "network_x0.5": {"network": 0.5},
    "network_x0.25": {"network": 0.25},
    "launch_10us": {"kernel_overhead_us": 10.0},
    "compute_x0.5": {"compute": 0.5},
}
PREFILL_WL = (8, 1000.0, 8192, 2, 44)
DECODE_WL = (64, 1000.0, 64, 64, 51)
FAST = {"moe_chunks": 8, "dispatch": "ranks"}
# variant: (deployment, prefill lowering options, decode lowering options, workload)
PP4 = Deployment(hardware=HW, prefill_ep=4, prefill_pp=4, prefill_budget=2048)
EP16 = Deployment(hardware=HW, prefill_budget=2048)
DEC = Deployment(hardware=HW, prefill_ep=4, prefill_pp=4, prefill_budget=2048, decode_budget=64, decode_max_batch=64)
VARIANTS = {
    "p4-base": (PP4, {}, {}, PREFILL_WL),
    "p4-mc8": (PP4, {"moe_chunks": 8}, {}, PREFILL_WL),
    "p4-mc8-rd": (PP4, FAST, {}, PREFILL_WL),
    "p4-mc8-rd-fp8": (PP4, {**FAST, "linear": "fp8"}, {}, PREFILL_WL),
    "p16-base": (EP16, {}, {}, PREFILL_WL),
    "p16-rd": (EP16, {"dispatch": "ranks"}, {}, PREFILL_WL),
    "d-bf16": (DEC, FAST, {}, DECODE_WL),
    "d-fp8": (DEC, FAST, {"linear": "fp8"}, DECODE_WL),
}
CLAIMS = {  # claim: (mechanism, variant with, variant without, metric)
    "MOECHUNK prefill PP4xEP4": ("M-MOECHUNK", "p4-mc8", "p4-base", "makespan_s"),
    "RANKDISPATCH prefill PP4xEP4 (+k8)": ("M-RANKDISPATCH", "p4-mc8-rd", "p4-mc8", "makespan_s"),
    "RANKDISPATCH prefill PP1xEP16": ("M-RANKDISPATCH", "p16-rd", "p16-base", "makespan_s"),
    "FP8LINEAR prefill PP4xEP4 (+k8, ranks)": ("M-FP8LINEAR", "p4-mc8-rd-fp8", "p4-mc8-rd", "makespan_s"),
    "FP8LINEAR decode EP16 TP4 b64": ("M-FP8LINEAR", "d-fp8", "d-bf16", "tpot_p50_ms"),
}


def lowering(opts):
    return {"kind": "grouped", **opts} if opts else "grouped"


def run(assumption, variant):
    dep, pre, dec, wl_spec = VARIANTS[variant]
    out = OUT / f"{assumption}-{variant}.jsonl"
    if not out.exists():
        cfg = tier2_config(dep, ASSUMPTIONS[assumption])
        for side, opts in (("prefill", pre), ("decode", dec)):
            for inst in cfg["system"][f"{side}_instances"]:
                inst["model_runner"]["lowering"] = lowering(opts)
        with tempfile.TemporaryDirectory(prefix="e0051-") as tmp:
            tmp = Path(tmp)
            te.write_config(cfg, tmp / "c.toml")
            wl = workloads.write(workloads.fixed(*wl_spec[:4], seed=wl_spec[4]), tmp / "w.jsonl")
            subprocess.run([str(SIMTHESIZER / "target/release/simthesizer"), "--config", str(tmp / "c.toml"),
                            "--workload", str(wl), "--output-jsonl", str(tmp / "r.jsonl")],
                           check=True, capture_output=True)
            OUT.mkdir(parents=True, exist_ok=True)
            out.write_text((tmp / "r.jsonl").read_text())
    cs = steady.calls(out)
    events = sorted([(c["emit"][0], 1) for c in cs] + [(c["emit"][-1], -1) for c in cs])
    live = peak = 0
    for _, step in events:
        live += step
        peak = max(peak, live)
    return {"makespan_s": max(c["emit"][0] for c in cs) - min(c["arrival"] for c in cs),
            "tpot_p50_ms": statistics.median(c["tpot"] for c in cs) * 1e3, "decode_concurrency": peak,
            "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}


def main():
    for a in ASSUMPTIONS:                         # launcher configs, serially
        for v in {VARIANTS[x][0].key: x for x in VARIANTS}.values():
            tier2_config(VARIANTS[v][0], ASSUMPTIONS[a])
    jobs = [(a, v) for a in ASSUMPTIONS for v in VARIANTS]
    with ThreadPoolExecutor(len(jobs)) as pool:
        res = dict(zip(jobs, pool.map(lambda j: run(*j), jobs)))
    bad = {a: [res[(a, v)]["decode_concurrency"] for v in ("d-bf16", "d-fp8")] for a in ASSUMPTIONS}
    bad = {a: c for a, c in bad.items() if c != [64, 64]}
    if bad:
        raise SystemExit(f"decode rounds not at 64 in both variants: {bad}")
    claims = {}
    for name, (mech, on, off, metric) in CLAIMS.items():
        gains = {a: 1 - res[(a, on)][metric] / res[(a, off)][metric] for a in ASSUMPTIONS}
        claims[name] = {"mechanism": mech, "hardware": HW, "gains": gains, "min": min(gains.values()), "max": max(gains.values()),
                        "robust": all(g > 0 for g in gains.values())}
        print(f"{name:42s} " + " ".join(f"{a}:{g:+.1%}" for a, g in gains.items())
              + f"  -> {'ROBUST' if claims[name]['robust'] else 'NOT robust'}", flush=True)
    cmv = cost_model_version()
    (HERE / (f"result-{cmv[:12]}.json" if HW == "h200" else f"result-{HW}-{cmv[:12]}.json")).write_text(json.dumps(
        {"experiment": "E0051", "hardware": HW, "cost_model_version": cmv, "source_identity": source_identity(),
         "assumptions": ASSUMPTIONS, "runs": {f"{a}/{v}": r for (a, v), r in res.items()}, "claims": claims}, indent=1))


if __name__ == "__main__":
    main()
