"""E0044: A-E evaluation of M-MOECHUNK (mechanisms/specs/M-MOECHUNK.toml).

Runs against the engine tree named by INFEREVO_SIMTHESIZER (the candidate,
set by `mechanisms.manage evaluate`). Tier-2 (grouped reference lowering).

  A  parent semantics: moe_chunks = 1 on every case (the parent lowering;
     byte identity with the parent engine is already a `check` receipt)
  B  best existing prefill layout at k = 1 (PP4 x EP4, E0040/MC-5 optimum)
  C  mechanism on: k in {2, 4, 8} on each prefill layout
  D  co-search of the new variable with the layout (best (layout, k))
  E  controls: work conservation (per-resource totals k=4 vs k=1), decode
     side (expected no gain), launch-cost robustness (CM-4b 5 us/kernel)

Prefill case: P1:D1 TP4, budget 2048, burst of 8 x 8192-token prompts
(OSL 2); metric = prefill makespan (last first token - first arrival) and
TTFT p50. Decode control: 16 x (2048 in, 64 out), TPOT p50.
Verdict VERIFIED iff conservation holds, some k > 1 cuts makespan >= 5 % on
some layout, and that gain stays > 0 with 5 us launch cost. Prints one JSON
verdict line last. `--replay` re-runs the best case and compares bytes.
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
from adapters.engines import SIMTHESIZER, enable_bridge_imports, source_identity
from adapters.topologies import Deployment, tier2_config

HERE = Path(__file__).parent
OUT = HERE / "tier2"
LAYOUTS = {"PP1xEP16": (16, 1), "PP2xEP8": (8, 2), "PP4xEP4": (4, 4)}
KS = (1, 2, 4, 8)
PREFILL_WL = (8, 1000.0, 8192, 2, 44)
DECODE_WL = (16, 1000.0, 2048, 64, 45)


def dep(layout, side="prefill"):
    ep, pp = LAYOUTS[layout]
    return Deployment(hardware="h200", prefill_budget=2048,
                      **{f"{side}_ep": ep, f"{side}_pp": pp})


def config(d, side, k, overhead=0.0):
    cfg = tier2_config(d)
    for inst in cfg["system"][f"{side}_instances"]:
        runner = inst["model_runner"]
        if k > 1:
            runner["lowering"] = {"kind": "grouped", "moe_chunks": k}
        if overhead:
            runner["hardware"]["efficiency"] = {"kernel_overhead_us": overhead}
    return cfg


def run(name, cfg, wl_spec):
    out = OUT / f"{name}.jsonl"
    if not out.exists():
        with tempfile.TemporaryDirectory(prefix="e0044-") as tmp:
            tmp = Path(tmp)
            te.write_config(cfg, tmp / "c.toml")
            wl = workloads.write(workloads.fixed(*wl_spec[:4], seed=wl_spec[4]), tmp / "w.jsonl")
            subprocess.run([str(SIMTHESIZER / "target/release/simthesizer"), "--config", str(tmp / "c.toml"),
                            "--workload", str(wl), "--output-jsonl", str(tmp / "r.jsonl")],
                           check=True, capture_output=True)
            OUT.mkdir(exist_ok=True)
            out.write_text((tmp / "r.jsonl").read_text())
    cs = steady.calls(out)
    first = min(c["arrival"] for c in cs)
    return {"makespan_s": max(c["emit"][0] for c in cs) - first,
            "ttft_p50_s": statistics.median(c["ttft"] for c in cs),
            "tpot_p50_ms": statistics.median(c["tpot"] for c in cs) * 1e3,
            "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}


def conservation():
    """Per-resource-kind totals of one prefill round, k = 4 vs k = 1."""
    enable_bridge_imports()
    from rooflang_bridge.kimi_k3_shared import KimiTemplate
    out = {}
    for layout, (ep, pp) in LAYOUTS.items():
        totals = {}
        for k in (1, 4):
            lowering = "grouped" if k == 1 else {"kind": "grouped", "moe_chunks": k}
            t = KimiTemplate(4, ep, {"gpu_model": "h200", "gpu_count": ep * pp}, 0, stage="p", pp=pp,
                             lowering=lowering)
            ops = t.export([{"q_len": 2048, "kv_len": 0, "lm_head_len": 1}] * 2, [0, 1], details=False)["operations"]
            sums, transfers = {}, 0
            for op in ops:
                for d in op["work"]["demands"]:
                    for r, v in d["resources"].items():
                        kind = r.split(":")[0]
                        sums[kind] = sums.get(kind, 0.0) + v
            totals[k] = {"resources": sums, "operations": len(ops)}
        rel = {kind: abs(totals[4]["resources"].get(kind, 0) / v - 1) for kind, v in totals[1]["resources"].items() if v}
        out[layout] = {"max_rel_diff": max(rel.values()), "by_kind": rel,
                       "operations": {k: v["operations"] for k, v in totals.items()}}
    return out


def main(replay=False):
    if replay:
        best = json.loads(max(HERE.glob("result-*.json"), key=lambda p: p.stat().st_mtime).read_text())["best"]
        stored = best["sha256"]
        (OUT / f"{best['case']}.jsonl").rename(OUT / f"{best['case']}.jsonl.replayed-from")
        layout, k = best["layout"], best["k"]
        got = run(best["case"], config(dep(layout), "prefill", k), PREFILL_WL)["sha256"]
        (OUT / f"{best['case']}.jsonl.replayed-from").unlink()
        print(json.dumps({"verdict": "REPRODUCED" if got == stored else "DIFFERS", "case": best["case"]}))
        return
    jobs = {}
    for layout in LAYOUTS:
        for k in KS:
            jobs[f"p-{layout}-k{k}"] = (config(dep(layout), "prefill", k), PREFILL_WL)
    for k in (1, 4):
        jobs[f"d-PP1xEP16-k{k}"] = (config(dep("PP1xEP16", "decode"), "decode", k), DECODE_WL)
    with ThreadPoolExecutor(len(jobs) + 2) as pool:
        futs = {n: pool.submit(run, n, *a) for n, a in jobs.items()}
        cons = pool.submit(conservation)
        res = {n: f.result() for n, f in futs.items()}
        cons = cons.result()
    gain = {(layout, k): 1 - res[f"p-{layout}-k{k}"]["makespan_s"] / res[f"p-{layout}-k1"]["makespan_s"]
            for layout in LAYOUTS for k in KS}
    (bl, bk), bgain = max(gain.items(), key=lambda kv: kv[1])
    robust = {}
    if bk > 1:
        with ThreadPoolExecutor(2) as pool:
            r = {k: pool.submit(run, f"p-{bl}-k{k}-oh5", config(dep(bl), "prefill", k, 5.0), PREFILL_WL)
                 for k in (1, bk)}
            r = {k: f.result() for k, f in r.items()}
        robust = {"layout": bl, "k": bk, "overhead_us": 5.0, "gain": 1 - r[bk]["makespan_s"] / r[1]["makespan_s"]}
    b_layout = min(LAYOUTS, key=lambda l: res[f"p-{l}-k1"]["makespan_s"])
    d_layout, d_k = min(((l, k) for l in LAYOUTS for k in KS), key=lambda lk: res[f"p-{lk[0]}-k{lk[1]}"]["makespan_s"])
    conserved = all(v["max_rel_diff"] < 1e-9 for v in cons.values())
    ok = conserved and bgain >= 0.05 and robust.get("gain", 0) > 0
    summary = {
        "A_parent_k1": {l: res[f"p-{l}-k1"] for l in LAYOUTS},
        "B_best_existing": {"layout": b_layout, **res[f"p-{b_layout}-k1"]},
        "C_gain_by_layout_k": {f"{l} k{k}": g for (l, k), g in gain.items()},
        "D_best_cosearch": {"layout": d_layout, "k": d_k, **res[f"p-{d_layout}-k{d_k}"],
                            "vs_B": 1 - res[f"p-{d_layout}-k{d_k}"]["makespan_s"] / res[f"p-{b_layout}-k1"]["makespan_s"]},
        "E_controls": {"conservation": cons, "decode_tpot_ms": {k: res[f"d-PP1xEP16-k{k}"]["tpot_p50_ms"] for k in (1, 4)},
                       "launch_cost": robust},
    }
    best = {"case": f"p-{bl}-k{bk}", "layout": bl, "k": bk, "gain": bgain, "sha256": res[f"p-{bl}-k{bk}"]["sha256"]}
    cmv = cost_model_version()
    (HERE / f"result-{cmv[:12]}.json").write_text(json.dumps(
        {"experiment": "E0044", "mechanism": "M-MOECHUNK", "cost_model_version": cmv,
         "source_identity": source_identity(), "runs": res, "summary": summary, "best": best}, indent=1))
    for l in LAYOUTS:
        print(l, " ".join(f"k{k}: {res[f'p-{l}-k{k}']['makespan_s']*1e3:.0f} ms ({gain[(l, k)]:+.1%})" for k in KS))
    print("decode TPOT p50 ms", summary["E_controls"]["decode_tpot_ms"], "| launch-cost", robust,
          "| conservation max", max(v["max_rel_diff"] for v in cons.values()))
    print(json.dumps({"verdict": "VERIFIED" if ok else "EXPLORATORY" if conserved else "FAILED",
                      "best": best, "D_vs_B": summary["D_best_cosearch"]["vs_B"], "conserved": conserved,
                      "launch_cost_gain": robust.get("gain")}))


if __name__ == "__main__":
    main(replay="--replay" in sys.argv)
