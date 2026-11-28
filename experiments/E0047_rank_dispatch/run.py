"""E0047: A-E evaluation of M-RANKDISPATCH (mechanisms/specs/M-RANKDISPATCH.toml).

Runs against the tree named by INFEREVO_SIMTHESIZER (the candidate). Tier-2.

  A  parent semantics: dispatch = leader (k = 1 and k = 8 as promoted)
  B  best existing prefill configuration: PP4 x EP4, moe_chunks 8 (E0044/E0045)
  C  mechanism on: dispatch = ranks, per layout, with k = 1 and k = 8
  D  co-search: best (layout, k, dispatch)
  E  controls: conservation (compute exact, copies within 0.1 %: local shares are free),
     decode side, 5 us launch cost

Prefill case and metric as E0044 (P1:D1 TP4, budget 2048, 8 x 8192 burst,
makespan). Decode control: 64 x (256 in, 128 out), TPOT p50; must reach >= 2
calls per attention group at once. Verdict VERIFIED iff conserved, dispatch = ranks cuts
makespan >= 5 % on some layout at equal k, and that gain stays > 0 with 5 us
launch cost. `--replay` re-runs the best case and compares bytes.
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
VARIANTS = {"leader-k1": {}, "ranks-k1": {"dispatch": "ranks"},
            "leader-k8": {"moe_chunks": 8}, "ranks-k8": {"dispatch": "ranks", "moe_chunks": 8}}
PREFILL_WL = (8, 1000.0, 8192, 2, 44)
# Short prompts so many calls decode at once (16 x 2048 left prefill ~0.6 s
# apart and decoded alone: nothing to split; r0 of this mechanism).
DECODE_WL = (64, 1000.0, 256, 128, 45)
GROUPS = 4                                    # attention groups per EP16 TP4 instance


def dep(layout, side="prefill"):
    ep, pp = LAYOUTS[layout]
    return Deployment(hardware="h200", prefill_budget=2048, **{f"{side}_ep": ep, f"{side}_pp": pp})


def lowering(variant):
    opts = VARIANTS[variant]
    return {"kind": "grouped", **opts} if opts else "grouped"


def config(d, side, variant, overhead=0.0):
    cfg = tier2_config(d)
    for inst in cfg["system"][f"{side}_instances"]:
        inst["model_runner"]["lowering"] = lowering(variant)
        if overhead:
            inst["model_runner"]["hardware"]["efficiency"] = {"kernel_overhead_us": overhead}
    return cfg


def run(name, cfg, wl_spec):
    out = OUT / f"{name}.jsonl"
    if not out.exists():
        with tempfile.TemporaryDirectory(prefix="e0047-") as tmp:
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
    events = sorted([(c["emit"][0], 1) for c in cs] + [(c["emit"][-1], -1) for c in cs])
    live = peak = 0
    for _, step in events:
        live += step
        peak = max(peak, live)
    return {"makespan_s": max(c["emit"][0] for c in cs) - first, "decode_concurrency": peak,
            "ttft_p50_s": statistics.median(c["ttft"] for c in cs),
            "tpot_p50_ms": statistics.median(c["tpot"] for c in cs) * 1e3,
            "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}


def conservation():
    enable_bridge_imports()
    from rooflang_bridge.kimi_k3_shared import KimiTemplate
    out = {}
    for layout, (ep, pp) in LAYOUTS.items():
        sums = {}
        for v in ("leader-k1", "ranks-k1"):
            t = KimiTemplate(4, ep, {"gpu_model": "h200", "gpu_count": ep * pp}, 0, stage="p", pp=pp,
                             lowering=lowering(v))
            ops = t.export([{"q_len": 2048, "kv_len": 0, "lm_head_len": 1}] * 2, [0, 1], details=False)["operations"]
            s = {}
            for op in ops:
                for d in op["work"]["demands"]:
                    for r, x in d["resources"].items():
                        s[r.split(":")[0]] = s.get(r.split(":")[0], 0.0) + x
            sums[v] = (s, len(ops))
        (a, na), (b, nb) = sums["leader-k1"], sums["ranks-k1"]
        out[layout] = {"rel_diff": {k: b.get(k, 0) / v - 1 for k, v in a.items() if v}, "operations": [na, nb]}
    return out


def main(replay=False):
    if replay:
        best = json.loads(max(HERE.glob("result-*.json"), key=lambda p: p.stat().st_mtime).read_text())["best"]
        moved = OUT / f"{best['case']}.jsonl.replayed-from"
        (OUT / f"{best['case']}.jsonl").rename(moved)
        got = run(best["case"], config(dep(best["layout"]), "prefill", best["variant"]), PREFILL_WL)["sha256"]
        moved.unlink()
        print(json.dumps({"verdict": "REPRODUCED" if got == best["sha256"] else "DIFFERS", "case": best["case"]}))
        return
    jobs = {f"p-{l}-{v}": (config(dep(l), "prefill", v), PREFILL_WL) for l in LAYOUTS for v in VARIANTS}
    jobs |= {f"d-PP1xEP16-{v}": (config(dep("PP1xEP16", "decode"), "decode", v), DECODE_WL)
             for v in ("leader-k1", "ranks-k1")}
    with ThreadPoolExecutor(len(jobs) + 1) as pool:
        futs = {n: pool.submit(run, n, *a) for n, a in jobs.items()}
        cons = pool.submit(conservation)
        res = {n: f.result() for n, f in futs.items()}
        cons = cons.result()
    ms = lambda l, v: res[f"p-{l}-{v}"]["makespan_s"]
    gain = {(l, k): 1 - ms(l, f"ranks-k{k}") / ms(l, f"leader-k{k}") for l in LAYOUTS for k in (1, 8)}
    (gl, gk), g = max(gain.items(), key=lambda kv: kv[1])
    with ThreadPoolExecutor(2) as pool:
        r = {v: pool.submit(run, f"p-{gl}-{v}-oh5", config(dep(gl), "prefill", v, 5.0), PREFILL_WL)
             for v in (f"leader-k{gk}", f"ranks-k{gk}")}
        r = {v: f.result() for v, f in r.items()}
    robust = {"layout": gl, "k": gk, "overhead_us": 5.0,
              "gain": 1 - r[f"ranks-k{gk}"]["makespan_s"] / r[f"leader-k{gk}"]["makespan_s"]}
    b = ("PP4xEP4", "leader-k8")
    d = min(((l, v) for l in LAYOUTS for v in VARIANTS), key=lambda lv: ms(*lv))
    # compute exact; copies may differ where a share is already local to its head
    concurrency = min(res[f"d-PP1xEP16-{v}"]["decode_concurrency"] for v in ("leader-k1", "ranks-k1"))
    if concurrency < 2 * GROUPS:
        raise SystemExit(f"decode control degenerate: peak concurrency {concurrency} < {2 * GROUPS}")
    conserved = all(abs(v["rel_diff"].get("compute", 0)) < 1e-9 and abs(v["rel_diff"].get("hbm", 0)) < 1e-3
                    for v in cons.values())
    ok = conserved and g >= 0.05 and robust["gain"] > 0
    best = {"case": f"p-{d[0]}-{d[1]}", "layout": d[0], "variant": d[1], "sha256": res[f"p-{d[0]}-{d[1]}"]["sha256"]}
    summary = {
        "A_parent": {l: {v: res[f"p-{l}-{v}"] for v in ("leader-k1", "leader-k8")} for l in LAYOUTS},
        "B_best_existing": {"layout": b[0], "variant": b[1], **res[f"p-{b[0]}-{b[1]}"]},
        "C_gain_ranks_vs_leader": {f"{l} k{k}": x for (l, k), x in gain.items()},
        "D_best_cosearch": {"layout": d[0], "variant": d[1], **res[f"p-{d[0]}-{d[1]}"],
                            "vs_B": 1 - ms(*d) / ms(*b)},
        "E_controls": {"conservation": cons, "launch_cost": robust,
                       "decode_tpot_ms": {v: res[f"d-PP1xEP16-{v}"]["tpot_p50_ms"] for v in ("leader-k1", "ranks-k1")}},
    }
    cmv = cost_model_version()
    (HERE / f"result-{cmv[:12]}.json").write_text(json.dumps(
        {"experiment": "E0047", "mechanism": "M-RANKDISPATCH", "cost_model_version": cmv,
         "source_identity": source_identity(), "runs": res, "summary": summary, "best": best}, indent=1))
    for l in LAYOUTS:
        print(l, " ".join(f"{v}: {ms(l, v)*1e3:.0f} ms" for v in VARIANTS),
              f"| ranks gain k1 {gain[(l, 1)]:+.1%} k8 {gain[(l, 8)]:+.1%}")
    print("decode TPOT", summary["E_controls"]["decode_tpot_ms"], "| launch-cost", robust,
          "| conservation", json.dumps({l: {k: f"{x:.1e}" for k, x in v["rel_diff"].items()} for l, v in cons.items()}))
    print(json.dumps({"verdict": "VERIFIED" if ok else "EXPLORATORY" if conserved else "FAILED",
                      "best": best, "max_gain": g, "D_vs_B": summary["D_best_cosearch"]["vs_B"],
                      "conserved": conserved, "launch_cost_gain": robust["gain"]}))


if __name__ == "__main__":
    main(replay="--replay" in sys.argv)
