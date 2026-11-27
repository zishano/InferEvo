"""E0029: Tier-2 confirmation of E0028 (steady-state capacity).

Per workload, 3 deployments at their own E0028 capacity rate, on reduced
workloads (Tier-2 cost): short 300 requests, long 60 requests, agent 20
sessions x 8 turns. Tier-1 and Tier-2 run the identical workload file;
compare steady-window metrics and the ranking. Tier-2 result.jsonl kept
(tier2/), everything else in temp dirs.

    ./ie.sh experiments/E0029_steady_tier2/run.py
"""
import json, subprocess, sys, tempfile, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from adapters import steady, tier1_engine as te, workloads
from adapters.engines import SIMTHESIZER, source_identity
from adapters.topologies import CACHE, Deployment, tier2_config
from agent.controller import models_for

here = Path(__file__).parent
cap = json.loads((here.parent / "E0028_steady_capacity/result.json").read_text())
SLO = {k: v["slo"] for k, v in cap.items()}
MAKE = {"short": lambda r: workloads.tiled(3, r, seed=17),
        "long": lambda r: workloads.long_input(60, r, seed=18),
        "agent": lambda r: workloads.agent_sessions(20, turns=8, rate=r, seed=19)}
D = lambda p, b, t=4: Deployment(prefill_instances=p, decode_instances=4 - p, prefill_budget=b, prefill_tp=t, decode_tp=t)
POINTS = {"short": [D(1, 2048), D(2, 2048, 8), D(1, 128)],
          "long": [D(3, 2048), D(2, 2048), D(1, 2048)],
          "agent": [D(3, 2048), D(2, 2048), D(1, 128)]}
BIN = SIMTHESIZER / "target/release/simthesizer"


def wrap(cfg, name):
    return {"system": {"kind": "agent_dag", "inner": cfg["system"]}} if name == "agent" else cfg


def job(args):
    name, dep = args
    rate = cap[name]["capacity"][dep.key]["rate"]
    keep = here / "tier2" / f"{name}-{dep.key}.jsonl"
    out = {"workload": name, "key": dep.key, "rate": rate}
    with tempfile.TemporaryDirectory(prefix="e0029-") as tmp:
        tmp = Path(tmp)
        wl = workloads.write(MAKE[name](rate), tmp / "w.jsonl")
        t2 = wrap(tier2_config(dep), name)
        te.write_config(t2, tmp / "t2.toml")
        t1 = wrap(te.tier1_config(tier2_config(dep), models_for(dep)), name)
        te.write_config(t1, tmp / "t1.toml")
        te.run(tmp / "t1.toml", wl, tmp / "t1.jsonl")
        out["tier1"] = steady.metrics(tmp / "t1.jsonl", dep.gpus, SLO[name])
        if not keep.exists():
            t0 = time.perf_counter()
            subprocess.run([str(BIN), "--config", str(tmp / "t2.toml"), "--workload", str(wl),
                            "--output-jsonl", str(tmp / "t2.jsonl")], check=True, capture_output=True)
            out["tier2_wall_s"] = time.perf_counter() - t0
            keep.write_text((tmp / "t2.jsonl").read_text())
    out["tier2"] = steady.metrics(keep, dep.gpus, SLO[name])
    return out


if __name__ == "__main__":
    (here / "tier2").mkdir(exist_ok=True)
    jobs = [(n, d) for n, ds in POINTS.items() for d in ds]
    res = []
    with ProcessPoolExecutor(len(jobs)) as pool:
        for r in pool.map(job, jobs):
            res.append(r)
            a, b = r["tier1"], r["tier2"]
            print("%-5s %-36s rate %.3f | attain T1 %.2f T2 %.2f | goodput T1 %6.2f T2 %6.2f | TTFT p50 T1 %7.0f T2 %7.0f | TPOT p50 T1 %5.1f T2 %5.1f | wall %.1f h" % (
                r["workload"], r["key"], r["rate"], a["slo_attainment"], b["slo_attainment"],
                a["goodput_tokens_per_s_gpu"], b["goodput_tokens_per_s_gpu"], a["ttft_ms"]["p50"], b["ttft_ms"]["p50"],
                a["tpot_ms"]["p50"], b["tpot_ms"]["p50"], r.get("tier2_wall_s", 0) / 3600), flush=True)
    (here / "result.json").write_text(json.dumps({"experiment": "E0029", "points": res,
                                                  "source_identity": source_identity()}, indent=1))
