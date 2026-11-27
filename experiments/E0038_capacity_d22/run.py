"""E0038: E0028 rerun under D22 capacity probes (duration-scaled, decode-backlog check).

Same 64xH200 space (P:D {1:3,2:2,3:1} x budget {128,512,2048} x TP {4,8},
EP16), workloads and SLOs as E0028; each probe now covers >= 300 s of
arrivals (short: >= 20 tiles, long: >= 600 requests, agent: >= 150 sessions)
and must show no decode backlog. Compares with E0028's capacities.

    ./ie.sh experiments/E0038_capacity_d22/run.py [workload ...]
"""
import json, math, sys, tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from adapters import steady, tier1_engine as te, workloads
from adapters.engines import source_identity
from adapters.topologies import Deployment, tier2_config
from agent.controller import models_for

here = Path(__file__).parent
DURATION = 300.0
WORKLOADS = {
    "short": {"make": lambda rate: workloads.tiled(max(20, math.ceil(rate * DURATION / 100)), rate, seed=7),
              "slo": {"ttft_s": 2.0, "tpot_s": 0.05}, "lo": 8.0, "hi": 32.0, "unit": "req/s"},
    "long": {"make": lambda rate: workloads.long_input(max(600, round(rate * DURATION)), rate, seed=8),
             "slo": {"ttft_s": 10.0, "tpot_s": 0.05}, "lo": 0.5, "hi": 2.0, "unit": "req/s"},
    "agent": {"make": lambda rate: workloads.agent_sessions(max(150, round(rate * DURATION)), turns=8, rate=rate, seed=9),
              "slo": {"ttft_s": 5.0, "tpot_s": 0.05}, "lo": 0.25, "hi": 1.0, "unit": "sessions/s"},
}
SPACE = [Deployment(prefill_instances=p, decode_instances=4 - p, prefill_budget=b, prefill_tp=t, decode_tp=t)
         for p in (1, 2, 3) for b in (128, 512, 2048) for t in (4, 8)]


def run_once(dep, name, rate):
    w = WORKLOADS[name]
    cfg = te.tier1_config(tier2_config(dep), models_for(dep))
    if name == "agent":
        cfg = dict(cfg, system={"kind": "agent_dag", "inner": cfg["system"]})
    with tempfile.TemporaryDirectory(prefix="e0038-") as tmp:
        tmp = Path(tmp)
        te.write_config(cfg, tmp / "c.toml")
        te.run(tmp / "c.toml", workloads.write(w["make"](rate), tmp / "w.jsonl"), tmp / "r.jsonl")
        return steady.metrics(tmp / "r.jsonl", dep.gpus, w["slo"])


def job(args):
    dep, name = args
    w = WORKLOADS[name]
    return dep.key, steady.capacity(lambda r: run_once(dep, name, r), w["lo"], w["hi"], steps=6)


if __name__ == "__main__":
    names = sys.argv[1:] or list(WORKLOADS)
    old = json.loads((here.parent / "E0028_steady_capacity/result.json").read_text())
    path = here / "result.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    for name in names:
        with ProcessPoolExecutor(18) as pool:
            res = dict(pool.map(job, [(d, name) for d in SPACE]))
        rank = sorted(res, key=lambda k: -res[k]["rate"])
        out[name] = {"slo": WORKLOADS[name]["slo"], "unit": WORKLOADS[name]["unit"], "ranking": rank,
                     "capacity": {k: {"rate": v["rate"], "at_rate": v.get("at_rate")} for k, v in res.items()},
                     "e0028_rate": {k: old[name]["capacity"][k]["rate"] for k in res},
                     "source_identity": source_identity()}
        print(f"\n== {name} (capacity, {WORKLOADS[name]['unit']}, 64 GPUs): D22 vs E0028", flush=True)
        for k in rank:
            a, was = res[k].get("at_rate"), old[name]["capacity"][k]["rate"]
            print("  %-36s %7.3f (E0028 %7.3f, %+5.0f%%)  %s" % (
                k, res[k]["rate"], was, (res[k]["rate"] / was - 1) * 100 if was else 0,
                "" if not a else "%5.1f tok/s/GPU TTFT p99 %6.0f ms TPOT p50 %5.1f ms" % (
                    a["tokens_per_s_gpu"], a["ttft_ms"]["p99"], a["tpot_ms"]["p50"])), flush=True)
        path.write_text(json.dumps(out, indent=1))
