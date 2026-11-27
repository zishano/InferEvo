"""E0028: steady-state SLO capacity of 64xH200 deployments on three workloads.

Workloads (defaults; synthetic shapes are assumptions):
  short : realistic token shapes x20 (2000 requests), Poisson; SLO TTFT<=2 s, TPOT<=50 ms
  long  : 600 requests, input log-uniform 4k-32k, output 100-1000, Poisson; SLO TTFT<=10 s, TPOT<=50 ms
  agent : 150 coding-agent sessions x 8 turns (context 2k -> ~20k, tools 0.2-3 s), Poisson sessions; SLO TTFT<=5 s, TPOT<=50 ms
Space: P:D {1:3,2:2,3:1} x prefill budget {128,512,2048} x attention TP {4,8} (equal P/D), EP16.
Metric: capacity = highest offered rate with >= 90 % of steady-window calls in SLO and no queue growth
(adapters/steady.py); reported per GPU plus goodput/TTFT/TPOT at that rate. Tier-1 (E0027 surrogates).

    ./ie.sh experiments/E0028_steady_capacity/run.py [workload ...]
"""
import json, sys, tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from adapters import steady, tier1_engine as te, workloads
from adapters.engines import source_identity
from adapters.topologies import Deployment, tier2_config
from agent.controller import models_for

here = Path(__file__).parent
WORKLOADS = {
    "short": {"make": lambda rate: workloads.tiled(20, rate, seed=7), "slo": {"ttft_s": 2.0, "tpot_s": 0.05},
              "lo": 8.0, "hi": 32.0, "unit": "req/s"},
    "long": {"make": lambda rate: workloads.long_input(600, rate, seed=8), "slo": {"ttft_s": 10.0, "tpot_s": 0.05},
             "lo": 0.5, "hi": 2.0, "unit": "req/s"},
    "agent": {"make": lambda rate: workloads.agent_sessions(150, turns=8, rate=rate, seed=9),
              "slo": {"ttft_s": 5.0, "tpot_s": 0.05}, "lo": 0.25, "hi": 1.0, "unit": "sessions/s"},
}
SPACE = [Deployment(prefill_instances=p, decode_instances=4 - p, prefill_budget=b, prefill_tp=t, decode_tp=t)
         for p in (1, 2, 3) for b in (128, 512, 2048) for t in (4, 8)]


def run_once(dep, name, rate):
    w = WORKLOADS[name]
    cfg = te.tier1_config(tier2_config(dep), models_for(dep))
    if name == "agent":                       # agent programs via the agent_dag front end (D20)
        cfg = dict(cfg, system={"kind": "agent_dag", "inner": cfg["system"]})
    with tempfile.TemporaryDirectory(prefix="e0028-") as tmp:
        tmp = Path(tmp)
        te.write_config(cfg, tmp / "c.toml")
        te.run(tmp / "c.toml", workloads.write(w["make"](rate), tmp / "w.jsonl"), tmp / "r.jsonl")
        return steady.metrics(tmp / "r.jsonl", dep.gpus, w["slo"])


def job(args):
    dep, name = args
    w = WORKLOADS[name]
    cap = steady.capacity(lambda r: run_once(dep, name, r), w["lo"], w["hi"], steps=6)
    return dep.key, cap


if __name__ == "__main__":
    names = sys.argv[1:] or list(WORKLOADS)
    path = here / "result.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    for name in names:
        with ProcessPoolExecutor(18) as pool:
            res = dict(pool.map(job, [(d, name) for d in SPACE]))
        rank = sorted(res, key=lambda k: -res[k]["rate"])
        out[name] = {"slo": WORKLOADS[name]["slo"], "unit": WORKLOADS[name]["unit"], "ranking": rank,
                     "capacity": res, "source_identity": source_identity()}
        print(f"\n== {name} (capacity, {WORKLOADS[name]['unit']}, 64 GPUs)")
        for k in rank:
            c = res[k]
            a = c.get("at_rate")
            print("  %-36s %7.3f  %s" % (k, c["rate"], "" if not a else
                  "goodput %6.2f tok/s/GPU TTFT p99 %7.0f ms TPOT p50 %5.1f ms attain %.2f" % (
                      a["goodput_tokens_per_s_gpu"], a["ttft_ms"]["p99"], a["tpot_ms"]["p50"], a["slo_attainment"])),
                  flush=True)
        path.write_text(json.dumps(out, indent=1))
