"""Tier-1 load sweep for a zoo layout (E0014 models), same metrics as run.py.

    ./ie.sh experiments/E0013_load_sweep/sweep.py b300 4
"""
import json, sys, tempfile, time
from pathlib import Path
from adapters import tier1, tier1_engine as te, workloads
from adapters.engines import ROOT, source_identity
from run import MULTIPLIERS, metrics

hw, tp = sys.argv[1], int(sys.argv[2])
zoo = ROOT / "experiments" / "E0014_tier1_zoo" / f"{hw}-tp{tp}"
z = json.loads((zoo / "result.json").read_text())
models = {"p": tier1.Tier1Model(**z["prefill"]), "d": tier1.Tier1Model(**z["decode"])}
cfg = te.tier1_config(te.load(zoo / "tier2_config.toml"), models)
rows_out = []
with tempfile.TemporaryDirectory(prefix="sweep-") as tmp:
    te.write_config(cfg, Path(tmp) / "config.toml")
    for mult in MULTIPLIERS:
        rows = workloads.scaled(mult)
        wl = workloads.write(rows, Path(tmp) / f"w{mult}.jsonl")
        te.run(Path(tmp) / "config.toml", wl, Path(tmp) / f"r{mult}.jsonl")
        res = metrics(Path(tmp) / f"r{mult}.jsonl", rows)
        res.update(multiplier=mult, offered_req_s=workloads.offered_rate(rows))
        rows_out.append(res)
        print(f"{hw} tp{tp} x{mult:<4} {res['tokens_per_s_gpu']:7.2f} tok/s/GPU {res['tokens_per_s_user']:6.1f} tok/s/user "
              f"TTFT p50/p99 {res['ttft_ms']['p50']:7.1f}/{res['ttft_ms']['p99']:7.1f} ms TPOT p50 {res['tpot_ms']['p50']:5.2f} ms", flush=True)
(Path(__file__).with_name(f"result-{hw}-tp{tp}.json")).write_text(json.dumps(
    {"experiment": "E0013", "tier": 1, "layout": f"{hw} TP{tp}/EP16 P1+D1", "rows": rows_out,
     "models": {k: v.key for k, v in models.items()}, "source_identity": source_identity()}, indent=1))
