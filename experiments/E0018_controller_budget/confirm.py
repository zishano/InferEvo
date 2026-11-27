"""Tier-2 confirmation of E0018 points (shared Rooflang, same configs).

Keeps result.jsonl per point (tier2/<key>.jsonl) for later decode refits;
all other run output lives in a temporary directory that is always removed.

    ./ie.sh experiments/E0018_controller_budget/confirm.py
"""
import json, subprocess, tempfile, time
from pathlib import Path
from adapters import feedback, workloads
from adapters.engines import SIMTHESIZER, source_identity
from adapters.topologies import CACHE, Deployment, tier2_config

here = Path(__file__).parent
t1 = {e["key"]: e for e in json.loads((here / "result.json").read_text())["realistic-x4"]["grid"]}
t1 |= {k: {"key": k, **o} for k, o in
       json.loads((here.parent / "E0021_tp_space/result.json").read_text())["grid"].items()}
POINTS = [Deployment(prefill_instances=1, decode_instances=3, prefill_budget=b) for b in (128, 2048)]
POINTS.append(Deployment(prefill_instances=1, decode_instances=3, prefill_budget=2048,
                         prefill_tp=8, decode_tp=8))      # E0021 goodput-best
out = json.loads((here / "confirm.json").read_text()) if (here / "confirm.json").exists() else {}
for dep in POINTS:
    if dep.key in out:
        continue
    tier2_config(dep)
    keep = here / "tier2" / f"{dep.key}.jsonl"
    rows = workloads.scaled(4)
    with tempfile.TemporaryDirectory(prefix="e0018c-") as tmp:
        wl = workloads.write(rows, Path(tmp) / "w.jsonl")
        t0 = time.perf_counter()
        subprocess.run([str(SIMTHESIZER / "target/release/simthesizer"), "--config", str(CACHE / f"{dep.key}.toml"),
                        "--workload", str(wl), "--output-jsonl", str(Path(tmp) / "r.jsonl")],
                       check=True, capture_output=True)
        wall = time.perf_counter() - t0
        keep.write_text((Path(tmp) / "r.jsonl").read_text())
    r = feedback.requests(keep, dep.gpus)
    out[dep.key] = {"tier2": {"tokens_per_s_gpu": r["tokens_per_s_gpu"], "ttft_p99_ms": r["ttft_ms"]["p99"],
                              "tpot_p50_ms": r["tpot_ms"]["p50"]}, "tier1": t1[dep.key], "wall_s": wall}
    print(dep.key, json.dumps(out[dep.key]), flush=True)
    (here / "confirm.json").write_text(json.dumps(out, indent=1))
