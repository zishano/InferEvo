"""Confirm selected E0013 points at Tier-2 (shared Rooflang, same layout).

    ./ie.sh experiments/E0013_load_sweep/confirm.py 1 4
"""
import json, subprocess, sys, tempfile, time
from pathlib import Path
from adapters import workloads
from adapters.engines import SIMTHESIZER, source_identity
from run import metrics  # same metric definitions as the Tier-1 sweep

here = Path(__file__).parent
t1 = {r["multiplier"]: r for r in json.loads((here / "result.json").read_text())["rows"]}
out = []
for mult in map(float, sys.argv[1:]):
    with tempfile.TemporaryDirectory(prefix="e0013c-") as tmp:
        rows = workloads.scaled(mult)
        wl = workloads.write(rows, Path(tmp) / "w.jsonl")
        t0 = time.perf_counter()
        subprocess.run(["bash", "run_kimi_k3_shared_workload.sh", "--workload", str(wl),
                        "--output-dir", str(Path(tmp) / "o")], cwd=SIMTHESIZER, check=True,
                       capture_output=True)
        res = metrics(Path(tmp) / "o" / "result.jsonl", rows)
        res.update(multiplier=mult, wall_s=time.perf_counter() - t0)
    ref = t1.get(mult if mult not in (1.0, 2.0, 4.0, 8.0, 16.0) else int(mult)) or t1.get(mult)
    res["tier1"] = ref
    out.append(res)
    print(json.dumps({"multiplier": mult, "tier2": {k: res[k] for k in ("tokens_per_s_gpu", "ttft_ms", "tpot_ms", "wall_s")},
                      "tier1": {k: ref[k] for k in ("tokens_per_s_gpu", "ttft_ms", "tpot_ms")} if ref else None}), flush=True)
    (here / "confirm.json").write_text(json.dumps({"rows": out, "source_identity": source_identity()}, indent=1))
