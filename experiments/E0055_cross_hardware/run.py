"""E0055: cross-hardware optima and configuration transfer (Phase 6).

For each (hardware, precision) search ledger (128 GPUs, 8K/1K, SLO 5 s / 50
ms), the best deployment is re-scored with 600 s probes (D22 noted 300 s
probes overestimate decode-bound capacity) on its own hardware and, with the
hardware swapped, on the other one (configuration transfer). FP8 rows carry
M-FP8LINEAR's quality flag (UNVERIFIED).

    ./ie.sh experiments/E0055_cross_hardware/run.py <ledger> [<ledger> ...]
"""
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

from adapters.cost_model import cost_model_version
from adapters.engines import source_identity
from adapters.topologies import Deployment, precision_changed, tier2_config
from agent import search

HERE = Path(__file__).parent


def best(ledger):
    rows = [json.loads(l) for l in Path(ledger).read_text().splitlines() if l.strip()]
    return Deployment.of(max(rows, key=lambda e: e["capacity_rps"])["deployment"])


def score(args):
    dep, hw = args
    d = replace(dep, hardware=hw)
    pb = search.Problem(hardware=hw, duration=600.0, allow_precision=precision_changed(d))
    if not search.legal(pb, d):
        return None
    e = search.evaluate(pb, d)
    return {k: e[k] for k in ("capacity_rps", "tokens_per_s_gpu", "tokens_per_s_user", "ttft_p99_ms",
                              "tpot_p50_ms", "prefill_tokens_per_s_gpu", "decode_tokens_per_s_gpu", "bottleneck")}


def main(ledgers):
    optima = {Path(l).name: best(l) for l in ledgers}
    jobs = [(name, dep, hw) for name, dep in optima.items() for hw in ("h200", "b300")]
    for _, dep, hw in jobs:
        tier2_config(replace(dep, hardware=hw))          # launcher configs, serially
    with ProcessPoolExecutor(len(jobs), mp_context=multiprocessing.get_context("fork")) as pool:
        scores = list(pool.map(score, [(dep, hw) for _, dep, hw in jobs]))
    rows = []
    for (name, dep, hw), s in zip(jobs, scores):
        rows.append({"ledger": name, "found_on": dep.hardware, "run_on": hw, "deployment": dep.key,
                     "precision_changed": precision_changed(dep), "score": s})
        print(f"{dep.hardware} optimum {dep.key} on {hw}: "
              + (f"{s['capacity_rps']:.2f} req/s, {s['tokens_per_s_gpu']:.1f} tok/s/GPU, "
                 f"{s['tokens_per_s_user']:.1f} tok/s/user, TTFT p99 {s['ttft_p99_ms']:.0f} ms ({s['bottleneck']})"
                 if s else "not legal / not covered by Tier-1")
              + ("  [FP8: quality UNVERIFIED]" if precision_changed(dep) else ""), flush=True)
    cmv = cost_model_version()
    (HERE / f"result-{cmv[:12]}.json").write_text(json.dumps(
        {"experiment": "E0055", "cost_model_version": cmv, "source_identity": source_identity(),
         "probe_duration_s": 600, "optima": {n: asdict(d) for n, d in optima.items()}, "rows": rows}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
