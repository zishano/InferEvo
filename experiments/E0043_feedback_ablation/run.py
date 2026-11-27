"""E0043: feedback-granularity ablation on whole searches (held-out problems).

E0042 checked single diagnoses on the five regimes the rules were tuned on.
Here the full best-first agent (tools/optimize.py, 16 evaluations, 4 in
parallel, same start) runs on three problems not used to design the rules,
once per feedback granularity (resource / queue / slo / none). Reported:
best capacity after k evaluations and evaluations needed to reach 95 % of
the best capacity any search found. P-a also gets an exhaustive grid as the
reference optimum. Ledgers are kept under ledgers/.

    ./ie.sh experiments/E0043_feedback_ablation/run.py
"""

import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from adapters.cost_model import cost_model_version
from adapters.engines import ROOT, source_identity
from agent.search import GRANULARITIES

HERE = Path(__file__).parent
LEDGERS = HERE / "ledgers"
PROBLEMS = {
    "Pa-64gpu-chat": ["--gpus", "64", "--isl", "2000", "--osl", "500", "--ttft-slo", "2"],
    "Pb-96gpu-decode-heavy": ["--gpus", "96", "--isl", "4096", "--osl", "2048"],
    "Pc-128gpu-long-prompt": ["--gpus", "128", "--isl", "24576", "--osl", "512", "--ttft-slo", "10"],
}
EVALS = 16
GRID = ("Pa-64gpu-chat",)


def search(problem, feedback):
    ledger = LEDGERS / f"{problem}-{feedback}.jsonl"
    if not ledger.exists():
        tmp = ledger.with_suffix(".partial")
        tmp.unlink(missing_ok=True)
        extra = ["--grid", "--parallel", "64"] if feedback == "grid" else \
                ["--feedback", feedback, "--evals", str(EVALS), "--parallel", "4"]
        subprocess.run([str(ROOT / "optimize.sh"), *PROBLEMS[problem], *extra, "--ledger", str(tmp)],
                       check=True, capture_output=True)
        tmp.rename(ledger)
    return [json.loads(l) for l in ledger.read_text().splitlines()]


def main():
    t0 = time.perf_counter()
    LEDGERS.mkdir(exist_ok=True)
    jobs = [(p, g) for p in PROBLEMS for g in GRANULARITIES] + [(p, "grid") for p in GRID]
    with ThreadPoolExecutor(len(jobs)) as pool:
        runs = dict(zip(jobs, pool.map(lambda j: search(*j), jobs)))
    out = {}
    for p in PROBLEMS:
        best_any = max(e["capacity_rps"] for (q, g), r in runs.items() if q == p and g != "grid" for e in r)
        ref = max((e["capacity_rps"] for e in runs.get((p, "grid"), [])), default=None)
        target = 0.95 * (ref or best_any)
        rows = {}
        for g in GRANULARITIES:
            caps = [e["capacity_rps"] for e in runs[(p, g)]]
            curve = [max(caps[:k + 1]) for k in range(len(caps))]
            best = max(runs[(p, g)], key=lambda e: e["capacity_rps"])
            rows[g] = {"best_capacity": curve[-1], "best_key": best["key"],
                       "best_tokens_per_s_gpu": best["tokens_per_s_gpu"],
                       "curve": curve, "evals_to_95pct": next((k + 1 for k, c in enumerate(curve) if c >= target), None)}
        out[p] = {"reference_grid": ref, "grid_size": len(runs.get((p, "grid"), [])) or None,
                  "best_any_search": best_any, "target_95pct": target, "searches": rows}
        print(f"\n{p}: grid optimum {ref}  best of searches {best_any:.3f} req/s")
        for g, r in rows.items():
            print(f"  {g:8} best {r['best_capacity']:.3f} req/s ({r['best_key']})  "
                  f"evals-to-95% {r['evals_to_95pct']}  curve " + " ".join(f"{c:.2f}" for c in r["curve"][::3]))
    cmv = cost_model_version()
    (HERE / f"result-{cmv[:12]}.json").write_text(json.dumps(
        {"experiment": "E0043", "evals": EVALS, "cost_model_version": cmv, "source_identity": source_identity(),
         "problems": {p: PROBLEMS[p] for p in PROBLEMS}, "results": out, "wall_s": time.perf_counter() - t0}, indent=1))


if __name__ == "__main__":
    main()
