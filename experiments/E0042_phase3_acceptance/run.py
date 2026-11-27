"""E0042: Phase 3 acceptance — does the agent's diagnosis point where the
counterfactuals say the bottleneck is? Plus the feedback-granularity ablation.

Per regime (128 x H200, Tier-1, frozen evaluator = agent.search.Problem):
  1. evaluate the start deployment -> feedback just above capacity;
  2. side counterfactuals: capacity with prefill (resp. decode) round time
     / 2 -> true bottleneck side = the larger capacity gain (`both` if the two
     gains are within 25 % of each other, `none` if both < 5 %);
  3. direction check: evaluate every legal one-step move; the agent's
     ordered list at each feedback granularity (queue / slo / none) gives the
     number of evaluations until the best move, and the gain of its first move;
  4. resource attribution per side: graph-bound elasticity of one
     representative round to compute / memory / network x0.5 (CM-4 derate;
     graph bound, no contention — screening evidence). The `resource`
     granularity uses it to order moves (agent.search.moves).

    ./ie.sh experiments/E0042_phase3_acceptance/run.py
"""

import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path

from adapters.cost_model import cost_model_version
from adapters.engines import source_identity
from adapters.topologies import Deployment, tier2_config
from agent import search
from agent.search import FACTOR, GRANULARITIES, Problem

HERE = Path(__file__).parent
BASE = Problem()                                   # 128 x H200, 8K/1K, SLO 5 s / 50 ms


def dep(p, *, p_layout=(16, 1), b=512, batch=32, tp=4):
    return Deployment(hardware="h200", prefill_instances=p, decode_instances=8 - p, prefill_tp=tp,
                      decode_tp=tp, prefill_ep=p_layout[0], prefill_pp=p_layout[1],
                      prefill_budget=b, decode_budget=batch, decode_max_batch=batch)


REGIMES = {
    # too few prefill instances: the prefill queue grows
    "R1-prefill-throughput": (BASE, dep(2)),
    # too few decode instances: calls wait for decode slots (backlog, D22)
    "R2-decode-throughput": (BASE, dep(6, p_layout=(4, 4), b=2048)),
    # long prompts, tight TTFT: a single prefill is close to the SLO
    "R3-ttft-latency": (replace(BASE, isl=16384, osl=256, ttft_s=8.0), dep(4)),
    # tight TPOT with large decode batches: per-token latency binds
    "R4-tpot-latency": (replace(BASE, tpot_s=0.02), dep(4, p_layout=(4, 4), b=2048, batch=64)),
    # the MC-5 optimum: both sides close to their limits
    "R5-balanced-optimum": (BASE, dep(4, p_layout=(4, 4), b=2048, batch=64)),
}


def truth(base, gain_p, gain_d):
    if max(gain_p, gain_d) < 0.05:
        return "none"
    lo, hi = sorted((gain_p, gain_d))
    if lo >= 0.75 * hi:
        return "both"
    return "prefill" if gain_p > gain_d else "decode"


def main():
    t0 = time.perf_counter()
    jobs, plan = [], {}
    for name, (pb, d) in REGIMES.items():
        assert search.legal(pb, d), name
        tier2_config(d)
        cands = {}
        for g in ("prefill", "decode", "none"):          # every legal one-step move
            for c, _ in search.ordered(pb, d, g):
                cands[c.key] = c
        plan[name] = list(cands.values())
        jobs += [(name, "start", pb, d, None), (name, "cf-prefill", pb, d, {"p": FACTOR}),
                 (name, "cf-decode", pb, d, {"d": FACTOR})]
        jobs += [(name, f"move:{c.key}", pb, c, None) for c in plan[name]]
    for _, _, _, c, _ in jobs:
        tier2_config(c)                                    # launcher configs, serially
    print(f"{len(jobs)} capacity evaluations over {len(REGIMES)} regimes", flush=True)
    with ProcessPoolExecutor(min(96, len(jobs)), mp_context=multiprocessing.get_context("fork")) as pool:
        futs = {(n, tag): pool.submit(search.evaluate, pb, c, sp) for n, tag, pb, c, sp in jobs}
        res = {k: f.result() for k, f in futs.items()}

    rows, attrib_jobs = {}, {}
    for name, (pb, d) in REGIMES.items():
        start = res[(name, "start")]
        cap = start["capacity_rps"]
        gain = {s: res[(name, f"cf-{s}")]["capacity_rps"] / cap - 1 for s in ("prefill", "decode")}
        true_side = truth(cap, gain["prefill"], gain["decode"])
        move_gain = {c.key: res[(name, f"move:{c.key}")]["capacity_rps"] / cap - 1 for c in plan[name]}
        best = max(move_gain, key=move_gain.get)
        per_g = {}
        for g in GRANULARITIES:
            side = search.diagnose(start["evidence"], g)
            res_ = start["resource"] if g == "resource" else None
            order = [c.key for c, _ in search.ordered(pb, d, side, res_)]
            directed = [c.key for p, c, _ in search.moves(pb, d, side, res_) if p == 0 and c.key in move_gain]
            per_g[g] = {
                "diagnosis": side,
                "matches_counterfactual": side == true_side or (true_side == "both" and side in ("prefill", "decode")),
                "evals_to_best_move": order.index(best) + 1,
                "first_move": order[0], "first_move_gain": move_gain[order[0]],
                "best_directed_gain": max((move_gain[k] for k in directed), default=None),
            }
        rows[name] = {"problem": asdict(pb), "start": start, "counterfactual_gain": gain,
                      "true_side": true_side, "moves": move_gain, "best_move": best,
                      "best_move_gain": move_gain[best], "granularity": per_g,
                      "counterfactual_runs": {s: res[(name, f"cf-{s}")] for s in ("prefill", "decode")}}
        rows[name]["resource_attribution"] = {s: search.attribution(pb, d, s) for s in ("prefill", "decode")}

    for name, r in rows.items():
        g = r["granularity"]
        print(f"\n{name}: start {r['start']['key']} {r['start']['capacity_rps']:.3f} req/s; "
              f"counterfactual x2 gain P {r['counterfactual_gain']['prefill']:+.0%} D {r['counterfactual_gain']['decode']:+.0%}"
              f" -> true side {r['true_side']}")
        print(f"  best move {r['best_move']} ({r['best_move_gain']:+.0%})")
        for k in GRANULARITIES:
            print(f"  {k:5}: diagnosis {g[k]['diagnosis']:8} match={g[k]['matches_counterfactual']!s:5} "
                  f"evals-to-best {g[k]['evals_to_best_move']:2d}  first move {g[k]['first_move_gain']:+.0%}")
        for s, a in r.get("resource_attribution", {}).items():
            print(f"  {s} round {a['round_ms']:.1f} ms elasticity " +
                  " ".join(f"{k} {v:.2f}" for k, v in a["elasticity"].items()))
    summary = {g: {"matches": sum(r["granularity"][g]["matches_counterfactual"] for r in rows.values()),
                   "mean_evals_to_best": sum(r["granularity"][g]["evals_to_best_move"] for r in rows.values()) / len(rows)}
               for g in GRANULARITIES}
    print("\nsummary:", json.dumps(summary))
    cmv = cost_model_version()
    (HERE / f"result-{cmv[:12]}.json").write_text(json.dumps(
        {"experiment": "E0042", "factor": FACTOR, "cost_model_version": cmv, "source_identity": source_identity(),
         "regimes": rows, "summary": summary, "wall_s": time.perf_counter() - t0}, indent=1))


if __name__ == "__main__":
    main()
