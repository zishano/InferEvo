"""E0058: do the fixed-workload conclusions hold on the agent workload?

Workload: closed-loop coding-agent sessions (workloads.agent_sessions, 8
turns, context 2k -> ~20k, tools 0.2-3 s; every turn re-prefills the whole
history: ~50:1 prefill:decode tokens vs 8:1 for 8K/1K). SLO per call TTFT
<= 5 s, TPOT <= 50 ms; capacity in sessions/s, 600 s probes (D22), Tier-1.
Rows (128 GPUs, H200 and B300):
  agent optimum (searches 20261127-204827-*-agent8) and its ablations:
    rank dispatch off, chunked MoE off, decode batch 64 <-> 128, P:D +-1;
  the 8K/1K optima (E0057) deployed on agent traffic (transfer).
FP8 not included (opt-in; separate question).

    ./ie.sh experiments/E0058_agent_workload/run.py
"""
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

from adapters.cost_model import cost_model_version
from adapters.engines import source_identity
from adapters.topologies import Deployment, tier2_config
from agent import search

HERE = Path(__file__).parent


def dep(hw, p, batch, mc=8, rd=False):
    return Deployment(hardware=hw, prefill_instances=p, decode_instances=8 - p, prefill_ep=4, prefill_pp=4,
                      prefill_budget=2048, prefill_moe_chunks=mc, prefill_dispatch="ranks" if rd else "leader",
                      decode_budget=batch, decode_max_batch=batch)


ROWS = {
    # name: deployment
    "h200 agent optimum (P5 D3 b64 mc8 rd)": dep("h200", 5, 64, rd=True),
    "h200  - rank dispatch": dep("h200", 5, 64),
    "h200  - rank dispatch - chunks": dep("h200", 5, 64, mc=1),
    "h200  b128": dep("h200", 5, 128, rd=True),
    "h200  P6 D2": dep("h200", 6, 64, rd=True),
    "h200  P4 D4": dep("h200", 4, 64, rd=True),
    "h200 8K/1K optimum P3 D5 b128 rd": dep("h200", 3, 128, rd=True),
    "h200 8K/1K max-throughput P4 D4 b128": dep("h200", 4, 128),
    "b300 agent optimum (P6 D2 b128 mc8)": dep("b300", 6, 128),
    "b300  + rank dispatch": dep("b300", 6, 128, rd=True),
    "b300  - chunks": dep("b300", 6, 128, mc=1),
    "b300  b64": dep("b300", 6, 64),
    "b300  P7 D1": dep("b300", 7, 128),
    "b300  P5 D3": dep("b300", 5, 128),
    "b300 8K/1K optimum P3 D5 b128 rd": dep("b300", 3, 128, rd=True),
}


def score(d):
    pb = search.Problem(hardware=d.hardware, workload="agent", duration=600.0)
    if not search.legal(pb, d):
        return None
    e = search.evaluate(pb, d)
    keep = ("capacity_rps", "tokens_per_s_gpu", "tokens_per_s_user", "ttft_p99_ms", "tpot_p50_ms",
            "jct_p50_s", "jct_p99_s", "prefill_tokens_per_s_gpu", "decode_tokens_per_s_gpu", "bottleneck")
    return {k: e[k] for k in keep}


def main():
    for d in ROWS.values():
        tier2_config(d)
    with ProcessPoolExecutor(len(ROWS), mp_context=multiprocessing.get_context("fork")) as pool:
        res = dict(zip(ROWS, pool.map(score, ROWS.values())))
    for name, r in res.items():
        print(f"{name:42s} " + ("n/a" if r is None else
              f"{r['capacity_rps']:5.2f} sess/s  {r['tokens_per_s_gpu']:5.1f} tok/s/GPU  "
              f"JCT p50/p99 {r['jct_p50_s']:.0f}/{r['jct_p99_s']:.0f} s  TTFT p99 {r['ttft_p99_ms']:.0f} ms  "
              f"TPOT {r['tpot_p50_ms']:.1f} ms  ({r['bottleneck']})"), flush=True)
    cmv = cost_model_version()
    (HERE / f"result-{cmv[:12]}.json").write_text(json.dumps(
        {"experiment": "E0058", "cost_model_version": cmv, "source_identity": source_identity(),
         "rows": {n: {"deployment": d.key, "score": res[n]} for n, d in ROWS.items()}}, indent=1))


if __name__ == "__main__":
    main()
