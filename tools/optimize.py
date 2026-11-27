"""Agent interface: find the best deployment for a GPU count, workload and SLO.

    ./optimize.sh --gpus 128 --isl 8192 --osl 1024
    ./optimize.sh --gpus 64 --isl 2000 --osl 500 --ttft-slo 2 --evals 16
    ./optimize.sh --gpus 128 --isl 8192 --osl 1024 --grid      # exhaustive check

Score of a deployment = SLO capacity: max Poisson rate with >= 90 % of
steady-window requests meeting TTFT/TPOT and no queue growth (Tier-1).
Agent loop (best-first): evaluate -> diagnose the side that breaks just above
capacity (growing queue, else TTFT -> prefill / TPOT -> decode) and the
resource bounding its rounds -> propose targeted moves (layouts with EP inside
a node first when network-bound, prefill budget, P<->D instances, TP, decode
batch) plus one probe per dimension -> expand the best-scoring deployment
next (agent/search.py). Every evaluation and its rationale go to a
JSONL ledger. Space: instances of 16 GPUs; per side PP1 x EP16, PP2 x EP8 or
PP4 x EP4; decode max batch 32/64/128 — wherever Tier-1 surrogates exist.
"""

import argparse
import heapq
import itertools
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

from adapters.engines import ROOT
from adapters.topologies import Deployment, check, tier2_config
from agent import search
from agent.search import BUDGETS, DECODE_BATCHES, INSTANCE_GPUS as EP, LAYOUTS


def args_():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--gpus", type=int, default=128, help="total GPUs (multiple of 32)")
    a.add_argument("--hardware", default="h200", choices=["h200", "b300"])
    a.add_argument("--isl", type=int, default=8192)
    a.add_argument("--osl", type=int, default=1024)
    a.add_argument("--requests", type=int, default=600, help="minimum requests per probe")
    a.add_argument("--duration", type=float, default=300.0,
                   help="seconds of arrivals per probe (requests = max(--requests, rate x duration)); "
                        "short runs hide slowly growing queues")
    a.add_argument("--ttft-slo", type=float, default=5.0, help="seconds")
    a.add_argument("--tpot-slo", type=float, default=0.05, help="seconds")
    a.add_argument("--evals", type=int, default=16, help="agent evaluation budget")
    a.add_argument("--parallel", type=int, default=4, help="candidates evaluated concurrently")
    a.add_argument("--feedback", default="resource", choices=search.GRANULARITIES,
                   help="feedback granularity the agent sees (Phase 3 ablation; default resource)")
    a.add_argument("--grid", action="store_true", help="evaluate the whole space instead")
    a.add_argument("--ledger", type=Path, default=None, help="JSONL ledger (default experiments/_optimize/)")
    return a.parse_args()


A = args_()
PB = search.Problem(hardware=A.hardware, gpus=A.gpus, isl=A.isl, osl=A.osl, ttft_s=A.ttft_slo,
                    tpot_s=A.tpot_slo, requests=A.requests, duration=A.duration)
N = PB.instances
if A.gpus % EP or N < 2:
    raise SystemExit("--gpus must be a multiple of 16 and >= 32")
TPS = PB.tps
evaluate = partial(search.evaluate, PB)
legal = partial(search.legal, PB)


def layout(d):
    return "\n".join(
        f"  {side:7}: {getattr(d, side + '_instances')} x {d.width(side)} GPUs | PP{getattr(d, side + '_pp')} x "
        f"(attention TP{getattr(d, side + '_tp')} x DP{d.stage_gpus(side) // getattr(d, side + '_tp')}), "
        f"MoE EP{getattr(d, side + '_ep')} x ETP{getattr(d, side + '_etp')} | "
        f"{getattr(d, side + '_budget')} tokens/round, max batch {getattr(d, side + '_max_batch')}"
        + (f", MoE chunks {getattr(d, side + '_moe_chunks')}" if getattr(d, side + "_moe_chunks") > 1 else "")
        for side in ("prefill", "decode"))


def moves(dep, e):
    side = search.diagnose(e["evidence"], A.feedback)
    return search.moves(PB, dep, side, e.get("resource") if A.feedback == "resource" else None)


def main():
    ledger = A.ledger or ROOT / "experiments" / "_optimize" / time.strftime(
        f"%Y%m%d-%H%M%S-{A.hardware}-{A.gpus}gpu-isl{A.isl}-osl{A.osl}.jsonl")
    ledger.parent.mkdir(parents=True, exist_ok=True)
    start = Deployment(hardware=A.hardware, prefill_instances=N // 2, decode_instances=N - N // 2, prefill_budget=512)
    print(f"{A.gpus} x {A.hardware.upper()} ({N} instances x {EP} GPUs), ISL {A.isl} OSL {A.osl}, "
          f"SLO TTFT<={A.ttft_slo}s TPOT<={A.tpot_slo*1e3:.0f}ms; score = SLO capacity (Tier-1)")
    if not legal(start):
        raise SystemExit("no legal start deployment:\n  " + "\n  ".join(check(start, A.isl + A.osl)))
    seen, tie = {}, itertools.count()
    if A.grid:
        todo = [(Deployment(hardware=A.hardware, prefill_instances=p, decode_instances=N - p, prefill_budget=b,
                            prefill_tp=t, decode_tp=t, prefill_ep=pl[0], prefill_pp=pl[1],
                            decode_ep=dl[0], decode_pp=dl[1], decode_max_batch=m, decode_budget=m), "grid")
                for p in range(1, N) for t in TPS for b in BUDGETS for pl in LAYOUTS for dl in LAYOUTS
                for m in DECODE_BATCHES]
        todo = [(d, why) for d, why in todo if legal(d)]
    heap = [(0.0, 0, next(tie), start, "start: balanced P:D, budget 512")]
    workers = A.parallel if not A.grid else min(64, len(todo))
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("fork")) as pool:
        def batch(items):
            for d, _ in items:
                tier2_config(d)                      # launcher configs, serially
            for (d, why), e in zip(items, pool.map(evaluate, [d for d, _ in items])):
                e["rationale"] = why
                seen[d.key] = e
                with ledger.open("a") as f:
                    f.write(json.dumps(e) + "\n")
                print("  [%2d] %-46s %6.3f req/s  %5.1f tok/s/GPU  %6.1f tok/s/user  bottleneck=%-7s | %s" % (
                    len(seen), d.key, e["capacity_rps"], e["tokens_per_s_gpu"], e["tokens_per_s_user"],
                    e["bottleneck"], why), flush=True)
                yield d, e
        if A.grid:
            list(batch(todo))
        else:
            while heap and len(seen) < A.evals:
                picked = []
                while heap and len(picked) < min(A.parallel, A.evals - len(seen)):
                    _, _, _, d, why = heapq.heappop(heap)
                    if d.key not in seen and d.key not in {p.key for p, _ in picked}:
                        picked.append((d, why))
                for d, e in batch(picked):
                    for prio, nxt, why in moves(d, e):
                        if nxt.key not in seen and legal(nxt):
                            heapq.heappush(heap, (-e["capacity_rps"], prio, next(tie), nxt, why))
    ranked = sorted(seen.values(), key=lambda e: -e["capacity_rps"])
    best = ranked[0]
    front = [e for e in ranked if e["capacity_rps"] > 0 and not any(
        o["tokens_per_s_gpu"] >= e["tokens_per_s_gpu"] and o["tokens_per_s_user"] >= e["tokens_per_s_user"]
        and (o["tokens_per_s_gpu"] > e["tokens_per_s_gpu"] or o["tokens_per_s_user"] > e["tokens_per_s_user"])
        for o in ranked)]
    d = Deployment.of(best["deployment"])
    print(f"\nbest: {best['key']}  capacity {best['capacity_rps']:.3f} req/s, {best['tokens_per_s_gpu']:.1f} tok/s/GPU, "
          f"{best['tokens_per_s_user']:.1f} tok/s/user, TTFT p99 {best['ttft_p99_ms']:.0f} ms, TPOT p50 {best['tpot_p50_ms']:.1f} ms")
    print(f"  per side: prefill {best['prefill_tokens_per_s_gpu']:.0f} input tok/s per prefill GPU, "
          f"decode {best['decode_tokens_per_s_gpu']:.1f} output tok/s per decode GPU")
    print(layout(d))
    print("Pareto (tok/s/GPU vs tok/s/user):")
    for e in sorted(front, key=lambda e: -e["tokens_per_s_gpu"]):
        print("  %-46s %5.1f tok/s/GPU  %6.1f tok/s/user  %6.3f req/s" % (
            e["key"], e["tokens_per_s_gpu"], e["tokens_per_s_user"], e["capacity_rps"]))
    print(f"\nevaluations: {len(seen)}  ledger: {ledger}")
    print(f"reproduce: ./simulate.sh --hardware {A.hardware} --p {d.prefill_instances} --d {d.decode_instances} "
          f"--p-tp {d.prefill_tp} --p-ep {d.prefill_ep} --p-etp {d.prefill_etp} --p-pp {d.prefill_pp} "
          f"--d-tp {d.decode_tp} --d-ep {d.decode_ep} --d-etp {d.decode_etp} --d-pp {d.decode_pp} "
          f"--p-budget {d.prefill_budget} --d-max-batch {d.decode_max_batch} --p-moe-chunks {d.prefill_moe_chunks} "
          f"--isl {A.isl} --osl {A.osl} "
          f"--rate {best['capacity_rps']!r} --requests {max(A.requests, round(best['capacity_rps'] * A.duration))} "
          f"--ttft-slo {A.ttft_slo} --tpot-slo {A.tpot_slo}")


if __name__ == "__main__":
    main()
