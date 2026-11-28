"""E0057: head-to-head 600 s capacity of the batch-128 candidates (H200 / B300;
P3:D5 vs P4:D4; leader vs rank dispatch; b64 vs b128; FP8 both sides) after the
E0056 batch-128 surrogates. Searches (300 s probes) are in experiments/_optimize
20261127-192503-*. FP8 rows: quality UNVERIFIED. n/a = no surrogate for that
option combination.

    ./ie.sh experiments/E0057_batch128_optima/run.py
"""
import json, multiprocessing, sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from adapters.topologies import Deployment, precision_changed, tier2_config
from agent import search

def dep(hw, p, b, rd, fp8p=False, fp8d=False):
    return Deployment(hardware=hw, prefill_instances=p, decode_instances=8 - p, prefill_ep=4, prefill_pp=4,
                      prefill_budget=2048, prefill_moe_chunks=8, prefill_dispatch="ranks" if rd else "leader",
                      decode_budget=b, decode_max_batch=b, prefill_linear="fp8" if fp8p else "bf16",
                      decode_linear="fp8" if fp8d else "bf16")

def score(d):
    pb = search.Problem(hardware=d.hardware, duration=600.0, allow_precision=precision_changed(d))
    if not search.legal(pb, d):
        return d.key, None
    e = search.evaluate(pb, d)
    return d.key, {k: round(e[k], 2) if isinstance(e[k], float) else e[k]
                   for k in ("capacity_rps", "tokens_per_s_gpu", "tokens_per_s_user", "ttft_p99_ms", "tpot_p50_ms", "bottleneck")}

if __name__ == "__main__":
    C = []
    for hw in ("h200", "b300"):
        for p in (3, 4):
            for rd in (False, True):
                for b in (64, 128):
                    C.append(dep(hw, p, b, rd))
                C.append(dep(hw, p, 128, rd, fp8p=True, fp8d=True))
    for d in C:
        tier2_config(d)
    with ProcessPoolExecutor(len(C), mp_context=multiprocessing.get_context("fork")) as pool:
        for key, s in pool.map(score, C):
            print(key, json.dumps(s), flush=True)
