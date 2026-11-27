"""Tier-1 Cost Query (DECISIONS D3): fitted StepTime(batch) surrogate.

Interface follows BLIS `LatencyModel.StepTime(batch)`. Targets are lower
bounds of the exact exported RoofLang graph (validation/round_bounds), which
track Tier-2 engine rounds within ~1 % for steady decode (E0008); the fit
maps batch aggregates to that bound in microseconds per query. A model is
valid only for its key (hardware, attention TP, EP, stage, efficiency
profile, cost_model_version) and must be re-fitted when any changes.
Screening only: promoted candidates are re-evaluated at Tier-2.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

from adapters.cost_model import cost_model_version
from adapters.engines import enable_bridge_imports

N_EXPERTS, TOPK = 896, 16


def features(batch, groups: int, stage: str) -> list[float]:
    """batch: list of (request_id, q_len, kv_len)."""
    per_n = [0] * groups
    per_kv = [0] * groups
    per_q = [0] * groups
    per_attn = [0.0] * groups
    for request, q, kv in batch:
        g = request % groups
        per_n[g] += 1
        per_kv[g] += kv
        per_q[g] += q
        per_attn[g] += q * (kv + (q + 1) / 2)       # causal query-key pairs
    tokens = sum(q for _, q, _ in batch)
    # expected fraction of experts touched by `tokens` top-k routings
    touched = 1 - (1 - TOPK / N_EXPERTS) ** tokens
    return [1.0, max(per_n), max(per_kv), max(per_q), max(per_attn),
            touched, float(tokens)]


FEATURE_NAMES = ["const", "max_group_requests", "max_group_kv", "max_group_q",
                 "max_group_attn_pairs", "expert_fraction_touched", "tokens"]


@dataclass
class Tier1Model:
    key: dict
    coef: list
    train_n: int
    train_median_rel_err: float
    train_p90_rel_err: float
    notes: list = field(default_factory=list)

    def step_time(self, batch) -> float:
        x = features(batch, self.key["groups"], self.key["stage"])
        return max(1e-9, float(np.dot(self.coef, x)))


def _template(hardware, tp, ep, stage, efficiency, pp=1):
    """ep: MoE ranks per PP stage; the instance has pp x ep GPUs. Uses the
    reference Tier-2 lowering, so graph bounds match Tier-2 runs."""
    enable_bridge_imports()
    from rooflang_bridge.kimi_k3_shared import KimiTemplate
    from adapters.topologies import LOWERING
    hw = {"gpu_model": hardware, "gpu_count": ep * pp}
    if efficiency:
        hw["efficiency"] = dict(efficiency)
    return KimiTemplate(tp, ep, hw, 0, stage=stage, pp=pp, lowering=LOWERING)


def sample_batches(stage: str, count: int, seed: int, max_requests: int,
                   max_kv: int, max_q: int):
    """Random batches. Prefill respects a total token budget of max_q (the
    scheduler's max_num_batched_tokens); chunks continue (kv>0) or start."""
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        n = rng.randint(1, max_requests)
        ids = rng.sample(range(10_000), n)
        if stage == "d":
            out.append([(i, 1, rng.randint(1, max_kv)) for i in ids])
            continue
        cuts = sorted(rng.sample(range(1, max_q), min(n - 1, max_q - 1)))
        budget = rng.randint(n, max_q)
        sizes = [max(1, round((b - a) * budget / max_q))
                 for a, b in zip([0, *cuts], [*cuts, max_q])]
        out.append([(i, q, rng.choice((0, rng.randint(1, max_kv))))
                    for i, q in zip(ids, sizes)])
    return out


def bound_of(template, batch, hardware) -> float:
    from validation.round_bounds import round_bounds
    ops = template.export(
        [{"q_len": q, "kv_len": kv, "lm_head_len": 1} for _, q, kv in batch],
        [r for r, _, _ in batch], details=False)["operations"]
    return round_bounds(ops, hardware).lower_s


def fit(hardware="h200", tp=4, ep=16, stage="d", efficiency=None, samples=120,
        seed=0, max_requests=32, max_kv=8192, max_q=512) -> Tier1Model:
    template = _template(hardware, tp, ep, stage, efficiency)
    groups = ep // tp
    batches = sample_batches(stage, samples, seed, max_requests, max_kv, max_q)
    x = np.array([features(b, groups, stage) for b in batches])
    y = np.array([bound_of(template, b, hardware) for b in batches])
    # relative-error weighting: scale rows by 1/y
    coef, *_ = np.linalg.lstsq(x / y[:, None], np.ones_like(y), rcond=None)
    rel = np.abs(x @ coef - y) / y
    key = {"hardware": hardware, "tp": tp, "ep": ep, "groups": groups, "stage": stage,
           "efficiency": efficiency or {}, "cost_model_version": cost_model_version(),
           "target": "exported-graph lower bound (validation/round_bounds)"}
    return Tier1Model(key, coef.tolist(), len(y), float(np.median(rel)),
                      float(np.quantile(rel, 0.9)))


def fit_samples(batches, targets, key: dict) -> Tier1Model:
    """Fit on given (batch, target seconds) pairs, e.g. Tier-2 engine rounds."""
    groups, stage = key["groups"], key["stage"]
    x = np.array([features(b, groups, stage) for b in batches])
    y = np.array(targets)
    coef, *_ = np.linalg.lstsq(x / y[:, None], np.ones_like(y), rcond=None)
    rel = np.abs(x @ coef - y) / y
    return Tier1Model(dict(key), coef.tolist(), len(y), float(np.median(rel)),
                      float(np.quantile(rel, 0.9)))


def fit_round_sums(samples, targets, key: dict) -> Tier1Model:
    """Fit when each target is the summed duration of known rounds
    (samples: list of rounds; round = batch). Linear model => sum features."""
    groups, stage = key["groups"], key["stage"]
    x = np.array([np.sum([features(b, groups, stage) for b in rounds], axis=0)
                  for rounds in samples])
    y = np.array(targets)
    coef, *_ = np.linalg.lstsq(x / y[:, None], np.ones_like(y), rcond=None)
    rel = np.abs(x @ coef - y) / y
    return Tier1Model(dict(key), coef.tolist(), len(y), float(np.median(rel)),
                      float(np.quantile(rel, 0.9)))
