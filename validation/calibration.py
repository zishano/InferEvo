"""Fit RoofLang-form efficiency parameters to public kernel measurements.

Model per kernel class (same arithmetic as RoofLang's per-kernel roofline,
plus a fixed per-kernel overhead):

    t = alpha + max(FLOPs / (eta_c * peak), bytes / (eta_m * hbm_bw))

Data: Frontier H200 Qwen3-235B kernel-only tables
(`references/frontier/data/profiling/compute/h200/Qwen3-235B-A22B`), BF16,
vLLM kernels, median over repetitions, milliseconds. Fitted parameters are
*calibrated assumptions* for Kimi-K3 shapes (transfer across models/kernels
is unverified); residuals quantify fit quality on the source data only.
"""

from __future__ import annotations

import csv
import itertools
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

from adapters.engines import REFERENCES

FRONTIER_H200 = (REFERENCES / "frontier" / "data" / "profiling" / "compute"
                 / "h200" / "Qwen3-235B-A22B")
H200_BF16_PEAK = 989.5e12      # FLOP/s, datasheet dense
H200_HBM_BW = 4.8e12           # B/s, datasheet


@dataclass(frozen=True)
class Sample:
    flops: float
    bytes: float
    measured_s: float
    label: str


@dataclass(frozen=True)
class Fit:
    kernel_class: str
    alpha_s: float
    eta_compute: float
    eta_memory: float
    n: int
    median_abs_rel_err: float
    p90_abs_rel_err: float

    def predict(self, s: Sample, peak=H200_BF16_PEAK, bw=H200_HBM_BW) -> float:
        return self.alpha_s + max(s.flops / (self.eta_compute * peak),
                                  s.bytes / (self.eta_memory * bw))


def dense_gemm_samples() -> list[Sample]:
    """`attn_post_proj`: one GEMM M=tokens, N=hidden, K=heads*128/TP."""
    out = []
    with open(FRONTIER_H200 / "linear_op_kernel_only.csv") as f:
        for r in csv.DictReader(f):
            t = r["time_stats.attn_post_proj.median"]
            if not t:
                continue
            m, tp = int(r["num_tokens"]), int(r["num_tensor_parallel_workers"])
            k, n = int(r["n_head"]) * 128 // tp, int(r["n_embd"])
            out.append(Sample(2.0 * m * n * k, 2.0 * (m * k + k * n + m * n),
                              float(t) * 1e-3, f"M={m},TP={tp}"))
    return out


def moe_grouped_gemm_samples() -> list[Sample]:
    """`moe_grouped_gemm` (gate+up+down), BF16, uniform routing.

    Expert intermediate is split by TP (vLLM TP-sharded experts); weights of
    every touched expert are read once.
    """
    out = []
    with open(FRONTIER_H200 / "moe_kernel_only.csv") as f:
        for r in csv.DictReader(f):
            t = r["time_stats.moe_grouped_gemm.median"]
            if (not t or r["moe_quantization_mode"] != "bf16"
                    or r["load_distribution"] != "uniform"):
                continue
            tokens = int(r["total_routed_tokens"])
            experts = int(r["num_experts_per_device"])
            h, tp = int(r["hidden_dim"]), int(r["num_tensor_parallel_workers"])
            ffn = int(r["expert_hidden_dim"]) // tp
            touched = min(experts, tokens)
            out.append(Sample(
                2.0 * tokens * 3 * h * ffn,
                2.0 * (touched * 3 * h * ffn + tokens * (2 * h + 2 * ffn)),
                float(t) * 1e-3, f"T={tokens},E={experts},TP={tp}"))
    return out


def _errors(fit_params, samples):
    alpha, ec, em = fit_params
    return [abs(alpha + max(s.flops / (ec * H200_BF16_PEAK),
                            s.bytes / (em * H200_HBM_BW)) - s.measured_s)
            / s.measured_s for s in samples]


def fit(kernel_class: str, samples: list[Sample]) -> Fit:
    """Grid search minimizing mean |log(pred/meas)|; eta bounded to (0, 1]."""
    def loss(p):
        a, ec, em = p
        return statistics.fmean(
            abs(math.log((a + max(s.flops / (ec * H200_BF16_PEAK),
                                  s.bytes / (em * H200_HBM_BW))) / s.measured_s))
            for s in samples)

    alphas = [i * 0.25e-6 for i in range(0, 41)]          # 0..10 us
    etas = [i / 50 for i in range(2, 51)]                 # 0.04..1.0
    best = min(itertools.product(alphas, etas, etas), key=loss)
    # local refinement
    a0, c0, m0 = best
    fine = min(itertools.product(
        [max(0.0, a0 + d * 0.05e-6) for d in range(-5, 6)],
        [min(1.0, max(0.01, c0 + d * 0.004)) for d in range(-5, 6)],
        [min(1.0, max(0.01, m0 + d * 0.004)) for d in range(-5, 6)]), key=loss)
    err = sorted(_errors(fine, samples))
    return Fit(kernel_class, fine[0], fine[1], fine[2], len(samples),
               statistics.median(err), err[int(0.9 * (len(err) - 1))])
