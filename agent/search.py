"""Deployment search primitives shared by optimize.sh and experiments (Phase 3).

A `Problem` fixes the frozen evaluator: hardware, GPU count, workload
(fixed ISL/OSL Poisson), SLO and probe length (D22). `evaluate` scores one
deployment by SLO capacity on Tier-1 and keeps the raw feedback just above
capacity; `diagnose` turns that feedback into a bottleneck side at a chosen
granularity; `moves` maps the side to ordered candidate deployments.

Feedback granularity (Phase 3 ablation):
  resource - queue + which hardware resource bounds the bottleneck side's
          round (graph-bound elasticity to compute/memory/network x0.5):
          a network-bound side tries layouts that keep EP inside one
          NVLink node first (E0042);
  queue - per-side queue growth (prefill TTFT drift, decode backlog), then the
          SLO split (D22);
  slo   - only which SLO breaks more (TTFT -> prefill, TPOT -> decode);
  none  - capacity only: no side, every move is an undirected probe.

`speedup` = {"p": k, "d": k} divides one side's Tier-1 round time by k: the
counterfactual "this side is k x faster" used to label the true bottleneck.
"""

from __future__ import annotations

import tempfile
import time
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path

from adapters import tier1, steady, tier1_engine as te, workloads
from adapters.topologies import DISPATCH, LINEAR, MOE_CHUNKS, Deployment, check, tier2_config
from agent.controller import models_for

BUDGETS = (128, 512, 2048)
INSTANCE_GPUS = 16
LAYOUTS = ((16, 1), (8, 2), (4, 4))          # (MoE ranks per stage, PP) with 16 GPUs
# Decode may use wider instances: EP32 / EP64 halve / quarter each GPU's expert
# bytes per token at the same per-GPU batch (E0062).
DECODE_LAYOUTS = LAYOUTS + ((32, 1), (64, 1))
DECODE_BATCHES = (32, 64, 128, 256, 512)
GRANULARITIES = ("resource", "queue", "slo", "none")
NODE_GPUS = 8                                # NVLink domain (H200 / B300 HGX)
NETWORK_BOUND = 0.5                          # network elasticity that reorders moves
FACTOR = 2.0                                 # counterfactual derate / speedup


@dataclass(frozen=True)
class Problem:
    hardware: str = "h200"
    gpus: int = 128
    isl: int = 8192
    osl: int = 1024
    ttft_s: float = 5.0
    tpot_s: float = 0.05
    requests: int = 600          # minimum requests per probe
    duration: float = 300.0      # seconds of arrivals per probe (D22)
    # "fixed": Poisson calls of isl/osl tokens (rate in req/s); "agent":
    # closed-loop coding sessions (workloads.agent_sessions, rate in
    # sessions/s, >= `sessions` per probe, system wrapped in agent_dag).
    workload: str = "fixed"
    turns: int = 8
    sessions: int = 150
    # Semantics-changing mechanisms (M-FP8LINEAR) are searched only if allowed:
    # the official checkpoint precision is the baseline (U1).
    allow_precision: bool = False

    @property
    def slo(self):
        return {"ttft_s": self.ttft_s, "tpot_s": self.tpot_s}

    @property
    def context(self):
        """Longest call context (input + output): the KV legality check."""
        return workloads.agent_max_context(self.turns) if self.workload == "agent" else self.isl + self.osl

    @property
    def instances(self):
        return self.gpus // INSTANCE_GPUS

    @property
    def tps(self):
        return (4, 8) if self.hardware == "h200" else (4,)


def side_layout(d, side):
    return getattr(d, side + "_ep"), getattr(d, side + "_pp")


def set_layout(d, side, layout):
    return replace(d, **{side + "_ep": layout[0], side + "_pp": layout[1]})


def _models(dep, speedup):
    models = models_for(dep)
    for side, k in (speedup or {}).items():
        m = models[side]
        models[side] = replace(m, coef=[c / k for c in m.coef])
    return models


def run(pb: Problem, dep: Deployment, rate: float, speedup=None) -> dict:
    cfg = te.tier1_config(tier2_config(dep), _models(dep, speedup), retention=dep.kv_retention)
    with tempfile.TemporaryDirectory(prefix="opt-") as tmp:
        tmp = Path(tmp)
        if pb.workload == "agent":
            cfg = dict(cfg, system={"kind": "agent_dag", "inner": cfg["system"]})
            rows = workloads.agent_sessions(max(pb.sessions, round(rate * pb.duration)), pb.turns, rate, seed=1,
                                            streams=dep.kv_retention)
        else:
            rows = workloads.fixed(max(pb.requests, round(rate * pb.duration)), rate, pb.isl, pb.osl, seed=1)
        te.write_config(cfg, tmp / "c.toml")
        wl = workloads.write(rows, tmp / "w.jsonl")
        te.run(tmp / "c.toml", wl, tmp / "r.jsonl")
        return steady.metrics(tmp / "r.jsonl", dep.gpus, pb.slo, sides=(
            dep.prefill_instances * dep.width("prefill"), dep.decode_instances * dep.width("decode")))


def representative_round(pb: Problem, d: Deployment, side: str):
    """One round of a side at its configured size: a prefill chunk of `budget`
    tokens at half the prompt as context; a full decode batch at mean context
    ISL + OSL/2. Returns (stage, tp, ep, pp, batch)."""
    if side == "prefill":
        q = min(d.prefill_budget, pb.isl)
        return "p", d.prefill_tp, d.prefill_ep, d.prefill_pp, [(0, q, max(0, pb.isl // 2 - q))]
    kv = pb.isl + pb.osl // 2
    return "d", d.decode_tp, d.decode_ep, d.decode_pp, [(i, 1, kv) for i in range(d.decode_max_batch)]


@lru_cache(maxsize=256)
def _elasticity(hw, stage, tp, ep, pp, batch):
    base = tier1.bound_of(tier1._template(hw, tp, ep, stage, None, pp=pp), list(batch), hw)
    out = {}
    for kind in ("compute", "memory", "network"):
        t = tier1.bound_of(tier1._template(hw, tp, ep, stage, {kind: 1 / FACTOR}, pp=pp), list(batch), hw)
        out[kind] = (t / base - 1) / (FACTOR - 1)
    return {"round_ms": base * 1e3, "elasticity": out}


def attribution(pb: Problem, d: Deployment, side: str) -> dict:
    """Graph-bound resource elasticity of one representative round of `side`
    (1 = round time scales fully with that resource; screening evidence)."""
    stage, tp, ep, pp, batch = representative_round(pb, d, side)
    return _elasticity(d.hardware, stage, tp, ep, pp, tuple(batch))


def diagnose(evidence: dict | None, granularity: str = "queue") -> str:
    """Bottleneck side from the feedback of the first failing probe above capacity."""
    if granularity == "none":
        return "unknown"
    if evidence is None:
        return "none"
    if granularity in ("queue", "resource") and evidence.get("queue"):
        return evidence["queue"]
    return "prefill" if evidence["ttft_violation"] >= evidence["tpot_violation"] else "decode"


def evaluate(pb: Problem, dep: Deployment, speedup=None) -> dict:
    t0 = time.perf_counter()
    cap = steady.capacity(lambda r: run(pb, dep, r, speedup), 0.25, 1.0, steps=6)
    fails = [m for r, m in cap["probes"].items() if not steady.meets(m) and float(r) > cap["rate"]]
    above = min(fails, key=lambda m: m["offered_rate"]) if fails else None
    evidence = None if above is None else {
        "rate": above["offered_rate"], "ttft_violation": above["ttft_violation"],
        "tpot_violation": above["tpot_violation"], "queue": above.get("queue"),
        "decode_growth": above.get("decode_growth"), "queue_growth": above.get("queue_growth")}
    m = cap.get("at_rate")
    side = diagnose(evidence)
    return {"key": dep.key, "deployment": asdict(dep), "capacity_rps": cap["rate"],
            "tokens_per_s_gpu": m["tokens_per_s_gpu"] if m else 0.0,
            "tokens_per_s_user": m["tokens_per_s_user"]["p50"] if m else 0.0,
            "prefill_tokens_per_s_gpu": m["prefill_tokens_per_s_gpu"] if m else 0.0,
            "decode_tokens_per_s_gpu": m["decode_tokens_per_s_gpu"] if m else 0.0,
            "ttft_p99_ms": m["ttft_ms"]["p99"] if m else None, "tpot_p50_ms": m["tpot_ms"]["p50"] if m else None,
            "jct_p50_s": m["jct_s"]["p50"] if m else None, "jct_p99_s": m["jct_s"]["p99"] if m else None,
            "bottleneck": side, "evidence": evidence, "speedup": speedup or {},
            "resource": attribution(pb, dep, side) if side in ("prefill", "decode") else None,
            "wall_s": time.perf_counter() - t0}


def legal(pb: Problem, dep: Deployment) -> bool:
    """Deployments the engine accepts, memory holds (same check as
    simulate.sh) and Tier-1 surrogates cover."""
    if check(dep, pb.context):
        return False
    try:
        models_for(dep)
    except KeyError:
        return False
    return True


def refill(pb: Problem, d: Deployment) -> Deployment | None:
    """`d` with as many decode instances as fill the GPU budget exactly (None
    if the remainder is not a positive multiple of the decode width)."""
    if d is None:
        return None
    rest = pb.gpus - d.prefill_instances * d.width("prefill")
    width = d.width("decode")
    if rest < width or rest % width:
        return None
    return replace(d, decode_instances=rest // width)


def moves(pb: Problem, dep: Deployment, side: str, resource: dict | None = None) -> list[tuple[int, Deployment, str]]:
    """Moves targeted at `side` first (priority 0), then one probe per dimension
    (1). `resource` = attribution(side): if the side is network-bound, layouts
    whose MoE group fits one NVLink node lead, smallest EP first."""
    out = []
    b = BUDGETS.index(dep.prefill_budget)
    # one decode instance <-> its width in prefill instances
    k = max(1, dep.width("decode") // dep.width("prefill"))
    shift_p = refill(pb, replace(dep, prefill_instances=dep.prefill_instances + k))
    shift_d = refill(pb, replace(dep, prefill_instances=dep.prefill_instances - k)) \
        if dep.prefill_instances > k else None
    other_tp = [replace(dep, prefill_tp=t, decode_tp=t) for t in pb.tps if t != dep.prefill_tp]
    k = DECODE_BATCHES.index(dep.decode_max_batch)
    batch = lambda j: replace(dep, decode_max_batch=DECODE_BATCHES[j], decode_budget=DECODE_BATCHES[j])
    other_layout = {"prefill": [(set_layout(dep, "prefill", l), f"PP{l[1]} x EP{l[0]}") for l in LAYOUTS
                                if l != side_layout(dep, "prefill")],
                    "decode": [(d, f"PP{l[1]} x EP{l[0]}") for l in DECODE_LAYOUTS if l != side_layout(dep, "decode")
                               for d in [refill(pb, set_layout(dep, "decode", l))] if d is not None]}
    chunks = [(replace(dep, prefill_moe_chunks=c), c) for c in MOE_CHUNKS if c != dep.prefill_moe_chunks]
    dispatch = [(replace(dep, prefill_dispatch=v), v) for v in DISPATCH if v != dep.prefill_dispatch]
    linear = {s: [(replace(dep, **{f"{s}_linear": v}), v) for v in LINEAR if v != getattr(dep, f"{s}_linear")]
              if pb.allow_precision else [] for s in ("prefill", "decode")}
    # M-KVRETAIN keys on token streams, which only agent sessions declare.
    retain = [replace(dep, kv_retention=not dep.kv_retention)] if pb.workload == "agent" else []
    if side == "prefill":
        if b + 1 < len(BUDGETS):
            out.append((0, replace(dep, prefill_budget=BUDGETS[b + 1]), "TTFT breaks first: larger prefill chunks"))
        if shift_p is not None:
            out.append((0, shift_p, "TTFT breaks first: move one instance D->P"))
        out += [(0, d, f"TTFT breaks first: prefill layout {name}") for d, name in other_layout["prefill"]]
        out += [(0, d, f"TTFT breaks first: MoE chunks {c} (M-MOECHUNK)") for d, c in chunks
                if c > dep.prefill_moe_chunks]
        out += [(0, d, "TTFT breaks first: dispatch from every TP rank (M-RANKDISPATCH)")
                for d, v in dispatch if v == "ranks"]
        out += [(0, d, "TTFT breaks first: FP8 linears (M-FP8LINEAR, quality UNVERIFIED)")
                for d, v in linear["prefill"] if v == "fp8"]
        out += [(0, d, "prefill re-computes session history: keep it (M-KVRETAIN)")
                for d in retain if d.kv_retention]
    elif side == "decode":
        if shift_d is not None:
            out.append((0, shift_d, "TPOT breaks first: move one instance P->D"))
        if k + 1 < len(DECODE_BATCHES):
            out.append((0, batch(k + 1), "decode saturates: larger decode batch"))
        out += [(0, d, "TPOT breaks first: other attention TP") for d in other_tp]
        out += [(0, d, f"TPOT breaks first: decode layout {name}") for d, name in other_layout["decode"]]
        out += [(0, d, "decode HBM-bound: FP8 linears (M-FP8LINEAR, quality UNVERIFIED)")
                for d, v in linear["decode"] if v == "fp8"]
    el = (resource or {}).get("elasticity", {})
    if side in ("prefill", "decode") and el and max(el, key=el.get) == "network" and el["network"] >= NETWORK_BOUND:
        local = sorted((d.key for d, _ in other_layout[side] if getattr(d, side + "_ep") <= NODE_GPUS
                        < getattr(dep, side + "_ep")), key=lambda k: next(
                            getattr(d, side + "_ep") for d, _ in other_layout[side] if d.key == k))
        lead = [m for k in local for m in out if m[1].key == k]
        out = [(0, d, f"{side} round network-bound (elasticity {el['network']:.2f}): {why.split(': ')[-1]}, EP inside one node")
               for _, d, why in lead] + [m for m in out if m[1].key not in local]
    probes = ([(1, shift_p, "probe D->P")] if shift_p is not None else []) + \
             ([(1, shift_d, "probe P->D")] if shift_d is not None else []) + \
             [(1, d, f"probe TP{d.prefill_tp}") for d in other_tp] + \
             [(1, replace(dep, prefill_budget=BUDGETS[j]), f"probe budget {BUDGETS[j]}")
              for j in (b - 1, b + 1) if 0 <= j < len(BUDGETS)] + \
             [(1, d, f"probe {s} layout {name}") for s, ds in other_layout.items() for d, name in ds] + \
             [(1, batch(j), f"probe decode batch {DECODE_BATCHES[j]}") for j in (k - 1, k + 1)
              if 0 <= j < len(DECODE_BATCHES)] + \
             [(1, d, f"probe prefill MoE chunks {c}") for d, c in chunks] + \
             [(1, d, f"probe prefill dispatch {v}") for d, v in dispatch] + \
             [(1, d, f"probe {s} linear {v}") for s, ds in linear.items() for d, v in ds] + \
             [(1, d, f"probe kv retention {d.kv_retention}") for d in retain]
    return [m for m in out + probes if m[1] is not None]


def ordered(pb: Problem, dep: Deployment, side: str, resource: dict | None = None) -> list[tuple[Deployment, str]]:
    """The legal candidates in the order the agent would evaluate them (deduped)."""
    seen, out = {dep.key}, []
    for _, d, why in sorted(moves(pb, dep, side, resource), key=lambda m: m[0]):
        if d.key not in seen and legal(pb, d):
            seen.add(d.key)
            out.append((d, why))
    return out
