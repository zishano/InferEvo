"""Kimi-K3 resident-memory feasibility across GPU counts and layouts.

Weight placement rules come from Simthesizer's executed Kimi bridge
(`rooflang_bridge.kimi_k3._weight_layout`, `state_schema`); this module only
generalizes the deployment grid beyond the bridge's fixed 16xH200 inspector.

Layout per instance of N GPUs: MoE EP = N with whole experts, Attention
TP x DP = N, PP = CP = DCP = 1. Results are capacity upper bounds: runtime
workspace, communication buffers and allocator fragmentation are excluded
and must be reserved via `reserve_gb` (RESEARCH_SPEC 5.3).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from validation.hardware import SPECS
from adapters.engines import enable_bridge_imports


@dataclass(frozen=True)
class Layout:
    """n_gpus per PP stage (= MoE ranks); pp stages hold disjoint layers."""
    hardware: str
    n_gpus: int
    attn_tp: int
    pp: int = 1

    @property
    def ep(self) -> int:
        return self.n_gpus

    @property
    def attn_dp(self) -> int:
        return self.n_gpus // self.attn_tp


@dataclass
class CapacityResult:
    layout: Layout
    status: str            # feasible | infeasible_* reason
    weight_bytes_per_gpu: int = 0
    usable_bytes_per_gpu: int = 0
    free_bytes_per_gpu: int = 0
    kda_state_bytes_per_request_per_gpu: int = 0
    mla_kv_bytes_per_token_per_gpu: int = 0
    max_requests: dict | None = None   # context tokens -> concurrent requests

    def row(self) -> dict:
        d = asdict(self)
        d.update(d.pop("layout"))
        d["ep"] = self.layout.ep
        d["attn_dp"] = self.layout.attn_dp
        return d


_GRAPH = None


def _graph():
    global _GRAPH
    if _GRAPH is None:
        enable_bridge_imports()
        from rooflang_bridge import kimi_k3 as K
        _GRAPH = K.compact_graph(1, 1, 1)[0]
    return _GRAPH


def state_per_request() -> tuple[int, int]:
    """(fixed KDA bytes, MLA bytes per context token), logical, one request."""
    enable_bridge_imports()
    from rooflang_bridge import kimi_k3 as K
    fixed = growing = 0
    for item in K.state_schema(1, 1):
        if item["growth"] == "fixed_per_request":
            fixed += item["bytes_per_request"]
        else:
            growing += item["bytes_per_token_per_request"]
    return fixed, growing


def evaluate(layout: Layout, contexts=(8192, 32768, 131072),
             utilization: float = 0.9, reserve_gb: float = 0.0,
             capacity_gb: float | None = None) -> CapacityResult:
    enable_bridge_imports()
    from rooflang_bridge import kimi_k3 as K
    from rooflang.programs.models.kimi_k3 import config as C

    if C.N_EXPERTS % layout.ep:
        return CapacityResult(layout, "infeasible_uneven_whole_expert_shards")
    if layout.n_gpus % layout.attn_tp or C.H % layout.attn_tp:
        return CapacityResult(layout, "infeasible_attention_tp")
    cap = capacity_gb if capacity_gb is not None else \
        SPECS[layout.hardware].facts["hbm_capacity_gb"].value
    usable = int(cap * 1e9 * utilization - reserve_gb * 1e9)
    if layout.pp > 1:
        # Worst PP stage, from the executed bridge's per-stage placement.
        from rooflang_bridge.kimi_k3_shared import stage_memory
        stages = stage_memory(layout.attn_tp, layout.ep, layout.pp)
        worst = min(stages, key=lambda st: (usable - st["weights"]) // (st["fixed"] + st["token"] * max(contexts)))
        weights, kda_gpu, mla_gpu = worst["weights"], worst["fixed"], worst["token"]
    else:
        try:
            inventory = K._weight_layout(_graph(), layout.attn_tp, layout.ep)
        except ValueError as exc:
            return CapacityResult(layout, f"infeasible_weight_partition: {exc}")
        weights = sum(r["local_bytes_per_gpu"] for r in inventory)
        fixed, per_token = state_per_request()
        # KDA state is head-sharded across the attention TP group; MLA compressed
        # KV is replicated on every TP rank (bridge contract).
        kda_gpu = fixed // layout.attn_tp
        mla_gpu = per_token
    free = usable - weights
    result = CapacityResult(layout, "feasible" if free > 0 else "infeasible_weights_oom",
                            weights, usable, free, kda_gpu, mla_gpu)
    if free > 0:
        result.max_requests = {
            ctx: layout.attn_dp * (free // (kda_gpu + mla_gpu * ctx))
            for ctx in contexts
        }
    return result


def sweep(hardware=("h200", "b300"), gpu_counts=(8, 16, 32, 64, 128, 256),
          attn_tps=(1, 2, 4, 8), **kwargs) -> list[CapacityResult]:
    return [evaluate(Layout(h, n, tp), **kwargs)
            for h in hardware for n in gpu_counts for tp in attn_tps
            if tp <= n]
