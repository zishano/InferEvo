"""Check the RoofLang Kimi-K3 definition against the official HF config.

Official source: https://huggingface.co/moonshotai/Kimi-K3 (config.json,
snapshot in inferevo/sources/kimi_k3_hf_config.json, sha256 recorded in
TECH_SOURCE_MAP.md). The official config indexes layers from 1; RoofLang
indexes from 0.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from adapters.engines import enable_bridge_imports

OFFICIAL = Path(__file__).parent / "sources" / "kimi_k3_hf_config.json"

# official text_config key -> RoofLang config constant
STRUCTURE = {
    "hidden_size": "D",
    "num_hidden_layers": "N_LAYERS",
    "num_attention_heads": "H",
    "q_lora_rank": "Q_LORA",
    "kv_lora_rank": "KV_LORA",
    "qk_nope_head_dim": "QK_NOPE_HD",
    "qk_rope_head_dim": "QK_ROPE_HD",
    "v_head_dim": "V_HD",
    "intermediate_size": "DENSE_INTER",
    "first_k_dense_replace": "DENSE_LAYERS",
    "num_experts": "N_EXPERTS",
    "num_experts_per_token": "TOPK",
    "num_shared_experts": "N_SHARED_EXPERTS",
    "moe_intermediate_size": "MOE_INTER",
    "routed_expert_hidden_size": "ROUTED_D",
    "moe_router_activation_func": "ROUTER_SCORING_FUNC",
    "vocab_size": "V",
    "attn_res_block_size": "ATTN_RES_BLOCK",
}


def official_text_config() -> dict:
    return json.loads(OFFICIAL.read_text())["text_config"]


def structure_findings() -> list[dict]:
    """Mismatches between official structure and RoofLang constants."""
    enable_bridge_imports()
    from rooflang.programs.models.kimi_k3 import config as C

    t = official_text_config()
    la = t["linear_attn_config"]
    findings = []
    for key, const in STRUCTURE.items():
        if t[key] != getattr(C, const):
            findings.append({"field": key, "official": t[key],
                             "rooflang": getattr(C, const)})
    pairs = {
        "full_attn_layers(0-based)": (
            tuple(i - 1 for i in la["full_attn_layers"]), tuple(C.FULL_ATTN_LAYERS)),
        "kda_layers(0-based)": (
            tuple(i - 1 for i in la["kda_layers"]), tuple(C.KDA_LAYERS)),
        "kda_head_dim": (la["head_dim"], C.KDA_HD),
        "kda_num_heads": (la["num_heads"], C.H),
        "kda_short_conv": (la["short_conv_kernel_size"], C.KDA_CONV),
    }
    for name, (official, ours) in pairs.items():
        if official != ours:
            findings.append({"field": name, "official": official, "rooflang": ours})
    return findings


def precision_findings() -> dict:
    """Compare per-weight storage dtypes with the official quantization scope.

    Official: only routed-expert Linear layers are MXFP4 (group 32, uint8
    scales); attention, shared experts, dense MLP and lm_head are ignored by
    quantization and stay BF16; kv_cache_scheme is null.
    """
    enable_bridge_imports()
    from rooflang_bridge import kimi_k3 as K

    q = official_text_config()["quantization_config"]
    group = q["config_groups"]["group_0"]["weights"]
    nodes, _ = K.compact_graph(1, 1, 1)
    inventory = K._weight_layout(nodes, 1, 1)
    by_dtype = Counter()
    for row in inventory:
        by_dtype[row["dtype"]] += row["local_bytes_per_gpu"]
    # Official: only routed experts are quantized; everything else BF16 (or
    # FP32 for small KDA parameters).
    nonexpert_quantized = sum(
        r["local_bytes_per_gpu"] for r in inventory
        if r["partition"] != "whole_expert_ep_shard"
        and r["dtype"] in ("fp8", "fp4", "ue8m0"))
    expert_dtypes = sorted({r["dtype"] for r in inventory
                            if r["partition"] == "whole_expert_ep_shard"})
    mla = next(s for s in K.state_schema(1, 1) if s["kind"] == "mla_compressed_kv")
    deviations = []
    if nonexpert_quantized:
        deviations.append(f"nonexpert weights quantized: {nonexpert_quantized} bytes")
    if expert_dtypes != ["fp4", "ue8m0"]:
        deviations.append(f"routed expert dtypes {expert_dtypes} != MXFP4 + ue8m0 scales")
    if mla["dtype"] != "bf16":
        deviations.append(f"MLA KV cache {mla['dtype']} != bf16 (kv_cache_scheme=null)")
    total = sum(by_dtype.values())
    return {
        "official_quantized_targets": q["config_groups"]["group_0"]["targets"],
        "official_ignore": q.get("ignore", []),
        "official_group_size": group["group_size"],
        "official_kv_cache_scheme": q.get("kv_cache_scheme"),
        "rooflang_bytes_by_dtype": dict(by_dtype),
        "rooflang_total_bytes": total,
        "nonexpert_quantized_bytes": nonexpert_quantized,
        # Evidence: sources/kimi_k3_tensor_names_layers_0_1.txt.
        "deviations": deviations,
    }
