# CM-5 MLA prefill uses full-MHA arithmetic

- Source: upstream yzygitzh/rooflang #10 (4ff475d), `github/rooflang`.
- Change: `KimiK3MlaAttn.attention_mode` (auto: causal → prefill); prefill
  width `qk_head_dim + v_head_dim` (Kimi 320) instead of absorbed
  `kv_cache_dim + kv_lora_rank` (1088). Split passes propagate the mode. Class
  rename to `MlaAttn` deferred (alias only) to avoid bridge churn.
- Effect (E0003 ablation, on top of MC-1): prefill 8k ×0.925, 32k ×0.77
  (H200), ×0.76 (B300); decode ×1.000.
- Validation: 4 ported upstream kernel tests; Simthesizer MLA shard FLOPs
  re-derived (2·8·512·48·96 + 2·48·21·320).
