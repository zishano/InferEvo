# MC-1 Kimi-K3 official checkpoint precision

- Decision: user, 2026-11-24 (DECISIONS U1).
- Evidence: HF `moonshotai/Kimi-K3` rev c7945ea `config.json` quantization
  (MXFP4 only for routed experts, `kv_cache_scheme=null`) and checkpoint tensor
  names (`validation/sources/kimi_k3_tensor_names_layers_0_1.txt`).
- Change: dtypes moved to RoofLang `programs/models/kimi_k3/config.py`
  (`NONEXPERT_W_DTYPE=bf16`, `EXPERT_W_DTYPE=fp4`, `MLA_KV_DTYPE=bf16`,
  `MLA_COMPUTE_DTYPE=bf16`); `model.py` and Simthesizer bridge
  (`kimi_k3.py`, `kimi_k3_shared.py`) read them; state bytes use `dtype_bytes`.
- Consequences: logical weights 1.511 → 1.560 TB (matches published ~1.56 TB);
  16×H200 TP2/EP16 no longer fits (147.2 GB/GPU) → shared Kimi path generalized
  (TP 1/2/4/8, EP | 896, H200/B300), default TP4/EP16. MLA KV 13,824 → 27,648
  B/token/rank.
- Effect (E0003, serial roofline ratio, MC-1 only): decode b64/8k ×1.04,
  decode b512/128k ×1.59, prefill 8k ×1.44.
- Validation: RoofLang suite (only pre-existing stale 512k test fails);
  Simthesizer Kimi tests re-derived from first principles; see PROGRESS.md.
