# MC-2 KDA recurrent state is FP32

- Evidence: fla (`../github/flash-linear-attention` @0c5e7d9) allocates KDA
  final state FP32 in `fla/ops/common/chunk_delta_h.py:720-723` (chunk) and
  `fla/ops/kda/fused_recurrent.py:284-286`; short-conv final state uses the
  input dtype (`fla/modules/causal_conv1d/ops.py:843`). Kimi-K3
  `modeling_kimi_linear.py` stores the returned state unchanged.
- Change: RoofLang config `KDA_STATE_DTYPE="fp32"`, `KDA_CONV_STATE_DTYPE="bf16"`;
  model and Simthesizer bridge read them. Commits: rooflang 277bc2f.
- Consequences: KDA state 237,404,160 → 454,459,392 B/request (logical);
  TP4 rank 113,614,848 B; 5-token TP4 P→D handoff 455,012,352 B.
- Note: serving frameworks may choose other state dtypes (e.g. configurable
  SSM cache dtype); a lower-precision state is a semantics-changing mechanism
  candidate with a quality flag, not the baseline.
