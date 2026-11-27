# MC-5: grouped operator lowering is the Tier-2 reference

Kind: model contract (execution semantics). Status: APPLIED (user decision,
option 1). Branch: simthesizer 996bee1 (grouped lowering), 8c62d1f (launcher
default `--lowering grouped`); InferEvo `adapters.topologies.LOWERING`
writes it into every Tier-2 config (also cached ones).

**Before.** One operator chain per request (weights read once per Attention
group); MoE as separate dispatch + up/down kernels per (request, expert).
~39k operators per request per round; a 32-request decode round 1.24M.

**After.** One chain per Attention group: GEMMs and collectives over the
group's packed tokens; attention priced per request on its own shape and
fused into one varlen kernel carrying every request's state access; MoE as
one dispatch, one grouped up/down kernel per rank and one combine per expert
group (fused_moe / DeepEP style). Work per resource is conserved (AttnRes
has a per-call term: ~1e-5 of the total); graphs shrink 8–14x.

**Effect (E0039, 11 kept runs).** Tier-2 wall ~4–5x lower on loaded runs.
TPOT within ±1 % except batched decode (−6..−9 %: fused varlen attention
overlaps compute and HBM). TTFT +0.7..+1.8 % with EP16, +5..+10 % with
in-node EP (PP2xEP8, PP4xEP4, TP8 PP2xEP8), +23 % for multi-request prefill
rounds: grouped-GEMM MoE waits for the expert group's whole dispatch, where
the per-request lowering overlapped dispatch and compute expert by expert.
That overlap is now a mechanism to model explicitly (C-OVERLAP), not an
implicit property of the lowering.

**Re-baselined.** All Tier-1 surrogates re-fitted (E0040: six layouts,
prefill held-out within 1 %; decode passes D23 for EP16 H200 TP4/TP8, B300
TP4 and PP2xEP8 TP4; PP4xEP4 and TP8 PP2xEP8 stay prefill-only), decode
batch > 32 (E0041). Results computed with the per-request lowering
(E0012–E0039, optimize runs before 2026-11-27) keep their recorded values
under the old contract; rankings must be re-derived before reuse.
