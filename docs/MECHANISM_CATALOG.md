# Mechanism Catalog

Classes: PUB (existing public mechanism), COMB (new combination), IMPR
(improvement of existing), NEW (proposed here), HYP (unverified hypothesis).
Only mechanisms executable in the current integrated path are listed as
available; nothing here is claimed as InferEvo original work.

## Available (Simthesizer shared path, audited)

| Id | Mechanism | Class | Where | Kimi | Notes |
|---|---|---|---|---|---|
| M-PD | Prefill/decode disaggregation, whole or per-layer state handoff | PUB | `pd_disaggregation.rs`, `kv.rs` | yes | single-LLM requests only |
| M-CPF | Chunked prefill, FCFS continuous batching | PUB | `schedulers/` | yes | token budgets |
| M-ATP | Attention TP (head split) + DP groups | PUB | bridge | TP 1/2/4/8, DP = GPUs/TP | |
| M-EP | Whole-expert EP, dispatch/combine | PUB | bridge | EP power of two ≤ 128 | seeded uniform routing |
| M-ETP | Expert TP: each expert over ETP GPUs, head scatter + reduce | PUB (Megatron EP×ETP) | bridge | ETP 1/2/4/8 (MC-3) | Tier-2 only; E0031 |
| M-PP | Pipeline stages of contiguous layers, boundary sends, per-stage KV ownership, PP batches in flight | PUB (vLLM/SGLang PP) | bridge, `kv.rs`, scheduler | PP 1..8 (MC-4) | E0032/E0033; Tier-1 for PP2xEP8 TP4 (E0034) |
| M-MOECHUNK | Expert-chunk pipelined grouped MoE: each expert group's experts in k token-balanced runs, in-order dispatch, per-chunk GEMMs/combine; work conserved | COMB (Tutel pipelining degree / Comet-style overlap) | bridge `lowering = {kind = "grouped", moe_chunks = k}` | k 1/2/4/8 (prefill) | ROBUST (E0051: +8..+18 % across 7 cost assumptions); first mechanism through the Phase 4 lifecycle (D25): VERIFIED + PROMOTED (simthesizer 60dc2d0); E0044 prefill makespan PP4xEP4 −17.7 % (k8), PP2xEP8 −12.7 %, EP16 −2.9 % (network-bound: little compute to hide); batch-1 decode TPOT +2.2 %; Tier-1: E0045 |
| M-RANKDISPATCH | MoE dispatch/combine from every attention TP rank (1/TP token share each, remainders rotated by expert), per-share combine + all-gather instead of leader broadcast | PUB (DeepEP-style all-rank all-to-all) | bridge `lowering = {kind = "grouped", dispatch = "ranks"}` | TP > 1, grouped | ROBUST (E0051: in-node +2..+38 %, EP16 +37..+63 %); PROMOTED (simthesizer 8d70c15; r0 withdrawn: degenerate decode control, rank-0 share bias). E0047 prefill makespan PP1xEP16 −56 %, PP2xEP8 −29 %, PP4xEP4 −23 %; with M-MOECHUNK k8 PP4xEP4 1.94 s (−11.6 % vs best existing); decode (64 concurrent) TPOT −0.6 %; Tier-1: E0048 |
| M-FP8LINEAR | Block-FP8 (E4M3, 128x128 ue8m0 scales) W8A8 non-expert projections (wq_a/b, wkv, kda_wq/k/v, output_gate, wo, routed_up/down, sw_up/down, dense); embedding, LM head, router, KDA gates, norms, AttnRes, attention, MXFP4 experts unchanged | SEM (DeepSeek-V3 FP8 recipe) — **quality UNVERIFIED, opt-in** | bridge `lowering = {..., linear = "fp8"}` | per side | performance ROBUST (E0051: prefill +4..+22 %, decode +5..+14 %), quality UNVERIFIED; PROMOTED (simthesizer 88f77c2, quality flag in lineage). E0049 r1 (equal round size): decode TPOT −12.7 % (batch 64), −16.6 % (32); decode HBM at 64 calls x 8.7K −13.5 %; prefill −10 %. r0 verdict withdrawn (concurrency confound). Admission still charges BF16 weights (conservative) |
| M-KVRETAIN | Session KV retention on prefill instances: `token_stream` workloads, stream prefix cache with KDA state snapshots every 64 tokens (LRU, charged to HBM), cache-aware least-work routing | PUB/COMB (prefix caching, Mooncake KV reuse, hybrid-state snapshots) | scheduler `enable_prefix_caching` + `state_checkpoint_tokens`; P/D `prefill_routing = "stream"`; Deployment `kv_retention` | Tier-1 only | PROMOTED (simthesizer a2f0da6). E0059 agent workload: H200 +26 % at the old optimum, +68 % re-searched (P4 D4 5.97 sess/s); B300 +49 % re-searched (10.49); graceful under memory pressure |
| M-POL | Causal policies: route, batch select, KV submit order/window | PUB/IMPR | `policies/` | yes | agent-searchable |
| M-STATE | Versioned MLA append + KDA fixed state, pins/ACK | PUB | `kv.rs` | yes | |

## Available only in RoofLang standalone (static batch, no serving)

DCP (MLA context split), KDA TP, hybrid TP/DCP, comm fusion
passes (`programs/models/kimi_k3/{dcp,kda_tp,hybrid,optimization}.py`).

## Evidence-driven candidates (from E0001)

| Id | Mechanism | Class | Trigger evidence |
|---|---|---|---|
| C-EXPPLACE | Non-uniform expert placement / expert replication (EP<N with DP replicas) | PUB (EPLB-style) | 256 GPUs infeasible for uniform whole-expert EP (896/256) |
| C-PREC | FP8 non-expert weights / FP8 KV as semantics-changing mechanism with quality flag | PUB | U1; +57 GB vs official precision |
| C-TPMLA | Decouple MLA cache replication from attention TP (e.g. DCP-style split) | PUB/COMB | concurrency peaks at TP2–4 because MLA KV replicates per TP rank |
| C-DISPATCH | Dispatch/combine from every Attention TP rank (token share per rank) instead of the group leader only; DeepEP-style multi-NIC all-to-all | PUB (DeepEP) | E0011 diagnosis: 128-token prefill round 50.2 ms vs 28.9 ms bound; dispatch stretched 22x, combine 11x; leader NIC serializes ~7 MB/layer cross-node; E0005 TTFT network elasticity 0.33–0.62 |
| C-OVERLAP | Overlap MoE all-to-all with attention/compute of the next micro-batch (two-batch overlap) | PUB (SGLang TBO / DeepSeek) | same evidence; dispatch→expert→combine serialized per layer. Intra-MoE part realised as M-MOECHUNK. Cross-micro-batch TBO screened out without a candidate: its optimistic bound (per-request lowering = one chain per request, weights read once) matched grouped_dense TTFT within 0.4 % and was 9 % slower on batched decode (E0039); real TBO also re-reads weights per micro-batch. Revisit only for network-bound decode (multi-node EP decode) |
| C-EPNODE | Keep EP within one NVLink node (EP8) with expert replication / non-uniform placement | PUB/COMB | cross-node IB dominates MoE traffic at EP16; E0032: PP2xEP8 (EP inside one node, via M-PP) TTFT −56 % at 16 GPUs |
| C-KVRETAIN | Cross-call KV / KDA-state retention for agent sessions (avoid re-prefilling the history every turn) | PUB (prefix caching) / COMB with KDA state | realised as M-KVRETAIN (Tier-1); Tier-2 shared execution still lacks prefix caching (KV runtime + KDA state restore) |
| C-PDRATIO | Non-1:1 P:D instance ratio / dynamic P-D reassignment | PUB | E0013: P saturates while D under-used (TTFT explodes, TPOT flat) |
| C-STATERESHARD | P/D with different attention TP: reshard MLA cache + KDA state at handoff (TP_p ≠ TP_d) | PUB (heterogeneous P/D, e.g. Splitwise/DistServe layouts) | Engine rejects mixed TP ("heterogeneous state requires equal P/D attention TP"); E0021: TP8 wins on both sides, mixed layouts untestable |
