# Phase 2 report — generalized execution, Tier-1 cost query, agent workloads

Date 2026-11-25. Engines: simthesizer `600c112`, rooflang `dbe1486`
(branches `inferevo/phase1-base`). Evidence: SIMULATED (Tier-2) and Tier-1
surrogate (screening) — no hardware verification.

## Delivered

| Item | Result | Evidence |
|---|---|---|
| Kimi layouts H200/B300, TP 1/2/4/8, EP \| 896 | done (Phase 1, D15) | E0004 |
| SIM-DEFECT-1 batched weight reads | fixed; decode with 2 req/group ~15 → 8.7 ms | E0008, re-baselined E0007 |
| SIM-DEFECT-2 fixed state in non-shared handoff | fixed | Rust tests |
| Tier-1 `step_model` runner (D18/D19) | engine-native; scheduler, P/D, KV topology unchanged | E0009–E0012 |
| Tier-1 accuracy, frozen workload | H200 TP4: TTFT 1.7 %, TPOT 2.2 %, e2e 2.2 % (median); B300 TP4: 2.0 / 1.1 / 1.1 % | E0011, E0014 |
| Tier-1 accuracy under load (100 req) | x1: tok/s/GPU +3.8 %, TTFT p50 +0.8 %, TPOT p50 −6.1 %; x4: +4.1 / +0.1 / −6.8 % | E0013 confirm |
| Agent programs through P/D (`agent_dag`, D20) | exact vs native DAG execution | Rust test |
| Tier-1 on agent sessions (contexts ≤ ~10k) | mean JCT −0.6 %, session JCT median 5.6 % (max 22 %), call TTFT median 6 % (max 120 %), TPOT 2.2 % | E0015 |
| Speed | Tier-1 ≈1 s per run vs 162 s – 76 min Tier-2 (150–4000×) | E0011–E0015 |

## Findings

1. Saturation is prefill-bound (H200 TP4 plateau 44.8, B300 TP4 69.2
   tok/s/GPU at Tier-1; TTFT explodes, TPOT flat) — E0013.
2. Prefill MoE dispatch/combine serialize on the Attention-group leader's
   NIC across nodes (128-token round 50 ms vs 29 ms graph bound) — E0011.
3. Default prefill budget (128 tokens) makes long prompts slow (14k tokens
   → 5.6 s TTFT) — E0015.
4. Agent sessions: tails are queue-order sensitive; aggregate metrics are
   robust at Tier-1.

## Known limits

Tier-1 TPOT ~6–7 % optimistic under load (decode fit on graph bounds);
Tier-1 per-call tails unreliable in closed loops; prefill surrogate fitted
for budget 128 only; B300 prefill/decode fits only TP4; agent sessions are
synthetic (no public Kimi agent trace); calls re-prefill full history (no
KV retention); routing seeded uniform.
