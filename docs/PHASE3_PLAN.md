# Phase 3 plan — dense feedback and autonomous configuration search

Inputs from Phase 2: Tier-1 (≈1 s/run, 1–7 % error, TPOT ~6 % optimistic
under load) and Tier-2 (minutes–hours/run) on the same engine; agent_dag for
closed-loop workloads; load sweeps showing prefill-bound saturation.

## 1. Cross-layer feedback record (one per run)

| Level | Fields | Source (Tier-1 / Tier-2) |
|---|---|---|
| session | JCT, calls, tool time, think time | result DAG / same |
| request/call | TTFT, TPOT p50/p99, queue = TTFT − own prefill service | result + step log / result + rounds |
| instance | busy fraction, mean batch, tokens/round, P vs D share | step-node trace / rounds.csv |
| round | duration, batch aggregates, features | step trace / RoundBegin + rounds |
| operator class | base vs elapsed (contention stretch), bound | — / operator-costs.csv (targeted) |
| link / GPU | traffic, HBM/compute load | — / reports (targeted) |
| sensitivity | elasticity per resource (CM-4 probes) | — / Tier-2 small workloads |

Bottleneck classes (spec 4.4.3): queueing (TTFT ≫ service), instance
saturation (busy ≈ 1 on one side), contention-exposed (stretch ≫ 1 on an
op class), imbalance (group/instance spread), resource-bound (elasticity).

## 2. Controller loop (agent/)

State machine (D12): BASELINE → FEEDBACK → DIAGNOSE → PROPOSE → SCREEN
(Tier-1) → CONFIRM (Tier-2, Pareto candidates only) → ARCHIVE → next.
Ledger (append-only JSONL) records every proposal, its rationale (which
bottleneck), runs, receipts and verdicts. The LLM agent proposes; fixed code
evaluates (frozen workloads, SLOs, metric formulas) and writes receipts.

## 3. First design space (parameter level, existing mechanisms)

P:D instance counts (fixed 32/64 GPUs), Attention TP per side, prefill token
budget, max batch, EP (16/32). Objectives: tok/s/GPU, tok/s/user, TTFT p99,
agent JCT; feasibility from validation/capacity. Expected first wins from
evidence: larger prefill budget (TTFT of long prompts: 14k tokens = 5.6 s at
budget 128), more P instances (E0013 prefill-bound).

## 4. Mechanism queue (Phase 4 entry criteria)

C-PDRATIO, C-DISPATCH, C-OVERLAP, C-EPNODE, C-KVRETAIN (MECHANISM_CATALOG).
A mechanism enters when the diagnosed bottleneck persists after parameter
search exhausts the existing design space.

## 5. Tier-1 maintenance

Refit decode on engine rounds of loaded Tier-2 runs (fix −6 % TPOT bias);
fit per (hw, TP, EP, budget) as layouts are added; every confirmation run
saves result.jsonl for refits.
