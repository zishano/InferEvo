# Phase 3 report — dense feedback and autonomous configuration search

Date 2026-11-27. Reference Tier-2 lowering: grouped (MC-5). Evidence level
of all capacities: SIMULATED (Tier-1 surrogates validated per D23 against
Tier-2; Tier-2 = RoofLang roofline + Simthesizer DES, no hardware data).

## Acceptance (ROADMAP Phase 3)

| Criterion | Status | Evidence |
|---|---|---|
| Agent direction matches counterfactual sensitivities on >= 3 regimes | met (5/5) | E0042: prefill/decode throughput, TTFT/TPOT latency, balanced optimum |
| Ablation with/without feedback granularity | met | E0042 (diagnosis), E0043 (whole searches, held-out problems) |
| Frozen evaluator + ledger | met | `agent/search.Problem`, `adapters/steady.py` (D22), JSONL ledgers |

## What the agent sees and does

- Evaluator: SLO capacity (max Poisson rate with >= 90 % attainment and no
  growing prefill queue or decode backlog; D22), per-side throughput.
- Feedback (D24): failing probe just above capacity -> side (queue growth,
  else TTFT vs TPOT) -> resource of that side's round (elasticity).
- Moves: side-targeted first (budget, P<->D, TP, decode batch, layout; layouts
  with EP inside a node first when network-bound), then one probe per
  dimension; best-first expansion (D21).

## E0042 — diagnosis vs counterfactuals (128 x H200)

| Regime | Counterfactual gain P x2 / D x2 | True side | resource | queue | slo | none |
|---|---|---|---|---|---|---|
| R1 prefill throughput | +177 % / 0 % | prefill | ✓ 1 | ✓ 4 | ✓ 4 | ✗ 7 |
| R2 decode throughput | 0 % / +100 % | decode | ✓ 1 | ✓ 1 | ✗ 4 | ✗ 2 |
| R3 TTFT latency (16K prompts) | +199 % / 0 % | prefill | ✓ 1 | ✓ 4 | ✓ 4 | ✗ 7 |
| R4 TPOT latency (20 ms) | −5 % / +113 % | decode | ✓ 1 | ✓ 1 | ✓ 1 | ✗ 2 |
| R5 MC-5 optimum | −7 % / +31 % | decode | ✓ 1 | ✓ 1 | ✗ 4 | ✗ 2 |

✓/✗ = diagnosis matches; number = evaluations until the best one-step move.
Resource attribution: EP16 prefill rounds network-bound (0.73–0.75), decode
rounds memory-bound (0.95). At R5 every one-step move loses (best −4 %): the
optimum is a local optimum of the space.

## E0043 — whole searches on held-out problems (16 evaluations)

| Problem | resource | queue | slo | none |
|---|---|---|---|---|
| 64 GPU, 2000/500, TTFT 2 s (grid: 99 legal, optimum 13.02) | 13.02 (eval 16) | 13.02 (16) | 10.72 | 7.50 |
| 96 GPU, 4096/2048 | 5.13 (eval 10) | 5.13 (14) | 3.26 | 3.96 |
| 128 GPU, 24576/512, TTFT 10 s | 6.51 (14) | 6.51 (14) | 6.51 (14) | 0.61 |

Queue-level feedback carries most of the value; resource attribution adds a
small held-out gain. Without directed feedback the search reaches 9–77 % of
the best capacity.

## Limits

- Tier-1 coverage bounds the space (ETP, some decode layouts, batch 128 are
  Tier-2-only); SIM-DEFECT-3 is guarded, not fixed.
- Resource attribution is a graph bound of one representative round (no
  contention); it ranks moves, it is not a performance claim.
- The agent policy is rule-based; the LLM policy path exists but has not been
  run against a model service (needs user approval).
