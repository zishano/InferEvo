# Phase 4 report — controlled mechanism extension

Date 2026-11-27. Engine simthesizer `88f77c2` (branch `inferevo/phase1-base`).
Evidence levels per D11: performance claims are SIMULATED and, where
stated, ROBUST (E0051); FP8 numerics are quality UNVERIFIED (D26).

## Acceptance (ROADMAP Phase 4)

| Criterion | Status | Evidence |
|---|---|---|
| Component manifest + impact closure | met | `mechanisms/registry.py` (290 components, ~850 scanned edges + `components.toml`); impact on past commits 996bee1 / 53d777f selects 26 / 42 test targets |
| Mechanism specs, worktree candidates, regression selection | met | `mechanisms/spec.py`, `manage.py` (D25); candidates under `mechanisms/candidates/` (removed after promote/reject, branches kept) |
| One mechanism promoted to VERIFIED with A–E ablation | met (3) | M-MOECHUNK, M-RANKDISPATCH, M-FP8LINEAR; receipts `mechanisms/receipts/`, lineage `mechanisms/lineage.jsonl` |

## Lifecycle as run

spec -> `plan` (validation, extension order, predicted tests) -> `new`
(worktree) -> implement + commit -> `check` (declared vs actual diff,
release+debug build, selected regression tests, default-off byte identity on
3 frozen cases) -> `evaluate` (A–E script, JSON verdict) -> `replay` (byte
reproduction) -> `promote` (fast-forward). Corrections are first-class:
`reject` + revision (M-RANKDISPATCH r0), `reopen` with reason
(M-FP8LINEAR r0); earlier receipts are archived.

## Mechanisms

| Mechanism | What | Key result (Tier-2) | Robustness (E0051, H200 / B300) |
|---|---|---|---|
| M-MOECHUNK | expert-chunk pipelined grouped MoE (k token-balanced expert runs, in-order dispatch) | prefill PP4xEP4 −17.7 %, EP16 −2.9 % (network-bound: little compute to hide) | +8..+18 % / +10..+18 % |
| M-RANKDISPATCH | dispatch/combine from every attention TP rank, all-gather of shares | prefill EP16 −56 %, PP4xEP4 −23 % (−11.6 % on top of k8) | in-node +2..+38 %, EP16 +37..+63 % / +6..+36 %, +51..+63 % |
| M-FP8LINEAR | block-FP8 W8A8 non-expert linears (DeepSeek-V3 recipe), opt-in, quality UNVERIFIED | decode TPOT −12.7 % (b64), prefill −10 % | prefill +4..+22 %, decode +5..+14 % / +3..+17 %, +4..+13 % |

Screened out without a candidate: TBO (E0039 upper bound shows no gain).

## Capacity (128 GPUs, 8K/1K, SLO 5 s / 50 ms, 600 s probes, Tier-1)

| | H200 | B300 |
|---|---|---|
| Phase 3 end (no new mechanisms) | 9.9 req/s | — |
| + mechanisms, official precision | 13.3 req/s (105 tok/s/GPU) | 21.0 req/s (167 tok/s/GPU) |
| + FP8 (opt-in, quality UNVERIFIED) | 15.2 req/s | 23.9 req/s |
| + decode batch 128 (E0056/E0057, NNLS surrogates), official | 15.0 req/s (P4 D4, TPOT 43 ms) / 13.6 balanced (P3 D5, 38 tok/s/user) | 24.9 req/s (P3 D5, TPOT 18 ms) |
| + decode batch 128, FP8 | 16.2 req/s | 26.6 req/s |

Same optimum structure on both (P3 PP4xEP4 + D5 EP16 b64). B300 is
decode-bound (prefill 1.9x faster): prefill mechanisms keep their component
gains but add no B300 capacity; H200 optima transfer to B300 losslessly, the
reverse loses 10 % (E0055).

## Lessons recorded in the framework

- Controls must exercise the mechanism (D25 amendment): three flawed
  controls in this phase (decode at 1 call per group; FP8 decode at unequal
  concurrency; held-out decode rounds below the target batch) all from a
  slow prefill feeding decode. Evaluations now guard round size/concurrency.
- New design variables sit behind the expansion frontier: searches needed
  36–48 evaluations to reach them.
- Concurrent searches need distinct ledgers (exclusive create).

## Limits

- Capacity numbers are Tier-1; per instance under load Tier-1 is within ~1 %
  of Tier-2 for the official and FP8 optima (E0052). Tier-1 surrogates are
  non-negative fits (D27, SIM-DEFECT-3 fixed).
- FP8: model quality not simulated; activation quantization passes not
  priced; freed weight memory not credited to admission.
- Gains are model predictions under RoofLang roofline pricing; ROBUST means
  stable across the swept assumptions, not hardware-verified.
