# Phase 1 report — trustworthy cost model and Kimi-K3 baseline

Date 2026-11-25. Final cost model `62e82ce9837c`; engines: simthesizer
`d44343c`, rooflang `dbe1486` (branches `inferevo/phase1-base`).
Evidence levels per DECISIONS D11.

## Acceptance (ROADMAP Phase 1)

| Criterion | Status | Evidence |
|---|---|---|
| Model structure = official | met | E0001 structure findings = [] |
| Precision = official checkpoint | met (MC-1, MC-2) | checkpoint tensor names; fla source; logical weights 1.560 TB |
| Hardware facts sourced; invalid paths removed | met (CM-1, CM-6) | Blackwell Ultra datasheet; driver-reported capacities; audit = [] |
| Upstream cost fixes adopted | met (CM-5) | rooflang #10 port, ablation E0003 |
| Kimi differential within tolerance | met | E0004: 12/12 rounds within independent bounds; pricing identity |
| CPU math references | met (MATH_VERIFIED) | MLA prefill≡absorbed 2e-14; KDA chunked≡single pass 2e-15; negative controls 3.2/4.7 |
| Capacity regenerated under contract | met | E0001 `capacity-62e82ce9837c.md` |
| Efficiency/overhead knobs + calibration | met with limits (CM-4, CM-4b, CM-7) | E0006 dense GEMM 3.3 % median error; MoE not calibrated |
| Layouts / hardware coverage | met (D15) | H200/B300, TP 1/2/4/8, EP | 896 |

## Reference baseline (E0007; 10-request realistic sample, P1+D1 × 16 GPUs)

| Layout | Profile | TTFT ms | TPOT ms | tok/s/user | tok/s/GPU |
|---|---|---|---|---|---|
| H200 TP4/EP16 | peak (upper bound) | 50.8 | 13.5 | 81.4 | 6.55 |
| H200 TP4/EP16 | h200_dense_v1 | 58.8 | 18.1 | 58.7 | 6.19 |
| B300 TP4/EP16 | peak | 29.7 | 8.7 | 124.8 | 6.75 |
| B300 TP4/EP16 | b300_transfer_v1 | 34.2 | 10.9 | 100.6 | 6.66 |
| B300 TP2/EP16 | peak | 35.0 | 10.9 | 95.3 | 6.49 |

Evidence level: SIMULATED (peak) / calibrated assumption (profiles); no
row is hardware-verified. tok/s/GPU is arrival-limited for this workload
and does not discriminate configurations.

## Findings carried into Phase 2–3

1. Decode is HBM-bound; serving amplifies it (TPOT memory elasticity >1, E0005).
2. TTFT is network-sensitive because EP16 spans two nodes (MoE dispatch/combine over IB).
3. First decode step exposes P→D state transfer (~22 %, E0004).
4. Small-batch decode prefers higher Attention TP over a newer GPU (B300 TP2 ≈ H200 TP4).
5. H200 prefill ~2.6× B300 after removing invalid FP4 pricing (E0003).
6. Graph operators are finer than real kernels (≈4k ops on the prefill critical path): fusion modelling needed before kernel-overhead calibration.

## Known limits (labelled, not fixed)

MoE kernel efficiency uncalibrated; B300 uncalibrated (no public kernel
tables); routing is seeded uniform (no learned-router trace); KDA chunk size
64 is kernel-level; Kimi P/D supports single-LLM requests only; host cost
100–210 s per 10-request run (Tier-1 needed); one pre-existing stale RoofLang
test (512k workloads) left untouched.

## Post-report correction (2026-11-25)

SIM-DEFECT-1 (docs/changes/SIM-DEFECT-1.md) made TPOT pessimistic whenever
two requests shared an Attention group. Corrected E0007 TPOT (peak / profile):
H200 TP4 9.65 / 11.85 ms; B300 TP4 6.29 / 7.82 ms; B300 TP2 10.03 / 12.45 ms.
TTFT values above are unchanged. The table above is retained as recorded.
