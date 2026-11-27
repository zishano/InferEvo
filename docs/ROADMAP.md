# Roadmap

Ordering differs from RESEARCH_SPEC §14 where the audit showed it should:
integration (spec Phase 2) already exists for Kimi on H200, so Phase 1 and 2
merge into "generalize the integrated path"; a Tier-1 cost layer is pulled
forward because Tier-2 speed blocks every later phase.

## Minimum viable system (MVS)

Kimi-K3, H200 and B300, 16–32 GPUs, P/D, frozen workload set; one controller
iteration that goes baseline → feedback → diagnosis → one *new* mechanism
(not a parameter) → validation → search → Tier-2 evaluation → archive, with
every result pinned to source identity and cost-model version.

## Phase 0 — Audit (DONE 2026-11-24)
REFERENCE_ANALYSIS.md, ARCHITECTURE.md, DECISIONS.md, E0001 static audit,
InferEvo skeleton + tests.

## Phase 1 — Trustworthy cost and model baseline (DONE 2026-11-25, docs/PHASE1_REPORT.md)
1. Cost-model change process (D6): `cost_models/` change records, version
   hash over vendored RoofLang presets/kernels; drift checker between
   `../rooflang` and Simthesizer's vendored copy.
2. CM-1..3: H200 MXFP4 emulation path, H200 141 GB, B300 8 TB/s — as a new
   cost-model version on a worktree branch; old version retained.
3. Kimi precision profiles (`official`, `fp8_nonexpert`) selectable;
   default per U1.
4. Read official `modeling_kimi_linear.py`; verify KDA/MLA/AttnRes/latent MoE
   operator semantics and KDA state dtype; CPU reference for KDA chunk vs
   recurrent equivalence and MLA absorbed vs naive attention.
5. Kimi single-iteration differential: shared-path isolated round vs RoofLang
   reference DES (currently missing for Kimi).
Accept: all deviations either fixed or labelled; differential within stated
tolerance; capacity table regenerated under the selected precision.

## Phase 2 — Generalized integrated execution (DONE 2026-11-25, docs/PHASE2_REPORT.md)
1. Remove Kimi shared-path gating: B300, attn TP ∈ {1,2,4,8}, EP ∈ divisors,
   16/32/64/128 GPUs, colocated (non-P/D) instances.
2. Agent workloads through P/D (multi-LLM requests with tool calls, cross-
   turn KV/state retention).
3. Tier-1 Cost Query: memoized RoofLang iteration cost + interpolation;
   measured error vs Tier-2 and speedup.
Accept: Llama/Qwen regression unchanged (existing paired harness); Kimi B300
end-to-end; Tier-1 error budget documented.

## Phase 3 — Dense feedback and autonomous configuration search
Cross-layer feedback schema (D10), bottleneck classifier with counterfactual
probes (×2 HBM / FLOPs / link), frozen evaluator + receipts, controller state
machine; Agent chooses parameter-level search from diagnosis.
Accept: Agent's chosen direction matches counterfactual sensitivities on
≥3 distinct bottleneck regimes; ablation: with/without feedback granularity.

## Phase 4 — Controlled mechanism extension
Component manifest + impact closure, mechanism specs, worktree candidates,
regression selection, CPU math references. First targets from evidence:
non-uniform / replicated expert placement (unlocks 256 GPUs, expert hot-spot
relief), persistent attention-group placement (designed in
qwen_moe_optimization), KV/state retention for agent sessions.
Accept: one agent-implemented mechanism promoted to VERIFIED with A–E ablation.

## Phase 5 — Open cross-layer exploration
Mechanisms spanning serving DAG + RoofLang graph + placement (e.g. KDA/MLA
separate parallelism, A/F disaggregation, DCP variants). QD archive
(MAP-Elites over bottleneck/resource descriptors) + Pareto per workload.
Accept: Pareto frontier pushed under identical constraints, robust across
sensitivity band.

## Phase 6 — Cross-hardware prediction
Independent H200 and B300 optimization, mechanism transfer both ways,
parametric future hardware (HBM bw/capacity, FP4 rate, scale-up domain),
long-term knowledge base.

## Per-module source audits (spec 4.5) scheduled before development
| Module | Audit before | References |
|---|---|---|
| Cost-model change / calibration | Phase 1 | Frontier, InferSim, ServeTwin, InferaSim |
| Tier-1 Cost Query | Phase 2 | BLIS, AISimulate |
| Agent workloads | Phase 2 | AgentServeSim, Contexture, ServeTwin |
| Feedback / diagnosis | Phase 3 | ServingStudio, KD-DSE, CAFEML |
| Mechanism manager / evaluator | Phase 4 | BLIS, ServingStudio, RSIBench-Infra |
| Search / archive | Phase 5 | KD-DSE, InferaSim, AISimulate |
| Model/hardware | Phase 1 | Kimi-K3 official repo, vLLM, SGLang |
