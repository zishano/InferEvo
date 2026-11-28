# Progress

Last update: 2026-11-27. Phase: **0, 1, 2, 3 DONE** (docs/PHASE3_REPORT.md); **Phase 4 in progress**:
lifecycle framework done (D25), first mechanism M-MOECHUNK PROMOTED. Engines: simthesizer
60dc2d0, rooflang dbe1486 (Phase 1 cost model 62e82ce9837c; later bridge commits change its hash).

## How to resume
1. Read `RESEARCH_SPEC.md`, `AGENTS.md` (layout), `docs/ARCHITECTURE.md`,
   `docs/DECISIONS.md`, this file, `docs/ROADMAP.md`, `docs/changes/README.md`.
2. Engines (`engines/`, D17): `simthesizer` = branch `inferevo/phase1-base`
   of `../simthesizer`; `rooflang` = branch `inferevo/phase1-base` of
   `../rooflang`. User trees are never modified.
3. Tests: `./run_tests.sh` (InferEvo). Simthesizer: in `engines/simthesizer`
   `cargo test --workspace --locked` and `.venv/bin/python -m unittest
   discover -s tests` (~10 min; some tests need `cd tests`). RoofLang: in
   `engines/`, `PYTHONPATH=$PWD ../../.venv/bin/python -m pytest rooflang/tests`.
4. Python entry: `./ie.sh <script>`. Candidates:
   `./ie.sh -m mechanisms.isolation candidate <repo> <name>`.

## Done
- Phase 0 audit and docs (see REFERENCE_ANALYSIS.md); E0001, E0002.
- U1 decided by user: official precision.
- MC-1 official precision (cost_models/MC-1.md): logical weights 1.560 TB
  (matches checkpoint); dtypes single-sourced in RoofLang Kimi `config.py`.
- CM-5 MLA prefill full-MHA width, ported from upstream rooflang #10
  (cost_models/CM-5.md).
- D15 layout generalization: H200/B300, TP 1/2/4/8, EP | 896; default
  H200 TP4/EP16. Smoke: H200 TP4 and B300 TP2 P/D runs complete; H200 TP2
  fails with explicit OOM.
- Tests on branch: Simthesizer Rust 252/252, Python 242 OK (6 skipped);
  RoofLang 938 pass + 1 pre-existing stale (512k workloads); InferEvo 13 OK.
- E0003 (experiments/E0003_official_precision): base vs phase1 serial
  roofline — decode b512/128k ×1.59, prefill 8k ×1.33; CM-5 ablation:
  prefill 8k ×0.925, 32k ×0.77; capacity table under official precision
  (16×H200 only TP4/TP8 fit; 8×B300 TP2–8 fit).
- Observation: at batch 1, B300 TP2 TPOT (9.8 ms) > H200 TP4 (8.8 ms):
  small-batch decode is bound by per-rank BF16 nonexpert weight reads, so
  Attention TP matters more than HBM generation (hypothesis for Phase 3).

- CM-1..3 (cost_models/CM-1-3.md): H200 FP4 priced at BF16 rate (W4A16),
  H200 141 GB, B300 8 TB/s; scheduler capacity read from presets. Qwen3-235B
  EP4 no longer fits H200. InferEvo preset audit now reports no deviations.
- MC-2 (cost_models/MC-2.md): KDA recurrent state FP32 per fla source
  (`../github/flash-linear-attention` @0c5e7d9); KDA state 454,459,392 B/request.
- Full suites on branch after all changes: Rust 252/252, Python 242 OK
  (6 skipped), RoofLang 938 + 1 pre-existing stale, InferEvo 13 OK.
- E0003 `comparison-phase1b.md`: full Phase 1 contract vs audit baseline.
  H200 prefill 8k ×1.69, B300 ×1.27 → predicted H200/B300 prefill gap widens
  1.97× → 2.62× (serial-roofline ratio); decode b512/128k ×1.6–1.7 on both.
  16×H200: only TP4 (28 req @32k) / TP8 (46) fit; 16×B300 TP2: 784.

- D17 restructure: engines/ (worktrees), references/, adapters/,
  mechanisms/, validation/, agent/, experiments/, docs/.
- E0004 Kimi round differential (`validation/round_bounds.py`): all 12 engine
  rounds (H200 TP4/8, B300 TP2/4; prefill + 2 decode) within independent
  bounds; pricing identity holds. First decode round exceeds its graph bound
  by 21–29 % = exposed P→D state transfer (IB), confirmed from kv-transfers.
- CM-4 efficiency factors (docs/changes/CM-4.md); engine HEAD 3f16e77,
  rooflang a337484. Suites: Rust 252/252, Python 244 OK (5 skipped),
  RoofLang 940 + 1 pre-existing stale, InferEvo 16 OK.
- E0005 probes (smoke request): memory elasticity 0.67–0.83, network
  0.03–0.14, compute 0 → need larger workloads for compute-side diagnosis.

- E0005 on the frozen 10-request workload (pre-CM-6 cost model): TPOT
  memory elasticity 0.98–1.52 (>1: slower decode → more concurrent requests →
  larger KV reads; serving amplifies the HBM bottleneck); TTFT network
  elasticity 0.33–0.62 (EP16 spans two nodes; MoE dispatch/combine over IB);
  compute ≈ 0. Host wall 100–210 s per run.
- Calibration sources cloned (references/: frontier, infersim, llmservingsim,
  aisimulate, inference-sim) and inventoried (REFERENCE_ANALYSIS §9).
- CM-6 corrects CM-2/CM-3 from primary/measured sources: H200 150.11 GB
  driver total; B300 7.7 TB/s and FP4 14 PF (Blackwell Ultra datasheet).
  Engine HEAD 95ef7ac, rooflang dbe1486. Suites: Rust 252/252, Python 244 OK,
  RoofLang 940 + 1 pre-existing stale, InferEvo 16 OK.

- E0006 calibration (CM-7), CM-4b kernel overhead (default 0; graph ops
  finer than real kernels), math references (MLA, KDA + negative controls),
  E0001/E0004 re-run and E0007 baseline under the final model.

## Phase 2 progress
- Host-time profile (metrics-only, frozen workload, 818 rounds, 194 s):
  Rust resource DES ~79 s, Python graph build + JSON 66 s, engine/graph parse
  ~40 s; cost ∝ graph size (~60k ops per request per round).
- BLIS/inference-sim audit: `LatencyModel.StepTime(batch)` interface with a
  separate analytic latency kernel (per-step evaluation, no fitting).
- E0008: exported-graph lower bound predicts engine decode rounds within
  0.6–1.5 % but costs 0.24–2.5 s per round → Tier-1 = fitted surrogate.
- SIM-DEFECT-1 found (docs/changes/SIM-DEFECT-1.md): no cross-request
  batching in Kimi GEMMs. Blocks all throughput work.

- SIM-DEFECT-1 fixed (simthesizer c623652); E0004 unchanged, E0007 TPOT
  re-baselined (H200 TP4 9.65 ms, B300 TP4 6.29 ms peak).
- E0009 Tier-1 decode surrogate (adapters/tier1.py): vs Tier-2 engine rounds
  median 1.07 %, p90 4.0 %; 3 µs per query (D18).

- E0010 prefill surrogate on graph bounds: held-out 1.5–2.0 % median.
- Native Tier-1 runner `step_model` + attention-DP capability + SIM-DEFECT-2
  fix + scalar Kimi model + scheduler-limit conversion (D19). Rust 254/254.
- E0011 Tier-1 vs Tier-2 (frozen workload, H200 TP4): TPOT median 2.0 %,
  e2e 2.4 %; TTFT −19 % mean (prefill graph bound optimistic); wall 1.08 s
  vs 194 s.
- Diagnosis: prefill MoE dispatch/combine serialized on the group leader NIC
  (cross-node IB) → mechanism candidates C-DISPATCH/C-OVERLAP/C-EPNODE.
- E0012 prefill surrogate on engine targets (TTFT-derived rounds, no
  reports): held-out median 2.2 %. E0011 re-run: Tier-1 vs Tier-2 TTFT 1.7 %,
  TPOT 2.2 %, e2e 2.2 % median (frozen workload, not in training).
- E0013 Tier-1 load sweep (realistic 100-request trace, x0.5–x16): plateau
  ≈44.8 tok/s/GPU from ~22 req/s; TTFT p50 0.1 → 3.2 s while TPOT ≈18 ms →
  single P instance is the bottleneck (P:D ratio, prefill MoE dispatch).
  Tier-2 confirmation of x1/x4 running (confirm.py).

- Engine Tier-1 changes committed (simthesizer c39474d); suites Rust
  254/254, Python 246 OK.
- E0014 zoo: B300 TP4 Tier-1 vs Tier-2 frozen workload TTFT 2.0 %, TPOT
  1.1 %, e2e 1.1 % median. E0013 B300 TP4 sweep: plateau 69.2 tok/s/GPU vs
  H200 TP4 44.8 (Tier-1, peak); both prefill-bound at saturation.
- Tier-2 confirmation of E0013 H200 x1/x4 (100 requests) still running
  (`experiments/E0013_load_sweep/confirm.py`, log in scratch); takes hours —
  re-run with `../../ie.sh confirm.py 1 4` from that dir if lost.

- E0013 Tier-2 confirmation x1 (73 min Tier-2 vs 1.1 s Tier-1): tok/s/GPU
  +3.8 %, TTFT p50/p99 +0.8/+2.7 %, TPOT p50 −6.1 %, p99 −16 % (Tier-1 vs
  Tier-2). x4 still running.
- agent_dag (simthesizer 600c112, D20): closed-loop agent programs over P/D;
  differential test vs native DAG execution exact. Rust 255, Python 246 OK.
- adapters/workloads.agent_sessions: synthetic coding-agent sessions
  (assumed token/time ranges; growing history; tool-gated release).
- E0015 running: long-context Tier-1 refit (decode kv ≤32k, prefill ≤16k)
  and agent-session Tier-1 vs Tier-2 validation (JCT, per-call TTFT/TPOT).

- E0013 x4 Tier-2 confirm: tok/s/GPU +4.1 %, TTFT p50 +0.1 %, TPOT −6.8 %.
- E0015: agent sessions Tier-1 vs Tier-2 mean JCT −0.6 %, session median
  5.6 %; long-context prefill refit held-out 0.7 % median.
- Phase 2 report: docs/PHASE2_REPORT.md.

## Phase 3 progress
- adapters/feedback.py: run record (calls, sessions, instances) + labels
  prefill_saturated (busy over active window), decode_batch_saturated
  (occupancy), instance imbalance, ttft_queueing (TTFT share not explained by
  own prefill service). Coherent on load sweep (queue share 0.00 → 0.96).
- step_model trace carries instance id + batch stats (simthesizer 09dc44a).
- adapters/topologies.py: exact Tier-2 configs per deployment via the
  launcher (cached in experiments/_configs).
- agent/controller.py v0: ledger, Pareto archive, rule policy (pluggable),
  search(). E0016 vs grid: same archive at x4; budget is the missing knob.

- E0017 joint prefill surrogate (budgets 128/512/2048): held-out median
  0.7 / 2.2 / 1.5 %.
- E0018 controller vs grid (64×H200 TP4, P:D × budget): finding — raising
  the prefill budget beats shifting instances (P1:D3 b2048 34.1 tok/s/GPU,
  TTFT p99 1.4 s vs baseline 27.8 / 5.5 s; P2:D2 b128 30.1). Controller found
  the throughput- and TPOT-best points in 4–6 of 9 evals; TTFT-best missed.
  Fixed proposal ordering (BFS, policy priority).

- SLO goodput + search-quality metrics (feedback/controller). E0019: rule
  policy HV 0.98 vs random 0.90 (6/9 evals); goodput at x4: budget 128 →
  2048 = 10.8 → 34.1 tok/s/GPU (Tier-1; SLO is an assumption).

- E0020 TP8 surrogates (prefill held-out median 1.9–2.8 %). Engine
  constraint found: P/D must share attention TP (Kimi MLA+KDA state
  handoff) → C-STATERESHARD candidate.
- E0021 (18-point space incl. TP): best goodput P1:D3 TP8 b2048
  39.5 tok/s/GPU at x4 (3.6x baseline, Tier-1). Search redesigned (D21:
  best-first + dimension coverage): HV 0.998 vs random 0.867.
- E0018 Tier-2 confirmation, P1:D3 b128: Tier-1 tok/s +2.1 %, TTFT p99
  −0.4 %, TPOT −4.0 % vs Tier-2 (result.jsonl kept). b2048: +1.6 %, −2.4 %,
  −4.6 %. Tier-2 confirms the budget finding: 27.21 → 33.57 tok/s/GPU
  (+23 %), TTFT p99 5.53 → 1.45 s. TP8 b2048 confirmation running.
- E0022: graph-bound decode surrogate was 3 % optimistic per loaded round
  (growing with batch: engine contention the lower bound ignores). Refit on
  ~6k loaded Tier-2 rounds (+ bound samples for coverage): end-to-end TPOT
  error −4 % → −0.3 %. Controller uses it for H200 TP4.

- agent/llm_policy.py: LLM policy plug-in (validated options only, ledger
  context, JSON answer, rule fallback); offline tests with a stub model.
- E0023 scale (32/64/128 GPUs, fixed per-instance load): best = P:D 1:1
  TP8 b2048 at every scale, goodput/GPU flat (~48–49, linear: instances
  only share the P->D link). Optimal P:D depends on the arrival process
  (bursty x4 trace → 1:3; sustained Poisson → 1:1). Controller found the
  goodput-best at all scales in 10 evals (of up to 42). Caveat: TP8 decode
  surrogate still bound-based (~3 % optimistic TPOT) until its loaded refit.
- E0024 B300 budget surrogates (held-out 1.3–2.0 %). E0025: B300 vs H200
  at 64 GPUs TP4: SLO goodput/GPU 1.55x (bursty) to 2.31x (sustained; the
  H200 config is SLO-limited by TTFT). Both prefer P1:D3 b2048.

- Tier-2 confirmation complete (3 points, result.jsonl kept). SLO goodput
  (TTFT ≤ 2 s, TPOT ≤ 50 ms): baseline 10.62 → P1:D3 TP4 b2048 33.57 (3.2x)
  → P1:D3 TP8 b2048 35.77 (3.4x; Tier-1 had 39.5 / 3.6x). Conclusion
  confirmed at Tier-2; TP8's margin over TP4 is 7 %, not 16 %.
- E0026: TP8 decode surrogate was 5.7 % optimistic under load; refit on
  loaded rounds → e2e TPOT 0.0 %. Residual: TTFT p99 −2.4 % (TP4) / −4.3 %
  (TP8), i.e. prefill under load; it matters near the SLO boundary.
- Workload realism review (user, 2026-11-26): E0016–E0026 used a 100-request
  burst (all arrivals within 2.3 s at x4; metric = drain speed) with short
  prompts (median 70 tokens). Added steady-state metrics/capacity search
  (adapters/steady.py) and long-input + agent workloads.
- E0027 long-context surrogate coverage (fixed a −28 % TP4 decode
  extrapolation at KV > 16k). E0028 steady capacity (Tier-1): optimum is
  workload-dependent — short: P1:D3 b2048 (1.8x baseline); long and agent:
  P3:D1 b2048 (prefill-bound). TP8 ≈ TP4 at steady state.
- Diagnosis: a 2048-token prefill round on one TP4 instance is bound by the
  attention-group leader's IB NIC (LB 216 ms; compute CP 126 ms; engine
  ~406 ms): all cross-node MoE dispatch/combine leaves through one NIC
  (C-DISPATCH). Modeled prefill ≈ 5k tok/s per 16 GPUs — pessimistic vs
  DeepEP-style all-rank dispatch; long-input/agent capacities inherit this.
- Decode capacity is capped by the launcher default max_batch 32 per
  16-GPU decode instance (surrogates validated only to 32 requests).
- E0029 (Tier-2 steady confirmation) stopped by user after 4.7 h (1/9
  done): Tier-2 too slow. Profile (30 requests, 21 min): Rust fluid resource
  rates 50 %, Python graph build + JSON 20 %; cause = per-request operator
  graphs (~39k ops/request/round, 1.24M ops for a 32-request decode round).
  Options awaiting user decision: batch-level lowering (semantics change,
  5–30x) and/or round-replay validation.
- Engine: attention-DP EP whitelist relaxed to power-of-two ≤ 128 (6e06298).
  Single 8K/1K request: EP64 (128 GPUs) TTFT +52 % vs EP16 — leader-NIC
  dispatch gets worse with more nodes (C-DISPATCH).
- E0030 128×H200 ISL 8K/OSL 1K optimization: best P7:D1 TP4 b2048
  (19.1 tok/s/GPU, 37 tok/s/user), 3.2x default balanced config; Pareto
  tok/s/GPU vs tok/s/user reported; steady metrics now include tok/s/user.
- Terminal interfaces: `./optimize.sh` (agent search) and `./simulate.sh`
  (manual, per-side TP/DP/EP/budget/batch). `Deployment` has per-side EP;
  `topologies.check` is the single legality check for both (engine rules,
  weights + KV memory); Tier-1 coverage enforced by `models_for`.
- MC-3 expert TP (simthesizer a468689): GPUs/instance = attention TP x DP =
  EP x ETP; `--p-etp/--d-etp` in simulate.sh (Tier-2 only, no Tier-1 fit).
  E0031: ETP costs prefill (head scatter/reduce, ETP8 +47 % TTFT), decode
  −1..−3 % at batch 1.
- MC-4 pipeline parallelism (simthesizer 53d777f, f462f7c, 29a6e84): GPUs/
  instance = PP x TP x DP = PP x EP x ETP; `--pp/--p-pp/--d-pp`; PP batches in
  flight per instance. E0032: PP2xEP8 at 16 GPUs TTFT −56 % vs EP16 (MoE
  inside one NVLink node); E0033: prefill 4.6x tok/s/GPU under a burst.
- Tier-1 PP (step_model stage split) + E0034 surrogates for PP2xEP8 TP4,
  load-calibrated in E0035 (held-out TTFT/TPOT within 1 %); optimize.sh
  searches per-side layout {PP1xEP16, PP2xEP8}.
- D22 capacity probes (duration-scaled, decode-backlog check). 128xH200
  8K/1K optimum: P4 PP2xEP8 b2048 + D4 EP16, 7.34 req/s, 57.4 tok/s/GPU
  (3.1x the EP16-only optimum).
- E0036 (fit_layout.py): PP4xEP4 TP4 and PP2xEP8 TP8 surrogates, prefill-only
  under D23 (decode held-out TPOT +6.6 / +15.5 %). E0038: E0028 under D22 —
  short capacities −4..−8 %, rankings unchanged.
- 128xH200 8K/1K optimum now P5 PP4xEP4 b512 + D3 EP16: 8.35 req/s,
  64.5 tok/s/GPU, decode-bound (TPOT p99 47 ms).
- E0037 decode max batch > 32 (EP16 TP4): loaded-round refit; batch-64 check
  (run C) pending — until it passes, decode batch stays <= 32.
- MC-5 (user chose option 1): grouped lowering is the Tier-2 reference
  (simthesizer 996bee1, 8c62d1f; InferEvo LOWERING). Tier-2 ~4–5x faster.
  E0040 re-fitted all six layouts (< 1 h), E0041 decode batch <= 64.
  Per-expert dispatch/compute overlap is now a mechanism (C-OVERLAP).
- SIM-DEFECT-3: Tier-1 decode preemption emitted zero-gap tokens (spurious
  156 tok/s/GPU optimum); trigger fixed (DP-aware scalar state, simthesizer
  kimi_k3_scalar), steady.py rejects such runs; root cause still open.
- 128xH200 8K/1K under MC-5: P4 PP4xEP4 b2048 + D4 EP16 batch 64, ~9–9.9
  req/s (decode-bound); 8.9 req/s stable at 70.7 tok/s/GPU, 39.7 tok/s/user.
  300 s probes can slightly overestimate decode-bound capacity.
- Per-side throughput (steady.metrics `sides=`; simulate/optimize print it):
  prefill = input tokens whose first token lands in the window / prefill GPUs,
  decode = output tokens emitted in the window / decode GPUs. Optimum above at
  8.9 req/s: prefill 1130 tok/s/GPU, decode 141 tok/s/GPU; 9.94: 1269 / 152.
  Prefill-side limit (OSL 2 capacity, TTFT SLO 5 s): 13.1 req/s = 1660
  tok/s/GPU, so prefill runs at ~68 % of its limit at 8.9 req/s.
- Search primitives moved to agent/search.py (Problem = frozen evaluator);
  optimize.sh --feedback {resource,queue,slo,none}. Phase 3 acceptance E0042
  (5/5 regimes match side counterfactuals) and held-out ablation E0043 (D24).
- Phase 4 framework (D25): mechanisms/registry.py (290 components, ~850
  scanned edges + components.toml), spec.py, identity.py, manage.py
  (lifecycle, receipts, lineage.jsonl). Validated impact selection on past
  commits 996bee1 / 53d777f (26 / 42 test targets).
- M-MOECHUNK (C-OVERLAP, intra-MoE): spec -> candidate -> check (27 tests,
  default-off identity) -> E0044 A–E VERIFIED -> replay -> PROMOTED. Prefill
  makespan −17.7 % PP4xEP4; registered as design variable
  (Deployment.prefill_moe_chunks, Tier-1 E0045). New 128xH200 8K/1K optimum:
  P3 PP4xEP4 mc8 + D5 EP16 b64, 11.8 req/s, 93 tok/s/GPU (+19 % vs 9.94);
  11.0 req/s recommended (stable, TTFT p99 2.6 s). Tier-1 evidence; no
  loaded Tier-2 confirmation of the new optimum yet.

- TBO screened out from E0039 evidence (catalog C-OVERLAP).
- M-RANKDISPATCH (C-DISPATCH): r0 verdict withdrawn (degenerate decode
  control, rank-0 share bias; D25 amendment), r1 VERIFIED + PROMOTED
  (simthesizer 8d70c15). Prefill EP16 −56 %, PP4xEP4 with chunks −11.6 % vs
  best existing. Lowering options generalised: Deployment
  `{side}_moe_chunks`, `{side}_dispatch`, `topologies.options()`;
  surrogates E0048. New optimum P3 PP4xEP4 mc8 + rank dispatch + D5 EP16
  b64: sustainable ~13.0 req/s, 103 tok/s/GPU (600 s); recommended 12.5
  req/s (99.4 tok/s/GPU, 43 tok/s/user, TTFT p99 2.2 s). Search needed 36
  evals: new variables sit behind the expansion frontier.

- User chose a precision-changing decode mechanism. Current precision (U1,
  official checkpoint): MXFP4 routed experts (W4A16 on H200), BF16 other
  linears, BF16 MLA KV, FP32 KDA state. Decode HBM at 64 calls x 8.7K:
  experts 63 %, BF16 linears 28 %, MLA KV ~4 %, KDA state ~4 %; KV capacity
  not binding at 8K/1K (176 requests/instance).
- M-FP8LINEAR (D26: opt-in, quality UNVERIFIED): block-FP8 W8A8 linears per
  DeepSeek-V3 3.3 (retention list verified in the primary source). r0
  verdict withdrawn (`reopen`: concurrency confound); r1 decode TPOT −12.7 %
  (batch 64), prefill −10 %; PROMOTED 88f77c2. Tier-1 E0050. With
  `--allow-fp8`: P3 PP4xEP4 mc8 rd fp8 + D5 EP16 b64 fp8 stable at 15.0
  req/s (600 s), 119 tok/s/GPU; recommended 14.5 (115 tok/s/GPU, 48.5
  tok/s/user, TTFT p99 2.8 s); official precision ~13.0.

- Credibility and Phase 6 (user: "确认", 2026-11-27): E0046 stopped (5 h,
  superseded) -> E0052 per-instance Tier-2 confirmation (running). E0051:
  all 5 mechanism claims ROBUST across 7 cost assumptions on H200 and 6 on
  B300 (tier2_config takes a CM-4 efficiency profile). B300: surrogates
  E0054, searches official / fp8, E0055 cross-hardware: B300 per GPU ~1.6x
  H200 (HBM-bound decode), same optimum structure (P3 PP4xEP4 + D5 EP16 b64);
  B300 is decode-bound, so prefill mechanisms add no B300 capacity; H200
  optimum transfers to B300 losslessly, the reverse loses 10 %.

- E0052 done: per-instance Tier-1 within ~1 % of Tier-2 for the official
  and FP8 optima (validates the Tier-1 capacity claims).
- Decode batch 128 validated (E0056). SIM-DEFECT-3 ROOT CAUSE FIXED (D27):
  Tier-1 surrogates had negative coefficients (collinear KV features) and
  priced decode recompute rows at <= 0; all 24 refitted with NNLS,
  validation preserved. Current optima (600 s, Tier-1 NNLS, E0057):
  H200 official P4 D4 b128 14.99 req/s (115 tok/s/GPU, TPOT 43 ms) or
  balanced P3 D5 b128 rd 13.60 (38 tok/s/user); H200 FP8 16.17;
  B300 official P3 D5 b128 rd 24.94 (199 tok/s/GPU, TPOT 18 ms); B300 FP8 26.62.

- Agent workload (user: "开始，继续推进"): `search.Problem(workload="agent")`,
  `optimize.sh / simulate.sh --workload agent`, session JCT in steady.metrics.
  E0058: P:D flips to 5:3 (H200) / 6:2 (B300); 8K/1K optima lose 41 % / 44 %
  on agent traffic (b128 ones are not even admissible at 20k contexts);
  chunked MoE and rank dispatch matter more (prefill-bound), batch 128 is
  +54 % on B300. Next: C-KVRETAIN (every turn re-prefills the history,
  ~88k prefill tokens per session vs ~19k with retention).

- M-KVRETAIN (C-KVRETAIN, Tier-1): token streams, KDA-state snapshot prefix
  cache, cache-aware least-work routing (hash affinity rejected in
  evaluation). PROMOTED a2f0da6. Agent optima with retention (600 s):
  H200 P4 D4 5.97 sess/s (+68 %), B300 P4 D4 b128 rd 11.69 (+66 %).

- E0060: agent workload Tier-1 vs Tier-2 per instance: Tier-1 optimistic
  (TPOT −8 %, JCT −4..−9 %) -> agent capacities carry a ~5–9 % optimism
  caveat; fix = long-context decode calibration run (open item).

- PAUSED 2026-11-28 by the user ("全部暂停"), nothing running. Phase 5 step 1
  (larger decode instances): graph bound EP16 b128 281 -> EP32 b256 387 ->
  EP64 b512 493 tok/s/GPU (H200); E0062 EP32 b256 surrogates VALIDATED (H200
  −1.6 %, B300 −2.0 %); search supports 32/64-GPU decode instances
  (`search.refill`, DECODE_LAYOUTS). Stopped, to rerun from scratch:
  E0062 EP64 b512 fits (`run.py h200-tp4-ep64-pp1-b512`, `b300-...`),
  E0061 held-out run (`heldout.py`), 128-GPU 8K/1K searches with wide decode
  (`optimize.sh --hardware h200|b300 --gpus 128 --evals 56`). Then: 600 s
  re-scores, AFD against the wide-decode baseline.

## Next (Phase 4)
0. Long-context decode calibration (agent contexts 2–20k) for the decode
   surrogates; re-score E0058/E0059 optima.
1. Precision follow-ups (opt-in): FP8 MLA KV / BF16 KDA state (~2 % each at
   8K, larger at long context and for KV capacity); credit freed FP8 weight
   bytes to admission. Official-precision mechanisms: C-EPNODE /
   C-EXPPLACE, C-KVRETAIN, C-STATERESHARD.
2. E0046 Tier-2 confirmation (running; mc8 optimum, 128 GPUs) and a Tier-2
   check of the rank-dispatch optimum.
3. Open defects: none known in Tier-1 surrogates after D27 (KV collinearity
   no longer produces negative weights; KV dependence still weak at q = 1).
4. Real LLM policy run vs rules — waiting on user OK for an external model
   service.

## Open questions for the user
None.

## Known risks
- Fine-grained Kimi simulation is slow (10 requests: 164–528 s) → Tier-1.
- All predictions are roofline upper bounds until CM-4.
- Agent scenario TOMLs on the branch use 141 GB; sealed baselines from the
  user's earlier agent runs are not comparable with branch results.
- A user Qwen agent run (Docker) may be active in `../simthesizer`; never
  touch its processes or files.
