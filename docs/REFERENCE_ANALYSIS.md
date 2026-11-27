# Phase 0 Audit: Simthesizer and RoofLang (source-verified)

Audit date: 2026-11-24. All claims below were checked in source, tests or
run artifacts; paths are relative to `../simthesizer` or `../rooflang`.
Experiment evidence: `experiments/E0001_phase0_audit/`.

## 0. Audited revisions

| Repo | Remote | Branch | HEAD | Working tree |
|---|---|---|---|---|
| Simthesizer | github.com/zishano/simthesizer (fork of casys-kaist/Simthesizer) | `test_dev` | `b05b76f` | 23 modified files (uncommitted user work) |
| RoofLang | github.com/zishano/rooflang (upstream yzygitzh/rooflang) | (local) | `8692f1d` | 9 modified + many untracked agent/analysis dirs |

Upstream Simthesizer HEAD `ebd06db` (git ls-remote, 2026-11-24) is an ancestor
of the fork: the fork = upstream + 15 local commits (RoofLang backend, P/D
shared execution, Kimi-K3, Qwen3-235B, policy search). Exact dirty-tree
digests are pinned by `inferevo.workspace.source_identity()` in every result.

Test baseline (this audit, unmodified trees):

| Suite | Result | Wall |
|---|---|---|
| Simthesizer `cargo test --workspace --locked` | 252 passed, 0 failed | 4 s (warm) |
| Simthesizer `python -m unittest discover -s tests` | 237 passed | 8.5 min |
| RoofLang `pytest tests` | 1060 passed, **1 failed** | 98 s |

The RoofLang failure (`test_dsv4_pro_find_pareto_frontier.py::test_workloads_include_intermediate_context_lengths`)
is a stale expectation: uncommitted edits added `decode-512k/prefill-512k`
workloads. Not a cost-model defect; left untouched (user's tree).

## 1. Simthesizer: actual capabilities

**Core (Rust, `crates/simthesizer-core`)**
- `engine.rs`: single-threaded DES over a mutable DAG. Four node kinds:
  compute, network, logical (opaque payload + owner handler that may add nodes
  / splice edges), resource (concurrent service demands + stream slots +
  memory reserve/release). Engine owns time only; deterministic (time, id) order.
- `resources.rs` (2.1k lines): fluid **progressive-filling** sharing of
  compute / HBM / directed-link resources, capacity reservations, stream
  exclusion. This is the actual timer for shared RoofLang execution.
- `kv.rs` (1.6k): `KvRuntime` — versioned state blocks (MLA append, KDA fixed
  versions), copy readiness, pins/ACK, release; P/D state handoff.
- Registries: each trait (`System`, `Scheduler`, `Router`, `ModelRunner`,
  `Model`, `ComputeSimulator`, `NetworkSimulator`) has a link-time `kind`
  registry (`registry.rs`). No central match. This is the natural basis for a
  Component Registry.
- Systems: `single_instance`, `multi_instance`, `moe_ep_multi_instance`,
  `pd_disaggregation`. Scheduler: `chunked_prefill` (+ prefix cache, legacy).
  Routers: round-robin, MoE load. Runners: `ring` (TP), `moe_ep`, `rooflang`.
- `policies/`: deterministic Rust policy runtime with three decisions
  (route arrived request, select batch, submit KV intents) and bounded,
  causal observations (`EnvironmentObservation`). Only for shared P/D.

**Workload**: native unit is an agent program — JSONL DAG of LLM calls and
tool calls with dependencies, think/tool durations. Closed-loop release is
real (dependency-gated). **Limitation**: `pd_disaggregation` (and therefore
every shared RoofLang / Kimi path) accepts only single-LLM requests.

**RoofLang bridge (`model_runners/rooflang.rs` + `scripts/rooflang_worker.py`
+ `scripts/rooflang_bridge/`)**, JSONL protocol v1 to a persistent Python worker:
- `exclusive` mode: worker runs a full RoofLang DES per iteration, returns µs;
  Llama-3.1-8B only.
- `shared` mode: worker builds the RoofLang graph for the batch, returns
  *work templates* (per-resource service seconds at full speed) + operation
  rows (parents, layer, KV rank, batch rows, state access). Rust expands them
  into native resource nodes; contention, KV readiness and time are native.
  Supported: Llama TP1–8, Qwen3-235B (attn TP1/2/4 × EP1–8), **Kimi-K3 only
  at H200, 16 GPU/instance, attn TP2 / DP8 / EP16, P/D (P1+D1 = 32 GPUs)**
  (`kimi_k3_shared.py:242,375`).
- Vendored RoofLang subset at `scripts/rooflang/` (rev `5cf122b`; Kimi files
  from `8692f1d`, hash manifest `programs/models/kimi_k3/SOURCE.json`).
  Drift vs `../rooflang` HEAD: comments in `runtime/simulator.py`; upstream
  `language/graph.py` adds a post-transform edge shape/dtype validation.

**Agent infrastructure**
- `.agents/skills/{task-design,sim-mapping,sim-dev,sim-validation}`: staged
  simulator-extension workflow (the Synthesizer-agent method of the paper).
- `agents/find_pareto_frontier/` (8.8k lines): Docker launcher, source
  freezing, local-baseline sealing, frozen evaluator, Pareto archive,
  independent host replay. Searches policy programs (Rust `policy/mod.rs`).
- `agents/qwen_moe_optimization/`: design for a **two-loop** system
  (inner policy search, outer mechanism extension) with a state machine
  `DRAFT→CHECKING→EVALUATING→REPLAYING→VERIFIED/FAILED/EXPLORATORY`, hypothesis
  cards, A–E ablation protocol. Largely design + partial code
  (`control.py`, `candidate.py`, `diagnostics.py`).
- Validation tooling: `tools/validate.py`, trace validators, timing checkers.

**Measured cost** (`output/kimi-perf-20261124-134752`): 10-request Kimi P/D
shared run = 528 s wall baseline; best candidate 164 s (target 53 s not met;
optimization stopped by user). Fine-grained Kimi simulation is ~10–50 s of
host time per request → unusable as the inner loop of large DSE.

## 2. RoofLang: actual capabilities

- IR (`language/`): `ComputeGraph` (networkx DAG, data edges with shape/dtype
  checks, control edges; mutation API: `insert_identity`, `fuse_kernels`,
  `split_kernel`, `dedup/dup`); `HardwareGraph` of Compute/Memory nodes with
  directed `FabricEdge`s (bw per direction, α latency, duplex) and route
  search; `Placement` (kernel→device/stream/resource_cap, tensor→memory).
- Kernels: forward (Gemm/StridedGemm, attentions incl. `KimiK3MlaAttn`,
  `KimiK3DeltaAttn`, state store, MoE dispatch/combine, norms, sampling),
  comm (AllReduce, ReduceScatter, AllGather, AllToAll, Broadcast, Scatter,
  Gather, Reduce, Send/Recv), identity, backward/optimizer.
- Passes (`language/optimization/`): column/row/head/context/batch splits,
  KV persistence split, comm fusion/elimination.
- Runtime (`runtime/simulator.py`): DES; per kernel `compute_time =
  Σ_dtype flops/peak`, `memory_time = bytes/bw` (overlapped: max), network α +
  bytes/eff_bw; device sharing by `resource_cap`, link sharing equal-split;
  OOM via tensor lifetimes. **No efficiency factors, no kernel-launch or
  software overhead**: predictions are roofline upper bounds.
- Presets: rtx6000d, h200, b300, gh200, gb300, ascend950dt.
- Models: dsv4_flash, dsv4_pro, glm_5_2_fp8, kimi_k3 (config/model + parallel
  variants `hybrid.py`, `dcp.py`, `kda_tp.py`, `mla_parallel.py`,
  `optimization.py`).
- Experiments: `find_pareto_frontier.py` sweeps TP/DCP/DP/EP/PP × batch per
  (workload, hardware, n_gpus) on single static batches (prefill/decode
  8k–512k). One Kimi case ≈ 30 s wall, ≈180k kernels. Decode cases use
  `cache_scope=snapshot_no_append` and `approximate_balanced_routing`.
- Agents: `find_pareto_frontier`, `find_new_arch` (architecture-discovery-v2:
  writable allowlist, P/S evidence tracks, review ledger), `kimi_k3_search*`,
  `kimi_k3_hybrid_search`; `analysis/dense_feedback.py` V0.1 (read-only
  per-case operator/layer/device/comm profile + baseline diff).

## 3. Kimi-K3 support and gaps

Official source: `huggingface.co/moonshotai/Kimi-K3` `config.json`
(sha256 `9710e121…9213`, snapshot `validation/sources/`).

- **Structure: matches.** All 18 mapped text_config fields plus layer maps
  (official 1-based `full_attn_layers=[4,…,92,93]` = RoofLang 0-based
  `3,…,91,92`; 69 KDA layers, 96 heads × 128, conv 4) agree
  (`tests/test_phase0.py`).
- **Precision: deviates.** Official quantization: MXFP4 (group 32, uint8
  scales) **only for routed experts**; `ignore` keeps attention, shared
  experts, dense MLP, lm_head in BF16; `kv_cache_scheme=null`. RoofLang/bridge
  stores non-expert projections as FP8 (+ue8m0 scales) and MLA cache as FP8.
  Quantified: RoofLang total 1.511 TB; with BF16 non-expert weights 1.568 TB,
  consistent with the ~1.56 TB published checkpoint. FP8 there is a deployment
  quantization choice without a cited source.
- Checkpoint evidence (HF rev `c7945ea`, `model.safetensors.index.json`,
  extract in `validation/sources/`): only `experts.N.w{1,2,3}` are
  `weight_packed`+`weight_scale` (MXFP4). `self_attn.*`, `shared_experts.*`,
  dense `mlp.*`, latent `routed_expert_{down,up}_proj`, router `gate` (+
  `e_score_correction_bias`) are plain BF16 `.weight`. RoofLang agrees on
  experts, latent projections and router; deviates (FP8) on attention,
  shared experts and dense MLP.
- `modeling_kimi_linear.py` (sha256 `9e3564c7…`) confirms KDA operator set:
  q/k/v proj + three ShortConvolution(4, silu), low-rank decay
  `f_a(7168→128)/f_b`, `b_proj` beta, full-rank output gate `g_proj`,
  gated RMSNorm, `o_proj`; chunk vs fused-recurrent kernels from fla;
  AttnRes with `attn_res_block_size=12`; latent MoE with norm. RoofLang byte
  counts for `f_a` (69·7168·128) and `b_proj` (69·7168·96·2 B) match.
- Still unverified: KDA recurrent-state dtype (set inside fla `chunk_kda`;
  bridge assumes BF16), chunk size 64 (fla kernel), real router distribution
  (only seeded uniform).

## 4. H200 / B300 support and gaps

| Item | RoofLang preset | Sourced value | Finding |
|---|---|---|---|
| H200 HBM | 144 GB | 141 GB (official) | preset over-states 3 GB/GPU |
| H200 FP4 | 1979 TFLOP/s native | no FP4 MMA on Hopper | **physics violation** (spec 5.2): MXFP4 experts priced as native |
| H200 BF16/FP8 dense | 989.5 / 1979 | same | ok |
| B300 HBM bw | 7750 GB/s | 8000 GB/s (official) | −3 % |
| B300 FP4 dense | 13500 | 13.5–15 PF (sources conflict) | open |
| B300 FP8/BF16 dense | 4500 / 2250 | third-party consistent | to verify |

Both presets: 8 GPU/node, NVSwitch, per-GPU IB NIC; collective cost uses
aggregate bandwidth (not NCCL algorithms); α 0.5 µs NVLink / 1 µs IB.
Simthesizer Kimi path: H200 only. RoofLang standalone: H200 and B300 at
any multiple of 8 GPUs.

Resident-memory feasibility (E0001; 0.9 utilization, no workspace reserve,
EP = N whole experts):

- 8×H200: infeasible at every attention TP (≥189 GB weights/GPU).
- 16×H200: TP1 infeasible; TP2 leaves only 4.2 GB/GPU (16 requests at 128k).
  The existing bridge uses the 144 GB preset, so it is more optimistic.
- 8×B300: feasible (14–70 GB free).
- 256 GPUs: **infeasible for uniform whole-expert EP** (896 / 256 not an
  integer) → requires non-uniform placement, expert replication or EP<N with
  DP replicas: a first concrete mechanism-extension target.
- Concurrency peaks at attention TP 2–4: MLA KV replicates per TP rank while
  DP shrinks.

## 5. Overlap, integration difficulty, reuse

- **Overlap**: both have a DES and contention model. In shared mode RoofLang
  supplies isolated demands and Simthesizer's resource domain does timing
  (RoofLang's own DES is bypassed). Contention semantics differ (RoofLang
  equal split vs progressive filling); identical in isolation. Llama has an
  isolated-reference differential check; **Kimi rejects
  `--reference-every-round`** → no differential check on the target model.
- **Hard parts**: (1) speed — fine-grained Kimi is ~10² slower than needed
  for DSE; (2) Kimi layouts hard-coded; (3) P/D + agent workloads unsupported
  together; (4) dual RoofLang copies; (5) no calibration layer; (6) no
  mechanism/component manifest spanning both repos.
- **Reuse directly**: engine, resource domain, KvRuntime, systems,
  scheduler, policy runtime, bridge/worker protocol, Kimi shared lowering,
  RoofLang IR/kernels/passes/presets, Pareto sweep, dense feedback V0.1,
  launchers (Docker, freezing, baseline sealing, replay), skills, validators.
- **Extend**: Kimi layouts (TP/EP/DP/DCP/PP), B300 in Simthesizer Kimi path,
  H200 MXFP4 emulation path, calibration/efficiency parameters, Tier-1 cost
  query, agent workloads through P/D, unified feedback schema, Kimi
  differential reference.
- **New**: mechanism registry/specs/dependency graph, candidate isolation
  over two repos, cross-repo frozen evaluator, experiment ledger and
  QD/Pareto archive, research controller over existing launchers.

## 6. Answers to RESEARCH_SPEC §19

1. *Actual capabilities* — §1–2.
2. *Connection level*: operator-demand level (RoofLang prices isolated work
   per operator / transfer; Simthesizer executes it on shared resources). Keep
   it. Add a cached iteration-level Cost Query for search. See DECISIONS D2/D3.
3. *RoofLang as sole backend*: yes; already true on the shared path (no
   profiles, no LLMCompass). Condition: every cost the engine charges must
   originate from a RoofLang demand or a declared native transfer, never both
   (D4), and calibration must live in RoofLang parameters (D6).
4. *Kimi/H200/B300 gaps*: §3–4.
5–6. *Reuse / extend / redesign*: §5.
7. *Extension without breaking*: kind registries + component manifest +
   per-candidate git worktrees of both repos + frozen evaluator outside
   candidates + regression selection by reverse dependencies (D7–D9).
8. *Feedback and validation*: normalize existing outputs (operator CSV,
   rounds, KV lifecycle, dense feedback) into one cross-layer schema (D10);
   validator = existing checks + InferEvo invariants + differential + sensitivity.
9. *Unimplemented mechanisms*: expressed as new logical/resource nodes,
   policies, RoofLang kernels/passes; must satisfy math reference (CPU) and
   physical checks; labelled by evidence level.
10. *Credibility without hardware*: evidence ladder (math → physical →
    simulated → calibrated-to-public-data → hardware), sensitivity bands,
    robustness criterion for promotion (D11).
11. *MVS* and 12. *growth path*: ROADMAP.md.

## 7. Reference projects (status)

Not yet source-audited this session: BLIS, AISimulate, ServingStudio,
Frontier, LLMServingSim 2.0, InferSim, ServeTwin, InferaSim, KD-DSE,
RSIBench-Infra, CAFEML, AgentServeSim, Contexture. Reachability checked for
BLIS (`inference-sim/inference-sim` HEAD `e7597a8`). Audits are scheduled per
module (ROADMAP) and recorded in TECH_SOURCE_MAP.md before that module's
development, per spec §4.5. No capabilities are attributed to them yet.

## 8. Phase 1 resolution of audit findings (2026-11-25)

Resolved on branches `inferevo/phase1-base` (see cost_models/):
precision (MC-1), KDA state dtype FP32 (MC-2), H200 FP4/capacity and B300
bandwidth (CM-1..3), MLA prefill arithmetic from upstream #10 (CM-5), Kimi
layout gating (D15). Still open: Kimi differential check, efficiency/launch
modelling (CM-4), P/D + agent workloads, vendored-copy drift tooling,
reference-project audits.

## 9. Calibration data inventory (2026-11-25)

| Need | Best available source | Gap |
|---|---|---|
| H200 per-op efficiency (GEMM/attention/MoE) | Frontier H200 Qwen3-235B kernel tables (measured) | Kimi-specific shapes (MLA, KDA, latent MoE) absent; transfer by shape/arith. intensity is an assumption |
| H200 MLA / linear attention kernels | InferSim H800 MLA, GDN tables | different SKU (same compute die, lower HBM bw); GDN ≠ KDA exactly |
| Launch / host overhead | Frontier kernel_gap (H200: eager 1.2–1.5 µs, CUDA graph 0.43 µs) | per-framework; used for CM-4b band |
| B300 | datasheet (rates); AISimulate sm100 probe (sparse) | no measured kernel tables → B300 results stay uncalibrated |
| End-to-end serving | Frontier published_readings (22 cells, non-Kimi) | no public Kimi-K3 H200/B300 serving measurement found yet |
