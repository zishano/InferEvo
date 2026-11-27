# Technology Source Map

Reuse type: A direct integration, B selective code reuse, C idea/algorithm,
D own development. Status: AUDITED (source read), PLANNED (audit scheduled,
nothing attributed yet), IMPLEMENTED (InferEvo code exists, with evidence).

| Module | Source | Repo / version | Relevant files / functions | Borrowed | Type | Adaptation | Status | Evidence |
|---|---|---|---|---|---|---|---|---|
| System DES, DAG, time | Simthesizer | zishano/simthesizer `test_dev` @b05b76f (+dirty) | `crates/simthesizer-core/src/engine.rs` | engine, logical/resource nodes | A | none | AUDITED | 252 Rust tests pass |
| Resource contention | Simthesizer | same | `resources.rs` (`Resources::submit/admit/advance`, progressive filling) | shared compute/HBM/link | A | calibration knobs later | AUDITED | tests |
| State runtime | Simthesizer | same | `kv.rs` `KvRuntime` | MLA/KDA versions, P/D copies | A | agent-session retention (Phase 2) | AUDITED | tests |
| Execution contract | Simthesizer bridge | same | `model_runners/rooflang.rs`, `scripts/rooflang_worker.py`, `rooflang_bridge/kimi_k3_shared.py` | protocol v1, work templates | A | layouts, B300, Tier-1 | AUDITED | 237 Py tests pass |
| Policy runtime | Simthesizer | same | `policies/mod.rs` (`PolicyRuntime::route`, batch, KV) | causal decisions | A | — | AUDITED | tests |
| Synthesizer agent workflow | Simthesizer | same | `.agents/skills/*`, `architectures/agent-workflow.md` | staged extension | C/A | used inside agent sessions | AUDITED | — |
| Policy search, freezing, replay | Simthesizer | same | `agents/find_pareto_frontier/{launcher,evaluate,verify,archive}.py` | frozen evaluator, sealed baselines | A/B | cross-repo identity | AUDITED | its tests |
| Two-loop design | Simthesizer | same | `agents/qwen_moe_optimization/DESIGN.md`, `control.py` | candidate state machine | C/B | controller (D12) | AUDITED | — |
| Operator cost, IR, placement | RoofLang | zishano/rooflang @8692f1d (+dirty); vendored subset 5cf122b | `language/{graph,placement,tensor}.py`, `kernels/*`, `runtime/simulator.py` (`_base_times`, `compute_time_us`) | sole operator backend | A | calibration (CM-4) | AUDITED | 1060/1061 tests |
| Kimi-K3 graph | RoofLang | same | `programs/models/kimi_k3/{config,model}.py` | structure | A | precision profiles (U1) | AUDITED | E0001 structure match |
| Static-batch DSE | RoofLang | same | `programs/experiments/find_pareto_frontier.py` | layout × batch sweep | A | Tier-1 seed data | AUDITED | — |
| Dense feedback V0.1 | RoofLang | same (untracked) | `analysis/dense_feedback.py` | operator/layer/device profile | B | cross-layer schema (D10) | AUDITED (doc) | — |
| Arch discovery agent | RoofLang | same | `agents/find_new_arch/AGENTS.md` | P/S tracks, review ledger, writable allowlist | C | mechanism promotion rules | AUDITED | — |
| Kimi-K3 official config | Moonshot AI | huggingface.co/moonshotai/Kimi-K3 config.json sha256 9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213 (fetched 2026-11-24) | `text_config`, `quantization_config` | ground truth | A | — | IMPLEMENTED | `validation/kimi_conformance.py` |
| H200 facts | NVIDIA | nvidia.com/en-us/data-center/h200 | datasheet | capacity, bw, peaks | — | dense = sparse/2 | IMPLEMENTED | `validation/hardware.py` |
| B300 facts | NVIDIA HGX page; SemiAnalysis; Tom's Hardware | see hardware.py | — | capacity, bw, FP4 | — | conflicts recorded | IMPLEMENTED (partial) | `audit_preset` |
| Capacity model | own, over bridge | — | `rooflang_bridge.kimi_k3._weight_layout`, `state_schema` | layout rules | A + D | grid generalization | IMPLEMENTED | E0001, exact 16×H200 match |
| Serving/latency interface | BLIS | github.com/inference-sim/inference-sim HEAD e7597a8 (reachable) | — | — | C | — | PLANNED (Phase 2) | — |
| Perf query + serving search | AISimulate | github.com/ai-dynamo/aisimulate | — | — | C | — | PLANNED (Phase 2) | — |
| Agent skills, bottleneck workflow | ServingStudio | github.com/SyFI-ServingStudio/ServingStudio | — | — | C | — | PLANNED (Phase 3) | — |
| Calibration, regression | Frontier | github.com/NetX-lab/Frontier | — | — | C | — | PLANNED (Phase 1) | — |
| Knowledge-driven DSE | KD-DSE | arXiv 2611.22411 | — | — | C | — | PLANNED (Phase 3/5) | — |
| Contention-aware overlap | CAFEML | arXiv 2611.26840 | — | — | C | — | PLANNED (Phase 3) | — |
| Independent verification | RSIBench-Infra | arXiv 2611.24395 | — | — | C | — | PLANNED (Phase 4) | — |
| No-target-hardware prediction | ServeTwin | arXiv 2610.02732 | — | — | C | — | PLANNED (Phase 1/6) | — |
| Low-cost calibration/search | ROCm InferaSim | arXiv 2611.04032 | — | — | C | — | PLANNED (Phase 1/5) | — |
| Agent-program workloads | AgentServeSim / Contexture | arXiv 2606.09613 / 2611.26980 | — | — | C | — | PLANNED (Phase 2) | — |
| Model catalog, public data | LLMServingSim 2.0, InferSim, vLLM, SGLang | see spec 4.4.6 | — | — | C | — | PLANNED (Phase 1) | — |
| KDA state dtype | flash-linear-attention (fla) | github.com/fla-org/flash-linear-attention @0c5e7d9 (`../github`) | `fla/ops/common/chunk_delta_h.py:720-723`, `fla/ops/kda/fused_recurrent.py:284-286`, `fla/modules/causal_conv1d/ops.py:843` | FP32 recurrent state, input-dtype conv state | — | MC-2 | IMPLEMENTED | Simthesizer/RoofLang branch tests |
| MLA prefill arithmetic | RoofLang upstream | github.com/yzygitzh/rooflang @4ff475d (`../github/rooflang`) #10 | `language/kernels/forward.py` `MlaAttn.attention_width` | prefill full MHA, decode absorbed | B | ported without class rename | IMPLEMENTED (CM-5) | 4 ported tests, E0003 ablation |
| Kimi-K3 checkpoint precision | Moonshot AI | HF rev c7945ea `model.safetensors.index.json` | tensor names layers 0–1 | MXFP4 only for experts | — | MC-1 | IMPLEMENTED | `validation/sources/` |
| Calibration: H200 kernels | Frontier | github.com/NetX-lab/Frontier @db5a85b (`references/frontier`) | `data/profiling/compute/h200/Qwen3-235B-A22B/{attention,linear_op,moe,kernel_gap}.csv`; `tests/comparison/e2e_gate/cases/published_readings.json` (22 e2e cells) | measured kernel latency, launch gaps (eager 1.24–1.53 µs, CUDA graph 0.43 µs) | C/data | per-op-class efficiency + CM-4b launch overhead | AUDITED (data inventory) | — |
| Calibration: Hopper MLA/GDN/GEMM | InferSim | github.com/alibaba/InferSim @d12ec47 (`references/infersim`) | `bench_data/{gemm,grouped_gemm,mla,gdn,mha}/{h800,h20,pro5000}` (latency_us, mfu) | measured MFU curves; H800 = H200 compute die (transfer is an assumption) | C/data | compute-efficiency curves | AUDITED (data inventory) | — |
| Hardware descriptors with provenance | inference-sim (BLIS) | github.com/inference-sim/inference-sim @e7597a8 (`references/inference-sim`) | `engines/sglang/lower/testdata/{h200,b300}.yaml` | datasheet + driver-measured capacity | C | CM-6 | IMPLEMENTED | CM-6.md |
| B300 primary spec | NVIDIA | Blackwell Ultra datasheet 4169750 (PDF sha256 ccdffc06…) | HGX B300 column | 7.7 TB/s, FP4 14 PF dense, 270 GB | — | CM-6 | IMPLEMENTED | validation/hardware.py |
| Blackwell op probes | AISimulate | github.com/ai-dynamo/aisimulate @c348ae5 (`references/aisimulate`) | `benchmarks/evidence/collector/op-probe-2026-10/results/sm100/trtllm-1.3.0rc29.yaml` (incl. Kimi-K2.5) | sparse sm100 measurements | C/data | B300 sanity checks only | AUDITED (inventory) | — |
| Serving sim + model catalog | LLMServingSim | github.com/casys-kaist/LLMServingSim @4c7f0ad (`references/llmservingsim`) | `profiler/perf/{RTX4090,RTXPRO6000}` | no H200/B300 data | C | — | AUDITED (inventory) | — |
