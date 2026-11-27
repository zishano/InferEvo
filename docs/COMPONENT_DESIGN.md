# Component Design

Implemented InferEvo components only; planned components are in
ARCHITECTURE.md / ROADMAP.md.

## engines (`adapters/engines.py`)
- Responsibility: locate `engines/simthesizer`, `engines/rooflang` (env overrides
  `INFEREVO_SIMTHESIZER`, `INFEREVO_ROOFLANG`); pin identity; enable imports of
  the vendored RoofLang + bridge actually executed by the simulator.
- Output: `{repo: {commit, branch, dirty, dirty_digest}}`.
- Failure: missing repo or git → exception (results must not be unpinned).
- Validation: `tests/test_phase0.py::SourceIdentity`.

## hardware facts (`validation/hardware.py`)
- Responsibility: sourced per-GPU facts with status; native MMA dtype sets;
  `audit_preset(h)` diff vs RoofLang preset.
- Contract: facts are never fed into simulations directly; deviations become
  cost-model change records (D5/D6).
- Validation: H200 FP4 finding, B300 FP4 native.

## Kimi conformance (`validation/kimi_conformance.py`)
- Responsibility: official HF config vs RoofLang constants (structure), and
  official quantization scope vs bridge weight dtypes (precision).
- Validation: structure has zero findings; precision deviation quantified.

## Capacity (`validation/capacity.py`)
- Responsibility: resident weights, KDA fixed state, MLA growth per GPU and
  max concurrent requests for a layout grid; status strings name the failing
  constraint.
- Dependencies: `rooflang_bridge.kimi_k3._weight_layout`, `state_schema`
  (executed code), `hardware.SPECS` capacities.
- Limits: no workspace/buffer reserve by default (upper bound); PP/DCP not
  modelled; layout = EP over all N GPUs.
- Validation: exact match of bridge-documented 122,651,032,104 B/GPU.

## cost model (`adapters/cost_model.py`)
- Responsibility: `cost_model_version()` (hash of vendored RoofLang + bridge +
  worker), `vendored_drift()`, `preset_gpu()`, serial-roofline ratios.
- Limit: roofline ratios are for comparisons only, never latency predictions.

## isolation (`mechanisms/isolation.py`)
- baseline: user checkout → `inferevo/<name>` worktree (HEAD + diff [+ untracked]).
- candidate: engine HEAD → `inferevo/cand/<name>` in `mechanisms/candidates/`.
- Records in `mechanisms/versions/`; refuses dirty engine parents.

## round bounds (`validation/round_bounds.py`)
- Responsibility: engine-independent differential for one shared-path round:
  operator pricing identity (FLOPs/peak, demand vectors) and
  max(critical path, resource load, stream serial) ≤ T_engine ≤ Σ isolated.
- Input: worker-exported operations (`KimiTemplate.export`); never imports Rust.
- Limit: brackets time; exposed waits outside the exported graph (P→D state
  copies) show up as engine − lower bound, not as errors.

## calibration (`validation/calibration.py`, `validation/cost_profiles.json`)
- Fits t = α + max(F/(η_c·peak), B/(η_m·BW)) per kernel class to Frontier H200
  tables; profiles are named efficiency sets the engine accepts
  (`hardware.efficiency`). Default `peak` = roofline upper bound.

## math references (`validation/math_reference.py`)
- numpy float64 references at small dimensions: MLA prefill (materialized
  MHA) ≡ absorbed latent; KDA chunked/decode with carried recurrent + conv
  state ≡ single pass; negative controls must fail.
