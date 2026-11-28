# Cost-model and model-contract change records (DECISIONS D6)

One file per change. Each record: motivation + evidence, scope (files, repos,
branch commits), version ids before/after, re-evaluated baselines, effect
(ablation), validation. Ordinary mechanism candidates may not edit the files
listed here; a change here re-evaluates every compared baseline.

| Id | Kind | Status | Branch commits |
|---|---|---|---|
| MC-1 | model contract: official precision | APPLIED (phase1) | rooflang 9c35746; simthesizer b940a7b |
| CM-5 | MLA prefill full-MHA width (upstream #10) | APPLIED (phase1) | rooflang 1f1d354; simthesizer b940a7b |
| CM-1..3 | H200 FP4 at BF16 rate (CM-1 kept); CM-2/CM-3 superseded by CM-6 (CM-1-3.md) | APPLIED (phase1) | rooflang 838541f; simthesizer 7fc31a3 |
| MC-2 | KDA recurrent state FP32 (fla) | APPLIED (phase1) | rooflang 277bc2f; simthesizer dc426a9 |
| CM-4 | efficiency factors (compute/memory/network), default peak (CM-4.md) | APPLIED (phase1) | rooflang a337484; simthesizer 3f16e77 |
| CM-4b | per-kernel fixed overhead, default 0 (CM-4b.md) | APPLIED (phase1) | simthesizer d44343c |
| CM-6 | H200 capacity 150.11 GB (driver total), B300 7.7 TB/s + FP4 14 PF (datasheet) (CM-6.md) | APPLIED (phase1) | rooflang dbe1486; simthesizer 95ef7ac |
| CM-7 | calibration profiles from Frontier H200 data (CM-7.md); default stays peak | APPLIED (InferEvo data) | validation/cost_profiles.json |
| MC-3 | model contract: expert TP layout for Kimi-K3 (MC-3.md) | APPLIED (InferEvo) | simthesizer a468689 |
| MC-4 | model contract: pipeline parallelism for Kimi-K3 (MC-4.md) | APPLIED (InferEvo) | simthesizer 53d777f |
| MC-5 | model contract: grouped Tier-2 lowering is the reference (MC-5.md) | APPLIED (InferEvo) | simthesizer 996bee1, 8c62d1f |
| SIM-DEFECT-3 | Tier-1 preempted P/D decode emits zero-gap tokens; trigger fixed (DP-aware scalar state), guard in steady (SIM-DEFECT-3.md) | MITIGATED | simthesizer kimi_k3_scalar |
| M-MOECHUNK | mechanism, default off (lowering option moe_chunks; spec mechanisms/specs/M-MOECHUNK.toml, receipts mechanisms/receipts/M-MOECHUNK/) | PROMOTED | simthesizer 60dc2d0; default-off identity byte-identical |
| M-RANKDISPATCH | mechanism, default off (lowering option dispatch; spec mechanisms/specs/M-RANKDISPATCH.toml) | PROMOTED | simthesizer 8d70c15; default-off identity byte-identical |
| M-FP8LINEAR | mechanism, default off, semantics-changing (lowering option linear; quality UNVERIFIED, opt-in) | PROMOTED | simthesizer 88f77c2; default-off identity byte-identical |
| M-KVRETAIN | mechanism, default off (workload token_stream, scheduler state snapshots, P/D stream routing; spec mechanisms/specs/M-KVRETAIN.toml) | PROMOTED | simthesizer 9a9b3b7 + a2f0da6; default-off identity byte-identical |
