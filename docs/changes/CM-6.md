# CM-6 Correct CM-2/CM-3 against primary and measured sources

Supersedes CM-2 (H200 141 GB) and CM-3 (B300 8 TB/s). Both were derived from
secondary summaries; the audit of `references/inference-sim` (e7597a8) and
the primary NVIDIA Blackwell Ultra datasheet showed them wrong.

| Field | Before | After | Source / status |
|---|---|---|---|
| H200 HBM capacity | 141 GB (CM-2) | 150.11 GB | CUDA driver total 150,109,880,320 B (vLLM allocates from it); datasheet label "141GB" is GiB-like. public_measurement, cited by inference-sim |
| B300 HBM bandwidth | 8000 GB/s (CM-3) | 7700 GB/s | Datasheet 4169750 HGX B300: "270 GB HBM3E \| 7.7 TB/s"; 8 TB/s is GB300. official_spec (PDF sha256 ccdffc06…) |
| B300 FP4 dense | 13500 | 14000 TFLOP/s | Datasheet: FP4 18 \| 14 PFLOPS (sparse \| dense). official_spec |
| B300 HBM capacity | 288 GB | 288 GB (kept) | Driver total 287.43 GB; within 0.2 % — not changed |

Lesson recorded: hardware numbers need a primary or measured source before
entering the cost model; secondary web summaries mix HGX and GB300 SKUs and
GB vs GiB. `validation/hardware.py` now cites these sources.

Datasheet inconsistency: per-GPU FP4 dense is 14 PFLOPS, but the HGX B300
system total is 108 PFLOPS dense (= 8 × 13.5). Per-GPU `B300Cluster` uses
14000 (per-GPU table, also inference-sim); aggregated `B300SuperChip` keeps
the stated system total 108000. Treat ±3.6 % FP4 as a sensitivity band.

Consequence: 16×H200 TP2/EP16 official-precision weights (147.16 GB) now fit
the 150.11 GB driver total with < 3 GB left; with any utilization headroom
(e.g. 0.9) TP2 remains infeasible, so the default stays TP4.
Commits: rooflang dbe1486; simthesizer (see README table).
