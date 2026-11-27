# MC-3: expert tensor parallelism (ETP) for Kimi-K3

Kind: model contract (parallel layout extension). Status: APPLIED.
Branch: simthesizer `inferevo/phase1-base` a468689.

**Motivation.** The executable path only had whole-expert EP (expert TP fixed
1); ETP is a standard MoE layout (Megatron EP x ETP) the user asked to model.

**Representation.** Instance GPUs = attention TP x DP = EP x ETP. EP = expert
groups (896 / EP experts each); every expert spans ETP consecutive GPUs
(gate/up column shards, down row shards of MOE_INTER). Engine/runner `ep`
remains the MoE rank count (EP x ETP); InferEvo `Deployment` stores EP and
ETP and passes `--{side}-ep EPxETP --{side}-expert-tp ETP`.

**Execution (assumption).** Leader sends each expert's tokens once to the
expert's head rank; head forwards to the other ETP-1 ranks; each rank runs
its shards; down partials go to the head, ETP-1 pairwise BF16 adds, one
result returns to the leader. An all-gather/reduce-scatter variant would
spread the head's NVLink load (not modelled).

**Scope.** `kimi_k3_shared.py` (validate_parallel, expert shards, reduce,
`_lower_expert_tp`), Rust runner validation, CLI allocation check, launcher
`--{side}-expert-tp 1/2/4/8`, `configuration.py` pass-through. ETP1 lowering
is untouched (separate branch), so no baseline changes: E0031 ETP1 single
8K request TTFT 2366 ms equals the earlier single 8K-request runs (2.37 s).

**Memory.** Per-GPU expert bytes = 896/EP x 1/ETP = 896/(GPUs) experts:
unchanged at equal GPU count (test: TP4, 16 GPUs, ETP2 = 118.79 GB).

**Validation.** Bridge tests: FLOPs and weight bytes conserved vs ETP1,
shards on head..head+ETP-1, dispatch/combine bytes, scatter/gather bytes,
16 reduces per layer, program-mode export; Rust validation test; full
cargo suite and Kimi bridge suites pass.

**Effect (E0031, 1 request 8K/64, TP4, peak).** 16 GPUs: ETP2/4/8 TTFT
+0.4/+5.0/+47 %, TPOT −1.2/−2.9/−3.1 %. 32 GPUs ETP2: TTFT +0.4 %, TPOT −1.0 %.
ETP does not reduce the leader-NIC dispatch bytes (C-DISPATCH unchanged).

**Not covered.** Tier-1 surrogates (fitted at ETP1): ETP runs need `--tier 2`
and are excluded from `optimize.sh` search.
