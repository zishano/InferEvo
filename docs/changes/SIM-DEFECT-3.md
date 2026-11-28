# SIM-DEFECT-3: preempted P/D decode requests emit zero-gap tokens (Tier-1)

Status: FIXED 2026-11-27 (root cause: Tier-1 surrogate extrapolation; NNLS refit, D27). Found 2026-11-27 while checking an
implausible optimum (P6 PP4xEP4 + D2 EP16 batch 64: 156 tok/s/GPU, 2.6x the
decode bound 2 x 64 / 23 ms).

**Symptom.** In Tier-1 (legacy chunked-prefill scheduler, non-shared P/D),
when a decode instance preempts requests for memory, those requests later
emit most of their tokens with zero inter-token gap (e.g. 990 of 1023);
throughput and TPOT become fictitious. Reproduced: P3 PP4xEP4 + D1 EP16,
batch 64, 160 x (1024|8192)/256 burst: 24 % / 40 % zero gaps; none at batch
32; none with 10x decode device memory.

**Trigger fixed.** The scalar Kimi description charged every request's state
on every device, though with Attention DP a device holds only its group's
requests (simthesizer kimi_k3_scalar: state / groups). Zero gaps: 0 after
the fix in the same runs.

**Guard.** adapters/steady.py marks a run `queue = "invalid"` when > 1 % of
gaps are zero; it never counts as a capacity point.

**Open.** The preempt -> recompute token accounting itself (legacy
scheduler + P/D decode requests) still needs a root-cause fix; genuine
memory overcommit would trigger it again (caught by the guard).

**Affected results.** Only the 2026-11-27 optimize run with decode batch 64
(discarded). Batch <= 32 runs and E0040/E0041 calibrations (no preemption:
memory headroom) are unaffected.

**Root cause (2026-11-27).** Not the scheduler: commits emit at most one
token per request per iteration and recompute is correct. The Tier-1
decode surrogates were unconstrained least-squares fits on q = 1 rows; the
KV features are collinear there (attention pairs = kv + 1), so the fits
cancelled them with opposite signs (e.g. +0.0147 / −0.0147 ms) and gave the
token count a negative weight. A preempted request returns to decode as a
recompute row (q > 1): the negative terms dominate, the round time is
clamped to ~0, and every token of those rounds has a zero gap. Found when
H200 decode batch 128 (KV pressure -> preemption) turned capacity probes
`invalid` (E0057). 33/35 prefill and 35/35 decode surrogates had a negative
coefficient.

**Fix.** `adapters.tier1.nnls` (Lawson–Hanson, column-scaled): every
surrogate coefficient >= 0, so round time never falls as work grows. All
surrogates refitted from their cached Tier-2 samples (old fits kept as
`result-lstsq.json`); held-out checks re-run (D23). Prototype (E0056 H200
b128): held-out TPOT −0.8 % before and after; a recompute row q=512,
kv=8000 now adds +28.6 ms instead of clamping the round to 0.
