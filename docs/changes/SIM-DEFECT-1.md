# SIM-DEFECT-1 Kimi decode does not batch requests in GEMMs (FIXED, simthesizer c623652)

Evidence (E0008, H200 TP4/EP16, frozen workload, 792 steady decode rounds
reconstructed from result timings): round time ≈ 7.1 ms × (max requests
sharing one Attention group) + small terms — 8.3–9.0 ms (1/group),
15.2–16.3 ms (2/group), 22.3 ms (3/group). Cause: the Kimi lowering is
documented "unfused_request_graphs": each request issues its own GEMMs and
re-reads nonexpert weights; requests in one Attention group serialize.

Impact: batching gives no weight reuse, so tokens/s/GPU cannot improve with
batch size beyond the number of Attention groups. Every throughput result
(E0005, E0007 tok/s/GPU) is pessimistic and must be re-evaluated after the
fix. Latency results at batch ≤ #groups are unaffected.

Classification: simulator fidelity defect (not an optimization mechanism):
real continuous batching (vLLM/SGLang) runs one GEMM over all batch tokens.
Fix plan: within each Attention group, token-wise operators (projections,
norms, gates, MoE dispatch inputs, output head) run once on the concatenated
tokens of the group's requests; per-request attention/state ops stay
separate (they read each request's own KV/state). Validate: batch=1 graphs
unchanged; weight bytes read per round independent of batch for token-wise
ops; FLOPs conserved; E0004/E0008 re-run.

## Fix (simthesizer c623652)

Cost-equivalent weight sharing instead of graph restructuring: per round, the
first request of each Attention group pays nonexpert weight reads (kernels of
later requests get weight_read_fraction 0), and the first invocation of each
(layer, direction, expert) pays expert weights. Per-request dependencies and
state bindings unchanged. Checks: FLOPs exactly conserved; HBM saving equals
one nonexpert weight read per group rank; single-request graphs unchanged
(E0004 identical); suites Rust 252/252, Python 246 OK.

Effect: decode bound for 2 requests in one group ~15 → 8.74 ms; 16 requests
13.7 ms. Re-baseline E0007 (frozen workload): TPOT H200 TP4 13.52 → 9.65 ms,
B300 TP4 8.69 → 6.29 ms, B300 TP2 10.88 → 10.03 ms; TTFT unchanged.
Residual conservatism: later requests' kernels still serialize on the stream
(each pays its own activation reads), slightly above one fused GEMM.
