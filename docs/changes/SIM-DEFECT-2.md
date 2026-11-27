# SIM-DEFECT-2 Non-shared P/D handoff omitted fixed per-request state (FIXED)

`pd_disaggregation` sized the non-shared KV transfer as input tokens × bytes
per token, ignoring `kv_bytes_per_request_per_device` (recurrent/conv state
of hybrid models: Kimi KDA 454 MB/request, Mamba-style models). Fix: add the
fixed state (whole transfer) or fixed/shards per head slice (topology); shard
divisibility now checks both. Shared execution was already correct (state
schema). Affects only non-shared P/D with fixed-state models (none of the
earlier baselines). Rust suite 254/254.
