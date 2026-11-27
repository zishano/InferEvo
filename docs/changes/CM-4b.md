# CM-4b Per-kernel fixed overhead (default 0)

- Change: `hardware.efficiency.kernel_overhead_us` sets demand latency on
  every priced compute/HBM component of a Kimi operator, so operator time is
  overhead + max(compute, memory) (matches the E0006 fit form). Launcher flag
  `--kernel-overhead-us`. Scope: Kimi shared path (Qwen/Llama not yet).
- Finding: 4 µs per operator raises the H200 TP4 prefill critical path
  10.29 → 26.50 ms (~4,000 operators on the path). The Kimi lowering is
  unfused per request, much finer than real fused/CUDA-graph kernels, so a
  per-real-kernel overhead cannot be applied per graph operator. Default 0;
  use 0.43 µs (Frontier H200 CUDA-graph gap) as a band. Operator-fusion
  modelling (RoofLang `fuse_kernels`) is required before calibrating it.
