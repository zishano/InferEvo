# CM-4 Explicit achievable-rate (efficiency) factors

- Change: RoofLang `HardwareGraph.derate(compute, memory, network)` scales
  Compute peaks, Memory-touching fabrics, other fabrics and preset aggregate
  collective bandwidth (rooflang a337484). Simthesizer bridge accepts an
  optional `hardware.efficiency`; the P/D composer derates the shared
  topology once so native KV links and workers agree; launcher flags
  `--{compute,memory,network}-efficiency` (default 1.0 = peak, results
  unchanged).
- Not yet modelled: kernel launch / host overhead per operator, latency (α)
  scaling, per-kernel-class efficiency curves. Planned as CM-4b.
- Validation: RoofLang derate tests (scaling, identity, single application);
  Simthesizer tests (exact demand scaling, composer KV-link derating);
  smoke: network 0.5 doubles the P→D transfer span (2.279 → 4.554 ms).
- Purpose: sensitivity bands (D11 `ROBUST`) and counterfactual bottleneck
  probes (E0005), not calibration; calibrated values require public data.
