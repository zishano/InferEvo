# CM-7 Calibration profiles (no default change)

- E0006 fits t = α + max(F/(η_c·peak), B/(η_m·BW)) to Frontier H200 kernel
  tables (Frontier db5a85b). Dense GEMM BF16: α 4.05 µs, η_c 0.788,
  η_m 0.784, median |rel err| 3.3 % (p90 8.2 %, n=774). MoE grouped GEMM
  (vLLM fused, TP-split experts, intermediate 192): η_c 0.196, η_m 0.588,
  median 17.8 % (p90 48 %, n=838) — single scalars do not fit; Kimi's whole
  experts (3072/3584) are wider, so this data does not transfer.
- Profiles in `validation/cost_profiles.json`: `peak` (default, upper
  bound), `h200_dense_v1`, `h200_pessimistic_v1`, `b300_transfer_v1`.
  Results must name the profile used; only `peak` vs profile bands are
  comparable across hardware.
