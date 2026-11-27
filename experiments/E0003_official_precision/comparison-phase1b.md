| hw | stage | batch | tokens | ctx | base ms | phase1b ms | ratio |
|---|---|---|---|---|---|---|---|
| h200 | decode | 64 | 1 | 8192 | 334.0 | 360.3 | 1.079 |
| h200 | decode | 512 | 1 | 131072 | 1027.9 | 1725.5 | 1.679 |
| h200 | prefill | 1 | 8192 | None | 1119.0 | 1890.9 | 1.690 |
| h200 | prefill | 1 | 32768 | None | 5492.6 | 8160.3 | 1.486 |
| h200 | prefill | 4 | 8192 | None | 4472.7 | 7560.5 | 1.690 |
| b300 | decode | 64 | 1 | 8192 | 206.8 | 216.2 | 1.045 |
| b300 | decode | 512 | 1 | 131072 | 630.9 | 1026.2 | 1.626 |
| b300 | prefill | 1 | 8192 | None | 568.1 | 720.8 | 1.269 |
| b300 | prefill | 1 | 32768 | None | 2159.2 | 2602.5 | 1.205 |
| b300 | prefill | 4 | 8192 | None | 1710.7 | 2338.8 | 1.367 |

| hw | GPUs | aTP | base weights GB | phase1b weights GB | phase1b status | phase1b req@32k |
|---|---|---|---|---|---|---|
| h200 | 8 | 1 | 245.3 | 294.3 | infeasible_weights_oom | - |
| h200 | 8 | 2 | 213.1 | 237.6 | infeasible_weights_oom | - |
| h200 | 8 | 4 | 196.9 | 209.2 | infeasible_weights_oom | - |
| h200 | 8 | 8 | 188.9 | 195.0 | infeasible_weights_oom | - |
| h200 | 16 | 1 | 154.9 | 203.9 | infeasible_weights_oom | - |
| h200 | 16 | 2 | 122.7 | 147.2 | infeasible_weights_oom | - |
| h200 | 16 | 4 | 106.5 | 118.8 | feasible | 28 |
| h200 | 16 | 8 | 98.5 | 104.6 | feasible | 46 |
| h200 | 32 | 1 | 109.7 | 158.7 | infeasible_weights_oom | - |
| h200 | 32 | 2 | 77.4 | 102.0 | feasible | 352 |
| h200 | 32 | 4 | 61.3 | 73.6 | feasible | 416 |
| h200 | 32 | 8 | 53.3 | 59.4 | feasible | 280 |
| h200 | 64 | 1 | 87.1 | 136.1 | infeasible_weights_oom | - |
| h200 | 64 | 2 | 54.8 | 79.4 | feasible | 1312 |
| h200 | 64 | 4 | 38.7 | 51.0 | feasible | 1184 |
| h200 | 64 | 8 | 30.7 | 36.8 | feasible | 744 |
| h200 | 128 | 1 | 75.8 | 124.8 | feasible | 128 |
| h200 | 128 | 2 | 43.5 | 68.1 | feasible | 3264 |
| h200 | 128 | 4 | 27.4 | 39.7 | feasible | 2720 |
| h200 | 128 | 8 | 19.4 | 25.5 | feasible | 1680 |
| b300 | 8 | 1 | 245.3 | 294.3 | infeasible_weights_oom | - |
| b300 | 8 | 2 | 213.1 | 237.6 | feasible | 76 |
| b300 | 8 | 4 | 196.9 | 209.2 | feasible | 98 |
| b300 | 8 | 8 | 188.9 | 195.0 | feasible | 66 |
| b300 | 16 | 1 | 154.9 | 203.9 | feasible | 640 |
| b300 | 16 | 2 | 122.7 | 147.2 | feasible | 784 |
| b300 | 16 | 4 | 106.5 | 118.8 | feasible | 548 |
| b300 | 16 | 8 | 98.5 | 104.6 | feasible | 320 |
| b300 | 32 | 1 | 109.7 | 158.7 | feasible | 2336 |
| b300 | 32 | 2 | 77.4 | 102.0 | feasible | 2208 |
| b300 | 32 | 4 | 61.3 | 73.6 | feasible | 1456 |
| b300 | 32 | 8 | 53.3 | 59.4 | feasible | 828 |
| b300 | 64 | 1 | 87.1 | 136.1 | feasible | 5760 |
| b300 | 64 | 2 | 54.8 | 79.4 | feasible | 5056 |
| b300 | 64 | 4 | 38.7 | 51.0 | feasible | 3264 |
| b300 | 64 | 8 | 30.7 | 36.8 | feasible | 1848 |
| b300 | 128 | 1 | 75.8 | 124.8 | feasible | 12544 |
| b300 | 128 | 2 | 43.5 | 68.1 | feasible | 10752 |
| b300 | 128 | 4 | 27.4 | 39.7 | feasible | 6880 |
| b300 | 128 | 8 | 19.4 | 25.5 | feasible | 3872 |
