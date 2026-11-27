| hw | stage | batch | tokens | ctx | base ms | phase1 ms | ratio |
|---|---|---|---|---|---|---|---|
| h200 | decode | 64 | 1 | 8192 | 334.0 | 348.7 | 1.044 |
| h200 | decode | 512 | 1 | 131072 | 1027.9 | 1632.9 | 1.589 |
| h200 | prefill | 1 | 8192 | None | 1119.0 | 1488.4 | 1.330 |
| h200 | prefill | 1 | 32768 | None | 5492.6 | 6550.1 | 1.193 |
| h200 | prefill | 4 | 8192 | None | 4472.7 | 5950.3 | 1.330 |
| b300 | decode | 64 | 1 | 8192 | 206.8 | 216.0 | 1.044 |
| b300 | decode | 512 | 1 | 131072 | 630.9 | 1001.1 | 1.587 |
| b300 | prefill | 1 | 8192 | None | 568.1 | 730.6 | 1.286 |
| b300 | prefill | 1 | 32768 | None | 2159.2 | 2624.1 | 1.215 |
| b300 | prefill | 4 | 8192 | None | 1710.7 | 2360.4 | 1.380 |

| hw | GPUs | aTP | base weights GB | phase1 weights GB | phase1 status | phase1 req@32k |
|---|---|---|---|---|---|---|
| h200 | 8 | 1 | 245.3 | 294.3 | infeasible_weights_oom | - |
| h200 | 8 | 2 | 213.1 | 237.6 | infeasible_weights_oom | - |
| h200 | 8 | 4 | 196.9 | 209.2 | infeasible_weights_oom | - |
| h200 | 8 | 8 | 188.9 | 195.0 | infeasible_weights_oom | - |
| h200 | 16 | 1 | 154.9 | 203.9 | infeasible_weights_oom | - |
| h200 | 16 | 2 | 122.7 | 147.2 | infeasible_weights_oom | - |
| h200 | 16 | 4 | 106.5 | 118.8 | feasible | 32 |
| h200 | 16 | 8 | 98.5 | 104.6 | feasible | 46 |
| h200 | 32 | 1 | 109.7 | 158.7 | infeasible_weights_oom | - |
| h200 | 32 | 2 | 77.4 | 102.0 | feasible | 384 |
| h200 | 32 | 4 | 61.3 | 73.6 | feasible | 440 |
| h200 | 32 | 8 | 53.3 | 59.4 | feasible | 288 |
| h200 | 64 | 1 | 87.1 | 136.1 | infeasible_weights_oom | - |
| h200 | 64 | 2 | 54.8 | 79.4 | feasible | 1472 |
| h200 | 64 | 4 | 38.7 | 51.0 | feasible | 1248 |
| h200 | 64 | 8 | 30.7 | 36.8 | feasible | 768 |
| h200 | 128 | 1 | 75.8 | 124.8 | feasible | 128 |
| h200 | 128 | 2 | 43.5 | 68.1 | feasible | 3648 |
| h200 | 128 | 4 | 27.4 | 39.7 | feasible | 2880 |
| h200 | 128 | 8 | 19.4 | 25.5 | feasible | 1728 |
| b300 | 8 | 1 | 245.3 | 294.3 | infeasible_weights_oom | - |
| b300 | 8 | 2 | 213.1 | 237.6 | feasible | 84 |
| b300 | 8 | 4 | 196.9 | 209.2 | feasible | 102 |
| b300 | 8 | 8 | 188.9 | 195.0 | feasible | 68 |
| b300 | 16 | 1 | 154.9 | 203.9 | feasible | 768 |
| b300 | 16 | 2 | 122.7 | 147.2 | feasible | 872 |
| b300 | 16 | 4 | 106.5 | 118.8 | feasible | 580 |
| b300 | 16 | 8 | 98.5 | 104.6 | feasible | 330 |
| b300 | 32 | 1 | 109.7 | 158.7 | feasible | 2784 |
| b300 | 32 | 2 | 77.4 | 102.0 | feasible | 2448 |
| b300 | 32 | 4 | 61.3 | 73.6 | feasible | 1536 |
| b300 | 32 | 8 | 53.3 | 59.4 | feasible | 852 |
| b300 | 64 | 1 | 87.1 | 136.1 | feasible | 6848 |
| b300 | 64 | 2 | 54.8 | 79.4 | feasible | 5600 |
| b300 | 64 | 4 | 38.7 | 51.0 | feasible | 3440 |
| b300 | 64 | 8 | 30.7 | 36.8 | feasible | 1896 |
| b300 | 128 | 1 | 75.8 | 124.8 | feasible | 14976 |
| b300 | 128 | 2 | 43.5 | 68.1 | feasible | 11904 |
| b300 | 128 | 4 | 27.4 | 39.7 | feasible | 7264 |
| b300 | 128 | 8 | 19.4 | 25.5 | feasible | 3984 |
