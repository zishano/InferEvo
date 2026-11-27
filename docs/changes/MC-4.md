# MC-4: pipeline parallelism (PP) for Kimi-K3

Kind: model contract (parallel layout extension). Status: APPLIED.
Branch: simthesizer `inferevo/phase1-base` 53d777f (layout), f462f7c (batch
overlap), 29a6e84 (Tier-1 PP).

**Representation.** Instance GPUs = PP x (attention TP x DP) = PP x (EP x
ETP). Stage s owns ranks [s*E, (s+1)*E), E = EP x ETP, and contiguous layers
`layer_stages(pp)` (93 layers, earlier stages one larger); embedding on the
first stage, output head on the last. Runner `pp` + per-stage `ep`;
`attention_groups` are stage-0 offsets. KV plans list owners stage-major
(PP x TP) with per-side layer->stage maps (`prefill/decode_layer_stages` in
`kv_topology`); `Plan::owners(layer)` is the single lookup.

**Execution (assumptions).** Every logical edge crossing a stage boundary
sends the producer's per-rank output to the same TP ranks of the next stage,
once per request (hidden state and AttnRes residual bank alike). Requests in
a round pipeline through stages (per-request graphs). Up to PP disjoint
batches are in flight per instance (vLLM-style virtual engines; a request
waits for its batch to commit; commits FIFO). Each GPU stream runs
overlapping rounds in issue order (first op of a round waits for the
previous round's last op on that stream).

**Memory.** Each stage holds its layers' weights and state: per-stage
reservations in `kv.rs`; scheduler admission uses the worst stage
(`scheduler_kv_*`); `kv_bytes_*_per_device` keep the all-layer contract.
TP4 16 GPUs/stage PP2: 59.8 / 59.0 GB weights (PP1 118.8 GB). EP8 TP4 becomes
feasible with PP2 (PP1: 209 GB > 135 GB usable).

**Validation.** Bridge: stage split, weights/state conserved across stages,
every layer op on its stage's GPUs, boundary sends only forward, FLOPs equal
to PP1, stage-major KV groups; Rust: plan owners/rank and per-stage
reservations; launcher smoke runs PP2/PP2, PP2/PP1, PP3/PP2+ETP2 end to end;
all cargo, Kimi and P/D suites pass. PP1 lowering unchanged (E0032 PP1 rows
equal E0031 ETP1 rows).

**Effect (E0032, 1 request 8K/64, TP4).** At 16 GPUs: PP2xEP8 TTFT −56 %
(2366 → 1037 ms), PP4xEP4 −54 %; at 32 GPUs PP2xEP16 −26 %, PP4xEP8 −67 %.
Cause: each stage's MoE all-to-all stays inside one NVLink node instead of
crossing the leader NIC (C-DISPATCH); the boundary send itself costs ~2 ms
(PP2xEP16 vs PP1xEP16 equal). TPOT within −2..+6 %.

**Overlap (E0033, burst 8 x 2048 tokens, budget 2048).** PP1xEP16 one
round per 588 ms; PP2xEP16 first 590 ms then every 289 ms (2x throughput,
equal per GPU); PP2xEP8 (16 GPUs) first 257 ms then every 127 ms: prefill
1010 vs 218 tok/s/GPU (4.6x). PP1 Tier-1/Tier-2 regression runs unchanged.

**Tier-1.** `step_model` splits a fitted whole-round time by stage layer
share and overlaps PP rounds; surrogates per layout (E0034: PP2xEP8 TP4).
