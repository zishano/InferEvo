"""Manual interface: one simulation with every parallel parameter explicit.

    ./simulate.sh --p 7 --d 1 --tp 4 --ep 16 --p-budget 2048 --isl 8192 --osl 1024 --rate 2.1
    ./simulate.sh --p 2 --d 2 --p-tp 8 --p-ep 16 --d-tp 8 --d-dp 4 --d-max-batch 32 --rate 0.5
    ./simulate.sh ... --capacity              # bisect the max SLO rate (Tier-1)
    ./simulate.sh ... --tier 2 --requests 1   # full simulation (slow)
    ./simulate.sh --p 1 --d 1 --d-ep 8 --d-etp 2 --tier 2 --requests 1   # decode experts split over 2 GPUs
    ./simulate.sh --p 1 --d 1 --ep 8 --pp 2 --tier 2 --requests 1        # 2 stages x 8 GPUs (EP8 in one node)

Per instance: GPUs = PP x (attention TP x DP) = PP x (MoE EP x ETP). PP
stages hold contiguous layers; EP = expert groups per stage (896 / EP experts
each), ETP = GPUs each expert is split over. Per side give TP plus DP or EP
(ETP, PP default 1); default 16 GPUs per stage, TP4. --tp/--ep/--etp/--pp set
both sides. The configuration is checked before running (launcher/engine
rules and resident memory); Tier-1 additionally needs a fitted surrogate
(H200 TP4/8, B300 TP4, EP16, ETP1, PP1, max batch <= 32), otherwise --tier 2.
Temporary files are removed; nothing is written to the project.
"""

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

from adapters import steady, tier1_engine as te, workloads
from adapters.engines import SIMTHESIZER
from adapters.topologies import Deployment, check, tier2_config
from agent.controller import models_for

a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
a.add_argument("--hardware", default="h200", choices=["h200", "b300"])
a.add_argument("--p", type=int, default=1, help="prefill instances")
a.add_argument("--d", type=int, default=1, help="decode instances")
a.add_argument("--tp", type=int, help="attention TP, both sides (default 4)")
a.add_argument("--ep", type=int, help="MoE EP (expert groups), both sides")
a.add_argument("--etp", type=int, help="expert TP, both sides (default 1)")
a.add_argument("--pp", type=int, help="pipeline stages, both sides (default 1)")
for s, side in (("p", "prefill"), ("d", "decode")):
    a.add_argument(f"--{s}-tp", type=int, help=f"{side} attention TP")
    a.add_argument(f"--{s}-dp", type=int, help=f"{side} attention DP (groups per instance)")
    a.add_argument(f"--{s}-ep", type=int, help=f"{side} MoE EP (expert groups)")
    a.add_argument(f"--{s}-etp", type=int, help=f"{side} expert TP (GPUs per expert)")
    a.add_argument(f"--{s}-pp", type=int, help=f"{side} pipeline stages")
    a.add_argument(f"--{s}-max-batch", type=int, default=32, help=f"{side} max requests per round")
    a.add_argument(f"--{s}-moe-chunks", type=int, default=1,
                   help=f"{side} expert-chunk pipelined MoE (M-MOECHUNK): 1 (off), 2, 4, 8")
a.add_argument("--p-budget", "--budget", type=int, default=2048, help="prefill tokens per round")
a.add_argument("--d-budget", type=int, help="decode tokens per round (default = decode max batch)")
a.add_argument("--isl", type=int, default=8192)
a.add_argument("--osl", type=int, default=1024)
a.add_argument("--requests", type=int, default=600)
a.add_argument("--rate", type=float, default=1.0, help="req/s (Poisson)")
a.add_argument("--ttft-slo", type=float, default=5.0, help="seconds")
a.add_argument("--tpot-slo", type=float, default=0.05, help="seconds")
a.add_argument("--tier", type=int, default=1, choices=[1, 2])
a.add_argument("--capacity", action="store_true", help="bisect the max rate meeting the SLO (90 %%)")
a.add_argument("--duration", type=float, default=300.0,
               help="with --capacity: seconds of arrivals per probe (requests = max(--requests, rate x duration))")
args = a.parse_args()


def layout(s):
    """(TP, EP, ETP, PP) of one side: stage GPUs = TP x DP = EP x ETP (default 16, TP4, ETP1, PP1)."""
    get = lambda f: getattr(args, f"{s}_{f}") or getattr(args, f, None)
    tp, dp, ep, etp, pp = get("tp"), getattr(args, f"{s}_dp"), get("ep"), get("etp") or 1, get("pp") or 1
    gpus = ep * etp if ep else tp * dp if tp and dp else 16
    tp = tp or (gpus // dp if dp else 4)
    if dp is not None and tp * dp != gpus:
        sys.exit(f"invalid configuration:\n  {s.upper()}: attention TP{tp} x DP{dp} must equal "
                 f"EP{ep} x ETP{etp} = {gpus} GPUs per stage")
    if gpus % etp:
        sys.exit(f"invalid configuration:\n  {s.upper()}: {gpus} GPUs per stage not divisible by ETP{etp}")
    return tp, gpus // etp, etp, pp


(ptp, pep, petp, ppp), (dtp, dep_, detp, dpp) = layout("p"), layout("d")
dep = Deployment(hardware=args.hardware, prefill_instances=args.p, decode_instances=args.d,
                 prefill_tp=ptp, decode_tp=dtp, prefill_ep=pep, decode_ep=dep_,
                 prefill_etp=petp, decode_etp=detp, prefill_pp=ppp, decode_pp=dpp,
                 prefill_budget=args.p_budget, decode_budget=args.d_budget or args.d_max_batch,
                 prefill_max_batch=args.p_max_batch, decode_max_batch=args.d_max_batch,
                 prefill_moe_chunks=args.p_moe_chunks, decode_moe_chunks=args.d_moe_chunks)
slo = {"ttft_s": args.ttft_slo, "tpot_s": args.tpot_slo}

hw = args.hardware.upper()
print(f"GPUs: {dep.gpus} {hw} = prefill {args.p} x {dep.width('prefill')} + decode {args.d} x {dep.width('decode')}")
for name, side, n in (("prefill", "prefill", args.p), ("decode ", "decode", args.d)):
    tp, ep, etp, pp = (getattr(dep, f"{side}_{f}") for f in ("tp", "ep", "etp", "pp"))
    w = dep.stage_gpus(side)
    print(f"  {name}: {n} instance{'s' * (n != 1)} x {dep.width(side)} GPUs | PP{pp} x (attention TP{tp} x DP{w // tp}), "
          f"MoE EP{ep} x ETP{etp} ({896 // ep} experts/group, {'whole experts' if etp == 1 else f'1/{etp} of each'} per GPU) | "
          f"{getattr(dep, side + '_budget')} tokens/round, max batch {getattr(dep, side + '_max_batch')}"
          + (f", MoE chunks {getattr(dep, side + '_moe_chunks')}" if getattr(dep, side + "_moe_chunks") > 1 else ""))
print(f"  workload: {args.requests} requests, ISL {args.isl}, OSL {args.osl}, {args.rate} req/s | "
      f"SLO TTFT<={args.ttft_slo}s TPOT<={args.tpot_slo*1e3:.0f}ms | Tier-{args.tier}")

errors = check(dep, args.isl + args.osl)
if errors:
    sys.exit("invalid configuration:\n" + "\n".join(f"  {e}" for e in errors))
models = None
if args.tier == 1:
    try:
        models = models_for(dep)
    except KeyError as e:
        sys.exit(f"legal, but outside Tier-1 coverage: {e.args[0]}\n  run it with --tier 2 (full simulation, slow)")


def run(rate, requests=None):
    requests = requests or args.requests
    t2 = tier2_config(dep)
    cfg = t2 if args.tier == 2 else te.tier1_config(t2, models)
    with tempfile.TemporaryDirectory(prefix="sim-") as tmp:
        tmp = Path(tmp)
        te.write_config(cfg, tmp / "c.toml")
        wl = workloads.write(workloads.fixed(requests, rate, args.isl, args.osl, seed=1), tmp / "w.jsonl")
        if args.tier == 2:
            subprocess.run([str(SIMTHESIZER / "target/release/simthesizer"), "--config", str(tmp / "c.toml"),
                            "--workload", str(wl), "--output-jsonl", str(tmp / "r.jsonl")], check=True)
        else:
            te.run(tmp / "c.toml", wl, tmp / "r.jsonl")
        if requests < 10:               # too few for a steady window: report per request
            for line in (tmp / "r.jsonl").read_text().splitlines():
                lat = json.loads(line)["sub_requests"][0]["token_latency"]
                print(f"  request: TTFT {lat[0]*1e3:.1f} ms, TPOT {statistics.fmean(lat[1:])*1e3:.2f} ms, "
                      f"tok/s/user {1/statistics.fmean(lat[1:]):.1f}, e2e {sum(lat):.2f} s")
            return None
        return steady.metrics(tmp / "r.jsonl", dep.gpus, slo,
                              sides=(args.p * dep.width("prefill"), args.d * dep.width("decode")))


def show(m, rate):
    print(f"  rate {rate:.3f} req/s | tok/s/GPU {m['tokens_per_s_gpu']:.1f} | tok/s/user {m['tokens_per_s_user']['p50']:.1f} | "
          f"TTFT p50/p99 {m['ttft_ms']['p50']:.0f}/{m['ttft_ms']['p99']:.0f} ms | TPOT p50/p99 "
          f"{m['tpot_ms']['p50']:.1f}/{m['tpot_ms']['p99']:.1f} ms | SLO attain {m['slo_attainment']:.2f} | "
          f"{'stable' if m['stable'] else 'queue growing'}")
    print(f"    prefill {m['prefill_tokens_per_s']:.0f} input tok/s = {m['prefill_tokens_per_s_gpu']:.1f} tok/s per prefill GPU | "
          f"decode {m['decode_tokens_per_s']:.0f} output tok/s = {m['decode_tokens_per_s_gpu']:.1f} tok/s per decode GPU")


try:
    tier2_config(dep)
except subprocess.CalledProcessError as e:        # launcher rejected a case check() misses
    sys.exit("invalid configuration (launcher):\n  " + (e.stderr or b"").decode().strip().splitlines()[-1])
if args.capacity:
    c = steady.capacity(lambda r: run(r, max(args.requests, round(r * args.duration))), args.rate, args.rate * 2, steps=7)
    print(f"capacity: {c['rate']:.3f} req/s")
    if c["rate"]:
        show(c["at_rate"], c["rate"])
else:
    m = run(args.rate)
    if m:
        show(m, args.rate)
