import json, sys
from pathlib import Path
d = Path(__file__).parent
a, b = (json.loads((d / f"result-{x}.json").read_text()) for x in sys.argv[1:3])
print(f"| hw | stage | batch | tokens | ctx | {a['label']} ms | {b['label']} ms | ratio |\n|---|---|---|---|---|---|---|---|")
for x, y in zip(a["roofline"], b["roofline"]):
    print(f"| {x['hw']} | {x['stage']} | {x['batch']} | {x['tokens']} | {x['context']} | "
          f"{x['serial_roofline_us']/1e3:.1f} | {y['serial_roofline_us']/1e3:.1f} | "
          f"{y['serial_roofline_us']/x['serial_roofline_us']:.3f} |")
print(f"\n| hw | GPUs | aTP | {a['label']} weights GB | {b['label']} weights GB | {b['label']} status | {b['label']} req@32k |\n|---|---|---|---|---|---|---|")
for x, y in zip(a["capacity"], b["capacity"]):
    m = (y["max_requests"] or {})
    print(f"| {x['hardware']} | {x['n_gpus']} | {x['attn_tp']} | {x['weight_bytes_per_gpu']/1e9:.1f} | "
          f"{y['weight_bytes_per_gpu']/1e9:.1f} | {y['status']} | {m.get(32768, m.get('32768','-'))} |")
