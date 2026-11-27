import json
import tempfile
import unittest
from pathlib import Path

from adapters import feedback


def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


class FeedbackTests(unittest.TestCase):
    def test_instance_busy_window_and_saturation_labels(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            res = write(t / "r.jsonl", [{"request_id": 0, "arrival_time": 0.0, "initial": [0],
                                        "sub_requests": [{"kind": "llm", "input_tokens": 4, "output_tokens": 3,
                                                          "next": [], "token_latency": [1.0, 0.5, 0.5]}]}])
            trace = write(t / "t.jsonl", [
                {"event": "node_create", "node": 1, "details": "StepModel(instance=p0 seconds=0.9 requests=1 tokens=4)"},
                {"event": "dispatch", "node": 1, "time_s": 0.0},
                {"event": "node_create", "node": 2, "details": "StepModel(instance=d0 seconds=0.5 requests=32 tokens=32)"},
                {"event": "dispatch", "node": 2, "time_s": 1.0},
            ])
            config = {"system": {"kind": "pd_disaggregation",
                                 "prefill_instances": [{"model_runner": {"kind": "step_model", "ep": 16},
                                                        "scheduler": {"max_batch": 2}}],
                                 "decode_instances": [{"model_runner": {"kind": "step_model", "ep": 16},
                                                       "scheduler": {"max_batch": 2}}]}}
            rec = feedback.record(res, gpus=2, trace_jsonl=trace, config=config)
            self.assertAlmostEqual(rec["instances"]["p0"]["busy_s"], 0.9)
            labels = {b["class"] for b in rec["bottlenecks"]}
            self.assertIn("prefill_saturated", labels)       # 0.9 / 0.9 window
            self.assertIn("decode_batch_saturated", labels)  # 32 / (2*16)
            self.assertAlmostEqual(rec["requests"]["jct_s"]["mean"], 2.0)


if __name__ == "__main__":
    unittest.main()
