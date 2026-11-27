import json
import unittest
from pathlib import Path

from validation import calibration as c


class CalibrationTests(unittest.TestCase):
    def test_frontier_samples_parse_with_positive_values(self):
        for samples in (c.dense_gemm_samples(), c.moe_grouped_gemm_samples()):
            self.assertGreater(len(samples), 500)
            self.assertTrue(all(s.flops > 0 and s.bytes > 0 and s.measured_s > 0
                                for s in samples))

    def test_recorded_dense_fit_reproduces_measurements(self):
        r = json.loads((Path(__file__).parents[1] / "experiments"
                        / "E0006_h200_calibration" / "result.json").read_text())
        dense = next(f for f in r["fits"] if f["kernel_class"] == "dense_gemm_bf16")
        fit = c.Fit(**dense)
        errs = sorted(abs(fit.predict(s) - s.measured_s) / s.measured_s
                      for s in c.dense_gemm_samples())
        self.assertLess(errs[len(errs) // 2], 0.05)

    def test_profiles_are_valid_engine_efficiency(self):
        p = json.loads((Path(c.__file__).parent / "cost_profiles.json").read_text())
        for name, prof in p.items():
            if name.startswith("_"):
                continue
            for k, v in prof["efficiency"].items():
                self.assertIn(k, ("compute", "memory", "network", "kernel_overhead_us"))
                self.assertTrue(v >= 0 if k == "kernel_overhead_us" else 0 < v <= 1)


if __name__ == "__main__":
    unittest.main()
