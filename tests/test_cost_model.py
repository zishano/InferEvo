import unittest

from adapters import cost_model


class CostModelIdentity(unittest.TestCase):
    def test_version_is_stable_sha256(self):
        v = cost_model.cost_model_version()
        self.assertRegex(v, r"^[0-9a-f]{64}$")
        self.assertEqual(v, cost_model.cost_model_version())

    def test_drift_lists_only_known_kinds(self):
        for d in cost_model.vendored_drift():
            self.assertIn(d["kind"], {"content_differs", "missing_upstream"})


class Fp4PathSensitivity(unittest.TestCase):
    def test_decode_is_hbm_bound_and_insensitive(self):
        a = cost_model.roofline_breakdown("h200", 64, 1, 8192)
        b = cost_model.roofline_breakdown("h200", 64, 1, 8192, fp4_rate=989.5)
        self.assertAlmostEqual(a["serial_roofline_us"], b["serial_roofline_us"])

    def test_long_prefill_is_slower_than_old_native_fp4_pricing(self):
        # CM-1: preset FP4 runs at the BF16 rate; 1979 was the former native rate.
        emulated = cost_model.roofline_breakdown("h200", 1, 8192, None)
        native = cost_model.roofline_breakdown("h200", 1, 8192, None, fp4_rate=1979.0)
        self.assertGreater(
            emulated["serial_roofline_us"] / native["serial_roofline_us"], 1.2)


if __name__ == "__main__":
    unittest.main()
