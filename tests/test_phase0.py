import unittest

from validation import capacity, hardware, kimi_conformance
from adapters.engines import source_identity


class SourceIdentity(unittest.TestCase):
    def test_pins_both_integrated_repos(self):
        ids = source_identity()
        for name in ("simthesizer", "rooflang"):
            self.assertRegex(ids[name]["commit"], r"^[0-9a-f]{40}$")
            if ids[name]["dirty"]:
                self.assertEqual(len(ids[name]["dirty_digest"]), 64)


class KimiConformance(unittest.TestCase):
    def test_structure_matches_official_config(self):
        self.assertEqual(kimi_conformance.structure_findings(), [])

    def test_precision_matches_official_checkpoint(self):
        # MC-1 (cost_models/MC-1.md); total matches the ~1.56 TB checkpoint.
        p = kimi_conformance.precision_findings()
        self.assertEqual(p["official_group_size"], 32)
        self.assertIsNone(p["official_kv_cache_scheme"])
        self.assertEqual(p["deviations"], [])
        self.assertEqual(p["rooflang_total_bytes"], 1_559_951_638_400)


class HardwareAudit(unittest.TestCase):
    def test_active_presets_match_sourced_facts(self):
        # CM-1..3 applied on the active branch: no deviations remain.
        self.assertEqual(hardware.audit_preset("h200"), [])
        self.assertEqual(hardware.audit_preset("b300"), [])

    def test_b300_fp4_is_native(self):
        kinds = {(f["kind"], f["field"]) for f in hardware.audit_preset("b300")}
        self.assertNotIn(("non_native_precision_priced_natively", "fp4"), kinds)


class Capacity(unittest.TestCase):
    def test_official_precision_16xh200(self):
        # experts/16 + BF16 nonexpert/TP + replicated vectors (kimi-k3.md).
        r = capacity.evaluate(capacity.Layout("h200", 16, 2))
        self.assertEqual(r.weight_bytes_per_gpu,
                         90_403_504_128 + 113_495_572_352 // 2 + 10_467_328 // 2)
        self.assertEqual(r.status, "infeasible_weights_oom")
        # MC-2: FP32 recurrent + BF16 conv state, 69 KDA layers, split over TP2.
        self.assertEqual(r.kda_state_bytes_per_request_per_gpu,
                         69 * (96 * 128 * 128 * 4 + 96 * 3 * 128 * 4 * 2) // 2)
        self.assertEqual(r.mla_kv_bytes_per_token_per_gpu, 24 * 576 * 2)
        self.assertEqual(capacity.evaluate(capacity.Layout("h200", 16, 4)).status,
                         "feasible")

    def test_rejects_uneven_expert_shards(self):
        r = capacity.evaluate(capacity.Layout("b300", 256, 8))
        self.assertEqual(r.status, "infeasible_uneven_whole_expert_shards")

    def test_eight_h200_cannot_hold_weights(self):
        r = capacity.evaluate(capacity.Layout("h200", 8, 8))
        self.assertEqual(r.status, "infeasible_weights_oom")

    def test_capacity_monotone_in_context(self):
        r = capacity.evaluate(capacity.Layout("b300", 16, 2))
        m = r.max_requests
        self.assertGreaterEqual(m[8192], m[32768])
        self.assertGreaterEqual(m[32768], m[131072])


if __name__ == "__main__":
    unittest.main()
