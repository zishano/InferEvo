import unittest

from mechanisms import spec as specs
from mechanisms.registry import CLI, Registry

KV = "sim:crates/simthesizer-core/src/kv.rs"
PD = "sim:crates/simthesizer-core/src/systems/pd_disaggregation.rs"
BRIDGE = "sim:scripts/rooflang_bridge/kimi_k3_shared.py"


class RegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reg = Registry()

    def test_scanned_edges(self):
        c = self.reg.c
        self.assertIn(KV, c[PD].deps)                                  # Rust `use crate::...`
        self.assertEqual(c[PD].kind, "system")                         # register_system!
        self.assertIn("pd_disaggregation", c[PD].names)
        self.assertIn(PD, c[CLI].deps)                                 # link-time registry
        self.assertIn(BRIDGE, c["ie:adapters/tier1.py"].deps)          # Python, vendored bridge
        self.assertIn(CLI, c["ie:adapters/tier1_engine.py"].deps)      # runs the binary
        self.assertIn("sim:scripts/rooflang_worker.py",                # overlay process edge
                      c["sim:crates/simthesizer-core/src/model_runners/rooflang.rs"].deps)

    def test_impact_selects_dependent_tests(self):
        tests = {t.id for t in self.reg.tests_for({BRIDGE})}
        self.assertIn("sim:tests/test_kimi_k3_shared.py", tests)
        self.assertIn("ie:tests/test_tier1.py", tests)
        self.assertNotIn("rl:tests/unit_tests/__init__.py", tests)
        self.assertNotIn("ie:tests/test_steady.py", tests)               # does not reach the bridge

    def test_topological_order(self):
        order = self.reg.topological({KV, PD})
        self.assertLess(order.index(KV), order.index(PD))


class SpecTests(unittest.TestCase):
    def test_reference_spec_is_valid(self):
        self.assertEqual(specs.validate(specs.load("M-MOECHUNK")), [])

    def test_rejects_bad_spec(self):
        bad = {"id": "X", "components": {"modified": ["sim:nope.rs"], "added": [KV]},
               "design_variables": {"k": {"values": [1, 2], "default": 3}},
               "validation": {"default_off_identity": True}, "evaluation": {"script": "experiments/none.py"}}
        errors = " | ".join(specs.validate(bad))
        for needle in ("missing field `hypothesis`", "unknown component sim:nope.rs", "already exists",
                       "default not among values", "does not exist"):
            self.assertIn(needle, errors)


if __name__ == "__main__":
    unittest.main()
