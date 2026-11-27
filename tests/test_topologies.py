import unittest

from adapters.topologies import Deployment, check


class CheckTests(unittest.TestCase):
    def test_default_is_legal(self):
        self.assertEqual(check(Deployment(), 9216), [])

    def test_rejections(self):
        cases = {"equal attention TP": Deployment(prefill_tp=8, decode_tp=4),
                 "divide 896": Deployment(prefill_ep=12),
                 "expert TP must be": Deployment(decode_etp=3),
                 "power of two": Deployment(prefill_ep=7, prefill_etp=2),
                 "exceed usable": Deployment(prefill_ep=4, decode_ep=4),
                 "instances 0": Deployment(prefill_instances=0)}
        for why, dep in cases.items():
            self.assertTrue(any(why in e for e in check(dep, 9216)), why)
        self.assertTrue(any("KV" in e for e in check(Deployment(), 10**6)))

    def test_per_side_ep_and_legacy_ledger(self):
        d = Deployment(decode_ep=32)
        self.assertEqual(d.gpus, 48)
        self.assertIn("ep16-32", d.key)
        self.assertEqual(Deployment.of({"ep": 16}).key, Deployment().key)

    def test_expert_tp_layout(self):
        d = Deployment(decode_ep=8, decode_etp=2)        # 16 GPUs, experts on 2 GPUs each
        self.assertEqual((d.width("decode"), d.gpus), (16, 32))
        self.assertIn("ep16-8-etp1-2", d.key)
        self.assertEqual(check(d, 9216), [])
        self.assertEqual(check(Deployment(prefill_etp=8, decode_etp=8), 9216), [])

    def test_pipeline_layout_and_memory(self):
        d = Deployment(prefill_ep=8, decode_ep=8, prefill_pp=2, decode_pp=2)   # EP8 fits only with PP2
        self.assertEqual((d.stage_gpus("prefill"), d.width("prefill"), d.gpus), (8, 16, 32))
        self.assertIn("ep8-pp2", d.key)
        self.assertEqual(check(d, 9216), [])
        self.assertTrue(any("exceed" in e for e in check(Deployment(prefill_ep=8, decode_ep=8), 9216)))
        self.assertTrue(any("PP9" in e for e in check(Deployment(prefill_pp=9), 9216)))


if __name__ == "__main__":
    unittest.main()
