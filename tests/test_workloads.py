import unittest

from adapters import workloads as w


class WorkloadTests(unittest.TestCase):
    def test_scaling_preserves_shapes_and_multiplies_rate(self):
        base, fast = w.scaled(1.0), w.scaled(4.0)
        self.assertEqual([r["sub_requests"] for r in base], [r["sub_requests"] for r in fast])
        self.assertAlmostEqual(w.offered_rate(fast) / w.offered_rate(base), 4.0, places=9)

    def test_poisson_is_seeded(self):
        self.assertEqual(w.poisson(5.0, seed=3), w.poisson(5.0, seed=3))
        self.assertNotEqual(w.poisson(5.0, seed=3), w.poisson(5.0, seed=4))


if __name__ == "__main__":
    unittest.main()


class AgentSessionTests(unittest.TestCase):
    def test_chain_structure_and_growing_context(self):
        rows = w.agent_sessions(3, turns=4, rate=1.0, seed=1)
        self.assertEqual(len(rows), 3)
        subs = rows[0]["sub_requests"]
        self.assertEqual([s["kind"] for s in subs], ["llm", "tool_call"] * 3 + ["llm"])
        llm = [s for s in subs if s["kind"] == "llm"]
        self.assertTrue(all(b["input_tokens"] > a["input_tokens"] for a, b in zip(llm, llm[1:])))
        for i, s in enumerate(subs[:-1]):
            self.assertEqual(s["next"], [i + 1])
        self.assertEqual(subs[-1]["next"], [])


class TiledTests(unittest.TestCase):
    def test_unique_ids_and_rate(self):
        rows = w.tiled(3, rate=10.0, seed=2)
        n = len(w.load())
        self.assertEqual(len(rows), 3 * n)
        self.assertEqual(len({r["request_id"] for r in rows}), 3 * n)
        self.assertAlmostEqual(w.offered_rate(rows), 10.0, delta=2.0)
