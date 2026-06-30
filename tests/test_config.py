import unittest

from viral_slop.config import _normalize_config_values


class ConfigTests(unittest.TestCase):
    def test_normalize_does_not_disable_missing_min_solution_steps(self) -> None:
        self.assertNotIn("min_solution_steps", _normalize_config_values({}))

    def test_blank_min_solution_steps_disables_minimum(self) -> None:
        normalized = _normalize_config_values({"min_solution_steps": ""})

        self.assertEqual(normalized["min_solution_steps"], 0)


if __name__ == "__main__":
    unittest.main()
