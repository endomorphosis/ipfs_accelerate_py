"""Pure environment equivalence/refusal controls; no native/provider effects."""
import importlib.util
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
SOURCE = Path(__file__).parent / "qualify_native_gate.py"
spec = importlib.util.spec_from_file_location("nonmodel_environment_memory_controls", SOURCE)
fixture = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fixture
spec.loader.exec_module(fixture)
EXPECTED = ["PATH=/usr/bin:/bin", "HOME=/workspace", "TMPDIR=/workspace", "LANG=C.UTF-8",
            "LC_ALL=C.UTF-8", "PYTHONDONTWRITEBYTECODE=1"]


class ExactEnvironmentControls(unittest.TestCase):
    def refusal(self, actual, expected=EXPECTED):
        with self.assertRaises(RuntimeError):
            fixture.check_exact_environment(actual, expected)

    def test_exact_bindings_pass(self):
        self.assertIsNone(fixture.check_exact_environment(list(EXPECTED), EXPECTED))

    def test_reversed_order_passes_without_changing_raw_list(self):
        actual = list(reversed(EXPECTED))
        before = list(actual)
        self.assertIsNone(fixture.check_exact_environment(actual, EXPECTED))
        self.assertEqual(actual, before)

    def test_rotated_order_passes(self):
        self.assertIsNone(fixture.check_exact_environment(EXPECTED[2:] + EXPECTED[:2], EXPECTED))

    def test_identical_duplicate_refuses(self):
        self.refusal(EXPECTED + [EXPECTED[0]])

    def test_changed_duplicate_refuses(self):
        self.refusal(EXPECTED + ["PATH=/unexpected"])

    def test_extra_unknown_key_refuses(self):
        self.refusal(EXPECTED + ["UNKNOWN=extra"])

    def test_missing_key_refuses(self):
        self.refusal(EXPECTED[:-1])

    def test_changed_value_refuses(self):
        self.refusal(["PATH=/unexpected", *EXPECTED[1:]])

    def test_entry_without_equals_refuses(self):
        self.refusal(["PATH", *EXPECTED[1:]])

    def test_empty_key_refuses(self):
        self.refusal(["=/usr/bin:/bin", *EXPECTED[1:]])

    def test_non_string_entry_refuses(self):
        self.refusal([None, *EXPECTED[1:]])

    def test_non_list_inventory_refuses(self):
        self.refusal(tuple(EXPECTED))

    def test_unknown_key_spelling_refuses(self):
        self.refusal(["BAD KEY=value", *EXPECTED[1:]])

    def test_duplicate_expected_inventory_refuses(self):
        self.refusal(EXPECTED, EXPECTED + [EXPECTED[0]])

    def test_first_equals_preserves_complete_value(self):
        self.assertIsNone(fixture.check_exact_environment(["EXAMPLE=left=right"], ["EXAMPLE=left=right"]))


if __name__ == "__main__":
    unittest.main()
