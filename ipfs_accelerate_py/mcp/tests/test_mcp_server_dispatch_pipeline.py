#!/usr/bin/env python3
"""Unit tests for canonical unified dispatch pipeline helpers."""

import unittest

from ipfs_accelerate_py.mcp_server.dispatch_pipeline import (
    coerce_dispatch_bool,
    coerce_dispatch_dict,
    coerce_dispatch_list,
    compute_dispatch_intent_cid,
    normalize_dispatch_parameters,
)


class TestDispatchPipelineHelpers(unittest.TestCase):
    def test_normalize_dispatch_parameters(self) -> None:
        self.assertEqual(normalize_dispatch_parameters({"k": 1}), {"k": 1})
        self.assertEqual(normalize_dispatch_parameters(None), {})
        self.assertEqual(normalize_dispatch_parameters("x"), {})

    def test_coerce_dispatch_bool(self) -> None:
        self.assertTrue(coerce_dispatch_bool(True, field_name="f"))
        self.assertFalse(coerce_dispatch_bool("false", field_name="f"))
        self.assertTrue(coerce_dispatch_bool("yes", field_name="f"))
        self.assertFalse(coerce_dispatch_bool(0, field_name="f"))

        with self.assertRaises(ValueError):
            coerce_dispatch_bool("maybe", field_name="f")

    def test_coerce_dispatch_list_and_dict(self) -> None:
        self.assertEqual(coerce_dispatch_list([1, 2], field_name="l"), [1, 2])
        self.assertEqual(coerce_dispatch_list(None, field_name="l"), [])
        self.assertEqual(coerce_dispatch_dict({"a": 1}, field_name="d"), {"a": 1})
        self.assertEqual(coerce_dispatch_dict(None, field_name="d"), {})

        with self.assertRaises(ValueError):
            coerce_dispatch_list({"a": 1}, field_name="l")
        with self.assertRaises(ValueError):
            coerce_dispatch_dict([1, 2], field_name="d")

    def test_compute_dispatch_intent_cid_deterministic(self) -> None:
        first = compute_dispatch_intent_cid("smoke", "echo", {"value": "x"})
        second = compute_dispatch_intent_cid("smoke", "echo", {"value": "x"})
        different = compute_dispatch_intent_cid("smoke", "echo", {"value": "y"})

        self.assertIsInstance(first, str)
        self.assertEqual(first, second)
        self.assertNotEqual(first, different)


if __name__ == "__main__":
    unittest.main()
