#!/usr/bin/env python3
"""UNI-205 compliance checker adapter parity tests."""

from __future__ import annotations

import unittest

from ipfs_accelerate_py.mcp_server import compliance_checker


class TestMCPServerUNI205ComplianceCheckerAdapter(unittest.TestCase):
    def test_adapter_exposes_expected_symbols(self) -> None:
        self.assertTrue(hasattr(compliance_checker, "ComplianceChecker"))
        self.assertTrue(hasattr(compliance_checker, "_COMPLIANCE_RULE_VERSION"))
        self.assertTrue(hasattr(compliance_checker, "make_default_checker"))
        self.assertTrue(hasattr(compliance_checker, "make_default_compliance_checker"))

    def test_default_checker_supports_basic_check_flow(self) -> None:
        checker = compliance_checker.make_default_checker()
        report = checker.check({"tool_name": "storage_tools"})

        # Source implementation returns an object with summary/passed and to_dict.
        # Fallback stub returns a dict with summary field.
        if isinstance(report, dict):
            self.assertIn("summary", report)
        else:
            self.assertTrue(hasattr(report, "summary"))


if __name__ == "__main__":
    unittest.main()
