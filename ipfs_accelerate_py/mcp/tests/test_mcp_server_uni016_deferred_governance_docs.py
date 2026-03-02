#!/usr/bin/env python3
"""Doc-contract tests for UNI-016 deferred module governance tracking."""

from __future__ import annotations

from pathlib import Path
import unittest


DEFERRED_MODULES = (
    "enterprise_api.py",
    "grpc_transport.py",
    "nl_ucan_policy.py",
    "trio_bridge.py",
    "compliance_checker.py",
    "investigation_mcp_client.py",
)


class TestMCPServerUNI016DeferredGovernanceDocs(unittest.TestCase):
    def setUp(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        self.plan_text = (repo_root / "MCP_SERVER_UNIFICATION_PLAN.md").read_text(encoding="utf-8")
        self.matrix_text = (repo_root / "mcpplusplus" / "SPEC_GAP_MATRIX.md").read_text(encoding="utf-8")

    def test_plan_contains_uni016_deferred_governance_ledger(self) -> None:
        self.assertIn("UNI-016", self.plan_text)
        self.assertIn("Deferred governance ledger", self.plan_text)
        self.assertIn("Primary Risk if Untracked", self.plan_text)
        self.assertIn("Target Milestone", self.plan_text)

    def test_plan_ledger_lists_all_deferred_modules(self) -> None:
        for module in DEFERRED_MODULES:
            self.assertIn(module, self.plan_text)

    def test_matrix_contains_uni016_governance_section(self) -> None:
        self.assertIn("Deferred Module Governance (`UNI-016`)", self.matrix_text)
        self.assertIn("Risk if Untracked", self.matrix_text)
        self.assertIn("Target Milestone", self.matrix_text)

    def test_matrix_governance_lists_all_deferred_modules(self) -> None:
        for module in DEFERRED_MODULES:
            self.assertIn(module, self.matrix_text)


if __name__ == "__main__":
    unittest.main()
