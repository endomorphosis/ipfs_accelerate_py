#!/usr/bin/env python3
"""Audit tests for UNI-204 trio bridge deferred-governance closure."""

from __future__ import annotations

from pathlib import Path
import unittest


class TestMCPServerUNI204TrioBridgeAudit(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[3]
        self.plan_text = (self.repo_root / "MCP_SERVER_UNIFICATION_PLAN.md").read_text(encoding="utf-8")
        self.matrix_text = (self.repo_root / "mcpplusplus" / "SPEC_GAP_MATRIX.md").read_text(encoding="utf-8")

    def test_uni204_completion_is_recorded_in_plan_and_matrix(self) -> None:
        self.assertIn("UNI-204", self.plan_text)
        self.assertIn("UNI-204", self.matrix_text)
        self.assertIn("trio_bridge.py", self.plan_text)
        self.assertIn("trio_bridge.py", self.matrix_text)
        self.assertIn("COMPLETE", self.plan_text)
        self.assertIn("COMPLETE", self.matrix_text)

    def test_canonical_runtime_does_not_import_source_trio_bridge(self) -> None:
        forbidden = "ipfs_datasets_py.mcp_server.trio_bridge"
        scan_roots = (
            self.repo_root / "ipfs_accelerate_py" / "mcp_server",
            self.repo_root / "ipfs_accelerate_py" / "mcp",
        )

        offenders: list[str] = []
        for root in scan_roots:
            for file_path in root.rglob("*.py"):
                text = file_path.read_text(encoding="utf-8")
                if forbidden in text:
                    offenders.append(str(file_path.relative_to(self.repo_root)))

        self.assertEqual(offenders, [], f"unexpected source trio_bridge imports: {offenders}")


if __name__ == "__main__":
    unittest.main()
