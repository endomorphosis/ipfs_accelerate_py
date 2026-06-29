#!/usr/bin/env python3
"""Compatibility adapter tests for UNI-012 artifact/IDL/event-dag surfaces."""

from __future__ import annotations

import unittest

from ipfs_accelerate_py.mcp_server.cid_artifacts import artifact_cid
from ipfs_accelerate_py.mcp_server.event_dag import EventDAG, EventNode
from ipfs_accelerate_py.mcp_server.interface_descriptor import (
    InterfaceRepository,
    build_descriptor,
)


class TestMCPServerUNI012LegacyAdapters(unittest.TestCase):
    def test_artifact_cid_deterministic(self) -> None:
        payload = {"b": 2, "a": 1}
        cid1 = artifact_cid(payload)
        cid2 = artifact_cid({"a": 1, "b": 2})
        self.assertEqual(cid1, cid2)
        self.assertTrue(cid1.startswith("cidv1-sha256-"))

    def test_interface_repository_register_and_compat(self) -> None:
        repo = InterfaceRepository(supported_capabilities={"cap.read"})

        candidate = build_descriptor(
            name="candidate",
            namespace="demo",
            version="1.0.0",
            methods=[],
            requires=["cap.read"],
        )
        required = build_descriptor(
            name="required",
            namespace="demo",
            version="1.0.0",
            methods=[],
            requires=["cap.read", "cap.write"],
        )
        candidate_cid = repo.register(candidate)
        required_cid = repo.register(required)

        verdict = repo.compat(candidate_cid, required_cid=required_cid)
        self.assertFalse(verdict.compatible)
        self.assertIn("cap.write", verdict.requires_missing)

    def test_event_dag_append_frontier_and_walk(self) -> None:
        dag = EventDAG(strict=True)

        root = EventNode(intent_cid="intent-root", decision_cid="dec-root")
        root_cid = dag.append(root)

        child = EventNode(parents=[root_cid], intent_cid="intent-child", decision_cid="dec-child")
        child_cid = dag.append(child)

        self.assertIn(child_cid, dag.frontier())
        walk = dag.walk(child_cid)
        self.assertEqual(walk, [child_cid, root_cid])


if __name__ == "__main__":
    unittest.main()
