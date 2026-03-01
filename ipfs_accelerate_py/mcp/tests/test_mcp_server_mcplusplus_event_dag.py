#!/usr/bin/env python3
"""Deterministic tests for Event DAG store primitives."""

from __future__ import annotations

import unittest

from ipfs_accelerate_py.mcp_server.mcplusplus.event_dag import EventDAGStore


class TestMCPServerMCPPlusPlusEventDAG(unittest.TestCase):
    """Validate parent integrity and lineage traversal."""

    def test_add_and_lineage(self) -> None:
        store = EventDAGStore()

        root = "cid-root"
        child = "cid-child"

        store.add_event(root, {"parents": [], "intent_cid": "i1"})
        store.add_event(child, {"parents": [root], "intent_cid": "i2"})

        self.assertTrue(store.has_event(root))
        self.assertTrue(store.has_event(child))
        self.assertEqual(store.get_lineage(child), [root, child])

    def test_rejects_missing_parent(self) -> None:
        store = EventDAGStore()
        with self.assertRaises(ValueError):
            store.add_event("cid-orphan", {"parents": ["cid-missing"]})


if __name__ == "__main__":
    unittest.main()
