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

    def test_snapshot_roundtrip_and_replay_rollback(self) -> None:
        store = EventDAGStore()
        store.add_event("cid-root", {"parents": [], "intent_cid": "i1"})
        store.add_event("cid-a", {"parents": ["cid-root"], "intent_cid": "i2"})
        store.add_event("cid-b", {"parents": ["cid-root"], "intent_cid": "i3"})
        store.add_event("cid-leaf", {"parents": ["cid-a"], "intent_cid": "i4"})

        snapshot = store.export_snapshot()
        self.assertEqual(snapshot.get("version"), 1)
        self.assertEqual((snapshot.get("stats") or {}).get("event_count"), 4)

        rebuilt = EventDAGStore.from_snapshot(snapshot)
        self.assertEqual(rebuilt.stats().get("event_count"), 4)
        self.assertEqual(rebuilt.get_lineage("cid-leaf"), ["cid-root", "cid-a", "cid-leaf"])

        replay_order = rebuilt.replay_from_root("cid-root")
        self.assertEqual(replay_order, ["cid-root", "cid-a", "cid-b", "cid-leaf"])

        rollback = rebuilt.rollback_path("cid-leaf")
        self.assertEqual(rollback, ["cid-leaf", "cid-a", "cid-root"])

    def test_snapshot_rebuild_rejects_unresolved_parents(self) -> None:
        with self.assertRaises(ValueError):
            EventDAGStore.from_snapshot(
                {
                    "version": 1,
                    "events": [
                        {
                            "event_cid": "cid-leaf",
                            "payload": {"parents": ["cid-missing"]},
                        }
                    ],
                }
            )


if __name__ == "__main__":
    unittest.main()
