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

    def test_snapshot_rebuild_tolerates_reordered_and_invalid_entries(self) -> None:
        store = EventDAGStore()
        store.add_event("cid-root", {"parents": [], "intent_cid": "root"})
        store.add_event("cid-a", {"parents": ["cid-root"], "intent_cid": "a"})
        store.add_event("cid-leaf", {"parents": ["cid-a"], "intent_cid": "leaf"})

        snapshot = store.export_snapshot()
        events = list(reversed(snapshot.get("events") or []))
        reordered_snapshot = {
            "version": snapshot.get("version"),
            "events": [
                "ignored-entry",
                {"event_cid": "", "payload": {"parents": []}},
                {"event_cid": "cid-bad", "payload": None},
                *events,
            ],
            "stats": {"event_count": 999},
        }

        rebuilt = EventDAGStore.from_snapshot(reordered_snapshot)
        self.assertEqual(rebuilt.stats().get("event_count"), 3)
        self.assertEqual(rebuilt.get_lineage("cid-leaf"), ["cid-root", "cid-a", "cid-leaf"])
        self.assertEqual(rebuilt.replay_from_root("cid-root"), ["cid-root", "cid-a", "cid-leaf"])
        self.assertEqual(rebuilt.rollback_path("cid-leaf"), ["cid-leaf", "cid-a", "cid-root"])

    def test_merge_node_lineage_prefers_lexically_smallest_parent_path(self) -> None:
        store = EventDAGStore()
        store.add_event("cid-root", {"parents": [], "intent_cid": "root"})
        store.add_event("cid-z-branch", {"parents": ["cid-root"], "intent_cid": "z"})
        store.add_event("cid-a-branch", {"parents": ["cid-root"], "intent_cid": "a"})
        store.add_event(
            "cid-merge",
            {
                "parents": ["cid-z-branch", "cid-a-branch"],
                "intent_cid": "merge",
            },
        )

        self.assertEqual(store.get_lineage("cid-merge"), ["cid-root", "cid-a-branch", "cid-merge"])

    def test_merge_replay_deduplicates_shared_merge_descendant(self) -> None:
        store = EventDAGStore()
        store.add_event("cid-root", {"parents": [], "intent_cid": "root"})
        store.add_event("cid-a", {"parents": ["cid-root"], "intent_cid": "a"})
        store.add_event("cid-b", {"parents": ["cid-root"], "intent_cid": "b"})
        store.add_event("cid-merge", {"parents": ["cid-b", "cid-a"], "intent_cid": "merge"})

        replay = store.replay_from_root("cid-root")
        self.assertEqual(replay, ["cid-root", "cid-a", "cid-b", "cid-merge"])
        self.assertEqual(replay.count("cid-merge"), 1)

        rollback = store.rollback_path("cid-merge")
        self.assertEqual(rollback, ["cid-merge", "cid-a", "cid-root"])

    def test_snapshot_roundtrip_preserves_merge_fork_determinism(self) -> None:
        store = EventDAGStore()
        store.add_event("cid-root", {"parents": [], "intent_cid": "root"})
        store.add_event("cid-branch-2", {"parents": ["cid-root"], "intent_cid": "b2"})
        store.add_event("cid-branch-1", {"parents": ["cid-root"], "intent_cid": "b1"})
        store.add_event(
            "cid-merge",
            {"parents": ["cid-branch-2", "cid-branch-1"], "intent_cid": "merge"},
        )

        rebuilt = EventDAGStore.from_snapshot(store.export_snapshot())
        self.assertEqual(rebuilt.get_lineage("cid-merge"), ["cid-root", "cid-branch-1", "cid-merge"])
        self.assertEqual(rebuilt.replay_from_root("cid-root"), ["cid-root", "cid-branch-1", "cid-branch-2", "cid-merge"])

    def test_rejects_conflicting_duplicate_event_payload(self) -> None:
        store = EventDAGStore()
        store.add_event("cid-root", {"parents": [], "intent_cid": "i1"})

        # Idempotent duplicate is allowed.
        store.add_event("cid-root", {"parents": [], "intent_cid": "i1"})

        # Conflicting duplicate for same CID must be rejected explicitly.
        with self.assertRaises(ValueError):
            store.add_event("cid-root", {"parents": [], "intent_cid": "i2"})

    def test_add_event_isolated_from_external_nested_mutation(self) -> None:
        store = EventDAGStore()
        payload = {
            "parents": [],
            "intent_cid": "i1",
            "meta": {"attrs": {"priority": 1}},
        }

        store.add_event("cid-root", payload)
        payload["meta"]["attrs"]["priority"] = 9

        stored = store.get_event("cid-root") or {}
        self.assertEqual((((stored.get("meta") or {}).get("attrs") or {}).get("priority")), 1)

    def test_get_event_returns_deep_copy_for_nested_payloads(self) -> None:
        store = EventDAGStore()
        store.add_event(
            "cid-root",
            {
                "parents": [],
                "intent_cid": "i1",
                "meta": {"attrs": {"priority": 1}},
                "labels": ["root"],
            },
        )

        first = store.get_event("cid-root") or {}
        first["meta"]["attrs"]["priority"] = 7
        first["labels"].append("mutated")

        second = store.get_event("cid-root") or {}
        self.assertEqual((((second.get("meta") or {}).get("attrs") or {}).get("priority")), 1)
        self.assertEqual(second.get("labels"), ["root"])

    def test_export_snapshot_returns_deep_copy_payloads(self) -> None:
        store = EventDAGStore()
        store.add_event(
            "cid-root",
            {
                "parents": [],
                "intent_cid": "i1",
                "meta": {"attrs": {"priority": 1}},
            },
        )

        snapshot = store.export_snapshot()
        events = snapshot.get("events") or []
        events[0]["payload"]["meta"]["attrs"]["priority"] = 11

        fresh = store.export_snapshot()
        fresh_events = fresh.get("events") or []
        self.assertEqual(
            (((fresh_events[0].get("payload") or {}).get("meta") or {}).get("attrs") or {}).get("priority"),
            1,
        )

    def test_large_dag_replay_and_rollback_are_deterministic(self) -> None:
        store = EventDAGStore()
        store.add_event("cid-root", {"parents": [], "intent_cid": "root"})

        # Build a deterministic layered DAG:
        # layer 1: 10 children of root
        # layer 2: each layer-1 node has two children
        # layer 3: each layer-2 node has one child
        layer1 = []
        for i in range(10):
            cid = f"cid-l1-{i:02d}"
            store.add_event(cid, {"parents": ["cid-root"], "intent_cid": cid})
            layer1.append(cid)

        layer2 = []
        for parent in layer1:
            for branch in ("a", "b"):
                cid = f"cid-l2-{parent}-{branch}"
                store.add_event(cid, {"parents": [parent], "intent_cid": cid})
                layer2.append(cid)

        layer3 = []
        for parent in layer2:
            cid = f"cid-l3-{parent}"
            store.add_event(cid, {"parents": [parent], "intent_cid": cid})
            layer3.append(cid)

        replay_first = store.replay_from_root("cid-root")
        replay_second = store.replay_from_root("cid-root")
        self.assertEqual(replay_first, replay_second)
        self.assertEqual(len(replay_first), 1 + len(layer1) + len(layer2) + len(layer3))
        self.assertEqual(replay_first[0], "cid-root")

        leaf = sorted(layer3)[-1]
        rollback_first = store.rollback_path(leaf)
        rollback_second = store.rollback_path(leaf)
        self.assertEqual(rollback_first, rollback_second)
        self.assertEqual(rollback_first[0], leaf)
        self.assertEqual(rollback_first[-1], "cid-root")

        snapshot = store.export_snapshot()
        rebuilt = EventDAGStore.from_snapshot(snapshot)
        self.assertEqual(rebuilt.replay_from_root("cid-root"), replay_first)
        self.assertEqual(rebuilt.rollback_path(leaf), rollback_first)


if __name__ == "__main__":
    unittest.main()
