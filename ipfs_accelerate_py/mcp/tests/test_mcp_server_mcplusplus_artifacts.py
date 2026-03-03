#!/usr/bin/env python3
"""Deterministic tests for CID-native execution artifacts."""

from __future__ import annotations

import tempfile
import unittest

from ipfs_accelerate_py.mcp_server.mcplusplus.artifacts import (
    ArtifactStore,
    build_decision,
    build_event,
    build_intent,
    build_receipt,
    canonicalize_artifact,
    compute_artifact_cid,
    envelope_from_payloads,
)


class TestMCPServerMCPPlusPlusArtifacts(unittest.TestCase):
    """Validate Profile B helper determinism and chain integrity."""

    def test_canonicalization_and_cid_are_deterministic(self) -> None:
        a = {"z": [3, 2, 1], "a": {"k": "v"}}
        b = {"a": {"k": "v"}, "z": [3, 2, 1]}

        self.assertEqual(canonicalize_artifact(a), canonicalize_artifact(b))
        self.assertEqual(compute_artifact_cid(a), compute_artifact_cid(b))

    def test_artifact_builders_have_expected_fields(self) -> None:
        intent = build_intent(interface_cid="cid-iface", tool="demo.tool", input_cid="cid-input")
        self.assertEqual(intent["interface_cid"], "cid-iface")

        decision = build_decision(decision="allow", intent_cid="cid-intent")
        self.assertEqual(decision["decision"], "allow")

        receipt = build_receipt(intent_cid="cid-intent", output_cid="cid-output", decision_cid="cid-decision")
        self.assertEqual(receipt["decision_cid"], "cid-decision")

        event = build_event(
            interface_cid="cid-iface",
            intent_cid="cid-intent",
            proof_cid="cid-proof",
            decision_cid="cid-decision",
            output_cid="cid-output",
            receipt_cid="cid-receipt",
        )
        self.assertEqual(event["intent_cid"], "cid-intent")

    def test_envelope_chain_integrity(self) -> None:
        envelope = envelope_from_payloads(
            interface_cid="cid-interface",
            input_payload={"x": 1},
            tool="demo.call",
            output_payload={"ok": True},
            decision="allow",
            proof_cid="cid-proof",
            policy_cid="cid-policy",
            correlation_id="corr-1",
            parent_event_cids=["cid-parent"],
        )

        self.assertEqual(envelope["intent"]["input_cid"], envelope["input_cid"])
        self.assertEqual(envelope["decision"]["intent_cid"], envelope["intent_cid"])
        self.assertEqual(envelope["receipt"]["decision_cid"], envelope["decision_cid"])
        self.assertEqual(envelope["event"]["receipt_cid"], envelope["receipt_cid"])
        self.assertEqual(envelope["event"]["parents"], ["cid-parent"])

    def test_artifact_store_put_many_and_stats(self) -> None:
        store = ArtifactStore()
        written = store.put_many(
            {
                "cid-a": {"k": "a"},
                "cid-b": {"k": "b"},
            }
        )
        self.assertEqual(written, 2)
        self.assertEqual(store.stats().get("artifact_count"), 2)

    def test_artifact_store_returns_payload_copy(self) -> None:
        store = ArtifactStore()
        store.put("cid-x", {"k": "v"})
        payload = store.get("cid-x")
        self.assertEqual(payload, {"k": "v"})

        # Ensure callers cannot mutate internal store state via returned object.
        payload["k"] = "changed"
        self.assertEqual(store.get("cid-x"), {"k": "v"})

    def test_artifact_store_json_round_trip_is_deterministic(self) -> None:
        envelope = envelope_from_payloads(
            interface_cid="cid-iface",
            input_payload={"v": 1},
            tool="demo.call",
            output_payload={"ok": True},
            decision="allow",
            proof_cid="cid-proof",
            policy_cid="cid-policy",
            correlation_id="corr-rt",
        )

        records = {
            envelope["intent_cid"]: envelope["intent"],
            envelope["decision_cid"]: envelope["decision"],
            envelope["receipt_cid"]: envelope["receipt"],
            envelope["event_cid"]: envelope["event"],
        }

        store = ArtifactStore()
        written = store.put_many(records)
        self.assertEqual(written, 4)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/artifacts.json"
            saved = store.save_json(path)
            self.assertEqual(saved, 4)

            reloaded = ArtifactStore.load_json(path)
            self.assertEqual(reloaded.stats().get("artifact_count"), 4)
            self.assertEqual(reloaded.export_records(), store.export_records())

            saved_again = reloaded.save_json(path)
            self.assertEqual(saved_again, 4)
            self.assertEqual(ArtifactStore.load_json(path).export_records(), store.export_records())

    def test_replay_reconstructs_chain_after_json_reload(self) -> None:
        root = envelope_from_payloads(
            interface_cid="cid-iface",
            input_payload={"v": "root"},
            tool="demo.root",
            output_payload={"ok": True},
            correlation_id="corr-replay",
        )
        leaf = envelope_from_payloads(
            interface_cid="cid-iface",
            input_payload={"v": "leaf"},
            tool="demo.leaf",
            output_payload={"ok": True},
            correlation_id="corr-replay",
            parent_event_cids=[root["event_cid"]],
        )

        store = ArtifactStore()
        store.put_many(
            {
                root["intent_cid"]: root["intent"],
                root["decision_cid"]: root["decision"],
                root["receipt_cid"]: root["receipt"],
                root["event_cid"]: root["event"],
                leaf["intent_cid"]: leaf["intent"],
                leaf["decision_cid"]: leaf["decision"],
                leaf["receipt_cid"]: leaf["receipt"],
                leaf["event_cid"]: leaf["event"],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/artifacts-replay.json"
            store.save_json(path)
            reloaded = ArtifactStore.load_json(path)

        leaf_event = reloaded.get(leaf["event_cid"])
        self.assertIsNotNone(leaf_event)
        self.assertEqual((leaf_event or {}).get("parents"), [root["event_cid"]])

        leaf_receipt = reloaded.get((leaf_event or {}).get("receipt_cid", ""))
        self.assertIsNotNone(leaf_receipt)
        leaf_decision = reloaded.get((leaf_receipt or {}).get("decision_cid", ""))
        self.assertIsNotNone(leaf_decision)
        leaf_intent = reloaded.get((leaf_decision or {}).get("intent_cid", ""))
        self.assertIsNotNone(leaf_intent)

        root_event = reloaded.get(root["event_cid"])
        self.assertIsNotNone(root_event)
        self.assertEqual((root_event or {}).get("parents"), [])


if __name__ == "__main__":
    unittest.main()
