#!/usr/bin/env python3
"""Deterministic tests for CID-native execution artifacts."""

from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
