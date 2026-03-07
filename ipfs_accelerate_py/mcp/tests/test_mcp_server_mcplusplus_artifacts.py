#!/usr/bin/env python3
"""Deterministic tests for CID-native execution artifacts."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

import anyio

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
from ipfs_accelerate_py.mcp.server import create_mcp_server


class _DummyServer:
    def __init__(self) -> None:
        self.tools = {}
        self.mcp = None

    def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
        self.tools[name] = {
            "function": function,
            "description": description,
            "input_schema": input_schema,
            "execution_context": execution_context,
            "tags": tags,
        }


class TestMCPServerMCPPlusPlusArtifacts(unittest.TestCase):
    """Validate Profile B helper determinism and chain integrity."""

    def _create_unified_server(self, *, name: str, enable_cid_artifacts: bool = False, extra_env: dict[str, str] | None = None):
        env = {
            "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
            "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
        }
        if enable_cid_artifacts:
            env["IPFS_MCP_SERVER_ENABLE_CID_ARTIFACTS"] = "1"
        env.update({str(k): str(v) for k, v in (extra_env or {}).items()})

        with patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper", return_value=_DummyServer()):
            with patch.dict(os.environ, env, clear=False):
                return create_mcp_server(name=name)

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

    def test_artifact_store_get_returns_deep_copy_for_nested_payloads(self) -> None:
        store = ArtifactStore()
        store.put("cid-nested", {"meta": {"level": {"value": 1}}, "tags": ["a", "b"]})

        payload = store.get("cid-nested") or {}
        self.assertEqual(((payload.get("meta") or {}).get("level") or {}).get("value"), 1)

        payload["meta"]["level"]["value"] = 99
        payload["tags"].append("mutated")

        reloaded = store.get("cid-nested") or {}
        self.assertEqual((((reloaded.get("meta") or {}).get("level") or {}).get("value")), 1)
        self.assertEqual(reloaded.get("tags"), ["a", "b"])

    def test_artifact_store_export_records_returns_deep_copy(self) -> None:
        store = ArtifactStore()
        store.put_many(
            {
                "cid-a": {"meta": {"depth": 1}},
                "cid-b": {"items": [{"x": 1}]},
            }
        )

        exported = store.export_records()
        exported["cid-a"]["meta"]["depth"] = 7
        exported["cid-b"]["items"][0]["x"] = 9

        again = store.export_records()
        self.assertEqual(((again.get("cid-a") or {}).get("meta") or {}).get("depth"), 1)
        self.assertEqual((((again.get("cid-b") or {}).get("items") or [{}])[0]).get("x"), 1)

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

    def test_dispatch_artifact_emission_opt_in_persists_chain(self) -> None:
        server = self._create_unified_server(name="artifacts-opt-in")

        async def _run_flow() -> None:
            async def echo(value: str):
                return {"echo": value}

            server._unified_tool_manager.register_tool("smoke", "echo", echo, description="echo smoke")
            dispatch = server.tools["tools_dispatch"]["function"]

            response = await dispatch(
                "smoke",
                "echo",
                {
                    "value": "ok",
                    "__emit_artifacts": True,
                    "__correlation_id": "corr-203",
                    "__proof_cid": "cid-proof",
                    "__policy_cid": "cid-policy",
                    "__parent_event_cids": ["cid-parent-203"],
                },
            )

            self.assertTrue(response.get("ok"))
            self.assertEqual((response.get("result") or {}).get("echo"), "ok")

            artifacts = response.get("artifacts") or {}
            for key in ["input_cid", "intent_cid", "decision_cid", "output_cid", "receipt_cid", "event_cid"]:
                self.assertTrue(str(artifacts.get(key) or "").startswith("cidv1-sha256-"))

            payloads = response.get("artifact_payloads") or {}
            self.assertEqual(((payloads.get("intent") or {}).get("correlation_id")), "corr-203")
            self.assertEqual(((payloads.get("decision") or {}).get("policy_cid")), "cid-policy")
            self.assertEqual(((payloads.get("event") or {}).get("parents")), ["cid-parent-203"])

            artifact_store = response.get("artifact_store") or {}
            self.assertTrue(artifact_store.get("persisted"))
            self.assertEqual(int(artifact_store.get("written") or 0), 6)

            stored_event = server._unified_artifact_store.get(artifacts.get("event_cid", "")) or {}
            self.assertEqual(stored_event.get("intent_cid"), artifacts.get("intent_cid"))

        anyio.run(_run_flow)

    def test_dispatch_artifact_policy_is_deterministic_across_emit_modes(self) -> None:
        server = self._create_unified_server(name="artifacts-policy-modes")

        async def _run_flow() -> None:
            async def echo(value: str):
                return {"echo": value}

            server._unified_tool_manager.register_tool("smoke", "echo", echo, description="echo smoke")
            dispatch = server.tools["tools_dispatch"]["function"]

            base_payload = {
                "value": "ok",
                "__enforce_policy": True,
                "__policy_actor": "did:model:worker",
                "__policy_version": "v1",
                "__policy_clauses": [
                    {
                        "clause_type": "permission",
                        "actor": "did:model:worker",
                        "action": "smoke.echo",
                    },
                    {
                        "clause_type": "obligation",
                        "actor": "did:model:worker",
                        "action": "smoke.echo",
                        "obligation_deadline": "2030-01-01T00:00:00Z",
                        "metadata": {"migration": "phase-b"},
                    },
                ],
            }

            response_no_emit = await dispatch("smoke", "echo", dict(base_payload, **{"__emit_artifacts": False}))
            response_emit = await dispatch("smoke", "echo", dict(base_payload, **{"__emit_artifacts": True}))

            cid_no_emit = str((response_no_emit.get("policy_decision") or {}).get("decision_cid") or "")
            cid_emit = str((response_emit.get("policy_decision") or {}).get("decision_cid") or "")
            self.assertTrue(cid_no_emit.startswith("cidv1-sha256-"))
            self.assertEqual(cid_no_emit, cid_emit)

            stored_v1 = server._unified_artifact_store.get(cid_no_emit) or {}
            self.assertEqual(stored_v1.get("policy_version"), "v1")
            self.assertEqual(stored_v1.get("decision"), "allow_with_obligations")

            response_v2 = await dispatch(
                "smoke",
                "echo",
                dict(base_payload, **{"__policy_version": "v2", "__emit_artifacts": False}),
            )
            cid_v2 = str((response_v2.get("policy_decision") or {}).get("decision_cid") or "")
            self.assertTrue(cid_v2.startswith("cidv1-sha256-"))
            self.assertNotEqual(cid_no_emit, cid_v2)

            stored_v2 = server._unified_artifact_store.get(cid_v2) or {}
            self.assertEqual(stored_v2.get("policy_version"), "v2")

        anyio.run(_run_flow)

    def test_dispatch_artifact_default_policy_from_config(self) -> None:
        server = self._create_unified_server(name="artifacts-default-policy", enable_cid_artifacts=True)

        async def _run_flow() -> None:
            async def echo(value: str):
                return {"echo": value}

            server._unified_tool_manager.register_tool("smoke", "echo", echo, description="echo smoke")
            dispatch = server.tools["tools_dispatch"]["function"]
            response = await dispatch("smoke", "echo", {"value": "ok", "__correlation_id": "corr-default"})

            self.assertTrue(response.get("ok"))
            self.assertIn("artifacts", response)
            self.assertIn("artifact_payloads", response)
            self.assertTrue(((response.get("artifact_store") or {}).get("persisted")))

        anyio.run(_run_flow)

    def test_dispatch_artifact_json_backend_persists_and_reloads_chain(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = f"{tmpdir}/cid-artifacts.json"
            server = self._create_unified_server(
                name="artifacts-json-backend",
                enable_cid_artifacts=True,
                extra_env={
                    "IPFS_MCP_SERVER_ARTIFACT_STORE_BACKEND": "json",
                    "IPFS_MCP_SERVER_ARTIFACT_STORE_PATH": artifact_path,
                },
            )

            async def _run_flow() -> None:
                async def echo(value: str):
                    return {"echo": value}

                server._unified_tool_manager.register_tool("smoke", "echo", echo, description="echo smoke")
                dispatch = server.tools["tools_dispatch"]["function"]
                response = await dispatch("smoke", "echo", {"value": "ok", "__correlation_id": "corr-json"})

                self.assertTrue(response.get("ok"))
                artifact_store = response.get("artifact_store") or {}
                self.assertTrue(artifact_store.get("persisted"))
                self.assertEqual(artifact_store.get("backend"), "json")
                self.assertEqual(artifact_store.get("path"), artifact_path)
                self.assertTrue(artifact_store.get("durable"))
                self.assertGreaterEqual(int(artifact_store.get("saved") or 0), 6)

                event_cid = str(((response.get("artifacts") or {}).get("event_cid")) or "")
                self.assertTrue(event_cid.startswith("cidv1-sha256-"))

                reloaded = ArtifactStore.load_json(artifact_path)
                stored_event = reloaded.get(event_cid) or {}
                self.assertEqual(stored_event.get("receipt_cid"), (response.get("artifacts") or {}).get("receipt_cid"))

                rehydrated_server = self._create_unified_server(
                    name="artifacts-json-backend-reloaded",
                    enable_cid_artifacts=True,
                    extra_env={
                        "IPFS_MCP_SERVER_ARTIFACT_STORE_BACKEND": "json",
                        "IPFS_MCP_SERVER_ARTIFACT_STORE_PATH": artifact_path,
                    },
                )
                self.assertIsNotNone(rehydrated_server._unified_artifact_store.get(event_cid))
                self.assertEqual((rehydrated_server._unified_artifact_store_meta or {}).get("backend"), "json")

            anyio.run(_run_flow)


if __name__ == "__main__":
    unittest.main()
