#!/usr/bin/env python3
"""Deterministic tests for MCP++ UCAN delegation validation."""

from __future__ import annotations

import base64
import json
import os
import unittest
from unittest.mock import patch

import anyio

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ipfs_accelerate_py.mcp_server.mcplusplus.delegation import (
    HAVE_CRYPTO_ED25519,
    compute_delegation_proof_cid,
    compute_delegation_signature,
    compute_delegation_signature_ed25519,
    parse_delegation_chain,
    validate_delegation_chain,
    validate_raw_delegation_chain,
)
from ipfs_accelerate_py.mcp.server import create_mcp_server


_B58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def _base58btc_encode(raw: bytes) -> str:
    """Encode bytes into base58btc without multibase prefix."""
    data = bytes(raw)
    if not data:
        return ""

    zeros = 0
    for b in data:
        if b != 0:
            break
        zeros += 1

    value = int.from_bytes(data, "big")
    out = ""
    while value > 0:
        value, rem = divmod(value, 58)
        out = _B58_ALPHABET[rem] + out

    return ("1" * zeros) + (out or "")


class TestMCPServerMCPPlusPlusUCAN(unittest.TestCase):
    """Validate Profile C execution-time authorization checks."""

    def _create_unified_server(self, *, name: str):
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

        with patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper", return_value=_DummyServer()):
            with patch.dict(
                os.environ,
                {
                    "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                    "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                },
                clear=False,
            ):
                return create_mcp_server(name=name)

    def test_allows_valid_chain_for_leaf_actor_and_capability(self) -> None:
        raw_chain = [
            {
                "issuer": "did:user:alice",
                "audience": "did:model:planner",
                "capabilities": [{"resource": "*", "ability": "invoke"}],
            },
            {
                "issuer": "did:model:planner",
                "audience": "did:model:worker",
                "capabilities": [{"resource": "smoke.echo", "ability": "invoke"}],
            },
        ]
        result = validate_raw_delegation_chain(
            raw_chain=raw_chain,
            resource="smoke.echo",
            ability="invoke",
            actor="did:model:worker",
        )
        self.assertTrue(result.allowed)
        self.assertEqual(result.reason, "allowed")
        self.assertEqual(result.chain_length, 2)
        self.assertEqual(result.failure_hop, None)
        self.assertEqual(len(result.proof_lineage), 2)
        self.assertTrue(all(str(x).startswith("cidv1-sha256-") for x in result.proof_lineage))

    def test_allows_compact_token_envelope_with_att_claim(self) -> None:
        payload = {
            "iss": "did:user:alice",
            "aud": "did:model:worker",
            "exp": 4102444800,
            "att": {
                "smoke.echo": {
                    "invoke": [{}],
                }
            },
        }
        payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).decode("ascii").rstrip("=")
        token = "e30." + payload_b64 + ".sig"

        result = validate_raw_delegation_chain(
            raw_chain=[{"token": token}],
            resource="smoke.echo",
            ability="invoke",
            actor="did:model:worker",
        )
        self.assertTrue(result.allowed)
        self.assertEqual(result.reason, "allowed")

    def test_parses_token_prf_to_proof_cid(self) -> None:
        payload = {
            "iss": "did:user:alice",
            "aud": "did:model:worker",
            "exp": 4102444800,
            "prf": ["cid-proof-parent"],
            "capabilities": [{"with": "smoke.echo", "can": "invoke"}],
        }
        payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).decode("ascii").rstrip("=")
        token = "e30." + payload_b64 + ".sig"

        parsed = parse_delegation_chain([{"ucan": token}])
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0].proof_cid, "cid-proof-parent")
        self.assertEqual(parsed[0].issuer, "did:user:alice")
        self.assertEqual(parsed[0].audience, "did:model:worker")

    def test_parses_token_prf_string_to_proof_cid(self) -> None:
        payload = {
            "iss": "did:user:alice",
            "aud": "did:model:worker",
            "exp": 4102444800,
            "prf": "cid-proof-parent-string",
            "capabilities": [{"with": "smoke.echo", "can": "invoke"}],
        }
        payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).decode("ascii").rstrip("=")
        token = "e30." + payload_b64 + ".sig"

        parsed = parse_delegation_chain([{"jwt": token}])
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0].proof_cid, "cid-proof-parent-string")
        self.assertEqual(parsed[0].issuer, "did:user:alice")
        self.assertEqual(parsed[0].audience, "did:model:worker")

    def test_denies_capability_escalation(self) -> None:
        raw_chain = [
            {
                "issuer": "did:user:alice",
                "audience": "did:model:planner",
                "capabilities": [{"resource": "smoke.echo", "ability": "invoke"}],
            },
            {
                "issuer": "did:model:planner",
                "audience": "did:model:worker",
                "capabilities": [{"resource": "workflow.delete_workflow", "ability": "invoke"}],
            },
        ]
        result = validate_raw_delegation_chain(
            raw_chain=raw_chain,
            resource="workflow.delete_workflow",
            ability="invoke",
            actor="did:model:worker",
        )
        self.assertFalse(result.allowed)
        self.assertIn("capability_escalation", result.reason)

    def test_denies_expired_chain(self) -> None:
        chain = parse_delegation_chain(
            [
                {
                    "issuer": "did:user:alice",
                    "audience": "did:model:worker",
                    "capabilities": [{"resource": "*", "ability": "invoke"}],
                    "expiry": 1.0,
                }
            ]
        )
        result = validate_delegation_chain(
            chain=chain,
            resource="smoke.echo",
            ability="invoke",
            actor="did:model:worker",
            now=2.0,
        )
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, "expired_at_hop_0")

    def test_allows_valid_chain_with_required_signatures(self) -> None:
        chain = parse_delegation_chain(
            [
                {
                    "issuer": "did:user:alice",
                    "audience": "did:model:planner",
                    "capabilities": [{"resource": "*", "ability": "invoke"}],
                },
                {
                    "issuer": "did:model:planner",
                    "audience": "did:model:worker",
                    "capabilities": [{"resource": "smoke.echo", "ability": "invoke"}],
                },
            ]
        )

        # Attach deterministic proof/signature envelope material.
        issuer_keys = {
            "did:user:alice": "pk-alice",
            "did:model:planner": "pk-planner",
        }
        signed_raw = []
        for d in chain:
            proof_cid = compute_delegation_proof_cid(d)
            signature = compute_delegation_signature(delegation=d, issuer_key_hint=issuer_keys.get(d.issuer, ""))
            signed_raw.append(
                {
                    "issuer": d.issuer,
                    "audience": d.audience,
                    "capabilities": [{"resource": c.resource, "ability": c.ability} for c in d.capabilities],
                    "proof_cid": proof_cid,
                    "signature": signature,
                }
            )

        result = validate_raw_delegation_chain(
            raw_chain=signed_raw,
            resource="smoke.echo",
            ability="invoke",
            actor="did:model:worker",
            require_signatures=True,
            issuer_public_keys=issuer_keys,
        )
        self.assertTrue(result.allowed)
        self.assertEqual(result.reason, "allowed")

    def test_denies_revoked_proof_cid(self) -> None:
        chain = parse_delegation_chain(
            [
                {
                    "issuer": "did:user:alice",
                    "audience": "did:model:worker",
                    "capabilities": [{"resource": "smoke.echo", "ability": "invoke"}],
                }
            ]
        )
        proof_cid = compute_delegation_proof_cid(chain[0])
        signature = compute_delegation_signature(delegation=chain[0], issuer_key_hint="pk-alice")

        result = validate_raw_delegation_chain(
            raw_chain=[
                {
                    "issuer": "did:user:alice",
                    "audience": "did:model:worker",
                    "capabilities": [{"resource": "smoke.echo", "ability": "invoke"}],
                    "proof_cid": proof_cid,
                    "signature": signature,
                }
            ],
            resource="smoke.echo",
            ability="invoke",
            actor="did:model:worker",
            require_signatures=True,
            issuer_public_keys={"did:user:alice": "pk-alice"},
            revoked_proof_cids=[proof_cid],
        )
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, "revoked_proof_at_hop_0")
        self.assertEqual(result.failure_hop, 0)
        self.assertEqual(result.proof_lineage, [proof_cid])

    def test_dispatch_ucan_allow_includes_proof_lineage_telemetry(self) -> None:
        server = self._create_unified_server(name="ucan-allow-proof-lineage")

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
                    "__enforce_ucan": True,
                    "__ucan_actor": "did:model:worker",
                    "__ucan_proof_chain": [
                        {
                            "issuer": "did:user:alice",
                            "audience": "did:model:planner",
                            "capabilities": [{"resource": "*", "ability": "invoke"}],
                        },
                        {
                            "issuer": "did:model:planner",
                            "audience": "did:model:worker",
                            "capabilities": [{"resource": "smoke.echo", "ability": "invoke"}],
                        },
                    ],
                },
            )

            self.assertEqual(response.get("echo"), "ok")
            authorization = response.get("authorization") or {}
            self.assertEqual(authorization.get("scheme"), "ucan")
            self.assertTrue(authorization.get("allowed"))
            self.assertEqual(authorization.get("reason"), "allowed")
            self.assertEqual(authorization.get("failure_hop"), None)
            self.assertEqual(len(authorization.get("proof_lineage") or []), 2)
            self.assertTrue(all(str(x).startswith("cidv1-sha256-") for x in (authorization.get("proof_lineage") or [])))

        anyio.run(_run_flow)

    def test_dispatch_ucan_deny_includes_proof_lineage_telemetry(self) -> None:
        server = self._create_unified_server(name="ucan-deny-proof-lineage")

        async def _run_flow() -> None:
            async def echo(value: str):
                return {"echo": value}

            server._unified_tool_manager.register_tool("smoke", "echo", echo, description="echo smoke")
            dispatch = server.tools["tools_dispatch"]["function"]

            delegation = parse_delegation_chain(
                [
                    {
                        "issuer": "did:user:alice",
                        "audience": "did:model:worker",
                        "capabilities": [{"resource": "smoke.echo", "ability": "invoke"}],
                    }
                ]
            )[0]
            proof_cid = compute_delegation_proof_cid(delegation)
            signature = compute_delegation_signature(delegation=delegation, issuer_key_hint="pk-alice")

            response = await dispatch(
                "smoke",
                "echo",
                {
                    "value": "ok",
                    "__enforce_ucan": True,
                    "__ucan_actor": "did:model:worker",
                    "__ucan_proof_chain": [
                        {
                            "issuer": delegation.issuer,
                            "audience": delegation.audience,
                            "capabilities": [{"resource": "smoke.echo", "ability": "invoke"}],
                            "proof_cid": proof_cid,
                            "signature": signature,
                        }
                    ],
                    "__ucan_require_signatures": True,
                    "__ucan_issuer_public_keys": {"did:user:alice": "pk-alice"},
                    "__ucan_revoked_proof_cids": [proof_cid],
                },
            )

            self.assertFalse(response.get("ok"))
            authorization = response.get("authorization") or {}
            self.assertEqual(authorization.get("scheme"), "ucan")
            self.assertFalse(authorization.get("allowed"))
            self.assertEqual(authorization.get("reason"), "revoked_proof_at_hop_0")
            self.assertEqual(authorization.get("failure_hop"), 0)
            self.assertEqual(authorization.get("proof_lineage"), [proof_cid])

        anyio.run(_run_flow)

    def test_denies_caveat_mismatch(self) -> None:
        result = validate_raw_delegation_chain(
            raw_chain=[
                {
                    "issuer": "did:user:alice",
                    "audience": "did:model:worker",
                    "capabilities": [{"resource": "smoke.echo", "ability": "invoke"}],
                    "caveats": [{"resource_prefix": "workflow."}],
                }
            ],
            resource="smoke.echo",
            ability="invoke",
            actor="did:model:worker",
        )
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, "caveat_denied_at_hop_0")

    def test_allows_caveat_with_actor_and_context_constraints(self) -> None:
        result = validate_raw_delegation_chain(
            raw_chain=[
                {
                    "issuer": "did:user:alice",
                    "audience": "did:model:worker",
                    "capabilities": [{"resource": "smoke.echo", "ability": "invoke"}],
                    "caveats": [
                        {
                            "actor_equals": "did:model:worker",
                            "context_cids_all": ["cid-ctx-a", "cid-ctx-b"],
                        }
                    ],
                }
            ],
            resource="smoke.echo",
            ability="invoke",
            actor="did:model:worker",
            context_cids=["cid-ctx-a", "cid-ctx-b", "cid-ctx-extra"],
        )
        self.assertTrue(result.allowed)
        self.assertEqual(result.reason, "allowed")

    def test_denies_caveat_when_context_missing(self) -> None:
        result = validate_raw_delegation_chain(
            raw_chain=[
                {
                    "issuer": "did:user:alice",
                    "audience": "did:model:worker",
                    "capabilities": [{"resource": "smoke.echo", "ability": "invoke"}],
                    "caveats": [
                        {
                            "actor_equals": "did:model:worker",
                            "context_cids_all": ["cid-required"],
                        }
                    ],
                }
            ],
            resource="smoke.echo",
            ability="invoke",
            actor="did:model:worker",
            context_cids=["cid-other"],
        )
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, "caveat_denied_at_hop_0")

    def test_allows_caveat_with_regex_and_ability_set(self) -> None:
        result = validate_raw_delegation_chain(
            raw_chain=[
                {
                    "issuer": "did:user:alice",
                    "audience": "did:model:worker",
                    "capabilities": [{"resource": "smoke.echo", "ability": "invoke"}],
                    "caveats": [
                        {
                            "resource_regex": r"smoke\.[a-z_]+",
                            "ability_in": ["invoke", "read"],
                        }
                    ],
                }
            ],
            resource="smoke.echo",
            ability="invoke",
            actor="did:model:worker",
        )
        self.assertTrue(result.allowed)
        self.assertEqual(result.reason, "allowed")

    def test_allows_caveat_with_actor_in_and_context_any(self) -> None:
        result = validate_raw_delegation_chain(
            raw_chain=[
                {
                    "issuer": "did:user:alice",
                    "audience": "did:model:worker",
                    "capabilities": [{"resource": "smoke.echo", "ability": "invoke"}],
                    "caveats": [
                        {
                            "actor_in": ["did:model:worker", "did:model:fallback"],
                            "context_cids_any": ["cid-a", "cid-b"],
                        }
                    ],
                }
            ],
            resource="smoke.echo",
            ability="invoke",
            actor="did:model:worker",
            context_cids=["cid-b", "cid-extra"],
        )
        self.assertTrue(result.allowed)
        self.assertEqual(result.reason, "allowed")

    def test_denies_caveat_with_actor_regex_and_context_none(self) -> None:
        result = validate_raw_delegation_chain(
            raw_chain=[
                {
                    "issuer": "did:user:alice",
                    "audience": "did:model:worker",
                    "capabilities": [{"resource": "smoke.echo", "ability": "invoke"}],
                    "caveats": [
                        {
                            "actor_regex": r"did:model:[a-z]+",
                            "context_cids_none": ["cid-blocked"],
                        }
                    ],
                }
            ],
            resource="smoke.echo",
            ability="invoke",
            actor="did:model:worker",
            context_cids=["cid-blocked"],
        )
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, "caveat_denied_at_hop_0")

    @unittest.skipUnless(HAVE_CRYPTO_ED25519, "cryptography ed25519 unavailable")
    def test_allows_valid_chain_with_ed25519_signatures(self) -> None:
        chain = parse_delegation_chain(
            [
                {
                    "issuer": "did:user:alice",
                    "audience": "did:model:worker",
                    "capabilities": [{"resource": "smoke.echo", "ability": "invoke"}],
                }
            ]
        )

        private = Ed25519PrivateKey.generate()
        public = private.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        private_bytes = private.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )

        import base64

        private_b64 = base64.urlsafe_b64encode(private_bytes).decode("ascii").rstrip("=")
        public_b64 = base64.urlsafe_b64encode(public).decode("ascii").rstrip("=")

        delegation = chain[0]
        proof_cid = compute_delegation_proof_cid(delegation)
        signature = compute_delegation_signature_ed25519(
            delegation=delegation,
            private_key_b64=private_b64,
        )

        result = validate_raw_delegation_chain(
            raw_chain=[
                {
                    "issuer": delegation.issuer,
                    "audience": delegation.audience,
                    "capabilities": [{"resource": "smoke.echo", "ability": "invoke"}],
                    "proof_cid": proof_cid,
                    "signature": signature,
                }
            ],
            resource="smoke.echo",
            ability="invoke",
            actor="did:model:worker",
            require_signatures=True,
            issuer_public_keys={"did:user:alice": {"alg": "ed25519", "public_key": public_b64}},
        )
        self.assertTrue(result.allowed)
        self.assertEqual(result.reason, "allowed")

    @unittest.skipUnless(HAVE_CRYPTO_ED25519, "cryptography ed25519 unavailable")
    def test_denies_invalid_ed25519_signature(self) -> None:
        chain = parse_delegation_chain(
            [
                {
                    "issuer": "did:user:alice",
                    "audience": "did:model:worker",
                    "capabilities": [{"resource": "smoke.echo", "ability": "invoke"}],
                }
            ]
        )

        private = Ed25519PrivateKey.generate()
        public = private.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        private_bytes = private.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )

        import base64

        private_b64 = base64.urlsafe_b64encode(private_bytes).decode("ascii").rstrip("=")
        public_b64 = base64.urlsafe_b64encode(public).decode("ascii").rstrip("=")

        delegation = chain[0]
        proof_cid = compute_delegation_proof_cid(delegation)
        signature = compute_delegation_signature_ed25519(
            delegation=delegation,
            private_key_b64=private_b64,
        )

        bad_signature = "ed25519:" + ("A" * 86)

        result = validate_raw_delegation_chain(
            raw_chain=[
                {
                    "issuer": delegation.issuer,
                    "audience": delegation.audience,
                    "capabilities": [{"resource": "smoke.echo", "ability": "invoke"}],
                    "proof_cid": proof_cid,
                    "signature": bad_signature,
                }
            ],
            resource="smoke.echo",
            ability="invoke",
            actor="did:model:worker",
            require_signatures=True,
            issuer_public_keys={"did:user:alice": public_b64},
        )
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, "invalid_signature_at_hop_0")

    @unittest.skipUnless(HAVE_CRYPTO_ED25519, "cryptography ed25519 unavailable")
    def test_allows_valid_chain_with_did_key_public_key(self) -> None:
        chain = parse_delegation_chain(
            [
                {
                    "issuer": "did:user:alice",
                    "audience": "did:model:worker",
                    "capabilities": [{"resource": "smoke.echo", "ability": "invoke"}],
                }
            ]
        )

        private = Ed25519PrivateKey.generate()
        public = private.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        private_bytes = private.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )

        import base64

        private_b64 = base64.urlsafe_b64encode(private_bytes).decode("ascii").rstrip("=")
        did_key = "did:key:z" + _base58btc_encode(bytes([0xED, 0x01]) + public)

        delegation = chain[0]
        proof_cid = compute_delegation_proof_cid(delegation)
        signature = compute_delegation_signature_ed25519(
            delegation=delegation,
            private_key_b64=private_b64,
        )

        result = validate_raw_delegation_chain(
            raw_chain=[
                {
                    "issuer": delegation.issuer,
                    "audience": delegation.audience,
                    "capabilities": [{"resource": "smoke.echo", "ability": "invoke"}],
                    "proof_cid": proof_cid,
                    "signature": signature,
                }
            ],
            resource="smoke.echo",
            ability="invoke",
            actor="did:model:worker",
            require_signatures=True,
            issuer_public_keys={"did:user:alice": did_key},
        )
        self.assertTrue(result.allowed)
        self.assertEqual(result.reason, "allowed")

    @unittest.skipUnless(HAVE_CRYPTO_ED25519, "cryptography ed25519 unavailable")
    def test_allows_valid_chain_with_ed25519_hex_signature(self) -> None:
        chain = parse_delegation_chain(
            [
                {
                    "issuer": "did:user:alice",
                    "audience": "did:model:worker",
                    "capabilities": [{"resource": "smoke.echo", "ability": "invoke"}],
                }
            ]
        )

        private = Ed25519PrivateKey.generate()
        public = private.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        private_bytes = private.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )

        import base64

        private_b64 = base64.urlsafe_b64encode(private_bytes).decode("ascii").rstrip("=")
        public_b64 = base64.urlsafe_b64encode(public).decode("ascii").rstrip("=")

        delegation = chain[0]
        proof_cid = compute_delegation_proof_cid(delegation)
        signature = compute_delegation_signature_ed25519(
            delegation=delegation,
            private_key_b64=private_b64,
        )
        sig_token = signature.split(":", 1)[1]
        sig_padding = "=" * ((4 - (len(sig_token) % 4)) % 4)
        signature_hex = "ed25519-hex:" + base64.urlsafe_b64decode(sig_token + sig_padding).hex()

        result = validate_raw_delegation_chain(
            raw_chain=[
                {
                    "issuer": delegation.issuer,
                    "audience": delegation.audience,
                    "capabilities": [{"resource": "smoke.echo", "ability": "invoke"}],
                    "proof_cid": proof_cid,
                    "signature": signature_hex,
                }
            ],
            resource="smoke.echo",
            ability="invoke",
            actor="did:model:worker",
            require_signatures=True,
            issuer_public_keys={"did:user:alice": public_b64},
        )
        self.assertTrue(result.allowed)
        self.assertEqual(result.reason, "allowed")

    @unittest.skipUnless(HAVE_CRYPTO_ED25519, "cryptography ed25519 unavailable")
    def test_allows_valid_chain_with_raw_hex_signature(self) -> None:
        chain = parse_delegation_chain(
            [
                {
                    "issuer": "did:user:alice",
                    "audience": "did:model:worker",
                    "capabilities": [{"resource": "smoke.echo", "ability": "invoke"}],
                }
            ]
        )

        private = Ed25519PrivateKey.generate()
        public = private.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        private_bytes = private.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )

        private_b64 = base64.urlsafe_b64encode(private_bytes).decode("ascii").rstrip("=")
        public_b64 = base64.urlsafe_b64encode(public).decode("ascii").rstrip("=")

        delegation = chain[0]
        proof_cid = compute_delegation_proof_cid(delegation)
        signature = compute_delegation_signature_ed25519(
            delegation=delegation,
            private_key_b64=private_b64,
        )
        sig_token = signature.split(":", 1)[1]
        sig_padding = "=" * ((4 - (len(sig_token) % 4)) % 4)
        signature_raw_hex = base64.urlsafe_b64decode(sig_token + sig_padding).hex()

        result = validate_raw_delegation_chain(
            raw_chain=[
                {
                    "issuer": delegation.issuer,
                    "audience": delegation.audience,
                    "capabilities": [{"resource": "smoke.echo", "ability": "invoke"}],
                    "proof_cid": proof_cid,
                    "signature": signature_raw_hex,
                }
            ],
            resource="smoke.echo",
            ability="invoke",
            actor="did:model:worker",
            require_signatures=True,
            issuer_public_keys={"did:user:alice": public_b64},
        )
        self.assertTrue(result.allowed)
        self.assertEqual(result.reason, "allowed")

    def test_dispatch_combined_ucan_and_policy_denial_prefers_ucan(self) -> None:
        server = self._create_unified_server(name="ucan-policy-combined-deny")

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
                    "__enforce_ucan": True,
                    "__ucan_actor": "did:model:worker",
                    "__ucan_proof_chain": [],
                    "__enforce_policy": True,
                    "__policy_actor": "did:model:worker",
                    "__policy_clauses": [
                        {
                            "clause_type": "prohibition",
                            "actor": "did:model:worker",
                            "action": "smoke.echo",
                        }
                    ],
                },
            )

            self.assertFalse(response.get("ok"))
            self.assertEqual(response.get("error"), "authorization_denied")
            self.assertEqual(((response.get("authorization") or {}).get("scheme")), "ucan")
            self.assertEqual(((response.get("authorization") or {}).get("reason")), "missing_delegation_chain")
            self.assertNotIn("policy", response)
            self.assertNotIn("policy_decision", response)

        anyio.run(_run_flow)

    def test_dispatch_combined_ucan_denial_overrides_policy_allow(self) -> None:
        server = self._create_unified_server(name="ucan-policy-ucan-deny-policy-allow")

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
                    "__enforce_ucan": True,
                    "__ucan_actor": "did:model:worker",
                    "__ucan_proof_chain": [],
                    "__enforce_policy": True,
                    "__policy_actor": "did:model:worker",
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
                        },
                    ],
                },
            )

            self.assertFalse(response.get("ok"))
            self.assertEqual(response.get("error"), "authorization_denied")
            self.assertEqual(((response.get("authorization") or {}).get("scheme")), "ucan")
            self.assertEqual(((response.get("authorization") or {}).get("reason")), "missing_delegation_chain")
            self.assertNotIn("policy", response)
            self.assertNotIn("policy_decision", response)

        anyio.run(_run_flow)

    def test_dispatch_combined_policy_denial_applies_after_valid_ucan(self) -> None:
        server = self._create_unified_server(name="ucan-policy-ucan-allow-policy-deny")

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
                    "__enforce_ucan": True,
                    "__ucan_actor": "did:model:worker",
                    "__ucan_proof_chain": [
                        {
                            "issuer": "did:user:alice",
                            "audience": "did:model:planner",
                            "capabilities": [{"resource": "*", "ability": "invoke"}],
                        },
                        {
                            "issuer": "did:model:planner",
                            "audience": "did:model:worker",
                            "capabilities": [{"resource": "smoke.echo", "ability": "invoke"}],
                        },
                    ],
                    "__enforce_policy": True,
                    "__policy_actor": "did:model:worker",
                    "__policy_clauses": [
                        {
                            "clause_type": "prohibition",
                            "actor": "did:model:worker",
                            "action": "smoke.echo",
                        }
                    ],
                },
            )

            self.assertFalse(response.get("ok"))
            self.assertEqual(response.get("error"), "policy_denied")
            self.assertEqual(((response.get("policy") or {}).get("decision")), "deny")
            self.assertNotIn("authorization", response)

            decision_cid = str(((response.get("policy_decision") or {}).get("decision_cid")) or "")
            self.assertTrue(decision_cid.startswith("cidv1-sha256-"))
            stored = server._unified_artifact_store.get(decision_cid) or {}
            self.assertEqual(stored.get("decision"), "deny")

        anyio.run(_run_flow)


if __name__ == "__main__":
    unittest.main()
