#!/usr/bin/env python3
"""Deterministic tests for MCP++ UCAN delegation validation."""

from __future__ import annotations

import unittest

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


class TestMCPServerMCPPlusPlusUCAN(unittest.TestCase):
    """Validate Profile C execution-time authorization checks."""

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


if __name__ == "__main__":
    unittest.main()
