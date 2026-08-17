#!/usr/bin/env python3
"""MCPP-034: emit and verify ExecutionEnvelope@1 from the accelerate runtime.

Acceptance: create an envelope, compute its CID with mcpp-jcs-v1, and verify it.
No large spec reimplementation — the runtime adapter reuses shared JCS + validators.
"""

from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

from ipfs_accelerate_py.mcp_server.mcplusplus.envelope import (
    CANONICALIZATION,
    INTERFACE,
    SCHEMA_ENVELOPE,
    EnvelopeError,
    EnvelopeStore,
    RuntimeEnvelopeAdapter,
    canonicalize_envelope,
    compute_envelope_cid,
    create_envelope,
    emit_envelope,
    envelope_identity,
    verify_envelope,
)

# Shared mcpp-jcs-v1 for independent cross-check (not a reimplementation).
_TESTS_PY = (
    Path(__file__).resolve().parents[2] / "mcplusplus" / "tests-py"
)
if _TESTS_PY.is_dir() and str(_TESTS_PY) not in sys.path:
    sys.path.insert(0, str(_TESTS_PY))

from validators.canonical_jcs import (  # noqa: E402
    ALGORITHM_ID,
    artifact_cid,
    canonicalize_bytes,
    identity,
)


# Stable CIDs matching the four-language envelope suite (valid CIDv1 form).
CID_A = "bafkreigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
CID_B = "bafkreihtwdlu4jntm7yl2mgsfzqgr4on37vr7inuld2dql2p4rmqafybti"
CID_C = "bafkreicssskybdf32rmzlbtge5bxyv4v6c6eac322pbrsr3azlb4fkxiqi"
CID_D = "bafkreihdwdcefgh4dqkjv67uzcmw7ojee6xedzdetojuzjevtenxquvyku"
DID_REQUESTER = "did:key:z6MkrequesterExample0001"


def _minimal_kwargs(**overrides):
    base = {
        "interface_cid": CID_A,
        "input_cid": CID_B,
        "intent_cid": CID_C,
        "requester_did": DID_REQUESTER,
        "correlation_id": "task-mcpp-034",
        "method": "repo.status",
        "policy_cid": CID_D,
        "proof_cids": [CID_D],
        "proof_cid": CID_D,
        "constraints": {"timeout_ms": 30000, "max_retries": 3},
        "created_at_ms": 1783872000000,
        "parents": [],
    }
    base.update(overrides)
    return base


class TestMcplusplusRuntimeEnvelope(unittest.TestCase):
    """Runtime path: create → mcpp-jcs-v1 CID → verify (MCPP-034)."""

    def test_interface_markers(self) -> None:
        self.assertEqual(INTERFACE, "RuntimeEnvelopeAdapter@1")
        self.assertEqual(SCHEMA_ENVELOPE, "mcp++/execution/envelope@1")
        self.assertEqual(CANONICALIZATION, "mcpp-jcs-v1")
        self.assertEqual(ALGORITHM_ID, "mcpp-jcs-v1")

    def test_create_envelope_has_required_fields(self) -> None:
        envelope = create_envelope(**_minimal_kwargs())
        self.assertEqual(envelope["schema"], SCHEMA_ENVELOPE)
        self.assertEqual(envelope["canonicalization"], CANONICALIZATION)
        self.assertEqual(envelope["interface_cid"], CID_A)
        self.assertEqual(envelope["input_cid"], CID_B)
        self.assertEqual(envelope["intent_cid"], CID_C)
        self.assertEqual(envelope["correlation_id"], "task-mcpp-034")
        self.assertEqual(envelope["requester"]["did"], DID_REQUESTER)
        self.assertEqual(envelope["authority"]["proof_cids"], [CID_D])
        self.assertEqual(envelope["authority"]["proof_cid"], CID_D)
        self.assertEqual(envelope["parents"], [])
        self.assertEqual(envelope["created_at_ms"], 1783872000000)
        self.assertEqual(envelope["method"], "repo.status")
        self.assertEqual(envelope["state_refs"], [])

    def test_compute_cid_uses_mcpp_jcs_v1(self) -> None:
        envelope = create_envelope(**_minimal_kwargs())
        cid = compute_envelope_cid(envelope)
        self.assertTrue(cid.startswith("b"))
        self.assertGreaterEqual(len(cid), 59)

        # Independent recomputation via shared JCS must match exactly.
        shared_cid = artifact_cid(envelope)
        self.assertEqual(cid, shared_cid)
        self.assertEqual(canonicalize_envelope(envelope), canonicalize_bytes(envelope))

        ident = envelope_identity(envelope)
        self.assertEqual(ident["algorithm"], "mcpp-jcs-v1")
        self.assertEqual(ident["cid"], cid)
        self.assertEqual(identity(envelope).cid, cid)

    def test_verify_envelope_accepts_valid_mint(self) -> None:
        envelope = create_envelope(**_minimal_kwargs())
        cid = compute_envelope_cid(envelope)
        result = verify_envelope(envelope, expected_cid=cid)
        self.assertTrue(result.ok, msg=result.errors)
        self.assertEqual(result.cid, cid)
        self.assertEqual(result.algorithm, CANONICALIZATION)
        self.assertEqual(result.errors, [])

    def test_verify_envelope_rejects_tamper(self) -> None:
        envelope = create_envelope(**_minimal_kwargs())
        cid = compute_envelope_cid(envelope)
        tampered = copy.deepcopy(envelope)
        tampered["correlation_id"] = "tampered-correlation"
        result = verify_envelope(tampered, expected_cid=cid)
        self.assertFalse(result.ok)
        self.assertTrue(any("cid_mismatch" in e for e in result.errors))

    def test_verify_envelope_rejects_missing_required(self) -> None:
        envelope = create_envelope(**_minimal_kwargs())
        del envelope["intent_cid"]
        result = verify_envelope(envelope)
        self.assertFalse(result.ok)
        self.assertTrue(any("intent_cid" in e for e in result.errors))

    def test_emit_envelope_binds_cid(self) -> None:
        emitted = emit_envelope(**_minimal_kwargs())
        self.assertEqual(emitted.algorithm, CANONICALIZATION)
        self.assertEqual(emitted.cid, compute_envelope_cid(emitted.envelope))
        self.assertEqual(emitted.canonical_bytes, canonicalize_envelope(emitted.envelope))
        result = verify_envelope(emitted.envelope, expected_cid=emitted.cid)
        self.assertTrue(result.ok, msg=result.errors)

    def test_store_persist_and_verify_by_cid(self) -> None:
        store = EnvelopeStore()
        envelope = create_envelope(**_minimal_kwargs())
        cid = store.put_envelope(envelope)
        self.assertEqual(cid, compute_envelope_cid(envelope))
        self.assertTrue(store.has(cid))
        loaded = store.get(cid)
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded["correlation_id"], "task-mcpp-034")

        # Caller mutation must not affect store.
        loaded["correlation_id"] = "mutated"
        again = store.get(cid)
        assert again is not None
        self.assertEqual(again["correlation_id"], "task-mcpp-034")

        verified = store.verify_stored(cid)
        self.assertTrue(verified.ok, msg=verified.errors)
        self.assertEqual(verified.cid, cid)
        self.assertEqual(store.stats().get("envelope_count"), 1)

    def test_store_rejects_cid_mismatch_on_put(self) -> None:
        store = EnvelopeStore()
        envelope = create_envelope(**_minimal_kwargs())
        with self.assertRaises(EnvelopeError) as ctx:
            store.put(CID_A, envelope)
        self.assertEqual(ctx.exception.reason_code, "cid_mismatch")

    def test_store_json_round_trip(self) -> None:
        store = EnvelopeStore()
        envelope = create_envelope(**_minimal_kwargs(correlation_id="json-rt"))
        cid = store.put_envelope(envelope)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/envelopes.json"
            written = store.save_json(path)
            self.assertEqual(written, 1)
            reloaded = EnvelopeStore.load_json(path)
            self.assertEqual(reloaded.stats().get("envelope_count"), 1)
            self.assertEqual(reloaded.export_records(), store.export_records())
            verified = reloaded.verify_stored(cid)
            self.assertTrue(verified.ok, msg=verified.errors)

    def test_runtime_adapter_create_persist_verify(self) -> None:
        """Primary acceptance path: create → CID (mcpp-jcs-v1) → verify."""
        adapter = RuntimeEnvelopeAdapter()
        emitted = adapter.emit_and_persist(**_minimal_kwargs(correlation_id="adapter-path"))
        self.assertEqual(emitted.algorithm, "mcpp-jcs-v1")
        self.assertEqual(adapter.compute_cid(emitted.envelope), emitted.cid)

        # Shared JCS identity must agree with the runtime adapter.
        self.assertEqual(artifact_cid(emitted.envelope), emitted.cid)

        result = adapter.verify(emitted.envelope, expected_cid=emitted.cid)
        self.assertTrue(result.ok, msg=result.errors)

        stored = adapter.verify_stored(emitted.cid)
        self.assertTrue(stored.ok, msg=stored.errors)
        loaded = adapter.load(emitted.cid)
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded["schema"], SCHEMA_ENVELOPE)
        self.assertEqual(loaded["canonicalization"], "mcpp-jcs-v1")

    def test_create_rejects_invalid_did(self) -> None:
        with self.assertRaises(EnvelopeError) as ctx:
            create_envelope(**_minimal_kwargs(requester_did="not-a-did"))
        self.assertEqual(ctx.exception.reason_code, "invalid_did")

    def test_create_rejects_invalid_interface_cid(self) -> None:
        with self.assertRaises(EnvelopeError) as ctx:
            create_envelope(**_minimal_kwargs(interface_cid="cidv1-sha256-deadbeef"))
        self.assertEqual(ctx.exception.reason_code, "invalid_cid")

    def test_deterministic_cid_for_identical_envelopes(self) -> None:
        a = create_envelope(**_minimal_kwargs())
        b = create_envelope(**_minimal_kwargs())
        self.assertEqual(a, b)
        self.assertEqual(compute_envelope_cid(a), compute_envelope_cid(b))


if __name__ == "__main__":
    unittest.main()
