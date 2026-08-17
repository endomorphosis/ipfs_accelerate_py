"""SCH-001 wire-contract and MCP++ codec tests."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.mcp_server.mcplusplus.artifacts import (
    canonicalize_artifact,
    compute_artifact_cid,
)
from ipfs_accelerate_py.mcp_server.mcplusplus.kubo_cid import cid_for_bytes

from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import (
    AcceptanceDisposition,
    HarnessError,
    SemanticCapsuleRef,
    SemanticStateRootManifest,
    TestSelectionRef,
    validate_opaque_cid,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.wire import (
    SemanticStateWireCodec,
    cid_for_payload,
    interface_descriptor_cid,
    semantic_state_interface_descriptor,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = (
    REPO_ROOT
    / "ipfs_accelerate_py/agent_supervisor/semantic_state/schemas/semantic-state-harness.interface.json"
)


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


SAMPLE_CID = _cid("capsule")


def _manifest(**overrides: object) -> SemanticStateRootManifest:
    payload = {
        "repository_id": "example/repo",
        "base_tree_cid": _cid("base-tree"),
        "candidate_tree_cid": _cid("candidate-tree"),
        "datasets_state_cid": _cid("datasets-state"),
        "datasets_semantic_state_root_cid": _cid("datasets-root"),
        "capsule_index_cid": _cid("capsule-index"),
        "delta_cid": _cid("delta"),
        "invalidation_cid": _cid("invalidation"),
        "obligation_set_cid": _cid("obligations"),
        "test_selection_cid": _cid("selection"),
        "receipt_index_cid": _cid("receipts"),
        "environment_binding_cids": [_cid("env-a"), _cid("env-b")],
        "event_head_cid": _cid("event-head"),
        "versions": {
            "capsule_schema": "ipfs-datasets.software-contracts.semantic-capsule@1",
            "selection_schema": "ipfs-datasets.software-contracts.semantic-test-selection@1",
            "semantic_index_schema": "ipfs-datasets.software-contracts.semantic-index@2",
            "semantic_state_schema": "ipfs-datasets.software-contracts.semantic-state@1",
        },
        "acceptance_disposition": AcceptanceDisposition.CANDIDATE.value,
    }
    payload.update(overrides)
    return SemanticStateRootManifest.from_dict(payload)


def test_equivalent_payloads_have_identical_canonical_bytes_and_real_cidv1() -> None:
    left = _manifest().to_dict()
    right = json.loads(json.dumps(_manifest().to_dict()))
    assert canonicalize_artifact(left) == canonicalize_artifact(right)
    assert cid_for_payload(left) == cid_for_payload(right)
    assert cid_for_payload(left).startswith("b")
    assert cid_for_payload(left) == cid_for_bytes(canonicalize_artifact(left))
    assert cid_for_payload(left) != compute_artifact_cid(left)


def test_unknown_fields_and_enums_fail_closed() -> None:
    payload = _manifest().to_dict()
    payload["wall_clock_ms"] = 12
    with pytest.raises(HarnessError, match="fields must be exactly"):
        SemanticStateRootManifest.from_dict(payload)
    payload = _manifest().to_dict()
    payload["acceptance_disposition"] = "maybe"
    with pytest.raises(HarnessError, match="unsupported value"):
        SemanticStateRootManifest.from_dict(payload)


def test_forged_and_pseudo_cids_fail_closed() -> None:
    with pytest.raises(HarnessError, match="forged|CIDv1|base32"):
        validate_opaque_cid("cidv1-sha256-" + "ab" * 32, "cid")
    with pytest.raises(HarnessError, match="forged|CIDv1|base32"):
        validate_opaque_cid("sim:local-model", "cid")
    payload = _manifest().to_dict()
    payload["delta_cid"] = "not-a-cid"
    with pytest.raises(HarnessError):
        SemanticStateRootManifest.from_dict(payload)


def test_capsule_ref_is_admission_metadata_only() -> None:
    ref = SemanticCapsuleRef.from_dict(
        {
            "capsule_cid": SAMPLE_CID,
            "semantic_state_root_cid": _cid("root"),
            "stable_symbol_id": "mod:fn",
            "version_cid": _cid("version"),
            "source_cid": _cid("source"),
            "confidence": "exact",
            "validity_bindings": [_cid("bind")],
            "raw_source_required": False,
        }
    )
    assert set(ref.to_dict()) == {
        "capsule_cid",
        "semantic_state_root_cid",
        "stable_symbol_id",
        "version_cid",
        "source_cid",
        "confidence",
        "validity_bindings",
        "raw_source_required",
    }
    with pytest.raises(HarnessError, match="fields must be exactly"):
        SemanticCapsuleRef.from_dict({**ref.to_dict(), "facts": {"purity": "pure"}})


def test_test_selection_ref_does_not_copy_producer_facts() -> None:
    ref = TestSelectionRef.from_dict(
        {
            "selection_cid": _cid("selection"),
            "previous_semantic_state_root_cid": None,
            "current_semantic_state_root_cid": _cid("current"),
        }
    )
    assert ref.previous_semantic_state_root_cid is None
    with pytest.raises(HarnessError, match="fields must be exactly"):
        TestSelectionRef.from_dict(
            {
                **ref.to_dict(),
                "selected_node_ids": ["test_a"],
            }
        )


def test_operational_fields_cannot_alter_manifest_cid() -> None:
    manifest = _manifest()
    left = cid_for_payload(manifest.to_dict())
    # Local path / pid / duration live outside the CID-bound record.
    operational = {
        "local_path": "/tmp/work",
        "pid": 12,
        "duration_ms": 40,
        **manifest.to_dict(),
    }
    with pytest.raises(HarnessError):
        SemanticStateRootManifest.from_dict(operational)
    again = _manifest(
        environment_binding_cids=[_cid("env-b"), _cid("env-a")],
    )
    assert cid_for_payload(again.to_dict()) == left


def test_descriptor_and_interface_changes_have_detectable_cids() -> None:
    codec = SemanticStateWireCodec()
    descriptor = codec.interface_descriptor()
    assert descriptor["application_schema_cid"] == interface_descriptor_cid()
    assert descriptor["name"] == "semantic-state-harness"
    checked_in = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    for field in ("name", "namespace", "version", "methods", "errors", "requires", "compatibility"):
        assert field in checked_in
    mutated = semantic_state_interface_descriptor()
    mutated["version"] = "1.0.1"
    assert cid_for_payload({k: mutated[k] for k in mutated if k != "application_schema_cid"}) != (
        interface_descriptor_cid()
    )


def test_envelope_and_event_round_trip_and_reject_forged_payload_cid() -> None:
    codec = SemanticStateWireCodec()
    manifest = _manifest()
    envelope = codec.encode_root_manifest(manifest)
    assert envelope["interface_cid"] == interface_descriptor_cid()
    assert codec.decode_root_manifest(envelope) == manifest
    forged = dict(envelope)
    forged["payload_cid"] = _cid("forged-payload")
    with pytest.raises(HarnessError, match="does not match"):
        codec.decode_execution_envelope(forged)
    receipt = codec.encode_execution_receipt(manifest.to_dict())
    assert codec.decode_execution_receipt(receipt) == manifest.to_dict()
    event = codec.encode_dag_event(manifest.to_dict(), parent_event_cids=[], timestamp="0")
    decoded = codec.decode_dag_event(event)
    assert decoded["payload_cid"] == cid_for_payload(manifest.to_dict())
    assert decoded["event_cid"] == event["event_cid"]


def test_ordinary_import_performs_no_io(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    opened: list[str] = []
    real_open = Path.open

    def tracked_open(self, *args, **kwargs):
        opened.append(str(self))
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracked_open)
    importlib.reload(
        importlib.import_module("ipfs_accelerate_py.agent_supervisor.semantic_state")
    )
    assert opened == []
    assert list(tmp_path.iterdir()) == []
