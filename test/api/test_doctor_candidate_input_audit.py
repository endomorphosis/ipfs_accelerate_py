"""Historical Doctor inputs survive source edits without creating authority."""
import copy
import hashlib
import json

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_context_audit as audit
from benchmarks.agent_supervisor.container_coding import terminal_candidate_input_audit as candidate
from ipfs_accelerate_py.agent_supervisor.runtime import owner_candidate_dispatch_audit as archive
from test.api.test_candidate_input_audit import case as ordinary_case, replay
from test.api.test_doctor_candidate_dispatch import doctor_dispatch_case, doctor_service, retained, scenario  # noqa: F401


@pytest.fixture
def case(doctor_dispatch_case):
    return ordinary_case.__wrapped__(doctor_dispatch_case)


def test_exact_doctor_retention_and_replay_after_original_artifact_and_handoff_removal(case):
    prepared, row, expected = case
    handle = prepared.prepared.audit
    manifest = json.loads(handle.manifest_bytes)
    assert manifest["schema"] == archive.DOCTOR_SCHEMA
    assert handle.observation["schema"] == archive.DOCTOR_OBSERVATION_SCHEMA
    assert set(manifest["artifacts"]) == set(archive.DOCTOR_FILENAMES)
    captured = prepared.launch._original_requirements._doctor_requirements
    bodies = {name: (handle.manifest_path.parent / pin["path"]).read_bytes()
              for name, pin in manifest["artifacts"].items()}
    assert bodies["original_doctor_artifact"] == captured.artifact_bytes
    assert bodies["original_doctor_advisory"] == captured.advisory_bytes
    assert bodies["original_doctor_receipt"] == captured._receipt_bytes
    assert bodies["doctor_context"] == prepared.prepared.prompt._doctor_context.metadata_bytes
    assert manifest["binding"]["doctor_coding_policy"] == prepared.launch.doctor_coding_policy
    assert row["doctor_residual_context"] is None
    original = captured._request[0]
    from pathlib import Path
    original = Path(original)
    original.unlink()
    prepared.prepared.close()
    (prepared.original.workspace / "answer.py").write_text("# subsequent candidate edit\n")
    try:
        model, _, checks = replay(case)
        assert model == expected and all(checks.values())
        assert checks["retained_doctor_constraints"] is True
        assert checks["retained_doctor_coding_policy"] is True
        assert checks["retained_proposal_budget_unchanged"] is True
    finally:
        original.write_bytes(captured.artifact_bytes)
        original.chmod(0o444)


def test_doctor_retained_artifact_advice_receipt_context_and_policy_cannot_be_substituted(case):
    prepared, _, _ = case
    handle = prepared.prepared.audit
    manifest = json.loads(handle.manifest_bytes)
    for name in ("original_doctor_artifact", "original_doctor_advisory", "original_doctor_receipt", "doctor_context", "admission"):
        path = handle.manifest_path.parent / manifest["artifacts"][name]["path"]
        original = path.read_bytes()
        try:
            path.chmod(0o600)
            path.write_bytes(b"{}")
            path.chmod(0o400)
            with pytest.raises(ValueError):
                replay(case)
        finally:
            path.chmod(0o600)
            path.write_bytes(original)
            path.chmod(0o400)
    replacement = copy.deepcopy(manifest)
    replacement["binding"]["doctor_coding_policy"]["max_outer_router_calls"] = 3
    try:
        handle.manifest_path.chmod(0o600)
        handle.manifest_path.write_text(json.dumps(replacement))
        handle.manifest_path.chmod(0o400)
        with pytest.raises(ValueError):
            replay(case)
    finally:
        handle.manifest_path.chmod(0o600)
        handle.manifest_path.write_bytes(handle.manifest_bytes)
        handle.manifest_path.chmod(0o400)


def test_doctor_dispatch_receipt_cannot_drop_constraints_or_claim_a_native_token_cap(case):
    prepared, original, _ = case
    for change in ("missing", "schema", "cap", "budget_type", "budget", "extra"):
        row = copy.deepcopy(original)
        dispatch = row["candidate_round_dispatch"]
        doctor = dispatch["doctor_candidate_dispatch"]
        if change == "missing":
            del dispatch["doctor_candidate_dispatch"]
        elif change == "schema":
            dispatch["schema"] = "supervisor-candidate-dispatch-observation@1"
        elif change == "cap":
            doctor["native_token_cap_enforced"] = True
        elif change == "budget_type":
            doctor["proposal_budget"]["max_calls"] = True
        elif change == "budget":
            doctor["proposal_budget"]["max_calls"] = 2
        else:
            doctor["provider_usage"] = 0
        with pytest.raises((KeyError, ValueError)):
            audit._receipt(row, prepared.original.workspace.parent)


def test_self_consistent_rehashed_doctor_advice_does_not_replace_dispatched_input(case):
    prepared, original, _ = case
    handle = prepared.prepared.audit
    manifest = json.loads(handle.manifest_bytes)
    pin = manifest["artifacts"]["original_doctor_advisory"]
    path = handle.manifest_path.parent / pin["path"]
    raw = path.read_bytes() + b"\nNew unauthorized advice."
    path.chmod(0o600); path.write_bytes(raw); path.chmod(0o400)
    pin.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    handle.manifest_path.chmod(0o600)
    handle.manifest_path.write_text(json.dumps(manifest))
    handle.manifest_path.chmod(0o400)
    # The archive can be rehashed, but the independently dispatched handoff
    # and its receipt still bind the original exact advice.
    with pytest.raises(ValueError):
        candidate.replay(rendered=prepared.original.prompt, receipt=original,
            parent=handle.manifest_path.parent)
