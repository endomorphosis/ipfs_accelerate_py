"""Both generations retain closed identity and claim-time control bindings."""
import hashlib
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources.database_attempt_binding import validate_database_attempt_binding


def _signed(value):
    value = dict(value)
    value.pop("binding_id", None)
    value["binding_id"] = "sha256:" + hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode()).hexdigest()
    return value


def _binding(version=2):
    value = dict(schema=f"ipfs_accelerate_py/agent-supervisor/database-portal-attempt-binding@{version}",
                 interface="DatabasePortalExecutionBridge@1", attempt_id="attempt:1",
                 claim_id="claim:1", task_cid="sha256:task", task_alias="TASK-1",
                 goal_cid="", plan_cid="", task_revision=3, fencing_token=2,
                 fence_epoch=2, lease_id="lease:1", task_body_digest="sha256:body",
                 projection_seed_digest="sha256:seed", projection_immutable_digest="sha256:immutable",
                 authoritative_task_store="duckdb", projection_authority=False)
    if version == 2:
        value.update(control_binding_id="baguq-control", control_task_projection_cid="baguq-projection",
                     control_expected_revision=3, control_portal_binding_basis_cid="baguq-basis")
    return _signed(value)


@pytest.mark.parametrize("version", (1, 2))
def test_accepts_both_closed_generations_without_mutating_input(version):
    value = _binding(version)
    result = validate_database_attempt_binding(value)
    assert result == value and result is not value


@pytest.mark.parametrize("field,value", (
    ("schema", "ipfs_accelerate_py/agent-supervisor/database-portal-attempt-binding@3"),
    ("task_revision", True), ("fencing_token", False), ("fence_epoch", 0),
    ("control_expected_revision", True), ("control_expected_revision", 4),
    ("control_binding_id", ""), ("control_task_projection_cid", None),
    ("control_portal_binding_basis_cid", ""), ("projection_authority", 0),
    ("projection_authority", True), ("authoritative_task_store", "file"),
    ("interface", "Foreign@1"), ("unknown_field", "ignored"),
))
def test_rejects_resigned_invalid_bindings(field, value):
    binding = _binding()
    binding[field] = value
    with pytest.raises(ValueError):
        validate_database_attempt_binding(_signed(binding))


@pytest.mark.parametrize("field", ("claim_id", "lease_id", "control_binding_id", "control_task_projection_cid", "control_portal_binding_basis_cid"))
def test_digest_binds_claim_and_control_identity(field):
    binding = _binding()
    binding[field] += "-substituted"
    with pytest.raises(ValueError, match="digest"):
        validate_database_attempt_binding(binding)


@pytest.mark.parametrize("field", ("claim_id", "control_binding_id", "control_expected_revision", "control_portal_binding_basis_cid"))
def test_rejects_resigned_missing_fields(field):
    binding = _binding()
    del binding[field]
    with pytest.raises(ValueError):
        validate_database_attempt_binding(_signed(binding))
