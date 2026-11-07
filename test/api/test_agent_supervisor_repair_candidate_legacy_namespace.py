"""The existing v1 validation registry and candidate parser share one literal.

These are metadata parser tests. They create no Git checkout, repair attempt,
provider, native reservation, source adoption, or live runtime authorization.
"""
import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import repair_candidate as candidate
from ipfs_accelerate_py.agent_supervisor.rescue import repair_candidate_owners as owners


def candidate_policy(tmp_path, namespace):
    repository = tmp_path / "repository"
    state = tmp_path / "state"
    repository.mkdir()
    state.mkdir()
    return dict(schema=candidate.POLICY_SCHEMA, repository_root=str(repository),
        state_root=str(state), namespace=namespace, base_commit="a" * 40,
        source_paths=["app.py"], allowed_paths=["app.py"],
        max_source_bytes=4096, max_candidate_bytes=8192, max_files=2,
        git_timeout_seconds=15)


def test_legacy_v1_candidate_reaches_original_validation_registry(tmp_path):
    value = candidate_policy(tmp_path, "legal_ir_unattended_repair")
    parsed = candidate.parse_candidate_policy({"candidate_policy": value})
    assert parsed.namespace == "legal_ir_unattended_repair"
    assert parsed.to_dict() == value
    attempt = "a" * 32
    directory = owners.validation_control_directory(
        {"validation": {"attempt_root": str(tmp_path / "validation")}},
        {"binding": {"namespace": parsed.namespace, "board_id": "legal-ir",
                     "attempt_id": attempt}})
    assert directory == tmp_path / "validation/control-requests/legal_ir_unattended_repair/legal-ir" / attempt
    assert not directory.exists()


def test_existing_ptr_lane_parser_and_v3_registry_are_unchanged(tmp_path):
    value = candidate_policy(tmp_path, "ptr_lane_0")
    parsed = candidate.parse_candidate_policy({"candidate_policy": value})
    assert parsed.to_dict() == value
    value["namespace"] = "ptr_lane_27"
    assert candidate.parse_candidate_policy({"candidate_policy": value}).namespace == "ptr_lane_27"
    attempt = "b" * 32
    directory = owners.validation_control_directory(
        {"validation": {"attempt_root": str(tmp_path / "validation"),
                        "external_fixture_profile": {}}},
        {"binding": {"namespace": parsed.namespace, "board_id": "legal-ir",
                     "attempt_id": attempt}})
    assert directory == tmp_path / "validation/control-requests/legal_ir_unattended_repair/legal-ir" / attempt
    assert not directory.exists()


def test_other_namespace_spellings_remain_rejected_without_effects(tmp_path):
    value = candidate_policy(tmp_path, "ptr_lane_0")
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    for namespace in ("legal_ir_unattended_repair_extra", "LEGAL_IR_UNATTENDED_REPAIR",
                      "legal-ir-unattended-repair", "ptr_lane_x", "ptr_lane_-1", "../other"):
        value["namespace"] = namespace
        with pytest.raises(candidate.CandidateError, match="explicit repair lane namespace required"):
            candidate.parse_candidate_policy({"candidate_policy": value})
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == before
