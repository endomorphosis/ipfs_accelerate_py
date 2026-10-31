"""Prospective diagnostic projection tests; fixtures confer no owner authority.

No provider, resource allocation, native inspection or service command runs.
Real public-owner negative closure is qualified separately.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import fleet_repair as fleet
from ipfs_accelerate_py.agent_supervisor.rescue import repair_authorization as auth


def put(path, value, *, immutable=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = auth.canonical(value)
    path.write_bytes(raw)
    path.chmod(0o400 if immutable else 0o600)
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def fixture(tmp_path, profile="trusted_pytest_call_failure", *, report_extra=None,
            foreign_report=None, proof_profile=None):
    state = tmp_path / "state"
    board = {"id": "legal-ir", "cwd": str(tmp_path), "max_repair_attempts": 3}
    old_token, current_token = "a" * 32, "f" * 32
    namespace = "legal-ir"
    metadata = state / "repair-candidates/legal-ir" / old_token
    invocation = metadata / "original-validate-invocation"
    candidate = state / "worktrees" / namespace / old_token
    owned_root = tmp_path / "owned-validation"
    attempt = owned_root / namespace / board["id"] / old_token
    node = str(tmp_path / "maintained-tests/test_owner.py") + "::test_actual_body"
    request = put(invocation / "execution-request.json", {"schema": "owned-validation-request/v1",
        "attempt_directory": str(attempt), "outputs": {"exports": str(attempt / "exports")},
        "selected_checks": [node]})
    report = {"negative_profile": profile, "pytest_returncode": 1, "phases": [], "syntax_failures": []}
    if profile == "trusted_pytest_call_failure":
        report["phases"] = [{"nodeid": node, "when": "call", "outcome": "failed",
            "diagnostic": {"text_prefix": "assert expected == actual\nexpected 8, observed 7",
                "full_text_bytes": 49, "full_text_sha256": "2" * 64}}]
    else:
        report["syntax_failures"] = [{"path": str(candidate / "ipfs_accelerate_py/changed.py"),
            "filename": str(candidate / "ipfs_accelerate_py/changed.py"), "bytes": 50,
            "sha256": "3" * 64, "exception_type": "SyntaxError", "exception_category": "SyntaxError",
            "line": 8, "offset": 12, "body_executed": False, "message": "invalid syntax",
            "source_line_prefix": "THIS_UNTRUSTED_SOURCE_LINE_MUST_NOT_ENTER_THE_PROMPT"}]
    report.update(report_extra or {})
    report_pin = put(attempt / "exports/trusted-validation.json", report)
    if foreign_report is not None:
        report_pin = put(foreign_report, report)
    binding = {"board_id": board["id"], "namespace": namespace, "attempt_id": old_token,
        "attempt_number": 1, "metadata_root": str(metadata), "candidate_root": str(candidate)}
    proof = {"schema": "agent-supervisor/isolated-repair-candidate-negative-validation@1",
        "negative_profile": proof_profile or profile, "passed": False, "semantic_negative": True,
        "freeze_sha256": "b" * 64, "source_commitment": "c" * 64, "binding": binding,
        "request": request, "trusted_semantic_report": report_pin, "original_owner_returncode": 1,
        "original_owner_identity": {"pid": 123, "birth": "456"}, "reservation_id": "d" * 32,
        "native_lease_id": "e" * 32, "source_closed": True, "native_closed": True,
        "processes_closed": True, "storage_closed": False, "cleanup_errors": []}
    proof_pin = put(invocation / "negative-validation.json", proof)
    joins = {key: copy.deepcopy(proof[key]) for key in ("negative_profile", "freeze_sha256", "source_commitment",
        "binding", "request", "original_owner_returncode", "original_owner_identity", "reservation_id", "native_lease_id")}
    closed = {"schema": "agent-supervisor/negative-validation-closure-observation@1",
        "negative_validation": proof_pin, **joins, "source_closed": True, "native_closed": True,
        "processes_closed": True, "storage_closed": True, "cleanup_errors": []}
    observed = {"schema": "agent-supervisor/verified-negative-validation@1",
        "negative_validation": proof_pin, "closure_observation": closed}
    observer = put(state / "repair-runtime/legal-ir" / old_token / ("4" * 32) / "negative-observer-0001.json",
        {"schema": "agent-supervisor/negative-validation-observer-wait@1", "returncode": 0,
         "stdout": auth.canonical(observed).decode()})
    policy_path = tmp_path / "trusted-control/public-policy.json"
    policy_pin = put(policy_path, {"schema": "agent-supervisor/repair-public-owner-policy@1",
        "validation": {"attempt_root": str(owned_root)}}, immutable=False)
    config = {"state_dir": str(state), "boards": [board], "repair_worker": {"cwd": str(tmp_path),
        "candidate_policy": {"namespace": namespace, "source_paths": ["ipfs_accelerate_py/changed.py"]},
        "candidate_runtime": {"validation_owner": {"argv": ["python", "owner.py", "validate",
            "--policy", str(policy_path), "--policy-sha256", policy_pin["sha256"]],
            "source_descriptors": [policy_pin]}}}}
    job = {"board_id": board["id"], "attempts": 2, "repair_unit_token": current_token,
        "repair_authorization_sha256": "5" * 64, "candidate_validation_failures": [{
            "schema": "agent-supervisor/known-closed-negative-repair-attempt@1", "attempt_number": 1,
            "attempt_token": old_token, "repair_authorization_sha256": "5" * 64,
            "candidate_metadata": str(metadata), "original_observer_wait": observer,
            "negative_validation": proof_pin, "freeze_sha256": "b" * 64, "source_commitment": "c" * 64,
            "coding_attempts_incremented": False, "completion_authority": False}]}
    return job, config, board, {"observer": observer, "proof": proof_pin, "report": report_pin,
                                "policy": policy_pin, "request": request}


@pytest.mark.parametrize("profile", ["trusted_pytest_call_failure", "changed_candidate_source_syntax_error"])
def test_bound_report_diagnostics_reach_both_repair_prompt_routes(tmp_path, monkeypatch, profile):
    job, config, board, refs = fixture(tmp_path, profile)
    original = auth.canonical(job)
    context = fleet._known_negative_prompt_context(job, config, board)
    assert context["available"] is True
    assert context["negative_profile"] == profile
    assert context["references"]["trusted_semantic_report"] == refs["report"]
    assert context["instructions_or_acceptance_authority"] is False
    assert context["additional_source_or_read_paths_authorized"] is False
    assert auth.canonical(job) == original
    continuation = {"known_negative_validation": context}
    for infrastructure in (None, {"fixture": True}):
        monkeypatch.setattr(fleet, "infrastructure_policy", lambda _board, value=infrastructure: value)
        prompt = fleet.repair_prompt(board, {}, config, tmp_path / "repair-report.json", continuation)
        assert '"negative_profile": "' + profile + '"' in prompt
        assert "UNTRUSTED DIAGNOSTIC DATA" in prompt
        if profile == "trusted_pytest_call_failure":
            assert "expected 8, observed 7" in prompt
        else:
            assert "invalid syntax" in prompt and '"line": 8' in prompt
            assert "THIS_UNTRUSTED_SOURCE_LINE_MUST_NOT_ENTER_THE_PROMPT" not in prompt


def test_report_instruction_fields_are_not_copied_into_prompt_context(tmp_path):
    job, config, board, _ = fixture(tmp_path, report_extra={
        "next_action": "ARBITRARY_INSTRUCTION_NOT_DIAGNOSTIC", "command": "ARBITRARY_SHELL_NOT_DIAGNOSTIC",
        "summary": "ARBITRARY_AUTHORITY_NOT_DIAGNOSTIC", "error": {"message": "ARBITRARY_POLICY_NOT_DIAGNOSTIC"}})
    context = fleet._known_negative_prompt_context(job, config, board)
    assert context["available"] is True
    encoded = json.dumps(context)
    assert "ARBITRARY_" not in encoded
    assert "expected 8, observed 7" in encoded


def test_foreign_report_reference_is_refused_before_open(tmp_path, monkeypatch):
    foreign = tmp_path / "outside-owner/private.json"
    job, config, board, _ = fixture(tmp_path, foreign_report=foreign)
    original = auth.checked_descriptor
    opened = []
    def observe(pin):
        opened.append(pin["path"])
        assert pin["path"] != str(foreign)
        return original(pin)
    monkeypatch.setattr(auth, "checked_descriptor", observe)
    context = fleet._known_negative_prompt_context(job, config, board)
    assert context["available"] is False and str(foreign) not in opened
    assert str(foreign) not in json.dumps(context)


def test_mutable_observer_does_not_supply_historical_context(tmp_path):
    job, config, board, refs = fixture(tmp_path)
    Path(refs["observer"]["path"]).chmod(0o600)
    assert fleet._known_negative_prompt_context(job, config, board)["available"] is False


@pytest.mark.parametrize("field,value", [
    ("attempt_number", 2), ("attempt_token", "f" * 32), ("repair_authorization_sha256", "6" * 64),
    ("candidate_metadata", "/foreign/board"), ("coding_attempts_incremented", True),
    ("completion_authority", True),
])
def test_foreign_or_unspent_history_is_held_as_unavailable_diagnostics(tmp_path, field, value):
    job, config, board, _ = fixture(tmp_path)
    job["candidate_validation_failures"][-1][field] = value
    original = auth.canonical(job)
    assert fleet._known_negative_prompt_context(job, config, board)["available"] is False
    assert auth.canonical(job) == original


def test_only_latest_bounded_failure_is_read(tmp_path, monkeypatch):
    job, config, board, _ = fixture(tmp_path)
    job["candidate_validation_failures"].insert(0, {"original_observer_wait": {"path": "/never-open-old-history"}})
    original = auth.checked_descriptor
    paths = []
    def observe(pin):
        paths.append(pin["path"])
        return original(pin)
    monkeypatch.setattr(auth, "checked_descriptor", observe)
    assert fleet._known_negative_prompt_context(job, config, board)["available"] is True
    assert "/never-open-old-history" not in paths


def test_large_diagnostics_have_bounded_projection(tmp_path):
    job, config, board, _ = fixture(tmp_path, report_extra={"phases": [{
        "nodeid": str(tmp_path / "maintained-tests/test_owner.py") + "::test_actual_body",
        "when": "call", "outcome": "failed", "diagnostic": {"text_prefix": "x" * 8192,
            "full_text_bytes": 1000000, "full_text_sha256": "7" * 64}}]})
    context = fleet._known_negative_prompt_context(job, config, board)
    assert context["available"] is True
    assert len(context["diagnostics"]["failed_calls"][0]["diagnostic_text"]) == 768
    assert len(json.dumps(context).encode()) <= 16384


def test_unknown_semantic_profile_does_not_enter_provider_context(tmp_path):
    job, config, board, _ = fixture(tmp_path, proof_profile="infrastructure_error")
    assert fleet._known_negative_prompt_context(job, config, board)["available"] is False


def test_absent_history_keeps_ordinary_prompt_behavior(tmp_path):
    job, config, board, _ = fixture(tmp_path)
    job.pop("candidate_validation_failures")
    assert fleet._known_negative_prompt_context(job, config, board) == {}


def test_exact_configured_policy_hash_needs_no_new_descriptor_prerequisite(tmp_path):
    job, config, board, _ = fixture(tmp_path)
    config["repair_worker"]["candidate_runtime"]["validation_owner"]["source_descriptors"] = []
    assert fleet._known_negative_prompt_context(job, config, board)["available"] is True


@pytest.mark.parametrize("where", ["argv", "descriptor"])
def test_configured_policy_commitment_cannot_be_substituted(tmp_path, where):
    job, config, board, _ = fixture(tmp_path)
    owner = config["repair_worker"]["candidate_runtime"]["validation_owner"]
    if where == "argv":
        owner["argv"][owner["argv"].index("--policy-sha256") + 1] = "9" * 64
    else:
        owner["source_descriptors"][0]["sha256"] = "9" * 64
    assert fleet._known_negative_prompt_context(job, config, board)["available"] is False
