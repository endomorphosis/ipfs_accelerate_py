"""PCCE-023: typed failures, closed result transitions, and redacted errors."""

from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import os
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import pytest

from ipfs_accelerate_py.proof_context.compatibility import FROZEN_MATRIX
from ipfs_accelerate_py.proof_context.errors import (
    ALLOWED_DETAIL_KEYS,
    DISPOSITIONS,
    ERROR_SCHEMA,
    ERROR_SEMANTICS,
    ERROR_TAXONOMY_CONTENT_ID,
    ERROR_TYPES,
    ERRORS,
    REDACTED,
    BoundaryViolationError,
    ContextInsufficientError,
    HumanReviewRequiredError,
    InfrastructureFailureError,
    MalformedError,
    PartialEffectError,
    ProofContextError,
    ProofTimeoutError,
    RepairRequiredError,
    UnavailableCapabilityError,
    UnknownFieldError,
    admit_error,
    bound_details,
    classify_error,
    error_for,
    error_status,
    from_provider_error,
    frozen_error_taxonomy,
    redact_text,
)
from ipfs_accelerate_py.proof_context.policy import (
    ERRORS as POLICY_ERRORS,
    STATUSES as POLICY_STATUSES,
)
from ipfs_accelerate_py.proof_context.results import (
    COMPATIBILITY_MATRIX_CONTENT_ID,
    CONTRACT_VERSION,
    ESCALATION_STATUSES,
    FAILURE_STATUSES,
    IDENTITY_FIELDS,
    PCCE_006_CONTENT_ID,
    PATCH_BEARING_STATUSES,
    PROVENANCES,
    REPAIR_STATUSES,
    RESULT_DESCRIPTOR,
    RESULT_SCHEMA,
    RESULT_STATE_CID,
    RETRYABLE_STATUSES,
    REVIEW_STATUSES,
    SCHEMA,
    START,
    STATUSES,
    STATUS_SEMANTICS,
    STATUS_TAXONOMY_CONTENT_ID,
    TERMINAL_STATUSES,
    ResultIdentities,
    ResultRecord,
    admit_result,
    admit_status,
    admit_transition,
    classify_status,
    emit_result,
    frozen_result_taxonomy,
    is_failure,
    is_legal_transition,
    is_success,
    is_terminal,
    is_unavailable,
    legal_targets,
    mint_result_cid,
    result_descriptor,
    result_from_error,
    result_state_cid,
    status_semantics,
    transition_pairs,
    transition_table,
)

VALID_CID = "bafkreiapj52u5hi7pco5ebplvecv72olbnqglg2e7emwnmme4gguzsnpu4"
EVIDENCE_CID = "b" + "e" * 58
PATCH_ID = "patch-pcce-023"
UNKNOWN_STATUSES = ("failed", "ok", "passed", "success", "error", "pass", "running")
UNKNOWN_ERRORS = ("mystery", "internal", "provider_crash", "ok")


def _cid(label: str) -> str:
    body = "".join(ch if ch in "abcdefghijklmnopqrstuvwxyz234567" else "a" for ch in label)
    return "b" + (body + "a" * 58)[:58]


def _identities(**overrides: object) -> ResultIdentities:
    values = {
        "repository_id": "example/ordinary-python-repo",
        "repository_state_cid": VALID_CID,
        "task_id": "PCCE-023",
        "run_id": "run-pcce-023",
        "trace_id": "trace-pcce-023",
        "evidence_cid": EVIDENCE_CID,
        "patch_id": PATCH_ID,
        "artifact_id": _cid("artifact"),
        "contract_version": CONTRACT_VERSION,
    }
    values.update(overrides)
    return ResultIdentities(**values)


def _record(status: str = "succeeded", **overrides: object) -> ResultRecord:
    payload = {
        "schema": RESULT_SCHEMA,
        "status": status,
        "identities": _identities(),
        "provenance": "live",
        "error": None,
        "payload": {},
    }
    payload.update(overrides)
    return ResultRecord(**payload)


def test_every_required_status_is_represented_exactly_once() -> None:
    assert STATUSES == (
        "succeeded",
        "rejected",
        "verification_failed",
        "proof_failed",
        "assurance_failed",
        "context_insufficient",
        "model_escalation_required",
        "human_review_required",
        "unavailable",
        "timeout",
        "cancelled",
        "invalid",
        "stale",
        "simulated",
        "infrastructure_failure",
        "partial_effect",
        "repair_required",
    )
    assert len(STATUSES) == 17
    assert len(STATUSES) == len(set(STATUSES))
    assert tuple(STATUS_SEMANTICS) == STATUSES
    assert tuple(transition_table()) == STATUSES
    assert STATUSES == POLICY_STATUSES
    assert PROVENANCES == ("live", "replayed", "simulated")
    assert set(TERMINAL_STATUSES).issubset(STATUSES)
    assert status_semantics("succeeded")["accepted"] is True
    assert status_semantics("unavailable")["failed"] is False
    assert status_semantics("partial_effect")["repair"] is True
    with pytest.raises(TypeError):
        STATUSES[0] = "shadow"  # type: ignore[index]


def test_every_required_error_is_represented_exactly_once() -> None:
    assert ERRORS == (
        "unknown_field",
        "malformed",
        "identity_inconsistent",
        "stale_root",
        "simulated_promoted",
        "pseudo_cid",
        "schema_mismatch",
        "boundary_violation",
        "unavailable_capability",
        "timeout",
        "cancelled",
        "verification_failed",
        "proof_failed",
        "assurance_failed",
        "context_insufficient",
        "infrastructure_failure",
        "partial_effect",
        "repair_required",
        "human_review_required",
    )
    assert len(ERRORS) == 19
    assert len(ERRORS) == len(set(ERRORS))
    assert tuple(ERROR_SEMANTICS) == ERRORS
    assert tuple(ERROR_TYPES) == ERRORS
    assert ERRORS == POLICY_ERRORS
    assert DISPOSITIONS == ("retry", "escalation", "review", "repair", "reject")
    with pytest.raises(TypeError):
        ERRORS[0] = "internal"  # type: ignore[index]


def test_legal_transitions_are_closed_and_deterministic() -> None:
    table = transition_table()
    pairs = transition_pairs()
    reconstructed: list[tuple[str, str]] = []
    for source in STATUSES:
        targets = table[source]
        assert targets == legal_targets(source)
        assert targets == tuple(dict.fromkeys(targets))
        assert set(targets).issubset(STATUSES)
        assert source in targets
        for target in STATUSES:
            legal = target in targets
            assert is_legal_transition(source, target) is legal
            if legal:
                assert admit_transition(source, target) == target
                reconstructed.append((source, target))
            else:
                with pytest.raises(BoundaryViolationError) as exc:
                    admit_transition(source, target)
                assert exc.value.code == "boundary_violation"
                assert exc.value.accepted is False
    assert tuple((row["from"], row["to"]) for row in pairs) == tuple(reconstructed)
    assert admit_transition(START, "timeout") == "timeout"
    assert is_legal_transition(START, "succeeded") is True
    with pytest.raises(TypeError):
        table["succeeded"] = ("rejected",)  # type: ignore[index]


@pytest.mark.parametrize("source,target", (("unavailable", "succeeded"), ("unavailable", "rejected"), ("unavailable", "verification_failed"), ("unavailable", "invalid"), ("partial_effect", "succeeded"), ("simulated", "succeeded"), ("succeeded", "rejected"), ("rejected", "succeeded"), ("cancelled", "timeout"), ("stale", "succeeded"), ("invalid", "repair_required")))
def test_forbidden_transitions_fail_closed(source: str, target: str) -> None:
    assert is_legal_transition(source, target) is False
    with pytest.raises(BoundaryViolationError):
        admit_transition(source, target)


def test_unavailable_is_not_collapsed_into_failure_or_pass() -> None:
    record = _record("unavailable", error="unavailable_capability")
    assert record.status == "unavailable"
    assert record.accepted is False
    assert record.failed is False
    assert record.unavailable is True
    assert record.retryable is True
    assert is_success("unavailable") is False
    assert is_failure("unavailable") is False
    assert is_unavailable("unavailable") is True
    assert classify_status("unavailable") == "retry"
    with pytest.raises(BoundaryViolationError):
        record.transition("succeeded")
    with pytest.raises(BoundaryViolationError):
        record.transition("rejected")
    reviewed = record.transition("human_review_required")
    assert reviewed.status == "human_review_required"
    assert reviewed.accepted is False
    assert reviewed.identities.trace_id == record.identities.trace_id


def test_simulated_partial_and_review_cannot_claim_success() -> None:
    simulated = _record("simulated", provenance="simulated", error="simulated_promoted")
    assert simulated.accepted is False
    assert is_success("simulated", provenance="simulated") is False
    partial = _record("partial_effect", error="partial_effect")
    assert partial.accepted is False
    assert partial.partial_effect is True
    assert partial.repair is True
    assert classify_status("partial_effect") == "repair"
    repaired = partial.transition("repair_required", error="repair_required")
    assert repaired.status == "repair_required"
    assert repaired.accepted is False
    review = _record("human_review_required", error="human_review_required")
    assert review.human_review is True
    assert review.accepted is False
    with pytest.raises(ProofContextError):
        _record("succeeded", provenance="simulated")


def test_terminal_results_bind_trace_run_task_repository_patch_and_evidence() -> None:
    status_errors = {
        "succeeded": None,
        "rejected": "boundary_violation",
        "verification_failed": "verification_failed",
        "proof_failed": "proof_failed",
        "assurance_failed": "assurance_failed",
        "cancelled": "cancelled",
        "invalid": "malformed",
        "stale": "stale_root",
        "simulated": "simulated_promoted",
    }
    for status in TERMINAL_STATUSES:
        provenance = "simulated" if status == "simulated" else "live"
        record = emit_result(
            status,
            _identities(),
            provenance=provenance,
            error=status_errors.get(status),
        )
        payload = record.to_mapping()
        bound = payload["identities"]
        for field in IDENTITY_FIELDS:
            assert field in bound
        assert bound["trace_id"] == "trace-pcce-023"
        assert bound["run_id"] == "run-pcce-023"
        assert bound["task_id"] == "PCCE-023"
        assert bound["repository_id"] == "example/ordinary-python-repo"
        assert bound["patch_id"] == PATCH_ID
        assert bound["evidence_cid"] == EVIDENCE_CID
        assert record.terminal is True
        if status == "succeeded":
            assert record.accepted is True
        else:
            assert record.accepted is False
    with pytest.raises(ProofContextError):
        emit_result("succeeded", _identities(evidence_cid=None))
    with pytest.raises(ProofContextError):
        emit_result("succeeded", _identities(patch_id=None))


def test_schema_round_trips_for_every_status_and_error() -> None:
    for status in STATUSES:
        error = None
        if status != "succeeded":
            matching = [code for code in ERRORS if error_status(code) == status]
            error = matching[0] if matching else "malformed"
            if status == "rejected":
                error = "boundary_violation"
            if status == "model_escalation_required":
                error = "context_insufficient"
        record = _record(status, error=error, provenance="simulated" if status == "simulated" else "live")
        restored = ResultRecord.from_mapping(dict(record.to_mapping()))
        assert restored.status == status
        assert restored.identities.trace_id == record.identities.trace_id
        assert restored.accepted is record.accepted
        assert admit_result(dict(record.to_mapping())).status == status
    for code in ERRORS:
        typed = error_for(code, f"{code} occurred", details={"stage": "verify", "retry_count": 1})
        restored_error = ProofContextError.from_mapping(dict(typed.to_mapping()))
        assert restored_error.code == code
        assert restored_error.accepted is False
        assert restored_error.disposition == classify_error(code)
        assert restored_error.to_mapping()["schema"] == ERROR_SCHEMA


@pytest.mark.parametrize("status", UNKNOWN_STATUSES)
def test_unknown_status_is_rejected(status: str) -> None:
    with pytest.raises(UnknownFieldError) as exc:
        admit_status(status)
    assert exc.value.code == "unknown_field"
    assert exc.value.accepted is False
    with pytest.raises(UnknownFieldError):
        emit_result(status, _identities())
    with pytest.raises(UnknownFieldError):
        ResultRecord.from_mapping(
            {
                "schema": RESULT_SCHEMA,
                "status": status,
                "identities": dict(_identities().to_mapping()),
            }
        )


@pytest.mark.parametrize("code", UNKNOWN_ERRORS)
def test_unknown_error_is_rejected(code: str) -> None:
    with pytest.raises(UnknownFieldError) as exc:
        admit_error(code)
    assert exc.value.code == "unknown_field"
    with pytest.raises(UnknownFieldError):
        error_for(code)


def test_errors_are_bounded_redacted_and_never_claim_success() -> None:
    secret = "api_key=sk-live-not-a-real-key bearer tok_abc123"
    typed = error_for(
        "infrastructure_failure",
        secret,
        details={
            "capability": "provider",
            "token": "should-not-appear",
            "password": "x",
            "stage": "run",
            "authorization": "redacted",
        },
    )
    text = str(typed)
    mapping = typed.to_mapping()
    assert "sk-live-not-a-real-key" not in text
    assert "tok_abc123" not in text
    assert REDACTED in text
    assert mapping["accepted"] is False
    assert mapping["disposition"] == "retry"
    assert "token" not in mapping["details"]
    assert "password" not in mapping["details"]
    assert mapping["details"]["stage"] == "run"
    assert mapping["details"]["capability"] == "provider"
    long_message = "x" * 1000
    assert len(redact_text(long_message)) <= 241
    bounded = bound_details(
        {key: "v" for key in list(ALLOWED_DETAIL_KEYS) + ["extra", "token"]}
    )
    assert "extra" not in bounded
    assert "token" not in bounded
    assert len(bounded) <= 8
    for code in ERRORS:
        item = error_for(code)
        assert item.accepted is False
        assert classify_error(code) in DISPOSITIONS
        assert classify_error(code) != "success"
        assert item.to_mapping()["accepted"] is False


def test_errors_classify_retry_escalation_review_and_repair() -> None:
    assert classify_error("timeout") == "retry"
    assert classify_error("unavailable_capability") == "retry"
    assert classify_error("infrastructure_failure") == "retry"
    assert classify_error("context_insufficient") == "escalation"
    assert classify_error("human_review_required") == "review"
    assert classify_error("partial_effect") == "repair"
    assert classify_error("repair_required") == "repair"
    assert classify_error("malformed") == "reject"
    assert ProofTimeoutError("late").retryable is True
    assert ContextInsufficientError("thin").escalation is True
    assert HumanReviewRequiredError("needs person").human_review is True
    assert RepairRequiredError("fix").repair is True
    assert PartialEffectError("half").repair is True
    assert UnavailableCapabilityError("missing").status == "unavailable"
    retry = result_from_error(ProofTimeoutError("late"), _identities())
    assert retry.status == "timeout"
    assert retry.accepted is False
    assert retry.retryable is True
    review = result_from_error("human_review_required", _identities())
    assert review.human_review is True
    assert review.accepted is False
    repair = result_from_error("repair_required", _identities())
    assert repair.repair is True
    assert repair.accepted is False
    escalate = result_from_error("context_insufficient", _identities())
    assert escalate.escalation is True
    assert escalate.accepted is False


def test_provider_errors_are_not_exposed() -> None:
    class ProviderBoom(RuntimeError):
        pass

    wrapped = from_provider_error(ProviderBoom("classified provider failure dump"))
    assert isinstance(wrapped, InfrastructureFailureError)
    assert wrapped.code == "infrastructure_failure"
    assert wrapped.accepted is False
    assert "classified provider failure dump" not in str(wrapped)
    assert "failure dump" not in str(wrapped)
    assert wrapped.details["capability"] == "ProviderBoom"
    timeout = from_provider_error(TimeoutError("sleep 30s"))
    assert timeout.code == "timeout"
    assert "sleep 30s" not in str(timeout)
    existing = ProofTimeoutError("already typed")
    assert from_provider_error(existing) is existing


def test_generic_success_dictionaries_are_rejected() -> None:
    with pytest.raises(MalformedError):
        admit_result({"ok": True})
    with pytest.raises(MalformedError):
        admit_result({"success": True, "passed": True})
    with pytest.raises(BoundaryViolationError):
        admit_result({"ok": True, "status": "unavailable"})
    with pytest.raises(MalformedError):
        admit_result(["succeeded"])
    admitted = admit_result(_record("succeeded").to_mapping())
    assert admitted.accepted is True
    assert admitted.status == "succeeded"


def test_identity_is_preserved_across_legal_transitions() -> None:
    current = emit_result("timeout", _identities(), error="timeout")
    for target in ("unavailable", "infrastructure_failure", "repair_required", "human_review_required"):
        current = current.transition(target)
        assert current.identities.trace_id == "trace-pcce-023"
        assert current.identities.run_id == "run-pcce-023"
        assert current.identities.task_id == "PCCE-023"
        assert current.identities.repository_id == "example/ordinary-python-repo"
        assert current.identities.patch_id == PATCH_ID
        assert current.identities.evidence_cid == EVIDENCE_CID
        assert current.accepted is False
    repaired = emit_result("repair_required", _identities(), error="repair_required")
    succeeded = repaired.transition("succeeded")
    assert succeeded.accepted is True
    assert succeeded.error is None
    assert succeeded.identities.patch_id == PATCH_ID


def test_retry_escalation_review_repair_status_families() -> None:
    assert RETRYABLE_STATUSES == ("unavailable", "timeout", "infrastructure_failure")
    assert ESCALATION_STATUSES == ("context_insufficient", "model_escalation_required")
    assert REVIEW_STATUSES == ("human_review_required",)
    assert REPAIR_STATUSES == ("partial_effect", "repair_required")
    assert "unavailable" not in FAILURE_STATUSES
    assert "succeeded" not in FAILURE_STATUSES
    assert "partial_effect" not in TERMINAL_STATUSES
    assert "succeeded" in TERMINAL_STATUSES
    for status in PATCH_BEARING_STATUSES:
        assert status in STATUSES


def test_result_descriptor_cid_and_frozen_taxonomy() -> None:
    descriptor = result_descriptor()
    assert descriptor is RESULT_DESCRIPTOR
    assert descriptor["schema"].endswith("/result-state")
    assert descriptor["cid"] == RESULT_STATE_CID
    assert result_state_cid() == RESULT_STATE_CID
    assert RESULT_STATE_CID.startswith("b")
    body = {key: value for key, value in descriptor.items() if key != "cid"}
    assert mint_result_cid(body) == RESULT_STATE_CID
    taxonomy = frozen_result_taxonomy()
    assert taxonomy["pcce_006_content_id"] == PCCE_006_CONTENT_ID
    assert taxonomy["pcce_006_content_id"] == (
        "sha256:b5503d2c2ec22e34091b3f747241fbde0519a9f0b213a03e0456a8f980a43f37"
    )
    assert taxonomy["compatibility_matrix_content_id"] == COMPATIBILITY_MATRIX_CONTENT_ID
    assert taxonomy["compatibility_matrix_content_id"] == FROZEN_MATRIX["content_id"]
    assert taxonomy["status_taxonomy_content_id"] == STATUS_TAXONOMY_CONTENT_ID
    assert taxonomy["error_taxonomy_content_id"] == ERROR_TAXONOMY_CONTENT_ID
    assert taxonomy["statuses"] == STATUSES
    assert taxonomy["errors"] == ERRORS
    errors = frozen_error_taxonomy()
    assert errors["errors"] == ERRORS
    assert errors["error_taxonomy_content_id"] == ERROR_TAXONOMY_CONTENT_ID
    digest = hashlib.sha256(RESULT_STATE_CID.encode("utf-8")).hexdigest()
    assert len(digest) == 64
    with pytest.raises(TypeError):
        RESULT_DESCRIPTOR["cid"] = "mutated"  # type: ignore[index]


def test_unknown_result_fields_and_pseudo_cids_fail_closed() -> None:
    with pytest.raises(UnknownFieldError):
        ResultRecord.from_mapping(
            {
                "schema": RESULT_SCHEMA,
                "status": "succeeded",
                "identities": dict(_identities().to_mapping()),
                "mystery": True,
            }
        )
    with pytest.raises(ProofContextError):
        ResultIdentities(
            repository_id="repo",
            repository_state_cid="sha256:deadbeef",
            task_id="PCCE-023",
            run_id="run",
            trace_id="trace",
        )
    with pytest.raises(UnknownFieldError):
        ProofContextError.from_mapping(
            {"schema": ERROR_SCHEMA, "code": "malformed", "mystery": 1}
        )
    with pytest.raises(BoundaryViolationError):
        ProofContextError.from_mapping(
            {"schema": ERROR_SCHEMA, "code": "malformed", "accepted": True}
        )


def test_typed_exception_classes_cover_the_taxonomy() -> None:
    for code, cls in ERROR_TYPES.items():
        instance = cls("typed")
        assert instance.code == code
        assert issubclass(cls, ProofContextError)
        assert instance.disposition == classify_error(code)
        assert inspect.isclass(cls)
    source = Path(
        inspect.getsourcefile(error_for)
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    assigned = {
        node.targets[0].id
        for node in tree.body
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
    }
    assert "UnknownFieldError" in assigned
    assert "RepairRequiredError" in assigned


def test_mapping_results_are_immutable() -> None:
    record = _record("timeout", error="timeout")
    payload = record.to_mapping()
    with pytest.raises(TypeError):
        payload["status"] = "succeeded"  # type: ignore[index]
    with pytest.raises(TypeError):
        record.identities.to_mapping()["task_id"] = "other"  # type: ignore[index]
    error_payload = error_for("malformed").to_mapping()
    with pytest.raises(TypeError):
        error_payload["accepted"] = True  # type: ignore[index]
    assert isinstance(payload, Mapping)
    assert isinstance(payload, MappingProxyType)


def test_cold_import_has_no_side_effects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    before = set(tmp_path.rglob("*"))
    results = importlib.import_module("ipfs_accelerate_py.proof_context.results")
    errors = importlib.import_module("ipfs_accelerate_py.proof_context.errors")
    after = set(tmp_path.rglob("*"))
    assert after == before
    assert results.SCHEMA == SCHEMA
    assert errors.ERRORS == ERRORS
    assert os.getenv("PCCE_MODE") in {None, os.environ.get("PCCE_MODE")}


def test_modules_do_not_search_siblings_or_perform_io() -> None:
    results_source = Path(inspect.getsourcefile(admit_result)).read_text(encoding="utf-8")
    errors_source = Path(inspect.getsourcefile(error_for)).read_text(encoding="utf-8")
    for source in (results_source, errors_source):
        assert "Path.home(" not in source
        assert "requests." not in source
        assert "subprocess" not in source
        assert "socket" not in source
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in {"mkdir", "write_text", "write_bytes"}:
                raise AssertionError(f"unexpected filesystem write {node.attr}")
