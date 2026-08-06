"""WPD-003 conformance tests for Worker Planner–Doctor authority floors."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pytest

# Test file: external/ipfs_accelerate/test/api/<this file>
# parents[2] = accelerate package root
_ACCELERATE_ROOT = Path(__file__).resolve().parents[2]


def _discover_monorepo_root() -> Path:
    """Locate the monorepo root that owns the WPD authority policy."""

    here = Path(__file__).resolve()
    candidates: list[Path] = []
    # Prefer parents[4] (monorepo/external/ipfs_accelerate/test/api/...).
    if len(here.parents) > 4:
        candidates.append(here.parents[4])
    candidates.extend(here.parents)
    seen: set[Path] = set()
    for root in candidates:
        if root in seen:
            continue
        seen.add(root)
        policy = (
            root / "config" / "supervisor_worker_planner_doctor_authority_policy.json"
        )
        if policy.is_file():
            return root
    # Fall back to parents[4] so failure messages remain path-local.
    return here.parents[4] if len(here.parents) > 4 else here.parents[-1]


_MONOREPO_ROOT = _discover_monorepo_root()

_POLICY_PATH = (
    _MONOREPO_ROOT / "config" / "supervisor_worker_planner_doctor_authority_policy.json"
)
_THREAT_MODEL_PATH = (
    _ACCELERATE_ROOT
    / "docs"
    / "architecture"
    / "agent_supervisor_worker_planner_doctor_threat_model.md"
)
_TEST_REL = (
    "external/ipfs_accelerate/test/api/"
    "test_agent_supervisor_worker_planner_doctor_authority_policy.py"
)
_POLICY_REL = "config/supervisor_worker_planner_doctor_authority_policy.json"
_THREAT_REL = (
    "external/ipfs_accelerate/docs/architecture/"
    "agent_supervisor_worker_planner_doctor_threat_model.md"
)

_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.worker_planner_doctor.authority_policy@1"
)
_INTERFACE = "WorkerPlannerDoctorAuthorityPolicy@1"
_THREAT_INTERFACE = "WorkerPlannerDoctorThreatModel@1"
_BOARD_NAMESPACE = "agent-supervisor-worker-planner-doctor-v1"
_TASK_ID = "WPD-003"

# Acceptance-critical forbidden transitions (machine-checked floors).
_REQUIRED_FORBIDDEN_TRANSITIONS = frozenset(
    {
        "unauthorized_llm_invocation",
        "deterministic_mode_network_or_llm_load",
        "completion_from_provider_prose",
        "free_reprompt_after_typed_failure",
        "candidate_or_self_certification_to_authority",
    }
)

# Non-compensable safety floors from plan §8 and acceptance criteria.
_NON_COMPENSABLE_FLOORS = frozenset(
    {
        "unauthorized_llm_calls",
        "deterministic_doctor_network_or_llm_load",
        "completion_from_provider_prose",
        "free_reprompt_after_typed_failure",
        "candidate_self_certification_count",
        "false_fixed_point_count",
        "policy_scope_escape_count",
    }
)

_REQUIRED_SAFETY_FLOORS = frozenset(
    {
        *_NON_COMPENSABLE_FLOORS,
        "authority_violation_count",
        "deterministic_llm_invocation_count",
        "deterministic_network_access_count",
        "deterministic_remote_embedding_call_count",
        "deterministic_remote_provider_call_count",
        "false_completion_count",
        "false_fixed_point_rate",
        "llm_model_provider_call_rate",
        "llm_router_invocation_rate",
        "out_of_scope_sandbox_write_rate",
        "path_escape_count",
        "protected_anchor_mutation_count",
        "secret_exposure_count",
        "completion_from_synthetic_doctor_success",
        "authority_promotion_rate",
    }
)

_CLOSED_DISPOSITIONS = frozenset(
    {
        "closed_deterministic",
        "residual_llm_authorized",
        "abstain_review",
        "defer_capability",
    }
)

_DETERMINISTIC_HARD_FALSE = frozenset(
    {
        "llm_router_enabled",
        "llm_invocations_allowed",
        "remote_model_provider_calls_allowed",
        "remote_embeddings_allowed",
        "network_access_allowed",
        "network_client_import_allowed",
        "target_code_import_allowed",
        "knowledge_graph_semantic_authority",
        "vector_semantic_authority",
        "embedding_semantic_authority",
        "proof_cache_metadata_semantic_authority",
    }
)

_PROTECTED_CONTROL_PATHS = frozenset(
    {
        "implementation_plan/docs/47-supervisor-worker-planner-doctor-integration-plan-2026-08-06.md",
        "implementation_plan/docs/47-supervisor-worker-planner-doctor-integration.objectives.md",
        "implementation_plan/docs/47-supervisor-worker-planner-doctor-integration.todo.md",
        "config/supervisor_worker_planner_doctor_integration_scheduler.json",
        "config/supervisor_worker_planner_doctor_supervisor.json",
        "scripts/validate_supervisor_worker_planner_doctor_board.py",
        "scripts/supervisor_worker_planner_doctor_supervisor.sh",
    }
)


class _DuplicateKey(ValueError):
    """Raised when a JSON object is not a unique-key mapping."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey(key)
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    def reject_number(value: str) -> Any:
        raise ValueError(f"non-canonical JSON number: {value}")

    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
        parse_float=reject_number,
        parse_constant=reject_number,
    )
    assert isinstance(payload, dict)
    return payload


def _walk(value: Any) -> Iterable[Any]:
    yield value
    if isinstance(value, Mapping):
        for key in sorted(value):
            yield from _walk(value[key])
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk(item)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _transition_ids(policy: Mapping[str, Any]) -> set[str]:
    return {str(item["id"]) for item in policy["forbidden_transitions"]}


def _transition_by_id(policy: Mapping[str, Any], transition_id: str) -> Mapping[str, Any]:
    for item in policy["forbidden_transitions"]:
        if item["id"] == transition_id:
            assert isinstance(item, Mapping)
            return item
    raise KeyError(transition_id)


def test_declared_outputs_and_interfaces_exist() -> None:
    assert _POLICY_PATH.is_file(), f"missing policy at {_POLICY_PATH}"
    assert _THREAT_MODEL_PATH.is_file(), f"missing threat model at {_THREAT_MODEL_PATH}"
    assert Path(__file__).is_file()

    policy = _load_json(_POLICY_PATH)
    text = _THREAT_MODEL_PATH.read_text(encoding="utf-8")

    assert policy["schema"] == _SCHEMA
    assert policy["interface"] == _INTERFACE
    assert policy["task_id"] == _TASK_ID
    assert policy["board_namespace"] == _BOARD_NAMESPACE
    assert policy["policy_revision"] == "1"
    assert f"Interface: `{_THREAT_INTERFACE}`" in text
    assert f"Policy interface: `{_INTERFACE}`" in text
    assert "WPD-003" in text

    bindings = policy["artifact_bindings"]
    assert bindings["authority_policy"] == _POLICY_REL
    assert bindings["threat_model"] == _THREAT_REL
    assert bindings["authority_test"] == _TEST_REL


def test_policy_has_deterministic_unique_key_canonical_preimage() -> None:
    policy = _load_json(_POLICY_PATH)
    values = tuple(_walk(policy))
    assert not any(isinstance(value, float) for value in values)
    assert policy["canonicalization"] == {
        "encoding": "utf-8",
        "profile": "sorted-key compact JSON with no insignificant whitespace",
        "duplicate_keys_forbidden": True,
        "floating_point_forbidden": True,
        "non_finite_numbers_forbidden": True,
        "content_identity": "sha2-256 over canonical UTF-8 policy bytes",
        "stored_self_identity": False,
    }

    first = _canonical_bytes(policy)
    second = _canonical_bytes(_load_json(_POLICY_PATH))
    assert first == second
    assert hashlib.sha256(first).hexdigest() == hashlib.sha256(second).hexdigest()
    assert b"NaN" not in first and b"Infinity" not in first


def test_defaults_and_lifecycle_fail_closed() -> None:
    policy = _load_json(_POLICY_PATH)
    defaults = policy["defaults"]
    lifecycle = policy["lifecycle"]

    assert defaults["planner_mode"] == "shadow"
    assert defaults["doctor_mode"] == "report_only"
    assert defaults["rollout_mode"] == "shadow"
    assert defaults["mutation_authorized"] is False
    assert defaults["automatic_promotion_enabled"] is False
    assert defaults["derived_refill_enabled"] is False
    assert defaults["completion_authoritative"] is False
    assert defaults["provider_first_path_enabled"] is False
    assert defaults["free_reprompt_enabled"] is False
    assert defaults["unknown_or_unsupported_disposition"] == "abstain"
    assert defaults["ambiguous_disposition"] == "abstain"

    assert lifecycle["completion_mode"] == "manual"
    assert lifecycle["self_sealing_forbidden"] is True
    assert lifecycle["immutable_after_manual_completion"] is True
    assert lifecycle["candidate_mutation_forbidden"] is True
    assert lifecycle["policy_root_protected_after_review"] is True


def test_authority_ladder_never_grants_mutation_or_completion() -> None:
    policy = _load_json(_POLICY_PATH)
    ladder = policy["authority_ladder"]

    assert [item["name"] for item in ladder] == [
        "nomination",
        "observation",
        "bounded_check",
        "kernel_proof",
        "attestation",
    ]
    assert [item["order"] for item in ladder] == list(range(5))
    for stage in ladder:
        assert stage["mutation_authority"] is False
        assert stage["completion_authority"] is False

    nomination = ladder[0]
    assert nomination["semantic_authority"] is False
    assert {
        "admit",
        "authorize_write",
        "certify_self",
        "claim_completion",
        "invoke_unauthorized_llm",
        "promote",
        "satisfy_proof",
    } <= set(nomination["may_not"])


def test_pre_implementation_kernel_closes_dispositions_and_provider_gate() -> None:
    policy = _load_json(_POLICY_PATH)
    kernel = policy["pre_implementation_kernel"]
    bindings = policy["contract_bindings"]

    assert kernel["required_before_provider_call"] is True
    assert kernel["required_on_validation_or_merge_failure"] is True
    assert kernel["dispositions_closed"] is True
    assert set(kernel["closed_dispositions"]) == _CLOSED_DISPOSITIONS
    assert set(bindings["implementation_dispositions"]) == _CLOSED_DISPOSITIONS
    assert kernel["provider_invocation_requires_disposition"] == "residual_llm_authorized"
    assert kernel["closed_deterministic_provider_calls_forbidden"] is True
    assert kernel["abstain_review_provider_calls_forbidden"] is True
    assert kernel["defer_capability_provider_calls_forbidden"] is True
    assert kernel["free_form_provider_rediscovery_forbidden"] is True
    assert bindings["new_authority_lattice_forbidden"] is True
    assert bindings["reuse_existing_completion_authority"] is True


def test_deterministic_doctor_boundary_is_hard_false() -> None:
    policy = _load_json(_POLICY_PATH)
    boundary = policy["deterministic_doctor_boundary"]

    assert boundary["mode"] == "report_only"
    assert boundary["enabled_at_bootstrap"] is False
    assert boundary["mutation_authorized"] is False
    assert boundary["automatic_fallback_to_model_forbidden"] is True
    assert boundary["optional_provider_failure_disposition"] == "typed_abstention"
    assert boundary["cold_import_must_not_load_network_or_llm_clients"] is True
    for name in _DETERMINISTIC_HARD_FALSE:
        assert boundary[name] is False, name


def test_residual_llm_and_free_reprompt_policies_forbid_acceptance_violations() -> None:
    policy = _load_json(_POLICY_PATH)
    residual = policy["residual_llm_boundary"]
    free = policy["free_reprompt_policy"]
    completion = policy["completion_policy"]
    self_cert = policy["self_certification_policy"]

    assert residual["distinct_from_deterministic_doctor"] is True
    assert residual["enabled_by_default"] is False
    assert residual["requires_sealed_residual_packet"] is True
    assert residual["requires_disposition_residual_llm_authorized"] is True
    assert residual["model_output_authority"] == "nomination"
    assert residual["model_write_authority"] is False
    assert residual["model_completion_authority"] is False
    assert residual["completion_from_provider_prose_forbidden"] is True
    assert residual["completion_from_synthetic_doctor_success_forbidden"] is True
    assert residual["inherits_deterministic_doctor_authority"] is False

    assert free["free_reprompt_after_typed_failure_forbidden"] is True
    assert free["typed_failure_requires_doctor_inspect"] is True
    assert free["typed_failure_requires_formal_replan"] is True
    assert free["typed_failure_requires_residual_packet_before_provider"] is True
    assert free["provider_retry_without_residual_packet_forbidden"] is True
    assert free["full_task_prose_reinjection_forbidden"] is True
    assert free["disposition_on_violation"] == "reject"

    assert completion["provider_prose_is_completion_authority"] is False
    assert completion["provider_exit_zero_is_completion_authority"] is False
    assert completion["synthetic_doctor_success_is_completion_authority"] is False
    assert completion["candidate_claim_is_completion_authority"] is False
    assert completion["task_status_is_completion_authority"] is False
    assert completion["zero_safety_floors_required"] is True
    assert completion["completion_authority_owner"] == "existing_goal_completion_contract"

    assert self_cert["candidate_self_certification_forbidden"] is True
    assert self_cert["candidate_may_supply_own_proof_flag"] is False
    assert self_cert["candidate_may_supply_own_expected_oracle"] is False
    assert self_cert["candidate_may_issue_own_admission_receipt"] is False
    assert self_cert["candidate_may_mark_own_task_complete"] is False
    assert self_cert["generated_test_may_certify_its_generator"] is False
    assert self_cert["disposition"] == "reject"


def test_mutation_boundary_requires_permit_lease_and_exact_roots() -> None:
    policy = _load_json(_POLICY_PATH)
    mutation = policy["mutation_boundary"]

    assert mutation["write_requires_mutation_permit"] is True
    assert mutation["write_requires_writer_lease"] is True
    assert mutation["write_requires_exact_roots"] is True
    assert mutation["write_requires_fencing_epoch"] is True
    assert mutation["write_requires_expected_effects"] is True
    assert mutation["write_requires_checkpoint_and_rollback"] is True
    assert mutation["path_escape_forbidden"] is True
    assert mutation["scope_widening_forbidden"] is True
    assert mutation["delegation_forbidden"] is True


def test_protected_anchors_include_wpd_control_plane() -> None:
    policy = _load_json(_POLICY_PATH)
    anchors = policy["protected_anchors"]
    paths = set(anchors["paths"])

    assert _PROTECTED_CONTROL_PATHS <= paths
    assert _POLICY_REL in paths
    assert _THREAT_REL in paths
    assert _TEST_REL in paths
    assert anchors["candidate_write_allowed"] is False
    assert anchors["model_write_allowed"] is False
    assert anchors["automatic_controller_write_allowed"] is False
    assert anchors["manual_operator_change_requires_new_policy_revision"] is True
    assert {
        "authority_policy.safety_floors",
        "authority_policy.forbidden_transitions",
    } <= set(anchors["protected_fields"])


def test_required_forbidden_transitions_are_closed_and_floor_bound() -> None:
    policy = _load_json(_POLICY_PATH)
    ids = _transition_ids(policy)

    assert _REQUIRED_FORBIDDEN_TRANSITIONS <= ids
    for transition in policy["forbidden_transitions"]:
        assert transition["disposition"] in {
            "reject",
            "reject_quarantine_and_escalate",
            "rollback_or_quarantine",
        }
        assert isinstance(transition["floor_keys"], list)
        assert transition["floor_keys"]
        for key in transition["floor_keys"]:
            assert key in policy["safety_floors"], key

    # Acceptance criteria → exact transition bindings.
    assert "unauthorized_llm_calls" in _transition_by_id(
        policy, "unauthorized_llm_invocation"
    )["floor_keys"]
    assert "deterministic_doctor_network_or_llm_load" in _transition_by_id(
        policy, "deterministic_mode_network_or_llm_load"
    )["floor_keys"]
    assert "completion_from_provider_prose" in _transition_by_id(
        policy, "completion_from_provider_prose"
    )["floor_keys"]
    assert "free_reprompt_after_typed_failure" in _transition_by_id(
        policy, "free_reprompt_after_typed_failure"
    )["floor_keys"]
    assert "candidate_self_certification_count" in _transition_by_id(
        policy, "candidate_or_self_certification_to_authority"
    )["floor_keys"]


def test_safety_floors_are_exact_integer_zero_and_non_compensable() -> None:
    policy = _load_json(_POLICY_PATH)
    floors = policy["safety_floors"]
    non_compensable = set(policy["non_compensable_floors"])

    assert floors
    assert _REQUIRED_SAFETY_FLOORS <= set(floors)
    assert _NON_COMPENSABLE_FLOORS <= non_compensable
    assert non_compensable <= set(floors)

    for name, value in floors.items():
        assert type(value) is int, name
        assert value == 0, name

    # Explicit machine checks for acceptance floors.
    assert floors["unauthorized_llm_calls"] == 0
    assert floors["deterministic_doctor_network_or_llm_load"] == 0
    assert floors["completion_from_provider_prose"] == 0
    assert floors["free_reprompt_after_typed_failure"] == 0
    assert floors["candidate_self_certification_count"] == 0
    assert floors["false_fixed_point_count"] == 0
    assert floors["policy_scope_escape_count"] == 0


def test_floor_breach_is_detectable_as_machine_check() -> None:
    """Simulate a nonzero floor and prove the machine check rejects it."""

    policy = _load_json(_POLICY_PATH)
    floors = dict(policy["safety_floors"])

    def floors_hold(values: Mapping[str, int]) -> bool:
        return all(type(v) is int and v == 0 for v in values.values())

    assert floors_hold(floors) is True

    for key in sorted(_NON_COMPENSABLE_FLOORS):
        breached = dict(floors)
        breached[key] = 1
        assert floors_hold(breached) is False, key


def test_kill_switch_and_escalation_fail_closed() -> None:
    policy = _load_json(_POLICY_PATH)
    kill = policy["kill_switch"]
    escalation = policy["manual_escalation"]
    promotion = policy["promotion_policy"]

    assert kill["available"] is True
    assert kill["automatic_engage_on_noncompensable_gate_failure"] is True
    assert kill["candidate_may_clear"] is False
    assert kill["model_may_clear"] is False
    assert kill["automatic_controller_may_clear"] is False
    assert kill["operator_only_clear"] is True
    assert kill["blocks_model_provider_calls"] is True
    assert kill["engaged_mode"] == "report_only"

    assert escalation["candidate_approval_allowed"] is False
    assert escalation["model_approval_allowed"] is False
    assert escalation["automatic_self_approval_allowed"] is False
    assert {
        "free_reprompt_policy_violation",
        "unauthorized_llm_invocation",
        "zero_floor_breach",
        "protected_anchor_change",
    } <= set(escalation["triggers"])

    assert promotion["default_decision"] == "shadow"
    assert promotion["manual"] is True
    assert promotion["synthetic_evidence_may_promote"] is False
    assert promotion["quality_non_compensable"] is True
    assert promotion["candidate_may_change_judge"] is False


def test_threat_model_documents_acceptance_floors_and_transitions() -> None:
    text = _THREAT_MODEL_PATH.read_text(encoding="utf-8")
    policy = _load_json(_POLICY_PATH)

    for key in _NON_COMPENSABLE_FLOORS:
        assert f"`{key}`" in text, key

    for transition_id in _REQUIRED_FORBIDDEN_TRANSITIONS:
        assert f"`{transition_id}`" in text, transition_id

    assert "non-compensable" in text.lower() or "Non-compensable" in text
    assert "residual_llm_authorized" in text
    assert "config/supervisor_worker_planner_doctor_authority_policy.json" in text
    assert policy["interface"] in text
    assert "false fixed point" in text.lower() or "false_fixed_point" in text
    assert "scope escape" in text.lower() or "policy_scope_escape" in text


def test_policy_rejects_float_and_duplicate_key_payloads() -> None:
    raw = _POLICY_PATH.read_text(encoding="utf-8")
    # Structural integrity: re-parse must remain integer-only.
    policy = _load_json(_POLICY_PATH)
    assert "0.0" not in raw or all(
        type(v) is int for v in policy["safety_floors"].values()
    )

    with pytest.raises(_DuplicateKey):
        json.loads(
            '{"a": 1, "a": 2}',
            object_pairs_hook=_unique_object,
        )
