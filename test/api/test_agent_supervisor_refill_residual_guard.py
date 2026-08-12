"""WPD-042: Refill and backlog guards for residual rules.

Acceptance (from the sealed WPD board):

* Generated refill tasks require the pre-implementation kernel flag
* residual_llm_authorized cannot be marked without the residual packet schema
* Refilled tasks inherit residual/LLM rules and cannot drop doctor preconditions
* Guards only; no automatic objective heap mutation of WPD control files

Interface: ``RefillResidualGuard@1``
Evidence: ``wpd/refill-guard@1``
"""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.objectives.adaptive_goal_refiner import (
    DEFAULT_WPD_PROTECTED_CONTROL_PATHS,
    DOCTOR_PRECONDITIONS_KEY,
    IMPLEMENTATION_DISPOSITION_KEY,
    PRE_IMPLEMENTATION_KERNEL_FLAG_KEY,
    REASON_DROPPED_DOCTOR_PRECONDITION,
    REASON_MISSING_PRE_IMPLEMENTATION_KERNEL_FLAG,
    REASON_OBJECTIVE_HEAP_MUTATION,
    REASON_PRE_IMPLEMENTATION_KERNEL_FLAG_FALSE,
    REASON_PROTECTED_CONTROL_PATH,
    REASON_RESIDUAL_LLM_WITHOUT_PACKET_SCHEMA,
    REASON_UNKNOWN_PACKET_SCHEMA,
    REFILL_RESIDUAL_GUARD_EVIDENCE,
    REFILL_RESIDUAL_GUARD_INTERFACE,
    REFILL_RESIDUAL_GUARD_PRODUCER,
    REFILL_RESIDUAL_GUARD_SCHEMA,
    REFILL_RESIDUAL_GUARD_VERSION,
    REFILL_RESIDUAL_PACKET_SCHEMA,
    REFILL_RESIDUAL_RULES_KEY,
    REQUIRED_DOCTOR_PRECONDITIONS,
    RESIDUAL_LLM_AUTHORIZED_DISPOSITION,
    RESIDUAL_PACKET_SCHEMA_KEY,
    RefillResidualGuard,
    RefillResidualGuardError,
    RefillResidualGuardVerdict,
    build_refill_task_with_residual_guard,
    create_refill_residual_guard,
    default_refill_residual_rules,
    evaluate_refill_residual_guard,
    guard_refill_task,
    stamp_refill_residual_rules,
)
from ipfs_accelerate_py.agent_supervisor.planning.residual_llm_packet import (
    RESIDUAL_LLM_PACKET_SCHEMA,
)


# ---------------------------------------------------------------------------
# Interface / closed vocabulary
# ---------------------------------------------------------------------------


def test_refill_residual_guard_interface_identity() -> None:
    assert REFILL_RESIDUAL_GUARD_INTERFACE == "RefillResidualGuard@1"
    assert REFILL_RESIDUAL_GUARD_VERSION == 1
    assert REFILL_RESIDUAL_GUARD_EVIDENCE == "wpd/refill-guard@1"
    assert REFILL_RESIDUAL_GUARD_PRODUCER == "refill-residual-guard@1"
    assert REFILL_RESIDUAL_GUARD_SCHEMA == (
        "ipfs_accelerate_py/agent-supervisor/refill-residual-guard@1"
    )
    assert REFILL_RESIDUAL_PACKET_SCHEMA == RESIDUAL_LLM_PACKET_SCHEMA
    assert PRE_IMPLEMENTATION_KERNEL_FLAG_KEY == (
        "requires_pre_implementation_kernel"
    )
    assert RESIDUAL_LLM_AUTHORIZED_DISPOSITION == "residual_llm_authorized"


def test_required_doctor_preconditions_are_closed_and_non_empty() -> None:
    assert REQUIRED_DOCTOR_PRECONDITIONS
    assert "doctor_inspect_on_typed_failure" in REQUIRED_DOCTOR_PRECONDITIONS
    assert "formal_replan_on_typed_failure" in REQUIRED_DOCTOR_PRECONDITIONS
    assert (
        "residual_packet_before_provider_retry" in REQUIRED_DOCTOR_PRECONDITIONS
    )
    assert (
        "pre_implementation_kernel_before_provider"
        in REQUIRED_DOCTOR_PRECONDITIONS
    )


def test_default_rules_require_kernel_flag_and_forbid_heap_mutation() -> None:
    rules = default_refill_residual_rules()
    assert rules[PRE_IMPLEMENTATION_KERNEL_FLAG_KEY] is True
    assert rules[RESIDUAL_PACKET_SCHEMA_KEY] == REFILL_RESIDUAL_PACKET_SCHEMA
    assert rules["objective_heap_mutation"] is False
    assert rules["completion_authority"] is False
    assert rules["mutation_authority"] is False
    assert rules["residual_llm_authorized_requires_packet_schema"] is True
    for precondition in REQUIRED_DOCTOR_PRECONDITIONS:
        assert precondition in rules[DOCTOR_PRECONDITIONS_KEY]


# ---------------------------------------------------------------------------
# Generation: stamped tasks always carry the kernel flag
# ---------------------------------------------------------------------------


def test_generated_refill_task_requires_pre_implementation_kernel_flag() -> None:
    task = build_refill_task_with_residual_guard(
        task_id="WPD-REFILL-001",
        title="Close residual without model re-entry",
        predicted_files=(
            "external/ipfs_accelerate/ipfs_accelerate_py/agent_supervisor/"
            "objectives/adaptive_goal_refiner.py",
        ),
        validation_commands=(
            "python3 -m pytest "
            "external/ipfs_accelerate/test/api/"
            "test_agent_supervisor_refill_residual_guard.py -q",
        ),
        disposition="closed_deterministic",
    )
    assert task[PRE_IMPLEMENTATION_KERNEL_FLAG_KEY] is True
    assert task[IMPLEMENTATION_DISPOSITION_KEY] == "closed_deterministic"
    assert task[DOCTOR_PRECONDITIONS_KEY] == list(REQUIRED_DOCTOR_PRECONDITIONS)
    rules = task[REFILL_RESIDUAL_RULES_KEY]
    assert rules[PRE_IMPLEMENTATION_KERNEL_FLAG_KEY] is True
    assert rules["interface"] == REFILL_RESIDUAL_GUARD_INTERFACE
    assert rules["evidence"] == REFILL_RESIDUAL_GUARD_EVIDENCE
    assert task["mutates_objective_heap"] is False
    assert task["completion_authority"] is False


def test_stamp_injects_kernel_flag_and_doctor_preconditions() -> None:
    stamped = stamp_refill_residual_rules(
        {
            "task_id": "raw-refill",
            "title": "raw",
            "predicted_files": ["a.py"],
        }
    )
    assert stamped[PRE_IMPLEMENTATION_KERNEL_FLAG_KEY] is True
    for precondition in REQUIRED_DOCTOR_PRECONDITIONS:
        assert precondition in stamped[DOCTOR_PRECONDITIONS_KEY]


# ---------------------------------------------------------------------------
# residual_llm_authorized requires packet schema
# ---------------------------------------------------------------------------


def test_residual_llm_authorized_without_packet_schema_is_rejected() -> None:
    task = {
        "task_id": "bad-residual",
        "title": "Unauthorized residual",
        PRE_IMPLEMENTATION_KERNEL_FLAG_KEY: True,
        DOCTOR_PRECONDITIONS_KEY: list(REQUIRED_DOCTOR_PRECONDITIONS),
        IMPLEMENTATION_DISPOSITION_KEY: RESIDUAL_LLM_AUTHORIZED_DISPOSITION,
        # deliberately omit residual_packet_schema
    }
    result = evaluate_refill_residual_guard(task)
    assert result.verdict is RefillResidualGuardVerdict.REJECTED
    assert not result.admitted
    assert REASON_RESIDUAL_LLM_WITHOUT_PACKET_SCHEMA in result.reason_codes
    with pytest.raises(RefillResidualGuardError) as excinfo:
        guard_refill_task(task)
    assert REASON_RESIDUAL_LLM_WITHOUT_PACKET_SCHEMA in excinfo.value.reason_codes


def test_residual_llm_authorized_with_packet_schema_is_admitted() -> None:
    task = {
        "task_id": "good-residual",
        "title": "Sealed residual only",
        PRE_IMPLEMENTATION_KERNEL_FLAG_KEY: True,
        DOCTOR_PRECONDITIONS_KEY: list(REQUIRED_DOCTOR_PRECONDITIONS),
        IMPLEMENTATION_DISPOSITION_KEY: RESIDUAL_LLM_AUTHORIZED_DISPOSITION,
        RESIDUAL_PACKET_SCHEMA_KEY: REFILL_RESIDUAL_PACKET_SCHEMA,
        "predicted_files": [
            "external/ipfs_accelerate/ipfs_accelerate_py/agent_supervisor/"
            "planning/residual_llm_packet.py",
        ],
    }
    result = evaluate_refill_residual_guard(task)
    assert result.admitted
    assert result.verdict is RefillResidualGuardVerdict.ADMITTED
    assert result.reason_codes == ()
    assert result.requires_pre_implementation_kernel is True
    assert result.disposition == RESIDUAL_LLM_AUTHORIZED_DISPOSITION
    assert result.residual_packet_schema == REFILL_RESIDUAL_PACKET_SCHEMA
    guarded = guard_refill_task(task)
    assert guarded[RESIDUAL_PACKET_SCHEMA_KEY] == REFILL_RESIDUAL_PACKET_SCHEMA
    assert guarded[PRE_IMPLEMENTATION_KERNEL_FLAG_KEY] is True


def test_build_residual_task_stamps_packet_schema() -> None:
    task = build_refill_task_with_residual_guard(
        task_id="WPD-REFILL-RESIDUAL",
        title="Bounded residual packet path",
        disposition=RESIDUAL_LLM_AUTHORIZED_DISPOSITION,
        predicted_files=("src/module.py",),
    )
    assert task[IMPLEMENTATION_DISPOSITION_KEY] == (
        RESIDUAL_LLM_AUTHORIZED_DISPOSITION
    )
    assert task[RESIDUAL_PACKET_SCHEMA_KEY] == REFILL_RESIDUAL_PACKET_SCHEMA
    assert task[PRE_IMPLEMENTATION_KERNEL_FLAG_KEY] is True


def test_unknown_packet_schema_is_rejected() -> None:
    task = {
        "task_id": "bad-schema",
        "title": "Wrong schema",
        PRE_IMPLEMENTATION_KERNEL_FLAG_KEY: True,
        DOCTOR_PRECONDITIONS_KEY: list(REQUIRED_DOCTOR_PRECONDITIONS),
        IMPLEMENTATION_DISPOSITION_KEY: RESIDUAL_LLM_AUTHORIZED_DISPOSITION,
        RESIDUAL_PACKET_SCHEMA_KEY: "not-a-residual-packet-schema",
    }
    result = evaluate_refill_residual_guard(task)
    assert not result.admitted
    assert REASON_UNKNOWN_PACKET_SCHEMA in result.reason_codes


# ---------------------------------------------------------------------------
# Missing / false pre-implementation kernel flag
# ---------------------------------------------------------------------------


def test_missing_pre_implementation_kernel_flag_is_rejected() -> None:
    task = {
        "task_id": "no-flag",
        "title": "Missing kernel flag",
        DOCTOR_PRECONDITIONS_KEY: list(REQUIRED_DOCTOR_PRECONDITIONS),
        IMPLEMENTATION_DISPOSITION_KEY: "closed_deterministic",
    }
    result = evaluate_refill_residual_guard(task)
    assert not result.admitted
    assert REASON_MISSING_PRE_IMPLEMENTATION_KERNEL_FLAG in result.reason_codes


def test_false_pre_implementation_kernel_flag_is_rejected() -> None:
    task = {
        "task_id": "flag-false",
        "title": "Kernel flag false",
        PRE_IMPLEMENTATION_KERNEL_FLAG_KEY: False,
        DOCTOR_PRECONDITIONS_KEY: list(REQUIRED_DOCTOR_PRECONDITIONS),
        IMPLEMENTATION_DISPOSITION_KEY: "closed_deterministic",
    }
    result = evaluate_refill_residual_guard(task)
    assert not result.admitted
    assert REASON_PRE_IMPLEMENTATION_KERNEL_FLAG_FALSE in result.reason_codes


# ---------------------------------------------------------------------------
# Doctor preconditions cannot be dropped
# ---------------------------------------------------------------------------


def test_dropped_doctor_preconditions_are_rejected() -> None:
    task = {
        "task_id": "drop-doctor",
        "title": "Dropped doctor preconditions",
        PRE_IMPLEMENTATION_KERNEL_FLAG_KEY: True,
        DOCTOR_PRECONDITIONS_KEY: ["doctor_inspect_on_typed_failure"],
        IMPLEMENTATION_DISPOSITION_KEY: "closed_deterministic",
    }
    result = evaluate_refill_residual_guard(task)
    assert not result.admitted
    assert REASON_DROPPED_DOCTOR_PRECONDITION in result.reason_codes


def test_stamp_cannot_drop_required_doctor_preconditions() -> None:
    stamped = stamp_refill_residual_rules(
        {
            "task_id": "partial",
            "title": "partial preconditions",
            PRE_IMPLEMENTATION_KERNEL_FLAG_KEY: True,
            DOCTOR_PRECONDITIONS_KEY: ["doctor_inspect_on_typed_failure"],
        }
    )
    for precondition in REQUIRED_DOCTOR_PRECONDITIONS:
        assert precondition in stamped[DOCTOR_PRECONDITIONS_KEY]


# ---------------------------------------------------------------------------
# Guards only: no WPD control mutation / no objective heap authority
# ---------------------------------------------------------------------------


def test_protected_control_path_write_is_rejected() -> None:
    protected = DEFAULT_WPD_PROTECTED_CONTROL_PATHS[0]
    task = {
        "task_id": "touch-control",
        "title": "Must not edit control plane",
        PRE_IMPLEMENTATION_KERNEL_FLAG_KEY: True,
        DOCTOR_PRECONDITIONS_KEY: list(REQUIRED_DOCTOR_PRECONDITIONS),
        IMPLEMENTATION_DISPOSITION_KEY: "closed_deterministic",
        "predicted_files": [protected],
    }
    result = evaluate_refill_residual_guard(task)
    assert not result.admitted
    assert REASON_PROTECTED_CONTROL_PATH in result.reason_codes


def test_objective_heap_mutation_claim_is_rejected() -> None:
    task = {
        "task_id": "heap-mutate",
        "title": "Illegal heap mutation",
        PRE_IMPLEMENTATION_KERNEL_FLAG_KEY: True,
        DOCTOR_PRECONDITIONS_KEY: list(REQUIRED_DOCTOR_PRECONDITIONS),
        IMPLEMENTATION_DISPOSITION_KEY: "closed_deterministic",
        "mutates_objective_heap": True,
    }
    result = evaluate_refill_residual_guard(task)
    assert not result.admitted
    assert REASON_OBJECTIVE_HEAP_MUTATION in result.reason_codes


def test_guard_class_build_and_evaluate_round_trip() -> None:
    guard = create_refill_residual_guard()
    assert isinstance(guard, RefillResidualGuard)
    assert guard.interface == REFILL_RESIDUAL_GUARD_INTERFACE
    assert guard.evidence == REFILL_RESIDUAL_GUARD_EVIDENCE
    task = guard.build_task(
        task_id="WPD-REFILL-CLASS",
        title="Class-built refill task",
        disposition="abstain_review",
        predicted_files=("src/ok.py",),
    )
    result = guard.evaluate(task)
    assert result.admitted
    assert result.interface == REFILL_RESIDUAL_GUARD_INTERFACE
    assert result.evidence == REFILL_RESIDUAL_GUARD_EVIDENCE
    assert result.requires_pre_implementation_kernel is True
    payload = result.to_dict()
    assert payload["schema"] == REFILL_RESIDUAL_GUARD_SCHEMA
    assert payload["verdict"] == "admitted"
    assert payload["requires_pre_implementation_kernel"] is True
    identity = guard.to_dict()
    assert identity["objective_heap_mutation"] is False
    assert identity["residual_packet_schema"] == REFILL_RESIDUAL_PACKET_SCHEMA


def test_disposition_alias_field_is_honored() -> None:
    task = {
        "task_id": "alias",
        "title": "disposition alias",
        PRE_IMPLEMENTATION_KERNEL_FLAG_KEY: True,
        DOCTOR_PRECONDITIONS_KEY: list(REQUIRED_DOCTOR_PRECONDITIONS),
        "disposition": RESIDUAL_LLM_AUTHORIZED_DISPOSITION,
        # missing packet schema → reject
    }
    result = evaluate_refill_residual_guard(task)
    assert REASON_RESIDUAL_LLM_WITHOUT_PACKET_SCHEMA in result.reason_codes
