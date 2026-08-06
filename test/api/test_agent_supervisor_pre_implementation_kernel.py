"""WPD-020: PreImplementationKernel.evaluate hermetic disposition tests.

Acceptance (from the sealed WPD board):

* Fixture task with unique analytical repair yields ``closed_deterministic``
  and zero provider hooks.
* Ambiguous case yields ``abstain_review``.
* Missing backend yields ``defer_capability``.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_disposition import (
    ImplementationDisposition,
    ImplementationForestRoots,
    implementation_disposition_cid,
    provider_invocation_authorized,
    seal_pre_implementation_kernel_receipt,
    verify_pre_implementation_kernel_receipt,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.pre_implementation_kernel import (
    PRE_IMPLEMENTATION_KERNEL_INTERFACE,
    REASON_AMBIGUOUS_CANDIDATES,
    REASON_ANALYTICAL_UNIQUE_MAPPING,
    REASON_MISSING_BACKEND,
    REASON_RESIDUAL_AUTHORIZED,
    AnalyticalRepairCandidate,
    KernelEvaluationRequest,
    PreImplementationKernel,
    PreImplementationKernelInputError,
    build_pre_implementation_kernel,
    evaluate_pre_implementation,
)


def _cid(name: str) -> str:
    return implementation_disposition_cid({"fixture": name})


@pytest.fixture
def forest_roots() -> ImplementationForestRoots:
    return ImplementationForestRoots(
        repository_id="repository:sha256:test",
        repository_forest_cid=_cid("forest"),
        git_tree_id=_cid("tree"),
        policy_root=_cid("policy"),
        dirty_overlay_cid=_cid("overlay"),
        capability_catalog_root=_cid("capabilities"),
        configuration_root=_cid("config"),
    )


def _request(
    forest_roots: ImplementationForestRoots,
    **changes: Any,
) -> KernelEvaluationRequest:
    values: dict[str, Any] = {
        "task_cid": _cid("task"),
        "forest_roots": forest_roots,
        "attempt": 1,
        "policy_revision": "1",
    }
    values.update(changes)
    return KernelEvaluationRequest(**values)


# ---------------------------------------------------------------------------
# Interface / cold import
# ---------------------------------------------------------------------------


def test_interface_identity_is_stable() -> None:
    assert PRE_IMPLEMENTATION_KERNEL_INTERFACE == "PreImplementationKernel@1"


def test_cold_import_does_not_load_llm_client_modules() -> None:
    llm_markers = (
        "openai",
        "anthropic",
        "litellm",
        "groq",
        "together",
    )
    # The kernel module itself must not pull these in; ambient host packages
    # may already be present under pytest so we only assert the kernel import
    # graph does not *require* them.
    import importlib

    importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.pre_implementation_kernel"
    )
    # No hard dependency: module source must not reference client packages.
    import inspect
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
        pre_implementation_kernel as mod,
    )

    source = inspect.getsource(mod)
    for marker in llm_markers:
        assert marker not in source


# ---------------------------------------------------------------------------
# Unique analytical close → closed_deterministic, zero provider hooks
# ---------------------------------------------------------------------------


def test_unique_analytical_repair_closes_deterministic(
    forest_roots: ImplementationForestRoots,
) -> None:
    candidate = AnalyticalRepairCandidate(
        candidate_id="repair-1",
        reason_code=REASON_ANALYTICAL_UNIQUE_MAPPING,
        closes_claim=True,
        evidence_cids=(_cid("evidence-1"),),
    )
    result = evaluate_pre_implementation(
        _request(
            forest_roots,
            analytical_candidates=(candidate,),
        )
    )

    assert result.disposition is ImplementationDisposition.CLOSED_DETERMINISTIC
    assert result.provider_hook_count == 0
    assert result.analytical_candidate_count == 1
    assert result.reason_code == REASON_ANALYTICAL_UNIQUE_MAPPING
    assert not result.authorizes_provider
    assert not provider_invocation_authorized(result.disposition)
    assert result.receipt.residual_packet_cid == ""
    assert _cid("evidence-1") in result.receipt.evidence_cids
    # Round-trip seal
    verified = verify_pre_implementation_kernel_receipt(result.receipt.to_dict())
    assert verified.disposition is ImplementationDisposition.CLOSED_DETERMINISTIC


def test_analytical_probe_unique_mapping_closes(
    forest_roots: ImplementationForestRoots,
) -> None:
    def probe(_request: KernelEvaluationRequest) -> list[AnalyticalRepairCandidate]:
        return [
            AnalyticalRepairCandidate(
                candidate_id="probe-unique",
                reason_code=REASON_ANALYTICAL_UNIQUE_MAPPING,
            )
        ]

    kernel = PreImplementationKernel(analytical_probe=probe)
    result = kernel.evaluate(_request(forest_roots))
    assert result.disposition is ImplementationDisposition.CLOSED_DETERMINISTIC
    assert result.provider_hook_count == 0


# ---------------------------------------------------------------------------
# Ambiguous candidates → abstain_review
# ---------------------------------------------------------------------------


def test_ambiguous_candidates_abstain_review(
    forest_roots: ImplementationForestRoots,
) -> None:
    result = evaluate_pre_implementation(
        _request(
            forest_roots,
            analytical_candidates=(
                AnalyticalRepairCandidate(candidate_id="a", closes_claim=True),
                AnalyticalRepairCandidate(candidate_id="b", closes_claim=True),
            ),
        )
    )
    assert result.disposition is ImplementationDisposition.ABSTAIN_REVIEW
    assert result.provider_hook_count == 0
    assert result.reason_code == REASON_AMBIGUOUS_CANDIDATES
    assert not result.authorizes_provider
    assert result.receipt.residual_packet_cid == ""


# ---------------------------------------------------------------------------
# Missing backend → defer_capability
# ---------------------------------------------------------------------------


def test_missing_planner_backend_defers(
    forest_roots: ImplementationForestRoots,
) -> None:
    kernel = build_pre_implementation_kernel(planner_available=False)
    result = kernel.evaluate(
        _request(
            forest_roots,
            analytical_candidates=(
                AnalyticalRepairCandidate(candidate_id="would-close"),
            ),
        )
    )
    assert result.disposition is ImplementationDisposition.DEFER_CAPABILITY
    assert result.provider_hook_count == 0
    assert result.reason_code == REASON_MISSING_BACKEND
    assert not result.authorizes_provider


def test_missing_doctor_backend_via_request_flag_defers(
    forest_roots: ImplementationForestRoots,
) -> None:
    result = evaluate_pre_implementation(
        _request(forest_roots, doctor_available=False),
        doctor_available=True,  # request override wins
    )
    assert result.disposition is ImplementationDisposition.DEFER_CAPABILITY
    assert result.reason_code == REASON_MISSING_BACKEND
    assert result.provider_hook_count == 0


# ---------------------------------------------------------------------------
# Residual authorization path
# ---------------------------------------------------------------------------


def test_residual_packet_authorizes_provider_without_kernel_hooks(
    forest_roots: ImplementationForestRoots,
) -> None:
    packet_cid = _cid("residual-packet")
    result = evaluate_pre_implementation(
        _request(forest_roots, residual_packet_cid=packet_cid)
    )
    assert result.disposition is ImplementationDisposition.RESIDUAL_LLM_AUTHORIZED
    assert result.provider_hook_count == 0  # kernel itself never calls provider
    assert result.reason_code == REASON_RESIDUAL_AUTHORIZED
    assert result.authorizes_provider
    assert result.receipt.residual_packet_cid == packet_cid
    assert result.receipt.require_provider_gate() == packet_cid


# ---------------------------------------------------------------------------
# Input validation / fail closed
# ---------------------------------------------------------------------------


def test_missing_task_cid_fails_closed(
    forest_roots: ImplementationForestRoots,
) -> None:
    with pytest.raises(PreImplementationKernelInputError, match="task_cid"):
        KernelEvaluationRequest(task_cid="", forest_roots=forest_roots)


def test_mapping_request_round_trip(
    forest_roots: ImplementationForestRoots,
) -> None:
    result = evaluate_pre_implementation(
        {
            "task_cid": _cid("map-task"),
            "forest_roots": forest_roots.to_dict(),
            "analytical_candidates": [
                {
                    "candidate_id": "only",
                    "reason_code": REASON_ANALYTICAL_UNIQUE_MAPPING,
                    "closes_claim": True,
                }
            ],
        }
    )
    assert result.disposition is ImplementationDisposition.CLOSED_DETERMINISTIC
    assert result.provider_hook_count == 0


def test_no_candidates_and_no_residual_abstains(
    forest_roots: ImplementationForestRoots,
) -> None:
    result = evaluate_pre_implementation(_request(forest_roots))
    assert result.disposition is ImplementationDisposition.ABSTAIN_REVIEW
    assert result.provider_hook_count == 0


def test_receipt_is_body_free_and_content_addressed(
    forest_roots: ImplementationForestRoots,
) -> None:
    result = evaluate_pre_implementation(
        _request(
            forest_roots,
            analytical_candidates=(
                AnalyticalRepairCandidate(candidate_id="only"),
            ),
        )
    )
    payload = result.receipt.to_dict()
    # Top-level keys must not smuggle source/prompt bodies.
    for forbidden in ("source", "prompt", "body", "transcript", "code", "source_body"):
        assert forbidden not in payload
    assert "reason_code" in payload  # identifier field is allowed
    assert payload["content_id"] == result.receipt.content_id
    # Manual seal of the same dual-view identity must verify.
    sealed = seal_pre_implementation_kernel_receipt(
        task_cid=result.receipt.task_cid,
        disposition=result.receipt.disposition,
        forest_roots=result.receipt.forest_roots,
        plan_cid=result.receipt.plan_cid,
        doctor_cid=result.receipt.doctor_cid,
        obligation_graph_cid=result.receipt.dual_view.obligation_graph_cid,
        attempt=result.receipt.attempt,
        reason_code=result.receipt.reason_code,
        evidence_cids=result.receipt.evidence_cids,
        policy_revision=result.receipt.policy_revision,
        producer_id=result.receipt.producer_id,
    )
    assert sealed.disposition is ImplementationDisposition.CLOSED_DETERMINISTIC


def test_llm_client_modules_not_required_for_import() -> None:
    """Ensure the kernel module graph does not import optional LLM clients."""

    # Snapshot of modules that must not appear solely due to kernel import.
    before = set(sys.modules)
    import importlib

    importlib.reload(
        importlib.import_module(
            "ipfs_accelerate_py.agent_supervisor.todo_daemon.pre_implementation_kernel"
        )
    )
    added = set(sys.modules) - before
    forbidden = {
        name
        for name in added
        if name.split(".")[0]
        in {"openai", "anthropic", "litellm", "google", "cohere"}
    }
    assert not forbidden
