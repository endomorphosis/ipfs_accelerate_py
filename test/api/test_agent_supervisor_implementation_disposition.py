"""Contract tests for implementation disposition and dual-view kernel receipts.

WPD-001 / ImplementationDisposition@1 / PreImplementationKernelReceipt@1
"""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_disposition import (
    DualViewKernelContract,
    ForgedImplementationDispositionIdentityError,
    IMPLEMENTATION_DISPOSITION_EVIDENCE,
    IMPLEMENTATION_DISPOSITION_INTERFACE,
    PRE_IMPLEMENTATION_KERNEL_RECEIPT_INTERFACE,
    ImplementationDisposition,
    ImplementationDispositionAuthorityError,
    ImplementationDispositionError,
    ImplementationForestRoots,
    KernelViewKind,
    PreImplementationKernelReceipt,
    ResidualRequirement,
    UnauthorizedProviderInvocationError,
    assert_provider_invocation_allowed,
    closed_disposition_values,
    closed_dispositions,
    disposition_metric_labels,
    expected_provider_call_count,
    implementation_disposition_cid,
    parse_implementation_disposition,
    provider_invocation_authorized,
    residual_requirement_for,
    seal_pre_implementation_kernel_receipt,
    verify_pre_implementation_kernel_receipt,
)


_CLOSED = frozenset(
    {
        "closed_deterministic",
        "residual_llm_authorized",
        "abstain_review",
        "defer_capability",
    }
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


@pytest.fixture
def dual_view() -> DualViewKernelContract:
    return DualViewKernelContract(
        obligation_graph_cid=_cid("obligations"),
        plan_cid=_cid("plan-view"),
        doctor_cid=_cid("doctor-view"),
        shared_validation_command_cids=(_cid("validate"),),
        shared_edit_packet_cids=(_cid("edit-packet"),),
    )


def _receipt(
    forest_roots: ImplementationForestRoots,
    dual_view: DualViewKernelContract,
    **changes: object,
) -> PreImplementationKernelReceipt:
    values: dict[str, object] = {
        "task_cid": _cid("task"),
        "disposition": ImplementationDisposition.CLOSED_DETERMINISTIC,
        "forest_roots": forest_roots,
        "plan_cid": dual_view.plan_cid,
        "doctor_cid": dual_view.doctor_cid,
        "dual_view": dual_view,
        "attempt": 1,
        "reason_code": "analytical_unique_mapping",
        "evidence_cids": (_cid("evidence"),),
        "policy_revision": "1",
    }
    values.update(changes)
    return PreImplementationKernelReceipt(**values)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Closed disposition vocabulary
# ---------------------------------------------------------------------------


def test_closed_dispositions_match_authority_policy_vocabulary() -> None:
    assert closed_disposition_values() == _CLOSED
    assert {item.value for item in closed_dispositions()} == _CLOSED
    assert set(ImplementationDisposition) == closed_dispositions()
    assert IMPLEMENTATION_DISPOSITION_INTERFACE == "ImplementationDisposition@1"
    assert (
        PRE_IMPLEMENTATION_KERNEL_RECEIPT_INTERFACE
        == "PreImplementationKernelReceipt@1"
    )
    assert IMPLEMENTATION_DISPOSITION_EVIDENCE == "wpd/implementation-disposition@1"


@pytest.mark.parametrize("token", sorted(_CLOSED))
def test_parse_accepts_only_closed_disposition_tokens(token: str) -> None:
    assert parse_implementation_disposition(token).value == token
    assert parse_implementation_disposition(
        ImplementationDisposition(token)
    ).value == token


def test_unknown_disposition_fails_closed() -> None:
    with pytest.raises(ImplementationDispositionError, match="must be one of"):
        parse_implementation_disposition("free_form_llm")
    with pytest.raises(ImplementationDispositionError):
        parse_implementation_disposition("CLOSED_DETERMINISTIC")


def test_provider_authorization_is_residual_only() -> None:
    assert provider_invocation_authorized(
        ImplementationDisposition.RESIDUAL_LLM_AUTHORIZED
    )
    for disposition in (
        ImplementationDisposition.CLOSED_DETERMINISTIC,
        ImplementationDisposition.ABSTAIN_REVIEW,
        ImplementationDisposition.DEFER_CAPABILITY,
    ):
        assert not provider_invocation_authorized(disposition)
        assert expected_provider_call_count(disposition) == 0
        assert residual_requirement_for(disposition) is ResidualRequirement.FORBIDDEN

    assert residual_requirement_for(
        ImplementationDisposition.RESIDUAL_LLM_AUTHORIZED
    ) is ResidualRequirement.REQUIRED
    assert expected_provider_call_count(
        ImplementationDisposition.RESIDUAL_LLM_AUTHORIZED
    ) == -1

    with pytest.raises(UnauthorizedProviderInvocationError):
        assert_provider_invocation_allowed(
            ImplementationDisposition.CLOSED_DETERMINISTIC
        )
    with pytest.raises(ImplementationDispositionAuthorityError, match="residual"):
        assert_provider_invocation_allowed(
            ImplementationDisposition.RESIDUAL_LLM_AUTHORIZED,
            residual_packet_cid="",
        )
    assert (
        assert_provider_invocation_allowed(
            ImplementationDisposition.RESIDUAL_LLM_AUTHORIZED,
            residual_packet_cid=_cid("residual-packet"),
        )
        is ImplementationDisposition.RESIDUAL_LLM_AUTHORIZED
    )


# ---------------------------------------------------------------------------
# Forest roots and dual-view kernel
# ---------------------------------------------------------------------------


def test_forest_roots_are_content_addressed_and_replayable(
    forest_roots: ImplementationForestRoots,
) -> None:
    replayed = ImplementationForestRoots.from_dict(forest_roots.to_record())
    assert replayed == forest_roots
    assert replayed.content_id.startswith("b")
    assert replayed.repository_forest_cid == forest_roots.repository_forest_cid
    assert forest_roots.matches(replayed)


def test_forged_forest_root_identity_fails_closed(
    forest_roots: ImplementationForestRoots,
) -> None:
    payload = forest_roots.to_record()
    payload["content_id"] = "baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    with pytest.raises(ForgedImplementationDispositionIdentityError):
        ImplementationForestRoots.from_dict(payload)


def test_dual_view_binds_distinct_plan_and_doctor_to_one_obligation_graph(
    dual_view: DualViewKernelContract,
) -> None:
    assert dual_view.planner_view_kind is KernelViewKind.PLANNER
    assert dual_view.doctor_view_kind is KernelViewKind.DOCTOR
    assert dual_view.plan_cid != dual_view.doctor_cid
    assert dual_view.binds_plan(dual_view.plan_cid)
    assert dual_view.binds_doctor(dual_view.doctor_cid)
    assert DualViewKernelContract.from_dict(dual_view.to_record()) == dual_view


def test_dual_view_rejects_identical_plan_and_doctor_cids() -> None:
    shared = _cid("same-view")
    with pytest.raises(
        ImplementationDispositionAuthorityError, match="distinct dual-view"
    ):
        DualViewKernelContract(
            obligation_graph_cid=_cid("obligations"),
            plan_cid=shared,
            doctor_cid=shared,
        )


def test_dual_view_rejects_forged_identity(dual_view: DualViewKernelContract) -> None:
    payload = dual_view.to_record()
    payload["cid"] = "baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    with pytest.raises(ForgedImplementationDispositionIdentityError):
        DualViewKernelContract.from_dict(payload)


# ---------------------------------------------------------------------------
# Pre-implementation kernel receipt
# ---------------------------------------------------------------------------


def test_receipt_binds_task_forest_and_plan_doctor_cids(
    forest_roots: ImplementationForestRoots,
    dual_view: DualViewKernelContract,
) -> None:
    receipt = _receipt(forest_roots, dual_view)
    payload = receipt.to_dict()

    assert payload["task_cid"] == receipt.task_cid
    assert payload["forest_roots"]["repository_forest_cid"] == (
        forest_roots.repository_forest_cid
    )
    assert payload["forest_roots"]["git_tree_id"] == forest_roots.git_tree_id
    assert payload["plan_cid"] == dual_view.plan_cid
    assert payload["doctor_cid"] == dual_view.doctor_cid
    assert payload["dual_view"]["obligation_graph_cid"] == (
        dual_view.obligation_graph_cid
    )
    assert payload["disposition"] == "closed_deterministic"
    assert not receipt.authorizes_provider
    assert receipt.repository_forest_cid == forest_roots.repository_forest_cid

    replayed = PreImplementationKernelReceipt.from_dict(receipt.to_record())
    assert replayed == receipt
    assert replayed.content_id == receipt.content_id


@pytest.mark.parametrize(
    ("disposition", "residual"),
    [
        (ImplementationDisposition.CLOSED_DETERMINISTIC, ""),
        (ImplementationDisposition.ABSTAIN_REVIEW, ""),
        (ImplementationDisposition.DEFER_CAPABILITY, ""),
        (ImplementationDisposition.RESIDUAL_LLM_AUTHORIZED, "residual-packet"),
    ],
)
def test_each_closed_disposition_seals(
    forest_roots: ImplementationForestRoots,
    dual_view: DualViewKernelContract,
    disposition: ImplementationDisposition,
    residual: str,
) -> None:
    receipt = seal_pre_implementation_kernel_receipt(
        task_cid=_cid("task"),
        disposition=disposition,
        forest_roots=forest_roots,
        plan_cid=dual_view.plan_cid,
        doctor_cid=dual_view.doctor_cid,
        obligation_graph_cid=dual_view.obligation_graph_cid,
        residual_packet_cid=_cid(residual) if residual else "",
        reason_code=disposition.value,
        shared_validation_command_cids=dual_view.shared_validation_command_cids,
        shared_edit_packet_cids=dual_view.shared_edit_packet_cids,
    )
    assert receipt.disposition is disposition
    assert receipt.plan_cid == dual_view.plan_cid
    assert receipt.doctor_cid == dual_view.doctor_cid
    assert receipt.dual_view.obligation_graph_cid == dual_view.obligation_graph_cid
    labels = disposition_metric_labels(disposition)
    assert labels["disposition"] == disposition.value
    assert labels["provider_authorized"] == (
        "true" if disposition.authorizes_provider else "false"
    )


def test_residual_requires_packet_and_non_residual_forbids_packet(
    forest_roots: ImplementationForestRoots,
    dual_view: DualViewKernelContract,
) -> None:
    with pytest.raises(
        ImplementationDispositionAuthorityError, match="requires residual_packet"
    ):
        _receipt(
            forest_roots,
            dual_view,
            disposition=ImplementationDisposition.RESIDUAL_LLM_AUTHORIZED,
            residual_packet_cid="",
        )

    with pytest.raises(
        ImplementationDispositionAuthorityError, match="forbids residual_packet"
    ):
        _receipt(
            forest_roots,
            dual_view,
            disposition=ImplementationDisposition.CLOSED_DETERMINISTIC,
            residual_packet_cid=_cid("unexpected-packet"),
        )

    residual = _receipt(
        forest_roots,
        dual_view,
        disposition=ImplementationDisposition.RESIDUAL_LLM_AUTHORIZED,
        residual_packet_cid=_cid("residual-packet"),
        reason_code="bounded_residual_syntax",
    )
    assert residual.require_provider_gate() == residual.residual_packet_cid


def test_receipt_rejects_plan_doctor_dual_view_mismatch(
    forest_roots: ImplementationForestRoots,
    dual_view: DualViewKernelContract,
) -> None:
    with pytest.raises(
        ImplementationDispositionAuthorityError, match="plan_cid must match"
    ):
        _receipt(forest_roots, dual_view, plan_cid=_cid("other-plan"))

    with pytest.raises(
        ImplementationDispositionAuthorityError, match="doctor_cid must match"
    ):
        _receipt(forest_roots, dual_view, doctor_cid=_cid("other-doctor"))


def test_forged_receipt_fields_fail_closed(
    forest_roots: ImplementationForestRoots,
    dual_view: DualViewKernelContract,
) -> None:
    receipt = _receipt(forest_roots, dual_view)
    payload = receipt.to_record()

    forged_identity = dict(payload)
    forged_identity["content_id"] = (
        "baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    )
    with pytest.raises(ForgedImplementationDispositionIdentityError):
        PreImplementationKernelReceipt.from_dict(forged_identity)

    unknown_field = dict(payload)
    unknown_field["provider_authorized_override"] = True
    with pytest.raises(ImplementationDispositionError, match="unsupported fields"):
        PreImplementationKernelReceipt.from_dict(unknown_field)

    forged_disposition = dict(payload)
    forged_disposition.pop("content_id", None)
    forged_disposition["disposition"] = "provider_first"
    with pytest.raises(ImplementationDispositionError, match="must be one of"):
        PreImplementationKernelReceipt.from_dict(forged_disposition)

    forged_task = dict(payload)
    forged_task.pop("content_id", None)
    forged_task["task_cid"] = _cid("forged-task")
    # Identity still matches only when recomputed; content_id omitted so decode
    # rebuilds, but verification against expected task fails closed.
    rebuilt = PreImplementationKernelReceipt.from_dict(forged_task)
    with pytest.raises(
        ImplementationDispositionAuthorityError, match="task_cid does not match"
    ):
        verify_pre_implementation_kernel_receipt(
            rebuilt, expected_task_cid=receipt.task_cid
        )


def test_verify_receipt_rebinds_forest_and_provider_gate(
    forest_roots: ImplementationForestRoots,
    dual_view: DualViewKernelContract,
) -> None:
    residual = seal_pre_implementation_kernel_receipt(
        task_cid=_cid("task"),
        disposition=ImplementationDisposition.RESIDUAL_LLM_AUTHORIZED,
        forest_roots=forest_roots,
        plan_cid=dual_view.plan_cid,
        doctor_cid=dual_view.doctor_cid,
        obligation_graph_cid=dual_view.obligation_graph_cid,
        residual_packet_cid=_cid("residual-packet"),
    )
    verified = verify_pre_implementation_kernel_receipt(
        residual.to_record(),
        expected_task_cid=residual.task_cid,
        expected_forest_roots=forest_roots,
        require_provider=True,
    )
    assert verified == residual

    stale_roots = ImplementationForestRoots(
        repository_id=forest_roots.repository_id,
        repository_forest_cid=_cid("stale-forest"),
        git_tree_id=forest_roots.git_tree_id,
        policy_root=forest_roots.policy_root,
    )
    with pytest.raises(ImplementationDispositionAuthorityError, match="stale"):
        verify_pre_implementation_kernel_receipt(
            residual, expected_forest_roots=stale_roots
        )

    closed = _receipt(forest_roots, dual_view)
    with pytest.raises(UnauthorizedProviderInvocationError):
        verify_pre_implementation_kernel_receipt(closed, require_provider=True)


def test_source_bodies_and_secrets_are_rejected(
    forest_roots: ImplementationForestRoots,
    dual_view: DualViewKernelContract,
) -> None:
    payload = _receipt(forest_roots, dual_view).to_record()
    payload.pop("content_id", None)
    payload["prompt_body"] = "exfiltrate secrets"
    with pytest.raises(ImplementationDispositionError, match="secrets or source"):
        PreImplementationKernelReceipt.from_dict(payload)

    with pytest.raises(ImplementationDispositionError, match="compact identifier"):
        ImplementationForestRoots(
            repository_id="repository with spaces",
            repository_forest_cid=_cid("forest"),
            git_tree_id=_cid("tree"),
            policy_root=_cid("policy"),
        )


def test_metric_labels_attribute_closed_paths(
    forest_roots: ImplementationForestRoots,
    dual_view: DualViewKernelContract,
) -> None:
    receipt = _receipt(forest_roots, dual_view)
    labels = receipt.metric_labels()
    assert labels["disposition"] == "closed_deterministic"
    assert labels["provider_authorized"] == "false"
    assert labels["task_cid"] == receipt.task_cid
    assert labels["repository_forest_cid"] == forest_roots.repository_forest_cid
