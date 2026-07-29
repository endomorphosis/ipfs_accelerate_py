from __future__ import annotations

from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.context.decision_contracts import (
    ApplicabilityFact,
    ApplicabilityFactKind,
)
from ipfs_accelerate_py.agent_supervisor.proof.ir_adapters import IRAdapterRegistry
from ipfs_accelerate_py.agent_supervisor.proof.ir_registry import (
    IRFamily,
    IRLoadRequest,
    IRLoadStatus,
    IRRegistry,
    deterministic_ir_fixture,
)
from ipfs_accelerate_py.agent_supervisor.proof.legal_constraint_adapter import (
    LEGAL_APPLICABILITY_REQUIREMENT_ID,
    LegalApplicabilityOutcome,
    LegalApplicabilityQuery,
    LegalCompilationStatus,
    LegalConstraintAdapter,
    LegalConstraintError,
    LegalModality,
    compile_legal_constraints,
)


_SCOPE = {
    "jurisdiction": "US-CA",
    "subject": "source-code",
    "principal": "agent:builder",
    "action": "write",
    "resource": "repository:alpha",
    "effect": "file:update",
}
_SOURCE = ({"source_id": "statute:ca-1", "span_id": "section:10"},)


def _norm(
    provision_id: str,
    modality: str,
    **updates: object,
) -> dict[str, object]:
    result: dict[str, object] = {
        "declaration_id": provision_id,
        "kind": "norm",
        "modality": modality,
        **_SCOPE,
        "effective_from_ms": 100,
        "effective_until_ms": 1000,
        "source_references": _SOURCE,
    }
    result.update(updates)
    return result


def _normalized(
    declarations: tuple[dict[str, object], ...],
    *,
    formal_views: tuple[dict[str, object], ...] = (),
    assumptions: tuple[dict[str, object], ...] = (),
    obligations: tuple[dict[str, object], ...] = (),
):
    reference, encoded = deterministic_ir_fixture(
        IRFamily.LEGAL,
        declarations=declarations,
        formal_views=formal_views,
        assumptions=assumptions,
        obligations=obligations,
    )
    registry = IRRegistry()
    registry.register_local_artifact(reference, encoded)
    loaded = registry.load(
        IRLoadRequest(reference=reference, family=IRFamily.LEGAL)
    )
    assert loaded.status is IRLoadStatus.VERIFIED
    return IRAdapterRegistry().normalize(loaded).require_artifact()


def _query(artifact, **updates: object) -> LegalApplicabilityQuery:
    values: dict[str, object] = {
        "legal_root_artifact_id": artifact.root_artifact_id,
        "legal_root_cid_v1": artifact.root_cid_v1,
        "legal_root_supervisor_digest": artifact.root_supervisor_digest,
        **_SCOPE,
        "effective_at_ms": 500,
    }
    values.update(updates)
    return LegalApplicabilityQuery(**values)


def test_exact_selection_compiles_norms_views_dependencies_and_source_bindings() -> None:
    artifact = _normalized(
        (
            _norm(
                "norm:obligation",
                "obligation",
                formal_view_ids=("view:deontic",),
                assumption_ids=("assumption:reviewed-source",),
                proof_obligation_ids=("proof:compliance",),
            ),
            _norm(
                "norm:similar-but-wrong-jurisdiction",
                "permission",
                jurisdiction="US-NY",
            ),
            _norm(
                "norm:similar-but-wrong-action",
                "permission",
                action="delete",
            ),
        ),
        formal_views=(
            {
                "view_id": "view:deontic",
                "view_kind": "deontic",
                "source_references": _SOURCE,
            },
        ),
        assumptions=(
            {
                "assumption_id": "assumption:reviewed-source",
                "kind": "source_authenticity",
                "source_references": _SOURCE,
            },
        ),
        obligations=(
            {
                "obligation_id": "proof:compliance",
                "kind": "proof",
                "provision_ids": ("norm:obligation",),
                "discharged": False,
                "source_references": _SOURCE,
            },
        ),
    )
    query = _query(
        artifact,
        # Retrieval nominated only the similar, inapplicable provision.  The
        # compiler must still scan and select the authoritative obligation.
        semantic_candidate_ids=("norm:similar-but-wrong-action",),
    )

    result = compile_legal_constraints(artifact, query)

    assert result.status is LegalCompilationStatus.COMPLETE
    assert result.outcome is LegalApplicabilityOutcome.APPLICABLE
    assert [item.provision_id for item in result.obligations] == [
        "norm:obligation"
    ]
    assert {item.provision_id for item in result.inapplicable} == {
        "norm:similar-but-wrong-action",
        "norm:similar-but-wrong-jurisdiction",
    }
    assert result.selected_formal_view_ids == ("view:deontic",)
    assert [item.provision_id for item in result.assumptions] == [
        "assumption:reviewed-source"
    ]
    assert [item.obligation_id for item in result.proof_obligations] == [
        "proof:compliance"
    ]
    constraint = result.obligations[0]
    assert constraint.source_binding.legal_root_cid_v1 == artifact.root_cid_v1
    assert constraint.source_binding.source_references
    assert constraint.source_binding.provenance_references
    assert result.authoritative_scan_complete
    assert not result.to_dict()["semantic_candidates_are_authority"]
    assert (
        result.to_dict()["requirement_id"]
        == LEGAL_APPLICABILITY_REQUIREMENT_ID
    )


def test_legal_permission_without_securityir_authorization_never_admits_action() -> None:
    artifact = _normalized((_norm("norm:permission", "permission"),))

    result = LegalConstraintAdapter().compile(artifact, _query(artifact))

    assert result.status is LegalCompilationStatus.COMPLETE
    assert result.legally_permitted
    assert [item.modality for item in result.permissions] == [
        LegalModality.PERMISSION
    ]
    assert not result.grants_security_authorization
    assert not result.grants_execution_authority
    assert not result.action_admitted
    assert not result.admits_action
    assert not result.permissions[0].to_dict()["grants_security_authorization"]


def test_exceptions_precedence_conflict_expiry_and_supersession_are_explicit() -> None:
    artifact = _normalized(
        (
            _norm(
                "norm:excepted-prohibition",
                "prohibition",
                exception_ids=("norm:exception",),
            ),
            _norm(
                "norm:exception",
                "exception",
                exception_to=("norm:excepted-prohibition",),
                precedence=20,
                supersedes=("norm:superseded-permission",),
            ),
            _norm(
                "norm:old-permission",
                "permission",
                effective_until_ms=400,
            ),
            _norm(
                "norm:superseded-permission",
                "permission",
                precedence=1,
            ),
            _norm(
                "norm:conflict-permission",
                "permission",
                conflicts_with=("norm:conflict-prohibition",),
                precedence=5,
            ),
            _norm(
                "norm:conflict-prohibition",
                "prohibition",
                conflicts_with=("norm:conflict-permission",),
                precedence=5,
            ),
        )
    )

    result = compile_legal_constraints(artifact, _query(artifact))

    assert result.status is LegalCompilationStatus.CONFLICTING
    assert {item.provision_id for item in result.conflicting} == {
        "norm:conflict-permission",
        "norm:conflict-prohibition",
    }
    assert [item.provision_id for item in result.expired] == [
        "norm:old-permission"
    ]
    assert [item.provision_id for item in result.superseded] == [
        "norm:superseded-permission"
    ]
    excepted = next(
        item
        for item in result.constraints
        if item.provision_id == "norm:excepted-prohibition"
    )
    assert excepted.outcome is LegalApplicabilityOutcome.APPLICABLE
    assert not excepted.active
    assert excepted.defeated_by == ("norm:exception",)


def test_higher_precedence_norm_deterministically_defeats_opposed_norm() -> None:
    artifact = _normalized(
        (
            _norm("norm:permission", "permission", precedence=10),
            _norm("norm:prohibition", "prohibition", precedence=20),
        )
    )

    result = compile_legal_constraints(artifact, _query(artifact))

    assert result.status is LegalCompilationStatus.PROHIBITED
    assert [item.provision_id for item in result.prohibitions] == [
        "norm:prohibition"
    ]
    assert [item.provision_id for item in result.superseded] == [
        "norm:permission"
    ]
    assert result.superseded[0].defeated_by == ("norm:prohibition",)


def test_sourced_applicability_conditions_are_exact_and_missing_facts_are_unknown() -> None:
    artifact = _normalized(
        (
            _norm(
                "norm:conditional",
                "permission",
                required_fact_ids=("fact:consent",),
                conditions={"consent": {"state": "granted"}},
            ),
        )
    )
    source_reference, _ = deterministic_ir_fixture(IRFamily.LEGAL)
    fact = ApplicabilityFact(
        fact_id="fact:consent",
        kind=ApplicabilityFactKind.OTHER,
        predicate="consent",
        value={"state": "granted"},
        source=source_reference,
        jurisdiction="US-CA",
        effective_from_ms=100,
        effective_until_ms=1000,
    )

    applicable = compile_legal_constraints(
        artifact,
        _query(artifact, applicability_facts=(fact,)),
    )
    missing = compile_legal_constraints(artifact, _query(artifact))

    assert applicable.status is LegalCompilationStatus.COMPLETE
    assert [item.provision_id for item in applicable.permissions] == [
        "norm:conditional"
    ]
    assert missing.status is LegalCompilationStatus.UNKNOWN
    assert missing.unknown[0].reason_codes == (
        "missing_required_applicability_fact",
    )


@pytest.mark.parametrize(
    ("declaration", "reason"),
    (
        (
            _norm(
                "norm:missing-exception",
                "prohibition",
                exception_ids=("norm:not-present",),
            ),
            "unresolved_exception",
        ),
        (
            _norm(
                "norm:missing-conflict",
                "permission",
                conflicts_with=("norm:not-present",),
            ),
            "unresolved_conflict_reference",
        ),
        (
            _norm("norm:unsupported", "recommendation"),
            "unsupported_modality",
        ),
        (
            {
                **_norm("norm:no-source", "permission"),
                "source_references": (),
            },
            "missing_trusted_source_or_provenance",
        ),
    ),
)
def test_mandatory_exception_modality_and_source_fail_closed(
    declaration: dict[str, object],
    reason: str,
) -> None:
    artifact = _normalized((declaration,))

    result = compile_legal_constraints(artifact, _query(artifact))

    assert result.fail_closed
    assert result.status in {
        LegalCompilationStatus.UNKNOWN,
        LegalCompilationStatus.REVIEW_REQUIRED,
    }
    assert reason in {
        code for item in result.constraints for code in item.reason_codes
    }


def test_unknown_missing_scope_and_changed_or_missing_root_fail_closed() -> None:
    incomplete = _norm("norm:missing-effect", "permission")
    incomplete.pop("effect")
    artifact = _normalized((incomplete,))
    query = _query(artifact)

    unknown = compile_legal_constraints(artifact, query)
    changed = compile_legal_constraints(
        artifact,
        replace(query, legal_root_supervisor_digest="sha256:" + "0" * 64),
    )
    missing = compile_legal_constraints(None, query)

    assert unknown.status is LegalCompilationStatus.UNKNOWN
    assert unknown.unknown[0].outcome is LegalApplicabilityOutcome.UNKNOWN
    assert "missing_effect_selector" in unknown.unknown[0].reason_codes
    assert changed.reason_codes == ("changed_legal_root",)
    assert not changed.authoritative_scan_complete
    assert missing.reason_codes == ("missing_trusted_legal_source",)
    assert changed.fail_closed and missing.fail_closed


def test_query_rejects_non_exact_values_and_temporally_invalid_facts() -> None:
    artifact = _normalized((_norm("norm:permission", "permission"),))

    with pytest.raises(LegalConstraintError, match="jurisdiction"):
        _query(artifact, jurisdiction=" US-CA")
    with pytest.raises(LegalConstraintError, match="effective_at_ms"):
        _query(artifact, effective_at_ms=-1)
