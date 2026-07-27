from __future__ import annotations

from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.ir_adapters import IRAdapterRegistry
from ipfs_accelerate_py.agent_supervisor.ir_registry import (
    IRFamily,
    IRLoadRequest,
    IRLoadStatus,
    IRRegistry,
    deterministic_ir_fixture,
)
from ipfs_accelerate_py.agent_supervisor.security_constraint_adapter import (
    SecurityAuthorizationRequest,
    SecurityCompilationStatus,
    SecurityConstraintError,
    SecurityDecisionOutcome,
    compile_security_constraints,
    evaluate_security_authorization,
)


_SOURCE = ({"source_id": "security-review", "span_id": "policy:1"},)
_EFFECT = {"operation": "update", "path": "src/safe.py"}
_FLOW = {"classification": "source", "direction": "workspace_to_tool"}


def _node(node_id: str, kind: str, **values: object) -> dict[str, object]:
    return {
        "declaration_id": node_id,
        "kind": kind,
        "source_references": _SOURCE,
        **values,
    }


def _allow_policy(policy_id: str = "policy:allow-write") -> dict[str, object]:
    return _node(
        policy_id,
        "policy",
        decision="allow",
        principal="principal:worker",
        action="write",
        tool="tool:editor",
        target="resource:repository",
        data_flow=_FLOW,
        expected_effect=_EFFECT,
        requested_authority="mutation",
        source_zone="zone:workspace",
        channel="channel:local-tool",
        target_zone="zone:repository",
        state_machine_id="state:repository",
        from_state="clean",
        to_state="modified",
        assumption_ids=("assumption:tool-isolated",),
        claim_ids=("claim:tool-boundary",),
        obligation_ids=("obligation:audit",),
    )


def _normalized(
    *,
    policies: tuple[dict[str, object], ...] | None = None,
    assumptions: tuple[dict[str, object], ...] = (),
    claims: tuple[dict[str, object], ...] = (),
    obligations: tuple[dict[str, object], ...] = (),
):
    if policies is None:
        policies = (_allow_policy(),)
    declarations = (
        _node("principal:worker", "principal"),
        _node("tool:editor", "resource", resource_type="tool"),
        _node("resource:repository", "resource", resource_type="repository"),
        _node("asset:source", "asset"),
        _node("zone:workspace", "zone", trust_level="agent"),
        _node("zone:repository", "zone", trust_level="protected"),
        _node(
            "channel:local-tool",
            "channel",
            source_zone="zone:workspace",
            target_zone="zone:repository",
        ),
        _node(
            "state:repository",
            "state_machine",
            resource_id="resource:repository",
            states=("clean", "modified"),
            current_state="clean",
            state_version=7,
            transitions=(
                {
                    "id": "transition:write",
                    "from": "clean",
                    "to": "modified",
                    "action": "write",
                    "tool": "tool:editor",
                    "target": "resource:repository",
                    "expected_effect": _EFFECT,
                    "guard_assumption_ids": ("assumption:tool-isolated",),
                },
            ),
        ),
        *policies,
    )
    if not assumptions:
        assumptions = (
            {
                "assumption_id": "assumption:tool-isolated",
                "kind": "threat_assumption",
                "source_references": _SOURCE,
            },
        )
    if not claims:
        claims = (
            {
                "claim_id": "claim:tool-boundary",
                "claim_kind": "security_claim",
                "result_status": "proved",
                "result_authority_id": "result-authority:kernel",
                "source_references": _SOURCE,
            },
        )
    if not obligations:
        obligations = (
            {
                "obligation_id": "obligation:audit",
                "kind": "formal_obligation",
                "required": True,
                "discharged": True,
                "source_references": _SOURCE,
            },
        )
    reference, encoded = deterministic_ir_fixture(
        IRFamily.SECURITY,
        declarations=declarations,
        assumptions=assumptions,
        claims=claims,
        obligations=obligations,
        result_authority=(
            {
                "result_id": "result-authority:kernel",
                "kind": "result_authority",
                "source_references": _SOURCE,
            },
        ),
    )
    registry = IRRegistry()
    registry.register_local_artifact(reference, encoded)
    loaded = registry.load(IRLoadRequest(reference=reference, family=IRFamily.SECURITY))
    assert loaded.status is IRLoadStatus.VERIFIED
    return IRAdapterRegistry().normalize(loaded).require_artifact()


def _request(artifact, **updates: object) -> SecurityAuthorizationRequest:
    values: dict[str, object] = {
        "security_root_artifact_id": artifact.root_artifact_id,
        "security_root_cid_v1": artifact.root_cid_v1,
        "security_root_supervisor_digest": artifact.root_supervisor_digest,
        "principal": "principal:worker",
        "action": "write",
        "tool": "tool:editor",
        "target": "resource:repository",
        "data_flow": _FLOW,
        "expected_effect": _EFFECT,
        "current_state": {"state:repository": "clean"},
        "state_version": {"state:repository": 7},
        "requested_authority": "mutation",
        "evaluated_at_ms": 1000,
        "source_zone": "zone:workspace",
        "channel": "channel:local-tool",
        "target_zone": "zone:repository",
        "satisfied_assumption_ids": ("assumption:tool-isolated",),
        "accepted_claim_result_ids": ("result-authority:kernel",),
    }
    values.update(updates)
    return SecurityAuthorizationRequest(**values)


def test_compiles_every_security_declaration_and_emits_root_bound_receipts() -> None:
    artifact = _normalized()

    policy = compile_security_constraints(artifact)
    decision = evaluate_security_authorization(policy, _request(artifact))

    assert policy.status is SecurityCompilationStatus.COMPILED
    assert policy.successful
    assert [item.declaration_id for item in policy.principals] == [
        "principal:worker"
    ]
    assert [item.declaration_id for item in policy.assets] == ["asset:source"]
    assert {item.declaration_id for item in policy.resources} == {
        "resource:repository",
        "tool:editor",
    }
    assert policy.zones and policy.channels and policy.state_machines
    assert policy.threat_assumptions and policy.claims
    assert policy.formal_obligations and policy.result_authorities
    assert policy.authorization_policy is not None
    assert policy.authorization_policy.version == artifact.root_supervisor_digest
    assert decision.outcome is SecurityDecisionOutcome.PERMIT
    assert decision.permitted
    assert decision.authorization_decision is not None
    assert decision.authorization_decision.permitted
    assert decision.security_root_cid_v1 == artifact.root_cid_v1
    assert decision.request_id == _request(artifact).content_id
    assert decision.canonical_bytes == decision.canonical_bytes
    assert not policy.grants_execution_authority
    assert not decision.grants_execution_authority
    assert not decision.establishes_generated_code_correctness


def test_deny_overrides_allow_and_unknown_and_conflict_are_explicit() -> None:
    allow_record = _allow_policy("policy:allow")
    deny_record = {**allow_record, "declaration_id": "policy:deny", "decision": "deny"}
    artifact = _normalized(policies=(allow_record, deny_record))
    decision = evaluate_security_authorization(
        compile_security_constraints(artifact), _request(artifact)
    )
    assert decision.outcome is SecurityDecisionOutcome.DENY
    assert "deny_override" in decision.reason_codes

    unknown_record = {
        **allow_record,
        "declaration_id": "policy:unknown",
        "decision": "unknown",
    }
    artifact = _normalized(policies=(unknown_record,))
    decision = evaluate_security_authorization(
        compile_security_constraints(artifact), _request(artifact)
    )
    assert decision.outcome is SecurityDecisionOutcome.UNKNOWN
    assert "explicit_unknown_policy" in decision.reason_codes

    conflict_record = {
        **allow_record,
        "declaration_id": "policy:conflict",
        "decision": "conflict",
    }
    artifact = _normalized(policies=(conflict_record,))
    decision = evaluate_security_authorization(
        compile_security_constraints(artifact), _request(artifact)
    )
    assert decision.outcome is SecurityDecisionOutcome.CONFLICT
    assert "conflicting_policy" in decision.reason_codes


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"target": "resource:unknown"}, "unknown_resource"),
        ({"current_state": {"state:repository": "modified"}}, "stale_state"),
        ({"state_version": {"state:repository": 6}}, "stale_state"),
        (
            {"expected_effect": {"operation": "delete", "path": "src/safe.py"}},
            "changed_expected_effect",
        ),
        ({"asserted_grant_sources": ("intent",)}, "non_security_input_used_as_grant"),
        (
            {"asserted_grant_sources": ("legal_permission",)},
            "non_security_input_used_as_grant",
        ),
        (
            {"asserted_grant_sources": ("model_output",)},
            "non_security_input_used_as_grant",
        ),
        (
            {"asserted_grant_sources": ("retrieval_rank",)},
            "non_security_input_used_as_grant",
        ),
    ],
)
def test_exact_scope_state_effect_and_non_grant_sources_fail_closed(
    changes: dict[str, object], reason: str
) -> None:
    artifact = _normalized()
    decision = evaluate_security_authorization(
        compile_security_constraints(artifact), _request(artifact, **changes)
    )

    assert not decision.permitted
    assert reason in decision.reason_codes


def test_changed_root_and_undischarged_dependencies_fail_closed() -> None:
    artifact = _normalized(
        obligations=(
            {
                "obligation_id": "obligation:audit",
                "kind": "formal_obligation",
                "required": True,
                "discharged": False,
                "source_references": _SOURCE,
            },
        )
    )
    policy = compile_security_constraints(artifact)

    stale = evaluate_security_authorization(
        policy,
        replace(
            _request(artifact),
            security_root_supervisor_digest="sha256:" + "f" * 64,
        ),
    )
    blocked = evaluate_security_authorization(policy, _request(artifact))

    assert stale.outcome is SecurityDecisionOutcome.DENY
    assert "changed_security_root" in stale.reason_codes
    assert blocked.outcome is SecurityDecisionOutcome.DENY
    assert "formal_obligation_undischarged" in blocked.reason_codes


def test_wildcards_and_unsupported_or_contradictory_policy_never_broaden() -> None:
    artifact = _normalized()
    with pytest.raises(SecurityConstraintError, match="wildcard"):
        _request(artifact, target="*")

    wildcard_policy = _node(
        "policy:wildcard",
        "policy",
        decision="allow",
        principal="principal:worker",
        action="write",
        tool="tool:editor",
        target="*",
        data_flow=_FLOW,
        expected_effect=_EFFECT,
        requested_authority="mutation",
    )
    unsupported = compile_security_constraints(
        _normalized(policies=(wildcard_policy,))
    )
    assert unsupported.status is SecurityCompilationStatus.UNSUPPORTED
    assert unsupported.authorization_policy is None
    assert "malformed_security_declaration" in unsupported.reason_codes

    contradictory = _node(
        "policy:contradiction",
        "policy",
        decision="contradictory",
        principal="principal:worker",
        action="write",
        tool="tool:editor",
        target="resource:repository",
        data_flow=_FLOW,
        expected_effect=_EFFECT,
        requested_authority="mutation",
    )
    unsupported = compile_security_constraints(
        _normalized(policies=(contradictory,))
    )
    assert unsupported.status is SecurityCompilationStatus.UNSUPPORTED
    assert "malformed_security_declaration" in unsupported.reason_codes

    for source in ("intent", "legal permission", "model_output", "retrieval_rank"):
        forged_grant = {
            **_allow_policy(f"policy:forged:{source.replace(' ', '-')}"),
            "grant_source": source,
        }
        unsupported = compile_security_constraints(
            _normalized(policies=(forged_grant,))
        )
        assert unsupported.status is SecurityCompilationStatus.UNSUPPORTED
        assert unsupported.authorization_policy is None
