from __future__ import annotations

from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.code_security_facts import (
    ChangedCodeDiff,
    CodeSecurityDelta,
    CodeSecurityExtractionStatus,
    CodeSecurityFact,
    CodeSecurityFactKind,
    CodeSecurityFactSet,
    CodeSecurityIdentityBinding,
    CodeSecuritySourceScope,
    extract_code_security_facts,
)
from ipfs_accelerate_py.agent_supervisor.cve_security_gate import (
    CVESecurityGateError,
    CVESecurityGateFindingCode,
    CVESecurityGateOutcome,
    SecurityFactStream,
    SecurityRequestContext,
    SecurityRequestMappingStatus,
    correlate_security_requests,
    evaluate_cve_security_gate,
    map_code_security_requests,
    map_intent_security_requests,
)
from ipfs_accelerate_py.agent_supervisor.intent_constraint_adapter import (
    IntentCompilationStatus,
    compile_intent_constraints,
)
from ipfs_accelerate_py.agent_supervisor.ir_adapters import IRAdapterRegistry
from ipfs_accelerate_py.agent_supervisor.ir_registry import (
    IRFamily,
    IRLoadRequest,
    IRLoadStatus,
    IRRegistry,
    deterministic_ir_fixture,
)
from ipfs_accelerate_py.agent_supervisor.security_constraint_adapter import (
    SecurityDecisionOutcome,
    compile_security_constraints,
)


_SOURCE = ({"source_id": "security-review", "span_id": "policy:1"},)
_FLOW = "workspace->repository"
_EFFECT = "state_update"


def _verified(family: IRFamily, **kwargs: object):
    reference, encoded = deterministic_ir_fixture(family, **kwargs)
    registry = IRRegistry()
    registry.register_local_artifact(reference, encoded)
    result = registry.load(IRLoadRequest(reference=reference, family=family))
    assert result.status is IRLoadStatus.VERIFIED
    return result.require_artifact()


def _intent(**updates: object):
    action: dict[str, object] = {
        "id": "intent-action:write",
        "kind": "action",
        "action_id": "intent-action:write",
        "goal_id": "goal:change",
        "action": "write",
        "principal": "principal:worker",
        "tool": "tool:editor",
        "target": "resource:repository",
        "data_flow": _FLOW,
        "expected_effect": _EFFECT,
        "current_state": {"state:repository": "clean"},
        "requested_authority": "mutation",
        "grounded": True,
    }
    action.update(updates)
    intent = _verified(
        IRFamily.INTENT,
        declarations=(
            {"id": "goal:change", "kind": "goal", "grounded": True},
            action,
        ),
    )
    formalization = _verified(
        IRFamily.FORMALIZATION,
        formal_views=(
            {
                "id": "formal:write",
                "kind": "first_order",
                "grounded": True,
            },
        ),
    )
    result = compile_intent_constraints(intent, formalization)
    assert result.status is IntentCompilationStatus.COMPILED
    return result


def _node(node_id: str, kind: str, **values: object) -> dict[str, object]:
    return {
        "declaration_id": node_id,
        "kind": kind,
        "source_references": _SOURCE,
        **values,
    }


def _policy(*, deny_effect: str | None = None):
    policies = [
        _node(
            "policy:allow-write",
            "policy",
            decision="allow",
            principal="principal:worker",
            action="write",
            tool="tool:editor",
            target="resource:repository",
            data_flow=_FLOW,
            expected_effect=_EFFECT,
            requested_authority="mutation",
        )
    ]
    if deny_effect:
        policies.append(
            _node(
                "policy:deny-dangerous-write",
                "policy",
                decision="deny",
                principal="principal:worker",
                action="write",
                tool="tool:editor",
                target="resource:repository",
                data_flow=_FLOW,
                expected_effect=deny_effect,
                requested_authority="mutation",
            )
        )
    artifact = _verified(
        IRFamily.SECURITY,
        declarations=(
            _node("principal:worker", "principal"),
            _node("tool:editor", "resource", resource_type="tool"),
            _node(
                "resource:repository",
                "resource",
                resource_type="repository",
            ),
            *policies,
        ),
    )
    normalized = IRAdapterRegistry().normalize(artifact).require_artifact()
    result = compile_security_constraints(normalized)
    assert result.successful
    return result


def _context(policy=None) -> SecurityRequestContext:
    policy = policy or _policy()
    return SecurityRequestContext.from_policy(
        policy,
        principal="principal:worker",
        tool="tool:editor",
        current_state={"state:repository": "clean"},
        state_version={"state:repository": 7},
        requested_authority="mutation",
        evaluated_at_ms=1_000,
    )


def _code_facts(
    *,
    action: str = "write",
    target: str = "resource:repository",
    data_flow: str = _FLOW,
    effect: str = _EFFECT,
    extra_targets: tuple[str, ...] = (),
) -> CodeSecurityFactSet:
    binding = CodeSecurityIdentityBinding(
        tree_id="tree:candidate",
        diff_id="diff:candidate",
        blob_id="blob:after",
        source_sha256="sha256:after",
        ast_id="ast:write-call",
    )
    scope = CodeSecuritySourceScope(
        path="src/change.py",
        symbol="apply_change",
        line_start=4,
        line_end=4,
        delta=CodeSecurityDelta.ADDED,
    )
    values = (
        (CodeSecurityFactKind.ACTION, action),
        (CodeSecurityFactKind.TARGET, target),
        *((CodeSecurityFactKind.TARGET, item) for item in extra_targets),
        (CodeSecurityFactKind.DATA_FLOW, data_flow),
        (CodeSecurityFactKind.EFFECT, effect),
        (CodeSecurityFactKind.CAPABILITY, "state_mutation"),
    )
    facts = tuple(
        CodeSecurityFact(kind, value, binding, scope) for kind, value in values
    )
    return CodeSecurityFactSet(
        tree_id=binding.tree_id,
        diff_id=binding.diff_id,
        status=CodeSecurityExtractionStatus.EXTRACTED,
        facts=facts,
    )


def test_maps_pinned_intent_and_code_independently_to_exact_root_bound_requests():
    policy = _policy()
    context = _context(policy)

    intent_mapping = map_intent_security_requests(_intent(), context)
    code_mapping = map_code_security_requests(_code_facts(), context)

    assert len(intent_mapping) == len(code_mapping) == 1
    assert intent_mapping[0].stream is SecurityFactStream.INTENT
    assert code_mapping[0].stream is SecurityFactStream.CODE
    assert intent_mapping[0].status is SecurityRequestMappingStatus.EXACT
    assert code_mapping[0].status is SecurityRequestMappingStatus.EXACT
    for mapping in (*intent_mapping, *code_mapping):
        request = mapping.request
        assert request is not None
        assert request.security_root_artifact_id == policy.security_root_artifact_id
        assert request.security_root_cid_v1 == policy.security_root_cid_v1
        assert (
            request.security_root_supervisor_digest
            == policy.security_root_supervisor_digest
        )
        assert dict(request.exact_inputs) == {
            "principal": "principal:worker",
            "action": "write",
            "tool": "tool:editor",
            "target": "resource:repository",
            "data_flow": _FLOW,
            "expected_effect": _EFFECT,
            "requested_authority": "mutation",
        }
        assert request.current_state == {"state:repository": "clean"}
        assert request.state_version == {"state:repository": 7}

    assert not correlate_security_requests(intent_mapping, code_mapping)


def test_maps_real_extractor_action_scope_without_treating_metadata_as_actions():
    facts = extract_code_security_facts(
        ChangedCodeDiff(
            tree_id="tree:extractor",
            diff_id="diff:extractor",
            new_path="src/change.py",
            before_source="def apply_change():\n    pass\n",
            after_source="def apply_change(payload):\n    sink(payload)\n",
        )
    )
    context = _context()

    mappings = map_code_security_requests(facts, context)

    assert len(mappings) == 1
    assert mappings[0].exact
    assert mappings[0].request is not None
    assert dict(mappings[0].request.exact_inputs) == {
        "principal": "principal:worker",
        "action": "invoke",
        "tool": "tool:editor",
        "target": "sink",
        "data_flow": "name->argument:sink",
        "expected_effect": "call",
        "requested_authority": "mutation",
    }


@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        (
            {"action": "delete"},
            CVESecurityGateFindingCode.UNDECLARED_CODE_EFFECT,
        ),
        (
            {"target": "resource:secrets"},
            CVESecurityGateFindingCode.BROADENED_CODE_EFFECT,
        ),
        (
            {"effect": "destructive_update"},
            CVESecurityGateFindingCode.CONTRADICTORY_CODE_EFFECT,
        ),
    ],
)
def test_correlates_undeclared_broadened_and_contradictory_code_effects(
    updates: dict[str, str],
    expected: CVESecurityGateFindingCode,
):
    context = _context()
    findings = correlate_security_requests(
        map_intent_security_requests(_intent(), context),
        map_code_security_requests(_code_facts(**updates), context),
    )

    assert [item.code for item in findings] == [expected]
    assert findings[0].code_mapping_ids
    if expected is not CVESecurityGateFindingCode.UNDECLARED_CODE_EFFECT:
        assert findings[0].intent_mapping_ids


def test_ambiguous_or_incomplete_mappings_are_explicit_unknowns():
    context = _context()

    ambiguous_intent = map_intent_security_requests(
        _intent(resource="resource:other"),
        context,
    )
    ambiguous_code = map_code_security_requests(
        _code_facts(extra_targets=("resource:other",)),
        context,
    )
    incomplete_code = _code_facts()
    incomplete_code = replace(
        incomplete_code,
        facts=tuple(
            item
            for item in incomplete_code.facts
            if item.kind is not CodeSecurityFactKind.DATA_FLOW
        ),
    )
    incomplete_mapping = map_code_security_requests(incomplete_code, context)

    assert not ambiguous_intent[0].exact
    assert ambiguous_intent[0].reason_codes == (
        CVESecurityGateFindingCode.AMBIGUOUS_INTENT_MAPPING,
    )
    assert not ambiguous_code[0].exact
    assert ambiguous_code[0].reason_codes == (
        CVESecurityGateFindingCode.AMBIGUOUS_CODE_MAPPING,
    )
    assert not incomplete_mapping[0].exact
    assert incomplete_mapping[0].reason_codes == (
        CVESecurityGateFindingCode.INCOMPLETE_CODE_MAPPING,
    )
    findings = correlate_security_requests(ambiguous_intent, ambiguous_code)
    assert {
        item.code for item in findings
    } == {
        CVESecurityGateFindingCode.AMBIGUOUS_INTENT_MAPPING,
        CVESecurityGateFindingCode.AMBIGUOUS_CODE_MAPPING,
    }


def test_intent_permit_cannot_mask_code_deny_or_contradictory_effect():
    policy = _policy(deny_effect="destructive_update")
    result = evaluate_cve_security_gate(
        policy,
        _intent(),
        _code_facts(effect="destructive_update"),
        _context(policy),
    )

    outcomes = {
        item.stream: item.decision.outcome for item in result.decisions
    }
    assert outcomes[SecurityFactStream.INTENT] is SecurityDecisionOutcome.PERMIT
    assert outcomes[SecurityFactStream.CODE] is SecurityDecisionOutcome.DENY
    assert result.outcome is CVESecurityGateOutcome.REJECT
    assert not result.passed
    assert result.fail_closed
    assert not result.grants_execution_authority
    assert not result.authorizes_completion
    assert {
        item.code for item in result.findings
    } >= {
        CVESecurityGateFindingCode.CONTRADICTORY_CODE_EFFECT,
        CVESecurityGateFindingCode.CODE_SECURITY_REJECTED,
    }
    assert result.to_dict()["intent_pass_cannot_mask_code_fail"] is True
    assert result.canonical_bytes == result.canonical_bytes


def test_exact_matching_uses_existing_adapter_and_allows_both_streams():
    policy = _policy()
    result = evaluate_cve_security_gate(
        policy,
        _intent(),
        _code_facts(),
        _context(policy),
    )

    assert result.outcome is CVESecurityGateOutcome.PASS
    assert result.passed
    assert not result.findings
    assert len(result.decisions) == 2
    assert all(
        item.decision.outcome is SecurityDecisionOutcome.PERMIT
        for item in result.decisions
    )


def test_gate_rejects_context_bound_to_a_different_security_root():
    policy = _policy()
    context = replace(_context(policy), security_root_artifact_id="root:changed")

    with pytest.raises(CVESecurityGateError, match="evaluated Security IR root"):
        evaluate_cve_security_gate(policy, _intent(), _code_facts(), context)
