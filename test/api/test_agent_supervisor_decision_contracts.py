from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.context.decision_contracts import (
    ActionEnvelope,
    ApplicabilityFact,
    ApplicabilityFactKind,
    AuthorityEnvelope,
    CapabilityEnvelope,
    DecisionAuthority,
    DecisionBindingError,
    DecisionBoundsError,
    DecisionBudget,
    DecisionContractError,
    DecisionIdentityError,
    DecisionPathEscapeError,
    DecisionRequest,
    DecisionTarget,
    DuplicateReferenceError,
    EffectEnvelope,
    EffectKind,
    MissingSemanticRootError,
    NonCanonicalDecisionError,
    PinnedArtifactRef,
    ReferenceAuthority,
    SemanticRoot,
    SemanticRootKind,
    UnknownAuthorityError,
    WorktreeCoverage,
    canonical_artifact_bytes,
)


def _ref(
    name: str,
    *,
    authority: ReferenceAuthority = ReferenceAuthority.VERIFIED,
) -> PinnedArtifactRef:
    return PinnedArtifactRef.from_value(
        {"artifact": name, "version": 1},
        artifact_id=f"artifact:{name}",
        artifact_kind=name,
        artifact_schema=f"example/{name}@1",
        artifact_schema_version="1",
        producer_id="producer:test-suite",
        authority=authority,
    )


def _budget(**changes: int) -> DecisionBudget:
    values = {
        "max_input_tokens": 4_096,
        "max_output_tokens": 2_048,
        "max_serialized_bytes": 262_144,
        "max_artifact_bytes": 1_048_576,
        "max_graph_hops": 8,
        "max_retrieval_results": 64,
        "max_proof_attempts": 8,
        "max_latency_ms": 30_000,
        "max_expansions": 32,
        "max_items": 512,
        "max_depth": 16,
        "max_text_bytes": 8_192,
        "max_actions": 8,
        "max_effects": 16,
        "max_facts": 32,
        "max_capabilities": 16,
    }
    values.update(changes)
    return DecisionBudget(**values)


def _roots() -> tuple[SemanticRoot, ...]:
    coverage = tuple(sorted(WorktreeCoverage, key=lambda item: item.value))
    return tuple(
        sorted(
            (
                SemanticRoot(
                    kind=kind,
                    artifact=_ref(f"root-{kind.value}"),
                    coverage=(
                        coverage
                        if kind is SemanticRootKind.DIRTY_WORKTREE
                        else ()
                    ),
                )
                for kind in SemanticRootKind
            ),
            key=lambda item: item.kind.value,
        )
    )


def _capability(
    capability_id: str = "tool:filesystem-edit",
) -> CapabilityEnvelope:
    return CapabilityEnvelope(
        capability_id=capability_id,
        provider_id="provider:local",
        version="1.0.0",
        configuration=_ref(f"capability-{capability_id}"),
    )


def _authority(
    *,
    capability_ids: tuple[str, ...] = ("tool:filesystem-edit",),
) -> AuthorityEnvelope:
    return AuthorityEnvelope(
        principal_id="principal:implementation-daemon",
        requested_authority=DecisionAuthority.MUTATION,
        capability_ids=capability_ids,
        lease_id="lease:42",
        fencing_epoch=42,
        idempotency_key="ASI-124/attempt-1/write",
        authorization=_ref("authorization"),
    )


def _action(**changes: object) -> ActionEnvelope:
    values: dict[str, object] = {
        "action_id": "action:write-contract",
        "action": "write_file",
        "tool_id": "tool:filesystem-edit",
        "authority": DecisionAuthority.MUTATION,
        "arguments": {
            "path": "ipfs_accelerate_py/agent_supervisor/decision_contracts.py",
            "mode": "replace",
        },
        "targets": (
            DecisionTarget(
                target_id="target:decision-contract",
                resource_type="repository-file",
                repository_paths=(
                    "ipfs_accelerate_py/agent_supervisor/decision_contracts.py",
                ),
            ),
        ),
    }
    values.update(changes)
    return ActionEnvelope(**values)


def _effect(
    effect_id: str = "effect:decision-contract",
) -> EffectEnvelope:
    return EffectEnvelope(
        effect_id=effect_id,
        kind=EffectKind.WRITE,
        authority=DecisionAuthority.MUTATION,
        target_ids=("target:decision-contract",),
        repository_paths=(
            "ipfs_accelerate_py/agent_supervisor/decision_contracts.py",
        ),
        description="Write the canonical decision contracts",
        verification={
            "command": (
                "python -m pytest "
                "test/api/test_agent_supervisor_decision_contracts.py -q"
            )
        },
    )


def _fact() -> ApplicabilityFact:
    return ApplicabilityFact(
        fact_id="fact:effective-policy",
        kind=ApplicabilityFactKind.EFFECTIVE_TIME,
        predicate="policy-effective",
        value={"effective": True},
        source=_ref("applicability"),
        jurisdiction="US",
        effective_from_ms=1_000,
        effective_until_ms=3_000,
    )


def _request(**changes: object) -> DecisionRequest:
    values: dict[str, object] = {
        "decision_kind": "execute",
        "stage": "implementation",
        "objective_id": "ASI-124",
        "objective_revision": "sha256:objective",
        "acceptance_id": "sha256:acceptance",
        "repository_id": "repository:example",
        "repository_path": "/srv/repos/example",
        "jurisdiction": "US",
        "effective_at_ms": 2_000,
        "environment_id": "environment:linux-x86_64",
        "model_id": "model:codex",
        "toolchain_id": "toolchain:python-3.12",
        "authority": _authority(),
        "budget": _budget(),
        "action": _action(),
        "expected_effects": (_effect(),),
        "semantic_roots": _roots(),
        "applicability_facts": (_fact(),),
        "capabilities": (_capability(),),
    }
    values.update(changes)
    return DecisionRequest(**values)


def test_pinned_reference_preserves_two_independent_identities_for_same_bytes() -> None:
    value = {"a": [1, True, None], "z": "canonical"}
    encoded = canonical_artifact_bytes(value)
    reference = PinnedArtifactRef.from_canonical_bytes(
        encoded,
        artifact_id="artifact:fixture",
        artifact_kind="fixture",
        artifact_schema="example/fixture@1",
        artifact_schema_version="1",
        producer_id="producer:test-suite",
        authority=ReferenceAuthority.AUTHORITATIVE,
    )

    assert reference.cid_v1.startswith("baguqeera")
    assert reference.supervisor_digest.startswith("sha256:")
    assert reference.cid_v1 != reference.supervisor_digest
    assert reference.verify_canonical_bytes(encoded) is True
    assert reference.verify_canonical_bytes(encoded + b" ") is False
    assert PinnedArtifactRef.from_json(reference.to_json()) == reference


def test_pinned_reference_rejects_mismatched_cid_digest_and_changed_bytes() -> None:
    left = _ref("left")
    right = _ref("right")

    with pytest.raises(DecisionIdentityError, match="same bytes"):
        replace(left, supervisor_digest=right.supervisor_digest)
    with pytest.raises(NonCanonicalDecisionError, match="canonical round trip"):
        PinnedArtifactRef.from_canonical_bytes(
            b'{ "a": 1 }',
            artifact_id="artifact:noncanonical",
            artifact_kind="fixture",
            artifact_schema="example/fixture@1",
            artifact_schema_version="1",
            producer_id="producer:test-suite",
            authority=ReferenceAuthority.VERIFIED,
        )


def test_decision_request_is_deeply_immutable_canonical_and_versioned() -> None:
    request = _request()
    restored = DecisionRequest.from_json(request.to_json())

    assert restored == request
    assert restored.request_id == request.content_id
    assert restored.principal == "principal:implementation-daemon"
    assert restored.root("intent_ir").kind is SemanticRootKind.INTENT_IR
    assert restored.intent_ir_root == restored.root("intent_ir").artifact
    assert restored.ast_root == restored.program_root
    assert restored.dirty_worktree_root.artifact_kind == "root-dirty_worktree"
    assert json.loads(request.to_json())["contract_version"] == 1
    with pytest.raises(FrozenInstanceError):
        request.stage = "validation"  # type: ignore[misc]
    with pytest.raises(TypeError):
        request.action.arguments["path"] = "other.py"  # type: ignore[index]
    with pytest.raises(TypeError):
        request.expected_effects[0].verification["command"] = "true"  # type: ignore[index]


def test_every_decision_changing_binding_changes_the_request_identity() -> None:
    request = _request(applicability_facts=())
    variants = (
        replace(request, stage="validation"),
        replace(request, objective_revision="sha256:objective-2"),
        replace(request, acceptance_id="sha256:acceptance-2"),
        replace(request, repository_id="repository:other"),
        replace(request, jurisdiction="CA"),
        replace(request, effective_at_ms=2_001),
        replace(request, environment_id="environment:other"),
        replace(request, model_id="model:other"),
        replace(request, toolchain_id="toolchain:other"),
        replace(request, budget=_budget(max_graph_hops=7)),
        replace(request, action=replace(request.action, action="patch_file")),
        replace(
            request,
            authority=replace(
                request.authority,
                idempotency_key="ASI-124/attempt-1/write-2",
            ),
        ),
    )

    assert all(item.content_id != request.content_id for item in variants)


def test_all_mandatory_roots_and_dirty_worktree_coverage_are_required() -> None:
    roots = _roots()
    without_policy = tuple(
        item for item in roots if item.kind is not SemanticRootKind.POLICY
    )
    with pytest.raises(MissingSemanticRootError, match="policy"):
        _request(semantic_roots=without_policy)

    dirty = next(
        item for item in roots if item.kind is SemanticRootKind.DIRTY_WORKTREE
    )
    with pytest.raises(MissingSemanticRootError, match="untracked"):
        replace(
            dirty,
            coverage=tuple(
                item
                for item in dirty.coverage
                if item is not WorktreeCoverage.UNTRACKED
            ),
        )


def test_duplicate_and_conflicting_references_fail_closed() -> None:
    roots = list(_roots())
    roots.append(roots[0])
    roots.sort(key=lambda item: item.kind.value)
    with pytest.raises(DuplicateReferenceError, match="roles"):
        _request(semantic_roots=tuple(roots))

    request = _request()
    duplicate_configuration = replace(
        request.capabilities[0],
        configuration=request.semantic_roots[0].artifact,
    )
    with pytest.raises(DuplicateReferenceError, match="duplicate pinned"):
        replace(request, capabilities=(duplicate_configuration,))

    forged = request.to_record()
    forged["stage"] = "validation"
    with pytest.raises(DecisionIdentityError, match="identity"):
        DecisionRequest.from_dict(forged)


def test_authority_capability_lease_fence_and_idempotency_are_exact() -> None:
    with pytest.raises(UnknownAuthorityError, match="unknown authority"):
        replace(_authority(), requested_authority="administrator")
    with pytest.raises(DecisionBindingError, match="lease, fence"):
        replace(_authority(), lease_id=None)
    with pytest.raises(DecisionBindingError, match="capability IDs"):
        _request(
            authority=_authority(capability_ids=("tool:undeclared",)),
        )
    with pytest.raises(DecisionBindingError, match="declared capability"):
        _request(action=_action(tool_id="tool:undeclared"))
    with pytest.raises(UnknownAuthorityError, match="does not cover"):
        _request(
            authority=AuthorityEnvelope(
                principal_id="principal:reader",
                requested_authority="read",
                capability_ids=("tool:filesystem-edit",),
                lease_id=None,
                fencing_epoch=None,
                idempotency_key=None,
                authorization=None,
            )
        )


def test_exact_targets_effects_and_repository_paths_are_bound() -> None:
    with pytest.raises(DecisionBindingError, match="undeclared action target"):
        _request(
            expected_effects=(
                replace(_effect(), target_ids=("target:other",)),
            )
        )
    with pytest.raises(DecisionPathEscapeError, match="repository-relative"):
        _action(arguments={"path": "../../etc/passwd"})
    with pytest.raises(DecisionPathEscapeError, match="repository-relative"):
        replace(
            _action().targets[0],
            repository_paths=("/etc/passwd",),
        )
    with pytest.raises(DecisionPathEscapeError, match="filesystem root"):
        _request(repository_path="/")


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan"), 1.5])
def test_non_finite_or_fractional_budgets_are_rejected(value: float) -> None:
    with pytest.raises(DecisionContractError, match="finite integer"):
        _budget(max_latency_ms=value)  # type: ignore[arg-type]


def test_count_depth_text_and_serialized_byte_bounds_are_enforced() -> None:
    with pytest.raises(DecisionBoundsError, match="max_effects"):
        _budget(
            max_items=1,
            max_actions=1,
            max_effects=2,
            max_facts=1,
            max_capabilities=1,
        )
    nested: dict[str, object] = {"leaf": True}
    for _ in range(34):
        nested = {"nested": nested}
    with pytest.raises(DecisionBoundsError, match="nesting-depth"):
        _action(arguments=nested)
    with pytest.raises(DecisionBoundsError, match="nesting-depth"):
        _request(budget=_budget(max_depth=4))
    with pytest.raises(DecisionBoundsError, match="max_artifact_bytes"):
        _request(
            budget=_budget(
                max_artifact_bytes=10,
                max_serialized_bytes=262_144,
            )
        )
    # Action values are validated against the absolute depth first; the request
    # then applies its tighter declared serialized-byte boundary.
    with pytest.raises(DecisionBoundsError, match="max_serialized_bytes"):
        _request(budget=_budget(max_serialized_bytes=1_024))


def test_applicability_jurisdiction_and_effective_time_are_not_ambient() -> None:
    with pytest.raises(DecisionBindingError, match="jurisdiction"):
        _request(jurisdiction="")
    with pytest.raises(DecisionBindingError, match="effective_at_ms"):
        _request(effective_at_ms=None)
    with pytest.raises(DecisionBindingError, match="outside"):
        _request(effective_at_ms=5_000)
    with pytest.raises(DecisionContractError, match="bounded interval"):
        replace(_fact(), effective_until_ms=None)


def test_missing_fields_unknown_fields_and_changed_json_round_trips_are_rejected() -> None:
    request = _request()
    missing = request.to_dict()
    del missing["semantic_roots"]
    with pytest.raises(DecisionContractError, match="semantic_roots"):
        DecisionRequest.from_dict(missing)

    unknown = request.to_dict()
    unknown["ambient_policy"] = "latest"
    with pytest.raises(DecisionContractError, match="unsupported fields"):
        DecisionRequest.from_dict(unknown)

    noncanonical = json.dumps(request.to_dict(), indent=2, sort_keys=False)
    with pytest.raises(NonCanonicalDecisionError, match="round trip"):
        DecisionRequest.from_json(noncanonical)

    duplicate_key = request.to_json().replace(
        '"stage":"implementation"',
        '"stage":"implementation","stage":"validation"',
    )
    with pytest.raises(DuplicateReferenceError, match="duplicate object keys"):
        DecisionRequest.from_json(duplicate_key)


def test_reordered_or_normalized_decision_inputs_are_rejected_not_silently_changed() -> None:
    with pytest.raises(NonCanonicalDecisionError, match="canonically sorted"):
        replace(
            _authority(),
            capability_ids=("tool:z", "tool:a"),
        )
    with pytest.raises(DecisionPathEscapeError, match="repository-relative"):
        replace(
            _action().targets[0],
            repository_paths=("docs/../secret",),
        )
    with pytest.raises(NonCanonicalDecisionError, match="whitespace"):
        replace(_action(), action=" write_file")
