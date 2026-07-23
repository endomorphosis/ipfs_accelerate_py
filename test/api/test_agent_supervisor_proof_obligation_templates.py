from __future__ import annotations

from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.code_proof_obligations import (
    CodeObligationRequest,
    compile_candidate_proof_scopes,
    materialize_code_proof_obligation,
    obligation_cache_identity,
)
from ipfs_accelerate_py.agent_supervisor.proof_obligation_templates import (
    DEFAULT_PROOF_OBLIGATION_TEMPLATES,
    DEFAULT_TEMPLATE_REGISTRY,
    AmbiguousProofTemplateError,
    ProofObligationTemplate,
    ProofObligationTemplateRegistry,
    ReviewedCodeShape,
    TemplateSelectionStatus,
    TemplateValidationError,
    UnsupportedProofTemplateError,
)


EXPECTED_TEMPLATE_IDS = {
    "legal-state-transitions",
    "lease-uniqueness-and-fencing",
    "dag-acyclicity",
    "merge-idempotence",
    "cache-key-completeness",
    "evidence-freshness",
    "projection-equivalence",
    "unsupported-proof-fail-closed",
}


def _scope_set():
    return compile_candidate_proof_scopes(
        [
            {
                "new_path": "runtime/state.py",
                "status": "add",
                "after_source": """\
class Machine:
    def start(self):
        self.state = "running"
""",
            }
        ]
    )


def test_initial_registry_has_every_reviewed_invariant_and_complete_declarations() -> None:
    assert set(DEFAULT_TEMPLATE_REGISTRY.template_ids) == EXPECTED_TEMPLATE_IDS
    assert len(DEFAULT_PROOF_OBLIGATION_TEMPLATES) == 8

    semantic_hashes = set()
    for template in DEFAULT_TEMPLATE_REGISTRY.templates:
        assert callable(template.python_reference_predicate)
        assert template.canonical_statement
        assert template.supported_backends
        assert template.assumptions
        assert template.mutation_cases
        assert template.fallback_tests
        assert template.supported_code_shapes
        assert template.semantic_hash.startswith("sha256:")
        assert len(template.semantic_hash) == len("sha256:") + 64
        assert all(template.verify_mutation_cases().values())
        semantic_hashes.add(template.semantic_hash)

    assert len(semantic_hashes) == len(DEFAULT_TEMPLATE_REGISTRY.templates)


def test_registry_serialization_round_trip_verifies_all_claimed_identities() -> None:
    payload = DEFAULT_TEMPLATE_REGISTRY.to_dict()
    restored = ProofObligationTemplateRegistry.from_json(
        DEFAULT_TEMPLATE_REGISTRY.to_json()
    )

    assert restored.registry_id == DEFAULT_TEMPLATE_REGISTRY.registry_id
    assert [item.semantic_hash for item in restored.templates] == [
        item.semantic_hash for item in DEFAULT_TEMPLATE_REGISTRY.templates
    ]

    payload["templates"][0]["canonical_statement"] = "tampered"
    with pytest.raises(TemplateValidationError, match="semantic hash"):
        ProofObligationTemplateRegistry.from_dict(payload)


def test_reference_predicates_reject_malformed_inputs_and_mutated_invariants() -> None:
    state = DEFAULT_TEMPLATE_REGISTRY.require("legal-state-transitions")
    lease = DEFAULT_TEMPLATE_REGISTRY.require("lease-uniqueness-and-fencing")
    dag = DEFAULT_TEMPLATE_REGISTRY.require("dag-acyclicity")
    cache = DEFAULT_TEMPLATE_REGISTRY.require("cache-key-completeness")
    freshness = DEFAULT_TEMPLATE_REGISTRY.require("evidence-freshness")
    projection = DEFAULT_TEMPLATE_REGISTRY.require("projection-equivalence")

    assert not state.evaluate(
        previous_state="ready",
        next_state="done",
        allowed_transitions={"ready": ["running"]},
    )
    assert not lease.evaluate(
        active_leases=[
            {"resource_id": "r", "fencing_token": 2},
            {"resource_id": "r", "fencing_token": 3},
        ],
        mutation_fencing_token=3,
        current_fencing_token=3,
    )
    assert not dag.evaluate(adjacency={"a": ["b"], "b": ["a"]})
    assert not cache.evaluate(
        semantic_inputs={"tree": "t", "template_version": "2"},
        cache_key_inputs={"tree": "t"},
    )
    assert not freshness.evaluate(
        expected_bindings={"tree": "new"},
        evidence_bindings={"tree": "old"},
        verified_at=2,
        not_before=1,
        expires_at=3,
    )
    assert not projection.evaluate(
        source={"id": "a"},
        projection={"id": "b"},
        projected_fields=["id"],
    )
    # Exceptions and non-boolean predicate behavior can never turn into truth.
    assert not state.evaluate(previous_state="ready")


def test_exact_shape_selection_never_uses_near_name_heuristics() -> None:
    exact = DEFAULT_TEMPLATE_REGISTRY.select_for_code_shape(
        ReviewedCodeShape.DAG_ACYCLICITY
    )
    near_miss = DEFAULT_TEMPLATE_REGISTRY.select_for_code_shape(
        "directed_acyclic_graph.edge_updates"
    )
    unknown = DEFAULT_TEMPLATE_REGISTRY.select_for_code_shape(
        "custom.graph.edge_update"
    )

    assert exact.supported
    assert exact.template.template_id == "dag-acyclicity"
    assert near_miss.status is TemplateSelectionStatus.UNSUPPORTED
    assert unknown.status is TemplateSelectionStatus.UNSUPPORTED
    with pytest.raises(UnsupportedProofTemplateError):
        near_miss.require_supported()
    assert DEFAULT_TEMPLATE_REGISTRY.get("dag-cycle-check") is None


def test_ambiguous_shape_and_template_versions_fail_closed() -> None:
    original = DEFAULT_TEMPLATE_REGISTRY.require("dag-acyclicity")
    competing = replace(
        original,
        template_id="another-reviewed-dag-template",
        canonical_statement="A separately reviewed DAG statement.",
    )
    ambiguous_shape_registry = ProofObligationTemplateRegistry(
        (original, competing)
    )
    selection = ambiguous_shape_registry.select_for_code_shape(
        ReviewedCodeShape.DAG_ACYCLICITY
    )

    assert selection.status is TemplateSelectionStatus.AMBIGUOUS
    assert len(selection.candidate_template_ids) == 2
    with pytest.raises(AmbiguousProofTemplateError):
        selection.require_supported()

    second_version = replace(original, version="2.0.0")
    ambiguous_version_registry = ProofObligationTemplateRegistry(
        (original, second_version)
    )
    assert ambiguous_version_registry.get(original.template_id) is None
    with pytest.raises(AmbiguousProofTemplateError, match="multiple versions"):
        ambiguous_version_registry.require(original.template_id)
    assert (
        ambiguous_version_registry.require(original.template_id, "2.0.0")
        == second_version
    )


def test_materialization_uses_only_registry_statement_semantics_and_exact_scopes() -> None:
    scopes = _scope_set()
    state_scope = next(
        scope for scope in scopes.scopes if scope.kind.value == "state_transition"
    )
    request = CodeObligationRequest(
        template_id="legal-state-transitions",
        template_version="1.0.0",
        ast_scope_ids=(state_scope.scope_id,),
        code_shape=ReviewedCodeShape.LEGAL_STATE_TRANSITION.value,
        premise_ids=("premise:declared-transition-table",),
        task_id="REF-249",
    )
    assert CodeObligationRequest.from_json(request.to_json()) == request
    template = DEFAULT_TEMPLATE_REGISTRY.require(
        request.template_id, request.template_version
    )
    obligation = materialize_code_proof_obligation(
        scopes,
        repository_id="repo:test",
        repository_tree_id="git-tree:one",
        request=request,
        backend_id="lean4",
    )

    assert obligation.statement == template.canonical_statement
    assert obligation.ast_scope_ids == (state_scope.scope_id,)
    assert obligation.template_id == template.template_id
    assert obligation.template_version == template.version
    assert obligation.template_semantic_hash == template.semantic_hash
    assert obligation.fallback_checks == template.fallback_tests
    assert obligation.invariant_class == "state_transition"
    assert obligation.metadata["code_shape"] == request.code_shape

    with pytest.raises(ValueError, match="outside the compiled scope set"):
        materialize_code_proof_obligation(
            scopes,
            repository_tree_id="git-tree:one",
            request=replace(request, ast_scope_ids=("scope:not-present",)),
        )
    with pytest.raises(UnsupportedProofTemplateError, match="does not support exact"):
        materialize_code_proof_obligation(
            scopes,
            repository_tree_id="git-tree:one",
            request=replace(
                request,
                code_shape=ReviewedCodeShape.DAG_ACYCLICITY.value,
            ),
        )


def test_template_version_and_semantic_hash_change_obligation_and_cache_identity() -> None:
    scopes = _scope_set()
    original = DEFAULT_TEMPLATE_REGISTRY.require("legal-state-transitions")
    changed_version = replace(original, version="2.0.0")
    changed_semantics = replace(
        original,
        canonical_statement=(
            original.canonical_statement + " Self-transitions must also be declared."
        ),
    )

    def build(template: ProofObligationTemplate):
        registry = ProofObligationTemplateRegistry((template,))
        return materialize_code_proof_obligation(
            scopes,
            repository_tree_id="git-tree:one",
            template_id=template.template_id,
            template_version=template.version,
            registry=registry,
        )

    base_obligation = build(original)
    version_obligation = build(changed_version)
    semantics_obligation = build(changed_semantics)

    assert original.semantic_hash != changed_version.semantic_hash
    assert original.semantic_hash != changed_semantics.semantic_hash
    assert len(
        {
            base_obligation.obligation_id,
            version_obligation.obligation_id,
            semantics_obligation.obligation_id,
        }
    ) == 3
    assert len(
        {
            obligation_cache_identity(base_obligation, backend_id="lean4"),
            obligation_cache_identity(version_obligation, backend_id="lean4"),
            obligation_cache_identity(semantics_obligation, backend_id="lean4"),
        }
    ) == 3
    assert obligation_cache_identity(
        base_obligation, backend_id="lean4"
    ) != obligation_cache_identity(base_obligation, backend_id="smtlib2")


def test_conservative_or_unsupported_code_cannot_become_a_proof_obligation() -> None:
    unsupported = compile_candidate_proof_scopes(
        [
            {
                "new_path": "runtime/config.yaml",
                "status": "add",
                "after_source": "state: ready\n",
            }
        ]
    )
    assert unsupported.conservative

    with pytest.raises(UnsupportedProofTemplateError, match="no non-conservative"):
        materialize_code_proof_obligation(
            unsupported,
            repository_tree_id="git-tree:one",
            template_id="legal-state-transitions",
        )
    with pytest.raises(UnsupportedProofTemplateError, match="unknown reviewed"):
        materialize_code_proof_obligation(
            _scope_set(),
            repository_tree_id="git-tree:one",
            template_id="looks-like-state-transitions",
        )


def test_unsupported_proof_template_encodes_fail_closed_gate_behavior() -> None:
    template = DEFAULT_TEMPLATE_REGISTRY.require(
        "unsupported-proof-fail-closed"
    )

    assert template.evaluate(
        support_status="unsupported",
        matching_template_ids=[],
        gate_satisfied=False,
    )
    assert not template.evaluate(
        support_status="unsupported",
        matching_template_ids=[],
        gate_satisfied=True,
    )
    assert template.evaluate(
        support_status="ambiguous",
        matching_template_ids=["a@1", "b@1"],
        gate_satisfied=False,
    )
    assert not template.evaluate(
        support_status="ambiguous",
        matching_template_ids=["a@1", "b@1"],
        gate_satisfied=True,
    )
