"""CBP-030: obligation compiler cache-key binding tests.

Covers dependency, API-contract, security, semantic-equivalence, and residual-ref
cases without embedding secrets or gold bodies in premises or receipts.
"""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.code_claim_contracts import (
    ClaimFamily,
    ClaimStatus,
    InvalidationSelectorKind,
)
from ipfs_accelerate_py.agent_supervisor.code_proof_obligations import (
    CandidateDiffEntry,
    CodeProofCompileRequest,
    CodeProofObligationCompilation,
    DiffChangeKind,
    ObligationCompileStatus,
    PremiseValidationError,
    assumption_set_digest,
    build_code_proof_cache_key,
    compile_code_proof_obligations,
    compiled_obligation_cache_identity,
    normalize_premise_ids,
    normalize_residual_refs,
    premise_set_digest,
)
from ipfs_accelerate_py.agent_supervisor.code_property_catalog import (
    DEFAULT_CODE_PROPERTY_CATALOG,
)
from ipfs_accelerate_py.agent_supervisor.formal_verification_contracts import (
    AssuranceLevel,
)


PYTHON_SOURCE = """\
from typing import Protocol
import json as json_module

class Store(Protocol):
    def save(self, value: str) -> None: ...

class Worker:
    def __init__(self, store: Store):
        self.store = store
        self.state = "idle"

    def run(self, value: str) -> None:
        self.state = "running"
        self.store.save(json_module.dumps(value))
"""

TREE_A = "git-tree:cbp-030-a"
TREE_B = "git-tree:cbp-030-b"
REPO = "repository:sha256:cbp-030-demo"
TOOLCHAIN = "toolchain:nix-lock-sha256"
POLICY = "policy:formal-v1"
PREMISES = ("premise:typed-import-edge", "premise:interface-store")
ASSUMPTIONS = ("assumption:single-worker", "assumption:no-network")
RESIDUAL_REF = "residual-ref:sha256:deadbeefcafebabe"


def _entries(source: str = PYTHON_SOURCE) -> list[CandidateDiffEntry]:
    return [
        CandidateDiffEntry(
            new_path="src/runtime.py",
            change_kind=DiffChangeKind.ADD,
            after_source=source,
            after_blob_id="git:runtime-cbp-030",
        )
    ]


def _compile(**kwargs):
    values = {
        "candidate_diff": _entries(),
        "repository_tree_id": TREE_A,
        "repository_id": REPO,
        "premise_ids": PREMISES,
        "assumption_ids": ASSUMPTIONS,
        "toolchain_id": TOOLCHAIN,
        "policy_id": POLICY,
        "task_id": "CBP-030",
        "catalog": DEFAULT_CODE_PROPERTY_CATALOG,
    }
    values.update(kwargs)
    return compile_code_proof_obligations(**values)


def test_normalize_premise_ids_rejects_repository_wide_and_source_dumps() -> None:
    assert normalize_premise_ids(PREMISES) == tuple(sorted(PREMISES))
    with pytest.raises(PremiseValidationError, match="repository-wide"):
        normalize_premise_ids(["repo:*"])
    with pytest.raises(PremiseValidationError, match="repository-wide"):
        normalize_premise_ids(["**"])
    with pytest.raises(PremiseValidationError, match="opaque"):
        normalize_premise_ids(["def worker():\n    return 1\n"])
    with pytest.raises(PremiseValidationError, match="opaque|gold"):
        normalize_premise_ids(
            [{"premise_id": "premise:x", "gold_ir": "SECRET_GOLD_BODY"}]
        )
    with pytest.raises(PremiseValidationError, match="opaque"):
        normalize_premise_ids(["-----BEGIN PRIVATE KEY-----\nabc\n"])


def test_normalize_residual_refs_reject_gold_bodies_accept_handles() -> None:
    assert normalize_residual_refs([RESIDUAL_REF]) == (RESIDUAL_REF,)
    assert normalize_residual_refs(
        [{"residual_ref_id": RESIDUAL_REF, "facet": "non_vacuous_candidate"}]
    ) == (RESIDUAL_REF,)
    with pytest.raises(PremiseValidationError, match="gold|source"):
        normalize_residual_refs(
            [{"residual_ref_id": RESIDUAL_REF, "gold_body": "IR(...)"}]
        )
    with pytest.raises(PremiseValidationError, match="gold|source"):
        normalize_residual_refs([{"id": RESIDUAL_REF, "source_dump": "entire repo"}])


def test_compile_dependency_api_security_semantic_and_residual_ref_cases() -> None:
    compilation = _compile(
        claim_families=(
            "dependency_reachability",
            "api_contract",
            "security_property",
            "semantic_equivalence",
        ),
        residual_refs=(RESIDUAL_REF,),
        formal_plan_effects=("effect:persist-value",),
    )

    assert isinstance(compilation, CodeProofObligationCompilation)
    families = {item.claim_family for item in compilation.items}
    assert ClaimFamily.DEPENDENCY_REACHABILITY.value in families
    assert ClaimFamily.API_CONTRACT.value in families
    assert ClaimFamily.SECURITY_PROPERTY.value in families
    assert ClaimFamily.SEMANTIC_EQUIVALENCE.value in families
    assert ClaimFamily.SRT_STRUCTURAL.value in families

    dep = compilation.by_family("dependency_reachability")[0]
    api = compilation.by_family("api_contract")[0]
    sec = compilation.by_family("security_property")[0]
    sem = compilation.by_family("semantic_equivalence")[0]
    srt = compilation.by_family("srt_structural")[0]

    for item in (dep, api, sec, sem, srt):
        assert item.status is ObligationCompileStatus.OPEN
        assert item.obligation is not None
        assert item.claim is not None
        assert item.claim.status is ClaimStatus.OPEN
        assert item.premise_ids
        assert item.assumption_ids == tuple(sorted(ASSUMPTIONS))
        assert item.invalidation_selectors
        assert item.cache_key_id.startswith("proof-cache-key:")
        # No gold / secrets in serialized claim or obligation metadata.
        payload = item.to_dict()
        blob = str(payload)
        assert "SECRET" not in blob
        assert "gold_body" not in blob
        assert "BEGIN PRIVATE" not in blob

    assert RESIDUAL_REF in srt.residual_ref_ids
    assert RESIDUAL_REF in srt.premise_ids
    assert any(
        selector["kind"] == InvalidationSelectorKind.PREMISE_SET.value
        for selector in dep.invalidation_selectors
    )
    assert any(
        selector["kind"] == InvalidationSelectorKind.ASSUMPTION_SET.value
        for selector in dep.invalidation_selectors
    )
    assert compilation.plan_effect_ids == ("effect:persist-value",)
    assert compilation.premise_digest == premise_set_digest(compilation.premise_ids)
    assert compilation.assumption_digest == assumption_set_digest(ASSUMPTIONS)


def test_unsupported_and_not_measured_remain_distinguishable() -> None:
    unsupported = _compile(
        property_ids=("property:unsupported-proof-fail-closed",),
    )
    not_measured = _compile(
        requests=(
            CodeProofCompileRequest(
                claim_family="security_property",
                force_not_measured=True,
            ),
        ),
    )
    # Empty-scope candidate: only non-python change → no AST scopes.
    empty_ast = compile_code_proof_obligations(
        candidate_diff=[
            {
                "path": "schema/api.json",
                "status": "modify",
                "before_source": '{"v": 1}',
                "after_source": '{"v": 2}',
            }
        ],
        repository_tree_id=TREE_A,
        repository_id=REPO,
        claim_families=("dependency_reachability",),
        premise_ids=PREMISES,
        assumption_ids=ASSUMPTIONS,
        toolchain_id=TOOLCHAIN,
        policy_id=POLICY,
    )

    u_item = unsupported.items[0]
    n_item = not_measured.items[0]
    e_item = empty_ast.items[0]

    assert u_item.status is ObligationCompileStatus.UNSUPPORTED
    assert u_item.claim.status is ClaimStatus.UNSUPPORTED
    assert n_item.status is ObligationCompileStatus.NOT_MEASURED
    assert n_item.claim.status is ClaimStatus.NOT_MEASURED
    assert e_item.status is ObligationCompileStatus.NOT_MEASURED
    assert e_item.claim.status is ClaimStatus.NOT_MEASURED

    # Distinct lifecycle values — never collapse into each other or refuted.
    assert u_item.status != n_item.status
    assert u_item.claim.status is not ClaimStatus.REFUTED
    assert n_item.claim.status is not ClaimStatus.REFUTED
    assert e_item.claim.status is not ClaimStatus.REFUTED
    assert "not_measured" in n_item.reason_codes
    assert "unsupported" in u_item.reason_codes or u_item.claim_family == "unsupported"


def test_cache_key_binds_property_catalog_tree_scope_premises_toolchain_policy_assurance() -> None:
    translator = "translator:python-to-lean@1"
    solver = "solver:z3@4.13"
    kernel = "kernel:lean-4.19"
    registry = "registry:reviewed-v3"
    compilation = _compile(
        property_ids=("property:lease-uniqueness-and-fencing",),
        translator_id=translator,
        solver_id=solver,
        kernel_id=kernel,
        theorem_registry_id=registry,
    )
    item = compilation.items[0]
    obligation = item.obligation
    assert obligation is not None

    def _key(**overrides):
        values = {
            "translator_id": translator,
            "solver_id": solver,
            "kernel_id": kernel,
            "toolchain_id": TOOLCHAIN,
            "theorem_registry_id": registry,
            "policy_id": POLICY,
            "property_id": item.property_id,
            "catalog_version": compilation.catalog_version,
            "catalog_id": compilation.catalog_id,
            "assumption_ids": item.assumption_ids,
        }
        values.update(overrides)
        return build_code_proof_cache_key(obligation, **values)

    key = _key()
    assert key.key_id == item.cache_key_id

    # Tree / policy / toolchain / catalog / property / assumption drift.
    assert _key(candidate_tree=TREE_B).key_id != key.key_id
    assert _key(policy_id="policy:other").key_id != key.key_id
    assert _key(toolchain_id="toolchain:other").key_id != key.key_id
    assert _key(catalog_version="catalog-drift-9").key_id != key.key_id
    assert _key(property_id="property:other").key_id != key.key_id
    assert _key(assumption_ids=("assumption:changed",)).key_id != key.key_id

    # Required assurance is part of the obligation component.
    raised = obligation.__class__(
        **{
            **{
                field: getattr(obligation, field)
                for field in (
                    "repository_id",
                    "repository_tree_id",
                    "ast_scope_ids",
                    "statement",
                    "template_id",
                    "template_version",
                    "template_semantic_hash",
                    "premise_ids",
                    "invariant_class",
                    "task_id",
                    "fallback_checks",
                    "metadata",
                )
            },
            "required_assurance": AssuranceLevel.ATTESTED,
        }
    )
    other_assurance = build_code_proof_cache_key(
        raised,
        translator_id=translator,
        solver_id=solver,
        kernel_id=kernel,
        toolchain_id=TOOLCHAIN,
        theorem_registry_id=registry,
        policy_id=POLICY,
        property_id=item.property_id,
        catalog_version=compilation.catalog_version,
        catalog_id=compilation.catalog_id,
        assumption_ids=item.assumption_ids,
    )
    assert other_assurance.key_id != key.key_id

    # Compact identity also binds the required surface.
    compact = compiled_obligation_cache_identity(
        property_id=item.property_id,
        catalog_version=compilation.catalog_version,
        catalog_id=compilation.catalog_id,
        repository_tree_id=TREE_A,
        ast_scope_ids=obligation.ast_scope_ids,
        premise_ids=item.premise_ids,
        assumption_ids=item.assumption_ids,
        toolchain_id=TOOLCHAIN,
        policy_id=POLICY,
        required_assurance=obligation.required_assurance,
        template_id=obligation.template_id,
        template_version=obligation.template_version,
        template_semantic_hash=obligation.template_semantic_hash,
        obligation_id=obligation.obligation_id,
    )
    compact_tree = compiled_obligation_cache_identity(
        property_id=item.property_id,
        catalog_version=compilation.catalog_version,
        catalog_id=compilation.catalog_id,
        repository_tree_id=TREE_B,
        ast_scope_ids=obligation.ast_scope_ids,
        premise_ids=item.premise_ids,
        assumption_ids=item.assumption_ids,
        toolchain_id=TOOLCHAIN,
        policy_id=POLICY,
        required_assurance=obligation.required_assurance,
        template_id=obligation.template_id,
        template_version=obligation.template_version,
        template_semantic_hash=obligation.template_semantic_hash,
        obligation_id=obligation.obligation_id,
    )
    assert compact
    assert compact != compact_tree
    assert item.metadata.get("compact_cache_identity") == compact


def test_claim_records_carry_explicit_premise_assumption_and_invalidators() -> None:
    compilation = _compile(
        property_ids=("property:projection-equivalence",),
        residual_refs=({"residual_ref_id": RESIDUAL_REF, "facet": "rule_cardinality_preserved"},),
    )
    item = compilation.items[0]
    claim = item.claim
    assert claim.premise_ids == item.premise_ids
    assert claim.assumption_ids == item.assumption_ids
    assert claim.property_id == item.property_id
    assert claim.catalog_version == compilation.catalog_version
    assert claim.toolchain_id == TOOLCHAIN
    assert claim.policy_id == POLICY
    kinds = {selector.kind for selector in claim.invalidation_selectors}
    assert InvalidationSelectorKind.REPOSITORY_TREE in kinds
    assert InvalidationSelectorKind.PREMISE_SET in kinds
    assert InvalidationSelectorKind.ASSUMPTION_SET in kinds
    assert InvalidationSelectorKind.PROPERTY in kinds
    assert InvalidationSelectorKind.CATALOG in kinds
    assert InvalidationSelectorKind.TOOLCHAIN in kinds
    assert InvalidationSelectorKind.POLICY in kinds
    assert InvalidationSelectorKind.REQUIRED_ASSURANCE in kinds
    assert RESIDUAL_REF in claim.premise_ids


def test_formal_plan_effects_bind_as_premises_without_source_dumps() -> None:
    compilation = _compile(
        property_ids=("property:legal-state-transitions",),
        formal_plan_effects=(
            {"effect_id": "effect:transition-running"},
            "effect:emit-save",
        ),
    )
    item = compilation.items[0]
    assert "plan-effect:effect:transition-running" in item.premise_ids
    assert "plan-effect:effect:emit-save" in item.premise_ids
    assert compilation.plan_effect_ids == (
        "effect:emit-save",
        "effect:transition-running",
    )
    # Compiler refuses to treat a source dump as an effect-linked premise.
    with pytest.raises(PremiseValidationError):
        _compile(
            property_ids=("property:legal-state-transitions",),
            premise_ids=("def dump():\n    pass\n",),
        )


def test_compilation_is_content_addressed_and_deterministic() -> None:
    first = _compile(
        claim_families=("security_property", "semantic_equivalence"),
        residual_refs=(RESIDUAL_REF,),
    )
    second = _compile(
        claim_families=("semantic_equivalence", "security_property"),
        residual_refs=(RESIDUAL_REF,),
    )
    assert first.compilation_id == second.compilation_id
    assert tuple(item.cache_key_id for item in first.items) == tuple(
        item.cache_key_id for item in second.items
    )
    assert tuple(item.obligation_id for item in first.items) == tuple(
        item.obligation_id for item in second.items
    )
    payload = first.to_dict()
    assert payload["schema"]
    assert payload["premise_digest"] == first.premise_digest
    assert "gold" not in str(payload).lower() or "gold" not in str(
        [item.metadata for item in first.items]
    )
