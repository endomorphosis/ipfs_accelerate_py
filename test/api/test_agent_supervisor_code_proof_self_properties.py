"""CBP-120: supervisor self-properties (lease, merge, DAG, freshness).

Covers exact ReviewedCodeShape template selection, always-on / policy-gated
obligation compilation, prove→cache→reproof, mutation invalidation, and warm
cache hits for the closed self-property population.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.code_proof_obligations import (
    CandidateDiffEntry,
    DiffChangeKind,
    ObligationCompileStatus,
    ProofCacheMetrics,
    SupervisorSelfPropertyPolicy,
    compile_supervisor_self_properties,
    default_supervisor_self_property_ids,
    default_supervisor_self_property_shapes,
    evaluate_supervisor_self_property_mutations,
    prove_supervisor_self_properties,
    select_supervisor_self_templates,
)
from ipfs_accelerate_py.agent_supervisor.proof.code_proof_reproof import (
    InvalidationReason,
    ReproofDisposition,
)
from ipfs_accelerate_py.agent_supervisor.proof.code_property_catalog import (
    DEFAULT_CODE_PROPERTY_CATALOG,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_cache import (
    FormalVerificationCache,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    AssuranceLevel,
    EvidenceAuthority,
    EvidenceKind,
    EvidenceVerdict,
    ProofEvidence,
    ProofReceipt,
    ProofVerdict,
    ResourceBudget,
)
from ipfs_accelerate_py.agent_supervisor.proof_obligation_templates import (
    DEFAULT_TEMPLATE_REGISTRY,
    ReviewedCodeShape,
)


SELF_SOURCE = """\
class LeaseStore:
    def acquire_or_mutate(self, resource_id: str, fencing_token: int) -> bool:
        self.state = "held"
        return True

class MergeOperator:
    def apply(self, left, right):
        return left

class DirectedAcyclicGraph:
    def edge_update(self, src: str, dst: str) -> None:
        self.edges = (src, dst)

class ProofEvidenceGate:
    def validity_check(self, tree_id: str) -> bool:
        return True
"""

SELF_SOURCE_MUTATED = """\
class LeaseStore:
    def acquire_or_mutate(self, resource_id: str, fencing_token: int) -> bool:
        self.state = "held"
        return fencing_token > 0

class MergeOperator:
    def apply(self, left, right):
        return right

class DirectedAcyclicGraph:
    def edge_update(self, src: str, dst: str) -> None:
        self.edges = (dst, src)

class ProofEvidenceGate:
    def validity_check(self, tree_id: str) -> bool:
        return bool(tree_id)
"""

TREE_A = "git-tree:cbp-120-a"
TREE_B = "git-tree:cbp-120-b"
REPO = "repository:sha256:cbp-120-self"
TOOLCHAIN = "toolchain:self-v1"
POLICY = "policy:self-v1"
PREMISES = ("premise:self-lease", "premise:self-merge")
ASSUMPTIONS = ("assumption:serializable-snapshot",)

REQUIRED_SHAPES = (
    ReviewedCodeShape.LEASE_UNIQUENESS_AND_FENCING,
    ReviewedCodeShape.MERGE_IDEMPOTENCE,
    ReviewedCodeShape.DAG_ACYCLICITY,
    ReviewedCodeShape.EVIDENCE_FRESHNESS,
)

REQUIRED_TEMPLATE_IDS = (
    "lease-uniqueness-and-fencing",
    "merge-idempotence",
    "dag-acyclicity",
    "evidence-freshness",
)

REQUIRED_PROPERTY_IDS = (
    "property:lease-uniqueness-and-fencing",
    "property:merge-idempotence",
    "property:dag-acyclicity",
    "property:evidence-freshness",
)


def _budget() -> ResourceBudget:
    return ResourceBudget(
        wall_time_ms=10_000,
        cpu_time_ms=8_000,
        memory_bytes=64 * 1024 * 1024,
        max_processes=2,
        max_premises=8,
        network_allowed=False,
    )


REPROOF_KW = dict(
    translator_id="translator:python-to-lean@1",
    solver_id="solver:z3@4.13",
    kernel_id="kernel:lean-4.19",
    theorem_registry_id="registry:reviewed-v3",
    resource_budget=_budget(),
)


def _entries(source: str = SELF_SOURCE, *, blob: str = "blob:self-a") -> list[CandidateDiffEntry]:
    return [
        CandidateDiffEntry(
            new_path="src/supervisor_self_invariants.py",
            change_kind=DiffChangeKind.ADD,
            after_source=source,
            after_blob_id=blob,
        )
    ]


def _compile(
    *,
    tree: str = TREE_A,
    source: str = SELF_SOURCE,
    blob: str = "blob:self-a",
    premises: tuple[str, ...] = PREMISES,
    assumptions: tuple[str, ...] = ASSUMPTIONS,
    toolchain: str = TOOLCHAIN,
    policy: str = POLICY,
    self_policy: SupervisorSelfPropertyPolicy | bool | None = None,
    code_shapes=None,
    verify_mutation_cases: bool = True,
):
    return compile_supervisor_self_properties(
        candidate_diff=_entries(source, blob=blob),
        repository_tree_id=tree,
        repository_id=REPO,
        premise_ids=premises,
        assumption_ids=assumptions,
        toolchain_id=toolchain,
        policy_id=policy,
        task_id="CBP-120",
        catalog=DEFAULT_CODE_PROPERTY_CATALOG,
        policy=self_policy,
        code_shapes=code_shapes,
        verify_mutation_cases=verify_mutation_cases,
        **{
            k: v
            for k, v in REPROOF_KW.items()
            if k
            in (
                "translator_id",
                "solver_id",
                "kernel_id",
                "theorem_registry_id",
                "resource_budget",
            )
        },
    )


def _receipt_for(item, *, tree: str, toolchain: str = TOOLCHAIN, policy: str = POLICY):
    obligation = item.obligation
    assert obligation is not None
    kernel_id = "kernel:lean-4.19"
    return ProofReceipt(
        obligation_id=obligation.obligation_id,
        plan_id="plan:cbp-120-self",
        attempt_id="attempt:1",
        repository_id=obligation.repository_id or REPO,
        repository_tree_id=tree,
        ast_scope_ids=tuple(obligation.ast_scope_ids),
        premise_ids=tuple(obligation.premise_ids),
        translator_id="translator:python-to-lean@1",
        solver_id="solver:z3@4.13",
        kernel_id=kernel_id,
        toolchain_id=toolchain,
        theorem_registry_id="registry:reviewed-v3",
        policy_id=policy,
        resource_budget=_budget(),
        verdict=ProofVerdict.PROVED,
        evidence=(
            ProofEvidence(
                kind=EvidenceKind.KERNEL_VERIFICATION,
                authority=EvidenceAuthority.KERNEL,
                verdict=EvidenceVerdict.ACCEPTED,
                artifact_id="artifact:kernel-self",
                subject_id=obligation.obligation_id,
                verifier_id=kernel_id,
                independent=True,
                simulated=False,
            ),
        ),
        provider_id="provider:cbp-120-self",
        provider_claimed_assurance=AssuranceLevel.ATTESTED,
        started_at="2026-07-28T00:00:00Z",
        finished_at="2026-07-28T00:00:01Z",
        resource_usage={"wall_time_ms": 5, "peak_memory_bytes": 1000},
    )


def test_default_self_property_population_is_closed() -> None:
    shapes = default_supervisor_self_property_shapes()
    ids = default_supervisor_self_property_ids()
    assert shapes == REQUIRED_SHAPES
    assert ids == REQUIRED_PROPERTY_IDS
    for property_id in ids:
        assert DEFAULT_CODE_PROPERTY_CATALOG.get(property_id) is not None


def test_templates_selected_by_exact_reviewed_code_shape() -> None:
    selection = select_supervisor_self_templates(
        DEFAULT_TEMPLATE_REGISTRY,
        catalog=DEFAULT_CODE_PROPERTY_CATALOG,
    )
    assert set(selection.code_shapes) == {shape.value for shape in REQUIRED_SHAPES}
    assert set(selection.template_ids) == set(REQUIRED_TEMPLATE_IDS)
    assert set(selection.property_ids) == set(REQUIRED_PROPERTY_IDS)
    for spec in selection.specs:
        selected = DEFAULT_TEMPLATE_REGISTRY.select_for_code_shape(spec.code_shape)
        template = selected.require_supported()
        assert template.template_id == spec.template_id
        assert template.supports_code_shape(spec.code_shape)
        assert template.semantic_hash == spec.template_semantic_hash
        # Exact membership only — no fuzzy / partial shape matching.
        assert ReviewedCodeShape(spec.code_shape) in REQUIRED_SHAPES


def test_mutation_cases_pass_for_all_self_property_templates() -> None:
    outcomes = evaluate_supervisor_self_property_mutations(DEFAULT_TEMPLATE_REGISTRY)
    assert set(outcomes) == set(REQUIRED_TEMPLATE_IDS)
    for template_id, cases in outcomes.items():
        assert cases, f"{template_id} must declare mutation cases"
        assert all(cases.values()), f"{template_id} mutation failures: {cases}"


def test_policy_gate_can_disable_or_subset_self_properties() -> None:
    disabled = compile_supervisor_self_properties(
        candidate_diff=_entries(),
        repository_tree_id=TREE_A,
        repository_id=REPO,
        policy=False,
        verify_mutation_cases=False,
    )
    assert disabled.items == ()
    assert disabled.metadata.get("supervisor_self_properties") is True

    subset = select_supervisor_self_templates(
        policy=SupervisorSelfPropertyPolicy(
            enabled=True,
            always_on=False,
            enabled_property_ids=(
                "property:lease-uniqueness-and-fencing",
                "property:merge-idempotence",
            ),
        ),
        catalog=DEFAULT_CODE_PROPERTY_CATALOG,
    )
    assert set(subset.property_ids) == {
        "property:lease-uniqueness-and-fencing",
        "property:merge-idempotence",
    }

    by_shape = select_supervisor_self_templates(
        policy=SupervisorSelfPropertyPolicy(
            enabled=True,
            always_on=False,
            enabled_code_shapes=(
                ReviewedCodeShape.DAG_ACYCLICITY.value,
                ReviewedCodeShape.EVIDENCE_FRESHNESS.value,
            ),
        ),
        catalog=DEFAULT_CODE_PROPERTY_CATALOG,
    )
    assert set(by_shape.code_shapes) == {
        ReviewedCodeShape.DAG_ACYCLICITY.value,
        ReviewedCodeShape.EVIDENCE_FRESHNESS.value,
    }


def test_compile_always_on_self_properties_bind_exact_shapes() -> None:
    compilation = _compile()
    open_items = [
        item
        for item in compilation.items
        if item.status is ObligationCompileStatus.OPEN and item.obligation is not None
    ]
    assert {item.property_id for item in open_items} == set(REQUIRED_PROPERTY_IDS)
    assert {item.template_id for item in open_items} == set(REQUIRED_TEMPLATE_IDS)

    for item in open_items:
        obligation = item.obligation
        assert obligation is not None
        assert obligation.template_semantic_hash
        shape = str((obligation.metadata or {}).get("code_shape") or "")
        assert shape in {s.value for s in REQUIRED_SHAPES}
        assert DEFAULT_TEMPLATE_REGISTRY.require(
            obligation.template_id, obligation.template_version
        ).supports_code_shape(shape)
        assert item.cache_key_id
        assert "supervisor_self_property" in (item.metadata or {}) or (
            obligation.metadata or {}
        ).get("supervisor_self_property") is True


def test_prove_cache_reproof_warm_path_hits_for_all_self_shapes(
    tmp_path: Path,
) -> None:
    cache = FormalVerificationCache(tmp_path)
    compilation = _compile(tree=TREE_A)
    open_items = [
        item for item in compilation.items if item.obligation is not None
    ]
    assert len(open_items) == len(REQUIRED_PROPERTY_IDS)

    calls: dict[str, int] = {}

    def prove(item, key):
        property_id = item.property_id
        calls[property_id] = calls.get(property_id, 0) + 1
        return _receipt_for(
            item,
            tree=TREE_A,
            toolchain=str(key.toolchain),
            policy=str(key.policy),
        )

    metrics = ProofCacheMetrics()
    cold = prove_supervisor_self_properties(
        cache, compilation, prove=prove, metrics=metrics, **REPROOF_KW
    )
    warm = prove_supervisor_self_properties(
        cache, compilation, prove=prove, metrics=metrics, **REPROOF_KW
    )

    assert cold.re_solved == len(REQUIRED_PROPERTY_IDS)
    assert cold.cache_hits == 0
    assert warm.cache_hits == len(REQUIRED_PROPERTY_IDS)
    assert warm.re_solved == 0

    for result in warm.results:
        assert result.disposition is ReproofDisposition.CACHE_HIT
        assert result.from_cache is True
        assert result.provenance.get("provider_calls") == 0
        assert InvalidationReason.AUTHORITATIVE_CACHE_HIT.value in result.reason_codes

    # Provider invoked once per self-property on the cold path only.
    assert set(calls) == set(REQUIRED_PROPERTY_IDS)
    assert all(count == 1 for count in calls.values())


def test_mutations_invalidate_self_property_cache_entries(tmp_path: Path) -> None:
    cache = FormalVerificationCache(tmp_path)
    base = _compile(tree=TREE_A, blob="blob:base", source=SELF_SOURCE)
    mutated_source = _compile(
        tree=TREE_A,
        blob="blob:mutated",
        source=SELF_SOURCE_MUTATED,
    )
    mutated_tree = _compile(
        tree=TREE_B,
        blob="blob:tree-b",
        source=SELF_SOURCE,
    )
    mutated_premises = _compile(
        tree=TREE_A,
        blob="blob:base",
        source=SELF_SOURCE,
        premises=("premise:self-lease", "premise:self-merge", "premise:extra"),
    )
    mutated_toolchain = _compile(
        tree=TREE_A,
        blob="blob:base",
        source=SELF_SOURCE,
        toolchain="toolchain:self-v2",
    )

    calls = {"n": 0}

    def prove(item, key):
        calls["n"] += 1
        return _receipt_for(
            item,
            tree=str(key.candidate_tree),
            toolchain=str(key.toolchain),
            policy=str(key.policy),
        )

    prove_supervisor_self_properties(cache, base, prove=prove, **REPROOF_KW)
    cold_calls = calls["n"]
    assert cold_calls == len(REQUIRED_PROPERTY_IDS)

    # Warm identical binding must not re-invoke the provider.
    warm = prove_supervisor_self_properties(cache, base, prove=prove, **REPROOF_KW)
    assert warm.cache_hits == len(REQUIRED_PROPERTY_IDS)
    assert calls["n"] == cold_calls

    for mutated, expected_reasons in (
        (
            mutated_source,
            {
                InvalidationReason.CACHE_KEY_CHANGED.value,
                InvalidationReason.AST_SCOPE_CHANGED.value,
                InvalidationReason.PATH_CHANGED.value,
            },
        ),
        (
            mutated_tree,
            {
                InvalidationReason.REPOSITORY_TREE_CHANGED.value,
                InvalidationReason.CACHE_KEY_CHANGED.value,
            },
        ),
        (
            mutated_premises,
            {
                InvalidationReason.PREMISE_DIGEST_CHANGED.value,
                InvalidationReason.CACHE_KEY_CHANGED.value,
            },
        ),
        (
            mutated_toolchain,
            {
                InvalidationReason.TOOLCHAIN_CHANGED.value,
                InvalidationReason.CACHE_KEY_CHANGED.value,
            },
        ),
    ):
        before = calls["n"]
        report = prove_supervisor_self_properties(
            cache,
            mutated,
            prove=prove,
            previous=base,
            changed_paths=["src/supervisor_self_invariants.py"],
            **REPROOF_KW,
        )
        solved = [
            item
            for item in report.results
            if item.disposition is ReproofDisposition.RE_SOLVED
        ]
        assert solved, f"expected re-solve for mutation; got {report.to_dict()}"
        assert calls["n"] > before
        for item in solved:
            assert item.from_cache is False
            codes = set(item.reason_codes)
            assert codes & expected_reasons or (
                InvalidationReason.CACHE_KEY_CHANGED.value in codes
            ), codes


def test_policy_gated_subset_still_warms_cache(tmp_path: Path) -> None:
    cache = FormalVerificationCache(tmp_path)
    policy = SupervisorSelfPropertyPolicy(
        enabled=True,
        always_on=False,
        enabled_code_shapes=(
            ReviewedCodeShape.LEASE_UNIQUENESS_AND_FENCING.value,
            ReviewedCodeShape.EVIDENCE_FRESHNESS.value,
        ),
    )
    compilation = _compile(self_policy=policy)
    assert {item.property_id for item in compilation.items if item.obligation} == {
        "property:lease-uniqueness-and-fencing",
        "property:evidence-freshness",
    }

    calls = {"n": 0}

    def prove(item, key):
        calls["n"] += 1
        return _receipt_for(
            item,
            tree=TREE_A,
            toolchain=str(key.toolchain),
            policy=str(key.policy),
        )

    cold = prove_supervisor_self_properties(
        cache, compilation, prove=prove, **REPROOF_KW
    )
    warm = prove_supervisor_self_properties(
        cache, compilation, prove=prove, **REPROOF_KW
    )
    assert cold.re_solved == 2
    assert warm.cache_hits == 2
    assert calls["n"] == 2
