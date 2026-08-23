"""Hermetic PCAR-010 contract-candidate extraction tests."""

from __future__ import annotations

import json

import pytest

from ipfs_accelerate_py.agent_supervisor.architecture_refactorer.architecture_ir import (
    ArchitectureEdge,
    ArchitectureIR,
    ArchitectureNode,
)
from ipfs_accelerate_py.agent_supervisor.architecture_refactorer.contract_extractor import (
    CANDIDATE_TIER_ONLY,
    CLOSED_AMBIGUITY_KINDS,
    CLOSED_COMPARISON_DISPOSITIONS,
    CLOSED_CONTRACT_DIMENSIONS,
    CLOSED_CONTRACT_TIERS,
    CLOSED_DIMENSION_STATUSES,
    CLOSED_EVIDENCE_POLARITIES,
    CLOSED_EVIDENCE_SOURCES,
    CONTRACT_AMBIGUITY_EVIDENCE,
    CONTRACT_AMBIGUITY_SCHEMA,
    CONTRACT_CANDIDATE_EVIDENCE,
    CONTRACT_CANDIDATE_SCHEMA,
    CONTRACT_CANDIDATE_VERSION,
    CONTRACT_EXTRACTION_SCHEMA,
    DEFAULT_FRESHNESS,
    EFFECT_CLASS,
    EXTRACTOR_CAN_HIDE_CONFLICTS,
    EXTRACTOR_CAN_PROMOTE_REQUIREMENT,
    EXTRACTOR_CAN_RESOLVE_IMPLEMENTATION_WINS,
    EXTRACTOR_CAN_TREAT_TESTS_AS_COMPLETE,
    EXTRACTOR_CAN_TREAT_UNMARKED_DOCS_AS_AUTHORITY,
    EXTRACTOR_IDENTITY,
    HIDDEN_CONFLICT_PROHIBITED,
    IMPLEMENTATION_WINS_PROHIBITED,
    REPETITION_IS_NOT_A_REQUIREMENT,
    REQUIRED_AMBIGUITY_KINDS,
    REQUIRED_CONTRACT_DIMENSIONS,
    SOURCE_PRECEDENCE,
    SOURCE_PRECEDENCE_RANK,
    TASK_ID,
    TEST_COMPLETENESS_ASSUMPTION_PROHIBITED,
    UNMARKED_DOCUMENTATION_IS_NOT_AUTHORITY,
    AmbiguityKind,
    ComparisonDisposition,
    ContractAmbiguity,
    ContractCandidate,
    ContractCandidateExtractor,
    ContractDimension,
    ContractDimensionRecord,
    ContractEvidenceSource,
    ContractEvidenceUnit,
    ContractExtractionResult,
    ContractExtractorAuthorityError,
    ContractExtractorError,
    ContractTier,
    DimensionStatus,
    EvidencePolarity,
    build_contract_extraction_result,
    compare_contract_evidence,
    evidence_comparison,
    extract_contract_candidates,
    mine_graph_evidence,
    refuse_hidden_conflict,
    refuse_implementation_resolution,
    refuse_requirement_promotion,
    refuse_test_completeness,
    refuse_unmarked_documentation_authority,
    source_precedence_rank,
)
from ipfs_accelerate_py.agent_supervisor.architecture_refactorer.contracts import (
    Confidence,
    EdgeKind,
    NodeKind,
    SourceFactIdentity,
    SourceSpan,
)
from ipfs_accelerate_py.utils.cid_utils import cid_for_dag_json, validate_cid

_TREE = "a698da9e4b54e2929adacb613bc61ba3e72eed58"
_FRESHNESS = "pcar-010-fixture"
_EXTRACTOR = "pcar-010-fixture"
_TYPE_PATH = "ipfs_accelerate_py/agent_supervisor/architecture_refactorer/contracts.py"
_SCHEMA_PATH = "ipfs_accelerate_py/agent_supervisor/control/control_contracts.py"
_TEST_PATH = "test/api/architecture_refactorer/test_contract_extractor.py"
_PROOF_PATH = "ipfs_accelerate_py/agent_supervisor/proof/program_contracts.py"
_RECEIPT_PATH = "ipfs_accelerate_py/agent_supervisor/contracts/execution.py"
_DOC_PATH = "docs/architecture/contracts.md"
_IMPL_PATH = "ipfs_accelerate_py/agent_supervisor/control/control_plane.py"
_NEG_PATH = "test/api/architecture_refactorer/test_contract_extractor_negatives.py"
_MUTANT_PATH = "test/api/architecture_refactorer/test_contract_extractor_mutants.py"
_RUNTIME_PATH = "ipfs_accelerate_py/agent_supervisor/runtime/checks.py"
_PUBLIC_PATH = "ipfs_accelerate_py/agent_supervisor/control/control_contracts.py"
_SUBJECT = "control.plane.execute"


def _span(path: str, start: int, end: int | None = None) -> SourceSpan:
    return SourceSpan(path, start, start if end is None else end)


def _fact(
    path: str,
    start: int,
    *,
    confidence: Confidence = Confidence.EXACT,
    end: int | None = None,
) -> SourceFactIdentity:
    return SourceFactIdentity(
        extractor_identity=_EXTRACTOR,
        span=_span(path, start, end),
        confidence=confidence,
        freshness=_FRESHNESS,
        repository_tree=_TREE,
    )


def _unit(
    source: ContractEvidenceSource,
    dimension: ContractDimension,
    value: str,
    path: str,
    start: int,
    *,
    polarity: EvidencePolarity = EvidencePolarity.POSITIVE,
    marked_authoritative: bool = False,
    public_contract: bool = False,
    assumes_test_completeness: bool = False,
    subject: str = _SUBJECT,
) -> ContractEvidenceUnit:
    return ContractEvidenceUnit(
        subject=subject,
        source_kind=source,
        dimension=dimension,
        value=value,
        polarity=polarity,
        marked_authoritative=marked_authoritative,
        public_contract=public_contract,
        assumes_test_completeness=assumes_test_completeness,
        provenance=_fact(path, start),
    )


_DIMENSION_VALUES: dict[ContractDimension, str] = {
    ContractDimension.INPUTS: "schema:object{task_id:string}",
    ContractDimension.OUTPUTS: "schema:object{receipt:cid}",
    ContractDimension.PRECONDITIONS: "lease-held",
    ContractDimension.POSTCONDITIONS: "receipt-sealed",
    ContractDimension.EFFECTS: "writes:control_plane.store",
    ContractDimension.FRAMES: "frame:no-sibling-write",
    ContractDimension.ERRORS: "ArchitectureContractError",
    ContractDimension.IDEMPOTENCY: "idempotent-on-task-id",
    ContractDimension.REVERSIBILITY: "rollback-to-sealed-tree",
    ContractDimension.AUTHORITY: "DuckDB-plus-Quack",
    ContractDimension.POLICY: "fail-closed",
    ContractDimension.CONFIRMATION: "signed-receipt",
    ContractDimension.BOUNDS: "cpu_ms<=7200000",
    ContractDimension.FRESHNESS: "current-tree",
    ContractDimension.OBSERVATIONS: "observes:execution-receipt",
}


def _all_dimension_units(
    source: ContractEvidenceSource = ContractEvidenceSource.TYPE,
    path: str = _TYPE_PATH,
    *,
    marked_authoritative: bool = False,
) -> tuple[ContractEvidenceUnit, ...]:
    return tuple(
        _unit(source, dimension, value, path, 10 + index, marked_authoritative=marked_authoritative)
        for index, (dimension, value) in enumerate(_DIMENSION_VALUES.items())
    )


def _extract(
    units: tuple[ContractEvidenceUnit, ...] = (),
    *,
    architecture: ArchitectureIR | None = None,
    public_contracts: tuple[ContractEvidenceUnit, ...] = (),
) -> ContractExtractionResult:
    return extract_contract_candidates(
        tuple(units),
        repository_tree=_TREE,
        freshness=_FRESHNESS,
        architecture=architecture,
        public_contracts=tuple(public_contracts),
    )


def _node(node_id: str, kind: NodeKind, path: str, start: int) -> ArchitectureNode:
    return ArchitectureNode(node_id=node_id, kind=kind, provenance=_fact(path, start))


def _edge(
    edge_id: str,
    kind: EdgeKind,
    source: str,
    target: str,
    path: str,
    start: int,
) -> ArchitectureEdge:
    return ArchitectureEdge(
        edge_id=edge_id,
        kind=kind,
        source=source,
        target=target,
        provenance=_fact(path, start),
    )


def _graph() -> ArchitectureIR:
    operation = _node("n:operation:control.plane.execute", NodeKind.OPERATION, _IMPL_PATH, 10)
    authority = _node("n:authority:quack", NodeKind.AUTHORITY, _IMPL_PATH, 20)
    policy = _node("n:policy:fail-closed", NodeKind.POLICY, _IMPL_PATH, 30)
    state = _node("n:state:control_plane.store", NodeKind.STATE, _IMPL_PATH, 40)
    receipt = _node("n:receipt:execution", NodeKind.RECEIPT, _RECEIPT_PATH, 12)
    return ArchitectureIR.from_parts(
        repository_tree=_TREE,
        freshness=_FRESHNESS,
        nodes=(operation, authority, policy, state, receipt),
        edges=(
            _edge("e-auth", EdgeKind.AUTHORIZES, authority.node_id, operation.node_id, _IMPL_PATH, 20),
            _edge("e-pol", EdgeKind.EVALUATES_POLICY, policy.node_id, operation.node_id, _IMPL_PATH, 30),
            _edge("e-mut", EdgeKind.MUTATES, operation.node_id, state.node_id, _IMPL_PATH, 40),
            _edge("e-obs", EdgeKind.OBSERVES, operation.node_id, receipt.node_id, _IMPL_PATH, 50),
            _edge("e-conf", EdgeKind.CONFIRMS, receipt.node_id, operation.node_id, _RECEIPT_PATH, 12),
        ),
    )


def test_closed_vocabulary_and_candidate_tier_invariants() -> None:
    assert CONTRACT_CANDIDATE_SCHEMA == (
        "ipfs_accelerate_py/agent-supervisor/contract-candidate@1"
    )
    assert CONTRACT_CANDIDATE_VERSION == 1
    assert CONTRACT_CANDIDATE_EVIDENCE == "pcar/contract-candidate@1"
    assert CONTRACT_AMBIGUITY_SCHEMA.endswith("contract-ambiguity@1")
    assert CONTRACT_AMBIGUITY_EVIDENCE == "pcar/contract-ambiguity@1"
    assert CONTRACT_EXTRACTION_SCHEMA.endswith("contract-extraction-result@1")
    assert EXTRACTOR_IDENTITY == "pcar-010-contract-candidate-extractor"
    assert TASK_ID == "PCAR-010"
    assert DEFAULT_FRESHNESS == "pcar-010-contract-candidate"
    assert EFFECT_CLASS == "read_only_analysis"
    assert EXTRACTOR_CAN_PROMOTE_REQUIREMENT is False
    assert EXTRACTOR_CAN_RESOLVE_IMPLEMENTATION_WINS is False
    assert EXTRACTOR_CAN_HIDE_CONFLICTS is False
    assert EXTRACTOR_CAN_TREAT_TESTS_AS_COMPLETE is False
    assert EXTRACTOR_CAN_TREAT_UNMARKED_DOCS_AS_AUTHORITY is False
    assert CANDIDATE_TIER_ONLY is True
    assert IMPLEMENTATION_WINS_PROHIBITED is True
    assert HIDDEN_CONFLICT_PROHIBITED is True
    assert TEST_COMPLETENESS_ASSUMPTION_PROHIBITED is True
    assert UNMARKED_DOCUMENTATION_IS_NOT_AUTHORITY is True
    assert REPETITION_IS_NOT_A_REQUIREMENT is True
    assert tuple(item.value for item in REQUIRED_CONTRACT_DIMENSIONS) == (
        "inputs",
        "outputs",
        "preconditions",
        "postconditions",
        "effects",
        "frames",
        "errors",
        "idempotency",
        "reversibility",
        "authority",
        "policy",
        "confirmation",
        "bounds",
        "freshness",
        "observations",
    )
    assert CLOSED_CONTRACT_DIMENSIONS == {item.value for item in ContractDimension}
    assert CLOSED_EVIDENCE_SOURCES == {item.value for item in ContractEvidenceSource}
    assert CLOSED_EVIDENCE_POLARITIES == {"positive", "negative", "mutant"}
    assert CLOSED_DIMENSION_STATUSES == {"present", "absent", "conflicted", "negative"}
    assert CLOSED_AMBIGUITY_KINDS == {item.value for item in AmbiguityKind}
    assert tuple(item.value for item in REQUIRED_AMBIGUITY_KINDS) == (
        "conflicting_values",
        "implementation_wins_rejected",
        "unmarked_documentation",
        "negative_contradicts_positive",
        "mutant_survives",
        "test_completeness_assumed",
        "source_precedence_unresolved",
    )
    assert CLOSED_COMPARISON_DISPOSITIONS == {"agree", "conflict", "repetition"}
    assert CLOSED_CONTRACT_TIERS == {"candidate"}
    assert SOURCE_PRECEDENCE[-1] is ContractEvidenceSource.IMPLEMENTATION
    assert source_precedence_rank(ContractEvidenceSource.PUBLIC_CONTRACT) == 0
    assert SOURCE_PRECEDENCE_RANK[ContractEvidenceSource.IMPLEMENTATION] == 100
    with pytest.raises(ValueError):
        ContractDimension("aesthetic score")
    with pytest.raises(ValueError):
        ContractEvidenceSource("wiki")
    with pytest.raises(ValueError):
        AmbiguityKind("ignore")
    with pytest.raises(ValueError):
        ContractTier("requirement")
    with pytest.raises(ValueError):
        DimensionStatus("maybe")
    with pytest.raises(ValueError):
        ComparisonDisposition("winner")


def test_all_contract_dimensions_are_emitted_and_absent_remain_explicit() -> None:
    result = _extract(_all_dimension_units())
    candidate = result.candidate(_SUBJECT)
    assert tuple(item.dimension for item in candidate.dimensions) == REQUIRED_CONTRACT_DIMENSIONS
    for dimension, value in _DIMENSION_VALUES.items():
        record = candidate.dimension(dimension)
        assert record.status is DimensionStatus.PRESENT
        assert record.values == (value,)
        assert record.absent is False
        assert ContractEvidenceSource.TYPE in record.source_kinds
    partial = _extract(
        (
            _unit(
                ContractEvidenceSource.SCHEMA,
                ContractDimension.INPUTS,
                "schema:object{path:string}",
                _SCHEMA_PATH,
                4,
            ),
        )
    )
    missing = partial.candidate(_SUBJECT)
    present = missing.dimension(ContractDimension.INPUTS)
    assert present.status is DimensionStatus.PRESENT
    assert present.values == ("schema:object{path:string}",)
    for dimension in REQUIRED_CONTRACT_DIMENSIONS:
        if dimension is ContractDimension.INPUTS:
            continue
        record = missing.dimension(dimension)
        assert record.status is DimensionStatus.ABSENT
        assert record.values == ()
        assert record.absent is True
        assert record.source_kinds == ()
    assert missing.tier is ContractTier.CANDIDATE
    assert result.candidate_tier is True


def test_source_comparison_matrix_retains_precedence_without_resolution() -> None:
    units = (
        _unit(
            ContractEvidenceSource.PUBLIC_CONTRACT,
            ContractDimension.OUTPUTS,
            "bytes",
            _PUBLIC_PATH,
            8,
        ),
        _unit(
            ContractEvidenceSource.TYPE,
            ContractDimension.OUTPUTS,
            "bytes",
            _TYPE_PATH,
            12,
        ),
        _unit(
            ContractEvidenceSource.PROOF,
            ContractDimension.OUTPUTS,
            "bytes",
            _PROOF_PATH,
            16,
        ),
        _unit(
            ContractEvidenceSource.ACCEPTED_RECEIPT,
            ContractDimension.OUTPUTS,
            "bytes",
            _RECEIPT_PATH,
            20,
        ),
        _unit(
            ContractEvidenceSource.TEST,
            ContractDimension.OUTPUTS,
            "str",
            _TEST_PATH,
            24,
        ),
        _unit(
            ContractEvidenceSource.IMPLEMENTATION,
            ContractDimension.OUTPUTS,
            "str",
            _IMPL_PATH,
            28,
        ),
    )
    comparisons = evidence_comparison(units)
    assert comparisons == compare_contract_evidence(units)
    assert comparisons
    by_pair = {
        (item.left_source, item.right_source): item
        for item in comparisons
    }
    public_type = by_pair[(ContractEvidenceSource.PUBLIC_CONTRACT, ContractEvidenceSource.TYPE)]
    assert public_type.disposition is ComparisonDisposition.AGREE
    assert public_type.left_precedence == 0
    assert public_type.right_precedence == 1
    assert public_type.retained_conflict is False
    conflict = next(item for item in comparisons if item.disposition is ComparisonDisposition.CONFLICT)
    assert conflict.retained_conflict is True
    assert {conflict.left_value, conflict.right_value} == {"bytes", "str"}
    result = _extract(units)
    candidate = result.candidate(_SUBJECT)
    outputs = candidate.dimension(ContractDimension.OUTPUTS)
    assert outputs.status is DimensionStatus.CONFLICTED
    assert outputs.values == ("bytes", "str")
    kinds = {item.kind for item in candidate.ambiguities}
    assert AmbiguityKind.CONFLICTING_VALUES in kinds
    assert AmbiguityKind.SOURCE_PRECEDENCE_UNRESOLVED in kinds
    unresolved = candidate.ambiguities_of(AmbiguityKind.SOURCE_PRECEDENCE_UNRESOLVED)
    assert unresolved
    assert "does not select a winner" in unresolved[0].message


def test_conflicting_evidence_emits_typed_contract_ambiguity() -> None:
    result = _extract(
        (
            _unit(
                ContractEvidenceSource.SCHEMA,
                ContractDimension.INPUTS,
                "path:string",
                _SCHEMA_PATH,
                3,
            ),
            _unit(
                ContractEvidenceSource.TEST,
                ContractDimension.INPUTS,
                "path:bytes",
                _TEST_PATH,
                9,
            ),
        )
    )
    candidate = result.candidate(_SUBJECT)
    assert candidate.dimension(ContractDimension.INPUTS).status is DimensionStatus.CONFLICTED
    conflicts = candidate.ambiguities_of(AmbiguityKind.CONFLICTING_VALUES)
    assert conflicts
    assert conflicts[0].retained is True
    assert conflicts[0].values == ("path:bytes", "path:string")
    assert result.ambiguities == candidate.ambiguities
    assert all(item.retained for item in result.ambiguities)


def test_implementation_cannot_win_or_hide_conflicts() -> None:
    result = _extract(
        (
            _unit(
                ContractEvidenceSource.PUBLIC_CONTRACT,
                ContractDimension.ERRORS,
                "NotFound",
                _PUBLIC_PATH,
                6,
            ),
            _unit(
                ContractEvidenceSource.IMPLEMENTATION,
                ContractDimension.ERRORS,
                "RuntimeError",
                _IMPL_PATH,
                18,
            ),
        )
    )
    candidate = result.candidate(_SUBJECT)
    rejected = candidate.ambiguities_of(AmbiguityKind.IMPLEMENTATION_WINS_REJECTED)
    assert rejected
    assert "implementation-wins" in rejected[0].message
    assert candidate.dimension(ContractDimension.ERRORS).values == (
        "NotFound",
        "RuntimeError",
    )
    with pytest.raises(ContractExtractorError, match="implementation cannot be marked"):
        _unit(
            ContractEvidenceSource.IMPLEMENTATION,
            ContractDimension.ERRORS,
            "RuntimeError",
            _IMPL_PATH,
            18,
            marked_authoritative=True,
        )


def test_negative_and_mutant_evidence_is_retained() -> None:
    negative = _extract(
        (
            _unit(
                ContractEvidenceSource.TYPE,
                ContractDimension.OUTPUTS,
                "bytes",
                _TYPE_PATH,
                4,
            ),
            _unit(
                ContractEvidenceSource.NEGATIVE_TEST,
                ContractDimension.OUTPUTS,
                "bytes",
                _NEG_PATH,
                11,
                polarity=EvidencePolarity.NEGATIVE,
            ),
        )
    )
    denied = negative.candidate(_SUBJECT).ambiguities_of(
        AmbiguityKind.NEGATIVE_CONTRADICTS_POSITIVE
    )
    assert denied
    assert negative.candidate(_SUBJECT).dimension(ContractDimension.OUTPUTS).status is (
        DimensionStatus.CONFLICTED
    )
    mutant_same = _extract(
        (
            _unit(
                ContractEvidenceSource.PROOF,
                ContractDimension.IDEMPOTENCY,
                "pure",
                _PROOF_PATH,
                5,
            ),
            _unit(
                ContractEvidenceSource.MUTANT,
                ContractDimension.IDEMPOTENCY,
                "pure",
                _MUTANT_PATH,
                7,
                polarity=EvidencePolarity.MUTANT,
            ),
        )
    )
    survived = mutant_same.candidate(_SUBJECT).ambiguities_of(AmbiguityKind.MUTANT_SURVIVES)
    assert survived
    mutant_diff = _extract(
        (
            _unit(
                ContractEvidenceSource.TYPE,
                ContractDimension.REVERSIBILITY,
                "rollback",
                _TYPE_PATH,
                2,
            ),
            _unit(
                ContractEvidenceSource.MUTANT,
                ContractDimension.REVERSIBILITY,
                "irreversible",
                _MUTANT_PATH,
                8,
                polarity=EvidencePolarity.MUTANT,
            ),
        )
    )
    mutant_conflict = mutant_diff.candidate(_SUBJECT).ambiguities_of(
        AmbiguityKind.NEGATIVE_CONTRADICTS_POSITIVE
    )
    assert mutant_conflict
    only_negative = _extract(
        (
            _unit(
                ContractEvidenceSource.NEGATIVE_TEST,
                ContractDimension.BOUNDS,
                "unbounded",
                _NEG_PATH,
                3,
                polarity=EvidencePolarity.NEGATIVE,
            ),
        )
    )
    bounds = only_negative.candidate(_SUBJECT).dimension(ContractDimension.BOUNDS)
    assert bounds.status is DimensionStatus.NEGATIVE
    assert bounds.values == ("unbounded",)


def test_unmarked_documentation_is_not_authority() -> None:
    unmarked = _extract(
        (
            _unit(
                ContractEvidenceSource.AUTHORITATIVE_DOCUMENT,
                ContractDimension.POLICY,
                "best-effort",
                _DOC_PATH,
                4,
                marked_authoritative=False,
            ),
        )
    )
    candidate = unmarked.candidate(_SUBJECT)
    assert candidate.dimension(ContractDimension.POLICY).status is DimensionStatus.ABSENT
    docs = candidate.ambiguities_of(AmbiguityKind.UNMARKED_DOCUMENTATION)
    assert docs
    assert docs[0].message == "unmarked documentation is not authority"
    marked = _extract(
        (
            _unit(
                ContractEvidenceSource.AUTHORITATIVE_DOCUMENT,
                ContractDimension.POLICY,
                "fail-closed",
                _DOC_PATH,
                4,
                marked_authoritative=True,
            ),
        )
    )
    policy = marked.candidate(_SUBJECT).dimension(ContractDimension.POLICY)
    assert policy.status is DimensionStatus.PRESENT
    assert policy.values == ("fail-closed",)
    assert marked.ambiguities_of(AmbiguityKind.UNMARKED_DOCUMENTATION) == ()


def test_test_completeness_is_not_assumed_and_repetition_is_not_promoted() -> None:
    assumed = _extract(
        (
            _unit(
                ContractEvidenceSource.TEST,
                ContractDimension.FRESHNESS,
                "current-tree",
                _TEST_PATH,
                3,
                assumes_test_completeness=True,
            ),
        )
    )
    completeness = assumed.candidate(_SUBJECT).ambiguities_of(
        AmbiguityKind.TEST_COMPLETENESS_ASSUMED
    )
    assert completeness
    repeated = _extract(
        (
            _unit(
                ContractEvidenceSource.TEST,
                ContractDimension.CONFIRMATION,
                "signed-receipt",
                _TEST_PATH,
                4,
            ),
            _unit(
                ContractEvidenceSource.IMPLEMENTATION,
                ContractDimension.CONFIRMATION,
                "signed-receipt",
                _IMPL_PATH,
                9,
            ),
            _unit(
                ContractEvidenceSource.RUNTIME_CHECK,
                ContractDimension.CONFIRMATION,
                "signed-receipt",
                _RUNTIME_PATH,
                11,
            ),
        )
    )
    candidate = repeated.candidate(_SUBJECT)
    confirmation = candidate.dimension(ContractDimension.CONFIRMATION)
    assert confirmation.status is DimensionStatus.PRESENT
    assert confirmation.values == ("signed-receipt",)
    assert candidate.tier is ContractTier.CANDIDATE
    assert candidate.can_promote_requirement is False
    assert any(
        item.disposition is ComparisonDisposition.REPETITION for item in candidate.comparisons
    )
    assert candidate.ambiguities_of(AmbiguityKind.CONFLICTING_VALUES) == ()


def test_graph_correlation_adds_implementation_observations() -> None:
    architecture = _graph()
    mined = mine_graph_evidence(architecture)
    assert mined
    assert all(
        item.source_kind
        in {
            ContractEvidenceSource.IMPLEMENTATION,
            ContractEvidenceSource.ACCEPTED_RECEIPT,
        }
        for item in mined
    )
    result = _extract((), architecture=architecture)
    assert result.architecture_ir_identity == architecture.content_identity
    subjects = {item.subject for item in result.candidates}
    assert "n:operation:control.plane.execute" in subjects
    operation = result.candidate("n:operation:control.plane.execute")
    assert operation.dimension(ContractDimension.EFFECTS).status is DimensionStatus.PRESENT
    assert operation.dimension(ContractDimension.AUTHORITY).status is DimensionStatus.PRESENT
    assert operation.dimension(ContractDimension.POLICY).status is DimensionStatus.PRESENT
    assert operation.dimension(ContractDimension.OBSERVATIONS).status is DimensionStatus.PRESENT
    assert operation.dimension(ContractDimension.CONFIRMATION).status is DimensionStatus.PRESENT
    assert operation.dimension(ContractDimension.FRAMES).status is DimensionStatus.ABSENT
    assert operation.dimension(ContractDimension.EFFECTS).observation_only is True


def test_public_contracts_are_compared_with_every_declared_source() -> None:
    public = _unit(
        ContractEvidenceSource.PUBLIC_CONTRACT,
        ContractDimension.INPUTS,
        "task_id:string",
        _PUBLIC_PATH,
        1,
    )
    others = (
        _unit(ContractEvidenceSource.TYPE, ContractDimension.INPUTS, "task_id:string", _TYPE_PATH, 2),
        _unit(ContractEvidenceSource.SCHEMA, ContractDimension.INPUTS, "task_id:string", _SCHEMA_PATH, 3),
        _unit(ContractEvidenceSource.TEST, ContractDimension.INPUTS, "task_id:string", _TEST_PATH, 4),
        _unit(ContractEvidenceSource.PROOF, ContractDimension.INPUTS, "task_id:string", _PROOF_PATH, 5),
        _unit(
            ContractEvidenceSource.ACCEPTED_RECEIPT,
            ContractDimension.INPUTS,
            "task_id:string",
            _RECEIPT_PATH,
            6,
        ),
        _unit(
            ContractEvidenceSource.NEGATIVE_TEST,
            ContractDimension.INPUTS,
            "task_id:int",
            _NEG_PATH,
            7,
            polarity=EvidencePolarity.NEGATIVE,
        ),
        _unit(
            ContractEvidenceSource.AUTHORITATIVE_DOCUMENT,
            ContractDimension.INPUTS,
            "task_id:string",
            _DOC_PATH,
            8,
            marked_authoritative=True,
        ),
    )
    result = _extract(others, public_contracts=(public,))
    candidate = result.candidate(_SUBJECT)
    sources = {unit.source_kind for unit in candidate.evidence}
    assert ContractEvidenceSource.PUBLIC_CONTRACT in sources
    compared = {
        frozenset({item.left_source, item.right_source}) for item in candidate.comparisons
    }
    assert frozenset({ContractEvidenceSource.PUBLIC_CONTRACT, ContractEvidenceSource.TEST}) in compared
    assert frozenset({ContractEvidenceSource.PUBLIC_CONTRACT, ContractEvidenceSource.PROOF}) in compared
    assert (
        frozenset({ContractEvidenceSource.PUBLIC_CONTRACT, ContractEvidenceSource.ACCEPTED_RECEIPT})
        in compared
    )
    assert (
        frozenset({ContractEvidenceSource.PUBLIC_CONTRACT, ContractEvidenceSource.NEGATIVE_TEST})
        in compared
    )
    assert (
        frozenset(
            {ContractEvidenceSource.PUBLIC_CONTRACT, ContractEvidenceSource.AUTHORITATIVE_DOCUMENT}
        )
        in compared
    )
    assert candidate.ambiguities_of(AmbiguityKind.NEGATIVE_CONTRADICTS_POSITIVE) == ()


def test_extractor_cannot_promote_or_resolve() -> None:
    extractor = ContractCandidateExtractor()
    result = extractor.extract(_all_dimension_units(), repository_tree=_TREE, freshness=_FRESHNESS)
    assert result == build_contract_extraction_result(
        _all_dimension_units(), repository_tree=_TREE, freshness=_FRESHNESS
    )
    candidate = result.candidate(_SUBJECT)
    with pytest.raises(ContractExtractorAuthorityError, match="cannot promote"):
        extractor.promote_to_requirement("public contract")
    with pytest.raises(ContractExtractorAuthorityError, match="cannot apply"):
        extractor.resolve_by_implementation()
    with pytest.raises(ContractExtractorError, match="hidden conflict is prohibited"):
        extractor.hide_conflicts()
    with pytest.raises(ContractExtractorError, match="assumption is prohibited"):
        extractor.treat_tests_as_complete()
    with pytest.raises(ContractExtractorError, match="is not authority"):
        extractor.treat_unmarked_docs_as_authority()
    with pytest.raises(ContractExtractorAuthorityError, match="cannot promote"):
        result.promote_to_requirement()
    with pytest.raises(ContractExtractorAuthorityError, match="cannot apply"):
        result.resolve_by_implementation()
    with pytest.raises(ContractExtractorError, match="hidden conflict is prohibited"):
        result.hide_conflicts()
    with pytest.raises(ContractExtractorError, match="assumption is prohibited"):
        result.treat_tests_as_complete()
    with pytest.raises(ContractExtractorError, match="is not authority"):
        result.treat_unmarked_docs_as_authority()
    with pytest.raises(ContractExtractorAuthorityError, match="cannot promote"):
        candidate.promote_to_requirement()
    with pytest.raises(ContractExtractorAuthorityError, match="cannot apply"):
        candidate.resolve_by_implementation()
    with pytest.raises(ContractExtractorError, match="hidden conflict is prohibited"):
        candidate.hide_conflicts()
    with pytest.raises(ContractExtractorAuthorityError, match="cannot promote"):
        refuse_requirement_promotion("requirement")
    with pytest.raises(ContractExtractorAuthorityError, match="cannot apply"):
        refuse_implementation_resolution("implementation-wins resolution")
    with pytest.raises(ContractExtractorError, match="hidden conflict is prohibited"):
        refuse_hidden_conflict("hidden conflict")
    with pytest.raises(ContractExtractorError, match="assumption is prohibited"):
        refuse_test_completeness("test completeness")
    with pytest.raises(ContractExtractorError, match="is not authority"):
        refuse_unmarked_documentation_authority("unmarked documentation")
    with pytest.raises(ContractExtractorAuthorityError, match="cannot promote"):
        ContractCandidateExtractor(can_promote_requirement=True)
    with pytest.raises(ContractExtractorAuthorityError, match="cannot apply"):
        ContractCandidateExtractor(can_resolve_implementation_wins=True)
    with pytest.raises(ContractExtractorError, match="hidden contract conflict"):
        ContractCandidateExtractor(can_hide_conflicts=True)
    with pytest.raises(ContractExtractorError, match="test completeness"):
        ContractCandidateExtractor(can_treat_tests_as_complete=True)
    with pytest.raises(ContractExtractorError, match="unmarked documentation"):
        ContractCandidateExtractor(can_treat_unmarked_docs_as_authority=True)


def test_round_trip_and_canonical_identity() -> None:
    result = _extract(
        (
            *_all_dimension_units(),
            _unit(
                ContractEvidenceSource.TEST,
                ContractDimension.OUTPUTS,
                "schema:object{receipt:text}",
                _TEST_PATH,
                80,
            ),
        )
    )
    payload = result.to_dict()
    restored = ContractExtractionResult.from_mapping(payload)
    assert restored == result
    assert restored.to_dict() == payload
    assert restored.to_json() == json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    assert ContractExtractionResult.from_json(restored.to_json()) == result
    claimed = payload.pop("content_identity")
    validate_cid(claimed, codecs=("dag-json",))
    assert claimed == cid_for_dag_json(payload)
    assert claimed == result.content_identity
    assert not claimed.startswith("sha256:")
    candidate = result.candidate(_SUBJECT)
    candidate_payload = candidate.to_dict()
    assert ContractCandidate.from_mapping(candidate_payload) == candidate
    assert ContractCandidate.from_json(candidate.to_json()) == candidate


def test_deterministic_identity_is_order_independent() -> None:
    first = _all_dimension_units()
    second = tuple(reversed(first))
    left = _extract(first)
    right = _extract(second)
    assert left.content_identity == right.content_identity
    assert left.to_dict() == right.to_dict()
    assert left.candidates[0].content_identity == right.candidates[0].content_identity


def test_unknown_fields_and_identity_mismatch_are_rejected() -> None:
    payload = _extract(_all_dimension_units()).to_dict()
    unknown = dict(payload)
    unknown["hidden"] = True
    identity_payload = {
        key: value for key, value in unknown.items() if key != "content_identity"
    }
    unknown["content_identity"] = cid_for_dag_json(identity_payload)
    with pytest.raises(ContractExtractorError, match="unknown contract-candidate field"):
        ContractExtractionResult.from_mapping(unknown)
    missing = {key: value for key, value in payload.items() if key != "freshness"}
    with pytest.raises(ContractExtractorError, match="missing contract-candidate field"):
        ContractExtractionResult.from_mapping(missing)
    forged = dict(payload)
    forged["content_identity"] = "sha256:" + ("00" * 32)
    with pytest.raises(ContractExtractorError, match="content identity mismatch"):
        ContractExtractionResult.from_mapping(forged)
    schema = dict(payload)
    schema["schema"] = schema["schema"] + "-extra"
    identity_payload = {
        key: value for key, value in schema.items() if key != "content_identity"
    }
    schema["content_identity"] = cid_for_dag_json(identity_payload)
    with pytest.raises(ContractExtractorError, match="unexpected contract-extraction schema"):
        ContractExtractionResult.from_mapping(schema)
    promote = dict(payload)
    promote["can_promote_requirement"] = True
    identity_payload = {
        key: value for key, value in promote.items() if key != "content_identity"
    }
    promote["content_identity"] = cid_for_dag_json(identity_payload)
    with pytest.raises(ContractExtractorAuthorityError, match="cannot promote"):
        ContractExtractionResult.from_mapping(promote)
    with pytest.raises(ContractExtractorError, match="public_contract requires"):
        _unit(
            ContractEvidenceSource.TEST,
            ContractDimension.INPUTS,
            "x",
            _TEST_PATH,
            1,
            public_contract=True,
        )
    with pytest.raises(ContractExtractorError, match="content identity is not a contract subject"):
        _unit(
            ContractEvidenceSource.TYPE,
            ContractDimension.INPUTS,
            "x",
            _TYPE_PATH,
            1,
            subject="baguqeera" + ("a" * 50),
        )


def test_hidden_conflict_construction_fails_closed() -> None:
    units = (
        _unit(
            ContractEvidenceSource.TYPE,
            ContractDimension.EFFECTS,
            "writes:store",
            _TYPE_PATH,
            3,
        ),
        _unit(
            ContractEvidenceSource.TEST,
            ContractDimension.EFFECTS,
            "pure",
            _TEST_PATH,
            4,
        ),
    )
    result = _extract(units)
    candidate = result.candidate(_SUBJECT)
    effects = candidate.dimension(ContractDimension.EFFECTS)
    assert effects.status is DimensionStatus.CONFLICTED
    dimensions = []
    for record in candidate.dimensions:
        if record.dimension is ContractDimension.EFFECTS:
            dimensions.append(
                ContractDimensionRecord(
                    dimension=record.dimension,
                    status=DimensionStatus.PRESENT,
                    values=("writes:store",),
                    source_kinds=(ContractEvidenceSource.TYPE,),
                    observation_only=False,
                )
            )
        else:
            dimensions.append(record)
    with pytest.raises(ContractExtractorError, match="must match retained evidence"):
        ContractCandidate(
            subject=candidate.subject,
            repository_tree=_TREE,
            freshness=_FRESHNESS,
            dimensions=tuple(dimensions),
            evidence=candidate.evidence,
            ambiguities=(),
            comparisons=candidate.comparisons,
        )
    with pytest.raises(ContractExtractorError, match="hidden contract conflict"):
        ContractAmbiguity(
            subject=_SUBJECT,
            dimension=ContractDimension.EFFECTS,
            kind=AmbiguityKind.CONFLICTING_VALUES,
            values=("pure", "writes:store"),
            source_kinds=(ContractEvidenceSource.TEST, ContractEvidenceSource.TYPE),
            message="conflict",
            provenance=_fact(_TYPE_PATH, 3),
            retained=False,
        )
    present = ContractDimensionRecord(
        dimension=ContractDimension.INPUTS,
        status=DimensionStatus.ABSENT,
        values=(),
        source_kinds=(),
    )
    with pytest.raises(ContractExtractorError, match="every contract dimension"):
        ContractCandidate(
            subject=_SUBJECT,
            repository_tree=_TREE,
            freshness=_FRESHNESS,
            dimensions=(present,),
            evidence=(),
            ambiguities=(),
            comparisons=(),
        )
    empty = _extract(())
    assert empty.candidates == ()
    assert empty.ambiguities == ()
    assert empty.architecture_ir_identity == ""
