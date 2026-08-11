"""Adversarial tests for memory-safety and native-boundary evidence collection."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.contract_repair_contracts import (
    AuthorityRoots,
    EvidenceReference,
    MemorySafetyDisposition,
    SourceSpan,
)
from ipfs_accelerate_py.agent_supervisor.analysis.memory_safety_facets import (
    MemorySafetyEvidenceCollector,
    MemorySafetyEvidenceError,
    MemorySafetyPolicy,
    MemorySafetyReceiptKind,
    MemorySafetyReceiptState,
    NativeBoundary,
    NativeBoundaryKind,
    ProofEvidence,
)


@pytest.fixture
def roots() -> AuthorityRoots:
    return AuthorityRoots(
        repository_id="repository:memory", forest_id="forest:memory", tree_id="tree:current",
        graph_id="graph:memory", index_id="index:memory", model_id="model:memory",
        config_id="config:memory", translator_id="translator:memory",
        toolchain_id="toolchain:rust-1.80", policy_id="policy:memory",
    )


@pytest.fixture
def span() -> SourceSpan:
    return SourceSpan("pkg/native.rs", 0, 42, "blob:native")


def reference(name: str) -> EvidenceReference:
    return EvidenceReference("memory_safety_receipt", f"evidence:{name}", producer_id="collector")


def receipt(
    span: SourceSpan,
    kind: MemorySafetyReceiptKind,
    *,
    language: str = "rust",
    tree_id: str = "tree:current",
    toolchain_id: str = "toolchain:rust-1.80",
    state: MemorySafetyReceiptState = MemorySafetyReceiptState.PASSED,
) -> ProofEvidence:
    return ProofEvidence(reference(kind.value), kind, language, toolchain_id, tree_id, (span.content_id,), state)


def test_native_policy_proof_is_bound_to_language_runtime_toolchain_tree_and_scope(
    roots: AuthorityRoots, span: SourceSpan
) -> None:
    policy = MemorySafetyPolicy(
        native_proof_groups=((MemorySafetyReceiptKind.BORROW_CHECKER,), (MemorySafetyReceiptKind.MIRI,)),
    )
    result = MemorySafetyEvidenceCollector(roots, policy).assess(
        subject_span=span, language_runtime="rust",
        receipts=(receipt(span, MemorySafetyReceiptKind.BORROW_CHECKER), receipt(span, MemorySafetyReceiptKind.MIRI)),
    )

    assert result.facet.disposition is MemorySafetyDisposition.PROVED
    assert result.memory_safe is True
    assert len(result.facet.proof_refs) == 2


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        ({"tree_id": "tree:old"}, "stale_tree_or_toolchain_receipt"),
        ({"toolchain_id": "toolchain:old"}, "stale_tree_or_toolchain_receipt"),
        ({"language": "c"}, "native_proof_receipts_missing"),
    ],
)
def test_mismatched_receipts_fail_closed(roots: AuthorityRoots, span: SourceSpan, mutate: dict[str, str], reason: str) -> None:
    item = receipt(span, MemorySafetyReceiptKind.BORROW_CHECKER, **{
        {"language": "language", "tree_id": "tree_id", "toolchain_id": "toolchain_id"}[key]: value
        for key, value in mutate.items()
    })
    result = MemorySafetyEvidenceCollector(roots).assess(
        subject_span=span, language_runtime="rust", receipts=(item,)
    )

    expected = MemorySafetyDisposition.STALE if reason.startswith("stale") else MemorySafetyDisposition.UNSUPPORTED
    assert result.facet.disposition is expected
    assert result.memory_safe is False
    assert reason in result.reason_codes


def test_max_memory_bytes_and_passing_unit_test_never_make_memory_safe(
    roots: AuthorityRoots, span: SourceSpan
) -> None:
    result = MemorySafetyEvidenceCollector(roots).assess(
        subject_span=span, language_runtime="rust", max_memory_bytes=1024,
        receipts=(receipt(span, MemorySafetyReceiptKind.UNIT_TEST),),
    )

    assert result.facet.disposition is MemorySafetyDisposition.EMPIRICAL
    assert result.memory_safe is False
    assert "max_memory_bytes" not in result.facet.to_json()
    assert not result.facet.proof_refs


def test_missing_required_native_evidence_is_unsupported(roots: AuthorityRoots, span: SourceSpan) -> None:
    result = MemorySafetyEvidenceCollector(roots).assess(subject_span=span, language_runtime="rust")

    assert result.facet.disposition is MemorySafetyDisposition.UNSUPPORTED
    assert result.memory_safe is False
    assert "native_proof_receipts_missing" in result.facet.unsupported_refs


@pytest.mark.parametrize("language,boundary", [
    ("python", NativeBoundaryKind.REFLECTION),
    ("python", NativeBoundaryKind.NATIVE_EXTENSION),
    ("typescript", NativeBoundaryKind.FFI),
    ("typescript", NativeBoundaryKind.MONKEY_PATCH),
])
def test_managed_reflection_and_native_boundaries_cannot_claim_general_memory_safety(
    roots: AuthorityRoots, span: SourceSpan, language: str, boundary: NativeBoundaryKind
) -> None:
    result = MemorySafetyEvidenceCollector(roots).assess(
        subject_span=span, language_runtime=language,
        boundaries=(NativeBoundary(f"boundary:{boundary.value}", boundary, span),),
    )

    assert result.facet.disposition is MemorySafetyDisposition.UNSUPPORTED
    assert result.memory_safe is False
    assert result.facet.unsupported_refs


def test_managed_language_without_unmodeled_boundary_is_model_supported_not_safe(
    roots: AuthorityRoots, span: SourceSpan
) -> None:
    result = MemorySafetyEvidenceCollector(roots).assess(subject_span=span, language_runtime="python")

    assert result.facet.disposition is MemorySafetyDisposition.SUPPORTED
    assert result.memory_safe is False


def test_explicit_stale_and_error_receipts_are_visible_and_fail_closed(roots: AuthorityRoots, span: SourceSpan) -> None:
    stale = MemorySafetyEvidenceCollector(roots).assess(
        subject_span=span, language_runtime="rust",
        receipts=(receipt(span, MemorySafetyReceiptKind.MIRI, state=MemorySafetyReceiptState.STALE),),
    )
    error = MemorySafetyEvidenceCollector(roots).assess(
        subject_span=span, language_runtime="rust",
        receipts=(receipt(span, MemorySafetyReceiptKind.MIRI, state=MemorySafetyReceiptState.ERROR),),
    )

    assert stale.facet.disposition is MemorySafetyDisposition.STALE
    assert error.facet.disposition is MemorySafetyDisposition.ERROR
    assert not stale.memory_safe and not error.memory_safe


def test_policy_cannot_treat_a_unit_test_as_native_proof() -> None:
    with pytest.raises(MemorySafetyEvidenceError, match="observational"):
        MemorySafetyPolicy(native_proof_groups=((MemorySafetyReceiptKind.UNIT_TEST,),))


def test_scope_ids_are_required_on_receipts(roots: AuthorityRoots, span: SourceSpan) -> None:
    with pytest.raises(MemorySafetyEvidenceError, match="scope_ids"):
        ProofEvidence(
            reference("missing-scope"), MemorySafetyReceiptKind.MIRI, "rust",
            roots.toolchain_id, roots.tree_id, (),
        )
