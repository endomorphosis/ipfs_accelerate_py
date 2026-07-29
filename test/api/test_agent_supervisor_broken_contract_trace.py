"""Tests for the conservative broken-call trace adapter."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.broken_contract_trace import (
    BrokenCallSite,
    BrokenContractTraceBuilder,
    CallArgumentFact,
    CallPolicyContext,
    GraphEvidence,
    ResolverEvidence,
)
from ipfs_accelerate_py.agent_supervisor.analysis.contract_repair_contracts import (
    AuthorityRoots,
    EvidenceReference,
    SourceSpan,
    TraceDisposition,
)


@pytest.fixture
def roots() -> AuthorityRoots:
    return AuthorityRoots(
        repository_id="repository:one", forest_id="forest:one", tree_id="tree:one",
        graph_id="graph:one", index_id="index:one", model_id="model:one",
        config_id="config:one", translator_id="translator:one",
        toolchain_id="toolchain:one", policy_id="policy:one",
    )


@pytest.fixture
def evidence() -> EvidenceReference:
    return EvidenceReference("resolver_receipt", "evidence:resolver", "call:one", "test")


@pytest.fixture
def call(roots: AuthorityRoots, evidence: EvidenceReference) -> BrokenCallSite:
    return BrokenCallSite(
        SourceSpan("pkg/caller.py", 10, 30, "blob:caller"), "symbol:caller",
        "legacy.send", "attribute_call", "python", "cpython-3.11",
        actual_arguments=(
            CallArgumentFact(0, type_ref="str", value_range="nonempty", evidence_id="fact:arg0"),
            CallArgumentFact(1, "timeout", "int", "[1,30]", "fact:timeout"),
        ),
        awaited=True, result_uses=("assigned:receipt", "returned"),
        handled_error_refs=("error:TimeoutError",),
        policy_context=CallPolicyContext(
            permitted_effects=("network",), authorized_capabilities=("cap:send",),
            authorization_context_refs=("auth:request",), resource_budget_refs=("budget:30s",),
            cancellation_behavior="propagate",
        ),
        evidence_refs=(evidence,),
    )


def graph(evidence: EvidenceReference, *, complete: bool = True, graph_id: str = "graph:one") -> GraphEvidence:
    return GraphEvidence(graph_id, complete, frontier_refs=("frontier:imports",), exclusion_refs=("excluded:vendor",), evidence_refs=(evidence,))


@dataclass
class Resolver:
    result: ResolverEvidence

    def resolve_call(self, call_site: BrokenCallSite, graph: GraphEvidence) -> ResolverEvidence:
        return self.result


def test_resolved_mismatch_preserves_sender_facts(roots: AuthorityRoots, evidence: EvidenceReference, call: BrokenCallSite) -> None:
    target = SourceSpan("pkg/receiver.py", 4, 22, "blob:receiver")
    result = BrokenContractTraceBuilder().build(
        roots, call, graph=graph(evidence),
        resolver=Resolver(ResolverEvidence("resolved_mismatch", target, "symbol:receiver", route_closed=True, evidence_refs=(evidence,))),
    )

    assert result.trace.disposition is TraceDisposition.RESOLVED_MISMATCH
    assert result.trace.target_span == target
    assert [item.name for item in result.call_site.actual_arguments] == ["", "timeout"]
    assert result.call_site.awaited is True
    assert result.call_site.result_uses == ("assigned:receipt", "returned")
    assert result.call_site.handled_error_refs == ("error:TimeoutError",)
    assert result.call_site.policy_context.authorized_capabilities == ("cap:send",)
    assert result.trace.graph_frontier_refs == ("frontier:imports",)
    assert result.trace.excluded_refs == ("excluded:vendor",)


@pytest.mark.parametrize(
    ("claim", "kwargs", "expected"),
    [
        ("missing_local", {"local_scope_complete": True}, TraceDisposition.MISSING_LOCAL),
        ("likely_refactor", {"route_closed": True, "identity_kinds": ("history_lineage",), "target_span": SourceSpan("pkg/moved.py", 1, 9, "blob:moved")}, TraceDisposition.LIKELY_REFACTOR),
        ("adapter_required", {"route_closed": True, "adapter_kinds": ("adapter_mapping",), "target_span": SourceSpan("pkg/new_api.py", 1, 9, "blob:new")}, TraceDisposition.ADAPTER_REQUIRED),
        ("external", {}, TraceDisposition.EXTERNAL),
        ("dynamic", {}, TraceDisposition.DYNAMIC),
        ("ambiguous", {}, TraceDisposition.AMBIGUOUS),
    ],
)
def test_each_nonresolved_disposition_requires_its_bounded_evidence(
    roots: AuthorityRoots, evidence: EvidenceReference, call: BrokenCallSite,
    claim: str, kwargs: dict[str, object], expected: TraceDisposition,
) -> None:
    result = BrokenContractTraceBuilder().build(
        roots, call, graph=graph(evidence), resolver=Resolver(ResolverEvidence(claim, evidence_refs=(evidence,), **kwargs)),
    )
    assert result.trace.disposition is expected
    if expected in {TraceDisposition.LIKELY_REFACTOR, TraceDisposition.ADAPTER_REQUIRED}:
        assert result.trace.target_span is not None
    else:
        assert result.trace.target_span is None


def test_same_name_or_vector_evidence_cannot_resolve_a_call(roots: AuthorityRoots, evidence: EvidenceReference, call: BrokenCallSite) -> None:
    result = BrokenContractTraceBuilder().build(
        roots, call, graph=graph(evidence),
        resolver=Resolver(ResolverEvidence("likely_refactor", route_closed=True, same_name=True, vector_evidence=True, evidence_refs=(evidence,))),
    )
    assert result.trace.disposition is TraceDisposition.UNSUPPORTED
    assert result.trace.target_span is None


def test_incomplete_or_stale_graph_fails_closed(roots: AuthorityRoots, evidence: EvidenceReference, call: BrokenCallSite) -> None:
    resolver = Resolver(ResolverEvidence("missing_local", local_scope_complete=True, evidence_refs=(evidence,)))
    incomplete = BrokenContractTraceBuilder().build(roots, call, graph=graph(evidence, complete=False), resolver=resolver)
    stale = BrokenContractTraceBuilder().build(roots, call, graph=graph(evidence, graph_id="graph:stale"), resolver=resolver)
    assert incomplete.trace.disposition is TraceDisposition.UNSUPPORTED
    assert stale.trace.disposition is TraceDisposition.UNSUPPORTED
    assert "graph_root_mismatch" in stale.unknown_frontier_refs


def test_missing_or_incompatible_resolver_returns_unsupported_without_raising(
    roots: AuthorityRoots, evidence: EvidenceReference, call: BrokenCallSite,
) -> None:
    builder = BrokenContractTraceBuilder()
    missing = builder.build(roots, call, graph=graph(evidence), resolver=None)
    incompatible = builder.build(roots, call, graph=graph(evidence), resolver=object())  # type: ignore[arg-type]
    assert missing.trace.disposition is TraceDisposition.UNSUPPORTED
    assert incompatible.trace.disposition is TraceDisposition.UNSUPPORTED
    assert missing.unknown_frontier_refs == ("resolver_or_graph_unsupported",)


def test_invalid_positive_claims_are_downgraded_not_promoted(roots: AuthorityRoots, evidence: EvidenceReference, call: BrokenCallSite) -> None:
    result = BrokenContractTraceBuilder().build(
        roots, call, graph=graph(evidence),
        resolver=Resolver(ResolverEvidence("resolved_mismatch", evidence_refs=(evidence,))),
    )
    assert result.trace.disposition is TraceDisposition.UNSUPPORTED
