"""SCA-090 deterministic contract mismatch analyzer tests."""

from __future__ import annotations

from copy import deepcopy

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.contract_mismatch_analyzer import (
    ContractFinding,
    ContractMismatchAnalyzer,
    ContractMismatchError,
    FindingLifecycle,
    MismatchAnalysis,
    MismatchState,
    SourceOwner,
    bounded_impact_closure,
    merge_finding_evidence,
    resolve_source_ownership,
    route_source_owner,
)
from ipfs_accelerate_py.agent_supervisor.analysis.mcp_contract_analysis import (
    ContractCounterexample,
    ContractParityClaim,
    ParityState,
)
from ipfs_accelerate_py.agent_supervisor.analysis.mcp_contract_catalog import (
    McpClaimFamily,
)
from ipfs_accelerate_py.agent_supervisor.proof.code_claim_contracts import (
    ClaimStatus,
)
from ipfs_accelerate_py.agent_supervisor.proof.code_proof_query import (
    ClaimQueryHit,
    CodeProofQueryResult,
)


def _claim(
    *,
    state: ParityState = ParityState.REFUTED,
    actual: str = "integer",
) -> ContractParityClaim:
    counterexamples = ()
    if state is ParityState.REFUTED:
        counterexamples = (
            ContractCounterexample(
                reason_code="argument_type_changed",
                boundary_id="tools/call",
                path="input.limit",
                expected="string",
                actual=actual,
                source_ids=("source:schema",),
            ),
        )
    return ContractParityClaim(
        family=McpClaimFamily.ARGUMENTS_PRESERVED,
        state=state,
        operation_id="repo.inspect",
        premise_ids=("premise:descriptor", "premise:handler"),
        reason_codes=(
            "argument_type_changed"
            if state is ParityState.REFUTED
            else f"claim_{state.value}"
        ,),
        counterexamples=counterexamples,
    )


def _analyze(
    claim: ContractParityClaim | dict | None = None,
    **overrides,
) -> ContractFinding:
    arguments = {
        "snapshot_id": "snapshot:one",
        "contract_id": "contract:repo.inspect",
        "affected_symbols": ("handler:repo.inspect",),
        "affected_paths": (
            "external/ipfs_accelerate/ipfs_accelerate_py/mcp/inspect.py",
        ),
        "obligation_ids": ("obligation:arguments",),
        "cas_handles": ("bafy:contract-slice",),
        "reproduction_commands": ("python -m pytest test_contract.py -q",),
    }
    arguments.update(overrides)
    findings = ContractMismatchAnalyzer().analyze_claim(
        claim or _claim(), **arguments
    )
    assert len(findings) == 1
    return findings[0]


def test_dedupe_identity_binds_exact_required_dimensions() -> None:
    baseline = _analyze(
        affected_symbols=("symbol:b", "symbol:a", "symbol:a")
    )
    reordered = _analyze(affected_symbols=("symbol:a", "symbol:b"))
    assert baseline.finding_id == reordered.finding_id
    assert baseline.affected_symbols == ("symbol:a", "symbol:b")

    changes = (
        _analyze(snapshot_id="snapshot:two"),
        _analyze(contract_id="contract:other"),
        _analyze(
            {
                "claim_id": "claim:other-family",
                "family": McpClaimFamily.FAILURE_PARITY.value,
                "state": "refuted",
                "reason_codes": ["failure_collapsed"],
                "counterexample": {"kind": "failure", "actual": "500"},
            }
        ),
        _analyze(affected_symbols=("symbol:c",)),
        _analyze(_claim(actual="number")),
    )
    assert all(item.finding_id != baseline.finding_id for item in changes)

    payload = baseline.to_dict()
    assert payload["dedupe_id"] == baseline.finding_id
    assert payload["reproduction"]["snapshot_id"] == "snapshot:one"
    assert payload["reproduction"]["claim_id"] == _claim().claim_id
    assert payload["reproduction"]["obligation_ids"] == [
        "obligation:arguments"
    ]
    assert payload["reproduction"]["cas_handles"] == ["bafy:contract-slice"]
    assert payload["reproduction"]["commands"] == [
        "python -m pytest test_contract.py -q"
    ]


def test_changed_evidence_upserts_one_finding_and_round_trips() -> None:
    first = _analyze(evidence_ids=("evidence:old",))
    changed = _analyze(
        evidence_ids=("evidence:new",),
        previous=(first,),
    )

    assert changed.finding_id == first.finding_id
    assert len(changed.evidence) == 2
    assert changed.reproduction.evidence_ids == (
        "evidence:new",
        "evidence:old",
    )

    merged_again = merge_finding_evidence(first, changed)
    assert merged_again.to_dict() == changed.to_dict()
    restored = ContractFinding.from_dict(changed.to_dict())
    assert restored.to_dict() == changed.to_dict()
    assert ContractFinding.from_json(changed.to_json()).to_dict() == changed.to_dict()
    report = MismatchAnalysis(
        snapshot_id="snapshot:one", findings=(restored,)
    )
    assert MismatchAnalysis.from_dict(report.to_dict()).to_dict() == report.to_dict()
    assert MismatchAnalysis.from_json(report.to_json()).to_dict() == report.to_dict()


def test_source_ownership_uses_reviewed_prefixes_without_guessing() -> None:
    paths = (
        "external/ipfs_accelerate/ipfs_accelerate_py/api.py",
        "external/ipfs_kit/ipfs_kit_py/api.py",
        "external/ipfs_datasets/ipfs_datasets_py/api.py",
        "swissknife/src/mcp/client.ts",
        "Mcp-Plus-Plus/src/schema.ts",
        "docs/ipfs_kit_py-looking-name.md",
    )
    assert tuple(route_source_owner(path) for path in paths) == (
        SourceOwner.ACCELERATOR,
        SourceOwner.KIT,
        SourceOwner.DATASETS,
        SourceOwner.SWISSKNIFE,
        SourceOwner.MCP_PLUS_PLUS,
        SourceOwner.UNRESOLVED,
    )
    ownership = resolve_source_ownership(paths)
    assert tuple(item.path for item in ownership) == tuple(sorted(paths))
    unresolved = next(
        item for item in ownership if item.owner is SourceOwner.UNRESOLVED
    )
    assert unresolved.matched_prefix == ""
    assert unresolved.reason_code == "owner_prefix_unrecognized"

    mixed = _analyze(affected_paths=paths[:2])
    assert mixed.owners == (SourceOwner.ACCELERATOR, SourceOwner.KIT)
    assert mixed.source_owner is SourceOwner.UNRESOLVED


def test_cache_miss_unknown_and_satisfied_are_not_findings() -> None:
    analyzer = ContractMismatchAnalyzer(
        default_reproduction_commands=("python -m pytest -q",)
    )
    claims = tuple(
        {
            "claim_id": f"claim:{state}",
            "family": "ArgumentsPreserved",
            "state": state,
            "affected_symbols": ["symbol:handler"],
        }
        for state in ("cache_miss", "miss", "unknown", "open", "satisfied")
    )
    result = analyzer.analyze(
        claims,
        snapshot_id="snapshot:one",
        contract_id="contract:one",
    )
    assert result.findings == ()
    assert result.ignored_claim_ids == tuple(
        sorted(f"claim:{state}" for state in (
            "cache_miss", "miss", "unknown", "open", "satisfied"
        ))
    )
    assert result.reason_codes == (
        "cache_miss_and_unknown_are_not_refutations",
    )


def test_typed_code_proof_query_result_is_consumed_without_status_collapse() -> None:
    result = CodeProofQueryResult(
        query="properties_refuted",
        repository_tree_id="snapshot:query",
        hits=(
            ClaimQueryHit(
                property_id="contract:query",
                status=ClaimStatus.REFUTED,
                claim_id="claim:query",
                obligation_ids=("obligation:query",),
                evidence_ids=("evidence:query",),
                reason_codes=("query_refuted",),
                provenance={
                    "symbols": ["symbol:query"],
                    "paths": ["external/ipfs_kit/ipfs_kit_py/query.py"],
                },
                counterexample={"kind": "refuted_claim", "actual": False},
            ),
            ClaimQueryHit(
                property_id="contract:query",
                status=ClaimStatus.OPEN,
                claim_id="claim:cache-miss",
                reason_codes=("cache_miss",),
                provenance={"symbols": ["symbol:query"]},
            ),
        ),
    )
    analysis = ContractMismatchAnalyzer().analyze_code_proof_query(
        result,
        claim_family=McpClaimFamily.ARGUMENTS_PRESERVED,
    )
    assert len(analysis.findings) == 1
    assert analysis.findings[0].snapshot_id == "snapshot:query"
    assert analysis.findings[0].contract_id == "contract:query"
    assert analysis.findings[0].source_owner is SourceOwner.KIT
    assert analysis.ignored_claim_ids == ("claim:cache-miss",)


@pytest.mark.parametrize(
    ("state", "expected"),
    (
        ("refuted", MismatchState.REFUTED),
        ("stale", MismatchState.STALE),
        ("contradicted", MismatchState.CONTRADICTORY),
        ("ambiguous", MismatchState.AMBIGUOUS),
        ("unsupported", MismatchState.UNSUPPORTED),
        ("not_measured", MismatchState.NOT_MEASURED),
    ),
)
def test_all_required_mismatch_states_are_preserved(
    state: str, expected: MismatchState
) -> None:
    claim = {
        "claim_id": f"claim:{state}",
        "family": "FailureParity",
        "state": state,
        "reason_codes": [f"reason:{state}"],
    }
    if state == "refuted":
        claim["counterexample"] = {"kind": "failure", "actual": "collapsed"}
    finding = _analyze(claim)
    assert finding.state is expected
    assert finding.lifecycle is (
        FindingLifecycle.STALE
        if expected is MismatchState.STALE
        else FindingLifecycle.ACTIVE
    )
    assert finding.counterexample_id


def test_refutation_requires_counterexample_and_exact_reproduction() -> None:
    claim = {
        "claim_id": "claim:refuted",
        "family": "ArgumentsPreserved",
        "state": "refuted",
        "reason_codes": ["failed"],
    }
    with pytest.raises(
        ContractMismatchError, match="requires a compact counterexample"
    ):
        _analyze(claim)
    with pytest.raises(
        ContractMismatchError, match="exact obligation or reproduction"
    ):
        _analyze(obligation_ids=(), reproduction_commands=())


def test_stale_and_resolved_finding_reopens_without_duplicate() -> None:
    stale = _analyze(
        {
            "claim_id": _claim().claim_id,
            "family": McpClaimFamily.ARGUMENTS_PRESERVED.value,
            "state": "stale",
            "reason_codes": ["evidence_stale"],
            # Use the same semantic witness as the later current finding so the
            # test exercises lifecycle, not a new counterexample identity.
            "counterexample": _claim().counterexamples[0].to_dict(),
        }
    )
    active = _analyze(previous=(stale,))
    assert active.finding_id == stale.finding_id
    assert active.lifecycle is FindingLifecycle.REOPENED

    report = ContractMismatchAnalyzer().analyze(
        (),
        snapshot_id="snapshot:one",
        contract_id="contract:repo.inspect",
        previous=(active,),
    )
    assert len(report.findings) == 1
    assert report.findings[0].lifecycle is FindingLifecycle.RESOLVED


def test_bounded_impact_closure_is_sorted_deterministic_and_bounded() -> None:
    edges = {
        "symbol:a": ("symbol:c", "symbol:b"),
        "symbol:b": ("symbol:d",),
        "symbol:c": ("symbol:e",),
    }
    first = bounded_impact_closure(
        ("symbol:a",), edges, max_depth=1, max_symbols=10
    )
    second = bounded_impact_closure(
        ("symbol:a",), dict(reversed(tuple(edges.items()))),
        max_depth=1,
        max_symbols=10,
    )
    assert first == second
    assert first.symbols == ("symbol:a", "symbol:b", "symbol:c")
    assert first.truncated is True

    capped = bounded_impact_closure(
        ("symbol:a",), edges, max_depth=6, max_symbols=2
    )
    assert len(capped.symbols) == 2
    assert capped.truncated is True


def test_serialized_identity_and_ownership_claims_fail_closed() -> None:
    finding = _analyze()
    bad_id = deepcopy(finding.to_dict())
    bad_id["finding_id"] = "forged"
    with pytest.raises(ContractMismatchError, match="dedupe identity"):
        ContractFinding.from_dict(bad_id)

    bad_owner = deepcopy(finding.to_dict())
    bad_owner["ownership"][0]["owner"] = SourceOwner.KIT.value
    with pytest.raises(ContractMismatchError, match="ownership claim disagrees"):
        ContractFinding.from_dict(bad_owner)

    with pytest.raises(ContractMismatchError, match="repository-relative"):
        route_source_owner("../external/ipfs_kit/tool.py")
