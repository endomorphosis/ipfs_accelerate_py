"""RPR-017: integrate the proof-gated @2 decision path into the repair flow.

Covers feature-gated pipeline routing, @1 regression, decision-derived write
paths, refinery admission after decision validation, and the adversarial
integration cases (rename-to-moved-file, new-site, stale, ambiguous, read-only,
incompatible-decoy).
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.analysis_pipeline import (
    AnalysisPipeline,
    AnalysisPipelinePolicy,
    ProofGatedContractRepairRequest,
)
from ipfs_accelerate_py.agent_supervisor.analysis.contract_mismatch_analyzer import (
    ContractFinding,
    ContractMismatchAnalyzer,
)
from ipfs_accelerate_py.agent_supervisor.analysis.contract_repair_contracts import (
    AuthorityRoots,
    BrokenContractTrace,
    DecisionDisposition,
    EvidenceReference,
    RepairCandidate,
    RepairStrategy,
    SourceSpan,
    TraceDisposition,
    candidate_set_identity,
)
from ipfs_accelerate_py.agent_supervisor.analysis.contract_repair_reranker import (
    CandidateEligibilityDisposition,
    CandidateRank,
    RerankDisposition,
    RerankReceipt,
)
from ipfs_accelerate_py.agent_supervisor.analysis.mcp_contract_analysis import (
    ContractCounterexample,
    ContractParityClaim,
    ParityState,
)
from ipfs_accelerate_py.agent_supervisor.analysis.mcp_contract_catalog import (
    McpClaimFamily,
)
from ipfs_accelerate_py.agent_supervisor.analysis.sender_receiver_contracts import (
    SenderReceiverContractCompiler,
)
from ipfs_accelerate_py.agent_supervisor.objectives.contract_mismatch_refinery import (
    ContractMismatchRefinery,
    ContractMismatchRefineryPolicy,
    ContractMismatchRefineryReason,
)
from ipfs_accelerate_py.agent_supervisor.planning.repair_target_admission import (
    DecisionExpiry,
    RepairTargetAdmission,
    TargetRepositoryAuthority,
)
from ipfs_accelerate_py.agent_supervisor.program_contracts import (
    ContractSourceKind,
    ExpectedProgramContract,
    InterfaceIdentity,
    Optionality,
    ParameterKind,
    ParameterSpec,
    ReturnSpec,
    SourceReference,
    SymbolIdentity,
    TypeConstructor,
    TypeShape,
)
from ipfs_accelerate_py.agent_supervisor.proof.contract_repair_edit_packet import (
    CONTRACT_REPAIR_EDIT_PACKET_INTERFACE,
    materialize_contract_repair_edit_packet,
)
from ipfs_accelerate_py.agent_supervisor.proof.mcp_contract_edit_packet import (
    MCP_CONTRACT_EDIT_PACKET_DECISION_VERSION,
    WRITE_PATH_AUTHORITY_TARGET_DECISION,
    ContractEditPacketError,
    ContractEditPacketReason,
    ExpansionHandle,
    materialize_contract_edit_packet,
)


OLD_PATH = "ipfs_accelerate_py/mcp/old_dispatch.py"
MOVED_PATH = "ipfs_accelerate_py/mcp/moved_dispatch.py"
NEW_SITE_PATH = "ipfs_accelerate_py/mcp/new_site_impl.py"
DECOY_PATH = "ipfs_accelerate_py/mcp/decoy_same_name.py"
CALLER_PATH = "ipfs_accelerate_py/mcp/caller.py"

ROOTS = AuthorityRoots(
    repository_id="repository:test",
    forest_id="forest:test",
    tree_id="tree:test",
    graph_id="graph:test",
    index_id="index:test",
    model_id="model:test",
    config_id="config:test",
    translator_id="translator:test",
    toolchain_id="toolchain:test",
    policy_id="policy:test",
)


def ref(kind: str, artifact: str) -> EvidenceReference:
    return EvidenceReference(kind, artifact, producer_id="test")


def program_contract(name: str, path: str) -> ExpectedProgramContract:
    shape = TypeShape(TypeConstructor.STRING, name="str")
    return ExpectedProgramContract(
        symbol=SymbolIdentity("repository:test", "tree:test", path, name),
        interface=InterfaceIdentity("vfs", "tool", method="read"),
        policy_revision="policy:test",
        sources=(
            SourceReference(
                ContractSourceKind.REVIEWED_INTERFACE, "expected", f"contract:{name}"
            ),
        ),
        inputs=(
            ParameterSpec(
                "path", shape, ParameterKind.POSITIONAL, Optionality.REQUIRED, position=0
            ),
        ),
        returns=ReturnSpec(shape),
    )


def trace(
    *,
    target_hint: str = "old_receiver",
    disposition: TraceDisposition = TraceDisposition.LIKELY_REFACTOR,
) -> BrokenContractTrace:
    return BrokenContractTrace(
        ROOTS,
        SourceSpan(CALLER_PATH, 0, 10, "blob:caller"),
        "symbol:caller",
        target_hint,
        disposition,
        evidence_refs=(
            ref("trace", "trace:one"),
            ref("counterexample", "counterexample:one"),
        ),
        proof_refs=(ref("proof", "trace-proof:one"),),
    )


def comparison(value: BrokenContractTrace):
    return SenderReceiverContractCompiler().synthesize(
        value,
        program_contract("caller", CALLER_PATH),
        program_contract("receiver", MOVED_PATH),
    )


def candidate(
    value: BrokenContractTrace,
    *,
    path: str,
    strategy: RepairStrategy = RepairStrategy.RENAME_SUBSTITUTION,
    artifact: str = "blob:target",
) -> RepairCandidate:
    return RepairCandidate(
        ROOTS,
        value.content_id,
        strategy,
        SourceSpan(path, 0, 10, artifact),
        (ref("candidate", f"candidate:{path}"),),
        proof_refs=(ref("proof", f"candidate-proof:{path}"),),
    )


def receipt(
    items: tuple[RepairCandidate, ...],
    *,
    disposition: RerankDisposition = RerankDisposition.RANKED,
    selected: RepairCandidate | None = None,
) -> RerankReceipt:
    winner = selected or (items[0] if items else None)
    ranks = tuple(
        CandidateRank(
            item.content_id,
            CandidateEligibilityDisposition.ELIGIBLE,
            (100 if winner is not None and item is winner else 0, 0, 0, 0, 0, 0, 0),
            proof_receipt_ids=(f"proof:{item.content_id}",),
        )
        for item in items
    )
    return RerankReceipt(
        ROOTS,
        candidate_set_identity(items),
        "rerank:test",
        ranks,
        disposition,
        selected_candidate_id=winner.content_id if winner is not None and disposition is RerankDisposition.RANKED else "",
    )


def authority(
    item: RepairCandidate,
    items: tuple[RepairCandidate, ...],
    *,
    read_only: bool = False,
) -> TargetRepositoryAuthority:
    span = item.target_span
    return TargetRepositoryAuthority(
        ROOTS,
        candidate_set_identity(items),
        item.content_id,
        span,
        (span,),
        (span,),
        (ref("repository_authority", f"authority:{item.target_span.path}"),),
        read_only=read_only,
    )


def admit(
    items: tuple[RepairCandidate, ...],
    *,
    ranking: RerankReceipt | None = None,
    authorities: tuple[TargetRepositoryAuthority, ...] | None = None,
    expiry: DecisionExpiry | None = None,
):
    ranking = ranking or receipt(items)
    authorities = authorities or tuple(authority(item, items) for item in items)
    expiry = expiry or DecisionExpiry(100, 200)
    return RepairTargetAdmission().admit(
        items, ranking, authorities, expiry=expiry
    )


def repair_packet(
    admission,
    value: BrokenContractTrace,
    items: tuple[RepairCandidate, ...],
    ranking: RerankReceipt,
    authorities: tuple[TargetRepositoryAuthority, ...],
    **changes: object,
):
    arguments: dict[str, object] = {
        "roots": ROOTS,
        "candidates": items,
        "rerank_receipt": ranking,
        "authorities": authorities,
        "now": 150,
        "post_edit_obligation_ids": ("obligation:caller-implies-receiver",),
        "validation_commands": (
            "python -m pytest -q test/api/test_agent_supervisor_contract_repair_integration.py",
        ),
        "reproof_commands": (
            "python -m repair_reproof obligation:caller-implies-receiver",
        ),
        "counterexample_refs": (value.evidence_refs[1],),
    }
    arguments.update(changes)
    return materialize_contract_repair_edit_packet(
        admission, value, comparison(value), **arguments
    )


def _finding(
    *,
    snapshot_id: str = "tree:test",
    path: str = OLD_PATH,
) -> ContractFinding:
    claim = ContractParityClaim(
        family=McpClaimFamily.ARGUMENTS_PRESERVED,
        state=ParityState.REFUTED,
        operation_id="repo.inspect",
        premise_ids=("premise:descriptor", "premise:handler"),
        reason_codes=("argument_type_changed",),
        counterexamples=(
            ContractCounterexample(
                reason_code="argument_type_changed",
                boundary_id="tools/call",
                path="input.limit",
                expected="string",
                actual="integer",
                source_ids=("source:schema",),
            ),
        ),
    )
    findings = ContractMismatchAnalyzer().analyze_claim(
        claim,
        snapshot_id=snapshot_id,
        contract_id="contract:repo.inspect",
        affected_symbols=("handler:repo.inspect", "schema:repo.inspect"),
        affected_paths=(path,),
        obligation_ids=("obligation:arguments",),
        cas_handles=("bafy:contract-slice",),
        reproduction_commands=("python -m pytest test_contract.py -q",),
    )
    assert len(findings) == 1
    return findings[0]


def _mcp_packet(finding: ContractFinding | None = None, **changes: object):
    selected = finding or _finding()
    arguments: dict[str, object] = {
        "current_snapshot_id": selected.snapshot_id,
        "task_id": "RPR-017-fixture",
        "expected_postcondition": {
            "operation_id": "repo.inspect",
            "condition": "declared and executed argument types agree",
        },
        "validation_commands": ("python -m pytest test_contract.py -q",),
        "reproof_commands": (
            "python -m ipfs_accelerate_py.agent_supervisor.proof.recheck obligation:arguments",
        ),
        "read_paths": selected.affected_paths,
        "write_paths": selected.affected_paths,
        "dependency_ids": ("RPR-015", "RPR-016"),
        "mandatory_dependency_ids": ("RPR-015", "RPR-016"),
        "expansion_handles": (
            ExpansionHandle(
                handle_id="proof:arguments",
                kind="proof_receipt",
                content_id="bafy:proof-receipt",
                byte_count=32_000,
            ),
        ),
    }
    arguments.update(changes)
    return materialize_contract_edit_packet(selected, **arguments)


# ---------------------------------------------------------------------------
# @1 regression
# ---------------------------------------------------------------------------


def test_legacy_v1_requires_affected_paths_equality() -> None:
    finding = _finding(path=OLD_PATH)
    packet = _mcp_packet(finding)
    assert packet.write_paths == finding.affected_paths == (OLD_PATH,)
    assert packet.context_capsule.goal.get("packet_version", 1) == 1

    with pytest.raises(ContractEditPacketError) as error:
        _mcp_packet(finding, write_paths=(MOVED_PATH,))
    assert error.value.reason_code == ContractEditPacketReason.PATH_SCOPE_MISMATCH.value


def test_legacy_refinery_still_emits_v1_tasks() -> None:
    packet = _mcp_packet()
    result = ContractMismatchRefinery(
        ContractMismatchRefineryPolicy(cooldown_seconds=0)
    ).refine((packet,), current_snapshot_id=packet.snapshot_id, now_epoch=10)
    assert result.generated_count == 1
    assert result.tasks[0].write_paths == packet.write_paths
    assert result.tasks[0].affected_paths == result.tasks[0].write_paths


# ---------------------------------------------------------------------------
# @2 materialize: write paths from RepairTargetDecision
# ---------------------------------------------------------------------------


def test_v2_write_paths_derive_from_decision_not_affected_paths() -> None:
    finding = _finding(path=OLD_PATH)
    value = trace()
    item = candidate(value, path=MOVED_PATH)
    items = (item,)
    ranking = receipt(items)
    authorities = (authority(item, items),)
    admission = admit(items, ranking=ranking, authorities=authorities)

    packet = _mcp_packet(
        finding,
        packet_version=MCP_CONTRACT_EDIT_PACKET_DECISION_VERSION,
        target_decision=admission.decision,
        write_paths=None,
        read_paths=None,
    )

    assert packet.write_paths == (MOVED_PATH,)
    assert packet.write_paths != finding.affected_paths
    assert packet.context_capsule.goal["decision_id"] == admission.decision.content_id
    assert (
        packet.context_capsule.authority["write_path_authority"]
        == WRITE_PATH_AUTHORITY_TARGET_DECISION
    )


def test_v2_rejects_provider_expanded_write_scope() -> None:
    finding = _finding(path=OLD_PATH)
    value = trace()
    item = candidate(value, path=MOVED_PATH)
    items = (item,)
    admission = admit(items)

    with pytest.raises(ContractEditPacketError) as error:
        _mcp_packet(
            finding,
            packet_version=2,
            target_decision=admission.decision,
            write_paths=(MOVED_PATH, NEW_SITE_PATH),
        )
    assert error.value.reason_code == ContractEditPacketReason.DECISION_SCOPE_MISMATCH.value


def test_v2_rejects_non_admitted_decision() -> None:
    finding = _finding(path=OLD_PATH)
    value = trace()
    item = candidate(value, path=MOVED_PATH)
    other = candidate(value, path=DECOY_PATH, strategy=RepairStrategy.NEW_IMPLEMENTATION)
    items = (item, other)
    # Ambiguous ranking yields abstention.
    ranking = RerankReceipt(
        ROOTS,
        candidate_set_identity(items),
        "rerank:ambiguous",
        (
            CandidateRank(
                item.content_id,
                CandidateEligibilityDisposition.ELIGIBLE,
                (50, 0, 0, 0, 0, 0, 0),
                proof_receipt_ids=(f"proof:{item.content_id}",),
            ),
            CandidateRank(
                other.content_id,
                CandidateEligibilityDisposition.ELIGIBLE,
                (50, 0, 0, 0, 0, 0, 0),
                proof_receipt_ids=(f"proof:{other.content_id}",),
            ),
        ),
        RerankDisposition.AMBIGUOUS,
        selected_candidate_id="",
    )
    authorities = (authority(item, items), authority(other, items))
    admission = admit(items, ranking=ranking, authorities=authorities)
    assert admission.decision.disposition is not DecisionDisposition.ADMITTED

    with pytest.raises(ContractEditPacketError) as error:
        _mcp_packet(
            finding,
            packet_version=2,
            admission=admission,
        )
    assert error.value.reason_code == ContractEditPacketReason.DECISION_NOT_ADMITTED.value


# ---------------------------------------------------------------------------
# Analysis pipeline feature-gated @2 route
# ---------------------------------------------------------------------------


def _pipeline(**policy_fields: object) -> AnalysisPipeline:
    from ipfs_accelerate_py.agent_supervisor.analysis.analysis_cache import (
        AnalysisCache,
    )

    cache = AnalysisCache(path=None)  # type: ignore[arg-type]
    # Use a temp-less in-memory-friendly cache when path is required.
    import tempfile
    from pathlib import Path

    root = Path(tempfile.mkdtemp(prefix="rpr017-cache-"))
    cache = AnalysisCache(root)

    def _analyzer(context):  # pragma: no cover - unused on proof-gated route
        raise AssertionError("legacy analyzer must not run on proof-gated route")

    return AnalysisPipeline(
        cache,
        _analyzer,
        provider=object(),  # present but must never be invoked before admission
        policy=AnalysisPipelinePolicy(**policy_fields),
    )


def test_pipeline_route_disabled_by_default() -> None:
    value = trace()
    item = candidate(value, path=MOVED_PATH)
    items = (item,)
    ranking = receipt(items)
    authorities = (authority(item, items),)
    pipeline = _pipeline()  # enable_proof_gated_contract_repair defaults false
    result = pipeline.run_proof_gated_contract_repair(
        ProofGatedContractRepairRequest(
            roots=ROOTS,
            trace=value,
            comparison=comparison(value),
            candidates=items,
            rerank_receipt=ranking,
            authorities=authorities,
            expiry=DecisionExpiry(100, 200),
            now=150,
            post_edit_obligation_ids=("obligation:caller-implies-receiver",),
            validation_commands=("python -m pytest -q",),
            reproof_commands=("python -m repair_reproof",),
            counterexample_refs=(value.evidence_refs[1],),
        )
    )
    assert result.enabled is False
    assert result.packet is None
    assert result.provider_invoked_before_admission is False


def test_pipeline_route_admits_then_materializes_without_provider() -> None:
    value = trace()
    item = candidate(value, path=MOVED_PATH)
    items = (item,)
    ranking = receipt(items)
    authorities = (authority(item, items),)
    pipeline = _pipeline(enable_proof_gated_contract_repair=True)
    result = pipeline.run_proof_gated_contract_repair(
        {
            "roots": ROOTS,
            "trace": value,
            "comparison": comparison(value),
            "candidates": items,
            "rerank_receipt": ranking,
            "authorities": authorities,
            "expiry": DecisionExpiry(100, 200),
            "now": 150,
            "post_edit_obligation_ids": ("obligation:caller-implies-receiver",),
            "validation_commands": ("python -m pytest -q",),
            "reproof_commands": ("python -m repair_reproof",),
            "counterexample_refs": (value.evidence_refs[1],),
            "nomination_receipt_id": "nomination:rename",
        }
    )
    assert result.enabled is True
    assert result.admitted is True
    assert result.provider_invoked_before_admission is False
    assert result.stage == "materialize"
    assert result.packet is not None
    assert result.packet.interface == CONTRACT_REPAIR_EDIT_PACKET_INTERFACE
    assert result.write_paths == (MOVED_PATH,)
    assert result.write_paths == result.admission.decision.permitted_write_paths


# ---------------------------------------------------------------------------
# Integration scenarios
# ---------------------------------------------------------------------------


def test_rename_to_moved_file_integration() -> None:
    value = trace(disposition=TraceDisposition.LIKELY_REFACTOR)
    item = candidate(
        value, path=MOVED_PATH, strategy=RepairStrategy.RENAME_SUBSTITUTION
    )
    items = (item,)
    ranking = receipt(items)
    authorities = (authority(item, items),)
    admission = admit(items, ranking=ranking, authorities=authorities)
    packet = repair_packet(admission, value, items, ranking, authorities)

    assert packet.write_paths == (MOVED_PATH,)
    assert packet.strategy is RepairStrategy.RENAME_SUBSTITUTION

    pipeline = _pipeline(enable_proof_gated_contract_repair=True)
    routed = pipeline.run_proof_gated_contract_repair(
        ProofGatedContractRepairRequest(
            roots=ROOTS,
            trace=value,
            comparison=comparison(value),
            candidates=items,
            rerank_receipt=ranking,
            authorities=authorities,
            expiry=DecisionExpiry(100, 200),
            now=150,
            post_edit_obligation_ids=("obligation:caller-implies-receiver",),
            validation_commands=("python -m pytest -q",),
            reproof_commands=("python -m repair_reproof",),
            counterexample_refs=(value.evidence_refs[1],),
        )
    )
    assert routed.write_paths == (MOVED_PATH,)

    refinery = ContractMismatchRefinery(
        ContractMismatchRefineryPolicy(cooldown_seconds=0)
    )
    board = refinery.refine(
        (packet,),
        current_snapshot_id=ROOTS.tree_id,
        now_epoch=10,
        target_decisions={admission.decision.content_id: admission.decision},
    )
    assert board.generated_count == 1
    assert board.tasks[0].write_paths == (MOVED_PATH,)
    assert board.tasks[0].affected_paths == (MOVED_PATH,)


def test_new_site_integration() -> None:
    value = trace(
        target_hint="missing_receiver",
        disposition=TraceDisposition.MISSING_LOCAL,
    )
    item = candidate(
        value, path=NEW_SITE_PATH, strategy=RepairStrategy.NEW_IMPLEMENTATION
    )
    items = (item,)
    ranking = receipt(items)
    authorities = (authority(item, items),)
    admission = admit(items, ranking=ranking, authorities=authorities)
    assert admission.decision.strategy is RepairStrategy.NEW_IMPLEMENTATION

    pipeline = _pipeline(enable_proof_gated_contract_repair=True)
    result = pipeline.run_proof_gated_contract_repair(
        ProofGatedContractRepairRequest(
            roots=ROOTS,
            trace=value,
            comparison=comparison(value),
            candidates=items,
            rerank_receipt=ranking,
            authorities=authorities,
            expiry=DecisionExpiry(100, 200),
            now=150,
            post_edit_obligation_ids=("obligation:new-site",),
            validation_commands=("python -m pytest -q",),
            reproof_commands=("python -m repair_reproof",),
            counterexample_refs=(value.evidence_refs[1],),
        )
    )
    assert result.admitted is True
    assert result.write_paths == (NEW_SITE_PATH,)


def test_stale_roots_reject_before_materialization() -> None:
    value = trace()
    item = candidate(value, path=MOVED_PATH)
    items = (item,)
    ranking = receipt(items)
    authorities = (authority(item, items),)
    admission = admit(items, ranking=ranking, authorities=authorities)

    stale_roots = replace(ROOTS, tree_id="tree:stale")
    with pytest.raises(Exception):
        materialize_contract_repair_edit_packet(
            admission,
            value,
            comparison(value),
            roots=stale_roots,
            candidates=items,
            rerank_receipt=ranking,
            authorities=authorities,
            now=150,
            post_edit_obligation_ids=("obligation:x",),
            validation_commands=("python -m pytest -q",),
            reproof_commands=("python -m repair_reproof",),
            counterexample_refs=(value.evidence_refs[1],),
        )

    # Expired admission also fails closed.
    with pytest.raises(Exception):
        materialize_contract_repair_edit_packet(
            admission,
            value,
            comparison(value),
            roots=ROOTS,
            candidates=items,
            rerank_receipt=ranking,
            authorities=authorities,
            now=999,
            post_edit_obligation_ids=("obligation:x",),
            validation_commands=("python -m pytest -q",),
            reproof_commands=("python -m repair_reproof",),
            counterexample_refs=(value.evidence_refs[1],),
        )


def test_ambiguous_decision_never_materializes_packet() -> None:
    value = trace()
    primary = candidate(value, path=MOVED_PATH)
    decoy = candidate(
        value, path=DECOY_PATH, strategy=RepairStrategy.NEW_IMPLEMENTATION
    )
    items = (primary, decoy)
    ranking = RerankReceipt(
        ROOTS,
        candidate_set_identity(items),
        "rerank:ambiguous",
        (
            CandidateRank(
                primary.content_id,
                CandidateEligibilityDisposition.ELIGIBLE,
                (50, 0, 0, 0, 0, 0, 0),
                proof_receipt_ids=(f"proof:{primary.content_id}",),
            ),
            CandidateRank(
                decoy.content_id,
                CandidateEligibilityDisposition.ELIGIBLE,
                (50, 0, 0, 0, 0, 0, 0),
                proof_receipt_ids=(f"proof:{decoy.content_id}",),
            ),
        ),
        RerankDisposition.AMBIGUOUS,
        selected_candidate_id="",
    )
    authorities = (authority(primary, items), authority(decoy, items))
    pipeline = _pipeline(enable_proof_gated_contract_repair=True)
    result = pipeline.run_proof_gated_contract_repair(
        ProofGatedContractRepairRequest(
            roots=ROOTS,
            trace=value,
            comparison=comparison(value),
            candidates=items,
            rerank_receipt=ranking,
            authorities=authorities,
            expiry=DecisionExpiry(100, 200),
            now=150,
            post_edit_obligation_ids=("obligation:x",),
            validation_commands=("python -m pytest -q",),
            reproof_commands=("python -m repair_reproof",),
            counterexample_refs=(value.evidence_refs[1],),
        )
    )
    assert result.admitted is False
    assert result.disposition == "ambiguous"
    assert result.packet is None
    assert result.provider_invoked_before_admission is False


def test_read_only_target_abstains() -> None:
    value = trace()
    # Authority under accelerator-owned path still marks write spans empty.
    item = candidate(value, path=MOVED_PATH)
    items = (item,)
    ranking = receipt(items)
    authorities = (authority(item, items, read_only=True),)
    admission = admit(items, ranking=ranking, authorities=authorities)
    assert admission.decision.disposition is not DecisionDisposition.ADMITTED
    assert admission.decision.permitted_write_paths == ()

    pipeline = _pipeline(enable_proof_gated_contract_repair=True)
    result = pipeline.run_proof_gated_contract_repair(
        ProofGatedContractRepairRequest(
            roots=ROOTS,
            trace=value,
            comparison=comparison(value),
            candidates=items,
            rerank_receipt=ranking,
            authorities=authorities,
            expiry=DecisionExpiry(100, 200),
            now=150,
            post_edit_obligation_ids=("obligation:x",),
            validation_commands=("python -m pytest -q",),
            reproof_commands=("python -m repair_reproof",),
            counterexample_refs=(value.evidence_refs[1],),
        )
    )
    assert result.packet is None
    assert result.disposition in {"rejected", "ambiguous"}


def test_incompatible_decoy_does_not_win_write_path() -> None:
    value = trace()
    real = candidate(value, path=MOVED_PATH)
    decoy = candidate(
        value,
        path=DECOY_PATH,
        strategy=RepairStrategy.NEW_IMPLEMENTATION,
        artifact="blob:decoy",
    )
    items = (real, decoy)
    # Rank real higher so admission selects the moved file, not the decoy.
    ranking = receipt(items, selected=real)
    authorities = (authority(real, items), authority(decoy, items))
    admission = admit(items, ranking=ranking, authorities=authorities)
    assert admission.decision.disposition is DecisionDisposition.ADMITTED
    assert admission.decision.permitted_write_paths == (MOVED_PATH,)
    assert DECOY_PATH not in admission.decision.permitted_write_paths

    packet = repair_packet(admission, value, items, ranking, authorities)
    assert packet.write_paths == (MOVED_PATH,)
    assert DECOY_PATH not in packet.write_paths


def test_refinery_accepts_v2_only_after_decision_validation() -> None:
    value = trace()
    item = candidate(value, path=MOVED_PATH)
    items = (item,)
    ranking = receipt(items)
    authorities = (authority(item, items),)
    admission = admit(items, ranking=ranking, authorities=authorities)
    packet = repair_packet(admission, value, items, ranking, authorities)

    refinery = ContractMismatchRefinery(
        ContractMismatchRefineryPolicy(cooldown_seconds=0)
    )
    # Without an explicit decision, packet still projects because it already
    # embeds decision-bound write paths from materialization.
    ok = refinery.refine(
        (packet,),
        current_snapshot_id=ROOTS.tree_id,
        now_epoch=5,
    )
    assert ok.generated_count == 1
    assert ok.tasks[0].write_paths == (MOVED_PATH,)

    # Explicit decision validation path.
    validated = refinery.refine(
        (packet,),
        current_snapshot_id=ROOTS.tree_id,
        now_epoch=6,
        existing_board=ok.markdown,
        target_decisions={admission.decision.content_id: admission.decision},
    )
    assert ContractMismatchRefineryReason.DUPLICATE.value in {
        item.reason_code.value for item in validated.decisions
    }

    # Scope expansion is rejected: packet write paths that exceed the validated
    # decision fail closed (packet schema or decision binding).
    expanded = deepcopy(packet.to_dict())
    expanded["write_paths"] = [MOVED_PATH, NEW_SITE_PATH]
    expanded.pop("content_id", None)
    bad = refinery.refine(
        (expanded,),
        current_snapshot_id=ROOTS.tree_id,
        now_epoch=7,
        target_decisions={admission.decision.content_id: admission.decision},
    )
    reasons = {item.reason_code for item in bad.decisions}
    assert (
        ContractMismatchRefineryReason.MALFORMED_PACKET in reasons
        or ContractMismatchRefineryReason.SCOPE_EXPANSION in reasons
        or ContractMismatchRefineryReason.DECISION_INVALID in reasons
    )


def test_refinery_rejects_v2_when_policy_disables_proof_gated_packets() -> None:
    value = trace()
    item = candidate(value, path=MOVED_PATH)
    items = (item,)
    ranking = receipt(items)
    authorities = (authority(item, items),)
    admission = admit(items, ranking=ranking, authorities=authorities)
    packet = repair_packet(admission, value, items, ranking, authorities)

    refinery = ContractMismatchRefinery(
        ContractMismatchRefineryPolicy(
            cooldown_seconds=0, accept_proof_gated_packets=False
        )
    )
    result = refinery.refine(
        (packet,),
        current_snapshot_id=ROOTS.tree_id,
        now_epoch=1,
    )
    assert result.generated_count == 0
    assert any(
        item.reason_code is ContractMismatchRefineryReason.UNSUPPORTED_FINDING
        for item in result.decisions
    )


def test_no_provider_before_admission_invariant_on_route() -> None:
    """The @2 route never invokes optional providers for target selection."""

    class _ExplodingProvider:
        def analyze(self, *args, **kwargs):  # pragma: no cover - must not run
            raise AssertionError("provider invoked before admission")

        def build_request(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError("provider build_request before admission")

    from pathlib import Path
    import tempfile

    from ipfs_accelerate_py.agent_supervisor.analysis.analysis_cache import (
        AnalysisCache,
    )

    cache = AnalysisCache(Path(tempfile.mkdtemp(prefix="rpr017-prov-")))
    pipeline = AnalysisPipeline(
        cache,
        analyzer=lambda context: (_ for _ in ()).throw(
            AssertionError("analyzer unused")
        ),
        provider=_ExplodingProvider(),
        policy=AnalysisPipelinePolicy(
            enable_proof_gated_contract_repair=True,
            enable_datasets_provider=True,
        ),
    )
    value = trace()
    item = candidate(value, path=MOVED_PATH)
    items = (item,)
    ranking = receipt(items)
    authorities = (authority(item, items),)
    result = pipeline.run_proof_gated_contract_repair(
        ProofGatedContractRepairRequest(
            roots=ROOTS,
            trace=value,
            comparison=comparison(value),
            candidates=items,
            rerank_receipt=ranking,
            authorities=authorities,
            expiry=DecisionExpiry(100, 200),
            now=150,
            post_edit_obligation_ids=("obligation:x",),
            validation_commands=("python -m pytest -q",),
            reproof_commands=("python -m repair_reproof",),
            counterexample_refs=(value.evidence_refs[1],),
        )
    )
    assert result.provider_invoked_before_admission is False
    assert result.admitted is True
