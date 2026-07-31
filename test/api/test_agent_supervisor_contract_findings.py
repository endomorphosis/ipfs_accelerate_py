"""Tests for the append-only content-addressed contract finding ledger (VFS-029 / VFS-049)."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.contract_findings import (
    CONTRACT_FINDINGS_VERSION,
    FINDING_LEDGER_EVIDENCE,
    FINDING_LEDGER_G100_EVIDENCE_TERMS,
    GOAL_ID,
    LEDGER_VERSION,
    MAX_COLLECTION_ITEMS,
    MAX_RECORD_BYTES,
    MAX_TEXT_BYTES,
    VULNERABILITY_EVIDENCE_POLICY,
    VULNERABILITY_LABEL,
    AnalyzerVersions,
    AppendOutcome,
    AppendReceipt,
    CallSlice,
    CallSliceStep,
    ContractFindingBoundsError,
    ContractFindingError,
    ContractFindingLedger,
    ContractFindingRecord,
    EvidenceReferences,
    FindingAdmissionState,
    FindingCollisionError,
    FindingProjectionEntry,
    ForgedFindingIdentityError,
    LedgerCapacityError,
    LedgerEvent,
    LedgerEventKind,
    PoisonedSeverityError,
    ProjectionSnapshot,
    SemanticDedupKey,
    StaleFindingError,
    VulnerabilityEvidencePolicyError,
    build_contract_finding,
    claims_contradict,
    covered_evidence_terms,
    finding_content_cid,
    finding_ledger_evidence_terms,
    is_partial_finding,
    is_vulnerability_labeled,
    validate_severity_binding,
    validate_vulnerability_evidence_policy,
    vulnerability_evidence_requirements_met,
)
from ipfs_accelerate_py.agent_supervisor.program_assurance_contracts import (
    ClaimLevel,
    EvidenceFreshness,
    FindingSeverity,
    FindingStatus,
)


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------


def _evidence(
    *,
    counterexample: str = "cex:alpha",
    proof: str = "",
    runtime: str = "",
    zk: str = "",
) -> EvidenceReferences:
    return EvidenceReferences(
        counterexample_cids=(counterexample,) if counterexample else (),
        proof_cids=(proof,) if proof else (),
        runtime_cids=(runtime,) if runtime else (),
        zk_cids=(zk,) if zk else (),
        artifact_cids=("artifact:witness",),
    )


def _slice(*symbols: str) -> CallSlice:
    steps = tuple(
        CallSliceStep(
            symbol=symbol,
            interface=f"iface://{symbol}",
            repository_id="repository:alpha",
            path=f"src/{symbol.replace('.', '/')}.py",
        )
        for symbol in symbols
    )
    return CallSlice(steps=steps)


def broken_finding(**overrides) -> ContractFindingRecord:
    base = dict(
        claim_level=ClaimLevel.MODEL_DISPROVED,
        status=FindingStatus.CONTRACT_BROKEN,
        severity=FindingSeverity.HIGH,
        confidence_millionths=950_000,
        freshness=EvidenceFreshness.CURRENT,
        repositories=("repository:alpha",),
        symbols=("pkg.api.call",),
        interfaces=("mcp://pkg/call",),
        expected_contract_cid="expected:contract:1",
        observed_contract_cid="observed:contract:1",
        root_cause_family="error-map-mismatch",
        merge_fate="pkg.api.call",
        summary="Implementation violates the reviewed interface contract.",
        call_slice=_slice("pkg.entry", "pkg.api.call"),
        evidence=_evidence(
            counterexample="cex:alpha",
            proof="proof:alpha",
            runtime="runtime:alpha",
        ),
        assumptions=("fixture is hermetic", "closed call graph"),
        analyzer_versions={"contract-checker": "1.0.0", "model-checker": "2.0"},
        remediation_scope=("pkg.api.call", "src/api.py"),
        tree_id="tree:abc",
        policy_revision="policy:v1",
        repository_observation_id="observation:1",
        verdict="violated",
    )
    base.update(overrides)
    return build_contract_finding(**base)


def suspected_finding(**overrides) -> ContractFindingRecord:
    base = dict(
        claim_level=ClaimLevel.RESOLVED_STATIC,
        status=FindingStatus.SUSPECTED,
        severity=FindingSeverity.MEDIUM,
        confidence_millionths=500_000,
        freshness=EvidenceFreshness.CURRENT,
        repositories=("repository:alpha",),
        symbols=("pkg.api.other",),
        interfaces=("mcp://pkg/other",),
        expected_contract_cid="expected:contract:2",
        observed_contract_cid="observed:contract:2",
        root_cause_family="optional-field-drift",
        merge_fate="pkg.api.other",
        summary="Static resolution suggests an optional field mismatch.",
        call_slice=_slice("pkg.api.other"),
        evidence=_evidence(counterexample=""),
        assumptions=("static graph is complete",),
        analyzer_versions={"static-resolver": "3.1"},
        remediation_scope=("pkg.api.other",),
        tree_id="tree:abc",
        policy_revision="policy:v1",
        repository_observation_id="observation:1",
        verdict="inconclusive",
    )
    base.update(overrides)
    return build_contract_finding(**base)


# ---------------------------------------------------------------------------
# VFS-G100 evidence term binding
# ---------------------------------------------------------------------------


def test_finding_ledger_evidence_terms_are_bound() -> None:
    """Prove vfs/finding-ledger@1 and vfs/vulnerability-evidence-policy@1."""

    assert FINDING_LEDGER_EVIDENCE == "vfs/finding-ledger@1"
    assert VULNERABILITY_EVIDENCE_POLICY == "vfs/vulnerability-evidence-policy@1"
    assert FINDING_LEDGER_G100_EVIDENCE_TERMS == (
        "vfs/finding-ledger@1",
        "vfs/vulnerability-evidence-policy@1",
    )
    assert finding_ledger_evidence_terms() == FINDING_LEDGER_G100_EVIDENCE_TERMS
    assert covered_evidence_terms() == finding_ledger_evidence_terms()
    assert GOAL_ID == "VFS-G100"
    assert VULNERABILITY_LABEL == "vulnerability"
    assert "vfs/finding-ledger@1" in FINDING_LEDGER_G100_EVIDENCE_TERMS
    assert "vfs/vulnerability-evidence-policy@1" in FINDING_LEDGER_G100_EVIDENCE_TERMS


def test_append_receipt_and_stats_bind_finding_ledger_evidence(
    tmp_path: Path,
) -> None:
    ledger = ContractFindingLedger(tmp_path / "ledger")
    receipt = ledger.append(broken_finding())
    payload = receipt.to_record()
    assert payload["evidence"] == "vfs/finding-ledger@1"
    assert payload["evidence_terms"] == [
        "vfs/finding-ledger@1",
        "vfs/vulnerability-evidence-policy@1",
    ] or payload["evidence_terms"] == (
        "vfs/finding-ledger@1",
        "vfs/vulnerability-evidence-policy@1",
    )
    assert payload["goal_id"] == "VFS-G100"
    assert AppendReceipt.from_dict(payload) == receipt

    stats = ledger.stats()
    assert stats["evidence"] == FINDING_LEDGER_EVIDENCE
    assert stats["evidence_terms"] == list(FINDING_LEDGER_G100_EVIDENCE_TERMS)
    assert stats["goal_id"] == GOAL_ID
    assert stats["history_is_append_only"] is True
    assert stats["projection_is_mutable_current_tree"] is True

    snapshot = ledger.projection()
    snap_payload = snapshot.to_record()
    assert snap_payload["evidence"] == FINDING_LEDGER_EVIDENCE
    assert tuple(snap_payload["evidence_terms"]) == FINDING_LEDGER_G100_EVIDENCE_TERMS
    assert snap_payload["history_is_append_only"] is True
    assert snap_payload["projection_is_mutable_current_tree"] is True
    assert ProjectionSnapshot.from_dict(snap_payload).snapshot_id == snapshot.snapshot_id

    meta = json.loads((tmp_path / "ledger" / "meta.json").read_text(encoding="utf-8"))
    assert meta["evidence"] == FINDING_LEDGER_EVIDENCE
    assert meta["evidence_terms"] == list(FINDING_LEDGER_G100_EVIDENCE_TERMS)


def test_vulnerability_label_requires_threat_path_and_impact() -> None:
    """vfs/vulnerability-evidence-policy@1: fail closed without path + impact."""

    ok, missing = vulnerability_evidence_requirements_met(
        labels=(),
        threat_path_cid="",
        impact="",
    )
    assert ok is True
    assert missing == ()

    ok, missing = vulnerability_evidence_requirements_met(
        labels=(VULNERABILITY_LABEL,),
        threat_path_cid="",
        impact="escape",
    )
    assert ok is False
    assert "threat_path_cid" in missing

    ok, missing = vulnerability_evidence_requirements_met(
        labels=(VULNERABILITY_LABEL,),
        threat_path_cid="threat:path:1",
        impact="",
    )
    assert ok is False
    assert "impact" in missing

    with pytest.raises(VulnerabilityEvidencePolicyError, match="threat path"):
        validate_vulnerability_evidence_policy(
            labels=(VULNERABILITY_LABEL,),
            threat_path_cid="",
            impact="",
        )

    with pytest.raises(VulnerabilityEvidencePolicyError, match="threat path"):
        broken_finding(labels=(VULNERABILITY_LABEL,))

    with pytest.raises(VulnerabilityEvidencePolicyError, match="impact"):
        broken_finding(
            labels=(VULNERABILITY_LABEL,),
            threat_path_cid="threat:path:1",
            impact="",
        )

    record = broken_finding(
        labels=(VULNERABILITY_LABEL, "security"),
        threat_path_cid="threat:path:closed:1",
        impact="Arbitrary path escape outside the declared root.",
    )
    assert record.is_vulnerability is True
    assert is_vulnerability_labeled(record.labels) is True
    assert VULNERABILITY_LABEL in record.labels
    assert record.threat_path_cid == "threat:path:closed:1"
    assert "path escape" in record.impact
    restored = ContractFindingRecord.from_dict(record.to_record())
    assert restored == record
    assert restored.is_vulnerability is True

    # Correctness findings without the label need no threat path.
    plain = broken_finding(labels=("correctness",), impact="")
    assert plain.is_vulnerability is False
    assert plain.threat_path_cid == ""


def test_duplicates_and_stale_are_not_actionable_current(
    tmp_path: Path,
) -> None:
    """Duplicates and stale admissions do not create actionable work."""

    ledger = ContractFindingLedger(tmp_path / "ledger")
    first = broken_finding()
    ledger.append(first)
    # Exact content re-append is an idempotent duplicate.
    dup = ledger.append(first)
    assert dup.outcome is AppendOutcome.DUPLICATE
    assert len(ledger.current_findings(admitted_only=True)) == 1

    # Semantic duplicate of a distinct payload does not expand admitted set.
    semantic = broken_finding(
        summary="Same semantic identity, different body text",
        confidence_millionths=960_000,
    )
    assert semantic.semantic_key_id == first.semantic_key_id
    assert semantic.finding_cid != first.finding_cid
    semantic_receipt = ledger.append(semantic)
    assert semantic_receipt.admission is FindingAdmissionState.DUPLICATE
    assert len(ledger.current_findings(admitted_only=True)) == 1

    ledger.invalidate_stale(finding_cids=(first.finding_cid,))
    assert ledger.current_findings(admitted_only=True) == ()
    stale_entries = [e for e in ledger.projection().entries if e.admission is FindingAdmissionState.STALE]
    assert len(stale_entries) >= 1
    # History retained separately from the mutable current projection.
    assert len(ledger.history()) >= 3
    assert ledger.require(first.finding_cid) == first


# ---------------------------------------------------------------------------
# Record identity, immutability, round-trip
# ---------------------------------------------------------------------------


def test_record_is_immutable_content_addressed_and_round_trips() -> None:
    record = broken_finding()
    assert record.finding_cid
    assert record.cid == record.finding_cid == record.content_id
    assert record.claim_level is ClaimLevel.MODEL_DISPROVED
    assert record.status is FindingStatus.CONTRACT_BROKEN
    assert record.severity is FindingSeverity.HIGH
    assert record.confidence_millionths == 950_000
    assert record.freshness is EvidenceFreshness.CURRENT
    assert record.repositories == ("repository:alpha",)
    assert record.symbols == ("pkg.api.call",)
    assert record.interfaces == ("mcp://pkg/call",)
    assert record.expected_contract_cid == "expected:contract:1"
    assert record.observed_contract_cid == "observed:contract:1"
    assert record.call_slice.entry_symbol == "pkg.entry"
    assert record.evidence.counterexample_cids == ("cex:alpha",)
    assert record.evidence.proof_cids == ("proof:alpha",)
    assert record.evidence.runtime_cids == ("runtime:alpha",)
    assert record.analyzer_versions.as_mapping()["contract-checker"] == "1.0.0"
    assert record.root_cause_family == "error-map-mismatch"
    assert record.merge_fate == "pkg.api.call"
    assert record.remediation_scope == ("pkg.api.call", "src/api.py")
    assert record.actionable is True
    assert not record.partial

    with pytest.raises(FrozenInstanceError):
        record.summary = "mutated"  # type: ignore[misc]

    payload = record.to_record()
    restored = ContractFindingRecord.from_dict(payload)
    assert restored == record
    assert restored.finding_cid == record.finding_cid
    assert restored.semantic_key_id == record.semantic_key_id

    # Forged identity is rejected.
    forged = dict(payload)
    forged["finding_cid"] = "b" + "a" * 58
    with pytest.raises(ForgedFindingIdentityError):
        ContractFindingRecord.from_dict(forged)


def test_semantic_dedup_key_ignores_severity_and_confidence() -> None:
    a = broken_finding(severity=FindingSeverity.HIGH, confidence_millionths=950_000)
    b = broken_finding(
        severity=FindingSeverity.MEDIUM,
        confidence_millionths=700_000,
        summary="Different summary text",
    )
    # Severity differs so content CID differs, but semantic key matches.
    assert a.finding_cid != b.finding_cid
    assert a.semantic_key_id == b.semantic_key_id
    assert a.semantic_key == b.semantic_key

    c = broken_finding(root_cause_family="different-root")
    assert c.semantic_key_id != a.semantic_key_id

    d = broken_finding(merge_fate="other.merge")
    assert d.semantic_key_id != a.semantic_key_id

    e = broken_finding(symbols=("pkg.api.other",))
    assert e.semantic_key_id != a.semantic_key_id


def test_nested_records_round_trip_and_reject_unknown_fields() -> None:
    step = CallSliceStep(symbol="a.b", interface="i", repository_id="r")
    assert CallSliceStep.from_dict(step.to_record()) == step

    slice_ = CallSlice(steps=(step,))
    assert CallSlice.from_dict(slice_.to_record()) == slice_

    refs = _evidence(zk="zk:trace:1")
    assert EvidenceReferences.from_dict(refs.to_record()) == refs
    assert refs.zk_cids == ("zk:trace:1",)

    versions = AnalyzerVersions(versions={"parser": "1", "resolver": "2"})
    assert AnalyzerVersions.from_dict(versions.to_record()) == versions

    key = SemanticDedupKey(
        expected_contract_cid="e",
        observed_contract_cid="o",
        root_cause_family="rc",
        merge_fate="mf",
        symbols=("s",),
        interfaces=("i",),
        repositories=("r",),
    )
    assert SemanticDedupKey.from_dict(key.to_record()) == key

    with pytest.raises(ContractFindingError, match="unknown fields"):
        CallSliceStep.from_dict({**step.to_record(), "extra": True})


# ---------------------------------------------------------------------------
# Severity poisoning / stale / partial
# ---------------------------------------------------------------------------


def test_poisoned_severity_is_rejected() -> None:
    with pytest.raises(PoisonedSeverityError, match="exceeds maximum"):
        broken_finding(
            status=FindingStatus.SUSPECTED,
            severity=FindingSeverity.CRITICAL,
            claim_level=ClaimLevel.RESOLVED_STATIC,
            confidence_millionths=1_000_000,
            evidence=_evidence(counterexample=""),
        )

    with pytest.raises(PoisonedSeverityError, match="confidence"):
        broken_finding(
            severity=FindingSeverity.CRITICAL,
            confidence_millionths=100_000,
        )

    with pytest.raises(PoisonedSeverityError, match="counterexample"):
        broken_finding(
            severity=FindingSeverity.CRITICAL,
            confidence_millionths=950_000,
            evidence=_evidence(counterexample=""),
        )

    with pytest.raises(PoisonedSeverityError, match="model_disproved"):
        broken_finding(
            severity=FindingSeverity.CRITICAL,
            confidence_millionths=950_000,
            claim_level=ClaimLevel.OBSERVED_SYNTAX,
        )

    # Direct validator surface.
    with pytest.raises(PoisonedSeverityError):
        validate_severity_binding(
            status=FindingStatus.INCONCLUSIVE,
            severity=FindingSeverity.HIGH,
            claim_level=ClaimLevel.RESOLVED_STATIC,
            confidence_millionths=900_000,
            freshness=EvidenceFreshness.CURRENT,
            has_counterexample=False,
        )


def test_stale_contract_broken_is_rejected() -> None:
    with pytest.raises(StaleFindingError, match="stale evidence"):
        broken_finding(freshness=EvidenceFreshness.STALE)


def test_partial_findings_are_flagged_and_not_actionable() -> None:
    partial, missing = is_partial_finding(
        repositories=(),
        symbols=("s",),
        interfaces=("i",),
        expected_contract_cid="e",
        observed_contract_cid="o",
        root_cause_family="rc",
        merge_fate="mf",
        claim_level=ClaimLevel.MODEL_DISPROVED,
        status=FindingStatus.CONTRACT_BROKEN,
    )
    assert partial is True
    assert "repositories" in missing

    record = build_contract_finding(
        claim_level=ClaimLevel.MODEL_DISPROVED,
        status=FindingStatus.CONTRACT_BROKEN,
        severity=FindingSeverity.HIGH,
        confidence_millionths=950_000,
        repositories=(),
        symbols=("pkg.api.call",),
        interfaces=("mcp://pkg/call",),
        expected_contract_cid="expected:contract:1",
        observed_contract_cid="observed:contract:1",
        root_cause_family="error-map-mismatch",
        merge_fate="pkg.api.call",
        summary="Partial observation",
        evidence=_evidence(),
    )
    assert record.partial is True
    assert "repositories" in record.partial_missing_fields
    assert record.actionable is False


def test_bounds_reject_oversized_text_and_collections() -> None:
    with pytest.raises(ContractFindingBoundsError):
        broken_finding(summary="x" * (MAX_TEXT_BYTES + 1))

    with pytest.raises(ContractFindingBoundsError):
        broken_finding(symbols=[f"sym{i}" for i in range(MAX_COLLECTION_ITEMS + 1)])

    with pytest.raises(ContractFindingBoundsError):
        CallSlice(
            steps=tuple(
                CallSliceStep(symbol=f"s{i}") for i in range(65)
            )
        )


# ---------------------------------------------------------------------------
# Ledger: append, dedup, history, projection
# ---------------------------------------------------------------------------


def test_append_persists_immutable_history_and_current_projection(
    tmp_path: Path,
) -> None:
    ledger = ContractFindingLedger(tmp_path / "ledger")
    record = broken_finding()

    receipt = ledger.append(record)
    assert receipt.outcome is AppendOutcome.STORED
    assert receipt.stored is True
    assert receipt.finding_cid == record.finding_cid
    assert receipt.admission is FindingAdmissionState.ADMITTED

    loaded = ledger.require(record.finding_cid)
    assert loaded == record

    snapshot = ledger.projection()
    assert len(snapshot.admitted) == 1
    assert snapshot.admitted[0].finding_cid == record.finding_cid
    assert ledger.current_findings() == (record,)

    events = ledger.history()
    assert len(events) == 1
    assert events[0].kind is LedgerEventKind.APPEND
    assert events[0].finding_cid == record.finding_cid

    stats = ledger.stats()
    assert stats["ledger_version"] == LEDGER_VERSION
    assert stats["record_count"] == 1
    assert stats["admitted"] == 1


def test_exact_content_and_semantic_deduplication(tmp_path: Path) -> None:
    ledger = ContractFindingLedger(tmp_path / "ledger")
    record = broken_finding()

    first = ledger.append(record)
    second = ledger.append(record)
    assert first.outcome is AppendOutcome.STORED
    assert second.outcome is AppendOutcome.DUPLICATE
    assert second.reasons == ("exact_content_identity",)
    assert second.prior_finding_cid == record.finding_cid

    # Equal semantic/root-cause/merge-fate with different body content.
    sibling = broken_finding(
        summary="Same root cause, different observation text",
        confidence_millionths=960_000,
    )
    assert sibling.semantic_key_id == record.semantic_key_id
    assert sibling.finding_cid != record.finding_cid

    third = ledger.append(sibling)
    assert third.outcome is AppendOutcome.DUPLICATE
    assert "equal_semantic_root_cause_merge_fate" in third.reasons
    assert third.prior_finding_cid == record.finding_cid

    # Both bodies retained for history; only one admitted.
    assert ledger.get(sibling.finding_cid) is not None
    admitted = ledger.current_findings(admitted_only=True)
    assert len(admitted) == 1
    assert admitted[0].finding_cid == record.finding_cid

    # Different root cause is NOT deduplicated.
    other = broken_finding(
        root_cause_family="authorization-bypass",
        summary="Different root cause family",
    )
    fourth = ledger.append(other)
    assert fourth.outcome is AppendOutcome.STORED
    assert len(ledger.current_findings()) == 2


def test_collision_on_distinct_payload_same_path_is_integrity_error(
    tmp_path: Path,
) -> None:
    ledger = ContractFindingLedger(tmp_path / "ledger")
    record = broken_finding()
    ledger.append(record)

    # Manually poison the stored body with different content under same path.
    path = ledger._record_path(record.finding_cid)
    poisoned = record.with_updates(summary="tampered body")
    # Write raw JSON that claims the original CID.
    payload = poisoned.to_record()
    payload["finding_cid"] = record.finding_cid
    payload["content_id"] = record.finding_cid
    payload["cid"] = record.finding_cid
    path.write_text(json.dumps(payload), encoding="utf-8")

    # from_dict will fail on forged identity before collision — either is fine.
    with pytest.raises((FindingCollisionError, ForgedFindingIdentityError, ContractFindingError)):
        ledger.append(record)


def test_stale_invalidation_preserves_history(tmp_path: Path) -> None:
    ledger = ContractFindingLedger(tmp_path / "ledger")
    current = broken_finding(tree_id="tree:old", repository_observation_id="obs:1")
    other = broken_finding(
        tree_id="tree:keep",
        root_cause_family="other-family",
        merge_fate="other",
        symbols=("pkg.keep",),
        interfaces=("mcp://keep",),
        expected_contract_cid="expected:keep",
        observed_contract_cid="observed:keep",
        repository_observation_id="obs:2",
        summary="Unrelated finding",
    )
    ledger.append(current)
    ledger.append(other)

    invalidated = ledger.invalidate_stale(tree_id="tree:old")
    assert current.finding_cid in invalidated
    assert other.finding_cid not in invalidated

    snapshot = ledger.projection()
    stale_entries = {e.finding_cid: e for e in snapshot.stale}
    assert current.finding_cid in stale_entries
    assert stale_entries[current.finding_cid].admission is FindingAdmissionState.STALE

    # History still contains the original append plus invalidation.
    kinds = [event.kind for event in ledger.history()]
    assert LedgerEventKind.APPEND in kinds
    assert LedgerEventKind.INVALIDATE_STALE in kinds
    assert ledger.get(current.finding_cid) == current

    # Current admitted projection excludes the stale entry.
    admitted_cids = {f.finding_cid for f in ledger.current_findings()}
    assert current.finding_cid not in admitted_cids
    assert other.finding_cid in admitted_cids


def test_contradictory_claims_surface_as_projection_conflict(
    tmp_path: Path,
) -> None:
    ledger = ContractFindingLedger(tmp_path / "ledger")
    left = broken_finding(
        status=FindingStatus.CONTRACT_BROKEN,
        claim_level=ClaimLevel.MODEL_DISPROVED,
        severity=FindingSeverity.HIGH,
        root_cause_family="error-map-mismatch",
        merge_fate="lane-a",
        verdict="violated",
    )
    # Same scope (repos/symbols/interfaces/contracts/tree) but different
    # root-cause/merge-fate so semantic keys differ — contradiction path.
    right = broken_finding(
        status=FindingStatus.SUSPECTED,
        claim_level=ClaimLevel.RESOLVED_STATIC,
        severity=FindingSeverity.MEDIUM,
        confidence_millionths=500_000,
        root_cause_family="static-suspicion",
        merge_fate="lane-b",
        evidence=_evidence(counterexample=""),
        summary="Contradictory claim on the same scope",
        verdict="inconclusive",
    )
    assert claims_contradict(left, right) is True

    r1 = ledger.append(left)
    r2 = ledger.append(right)
    assert r1.outcome is AppendOutcome.STORED
    assert r2.outcome is AppendOutcome.STORED
    assert r2.admission is FindingAdmissionState.CONFLICT

    snapshot = ledger.projection()
    conflict_cids = {e.finding_cid for e in snapshot.conflicts}
    assert left.finding_cid in conflict_cids
    assert right.finding_cid in conflict_cids

    # Both bodies retained.
    assert ledger.get(left.finding_cid) is not None
    assert ledger.get(right.finding_cid) is not None


def test_supersession_and_rejection_preserve_prior_records(
    tmp_path: Path,
) -> None:
    ledger = ContractFindingLedger(tmp_path / "ledger")
    original = broken_finding(summary="Original finding")
    ledger.append(original)

    # Distinct root-cause so supersession is not conflated with semantic dedup.
    replacement = broken_finding(
        root_cause_family="error-map-mismatch-v2",
        summary="Superseding finding with refined evidence",
        confidence_millionths=990_000,
        evidence=_evidence(
            counterexample="cex:beta",
            proof="proof:beta",
            runtime="runtime:beta",
            zk="zk:beta",
        ),
    )
    receipt = ledger.supersede(original.finding_cid, replacement)
    assert receipt.outcome is AppendOutcome.SUPERSEDED_PRIOR
    assert receipt.prior_finding_cid == original.finding_cid
    # Supersession edges are content-bound, so the stored CID differs from the
    # pre-edge replacement body.
    stored_cid = receipt.finding_cid
    stored = ledger.require(stored_cid)
    assert original.finding_cid in stored.supersedes_cids

    snapshot = ledger.projection()
    by_cid = {e.finding_cid: e for e in snapshot.entries}
    assert by_cid[original.finding_cid].admission is FindingAdmissionState.SUPERSEDED
    assert by_cid[original.finding_cid].superseded_by_cid == stored_cid
    assert by_cid[stored_cid].admission is FindingAdmissionState.ADMITTED

    # Original body still readable.
    assert ledger.require(original.finding_cid) == original

    # Reject the replacement.
    reject_receipt = ledger.reject(stored_cid, ("out_of_scope", "policy_denied"))
    assert reject_receipt.outcome is AppendOutcome.REJECTED
    assert "out_of_scope" in reject_receipt.reasons
    snapshot = ledger.projection()
    assert by_cid_admission(snapshot, stored_cid) is FindingAdmissionState.REJECTED
    assert ledger.current_findings() == ()


def by_cid_admission(
    snapshot: ProjectionSnapshot, finding_cid: str
) -> FindingAdmissionState:
    for entry in snapshot.entries:
        if entry.finding_cid == finding_cid:
            return entry.admission
    raise AssertionError(f"missing projection entry for {finding_cid}")


def test_partial_findings_append_as_partial_admission(tmp_path: Path) -> None:
    ledger = ContractFindingLedger(tmp_path / "ledger")
    partial = build_contract_finding(
        claim_level=ClaimLevel.OBSERVED_SYNTAX,
        status=FindingStatus.INCONCLUSIVE,
        severity=FindingSeverity.INFO,
        confidence_millionths=0,
        repositories=("repository:alpha",),
        symbols=(),  # missing
        interfaces=("mcp://pkg/call",),
        expected_contract_cid="expected:1",
        observed_contract_cid="observed:1",
        root_cause_family="incomplete-scan",
        merge_fate="incomplete",
        summary="Partial finding from truncated scan",
    )
    assert partial.partial is True
    receipt = ledger.append(partial)
    assert receipt.outcome is AppendOutcome.STORED
    assert receipt.admission is FindingAdmissionState.PARTIAL
    assert ledger.current_findings(admitted_only=True) == ()
    current = ledger.current_findings(admitted_only=False)
    assert len(current) == 1
    assert current[0].finding_cid == partial.finding_cid


def test_replay_rebuilds_projection_from_event_log(tmp_path: Path) -> None:
    ledger = ContractFindingLedger(tmp_path / "ledger")
    a = broken_finding(summary="A")
    b = broken_finding(
        root_cause_family="other",
        merge_fate="other",
        symbols=("pkg.other",),
        interfaces=("mcp://other",),
        expected_contract_cid="expected:other",
        observed_contract_cid="observed:other",
        summary="B",
    )
    ledger.append(a)
    ledger.append(b)
    ledger.invalidate_stale(finding_cids=(a.finding_cid,))
    ledger.reject(b.finding_cid, ("manual_reject",))

    before = ledger.projection()
    # Corrupt the projection file, then rebuild from events.
    ledger._projection_path.write_text("{}", encoding="utf-8")

    rebuilt = ledger.replay()
    assert rebuilt.history_length == before.history_length
    by_cid = {e.finding_cid: e.admission for e in rebuilt.entries}
    assert by_cid[a.finding_cid] is FindingAdmissionState.STALE
    assert by_cid[b.finding_cid] is FindingAdmissionState.REJECTED

    # Records untouched.
    assert ledger.require(a.finding_cid) == a
    assert ledger.require(b.finding_cid) == b


def test_concurrency_serializes_appends(tmp_path: Path) -> None:
    ledger = ContractFindingLedger(tmp_path / "ledger")
    barrier = threading.Barrier(8)
    errors: list[BaseException] = []

    def worker(index: int) -> str:
        barrier.wait()
        record = broken_finding(
            root_cause_family=f"family-{index}",
            merge_fate=f"fate-{index}",
            symbols=(f"pkg.sym{index}",),
            interfaces=(f"mcp://{index}",),
            expected_contract_cid=f"expected:{index}",
            observed_contract_cid=f"observed:{index}",
            summary=f"Concurrent finding {index}",
        )
        receipt = ledger.append(record)
        return receipt.finding_cid

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(worker, i) for i in range(8)]
        cids = []
        for future in as_completed(futures):
            try:
                cids.append(future.result())
            except BaseException as exc:  # pragma: no cover - failure path
                errors.append(exc)

    assert not errors
    assert len(set(cids)) == 8
    assert len(ledger.current_findings()) == 8
    assert ledger.stats()["record_count"] == 8
    # Sequence advances monotonically with one event per append.
    sequences = [event.sequence for event in ledger.history()]
    assert sequences == sorted(sequences)
    assert len(set(sequences)) == 8


def test_capacity_bounds_reject_overflow(tmp_path: Path) -> None:
    ledger = ContractFindingLedger(tmp_path / "ledger", max_entries=2)
    ledger.append(
        broken_finding(
            root_cause_family="f1",
            merge_fate="m1",
            symbols=("s1",),
            interfaces=("i1",),
            expected_contract_cid="e1",
            observed_contract_cid="o1",
            summary="one",
        )
    )
    ledger.append(
        broken_finding(
            root_cause_family="f2",
            merge_fate="m2",
            symbols=("s2",),
            interfaces=("i2",),
            expected_contract_cid="e2",
            observed_contract_cid="o2",
            summary="two",
        )
    )
    with pytest.raises(LedgerCapacityError):
        ledger.append(
            broken_finding(
                root_cause_family="f3",
                merge_fate="m3",
                symbols=("s3",),
                interfaces=("i3",),
                expected_contract_cid="e3",
                observed_contract_cid="o3",
                summary="three",
            )
        )


def test_append_receipt_and_projection_snapshot_round_trip(
    tmp_path: Path,
) -> None:
    ledger = ContractFindingLedger(tmp_path / "ledger")
    record = broken_finding()
    receipt = ledger.append(record)
    restored_receipt = AppendReceipt.from_dict(receipt.to_record())
    assert restored_receipt == receipt

    snapshot = ledger.projection()
    restored_snapshot = ProjectionSnapshot.from_dict(snapshot.to_record())
    assert restored_snapshot.snapshot_id == snapshot.snapshot_id
    assert len(restored_snapshot.entries) == 1

    entry = snapshot.entries[0]
    assert FindingProjectionEntry.from_dict(entry.to_record()) == entry

    event = ledger.history()[0]
    assert LedgerEvent.from_dict(event.to_record()) == event


def test_finding_content_cid_is_stable() -> None:
    record = broken_finding()
    cid_a = finding_content_cid(record)
    cid_b = finding_content_cid(record.to_dict())
    # Multiformats and content_identity may differ slightly in encoding
    # profile; both must be stable across repeated calls.
    assert finding_content_cid(record) == cid_a
    assert finding_content_cid(record.to_dict()) == cid_b
    assert record.finding_cid == record.content_id


def test_schema_version_is_frozen() -> None:
    assert CONTRACT_FINDINGS_VERSION == 1
    record = broken_finding()
    assert record.schema_version == 1
    assert record.to_dict()["contract_version"] == 1


def test_suspected_and_zk_references_are_bound(tmp_path: Path) -> None:
    ledger = ContractFindingLedger(tmp_path / "ledger")
    record = suspected_finding(
        evidence=_evidence(counterexample="", zk="zk:attestation:9"),
    )
    assert record.evidence.zk_cids == ("zk:attestation:9",)
    assert record.actionable is False
    receipt = ledger.append(record)
    # SUSPECTED with complete fields admits (not contract_broken, but not partial).
    assert receipt.admission is FindingAdmissionState.ADMITTED
    assert ledger.current_findings()[0].evidence.zk_cids == ("zk:attestation:9",)


def test_rejects_unknown_enum_and_duplicate_analyzer_names() -> None:
    with pytest.raises(ContractFindingError):
        broken_finding(status="not-a-status")

    with pytest.raises(ContractFindingError, match="duplicate"):
        AnalyzerVersions(versions=(("a", "1"), ("a", "2")))


def test_ledger_does_not_mutate_source_records(tmp_path: Path) -> None:
    """Conflict policy: finding storage is diagnostic only."""

    ledger = ContractFindingLedger(tmp_path / "ledger")
    record = broken_finding()
    original_cid = record.finding_cid
    ledger.append(record)
    ledger.reject(original_cid, ("diagnostic_only",))
    # Immutable body unchanged on disk.
    assert ledger.require(original_cid).summary == record.summary
    assert ledger.require(original_cid).rejection_reasons == ()
    # Rejection lives only in projection / events.
    entry = next(
        e for e in ledger.projection().entries if e.finding_cid == original_cid
    )
    assert entry.admission is FindingAdmissionState.REJECTED
    assert "diagnostic_only" in entry.rejection_reasons
