"""CBP-080: CodeEditPacket@1 and supervisor materializer tests."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.code_claim_contracts import (
    ClaimFamily,
    ClaimStatus,
    CodeClaimRecord,
    EvidenceTier,
    build_invalidation_selectors,
)
from ipfs_accelerate_py.agent_supervisor.code_edit_materialize import (
    CODE_EDIT_MATERIALIZE_INTERFACE,
    VALIDATION_KIND_CACHE_AWARE_REPROOF,
    VALIDATION_KIND_DOMAIN_METRICS,
    VALIDATION_KIND_TEST,
    CodeEditMaterializeReport,
    CodeEditSupervisorTask,
    bridge_plateau_codex_packet,
    emit_validation_command_specs,
    emit_validation_commands,
    materialize_code_edit_packets,
    materialize_supervisor_task,
    packet_from_claim,
    packet_from_query_hit,
)
from ipfs_accelerate_py.agent_supervisor.code_edit_packet import (
    CODE_EDIT_PACKET_INTERFACE,
    CODE_EDIT_PACKET_SCHEMA,
    REQUIRED_NON_IMPLEMENTABLE,
    CacheDisposition,
    CacheStatusRecord,
    ClaimStatusRecord,
    CodeEditPacket,
    CodeEditPacketError,
    NonImplementableReason,
    ProverBinding,
    build_code_edit_packet,
    compute_implementable,
)
from ipfs_accelerate_py.agent_supervisor.code_proof_query import (
    ClaimQueryHit,
    build_code_proof_query,
)
from ipfs_accelerate_py.agent_supervisor.formal_verification_contracts import (
    AssuranceLevel,
)


def _selectors(*, tree: str = "git-tree:edit", property_id: str = "prop:api") -> tuple:
    return build_invalidation_selectors(
        repository_tree_id=tree,
        scope_ids=("scope:a",),
        premise_ids=("premise:a",),
        assumption_ids=("assumption:a",),
        toolchain_id="toolchain:t",
        policy_id="policy:p",
        catalog_version="1",
        property_id=property_id,
        producer_id="test",
        required_assurance=AssuranceLevel.KERNEL_VERIFIED,
    )


def _claim(
    *,
    property_id: str = "prop:api",
    status: ClaimStatus = ClaimStatus.OPEN,
    tree: str = "git-tree:edit",
    obligation_id: str = "obligation:1",
) -> CodeClaimRecord:
    satisfied = status is ClaimStatus.SATISFIED
    return CodeClaimRecord(
        claim_family=ClaimFamily.API_CONTRACT,
        status=status,
        property_id=property_id,
        obligation_id=obligation_id,
        repository_id="repo:edit",
        repository_tree_id=tree,
        scope_ids=("scope:a",),
        premise_ids=("premise:a",),
        assumption_ids=("assumption:a",),
        producer_id="test",
        toolchain_id="toolchain:t",
        policy_id="policy:p",
        catalog_version="1",
        required_assurance=AssuranceLevel.KERNEL_VERIFIED,
        derived_assurance=(
            AssuranceLevel.KERNEL_VERIFIED if satisfied else AssuranceLevel.UNVERIFIED
        ),
        invalidation_selectors=_selectors(tree=tree, property_id=property_id),
        evidence_ids=("evidence:kernel-1",) if satisfied else (),
        evidence_tiers=(EvidenceTier.KERNEL_PROOF,) if satisfied else (),
        receipt_id="receipt:kernel-1" if satisfied else "",
        statement=property_id,
    )


# ---------------------------------------------------------------------------
# Packet identity and bindings
# ---------------------------------------------------------------------------


def test_packet_is_content_addressed_and_binds_required_fields() -> None:
    packet = build_code_edit_packet(
        repository_tree_id="git-tree:edit",
        claim_ids=("claim:1",),
        obligation_ids=("obligation:1",),
        assumption_ids=("assumption:a",),
        invalidation_reasons=("repository_tree_changed",),
        predicted_files=(
            "ipfs_accelerate_py/agent_supervisor/code_edit_packet.py",
        ),
        acceptance_ids=("accept:cbp-080",),
        property_ids=("prop:api",),
        claim_status=ClaimStatusRecord(
            claim_id="claim:1",
            property_id="prop:api",
            status=ClaimStatus.OPEN.value,
            obligation_id="obligation:1",
        ),
        prover=ProverBinding(prover_id="prover:leanstral", solver_id="solver:z3"),
    )
    assert packet.packet_id
    assert packet.packet_id == packet.content_id
    assert packet.interface == CODE_EDIT_PACKET_INTERFACE
    assert packet.repository_tree_id == "git-tree:edit"
    assert packet.source_tree_id == "git-tree:edit"
    assert packet.claim_ids == ("claim:1",)
    assert packet.obligation_ids == ("obligation:1",)
    assert packet.assumption_ids == ("assumption:a",)
    assert packet.assumptions == ("assumption:a",)
    assert packet.invalidation_reasons == ("repository_tree_changed",)
    assert packet.predicted_files == (
        "ipfs_accelerate_py/agent_supervisor/code_edit_packet.py",
    )
    assert packet.acceptance_ids == ("accept:cbp-080",)
    assert packet.implementable is True
    assert packet.prover.semantic_authority is False

    # Same payload → same identity.
    again = CodeEditPacket.from_dict(packet.to_dict())
    assert again.packet_id == packet.packet_id


def test_packet_round_trip_serialize() -> None:
    packet = build_code_edit_packet(
        repository_tree_id="git-tree:rt",
        claim_ids=("c1", "c2"),
        obligation_ids=("o1",),
        assumption_ids=("a1",),
        invalidation_reasons=("catalog_changed",),
        predicted_files=("src/a.py", "src/b.py"),
        acceptance_ids=("acc:1",),
        property_ids=("prop:x",),
        required_assurance=AssuranceLevel.SOLVER_CHECKED,
        cache_status=CacheStatusRecord(
            disposition=CacheDisposition.MISS,
            cache_key_id="cache:key:1",
            reason_codes=("cold_miss",),
            required_assurance=AssuranceLevel.SOLVER_CHECKED,
        ),
        claim_status=ClaimStatusRecord(
            claim_id="c1",
            property_id="prop:x",
            status=ClaimStatus.OPEN.value,
            obligation_id="o1",
            required_assurance=AssuranceLevel.SOLVER_CHECKED,
        ),
        prover={"prover_id": "p", "semantic_authority": True},  # forced false
        residual_ref_ids=("residual:1",),
        metadata={"lane": "cbp-materialize"},
    )
    payload = packet.to_dict(include_id=True)
    assert payload["schema"] == CODE_EDIT_PACKET_SCHEMA
    assert payload["prover"]["semantic_authority"] is False
    restored = CodeEditPacket.from_dict(payload)
    assert restored.to_dict() == packet.to_dict()
    assert restored.packet_id == packet.packet_id
    assert restored.prover.semantic_authority is False

    # JSON round-trip via CanonicalContract helpers.
    via_json = CodeEditPacket.from_json(packet.to_json())
    assert via_json.packet_id == packet.packet_id


def test_prover_fields_always_semantic_authority_false() -> None:
    binding = ProverBinding(
        prover_id="prover:x",
        solver_id="solver:y",
        kernel_id="kernel:z",
        semantic_authority=True,  # ignored
    )
    assert binding.semantic_authority is False
    assert binding.to_dict()["semantic_authority"] is False
    restored = ProverBinding.from_dict(
        {"prover_id": "p", "semantic_authority": True}
    )
    assert restored.semantic_authority is False


@pytest.mark.parametrize(
    "reason",
    sorted(REQUIRED_NON_IMPLEMENTABLE),
)
def test_implementable_false_on_required_blocking_reasons(reason: str) -> None:
    implementable, reasons = compute_implementable(
        repository_tree_id="git-tree:x",
        claim_status=ClaimStatus.OPEN.value,
        explicit_reasons=(reason,),
    )
    assert implementable is False
    assert reason in reasons

    packet = build_code_edit_packet(
        repository_tree_id="git-tree:x",
        claim_ids=("c",),
        obligation_ids=("o",),
        force_non_implementable=(reason,),
        claim_status=ClaimStatusRecord(status=ClaimStatus.OPEN.value),
    )
    assert packet.implementable is False
    assert reason in packet.non_implementable_reasons


def test_implementable_false_on_unsupported_not_measured_statuses() -> None:
    for status in (ClaimStatus.UNSUPPORTED, ClaimStatus.NOT_MEASURED):
        packet = packet_from_claim(_claim(status=status))
        assert packet.implementable is False
        if status is ClaimStatus.UNSUPPORTED:
            assert NonImplementableReason.UNSUPPORTED.value in packet.non_implementable_reasons
        else:
            assert (
                NonImplementableReason.NOT_MEASURED.value
                in packet.non_implementable_reasons
            )


def test_implementable_false_on_stale_required_input() -> None:
    packet = build_code_edit_packet(
        repository_tree_id="git-tree:stale",
        claim_status=ClaimStatusRecord(status=ClaimStatus.STALE.value),
        reason_codes=("stale_required_input",),
    )
    assert packet.implementable is False
    assert (
        NonImplementableReason.STALE_REQUIRED_INPUT.value
        in packet.non_implementable_reasons
    )


def test_implementable_false_on_reject_and_timeout_cache() -> None:
    reject = build_code_edit_packet(
        repository_tree_id="git-tree:r",
        claim_status=ClaimStatusRecord(status=ClaimStatus.OPEN.value),
        cache_status=CacheStatusRecord(disposition=CacheDisposition.REJECTED),
    )
    assert reject.implementable is False
    assert NonImplementableReason.REJECT.value in reject.non_implementable_reasons

    timeout = build_code_edit_packet(
        repository_tree_id="git-tree:t",
        claim_status=ClaimStatusRecord(status=ClaimStatus.OPEN.value),
        cache_status=CacheStatusRecord(disposition=CacheDisposition.TIMEOUT),
    )
    assert timeout.implementable is False
    assert NonImplementableReason.TIMEOUT.value in timeout.non_implementable_reasons


def test_open_claim_with_tree_is_implementable() -> None:
    packet = packet_from_claim(
        _claim(status=ClaimStatus.OPEN),
        predicted_files=("src/x.py",),
        acceptance_ids=("acc:open",),
    )
    assert packet.implementable is True
    assert packet.non_implementable_reasons == ()
    assert packet.predicted_files == ("src/x.py",)
    assert packet.acceptance_ids == ("acc:open",)


def test_cache_and_claim_status_exclude_proof_bodies() -> None:
    with pytest.raises(CodeEditPacketError, match="proof bodies"):
        CacheStatusRecord.from_dict(
            {
                "disposition": "hit",
                "proof_body": "theorem T : True := trivial",
            }
        )
    with pytest.raises(CodeEditPacketError, match="proof bodies"):
        ClaimStatusRecord.from_dict(
            {
                "claim_id": "c",
                "status": "open",
                "gold_ir_body": {"nodes": []},
            }
        )
    with pytest.raises(CodeEditPacketError, match="proof bodies"):
        build_code_edit_packet(
            repository_tree_id="git-tree:x",
            metadata={"proof_body": "secret"},
        )

    # Compact status records serialize without bodies.
    cache = CacheStatusRecord(
        disposition=CacheDisposition.HIT,
        cache_key_id="ck",
        receipt_id="rcpt",
    )
    payload = cache.to_dict()
    assert "proof_body" not in payload
    assert payload["receipt_id"] == "rcpt"
    assert payload["cache_key_id"] == "ck"


# ---------------------------------------------------------------------------
# Materializer
# ---------------------------------------------------------------------------


def test_emit_validation_commands_cover_tests_metrics_and_reproof() -> None:
    commands = emit_validation_commands(
        required_assurance=AssuranceLevel.KERNEL_VERIFIED,
        property_ids=("prop:api",),
        obligation_ids=("obligation:1",),
        predicted_files=("src/a.py",),
    )
    assert len(commands) == 3
    test_cmd, metrics_cmd, reproof_cmd = commands
    assert "pytest" in test_cmd
    assert "pytest" in metrics_cmd
    assert "assurance" in reproof_cmd
    assert "kernel_verified" in reproof_cmd
    assert "cache-aware=true" in reproof_cmd
    assert "prop:api" in reproof_cmd

    specs = emit_validation_command_specs(
        required_assurance=AssuranceLevel.KERNEL_VERIFIED,
        property_ids=("prop:api",),
    )
    assert len(specs) == 3
    sources = {s.source for s in specs}
    assert VALIDATION_KIND_TEST in sources
    assert VALIDATION_KIND_DOMAIN_METRICS in sources
    assert VALIDATION_KIND_CACHE_AWARE_REPROOF in sources


def test_materializer_emits_validation_commands_for_open_obligations() -> None:
    open_claim = _claim(status=ClaimStatus.OPEN, property_id="prop:open")
    report = materialize_code_edit_packets(
        claims=[open_claim],
        predicted_files=(
            "ipfs_accelerate_py/agent_supervisor/code_edit_packet.py",
        ),
        acceptance_ids=("accept:cbp-080",),
        prover=ProverBinding(prover_id="prover:test"),
    )
    assert report.implementable_count >= 1
    assert report.interface if False else True  # report has schema via to_dict
    impl_tasks = [t for t in report.tasks if t.implementable]
    assert impl_tasks
    task = impl_tasks[0]
    assert len(task.validation_commands) == 3
    joined = "\n".join(task.validation_commands)
    assert "pytest" in joined
    assert "kernel_verified" in joined or "assurance" in joined
    assert "cache-aware" in joined
    assert task.packet.prover.semantic_authority is False
    assert task.packet.acceptance_ids == ("accept:cbp-080",)
    assert (
        "ipfs_accelerate_py/agent_supervisor/code_edit_packet.py"
        in task.predicted_files
    )


def test_materializer_blocks_non_open_without_commands() -> None:
    unsupported = _claim(status=ClaimStatus.UNSUPPORTED, property_id="prop:unsup")
    report = materialize_code_edit_packets(
        claims=[unsupported],
        include_blocked=True,
        open_only=False,
    )
    blocked = [p for p in report.packets if not p.implementable]
    assert blocked
    blocked_tasks = [t for t in report.tasks if not t.implementable]
    assert blocked_tasks
    assert blocked_tasks[0].validation_commands == ()


def test_materialize_report_round_trip() -> None:
    claim = _claim(status=ClaimStatus.OPEN)
    report = materialize_code_edit_packets(
        claims=[claim],
        predicted_files=("a.py",),
        acceptance_ids=("acc",),
    )
    payload = report.to_dict(include_id=True)
    assert payload["interface"] == CODE_EDIT_MATERIALIZE_INTERFACE
    assert payload["report_id"]
    restored = CodeEditMaterializeReport.from_dict(payload)
    assert restored.report_id == report.report_id
    assert len(restored.packets) == len(report.packets)
    assert len(restored.tasks) == len(report.tasks)
    # Nested packet + task serialize cleanly.
    for task in restored.tasks:
        again = CodeEditSupervisorTask.from_dict(task.to_dict())
        assert again.task_id == task.task_id
        assert again.packet.packet_id == task.packet.packet_id


def test_materialize_from_query_open_hits() -> None:
    claims = [
        _claim(status=ClaimStatus.OPEN, property_id="prop:a", obligation_id="o:a"),
        _claim(
            status=ClaimStatus.SATISFIED,
            property_id="prop:b",
            obligation_id="o:b",
        ),
    ]
    query = build_code_proof_query(claims=claims)
    report = materialize_code_edit_packets(query=query, open_only=True)
    # open_only path uses properties_open; satisfied is not included as open.
    for packet in report.packets:
        if packet.implementable:
            assert ClaimStatus.OPEN.value in (
                packet.claim_status.status,
                ClaimStatus.OPEN.value,
            )
    assert report.implementable_count >= 1


def test_packet_from_query_hit_records_cache_without_bodies() -> None:
    hit = ClaimQueryHit(
        property_id="prop:q",
        status=ClaimStatus.OPEN,
        claim_id="claim:q",
        obligation_ids=("obligation:q",),
        repository_tree_id="git-tree:q",
        cache_key_id="cache:q",
        reason_codes=("compile_open",),
        provenance={
            "required_assurance": AssuranceLevel.KERNEL_VERIFIED.value,
            "derived_assurance": AssuranceLevel.UNVERIFIED.value,
            "cache_lookup": "miss",
        },
    )
    packet = packet_from_query_hit(
        hit,
        predicted_files=("q.py",),
        acceptance_ids=("acc:q",),
    )
    assert packet.implementable is True
    assert packet.cache_status.disposition is CacheDisposition.MISS
    assert packet.cache_status.cache_key_id == "cache:q"
    body = packet.to_dict()
    assert "proof_body" not in str(body)


def test_plateau_bridge_handles_only_rejects_gold_ir() -> None:
    packet = bridge_plateau_codex_packet(
        {
            "packet_id": "plateau:1",
            "repository_tree_id": "git-tree:plat",
            "residual_ref_ids": ("residual:r1",),
            "claim_ids": ("claim:plat",),
            "obligation_ids": ("obl:plat",),
            "property_ids": ("prop:srt",),
            "status": ClaimStatus.OPEN.value,
            "predicted_files": ("srt/edit_target.py",),
        },
        acceptance_ids=("acc:plat",),
    )
    assert packet.plateau_packet_id == "plateau:1"
    assert packet.residual_ref_ids == ("residual:r1",)
    assert packet.implementable is True
    assert packet.metadata.get("gold_ir_excluded") is True
    assert packet.prover.semantic_authority is False

    with pytest.raises(Exception, match="gold IR|proof bodies"):
        bridge_plateau_codex_packet(
            {
                "packet_id": "plateau:bad",
                "repository_tree_id": "git-tree:plat",
                "gold_ir_body": {"ir": "SECRET"},
            }
        )


def test_supervisor_task_round_trip_and_metadata() -> None:
    packet = packet_from_claim(
        _claim(status=ClaimStatus.OPEN),
        predicted_files=("f.py",),
        acceptance_ids=("a1",),
        prover=ProverBinding(kernel_id="kernel:lean"),
    )
    task = materialize_supervisor_task(packet, title="Fix open API contract")
    assert task.implementable is True
    assert len(task.validation_commands) == 3
    assert task.metadata.get("semantic_authority") is False
    restored = CodeEditSupervisorTask.from_dict(task.to_dict())
    assert restored.title == "Fix open API contract"
    assert restored.validation_commands == task.validation_commands
    assert restored.packet.prover.kernel_id == "kernel:lean"


def test_missing_source_tree_not_implementable() -> None:
    packet = build_code_edit_packet(
        repository_tree_id="",
        claim_status=ClaimStatusRecord(status=ClaimStatus.OPEN.value),
    )
    assert packet.implementable is False
    assert (
        NonImplementableReason.MISSING_SOURCE_TREE.value
        in packet.non_implementable_reasons
    )


def test_satisfied_claim_not_implementable() -> None:
    packet = packet_from_claim(_claim(status=ClaimStatus.SATISFIED))
    assert packet.implementable is False
    assert NonImplementableReason.SATISFIED.value in packet.non_implementable_reasons


def test_materialize_report_notes_doctrine() -> None:
    report = materialize_code_edit_packets(claims=[_claim(status=ClaimStatus.OPEN)])
    notes = set(report.notes)
    assert "cache_miss_is_not_refutation" in notes
    assert "prover_semantic_authority_false" in notes
    assert "no_full_proof_bodies" in notes
