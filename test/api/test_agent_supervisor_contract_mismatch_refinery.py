"""SCA-101 generated contract repair-board tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.contract_mismatch_analyzer import (
    ContractFinding,
    ContractMismatchAnalyzer,
    FindingEvidence,
    MismatchState,
)
from ipfs_accelerate_py.agent_supervisor.analysis.mcp_contract_analysis import (
    ContractCounterexample,
    ContractParityClaim,
    ParityState,
)
from ipfs_accelerate_py.agent_supervisor.analysis.mcp_contract_catalog import (
    McpClaimFamily,
)
from ipfs_accelerate_py.agent_supervisor.objectives.contract_mismatch_refinery import (
    ContractMismatchRefinery,
    ContractMismatchRefineryError,
    ContractMismatchRefineryPolicy,
    ContractMismatchRefineryReason,
    ContractRepairTask,
    deterministic_repair_task_id,
    main,
    parse_contract_repair_board,
    render_contract_repair_board,
)
from ipfs_accelerate_py.agent_supervisor.proof.mcp_contract_edit_packet import (
    ExpansionHandle,
    McpContractEditPacket,
    materialize_contract_edit_packet,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    parse_task_file,
)


ACCELERATOR_PATH = (
    "external/ipfs_accelerate/ipfs_accelerate_py/mcp/dispatch.py"
)
TEST_PATH = "external/ipfs_accelerate/test/api/test_contract_dispatch.py"


def _finding(
    *,
    snapshot_id: str = "git-tree:current",
    actual: object = "integer",
    path: str = ACCELERATOR_PATH,
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
                actual=actual,
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


def _packet(
    finding: ContractFinding | None = None,
    **changes: object,
) -> McpContractEditPacket:
    selected = finding or _finding()
    read_paths = (
        selected.affected_paths
        if selected.affected_paths != (ACCELERATOR_PATH,)
        else (ACCELERATOR_PATH, TEST_PATH)
    )
    arguments: dict[str, object] = {
        "current_snapshot_id": selected.snapshot_id,
        "task_id": "SCA-100-fixture",
        "expected_postcondition": {
            "operation_id": "repo.inspect",
            "condition": "declared and executed argument types agree",
        },
        "validation_commands": (
            "python -m pytest test_contract.py -q",
        ),
        "reproof_commands": (
            "python -m ipfs_accelerate_py.agent_supervisor.proof.recheck "
            "obligation:arguments",
        ),
        "read_paths": read_paths,
        "write_paths": selected.affected_paths,
        "dependency_ids": ("SCA-090", "SCA-100"),
        "mandatory_dependency_ids": ("SCA-090", "SCA-100"),
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


def _reason_values(result: object) -> list[str]:
    return [item.reason_code.value for item in result.decisions]


def test_emits_targeted_exact_scope_task_and_round_trips_board() -> None:
    packet = _packet()
    refinery = ContractMismatchRefinery(
        ContractMismatchRefineryPolicy(cooldown_seconds=0)
    )

    first = refinery.refine(
        (packet,),
        current_snapshot_id=packet.snapshot_id,
        now_epoch=100,
    )
    again = refinery.refine(
        (packet,),
        current_snapshot_id=packet.snapshot_id,
        now_epoch=100,
    )

    assert first.markdown == again.markdown
    assert first.generated_count == 1
    assert first.final_open_work == 1
    assert first.safe_for_completion_reasoning is False
    task = first.tasks[0]
    assert task.task_id == deterministic_repair_task_id(packet.finding_id)
    assert task.write_paths == (ACCELERATOR_PATH,)
    assert task.affected_paths == task.write_paths
    assert set(task.write_paths).issubset(task.read_paths)
    assert task.contract_ids == ("contract:repo.inspect",)
    assert task.obligation_ids == ("obligation:arguments",)
    assert task.validation_commands and task.reproof_commands
    assert task.reproduction == task.validation_commands
    assert task.expansion_handles[0]["body_embedded"] is False
    assert "review source" not in task.title.casefold()
    assert "scan" not in task.title.casefold()
    assert task.can_certify_completion is False
    assert "- Completion authoritative: false" in first.markdown
    assert "- Generated completion evidence: none" in first.markdown
    assert parse_contract_repair_board(first.markdown).tasks == first.tasks


def test_deterministic_identity_deduplicates_and_updates_evidence_revision() -> None:
    finding = _finding()
    packet = _packet(finding)
    first = ContractMismatchRefinery().refine(
        (packet, packet),
        current_snapshot_id=finding.snapshot_id,
        now_epoch=1_000,
    )
    assert len(first.tasks) == 1

    revision = FindingEvidence(
        claim_id=finding.reproduction.claim_id,
        state=MismatchState.REFUTED,
        reason_codes=("argument_type_changed", "new_receipt"),
        premise_ids=("premise:descriptor",),
        obligation_ids=("obligation:arguments",),
        evidence_ids=("evidence:new",),
    )
    revised_finding = replace(
        finding,
        evidence=(*finding.evidence, revision),
    )
    revised_packet = _packet(revised_finding)
    updated = ContractMismatchRefinery().refine(
        (revised_packet,),
        current_snapshot_id=finding.snapshot_id,
        existing_board=first.markdown,
        now_epoch=1_001,
    )

    assert len(updated.tasks) == 1
    assert updated.tasks[0].task_id == first.tasks[0].task_id
    assert set(updated.tasks[0].evidence_record_ids) == {
        packet.finding_record_id,
        revised_packet.finding_record_id,
    }
    assert ContractMismatchRefineryReason.EVIDENCE_UPDATED.value in _reason_values(
        updated
    )

    completed_board = first.markdown.replace(
        "- Status: todo", "- Status: completed", 1
    )
    invalidated_completion = ContractMismatchRefinery().refine(
        (revised_packet,),
        current_snapshot_id=finding.snapshot_id,
        existing_board=completed_board,
        now_epoch=1_001,
    )
    assert invalidated_completion.tasks[0].status == "blocked"
    assert invalidated_completion.tasks[0].blocked_reason == "stale_finding"
    assert _reason_values(invalidated_completion) == ["stale_finding"]


def test_stale_packet_blocks_existing_task_and_never_creates_one() -> None:
    packet = _packet()
    refinery = ContractMismatchRefinery()
    current = refinery.refine(
        (packet,),
        current_snapshot_id=packet.snapshot_id,
        now_epoch=10,
    )

    stale = refinery.refine(
        (packet,),
        current_snapshot_id="git-tree:new",
        existing_board=current.markdown,
        now_epoch=20,
    )
    without_existing = refinery.refine(
        (packet,),
        current_snapshot_id="git-tree:new",
        now_epoch=20,
    )

    assert stale.tasks[0].status == "blocked"
    assert stale.tasks[0].blocked_reason == "stale_finding"
    assert stale.generated_count == 0
    assert without_existing.tasks == ()
    assert _reason_values(stale) == ["stale_finding"]


def test_snapshot_advance_blocks_old_work_before_emitting_current_identity() -> None:
    old_packet = _packet(_finding(snapshot_id="git-tree:old"))
    refinery = ContractMismatchRefinery(
        ContractMismatchRefineryPolicy(cooldown_seconds=0)
    )
    old_board = refinery.refine(
        (old_packet,),
        current_snapshot_id="git-tree:old",
        now_epoch=10,
    )
    current_packet = _packet(_finding(snapshot_id="git-tree:current"))

    advanced = refinery.refine(
        (current_packet,),
        current_snapshot_id="git-tree:current",
        existing_board=old_board.markdown,
        now_epoch=20,
    )

    assert len(advanced.tasks) == 2
    assert sorted(task.status for task in advanced.tasks) == ["blocked", "todo"]
    blocked = next(task for task in advanced.tasks if task.status == "blocked")
    assert blocked.finding_id == old_packet.finding_id
    assert blocked.blocked_reason == "stale_finding"
    assert sorted(_reason_values(advanced)) == ["emitted", "stale_finding"]


def test_open_work_finding_and_cooldown_bounds_are_independent() -> None:
    one = _packet(_finding(actual="integer"))
    two = _packet(_finding(actual="boolean"))

    finding_bound = ContractMismatchRefinery(
        ContractMismatchRefineryPolicy(
            max_open_work=10,
            max_findings_per_run=1,
            cooldown_seconds=0,
        )
    ).refine(
        (two, one),
        current_snapshot_id="git-tree:current",
        now_epoch=100,
    )
    assert finding_bound.generated_count == 1
    assert _reason_values(finding_bound).count("finding_limit") == 1

    open_bound = ContractMismatchRefinery(
        ContractMismatchRefineryPolicy(
            max_open_work=1,
            max_findings_per_run=10,
            cooldown_seconds=0,
        )
    ).refine(
        (one, two),
        current_snapshot_id="git-tree:current",
        current_open_work=1,
        now_epoch=100,
    )
    assert open_bound.generated_count == 0
    assert _reason_values(open_bound) == ["open_work_limit", "open_work_limit"]

    initial = ContractMismatchRefinery(
        ContractMismatchRefineryPolicy(cooldown_seconds=60)
    ).refine(
        (one,),
        current_snapshot_id="git-tree:current",
        now_epoch=100,
    )
    cooldown = ContractMismatchRefinery(
        ContractMismatchRefineryPolicy(cooldown_seconds=60)
    ).refine(
        (one, two),
        current_snapshot_id="git-tree:current",
        existing_board=initial.markdown,
        now_epoch=101,
    )
    assert len(cooldown.tasks) == 1
    assert "cooldown" in _reason_values(cooldown)
    # Existing evidence is still observed during cooldown.
    assert any(
        reason in _reason_values(cooldown)
        for reason in ("duplicate", "evidence_updated")
    )


@pytest.mark.parametrize(
    ("packet", "reason"),
    (
        (
            lambda: _packet(
                _finding(path="external/ipfs_kit/ipfs_kit_py/mcp/dispatch.py")
            ),
            ContractMismatchRefineryReason.OWNER_MISMATCH,
        ),
        (
            lambda: _packet(
                dependency_ids=("SCA-090", "SCA-100", "bad dependency")
            ),
            ContractMismatchRefineryReason.MALFORMED_DEPENDENCY,
        ),
    ),
)
def test_non_accelerator_paths_and_malformed_dependencies_are_rejected(
    packet: object,
    reason: ContractMismatchRefineryReason,
) -> None:
    selected = packet()
    result = ContractMismatchRefinery().refine(
        (selected,),
        current_snapshot_id=selected.snapshot_id,
        now_epoch=10,
    )

    assert result.tasks == ()
    assert result.decisions[0].reason_code is reason


def test_self_dependency_is_rejected_at_projection_boundary() -> None:
    finding = _finding()
    task_id = deterministic_repair_task_id(finding.finding_id)
    packet = _packet(
        finding,
        dependency_ids=(task_id,),
        mandatory_dependency_ids=(task_id,),
    )
    result = ContractMismatchRefinery().refine(
        (packet,),
        current_snapshot_id=packet.snapshot_id,
    )

    assert result.tasks == ()
    assert _reason_values(result) == ["self_dependency"]


def test_persisted_task_cannot_grant_itself_completion_authority() -> None:
    packet = _packet()
    result = ContractMismatchRefinery().refine(
        (packet,),
        current_snapshot_id=packet.snapshot_id,
    )
    payload = result.tasks[0].to_dict()
    payload["completion_authoritative"] = True

    with pytest.raises(
        ContractMismatchRefineryError, match="completion authority"
    ):
        ContractRepairTask.from_dict(payload)

    # A visible status change does not alter the machine authority contract.
    externally_completed = result.markdown.replace(
        "- Status: todo", "- Status: completed", 1
    )
    parsed = parse_contract_repair_board(externally_completed)
    assert parsed.tasks[0].status == "completed"
    assert parsed.tasks[0].can_certify_completion is False


def test_empty_generated_board_is_valid_and_non_authoritative() -> None:
    board = render_contract_repair_board(())
    parsed = parse_contract_repair_board(board)

    assert parsed.tasks == ()
    assert "admitted CodeEditPacket@1 records only" in board
    assert "Generated evidence authoritative: false" in board


def test_cli_derives_snapshot_from_empty_content_addressed_findings(
    tmp_path: Path,
) -> None:
    findings = tmp_path / "findings.json"
    output = tmp_path / "generated.todo.md"
    findings.write_text(
        json.dumps(
            {
                "schema": "ipfs_accelerate_py/agent-supervisor/"
                "contract-mismatch-analysis@1",
                "snapshot_id": "sca-repository-snapshot:sha256:current",
                "findings": [],
            }
        ),
        encoding="utf-8",
    )

    assert main(["--findings", str(findings), "--output", str(output)]) == 0
    assert parse_contract_repair_board(output.read_text(encoding="utf-8")).tasks == ()


def test_cli_rejects_missing_or_conflicting_inferred_snapshot(
    tmp_path: Path,
) -> None:
    output = tmp_path / "generated.todo.md"
    missing = tmp_path / "missing.json"
    missing.write_text(json.dumps({"findings": []}), encoding="utf-8")
    with pytest.raises(SystemExit, match="2"):
        main(["--findings", str(missing), "--output", str(output)])

    conflicting = tmp_path / "conflicting.json"
    conflicting.write_text(
        json.dumps(
            {
                "snapshot_id": "git-tree:document",
                "findings": [{"snapshot_id": "git-tree:record"}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="2"):
        main(["--findings", str(conflicting), "--output", str(output)])


def test_emitted_board_is_consumable_by_existing_markdown_task_source(
    tmp_path: Path,
) -> None:
    packet = _packet()
    result = ContractMismatchRefinery().refine(
        (packet,),
        current_snapshot_id=packet.snapshot_id,
    )
    path = tmp_path / "generated.todo.md"
    path.write_text(result.markdown, encoding="utf-8")

    parsed = parse_task_file(path, task_header_prefix="## SCA-REPAIR-")

    assert len(parsed) == 1
    assert parsed[0].task_id == result.tasks[0].task_id
    assert tuple(parsed[0].outputs) == result.tasks[0].write_paths
    assert tuple(parsed[0].depends_on) == result.tasks[0].dependency_ids
    assert parsed[0].metadata["completion authoritative"] == "false"
