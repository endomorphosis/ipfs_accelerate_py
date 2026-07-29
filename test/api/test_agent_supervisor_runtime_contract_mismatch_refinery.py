"""SCA-178 runtime contract repair-board projection tests."""

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
    ContractMismatchRefineryPolicy,
    ContractRepairTask,
    deterministic_repair_task_id,
    parse_contract_repair_board,
)
from ipfs_accelerate_py.agent_supervisor.objectives.runtime_contract_mismatch_refinery import (
    DEFAULT_RUNTIME_GOAL_ID,
    RUNTIME_CONTRACT_MISMATCH_REFINERY_INTERFACE,
    RUNTIME_CONTRACT_MISMATCH_TRIAGE_SCHEMA,
    RuntimeContractMismatchRefinery,
    RuntimeContractMismatchRefineryPolicy,
    RuntimeContractMismatchRefineryReason,
    build_runtime_contract_mismatch_triage,
    main,
    refine_runtime_contract_mismatch_packets,
    render_runtime_contract_repair_board,
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
    "external/ipfs_accelerate/ipfs_accelerate_py/agent_supervisor/runtime.py"
)
TEST_PATH = (
    "external/ipfs_accelerate/test/api/"
    "test_agent_supervisor_runtime_contract_mismatch_refinery.py"
)
BASELINE_ROOT = Path(
    "data/agent_supervisor/swissknife_contract_assurance/baseline"
)
GENERATED_BOARD = Path(
    "data/agent_supervisor/swissknife_contract_assurance/generated/"
    "ipfs_accelerate_contract_repairs.todo.md"
)


def _finding(
    *,
    snapshot_id: str = "git-tree:runtime-current",
    actual: object = "direct_dispatch",
    path: str = ACCELERATOR_PATH,
    reason_code: str = "direct_dispatch",
) -> ContractFinding:
    claim = ContractParityClaim(
        family=McpClaimFamily.POLICY_BEFORE_EFFECT,
        state=ParityState.REFUTED,
        operation_id="runtime.tools_dispatch",
        premise_ids=("premise:route", "premise:policy"),
        reason_codes=(reason_code,),
        counterexamples=(
            ContractCounterexample(
                reason_code=reason_code,
                boundary_id="tools/call",
                path="runtime.route",
                expected="guarded_mediated_path",
                actual=actual,
                source_ids=("source:runtime-trace",),
            ),
        ),
    )
    findings = ContractMismatchAnalyzer().analyze_claim(
        claim,
        snapshot_id=snapshot_id,
        contract_id="contract:runtime.tools_dispatch",
        affected_symbols=(
            "handler:runtime.tools_dispatch",
            "route:tools_dispatch",
        ),
        affected_paths=(path,),
        obligation_ids=("obligation:runtime-policy",),
        cas_handles=("bafy:runtime-contract-slice",),
        reproduction_commands=(
            "python -m pytest "
            "external/ipfs_accelerate/test/api/"
            "test_agent_supervisor_runtime_contract_mismatch_refinery.py -q",
        ),
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
        "task_id": "SCA-178-fixture",
        "expected_postcondition": {
            "operation_id": "runtime.tools_dispatch",
            "condition": (
                "runtime dispatch closes through the reviewed mediated path"
            ),
        },
        "validation_commands": (
            "python -m pytest "
            "external/ipfs_accelerate/test/api/"
            "test_agent_supervisor_runtime_contract_mismatch_refinery.py -q",
        ),
        "reproof_commands": (
            "python -m ipfs_accelerate_py.agent_supervisor.proof.recheck "
            "obligation:runtime-policy",
        ),
        "read_paths": read_paths,
        "write_paths": selected.affected_paths,
        "dependency_ids": ("SCA-177", "SCA-178"),
        "mandatory_dependency_ids": ("SCA-177", "SCA-178"),
        "expansion_handles": (
            ExpansionHandle(
                handle_id="proof:runtime-policy",
                kind="proof_receipt",
                content_id="bafy:runtime-proof-receipt",
                byte_count=32_000,
            ),
        ),
    }
    arguments.update(changes)
    return materialize_contract_edit_packet(selected, **arguments)


def _reason_values(result: object) -> list[str]:
    return [item.reason_code.value for item in result.decisions]


def test_one_impact_cluster_yields_one_exact_scope_runtime_task() -> None:
    packet = _packet()
    refinery = RuntimeContractMismatchRefinery(
        RuntimeContractMismatchRefineryPolicy(cooldown_seconds=0)
    )

    first = refinery.refine(
        (packet, packet),
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
    assert len(first.tasks) == 1
    task = first.tasks[0]
    assert task.task_id == deterministic_repair_task_id(packet.finding_id)
    assert task.goal_id == DEFAULT_RUNTIME_GOAL_ID
    assert task.write_paths == (ACCELERATOR_PATH,)
    assert task.affected_paths == task.write_paths
    assert set(task.write_paths).issubset(task.read_paths)
    assert task.contract_ids == ("contract:runtime.tools_dispatch",)
    assert task.obligation_ids == ("obligation:runtime-policy",)
    assert task.affected_symbols
    assert task.validation_commands and task.reproof_commands
    assert task.reproduction == task.validation_commands
    assert task.expected_postcondition
    assert task.expansion_handles[0]["body_embedded"] is False
    assert "no repository corpus" in first.markdown
    assert "review source" not in task.title.casefold()
    assert task.can_certify_completion is False
    assert "- Completion authoritative: false" in first.markdown
    assert "- Track: runtime-repair" in first.markdown
    assert RUNTIME_CONTRACT_MISMATCH_REFINERY_INTERFACE in first.markdown
    assert "no repository corpus" in first.markdown
    assert parse_contract_repair_board(first.markdown).tasks == first.tasks


def test_deterministic_identity_deduplicates_and_updates_evidence() -> None:
    finding = _finding()
    packet = _packet(finding)
    first = RuntimeContractMismatchRefinery().refine(
        (packet, packet),
        current_snapshot_id=finding.snapshot_id,
        now_epoch=1_000,
    )
    assert len(first.tasks) == 1

    revision = FindingEvidence(
        claim_id=finding.reproduction.claim_id,
        state=MismatchState.REFUTED,
        reason_codes=("direct_dispatch", "new_runtime_receipt"),
        premise_ids=("premise:route",),
        obligation_ids=("obligation:runtime-policy",),
        evidence_ids=("evidence:runtime-new",),
    )
    revised_finding = replace(finding, evidence=(*finding.evidence, revision))
    revised_packet = _packet(revised_finding)
    updated = RuntimeContractMismatchRefinery().refine(
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
    assert (
        RuntimeContractMismatchRefineryReason.EVIDENCE_UPDATED.value
        in _reason_values(updated)
    )


def test_stale_packet_blocks_existing_and_never_creates_work() -> None:
    packet = _packet()
    refinery = RuntimeContractMismatchRefinery()
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


def test_appends_to_existing_baseline_board_without_dropping_history() -> None:
    baseline_packet = _packet(
        _finding(actual="baseline-actual", reason_code="argument_type_changed")
    )
    # Force a distinct finding identity via a different counterexample actual.
    runtime_packet = _packet(
        _finding(actual="runtime-bypass", reason_code="policy_bypass")
    )
    assert baseline_packet.finding_id != runtime_packet.finding_id

    baseline = ContractMismatchRefinery(
        ContractMismatchRefineryPolicy(
            cooldown_seconds=0,
            goal_id="SCA-G101",
        )
    ).refine(
        (baseline_packet,),
        current_snapshot_id=baseline_packet.snapshot_id,
        now_epoch=50,
    )
    assert len(baseline.tasks) == 1
    assert baseline.tasks[0].goal_id == "SCA-G101"

    runtime = RuntimeContractMismatchRefinery(
        RuntimeContractMismatchRefineryPolicy(cooldown_seconds=0)
    ).refine(
        (runtime_packet,),
        current_snapshot_id=runtime_packet.snapshot_id,
        existing_board=baseline.markdown,
        now_epoch=60,
    )

    assert len(runtime.tasks) == 2
    by_finding = {task.finding_id: task for task in runtime.tasks}
    assert by_finding[baseline_packet.finding_id].goal_id == "SCA-G101"
    assert by_finding[runtime_packet.finding_id].goal_id == DEFAULT_RUNTIME_GOAL_ID
    assert "- Track: contract-repair" in runtime.markdown
    assert "- Track: runtime-repair" in runtime.markdown
    assert runtime.generated_count == 1


def test_open_work_finding_and_cooldown_bounds_are_independent() -> None:
    one = _packet(_finding(actual="integer"))
    two = _packet(_finding(actual="boolean"))

    finding_bound = RuntimeContractMismatchRefinery(
        RuntimeContractMismatchRefineryPolicy(
            max_open_work=10,
            max_findings_per_run=1,
            cooldown_seconds=0,
        )
    ).refine(
        (two, one),
        current_snapshot_id="git-tree:runtime-current",
        now_epoch=100,
    )
    assert finding_bound.generated_count == 1
    assert _reason_values(finding_bound).count("finding_limit") == 1

    open_bound = RuntimeContractMismatchRefinery(
        RuntimeContractMismatchRefineryPolicy(
            max_open_work=1,
            max_findings_per_run=10,
            cooldown_seconds=0,
        )
    ).refine(
        (one, two),
        current_snapshot_id="git-tree:runtime-current",
        current_open_work=1,
        now_epoch=100,
    )
    assert open_bound.generated_count == 0
    assert set(_reason_values(open_bound)) == {"open_work_limit"}


@pytest.mark.parametrize(
    ("packet", "reason"),
    (
        (
            lambda: _packet(
                _finding(
                    path="external/ipfs_kit/ipfs_kit_py/mcp/dispatch.py"
                )
            ),
            RuntimeContractMismatchRefineryReason.OWNER_MISMATCH,
        ),
        (
            lambda: _packet(
                dependency_ids=("SCA-177", "SCA-178", "bad dependency")
            ),
            RuntimeContractMismatchRefineryReason.MALFORMED_DEPENDENCY,
        ),
    ),
)
def test_non_accelerator_paths_and_malformed_dependencies_are_rejected(
    packet: object,
    reason: RuntimeContractMismatchRefineryReason,
) -> None:
    selected = packet()
    result = RuntimeContractMismatchRefinery().refine(
        (selected,),
        current_snapshot_id=selected.snapshot_id,
        now_epoch=10,
    )

    assert result.tasks == ()
    assert result.decisions[0].reason_code is reason


def test_unsupported_stale_and_unknown_only_are_not_implementation_ready() -> None:
    snapshot_id = "git-tree:runtime-current"
    unsupported = {
        "finding_id": "sha256:unsupported-runtime",
        "state": "unsupported",
        "reason_code": "repository_index_missing",
        "contract_id": "contract:runtime-health",
        "affected_paths": ["external/ipfs_accelerate/ipfs_accelerate_py"],
        "counterexample": {"kind": "repository_index_missing"},
        "snapshot_root": snapshot_id,
    }
    unknown = {
        "finding_id": "sha256:unknown-runtime",
        "state": "unknown",
        "reason_code": "dynamic_unknown",
        "contract_id": "contract:runtime-dynamic",
        "affected_paths": ["external/ipfs_accelerate/ipfs_accelerate_py"],
        "counterexample": {"kind": "dynamic_unknown"},
        "snapshot_id": snapshot_id,
    }
    stale = {
        "finding_id": "sha256:stale-runtime",
        "state": "stale",
        "reason_code": "snapshot_advanced",
        "contract_id": "contract:runtime-stale",
        "affected_paths": ["external/ipfs_accelerate/ipfs_accelerate_py"],
        "counterexample": {"kind": "stale_replay"},
        "snapshot_id": "git-tree:old",
    }

    result = RuntimeContractMismatchRefinery().refine(
        (unsupported, unknown, stale),
        current_snapshot_id=snapshot_id,
        now_epoch=0,
    )

    assert result.tasks == ()
    assert result.generated_count == 0
    reasons = set(_reason_values(result))
    assert reasons == {
        "unsupported_finding",
        "unknown_only",
        "stale_finding",
    }
    assert all(
        "not implementation-ready" in decision.detail
        for decision in result.decisions
    )


def test_persisted_task_cannot_grant_itself_completion_authority() -> None:
    packet = _packet()
    result = RuntimeContractMismatchRefinery().refine(
        (packet,),
        current_snapshot_id=packet.snapshot_id,
    )
    payload = result.tasks[0].to_dict()
    payload["completion_authoritative"] = True

    from ipfs_accelerate_py.agent_supervisor.objectives.contract_mismatch_refinery import (
        ContractMismatchRefineryError,
    )

    with pytest.raises(ContractMismatchRefineryError, match="completion authority"):
        ContractRepairTask.from_dict(payload)

    assert result.tasks[0].can_certify_completion is False
    assert result.can_certify_completion is False


def test_empty_generated_board_is_valid_and_non_authoritative() -> None:
    board = render_runtime_contract_repair_board(())
    parsed = parse_contract_repair_board(board)

    assert parsed.tasks == ()
    assert "admitted CodeEditPacket@1 records only" in board
    assert "no repository corpus" in board
    assert "Generated evidence authoritative: false" in board
    assert RUNTIME_CONTRACT_MISMATCH_REFINERY_INTERFACE in board


def test_cli_triages_non_actionable_runtime_findings(
    tmp_path: Path,
) -> None:
    findings = tmp_path / "findings.json"
    output = tmp_path / "generated.todo.md"
    triage_output = tmp_path / "runtime_triage.json"
    snapshot_id = "sca-repository-snapshot:sha256:runtime-current"
    source_records = [
        {
            "affected_paths": ["swissknife"],
            "contract_id": "contract:runtime-analyzer-health",
            "counterexample": {"kind": "repository_index_missing"},
            "finding_id": "sha256:unsupported-runtime",
            "reason_code": "repository_index_missing",
            "snapshot_root": snapshot_id,
            "state": "unsupported",
        },
        {
            "affected_paths": ["swissknife"],
            "contract_id": "contract:runtime-dynamic",
            "counterexample": {"kind": "dynamic_unknown"},
            "finding_id": "sha256:unknown-runtime",
            "reason_code": "dynamic_unknown",
            "snapshot_id": snapshot_id,
            "state": "unknown",
        },
    ]
    findings.write_text(
        json.dumps(
            {
                "schema": (
                    "ipfs_accelerate_py/agent-supervisor/"
                    "sca-baseline-runtime-findings@1"
                ),
                "snapshot_root": snapshot_id,
                "findings": source_records,
            }
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "--findings",
                str(findings),
                "--output",
                str(output),
                "--triage-output",
                str(triage_output),
                "--now-epoch",
                "0",
            ]
        )
        == 0
    )
    assert parse_contract_repair_board(output.read_text(encoding="utf-8")).tasks == ()
    expected_result = RuntimeContractMismatchRefinery().refine(
        source_records,
        current_snapshot_id=snapshot_id,
        now_epoch=0,
    )
    triage = json.loads(triage_output.read_text(encoding="utf-8"))
    assert triage == build_runtime_contract_mismatch_triage(
        expected_result,
        current_snapshot_id=snapshot_id,
        owner="external/ipfs_accelerate",
        source_records=tuple(source_records),
    )
    assert triage["schema"] == RUNTIME_CONTRACT_MISMATCH_TRIAGE_SCHEMA
    assert triage["interface"] == RUNTIME_CONTRACT_MISMATCH_REFINERY_INTERFACE
    assert triage["generated_count"] == 0
    assert triage["completion_authoritative"] is False
    assert triage["llm_call_count"] == 0
    assert set(triage["reason_counts"]) == {
        "unsupported_finding",
        "unknown_only",
    }


def test_cli_does_not_downgrade_actionable_non_packet_to_unsupported(
    tmp_path: Path,
) -> None:
    findings = tmp_path / "findings.json"
    output = tmp_path / "generated.todo.md"
    findings.write_text(
        json.dumps(
            {
                "snapshot_root": "sca-repository-snapshot:sha256:current",
                "findings": [
                    {
                        "finding_id": "sha256:refuted-runtime",
                        "state": "refuted",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="2"):
        main(["--findings", str(findings), "--output", str(output)])


def test_emitted_board_is_consumable_by_existing_markdown_task_source(
    tmp_path: Path,
) -> None:
    packet = _packet()
    result = RuntimeContractMismatchRefinery().refine(
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


def test_functional_entry_point_matches_class_api() -> None:
    packet = _packet()
    via_fn = refine_runtime_contract_mismatch_packets(
        (packet,),
        current_snapshot_id=packet.snapshot_id,
        now_epoch=7,
        policy=RuntimeContractMismatchRefineryPolicy(cooldown_seconds=0),
    )
    via_cls = RuntimeContractMismatchRefinery(
        RuntimeContractMismatchRefineryPolicy(cooldown_seconds=0)
    ).refine(
        (packet,),
        current_snapshot_id=packet.snapshot_id,
        now_epoch=7,
    )
    assert via_fn.markdown == via_cls.markdown
    assert via_fn.tasks == via_cls.tasks


def test_baseline_runtime_artifacts_are_seeded_and_non_authoritative() -> None:
    """Committed SCA-178 artifacts must exist and stay non-authoritative."""

    triage_path = BASELINE_ROOT / "runtime_triage.json"
    assert GENERATED_BOARD.is_file()
    assert triage_path.is_file()

    board = GENERATED_BOARD.read_text(encoding="utf-8")
    triage = json.loads(triage_path.read_text(encoding="utf-8"))

    assert "Generated evidence authoritative: false" in board
    assert "admitted CodeEditPacket@1 records only" in board
    assert triage["schema"] == RUNTIME_CONTRACT_MISMATCH_TRIAGE_SCHEMA
    assert triage["interface"] == RUNTIME_CONTRACT_MISMATCH_REFINERY_INTERFACE
    assert triage["completion_authoritative"] is False
    assert triage["llm_call_count"] == 0
    assert triage["provider_call_count"] == 0
    assert triage["model_call_count"] == 0
    assert "triage_id" in triage
