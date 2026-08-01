"""SCA-110 supervisor contract-assurance refill tests."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.analysis.analyzer_health import (
    ANALYZER_CANARY_SCHEMA,
    ANALYZER_HEALTH_SCHEMA,
)
from ipfs_accelerate_py.agent_supervisor.analysis.contract_mismatch_analyzer import (
    ContractFinding,
    ContractMismatchAnalyzer,
)
from ipfs_accelerate_py.agent_supervisor.analysis.mcp_contract_analysis import (
    ContractCounterexample,
    ContractParityClaim,
    ParityState,
)
from ipfs_accelerate_py.agent_supervisor.analysis.mcp_contract_catalog import (
    McpClaimFamily,
)
from ipfs_accelerate_py.agent_supervisor.objectives.contract_assurance_refill import (
    ContractAnalyzerCapability,
    ContractAssuranceAnalysis,
    ContractAssuranceFinding,
    ContractAssuranceGoalLineage,
    ContractAssuranceRefill,
    ContractAssuranceRefillPolicy,
    ContractAssuranceRefillReason,
)
from ipfs_accelerate_py.agent_supervisor.objectives.contract_mismatch_refinery import (
    parse_contract_repair_board,
)
from ipfs_accelerate_py.agent_supervisor.proof.mcp_contract_edit_packet import (
    ExpansionHandle,
    McpContractEditPacket,
    materialize_contract_edit_packet,
)


SNAPSHOT = "git-tree:current"
REPOSITORY = "repository:swissknife"
TREE = "tree:current"
OBJECTIVE_REVISION = "objective:current"
ANALYZER_VERSION = "contract-analyzer/v1"
ACCELERATOR_PATH = (
    "external/ipfs_accelerate/ipfs_accelerate_py/mcp/dispatch.py"
)
TEST_PATH = "external/ipfs_accelerate/test/api/test_contract_dispatch.py"


def _finding(*, actual: object = "integer") -> ContractFinding:
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
        snapshot_id=SNAPSHOT,
        contract_id="contract:repo.inspect",
        affected_symbols=("handler:repo.inspect", "schema:repo.inspect"),
        affected_paths=(ACCELERATOR_PATH,),
        obligation_ids=("obligation:arguments",),
        cas_handles=("bafy:contract-slice",),
        reproduction_commands=("python -m pytest test_contract.py -q",),
    )
    assert len(findings) == 1
    return findings[0]


def _packet(*, actual: object = "integer") -> McpContractEditPacket:
    finding = _finding(actual=actual)
    return materialize_contract_edit_packet(
        finding,
        current_snapshot_id=SNAPSHOT,
        task_id="SCA-100-fixture",
        expected_postcondition={
            "operation_id": "repo.inspect",
            "condition": "declared and executed argument types agree",
        },
        validation_commands=("python -m pytest test_contract.py -q",),
        reproof_commands=(
            "python -m ipfs_accelerate_py.agent_supervisor.proof.recheck "
            "obligation:arguments",
        ),
        read_paths=(ACCELERATOR_PATH, TEST_PATH),
        write_paths=(ACCELERATOR_PATH,),
        dependency_ids=("SCA-090", "SCA-100"),
        mandatory_dependency_ids=("SCA-090", "SCA-100"),
        expansion_handles=(
            ExpansionHandle(
                handle_id="proof:arguments",
                kind="proof_receipt",
                content_id="bafy:proof-receipt",
                byte_count=32_000,
            ),
        ),
    )


def _lineage(**changes: object) -> ContractAssuranceGoalLineage:
    values: dict[str, object] = {
        "goal_id": "SCA-G101",
        "root_goal_id": "SCA-G000",
        "ancestor_goal_ids": ("SCA-G000", "SCA-G090"),
        "objective_revision": OBJECTIVE_REVISION,
    }
    values.update(changes)
    return ContractAssuranceGoalLineage(**values)


def _capability(**changes: object) -> ContractAnalyzerCapability:
    values: dict[str, object] = {
        "analyzer_id": "swissknife.contract-assurance",
        "analyzer_version": ANALYZER_VERSION,
        "capability_id": "capability:contract-v1",
        "repository_id": REPOSITORY,
        "tree_id": TREE,
        "snapshot_id": SNAPSHOT,
        "available": True,
        "supported_claim_families": ("ArgumentsPreserved",),
    }
    values.update(changes)
    return ContractAnalyzerCapability(**values)


def _health(*, healthy: bool = True) -> dict[str, object]:
    return {
        "schema": ANALYZER_HEALTH_SCHEMA,
        "status": "healthy" if healthy else "unhealthy",
        "healthy": healthy,
        "safe_for_completion_reasoning": healthy,
        "reasons": [] if healthy else ["canary_failure"],
        "thresholds": {},
        "metrics": {},
    }


def _canaries(*, passed: bool = True) -> dict[str, object]:
    return {
        "schema": ANALYZER_CANARY_SCHEMA,
        "analyzer_version": ANALYZER_VERSION,
        "registry_present": True,
        "registry_errors": [],
        "passed": passed,
        "fixture_count": 3,
        "failed_fixture_ids": [] if passed else ["descriptor-mismatch"],
        "results": [],
    }


def _coverage(*, complete: bool = True) -> dict[str, int]:
    if not complete:
        return {
            "tracked_file_count": 4,
            "eligible_file_count": 3,
            "excluded_file_count": 1,
            "parsed_file_count": 2,
            "cache_hit_count": 0,
            "parser_failure_count": 1,
        }
    return {
        "tracked_file_count": 4,
        "eligible_file_count": 3,
        "excluded_file_count": 1,
        "parsed_file_count": 2,
        "cache_hit_count": 1,
        "parser_failure_count": 0,
    }


def _analysis(
    *packets: McpContractEditPacket,
    evidence_channel: str = "",
    exhaustive: bool = False,
    coverage_complete: bool = False,
    health: dict[str, object] | None = None,
    canaries: dict[str, object] | None = None,
    capability: ContractAnalyzerCapability | None = None,
    lineage: ContractAssuranceGoalLineage | None = None,
) -> ContractAssuranceAnalysis:
    return ContractAssuranceAnalysis(
        snapshot_id=SNAPSHOT,
        repository_id=REPOSITORY,
        tree_id=TREE,
        analyzer_version=ANALYZER_VERSION,
        capability=capability or _capability(),
        analyzer_health=health or _health(),
        canary_report=canaries or _canaries(),
        findings=tuple(
            ContractAssuranceFinding(packet, lineage or _lineage())
            for packet in packets
        ),
        coverage=_coverage(complete=coverage_complete),
        coverage_complete=coverage_complete,
        exhaustive=exhaustive,
        evidence_channel=evidence_channel,
        current_finding_record_ids={
            packet.finding_id: packet.finding_record_id for packet in packets
        },
    )


def _policy(**changes: object) -> ContractAssuranceRefillPolicy:
    values: dict[str, object] = {
        "min_open_tasks": 2,
        "max_open_tasks": 4,
        "max_findings_per_run": 2,
        "timeout_seconds": 5,
        "cooldown_seconds": 0,
        "required_exhaustion_members": 2,
        "expected_analyzer_version": ANALYZER_VERSION,
    }
    values.update(changes)
    return ContractAssuranceRefillPolicy(**values)


def _run(
    refill: ContractAssuranceRefill,
    *,
    key: str,
    open_tasks: int = 0,
    now_epoch: int = 100,
):
    return refill.refill(
        current_open_tasks=open_tasks,
        snapshot_id=SNAPSHOT,
        repository_id=REPOSITORY,
        tree_id=TREE,
        objective_revision=OBJECTIVE_REVISION,
        idempotency_key=key,
        now_epoch=now_epoch,
    )


def test_low_backlog_only_triggers_current_healthy_goal_backed_analysis(
    tmp_path: Path,
) -> None:
    calls = 0
    packet = _packet()

    def analyzer(_request):
        nonlocal calls
        calls += 1
        return _analysis(packet)

    refill = ContractAssuranceRefill(
        analyzer, state_path=tmp_path / "refill.json", policy=_policy()
    )
    skipped = _run(refill, key="above-threshold", open_tasks=2)
    generated = _run(refill, key="below-threshold", open_tasks=1, now_epoch=101)

    assert skipped.reason is ContractAssuranceRefillReason.THRESHOLD_SATISFIED
    assert calls == 1
    assert generated.reason is ContractAssuranceRefillReason.GENERATED
    assert generated.generated_count == 1
    assert generated.safe_for_completion_reasoning is False
    assert generated.completion_authoritative is False
    task = generated.tasks[0]
    assert task.finding_id == packet.finding_id
    assert task.snapshot_id == SNAPSHOT
    assert task.goal_id == "SCA-G101"
    assert task.can_certify_completion is False
    assert parse_contract_repair_board(generated.board_markdown).tasks == (
        task,
    )


def test_missing_current_capability_health_or_lineage_fails_closed(
    tmp_path: Path,
) -> None:
    packet = _packet()
    analyses = iter(
        (
            _analysis(packet, capability=_capability(snapshot_id="git-tree:old")),
            _analysis(packet, health=_health(healthy=False)),
            _analysis(
                packet,
                lineage=_lineage(objective_revision="objective:stale"),
            ),
        )
    )
    refill = ContractAssuranceRefill(
        lambda _request: next(analyses),
        state_path=tmp_path / "refill.json",
        policy=_policy(),
    )

    stale = _run(refill, key="stale", now_epoch=100)
    unhealthy = _run(refill, key="unhealthy", now_epoch=101)
    unbacked = _run(refill, key="unbacked", now_epoch=102)

    assert stale.reason is ContractAssuranceRefillReason.CAPABILITY_STALE
    assert unhealthy.reason is ContractAssuranceRefillReason.ANALYZER_UNHEALTHY
    assert unbacked.reason is ContractAssuranceRefillReason.NO_GOAL_LINEAGE
    assert not stale.tasks and not unhealthy.tasks and not unbacked.tasks
    assert not parse_contract_repair_board(unbacked.board_markdown).tasks


def test_duplicate_evidence_and_bounds_do_not_create_task_storms(
    tmp_path: Path,
) -> None:
    one = _packet(actual="integer")
    two = _packet(actual="boolean")
    three = _packet(actual="array")
    analyses = iter(
        (
            _analysis(one, one),
            _analysis(one),
            _analysis(three, two, one),
        )
    )
    refill = ContractAssuranceRefill(
        lambda _request: next(analyses),
        state_path=tmp_path / "refill.json",
        policy=_policy(max_findings_per_run=2, max_open_tasks=2),
    )

    first = _run(refill, key="first", now_epoch=100)
    duplicate = _run(refill, key="duplicate", now_epoch=101)
    bounded = _run(refill, key="bounded", now_epoch=102)

    assert first.generated_count == 1
    assert duplicate.reason is ContractAssuranceRefillReason.DUPLICATE_ONLY
    assert duplicate.generated_count == 0
    board = parse_contract_repair_board(bounded.board_markdown)
    assert len(board.tasks) == 2
    assert len({task.finding_id for task in board.tasks}) == 2
    assert ContractAssuranceRefillReason.FINDING_LIMIT.value in bounded.reason_codes


def test_explicit_cycle_replays_exact_result_after_restart(tmp_path: Path) -> None:
    calls = 0

    def analyzer(_request):
        nonlocal calls
        calls += 1
        return _analysis(_packet())

    state_path = tmp_path / "refill.json"
    first = _run(
        ContractAssuranceRefill(analyzer, state_path=state_path, policy=_policy()),
        key="cycle:one",
    )
    restarted = ContractAssuranceRefill(
        analyzer, state_path=state_path, policy=_policy()
    )
    replay = _run(restarted, key="cycle:one", now_epoch=999)

    assert calls == 1
    assert replay.replayed is True
    assert replay.reason is first.reason
    assert replay.tasks == first.tasks
    assert replay.board_markdown == first.board_markdown
    assert replay.scan_result.receipt_cid == first.scan_result.receipt_cid


def test_corrupt_latest_state_recovers_last_good_cycle_and_quarantines_it(
    tmp_path: Path,
) -> None:
    calls = 0

    def analyzer(_request):
        nonlocal calls
        calls += 1
        return _analysis(_packet())

    state_path = tmp_path / "refill.json"
    refill = ContractAssuranceRefill(
        analyzer, state_path=state_path, policy=_policy()
    )
    original = _run(refill, key="cycle:recover", now_epoch=100)
    # The second atomic transaction promotes the first state to .bak.
    _run(refill, key="threshold", open_tasks=2, now_epoch=101)
    state_path.write_text('{"schema":', encoding="utf-8")

    recovered = _run(
        ContractAssuranceRefill(analyzer, state_path=state_path, policy=_policy()),
        key="cycle:recover",
        now_epoch=102,
    )

    assert calls == 1
    assert recovered.replayed is True
    assert recovered.recovered_state is True
    assert recovered.tasks == original.tasks
    assert ContractAssuranceRefillReason.STATE_RECOVERED.value in (
        recovered.reason_codes
    )
    assert list(tmp_path.glob("refill.json.corrupt-*"))


def test_empty_scan_is_not_exhaustion_without_full_coverage_and_canaries(
    tmp_path: Path,
) -> None:
    analyses = iter(
        (
            _analysis(exhaustive=True, evidence_channel="static"),
            _analysis(
                exhaustive=True,
                evidence_channel="static",
                coverage_complete=True,
                canaries=_canaries(passed=False),
            ),
        )
    )
    refill = ContractAssuranceRefill(
        lambda _request: next(analyses),
        state_path=tmp_path / "refill.json",
        policy=_policy(),
    )

    incomplete = _run(refill, key="coverage", now_epoch=100)
    canary_failure = _run(refill, key="canary", now_epoch=101)

    assert incomplete.reason is ContractAssuranceRefillReason.COVERAGE_INCOMPLETE
    assert canary_failure.reason is ContractAssuranceRefillReason.CANARIES_FAILED
    assert incomplete.safe_for_completion_reasoning is False
    assert canary_failure.safe_for_completion_reasoning is False


def test_exhaustion_requires_independent_healthy_current_quorum(
    tmp_path: Path,
) -> None:
    channels = iter(("static", "audit"))

    def analyzer(_request):
        return _analysis(
            exhaustive=True,
            coverage_complete=True,
            evidence_channel=next(channels),
        )

    state_path = tmp_path / "refill.json"
    first = _run(
        ContractAssuranceRefill(analyzer, state_path=state_path, policy=_policy()),
        key="quorum:one",
        now_epoch=100,
    )
    second = _run(
        ContractAssuranceRefill(analyzer, state_path=state_path, policy=_policy()),
        key="quorum:two",
        now_epoch=101,
    )

    assert first.reason is ContractAssuranceRefillReason.QUORUM_INCOMPLETE
    assert first.quorum["member_count"] == 1
    assert first.safe_for_completion_reasoning is False
    assert second.reason is ContractAssuranceRefillReason.EXHAUSTED
    assert second.quorum["member_count"] == 2
    assert second.quorum["satisfied"] is True
    assert second.safe_for_completion_reasoning is True
    assert second.generated_count == 0
    assert second.completion_authoritative is False


def test_snapshot_change_invalidates_old_tasks_and_old_exhaustion_votes(
    tmp_path: Path,
) -> None:
    packet = _packet()
    analyses = iter(
        (
            _analysis(packet),
            replace(
                _analysis(
                    exhaustive=True,
                    coverage_complete=True,
                    evidence_channel="static",
                ),
                snapshot_id="git-tree:new",
                capability=_capability(snapshot_id="git-tree:new"),
            ),
        )
    )
    refill = ContractAssuranceRefill(
        lambda _request: next(analyses),
        state_path=tmp_path / "refill.json",
        policy=_policy(),
    )
    generated = _run(refill, key="old", now_epoch=100)
    assert generated.generated_count == 1

    changed = refill.refill(
        current_open_tasks=0,
        snapshot_id="git-tree:new",
        repository_id=REPOSITORY,
        tree_id=TREE,
        objective_revision=OBJECTIVE_REVISION,
        idempotency_key="new",
        now_epoch=101,
    )
    assert changed.reason is ContractAssuranceRefillReason.QUORUM_INCOMPLETE
    assert changed.quorum["member_count"] == 1
    board = parse_contract_repair_board(changed.board_markdown)
    assert board.tasks[0].status == "blocked"
    assert board.tasks[0].blocked_reason == "stale_finding"


def test_timeout_and_cooldown_are_typed_and_do_not_launch_task_storms(
    tmp_path: Path,
) -> None:
    slow_calls = 0

    def slow_analyzer(_request):
        nonlocal slow_calls
        slow_calls += 1
        time.sleep(0.03)
        return _analysis(_packet())

    timed_out = _run(
        ContractAssuranceRefill(
            slow_analyzer,
            state_path=tmp_path / "timeout.json",
            policy=_policy(timeout_seconds=0.005),
        ),
        key="timeout",
        now_epoch=100,
    )
    assert timed_out.reason is ContractAssuranceRefillReason.TIMED_OUT
    assert timed_out.scan_result.terminal_reason.value == "timed_out"
    assert timed_out.generated_count == 0

    calls = 0

    def analyzer(_request):
        nonlocal calls
        calls += 1
        return _analysis(_packet())

    refill = ContractAssuranceRefill(
        analyzer,
        state_path=tmp_path / "cooldown.json",
        policy=_policy(cooldown_seconds=10),
    )
    assert _run(refill, key="first", now_epoch=100).generated_count == 1
    cooldown = _run(refill, key="second", now_epoch=105)
    assert cooldown.reason is ContractAssuranceRefillReason.COOLDOWN
    assert calls == 1
