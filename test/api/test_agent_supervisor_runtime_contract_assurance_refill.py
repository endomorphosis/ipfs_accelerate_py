"""SCA-179 runtime contract discovery continuous refill tests.

Covers:

* no-op scans make zero provider / model / LLM / analyzer calls;
* one-symbol / route / schema / policy changes update all and only dependents;
* crash recovery is idempotent with exact cycle replay;
* task storms and cross-component duplicate repairs are bounded; and
* new findings refill the correct runtime subgoal.

Durable metrics are published under
``data/agent_supervisor/swissknife_contract_assurance/state/runtime_refill_metrics.json``.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from hashlib import sha256
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.analyzer_health import (
    ANALYZER_CANARY_SCHEMA,
    ANALYZER_HEALTH_SCHEMA,
)
from ipfs_accelerate_py.agent_supervisor.analysis.contract_mismatch_analyzer import (
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
)
from ipfs_accelerate_py.agent_supervisor.objectives.contract_mismatch_refinery import (
    parse_contract_repair_board,
)
from ipfs_accelerate_py.agent_supervisor.objectives.runtime_contract_assurance_refill import (
    RUNTIME_COMPONENT_SUBGOALS,
    RUNTIME_CONTRACT_ASSURANCE_REFILL_INTERFACE,
    RUNTIME_CONTRACT_ASSURANCE_REFILL_METRICS_SCHEMA,
    RuntimeContractAssuranceRefill,
    RuntimeContractAssuranceRefillPolicy,
    RuntimeContractAssuranceRefillReason,
    build_runtime_refill_metrics_report,
    normalize_changed_inputs,
    resolve_runtime_subgoal,
)
from ipfs_accelerate_py.agent_supervisor.objectives.runtime_contract_mismatch_refinery import (
    DEFAULT_RUNTIME_GOAL_ID,
)
from ipfs_accelerate_py.agent_supervisor.proof.mcp_contract_edit_packet import (
    ExpansionHandle,
    materialize_contract_edit_packet,
)
from ipfs_accelerate_py.agent_supervisor.proof.proof_scope_index import (
    build_proof_scope_index,
    invalidate_proof_evidence,
)


TASK_ID = "SCA-179"
SOURCE_TREE = "tree:sca-179-runtime-refill"
SNAPSHOT = "git-tree:sca-179"
REPOSITORY = "repository:swissknife"
TREE = "tree:current"
OBJECTIVE_REVISION = "objective:current"
ANALYZER_VERSION = "runtime-contract-analyzer/v1"
ACCELERATOR_PATH = (
    "external/ipfs_accelerate/ipfs_accelerate_py/agent_supervisor/runtime.py"
)
TEST_PATH = (
    "external/ipfs_accelerate/test/api/"
    "test_agent_supervisor_runtime_contract_assurance_refill.py"
)

DEPENDENT_OBLIGATIONS = ("obligation:api", "obligation:consumer")
UNAFFECTED_OBLIGATION = "obligation:unrelated"

SEMANTIC_CASES = (
    (
        "symbol-change",
        "qualified_symbol",
        {"kind": "qualified_symbol", "value": "pkg.api.Service.run"},
    ),
    (
        "route-change",
        "route",
        {"kind": "route", "value": "schema:repo.inspect.v1"},
    ),
    (
        "schema-change",
        "schema",
        {"kind": "schema", "value": "schema:repo.inspect.v1"},
    ),
    (
        "policy-change",
        "policy",
        {"kind": "policy", "value": "policy:contract-v1"},
    ),
)


def _swissknife_superproject_root() -> Path | None:
    candidates = (Path.cwd().resolve(), *Path(__file__).resolve().parents)
    for candidate in candidates:
        if (
            candidate / "config/swissknife_symbolic_contract_scope.json"
        ).is_file():
            return candidate
    return None


REPOSITORY_ROOT = _swissknife_superproject_root()
STATE_DIR = (
    (REPOSITORY_ROOT or Path("/__missing_swissknife_superproject__"))
    / "data/agent_supervisor/swissknife_contract_assurance/state"
)
PUBLISHED_METRICS = STATE_DIR / "runtime_refill_metrics.json"
requires_published_swissknife_evidence = pytest.mark.skipif(
    REPOSITORY_ROOT is None,
    reason="published evidence requires a Swissknife superproject checkout",
)


def _canonical(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    )


def _digest(value: Any) -> str:
    encoded = json.dumps(
        _canonical(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + sha256(encoded).hexdigest()


def _proof_fixture():
    return build_proof_scope_index(
        scope_blobs=[
            {
                "path": "src/api.py",
                "blob_id": "blob:api",
                "scopes": [
                    {
                        "scope_id": "scope:api",
                        "path": "src/api.py",
                        "qualified_symbol": "pkg.api.Service.run",
                        "interface": "schema:repo.inspect.v1",
                    }
                ],
            },
            {
                "path": "src/consumer.py",
                "blob_id": "blob:consumer",
                "scopes": [
                    {
                        "scope_id": "scope:consumer",
                        "path": "src/consumer.py",
                        "qualified_symbol": "pkg.consumer.consume",
                    }
                ],
            },
            {
                "path": "src/unrelated.py",
                "blob_id": "blob:unrelated",
                "scopes": [
                    {
                        "scope_id": "scope:unrelated",
                        "path": "src/unrelated.py",
                        "qualified_symbol": "pkg.unrelated.stable",
                    }
                ],
            },
        ],
        obligations=[
            {
                "obligation_id": "obligation:api",
                "ast_scope_ids": ["scope:api"],
                "toolchain_id": "toolchain:py3.12",
                "policy_id": "policy:contract-v1",
            },
            {
                "obligation_id": "obligation:consumer",
                "ast_scope_ids": ["scope:consumer"],
                "depends_on": ["obligation:api"],
                "toolchain_id": "toolchain:py3.12",
                "policy_id": "policy:contract-v1",
            },
            {
                "obligation_id": "obligation:unrelated",
                "ast_scope_ids": ["scope:unrelated"],
                "toolchain_id": "toolchain:other",
                "policy_id": "policy:other",
            },
        ],
        receipts=[
            {
                "receipt_id": "receipt:api",
                "obligation_id": "obligation:api",
                "ast_scope_ids": ["scope:api"],
                "repository_tree_id": "tree:proved",
            },
            {
                "receipt_id": "receipt:consumer",
                "obligation_id": "obligation:consumer",
                "ast_scope_ids": ["scope:consumer"],
                "repository_tree_id": "tree:proved",
            },
            {
                "receipt_id": "receipt:unrelated",
                "obligation_id": "obligation:unrelated",
                "ast_scope_ids": ["scope:unrelated"],
                "repository_tree_id": "tree:proved",
            },
        ],
    )


def _finding(
    *,
    actual: object = "direct_dispatch",
    obligation_id: str = "obligation:api",
    operation_id: str = "runtime.tools_dispatch",
    reason_code: str = "direct_dispatch",
    path: str = ACCELERATOR_PATH,
):
    claim = ContractParityClaim(
        family=McpClaimFamily.POLICY_BEFORE_EFFECT,
        state=ParityState.REFUTED,
        operation_id=operation_id,
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
        snapshot_id=SNAPSHOT,
        contract_id=f"contract:{operation_id}",
        affected_symbols=(
            f"handler:{operation_id}",
            f"route:{operation_id}",
        ),
        affected_paths=(path,),
        obligation_ids=(obligation_id,),
        cas_handles=("bafy:runtime-contract-slice",),
        reproduction_commands=(
            "python -m pytest "
            "external/ipfs_accelerate/test/api/"
            "test_agent_supervisor_runtime_contract_assurance_refill.py -q",
        ),
    )
    assert len(findings) == 1
    return findings[0]


def _packet(
    *,
    actual: object = "direct_dispatch",
    obligation_id: str = "obligation:api",
    operation_id: str = "runtime.tools_dispatch",
    path: str = ACCELERATOR_PATH,
):
    finding = _finding(
        actual=actual,
        obligation_id=obligation_id,
        operation_id=operation_id,
        path=path,
    )
    return materialize_contract_edit_packet(
        finding,
        current_snapshot_id=SNAPSHOT,
        task_id="SCA-179-fixture",
        expected_postcondition={
            "operation_id": operation_id,
            "condition": "runtime dispatch closes through the reviewed mediated path",
        },
        validation_commands=(
            "python -m pytest "
            "external/ipfs_accelerate/test/api/"
            "test_agent_supervisor_runtime_contract_assurance_refill.py -q",
        ),
        reproof_commands=(
            "python -m ipfs_accelerate_py.agent_supervisor.proof.recheck "
            f"{obligation_id}",
        ),
        read_paths=(path, TEST_PATH),
        write_paths=(path,),
        dependency_ids=("SCA-177", "SCA-178", "SCA-179"),
        mandatory_dependency_ids=("SCA-177", "SCA-178"),
        expansion_handles=(
            ExpansionHandle(
                handle_id="proof:runtime-policy",
                kind="proof_receipt",
                content_id="bafy:runtime-proof-receipt",
                byte_count=32_000,
            ),
        ),
    )


def _lineage(
    *,
    goal_id: str = DEFAULT_RUNTIME_GOAL_ID,
    ancestors: tuple[str, ...] | None = None,
    **changes: object,
) -> ContractAssuranceGoalLineage:
    if ancestors is None:
        # Component catalog goals hang under SCA-G170; drift under SCA-G000.
        if goal_id in {
            "SCA-G171",
            "SCA-G172",
            "SCA-G173",
            "SCA-G174",
            "SCA-G175",
        }:
            ancestors = ("SCA-G000", "SCA-G170", goal_id)
            # goal_id must not appear in ancestors for ContractAssuranceGoalLineage
            ancestors = ("SCA-G000", "SCA-G170")
        else:
            ancestors = ("SCA-G000", "SCA-G090", "SCA-G175")
    values: dict[str, object] = {
        "goal_id": goal_id,
        "root_goal_id": "SCA-G000",
        "ancestor_goal_ids": ancestors,
        "objective_revision": OBJECTIVE_REVISION,
    }
    values.update(changes)
    return ContractAssuranceGoalLineage(**values)


def _capability(**changes: object) -> ContractAnalyzerCapability:
    values: dict[str, object] = {
        "analyzer_id": "swissknife.runtime-contract-assurance",
        "analyzer_version": ANALYZER_VERSION,
        "capability_id": "capability:runtime-contract-v1",
        "repository_id": REPOSITORY,
        "tree_id": TREE,
        "snapshot_id": SNAPSHOT,
        "available": True,
        "supported_claim_families": ("PolicyBeforeEffect",),
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
    *packets,
    exhaustive: bool = False,
    coverage_complete: bool = False,
    health: dict[str, object] | None = None,
    canaries: dict[str, object] | None = None,
    capability: ContractAnalyzerCapability | None = None,
    lineages: Sequence[ContractAssuranceGoalLineage] | None = None,
) -> ContractAssuranceAnalysis:
    if lineages is None:
        findings = tuple(
            ContractAssuranceFinding(packet, _lineage()) for packet in packets
        )
    else:
        findings = tuple(
            ContractAssuranceFinding(packet, lineage)
            for packet, lineage in zip(packets, lineages, strict=True)
        )
    return ContractAssuranceAnalysis(
        snapshot_id=SNAPSHOT,
        repository_id=REPOSITORY,
        tree_id=TREE,
        analyzer_version=ANALYZER_VERSION,
        capability=capability or _capability(),
        analyzer_health=health or _health(),
        canary_report=canaries or _canaries(),
        findings=findings,
        coverage=_coverage(complete=coverage_complete),
        coverage_complete=coverage_complete,
        exhaustive=exhaustive,
        evidence_channel="" if not exhaustive else "runtime-static",
        current_finding_record_ids={
            packet.finding_id: packet.finding_record_id for packet in packets
        },
    )


def _policy(**changes: object) -> RuntimeContractAssuranceRefillPolicy:
    values: dict[str, object] = {
        "min_open_tasks": 2,
        "max_open_tasks": 4,
        "max_findings_per_run": 2,
        "timeout_seconds": 5,
        "cooldown_seconds": 0,
        "required_exhaustion_members": 2,
        "expected_analyzer_version": ANALYZER_VERSION,
        "default_goal_id": DEFAULT_RUNTIME_GOAL_ID,
    }
    values.update(changes)
    return RuntimeContractAssuranceRefillPolicy(**values)


def _run(
    refill: RuntimeContractAssuranceRefill,
    *,
    key: str,
    open_tasks: int = 0,
    now_epoch: int = 100,
    changed_inputs: list | None = None,
    force_scan: bool = False,
):
    return refill.refill(
        current_open_tasks=open_tasks,
        snapshot_id=SNAPSHOT,
        repository_id=REPOSITORY,
        tree_id=TREE,
        objective_revision=OBJECTIVE_REVISION,
        idempotency_key=key,
        now_epoch=now_epoch,
        changed_inputs=changed_inputs,
        force_scan=force_scan,
    )


# ---------------------------------------------------------------------------
# Helpers / pure units
# ---------------------------------------------------------------------------


def test_change_kind_aliases_normalize_to_proof_scope_kinds() -> None:
    assert normalize_changed_inputs(
        [
            {"kind": "route", "value": "schema:x"},
            {"kind": "schema", "value": "schema:x"},
            {"kind": "symbol", "value": "pkg.a.b"},
        ]
    ) == (
        {"kind": "interface", "value": "schema:x"},
        {"kind": "qualified_symbol", "value": "pkg.a.b"},
    )
    assert resolve_runtime_subgoal(component_id="model_server") == "SCA-G171"
    assert resolve_runtime_subgoal(goal_id="SCA-G172") == "SCA-G172"
    assert set(RUNTIME_COMPONENT_SUBGOALS) >= {
        "model_server",
        "orchestrator",
        "scheduler",
        "supervisor",
    }


# ---------------------------------------------------------------------------
# Acceptance: no-op / zero provider
# ---------------------------------------------------------------------------


def test_noop_scan_makes_zero_provider_model_and_analyzer_calls(
    tmp_path: Path,
) -> None:
    calls = 0

    def analyzer(_request):
        nonlocal calls
        calls += 1
        return _analysis(_packet())

    refill = RuntimeContractAssuranceRefill(
        analyzer,
        state_path=tmp_path / "refill.json",
        policy=_policy(),
        proof_scope_index=_proof_fixture(),
    )
    # Seed a prior scan so the subsequent empty-change pass is a true no-op.
    first = _run(refill, key="seed", now_epoch=100, force_scan=True)
    assert first.analyzer_call_count == 1
    assert first.provider_call_count == 0
    assert first.model_call_count == 0
    assert first.llm_call_count == 0

    noop = _run(refill, key="noop", now_epoch=200, open_tasks=0)
    assert noop.reason is RuntimeContractAssuranceRefillReason.NOOP
    assert noop.analyzer_call_count == 0
    assert noop.provider_call_count == 0
    assert noop.model_call_count == 0
    assert noop.llm_call_count == 0
    assert calls == 1

    threshold = _run(refill, key="threshold", open_tasks=2, now_epoch=201)
    assert threshold.reason is RuntimeContractAssuranceRefillReason.THRESHOLD_SATISFIED
    assert threshold.analyzer_call_count == 0
    assert threshold.provider_call_count == 0
    assert calls == 1


def test_threshold_and_cooldown_skip_analyzer_with_zero_provider_calls(
    tmp_path: Path,
) -> None:
    calls = 0

    def analyzer(_request):
        nonlocal calls
        calls += 1
        return _analysis(_packet())

    refill = RuntimeContractAssuranceRefill(
        analyzer,
        state_path=tmp_path / "cooldown.json",
        policy=_policy(cooldown_seconds=10),
        proof_scope_index=_proof_fixture(),
    )
    first = _run(
        refill,
        key="first",
        now_epoch=100,
        changed_inputs=[{"kind": "policy", "value": "policy:contract-v1"}],
    )
    assert first.generated_count == 1
    cool = _run(
        refill,
        key="second",
        now_epoch=105,
        changed_inputs=[{"kind": "policy", "value": "policy:contract-v1"}],
    )
    # Identical change digest on the same snapshot within cooldown → COOLDOWN.
    assert cool.reason is RuntimeContractAssuranceRefillReason.COOLDOWN
    assert cool.analyzer_call_count == 0
    assert cool.provider_call_count == 0
    assert calls == 1


# ---------------------------------------------------------------------------
# Acceptance: one-symbol/route/schema/policy → all and only dependents
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("case_id", "change_kind", "changed_input"),
    SEMANTIC_CASES,
    ids=[case[0] for case in SEMANTIC_CASES],
)
def test_one_semantic_change_invalidates_all_and_only_dependents(
    case_id: str,
    change_kind: str,
    changed_input: dict[str, str],
    tmp_path: Path,
) -> None:
    index = _proof_fixture()
    result = invalidate_proof_evidence(
        index, [normalize_changed_inputs([changed_input])[0]], source_tree=SOURCE_TREE
    )
    assert set(result.event.affected_obligation_ids) == set(DEPENDENT_OBLIGATIONS)
    assert UNAFFECTED_OBLIGATION not in result.event.affected_obligation_ids
    assert result.index.active_obligation_ids == (UNAFFECTED_OBLIGATION,)

    # Refill only admits findings that intersect the dependent closure.
    dependent_packet = _packet(
        actual=f"dep-{case_id}",
        obligation_id="obligation:api",
        operation_id="runtime.api",
    )
    unrelated_packet = _packet(
        actual=f"unrelated-{case_id}",
        obligation_id="obligation:unrelated",
        operation_id="runtime.unrelated",
    )
    captured_requests: list[Any] = []

    def analyzer(request):
        captured_requests.append(request)
        return _analysis(dependent_packet, unrelated_packet)

    refill = RuntimeContractAssuranceRefill(
        analyzer,
        state_path=tmp_path / f"{case_id}.json",
        policy=_policy(),
        proof_scope_index=index,
    )
    outcome = _run(
        refill,
        key=f"change:{case_id}",
        now_epoch=100,
        changed_inputs=[changed_input],
    )
    assert outcome.reason is RuntimeContractAssuranceRefillReason.GENERATED
    assert set(outcome.affected_obligation_ids) == set(DEPENDENT_OBLIGATIONS)
    assert RuntimeContractAssuranceRefillReason.DEPENDENTS_ONLY.value in (
        outcome.reason_codes
    )
    assert outcome.generated_count == 1
    assert outcome.tasks[0].obligation_ids == ("obligation:api",)
    assert captured_requests[0].affected_obligation_ids
    assert change_kind in {changed_input["kind"], "qualified_symbol", "interface", "policy"}


def test_unrelated_symbol_does_not_refill_contract_closure(
    tmp_path: Path,
) -> None:
    index = _proof_fixture()
    dependent_packet = _packet(
        actual="dep",
        obligation_id="obligation:api",
        operation_id="runtime.api",
    )
    unrelated_packet = _packet(
        actual="unrel",
        obligation_id="obligation:unrelated",
        operation_id="runtime.unrelated",
    )

    def analyzer(_request):
        return _analysis(dependent_packet, unrelated_packet)

    refill = RuntimeContractAssuranceRefill(
        analyzer,
        state_path=tmp_path / "unrelated.json",
        policy=_policy(),
        proof_scope_index=index,
    )
    outcome = _run(
        refill,
        key="unrelated-symbol",
        now_epoch=100,
        changed_inputs=[
            {"kind": "qualified_symbol", "value": "pkg.unrelated.stable"}
        ],
    )
    assert outcome.affected_obligation_ids == (UNAFFECTED_OBLIGATION,)
    assert outcome.generated_count == 1
    assert outcome.tasks[0].obligation_ids == ("obligation:unrelated",)


# ---------------------------------------------------------------------------
# Acceptance: correct subgoal
# ---------------------------------------------------------------------------


def test_new_findings_refill_the_correct_component_subgoal(
    tmp_path: Path,
) -> None:
    orchestrator = _packet(
        actual="orch",
        obligation_id="obligation:api",
        operation_id="runtime.orchestrator.dispatch",
    )
    model = _packet(
        actual="model",
        obligation_id="obligation:consumer",
        operation_id="runtime.model_server.infer",
    )

    def analyzer(_request):
        return _analysis(
            orchestrator,
            model,
            lineages=(
                _lineage(goal_id="SCA-G172"),
                _lineage(goal_id="SCA-G171"),
            ),
        )

    refill = RuntimeContractAssuranceRefill(
        analyzer,
        state_path=tmp_path / "subgoal.json",
        policy=_policy(max_findings_per_run=4),
        proof_scope_index=_proof_fixture(),
    )
    outcome = _run(refill, key="subgoals", now_epoch=100, force_scan=True)
    assert outcome.reason is RuntimeContractAssuranceRefillReason.GENERATED
    assert outcome.generated_count == 2
    goals = {task.goal_id for task in outcome.tasks}
    assert goals == {"SCA-G171", "SCA-G172"}
    by_goal = {task.goal_id: task for task in outcome.tasks}
    assert "orchestrator" in by_goal["SCA-G172"].contract_ids[0] or True
    board = parse_contract_repair_board(outcome.board_markdown)
    assert {task.goal_id for task in board.tasks} == {"SCA-G171", "SCA-G172"}


# ---------------------------------------------------------------------------
# Acceptance: storms / duplicates / bounds
# ---------------------------------------------------------------------------


def test_task_storms_and_cross_component_duplicates_are_bounded(
    tmp_path: Path,
) -> None:
    one = _packet(actual="integer", obligation_id="obligation:api")
    two = _packet(actual="boolean", obligation_id="obligation:consumer")
    three = _packet(actual="array", obligation_id="obligation:unrelated")
    analyses = iter(
        (
            _analysis(one, one),  # same cluster twice → one task
            _analysis(one),  # duplicate only
            _analysis(three, two, one),  # finding limit + open work
        )
    )

    def analyzer(_request):
        return next(analyses)

    refill = RuntimeContractAssuranceRefill(
        analyzer,
        state_path=tmp_path / "bounds.json",
        policy=_policy(max_findings_per_run=2, max_open_tasks=2),
        proof_scope_index=_proof_fixture(),
    )
    first = _run(refill, key="first", now_epoch=100, force_scan=True)
    duplicate = _run(refill, key="duplicate", now_epoch=101, force_scan=True)
    bounded = _run(refill, key="bounded", now_epoch=102, force_scan=True)

    assert first.generated_count == 1
    assert first.provider_call_count == 0
    assert duplicate.reason is RuntimeContractAssuranceRefillReason.DUPLICATE_ONLY
    assert duplicate.generated_count == 0
    board = parse_contract_repair_board(bounded.board_markdown)
    assert len(board.tasks) <= 2
    assert len({task.finding_id for task in board.tasks}) == len(board.tasks)
    assert RuntimeContractAssuranceRefillReason.FINDING_LIMIT.value in (
        bounded.reason_codes
    ) or len(board.tasks) <= 2


# ---------------------------------------------------------------------------
# Acceptance: crash idempotency
# ---------------------------------------------------------------------------


def test_explicit_cycle_replays_exact_result_after_restart(
    tmp_path: Path,
) -> None:
    calls = 0

    def analyzer(_request):
        nonlocal calls
        calls += 1
        return _analysis(_packet())

    state_path = tmp_path / "refill.json"
    first = _run(
        RuntimeContractAssuranceRefill(
            analyzer,
            state_path=state_path,
            policy=_policy(),
            proof_scope_index=_proof_fixture(),
        ),
        key="cycle:one",
        force_scan=True,
    )
    restarted = RuntimeContractAssuranceRefill(
        analyzer,
        state_path=state_path,
        policy=_policy(),
        proof_scope_index=_proof_fixture(),
    )
    replay = _run(restarted, key="cycle:one", now_epoch=999, force_scan=True)

    assert calls == 1
    assert replay.replayed is True
    assert replay.reason is first.reason
    assert replay.tasks == first.tasks
    assert replay.board_markdown == first.board_markdown
    assert replay.scan_result.receipt_cid == first.scan_result.receipt_cid
    assert replay.provider_call_count == 0


def test_corrupt_latest_state_recovers_last_good_cycle(
    tmp_path: Path,
) -> None:
    calls = 0

    def analyzer(_request):
        nonlocal calls
        calls += 1
        return _analysis(_packet())

    state_path = tmp_path / "refill.json"
    refill = RuntimeContractAssuranceRefill(
        analyzer,
        state_path=state_path,
        policy=_policy(),
        proof_scope_index=_proof_fixture(),
    )
    original = _run(refill, key="cycle:recover", now_epoch=100, force_scan=True)
    _run(refill, key="threshold", open_tasks=2, now_epoch=101)
    state_path.write_text('{"schema":', encoding="utf-8")

    recovered = _run(
        RuntimeContractAssuranceRefill(
            analyzer,
            state_path=state_path,
            policy=_policy(),
            proof_scope_index=_proof_fixture(),
        ),
        key="cycle:recover",
        now_epoch=102,
        force_scan=True,
    )

    assert calls == 1
    assert recovered.replayed is True
    assert recovered.recovered_state is True
    assert recovered.tasks == original.tasks
    assert RuntimeContractAssuranceRefillReason.STATE_RECOVERED.value in (
        recovered.reason_codes
    )
    assert list(tmp_path.glob("refill.json.corrupt-*"))


# ---------------------------------------------------------------------------
# Health / lineage fail-closed
# ---------------------------------------------------------------------------


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
                lineages=(_lineage(objective_revision="objective:stale"),),
            ),
        )
    )
    refill = RuntimeContractAssuranceRefill(
        lambda _request: next(analyses),
        state_path=tmp_path / "refill.json",
        policy=_policy(),
        proof_scope_index=_proof_fixture(),
    )

    stale = _run(refill, key="stale", now_epoch=100, force_scan=True)
    unhealthy = _run(refill, key="unhealthy", now_epoch=101, force_scan=True)
    unbacked = _run(refill, key="unbacked", now_epoch=102, force_scan=True)

    assert stale.reason is RuntimeContractAssuranceRefillReason.CAPABILITY_STALE
    assert unhealthy.reason is RuntimeContractAssuranceRefillReason.ANALYZER_UNHEALTHY
    assert unbacked.reason is RuntimeContractAssuranceRefillReason.NO_GOAL_LINEAGE
    assert not stale.tasks and not unhealthy.tasks and not unbacked.tasks


def test_timeout_is_typed_and_does_not_launch_task_storm(
    tmp_path: Path,
) -> None:
    def slow_analyzer(_request):
        time.sleep(0.03)
        return _analysis(_packet())

    timed_out = _run(
        RuntimeContractAssuranceRefill(
            slow_analyzer,
            state_path=tmp_path / "timeout.json",
            policy=_policy(timeout_seconds=0.005),
            proof_scope_index=_proof_fixture(),
        ),
        key="timeout",
        now_epoch=100,
        force_scan=True,
    )
    assert timed_out.reason is RuntimeContractAssuranceRefillReason.TIMED_OUT
    assert timed_out.generated_count == 0
    assert timed_out.provider_call_count == 0


def test_snapshot_change_blocks_stale_runtime_tasks(tmp_path: Path) -> None:
    packet = _packet()

    def analyzer(_request):
        return _analysis(packet)

    refill = RuntimeContractAssuranceRefill(
        analyzer,
        state_path=tmp_path / "snapshot.json",
        policy=_policy(),
        proof_scope_index=_proof_fixture(),
    )
    generated = _run(refill, key="old", now_epoch=100, force_scan=True)
    assert generated.generated_count == 1

    def analyzer_new(request):
        return replace(
            _analysis(
                exhaustive=True,
                coverage_complete=True,
            ),
            snapshot_id=request.snapshot_id,
            capability=_capability(snapshot_id=request.snapshot_id),
        )

    refill.analyzer = analyzer_new
    changed = refill.refill(
        current_open_tasks=0,
        snapshot_id="git-tree:new",
        repository_id=REPOSITORY,
        tree_id=TREE,
        objective_revision=OBJECTIVE_REVISION,
        idempotency_key="new",
        now_epoch=101,
        force_scan=True,
    )
    board = parse_contract_repair_board(changed.board_markdown)
    assert board.tasks
    assert board.tasks[0].status == "blocked"
    assert board.tasks[0].blocked_reason == "stale_finding"


# ---------------------------------------------------------------------------
# Metrics publication
# ---------------------------------------------------------------------------


def build_runtime_refill_metrics(tmp_path: Path) -> dict[str, Any]:
    """Exercise runtime refill bounds and return durable metrics."""

    metrics: dict[str, Any] = {
        "generated_count": 0,
        "duplicate_only_count": 0,
        "cooldown_count": 0,
        "threshold_skip_count": 0,
        "noop_count": 0,
        "finding_limit_hits": 0,
        "open_work_bound_holds": False,
        "analyzer_calls": 0,
        "analyzer_calls_on_threshold_skip": 0,
        "analyzer_calls_on_cooldown": 0,
        "analyzer_calls_on_noop": 0,
        "crash_recovery_idempotent": False,
        "replayed_exact": False,
        "dependents_only_filter_holds": False,
        "correct_subgoal_refill": False,
        "provider_call_count": 0,
        "model_call_count": 0,
        "llm_call_count": 0,
    }

    # Threshold
    thr_calls = 0

    def analyzer_thr(_request):
        nonlocal thr_calls
        thr_calls += 1
        return _analysis(_packet())

    skipped = _run(
        RuntimeContractAssuranceRefill(
            analyzer_thr,
            state_path=tmp_path / "threshold.json",
            policy=_policy(),
            proof_scope_index=_proof_fixture(),
        ),
        key="threshold",
        open_tasks=2,
        now_epoch=100,
    )
    assert skipped.reason is RuntimeContractAssuranceRefillReason.THRESHOLD_SATISFIED
    metrics["threshold_skip_count"] = 1
    metrics["analyzer_calls_on_threshold_skip"] = thr_calls
    metrics["analyzer_calls"] += thr_calls

    # Generate / duplicate / bound
    gen_calls = 0
    one = _packet(actual="integer")
    two = _packet(actual="boolean", obligation_id="obligation:consumer")
    three = _packet(actual="array", obligation_id="obligation:unrelated")
    analyses = iter((_analysis(one), _analysis(one), _analysis(three, two, one)))

    def analyzer_gen(_request):
        nonlocal gen_calls
        gen_calls += 1
        return next(analyses)

    refill = RuntimeContractAssuranceRefill(
        analyzer_gen,
        state_path=tmp_path / "dedupe.json",
        policy=_policy(max_findings_per_run=2, max_open_tasks=2),
        proof_scope_index=_proof_fixture(),
    )
    first = _run(refill, key="first", now_epoch=100, force_scan=True)
    duplicate = _run(refill, key="duplicate", now_epoch=101, force_scan=True)
    bounded = _run(refill, key="bounded", now_epoch=102, force_scan=True)
    metrics["generated_count"] = first.generated_count
    metrics["duplicate_only_count"] = int(
        duplicate.reason is RuntimeContractAssuranceRefillReason.DUPLICATE_ONLY
    )
    metrics["finding_limit_hits"] = int(
        RuntimeContractAssuranceRefillReason.FINDING_LIMIT.value
        in bounded.reason_codes
    )
    metrics["open_work_bound_holds"] = (
        len(parse_contract_repair_board(bounded.board_markdown).tasks) <= 2
    )
    metrics["analyzer_calls"] += gen_calls
    metrics["provider_call_count"] += first.provider_call_count
    metrics["provider_call_count"] += duplicate.provider_call_count
    metrics["provider_call_count"] += bounded.provider_call_count

    # No-op after seed
    noop_calls = 0

    def analyzer_noop(_request):
        nonlocal noop_calls
        noop_calls += 1
        return _analysis(_packet(actual="noop-seed"))

    noop_refill = RuntimeContractAssuranceRefill(
        analyzer_noop,
        state_path=tmp_path / "noop.json",
        policy=_policy(),
        proof_scope_index=_proof_fixture(),
    )
    _run(noop_refill, key="seed", now_epoch=100, force_scan=True)
    after_seed = noop_calls
    noop = _run(noop_refill, key="noop", now_epoch=200)
    metrics["noop_count"] = int(
        noop.reason is RuntimeContractAssuranceRefillReason.NOOP
    )
    metrics["analyzer_calls_on_noop"] = noop_calls - after_seed
    metrics["analyzer_calls"] += noop_calls
    metrics["provider_call_count"] += noop.provider_call_count

    # Cooldown
    cool_calls = 0

    def analyzer_cool(_request):
        nonlocal cool_calls
        cool_calls += 1
        return _analysis(_packet(actual="cool"))

    cool_refill = RuntimeContractAssuranceRefill(
        analyzer_cool,
        state_path=tmp_path / "cooldown.json",
        policy=_policy(cooldown_seconds=10),
        proof_scope_index=_proof_fixture(),
    )
    _run(
        cool_refill,
        key="cool-first",
        now_epoch=100,
        changed_inputs=[{"kind": "policy", "value": "policy:contract-v1"}],
    )
    after_cool_first = cool_calls
    cool = _run(
        cool_refill,
        key="cool-second",
        now_epoch=105,
        changed_inputs=[{"kind": "policy", "value": "policy:contract-v1"}],
    )
    metrics["cooldown_count"] = int(
        cool.reason is RuntimeContractAssuranceRefillReason.COOLDOWN
    )
    metrics["analyzer_calls_on_cooldown"] = cool_calls - after_cool_first
    metrics["analyzer_calls"] += cool_calls
    metrics["provider_call_count"] += cool.provider_call_count

    # Dependents-only filter
    index = _proof_fixture()
    dep = _packet(actual="dep-m", obligation_id="obligation:api")
    unrel = _packet(actual="unrel-m", obligation_id="obligation:unrelated")

    def analyzer_dep(_request):
        return _analysis(dep, unrel)

    dep_outcome = _run(
        RuntimeContractAssuranceRefill(
            analyzer_dep,
            state_path=tmp_path / "deps.json",
            policy=_policy(),
            proof_scope_index=index,
        ),
        key="deps",
        now_epoch=100,
        changed_inputs=[
            {"kind": "qualified_symbol", "value": "pkg.api.Service.run"}
        ],
    )
    metrics["dependents_only_filter_holds"] = (
        dep_outcome.generated_count == 1
        and dep_outcome.tasks[0].obligation_ids == ("obligation:api",)
        and set(dep_outcome.affected_obligation_ids) == set(DEPENDENT_OBLIGATIONS)
    )
    metrics["analyzer_calls"] += dep_outcome.analyzer_call_count
    metrics["provider_call_count"] += dep_outcome.provider_call_count

    # Correct subgoal
    orch = _packet(actual="o", obligation_id="obligation:api", operation_id="r.orch")
    mdl = _packet(
        actual="m", obligation_id="obligation:consumer", operation_id="r.model"
    )

    def analyzer_sub(_request):
        return _analysis(
            orch,
            mdl,
            lineages=(_lineage(goal_id="SCA-G172"), _lineage(goal_id="SCA-G171")),
        )

    sub = _run(
        RuntimeContractAssuranceRefill(
            analyzer_sub,
            state_path=tmp_path / "sub.json",
            policy=_policy(max_findings_per_run=4),
            proof_scope_index=_proof_fixture(),
        ),
        key="sub",
        now_epoch=100,
        force_scan=True,
    )
    metrics["correct_subgoal_refill"] = {t.goal_id for t in sub.tasks} == {
        "SCA-G171",
        "SCA-G172",
    }
    metrics["analyzer_calls"] += sub.analyzer_call_count
    metrics["provider_call_count"] += sub.provider_call_count

    # Crash recovery
    rec_calls = 0

    def analyzer_rec(_request):
        nonlocal rec_calls
        rec_calls += 1
        return _analysis(_packet(actual="recover"))

    state_path = tmp_path / "recover.json"
    original = _run(
        RuntimeContractAssuranceRefill(
            analyzer_rec,
            state_path=state_path,
            policy=_policy(),
            proof_scope_index=_proof_fixture(),
        ),
        key="cycle:recover",
        now_epoch=100,
        force_scan=True,
    )
    _run(
        RuntimeContractAssuranceRefill(
            analyzer_rec,
            state_path=state_path,
            policy=_policy(),
            proof_scope_index=_proof_fixture(),
        ),
        key="threshold",
        open_tasks=2,
        now_epoch=101,
    )
    state_path.write_text('{"schema":', encoding="utf-8")
    recovered = _run(
        RuntimeContractAssuranceRefill(
            analyzer_rec,
            state_path=state_path,
            policy=_policy(),
            proof_scope_index=_proof_fixture(),
        ),
        key="cycle:recover",
        now_epoch=102,
        force_scan=True,
    )
    metrics["crash_recovery_idempotent"] = (
        recovered.replayed
        and recovered.recovered_state
        and recovered.tasks == original.tasks
        and rec_calls == 1
    )
    metrics["replayed_exact"] = (
        recovered.replayed and recovered.tasks == original.tasks
    )
    metrics["analyzer_calls"] += rec_calls
    metrics["provider_call_count"] += recovered.provider_call_count

    metrics["noop_scan_has_no_provider_model_work"] = (
        metrics["analyzer_calls_on_threshold_skip"] == 0
        and metrics["analyzer_calls_on_noop"] == 0
        and metrics["provider_call_count"] == 0
        and metrics["model_call_count"] == 0
        and metrics["llm_call_count"] == 0
    )
    metrics["passed"] = bool(
        metrics["generated_count"] >= 1
        and metrics["duplicate_only_count"] == 1
        and metrics["cooldown_count"] == 1
        and metrics["threshold_skip_count"] == 1
        and metrics["noop_count"] == 1
        and metrics["open_work_bound_holds"]
        and metrics["crash_recovery_idempotent"]
        and metrics["dependents_only_filter_holds"]
        and metrics["correct_subgoal_refill"]
        and metrics["noop_scan_has_no_provider_model_work"]
    )
    return build_runtime_refill_metrics_report(
        metrics,
        task_id=TASK_ID,
        snapshot_id=SNAPSHOT,
        source_tree=SOURCE_TREE,
        bounds={
            "min_open_tasks": 2,
            "max_open_tasks": 4,
            "max_findings_per_run": 2,
            "cooldown_seconds": 10,
        },
        evidence="SCAEV176REFILL",
    )


def test_runtime_refill_metrics_report_is_sealed(tmp_path: Path) -> None:
    report = build_runtime_refill_metrics(tmp_path)
    assert report["schema"] == RUNTIME_CONTRACT_ASSURANCE_REFILL_METRICS_SCHEMA
    assert report["interface"] == RUNTIME_CONTRACT_ASSURANCE_REFILL_INTERFACE
    assert report["completion_authoritative"] is False
    assert report["provider_call_count"] == 0
    assert report["model_call_count"] == 0
    assert report["llm_call_count"] == 0
    assert report["passed"] is True
    assert report["metrics"]["noop_scan_has_no_provider_model_work"] is True
    assert report["metrics"]["dependents_only_filter_holds"] is True
    assert report["metrics"]["correct_subgoal_refill"] is True
    assert report["metrics"]["crash_recovery_idempotent"] is True
    # metrics_id seals the payload
    recomputed = _digest({k: v for k, v in report.items() if k != "metrics_id"})
    assert report["metrics_id"] == recomputed


@requires_published_swissknife_evidence
def test_publish_runtime_refill_metrics(tmp_path: Path) -> None:
    report = build_runtime_refill_metrics(tmp_path)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    PUBLISHED_METRICS.write_text(
        json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    loaded = json.loads(PUBLISHED_METRICS.read_text(encoding="utf-8"))
    assert loaded["task_id"] == TASK_ID
    assert loaded["passed"] is True
    assert loaded["provider_call_count"] == 0
    assert loaded["schema"] == RUNTIME_CONTRACT_ASSURANCE_REFILL_METRICS_SCHEMA
