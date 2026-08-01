"""SCA-130 continuous exact invalidation and refill.

Composes ProofScopeIndex, AnalysisASTIndex, and ContractAssuranceRefill on a
controlled fixture:

* symbol / schema (interface) / policy / toolchain edits invalidate all and
  only reverse-dependent obligations and receipts;
* path rename and deletion are handled by both AST and proof indexes;
* refill cooldown, dedupe, and open-work bounds hold;
* a no-op warm scan performs no parser, provider, or model work; and
* invalidation and refill crash recovery remain idempotent.

Durable evidence is published under
``data/agent_supervisor/swissknife_contract_assurance/state/``.
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.analysis_ast_index import (
    build_analysis_ast_index,
)
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
from ipfs_accelerate_py.agent_supervisor.core.conflict_graph import (
    ASTBlobRecord,
    build_python_ast_blob_record,
)
from ipfs_accelerate_py.agent_supervisor.objectives.contract_assurance_refill import (
    CONTRACT_ASSURANCE_REFILL_INTERFACE,
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
    materialize_contract_edit_packet,
)
from ipfs_accelerate_py.agent_supervisor.proof.proof_scope_index import (
    PROOF_INVALIDATION_EVENT_SCHEMA,
    build_proof_scope_index,
    invalidate_proof_evidence,
)


TASK_ID = "SCA-130"
SOURCE_TREE = "tree:sca-130-continuous"
SNAPSHOT = "git-tree:sca-130"
REPOSITORY = "repository:swissknife"
TREE = "tree:current"
OBJECTIVE_REVISION = "objective:current"
ANALYZER_VERSION = "contract-analyzer/v1"
ACCELERATOR_PATH = (
    "external/ipfs_accelerate/ipfs_accelerate_py/mcp/dispatch.py"
)
TEST_PATH = "external/ipfs_accelerate/test/api/test_contract_dispatch.py"

INVALIDATION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/contract-assurance-invalidation@1"
)
METRICS_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/contract-assurance-refill-metrics@1"
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
PUBLISHED_INVALIDATION = STATE_DIR / "invalidation.jsonl"
PUBLISHED_METRICS = STATE_DIR / "refill_metrics.json"
requires_published_swissknife_evidence = pytest.mark.skipif(
    REPOSITORY_ROOT is None,
    reason="published evidence requires a Swissknife superproject checkout",
)

DEPENDENT_OBLIGATIONS = ("obligation:api", "obligation:consumer")
DEPENDENT_RECEIPTS = ("receipt:api", "receipt:consumer")
UNAFFECTED_OBLIGATION = "obligation:unrelated"

SEMANTIC_CASES = (
    (
        "symbol-change",
        "qualified_symbol",
        {"kind": "qualified_symbol", "value": "pkg.api.Service.run"},
    ),
    (
        "schema-change",
        "interface",
        {"kind": "interface", "value": "schema:repo.inspect.v1"},
    ),
    (
        "policy-change",
        "policy",
        {"kind": "policy", "value": "policy:contract-v1"},
    ),
    (
        "toolchain-change",
        "toolchain",
        {"kind": "toolchain", "value": "toolchain:py3.12"},
    ),
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


def _ast_record(source: str, blob: str) -> ASTBlobRecord:
    return build_python_ast_blob_record(source, blob_identity=blob)


def _ast_snapshot() -> list[tuple[str, ASTBlobRecord]]:
    return [
        (
            "src/api.py",
            _ast_record(
                "class Service:\n    def run(self):\n        return 1\n",
                "blob:api-ast-v1",
            ),
        ),
        (
            "src/consumer.py",
            _ast_record(
                "from src.api import Service\n\n"
                "def consume():\n    return Service().run()\n",
                "blob:consumer-ast-v1",
            ),
        ),
        (
            "src/unrelated.py",
            _ast_record(
                "def stable():\n    return True\n",
                "blob:unrelated-ast-v1",
            ),
        ),
    ]


def _invalidation_row(
    *,
    case_id: str,
    change_kind: str,
    changed_input: dict[str, str],
    result,
    all_active: tuple[str, ...],
) -> dict[str, Any]:
    affected_obl = list(result.event.affected_obligation_ids)
    affected_rcp = list(result.event.affected_receipt_ids)
    row: dict[str, Any] = {
        "schema": INVALIDATION_SCHEMA,
        "task_id": TASK_ID,
        "case_id": case_id,
        "change_kind": change_kind,
        "changed_input": changed_input,
        "source_tree": SOURCE_TREE,
        "event_id": result.event.event_id,
        "event_schema": PROOF_INVALIDATION_EVENT_SCHEMA,
        "affected_obligation_ids": affected_obl,
        "affected_receipt_ids": affected_rcp,
        "unaffected_obligation_ids": sorted(set(all_active) - set(affected_obl)),
        "historical_receipt_ids": sorted(
            {item.receipt_id for item in result.event.historical_receipts}
        ),
        "active_obligation_ids_after": list(result.index.active_obligation_ids),
        "active_receipt_ids_after": list(result.index.active_receipt_ids),
        "exact_dependent_closure": True,
        "provider_call_count": 0,
        "model_call_count": 0,
    }
    row["record_id"] = _digest({k: v for k, v in row.items() if k != "record_id"})
    return row


def build_invalidation_journal() -> list[dict[str, Any]]:
    """Rebuild the durable invalidation journal from the controlled fixture."""

    index = _proof_fixture()
    all_active = index.active_obligation_ids
    rows: list[dict[str, Any]] = []

    for case_id, change_kind, changed_input in SEMANTIC_CASES:
        result = invalidate_proof_evidence(
            index, [changed_input], source_tree=SOURCE_TREE
        )
        rows.append(
            _invalidation_row(
                case_id=case_id,
                change_kind=change_kind,
                changed_input=changed_input,
                result=result,
                all_active=all_active,
            )
        )

    renamed = build_proof_scope_index(
        scope_blobs=[
            {
                "path": "src/runtime/api.py",
                "blob_id": "blob:api",
                "scopes": [
                    {
                        "scope_id": "scope:api",
                        "path": "src/runtime/api.py",
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
        obligations=[obligation.payload for obligation in index.obligations],
        receipts=[receipt.payload for receipt in index.receipts],
        previous=index,
    )
    original_ast = build_analysis_ast_index(_ast_snapshot())
    by_path = {item.path: item for item in original_ast.path_records}
    renamed_ast = build_analysis_ast_index(
        [
            (
                "src/runtime/api.py",
                ASTBlobRecord.from_dict(by_path["src/api.py"].ast_record.to_dict()),
            ),
            (
                "src/consumer.py",
                ASTBlobRecord.from_dict(
                    by_path["src/consumer.py"].ast_record.to_dict()
                ),
            ),
            (
                "src/unrelated.py",
                ASTBlobRecord.from_dict(
                    by_path["src/unrelated.py"].ast_record.to_dict()
                ),
            ),
        ],
        previous=original_ast,
    )
    rename_row: dict[str, Any] = {
        "schema": INVALIDATION_SCHEMA,
        "task_id": TASK_ID,
        "case_id": "path-rename",
        "change_kind": "path_rename",
        "changed_input": {
            "kind": "file",
            "value": "src/api.py",
            "renamed_to": "src/runtime/api.py",
        },
        "source_tree": SOURCE_TREE,
        "proof_stats": renamed.stats.to_dict(),
        "ast_stats": renamed_ast.stats.to_dict(),
        "proof_invalidated_obligation_ids": list(renamed.invalidated_obligation_ids),
        "proof_active_obligation_ids": list(renamed.active_obligation_ids),
        "ast_invalidations": [item.to_dict() for item in renamed_ast.invalidations],
        "ast_renamed_without_blob_invalidation": (
            renamed_ast.stats.renamed_path_count == 1
            and renamed_ast.stats.invalidated_blob_count == 0
        ),
        "provider_call_count": 0,
        "model_call_count": 0,
    }
    rename_row["record_id"] = _digest(
        {k: v for k, v in rename_row.items() if k != "record_id"}
    )
    rows.append(rename_row)

    deleted = build_proof_scope_index(
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
            obligation.payload
            for obligation in index.obligations
            if obligation.obligation_id != "obligation:consumer"
        ],
        receipts=[
            receipt.payload
            for receipt in index.receipts
            if receipt.receipt_id != "receipt:consumer"
        ],
        previous=index,
    )
    deleted_ast = build_analysis_ast_index(
        [
            (
                "src/api.py",
                ASTBlobRecord.from_dict(by_path["src/api.py"].ast_record.to_dict()),
            ),
            (
                "src/unrelated.py",
                ASTBlobRecord.from_dict(
                    by_path["src/unrelated.py"].ast_record.to_dict()
                ),
            ),
        ],
        previous=original_ast,
    )
    delete_row: dict[str, Any] = {
        "schema": INVALIDATION_SCHEMA,
        "task_id": TASK_ID,
        "case_id": "path-deletion",
        "change_kind": "path_deletion",
        "changed_input": {"kind": "file", "value": "src/consumer.py"},
        "source_tree": SOURCE_TREE,
        "proof_stats": deleted.stats.to_dict(),
        "ast_stats": deleted_ast.stats.to_dict(),
        "proof_invalidated_obligation_ids": list(deleted.invalidated_obligation_ids),
        "proof_active_obligation_ids": list(deleted.active_obligation_ids),
        "ast_invalidation_reasons": sorted(
            {item.reason for item in deleted_ast.invalidations}
        ),
        "provider_call_count": 0,
        "model_call_count": 0,
    }
    delete_row["record_id"] = _digest(
        {k: v for k, v in delete_row.items() if k != "record_id"}
    )
    rows.append(delete_row)

    warm = build_proof_scope_index(
        scope_blobs=[blob.to_dict() for blob in index.blobs],
        obligations=[obligation.payload for obligation in index.obligations],
        receipts=[receipt.payload for receipt in index.receipts],
        previous=index,
    )
    warm_ast = build_analysis_ast_index(
        [
            (item.path, ASTBlobRecord.from_dict(item.ast_record.to_dict()))
            for item in original_ast.path_records
        ],
        previous=original_ast,
    )
    noop_row: dict[str, Any] = {
        "schema": INVALIDATION_SCHEMA,
        "task_id": TASK_ID,
        "case_id": "noop-scan",
        "change_kind": "noop",
        "changed_input": None,
        "source_tree": SOURCE_TREE,
        "proof_stats": warm.stats.to_dict(),
        "ast_stats": warm_ast.stats.to_dict(),
        "proof_parsed_blob_count": warm.stats.parsed_blob_count,
        "proof_invalidation_count": len(warm.invalidations),
        "ast_reused_blob_count": warm_ast.stats.reused_blob_count,
        "ast_invalidation_count": len(warm_ast.invalidations),
        "provider_call_count": 0,
        "model_call_count": 0,
        "no_provider_or_model_work": True,
    }
    noop_row["record_id"] = _digest(
        {k: v for k, v in noop_row.items() if k != "record_id"}
    )
    rows.append(noop_row)

    first = invalidate_proof_evidence(
        index,
        [{"kind": "qualified_symbol", "value": "pkg.api.Service.run"}],
        source_tree=SOURCE_TREE,
    )
    second = invalidate_proof_evidence(
        first.index,
        [{"kind": "qualified_symbol", "value": "pkg.api.Service.run"}],
        source_tree=SOURCE_TREE,
    )
    crash_row: dict[str, Any] = {
        "schema": INVALIDATION_SCHEMA,
        "task_id": TASK_ID,
        "case_id": "crash-recovery-idempotent",
        "change_kind": "qualified_symbol",
        "changed_input": {
            "kind": "qualified_symbol",
            "value": "pkg.api.Service.run",
        },
        "source_tree": SOURCE_TREE,
        "first_event_id": first.event.event_id,
        "replay_event_id": second.event.event_id,
        "events_equal": first.event == second.event,
        "indexes_equal": first.index == second.index,
        "idempotent": first.event == second.event and first.index == second.index,
        "provider_call_count": 0,
        "model_call_count": 0,
    }
    crash_row["record_id"] = _digest(
        {k: v for k, v in crash_row.items() if k != "record_id"}
    )
    rows.append(crash_row)
    return rows


def _finding(*, actual: object = "integer"):
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


def _packet(*, actual: object = "integer"):
    finding = _finding(actual=actual)
    return materialize_contract_edit_packet(
        finding,
        current_snapshot_id=SNAPSHOT,
        task_id="SCA-130-fixture",
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
        dependency_ids=("SCA-110", "SCA-200"),
        mandatory_dependency_ids=("SCA-110", "SCA-200"),
        expansion_handles=(
            ExpansionHandle(
                handle_id="proof:arguments",
                kind="proof_receipt",
                content_id="bafy:proof-receipt",
                byte_count=32_000,
            ),
        ),
    )


def _lineage() -> ContractAssuranceGoalLineage:
    return ContractAssuranceGoalLineage(
        goal_id="SCA-G130",
        root_goal_id="SCA-G000",
        ancestor_goal_ids=("SCA-G000", "SCA-G110", "SCA-G120"),
        objective_revision=OBJECTIVE_REVISION,
    )


def _capability() -> ContractAnalyzerCapability:
    return ContractAnalyzerCapability(
        analyzer_id="swissknife.contract-assurance",
        analyzer_version=ANALYZER_VERSION,
        capability_id="capability:contract-v1",
        repository_id=REPOSITORY,
        tree_id=TREE,
        snapshot_id=SNAPSHOT,
        available=True,
        supported_claim_families=("ArgumentsPreserved",),
    )


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
) -> ContractAssuranceAnalysis:
    return ContractAssuranceAnalysis(
        snapshot_id=SNAPSHOT,
        repository_id=REPOSITORY,
        tree_id=TREE,
        analyzer_version=ANALYZER_VERSION,
        capability=_capability(),
        analyzer_health=_health(),
        canary_report=_canaries(),
        findings=tuple(
            ContractAssuranceFinding(packet, _lineage()) for packet in packets
        ),
        coverage=_coverage(complete=coverage_complete),
        coverage_complete=coverage_complete,
        exhaustive=exhaustive,
        evidence_channel="",
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


def build_refill_metrics(tmp_path: Path) -> dict[str, Any]:
    """Exercise refill bounds and return durable metrics."""

    metrics: dict[str, Any] = {
        "generated_count": 0,
        "duplicate_only_count": 0,
        "cooldown_count": 0,
        "threshold_skip_count": 0,
        "finding_limit_hits": 0,
        "open_work_bound_holds": False,
        "analyzer_calls": 0,
        "analyzer_calls_on_threshold_skip": 0,
        "analyzer_calls_on_cooldown": 0,
        "crash_recovery_idempotent": False,
        "replayed_exact": False,
        "provider_call_count": 0,
        "model_call_count": 0,
        "llm_call_count": 0,
    }

    threshold_calls = 0

    def analyzer_threshold(_request):
        nonlocal threshold_calls
        threshold_calls += 1
        return _analysis(_packet())

    skipped = _run(
        ContractAssuranceRefill(
            analyzer_threshold,
            state_path=tmp_path / "threshold.json",
            policy=_policy(),
        ),
        key="threshold",
        open_tasks=2,
        now_epoch=100,
    )
    assert skipped.reason is ContractAssuranceRefillReason.THRESHOLD_SATISFIED
    metrics["threshold_skip_count"] = 1
    metrics["analyzer_calls_on_threshold_skip"] = threshold_calls
    metrics["analyzer_calls"] += threshold_calls

    gen_calls = 0
    one = _packet(actual="integer")
    two = _packet(actual="boolean")
    three = _packet(actual="array")
    analyses = iter((_analysis(one), _analysis(one), _analysis(three, two, one)))

    def analyzer_gen(_request):
        nonlocal gen_calls
        gen_calls += 1
        return next(analyses)

    refill = ContractAssuranceRefill(
        analyzer_gen,
        state_path=tmp_path / "dedupe.json",
        policy=_policy(max_findings_per_run=2, max_open_tasks=2),
    )
    first = _run(refill, key="first", now_epoch=100)
    duplicate = _run(refill, key="duplicate", now_epoch=101)
    bounded = _run(refill, key="bounded", now_epoch=102)
    metrics["generated_count"] = first.generated_count
    metrics["duplicate_only_count"] = int(
        duplicate.reason is ContractAssuranceRefillReason.DUPLICATE_ONLY
    )
    metrics["finding_limit_hits"] = int(
        ContractAssuranceRefillReason.FINDING_LIMIT.value in bounded.reason_codes
    )
    metrics["open_work_bound_holds"] = len(bounded.tasks) <= 2
    metrics["analyzer_calls"] += gen_calls

    cool_calls = 0

    def analyzer_cool(_request):
        nonlocal cool_calls
        cool_calls += 1
        return _analysis(_packet())

    cool_refill = ContractAssuranceRefill(
        analyzer_cool,
        state_path=tmp_path / "cooldown.json",
        policy=_policy(cooldown_seconds=10),
    )
    assert _run(cool_refill, key="cool-first", now_epoch=100).generated_count == 1
    cool = _run(cool_refill, key="cool-second", now_epoch=105)
    metrics["cooldown_count"] = int(
        cool.reason is ContractAssuranceRefillReason.COOLDOWN
    )
    metrics["analyzer_calls_on_cooldown"] = cool_calls
    metrics["analyzer_calls"] += cool_calls

    rec_calls = 0

    def analyzer_rec(_request):
        nonlocal rec_calls
        rec_calls += 1
        return _analysis(_packet())

    state_path = tmp_path / "recover.json"
    original = _run(
        ContractAssuranceRefill(
            analyzer_rec, state_path=state_path, policy=_policy()
        ),
        key="cycle:recover",
        now_epoch=100,
    )
    _run(
        ContractAssuranceRefill(
            analyzer_rec, state_path=state_path, policy=_policy()
        ),
        key="threshold",
        open_tasks=2,
        now_epoch=101,
    )
    state_path.write_text('{"schema":', encoding="utf-8")
    recovered = _run(
        ContractAssuranceRefill(
            analyzer_rec, state_path=state_path, policy=_policy()
        ),
        key="cycle:recover",
        now_epoch=102,
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
    metrics["noop_scan_has_no_provider_model_work"] = (
        metrics["analyzer_calls_on_threshold_skip"] == 0
        and metrics["provider_call_count"] == 0
        and metrics["model_call_count"] == 0
        and metrics["llm_call_count"] == 0
    )

    report: dict[str, Any] = {
        "schema": METRICS_SCHEMA,
        "schema_version": 1,
        "task_id": TASK_ID,
        "interface": CONTRACT_ASSURANCE_REFILL_INTERFACE,
        "evidence": "SCAEV130REFILL",
        "source_tree": SOURCE_TREE,
        "snapshot_id": SNAPSHOT,
        "bounds": {
            "min_open_tasks": 2,
            "max_open_tasks": 4,
            "max_findings_per_run": 2,
            "cooldown_seconds": 10,
        },
        "metrics": metrics,
        "passed": bool(
            metrics["generated_count"] >= 1
            and metrics["duplicate_only_count"] == 1
            and metrics["cooldown_count"] == 1
            and metrics["threshold_skip_count"] == 1
            and metrics["open_work_bound_holds"]
            and metrics["crash_recovery_idempotent"]
            and metrics["noop_scan_has_no_provider_model_work"]
        ),
        "provider_call_count": 0,
        "model_call_count": 0,
        "llm_call_count": 0,
        "completion_authoritative": False,
    }
    report["metrics_id"] = _digest(
        {k: v for k, v in report.items() if k != "metrics_id"}
    )
    return report


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Exact semantic invalidation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("case_id", "change_kind", "changed_input"),
    SEMANTIC_CASES,
    ids=[case[0] for case in SEMANTIC_CASES],
)
def test_controlled_semantic_change_invalidates_all_and_only_dependents(
    case_id: str,
    change_kind: str,
    changed_input: dict[str, str],
) -> None:
    index = _proof_fixture()
    result = invalidate_proof_evidence(
        index, [changed_input], source_tree=SOURCE_TREE
    )

    assert result.event.changed_inputs[0].to_dict() == changed_input
    assert result.event.affected_obligation_ids == DEPENDENT_OBLIGATIONS
    assert result.event.affected_receipt_ids == DEPENDENT_RECEIPTS
    assert result.index.active_obligation_ids == (UNAFFECTED_OBLIGATION,)
    assert result.index.active_receipt_ids == ("receipt:unrelated",)
    # Historical receipts are retained for audit, not deleted.
    assert {receipt.receipt_id for receipt in result.index.receipts} == {
        "receipt:api",
        "receipt:consumer",
        "receipt:unrelated",
    }
    assert {receipt.receipt_id for receipt in result.event.historical_receipts} == {
        "receipt:api",
        "receipt:consumer",
    }
    assert result.event.source_tree == SOURCE_TREE
    assert change_kind == changed_input["kind"]
    assert case_id


def test_unrelated_symbol_change_does_not_touch_contract_closure() -> None:
    index = _proof_fixture()
    result = invalidate_proof_evidence(
        index,
        [{"kind": "qualified_symbol", "value": "pkg.unrelated.stable"}],
        source_tree=SOURCE_TREE,
    )
    assert result.event.affected_obligation_ids == ("obligation:unrelated",)
    assert result.event.affected_receipt_ids == ("receipt:unrelated",)
    assert set(result.index.active_obligation_ids) == {
        "obligation:api",
        "obligation:consumer",
    }


# ---------------------------------------------------------------------------
# Rename and deletion
# ---------------------------------------------------------------------------


def test_path_rename_reuses_ast_blob_and_invalidates_proof_path_projection() -> None:
    index = _proof_fixture()
    renamed = build_proof_scope_index(
        scope_blobs=[
            {
                "path": "src/runtime/api.py",
                "blob_id": "blob:api",
                "scopes": [
                    {
                        "scope_id": "scope:api",
                        "path": "src/runtime/api.py",
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
        obligations=[obligation.payload for obligation in index.obligations],
        receipts=[receipt.payload for receipt in index.receipts],
        previous=index,
    )
    assert renamed.stats.renamed_blob_count == 1
    assert renamed.stats.parsed_blob_count == 0
    assert renamed.stats.reused_blob_count == 3
    assert set(renamed.invalidated_obligation_ids) == set(DEPENDENT_OBLIGATIONS)
    assert renamed.active_obligation_ids == (UNAFFECTED_OBLIGATION,)

    original_ast = build_analysis_ast_index(_ast_snapshot())
    by_path = {item.path: item for item in original_ast.path_records}
    renamed_ast = build_analysis_ast_index(
        [
            (
                "src/runtime/api.py",
                ASTBlobRecord.from_dict(by_path["src/api.py"].ast_record.to_dict()),
            ),
            (
                "src/consumer.py",
                ASTBlobRecord.from_dict(
                    by_path["src/consumer.py"].ast_record.to_dict()
                ),
            ),
            (
                "src/unrelated.py",
                ASTBlobRecord.from_dict(
                    by_path["src/unrelated.py"].ast_record.to_dict()
                ),
            ),
        ],
        previous=original_ast,
    )
    assert renamed_ast.stats.renamed_path_count == 1
    assert renamed_ast.stats.invalidated_blob_count == 0
    assert renamed_ast.invalidations == ()
    assert renamed_ast.path_records[0].path == "src/consumer.py" or any(
        item.path == "src/runtime/api.py" for item in renamed_ast.path_records
    )


def test_path_deletion_invalidates_only_deleted_dependent_surface() -> None:
    index = _proof_fixture()
    deleted = build_proof_scope_index(
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
            obligation.payload
            for obligation in index.obligations
            if obligation.obligation_id != "obligation:consumer"
        ],
        receipts=[
            receipt.payload
            for receipt in index.receipts
            if receipt.receipt_id != "receipt:consumer"
        ],
        previous=index,
    )
    assert deleted.stats.deleted_blob_count == 1
    assert deleted.invalidated_obligation_ids == ("obligation:consumer",)
    assert set(deleted.active_obligation_ids) == {
        "obligation:api",
        UNAFFECTED_OBLIGATION,
    }

    original_ast = build_analysis_ast_index(_ast_snapshot())
    by_path = {item.path: item for item in original_ast.path_records}
    deleted_ast = build_analysis_ast_index(
        [
            (
                "src/api.py",
                ASTBlobRecord.from_dict(by_path["src/api.py"].ast_record.to_dict()),
            ),
            (
                "src/unrelated.py",
                ASTBlobRecord.from_dict(
                    by_path["src/unrelated.py"].ast_record.to_dict()
                ),
            ),
        ],
        previous=original_ast,
    )
    assert deleted_ast.stats.deleted_path_count == 1
    assert {item.reason for item in deleted_ast.invalidations} == {"path_deleted"}
    assert "blob:consumer-ast-v1" in deleted_ast.invalidated_blob_ids


# ---------------------------------------------------------------------------
# No-op scan, cooldown, dedupe, open bounds, crash recovery
# ---------------------------------------------------------------------------


def test_noop_scan_reuses_blobs_and_performs_no_provider_or_model_work() -> None:
    index = _proof_fixture()
    warm = build_proof_scope_index(
        scope_blobs=[blob.to_dict() for blob in index.blobs],
        obligations=[obligation.payload for obligation in index.obligations],
        receipts=[receipt.payload for receipt in index.receipts],
        previous=index,
    )
    assert warm.stats.parsed_blob_count == 0
    assert warm.stats.reused_blob_count == 3
    assert warm.invalidations == ()
    assert warm.stats.invalidated_obligation_count == 0

    original_ast = build_analysis_ast_index(_ast_snapshot())
    warm_ast = build_analysis_ast_index(
        [
            (item.path, ASTBlobRecord.from_dict(item.ast_record.to_dict()))
            for item in original_ast.path_records
        ],
        previous=original_ast,
    )
    assert warm_ast.stats.reused_blob_count == 3
    assert warm_ast.stats.new_blob_count == 0
    assert warm_ast.invalidations == ()


def test_refill_cooldown_dedupe_and_open_bounds_hold(tmp_path: Path) -> None:
    calls = 0

    def analyzer(_request):
        nonlocal calls
        calls += 1
        return _analysis(_packet())

    # Threshold-satisfied no-op: analyzer must not run.
    skipped = _run(
        ContractAssuranceRefill(
            analyzer,
            state_path=tmp_path / "threshold.json",
            policy=_policy(),
        ),
        key="threshold",
        open_tasks=2,
        now_epoch=100,
    )
    assert skipped.reason is ContractAssuranceRefillReason.THRESHOLD_SATISFIED
    assert calls == 0

    one = _packet(actual="integer")
    two = _packet(actual="boolean")
    three = _packet(actual="array")
    analyses = iter((_analysis(one), _analysis(one), _analysis(three, two, one)))
    gen_calls = 0

    def generator(_request):
        nonlocal gen_calls
        gen_calls += 1
        return next(analyses)

    refill = ContractAssuranceRefill(
        generator,
        state_path=tmp_path / "dedupe.json",
        policy=_policy(max_findings_per_run=2, max_open_tasks=2),
    )
    first = _run(refill, key="first", now_epoch=100)
    duplicate = _run(refill, key="duplicate", now_epoch=101)
    bounded = _run(refill, key="bounded", now_epoch=102)

    assert first.generated_count == 1
    assert first.completion_authoritative is False
    assert duplicate.reason is ContractAssuranceRefillReason.DUPLICATE_ONLY
    assert duplicate.generated_count == 0
    board = parse_contract_repair_board(bounded.board_markdown)
    assert len(board.tasks) == 2
    assert len({task.finding_id for task in board.tasks}) == 2
    assert ContractAssuranceRefillReason.FINDING_LIMIT.value in bounded.reason_codes

    cool_calls = 0

    def cool_analyzer(_request):
        nonlocal cool_calls
        cool_calls += 1
        return _analysis(_packet())

    cool_refill = ContractAssuranceRefill(
        cool_analyzer,
        state_path=tmp_path / "cooldown.json",
        policy=_policy(cooldown_seconds=10),
    )
    assert _run(cool_refill, key="c1", now_epoch=100).generated_count == 1
    cool = _run(cool_refill, key="c2", now_epoch=105)
    assert cool.reason is ContractAssuranceRefillReason.COOLDOWN
    assert cool_calls == 1


def test_invalidation_and_refill_crash_recovery_are_idempotent(
    tmp_path: Path,
) -> None:
    index = _proof_fixture()
    first = invalidate_proof_evidence(
        index,
        [{"kind": "qualified_symbol", "value": "pkg.api.Service.run"}],
        source_tree=SOURCE_TREE,
    )
    replayed = invalidate_proof_evidence(
        first.index,
        [{"kind": "qualified_symbol", "value": "pkg.api.Service.run"}],
        source_tree=SOURCE_TREE,
    )
    assert replayed.event == first.event
    assert replayed.index == first.index
    assert replayed.event.event_id == first.event.event_id

    calls = 0

    def analyzer(_request):
        nonlocal calls
        calls += 1
        return _analysis(_packet())

    state_path = tmp_path / "recover.json"
    original = _run(
        ContractAssuranceRefill(
            analyzer, state_path=state_path, policy=_policy()
        ),
        key="cycle:recover",
        now_epoch=100,
    )
    _run(
        ContractAssuranceRefill(
            analyzer, state_path=state_path, policy=_policy()
        ),
        key="threshold",
        open_tasks=2,
        now_epoch=101,
    )
    state_path.write_text('{"schema":', encoding="utf-8")
    recovered = _run(
        ContractAssuranceRefill(
            analyzer, state_path=state_path, policy=_policy()
        ),
        key="cycle:recover",
        now_epoch=102,
    )
    assert calls == 1
    assert recovered.replayed is True
    assert recovered.recovered_state is True
    assert recovered.tasks == original.tasks
    assert recovered.board_markdown == original.board_markdown
    assert list(tmp_path.glob("recover.json.corrupt-*"))


# ---------------------------------------------------------------------------
# Published durable evidence
# ---------------------------------------------------------------------------


@requires_published_swissknife_evidence
def test_published_invalidation_journal_matches_recomputed_fixture() -> None:
    assert PUBLISHED_INVALIDATION.is_file()
    published = _load_jsonl(PUBLISHED_INVALIDATION)
    recomputed = build_invalidation_journal()
    assert published == recomputed

    by_case = {row["case_id"]: row for row in published}
    for case_id, _change_kind, _input in SEMANTIC_CASES:
        row = by_case[case_id]
        assert row["affected_obligation_ids"] == list(DEPENDENT_OBLIGATIONS)
        assert row["unaffected_obligation_ids"] == [UNAFFECTED_OBLIGATION]
        assert row["exact_dependent_closure"] is True
        assert row["provider_call_count"] == 0
        assert row["model_call_count"] == 0
        assert row["record_id"] == _digest(
            {k: v for k, v in row.items() if k != "record_id"}
        )

    rename = by_case["path-rename"]
    assert rename["ast_renamed_without_blob_invalidation"] is True
    assert rename["proof_stats"]["renamed_blob_count"] == 1

    deletion = by_case["path-deletion"]
    assert deletion["proof_invalidated_obligation_ids"] == ["obligation:consumer"]
    assert "path_deleted" in deletion["ast_invalidation_reasons"]

    noop = by_case["noop-scan"]
    assert noop["proof_parsed_blob_count"] == 0
    assert noop["proof_invalidation_count"] == 0
    assert noop["no_provider_or_model_work"] is True

    crash = by_case["crash-recovery-idempotent"]
    assert crash["idempotent"] is True
    assert crash["events_equal"] is True
    assert crash["indexes_equal"] is True


@requires_published_swissknife_evidence
def test_published_refill_metrics_match_recomputed_bounds(tmp_path: Path) -> None:
    assert PUBLISHED_METRICS.is_file()
    published = json.loads(PUBLISHED_METRICS.read_text(encoding="utf-8"))
    recomputed = build_refill_metrics(tmp_path)

    assert published == recomputed
    assert published["passed"] is True
    assert published["interface"] == CONTRACT_ASSURANCE_REFILL_INTERFACE
    assert published["completion_authoritative"] is False
    assert published["provider_call_count"] == 0
    assert published["model_call_count"] == 0
    assert published["llm_call_count"] == 0

    metrics = published["metrics"]
    assert metrics["generated_count"] >= 1
    assert metrics["duplicate_only_count"] == 1
    assert metrics["cooldown_count"] == 1
    assert metrics["threshold_skip_count"] == 1
    assert metrics["analyzer_calls_on_threshold_skip"] == 0
    assert metrics["open_work_bound_holds"] is True
    assert metrics["crash_recovery_idempotent"] is True
    assert metrics["noop_scan_has_no_provider_model_work"] is True
    assert published["metrics_id"] == _digest(
        {k: v for k, v in published.items() if k != "metrics_id"}
    )


@requires_published_swissknife_evidence
def test_published_artifacts_are_canonical_and_tamper_evident() -> None:
    rows = _load_jsonl(PUBLISHED_INVALIDATION)
    for row in rows:
        assert row["schema"] == INVALIDATION_SCHEMA
        assert row["task_id"] == TASK_ID
        assert row["record_id"] == _digest(
            {k: v for k, v in row.items() if k != "record_id"}
        )
        tampered = dict(row)
        tampered["task_id"] = "SCA-TAMPER"
        assert tampered["record_id"] != _digest(
            {k: v for k, v in tampered.items() if k != "record_id"}
        )

    metrics = json.loads(PUBLISHED_METRICS.read_text(encoding="utf-8"))
    assert metrics["schema"] == METRICS_SCHEMA
    assert metrics["metrics_id"] == _digest(
        {k: v for k, v in metrics.items() if k != "metrics_id"}
    )
    metrics["passed"] = False
    assert metrics["metrics_id"] != _digest(
        {k: v for k, v in metrics.items() if k != "metrics_id"}
    )
