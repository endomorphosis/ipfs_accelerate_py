from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import subprocess
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    open_duckdb_connection,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
    GOAL_COMPLETION_AUTHORITY_SPEC_SCHEMA,
    GOAL_ROOT_COMPLETION_GATE_SCHEMA,
    GOAL_RUNTIME_SETTLEMENT_BINDING_SCHEMA,
    GOAL_TERMINAL_REPORT_CONTRACT_SCHEMA,
    GOAL_TERMINAL_REPORT_EVIDENCE_SCHEMA,
    IntentRepository,
    IntentRepositoryError,
    IntentRepositoryIntegrityError,
    goal_authority_projection_on_connection,
    open_intent_repository,
)


GOAL_ALIASES = (
    "VRIF-G000",
    "VRIF-G010",
    "VRIF-G011",
    "VRIF-G020",
    "VRIF-G021",
    "VRIF-G030",
    "VRIF-G031",
    "VRIF-G040",
    "VRIF-G041",
)
PARENTS = {
    "VRIF-G010": "VRIF-G000",
    "VRIF-G011": "VRIF-G010",
    "VRIF-G020": "VRIF-G000",
    "VRIF-G021": "VRIF-G020",
    "VRIF-G030": "VRIF-G000",
    "VRIF-G031": "VRIF-G030",
    "VRIF-G040": "VRIF-G000",
    "VRIF-G041": "VRIF-G040",
}
DEPENDENCIES = {
    "VRIF-G020": "VRIF-G010",
    "VRIF-G021": "VRIF-G011",
    "VRIF-G030": "VRIF-G020",
    "VRIF-G031": "VRIF-G021",
    "VRIF-G040": "VRIF-G030",
    "VRIF-G041": "VRIF-G031",
}
TASK_GROUPS = {
    "VRIF-G011": range(0, 9),
    "VRIF-G021": range(9, 16),
    "VRIF-G031": range(16, 28),
    "VRIF-G041": range(28, 33),
}
COMPLETION_POLICY_FIELDS = (
    "all_task_dependencies_terminal_required",
    "goal_completion_contracts_required",
    "current_tree_required",
    "active_mutating_claims_empty_required",
    "merge_queue_settled_required",
    "blocking_obligations_empty_required",
    "required_receipts_and_seals_verify",
    "non_success_terminals_never_report_success",
    "ducklake_outage_cannot_block_core_completion",
    "final_report_required",
)
ROOT = Path(__file__).resolve().parents[3]
OPERATOR_PATH = ROOT / "scripts/run_agent_supervisor_residual_intelligence.py"
CONFIG_PATH = ROOT / "config/agent_supervisor_residual_intelligence_scheduler.json"
TASKBOARD_PATH = ROOT / "docs/architecture/agent_supervisor_residual_intelligence.todo.md"
TERMINAL_REPORT_PATHS = (
    "docs/architecture/residual_intelligence_inventory/final_release_report.json",
    "docs/architecture/residual_intelligence_inventory/final_release_report.md",
)


def _sha256_identity(value: object) -> str:
    payload = value if isinstance(value, bytes) else json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _operator():
    spec = importlib.util.spec_from_file_location("vrif_goal_operator", OPERATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _goal_cid(alias: str) -> str:
    return f"goal:vrif:{alias.removeprefix('VRIF-G')}"


def _task_cid(ordinal: int) -> str:
    digest = hashlib.sha256(f"VRIF-{ordinal:03d}".encode("utf-8")).digest()
    return "baguqeera" + base64.b32encode(digest).decode("ascii").lower().rstrip("=")


@lru_cache(maxsize=1)
def _sealed_task_fields() -> dict[str, dict[str, str]]:
    operator = _operator()
    from ipfs_accelerate_py.agent_supervisor.task_sources.todo_vector_index import (
        parse_todo_blocks,
    )

    return {
        task_alias: {
            key: operator._metadata_value(value) for key, value in fields.items()
        }
        for task_alias, _title, _source_line, fields in parse_todo_blocks(
            TASKBOARD_PATH.read_text(encoding="utf-8"),
            task_header_prefix="## VRIF-",
        )
    }


def _task_dependency_aliases(task_alias: str) -> list[str]:
    value = _sealed_task_fields()[task_alias].get("depends_on", "")
    return [item.strip() for item in value.split(",") if item.strip()]


def _portal_control_receipt(
    ordinal: int,
    *,
    implementation_commit: str | None = None,
) -> tuple[dict[str, object], str]:
    evidence_digest = "sha256:" + hashlib.sha256(
        f"portal-evidence:{ordinal}".encode("utf-8")
    ).hexdigest()
    attempt_id = f"attempt:vrif:{ordinal:03d}"
    task_cid = _task_cid(ordinal)
    implementation_commit = implementation_commit or hashlib.sha1(
        f"implementation:{ordinal}".encode("utf-8")
    ).hexdigest()
    portal_completion_binding: dict[str, object] = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "database-portal-completion-binding@1"
        ),
        "task_cid": task_cid,
        "attempt_id": attempt_id,
        "binding_id": "sha256:" + hashlib.sha256(
            f"binding:{ordinal}".encode("utf-8")
        ).hexdigest(),
        "portal_receipt_id": "sha256:" + hashlib.sha256(
            f"portal-receipt:{ordinal}".encode("utf-8")
        ).hexdigest(),
        "evidence_digest": evidence_digest,
        "baseline_commit": "d" * 40,
        "baseline_tree": "e" * 40,
        "implementation_commit": implementation_commit,
        "completion_event_id": "sha256:" + hashlib.sha256(
            f"completion-event:{ordinal}".encode("utf-8")
        ).hexdigest(),
    }
    portal_completion_binding["receipt_id"] = "sha256:" + hashlib.sha256(
        json.dumps(
            portal_completion_binding,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    validation = {
        "outcome": "passed",
        "evidence_digest": evidence_digest,
        "argv": ["portal-supervisor-gates"],
        "validator": "DatabasePortalExecutionBridge@1",
        "task_cid": task_cid,
        "attempt_id": attempt_id,
        "portal_receipt_id": portal_completion_binding["portal_receipt_id"],
        "portal_completion_binding": portal_completion_binding,
    }
    return (
        {
            "operation": "database_complete",
            "attempt_id": attempt_id,
            "claim_id": f"claim:vrif:{ordinal:03d}",
            "lease_id": f"lease:vrif:{ordinal:03d}",
            "owner_session_id": "session:pytest",
            "fencing_token": 1,
            "fence_epoch": 1,
            "evidence_digest": evidence_digest,
            "coordination_preparation": {"prepared": True},
            "validation": validation,
        },
        evidence_digest,
    )


def _terminal_control_receipt() -> tuple[dict[str, object], str]:
    return _portal_control_receipt(32, implementation_commit="c" * 40)


def _terminal_completion_binding() -> dict[str, object]:
    control_receipt, evidence_digest = _terminal_control_receipt()
    revision = 2
    completion_evidence_digest = content_identity(
        {
            "task_cid": _task_cid(32),
            "revision": revision,
            "receipt": control_receipt,
            "evidence_digests": [evidence_digest],
        }
    )
    return {
        "task_revision": revision,
        "completion_receipt_cid": content_identity(
            {
                "namespace": "completion-receipt",
                "task_cid": _task_cid(32),
                "revision": revision,
                "evidence_digest": completion_evidence_digest,
            }
        ),
        "completion_evidence_digest": completion_evidence_digest,
    }


def _specification() -> dict[str, object]:
    from ipfs_accelerate_py.agent_supervisor.validation.validation_commands import (
        split_validation_commands,
    )

    goals = [
        {
            "goal_cid": _goal_cid(alias),
            "goal_alias": alias,
            "parent_goal_cid": _goal_cid(PARENTS[alias]) if alias in PARENTS else "",
            "ordinal": ordinal,
        }
        for ordinal, alias in enumerate(GOAL_ALIASES, start=1)
    ]
    edges = [
        {
            "parent_goal_cid": _goal_cid(parent),
            "child_goal_cid": _goal_cid(child),
            "edge_kind": "goal_parent",
        }
        for child, parent in PARENTS.items()
    ]
    edges.extend(
        {
            "parent_goal_cid": _goal_cid(dependency),
            "child_goal_cid": _goal_cid(goal),
            "edge_kind": "goal_dependency",
        }
        for goal, dependency in DEPENDENCIES.items()
    )
    owner_by_task = {
        ordinal: goal_alias
        for goal_alias, ordinals in TASK_GROUPS.items()
        for ordinal in ordinals
    }
    tasks = [
        {
            "task_cid": _task_cid(ordinal),
            "task_alias": f"VRIF-{ordinal:03d}",
            "goal_cid": _goal_cid(owner_by_task[ordinal]),
        }
        for ordinal in range(33)
    ]
    task_dependencies = sorted(
        (
            {
                "task_cid": _task_cid(ordinal),
                "dependency_task_cid": _task_cid(
                    int(dependency_alias.removeprefix("VRIF-"))
                ),
                "kind": "depends_on",
            }
            for ordinal in range(33)
            for dependency_alias in _task_dependency_aliases(f"VRIF-{ordinal:03d}")
        ),
        key=lambda item: (
            item["task_cid"],
            item["dependency_task_cid"],
            item["kind"],
        ),
    )
    assert len(task_dependencies) == 111
    terminal_fields = _sealed_task_fields()["VRIF-032"]
    producer_output_paths = {
        task_alias: [
            item.strip()
            for item in _sealed_task_fields()[task_alias]["predicted_files"].split(",")
            if item.strip()
        ]
        for task_alias in ("VRIF-028", "VRIF-029", "VRIF-030", "VRIF-031")
    }
    producer_validation_commands = {
        task_alias: [
            [command]
            for command in split_validation_commands(
                _sealed_task_fields()[task_alias]["validation"]
            )
        ]
        for task_alias in ("VRIF-028", "VRIF-029", "VRIF-030", "VRIF-031")
    }
    declared_output_paths = [
        item.strip()
        for item in terminal_fields["predicted_files"].split(",")
        if item.strip()
    ]
    terminal_contract: dict[str, object] = {
        "schema": GOAL_TERMINAL_REPORT_CONTRACT_SCHEMA,
        "task_cid": _task_cid(32),
        "task_alias": "VRIF-032",
        "declared_output_paths": declared_output_paths,
        "declared_symbols": [
            item.strip()
            for item in terminal_fields["predicted_symbols"].replace(";", ",").split(",")
            if item.strip()
        ],
        "required_report_paths": list(TERMINAL_REPORT_PATHS),
        "producer_output_paths": producer_output_paths,
        "producer_validation_commands": producer_validation_commands,
        "acceptance_criteria": [terminal_fields["acceptance_subset"]],
        "validation_commands": [[terminal_fields["validation"]]],
    }
    terminal_contract["contract_id"] = content_identity(terminal_contract)
    spec: dict[str, object] = {
        "schema": GOAL_COMPLETION_AUTHORITY_SPEC_SCHEMA,
        "board_namespace": "agent-supervisor-verified-residual-intelligence-foundry-v1",
        "goal_count": 9,
        "task_count": 33,
        "root_goal_cid": _goal_cid("VRIF-G000"),
        "root_goal_alias": "VRIF-G000",
        "goals": goals,
        "goal_edges": sorted(
            edges,
            key=lambda item: (
                item["edge_kind"],
                item["parent_goal_cid"],
                item["child_goal_cid"],
            ),
        ),
        "tasks": tasks,
        "task_dependencies": task_dependencies,
        "terminal_report_contract": terminal_contract,
        "completion_policy": {
            **{field: True for field in COMPLETION_POLICY_FIELDS},
            "terminal_task_id": "VRIF-032",
        },
        "receipt_backfill_goal_cids": [
            _goal_cid("VRIF-G010"),
            _goal_cid("VRIF-G011"),
        ],
    }
    spec["authority_spec_id"] = content_identity(spec)
    return spec


def _producer_artifacts(
    specification: dict[str, object],
    *,
    content_for_path: Callable[[str], bytes] | None = None,
) -> dict[str, object]:
    contract = specification["terminal_report_contract"]
    assert isinstance(contract, dict)

    producer_artifact_tasks: list[dict[str, object]] = []
    for producer_alias, paths in sorted(contract["producer_output_paths"].items()):
        task_bundle: dict[str, object] = {
            "task_alias": producer_alias,
            "artifacts": [
                {
                    "path": path,
                    "blob_identity": _sha256_identity(
                        content_for_path(path)
                        if content_for_path is not None
                        else f"artifact:{path}".encode("utf-8")
                    ),
                }
                for path in sorted(paths)
            ],
        }
        task_bundle["bundle_id"] = _sha256_identity(task_bundle)
        producer_artifact_tasks.append(task_bundle)
    result: dict[str, object] = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "goal-terminal-producer-artifacts@1"
        ),
        "digest_algorithm": "sha256",
        "tasks": producer_artifact_tasks,
    }
    result["bundle_id"] = _sha256_identity(result)
    return result


def _runtime_settlement_binding(
    *,
    owner_generation: int = 1,
    variant: int = 0,
) -> dict[str, object]:
    def digest(offset: int) -> str:
        return "sha256:" + (f"{variant + offset:x}"[-1] * 64)

    binding: dict[str, object] = {
        "schema": GOAL_RUNTIME_SETTLEMENT_BINDING_SCHEMA,
        "settled": True,
        "receipt_cid": digest(1),
        "snapshot_cid": digest(2),
        "owner_generation": owner_generation,
        "target": {
            "binding_schema": (
                "ipfs_accelerate_py/agent-supervisor/merge-target-binding@1"
            ),
            "repository_id": (
                "repository:baguqeera"
                "ul4vqj7wze6dfjxogue57aadnvwrzw55527c2kfafyiyvuoaw2ca"
            ),
            "branch": "codex/verified-residual-intelligence-foundry-v1",
        },
        "config_cid": digest(3),
        "profile_cid": digest(4),
        "lane_snapshot_cids": [digest(index) for index in range(5, 9)],
        "merge_queue_receipt_cid": digest(9),
        "merge_queue_snapshot_cid": digest(10),
        "active_counts": {
            "coordination": 0,
            "execution": 0,
            "merge_queue": 0,
            "total": 0,
        },
        "retired_ready_task_cids": sorted(
            _task_cid(index) for index in (13, 14, 15)
        ),
    }
    binding["binding_id"] = _sha256_identity(binding)
    return binding


def _root_gate(
    specification: dict[str, object],
    connection: object,
) -> dict[str, object]:
    terminal_binding = _terminal_completion_binding()
    control_receipt, _evidence_digest = _terminal_control_receipt()
    lineage_rows = connection.execute(
        "SELECT vr.run_id, result.result_id "
        "FROM validation_runs AS vr "
        "JOIN validation_results AS result ON result.run_id = vr.run_id "
        "WHERE vr.task_cid = ? AND result.task_cid = ?",
        [_task_cid(32), _task_cid(32)],
    ).fetchall()
    assert len(lineage_rows) == 1
    validation_run_id = str(lineage_rows[0][0])
    validation_result_id = str(lineage_rows[0][1])
    validation_evidence_id = content_identity(
        {
            "task_cid": _task_cid(32),
            "evidence_kind": "validation",
            "digest": control_receipt["evidence_digest"],
            "run_id": validation_run_id,
        }
    )
    contract = specification["terminal_report_contract"]
    assert isinstance(contract, dict)
    producer_rows = connection.execute(
        "SELECT t.task_alias, receipt.receipt_cid "
        "FROM tasks AS t JOIN completion_receipts AS receipt "
        "ON receipt.task_cid = t.task_cid "
        "WHERE t.task_alias IN (?, ?, ?, ?) ORDER BY t.task_alias",
        ["VRIF-028", "VRIF-029", "VRIF-030", "VRIF-031"],
    ).fetchall()
    assert len(producer_rows) == 4
    producer_receipts = {str(row[0]): str(row[1]) for row in producer_rows}
    producer_artifacts = _producer_artifacts(specification)
    artifact_task_by_alias = {
        str(item["task_alias"]): item for item in producer_artifacts["tasks"]
    }
    producer_receipt_bindings: list[dict[str, object]] = []
    for producer_alias in sorted(producer_receipts):
        ordinal = int(producer_alias.removeprefix("VRIF-"))
        producer_control_receipt, _ = _portal_control_receipt(ordinal)
        producer_validation = producer_control_receipt["validation"]
        assert isinstance(producer_validation, dict)
        binding: dict[str, object] = {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "goal-terminal-producer-receipt-binding@1"
            ),
            "task_alias": producer_alias,
            "task_cid": _task_cid(ordinal),
            "completion_receipt_cid": producer_receipts[producer_alias],
            "portal_completion_binding": producer_validation[
                "portal_completion_binding"
            ],
            "artifact_bundle_id": artifact_task_by_alias[producer_alias][
                "bundle_id"
            ],
        }
        binding["binding_id"] = _sha256_identity(binding)
        producer_receipt_bindings.append(binding)
    terminal_evidence: dict[str, object] = {
        "schema": GOAL_TERMINAL_REPORT_EVIDENCE_SCHEMA,
        "terminal_report_contract_id": contract["contract_id"],
        "task_cid": _task_cid(32),
        "task_alias": "VRIF-032",
        "task_revision": terminal_binding["task_revision"],
        "completion_receipt_cid": terminal_binding["completion_receipt_cid"],
        "completion_evidence_digest": terminal_binding[
            "completion_evidence_digest"
        ],
        "control_receipt_id": content_identity(control_receipt),
        "portal_receipt_id": control_receipt["validation"]["portal_receipt_id"],
        "portal_completion_binding": control_receipt["validation"][
            "portal_completion_binding"
        ],
        "producer_receipts": producer_receipts,
        "validation_run_id": validation_run_id,
        "validation_result_id": validation_result_id,
        "validation_evidence_id": validation_evidence_id,
        "report_artifacts": [
            {
                "path": path,
                "blob_identity": "sha256:" + (str(index + 1) * 64),
                "bootstrap_blob_identity": "sha256:" + (str(index + 3) * 64),
            }
            for index, path in enumerate(TERMINAL_REPORT_PATHS)
        ],
        "producer_artifacts": producer_artifacts,
        "producer_receipt_bindings": producer_receipt_bindings,
    }
    terminal_evidence["evidence_id"] = content_identity(terminal_evidence)
    gate: dict[str, object] = {
        "schema": GOAL_ROOT_COMPLETION_GATE_SCHEMA,
        "authority_spec_id": specification["authority_spec_id"],
        "source_head": "1" * 40,
        "repository_tree_id": "2" * 40,
        "predecessor_gate_id": "",
        "owner_generation": 1,
        "owner_restart_admission_id": "sha256:" + ("3" * 64),
        "owner_restart_receipt_id": "sha256:" + ("4" * 64),
        "completion_policy": dict(specification["completion_policy"]),
        "runtime_settlement_binding": _runtime_settlement_binding(),
        "terminal_report_evidence": terminal_evidence,
    }
    gate["gate_id"] = content_identity(gate)
    return gate


def _renewed_root_gate(gate: dict[str, object]) -> dict[str, object]:
    renewed = dict(gate)
    renewed.pop("gate_id")
    renewed["source_head"] = "a" * 40
    renewed["repository_tree_id"] = "b" * 40
    renewed["predecessor_gate_id"] = gate["gate_id"]
    renewed["owner_generation"] = int(gate["owner_generation"]) + 1
    renewed["runtime_settlement_binding"] = _runtime_settlement_binding(
        owner_generation=int(renewed["owner_generation"]),
        variant=1,
    )
    renewed["owner_restart_admission_id"] = "sha256:" + ("6" * 64)
    renewed["owner_restart_receipt_id"] = "sha256:" + ("7" * 64)
    renewed["gate_id"] = content_identity(renewed)
    return renewed


def _seed(
    path: Path,
    *,
    skipped_task: int | None = None,
    arbitrary_terminal_receipt: bool = False,
) -> None:
    specification = _specification()
    with open_intent_repository(path, owner_id="owner:goal-authority-seed") as repository:
        for goal in specification["goals"]:
            alias = str(goal["goal_alias"])
            repository.upsert_goal(
                goal_cid=str(goal["goal_cid"]),
                goal_alias=alias,
                parent_goal_cid=str(goal["parent_goal_cid"]),
                ordinal=int(goal["ordinal"]),
                title=alias,
                status=("completed" if alias in {"VRIF-G010", "VRIF-G011"} else "waiting"),
            )
        for edge in specification["goal_edges"]:
            repository.link_goal_edge(
                parent_goal_cid=str(edge["parent_goal_cid"]),
                child_goal_cid=str(edge["child_goal_cid"]),
                edge_kind=str(edge["edge_kind"]),
            )
        for ordinal, task in enumerate(specification["tasks"]):
            dependencies = [
                str(item["dependency_task_cid"])
                for item in specification["task_dependencies"]
                if item["task_cid"] == task["task_cid"]
            ]
            terminal_contract = specification["terminal_report_contract"]
            assert isinstance(terminal_contract, dict)
            terminal = ordinal == 32
            producer_paths = terminal_contract["producer_output_paths"].get(
                str(task["task_alias"])
            )
            repository.upsert_task(
                task_cid=str(task["task_cid"]),
                task_alias=str(task["task_alias"]),
                goal_cid=str(task["goal_cid"]),
                ordinal=ordinal + 1,
                status="ready",
                dependencies=dependencies,
                outputs=(
                    [
                        {"path": path, "effect_id": f"effect:{ordinal}:{index}"}
                        for index, path in enumerate(
                            terminal_contract["declared_output_paths"]
                        )
                    ]
                    if terminal
                    else (
                        [
                            {"path": path, "effect_id": f"effect:{ordinal}:{index}"}
                            for index, path in enumerate(producer_paths)
                        ]
                        if producer_paths
                        else None
                    )
                ),
                acceptance=(
                    list(terminal_contract["acceptance_criteria"])
                    if terminal
                    else None
                ),
                validations=(
                    [list(item) for item in terminal_contract["validation_commands"]]
                    if terminal
                    else (
                        [
                            list(item)
                            for item in terminal_contract[
                                "producer_validation_commands"
                            ].get(str(task["task_alias"]), [])
                        ]
                        if producer_paths
                        else None
                    )
                ),
            )
            current = repository.get_task(str(task["task_cid"]))
            assert current is not None
            if (
                ordinal in {28, 29, 30, 31, 32}
                and ordinal != skipped_task
                and not (terminal and arbitrary_terminal_receipt)
            ):
                control_receipt, evidence_digest = (
                    _terminal_control_receipt()
                    if terminal
                    else _portal_control_receipt(ordinal)
                )
                validation = control_receipt["validation"]
                assert isinstance(validation, dict)
                repository.record_validation_result(
                    task_cid=str(task["task_cid"]),
                    outcome="passed",
                    evidence_digest=evidence_digest,
                    argv=["portal-supervisor-gates"],
                    attempt_id=str(control_receipt["attempt_id"]),
                    body=validation,
                )
                repository.cas_task_status(
                    task_cid=str(task["task_cid"]),
                    expected_revision=int(current["revision"]),
                    new_status="completed",
                    receipt=control_receipt,
                    evidence_digests=[evidence_digest],
                )
                continue
            repository.cas_task_status(
                task_cid=str(task["task_cid"]),
                expected_revision=int(current["revision"]),
                new_status=("skipped" if ordinal == skipped_task else "completed"),
                receipt={"operation": "pytest_completion", "task_alias": task["task_alias"]},
                allow_completion_without_evidence=True,
            )


def _bound_repository(path: Path) -> tuple[object, IntentRepository]:
    connection = open_duckdb_connection(path)
    repository = IntentRepository(
        path,
        bound_connection=connection,
        owner_id="owner:goal-authority",
        session_id="session:goal-authority",
        install_schema=False,
    )
    return connection, repository


def test_operator_reconstructs_exact_nine_goal_authority_from_seal() -> None:
    operator = _operator()
    board, config = operator._load_config(CONFIG_PATH)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "rev-parse", f"{head}^{{tree}}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    plan_root_cid = "plan:pytest-goal-authority"
    objective_text = subprocess.run(
        ["git", "show", f"{head}:{board.objectives_path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    goal_cids = {
        goal_alias: content_identity(
            {
                "goal_id": goal_alias,
                "title": title,
                "metadata": fields,
                "plan_root_cid": plan_root_cid,
            }
        )
        for goal_alias, title, fields in operator._goal_blocks(objective_text)
    }
    task_goal = {
        task_alias: goal_alias
        for goal_alias, task_aliases in config["task_groups"].items()
        for task_alias in task_aliases
    }
    from ipfs_accelerate_py.agent_supervisor.task_sources.todo_vector_index import (
        parse_todo_blocks,
    )

    taskboard_text = subprocess.run(
        ["git", "show", f"{head}:{board.taskboard_path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    task_cids = {
        task_alias: content_identity(
            {
                "task_id": task_alias,
                "title": title,
                "source_line": source_line,
                "metadata": {
                    key: operator._metadata_value(value)
                    for key, value in fields.items()
                },
                "plan_root_cid": plan_root_cid,
                "repository_tree_id": tree,
            }
        )
        for task_alias, title, source_line, fields in parse_todo_blocks(
            taskboard_text,
            task_header_prefix="## VRIF-",
        )
    }
    rows = [
        (
            task_cids[task_alias],
            task_alias,
            goal_cids[task_goal[task_alias]],
        )
        for task_alias in sorted(task_goal)
    ]

    class _Result:
        def fetchall(self):
            return rows

    class _Connection:
        def execute(self, sql, _parameters):
            assert "SELECT task_cid, task_alias, goal_cid FROM tasks" in sql
            return _Result()

    admission = {
        "bootstrap_source_head": head,
        "plan_root_cid": plan_root_cid,
        "database_authority": {
            "repository_tree_id": tree,
            "task_cids": sorted(row[0] for row in rows),
        },
    }
    specification = operator._vrif_goal_completion_authority_spec(
        board,
        json.loads(json.dumps(config)),
        admission,
        _Connection(),
    )
    assert specification["goal_count"] == 9
    assert specification["task_count"] == 33
    assert len(specification["task_dependencies"]) == 111
    assert specification["root_goal_alias"] == "VRIF-G000"
    assert specification["completion_policy"]["terminal_task_id"] == "VRIF-032"
    assert specification["terminal_report_contract"]["required_report_paths"] == list(
        TERMINAL_REPORT_PATHS
    )
    assert [item["goal_alias"] for item in specification["goals"]] == list(
        GOAL_ALIASES
    )
    assert specification["receipt_backfill_goal_cids"] == [
        goal_cids["VRIF-G010"],
        goal_cids["VRIF-G011"],
    ]
    identity_body = dict(specification)
    observed_identity = identity_body.pop("authority_spec_id")
    assert observed_identity == content_identity(identity_body)


def test_owner_reconciles_exact_goals_and_backfills_preseeded_receipts(
    tmp_path: Path,
) -> None:
    path = tmp_path / "control.duckdb"
    specification = _specification()
    _seed(path)
    connection, repository = _bound_repository(path)
    try:
        gate = _root_gate(specification, connection)
        result = repository.reconcile_goal_completion_authority(
            specification,
            root_completion_gate=gate,
        )
        assert result["changed_goal_ids"] == [
            "VRIF-G011",
            "VRIF-G010",
            "VRIF-G021",
            "VRIF-G020",
            "VRIF-G031",
            "VRIF-G030",
            "VRIF-G041",
            "VRIF-G040",
            "VRIF-G000",
        ]
        authority = result["goal_authority"]
        assert authority["all_goals_satisfied"] is True
        assert authority["goal_count"] == 9
        assert authority["incomplete_goal_ids"] == []
        assert authority["invalid_goal_ids"] == []
        assert authority["ready_goal_ids"] == []
        assert authority["root_goal"]["completion_receipt_id"].startswith("baguqeera")
        assert authority["projection_cid"].startswith("baguqeera")
        assert len(authority["task_dependencies"]) == 111
        assert authority["completion_policy"]["terminal_task_id"] == "VRIF-032"
        assert authority["completion_gates"][
            "terminal_report_completion_receipt_satisfied"
        ] is True
        assert authority["completion_gates"]["terminal_report_gate_satisfied"] is True
        assert authority["terminal_report_authority"]["task_alias"] == "VRIF-032"
        assert authority["terminal_report_authority"][
            "production_completion_receipt_satisfied"
        ] is True
        assert authority["terminal_report_authority"]["required_report_paths"] == list(
            TERMINAL_REPORT_PATHS
        )
        assert len(authority["terminal_report_authority"]["report_artifacts"]) == 2

        for alias in ("VRIF-G010", "VRIF-G011"):
            goal = repository.get_goal(alias)
            assert goal is not None
            assert goal["body"]["completion_receipt"]["completion_kind"] == (
                "preseeded_completion_receipt_backfill"
            )
        replay = repository.reconcile_goal_completion_authority(
            specification,
            root_completion_gate=gate,
        )
        assert replay["changed"] is False
        assert replay["goal_authority"]["projection_cid"] == authority["projection_cid"]

        readback = repository.goal_authority_projection(
            specification,
            root_gate_context={
                "current_tree_clean": True,
                "source_head": gate["source_head"],
                "repository_tree_id": gate["repository_tree_id"],
                "runtime_settlement_binding": gate[
                    "runtime_settlement_binding"
                ],
            },
        )
        assert readback["all_goals_satisfied"] is True
        assert readback["projection_cid"] == authority["projection_cid"]

        runtime_mismatch = repository.goal_authority_projection(
            specification,
            root_gate_context={
                "current_tree_clean": True,
                "source_head": gate["source_head"],
                "repository_tree_id": gate["repository_tree_id"],
                "runtime_settlement_binding": _runtime_settlement_binding(
                    variant=1
                ),
            },
        )
        assert runtime_mismatch["all_goals_satisfied"] is False
        assert runtime_mismatch["completion_gates"][
            "runtime_settlement_gate_satisfied"
        ] is False
        assert "completion_gate:runtime_settlement_gate_satisfied" in (
            runtime_mismatch["root_goal"]["incomplete_reasons"]
        )

        repository.rebuild_projections_from_events()
        rebuilt = repository.goal_authority_projection(
            specification,
            root_gate_context={
                "current_tree_clean": True,
                "source_head": gate["source_head"],
                "repository_tree_id": gate["repository_tree_id"],
                "runtime_settlement_binding": gate[
                    "runtime_settlement_binding"
                ],
            },
        )
        assert rebuilt["all_goals_satisfied"] is True
        assert rebuilt["projection_cid"] == authority["projection_cid"]

        root_before = repository.get_goal("VRIF-G000")
        assert root_before is not None
        renewed_gate = _renewed_root_gate(gate)
        renewed = repository.reconcile_goal_completion_authority(
            specification,
            root_completion_gate=renewed_gate,
        )
        assert renewed["changed_goal_ids"] == ["VRIF-G000"]
        assert renewed["goal_authority"]["all_goals_satisfied"] is True
        root_after = repository.get_goal("VRIF-G000")
        assert root_after is not None
        assert root_after["revision"] == root_before["revision"] + 1
        assert root_after["body"]["completion_receipt"]["root_completion_gate"] == (
            renewed_gate
        )

        projected = goal_authority_projection_on_connection(
            connection,
            specification,
            root_gate_context={
                "current_tree_clean": True,
                "source_head": renewed_gate["source_head"],
                "repository_tree_id": renewed_gate["repository_tree_id"],
                "runtime_settlement_binding": renewed_gate[
                    "runtime_settlement_binding"
                ],
            },
        )
        assert projected["all_goals_satisfied"] is True
        stale = goal_authority_projection_on_connection(
            connection,
            specification,
            root_gate_context={
                "current_tree_clean": True,
                "source_head": "5" * 40,
                "repository_tree_id": renewed_gate["repository_tree_id"],
                "runtime_settlement_binding": renewed_gate[
                    "runtime_settlement_binding"
                ],
            },
        )
        assert stale["all_goals_satisfied"] is False
        assert stale["invalid_goal_ids"] == ["VRIF-G000"]

        downgrade = repository.reconcile_goal_completion_authority(
            specification,
            root_completion_gate=gate,
        )
        assert downgrade["changed_goal_ids"] == []
        assert downgrade["goal_authority"]["all_goals_satisfied"] is False
        assert "root_completion_gate_predecessor_or_generation_invalid" in (
            downgrade["goal_authority"]["root_goal"]["incomplete_reasons"]
        )
        stored_after_downgrade = repository.get_goal("VRIF-G000")
        assert stored_after_downgrade is not None
        assert stored_after_downgrade["body"]["completion_receipt"][
            "root_completion_gate"
        ] == renewed_gate
        restored = repository.reconcile_goal_completion_authority(
            specification,
            root_completion_gate=renewed_gate,
        )
        assert restored["changed_goal_ids"] == []
        assert restored["goal_authority"]["all_goals_satisfied"] is True
    finally:
        repository.close()
        connection.close()


def test_skipped_task_never_satisfies_goal_completion(tmp_path: Path) -> None:
    path = tmp_path / "control.duckdb"
    specification = _specification()
    _seed(path, skipped_task=0)
    connection, repository = _bound_repository(path)
    try:
        result = repository.reconcile_goal_completion_authority(
            specification,
            root_completion_gate=_root_gate(specification, connection),
        )
        authority = result["goal_authority"]
        assert authority["all_goals_satisfied"] is False
        assert "VRIF-000" in authority["task_receipt_invalid_ids"]
        assert "VRIF-G011" in authority["invalid_goal_ids"]
        assert "VRIF-G010" in authority["invalid_goal_ids"]
        assert "VRIF-G000" in authority["incomplete_goal_ids"]
        assert result["changed_goal_ids"] == []
    finally:
        repository.close()
        connection.close()


def test_deleted_sealed_task_dependency_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "control.duckdb"
    specification = _specification()
    _seed(path)
    connection, repository = _bound_repository(path)
    try:
        deleted = specification["task_dependencies"][0]
        gate = _root_gate(specification, connection)
        connection.execute(
            "DELETE FROM task_dependencies "
            "WHERE task_cid = ? AND dependency_task_cid = ? AND kind = ?",
            [
                deleted["task_cid"],
                deleted["dependency_task_cid"],
                deleted["kind"],
            ],
        )
        with pytest.raises(
            IntentRepositoryIntegrityError,
            match="task dependencies differ from exact completion authority",
        ):
            repository.reconcile_goal_completion_authority(
                specification,
                root_completion_gate=gate,
            )
    finally:
        repository.close()
        connection.close()


def test_arbitrary_vrif_032_receipt_and_bootstrap_reports_keep_root_open(
    tmp_path: Path,
) -> None:
    path = tmp_path / "control.duckdb"
    specification = _specification()
    _seed(path, arbitrary_terminal_receipt=True)
    connection, repository = _bound_repository(path)
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        operator = _operator()
        assert operator._vrif_terminal_report_evidence(
            specification,
            {
                "bootstrap_source_head": head,
                "current_source_head": head,
            },
            connection,
        ) is None
        result = repository.reconcile_goal_completion_authority(
            specification,
            root_completion_gate=None,
        )
        authority = result["goal_authority"]
        assert result["changed_goal_ids"][-1] == "VRIF-G040"
        assert "VRIF-G000" not in result["changed_goal_ids"]
        assert authority["all_goals_satisfied"] is False
        assert authority["completion_gates"][
            "terminal_report_completion_receipt_satisfied"
        ] is False
        assert authority["completion_gates"]["terminal_report_gate_satisfied"] is False
        assert authority["terminal_report_authority"][
            "production_completion_receipt_satisfied"
        ] is False
        assert authority["terminal_report_authority"]["report_artifacts"] == []
        assert "completion_gate:terminal_report_gate_satisfied" in (
            authority["root_goal"]["incomplete_reasons"]
        )
    finally:
        repository.close()
        connection.close()


@pytest.mark.parametrize(
    "tamper",
    [
        "canonical",
        "lineage",
        "files_symbols",
        "corpus_rights_splits",
        "architecture_tokenizer_checkpoint",
        "expert_dispositions",
        "denominators",
        "costs",
        "proof_validation",
        "drift",
        "rollback_blockers",
        "incomplete_not_run",
        "omitted_section",
        "meaningless_markdown",
        "extra_changed_path",
        "producer_rewrite",
        "missing_family",
        "missing_required_kind",
        "hidden_training_case",
        "case_group_lineage",
        "case_input_identity",
        "case_expected_outcome",
        "benchmark_freeze_missing",
        "benchmark_binding_tamper",
        "fault_schedule_tamper",
        "paired_lineage_tamper",
        "stale_bootstrap_manifest",
        "later_catalog_change",
    ],
)
def test_terminal_report_semantics_require_exact_producer_anchors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    path = tmp_path / "control.duckdb"
    specification = _specification()
    _seed(path)
    connection, repository = _bound_repository(path)
    try:
        gate = _root_gate(specification, connection)
        terminal_evidence = gate["terminal_report_evidence"]
        assert isinstance(terminal_evidence, dict)

        def generic_producer_content(path: str) -> bytes:
            return f"producer:{path}".encode("utf-8")
        manifest = json.loads(
            (
                ROOT
                / "benchmarks/agent_supervisor/residual_intelligence/manifest.json"
            ).read_text(encoding="utf-8")
        )
        admission_payload = json.loads(
            (
                ROOT
                / "benchmarks/agent_supervisor/residual_intelligence/"
                "synthetic_training_admission.json"
            ).read_text(encoding="utf-8")
        )
        baseline = json.loads(
            (
                ROOT
                / "docs/architecture/residual_intelligence_inventory/baseline.json"
            ).read_text(encoding="utf-8")
        )
        split_manifest = json.loads(
            (
                ROOT
                / "benchmarks/agent_supervisor/residual_intelligence/"
                "synthetic_split_manifest.json"
            ).read_text(encoding="utf-8")
        )
        portal_completion_binding = terminal_evidence["portal_completion_binding"]
        assert isinstance(portal_completion_binding, dict)
        evaluated_tree = str(portal_completion_binding["baseline_tree"])
        terminal_contract = specification["terminal_report_contract"]
        assert isinstance(terminal_contract, dict)
        partitions = ["training", "development", "held_out", "adversarial"]
        required_kinds = [
            "boundary",
            "negative",
            "cross_repository",
            "unknown_ood",
        ]
        benchmark_producer_receipt, _ = _portal_control_receipt(30)
        benchmark_producer_binding = benchmark_producer_receipt["validation"][
            "portal_completion_binding"
        ]
        objective_path = (
            "docs/architecture/agent_supervisor_residual_intelligence.objectives.md"
        )
        config_path = "config/agent_supervisor_residual_intelligence_scheduler.json"
        operation_catalog_path = (
            "ipfs_accelerate_py/agent_supervisor/control/control_plane.py"
        )
        validation_policy_path = "test/api/residual_intelligence/test_benchmark.py"
        inventory_path = (
            "docs/architecture/residual_intelligence_inventory/"
            "residual_model_call_inventory.json"
        )
        taskboard_path = (
            "docs/architecture/agent_supervisor_residual_intelligence.todo.md"
        )
        operation_catalog_at_benchmark = (
            b"historical-catalog-before-vrif-029"
            if tamper == "later_catalog_change"
            else generic_producer_content(operation_catalog_path)
        )
        base_frozen_bindings = {
            "repository_states": _sha256_identity(
                {
                    "commit": benchmark_producer_binding["baseline_commit"],
                    "tree": benchmark_producer_binding["baseline_tree"],
                }
            ),
            "objective_revisions": _sha256_identity(
                {
                    "schema": (
                        "ipfs_accelerate_py/agent-supervisor/"
                        "residual-benchmark-objective-revisions@1"
                    ),
                    "artifacts": {
                        objective_path: _sha256_identity(
                            (ROOT / objective_path).read_bytes()
                        ),
                        taskboard_path: _sha256_identity(
                            (ROOT / taskboard_path).read_bytes()
                        ),
                    },
                }
            ),
            "operation_catalog": _sha256_identity(
                operation_catalog_at_benchmark
            ),
            "provider_policy": _sha256_identity((ROOT / config_path).read_bytes()),
            "tokenizer": _sha256_identity(
                {
                    "admission_id": admission_payload["admission_id"],
                    "disposition": "no_learned_tokenizer_admitted",
                }
            ),
            "model_versions": _sha256_identity(
                {
                    "inventory_blob_identity": _sha256_identity(
                        (ROOT / inventory_path).read_bytes()
                    ),
                    "disposition": "training_unavailable",
                }
            ),
            "validation_policy": _sha256_identity(
                {
                    "argv": terminal_contract["producer_validation_commands"][
                        "VRIF-030"
                    ],
                    "test_blob_identity": _sha256_identity(
                        generic_producer_content(validation_policy_path)
                    ),
                }
            ),
        }
        operator = _operator()
        frozen_benchmark = operator._vrif_frozen_benchmark_contract(
            task_families=manifest["task_families"],
            source_commit=benchmark_producer_binding["baseline_commit"],
            source_tree=benchmark_producer_binding["baseline_tree"],
            split_root=split_manifest["split_root"],
            base_bindings=base_frozen_bindings,
        )
        frozen_bindings = frozen_benchmark["bindings"]
        frozen_binding_set_id = frozen_benchmark["binding_set_id"]
        benchmark_cases = frozen_benchmark["cases"]
        benchmark_blob_cases = [dict(item) for item in benchmark_cases]
        benchmark_scores = frozen_benchmark["scores"]
        paired_baseline = frozen_benchmark["paired_baseline"]
        benchmark_freeze = frozen_benchmark["benchmark_freeze"]
        qualified_manifest = {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "residual-intelligence-benchmark-manifest@1"
            ),
            "program_identifier": (
                "agent-supervisor-verified-residual-intelligence-foundry-v1"
            ),
            "status": "staged_not_qualified",
            "owner_task": "VRIF-030",
            "source_revision": benchmark_producer_binding["baseline_commit"],
            "partitions": partitions,
            "required_case_kinds": required_kinds,
            "task_families": manifest["task_families"],
            "training_admission": "training_unavailable",
            "weights_committed": False,
            "large_corpus_committed": False,
            "promotion_evidence": False,
            "benchmark_freeze": benchmark_freeze,
        }
        benchmark_cases_bytes = (
            "\n".join(json.dumps(item, sort_keys=True) for item in benchmark_cases)
            + "\n"
        ).encode("utf-8")
        qualified_manifest_bytes = json.dumps(
            qualified_manifest, sort_keys=True
        ).encode("utf-8")
        benchmark_blob_manifest = json.loads(json.dumps(qualified_manifest))

        def producer_content(path: str) -> bytes:
            if path.endswith("/cases.jsonl"):
                return benchmark_cases_bytes
            if path.endswith("/manifest.json"):
                return qualified_manifest_bytes
            return generic_producer_content(path)

        producer_artifacts = _producer_artifacts(
            specification,
            content_for_path=producer_content,
        )
        exact_not_run = ["gpu_live_qualification", "promotion", "training"]
        exact_blockers = ["training_unavailable"]
        report: dict[str, object] = {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "residual-intelligence-release-report@2"
            ),
            "start_tree": baseline["source"]["tree"],
            "end_tree": evaluated_tree,
            "corpus_admission_id": admission_payload["admission_id"],
            "expert_dispositions": {
                family: "CAPABILITY_UNAVAILABLE"
                for family in manifest["task_families"]
            },
            "before": benchmark_scores,
            "after": benchmark_scores,
            "costs": {"tokens": 0, "break_even": 0},
            "promotion_eligible": False,
            "rollback_target": baseline["source"]["commit"],
            "gaps": {
                "blockers": exact_blockers,
                "unsupported_claims": [
                    "learned",
                    "verified",
                    "safe",
                    "autonomous",
                    "token-efficient",
                    "production-ready",
                ],
                "not_run": exact_not_run,
            },
            "producer_artifacts": producer_artifacts,
            "files_symbols": {
                "disposition": "current_tracked_blobs_bound",
                "declared_output_paths": terminal_contract["declared_output_paths"],
                "required_report_paths": list(TERMINAL_REPORT_PATHS),
                "declared_symbols": terminal_contract["declared_symbols"],
                "producer_artifact_bundle_id": producer_artifacts["bundle_id"],
            },
            "corpus_rights_splits": {
                "disposition": "training_unavailable",
                "admission_id": admission_payload["admission_id"],
                "corpus_root": admission_payload["corpus_root"],
                "source_rights_root": admission_payload["source_rights_root"],
                "split_root": split_manifest["split_root"],
                "partitions": partitions,
                "hidden_test_bodies_accessed": False,
                "privacy_disposition": "public_report_bounded",
            },
            "architecture_tokenizer_checkpoint": {
                "disposition": "training_unavailable",
                "architecture": "not_selected",
                "tokenizer": "no_learned_tokenizer_admitted",
                "checkpoint": "not_created",
                "training": "not_attempted",
            },
            "proof_validation": {
                "disposition": "owner_receipts_required",
                "validation_commands": terminal_contract["validation_commands"],
                "producer_artifact_bundle_id": producer_artifacts["bundle_id"],
                "benchmark_freeze_id": benchmark_freeze["freeze_id"],
                "benchmark_case_root": benchmark_freeze["case_root"],
                "benchmark_binding_set_id": frozen_binding_set_id,
                "paired_baseline_id": paired_baseline["paired_baseline_id"],
                "benchmark_case_payload_disposition": benchmark_freeze[
                    "case_payload_disposition"
                ],
                "benchmark_evaluation_disposition": benchmark_freeze[
                    "evaluation_disposition"
                ],
                "producer_database_portal_validations": "required",
                "terminal_database_portal_validation": "required",
                "report_authoritative": False,
            },
            "drift": {
                "disposition": "not_run_training_unavailable",
                "reference_tree": baseline["source"]["tree"],
                "evaluated_tree": evaluated_tree,
                "checkpoint_available": False,
                "detectors_run": [],
                "reason_codes": [
                    "no_admitted_checkpoint",
                    "training_unavailable",
                ],
            },
            "rollback_blocker_eligibility": {
                "promotion_eligible": False,
                "rollback_target": baseline["source"]["commit"],
                "blockers": exact_blockers,
                "not_run": exact_not_run,
                "report_authority": "non_authoritative",
            },
        }
        if tamper == "lineage":
            report["start_tree"] = "banana"
            report["end_tree"] = "banana"
        elif tamper == "files_symbols":
            report["files_symbols"]["declared_symbols"] = ["fake"]
        elif tamper == "corpus_rights_splits":
            report["corpus_rights_splits"]["split_root"] = "banana"
        elif tamper == "architecture_tokenizer_checkpoint":
            report["architecture_tokenizer_checkpoint"]["checkpoint"] = "invented"
        elif tamper == "expert_dispositions":
            report["expert_dispositions"][manifest["task_families"][0]] = "ACCEPT"
        elif tamper == "denominators":
            report["before"] = {**benchmark_scores, "accept": 1}
        elif tamper == "costs":
            report["costs"] = {"tokens": 1, "break_even": 0}
        elif tamper == "proof_validation":
            report["proof_validation"]["disposition"] = "self_asserted"
        elif tamper == "drift":
            report["drift"]["detectors_run"] = ["invented"]
        elif tamper == "rollback_blockers":
            report["rollback_blocker_eligibility"]["promotion_eligible"] = True
        elif tamper == "incomplete_not_run":
            report["gaps"]["not_run"] = ["gpu_live_qualification"]
        elif tamper == "omitted_section":
            report.pop("proof_validation")
        elif tamper == "missing_family":
            benchmark_blob_cases = [
                item
                for item in benchmark_blob_cases
                if item["family"] != manifest["task_families"][0]
            ]
        elif tamper == "missing_required_kind":
            benchmark_blob_cases = [
                item
                for item in benchmark_blob_cases
                if item["kind"] != "unknown_ood"
            ]
        elif tamper == "hidden_training_case":
            benchmark_blob_cases[0]["hidden_test"] = True
        elif tamper == "case_group_lineage":
            benchmark_blob_cases[0]["group_id"] = "sha256:" + "1" * 64
        elif tamper == "case_input_identity":
            benchmark_blob_cases[0]["input_identity"] = "sha256:" + "2" * 64
        elif tamper == "case_expected_outcome":
            benchmark_blob_cases[0]["expected_outcome"] = "ACCEPT"
        elif tamper == "benchmark_freeze_missing":
            benchmark_blob_manifest.pop("benchmark_freeze")
        elif tamper == "benchmark_binding_tamper":
            benchmark_blob_manifest["benchmark_freeze"]["bindings"][
                "objective_revisions"
            ] = "sha256:" + "3" * 64
        elif tamper == "fault_schedule_tamper":
            benchmark_blob_manifest["benchmark_freeze"]["fault_schedule"][
                "entries"
            ][0]["kind"] = "invented"
        elif tamper == "paired_lineage_tamper":
            benchmark_blob_manifest["benchmark_freeze"]["paired_baseline"][
                "evaluated_source"
            ]["tree"] = "f" * 40
        elif tamper == "stale_bootstrap_manifest":
            benchmark_blob_manifest = json.loads(json.dumps(manifest))

        operator = _operator()
        current_head = "a" * 40
        bootstrap_head = "b" * 40
        implementation_commit = "c" * 40
        producer_paths = {
            path
            for paths in specification["terminal_report_contract"][
                "producer_output_paths"
            ].values()
            for path in paths
        }
        producer_implementation_by_path: dict[str, str] = {}
        for producer_alias, paths in terminal_contract[
            "producer_output_paths"
        ].items():
            ordinal = int(producer_alias.removeprefix("VRIF-"))
            producer_control_receipt, _ = _portal_control_receipt(ordinal)
            producer_binding = producer_control_receipt["validation"][
                "portal_completion_binding"
            ]
            for producer_path in paths:
                producer_implementation_by_path[producer_path] = producer_binding[
                    "implementation_commit"
                ]
        rewritten_path = next(
            path
            for path in sorted(producer_paths)
            if not path.endswith("/cases.jsonl")
            and not path.endswith("/manifest.json")
        )

        def fake_git(*arguments: str, binary: bool = False) -> str | bytes:
            del binary
            if arguments and arguments[0] == "ls-tree":
                path = arguments[-1]
                return f"100644 blob {'f' * 40}\t{path}"
            if arguments and arguments[0] == "diff-tree":
                changed_paths = list(terminal_contract["declared_output_paths"])
                if tamper == "extra_changed_path":
                    changed_paths.append(
                        "ipfs_accelerate_py/agent_supervisor/"
                        "task_sources/intent_repository.py"
                    )
                return "\n".join(changed_paths)
            raise AssertionError(arguments)

        def fake_commit_tree(_commit: object, *, field: str) -> str:
            return (
                evaluated_tree
                if "Portal evaluated source" in field
                else str(baseline["source"]["tree"])
            )

        def fake_blob(*, head: str, path: Path, field: str) -> bytes:
            del field
            relative = path.relative_to(ROOT).as_posix()
            if relative == TERMINAL_REPORT_PATHS[0]:
                return (
                    b'{"bootstrap":"fixture"}'
                    if head == bootstrap_head
                    else json.dumps(report, sort_keys=True).encode("utf-8")
                )
            if relative == TERMINAL_REPORT_PATHS[1]:
                return (
                    b"bootstrap fixture"
                    if head == bootstrap_head
                    else (
                        b"x"
                        if tamper == "meaningless_markdown"
                        else operator._vrif_release_report_markdown(report).encode(
                            "utf-8"
                        )
                    )
                )
            if relative == (
                "benchmarks/agent_supervisor/residual_intelligence/cases.jsonl"
            ):
                if head == bootstrap_head:
                    return b'{"bootstrap":"cases"}\n'
                return (
                    "\n".join(
                        json.dumps(item, sort_keys=True)
                        for item in benchmark_blob_cases
                    )
                    + "\n"
                ).encode("utf-8")
            if relative == (
                "benchmarks/agent_supervisor/residual_intelligence/manifest.json"
            ):
                if head == bootstrap_head:
                    return b'{"bootstrap":"manifest"}'
                return json.dumps(benchmark_blob_manifest, sort_keys=True).encode(
                    "utf-8"
                )
            if (
                relative == operation_catalog_path
                and head == benchmark_producer_binding["implementation_commit"]
            ):
                return operation_catalog_at_benchmark
            if relative in producer_paths:
                if (
                    tamper == "producer_rewrite"
                    and relative == rewritten_path
                    and head == producer_implementation_by_path[relative]
                ):
                    return b"producer-original-before-terminal-rewrite"
                return (
                    f"bootstrap:{relative}".encode("utf-8")
                    if head == bootstrap_head
                    else producer_content(relative)
                )
            return (ROOT / relative).read_bytes()

        monkeypatch.setattr(operator, "_git", fake_git)
        monkeypatch.setattr(operator, "_git_commit_tree", fake_commit_tree)
        monkeypatch.setattr(operator, "_git_is_ancestor", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(operator, "_git_blob_at", fake_blob)
        observed = operator._vrif_terminal_report_evidence(
            specification,
            {
                "bootstrap_source_head": bootstrap_head,
                "current_source_head": current_head,
            },
            connection,
        )
        if tamper in {"canonical", "later_catalog_change"}:
            assert observed is not None
            assert observed["producer_artifacts"] == producer_artifacts
        else:
            assert observed is None
    finally:
        repository.close()
        connection.close()


def test_root_waits_for_active_maintenance_lease_settlement(tmp_path: Path) -> None:
    path = tmp_path / "control.duckdb"
    specification = _specification()
    _seed(path)
    connection, repository = _bound_repository(path)
    try:
        connection.execute(
            """
            INSERT INTO maintenance_leases (
                lease_id, scope, owner_session_id, process_birth_id,
                fencing_token, fence_epoch, acquired_at, expires_at,
                released_at, state, revision
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                "lease:still-active",
                "scope:vrif-settlement",
                "session:pytest",
                "process:pytest",
                1,
                1,
                "2026-01-01T00:00:00Z",
                "2099-01-01T00:00:00Z",
                None,
                "active",
                1,
            ],
        )
        repository.rebuild_projections_from_events()
        assert connection.execute(
            "SELECT COUNT(*) FROM maintenance_leases WHERE lease_id = ?",
            ["lease:still-active"],
        ).fetchone()[0] == 1
        first = repository.reconcile_goal_completion_authority(
            specification,
            root_completion_gate=_root_gate(specification, connection),
        )
        assert first["changed_goal_ids"][-1] == "VRIF-G040"
        assert "VRIF-G000" not in first["changed_goal_ids"]
        assert first["goal_authority"]["all_goals_satisfied"] is False
        assert first["goal_authority"]["settlement_counts"][
            "active_maintenance_leases"
        ] == 1
        assert "completion_gate:active_mutating_claims_empty" in (
            first["goal_authority"]["root_goal"]["incomplete_reasons"]
        )

        connection.execute(
            "UPDATE maintenance_leases SET state = 'released', released_at = ? "
            "WHERE lease_id = ? AND state = 'active'",
            ["2026-01-01T00:01:00Z", "lease:still-active"],
        )
        repository.rebuild_projections_from_events()
        second = repository.reconcile_goal_completion_authority(
            specification,
            root_completion_gate=_root_gate(specification, connection),
        )
        assert second["changed_goal_ids"] == ["VRIF-G000"]
        assert second["goal_authority"]["all_goals_satisfied"] is True
    finally:
        repository.close()
        connection.close()


def test_tampered_old_root_receipt_cannot_be_renewed(tmp_path: Path) -> None:
    path = tmp_path / "control.duckdb"
    specification = _specification()
    _seed(path)
    connection, repository = _bound_repository(path)
    try:
        gate = _root_gate(specification, connection)
        completed = repository.reconcile_goal_completion_authority(
            specification,
            root_completion_gate=gate,
        )
        assert completed["goal_authority"]["all_goals_satisfied"] is True
        root = repository.get_goal("VRIF-G000")
        assert root is not None
        body = json.loads(json.dumps(root["body"]))
        body["completion_receipt"]["goal_alias"] = "VRIF-G999"
        connection.execute(
            "UPDATE goals SET body_json = ? WHERE goal_cid = ? AND revision = ?",
            [
                json.dumps(body, sort_keys=True, separators=(",", ":")),
                root["goal_cid"],
                root["revision"],
            ],
        )
        renewed = repository.reconcile_goal_completion_authority(
            specification,
            root_completion_gate=_renewed_root_gate(gate),
        )
        assert renewed["changed_goal_ids"] == []
        assert renewed["goal_authority"]["all_goals_satisfied"] is False
        assert renewed["goal_authority"]["invalid_goal_ids"] == ["VRIF-G000"]
        assert "goal_current_completion_receipt_invalid" in (
            renewed["goal_authority"]["root_goal"]["incomplete_reasons"]
        )
    finally:
        repository.close()
        connection.close()


def test_generic_goal_cas_preserves_legacy_skipped_and_optional_receipt(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy-goal-cas.duckdb"
    with open_intent_repository(path, owner_id="owner:legacy-goal-cas") as repository:
        repository.upsert_goal(
            goal_cid="goal:legacy",
            goal_alias="LEGACY-G000",
            ordinal=1,
            title="legacy goal",
            status="waiting",
        )
        repository.upsert_task(
            task_cid="task:legacy",
            task_alias="LEGACY-000",
            goal_cid="goal:legacy",
            ordinal=1,
            status="skipped",
        )
        goal = repository.get_goal("goal:legacy")
        assert goal is not None
        receipt = repository.cas_goal_status(
            goal_cid="goal:legacy",
            expected_revision=int(goal["revision"]),
            new_status="completed",
        )
        assert receipt.changed is True
        completed = repository.get_goal("goal:legacy")
        assert completed is not None
        assert completed["status"] == "completed"
        assert "completion_receipt" not in completed["body"]


def test_started_attempt_blocks_root_until_it_has_a_finish_marker(
    tmp_path: Path,
) -> None:
    path = tmp_path / "started-attempt.duckdb"
    specification = _specification()
    _seed(path)
    connection, repository = _bound_repository(path)
    try:
        attempt = repository.record_attempt(task_cid=_task_cid(32))
        first = repository.reconcile_goal_completion_authority(
            specification,
            root_completion_gate=_root_gate(specification, connection),
        )
        assert first["goal_authority"]["all_goals_satisfied"] is False
        assert first["goal_authority"]["settlement_counts"][
            "running_task_attempts"
        ] == 1
        assert "VRIF-G000" not in first["changed_goal_ids"]

        connection.execute(
            "UPDATE task_attempts SET status = 'completed', finished_at = ? "
            "WHERE attempt_id = ?",
            ["2026-01-01T00:01:00Z", attempt.subject_id],
        )
        second = repository.reconcile_goal_completion_authority(
            specification,
            root_completion_gate=_root_gate(specification, connection),
        )
        assert second["changed_goal_ids"] == ["VRIF-G000"]
        assert second["goal_authority"]["all_goals_satisfied"] is True
    finally:
        repository.close()
        connection.close()


def test_attempt_status_matrix_records_and_replays_exact_finish_timestamps(
    tmp_path: Path,
) -> None:
    path = tmp_path / "terminal-attempt-replay.duckdb"
    _seed(path)
    connection, repository = _bound_repository(path)
    try:
        active = {"started", "running", "in_progress"}
        terminal = {
            "succeeded",
            "completed",
            "failed",
            "cancelled",
            "released",
            "expired",
        }
        before_by_attempt: dict[str, tuple[object, ...]] = {}
        for status in sorted(active | terminal):
            attempt = repository.record_attempt(
                task_cid=_task_cid(32),
                status=status,
            )
            before = connection.execute(
                "SELECT status, started_at, finished_at FROM task_attempts "
                "WHERE attempt_id = ?",
                [attempt.subject_id],
            ).fetchone()
            assert before is not None
            values = tuple(before[index] for index in range(3))
            assert values[0] == status
            assert str(values[1]).endswith("Z")
            assert values[2] == (values[1] if status in terminal else "")
            before_by_attempt[attempt.subject_id] = values

        with pytest.raises(IntentRepositoryError, match="closed set"):
            repository.record_attempt(
                task_cid=_task_cid(32),
                status="mystery_terminal",
            )

        repository.rebuild_projections_from_events()
        for attempt_id, before_values in before_by_attempt.items():
            after = connection.execute(
                "SELECT status, started_at, finished_at FROM task_attempts "
                "WHERE attempt_id = ?",
                [attempt_id],
            ).fetchone()
            assert after is not None
            assert tuple(after[index] for index in range(3)) == before_values
    finally:
        repository.close()
        connection.close()


def test_timestamped_settlement_projections_replay_exactly_under_changed_clock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "validation-outcome-replay.duckdb"
    _seed(path)
    connection, repository = _bound_repository(path)
    try:
        run_ids: list[str] = []
        for outcome in ("passed", "failed", "error", "skipped"):
            receipt = repository.record_validation_result(
                task_cid=_task_cid(0),
                outcome=outcome,
                evidence_digest=f"digest:validation:{outcome}",
                argv=[f"validate:{outcome}"],
            )
            event_row = connection.execute(
                "SELECT body_json FROM domain_events WHERE event_id = ?",
                [receipt.event_id],
            ).fetchone()
            assert event_row is not None
            envelope = json.loads(str(event_row[0]))
            run_ids.append(str(envelope["body"]["run_id"]))

        def observed_rows() -> list[tuple[object, ...]]:
            rows = connection.execute(
                "SELECT run_id, status, started_at, finished_at "
                "FROM validation_runs WHERE run_id IN (?, ?, ?, ?) "
                "ORDER BY run_id",
                run_ids,
            ).fetchall()
            return [tuple(row[index] for index in range(4)) for row in rows]

        before = observed_rows()
        assert len(before) == 4
        assert {str(row[1]) for row in before} == {
            "passed",
            "failed",
            "error",
            "skipped",
        }
        assert all(row[2] == row[3] and str(row[3]).endswith("Z") for row in before)
        counts = IntentRepository._goal_settlement_counts(connection)
        assert counts["invalid_validation_runs_rows"] == 0
        assert counts["running_validation_runs"] == 0

        evidence = repository.record_evidence(
            task_cid=_task_cid(0),
            evidence_kind="pytest",
            digest="digest:evidence:replay",
        )

        def observed_evidence() -> tuple[object, ...] | None:
            row = connection.execute(
                "SELECT evidence_id, created_at FROM evidence_nodes "
                "WHERE evidence_id = ?",
                [evidence.subject_id],
            ).fetchone()
            return None if row is None else tuple(row[index] for index in range(2))

        evidence_before = observed_evidence()
        assert evidence_before is not None
        assert str(evidence_before[1]).endswith("Z")

        cleared_block = repository.block_task(
            task_cid=_task_cid(0),
            blocker_kind="pytest",
            blocker_id="blocker:cleared",
            reason="exercise cleared replay",
        )
        repository.block_task(
            task_cid=_task_cid(0),
            blocker_kind="pytest",
            blocker_id="blocker:remains-active",
            reason="exercise selective unblock replay",
        )
        repository.unblock_task(
            task_cid=_task_cid(0), block_id=cleared_block.subject_id
        )
        repository.block_task(
            task_cid=_task_cid(1),
            blocker_kind="pytest",
            blocker_id="blocker:active",
            reason="exercise active replay",
        )

        def observed_blocks() -> list[tuple[object, ...]]:
            rows = connection.execute(
                "SELECT task_cid, state, cleared_at FROM task_blocks "
                "ORDER BY task_cid, block_id"
            ).fetchall()
            return [tuple(row[index] for index in range(3)) for row in rows]

        blocks_before = observed_blocks()
        assert {str(row[1]) for row in blocks_before} == {"active", "cleared"}
        assert all(
            (str(row[2] or "") == "") == (row[1] == "active")
            for row in blocks_before
        )

        from ipfs_accelerate_py.agent_supervisor.task_sources import (
            intent_repository as intent,
        )

        monkeypatch.setattr(
            intent,
            "_utc_iso",
            lambda *_args, **_kwargs: "2099-01-01T00:00:00Z",
        )
        repository.rebuild_projections_from_events()
        assert observed_rows() == before
        assert observed_evidence() == evidence_before
        assert observed_blocks() == blocks_before
        counts = IntentRepository._goal_settlement_counts(connection)
        assert counts["invalid_validation_runs_rows"] == 0
    finally:
        repository.close()
        connection.close()


@pytest.mark.parametrize(
    ("event_type", "timestamp_field"),
    (
        ("intent.task_blocked", "created_at"),
        ("intent.task_unblocked", "cleared_at"),
        ("intent.evidence_recorded", "created_at"),
    ),
)
def test_timestamped_projection_replay_rejects_noncanonical_event_timestamp(
    tmp_path: Path,
    event_type: str,
    timestamp_field: str,
) -> None:
    path = tmp_path / f"invalid-{timestamp_field}.duckdb"
    _seed(path)
    connection, repository = _bound_repository(path)
    try:
        if event_type == "intent.evidence_recorded":
            repository.record_evidence(
                task_cid=_task_cid(0),
                evidence_kind="pytest",
                digest="digest:evidence:timestamp-validation",
            )
        else:
            repository.block_task(
                task_cid=_task_cid(0),
                blocker_kind="pytest",
                blocker_id=f"blocker:{timestamp_field}",
                reason="exercise timestamp validation",
            )
            if event_type == "intent.task_unblocked":
                repository.unblock_task(task_cid=_task_cid(0))
        row = connection.execute(
            "SELECT event_id, body_json FROM domain_events "
            "WHERE event_type = ? ORDER BY global_sequence DESC LIMIT 1",
            [event_type],
        ).fetchone()
        assert row is not None
        envelope = json.loads(str(row[1]))
        envelope["body"][timestamp_field] = "not-a-canonical-timestamp"
        connection.execute(
            "UPDATE domain_events SET body_json = ? WHERE event_id = ?",
            [
                json.dumps(envelope, sort_keys=True, separators=(",", ":")),
                str(row[0]),
            ],
        )

        with pytest.raises(
            IntentRepositoryIntegrityError,
            match="canonical UTC timestamp",
        ):
            repository.rebuild_projections_from_events()
    finally:
        repository.close()
        connection.close()


def test_canonical_settlement_vocabularies_reject_orphans_and_bad_markers(
    tmp_path: Path,
) -> None:
    path = tmp_path / "settlement-vocabularies.duckdb"
    with open_intent_repository(path):
        pass
    connection = open_duckdb_connection(path)
    now = "2026-01-01T00:00:00Z"
    later = "2026-01-01T00:01:00Z"
    cases = [
        (
            "task_assignments",
            "INSERT INTO task_assignments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ["assignment:orphan", "task:orphan", "session:x", "daemon:x", now, None, "running", 1, 1],
        ),
        (
            "task_claims",
            "INSERT INTO task_claims VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ["claim:orphan", "task:orphan", "session:x", 1, 1, now, later, None, "accepted", 1, "idem:x"],
        ),
        (
            "resource_claims",
            "INSERT INTO resource_claims VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ["claim:resource", "kind:x", "resource:x", "session:x", "task:orphan", 1, 1, now, later, "accepted", 1],
        ),
        (
            "path_claims",
            "INSERT INTO path_claims VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ["claim:path", "repository:x", "worktree:x", "x.py", "session:x", "task:orphan", 1, 1, now, later, "accepted", 1],
        ),
        (
            "effect_claims",
            "INSERT INTO effect_claims VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ["effect:orphan", "task:orphan", "attempt:orphan", "write", "x.py", now, "completed", "{}"],
        ),
        (
            "attempt_phases",
            "INSERT INTO attempt_phases VALUES (?, ?, ?, ?, ?)",
            ["attempt:orphan", "provider", now, later, "completed"],
        ),
        (
            "provider_invocations",
            "INSERT INTO provider_invocations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ["invocation:orphan", "task:orphan", "attempt:orphan", "provider:x", now, later, "succeeded", "digest:in", "digest:out", "{}"],
        ),
        (
            "merge_attempts",
            "INSERT INTO merge_attempts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ["merge:orphan", "entry:orphan", "task:orphan", "worktree:x", now, later, "accepted", "a" * 40, "{}"],
        ),
        (
            "refill_epochs",
            "INSERT INTO refill_epochs VALUES (?, ?, ?, ?, ?, ?, ?)",
            ["epoch:orphan", "board:x", now, later, "completed", 1, "{}"],
        ),
        (
            "recovery_actions",
            "INSERT INTO recovery_actions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ["action:orphan", "task", "task:orphan", "task:orphan", "retry", now, "applied", "{}"],
        ),
        (
            "merge_queue_entries",
            "INSERT INTO merge_queue_entries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ["entry:orphan", "repository:x", "worktree:x", "task:orphan", "source", "target", "settled", 1, now, later, 1, 1],
        ),
        (
            "task_blocks",
            "INSERT INTO task_blocks VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ["block:unknown", "task:orphan", "manual", "blocker:x", "reason", now, None, "mystery"],
        ),
        (
            "task_blocks",
            "INSERT INTO task_blocks VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ["block:uppercase", "task:orphan", "manual", "blocker:x", "reason", now, None, "ACTIVE"],
        ),
        (
            "task_blocks",
            "INSERT INTO task_blocks VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ["block:whitespace", "task:orphan", "manual", "blocker:x", "reason", now, None, " active "],
        ),
        (
            "leases",
            "INSERT INTO leases (task_cid, claim_cid, resolution_cid, claimant_did, logical_epoch, fencing_token, expires_at_ms, attempt, state, started_at_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ["task:orphan", "claim:x", "resolution:x", "did:x", 1, 1, 1, 1, "superseded", 1],
        ),
        (
            "maintenance_leases",
            "INSERT INTO maintenance_leases (lease_id, scope, owner_session_id, process_birth_id, fencing_token, fence_epoch, acquired_at, expires_at, released_at, state, revision) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ["lease:unknown", "scope:x", "session:x", "process:x", 1, 1, now, later, None, "accepted", 1],
        ),
        (
            "task_attempts",
            "INSERT INTO task_attempts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ["attempt:unknown", "task:orphan", 1, "session:x", 1, 1, now, None, "mystery", 1],
        ),
        (
            "validation_runs",
            "INSERT INTO validation_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ["validation:unknown", "task:orphan", "attempt:orphan", now, later, "interrupted", "digest:command", "{}"],
        ),
        (
            "task_blocks",
            "INSERT INTO task_blocks VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ["block:bad-marker", "task:orphan", "manual", "blocker:x", "reason", now, None, "cleared"],
        ),
        (
            "maintenance_leases",
            "INSERT INTO maintenance_leases (lease_id, scope, owner_session_id, process_birth_id, fencing_token, fence_epoch, acquired_at, expires_at, released_at, state, revision) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ["lease:bad-marker", "scope:x", "session:x", "process:x", 1, 1, now, later, None, "released", 1],
        ),
        (
            "task_attempts",
            "INSERT INTO task_attempts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ["attempt:bad-marker", "task:orphan", 2, "session:x", 1, 1, now, None, "succeeded", 1],
        ),
        (
            "validation_runs",
            "INSERT INTO validation_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ["validation:bad-marker", "task:orphan", "attempt:orphan", now, None, "passed", "digest:command", "{}"],
        ),
    ]
    try:
        assert IntentRepository._goal_settlement_counts(connection)[
            "invalid_settlement_rows"
        ] == 0
        for state in ("accepted", "released", "expired", "completed"):
            connection.execute(
                "INSERT INTO leases (task_cid, claim_cid, resolution_cid, "
                "claimant_did, logical_epoch, fencing_token, expires_at_ms, "
                "attempt, state, started_at_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    f"task:lease:{state}",
                    f"claim:{state}",
                    f"resolution:{state}",
                    "did:x",
                    1,
                    1,
                    1,
                    1,
                    state,
                    1,
                ],
            )
            counts = IntentRepository._goal_settlement_counts(connection)
            assert counts["invalid_leases_rows"] == 0
            assert counts["active_leases"] == (1 if state == "accepted" else 0)
            connection.execute("DELETE FROM leases")
        for table, statement, parameters in cases:
            connection.execute(statement, parameters)
            counts = IntentRepository._goal_settlement_counts(connection)
            assert counts[f"invalid_{table}_rows"] == 1, table
            assert counts["invalid_settlement_rows"] == 1, table
            connection.execute(f"DELETE FROM {table}")
    finally:
        connection.close()


def test_unknown_canonical_claim_state_cannot_close_root(tmp_path: Path) -> None:
    path = tmp_path / "unknown-claim-state.duckdb"
    specification = _specification()
    _seed(path)
    connection, repository = _bound_repository(path)
    try:
        connection.execute(
            """
            INSERT INTO task_claims (
                claim_id, task_cid, owner_session_id, fencing_token,
                fence_epoch, claimed_at, expires_at, released_at,
                state, revision, idempotency_key
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                "claim:unprojectable",
                _task_cid(32),
                "session:pytest",
                1,
                1,
                "2026-01-01T00:00:00Z",
                "2099-01-01T00:00:00Z",
                None,
                "running",
                1,
                "pytest-unprojectable-claim",
            ],
        )
        first = repository.reconcile_goal_completion_authority(
            specification,
            root_completion_gate=_root_gate(specification, connection),
        )
        assert "VRIF-G000" not in first["changed_goal_ids"]
        assert first["goal_authority"]["all_goals_satisfied"] is False
        assert first["goal_authority"]["settlement_counts"][
            "invalid_task_claims_rows"
        ] == 1
        assert "completion_gate:settlement_state_integrity" in first[
            "goal_authority"
        ]["root_goal"]["incomplete_reasons"]

        connection.execute(
            "DELETE FROM task_claims WHERE claim_id = ?",
            ["claim:unprojectable"],
        )
        second = repository.reconcile_goal_completion_authority(
            specification,
            root_completion_gate=_root_gate(specification, connection),
        )
        assert second["changed_goal_ids"] == ["VRIF-G000"]
        assert second["goal_authority"]["all_goals_satisfied"] is True
    finally:
        repository.close()
        connection.close()


def test_root_requires_exact_retired_ready_task_lineage(tmp_path: Path) -> None:
    path = tmp_path / "retired-ready-lineage.duckdb"
    specification = _specification()
    _seed(path)
    connection, repository = _bound_repository(path)
    try:
        gate = _root_gate(specification, connection)
        binding = dict(gate["runtime_settlement_binding"])
        binding.pop("binding_id")
        binding["retired_ready_task_cids"] = binding[
            "retired_ready_task_cids"
        ][:-1]
        binding["binding_id"] = _sha256_identity(binding)
        gate.pop("gate_id")
        gate["runtime_settlement_binding"] = binding
        gate["gate_id"] = content_identity(gate)

        observed = repository.reconcile_goal_completion_authority(
            specification,
            root_completion_gate=gate,
        )
        assert "VRIF-G000" not in observed["changed_goal_ids"]
        assert observed["goal_authority"]["all_goals_satisfied"] is False
        assert observed["goal_authority"]["completion_gates"][
            "retired_ready_tasks_satisfied"
        ] is False
        assert "completion_gate:retired_ready_tasks_satisfied" in observed[
            "goal_authority"
        ]["root_goal"]["incomplete_reasons"]
    finally:
        repository.close()
        connection.close()


def test_assignment_orphan_survives_event_rebuild_and_blocks_root(
    tmp_path: Path,
) -> None:
    path = tmp_path / "assignment-rebuild.duckdb"
    specification = _specification()
    _seed(path)
    connection, repository = _bound_repository(path)
    try:
        connection.execute(
            "INSERT INTO task_assignments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                "assignment:orphan",
                _task_cid(32),
                "session:pytest",
                "daemon:pytest",
                "2026-01-01T00:00:00Z",
                None,
                "running",
                1,
                1,
            ],
        )
        repository.rebuild_projections_from_events()
        assert connection.execute(
            "SELECT COUNT(*) FROM task_assignments WHERE assignment_id = ?",
            ["assignment:orphan"],
        ).fetchone()[0] == 1
        first = repository.reconcile_goal_completion_authority(
            specification,
            root_completion_gate=_root_gate(specification, connection),
        )
        assert first["goal_authority"]["all_goals_satisfied"] is False
        assert first["goal_authority"]["settlement_counts"][
            "invalid_task_assignments_rows"
        ] == 1
    finally:
        repository.close()
        connection.close()


def test_completed_goal_receipts_refresh_after_current_task_revision(
    tmp_path: Path,
) -> None:
    path = tmp_path / "goal-receipt-refresh.duckdb"
    specification = _specification()
    _seed(path)
    connection, repository = _bound_repository(path)
    try:
        gate = _root_gate(specification, connection)
        first = repository.reconcile_goal_completion_authority(
            specification,
            root_completion_gate=gate,
        )
        assert first["goal_authority"]["all_goals_satisfied"] is True
        before = {
            alias: int(repository.get_goal(alias)["revision"])
            for alias in GOAL_ALIASES
        }

        task = repository.get_task(_task_cid(0))
        assert task is not None
        repository.cas_task_status(
            task_cid=_task_cid(0),
            expected_revision=int(task["revision"]),
            new_status="ready",
        )
        reopened = repository.get_task(_task_cid(0))
        assert reopened is not None
        repository.cas_task_status(
            task_cid=_task_cid(0),
            expected_revision=int(reopened["revision"]),
            new_status="completed",
            receipt={"operation": "pytest_recompletion", "task_alias": "VRIF-000"},
            allow_completion_without_evidence=True,
        )
        refreshed = repository.reconcile_goal_completion_authority(
            specification,
            root_completion_gate=gate,
        )
        assert refreshed["changed_goal_ids"] == [
            "VRIF-G011",
            "VRIF-G010",
            "VRIF-G021",
            "VRIF-G020",
            "VRIF-G031",
            "VRIF-G030",
            "VRIF-G041",
            "VRIF-G040",
            "VRIF-G000",
        ]
        assert refreshed["goal_authority"]["all_goals_satisfied"] is True
        assert all(
            int(repository.get_goal(alias)["revision"]) == before[alias] + 1
            for alias in GOAL_ALIASES
        )
    finally:
        repository.close()
        connection.close()


def test_projection_reuses_raw_duckdb_caller_transaction(tmp_path: Path) -> None:
    import duckdb

    path = tmp_path / "caller-transaction.duckdb"
    specification = _specification()
    _seed(path)
    connection = duckdb.connect(str(path))
    try:
        connection.execute("BEGIN TRANSACTION")
        assert _operator()._task_status(connection)["task_count"] == 33
        projected = goal_authority_projection_on_connection(
            connection,
            specification,
            transaction_owned_by_caller=True,
        )
        assert projected["goal_count"] == 9
        connection.execute("COMMIT")
        assert connection.execute("SELECT COUNT(*) FROM goals").fetchone()[0] == 9
    finally:
        connection.close()


@pytest.mark.parametrize("projection_fails", [False, True])
def test_status_uses_one_snapshot_and_resets_partial_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    projection_fails: bool,
) -> None:
    operator = _operator()
    trace: list[str] = []

    class TraceConnection:
        in_transaction = False

        def execute(self, sql: str, _parameters: object = None):
            normalized = " ".join(sql.strip().split()).upper()
            if normalized.startswith("BEGIN"):
                assert self.in_transaction is False
                self.in_transaction = True
                trace.append("BEGIN")
            elif normalized == "COMMIT":
                assert self.in_transaction is True
                self.in_transaction = False
                trace.append("COMMIT")
            elif normalized == "ROLLBACK":
                assert self.in_transaction is True
                self.in_transaction = False
                trace.append("ROLLBACK")
            else:
                raise AssertionError(sql)
            return self

        def close(self) -> None:
            trace.append("CLOSE")

    connection = TraceConnection()
    owner_dir = tmp_path / "owner"
    owner_dir.mkdir()
    (owner_dir / "quack-state-server.status.json").write_text(
        json.dumps({"lifecycle": "ready"}), encoding="utf-8"
    )
    paths = {
        "owner": owner_dir,
        "database": tmp_path / "control.duckdb",
        "bootstrap_receipt": tmp_path / "bootstrap.json",
        "ducklake_receipt": tmp_path / "ducklake.json",
    }
    program = SimpleNamespace(
        endpoint_secret_handle="secret-handle",
        quack_endpoint="quack://127.0.0.1:1",
    )
    board = SimpleNamespace(resolved_database_program=lambda: program)

    monkeypatch.setattr(operator, "_load_config", lambda _path: (board, {}))
    monkeypatch.setattr(operator, "_runtime_paths", lambda _board: paths)
    monkeypatch.setattr(operator, "_owner_liveness", lambda _status: "alive")
    monkeypatch.setattr(operator, "_read_owner_token", lambda _path: "token")
    def task_status(observed):
        trace.append("TASK")
        assert observed.in_transaction is True
        return {"task_count": 33}

    def admission(*_args):
        trace.append("ADMISSION")
        assert connection.in_transaction is True
        return {"admission_id": "admission"}

    def authority_spec(*_args):
        trace.append("SPEC")
        assert connection.in_transaction is True
        return {"authority_spec_id": "spec"}

    monkeypatch.setattr(operator, "_task_status", task_status)
    monkeypatch.setattr(operator, "_owner_restart_admission", admission)
    monkeypatch.setattr(
        operator, "_vrif_goal_completion_authority_spec", authority_spec
    )
    monkeypatch.setattr(
        operator,
        "_assert_clean_current_tree",
        lambda _config: ("a" * 40, "b" * 40),
    )
    import ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state as duckdb_state
    import ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository as intent

    monkeypatch.setattr(
        duckdb_state,
        "open_quack_transport_connection",
        lambda *_args, **_kwargs: connection,
    )

    def project(observed, _specification, **kwargs):
        trace.append("GOAL")
        assert observed.in_transaction is True
        assert kwargs["transaction_owned_by_caller"] is True
        assert kwargs["root_gate_context"]["runtime_settlement_binding"] is None
        if projection_fails:
            raise RuntimeError("projection failed")
        return {"goal_count": 9, "all_goals_satisfied": False}

    monkeypatch.setattr(intent, "goal_authority_projection_on_connection", project)
    observed = operator.status(tmp_path / "config.json")
    if projection_fails:
        assert trace == ["BEGIN", "TASK", "ADMISSION", "SPEC", "GOAL", "ROLLBACK", "CLOSE"]
        assert observed["task_authority"]["available"] is False
        assert observed["goal_authority"]["available"] is False
        assert observed["task_authority"]["reason_code"] == "goal_authority_probe_failed"
        assert observed["goal_authority"]["reason_code"] == "goal_authority_probe_failed"
    else:
        assert trace == ["BEGIN", "TASK", "ADMISSION", "SPEC", "GOAL", "COMMIT", "CLOSE"]
        assert observed["task_authority"]["available"] is True
        assert observed["goal_authority"]["available"] is True


def test_runtime_guard_is_held_through_root_reconciliation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    import ipfs_accelerate_py.agent_supervisor.runtime.vrif_runtime_settlement as runtime

    events: list[str] = []
    guard_active = {"value": False}
    binding = _runtime_settlement_binding()

    @contextmanager
    def hold_guard(*_args, **_kwargs):
        events.append("guard_enter")
        guard_active["value"] = True
        try:
            yield {"settled": True}
        finally:
            guard_active["value"] = False
            events.append("guard_exit")

    monkeypatch.setattr(runtime, "hold_vrif_runtime_settlement", hold_guard)
    monkeypatch.setattr(
        runtime,
        "vrif_runtime_settlement_binding",
        lambda *_args, **_kwargs: binding,
    )
    monkeypatch.setattr(
        operator,
        "_vrif_runtime_target",
        lambda _config: (
            binding["target"]["repository_id"],
            binding["target"]["branch"],
        ),
    )

    def root_gate(*_args, runtime_settlement_binding, **_kwargs):
        assert guard_active["value"] is True
        assert runtime_settlement_binding == binding
        return {"runtime_settlement_binding": dict(binding)}

    def current_gate(
        _config,
        _admission,
        gate,
        *,
        runtime_settlement_binding=None,
    ):
        assert guard_active["value"] is True
        return gate if runtime_settlement_binding == binding else None

    def reconcile(
        _repository,
        _specification,
        *,
        root_completion_gate,
        root_gate_current_validator,
        **_kwargs,
    ):
        events.append("root_cas")
        assert guard_active["value"] is True
        assert root_completion_gate is not None
        assert root_gate_current_validator(root_completion_gate) is True
        return {"changed": True, "changed_goal_ids": ["VRIF-G000"]}

    monkeypatch.setattr(operator, "_vrif_root_completion_gate", root_gate)
    monkeypatch.setattr(operator, "_current_vrif_root_completion_gate", current_gate)
    monkeypatch.setattr(operator, "_reconcile_vrif_goal_completion", reconcile)
    observed = operator._reconcile_vrif_goal_completion_under_runtime_guard(
        config_path=tmp_path / "config.json",
        config={"merge_target_branch": binding["target"]["branch"]},
        admission={},
        restart_receipt={"state_owner": {"generation": 1}},
        repository=object(),
        specification={},
        connection=object(),
    )
    assert observed["changed_goal_ids"] == ["VRIF-G000"]
    assert events == ["guard_enter", "root_cas", "guard_exit"]


def test_root_gate_reuses_exact_stored_gate_in_same_owner_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    binding = _runtime_settlement_binding()
    specification = {
        "authority_spec_id": "authority:pytest",
        "root_goal_cid": "goal:pytest",
        "completion_policy": {"terminal_task_id": "VRIF-032"},
    }
    admission = {
        "admission_id": "admission:pytest",
        "current_source_head": "a" * 40,
        "current_source_tree": "b" * 40,
    }
    restart_receipt = {
        "receipt_id": "restart:pytest",
        "state_owner": {"generation": 1},
    }
    terminal_evidence = {"evidence": "terminal:pytest"}
    monkeypatch.setattr(
        operator,
        "_vrif_terminal_report_evidence",
        lambda *_args, **_kwargs: terminal_evidence,
    )

    class Result:
        def __init__(self, rows):
            self._rows = rows

        def fetchall(self):
            return self._rows

    class Connection:
        def __init__(self, rows):
            self._rows = rows

        def execute(self, *_args, **_kwargs):
            return Result(self._rows)

    initial = operator._vrif_root_completion_gate(
        specification,
        admission,
        restart_receipt,
        Connection([("pending", "{}")]),
        runtime_settlement_binding=binding,
    )
    assert initial is not None
    stored_body = json.dumps(
        {"completion_receipt": {"root_completion_gate": initial}}
    )
    monkeypatch.setattr(
        operator,
        "_git_is_ancestor",
        lambda *_args, **_kwargs: pytest.fail(
            "same-generation exact gate reuse needs no ancestry refresh"
        ),
    )
    observed = operator._vrif_root_completion_gate(
        specification,
        admission,
        restart_receipt,
        Connection([("completed", stored_body)]),
        runtime_settlement_binding=binding,
    )
    assert observed == initial


def test_runtime_guard_exit_failure_after_root_cas_fails_closed_without_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    import ipfs_accelerate_py.agent_supervisor.runtime.vrif_runtime_settlement as runtime

    binding = _runtime_settlement_binding()

    @contextmanager
    def changing_guard(*_args, **_kwargs):
        yield {"settled": True}
        raise runtime.VRIFRuntimeSettlementError("runtime changed on guard exit")

    monkeypatch.setattr(runtime, "hold_vrif_runtime_settlement", changing_guard)
    monkeypatch.setattr(
        runtime,
        "vrif_runtime_settlement_binding",
        lambda *_args, **_kwargs: binding,
    )
    monkeypatch.setattr(
        operator,
        "_vrif_runtime_target",
        lambda _config: (
            binding["target"]["repository_id"],
            binding["target"]["branch"],
        ),
    )
    gate = {"runtime_settlement_binding": dict(binding)}
    monkeypatch.setattr(
        operator,
        "_vrif_root_completion_gate",
        lambda *_args, **_kwargs: gate,
    )
    monkeypatch.setattr(
        operator,
        "_current_vrif_root_completion_gate",
        lambda _config, _admission, candidate, **_kwargs: candidate,
    )
    reconcile_calls: list[object] = []

    def reconcile(
        _repository,
        _specification,
        *,
        root_completion_gate,
        **_kwargs,
    ):
        reconcile_calls.append(root_completion_gate)
        return {"changed": True, "changed_goal_ids": ["VRIF-G000"]}

    monkeypatch.setattr(operator, "_reconcile_vrif_goal_completion", reconcile)
    with pytest.raises(
        operator.OperatorError,
        match="runtime settlement changed across the root completion CAS",
    ):
        operator._reconcile_vrif_goal_completion_under_runtime_guard(
            config_path=tmp_path / "config.json",
            config={"merge_target_branch": binding["target"]["branch"]},
            admission={},
            restart_receipt={"state_owner": {"generation": 1}},
            repository=object(),
            specification={},
            connection=object(),
        )
    assert reconcile_calls == [gate]


@pytest.mark.parametrize("runtime_state", ["busy", "unsettled"])
def test_runtime_unavailable_or_unsettled_advances_only_nonroot_goals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime_state: str,
) -> None:
    operator = _operator()
    import ipfs_accelerate_py.agent_supervisor.runtime.vrif_runtime_settlement as runtime

    binding = _runtime_settlement_binding()

    @contextmanager
    def hold_guard(*_args, **_kwargs):
        if runtime_state == "busy":
            raise runtime.VRIFRuntimeSettlementError("busy")
        yield {
            "settled": False,
            "active_counts": {
                "coordination": 1,
                "execution": 0,
                "merge_queue": 0,
                "total": 1,
            },
        }

    monkeypatch.setattr(runtime, "hold_vrif_runtime_settlement", hold_guard)
    monkeypatch.setattr(
        runtime,
        "vrif_runtime_settlement_binding",
        lambda *_args, **_kwargs: pytest.fail(
            "unsettled runtime must not produce a root binding"
        ),
    )
    monkeypatch.setattr(
        operator,
        "_vrif_runtime_target",
        lambda _config: (
            binding["target"]["repository_id"],
            binding["target"]["branch"],
        ),
    )
    calls: list[object] = []

    def reconcile(
        _repository,
        _specification,
        *,
        root_completion_gate,
        root_gate_current_validator,
        **_kwargs,
    ):
        calls.append(root_completion_gate)
        assert root_completion_gate is None
        assert root_gate_current_validator is None or runtime_state == "unsettled"
        return {"changed": True, "changed_goal_ids": ["VRIF-G041"]}

    monkeypatch.setattr(operator, "_reconcile_vrif_goal_completion", reconcile)
    observed = operator._reconcile_vrif_goal_completion_under_runtime_guard(
        config_path=tmp_path / "config.json",
        config={"merge_target_branch": binding["target"]["branch"]},
        admission={},
        restart_receipt={"state_owner": {"generation": 1}},
        repository=object(),
        specification={},
        connection=object(),
    )
    assert observed["changed_goal_ids"] == ["VRIF-G041"]
    assert calls == [None]


def _status_runtime_fixture(
    operator: object,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, object], dict[str, object]]:
    owner_dir = tmp_path / "owner"
    owner_dir.mkdir()
    (owner_dir / "quack-state-server.status.json").write_text(
        json.dumps(
            {
                "lifecycle": "ready",
                "identity": {"generation": 1},
            }
        ),
        encoding="utf-8",
    )
    paths: dict[str, object] = {
        "owner": owner_dir,
        "database": tmp_path / "control.duckdb",
        "bootstrap_receipt": tmp_path / "bootstrap.json",
        "ducklake_receipt": tmp_path / "ducklake.json",
    }
    program = SimpleNamespace(
        endpoint_secret_handle="secret-handle",
        quack_endpoint="quack://127.0.0.1:1",
    )
    board = SimpleNamespace(resolved_database_program=lambda: program)
    config = {
        "merge_target_branch": "codex/verified-residual-intelligence-foundry-v1"
    }
    monkeypatch.setattr(operator, "_load_config", lambda _path: (board, config))
    monkeypatch.setattr(operator, "_runtime_paths", lambda _board: paths)
    monkeypatch.setattr(operator, "_owner_liveness", lambda _status: "alive")
    return config, paths


def test_status_runtime_busy_preserves_task_and_goal_projection_availability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    import ipfs_accelerate_py.agent_supervisor.runtime.vrif_runtime_settlement as runtime

    config, _paths = _status_runtime_fixture(operator, tmp_path, monkeypatch)
    binding = _runtime_settlement_binding()

    @contextmanager
    def busy_guard(*_args, **_kwargs):
        raise runtime.VRIFRuntimeSettlementError("busy")
        yield  # pragma: no cover

    monkeypatch.setattr(runtime, "hold_vrif_runtime_settlement", busy_guard)
    monkeypatch.setattr(
        operator,
        "_vrif_runtime_target",
        lambda _config: (
            binding["target"]["repository_id"],
            config["merge_target_branch"],
        ),
    )

    def snapshot(*, runtime_settlement_binding, **_kwargs):
        assert runtime_settlement_binding is None
        return (
            {"available": True, "task_count": 33},
            {"available": True, "all_goals_satisfied": False},
        )

    monkeypatch.setattr(operator, "_quack_status_authority_snapshot", snapshot)
    observed = operator.status(tmp_path / "config.json")
    assert observed["task_authority"]["available"] is True
    assert observed["goal_authority"]["available"] is True
    assert observed["goal_authority"]["all_goals_satisfied"] is False
    assert observed["runtime_settlement"]["available"] is False
    assert observed["runtime_settlement"]["reason_code"] == (
        "runtime_settlement_unavailable"
    )


def test_status_settled_binding_is_guarded_through_quack_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    import ipfs_accelerate_py.agent_supervisor.runtime.vrif_runtime_settlement as runtime

    config, _paths = _status_runtime_fixture(operator, tmp_path, monkeypatch)
    binding = _runtime_settlement_binding()
    guard_active = {"value": False}

    @contextmanager
    def settled_guard(*_args, **_kwargs):
        guard_active["value"] = True
        try:
            yield {"settled": True}
        finally:
            guard_active["value"] = False

    monkeypatch.setattr(runtime, "hold_vrif_runtime_settlement", settled_guard)
    monkeypatch.setattr(
        runtime,
        "vrif_runtime_settlement_binding",
        lambda *_args, **_kwargs: binding,
    )
    monkeypatch.setattr(
        operator,
        "_vrif_runtime_target",
        lambda _config: (
            binding["target"]["repository_id"],
            config["merge_target_branch"],
        ),
    )

    def snapshot(*, runtime_settlement_binding, **_kwargs):
        assert guard_active["value"] is True
        assert runtime_settlement_binding == binding
        return (
            {"available": True, "task_count": 33},
            {"available": True, "all_goals_satisfied": True},
        )

    monkeypatch.setattr(operator, "_quack_status_authority_snapshot", snapshot)
    observed = operator.status(tmp_path / "config.json")
    assert guard_active["value"] is False
    assert observed["task_authority"]["available"] is True
    assert observed["goal_authority"]["all_goals_satisfied"] is True
    assert observed["runtime_settlement"] == {
        "available": True,
        "settled": True,
        "reason_code": "settled",
        "binding": binding,
    }
