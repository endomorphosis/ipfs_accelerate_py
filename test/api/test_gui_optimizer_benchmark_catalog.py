"""VGO-083: exactly 15 controlled GUI optimizer benchmark tasks.

Acceptance coverage:

* the catalog rejects any count other than 15 and duplicate IDs
* every required kind is present once with one or two measurable objectives,
  bounded files, controlled fixtures, hard gates, an expected decision,
  a route, and an evidence class
* rerunning catalog creation is byte-identical
* the durable fixture matches the sealed builder
* subjective primary-action hierarchy cannot be auto-accepted
* fixtures stay nonproduction and screen-bounded
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.gui_optimizer.benchmark import (
    BENCHMARK_ID,
    CATALOG_ID,
    CONFLICT_POLICY,
    DEFAULT_CATALOG_RELATIVE_PATH,
    EXPECTED_TASK_COUNT,
    GUI_BENCHMARK_RESULT_INTERFACE,
    GUI_BENCHMARK_RESULT_SCHEMA,
    GUI_BENCHMARK_TASK_INTERFACE,
    GUI_OPTIMIZATION_BENCHMARK_INTERFACE,
    REQUIRED_TASK_KINDS,
    SUBJECTIVE_KINDS,
    BenchmarkDecision,
    BenchmarkReasonCode,
    BenchmarkRouteKind,
    BenchmarkTaskKind,
    GuiBenchmarkError,
    GuiBenchmarkResult,
    GuiBenchmarkTask,
    GuiOptimizationBenchmark,
    build_benchmark_catalog,
    default_catalog_path,
    empty_benchmark_result,
    load_benchmark_catalog,
    main,
    materialize_catalog_fixture,
    render_catalog_document,
    sealed_benchmark_tasks,
    write_catalog_fixture,
)
from ipfs_accelerate_py.agent_supervisor.gui_optimizer.cli import (
    BENCHMARK_REGISTRY,
)
from ipfs_accelerate_py.agent_supervisor.gui_optimizer.authority import (
    DEFAULT_ALLOWED_ROOTS,
    path_under_allowed_roots,
)


FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "gui_optimizer"
    / "benchmark-tasks.json"
)


def _task_payload(task: GuiBenchmarkTask | None = None, **overrides: Any) -> dict[str, Any]:
    payload = (task or sealed_benchmark_tasks()[0]).to_dict()
    payload.update(overrides)
    return payload


def _catalog_payload(
    catalog: GuiOptimizationBenchmark | None = None, **overrides: Any
) -> dict[str, Any]:
    payload = (catalog or build_benchmark_catalog()).to_dict()
    payload.update(overrides)
    return payload


def test_catalog_contains_exactly_fifteen_unique_required_kinds() -> None:
    catalog = build_benchmark_catalog()
    assert len(catalog.tasks) == EXPECTED_TASK_COUNT
    assert EXPECTED_TASK_COUNT == 15
    assert [task.kind for task in catalog.tasks] == list(REQUIRED_TASK_KINDS)
    assert len({task.task_id for task in catalog.tasks}) == 15
    assert len({task.kind for task in catalog.tasks}) == 15
    assert set(REQUIRED_TASK_KINDS) == {item.value for item in BenchmarkTaskKind}


def test_catalog_rejects_any_count_other_than_fifteen() -> None:
    tasks = sealed_benchmark_tasks()
    with pytest.raises(GuiBenchmarkError) as too_few:
        GuiOptimizationBenchmark(
            benchmark_id=BENCHMARK_ID,
            catalog_id=CATALOG_ID,
            tasks=tasks[:-1],
        )
    assert too_few.value.reason_code == BenchmarkReasonCode.TASK_COUNT_MISMATCH.value

    extra = GuiBenchmarkTask.from_mapping(
        _task_payload(tasks[0], task_id="task:extra-overflow", kind="responsive_overflow")
    )
    with pytest.raises(GuiBenchmarkError) as too_many:
        GuiOptimizationBenchmark(
            benchmark_id=BENCHMARK_ID,
            catalog_id=CATALOG_ID,
            tasks=tasks + (extra,),
        )
    assert too_many.value.reason_code == BenchmarkReasonCode.TASK_COUNT_MISMATCH.value

    payload = _catalog_payload()
    payload["tasks"] = payload["tasks"][:3]
    payload["expected_task_count"] = 3
    with pytest.raises(GuiBenchmarkError) as wire:
        GuiOptimizationBenchmark.from_mapping(payload)
    assert wire.value.reason_code == BenchmarkReasonCode.TASK_COUNT_MISMATCH.value


def test_catalog_rejects_duplicate_task_ids() -> None:
    tasks = list(sealed_benchmark_tasks())
    duplicate = GuiBenchmarkTask.from_mapping(
        _task_payload(tasks[1], task_id=tasks[0].task_id)
    )
    mutated = tuple(tasks[:1] + [duplicate] + tasks[2:])
    with pytest.raises(GuiBenchmarkError) as exc:
        GuiOptimizationBenchmark(
            benchmark_id=BENCHMARK_ID,
            catalog_id=CATALOG_ID,
            tasks=mutated,
        )
    assert exc.value.reason_code == BenchmarkReasonCode.DUPLICATE_TASK_ID.value


def test_every_task_has_required_bounded_contract_fields() -> None:
    catalog = build_benchmark_catalog()
    assert catalog.benchmark_id == BENCHMARK_ID
    assert catalog.catalog_id == CATALOG_ID
    assert catalog.application_id == "app:agent-supervisor"
    assert catalog.screen_id == "screen:agent-supervisor"
    assert catalog.conflict_policy == CONFLICT_POLICY
    assert catalog.uses_production_services is False
    assert catalog.uses_production_credentials is False
    for task in catalog.tasks:
        assert task.interface == GUI_BENCHMARK_TASK_INTERFACE
        assert 1 <= len(task.objective_ids) <= 2
        assert 1 <= len(task.objective_metric_ids) <= 2
        assert task.bounded_file_paths
        assert task.controlled_fixture_ids
        assert task.hard_gate_ids
        assert task.expected_decision in {
            BenchmarkDecision.ACCEPT.value,
            BenchmarkDecision.REJECT.value,
            BenchmarkDecision.HUMAN_REVIEW.value,
        }
        assert task.expected_route in {item.value for item in BenchmarkRouteKind}
        assert task.evidence_class
        assert task.baseline_id.startswith("baseline:")
        assert task.reference_id.startswith("ref:vgo-083-")
        assert task.raw_retrieval_token_estimate >= 1
        assert task.affected_component_ids
        assert task.affected_scenario_ids
        assert task.affected_check_ids
        assert task.route_id == "route:agent-supervisor"
        assert task.application_id == "app:agent-supervisor"
        assert task.screen_id == "screen:agent-supervisor"
        for path in task.bounded_file_paths:
            assert path_under_allowed_roots(path, allowed_roots=DEFAULT_ALLOWED_ROOTS)


def test_catalog_creation_is_byte_identical() -> None:
    first = build_benchmark_catalog()
    second = build_benchmark_catalog()
    assert first.canonical_bytes() == second.canonical_bytes()
    assert first.fixture_bytes() == second.fixture_bytes()
    assert first.catalog_identity() == second.catalog_identity()
    assert render_catalog_document() == render_catalog_document(first)
    assert first.to_dict() == second.to_dict()


def test_fixture_matches_sealed_builder() -> None:
    assert FIXTURE_PATH.is_file()
    fixture_bytes = FIXTURE_PATH.read_bytes()
    catalog = load_benchmark_catalog(FIXTURE_PATH)
    built = build_benchmark_catalog()
    assert json.loads(fixture_bytes.decode("utf-8")) == built.to_dict()
    assert catalog.canonical_bytes() == built.canonical_bytes()
    assert catalog.to_dict() == built.to_dict()
    assert default_catalog_path() == FIXTURE_PATH
    assert render_catalog_document(built) == render_catalog_document()
    assert FIXTURE_PATH.read_bytes() == render_catalog_document(built)


def test_cli_registry_points_at_the_declared_catalog() -> None:
    spec = BENCHMARK_REGISTRY["benchmark-v1"]
    assert spec.benchmark_id == BENCHMARK_ID
    assert spec.expected_tasks == EXPECTED_TASK_COUNT
    assert spec.catalog_path == DEFAULT_CATALOG_RELATIVE_PATH
    loaded = load_benchmark_catalog(
        Path(__file__).resolve().parents[4] / spec.catalog_path
    )
    assert loaded.canonical_bytes() == build_benchmark_catalog().canonical_bytes()


def test_primary_action_hierarchy_cannot_be_auto_accepted() -> None:
    catalog = build_benchmark_catalog()
    task = catalog.task_by_id("task:primary-action-hierarchy")
    assert task.kind in SUBJECTIVE_KINDS
    assert task.expected_decision == BenchmarkDecision.HUMAN_REVIEW.value
    assert task.expected_route == BenchmarkRouteKind.HUMAN_REVIEW.value
    assert task.declared_tier == "human"
    with pytest.raises(GuiBenchmarkError) as exc:
        GuiBenchmarkTask.from_mapping(
            _task_payload(
                task,
                expected_decision="accept",
                expected_route="deterministic_transform",
                declared_tier="deterministic",
            )
        )
    assert (
        exc.value.reason_code
        == BenchmarkReasonCode.SUBJECTIVE_AUTO_ACCEPT_FORBIDDEN.value
    )


def test_interaction_and_binding_tasks_keep_hard_gates() -> None:
    catalog = build_benchmark_catalog()
    steps = catalog.task_by_id("task:interaction-step-reduction")
    assert "gate:no-confirmation-regression" in steps.hard_gate_ids
    assert "check:confirmation" in steps.affected_check_ids
    binding = catalog.task_by_id("task:action-binding-integrity")
    assert "check:policy" in binding.affected_check_ids
    assert "check:confirmation" in binding.affected_check_ids
    assert "check:host-boundary" in binding.affected_check_ids


def test_closed_wire_inputs_reject_unknown_fields_and_wrong_containers() -> None:
    payload = _task_payload()
    payload["vendor"] = "hidden"
    with pytest.raises(GuiBenchmarkError) as unknown:
        GuiBenchmarkTask.from_mapping(payload)
    assert unknown.value.reason_code == BenchmarkReasonCode.UNKNOWN_FIELD.value

    catalog_payload = _catalog_payload()
    catalog_payload["tasks"] = tuple(catalog_payload["tasks"])
    with pytest.raises(GuiBenchmarkError) as tuples:
        GuiOptimizationBenchmark.from_mapping(catalog_payload)
    assert tuples.value.reason_code == BenchmarkReasonCode.INVALID_COLLECTION_TYPE.value

    result_payload = empty_benchmark_result(sealed_benchmark_tasks()[0]).to_dict()
    result_payload["artifact_ids"] = None
    with pytest.raises(GuiBenchmarkError) as nulls:
        GuiBenchmarkResult.from_mapping(result_payload)
    assert nulls.value.reason_code == BenchmarkReasonCode.INVALID_COLLECTION_TYPE.value


def test_paths_and_rewrite_language_are_fail_closed() -> None:
    with pytest.raises(GuiBenchmarkError) as outside:
        GuiBenchmarkTask.from_mapping(
            _task_payload(bounded_file_paths=["src/services/mcp/all-app-tool-gateway.ts"])
        )
    assert (
        outside.value.reason_code
        == BenchmarkReasonCode.PATH_OUTSIDE_ALLOWED_ROOTS.value
    )
    with pytest.raises(GuiBenchmarkError) as traversal:
        GuiBenchmarkTask.from_mapping(
            _task_payload(
                bounded_file_paths=["swissknife/web/js/apps/../../secrets.env"]
            )
        )
    assert (
        traversal.value.reason_code
        == BenchmarkReasonCode.PATH_ABSOLUTE_OR_TRAVERSAL.value
    )
    with pytest.raises(GuiBenchmarkError) as rewrite:
        GuiBenchmarkTask.from_mapping(
            _task_payload(title="Redesign the application to make the app generally better.")
        )
    assert rewrite.value.reason_code == BenchmarkReasonCode.WHOLE_APP_REWRITE.value


def test_write_and_materialize_catalog_fixture(tmp_path: Path) -> None:
    target = tmp_path / "gui_optimizer" / "benchmark-tasks.json"
    written = write_catalog_fixture(target)
    assert written == target
    built = build_benchmark_catalog()
    assert target.read_bytes() == render_catalog_document(built)
    assert load_benchmark_catalog(target).canonical_bytes() == built.canonical_bytes()
    again = tmp_path / "alias.json"
    assert materialize_catalog_fixture(again).read_bytes() == target.read_bytes()
    assert main(["write", str(tmp_path / "cli.json")]) == 0
    assert (tmp_path / "cli.json").read_bytes() == target.read_bytes()
    assert main(["materialize", str(tmp_path / "cli-materialize.json")]) == 0
    assert (tmp_path / "cli-materialize.json").read_bytes() == target.read_bytes()


def test_missing_catalog_is_unavailable(tmp_path: Path) -> None:
    missing = tmp_path / "missing-benchmark-tasks.json"
    with pytest.raises(GuiBenchmarkError) as exc:
        load_benchmark_catalog(missing)
    assert exc.value.reason_code == BenchmarkReasonCode.CATALOG_UNAVAILABLE.value


def test_benchmark_result_round_trip_and_pending_shell() -> None:
    task = sealed_benchmark_tasks()[0]
    pending = empty_benchmark_result(task)
    assert pending.interface == GUI_BENCHMARK_RESULT_INTERFACE
    assert pending.schema_version == GUI_BENCHMARK_RESULT_SCHEMA
    assert pending.decision == BenchmarkDecision.PENDING.value
    assert pending.task_id == task.task_id
    decoded = GuiBenchmarkResult.from_mapping(pending.to_dict())
    assert decoded.canonical_bytes() == pending.canonical_bytes()
    accepted = GuiBenchmarkResult.from_mapping(
        {
            **pending.to_dict(),
            "decision": "accept",
            "measurable_improvement": True,
            "hard_gate_passed": True,
            "receipt_id": "receipt:focus-restoration",
            "artifact_ids": ["cid:artifact-focus"],
            "reason_codes": ["accepted"],
            "metric_values": {"focus_restoration_coverage": 1},
        }
    )
    assert accepted.decision == BenchmarkDecision.ACCEPT.value
    assert accepted.metric_values["focus_restoration_coverage"] == 1


def test_production_surfaces_and_missing_kinds_are_rejected() -> None:
    payload = _catalog_payload(uses_production_services=True)
    with pytest.raises(GuiBenchmarkError) as production:
        GuiOptimizationBenchmark.from_mapping(payload)
    assert (
        production.value.reason_code
        == BenchmarkReasonCode.PRODUCTION_SURFACE_FORBIDDEN.value
    )
    tasks = list(sealed_benchmark_tasks())
    mutated = tuple(
        tasks[:1]
        + [
            GuiBenchmarkTask.from_mapping(
                _task_payload(tasks[1], kind=tasks[0].kind, task_id="task:dup-kind")
            )
        ]
        + tasks[2:]
    )
    with pytest.raises(GuiBenchmarkError) as duplicate_kind:
        GuiOptimizationBenchmark(
            benchmark_id=BENCHMARK_ID,
            catalog_id=CATALOG_ID,
            tasks=mutated,
        )
    assert (
        duplicate_kind.value.reason_code
        == BenchmarkReasonCode.DUPLICATE_TASK_KIND.value
    )


def test_catalog_interfaces_are_the_declared_board_contracts() -> None:
    catalog = build_benchmark_catalog()
    assert catalog.interface == GUI_OPTIMIZATION_BENCHMARK_INTERFACE
    assert catalog.interface == "GuiOptimizationBenchmark@1"
    assert sealed_benchmark_tasks()[0].interface == "GuiBenchmarkTask@1"
    assert empty_benchmark_result(sealed_benchmark_tasks()[0]).interface == (
        "GuiBenchmarkResult@1"
    )
    raw = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert raw["interface"] == "GuiOptimizationBenchmark@1"
    assert raw["expected_task_count"] == 15
    assert len(raw["tasks"]) == 15
