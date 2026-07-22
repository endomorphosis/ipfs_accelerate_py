from __future__ import annotations

from datetime import datetime, timezone

from ipfs_accelerate_py.agent_supervisor.conflict_graph import (
    compare_surface_evidence,
    detect_surface_contradictions,
)
from ipfs_accelerate_py.agent_supervisor.goal_coverage import (
    CoverageStatus,
    UNMAPPED_GOAL_ID,
    attach_findings_to_goals,
    build_goal_coverage_map,
)
from ipfs_accelerate_py.agent_supervisor.objective_graph import (
    ObjectiveCoverageGraph,
    parse_goal_heap,
)
from ipfs_accelerate_py.agent_supervisor.todo_vector_index import (
    build_todo_coverage_inputs,
    parse_todo_vector_records,
)


EVALUATED_AT = datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc)
TREE = "sha256:current-tree"


def _goal(goal_id: str, criterion: str, *, title: str = "Coverage goal") -> dict[str, object]:
    return {
        "goal_id": goal_id,
        "title": title,
        "fields": {"acceptance": criterion},
    }


def _task(task_id: str, goal_id: str, criterion: str) -> dict[str, object]:
    return {
        "task_id": task_id,
        "goal_id": goal_id,
        "acceptance_criteria": [criterion],
        "predicted_files": [f"src/{task_id.lower()}.py"],
        "changed_paths": [f"src/{task_id.lower()}.py"],
        "ast_symbols": [f"{task_id}Service.run"],
        "interfaces": [f"{task_id}Protocol"],
        "validation_commands": [f"pytest -q tests/test_{task_id.lower()}.py"],
    }


def _receipt(task_id: str, criterion: str, **changes: object) -> dict[str, object]:
    receipt: dict[str, object] = {
        "task_id": task_id,
        "acceptance_criterion": criterion,
        "command": f"pytest -q tests/test_{task_id.lower()}.py",
        "passed": True,
        "repository_tree": TREE,
        "observed_at": "2026-07-22T11:55:00+00:00",
        "provenance_cid": f"bafy-{task_id.lower()}",
    }
    receipt.update(changes)
    return receipt


def test_complete_criterion_maps_every_required_surface_and_provenance() -> None:
    criterion = "The public API returns a provenance-backed result."
    coverage = build_goal_coverage_map(
        [_goal("G10.S3", criterion)],
        [_task("REF-207", "G10.S3", criterion)],
        validation_receipts=[_receipt("REF-207", criterion)],
        repository_tree=TREE,
        evaluated_at=EVALUATED_AT,
    )

    row = coverage.criteria[0]
    assert row.status is CoverageStatus.VERIFIED
    assert row.task_ids == ["REF-207"]
    assert row.predicted_files == ["src/ref-207.py"]
    assert row.changed_files == ["src/ref-207.py"]
    assert row.ast_symbols == ["REF-207Service.run"]
    assert row.interfaces == ["REF-207Protocol"]
    assert row.validation_commands == ["pytest -q tests/test_ref-207.py"]
    assert row.validation_receipt_ids
    assert row.provenance_cids == ["bafy-ref-207"]
    assert row.missing_surfaces == []
    assert all(edge.explanation for edge in coverage.edges)
    assert any(edge.provenance_cid == "bafy-ref-207" for edge in coverage.verified)

    graph = coverage.to_objective_graph()
    assert isinstance(graph, ObjectiveCoverageGraph)
    assert graph.status_counts["verified"] > 0
    assert coverage.to_dict()["objective_graph"] == graph.to_dict()


def test_graph_separates_uncovered_weak_stale_contradicted_and_verified() -> None:
    criteria = {
        name: f"{name} coverage criterion"
        for name in ("uncovered", "weak", "stale", "contradicted", "verified")
    }
    goals = [_goal(f"G-{name}", criterion) for name, criterion in criteria.items()]
    tasks = [
        _task(f"T-{name}", f"G-{name}", criterion)
        for name, criterion in criteria.items()
        if name != "uncovered"
    ]
    receipts = [
        _receipt(
            "T-stale",
            criteria["stale"],
            observed_at="2026-07-19T00:00:00+00:00",
        ),
        _receipt("T-contradicted", criteria["contradicted"], passed=False),
        _receipt("T-verified", criteria["verified"]),
    ]

    coverage = build_goal_coverage_map(
        goals,
        tasks,
        validation_receipts=receipts,
        repository_tree=TREE,
        evaluated_at=EVALUATED_AT,
        evidence_max_age_seconds=3600,
    )
    statuses = {item.goal_id: item.status.value for item in coverage.criteria}

    assert statuses == {
        "G-contradicted": "contradicted",
        "G-stale": "stale",
        "G-uncovered": "uncovered",
        "G-verified": "verified",
        "G-weak": "weakly_inferred",
    }
    artifact = coverage.to_dict()
    assert set(artifact["surfaces_by_status"]) == {
        "uncovered",
        "weakly_inferred",
        "stale",
        "contradicted",
        "verified",
    }
    assert all(artifact["surfaces_by_status"][name] for name in artifact["surfaces_by_status"])
    assert artifact["criterion_status_counts"] == {
        "uncovered": 1,
        "weakly_inferred": 1,
        "stale": 1,
        "contradicted": 1,
        "verified": 1,
    }


def test_dynamic_findings_attach_to_best_registered_goal_and_keep_unmapped_bucket() -> None:
    goals = [
        _goal("G-CACHE", "Cache invalidation preserves runtime consistency.", title="Runtime cache invalidation"),
        _goal("G-UI", "Navigation controls meet accessibility rules.", title="Accessible navigation UI"),
    ]
    findings = [
        {
            "fingerprint": "finding-cache",
            "summary": "runtime cache invalidation inconsistency",
            "missing_evidence": ["cache consistency"],
        },
        {
            "fingerprint": "finding-orphan",
            "summary": "quantum satellite telemetry ephemeris",
        },
    ]

    assignments = attach_findings_to_goals(goals, findings)

    assert assignments[0].finding_id == "finding-cache"
    assert assignments[0].goal_id == "G-CACHE"
    assert assignments[0].inferred is True
    orphan = next(item for item in assignments if item.finding_id == "finding-orphan")
    assert orphan.goal_id == UNMAPPED_GOAL_ID
    assert "threshold" in orphan.explanation

    coverage = build_goal_coverage_map(goals, findings=findings, evaluated_at=EVALUATED_AT)
    bucket = coverage.to_dict()["unmapped_bucket"]
    assert bucket["goal_id"] == UNMAPPED_GOAL_ID
    assert bucket["label"] == "Unmapped dynamic findings"
    assert [item["finding_id"] for item in bucket["findings"]] == ["finding-orphan"]


def test_coverage_calculation_is_order_independent_and_edges_explain_evidence() -> None:
    criterion = "Persist deterministic coverage evidence."
    goals = [_goal("G-DETERMINISTIC", criterion)]
    tasks = [
        _task("T-B", "G-DETERMINISTIC", criterion),
        _task("T-A", "G-DETERMINISTIC", criterion),
    ]
    receipts = [_receipt("T-B", criterion), _receipt("T-A", criterion)]

    left = build_goal_coverage_map(
        goals,
        tasks,
        validation_receipts=receipts,
        repository_tree=TREE,
        evaluated_at=EVALUATED_AT,
    )
    right = build_goal_coverage_map(
        list(reversed(goals)),
        list(reversed(tasks)),
        validation_receipts=list(reversed(receipts)),
        repository_tree=TREE,
        evaluated_at=EVALUATED_AT,
    )

    assert left.graph_id == right.graph_id
    assert left.to_dict() == right.to_dict()
    assert [edge.edge_id for edge in left.edges] == sorted(edge.edge_id for edge in left.edges)
    for edge in left.edges:
        assert edge.explanation
        assert edge.evidence


def test_todo_vector_index_preserves_criterion_receipt_and_unmapped_inputs(tmp_path) -> None:
    source = tmp_path / "src" / "service.py"
    source.parent.mkdir()
    source.write_text("class Service:\n    def run(self):\n        return True\n", encoding="utf-8")
    todo = tmp_path / "todo.md"
    todo.write_text(
        """# Board

## REF-207 Coverage implementation
- Status: todo
- Priority: P0
- Track: g10
- Goal id: G10.S3
- Outputs: src/service.py
- Predicted files: src/service.py
- Changed paths: src/service.py
- Interfaces: CoverageProtocol
- Validation: pytest -q test/api/test_coverage.py
- Acceptance: Maps criterion one; maps criterion two
- Validation receipts: [{"receipt_cid":"bafy-receipt","provenance_cid":"bafy-proof","passed":true}]

## REF-ORPHAN Orphan scan
- Status: todo
- Acceptance: Investigate an unregistered surface
""",
        encoding="utf-8",
    )

    records = parse_todo_vector_records(
        repo_root=tmp_path,
        todo_path=todo,
        task_header_prefix="REF-",
    )
    payload = build_todo_coverage_inputs(records)

    registered = next(record for record in records if record.task_id == "REF-207")
    assert registered.acceptance_criteria == ["Maps criterion one", "maps criterion two"]
    assert registered.provenance_cids == ["bafy-proof"]
    assert registered.validation_receipts[0]["receipt_cid"] == "bafy-receipt"
    assert payload["by_goal"]["G10.S3"]["task_ids"] == ["REF-207"]
    assert payload["unmapped_task_ids"] == ["REF-ORPHAN"]
    assert payload == build_todo_coverage_inputs(list(reversed(records)))
    assert all(edge["explanation"]["deterministic"] for edge in payload["edges"])


def test_predicted_and_observed_surface_evidence_is_explainable_and_conservative() -> None:
    predicted = {
        "predicted_files": ["src"],
        "ast_symbols": ["Service.run", "Service.stop"],
        "interfaces": ["ServiceProtocol"],
    }
    observed = {
        "changed_paths": ["src/service.py", "docs/notes.md"],
        "observed_ast_symbols": ["Service.run"],
        "observed_interfaces": ["ServiceProtocol"],
    }

    comparison = compare_surface_evidence(predicted, observed)

    assert comparison.matched_paths == ["src/service.py"]
    assert comparison.missing_symbols == ["Service.stop"]
    assert comparison.unexpected_paths == ["docs/notes.md"]
    assert comparison.coverage_ratio == 0.75
    assert comparison.explanations
    assert detect_surface_contradictions(comparison).contradicted is False

    strict = detect_surface_contradictions(
        comparison,
        missing_is_contradiction=True,
        unexpected_is_contradiction=True,
    )
    assert strict.contradicted is True
    assert {item.kind for item in strict.contradictions} == {
        "missing_expected_surface",
        "unexpected_observed_surface",
    }


def test_parsed_objective_goal_criteria_feed_coverage_map() -> None:
    goals = parse_goal_heap(
        """# Heap

## G10.S3 Coverage objective
- Status: active
- Acceptance: criterion alpha; criterion beta
"""
    )

    coverage = build_goal_coverage_map(goals, evaluated_at=EVALUATED_AT)

    assert [item.criterion for item in coverage.criteria] == ["criterion alpha", "criterion beta"]
    assert all(item.status is CoverageStatus.UNCOVERED for item in coverage.criteria)
