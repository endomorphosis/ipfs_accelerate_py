from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.analyzer_health import (
    AnalysisEscalationPolicy,
)
from ipfs_accelerate_py.agent_supervisor.audit_scanner import (
    run_audit_scan,
    run_exhaustive_ast_coverage,
    run_low_backlog_analysis,
)
from ipfs_accelerate_py.agent_supervisor.objective_daemon import (
    run_objective_analysis_escalation,
)
from ipfs_accelerate_py.agent_supervisor.task_proposal_router import (
    StructuredPlanRouterConfig,
    generate_analysis_proposals,
)


def _branch(branch_id: str, *, source: str = "llm_router") -> dict[str, object]:
    return {
        "branch_id": branch_id,
        "summary": f"Implement {branch_id} with focused evidence.",
        "predicted_files": [f"src/{branch_id}.py"],
        "predicted_symbols": [f"build_{branch_id}"],
        "dependencies": [],
        "validation_commands": [f"pytest tests/test_{branch_id}.py -q"],
        "validation_proof": ["The focused test proves the objective behavior."],
        "estimated_cost": 1.0,
        "risk": 0.1,
        "expected_objective_delta": 0.8,
        "source": source,
    }


def _proposal(branch_id: str, *, confidence: float = 0.9, novelty: float = 0.9) -> dict[str, object]:
    return {
        "branch": _branch(branch_id),
        "confidence": confidence,
        "novelty": novelty,
        "objective_terms": ["prove cache invalidation"],
    }


def test_policy_escalates_static_ast_router_and_records_required_evidence(tmp_path: Path) -> None:
    calls: list[str] = []

    def static_scan() -> dict[str, object]:
        calls.append("static")
        return {
            "candidates": [],
            "healthy": True,
            "confidence": 1.0,
            "scope": {"files": 2},
            "cost": {"files_parsed": 2},
            "rejected_candidates": [{"reason": "seen", "candidate": "old"}],
        }

    def ast_scan() -> dict[str, object]:
        calls.append("ast")
        return {
            "candidates": [],
            "healthy": True,
            "confidence": 1.0,
            "scope": {"symbols": 4, "complete": True},
            "cost": {"source_bytes": 100},
        }

    def router(_prompt: str) -> str:
        calls.append("router")
        return json.dumps({"proposals": [_proposal("direct"), _proposal("layered")]})

    result = run_low_backlog_analysis(
        tmp_path,
        healthy_backlog_count=0,
        objective_terms=["prove cache invalidation"],
        policy=AnalysisEscalationPolicy(backlog_target=2, max_router_retries=0),
        incremental_scanner=static_scan,
        ast_scanner=ast_scan,
        router=router,
        router_config=StructuredPlanRouterConfig(
            repo_root=tmp_path,
            branch_count=2,
            max_new_tokens=128,
        ),
    )

    assert calls == ["static", "ast", "router"]
    assert result.backlog_satisfied
    assert not result.analysis_inconclusive
    assert not result.exhausted
    assert [item.stage.value for item in result.records] == [
        "incremental_static",
        "exhaustive_ast",
        "llm_router",
    ]
    for record in result.to_dict()["records"]:
        assert set(record) >= {
            "cost",
            "scope",
            "novelty",
            "confidence",
            "rejected_candidates",
            "objective_terms_attempted",
        }


def test_incremental_scan_short_circuits_more_expensive_analysis(tmp_path: Path) -> None:
    result = run_low_backlog_analysis(
        tmp_path,
        healthy_backlog_count=1,
        objective_terms=["term"],
        policy=AnalysisEscalationPolicy(backlog_target=2),
        incremental_scanner=lambda: [{"candidate": "new"}],
        ast_scanner=lambda: (_ for _ in ()).throw(AssertionError("AST must not run")),
        router=lambda _prompt: (_ for _ in ()).throw(AssertionError("router must not run")),
    )

    assert result.backlog_satisfied
    assert [item.stage.value for item in result.records] == ["incremental_static"]


def test_low_confidence_router_is_bounded_inconclusive_and_falls_back(tmp_path: Path) -> None:
    calls = 0

    def low_confidence_router(_prompt: str) -> str:
        nonlocal calls
        calls += 1
        return json.dumps({"proposals": [_proposal("weak", confidence=0.2)]})

    policy = AnalysisEscalationPolicy(
        backlog_target=3,
        max_router_calls=2,
        router_calls_per_window=10,
        max_router_tokens=128,
        max_router_retries=9,
        max_novel_proposals=2,
        min_confidence=0.8,
    )
    result = run_low_backlog_analysis(
        tmp_path,
        healthy_backlog_count=0,
        objective_terms=["prove cache invalidation"],
        policy=policy,
        incremental_scanner=lambda: [],
        ast_scanner=lambda: {"healthy": True, "complete": True, "candidates": []},
        router=low_confidence_router,
        router_config=StructuredPlanRouterConfig(
            repo_root=tmp_path,
            branch_count=1,
            max_new_tokens=64,
        ),
    )

    assert calls == 2  # retry and call caps agree with the aggregate token cap
    assert result.analysis_inconclusive
    assert result.deterministic_fallback
    assert not result.safe_for_completion_reasoning
    assert [item.stage.value for item in result.records][-2:] == [
        "llm_router",
        "deterministic_fallback",
    ]
    router_record = result.records[-2]
    assert router_record.cost["router_calls"] == 2
    assert router_record.cost["router_retries"] == 1
    assert router_record.cost["reserved_tokens"] == 128
    assert any(
        item["reason"] == "confidence_below_threshold"
        for item in router_record.rejected_candidates
    )


def test_rate_limit_prevents_router_call_but_still_returns_fallback(tmp_path: Path) -> None:
    routed = generate_analysis_proposals(
        {"task_id": "RATE-1", "predicted_files": ["src/rate.py"]},
        objective_terms=["prove cache invalidation"],
        router=lambda _prompt: (_ for _ in ()).throw(AssertionError("rate limited")),
        config=StructuredPlanRouterConfig(repo_root=tmp_path, branch_count=1, max_new_tokens=64),
        policy=AnalysisEscalationPolicy(
            router_calls_per_window=1,
            max_router_calls=2,
            max_router_tokens=64,
        ),
        router_calls_in_window=1,
    )

    assert routed.router_calls == 0
    assert routed.analysis_inconclusive
    assert routed.used_fallback
    assert routed.limit_reason == "router_rate_or_call_limit_reached"


def test_exhaustive_ast_coverage_uses_real_python_ast_and_reports_parse_health(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    source = tmp_path / "service.py"
    source.write_text("class Cache:\n    def invalidate(self):\n        return True\n", encoding="utf-8")
    subprocess.run(["git", "add", "service.py"], cwd=tmp_path, check=True)

    report = run_exhaustive_ast_coverage(
        tmp_path,
        objective_terms=["Cache.invalidate"],
        max_records=10,
        max_source_bytes=10000,
    )

    assert report.healthy
    assert report.complete
    assert report.expected_file_count == report.scanned_file_count == 1
    assert report.records[0]["ast_kind"] == "python_ast"
    assert report.term_evidence["Cache.invalidate"]


def test_inconclusive_escalation_downgrades_audit_exhaustion_receipt(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    source = tmp_path / "clean.py"
    source.write_text("def ready():\n    return True\n", encoding="utf-8")
    subprocess.run(["git", "add", "clean.py"], cwd=tmp_path, check=True)

    result = run_audit_scan(
        tmp_path,
        required_quorum=1,
        objective="prove clean runtime",
        persist=False,
        analysis_escalation={
            "status": "analysis_inconclusive",
            "analysis_inconclusive": True,
            "exhaustion_eligible": False,
        },
    )

    assert result.receipt.terminal_reason.value == "partial"
    assert not result.receipt.safe_for_completion_reasoning
    assert result.receipt.metadata["analysis_escalation"]["analysis_inconclusive"]


def test_objective_daemon_bridge_persists_complete_escalation_artifact(tmp_path: Path) -> None:
    objective_path = tmp_path / "objective.md"
    objective_path.write_text(
        "# Goals\n\n## G1 Cache proof\n\n- Status: active\n- Evidence: prove cache invalidation\n",
        encoding="utf-8",
    )
    artifact = tmp_path / "state" / "analysis.json"
    result = run_objective_analysis_escalation(
        repo_root=tmp_path,
        objective_path=objective_path,
        healthy_backlog_count=0,
        artifact_path=artifact,
        policy=AnalysisEscalationPolicy(
            backlog_target=1,
            max_router_calls=0,
        ),
        incremental_scanner=lambda: [{"candidate": "static work"}],
        ast_scanner=lambda: (_ for _ in ()).throw(AssertionError("already satisfied")),
    )

    persisted = json.loads(artifact.read_text(encoding="utf-8"))
    assert result.backlog_satisfied
    assert persisted["schema"].endswith("analysis-escalation@1")
    assert persisted["records"][0]["objective_terms_attempted"] == [
        "prove cache invalidation"
    ]
