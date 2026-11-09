"""Blocked goals get a legal next step without a board edit or a model call."""

from __future__ import annotations

import json
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.rescue.goal_redesign import (
    closed_lineage_planned,
    correct_completion_claims,
    enforce_completion_claims,
    redesign_goals,
    truncated_phase_retained,
    truncated_spent_phase,
    waiting_official_release,
    write_goal_redesign,
)


def _tasks() -> dict[str, dict[str, str]]:
    return {
        "JLR-021": {"status": "completed", "depends_on": "", "goal_id": "JLR-G004"},
        "JLR-022": {"status": "planned", "depends_on": "JLR-021", "goal_id": "JLR-G004"},
        "JLR-024": {"status": "pending", "depends_on": "JLR-021", "goal_id": "JLR-G005"},
        "JLR-025": {"status": "planned", "depends_on": "JLR-024", "goal_id": "JLR-G005"},
        "JLR-035": {"status": "pending", "depends_on": "JLR-021", "goal_id": "JLR-G009"},
    }


def _goals() -> dict[str, dict]:
    return {
        "JLR-022": {
            "id": "JLR-022",
            "goal_id": "JLR-G004",
            "work_package": {
                "action": "planned_experiment",
                "activation_requires": ["frozen candidate/source manifests"],
            },
        }
    }


def _by_action(plan: dict) -> dict[str, dict]:
    return {item["task_id"]: item for item in plan["decisions"]}


def test_dependency_ready_planned_work_stays_planned() -> None:
    plan = redesign_goals(_tasks(), goal_tasks=_goals())
    decision = _by_action(plan)["JLR-022"]
    assert decision["action"] == "review_execution_contract"
    assert decision["activation_requires"] == ["frozen candidate/source manifests"]
    assert decision["board_mutation"] is False
    assert plan["board_mutations"] == []
    assert plan["completion_authority"] is False
    assert plan["idle_incomplete"] is True
    subgoal = next(item for item in plan["subgoals"] if item["task_id"] == "JLR-022")
    assert subgoal["admission"] == "proposal_only"
    assert subgoal["subgoal_id"] == "JLR-022:review_execution_contract"
    assert subgoal["board_mutation"] is False
    assert plan["llm"]["used"] is False


def test_spent_phase_and_missing_cohort_are_named() -> None:
    plan = redesign_goals(
        _tasks(),
        gate_reasons={
            "JLR-024": "phase_already_attempted",
            "JLR-035": "official full cohort unavailable",
        },
    )
    decisions = _by_action(plan)
    assert decisions["JLR-024"]["action"] == "register_reviewed_phase"
    assert decisions["JLR-024"]["provider_calls"] is False
    assert decisions["JLR-025"]["action"] == "wait_dependencies"
    assert decisions["JLR-035"]["action"] == "retry_external_release"
    assert "invent" in decisions["JLR-035"]["reason"]


def test_provider_freeze_does_not_call_the_model() -> None:
    def completer(_request: dict) -> dict:
        raise AssertionError("completer must not run")

    plan = redesign_goals(_tasks(), llm_complete=completer)
    assert plan["mechanism"] == "deterministic"
    assert plan["llm"]["reason"] == "provider_calls_disabled"


def test_llm_proposals_cannot_rewrite_a_sealed_task() -> None:
    def completer(_request: dict) -> dict:
        return {"proposals": [
            {"task_id": "JLR-022", "action": "set_pending", "rationale": "flip the row"},
            {"task_id": "JLR-022B", "goal_id": "JLR-G004", "action": "prepare_contract",
             "rationale": "draft only", "provider_calls": True},
            {"task_id": "JLR-022C", "goal_id": "JLR-G004", "action": "prepare_contract",
             "rationale": "advisory draft"},
        ]}

    plan = redesign_goals(_tasks(), allow_provider_calls=True, llm_complete=completer)
    assert plan["mechanism"] == "deterministic+llm"
    assert plan["llm"]["proposals"] == [{
        "task_id": "JLR-022C",
        "goal_id": "JLR-G004",
        "action": "prepare_contract",
        "rationale": "advisory draft",
        "admission": "proposal_only",
        "board_mutation": False,
        "completion_authority": False,
    }]
    assert _by_action(plan)["JLR-022"]["status"] == "planned"


def _seal_closed_lineage(root: Path, *, live_contracts: dict | None = None) -> None:
    contracts = {"JLR-024": {"action": "pilot"}, "JLR-035": {"action": "release"}}
    baseline = root / "data/lean_refactor_arena_20261024/execution-v2/history/baseline.json"
    baseline.parent.mkdir(parents=True, exist_ok=True)
    baseline.write_text(json.dumps({"contracts": contracts}), encoding="utf-8")
    (root / "config").mkdir(exist_ok=True)
    (root / "config/lean_refactor_arena_20261024_execution_v2.json").write_text(
        json.dumps({"contracts": live_contracts or contracts}), encoding="utf-8")
    (root / "tools").mkdir(exist_ok=True)
    (root / "tools/lean_refactor_compiled_sources_v1.json").write_text(json.dumps({
        "baseline_profile": "data/lean_refactor_arena_20261024/execution-v2/history/baseline.json",
    }), encoding="utf-8")


def _official_release(**overrides: object) -> dict:
    document = {
        "status": "WAITING_RELEASE",
        "reason": "official_full_cohort_file_unavailable",
        "model_calls": 0,
        "completion_authority": False,
        "proof_authority": False,
        "full_campaign_complete": False,
    }
    document.update(overrides)
    return document


def test_closed_lineage_retains_the_planned_row(tmp_path: Path) -> None:
    _seal_closed_lineage(tmp_path)
    assert closed_lineage_planned(tmp_path) is True
    receipt = tmp_path / "lane" / "goal-redesign.json"
    plan = write_goal_redesign(tmp_path, _tasks(), receipt)
    decision = _by_action(plan)["JLR-022"]
    assert decision["action"] == "retain_closed_lineage"
    assert decision["status"] == "planned"
    assert decision["completion_authority"] is False
    assert decision["board_mutation"] is False
    assert plan["board_mutations"] == []
    assert plan["completion_authority"] is False
    assert "do not flip it to pending" in decision["reason"]
    subgoal = next(item for item in plan["subgoals"] if item["task_id"] == "JLR-022")
    assert subgoal["subgoal_id"] == "JLR-022:retain_closed_lineage"
    assert subgoal["admission"] == "proposal_only"


def test_drifted_contract_set_stays_reviewable(tmp_path: Path) -> None:
    _seal_closed_lineage(tmp_path, live_contracts={"JLR-022": {"action": "portfolio"}, "JLR-024": {"action": "pilot"}})
    assert closed_lineage_planned(tmp_path) is False
    receipt = tmp_path / "lane" / "goal-redesign.json"
    plan = write_goal_redesign(tmp_path, _tasks(), receipt)
    assert _by_action(plan)["JLR-022"]["action"] == "review_execution_contract"
    assert _by_action(plan)["JLR-022"]["status"] == "planned"


def test_official_release_is_retained_without_completion() -> None:
    document = _official_release()
    assert waiting_official_release(document) is True
    claimed = _official_release(completion_authority=True)
    assert waiting_official_release(claimed) is False
    called = _official_release(model_calls=1)
    assert waiting_official_release(called) is False
    plan = redesign_goals(_tasks(), phase_results={"JLR-035": document})
    decision = _by_action(plan)["JLR-035"]
    assert decision["action"] == "retain_official_release"
    assert decision["completion_authority"] is False
    assert decision["provider_calls"] is False
    assert "do not invent the file" in decision["reason"]
    assert plan["work_complete"] is False


def test_waiting_release_result_names_the_cohort_gate(tmp_path: Path) -> None:
    result = tmp_path / "data/lean_refactor_arena_20261024/execution-v2/JLR-035.result.json"
    result.parent.mkdir(parents=True)
    result.write_text(json.dumps({
        "status": "WAITING_RELEASE",
        "reason": "official_full_cohort_file_unavailable",
        "completion_authority": False,
    }), encoding="utf-8")
    receipt = tmp_path / "lane" / "goal-redesign.json"
    plan = write_goal_redesign(tmp_path, _tasks(), receipt)
    decision = _by_action(plan)["JLR-035"]
    assert decision["action"] == "retry_external_release"
    assert decision["board_mutation"] is False


def _attempt(body: str, **extra: object) -> dict:
    attempt = {
        "status": "RETURNED",
        "http_status": 200,
        "candidate_error": "Expecting value: line 1 column 1 (char 0)",
        "response": {"choices": [{"finish_reason": "length", "message": {"content": body}}]},
    }
    attempt.update(extra)
    return attempt


def _truncated_document() -> dict:
    body = '```json\n{\n  "proof": "intros H'
    return {
        "status": "NO_CONFIRMED_WINNER",
        "winner": None,
        "completion_authority": False,
        "proof_authority": False,
        "promoted": False,
        "full_campaign_complete": False,
        "local_inference_calls": 2,
        "limits": {"max_calls": 2, "max_output_tokens": 256},
        "attempts": [_attempt(body), _attempt(body + "\n  case abs ih =>")],
    }


def test_length_truncated_phase_is_retained_without_completion() -> None:
    document = _truncated_document()
    assert truncated_spent_phase(document) is True
    plan = redesign_goals(
        _tasks(),
        gate_reasons={"JLR-024": "phase_already_attempted"},
        phase_results={"JLR-024": document},
    )
    decision = _by_action(plan)["JLR-024"]
    assert decision["action"] == "retain_truncated_phase"
    assert decision["provider_calls"] is False
    assert decision["board_mutation"] is False
    assert decision["completion_authority"] is False
    assert plan["completion_authority"] is False
    assert plan["work_complete"] is False
    assert "do not spend another call" in decision["reason"]
    subgoal = next(item for item in plan["subgoals"] if item["task_id"] == "JLR-024")
    assert subgoal["subgoal_id"] == "JLR-024:retain_truncated_phase"
    assert subgoal["admission"] == "proposal_only"
    assert subgoal["completion_authority"] is False


def test_closed_or_unspent_phase_stays_reviewable() -> None:
    closed = _truncated_document()
    closed["attempts"][1]["response"]["choices"][0]["message"]["content"] = (
        '```json\n{"proof": "intros Hnorm; simp"}\n```'
    )
    assert truncated_spent_phase(closed) is False
    plan = redesign_goals(
        _tasks(),
        gate_reasons={"JLR-024": "phase_already_attempted"},
        phase_results={"JLR-024": closed},
    )
    assert _by_action(plan)["JLR-024"]["action"] == "register_reviewed_phase"
    claimed = _truncated_document()
    claimed["completion_authority"] = True
    assert truncated_spent_phase(claimed) is False
    winner = _truncated_document()
    winner["winner"] = {"source": "theorem extractedOldExprInVars"}
    assert truncated_spent_phase(winner) is False


def test_completion_claims_are_cleared_and_not_granted(tmp_path: Path) -> None:
    document = {
        "completion_authority": True,
        "work_complete": True,
        "board_mutations": [],
        "subgoals": [{"task_id": "JLR-024", "completion_authority": True, "action": "retain_truncated_phase"}],
        "decisions": [{
            "task_id": "JLR-024",
            "action": "retain_truncated_phase",
            "completion_authority": "yes",
            "board_mutation": False,
        }],
    }
    corrected, count = correct_completion_claims(document)
    assert count == 4
    assert corrected["completion_authority"] is False
    assert corrected["work_complete"] is False
    assert corrected["subgoals"][0]["completion_authority"] is False
    assert corrected["decisions"][0]["completion_authority"] is False
    assert correct_completion_claims(corrected)[1] == 0
    finished = {
        "completion_authority": False,
        "work_complete": True,
        "board_mutations": [],
        "subgoals": [],
        "decisions": [{
            "task_id": "JLR-001",
            "action": "task_status_complete",
            "completion_authority": False,
            "board_mutation": False,
        }],
    }
    kept, kept_count = correct_completion_claims(finished)
    assert kept_count == 0
    assert kept["work_complete"] is True
    assert kept["completion_authority"] is False
    receipt = tmp_path / "goal-redesign.json"
    receipt.write_text(json.dumps(document), encoding="utf-8")
    assert enforce_completion_claims(receipt) == 4
    stored = json.loads(receipt.read_text(encoding="utf-8"))
    assert stored["completion_authority"] is False
    assert stored["work_complete"] is False
    stamp = receipt.stat().st_mtime_ns
    assert enforce_completion_claims(receipt) == 0
    assert receipt.stat().st_mtime_ns == stamp
    link = tmp_path / "linked.json"
    link.symlink_to(receipt)
    assert enforce_completion_claims(link) == 0
    assert link.is_symlink()


def test_receipt_retains_a_truncated_result_without_editing_it(tmp_path: Path) -> None:
    result = tmp_path / "data/lean_refactor_arena_20261024/execution-v2/JLR-024.result.json"
    result.parent.mkdir(parents=True)
    original = json.dumps(_truncated_document())
    result.write_text(original, encoding="utf-8")
    gate = tmp_path / "data/lean_refactor_arena_20261024/execution-v2/iterative-admission-gate.json"
    gate.write_text(json.dumps({
        "task_id": "JLR-024",
        "reason": "phase_already_attempted",
        "completion_authority": False,
    }), encoding="utf-8")
    receipt = tmp_path / "lane" / "goal-redesign.json"
    plan = write_goal_redesign(tmp_path, _tasks(), receipt)
    assert truncated_phase_retained(tmp_path) is True
    assert _by_action(plan)["JLR-024"]["action"] == "retain_truncated_phase"
    assert plan["board_mutations"] == []
    assert plan["completion_authority"] is False
    assert result.read_text(encoding="utf-8") == original
    stamp = receipt.stat().st_mtime_ns
    assert write_goal_redesign(tmp_path, _tasks(), receipt)["decisions"] == plan["decisions"]
    assert receipt.stat().st_mtime_ns == stamp


def test_receipt_write_is_stable(tmp_path: Path) -> None:
    goals = tmp_path / "implementation_plan/lean_refactor_arena_20261025.continuation.goals.json"
    goals.parent.mkdir(parents=True)
    goals.write_text(json.dumps({"tasks": list(_goals().values())}), encoding="utf-8")
    receipt = tmp_path / "lane" / "goal-redesign.json"
    first = write_goal_redesign(tmp_path, _tasks(), receipt)
    stamp = receipt.stat().st_mtime_ns
    second = write_goal_redesign(tmp_path, _tasks(), receipt)
    assert first == second
    assert receipt.stat().st_mtime_ns == stamp
    assert json.loads(receipt.read_text())["board_mutations"] == []
