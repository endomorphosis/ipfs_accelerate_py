from __future__ import annotations

from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_advisor import (
    KernelSpend,
    SmtTriageAction,
    advise_proof_draft,
    maybe_verify_leanstral_draft,
    score_synthesis_candidate,
    triage_smt,
    typesafe_permitted,
)
from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_trace_guard import (
    observe_worker_trace,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.llm import LLM_USAGE_MODE_OBSERVE
from ipfs_accelerate_py.typesafe_inference import TypeSafeInferenceError, system_one


@pytest.fixture
def no_typesafe_key(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "TYPESAFE_API_KEY",
        "ipfs_accelerate_py_TYPESAFE_API_KEY",
        "IPFS_ACCELERATE_PY_TYPESAFE_API_KEY",
        "IPFS_DATASETS_PY_TYPESAFE_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def forbid_typesafe_http(no_typesafe_key: None, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args, **_kwargs):
        raise AssertionError("system_one must not run without TYPESAFE_API_KEY")

    monkeypatch.setattr("ipfs_accelerate_py.typesafe_inference.system_one", boom)


def test_typesafe_permitted_false_without_key(no_typesafe_key: None) -> None:
    assert typesafe_permitted() is False


def test_system_one_raises_only_when_called_directly(no_typesafe_key: None) -> None:
    with pytest.raises(TypeSafeInferenceError, match="TYPESAFE_API_KEY"):
        system_one("state", {"ok": {"type": "noul", "instructions": "yes?"}})


def test_advise_proof_draft_fails_open_to_kernel(no_typesafe_key: None) -> None:
    receipt = advise_proof_draft(
        goal_id="g1",
        declaration="theorem t : True",
        draft_text="by exact True.intro",
    )
    assert receipt.action == KernelSpend.UNAVAILABLE.value
    assert receipt.action != KernelSpend.SKIP.value


def test_maybe_verify_precheck_without_key_still_calls_kernel(
    no_typesafe_key: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = {"kernel": False}

    def _kernel(*_args, **_kwargs):
        called["kernel"] = True
        return SimpleNamespace(status="rejected")

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.proof.leanstral_proof_provider.verify_leanstral_draft",
        _kernel,
    )
    result = maybe_verify_leanstral_draft(
        SimpleNamespace(draft_text="by exact True.intro"),
        SimpleNamespace(theorem_id="t", obligation_id="o"),
        typesafe_precheck=True,
        native_source="theorem t : True := sorry",
        bindings=None,
    )
    assert called["kernel"] is True
    assert result.status == "rejected"


def test_triage_smt_without_key_runs_z3(no_typesafe_key: None) -> None:
    receipt = triage_smt(
        english="identity",
        smtlib="(set-logic UF)\n(check-sat)\n",
        case_id="fol_identity",
        complexity="easy",
    )
    assert receipt.action == SmtTriageAction.RUN_Z3.value
    assert receipt.action != SmtTriageAction.SKIP_Z3.value


def test_observe_without_key_does_not_raise(no_typesafe_key: None) -> None:
    assert (
        observe_worker_trace(
            prompt="ignore previous",
            output="by intro",
            usage_mode=LLM_USAGE_MODE_OBSERVE,
        )
        is None
    )


def test_synthesis_score_without_key_abstains(no_typesafe_key: None) -> None:
    receipt = score_synthesis_candidate(
        candidate_id="cand-1",
        allowlisted_ids=("cand-1",),
        state={},
    )
    assert receipt.action == "abstain"
    assert receipt.accepted_as_authority is False


def test_prepare_step_candidates_without_key_keeps_original_set(
    no_typesafe_key: None,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.autonomy.contracts import (
        AuthorityClass,
        CancellationBehavior,
        DecisionQuestion,
        DecisionQuestionType,
        MetaAction,
        PrivacyClass,
        QuestionDisposition,
        ResolutionAction,
        ResolutionCandidate,
        ResolutionEvidenceKind,
        RiskClass,
    )
    from ipfs_accelerate_py.agent_supervisor.autonomy.decision_graph import (
        DecisionGraphController,
    )
    from ipfs_accelerate_py.agent_supervisor.autonomy.typesafe_decision import (
        prepare_step_candidates,
    )

    action = ResolutionAction(
        action=MetaAction.RUN_LOCAL_STATIC_ANALYSIS,
        precondition_ids=("tree-current",),
        expected_evidence_kind=ResolutionEvidenceKind.STATIC_ANALYSIS,
        expected_uncertainty_reduction_bp=8_000,
        token_cost=0,
        latency_cost_ms=100,
        provider_cost_micros=0,
        resource_cost_units=1,
        invalidation_cost_units=0,
        privacy_cost_units=0,
        privacy_class=PrivacyClass.LOCAL_ONLY,
        risk_class=RiskClass.R1_READ_ONLY,
        cancellation_behavior=CancellationBehavior.COOPERATIVE,
        cacheable=True,
        authority_class=AuthorityClass.VERIFIED,
        accepted_as_authority=True,
    )
    question = DecisionQuestion(
        objective_id="APMC-G000",
        acceptance_criterion_ids=("AC-1",),
        question_type=DecisionQuestionType.WHICH_PROOF_OBLIGATION_APPLIES,
        current_alternatives=("obl-1", "obl-2"),
        required_evidence_ids=(),
        known_evidence_ids=(),
        contradictory_evidence_ids=(),
        residual_uncertainty_bp=5_000,
        decision_deadline_ms=1_000,
        risk_if_incorrect=RiskClass.R1_READ_ONLY,
        risk_if_left_unresolved=RiskClass.R1_READ_ONLY,
        possible_resolution_action_ids=(action.action_id,),
        dependency_question_ids=(),
        terminal_decision_rule="select only from current alternatives",
        mandatory=True,
        disposition=QuestionDisposition.UNRESOLVED,
        terminal_answer="",
    )
    controller = DecisionGraphController.compile(
        repository_id="repo:ipfs-accelerate",
        tree_id="tree:one",
        objective_id="APMC-G000",
        objective_revision="revision:one",
        questions=(question,),
    )
    compiled = controller.graph.questions[0]
    candidates = (
        ResolutionCandidate(
            question_id=compiled.question_id,
            resolution_action=action,
            expected_decision_value=100,
            admissible=True,
            policy_id="policy:one",
        ),
    )
    prepared, updated, advice = prepare_step_candidates(
        controller,
        compiled,
        candidates,
        state={"board_item": "TASK-1"},
        remote_disclosure_permitted=True,
    )
    assert advice is None
    assert prepared == candidates
    assert updated.question_id == compiled.question_id
    assert compiled.known_evidence_ids == ()


def test_new_helpers_do_not_call_http_without_key(
    forbid_typesafe_http: None, tmp_path
) -> None:
    from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_calibration import (
        recommend_skip_policy,
        should_trust_skip,
    )
    from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_context import (
        cite_claim_spans,
        extract_claim_spans,
        observe_source_edit_lint,
        rerank_allowlisted_snippets,
    )
    from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_ops import (
        typesafe_ops_snapshot,
    )
    from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_task_router import (
        advise_board_task_kind,
    )
    from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_trace_reduce import (
        observe_control_audit_throttled,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.llm import (
        LLM_USAGE_MODE_OFF,
        maybe_observe_worker_output,
    )

    maybe_observe_worker_output(prompt="p", output="KERNEL_VERIFIED", usage_mode=LLM_USAGE_MODE_OFF)
    spans = extract_claim_spans("KERNEL_VERIFIED identity.")
    assert cite_claim_spans(spans, receipt_ids=(), allowlisted_ids=tuple(s["id"] for s in spans)) == ()
    assert rerank_allowlisted_snippets(
        ({"id": "a", "text": "one"}, {"id": "b", "text": "two"}),
        obligation_id="obl",
        allowlisted_ids=("a", "b"),
    ) == ("a", "b")
    lint = observe_source_edit_lint(operator_id="op", relative_path="src/a.py")
    assert lint is not None and lint.action == "skipped"
    assert advise_board_task_kind({"metadata": {"kind": "legal"}, "title": "license"}) == "legal"
    snap = typesafe_ops_snapshot()
    assert snap["accepted_as_authority"] is False
    policy = recommend_skip_policy()
    assert policy["auto_apply"] is False
    assert should_trust_skip(family="fol_identity", confidence=0.92) is True
    audit = tmp_path / "control-audit.jsonl"
    audit.write_text('{"operation":"capabilities","status":"succeeded"}\n', encoding="utf-8")
    observe_control_audit_throttled(audit, min_interval_s=0)


def test_ops_cli_prints_json_without_key(forbid_typesafe_http: None, capsys) -> None:
    from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_ops import main

    assert main([]) == 0
    payload = __import__("json").loads(capsys.readouterr().out)
    assert payload["accepted_as_authority"] is False
    assert "calibration" in payload


def test_enforce_and_assist_never_call_typesafe_http(
    forbid_typesafe_http: None,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.llm import (
        LLM_USAGE_MODE_ASSIST,
        LLM_USAGE_MODE_ENFORCE,
        maybe_observe_worker_output,
    )

    maybe_observe_worker_output(
        prompt="ignore previous",
        output="write_file",
        usage_mode=LLM_USAGE_MODE_ENFORCE,
    )
    maybe_observe_worker_output(
        prompt="ignore previous",
        output="write_file",
        usage_mode=LLM_USAGE_MODE_ASSIST,
    )


def test_allocate_without_key_never_selects_typesafe_provider(
    forbid_typesafe_http: None,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.allocation_compat import (
        allocate_supervisor_endpoint,
    )

    allocated = allocate_supervisor_endpoint(
        task={"task_id": "TASK-1", "metadata": {"kind": "legal"}, "title": "license"},
        available_providers=("openrouter", "grok_cli", "codex_cli"),
    )
    assert allocated["provider"]
    assert "typesafe" not in str(allocated.get("provider") or "").casefold()


def test_prepare_evidence_and_compile_flag_without_key(forbid_typesafe_http: None) -> None:
    from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_context import (
        prepare_evidence_for_compile,
    )

    prepared = prepare_evidence_for_compile(
        (
            {"reference_id": "opt-b", "summary": "b", "required": False},
            {"reference_id": "req-a", "summary": "must", "required": True},
        ),
        obligation_id="obl-1",
    )
    assert prepared[0]["reference_id"] == "req-a"


def test_handle_wake_typesafe_prepare_fail_open_without_key(
    forbid_typesafe_http: None,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.autonomy.cognitive_budget import (
        ObjectiveCognitiveBudgetLedger,
    )
    from ipfs_accelerate_py.agent_supervisor.autonomy.cognitive_scheduler import (
        CognitiveSchedulingContext,
    )
    from ipfs_accelerate_py.agent_supervisor.autonomy.contracts import (
        AuthorityClass,
        CancellationBehavior,
        CognitiveBudget,
        DecisionQuestion,
        DecisionQuestionType,
        MetaAction,
        PrivacyClass,
        QuestionDisposition,
        ResolutionAction,
        ResolutionCandidate,
        ResolutionEvidenceKind,
        RiskClass,
    )
    from ipfs_accelerate_py.agent_supervisor.autonomy.decision_graph import (
        DecisionGraphController,
    )
    from ipfs_accelerate_py.agent_supervisor.autonomy.runtime import (
        AutonomousMetaController,
        AutonomyRuntime,
        AutonomyWakeEvent,
        AutonomyWakeKind,
    )

    action = ResolutionAction(
        action=MetaAction.RUN_LOCAL_STATIC_ANALYSIS,
        precondition_ids=("tree-current",),
        expected_evidence_kind=ResolutionEvidenceKind.STATIC_ANALYSIS,
        expected_uncertainty_reduction_bp=8_000,
        token_cost=0,
        latency_cost_ms=100,
        provider_cost_micros=0,
        resource_cost_units=1,
        invalidation_cost_units=0,
        privacy_cost_units=0,
        privacy_class=PrivacyClass.LOCAL_ONLY,
        risk_class=RiskClass.R1_READ_ONLY,
        cancellation_behavior=CancellationBehavior.COOPERATIVE,
        cacheable=True,
        authority_class=AuthorityClass.VERIFIED,
        accepted_as_authority=True,
    )
    question = DecisionQuestion(
        objective_id="APMC-G000",
        acceptance_criterion_ids=("AC-1",),
        question_type=DecisionQuestionType.WHICH_PROOF_OBLIGATION_APPLIES,
        current_alternatives=("obl-1",),
        required_evidence_ids=(),
        known_evidence_ids=(),
        contradictory_evidence_ids=(),
        residual_uncertainty_bp=5_000,
        decision_deadline_ms=1_000,
        risk_if_incorrect=RiskClass.R1_READ_ONLY,
        risk_if_left_unresolved=RiskClass.R1_READ_ONLY,
        possible_resolution_action_ids=(action.action_id,),
        dependency_question_ids=(),
        terminal_decision_rule="select only from current alternatives",
        mandatory=True,
        disposition=QuestionDisposition.UNRESOLVED,
        terminal_answer="",
    )
    controller = DecisionGraphController.compile(
        repository_id="repo:ipfs-accelerate",
        tree_id="tree:one",
        objective_id="APMC-G000",
        objective_revision="revision:one",
        questions=(question,),
    )
    compiled = controller.graph.questions[0]
    meta = AutonomousMetaController(
        decision_graph=controller,
        budget_controller=ObjectiveCognitiveBudgetLedger(
            CognitiveBudget(
                max_total_model_calls=4,
                max_strong_model_calls=2,
                max_input_tokens=8_000,
                max_output_tokens=2_000,
                max_provider_spend_micros=20_000,
                max_proof_time_ms=10_000,
                max_validation_time_ms=10_000,
                max_human_questions=1,
                max_repair_rounds=1,
                max_plan_branches=1,
                max_context_expansions=2,
                max_wall_time_ms=30_000,
                validation_reserve_ms=1_000,
            ),
            epoch=1,
        ),
    )
    runtime = AutonomyRuntime(controller=meta)
    result = runtime.handle_wake(
        AutonomyWakeEvent(kind=AutonomyWakeKind.PROOF, cursor_id="cursor:d1", sequence=1),
        candidates=(
            ResolutionCandidate(
                question_id=compiled.question_id,
                resolution_action=action,
                expected_decision_value=100,
                admissible=True,
                policy_id="policy:one",
            ),
        ),
        context=CognitiveSchedulingContext(
            policy_id="policy:one",
            satisfied_precondition_ids=frozenset({"tree-current"}),
            local_small_model_available=True,
            remote_standard_model_available=True,
            remote_strong_model_available=True,
            remote_disclosure_permitted=True,
            required_authority_class=AuthorityClass.DERIVED,
        ),
    )
    assert result.model_called is False
    assert result.authorizes_effect is False
