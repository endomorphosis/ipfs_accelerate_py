from ipfs_accelerate_py.agent_supervisor.entrypoints.refill_controller import (
    CompletionAuthorityDecision, RefillController, RefillDisposition, RefillObservation,
    RefillPolicy, RefillTrigger, ResidualEvidence, ResidualGap, refill_triggers,
)


def _gap(scope="scope"):
    return ResidualGap("goal", "evidence", scope, ("goal", "root"), 0,
                       {"priority": "P0", "track": "x", "parallel_lane": "lane", "resource_class": "cpu"})


def _controller(gaps=(), completion=None, policy=None):
    appended = []
    def evaluate(_observation, *, force_final_scan):
        assert force_final_scan
        return ResidualEvidence("tree", tuple(gaps), completion or CompletionAuthorityDecision(False))
    def append(work, cas):
        appended.append((work, cas)); return True
    return RefillController(evaluate, append, policy=policy), appended


def test_trigger_matrix_is_complete_and_deterministic():
    observation = RefillObservation("plan", 1, open_goals=1, validation_rejected=True,
        actionable_drift=True, retry_exhausted_with_refinement=True, rollout_threshold_missed=True)
    assert refill_triggers(observation, RefillPolicy()) == tuple(RefillTrigger)


def test_healthy_non_drained_board_does_not_refill():
    controller, appended = _controller()
    result = controller.decide(RefillObservation("plan", 1, ready_tasks=2))
    assert result.disposition is RefillDisposition.NO_REFILL
    assert not appended


def test_refill_has_lineage_scheduler_metadata_and_cas():
    controller, appended = _controller((_gap(),))
    result = controller.decide(RefillObservation("plan", 4, open_goals=1))
    assert result.disposition is RefillDisposition.REFILLED
    assert result.appended_count == 1 and result.cas.expected_revision == 4
    assert appended[0][0][0].lineage_goal_cids == ("goal", "root")


def test_initial_work_and_depth_caps_are_enforced():
    deep = ResidualGap("g2", "e2", "s2", ("g2",), 1, _gap().scheduler_metadata)
    controller, appended = _controller((_gap("a"), _gap("b"), deep), policy=RefillPolicy(max_findings_per_scan=2, max_new_work_per_epoch=1, max_refinement_depth=1))
    result = controller.decide(RefillObservation("plan", 1))
    assert result.appended_count == 1 and len(appended[0][0]) == 1


def test_branch_only_and_stale_completion_reopen_convergence():
    controller, _ = _controller()
    assert controller.decide(RefillObservation("plan", 1, branch_only_completion=True)).disposition is RefillDisposition.REOPEN_CONVERGENCE
    controller, _ = _controller()
    assert controller.decide(RefillObservation("plan", 1, stale_evidence=True)).disposition is RefillDisposition.REOPEN_CONVERGENCE


def test_unchanged_residuals_trip_circuit_breaker_without_spinning():
    policy = RefillPolicy(max_unchanged_epochs=1)
    controller, _ = _controller((_gap(),), policy=policy)
    assert controller.decide(RefillObservation("plan", 1)).disposition is RefillDisposition.REFILLED
    # A new controller state normally sees this as a duplicate.  A repeated evaluator
    # result therefore terminates safely instead of allocating another task.
    assert controller.decide(RefillObservation("plan", 2)).disposition is RefillDisposition.BLOCKED


def test_healthy_drain_and_cooldown_do_not_generate_more_work():
    completed = CompletionAuthorityDecision(True, True, True, True)
    controller, appended = _controller((), completed)
    assert controller.decide(RefillObservation("plan", 1)).disposition is RefillDisposition.NO_REFILL
    controller, appended = _controller((_gap(),), policy=RefillPolicy(cooldown_epochs=2))
    assert controller.decide(RefillObservation("plan", 1)).disposition is RefillDisposition.REFILLED
    assert controller.decide(RefillObservation("plan", 2)).reason == "cooldown"
    assert len(appended) == 1
