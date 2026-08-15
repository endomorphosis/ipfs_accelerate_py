"""IPS-034: cancellation, timeout, and process-tree fencing."""

from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.process_control import (
    EVIDENCE_SUBSET,
    CancellationToken,
    ControlOutcome,
    ProcessTerminationResult,
    ProofProcessController,
    TerminationStage,
)


def test_evidence_subset() -> None:
    assert EVIDENCE_SUBSET == "ips/process-fencing@1"


def test_timeout_and_cancel_never_satisfy_required_unit() -> None:
    alive = {"pid": True}
    signals: list[str] = []

    def terminate(pid: int) -> None:
        del pid
        signals.append("term")
        alive["pid"] = False

    def kill(pid: int) -> None:
        del pid
        signals.append("kill")
        alive["pid"] = False

    controller = ProofProcessController(
        terminate=terminate,
        kill=kill,
        poll_alive=lambda pid: alive["pid"],
    )
    controller.attach(4242)
    result = controller.fence(timeout=True)
    assert result.outcome is ControlOutcome.TIMEOUT
    assert result.admitted_proof is False
    assert result.satisfies_required_unit is False
    assert result.live_descendants == 0
    assert result.stage is TerminationStage.REAPED
    assert "term" in signals


def test_late_output_is_quarantined_across_generation() -> None:
    token = CancellationToken()
    controller = ProofProcessController(token)
    controller.attach(7)
    assert controller.observe_output(b"ok") is True
    token.cancel("operator")
    assert controller.observe_output(b"late") is False
    result = controller.fence()
    assert result.outcome is ControlOutcome.CANCELLED
    assert result.late_output_quarantined is True
    assert result.admitted_proof is False
    assert result.generation == token.generation
    assert result.satisfies_required_unit is False


def test_escalates_to_kill_when_still_alive() -> None:
    calls: list[str] = []

    def terminate(pid: int) -> None:
        del pid
        calls.append("term")

    def kill(pid: int) -> None:
        del pid
        calls.append("kill")

    controller = ProofProcessController(
        terminate=terminate,
        kill=kill,
        poll_alive=lambda pid: True,
        clock=lambda: 0.0,
    )
    controller.attach(9)
    result = controller.fence(timeout=True, grace_seconds=0.0)
    assert calls == ["term", "kill"]
    assert result.stage is TerminationStage.KILL
    assert result.live_descendants == 1
    assert result.outcome is ControlOutcome.TIMEOUT
    assert ProcessTerminationResult(
        outcome=ControlOutcome.UNKNOWN,
        stage=TerminationStage.NONE,
        live_descendants=0,
        generation=0,
        admitted_proof=False,
        late_output_quarantined=False,
    ).satisfies_required_unit is False
