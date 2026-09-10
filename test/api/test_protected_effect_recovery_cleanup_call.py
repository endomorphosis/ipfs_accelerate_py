"""Exercise recovery control flow with explicit admitted-boundary test doubles.

The strict cleanup double preserves the production keyword-only contract. These
tests qualify caller ordering/replay, not Docker or native authority admission.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py import llm_router
from ipfs_accelerate_py.agent_supervisor.control import provider_attempt_store
from ipfs_accelerate_py.agent_supervisor.runtime import grok_cli_runner as runner


def _recovery_fixture(monkeypatch, tmp_path: Path, *, status="exited", returncode=0):
    prompt = "exact recovery prompt"
    invocation = SimpleNamespace(
        task_id="task:recovery",
        attempt=2,
        task_revision_cid="revision:2",
        logical_attempt_id="logical:2",
        worktree_id="worktree:2",
        prompt_cid=runner._agent_prompt_cid(prompt),
        workspace_path=str(tmp_path),
        provider_attempt_store=str(tmp_path / "provider-attempts"),
        provider_attempt_store_identity="store:identity",
        route_id="route:2",
        control_plane="sealed-control-plane",
    )
    locator = {
        name: getattr(invocation, name)
        for name in (
            "task_id",
            "attempt",
            "task_revision_cid",
            "logical_attempt_id",
            "worktree_id",
            "prompt_cid",
            "workspace_path",
            "provider_attempt_store",
            "provider_attempt_store_identity",
        )
    }
    locator.update(
        schema=runner._PROTECTED_EFFECT_RECOVERY_LOCATOR_SCHEMA,
        board_namespace="test:protected",
    )
    locator["locator_id"] = runner._effect_receipt_identity(locator)
    context = SimpleNamespace(
        route=SimpleNamespace(invocation_binding=invocation, authorization=None),
        decision=SimpleNamespace(content_id="decision:2", verifier_status="accepted"),
        failure_receipt={"fixture": "failure"},
        quota_evidence={},
    )
    active = SimpleNamespace(
        state="effect_started",
        terminal=False,
        task_id=invocation.task_id,
        worktree_id=invocation.worktree_id,
        route_id=invocation.route_id,
        decision_id=context.decision.content_id,
        reservation_id="reservation:2",
        authorization_context="exact historical authorization",
        effect_launch_receipt={"effect_owner_pid": 123, "cleanup_id": "cleanup:2"},
        effect_started_at_ms=1,
        effect_adoption_receipt={
            "inspection_status": status,
            "container_returncode": returncode,
        },
        quarantine_receipt={},
        quarantine_terminalization_receipt={},
    )
    state = {
        "current": active,
        "adoptions": 0,
        "completions": 0,
        "cleanup_calls": [],
        "events": [],
    }

    class Store:
        def read(self, logical_id):
            assert logical_id == invocation.logical_attempt_id
            return state["current"]

        def adopt_effect(self, reservation):
            assert reservation is state["current"]
            state["adoptions"] += 1
            state["events"].append("adopt")
            return SimpleNamespace(
                adoption_authorized=True,
                reservation=reservation,
                completion_capability="completion:2",
            )

        def complete(
            self,
            reservation,
            *,
            returncode,
            outcome,
            completion_capability,
            terminal_cleanup_evidence,
        ):
            assert reservation is state["current"]
            assert completion_capability == "completion:2"
            assert terminal_cleanup_evidence == {"binding_record_id": "binding:2"}
            state["completions"] += 1
            state["events"].append("complete")
            terminal = SimpleNamespace(**vars(reservation))
            terminal.state = "terminal"
            terminal.terminal = True
            terminal.terminal_returncode = returncode
            terminal.terminal_outcome = outcome
            state["current"] = terminal
            return terminal

    store = Store()

    def open_store(path, *, expected_directory_identity):
        assert path == invocation.provider_attempt_store
        assert expected_directory_identity == invocation.provider_attempt_store_identity
        return store

    def release(launch_receipt, *, terminal_observer, terminal_reservation):
        assert terminal_observer is store
        assert terminal_reservation is state["current"]
        assert terminal_reservation.terminal is True
        assert launch_receipt is terminal_reservation.effect_launch_receipt
        state["cleanup_calls"].append((terminal_observer, terminal_reservation))
        state["events"].append("cleanup")
        if state.get("cleanup_unknown"):
            raise ValueError("exact cleanup outcome remains unknown")

    def parse_context(value, **kwargs):
        assert value == active.authorization_context
        assert kwargs["expected_signer_parent_pid"] == 123
        return context

    monkeypatch.setattr(provider_attempt_store, "DurableProviderAttemptCAS", open_store)
    monkeypatch.setattr(
        llm_router,
        "parse_agent_implementation_effect_authorization_context",
        parse_context,
    )
    monkeypatch.setattr(
        llm_router,
        "verify_agent_implementation_sealed_control_plane",
        lambda control, fd: (
            "/proc/self/fd/99"
            if (control == invocation.control_plane and fd == 99)
            else ""
        ),
    )

    def build_outcome(**kwargs):
        # The real builder canonically represents absent quarantine evidence.
        for field in (
            "effect_quarantine_receipt",
            "effect_quarantine_terminalization_receipt",
        ):
            if kwargs[field] is None:
                kwargs[field] = {}
        return kwargs

    monkeypatch.setattr(
        llm_router, "build_agent_implementation_route_outcome", build_outcome
    )
    monkeypatch.setattr(
        llm_router,
        "valid_agent_implementation_route_outcome",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        llm_router,
        "render_agent_implementation_route_outcome",
        lambda value: "recorded-route-outcome:" + str(value["fallback_returncode"]),
    )
    monkeypatch.setattr(
        runner, "_recorded_codex_terminal_capacity_evidence", lambda _r: {}
    )
    monkeypatch.setattr(
        runner,
        "_recorded_codex_terminal_cleanup_evidence",
        lambda _r: {"binding_record_id": "binding:2"},
    )
    monkeypatch.setattr(runner, "_release_recorded_codex_effect_cleanup", release)
    monkeypatch.setattr(runner.sys, "argv", ["/proc/self/fd/99"])

    def run():
        monkeypatch.setattr(runner.sys, "stdin", io.StringIO(prompt))
        return runner._run_protected_effect_recovery(
            raw_locator=json.dumps(locator), workspace=tmp_path
        )

    return state, run


@pytest.mark.parametrize("returncode", [0, 78])
def test_successful_scoped_cleanup_returns_recorded_result_once(
    monkeypatch,
    tmp_path,
    capsys,
    returncode,
):
    state, run = _recovery_fixture(monkeypatch, tmp_path, returncode=returncode)
    assert run() == returncode
    assert state["events"] == ["adopt", "complete", "cleanup"]
    assert len(state["cleanup_calls"]) == 1
    assert "protected effect recovery denied" not in capsys.readouterr().err


def test_terminal_replay_uses_same_scoped_cleanup_without_new_completion(
    monkeypatch,
    tmp_path,
):
    state, run = _recovery_fixture(monkeypatch, tmp_path)
    assert run() == 0
    original_terminal = state["current"]
    assert run() == 0
    assert state["adoptions"] == state["completions"] == 1
    assert len(state["cleanup_calls"]) == 2
    assert all(call[1] is original_terminal for call in state["cleanup_calls"])


def test_cleanup_unknown_retains_terminal_and_replays_only_exact_cleanup(
    monkeypatch,
    tmp_path,
    capsys,
):
    state, run = _recovery_fixture(monkeypatch, tmp_path)
    state["cleanup_unknown"] = True
    assert run() == 125
    original_terminal = state["current"]
    assert original_terminal.terminal is True
    assert state["adoptions"] == state["completions"] == 1
    assert "exact cleanup outcome remains unknown" in capsys.readouterr().err
    state["cleanup_unknown"] = False
    assert run() == 0
    assert state["current"] is original_terminal
    assert state["adoptions"] == state["completions"] == 1
    assert len(state["cleanup_calls"]) == 2


def test_unknown_inspection_never_completes_or_cleans(monkeypatch, tmp_path):
    state, run = _recovery_fixture(monkeypatch, tmp_path, status="unknown")
    original = state["current"]
    assert run() == 125
    assert state["current"] is original
    assert state["completions"] == 0
    assert state["cleanup_calls"] == []


def test_invalid_terminal_replay_never_cleans(monkeypatch, tmp_path):
    state, run = _recovery_fixture(monkeypatch, tmp_path)
    state["cleanup_unknown"] = True
    assert run() == 125
    state["current"].terminal_outcome["reservation_id"] = "foreign:reservation"
    state["cleanup_unknown"] = False
    assert run() == 125
    assert state["adoptions"] == state["completions"] == 1
    assert len(state["cleanup_calls"]) == 1
