"""Keep policy findings observable without granting callback replay authority."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.candidate_failure_diagnostics import (
    MAX_SUMMARY_BYTES,
    normalize_candidate_failure_diagnostics,
    summarize_candidate_failure,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
    DatabasePortalBridgeError,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DatabaseImplementationProviderDispatchError,
)
from test.api.test_agent_supervisor_database_portal_bridge import (
    _seed_interrupted_database_portal_attempt,
)


def _rejected_implementation() -> dict:
    # PCTDD-035's actual failure shape. Prose and paths are deliberately
    # excluded from the durable diagnostic vocabulary.
    return {
        "returncode": 78,
        "provider_dispatched": True,
        "attempt_consumed": True,
        "validation_result": {
            "attempted": False,
            "error": "proposal_validation_failed",
            "failure_review": {
                "finding_codes": [
                    "path_outside_scope",
                    "validation_channel_tampering_forbidden",
                    "invented-authority",
                ],
                "reason_codes": ["scope_expansion_denied", "proposal_gate_failed"],
                "guidance_markdown": "untrusted text containing secret-token",
                "out_of_scope_paths": ["untrusted/path.py"],
            },
        },
    }


EXPECTED = {
    "reason_codes": ["proposal_gate_failed", "scope_expansion_denied"],
    "finding_codes": ["path_outside_scope", "validation_channel_tampering_forbidden"],
}


def test_failure_summary_uses_only_closed_native_codes() -> None:
    summary = summarize_candidate_failure(_rejected_implementation())
    assert summary == EXPECTED
    assert len(json.dumps(summary).encode()) < MAX_SUMMARY_BYTES
    assert "secret-token" not in json.dumps(summary)
    assert "untrusted/path.py" not in json.dumps(summary)


@pytest.mark.parametrize("value", [None, [], "path_outside_scope", 78, True])
def test_nonobject_diagnostics_do_not_acquire_meaning(value) -> None:
    assert normalize_candidate_failure_diagnostics(value) == {}
    assert summarize_candidate_failure(value) == {}


def test_summary_does_not_invoke_candidate_mapping_hooks() -> None:
    class UntrustedDict(dict):
        def get(self, *args):
            raise AssertionError("candidate object hook executed")

    assert normalize_candidate_failure_diagnostics(UntrustedDict()) == {}
    assert summarize_candidate_failure(UntrustedDict()) == {}
    assert summarize_candidate_failure({"validation_result": UntrustedDict()}) == {}


def test_diagnostic_rejection_is_retained_without_provider_replay(tmp_path: Path) -> None:
    _repo, daemon, bridge, attempt, paths = _seed_interrupted_database_portal_attempt(
        tmp_path, seed_nested_state=False,
    )
    calls = []
    closed = []

    def portal_run():
        calls.append("provider")
        return {"implementation_result": _rejected_implementation()}

    bridge.portal_factory = lambda *_: SimpleNamespace(
        run_once=portal_run, close=lambda: closed.append(True),
    )
    before_task = daemon.task_source.get_task(attempt.task_cid).to_dict()
    try:
        with pytest.raises(DatabasePortalBridgeError) as caught:
            daemon.run_provider(attempt)
        assert caught.value.diagnostic_summary == EXPECTED
        journal = daemon._dispatch_journal_entry(
            attempt, dispatch_kind="provider", idempotency_key=f"provider:{attempt.attempt_id}",
        )
        assert journal["outcome"] == "raised"
        assert journal["body"] == {
            "exception_type": "DatabasePortalBridgeError",
            "candidate_failure_diagnostics": EXPECTED,
        }
        assert not paths.state.exists()
        assert not daemon.get_attempt(attempt.attempt_id).phase_committed("provider")
        assert daemon.provider_invocation_recorded(
            attempt.attempt_id, idempotency_key=f"provider:{attempt.attempt_id}",
        ) is None
        # The codes do not invent terminal proof for the missing nested state.
        # A subsequent pass must recover evidence, never dispatch again.
        with pytest.raises(DatabaseImplementationProviderDispatchError, match="outcome is unknown"):
            daemon.run_provider(attempt)
        assert calls == ["provider"]
        assert closed == [True]
        assert daemon.task_source.get_task(attempt.task_cid).to_dict() == before_task
        assert daemon._dispatch_journal_entry(
            attempt, dispatch_kind="provider", idempotency_key=f"provider:{attempt.attempt_id}",
        ) == journal
        assert daemon.claim_next() is None
    finally:
        daemon.close()


def test_generic_bridge_error_keeps_existing_journal_shape(tmp_path: Path) -> None:
    _repo, daemon, bridge, attempt, _paths = _seed_interrupted_database_portal_attempt(
        tmp_path, seed_nested_state=False,
    )
    bridge.portal_factory = lambda *_: SimpleNamespace(
        run_once=lambda: {"implementation_result": {"returncode": 78}},
    )
    try:
        with pytest.raises(DatabasePortalBridgeError):
            daemon.run_provider(attempt)
        journal = daemon._dispatch_journal_entry(
            attempt, dispatch_kind="provider", idempotency_key=f"provider:{attempt.attempt_id}",
        )
        assert journal["body"] == {"exception_type": "DatabasePortalBridgeError"}
    finally:
        daemon.close()
