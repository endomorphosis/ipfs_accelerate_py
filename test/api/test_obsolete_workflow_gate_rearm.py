from __future__ import annotations

import json

from ipfs_accelerate_py.agent_supervisor.todo_daemon.obsolete_workflow_gate_rearm import (
    close_open_receipt,
    events_prove_obsolete_workflow_weakening,
    preserved_workflow_revisions,
    rearm,
    stale_unknown_callback_yields_to_closed_rearm,
)
from ipfs_accelerate_py.agent_supervisor.validation.proposal_validation import (
    workflow_gate_checker_revision,
)

ATTEMPT = "attempt:" + "ab" * 16
TASK = "sha256:" + "cd" * 32


class _Rows:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _Connection:
    def __init__(self, rows=None):
        self.rows = rows or []

    def execute(self, _sql, _params=None):
        return _Rows(self.rows)


class _Changed:
    def __init__(self, changed=True):
        self.changed = changed

    def to_dict(self):
        return {
            "task": {"task_cid": TASK, "status": "retrying", "revision": 10},
            "changed": True,
            "previous_status": "blocked",
            "revision": 10,
            "receipt_cid": "sha256:" + "ab" * 32,
        }


class _Coordinator:
    def __init__(self):
        self.calls = []

    def rearm_failed_task(self, *, failure_receipt, control_task_observation, now_ms=None):
        self.calls.append((dict(failure_receipt), dict(control_task_observation), now_ms))
        return {"ready": True}


class _Source:
    def __init__(self):
        self.receipts = []
        self.closed = []
        self.refreshed = None

    def rearm_blocked_task(self, task_cid, *, receipt):
        self.receipts.append((task_cid, receipt))
        return _Changed(True)

    def replace_retrying_rearm_receipt(self, task_cid, *, expected_revision, expected_operation, receipt):
        self.closed.append((task_cid, expected_revision, expected_operation, dict(receipt)))
        return True

    def get_task(self, task_cid):
        return self.refreshed


class _Task:
    def __init__(self, receipt, *, status="blocked", revision=9):
        self.status = status
        self.revision = revision
        self.task_cid = TASK
        self.task_alias = "WWL-111"
        self.body = {"completion_receipt": receipt}


class _Owner:
    def __init__(self, rows=None):
        self.task_source = _Source()
        self.coordinator = _Coordinator()
        self.connection = _Connection(rows)
        self.events = []
        self.running = []
        self.max_task_attempts = 3
        self._now_ms = lambda: 10

    def _require_connection(self):
        return self.connection

    def list_running_attempts(self):
        return self.running

    def _record_event(self, event_type, *, attempt_id, task_cid, body):
        self.events.append((event_type, attempt_id, task_cid, body))


def _receipt(**overrides):
    receipt = {
        "schema": "ipfs_accelerate_py/agent-supervisor/task-claim-failure-settlement@1",
        "operation": "database_task_claim_failure",
        "failure_kind": "terminal_portal_bridge_error",
        "automatic_retry_admitted": False,
        "provider_invocation_count": 1,
        "effect_claim_count": 0,
        "attempt_number": 3,
        "attempt_id": ATTEMPT,
        "settlement_id": "baguqeera" + "ab" * 16,
        "control_expected_revision": 8,
    }
    receipt.update(overrides)
    return receipt


def _pairs(after_command="run: npx tsc --noEmit --pretty false"):
    before = "jobs:\n  build:\n    steps:\n      - run: npx tsc --noEmit\n"
    after = (
        "jobs:\n  build:\n    steps:\n      - name: Typecheck\n        "
        + after_command
        + "\n"
    )
    return [(".github/workflows/deploy.yml", before, after)]


def test_preserved_command_extension_rearms_once() -> None:
    owner = _Owner()
    result = rearm(owner, _Task(_receipt()), load_pairs=lambda *_args: _pairs())

    assert result is not None
    assert result["rearmed"] is True
    assert result["completion_authority"] is False
    assert result["paid_attempt_budget_reset"] is False
    assert owner.task_source.receipts[0] == (
        TASK,
        {
            "operation": "operator_control_plane_repair",
            "settlement_id": "baguqeera" + "ab" * 16,
        },
    )
    assert owner.coordinator.calls[0][0]["operation"] == "database_task_claim_failure"
    assert owner.coordinator.calls[0][2] == 10
    assert owner.events[0][0] == "obsolete_workflow_gate_rearmed"
    recorded = json.dumps(owner.events[0][3])
    owner.connection.rows = [(recorded,)]
    assert rearm(owner, _Task(_receipt()), load_pairs=lambda *_args: _pairs()) is None
    assert len(owner.task_source.receipts) == 1


def test_real_weakening_and_live_work_stay_blocked() -> None:
    owner = _Owner()
    dropped = rearm(
        owner,
        _Task(_receipt()),
        load_pairs=lambda *_args: _pairs(after_command="run: echo skipped"),
    )
    owner.running = [object()]
    live = rearm(owner, _Task(_receipt()), load_pairs=lambda *_args: _pairs())
    owner.running = []
    other = rearm(
        owner,
        _Task(_receipt(failure_kind="declared_validation_failed")),
        load_pairs=lambda *_args: _pairs(),
    )
    early = rearm(
        owner,
        _Task(_receipt(attempt_number=1)),
        load_pairs=lambda *_args: _pairs(),
    )

    assert dropped is None
    assert live is None
    assert other is None
    assert early is None
    assert owner.task_source.receipts == []


def test_open_retrying_receipt_closes_at_the_same_revision() -> None:
    owner = _Owner()
    failure = _receipt()
    closed = {
        "operation": "operator_control_plane_repair",
        "settlement_id": failure["settlement_id"],
    }
    owner.task_source.refreshed = _Task(closed, status="retrying", revision=10)
    updated = close_open_receipt(
        owner,
        _Task(
            {
                "operation": "obsolete_workflow_gate_rearm",
                "attempt_id": ATTEMPT,
                "reason": "proposal_gate_failed",
                "completion_authority": False,
                "paid_attempt_budget_reset": False,
            },
            status="retrying",
            revision=10,
        ),
        failure,
    )

    assert updated is owner.task_source.refreshed
    assert owner.task_source.closed == [
        (TASK, 10, "obsolete_workflow_gate_rearm", closed)
    ]
    unrelated = _Task(
        {"operation": "database_task_claim_failure"},
        status="retrying",
        revision=10,
    )
    assert close_open_receipt(owner, unrelated, failure) is unrelated
    moved = _Task(
        {"operation": "obsolete_workflow_gate_rearm"},
        status="retrying",
        revision=11,
    )
    assert close_open_receipt(owner, moved, failure) is moved
    assert len(owner.task_source.closed) == 1


def test_closed_rearm_yields_only_failed_attempts_without_effects() -> None:
    settlement = "baguqeera" + "ab" * 16
    closed = {"operation": "operator_control_plane_repair", "settlement_id": settlement}
    common = dict(
        task_status="retrying",
        control_receipt=closed,
        attempt_status="failed",
        attempt_task_cid=TASK,
        task_cid=TASK,
        attempt_owner_session_id="worldwide-legal-search-v1",
        owner_session_id="worldwide-legal-search-v1",
        effect_claim_present=False,
        claim_state="released",
        rearm_settlement_id=settlement,
    )
    assert stale_unknown_callback_yields_to_closed_rearm(**common)
    assert not stale_unknown_callback_yields_to_closed_rearm(
        **{**common, "rearm_settlement_id": ""}
    )
    assert not stale_unknown_callback_yields_to_closed_rearm(
        **{**common, "effect_claim_present": True}
    )
    assert not stale_unknown_callback_yields_to_closed_rearm(
        **{**common, "attempt_status": "running"}
    )
    assert not stale_unknown_callback_yields_to_closed_rearm(
        **{**common, "claim_state": "accepted"}
    )
    assert not stale_unknown_callback_yields_to_closed_rearm(
        **{**common, "control_receipt": {"operation": "database_task_claim_failure", "settlement_id": settlement}}
    )


def test_revision_fingerprint_uses_the_loaded_checker() -> None:
    assert workflow_gate_checker_revision().startswith("sha256:")
    baseline, commit = preserved_workflow_revisions(
        [
            {
                "type": "implementation_finished",
                "baseline_ref": "a" * 40,
                "implementation_commit": "b" * 40,
            }
        ]
    )
    assert baseline == "a" * 40
    assert commit == "b" * 40
    assert preserved_workflow_revisions(
        [{"type": "implementation_finished", "baseline_ref": "HEAD", "implementation_commit": "b" * 40}]
    ) is None
    weakening = {
        "type": "implementation_proposal_rejected",
        "accepted": False,
        "reason_codes": ["validation_weakening_forbidden"],
        "changed_paths": [".github/workflows/deploy.yml"],
    }
    review = {
        "type": "implementation_failure_reviewed",
        "decision": "reject",
        "finding_codes": ["validation_weakening_forbidden"],
    }
    finished = {"type": "implementation_finished", "returncode": 78}
    assert events_prove_obsolete_workflow_weakening([weakening, review, finished])
    assert not events_prove_obsolete_workflow_weakening(
        [weakening, {**review, "finding_codes": ["validation_command_failed"]}, finished]
    )
