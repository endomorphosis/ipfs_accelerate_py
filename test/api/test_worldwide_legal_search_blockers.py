"""WWL-104: classify blockers and continue independent ready work.

Eight kinds stay distinct. Ready work that does not depend on a blocker stays
in the continuation set. A blocked task is rechecked only by an admitted probe
or a changed prerequisite.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_blockers import (
    ADMITTED_RECHECK_REASONS,
    BLOCKER_KINDS,
    EXTERNAL_KINDS,
    HOLD_MARKERS,
    SCHEMA,
    BlockerBoard,
    BlockerContractError,
    canonical_blocker_kind,
    reproducer_receipt,
)
from ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_progress import (
    DEFAULT_LIMITS,
    ProgressLedger,
    digest,
)
from ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_recovery import (
    HOLD_MARKERS as RECOVERY_HOLD_MARKERS,
)

RUN = "wwl-run-104"
FREE = "WWL-900"
CHILD = "WWL-910"
SUBJECT = "WWL-800"
RUNNING = "WWL-700"
TOKEN = "sha256:" + ("a" * 64)
TOKEN_NEXT = "sha256:" + ("b" * 64)


def board_of(*tasks, at: float = 100, board: BlockerBoard | None = None, **extra):
    owner = board or BlockerBoard(RUN)
    payload = {"observed_at": at, "tasks": list(tasks)}
    payload.update(extra)
    return owner, owner.classify(payload)


def row(decision, task_id):
    found = [item for item in decision["tasks"] if item["task_id"] == task_id]
    assert len(found) == 1
    return found[0]


def free_task():
    return {"task_id": FREE, "status": "ready", "depends_on": []}


def child_of(task_id):
    return {"task_id": CHILD, "status": "ready", "depends_on": [task_id]}


def reproducer(case: str = "router"):
    body = {"failure_key": "router_crash", "case": case}
    return body, reproducer_receipt(body)


def subject_for(kind: str):
    """One blocked subject, its waiter, and one independent ready task."""
    task = {"task_id": SUBJECT, "status": "blocked", "depends_on": [], "blocked_kind": kind}
    leading = []
    if kind == "credentials":
        task.update(credential_ref="hf-read", credential_state="missing")
    elif kind == "permission":
        task.update(permission_ref="hf-write", permission_state="revoked")
    elif kind == "gated-input":
        task.update(gate_ref="corpus-nl", gate_state="closed")
    elif kind == "source-outage":
        task.update(source_ref="hf-nl", source_state="outage", source_status=503)
    elif kind == "hold":
        task.update(hold=True, hold_markers=["HOLD"])
    elif kind == "exhausted-budget":
        task.update(budgets={"repair": {"spent": DEFAULT_LIMITS["repair"], "limit": DEFAULT_LIMITS["repair"]}})
    elif kind == "software-defect":
        body, receipt = reproducer(kind)
        task.update(failure_key="router_crash", reproducer=body, reproducer_receipt=receipt)
    elif kind == "dependency-running":
        task = {"task_id": SUBJECT, "status": "ready", "depends_on": [RUNNING], "blocked_kind": "dependency_running"}
        leading = [{"task_id": RUNNING, "status": "running", "depends_on": [], "progress_token": TOKEN}]
    else:
        raise AssertionError(kind)
    return [*leading, task, child_of(SUBJECT), free_task()]


def test_constants_match_the_campaign_contract():
    assert BLOCKER_KINDS == (
        "dependency-running",
        "software-defect",
        "source-outage",
        "gated-input",
        "credentials",
        "permission",
        "hold",
        "exhausted-budget",
    )
    assert HOLD_MARKERS == RECOVERY_HOLD_MARKERS == ("HOLD", "STOP", "OPERATOR_STOP")
    assert ADMITTED_RECHECK_REASONS == ("admitted_probe", "prerequisite_changed")
    assert canonical_blocker_kind("source_unavailable") == "source-outage"
    assert canonical_blocker_kind("rights") == "permission"
    assert canonical_blocker_kind("budget") == "exhausted-budget"
    with pytest.raises(BlockerContractError, match="not recognized"):
        canonical_blocker_kind("resource_quota")
    owner, decision = board_of(at=1)
    assert decision["schema"] == SCHEMA
    assert decision["continue_task_ids"] == []
    assert decision["execution_completed"] is False
    assert decision["success"] is False
    assert decision["completion_authority"] is False
    assert decision["repair_admitted"] is False
    assert owner.completion_authority is False


@pytest.mark.parametrize("kind", BLOCKER_KINDS)
def test_each_kind_stays_distinct_and_ready_work_continues(kind):
    _owner, decision = board_of(*subject_for(kind))
    subject = row(decision, SUBJECT)
    child = row(decision, CHILD)
    assert subject["classification"] == kind
    assert subject["kind"] == kind
    assert subject["matched_kinds"] == [kind]
    assert subject["continues"] is False
    assert subject["success"] is False
    assert subject["repair_admitted"] is False
    assert subject["lifecycle_mutation"] is False
    assert subject["idle_failure"] is False
    assert subject["recoverable_idleness"] is False
    assert subject["completion_authority"] is False
    assert subject["recheck_admitted"] is False
    assert subject["recheck_reason"] == "initial"
    assert child["continues"] is False
    assert FREE in decision["continue_task_ids"]
    assert SUBJECT not in decision["continue_task_ids"]
    assert set(decision["continue_task_ids"]).isdisjoint(decision["blocked_task_ids"])
    if kind == "dependency-running":
        assert subject["dependency_running"] is True
        assert subject["blocker"] is False
        assert subject["dependency_chain"] == [RUNNING]
        assert child["classification"] == "dependency-running"
        assert RUNNING in decision["running_task_ids"]
        assert RUNNING not in decision["continue_task_ids"]
    else:
        assert subject["blocker"] is True
        assert subject["dependency_running"] is False
        assert child["classification"] == "waiting-on-blocked-dependency"
        assert SUBJECT in child["blocked_dependencies"]
    if kind in EXTERNAL_KINDS:
        assert subject["external"] is True
        assert subject["failure"] is False
    if kind == "software-defect":
        assert subject["failure"] is True
        assert subject["external"] is False
    if kind == "hold":
        assert subject["held"] is True
    if kind == "exhausted-budget":
        assert subject["exhausted_budgets"] == ["repair"]
    assert decision["digest"] == digest({key: value for key, value in decision.items() if key != "digest"})


def test_aliases_and_status_codes_do_not_collapse_kinds():
    rights, _decision = board_of({
        "task_id": SUBJECT, "status": "blocked", "depends_on": [], "blocked_kind": "rights",
        "permission_ref": "corpus-rights", "permission_state": "restricted",
    })
    del rights
    permission = _decision["tasks"][0]
    assert permission["classification"] == "permission"
    assert "gated-input" not in permission["matched_kinds"]

    unavailable = board_of({
        "task_id": SUBJECT, "status": "blocked", "depends_on": [], "blocked_kind": "source_unavailable",
        "source_ref": "hf-nl", "source_state": "outage",
    })[1]["tasks"][0]
    assert unavailable["classification"] == "source-outage"

    budget = board_of({
        "task_id": SUBJECT, "status": "blocked", "depends_on": [], "blocked_kind": "budget",
        "budget": "provider", "budgets": {"provider": {"spent": DEFAULT_LIMITS["provider"], "limit": DEFAULT_LIMITS["provider"]}},
    })[1]["tasks"][0]
    assert budget["classification"] == "exhausted-budget"
    assert budget["exhausted_budgets"] == ["provider"]

    credentials = board_of({
        "task_id": SUBJECT, "status": "blocked", "depends_on": [], "source_ref": "hf-nl", "source_status": 401,
    })[1]["tasks"][0]
    permission_status = board_of({
        "task_id": SUBJECT, "status": "blocked", "depends_on": [], "source_ref": "hf-nl", "source_status": 403,
    })[1]["tasks"][0]
    outage = board_of({
        "task_id": SUBJECT, "status": "blocked", "depends_on": [], "source_ref": "hf-nl", "source_status": 429,
    })[1]["tasks"][0]
    assert credentials["classification"] == "credentials"
    assert permission_status["classification"] == "permission"
    assert outage["classification"] == "source-outage"
    assert credentials["matched_kinds"] == ["credentials"]
    assert permission_status["matched_kinds"] == ["permission"]
    assert outage["matched_kinds"] == ["source-outage"]


def test_stacked_evidence_keeps_precedence_and_every_match():
    body, receipt = reproducer("stacked")
    shared = {
        "task_id": SUBJECT, "status": "blocked", "depends_on": [],
        "credential_ref": "hf-read", "credential_state": "missing",
        "permission_ref": "hf-write", "permission_state": "revoked",
        "gate_ref": "corpus-nl", "gate_state": "closed",
        "source_ref": "hf-nl", "source_state": "outage", "source_status": 503,
        "failure_key": "router_crash", "reproducer": body, "reproducer_receipt": receipt,
        "budgets": {"task": {"spent": DEFAULT_LIMITS["task"], "limit": DEFAULT_LIMITS["task"]}},
    }
    stacked = board_of(shared, free_task())[1]
    subject = row(stacked, SUBJECT)
    assert subject["classification"] == "credentials"
    assert subject["matched_kinds"] == [
        "credentials", "permission", "gated-input", "source-outage", "exhausted-budget", "software-defect",
    ]
    assert subject["external"] is True
    assert subject["failure"] is False
    assert FREE in stacked["continue_task_ids"]
    held = board_of({**shared, "hold": True, "hold_markers": ["STOP"]}, free_task())[1]
    held_row = row(held, SUBJECT)
    assert held_row["classification"] == "hold"
    assert held_row["matched_kinds"][0] == "hold"
    assert "credentials" in held_row["matched_kinds"]
    assert held_row["recoverable_idleness"] is False
    assert FREE in held["continue_task_ids"]


def test_independent_ready_work_continues_beside_every_blocker():
    tasks = [
        {"task_id": "WWL-901", "status": "done", "depends_on": []},
        {"task_id": "WWL-902", "status": "ready", "depends_on": ["WWL-901"]},
        free_task(),
    ]
    subjects = {}
    for index, kind in enumerate(BLOCKER_KINDS):
        subject = f"WWL-{210 + index}"
        child = f"WWL-{310 + index}"
        subjects[kind] = subject
        if kind == "dependency-running":
            tasks.append({"task_id": RUNNING, "status": "running", "depends_on": [], "progress_token": TOKEN})
            tasks.append({"task_id": subject, "status": "ready", "depends_on": [RUNNING]})
        elif kind == "credentials":
            tasks.append({
                "task_id": subject, "status": "blocked", "depends_on": [],
                "credential_ref": "hf-read", "credential_state": "invalid",
            })
        elif kind == "permission":
            tasks.append({
                "task_id": subject, "status": "blocked", "depends_on": [],
                "permission_ref": "hf-write", "permission_state": "denied",
            })
        elif kind == "gated-input":
            tasks.append({
                "task_id": subject, "status": "blocked", "depends_on": [],
                "gate_ref": "corpus-nl", "gate_state": "closed",
            })
        elif kind == "source-outage":
            tasks.append({
                "task_id": subject, "status": "blocked", "depends_on": [],
                "source_ref": "hf-de", "source_state": "outage", "source_status": 503,
            })
        elif kind == "hold":
            tasks.append({
                "task_id": subject, "status": "blocked", "depends_on": [],
                "hold": True, "hold_markers": ["OPERATOR_STOP"],
            })
        elif kind == "exhausted-budget":
            tasks.append({
                "task_id": subject, "status": "blocked", "depends_on": [],
                "budgets": {"merge": {"spent": DEFAULT_LIMITS["merge"], "limit": DEFAULT_LIMITS["merge"]}},
            })
        else:
            body, receipt = reproducer("board")
            tasks.append({
                "task_id": subject, "status": "blocked", "depends_on": [],
                "failure_key": "router_crash", "reproducer": body, "reproducer_receipt": receipt,
            })
        tasks.append({"task_id": child, "status": "ready", "depends_on": [subject]})
    decision = board_of(*tasks)[1]
    assert decision["continue_task_ids"] == [FREE, "WWL-902"]
    assert decision["success"] is False
    assert decision["execution_completed"] is False
    for kind, subject in subjects.items():
        classified = row(decision, subject)
        assert classified["classification"] == kind
        assert classified["continues"] is False
        child = row(decision, f"WWL-{310 + BLOCKER_KINDS.index(kind)}")
        assert child["continues"] is False
        if kind == "dependency-running":
            assert child["classification"] == "dependency-running"
            assert classified["dependency_chain"] == [RUNNING]
            assert subject not in decision["blocked_task_ids"]
        else:
            assert child["classification"] == "waiting-on-blocked-dependency"
            assert subject in decision["blocked_task_ids"]
    assert row(decision, RUNNING)["classification"] == "running"
    assert row(decision, "WWL-901")["classification"] == "done"
    assert "WWL-901" not in decision["continue_task_ids"]


def test_dependency_running_reports_the_chain_and_is_not_idle_failure():
    owner = BlockerBoard(RUN)
    first = owner.classify({
        "observed_at": 50,
        "tasks": [
            {"task_id": "WWL-100", "status": "running", "depends_on": [], "progress_token": TOKEN},
            {"task_id": "WWL-110", "status": "ready", "depends_on": ["WWL-100"]},
            {"task_id": SUBJECT, "status": "ready", "depends_on": ["WWL-110", RUNNING]},
            {"task_id": RUNNING, "status": "running", "depends_on": [], "progress_token": TOKEN},
            free_task(),
            {"task_id": "WWL-120", "status": "blocked", "depends_on": [], "credential_ref": "hf-read", "credential_state": "missing"},
        ],
    })
    subject = row(first, SUBJECT)
    assert subject["classification"] == "dependency-running"
    assert subject["dependency_chain"] == ["WWL-110", "WWL-100", RUNNING]
    assert subject["idle_failure"] is False
    assert subject["continues"] is False
    assert subject["recheck_reason"] == "initial"
    assert first["continue_task_ids"] == [FREE]
    assert "WWL-120" in first["blocked_task_ids"]
    moved = owner.classify({
        "observed_at": 80,
        "pid": 44,
        "heartbeat_at": 80,
        "log_bytes": 10,
        "attempt": 3,
        "log": "heartbeat only",
        "tasks": [
            {"task_id": "WWL-100", "status": "running", "depends_on": [], "progress_token": TOKEN_NEXT, "pid": 44, "attempt": 3},
            {"task_id": "WWL-110", "status": "ready", "depends_on": ["WWL-100"]},
            {"task_id": SUBJECT, "status": "ready", "depends_on": ["WWL-110", RUNNING], "heartbeat_at": 80},
            {"task_id": RUNNING, "status": "running", "depends_on": [], "progress_token": TOKEN},
            free_task(),
            {"task_id": "WWL-120", "status": "blocked", "depends_on": [], "credential_ref": "hf-read", "credential_state": "missing"},
        ],
    })
    assert row(moved, SUBJECT)["recheck_admitted"] is True
    assert row(moved, SUBJECT)["recheck_reason"] == "prerequisite_changed"
    assert row(moved, SUBJECT)["classification"] == "dependency-running"
    assert row(moved, SUBJECT)["continues"] is False
    assert row(moved, "WWL-100")["recheck_admitted"] is False
    assert moved["continue_task_ids"] == [FREE]
    mixed = owner.classify({
        "observed_at": 90,
        "tasks": [
            {"task_id": "WWL-100", "status": "running", "depends_on": [], "progress_token": TOKEN_NEXT},
            {"task_id": "WWL-110", "status": "blocked", "depends_on": [], "credential_ref": "hf-read", "credential_state": "missing"},
            {"task_id": SUBJECT, "status": "ready", "depends_on": ["WWL-100", "WWL-110"]},
            free_task(),
        ],
    })
    assert row(mixed, SUBJECT)["classification"] == "waiting-on-blocked-dependency"
    assert row(mixed, SUBJECT)["dependency_running"] is False
    assert "WWL-110" in row(mixed, SUBJECT)["blocked_dependencies"]
    assert "WWL-100" in row(mixed, SUBJECT)["dependency_chain"]
    assert mixed["continue_task_ids"] == [FREE]


def test_recheck_requires_an_admitted_probe_or_a_changed_prerequisite():
    owner = BlockerBoard(RUN)
    outage = {
        "task_id": SUBJECT, "status": "blocked", "depends_on": [],
        "source_ref": "hf-nl", "source_state": "outage", "source_status": 503,
    }
    first = owner.classify({"observed_at": 100, "tasks": [outage, free_task()]})
    initial = row(first, SUBJECT)
    assert initial["recheck_reason"] == "initial"
    assert initial["recheck_admitted"] is False
    assert first["continue_task_ids"] == [FREE]
    noise = owner.classify({
        "observed_at": 120, "pid": 9, "heartbeat_at": 120, "log_bytes": 400, "attempt": 2,
        "log": "still down", "tasks": [{**outage, "pid": 9, "heartbeat_at": 120, "attempt": 2}, free_task()],
    })
    assert row(noise, SUBJECT)["prerequisite_digest"] == initial["prerequisite_digest"]
    assert row(noise, SUBJECT)["recheck_admitted"] is False
    assert row(noise, SUBJECT)["recheck_reason"] == "prerequisite_unchanged"
    assert noise["continue_task_ids"] == [FREE]
    changed = owner.classify({
        "observed_at": 130,
        "tasks": [{**outage, "source_status": 429}, free_task()],
    })
    assert row(changed, SUBJECT)["recheck_admitted"] is True
    assert row(changed, SUBJECT)["recheck_reason"] == "prerequisite_changed"
    assert row(changed, SUBJECT)["classification"] == "source-outage"
    assert row(changed, SUBJECT)["continues"] is False
    assert row(changed, SUBJECT)["prerequisite_digest"] != initial["prerequisite_digest"]
    owner.admit_probe(task_id=SUBJECT, probe_id="probe-1", not_before=200, reason="source-probe")
    early = owner.classify({
        "observed_at": 150, "probe_id": "probe-1", "tasks": [{**outage, "source_status": 429}, free_task()],
    })
    assert row(early, SUBJECT)["recheck_admitted"] is False
    assert owner.probes["probe-1"]["consumed"] is False
    with pytest.raises(BlockerContractError, match="not admitted"):
        owner.classify({"observed_at": 160, "probe_id": "probe-missing", "tasks": [{**outage, "source_status": 429}, free_task()]})
    assert owner.probes["probe-1"]["consumed"] is False
    due = owner.classify({
        "observed_at": 200, "probe_id": "probe-1", "tasks": [{**outage, "source_status": 429}, free_task()],
    })
    assert row(due, SUBJECT)["recheck_admitted"] is True
    assert row(due, SUBJECT)["recheck_reason"] == "admitted_probe"
    assert row(due, SUBJECT)["classification"] == "source-outage"
    assert row(due, SUBJECT)["continues"] is False
    assert due["probe_consumed"] is True
    assert due["continue_task_ids"] == [FREE]
    assert owner.probes["probe-1"]["consumed"] is True
    with pytest.raises(BlockerContractError, match="already consumed"):
        owner.classify({"observed_at": 205, "probe_id": "probe-1", "tasks": [{**outage, "source_status": 429}, free_task()]})
    cleared = owner.classify({
        "observed_at": 210,
        "tasks": [{
            "task_id": SUBJECT, "status": "ready", "depends_on": [],
            "source_ref": "hf-nl", "source_state": "available", "source_status": 200,
        }, free_task()],
    })
    assert row(cleared, SUBJECT)["classification"] == "ready"
    assert row(cleared, SUBJECT)["recheck_admitted"] is True
    assert row(cleared, SUBJECT)["recheck_reason"] == "prerequisite_changed"
    assert cleared["continue_task_ids"] == sorted([FREE, SUBJECT])


def test_hold_is_not_recoverable_idleness():
    owner = BlockerBoard(RUN)
    held = {
        "task_id": SUBJECT, "status": "blocked", "depends_on": [],
        "hold": True, "hold_markers": ["HOLD"],
    }
    first = owner.classify({"observed_at": 10, "tasks": [held, free_task()]})
    assert row(first, SUBJECT)["classification"] == "hold"
    assert row(first, SUBJECT)["recoverable_idleness"] is False
    assert row(first, SUBJECT)["idle_failure"] is False
    assert first["continue_task_ids"] == [FREE]
    owner.admit_probe(task_id=SUBJECT, probe_id="hold-probe", not_before=20, reason="hold-probe")
    probed = owner.classify({"observed_at": 20, "probe_id": "hold-probe", "tasks": [held, free_task()]})
    assert row(probed, SUBJECT)["classification"] == "hold"
    assert row(probed, SUBJECT)["continues"] is False
    assert row(probed, SUBJECT)["recheck_reason"] == "admitted_probe"
    assert probed["continue_task_ids"] == [FREE]
    released = owner.classify({"observed_at": 30, "tasks": [
        {"task_id": SUBJECT, "status": "ready", "depends_on": []}, free_task(),
    ]})
    assert row(released, SUBJECT)["classification"] == "ready"
    assert released["continue_task_ids"] == sorted([FREE, SUBJECT])
    campaign = BlockerBoard(RUN).classify({
        "observed_at": 40, "hold": True, "hold_markers": ["STOP"],
        "tasks": [free_task(), {"task_id": SUBJECT, "status": "running", "depends_on": [], "progress_token": TOKEN}],
    })
    assert campaign["continue_task_ids"] == []
    assert campaign["running_task_ids"] == []
    assert row(campaign, FREE)["classification"] == "hold"
    assert row(campaign, SUBJECT)["classification"] == "hold"
    assert row(campaign, FREE)["recoverable_idleness"] is False
    resumed = BlockerBoard(RUN)
    resumed.classify({
        "observed_at": 50, "hold": True,
        "tasks": [free_task(), {"task_id": SUBJECT, "status": "ready", "depends_on": []}],
    })
    opened = resumed.classify({"observed_at": 60, "tasks": [free_task(), {"task_id": SUBJECT, "status": "ready", "depends_on": []}]})
    assert opened["continue_task_ids"] == sorted([FREE, SUBJECT])
    assert row(opened, FREE)["recheck_admitted"] is True
    assert row(opened, FREE)["recheck_reason"] == "prerequisite_changed"


def test_exhausted_budget_is_visible_and_cannot_be_refunded(tmp_path: Path):
    owner = BlockerBoard(RUN)
    blocked = {
        "task_id": SUBJECT, "status": "blocked", "depends_on": [],
        "budgets": {"task": {"spent": DEFAULT_LIMITS["task"], "limit": DEFAULT_LIMITS["task"]}},
        "incident": {"stage": "task", "defect": {"failure_key": "router_crash"}},
    }
    first = owner.classify({"observed_at": 10, "tasks": [blocked, free_task()]})
    assert row(first, SUBJECT)["classification"] == "exhausted-budget"
    assert row(first, SUBJECT)["success"] is False
    assert first["continue_task_ids"] == [FREE]
    owner.admit_probe(task_id=SUBJECT, probe_id="budget-probe", not_before=20, reason="budget-probe")
    probed = owner.classify({"observed_at": 20, "probe_id": "budget-probe", "tasks": [blocked, free_task()]})
    assert row(probed, SUBJECT)["classification"] == "exhausted-budget"
    assert row(probed, SUBJECT)["continues"] is False
    assert row(probed, SUBJECT)["success"] is False
    with pytest.raises(BlockerContractError, match="refunded"):
        owner.classify({"observed_at": 30, "tasks": [
            {"task_id": SUBJECT, "status": "ready", "depends_on": [], "incident": blocked["incident"]},
            free_task(),
        ]})
    with pytest.raises(BlockerContractError, match="refunded"):
        owner.classify({"observed_at": 30, "tasks": [
            {"task_id": SUBJECT, "status": "blocked", "depends_on": [], "incident": {
                "stage": "validation", "defect": {"failure_key": "router_crash"},
            }},
            free_task(),
        ]})
    path = tmp_path / "blockers.json"
    owner.save(path)
    restored = BlockerBoard.load(path)
    again = restored.classify({"observed_at": 40, "pid": 3, "tasks": [blocked, free_task()]})
    assert row(again, SUBJECT)["classification"] == "exhausted-budget"
    assert row(again, SUBJECT)["recheck_admitted"] is False
    assert again["continue_task_ids"] == [FREE]
    with pytest.raises(BlockerContractError, match="already consumed"):
        restored.classify({"observed_at": 41, "probe_id": "budget-probe", "tasks": [blocked, free_task()]})
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["records"][0]["exhausted_budgets"] = []
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(BlockerContractError, match="digest"):
        BlockerBoard.load(path)


def test_checkpoint_rejects_completion_authority(tmp_path: Path):
    owner = BlockerBoard(RUN)
    owner.classify({"observed_at": 10, "tasks": [free_task()]})
    path = tmp_path / "blockers.json"
    owner.save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["completion_authority"] = True
    body = {key: payload[key] for key in ("schema", "run_id", "records", "probes", "completion_authority")}
    payload["digest"] = digest(body)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(BlockerContractError, match="completion authority"):
        BlockerBoard.load(path)


def test_ledger_exhaustion_is_authoritative():
    defect = {"failure_key": "router_crash"}
    ledger = ProgressLedger(RUN)
    observed = ledger.observe({
        "run_id": RUN, "task_id": SUBJECT, "stage": "task", "observed_at": 10, "defect": defect,
    })
    for _ in range(DEFAULT_LIMITS["repair"]):
        ledger.charge(observed["incident_id"], "repair")
    owner = BlockerBoard(RUN, ledger=ledger)
    incident = {"stage": "task", "defect": defect}
    blocked = {"task_id": SUBJECT, "status": "blocked", "depends_on": [], "incident": incident}
    first = owner.classify({"observed_at": 20, "tasks": [blocked, free_task()]})
    assert row(first, SUBJECT)["classification"] == "exhausted-budget"
    assert row(first, SUBJECT)["exhausted_budgets"] == ["repair"]
    assert first["continue_task_ids"] == [FREE]
    assert ledger.incidents[observed["incident_id"]]["spent"]["repair"] == DEFAULT_LIMITS["repair"]
    with pytest.raises(BlockerContractError, match="progress ledger"):
        owner.classify({"observed_at": 21, "tasks": [{
            **blocked, "budgets": {"repair": {"spent": 0, "limit": DEFAULT_LIMITS["repair"]}},
        }, free_task()]})
    same = owner.classify({"observed_at": 22, "tasks": [blocked, free_task()]})
    assert row(same, SUBJECT)["classification"] == "exhausted-budget"
    assert row(same, SUBJECT)["recheck_admitted"] is False
    validation = ledger.observe({
        "run_id": RUN, "task_id": SUBJECT, "stage": "validation", "observed_at": 30, "defect": defect,
    })
    assert validation["incident_id"] != observed["incident_id"]
    moved = owner.classify({"observed_at": 40, "tasks": [
        {"task_id": SUBJECT, "status": "ready", "depends_on": [], "incident": {"stage": "validation", "defect": defect}},
        free_task(),
    ]})
    assert row(moved, SUBJECT)["classification"] == "ready"
    assert row(moved, SUBJECT)["recheck_reason"] == "prerequisite_changed"
    assert moved["continue_task_ids"] == sorted([FREE, SUBJECT])
    assert ledger.incidents[observed["incident_id"]]["spent"]["repair"] == DEFAULT_LIMITS["repair"]
    returned = owner.classify({"observed_at": 50, "tasks": [blocked, free_task()]})
    assert row(returned, SUBJECT)["classification"] == "exhausted-budget"
    assert row(returned, SUBJECT)["success"] is False


def test_unknown_effects_are_not_a_software_defect():
    body, receipt = reproducer("effects")
    owner = BlockerBoard(RUN)
    unknown = {
        "task_id": SUBJECT, "status": "blocked", "depends_on": [], "effects_unknown": True,
        "failure_key": "router_crash", "reproducer": body, "reproducer_receipt": receipt,
    }
    first = owner.classify({"observed_at": 10, "tasks": [unknown, free_task()]})
    subject = row(first, SUBJECT)
    assert subject["classification"] == "effects-unknown"
    assert subject["kind"] == ""
    assert subject["failure"] is False
    assert "software-defect" in subject["matched_kinds"]
    assert subject["repair_admitted"] is False
    assert subject["continues"] is False
    assert SUBJECT in first["unresolved_task_ids"]
    assert first["continue_task_ids"] == [FREE]
    owner.admit_probe(task_id=SUBJECT, probe_id="effects-probe", not_before=20, reason="effects-probe")
    probed = owner.classify({"observed_at": 20, "probe_id": "effects-probe", "tasks": [unknown, free_task()]})
    assert row(probed, SUBJECT)["classification"] == "effects-unknown"
    assert row(probed, SUBJECT)["effects_unknown"] is True
    assert probed["continue_task_ids"] == [FREE]
    known = owner.classify({"observed_at": 30, "tasks": [
        {**unknown, "effects_unknown": False}, free_task(),
    ]})
    assert row(known, SUBJECT)["classification"] == "software-defect"
    assert row(known, SUBJECT)["failure"] is True
    assert row(known, SUBJECT)["repair_admitted"] is False
    assert row(known, SUBJECT)["continues"] is False
    assert known["continue_task_ids"] == [FREE]


def test_fail_closed_on_cycles_secrets_completion_and_mismatches():
    with pytest.raises(BlockerContractError, match="completion authority"):
        board_of(free_task(), completion_authority=True)
    with pytest.raises(BlockerContractError, match="reset"):
        board_of(free_task(), reset_budget=True)
    with pytest.raises(BlockerContractError, match="credential material"):
        board_of({"task_id": SUBJECT, "status": "ready", "depends_on": [], "api_key": "hf-secret"})
    body, _receipt = reproducer("secret")
    with pytest.raises(BlockerContractError, match="credential material"):
        board_of({
            "task_id": SUBJECT, "status": "blocked", "depends_on": [], "failure_key": "router_crash",
            "reproducer": {**body, "api_key": "hf-secret"}, "reproducer_receipt": "sha256:" + ("c" * 64),
        })
    with pytest.raises(BlockerContractError, match="cycle"):
        board_of(
            {"task_id": "WWL-1", "status": "ready", "depends_on": ["WWL-2"]},
            {"task_id": "WWL-2", "status": "ready", "depends_on": ["WWL-1"]},
        )
    with pytest.raises(BlockerContractError, match="unknown task"):
        board_of({"task_id": SUBJECT, "status": "ready", "depends_on": ["WWL-missing"]})
    with pytest.raises(BlockerContractError, match="distinguished blocker"):
        board_of({"task_id": SUBJECT, "status": "blocked", "depends_on": []})
    with pytest.raises(BlockerContractError, match="ready work"):
        board_of({
            "task_id": SUBJECT, "status": "ready", "depends_on": [],
            "credential_ref": "hf-read", "credential_state": "missing",
        })
    with pytest.raises(BlockerContractError, match="does not match"):
        board_of({
            "task_id": SUBJECT, "status": "blocked", "depends_on": [], "blocked_kind": "software-defect",
            "credential_ref": "hf-read", "credential_state": "missing",
        })
    with pytest.raises(BlockerContractError, match="verified reproducer"):
        board_of({
            "task_id": SUBJECT, "status": "blocked", "depends_on": [],
            "failure_key": "router_crash", "reproducer_receipt": "sha256:" + ("d" * 64),
        })
    body, receipt = reproducer("mismatch")
    with pytest.raises(BlockerContractError, match="does not match"):
        board_of({
            "task_id": SUBJECT, "status": "blocked", "depends_on": [], "failure_key": "router_crash",
            "reproducer": body, "reproducer_receipt": "sha256:" + ("e" * 64),
        })
    with pytest.raises(BlockerContractError, match="source status"):
        board_of({"task_id": SUBJECT, "status": "blocked", "depends_on": [], "source_ref": "hf-nl", "source_status": 404})
    with pytest.raises(BlockerContractError, match="campaign ledger"):
        board_of({
            "task_id": SUBJECT, "status": "blocked", "depends_on": [],
            "budgets": {"task": {"spent": 1, "limit": 9}},
        })
    with pytest.raises(BlockerContractError, match="exceeds"):
        board_of({
            "task_id": SUBJECT, "status": "blocked", "depends_on": [],
            "budgets": {"task": {"spent": 4, "limit": DEFAULT_LIMITS["task"]}},
        })
    with pytest.raises(BlockerContractError, match="repeats"):
        board_of(free_task(), free_task())
    with pytest.raises(BlockerContractError, match="progress token"):
        board_of({"task_id": RUNNING, "status": "running", "depends_on": []})
    with pytest.raises(BlockerContractError, match="done work"):
        board_of(
            {"task_id": RUNNING, "status": "ready", "depends_on": []},
            {"task_id": SUBJECT, "status": "done", "depends_on": [RUNNING]},
        )
    owner = BlockerBoard(RUN)
    with pytest.raises(BlockerContractError, match="classified blocker"):
        owner.admit_probe(task_id=SUBJECT, probe_id="early", not_before=1, reason="early")
    owner.classify({"observed_at": 100, "tasks": [subject_for("credentials")[0], free_task()]})
    with pytest.raises(BlockerContractError, match="predates"):
        owner.admit_probe(task_id=SUBJECT, probe_id="probe-a", not_before=50, reason="early")
    owner.admit_probe(task_id=SUBJECT, probe_id="probe-a", not_before=100, reason="credential-probe")
    with pytest.raises(BlockerContractError, match="already has"):
        owner.admit_probe(task_id=SUBJECT, probe_id="probe-b", not_before=100, reason="credential-probe")
    with pytest.raises(BlockerContractError, match="classified blocker"):
        owner.admit_probe(task_id=FREE, probe_id="probe-c", not_before=100, reason="ready-probe")
    mismatched = ProgressLedger(RUN, limits={"repair": 1})
    with pytest.raises(BlockerContractError, match="campaign ledger"):
        BlockerBoard(RUN, ledger=mismatched)


def test_receipt_matches_the_blocker_contract():
    receipt = json.loads(Path("artifacts/worldwide-legal-search/receipts/WWL-104.json").read_text(encoding="utf-8"))
    assert receipt["schema"] == "worldwide-legal-search.task-receipt@1"
    assert receipt["task_id"] == "WWL-104"
    assert receipt["title"] == "Classify blockers and continue independent ready work"
    assert receipt["goal_id"] == "WWL-G090"
    assert receipt["subgoal_id"] == "WWL-G090.S1"
    assert receipt["root_goal_id"] == "WWL-G000"
    assert receipt["depends_on"] == ["WWL-101"]
    assert receipt["blocker_kinds"] == list(BLOCKER_KINDS)
    assert receipt["recheck_reasons"] == list(ADMITTED_RECHECK_REASONS)
    assert receipt["hold_markers"] == list(HOLD_MARKERS)
    assert receipt["completion_authority"] is False
    assert receipt["self_certified_completion"] is False
    assert receipt["execution_completed"] is False
    assert receipt["outputs"] == [
        "test/api/test_worldwide_legal_search_blockers.py",
        "ipfs_accelerate_py/agent_supervisor/project_adapters/worldwide_legal_search_blockers.py",
        "artifacts/worldwide-legal-search/receipts/WWL-104.json",
    ]
    assert receipt["validation_argv"] == ["python3", "-m", "pytest", "test/api/test_worldwide_legal_search_blockers.py", "-q"]
    assert "does not grant campaign completion" in receipt["limitations"][-1]
    text = "\n".join(receipt["limitations"])
    assert "admitted probe" in text
    assert "independent" in text
    assert "/usr/bin/python3.12" in text
