"""WWL-101: bounded deterministic recovery and lifecycle ownership.

Native recovery admits a worker only when the kernel PID birth matches and
both the process generation and the lease generation are current. It releases
stale claims, fences a dead worker, and wakes an idle runnable queue. Unknown
effects stay unknown. Holds and deadlines admit no mutation. The recover
budget is the campaign limit and a checkpoint cannot refund it.
"""
from __future__ import annotations

import hashlib
import json
import os

import pytest

from ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_progress import (
    DEFAULT_LIMITS, ProgressLedger, incident_identity,
)
from ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_recovery import (
    DISPOSITIONS, EFFECT_STATUSES, HOLD_MARKERS, RECOVERY_LIMIT, RecoveryContractError, RecoveryOwner,
    read_pid_birth, verify_generations, verify_pid_birth,
)

RUN = "wwl-run-101"
TASK = "WWL-101"


def sha(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


class Births:
    def __init__(self, rows: dict[int, tuple[bool, int]] | None = None):
        self.rows = dict(rows or {})

    def set(self, pid: int, *, alive: bool, birth: int) -> None:
        self.rows[pid] = (alive, birth)

    def __call__(self, pid: int) -> dict[str, object]:
        row = self.rows.get(pid)
        if row is None:
            return {"pid": pid, "alive": False, "start_ticks": 0, "state": "missing"}
        alive, birth = row
        return {"pid": pid, "alive": alive, "start_ticks": birth, "state": "R" if alive else "dead"}


def effect(effect_id: str, task_id: str, status: str, text: str = "payload") -> dict:
    return {"effect_id": effect_id, "task_id": task_id, "status": status, "digest": sha(text)}


def claim(claim_id: str, task_id: str, *, pid: int = 10, birth: int = 100, process: int = 2, lease: int = 5,
          worker: str = "worker-a", effect_id: str = "") -> dict:
    body = {
        "claim_id": claim_id, "task_id": task_id, "worker_id": worker, "pid": pid, "birth": birth,
        "process_generation": process, "lease_generation": lease,
    }
    if effect_id:
        body["effect_id"] = effect_id
    return body


def owner_binding(*, pid: int = 10, birth: int = 100, process: int = 2, lease: int = 5, worker: str = "worker-a") -> dict:
    return {
        "worker_id": worker, "pid": pid, "birth": birth,
        "process_generation": process, "lease_generation": lease,
    }


def observation(action: str, at: float, deadline: float, **extra) -> dict:
    body = {
        "action_id": action,
        "observed_at": at,
        "deadline_at": deadline,
        "incident": {"task_id": TASK, "stage": "task", "defect": {"failure_key": "worker_lost"}},
        "completion_authority": False,
    }
    body.update(extra)
    return body


def bind(owner: RecoveryOwner, births: Births, **identity) -> dict:
    fields = owner_binding(**identity)
    births.set(fields["pid"], alive=True, birth=fields["birth"])
    return owner.bind_owner(
        worker_id=fields["worker_id"], pid=fields["pid"], birth=fields["birth"],
        process_generation=fields["process_generation"], lease_generation=fields["lease_generation"],
    )


def test_constants_match_the_campaign_recover_budget():
    assert RECOVERY_LIMIT == DEFAULT_LIMITS["recover"] == 2
    assert HOLD_MARKERS == ("HOLD", "STOP", "OPERATOR_STOP")
    assert EFFECT_STATUSES == ("unknown", "known_none", "applied")
    assert "dead_worker" in DISPOSITIONS
    assert "stale_claim" in DISPOSITIONS
    assert "idle_runnable_queue" in DISPOSITIONS
    assert "effects_preserved" in DISPOSITIONS
    assert "held" in DISPOSITIONS
    assert "deadline_exceeded" in DISPOSITIONS
    assert "recovery_bound_exhausted" in DISPOSITIONS


def test_pid_birth_matches_the_kernel_and_rejects_reuse_and_death():
    observed = read_pid_birth(os.getpid())
    assert observed["alive"] is True
    assert observed["start_ticks"] > 0
    matched = verify_pid_birth(os.getpid(), observed["start_ticks"])
    assert matched["pid_birth_matches"] is True
    assert matched["pid_reused"] is False
    assert matched["observed_birth"] == observed["start_ticks"]
    reused = verify_pid_birth(os.getpid(), observed["start_ticks"] + 1)
    assert reused["pid_birth_matches"] is False
    assert reused["pid_reused"] is True
    assert reused["alive"] is True
    missing = read_pid_birth(2**31 - 1)
    assert missing["alive"] is False
    assert missing["start_ticks"] == 0
    dead = verify_pid_birth(2**31 - 1, observed["start_ticks"])
    assert dead["pid_birth_matches"] is False
    assert dead["pid_reused"] is False
    assert dead["alive"] is False

    def zombie(pid: int) -> dict:
        return {"pid": pid, "alive": True, "start_ticks": 50, "state": "Z"}

    reported = verify_pid_birth(7, 50, probe=zombie)
    assert reported["alive"] is False
    assert reported["pid_birth_matches"] is False
    births = Births({10: (True, 100)})
    assert verify_pid_birth(10, 100, probe=births)["pid_birth_matches"] is True
    births.set(10, alive=True, birth=250)
    reused_script = verify_pid_birth(10, 100, probe=births)
    assert reused_script["pid_reused"] is True
    assert reused_script["pid_birth_matches"] is False
    births.set(10, alive=False, birth=100)
    assert verify_pid_birth(10, 100, probe=births)["alive"] is False
    with pytest.raises(RecoveryContractError, match="pid birth"):
        verify_pid_birth(10, 0, probe=births)
    with pytest.raises(RecoveryContractError, match="pid birth"):
        verify_pid_birth(10, True, probe=births)  # type: ignore[arg-type]


def test_process_and_lease_generations_are_verified_independently():
    current = verify_generations(2, 5, owner_process_generation=2, owner_lease_generation=5)
    assert current["match"] is True
    assert current["process_generation_current"] is True
    assert current["lease_generation_current"] is True
    assert current["stale"] is False
    assert current["future"] is False
    stale_lease = verify_generations(2, 4, owner_process_generation=2, owner_lease_generation=5)
    assert stale_lease["stale"] is True
    assert stale_lease["lease_generation_current"] is False
    assert stale_lease["process_generation_current"] is True
    stale_process = verify_generations(1, 5, owner_process_generation=2, owner_lease_generation=5)
    assert stale_process["stale"] is True
    assert stale_process["process_generation_current"] is False
    future = verify_generations(3, 5, owner_process_generation=2, owner_lease_generation=5)
    assert future["future"] is True
    assert future["stale"] is False
    conflict = verify_generations(1, 6, owner_process_generation=2, owner_lease_generation=5)
    assert conflict["conflict"] is True
    assert conflict["match"] is False
    with pytest.raises(RecoveryContractError, match="lease generation"):
        verify_generations(2, False, owner_process_generation=2, owner_lease_generation=5)  # type: ignore[arg-type]


def test_owner_binding_requires_a_matching_birth_and_refuses_generation_jumps():
    births = Births()
    owner = RecoveryOwner(RUN, probe=births)
    bound = bind(owner, births)
    assert bound["disposition"] == "owner_bound"
    assert bound["pid_birth_matches"] is True
    assert bound["process_generation"] == 2
    assert bound["lease_generation"] == 5
    assert bound["completion_authority"] is False
    again = bind(owner, births)
    assert again["deduplicated"] is True
    assert owner.process_generation == 2
    births.set(10, alive=True, birth=101)
    with pytest.raises(RecoveryContractError, match="pid birth"):
        owner.bind_owner(worker_id="worker-a", pid=10, birth=100, process_generation=2, lease_generation=5)
    births.set(10, alive=True, birth=100)
    births.set(11, alive=True, birth=100)
    with pytest.raises(RecoveryContractError, match="already bound"):
        owner.bind_owner(worker_id="worker-b", pid=11, birth=100, process_generation=2, lease_generation=5)
    with pytest.raises(RecoveryContractError, match="regressed"):
        owner.bind_owner(worker_id="worker-a", pid=10, birth=100, process_generation=1, lease_generation=5)
    births.set(12, alive=True, birth=300)
    with pytest.raises(RecoveryContractError, match="jumped"):
        owner.bind_owner(worker_id="worker-c", pid=12, birth=300, process_generation=4, lease_generation=7)


def test_stale_claim_is_released_and_the_live_owner_keeps_its_generation():
    births = Births()
    owner = RecoveryOwner(RUN, probe=births)
    bind(owner, births)
    current = claim("claim-current", "task-current", effect_id="effect-current")
    stale = claim("claim-stale", "task-stale", lease=4, effect_id="effect-stale")
    older_process = claim("claim-process", "task-process", process=1, effect_id="effect-process")
    decision = owner.recover(observation(
        "action-stale", 100, 200,
        owner=owner_binding(),
        claims=[current, stale, older_process],
        effects=[
            effect("effect-current", "task-current", "known_none", "current"),
            effect("effect-stale", "task-stale", "known_none", "stale"),
            effect("effect-process", "task-process", "known_none", "process"),
        ],
    ))
    assert decision["disposition"] == "stale_claim"
    assert decision["stale_claim_ids"] == ["claim-stale", "claim-process"]
    assert decision["dead_worker"] is False
    assert decision["owner_live"] is True
    assert decision["generation_advanced"] is False
    assert decision["process_generation"] == 2
    assert decision["lease_generation"] == 5
    assert decision["pid_birth_matches"] is True
    assert decision["applied"] is True
    assert decision["budget"]["charged"] is True
    assert decision["budget"]["spent"] == 1
    assert decision["budget"]["remaining"] == 1
    assert decision["completion_authority"] is False
    assert owner.owner is not None and owner.owner["pid"] == 10 and owner.owner["fenced"] is False
    assert owner.claims["claim-current"]["state"] == "current"
    assert owner.claims["claim-stale"]["state"] == "released"
    assert owner.claims["claim-process"]["state"] == "released"
    assert decision["runnable_task_ids"] == ["task-process", "task-stale"]
    assert "task-current" not in owner.queue
    replay = owner.recover(observation(
        "action-stale", 100, 200,
        owner=owner_binding(),
        claims=[current, stale, older_process],
        effects=[
            effect("effect-current", "task-current", "known_none", "current"),
            effect("effect-stale", "task-stale", "known_none", "stale"),
            effect("effect-process", "task-process", "known_none", "process"),
        ],
    ))
    assert replay["deduplicated"] is True
    assert owner.spent[decision["incident_id"]] == 1
    with pytest.raises(RecoveryContractError, match="ahead of the durable lifecycle"):
        owner.recover(observation(
            "action-future", 110, 200,
            owner=owner_binding(),
            claims=[claim("claim-future", "task-future", lease=9)],
        ))


def test_dead_worker_and_pid_reuse_advance_one_generation_when_effects_are_known():
    births = Births()
    owner = RecoveryOwner(RUN, probe=births)
    bind(owner, births)
    births.set(10, alive=False, birth=0)
    decision = owner.recover(observation(
        "action-dead", 100, 200,
        owner=owner_binding(),
        claims=[claim("claim-dead", "task-dead", effect_id="effect-dead")],
        effects=[effect("effect-dead", "task-dead", "known_none", "none")],
    ))
    assert decision["disposition"] == "dead_worker"
    assert decision["dead_worker"] is True
    assert decision["pid_birth_matches"] is False
    assert "pid_birth_checked" in decision["findings"]
    assert "process_generation_checked" in decision["findings"]
    assert "lease_generation_checked" in decision["findings"]
    assert decision["generation_advanced"] is True
    assert decision["process_generation"] == 3
    assert decision["lease_generation"] == 6
    assert decision["owner_live"] is False
    assert decision["effects_unknown"] is False
    assert decision["runnable_task_ids"] == ["task-dead"]
    assert decision["budget"]["spent"] == 1
    assert owner.owner is None
    assert owner.claims["claim-dead"]["state"] == "released"
    assert owner.fences[0]["birth"] == 100
    assert owner.fences[0]["process_generation"] == 2
    with pytest.raises(RecoveryContractError, match="fenced pid birth"):
        births.set(10, alive=True, birth=100)
        owner.bind_owner(worker_id="worker-a", pid=10, birth=100, process_generation=3, lease_generation=6)
    births.set(11, alive=True, birth=400)
    rebound = owner.bind_owner(worker_id="worker-b", pid=11, birth=400, process_generation=3, lease_generation=6)
    assert rebound["disposition"] == "owner_bound"
    assert owner.process_generation == 3 and owner.lease_generation == 6
    births.set(11, alive=True, birth=401)
    reused = owner.recover(observation(
        "action-reuse", 120, 200,
        incident={"task_id": TASK, "stage": "task", "defect": {"failure_key": "pid_reused"}},
        owner=owner_binding(pid=11, birth=400, process=3, lease=6, worker="worker-b"),
        claims=[claim("claim-reuse", "task-reuse", pid=11, birth=400, process=3, lease=6, worker="worker-b", effect_id="effect-reuse")],
        effects=[effect("effect-reuse", "task-reuse", "known_none", "reused")],
    ))
    assert reused["disposition"] == "dead_worker"
    assert reused["pid_reused"] is True
    assert reused["pid_birth_matches"] is False
    assert reused["process_generation"] == 4
    assert reused["lease_generation"] == 7
    assert owner.owner is None
    assert "task-reuse" in owner.queue


def test_unknown_effects_are_preserved_and_block_requeue_and_rebind():
    births = Births()
    owner = RecoveryOwner(RUN, probe=births)
    bind(owner, births)
    births.set(10, alive=False, birth=0)
    original = effect("effect-unknown", "task-unknown", "unknown", "in-flight")
    decision = owner.recover(observation(
        "action-unknown", 100, 200,
        owner=owner_binding(),
        claims=[claim("claim-unknown", "task-unknown", effect_id="effect-unknown")],
        effects=[original],
        runnable_task_ids=["task-ready"],
        idle_since=10,
        queue_deadline_at=50,
    ))
    assert decision["disposition"] == "effects_preserved"
    assert decision["effects_unknown"] is True
    assert decision["effects_preserved"] is True
    assert decision["dead_worker"] is True
    assert decision["generation_advanced"] is False
    assert decision["process_generation"] == 2
    assert decision["lease_generation"] == 5
    assert "task-unknown" not in decision["runnable_task_ids"]
    assert decision["wake_authorized"] is False
    assert decision["disposition"] != "idle_runnable_queue"
    assert owner.effects["effect-unknown"] == original
    assert owner.owner is not None and owner.owner["fenced"] is True
    forged = effect("effect-unknown", "task-unknown", "applied", "in-flight")
    follow = owner.recover(observation(
        "action-unknown-follow", 110, 200,
        owner=owner_binding(),
        claims=[claim("claim-unknown", "task-unknown", effect_id="effect-unknown")],
        effects=[forged],
    ))
    assert follow["budget"]["charged"] is False
    assert owner.spent[decision["incident_id"]] == 1
    assert owner.effects["effect-unknown"]["status"] == "unknown"
    assert owner.effects["effect-unknown"]["digest"] == original["digest"]
    assert owner.process_generation == 2
    births.set(12, alive=True, birth=500)
    with pytest.raises(RecoveryContractError, match="unknown effects"):
        owner.bind_owner(worker_id="worker-c", pid=12, birth=500, process_generation=3, lease_generation=6)
    with pytest.raises(RecoveryContractError, match="effect identity changed"):
        owner.recover(observation(
            "action-conflict", 120, 200,
            owner=owner_binding(),
            effects=[effect("effect-unknown", "task-other", "unknown", "in-flight")],
        ))


def test_idle_runnable_queue_waits_for_its_deadline_and_does_not_hide_a_dead_worker():
    births = Births()
    owner = RecoveryOwner(RUN, probe=births)
    waiting = owner.recover(observation(
        "action-wait", 100, 500,
        runnable_task_ids=["task-ready", "task-other"],
        idle_since=90,
        queue_deadline_at=150,
    ))
    assert waiting["disposition"] == "idle_within_deadline"
    assert waiting["idle_runnable"] is True
    assert waiting["wake_authorized"] is False
    assert waiting["applied"] is False
    assert waiting["budget"]["charged"] is False
    assert waiting["budget"]["spent"] == 0
    assert waiting["runnable_task_ids"] == ["task-other", "task-ready"]
    assert waiting["owner_live"] is False
    woke = owner.recover(observation(
        "action-wake", 150, 500,
        runnable_task_ids=["task-ready", "task-other"],
        idle_since=90,
        queue_deadline_at=150,
    ))
    assert woke["disposition"] == "idle_runnable_queue"
    assert woke["wake_authorized"] is True
    assert woke["applied"] is True
    assert woke["budget"]["spent"] == 1
    assert woke["budget"]["remaining"] == 1
    assert woke["runnable_task_ids"] == ["task-other", "task-ready"]
    assert owner.owner is None
    repeat = owner.recover(observation(
        "action-wake-again", 160, 500,
        runnable_task_ids=["task-ready", "task-other"],
        idle_since=90,
        queue_deadline_at=150,
    ))
    assert repeat["wake_authorized"] is True
    assert repeat["applied"] is False
    assert repeat["budget"]["charged"] is False
    assert owner.spent[woke["incident_id"]] == 1

    births = Births()
    live = RecoveryOwner(RUN, probe=births)
    bind(live, births)
    covered = live.recover(observation(
        "action-covered", 400, 500,
        owner=owner_binding(),
        runnable_task_ids=["task-ready"],
        idle_since=10,
        queue_deadline_at=20,
    ))
    assert covered["disposition"] == "owner_live"
    assert covered["wake_authorized"] is False
    assert covered["dead_worker"] is False
    assert covered["budget"]["charged"] is False
    assert covered["process_generation"] == 2
    assert covered["pid_birth_matches"] is True
    births.set(10, alive=False, birth=0)
    masked = live.recover(observation(
        "action-masked", 420, 500,
        owner=owner_binding(),
        claims=[claim("claim-masked", "task-ready", effect_id="effect-ready")],
        effects=[effect("effect-ready", "task-ready", "known_none")],
        runnable_task_ids=["task-ready"],
        idle_since=10,
        queue_deadline_at=20,
    ))
    assert masked["disposition"] == "dead_worker"
    assert masked["idle_runnable"] is False
    assert masked["generation_advanced"] is True
    assert masked["wake_authorized"] is False


def test_hold_and_deadline_admit_no_lifecycle_mutation():
    births = Births()
    owner = RecoveryOwner(RUN, probe=births)
    bind(owner, births)
    births.set(10, alive=False, birth=0)
    owner.set_hold(True, marker="OPERATOR_STOP")
    held = owner.recover(observation(
        "action-held", 100, 200,
        owner=owner_binding(),
        claims=[claim("claim-held", "task-held", effect_id="effect-held")],
        effects=[effect("effect-held", "task-held", "known_none")],
        runnable_task_ids=["task-held"],
        idle_since=1,
        queue_deadline_at=2,
    ))
    assert held["disposition"] == "held"
    assert held["held"] is True
    assert held["applied"] is False
    assert held["dead_worker"] is True
    assert held["budget"]["charged"] is False
    assert held["budget"]["spent"] == 0
    assert held["generation_advanced"] is False
    assert owner.process_generation == 2
    assert owner.lease_generation == 5
    assert owner.owner is not None and owner.owner["fenced"] is False
    assert owner.effects == {}
    assert owner.queue == []
    assert owner.fences == []
    with pytest.raises(RecoveryContractError, match="hold blocks"):
        owner.bind_owner(worker_id="worker-a", pid=10, birth=100, process_generation=2, lease_generation=5)
    owner.set_hold(False)
    births.set(10, alive=True, birth=100)
    marked = owner.recover(observation(
        "action-stop", 110, 200,
        hold_markers=["STOP"],
        owner=owner_binding(),
    ))
    assert marked["disposition"] == "held"
    assert marked["applied"] is False
    assert owner.owner is not None and owner.owner["fenced"] is False
    with pytest.raises(RecoveryContractError, match="hold marker"):
        owner.recover(observation("action-pause", 110, 200, hold_markers=["PAUSE"], owner=owner_binding()))
    births.set(10, alive=False, birth=0)
    late_body = observation(
        "action-late", 200, 200,
        owner=owner_binding(),
        claims=[claim("claim-late", "task-late", effect_id="effect-late")],
        effects=[effect("effect-late", "task-late", "known_none")],
    )
    late = owner.recover(late_body)
    assert late["disposition"] == "deadline_exceeded"
    assert late["deadline_exceeded"] is True
    assert late["dead_worker"] is True
    assert late["applied"] is False
    assert late["budget"]["spent"] == 0
    assert owner.process_generation == 2
    assert owner.effects == {}
    assert owner.claims == {}
    assert owner.owner is not None and owner.owner["fenced"] is False
    still = owner.recover(late_body)
    assert still["deduplicated"] is True
    assert owner.owner is not None and owner.owner["fenced"] is False


def test_durable_bound_survives_restart_and_tracks_one_incident():
    births = Births()
    ledger = ProgressLedger(RUN)
    opened = ledger.observe({
        "run_id": RUN, "task_id": TASK, "stage": "task", "observed_at": 10,
        "defect": "queue idle pid=4 attempt=1",
    })
    token = ledger.progress_token
    owner = RecoveryOwner(RUN, ledger=ledger, probe=births)
    left = incident_identity(run_id=RUN, task_id=TASK, stage="task", defect="queue idle pid=4 attempt=1")
    right = incident_identity(run_id=RUN, task_id=TASK, stage="implementation", defect="queue idle pid=9 attempt=8")
    assert left == right
    assert left == opened["incident_id"]

    def wake(action: str, idle_since: float, defect: str) -> dict:
        return owner.recover(observation(
            action, 100, 500,
            incident={"task_id": TASK, "stage": "task", "defect": defect},
            runnable_task_ids=["task-ready"],
            idle_since=idle_since,
            queue_deadline_at=100,
        ))

    first = wake("bound-1", 10, "queue idle pid=4 attempt=1")
    second = wake("bound-2", 20, "queue idle pid=9 attempt=8")
    assert first["incident_id"] == second["incident_id"] == opened["incident_id"]
    assert first["budget"]["spent"] == 1 and first["budget"]["remaining"] == 1
    assert second["budget"]["spent"] == 2 and second["budget"]["remaining"] == 0
    assert second["budget"]["limit"] == 2
    assert ledger.incidents[opened["incident_id"]]["spent"]["recover"] == 2
    assert ledger.progress_token == token
    ledger.observe({
        "run_id": RUN, "task_id": TASK, "stage": "task", "observed_at": 120, "kind": "retry", "attempt": 4, "pid": 90,
        "defect": {"failure_key": "worker_lost", "message": "pid=90 attempt=4"},
    })
    assert ledger.incidents[opened["incident_id"]]["spent"]["recover"] == 2
    assert ledger.progress_token == token
    blocked = wake("bound-3", 30, "queue idle pid=15 attempt=2")
    assert blocked["disposition"] == "recovery_bound_exhausted"
    assert blocked["applied"] is False
    assert blocked["wake_authorized"] is False
    assert blocked["budget"]["status"] == "blocked"
    assert blocked["budget"]["spent"] == 2
    assert blocked["completion_authority"] is False
    assert owner.woken_epochs.__len__() == 2
    snapshot = owner.to_dict()
    assert snapshot["completion_authority"] is False
    assert snapshot["recover_limit"] == 2
    restored = RecoveryOwner.from_dict(json.loads(json.dumps(snapshot)), probe=births)
    assert restored.to_dict() == snapshot
    assert restored.ledger is not None
    assert restored.ledger.incidents[opened["incident_id"]]["spent"]["recover"] == 2
    again = restored.recover(observation(
        "bound-4", 140, 500,
        incident={"task_id": TASK, "stage": "task", "defect": "queue idle pid=1 attempt=1"},
        runnable_task_ids=["task-fresh"],
        idle_since=40,
        queue_deadline_at=50,
    ))
    assert again["disposition"] == "recovery_bound_exhausted"
    assert restored.ledger.incidents[opened["incident_id"]]["spent"]["recover"] == 2
    tampered = json.loads(json.dumps(snapshot))
    tampered["recover_limit"] = 9
    with pytest.raises(RecoveryContractError, match="durable recovery bound"):
        RecoveryOwner.from_dict(tampered, probe=births)
    tampered = json.loads(json.dumps(snapshot))
    tampered["completion_authority"] = True
    with pytest.raises(RecoveryContractError, match="completion authority"):
        RecoveryOwner.from_dict(tampered, probe=births)
    widened = ProgressLedger(RUN, limits={"recover": 4})
    with pytest.raises(RecoveryContractError, match="durable recovery bound"):
        RecoveryOwner(RUN, ledger=widened, probe=births)
    with pytest.raises(RecoveryContractError, match="completion authority"):
        owner.recover(observation("bound-complete", 150, 500, completion_authority=True))
    assert "bound-complete" not in owner.to_dict()["actions"]


def test_applied_effects_are_not_replayed_and_completion_stays_closed():
    births = Births()
    owner = RecoveryOwner(RUN, probe=births)
    bind(owner, births)
    births.set(10, alive=False, birth=0)
    decision = owner.recover(observation(
        "action-applied", 100, 200,
        owner=owner_binding(),
        claims=[claim("claim-applied", "task-applied", effect_id="effect-applied")],
        effects=[effect("effect-applied", "task-applied", "applied", "landed")],
    ))
    assert decision["disposition"] == "dead_worker"
    assert "task-applied" not in decision["runnable_task_ids"]
    assert decision["completion_authority"] is False
    assert owner.effects["effect-applied"]["status"] == "applied"
    assert decision["generation_advanced"] is True
