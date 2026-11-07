"""WWL-103: quota fallback, unknown effects, and repair nonconvergence.

Inject typed quota, authentication, and launch unavailability; transport loss
after a mutation; malformed output; a dead supervisor; a live heartbeat that
does not move progress; and repeated repair of one defect. A second writer, a
false success, and an attempt refund are refused.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py import llm_router
from ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search import (
    CANONICAL_ROUTING,
    ROUTE_POLICY,
    WorldwideRouteError,
    evaluate_emergency_fallback,
    snapshot_workspace,
    trusted_pre_effect_receipt,
)
from ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_blockers import (
    BlockerBoard,
    reproducer_receipt,
)
from ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_progress import (
    DEFAULT_LIMITS,
    ProgressIdentityError,
    ProgressLedger,
)
from ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_recovery import (
    RecoveryContractError,
    RecoveryOwner,
)
from ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_repair import (
    CANARY_SCHEMA,
    CANONICAL_PYTHON,
    REGRESSION_SCHEMA,
    REPAIR_LIMIT,
    RepairContractError,
    RepairSession,
    repair_task_identity,
)
from ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_watchdog import (
    WorldwideWatchdogBridge,
)

RUN = "wwl-run-103"
TASK = "WWL-103"
FREE = "WWL-900"
DEFECT = "native_supervisor_defect"
SOURCE = "ipfs_accelerate_py/llm_router.py"
REPRO = "test/api/test_worldwide_legal_search_faults_repro.py"
BASELINE_REV = "a" * 40
BASELINE_TEXT = "def route():\n    return 'broken'\n"
FIXED_TEXT = "def route():\n    return 'healthy'\n"
REPRO_TEXT = "def test_reproduced_native_defect():\n    assert 'native_supervisor_defect'\n"
ARGV = ["python3", "-m", "pytest", REPRO, "-q"]
RECEIPT = Path("artifacts/worldwide-legal-search/receipts/WWL-103.json")
TYPED = (
    ("grok_quota_exhausted", "grok_provider_insufficient_quota"),
    ("authentication_failure", "grok_authentication_failure"),
    ("launch_failure", "grok_process_did_not_launch"),
)


def sha(text: str) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def workspace_of(tmp_path: Path) -> Path:
    workspace = tmp_path / "worktrees" / "task-1"
    workspace.mkdir(parents=True)
    (workspace / "seed.txt").write_text("seed\n", encoding="utf-8")
    return workspace


def provider_result(returncode, *, launched=True, activity="unknown"):
    return llm_router.AgentCLIProviderResult(
        returncode,
        launched=launched,
        activity_state=llm_router.AgentCLIActivityState(activity),
    )


def opened_ledger() -> tuple[ProgressLedger, str]:
    ledger = ProgressLedger(RUN)
    decision = ledger.observe({
        "run_id": RUN,
        "task_id": TASK,
        "stage": "task",
        "observed_at": 10,
        "attempt": 1,
        "pid": 10,
        "defect": {"failure_key": DEFECT, "message": "pid=10 attempt=1"},
    })
    return ledger, decision["incident_id"]


def spent(ledger: ProgressLedger, incident_id: str) -> dict[str, int]:
    return dict(ledger.incidents[incident_id]["spent"])


def assert_not_refunded(before: dict[str, int], ledger: ProgressLedger, incident_id: str) -> None:
    current = spent(ledger, incident_id)
    assert current.keys() == before.keys()
    for name, used in before.items():
        assert current[name] >= used
        assert ledger.incidents[incident_id]["limits"][name] - current[name] <= DEFAULT_LIMITS[name] - used


def assert_closed(decision: dict) -> None:
    assert decision["completion_authority"] is False
    assert decision.get("promoted") is not True
    assert decision.get("reloaded") is not True
    assert decision.get("success") is not True
    assert decision.get("budgets_reset") is not True
    assert decision.get("progress_reset") is not True


def route(tmp_path: Path, workspace: Path, kind: str, reason: str, *, activity="no_activity", attempt=1, name="route",
          launched=True, returncode=19, unavailable=None, before=None):
    snapshot = before if before is not None else snapshot_workspace(workspace)
    receipt = trusted_pre_effect_receipt(kind, returncode=returncode if returncode is not None else 19, activity=activity, reason=reason)
    return evaluate_emergency_fallback(
        workspace,
        task_id=TASK,
        attempt=attempt,
        stage="implementation",
        primary_result=provider_result(returncode, launched=launched, activity=activity),
        workspace_before=snapshot,
        telemetry_path=tmp_path / f"{name}.json",
        trusted_failure_receipt=receipt,
        primary_unavailable_kind=unavailable,
    )


class Births:
    def __init__(self) -> None:
        self.rows: dict[int, tuple[bool, int]] = {}

    def set(self, pid: int, *, alive: bool, birth: int) -> None:
        self.rows[pid] = (alive, birth)

    def __call__(self, pid: int) -> dict[str, object]:
        row = self.rows.get(pid)
        if row is None:
            return {"pid": pid, "alive": False, "start_ticks": 0, "state": "missing"}
        alive, birth = row
        return {"pid": pid, "alive": alive, "start_ticks": birth, "state": "R" if alive else "dead"}


def bind_supervisor(owner: RecoveryOwner, births: Births) -> None:
    births.set(10, alive=True, birth=100)
    bound = owner.bind_owner(worker_id="supervisor", pid=10, birth=100, process_generation=2, lease_generation=5)
    assert bound["disposition"] == "owner_bound"
    assert bound["deduplicated"] is not True or owner.process_generation == 2


def recovery_observation(action: str, **extra) -> dict:
    body = {
        "action_id": action,
        "observed_at": 100,
        "deadline_at": 500,
        "incident": {"task_id": TASK, "stage": "task", "defect": {"failure_key": DEFECT, "message": "supervisor dead"}},
        "owner": {
            "worker_id": "supervisor", "pid": 10, "birth": 100,
            "process_generation": 2, "lease_generation": 5,
        },
        "claims": [{
            "claim_id": "claim-supervisor", "task_id": TASK, "worker_id": "supervisor",
            "pid": 10, "birth": 100, "process_generation": 2, "lease_generation": 5,
            "effect_id": "effect-supervisor",
        }],
        "effects": [{
            "effect_id": "effect-supervisor", "task_id": TASK, "status": "unknown", "digest": sha("in-flight"),
        }],
        "completion_authority": False,
    }
    body.update(extra)
    return body


class Gates:
    def __init__(self) -> None:
        self.regressions = 0
        self.canaries = 0
        self.merges = 0
        self.reloads = 0
        self.fail_regression = False
        self.merge_unknown = False
        self.merge_without_receipt = False
        self.nested_request = None
        self.inner = None
        self.session: RepairSession | None = None

    def regression(self, context: dict) -> dict:
        self.regressions += 1
        if self.session is not None and self.regressions == 1 and self.nested_request is not None and self.inner is None:
            self.inner = self.session.repair(self.nested_request)
        passed = not self.fail_regression
        return {
            "schema": REGRESSION_SCHEMA,
            "baseline_failure_verified": passed,
            "candidate_passed": passed,
            "focused_passed": passed,
            "protected_hashes_unchanged": True,
            "exit_status": 0 if passed else 1,
            "argv": list(context["reproducer_argv"]),
            "completion_authority": False,
        }

    def canary(self, context: dict) -> dict:
        self.canaries += 1
        return {
            "schema": CANARY_SCHEMA,
            "baseline_failure_verified": True,
            "candidate_regressions_verified": True,
            "repair_canary_verified": True,
            "validated_promotion": False,
            "paid_provider_calls": 0,
            "hosted_model_invocations": 0,
            "completion_authority": False,
        }

    def merge(self, context: dict) -> dict:
        self.merges += 1
        if self.merge_unknown:
            return {"merged": True, "effects_unknown": True, "exit_status": 0, "completion_authority": False}
        if self.merge_without_receipt:
            return {
                "merged": True,
                "effects_unknown": False,
                "exit_status": 0,
                "merge_commit": "c" * 40,
                "merge_receipt": "",
                "tree": "",
                "completion_authority": False,
            }
        return {
            "merged": True,
            "merge_commit": "b" * 40,
            "merge_receipt": sha("merged"),
            "tree": sha("tree"),
            "effects_unknown": False,
            "completion_authority": False,
        }

    def reload(self, context: dict) -> dict:
        self.reloads += 1
        return {
            "reloaded": True,
            "fenced": True,
            "lane_revision": context["merge"]["merge_commit"],
            "completion_authority": False,
        }


def campaign_plan() -> dict:
    routing = json.loads(json.dumps(CANONICAL_ROUTING))
    return {"schema": "worldwide-legal-search.plan@1", "routing": routing}


def failure(**extra) -> dict:
    body = {
        "schema": "wwl.native-failure@1",
        "run_id": RUN,
        "task_id": TASK,
        "stage": "task",
        "defect": {"failure_key": DEFECT, "message": "pid=10 attempt=1"},
        "effects_unknown": False,
        "source_revision": BASELINE_REV,
        "reproducer": {"path": REPRO, "argv": list(ARGV)},
        "source_frames": [{"path": SOURCE, "source_sha256": sha(BASELINE_TEXT).split(":", 1)[1]}],
        "baseline_files": {SOURCE: BASELINE_TEXT},
        "receipt_id": sha("failure"),
        "pid": 10,
        "attempt": 1,
    }
    body.update(extra)
    return body


def repair_request(tmp_path: Path, action: str, **extra) -> dict:
    campaign = tmp_path / "campaign"
    live = tmp_path / "live"
    campaign.mkdir(parents=True, exist_ok=True)
    live.mkdir(parents=True, exist_ok=True)
    sentinel = live / "sentinel.txt"
    if not sentinel.exists():
        sentinel.write_text("live\n", encoding="utf-8")
    body = {
        "action_id": action,
        "observed_at": 100,
        "deadline_at": 500,
        "failure": failure(),
        "campaign_root": str(campaign),
        "live_checkout": str(live),
        "plan": campaign_plan(),
        "provider": {
            "grok_bin": "/usr/bin/grok",
            "codex_bin": "/usr/bin/codex",
            "codex_model": "deployment-configured-model",
            "codex_reasoning_effort": "high",
        },
        "python_executable": CANONICAL_PYTHON,
        "edits": [
            {"path": REPRO, "role": "reproducer", "text": REPRO_TEXT},
            {"path": SOURCE, "role": "source", "text": FIXED_TEXT},
        ],
        "completion_authority": False,
    }
    body.update(extra)
    return body


def open_repair(gates: Gates | None = None, **kwargs) -> tuple[RepairSession, Gates]:
    gates = gates or Gates()
    session = RepairSession(
        RUN,
        regression_runner=gates.regression,
        canary_runner=gates.canary,
        merger=gates.merge,
        reloader=gates.reload,
        **kwargs,
    )
    gates.session = session
    return session, gates


def blocker_row(decision: dict, task_id: str) -> dict:
    found = [item for item in decision["tasks"] if item["task_id"] == task_id]
    assert len(found) == 1
    return found[0]


@pytest.mark.parametrize(("kind", "reason"), TYPED)
def test_quota_auth_and_unavailable_are_not_success_and_do_not_refund(tmp_path: Path, kind: str, reason: str):
    ledger, incident = opened_ledger()
    charged = ledger.charge(incident, "provider")
    assert charged["charged"] is True and charged["remaining"] == 3
    before = spent(ledger, incident)
    token = ledger.progress_token
    workspace = workspace_of(tmp_path)
    prior = snapshot_workspace(workspace)
    directory = tmp_path / "telemetry"
    directory.mkdir()
    first = evaluate_emergency_fallback(
        workspace,
        task_id=TASK,
        attempt=1,
        stage="implementation",
        primary_result=provider_result(19, activity="no_activity"),
        workspace_before=prior,
        telemetry_path=directory / "attempt-1.json",
        trusted_failure_receipt=trusted_pre_effect_receipt(kind, returncode=19, activity="no_activity", reason=reason),
    )
    second = evaluate_emergency_fallback(
        workspace,
        task_id=TASK,
        attempt=1,
        stage="implementation",
        primary_result=provider_result(19, activity="no_activity"),
        workspace_before=prior,
        telemetry_path=directory / "attempt-1.json",
        trusted_failure_receipt=trusted_pre_effect_receipt(kind, returncode=19, activity="no_activity", reason=reason),
    )
    assert first["decision"] == second["decision"] == "fallback"
    assert first["failure_kind"] == kind
    assert first["route_record"]["route"] == "fallback"
    assert first["route_record"]["side_effects_started"] is False
    assert first["route_record"]["attempt"] == 1
    assert first["effects_ambiguous"] is False
    assert first["paid_provider_calls"] == second["paid_provider_calls"] == 0
    assert first["hosted_model_invocations"] == 0
    assert_closed(first)
    assert_closed(second)
    assert list(directory.iterdir()) == [directory / "attempt-1.json"]
    assert snapshot_workspace(workspace).digest == prior.digest
    noise = ledger.observe({
        "run_id": RUN, "kind": "heartbeat", "heartbeat_at": 80, "pid": 11, "attempt": 1,
    })
    assert noise["meaningful"] is False
    assert noise["progress_token"] == token
    assert noise["budgets_reset"] is False
    assert_not_refunded(before, ledger, incident)
    assert spent(ledger, incident)["provider"] == 1
    with pytest.raises(ProgressIdentityError, match="cannot reset"):
        ledger.observe({"run_id": RUN, "task_id": TASK, "stage": "task", "observed_at": 90, "refund": True})
    assert spent(ledger, incident)["provider"] == 1


@pytest.mark.parametrize(("kind", "reason"), TYPED)
@pytest.mark.parametrize("activity", ("started", "unknown"))
def test_quota_auth_and_unavailable_with_unknown_effects_reconcile(tmp_path: Path, kind: str, reason: str, activity: str):
    ledger, incident = opened_ledger()
    ledger.charge(incident, "task")
    before = spent(ledger, incident)
    workspace = workspace_of(tmp_path)
    decision = route(tmp_path, workspace, kind, reason, activity=activity, name=f"{kind}-{activity}")
    assert decision["decision"] == "reconcile"
    assert decision["route_record"] is None
    assert decision["terminal_reason"] == "side_effects_started"
    assert decision["effects_ambiguous"] is True
    assert decision["side_effects_started"] is True
    assert decision["paid_provider_calls"] == 0
    assert_closed(decision)
    assert_not_refunded(before, ledger, incident)
    assert spent(ledger, incident)["task"] == 1


def test_transport_loss_after_mutation_does_not_fallback_or_refund(tmp_path: Path):
    ledger, incident = opened_ledger()
    ledger.charge(incident, "provider")
    ledger.charge(incident, "task")
    before = spent(ledger, incident)
    token = ledger.progress_token
    workspace = workspace_of(tmp_path)
    prior = snapshot_workspace(workspace)
    (workspace / "seed.txt").write_text("mutated-after-dispatch\n", encoding="utf-8")
    decision = route(
        tmp_path, workspace, "transport_failure", "grok_transport_failure",
        activity="no_activity", name="transport", before=prior,
    )
    assert decision["decision"] == "reconcile"
    assert decision["route_record"] is None
    assert decision["workspace_unchanged"] is False
    assert decision["failure_kind"] == "transport_failure"
    assert decision["paid_provider_calls"] == 0
    assert decision["hosted_model_invocations"] == 0
    assert_closed(decision)
    assert snapshot_workspace(workspace).digest != prior.digest
    denied = evaluate_emergency_fallback(
        workspace,
        task_id=TASK,
        attempt=1,
        stage="implementation",
        primary_result=provider_result(None, launched=False, activity="pre_dispatch"),
        workspace_before=prior,
        telemetry_path=tmp_path / "unavailable-after-mutation.json",
        primary_unavailable_kind="launch_failure",
    )
    assert denied["decision"] == "reconcile"
    assert denied["route_record"] is None
    assert denied["terminal_reason"] == "side_effects_started"
    assert denied["effects_ambiguous"] is True
    assert_closed(denied)
    beat = ledger.observe({"run_id": RUN, "kind": "heartbeat", "heartbeat_at": 500, "pid": 99, "attempt": 1})
    assert beat["meaningful"] is False
    assert beat["progress_token"] == token
    assert beat["last_progress_at"] == 10
    assert_not_refunded(before, ledger, incident)
    files = sorted(path.name for path in tmp_path.glob("*.json"))
    assert files == ["transport.json", "unavailable-after-mutation.json"]


def test_malformed_output_is_not_a_fallback_or_a_success(tmp_path: Path):
    ledger, incident = opened_ledger()
    ledger.charge(incident, "task")
    before = spent(ledger, incident)
    workspace = workspace_of(tmp_path)
    decision = route(
        tmp_path, workspace, "malformed_output", "grok_output_not_valid_text",
        activity="no_activity", name="malformed",
    )
    assert decision["decision"] == "reconcile"
    assert decision["failure_kind"] == "malformed_output"
    assert decision["terminal_reason"] == "failure_not_fallback_eligible"
    assert decision["route_record"] is None
    assert decision["effects_ambiguous"] is False
    assert_closed(decision)
    receipt = trusted_pre_effect_receipt(
        "authentication_failure", returncode=19, activity="no_activity", reason="grok_authentication_failure",
    )
    forged = json.dumps({**json.loads(receipt), "provider_body": "authentication_failure"})
    forged_decision = evaluate_emergency_fallback(
        workspace,
        task_id=TASK,
        attempt=1,
        stage="implementation",
        primary_result=provider_result(19, activity="no_activity"),
        workspace_before=snapshot_workspace(workspace),
        telemetry_path=tmp_path / "forged.json",
        trusted_failure_receipt=forged,
    )
    assert forged_decision["decision"] == "reconcile"
    assert forged_decision["failure_kind"] == "malformed_output"
    assert forged_decision["route_record"] is None
    assert_closed(forged_decision)
    replay = ledger.observe({
        "run_id": RUN, "task_id": TASK, "stage": "task", "observed_at": 40,
        "kind": "retry", "attempt": 1, "pid": 4, "defect": {"failure_key": DEFECT, "message": "attempt=1"},
    })
    assert replay["meaningful"] is False
    assert replay["budgets_reset"] is False
    assert_not_refunded(before, ledger, incident)
    assert spent(ledger, incident)["task"] == 1


def test_dead_supervisor_and_live_heartbeat_do_not_advance_or_refund(tmp_path: Path):
    ledger, incident = opened_ledger()
    ledger.charge(incident, "task")
    births = Births()
    owner = RecoveryOwner(RUN, ledger=ledger, probe=births)
    bind_supervisor(owner, births)
    births.set(10, alive=False, birth=0)
    token = ledger.progress_token
    progress_at = ledger.last_progress_at
    dead = owner.recover(recovery_observation("supervisor-dead"))
    assert dead["disposition"] == "effects_preserved"
    assert dead["dead_worker"] is True
    assert dead["effects_unknown"] is True
    assert dead["effects_preserved"] is True
    assert dead["generation_advanced"] is False
    assert dead["process_generation"] == 2
    assert dead["lease_generation"] == 5
    assert dead["wake_authorized"] is False
    assert dead["applied"] is True
    assert dead["budget"]["spent"] == 1
    assert dead["budget"]["charged"] is True
    assert dead["budget"]["budgets_reset"] is False
    assert_closed(dead)
    assert TASK not in dead["runnable_task_ids"]
    assert owner.owner is not None and owner.owner["fenced"] is True
    assert owner.effects["effect-supervisor"]["status"] == "unknown"
    before = spent(ledger, incident)
    assert before["recover"] == 1
    assert before["task"] == 1
    for index in range(6):
        beat = ledger.observe({
            "run_id": RUN, "kind": "heartbeat", "heartbeat_at": 1_000 + index, "pid": 400 + index, "attempt": 1,
        })
        assert beat["meaningful"] is False
        assert beat["reason"] == "heartbeat"
        assert beat["progress_advanced"] is False
        assert beat["progress_token"] == token
        assert beat["last_progress_at"] == progress_at
        assert beat["incident_id"] == incident
        assert_closed(beat)
    assert ledger.noise["heartbeat"] == 6
    assert spent(ledger, incident) == before
    replay = owner.recover(recovery_observation("supervisor-dead"))
    assert replay["deduplicated"] is True
    assert replay["budget"]["spent"] == 1
    assert spent(ledger, incident)["recover"] == 1
    follow = owner.recover(recovery_observation("supervisor-dead-again", observed_at=120))
    assert follow["budget"]["charged"] is False
    assert follow["generation_advanced"] is False
    assert owner.effects["effect-supervisor"]["status"] == "unknown"
    assert spent(ledger, incident)["recover"] == 1
    births.set(12, alive=True, birth=500)
    with pytest.raises(RecoveryContractError, match="unknown effects"):
        owner.bind_owner(worker_id="replacement", pid=12, birth=500, process_generation=3, lease_generation=6)
    with pytest.raises(RecoveryContractError, match="cannot reset"):
        owner.recover(recovery_observation("refund-recovery", refund=True))
    assert spent(ledger, incident) == before
    snapshot = owner.to_dict()
    tampered = json.loads(json.dumps(snapshot))
    tampered["spent"][incident] = 0
    with pytest.raises(RecoveryContractError, match="diverged"):
        RecoveryOwner.from_dict(tampered, probe=births)
    restored = RecoveryOwner.from_dict(json.loads(json.dumps(snapshot)), probe=births)
    assert restored.spent[incident] == 1
    assert restored.ledger is not None
    assert restored.ledger.incidents[incident]["spent"]["recover"] == 1
    assert restored.ledger.progress_token == token

    repository, state = tmp_path / "repo", tmp_path / "state"
    repository.mkdir()
    state.mkdir()
    readiness = {
        "schema": "wwl.watchdog.ready@1", "run_id": RUN, "repository": str(repository),
        "policy_id": "wwl-policy", "native_revision": "accepted-native-revision",
        "admitted_plan_revision": "plan-1", "required_task_ids": [TASK],
        "readiness_receipt": "native:fault-evidence",
        "bindings": {"task_source": True, "meaningful_progress": True, "reconcile": True, "lifecycle_restart": True, "repair_dispatch": False},
    }
    status = {
        "schema": "wwl.watchdog.snapshot@1", "run_id": RUN, "observed_at": 10040,
        "authority_revision": "task-source-1", "accepted_tree": "accepted-tree-1", "hold": False,
        "admitted_plan_revision": "plan-1", "heartbeat_at": 10050, "progress_token": token,
        "owner_pid": 4242, "completion_verified": True, "completion_receipt": "forged-success",
    }
    (state / "campaign-readiness.json").write_text(json.dumps(readiness), encoding="utf-8")
    (state / "campaign-status.json").write_text(json.dumps(status), encoding="utf-8")

    class DeadSupervisor:
        def probe(self) -> dict:
            return {
                "unit": "wwl.service", "registered": True, "main_pid": 0, "start_ticks": 0,
                "owner_alive": False, "owner_identity_verified": True, "control_group": "",
                "active_state": "inactive", "fresh_owner_projection": False, "observed_at": 10050,
            }

    bridge = WorldwideWatchdogBridge(
        repository=repository, state_root=state, run_id=RUN, clock=lambda: 10050, host_lifecycle=DeadSupervisor(),
    )
    observed = bridge.observe({"policy_id": "wwl-policy", "run_id": RUN})
    assert observed["owner_alive"] is False
    assert observed["heartbeat_at"] == 10050
    assert observed["progress_token"] == token
    assert observed["completion_verified"] is False
    assert observed["completion_receipt"] == ""
    assert not (state / "control" / "requests").exists()
    stored = json.loads((state / "campaign-status.json").read_text(encoding="utf-8"))
    assert stored["progress_token"] == token
    assert stored["heartbeat_at"] == 10050
    again = bridge.observe({"policy_id": "wwl-policy", "run_id": RUN})
    assert again["owner_alive"] is False
    assert again["completion_verified"] is False
    assert again["progress_token"] == token


def test_live_heartbeat_freezes_progress_across_repeated_attempts():
    ledger, incident = opened_ledger()
    ledger.charge(incident, "task")
    ledger.charge(incident, "repair")
    token = ledger.progress_token
    progress_at = ledger.last_progress_at
    before = spent(ledger, incident)
    for attempt in range(1, 8):
        beat = ledger.observe({
            "run_id": RUN, "kind": "heartbeat", "heartbeat_at": 100 + attempt, "pid": 20 + attempt,
        })
        retry = ledger.observe({
            "run_id": RUN, "task_id": TASK, "stage": "task", "observed_at": 200 + attempt,
            "kind": "same_stage_retry", "attempt": attempt, "pid": 30 + attempt,
            "defect": {"failure_key": DEFECT, "message": f"pid={30 + attempt} attempt={attempt}"},
            "log": "still failing", "log_bytes": attempt,
        })
        assert beat["meaningful"] is False and retry["meaningful"] is False
        assert beat["progress_advanced"] is False and retry["progress_advanced"] is False
        assert retry["same_stage_retry"] is True
        assert beat["progress_token"] == retry["progress_token"] == token
        assert beat["last_progress_at"] == retry["last_progress_at"] == progress_at
        assert beat["incident_id"] == retry["incident_id"] == incident
        assert_closed(beat)
        assert_closed(retry)
    rewound = ledger.observe({
        "run_id": RUN, "task_id": TASK, "stage": "task", "observed_at": 400,
        "kind": "retry", "attempt": 1, "pid": 10,
        "defect": {"failure_key": DEFECT, "message": "attempt=1"},
    })
    assert rewound["meaningful"] is False
    assert rewound["budgets_reset"] is False
    assert ledger.attempt == 1
    assert spent(ledger, incident) == before
    assert ledger.progress_token == token
    assert ledger.last_progress_at == progress_at
    with pytest.raises(ProgressIdentityError, match="cannot reset"):
        ledger.observe({
            "run_id": RUN, "task_id": TASK, "stage": "task", "observed_at": 410,
            "reset_budget": True, "attempt": 0,
        })
    with pytest.raises(ProgressIdentityError, match="completion"):
        ledger.observe({
            "run_id": RUN, "task_id": TASK, "stage": "task", "observed_at": 420,
            "completion_authority": True,
        })
    assert spent(ledger, incident) == before
    saved = ledger.to_dict()
    restored = ProgressLedger.from_dict(json.loads(json.dumps(saved)))
    assert restored.incidents[incident]["spent"] == before
    assert restored.progress_token == token
    widened = json.loads(json.dumps(saved))
    widened["incidents"][0]["spent"]["task"] = 0
    widened["incidents"][0]["spent"]["repair"] = 0
    loaded = ProgressLedger.from_dict(widened)
    assert loaded.incidents[incident]["spent"]["task"] == 0
    rebound = ProgressLedger.from_dict(json.loads(json.dumps(saved)))
    follow = rebound.observe({
        "run_id": RUN, "kind": "heartbeat", "heartbeat_at": 900, "pid": 70,
    })
    assert follow["meaningful"] is False
    assert rebound.incidents[incident]["spent"] == before


def test_repeated_repair_defects_do_not_converge_or_duplicate_a_writer(tmp_path: Path):
    ledger, incident = opened_ledger()
    assert incident == repair_task_identity(run_id=RUN, task_id=TASK, stage="task", defect={"failure_key": DEFECT})
    gates = Gates()
    gates.fail_regression = True
    session, gates = open_repair(gates, ledger=ledger)
    gates.nested_request = repair_request(tmp_path, "action-nested", failure=failure(pid=11, attempt=2))
    token = ledger.progress_token
    first = session.repair(repair_request(tmp_path, "action-1", failure=failure(pid=10, attempt=1)))
    assert gates.inner is not None
    assert gates.inner["disposition"] == "duplicate_writer"
    assert gates.inner["duplicate_writer"] is True
    assert gates.inner["budget"]["repair"]["charged"] is False
    assert gates.inner["budget"]["repair"]["spent"] == 1
    assert_closed(gates.inner)
    assert first["disposition"] == "regressions_failed"
    assert first["repair_task_id"] == gates.inner["repair_task_id"]
    assert first["promoted"] is False
    assert first["applied"] is True
    assert first["route"]["launched"] is False
    assert first["route"]["paid_provider_calls"] == 0
    assert first["route"]["hosted_model_invocations"] == 0
    assert first["budget"]["repair"]["spent"] == 1
    assert_closed(first)
    assert ledger.incidents[incident]["spent"]["repair"] == 1
    replay = session.repair(repair_request(tmp_path, "action-1", failure=failure(pid=99, attempt=9)))
    assert replay["deduplicated"] is True
    assert replay["disposition"] == "regressions_failed"
    assert ledger.incidents[incident]["spent"]["repair"] == 1
    second = session.repair(repair_request(tmp_path, "action-2", failure=failure(pid=12, attempt=3)))
    assert second["disposition"] == "regressions_failed"
    assert second["repair_task_id"] == first["repair_task_id"]
    assert second["promoted"] is False
    assert second["budget"]["repair"]["spent"] == REPAIR_LIMIT == 2
    assert second["budget"]["repair"]["charged"] is True
    assert_closed(second)
    assert gates.regressions == 2
    assert gates.merges == 0
    assert gates.reloads == 0
    blocked = session.repair(repair_request(tmp_path, "action-3", failure=failure(pid=13, attempt=4)))
    assert blocked["disposition"] == "repair_budget_exhausted"
    assert blocked["duplicate_writer"] is False
    assert blocked["budget"]["repair"]["spent"] == 2
    assert blocked["budget"]["repair"]["charged"] is False
    assert blocked["budget"]["repair"]["remaining"] == 0
    assert_closed(blocked)
    assert gates.regressions == 2
    assert session.campaign_repairs == 2
    writers = list((tmp_path / "campaign" / "repair-worktrees").iterdir())
    assert len(writers) == 1
    assert writers[0].name == first["repair_task_id"]
    assert (tmp_path / "live" / "sentinel.txt").read_text(encoding="utf-8") == "live\n"
    for index in range(4):
        beat = ledger.observe({"run_id": RUN, "kind": "heartbeat", "heartbeat_at": 300 + index, "pid": 80 + index})
        assert beat["meaningful"] is False
        assert beat["progress_token"] == token
    assert ledger.incidents[incident]["spent"]["repair"] == 2
    assert ledger.incidents[incident]["spent"]["provider"] == 2
    assert ledger.incidents[incident]["spent"]["validation"] == 2
    assert ledger.incidents[incident]["spent"]["merge"] == 0
    with pytest.raises(RepairContractError, match="cannot reset"):
        session.repair(repair_request(tmp_path, "action-refund", refund=True))
    checkpoint = tmp_path / "repair-state.json"
    session.save(checkpoint)
    raw = json.loads(checkpoint.read_text(encoding="utf-8"))
    raw["spent"]["repair"][incident] = 0
    checkpoint.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(RepairContractError, match="rewritten"):
        RepairSession.load(
            checkpoint,
            regression_runner=gates.regression,
            canary_runner=gates.canary,
            merger=gates.merge,
            reloader=gates.reload,
        )
    assert ledger.incidents[incident]["spent"]["repair"] == 2

    board = BlockerBoard(RUN, ledger=ledger)
    body = {"failure_key": DEFECT, "case": "unknown-effects"}
    receipt = reproducer_receipt(body)
    unknown_task = {
        "task_id": TASK, "status": "blocked", "depends_on": [], "effects_unknown": True,
        "failure_key": DEFECT, "reproducer": body, "reproducer_receipt": receipt,
        "incident": {"stage": "task", "defect": {"failure_key": DEFECT}},
    }
    ready = {"task_id": FREE, "status": "ready", "depends_on": []}
    unknown = board.classify({"observed_at": 20, "tasks": [unknown_task, ready]})
    unknown_row = blocker_row(unknown, TASK)
    assert unknown_row["classification"] == "effects-unknown"
    assert unknown_row["failure"] is False
    assert unknown_row["repair_admitted"] is False
    assert unknown_row["success"] is False
    assert unknown_row["continues"] is False
    assert "software-defect" in unknown_row["matched_kinds"]
    assert unknown["continue_task_ids"] == [FREE]
    frozen_unknown = board.classify({
        "observed_at": 30, "heartbeat_at": 999, "pid": 70, "attempt": 9,
        "tasks": [unknown_task, ready],
    })
    frozen_unknown_row = blocker_row(frozen_unknown, TASK)
    assert frozen_unknown_row["classification"] == "effects-unknown"
    assert frozen_unknown_row["recheck_admitted"] is False
    assert frozen_unknown_row["recheck_reason"] == "prerequisite_unchanged"
    assert frozen_unknown_row["success"] is False
    assert frozen_unknown_row["repair_admitted"] is False
    assert frozen_unknown["continue_task_ids"] == [FREE]
    exhausted_task = {
        "task_id": TASK, "status": "blocked", "depends_on": [],
        "incident": {"stage": "task", "defect": {"failure_key": DEFECT}},
    }
    exhausted = board.classify({"observed_at": 40, "tasks": [exhausted_task, ready]})
    exhausted_row = blocker_row(exhausted, TASK)
    assert exhausted_row["classification"] == "exhausted-budget"
    assert exhausted_row["exhausted_budgets"] == ["repair"]
    assert exhausted_row["success"] is False
    assert exhausted_row["repair_admitted"] is False
    assert exhausted_row["recheck_admitted"] is True
    assert exhausted_row["recheck_reason"] == "prerequisite_changed"
    assert exhausted["continue_task_ids"] == [FREE]
    frozen = board.classify({
        "observed_at": 50, "heartbeat_at": 1000, "pid": 71, "attempt": 9,
        "tasks": [exhausted_task, ready],
    })
    frozen_row = blocker_row(frozen, TASK)
    assert frozen_row["classification"] == "exhausted-budget"
    assert frozen_row["exhausted_budgets"] == ["repair"]
    assert frozen_row["recheck_admitted"] is False
    assert frozen_row["success"] is False
    assert frozen_row["repair_admitted"] is False
    assert frozen["continue_task_ids"] == [FREE]
    assert ledger.incidents[incident]["spent"]["repair"] == 2
    from ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_blockers import BlockerContractError
    with pytest.raises(BlockerContractError, match="progress ledger"):
        board.classify({"observed_at": 60, "tasks": [
            {
                "task_id": TASK, "status": "blocked", "depends_on": [],
                "incident": {"stage": "task", "defect": {"failure_key": DEFECT}},
                "budgets": {"repair": {"spent": 0, "limit": REPAIR_LIMIT}},
            },
            {"task_id": FREE, "status": "ready", "depends_on": []},
        ]})
    assert ledger.incidents[incident]["spent"]["repair"] == 2


def test_unknown_merge_and_a_bare_exit_are_not_promotion(tmp_path: Path):
    ledger, incident = opened_ledger()
    gates = Gates()
    gates.merge_unknown = True
    session, gates = open_repair(gates, ledger=ledger)
    unknown = session.repair(repair_request(tmp_path, "merge-unknown"))
    assert unknown["disposition"] == "effects_unknown"
    assert unknown["promoted"] is False
    assert unknown["reloaded"] is False
    assert unknown["fenced"] is True
    assert gates.reloads == 0
    assert unknown["budget"]["merge"]["spent"] == 1
    assert unknown["budget"]["repair"]["spent"] == 1
    assert_closed(unknown)
    again = session.repair(repair_request(tmp_path / "second", "merge-unknown-2"))
    assert again["disposition"] == "effects_unknown"
    assert again["promoted"] is False
    assert again["repair_task_id"] == unknown["repair_task_id"]
    assert again["budget"]["repair"]["spent"] == 2
    assert ledger.incidents[incident]["spent"]["repair"] == 2
    blocked = session.repair(repair_request(tmp_path / "third", "merge-unknown-3"))
    assert blocked["disposition"] == "repair_budget_exhausted"
    assert blocked["budget"]["repair"]["charged"] is False
    assert ledger.incidents[incident]["spent"]["repair"] == 2
    beat = ledger.observe({"run_id": RUN, "kind": "heartbeat", "heartbeat_at": 700, "attempt": 1})
    assert beat["meaningful"] is False
    assert ledger.incidents[incident]["spent"]["repair"] == 2

    bare_ledger, bare_incident = opened_ledger()
    bare_gates = Gates()
    bare_gates.merge_without_receipt = True
    bare, bare_gates = open_repair(bare_gates, ledger=bare_ledger)
    refused = bare.repair(repair_request(tmp_path / "bare", "bare-exit"))
    assert refused["disposition"] == "promotion_refused"
    assert refused["promoted"] is False
    assert refused["reloaded"] is False
    assert bare_gates.reloads == 0
    assert refused["budget"]["repair"]["spent"] == 1
    assert refused["budget"]["merge"]["charged"] is True
    assert_closed(refused)
    assert bare_ledger.incidents[bare_incident]["spent"]["repair"] == 1
    assert (tmp_path / "bare" / "live" / "sentinel.txt").read_text(encoding="utf-8") == "live\n"


def test_receipt_records_the_fault_injections_without_completion_authority():
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["schema"] == "worldwide-legal-search.task-receipt@1"
    assert payload["task_id"] == "WWL-103"
    assert payload["title"] == "Test quota fallback unknown effects and repair nonconvergence"
    assert payload["namespace"] == "worldwide-legal-search-v1"
    assert payload["root_goal_id"] == "WWL-G000"
    assert payload["goal_id"] == "WWL-G090"
    assert payload["subgoal_id"] == "WWL-G090.S2"
    assert payload["depends_on"] == ["WWL-102"]
    assert payload["fallback_policy"] == ROUTE_POLICY == "grok_quota_auth_or_unavailable"
    assert payload["repair_limit"] == REPAIR_LIMIT == DEFAULT_LIMITS["repair"] == 2
    assert payload["recover_limit"] == DEFAULT_LIMITS["recover"] == 2
    assert payload["injected_faults"] == [
        "quota",
        "auth",
        "unavailable",
        "transport_loss_after_mutation",
        "malformed_output",
        "dead_supervisor",
        "live_heartbeat_frozen_progress",
        "repeated_repair_defects",
    ]
    assert payload["invariants"] == [
        "no duplicate writer",
        "no false success",
        "no attempt refund",
    ]
    assert payload["execution_completed"] is False
    assert payload["completion_authority"] is False
    assert payload["self_certified_completion"] is False
    assert payload["outputs"] == [
        "test/api/test_worldwide_legal_search_faults.py",
        "artifacts/worldwide-legal-search/receipts/WWL-103.json",
    ]
    assert payload["validation_argv"] == ["python3", "-m", "pytest", "test/api/test_worldwide_legal_search_faults.py", "-q"]
    text = "\n".join(payload["limitations"])
    assert "duplicate writer" in text
    assert "false success" in text
    assert "refund" in text
    assert "/usr/bin/python3.12" in text
    assert "does not grant campaign completion" in text
    with pytest.raises(WorldwideRouteError):
        evaluate_emergency_fallback(
            Path("."),
            task_id=TASK,
            attempt=0,
            stage="implementation",
            primary_result=provider_result(1),
            workspace_before=llm_router.AgentCLIWorkspaceSnapshot("", False, "workspace_snapshot_unavailable"),
            telemetry_path=Path("unused.json"),
        )
