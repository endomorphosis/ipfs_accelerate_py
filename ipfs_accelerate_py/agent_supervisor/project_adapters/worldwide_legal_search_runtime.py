"""Exact-plan worldwide campaign on the native database/Portal execution path.

The process owns its embedded task store. Monitoring reads atomic projections
and submits durable requests; it never opens the live DuckDB file. Provider,
effect, validation and completion remain the existing native daemon gates.
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from types import SimpleNamespace
from typing import Any, Mapping

MODULE = "ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_runtime"
COMPLETED = {"completed", "complete", "done"}


def identity(value: Any) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if path.stat().st_size > 4_194_304:
        raise ValueError("campaign JSON exceeds its size bound")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("campaign JSON must be an object")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".wwl-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, ensure_ascii=False, default=str)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def git(root: Path, *argv: str) -> str:
    result = subprocess.run(["git", "-C", str(root), *argv], capture_output=True, text=True, timeout=30, check=False)
    if result.returncode:
        raise ValueError("Git observation failed: " + " ".join(argv[:2]))
    return result.stdout.strip()


@dataclass(frozen=True)
class CampaignConfig:
    site_root: Path
    framework_root: Path
    state_root: Path
    plan_path: Path
    config_path: Path
    python_executable: str = sys.executable
    implementation_command: str = ""
    run_id: str = "worldwide-legal-search-v1"
    poll_seconds: float = 5.0
    task_timeout_seconds: int = 5400
    max_task_attempts: int = 3
    repair_command: tuple[str, ...] = ()

    @property
    def database(self) -> Path:
        return self.state_root / "control.duckdb"

    @property
    def objective_path(self) -> Path:
        explicit = read_json(self.config_path).get("objectives_path") if self.config_path.is_file() else None
        return Path(explicit) if explicit else self.plan_path.with_name("goals.objectives.md")

    def serializable(self) -> dict[str, Any]:
        return {key: str(value) if isinstance(value, Path) else list(value) if isinstance(value, tuple) else value for key, value in asdict(self).items()}


def config_from_args(args: argparse.Namespace) -> CampaignConfig:
    raw = read_json(args.config) if args.config else {}
    for key in ("site_root", "framework_root", "state_root", "plan_path", "implementation_command", "python_executable", "run_id"):
        value = getattr(args, key, None)
        if value:
            raw[key] = str(value)
    required = ("site_root", "framework_root", "state_root")
    if any(not raw.get(key) for key in required):
        raise ValueError("site_root, framework_root and state_root are required")
    site, framework, state = (Path(raw[key]).expanduser().resolve() for key in required)
    for repository in (site, framework):
        if not repository.is_dir() or git(repository, "rev-parse", "--show-toplevel") != str(repository):
            raise ValueError("campaign repository must be an exact existing Git root")
        if state == repository or repository in state.parents:
            raise ValueError("native state must remain outside campaign repositories")
    path = Path(raw.get("plan_path") or site / "implementation_plan/worldwide_legal_search/plan.json").resolve()
    config_path = args.config.resolve() if args.config else state / "runtime-config.json"
    config = CampaignConfig(site, framework, state, path, config_path,
        python_executable=str(raw.get("python_executable") or sys.executable),
        implementation_command=str(raw.get("implementation_command") or ""),
        run_id=str(raw.get("run_id") or "worldwide-legal-search-v1"),
        poll_seconds=float(raw.get("poll_seconds", 5)), task_timeout_seconds=int(raw.get("task_timeout_seconds", 5400)),
        max_task_attempts=int(raw.get("max_task_attempts", 3)), repair_command=tuple(raw.get("repair_command") or ()))
    if not config.run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for c in config.run_id):
        raise ValueError("campaign run ID is not a safe identifier")
    return config


def topological_tasks(plan: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    pending = {str(task["id"]): task for task in plan["tasks"]}
    if len(pending) != len(plan["tasks"]):
        raise ValueError("duplicate campaign task IDs")
    ordered, finished = [], set()
    while pending:
        ready = [task for task in pending.values() if set(task["depends_on"]) <= finished]
        if not ready:
            raise ValueError("campaign dependencies are cyclic or missing")
        for task in ready:
            ordered.append(task)
            finished.add(str(task["id"]))
            del pending[str(task["id"])]
    return ordered


def build_population(plan: Mapping[str, Any], *, source_forest: Mapping[str, Any]) -> dict[str, Any]:
    if plan.get("schema") != "worldwide-legal-search.plan@1":
        raise ValueError("unsupported worldwide plan")
    root = identity({"plan": plan, "source_forest": source_forest})
    tree = identity(source_forest)
    goals = list(plan["goals"])
    goal_cids = {goal["id"]: identity({"root": root, "goal": goal}) for goal in goals}
    task_cids = {task["id"]: identity({"root": root, "task": task}) for task in plan["tasks"]}
    objects = [{"goal_cid": goal_cids[g["id"]], "goal_id": g["id"], "goal_alias": g["id"], "title": g["title"],
                "objective_id": "objective:" + str(plan["namespace"]), "status": "active", "ordinal": n,
                "parent_goal_cid": goal_cids[g["parent_ids"][0]] if g["parent_ids"] else "", "declaration": dict(g)} for n, g in enumerate(goals, 1)]
    edges = [{"parent_goal_cid": goal_cids[parent], "child_goal_cid": goal_cids[g["id"]], "edge_kind": "goal_parent"} for g in goals for parent in g["parent_ids"]]
    tasks = []
    for ordinal, task in enumerate(topological_tasks(plan), 1):
        outputs = [*task["outputs"], task["evidence_output"]]
        validations = [shlex.join(argv) for argv in task["validation_argv"]]
        tasks.append({"task_cid": task_cids[task["id"]], "task_id": task["id"], "task_alias": task["id"], "title": task["title"],
            "goal_cid": goal_cids[task["subgoal_id"]], "goal_id": task["subgoal_id"] , "plan_cid": root,
            "objective_id": "objective:" + str(plan["namespace"]), "ordinal": ordinal, "status": "todo", "priority": "P1",
            "depends_on": [task_cids[alias] for alias in task["depends_on"]],
            "outputs": [{"path": path, "effect_id": identity({"task_cid": task_cids[task["id"]], "path": path})} for path in outputs],
            "acceptance": [task["acceptance"]], "validations": validations, "validation": " && ".join(validations),
            "completion": "auto", "board_namespace": plan["namespace"], "owning_repository": task["repository"],
            "source_plan_task": dict(task), "accepted_plan_root_cid": root, "base_repository_tree_id": tree,
            "base_revision": source_forest[task["repository"]]["head"],
            "implementation_timeout_seconds": task["bounds"]["wall_seconds_per_attempt"]})
    return {"schema": "worldwide-legal-search.native-population@1", "repository_tree_id": tree, "plan_root_cid": root,
            "source_forest": dict(source_forest), "objectives": objects, "goal_edges": edges,
            "plans": [{"plan_cid": root, "plan_alias": plan["namespace"], "goal_cid": goal_cids["WWL-G000"], "status": "active"}],
            "tasks": tasks, "task_cids_by_alias": task_cids, "goal_cids_by_alias": goal_cids}


def observed_forest(config: CampaignConfig) -> dict[str, Any]:
    return {key: {"path": str(root), "head": git(root, "rev-parse", "HEAD"), "tree": git(root, "rev-parse", "HEAD^{tree}")}
            for key, root in (("site", config.site_root), ("accelerate", config.framework_root))}


@contextlib.contextmanager
def owner_lock(config: CampaignConfig):
    config.state_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (config.state_root / "native-owner.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("native campaign state already has a live owner") from exc
        yield


def bootstrap(config: CampaignConfig, *, already_locked: bool = False) -> dict[str, Any]:
    from ..task_sources.database_task_source import DatabaseTaskSource
    from .worldwide_legal_search_goals import admit_typed_goals

    with contextlib.nullcontext() if already_locked else owner_lock(config):
        if not config.config_path.is_file():
            atomic_json(config.config_path, config.serializable())
        plan = read_json(config.plan_path)
        population = build_population(plan, source_forest=observed_forest(config))
        receipt_path = config.state_root / "bootstrap-receipt.json"
        if config.database.exists():
            prior = read_json(receipt_path)
            if prior["plan_definition_id"] != identity(plan):
                raise ValueError("existing native store belongs to a different plan definition")
            # Restart reuses exact admitted identities, not a new HEAD-derived
            # population that would reset progress after a validated merge.
            return {**prior, "idempotent_replay": True}
        typed = admit_typed_goals(plan, config.objective_path.read_text(), config_path=config.config_path, python_executable=config.python_executable)
        atomic_json(config.state_root / "typed-goals.json", typed)
        with DatabaseTaskSource(config.database, owner_id=config.run_id + ":bootstrap", repository_tree_id=population["repository_tree_id"], plan_root_cid=population["plan_root_cid"]) as source:
            materialization = dict(source.materialize(population))
            snapshot = source.snapshot().to_dict()
            ready = [task.task_alias for task in source.ready_tasks(limit=256).tasks]
        expected = [task["id"] for task in plan["tasks"] if not task["depends_on"]]
        if ready != expected or snapshot["task_count"] != len(plan["tasks"]) or snapshot["goal_count"] != len(plan["goals"]):
            raise ValueError("native materialization lost the exact task/goal population or readiness frontier")
        receipt = {"schema": "worldwide-legal-search.native-bootstrap@1", "run_id": config.run_id, "plan_definition_id": identity(plan),
            "admitted_plan_revision": population["plan_root_cid"], "source_forest": population["source_forest"],
            "native_materialization": materialization, "snapshot": snapshot, "ready_task_ids": ready, "typed_goals_accepted": typed["all_accepted"],
            "created_at": time.time(), "execution_completed": False}
        receipt["receipt_id"] = identity(receipt)
        atomic_json(receipt_path, receipt)
        return receipt


class CampaignRuntime:
    def __init__(self, config: CampaignConfig, bootstrap_receipt: Mapping[str, Any]):
        self.config, self.bootstrap_receipt = config, dict(bootstrap_receipt)
        self.plan = read_json(config.plan_path)
        self.tasks = {t["id"]: t for t in self.plan["tasks"]}
        self.stop, self.cancel = threading.Event(), threading.Event()
        self._cache_lock = threading.RLock()
        self._cached: dict[str, Any] = {}
        self._actions: dict[str, dict[str, Any]] = {}
        ledger = config.state_root / "control/action-ledger.json"
        if ledger.is_file():
            self._actions = read_json(ledger)
        self._last_failure: dict[str, Any] = {}
        self._last_result: dict[str, Any] = {}
        self._worker: threading.Thread | None = None
        self._daemon: Any = None
        self._progress_at = time.time()
        self._progress_token = ""
        self._passes = 0
        self._provider_callback_entered = False
        self._retained_attempt_before_pass = False
        self._pending_controls: list[str] = []
        self._repair_thread: threading.Thread | None = None

    def repository_for_task(self, alias: str) -> Path:
        task = self.tasks.get(alias)
        if task is None:
            raise ValueError("unknown campaign task alias")
        if task["repository"] == "site":
            return self.config.site_root
        if task["repository"] == "accelerate":
            return self.config.framework_root
        raise ValueError("read-only reference cannot own an execution task")

    def open(self) -> None:
        from ..todo_daemon.implementation_daemon import DatabaseImplementationDaemon, PortalImplementationDaemon, parse_args
        from ..todo_daemon.implementation_daemon_runner import bind_database_portal_execution_from_args

        self._daemon = DatabaseImplementationDaemon(database_path=self.config.database,
            coordination_path=self.config.state_root / "coordination.duckdb", execution_path=self.config.state_root / "execution.duckdb",
            state_dir=self.config.state_root / "native", state_prefix="wwl", owner_session_id=self.config.run_id,
            authority_mode="embedded_exclusive", require_real_execution=True, max_task_attempts=self.config.max_task_attempts,
            task_prefix="WWL-").open()
        command = self.config.implementation_command
        if not command:
            from .worldwide_legal_search_provider import implementation_command
            command = implementation_command(self.config.python_executable, self.config.config_path)
        args = parse_args(["--implement", "--state-dir", str(self.config.state_root / "native"), "--state-prefix", "wwl",
            "--task-prefix", "## WWL-", "--board-namespace", self.plan["namespace"], "--implementation-command", command,
            "--worktree-root", str(self.config.state_root / "worktrees"), "--merge-queue-dir", str(self.config.state_root / "merge-queue"),
            "--merge-target-branch", git(self.config.site_root, "symbolic-ref", "--short", "HEAD"),
            "--max-task-attempts", str(self.config.max_task_attempts), "--implementation-timeout", str(self.config.task_timeout_seconds)])
        for protected in self.plan["execution"]["protected_acceptance_inputs"]:
            args.implementation_protected_path.append(protected)
        parent = self

        class CampaignPortal(PortalImplementationDaemon):
            def __init__(self, **kwargs: Any):
                kwargs["implementation_cancelled"] = parent.cancel.is_set
                super().__init__(**kwargs)

        captured: dict[str, Any] = {}
        proxy = SimpleNamespace(task_source=self._daemon.task_source,
            authorize_superseded_portal_attempt_binding=self._daemon.authorize_superseded_portal_attempt_binding,
            bind_execution_callbacks=lambda **callbacks: captured.update(callbacks))
        bridge = bind_database_portal_execution_from_args(proxy, args, repo_root=self.config.site_root, portal_daemon_class=CampaignPortal,
            task_repo_root_resolver=self.repository_for_task)
        self._bridge = bridge
        callbacks = _ProgressCallbacks(self, bridge)
        self._daemon.bind_execution_callbacks(provider_fn=callbacks.run_provider, effect_fn=callbacks.apply_effect, validation_fn=callbacks.validate_effect)

    def capture(self, stage: str = "", attempt: Any = None) -> None:
        """Called only by the native DB owner worker, never the watcher."""
        source = self._daemon.task_source
        snapshot = source.snapshot().to_dict()
        records = source.list_tasks(limit=256).tasks
        evidence = dict(source.intent.completion_evidence_projection())
        receipt_cids = {item["task_cid"]: item["receipt_cid"] for item in evidence["completion_receipts"]}
        verified = [task.task_alias for task in records if task.status in COMPLETED and task.task_cid in receipt_cids]
        ready = [task.task_alias for task in source.ready_tasks(limit=256).tasks]
        running = self._daemon.list_running_attempts()
        current = attempt or (running[0] if running else None)
        active = str(getattr(current, "task_alias", "") or "")
        phase = stage or str(getattr(current, "committed_phase", "") or "idle")
        signature = {"tasks": [(r.task_alias, r.status) for r in records], "verified": verified,
                     "active": active, "phase": phase, "receipts": receipt_cids}
        token = identity(signature)
        if token != self._progress_token:
            self._progress_token, self._progress_at = token, time.time()
        with self._cache_lock:
            self._cached = {"authority_revision": str(snapshot["revision"]), "progress_token": token,
                "task_statuses": {r.task_alias: r.status for r in records}, "ready_task_ids": ready,
                "verified_task_ids": verified, "native_completion_receipts": receipt_cids,
                "native_completion_receipts_by_alias": {task.task_alias: receipt_cids[task.task_cid] for task in records if task.task_alias in verified},
                "native_completion_projection": evidence["projection_cid"], "active_task_id": active,
                "active_task_cid": str(getattr(current, "task_cid", "") or ""), "stage": phase,
                "active_attempt_id": str(getattr(current, "attempt_id", "") or ""),
                "stage_deadline_at": (int(getattr(current, "started_at_ms", 0)) / 1000 + self.config.task_timeout_seconds) if current else 0,
                "source_snapshot": snapshot}

    def publish(self, *, alive: bool = True) -> dict[str, Any]:
        now = time.time()
        with self._cache_lock:
            cached = dict(self._cached)
        failure = self._last_failure
        hold = any((self.config.state_root / name).exists() for name in ("HOLD", "STOP", "OPERATOR_STOP"))
        verified = list(cached.get("verified_task_ids", []))
        required = list(self.tasks)
        # Root completion must eventually come from the native typed-goal
        # proof validator. A host-written boolean receipt cannot grant it.
        completed = False
        completion_receipt = ""
        snapshot = {"schema": "wwl.watchdog.snapshot@1", "run_id": self.config.run_id, "observed_at": now,
            "authority_revision": str(cached.get("authority_revision") or "bootstrap"),
            "admitted_plan_revision": self.bootstrap_receipt["admitted_plan_revision"],
            "accepted_tree": identity(observed_forest(self.config)), "hold": hold,
            "completion_verified": completed, "completion_receipt": completion_receipt,
            "required_task_ids": required, "verified_task_ids": verified,
            "stage": str(cached.get("stage") or "startup"), "task_id": str(cached.get("active_task_id") or ""),
            "progress_token": str(cached.get("progress_token") or self.bootstrap_receipt["receipt_id"]),
            "heartbeat_at": now, "owner_alive": alive, "owner_identity_verified": alive,
            "runnable_count": len(cached.get("ready_task_ids", [])), "blocked_kind": "software_defect" if failure else "none",
            "dependency_running": False, "stage_deadline_at": cached.get("stage_deadline_at", 0),
            "effects_unknown": bool(failure.get("effects_unknown", False)), "failure_key": failure.get("failure_key", ""),
            "reproducer_receipt": failure.get("receipt_id", ""),
            "action_observations": [value["observation"] for value in self._actions.values() if value.get("observation")],
            "owner_pid": os.getpid(), "meaningful_progress_at_ms": int(self._progress_at * 1000),
            "ready_task_ids": cached.get("ready_task_ids", []), "task_statuses": cached.get("task_statuses", {}),
            "native_completion_receipts": cached.get("native_completion_receipts", {}), "native_completion_projection": cached.get("native_completion_projection", ""),
            "native_completion_receipts_by_alias": cached.get("native_completion_receipts_by_alias", {}),
            "native_error": failure, "native_passes": self._passes, "completion_authority": False}
        for name in ("campaign-status.json", "status.json"):
            atomic_json(self.config.state_root / name, snapshot)
        return snapshot

    def publish_readiness(self) -> None:
        enabled_repair = bool(self.config.repair_command)
        value = {"schema": "wwl.watchdog.ready@1", "run_id": self.config.run_id,
            "policy_id": "worldwide-legal-search-recovery-v1", "repository": str(self.config.site_root),
            "native_revision": git(self.config.framework_root, "rev-parse", "HEAD"),
            "admitted_plan_revision": self.bootstrap_receipt["admitted_plan_revision"],
            "readiness_receipt": self.bootstrap_receipt["receipt_id"], "required_task_ids": list(self.tasks),
            "bindings": {"task_source": True, "reconcile": True, "lifecycle_restart": True,
                "route_policy": True, "independent_validation": True, "validated_promotion": True,
                "incident_deduplication": True, "meaningful_progress": True, "repair_dispatch": enabled_repair,
                "rescue_planner": False, "rescue_orchestrator": False},
            "control_canary_verified": False, "bootstrap_only": True}
        atomic_json(self.config.state_root / "campaign-readiness.json", value)

    def _run_pass(self) -> None:
        self._provider_callback_entered = False
        try:
            self.capture()
            self._retained_attempt_before_pass = bool(self._daemon.list_running_attempts())
            self._reconcile_controls()
            self._last_result = dict(self._daemon.run_once())
            self.capture()
            previous_task = str(self._last_failure.get("task_id") or "")
            if self._last_failure and (previous_task in self._cached.get("verified_task_ids", []) or (not previous_task and not self._cached.get("active_task_id"))):
                self._last_failure = {}
        except Exception as error:
            # The wrapper knows whether its only provider callback was ever
            # entered. A prior retained attempt can still have unknown effects,
            # so never infer non-dispatch from zero accepted receipt counts.
            unknown = self._provider_callback_entered or self._retained_attempt_before_pass
            trace = traceback.TracebackException.from_exception(error, capture_locals=False)
            frames = []
            for frame in list(trace.stack)[-20:]:
                frame_path = Path(frame.filename).resolve()
                if frame_path.is_relative_to(self.config.framework_root):
                    relative = frame_path.relative_to(self.config.framework_root).as_posix()
                    frames.append({"path": relative, "line": frame.lineno, "function": frame.name,
                        "source": (frame.line or "")[:500], "source_sha256": hashlib.sha256(frame_path.read_bytes()).hexdigest() if frame_path.is_file() else ""})
            payload = {"schema": "wwl.native-failure@1", "run_id": self.config.run_id, "error_class": type(error).__name__,
                "reason": str(error)[:1500], "failure_key": identity([type(error).__name__, str(error)[:500]]),
                "source_revision": git(self.config.framework_root, "rev-parse", "HEAD"), "observed_at": time.time(), "effects_unknown": unknown,
                "task_id": self._cached.get("active_task_id", ""), "provider_callback_entered": self._provider_callback_entered,
                "retained_attempt_before_pass": self._retained_attempt_before_pass, "phase": self._cached.get("stage", "preselection"),
                "traceback": "".join(trace.format())[-12000:], "source_frames": frames,
                "reproduce_argv": [self.config.python_executable, "-m", MODULE, "run", "--config", str(self.config.config_path), "--once"],
                "reproduce_requires_exclusive_owner_and_effect_reconciliation": True}
            payload["receipt_id"] = identity(payload)
            atomic_json(self.config.state_root / "failures" / (payload["failure_key"].split(":")[1] + ".json"), payload)
            self._last_failure = payload
        finally:
            self._passes += 1

    def _reconcile_controls(self) -> None:
        # Only the native execution thread touches the embedded DB. Control
        # requests are admitted by the owner and consumed at this boundary.
        pending = list(self._pending_controls)
        self._pending_controls.clear()
        for action_id in pending:
            action = self._actions[action_id]
            try:
                evidence = {
                    "prepared_completions": self._daemon.reconcile_prepared_task_completions(),
                    "terminal_failures": self._daemon.reconcile_terminal_portal_failures(),
                    "expired_attempts": self._daemon.reconcile_expired_running_attempts(),
                    "failure_rearms": self._daemon.reconcile_recoverable_portal_failure_rearms(),
                }
                self.capture()
                if self._last_failure and not self._last_failure.get("effects_unknown"):
                    # Keep the reproducible defect visible for repair. Its
                    # pre-callback proof establishes known no-provider effects.
                    evidence["known_pre_provider_failure"] = self._last_failure["receipt_id"]
                outcome = "succeeded" if not self._last_failure.get("effects_unknown", False) else "blocked"
                action["observation"] = {"schema": "wwl.watchdog.action-observation@1", "action_id": action_id,
                    "status": outcome, "receipt": identity(evidence), "effects_verified": True}
            except Exception as error:
                action["observation"] = {"schema": "wwl.watchdog.action-observation@1", "action_id": action_id,
                    "status": "unknown", "receipt": identity([type(error).__name__, str(error)]), "effects_verified": False}
        if pending:
            atomic_json(self.config.state_root / "control/action-ledger.json", self._actions)

    def process_requests(self) -> None:
        requests = self.config.state_root / "control/requests"
        requests.mkdir(mode=0o700, parents=True, exist_ok=True)
        for path in sorted(requests.glob("*.json")):
            request = read_json(path)
            action_id = str(request.get("action_id") or "")
            if not action_id or path.stem != action_id or request.get("run_id") != self.config.run_id or request.get("schema") != "wwl.watchdog.action@1":
                continue
            if action_id in self._actions:
                continue
            kind = request.get("kind")
            receipt = {"schema": "wwl.watchdog.action-result@1", "action_id": action_id, "status": "blocked", "receipt": identity(request)}
            reason = "native operation is not currently bound"
            if kind == "reconcile":
                # A fresh owner observation is safe while a provider is
                # active. It cannot certify/clear ambiguous provider effects.
                reason = "native pass reconciliation is retained at the next safe owner boundary"
                receipt.update(status="accepted", deadline_at=float(request["issued_at"]) + 120)
                self._pending_controls.append(action_id)
            elif kind == "recover":
                self.cancel.set()
                reason = "cancellation requested; native ownership/effect reconciliation required before another writer"
                receipt.update(status="accepted", deadline_at=float(request["issued_at"]) + 120)
            elif kind == "repair" and self.config.repair_command:
                self.cancel.set()
                reason = "repair queued behind drained native attempt"
                receipt.update(status="accepted", deadline_at=float(request["issued_at"]) + self.config.task_timeout_seconds)
            self._actions[action_id] = {"request": request, "result": receipt, "reason": reason,
                "observation": {"schema": "wwl.watchdog.action-observation@1", "action_id": action_id,
                    "status": "running" if receipt["status"] == "accepted" else "blocked", "receipt": receipt["receipt"],
                    "effects_verified": receipt["status"] == "blocked", **({"deadline_at": receipt["deadline_at"]} if receipt["status"] == "accepted" else {})}}
            atomic_json(self.config.state_root / "control/action-ledger.json", self._actions)
            atomic_json(self.config.state_root / "control/results" / path.name, receipt)

    def settle_controls(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        for action_id, action in self._actions.items():
            if action["result"]["status"] != "accepted" or action["observation"]["status"] != "running":
                continue
            request = action["request"]
            if request["kind"] == "recover":
                # Native reconciliation is the next pass, never a guessed
                # task status write or a second provider launch.
                action["observation"] = {"schema": "wwl.watchdog.action-observation@1", "action_id": action_id,
                    "status": "blocked", "receipt": identity({"action": action_id, "native_worker_drained": True, "requires_next_native_pass": True}), "effects_verified": True}
                self.cancel.clear()
            elif request["kind"] == "repair" and (self._repair_thread is None or not self._repair_thread.is_alive()):
                # The host binds the native bounded dispatcher, which runs
                # repair-once below; its exit code is never repair evidence.
                self._repair_thread = threading.Thread(target=self._execute_repair, args=(action_id, action), name="wwl-native-repair", daemon=False)
                self._repair_thread.start()
        atomic_json(self.config.state_root / "control/action-ledger.json", self._actions)

    def _execute_repair(self, action_id: str, action: dict[str, Any]) -> None:
        from ..rescue.repair_dispatcher import Config, run_worker

        request_path = self.config.state_root / "control/requests" / (action_id + ".json")
        argv = [part.replace("{request_path}", str(request_path)) for part in self.config.repair_command]
        manifest = self.config.state_root / "repair-manifest.json"
        atomic_json(manifest, {"repo_root": str(self.config.framework_root), "run_id": self.config.run_id})
        config = Config(manifest_path=manifest, repo_root=self.config.framework_root,
            state_dir=self.config.state_root / "repairs" / action_id, worker_argv=argv,
            prompt_path=request_path, worker_timeout=self.config.task_timeout_seconds)
        config.state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        result = run_worker(config, config.state_dir / "dispatcher.log", lambda process: atomic_json(config.state_dir / "dispatcher-process.json", {key: value for key, value in process.items() if key != "argv"}))
        evidence_path = config.state_dir / "native-repair-result.json"
        evidence = read_json(evidence_path) if evidence_path.is_file() else {}
        passed = bool(evidence.get("native_completion_receipt") and evidence.get("validated") is True and evidence.get("action_id") == action_id)
        action["observation"] = {"schema": "wwl.watchdog.action-observation@1", "action_id": action_id,
            "status": "succeeded" if passed else "failed", "receipt": identity({"dispatcher": result, "native_evidence": evidence}), "effects_verified": True}
        self.cancel.clear()
        atomic_json(self.config.state_root / "control/action-ledger.json", self._actions)

    def run(self, *, once: bool = False) -> dict[str, Any]:
        self.open()
        self.publish_readiness()
        try:
            while not self.stop.is_set():
                self.process_requests()
                self.settle_controls()
                hold = any((self.config.state_root / name).exists() for name in ("HOLD", "STOP", "OPERATOR_STOP"))
                repairing = self._repair_thread is not None and self._repair_thread.is_alive()
                if not hold and not repairing and (self._worker is None or not self._worker.is_alive()):
                    if once and self._passes:
                        break
                    self.cancel.clear()
                    self._worker = threading.Thread(target=self._run_pass, name="wwl-native-db-owner", daemon=False)
                    self._worker.start()
                self.publish()
                if once and self._worker is not None and not self._worker.is_alive():
                    break
                self.stop.wait(min(1.0, self.config.poll_seconds))
            self.cancel.set()
            if self._worker is not None:
                self._worker.join(timeout=60)
                if self._worker.is_alive():
                    raise RuntimeError("native worker has not drained; refusing DB ownership release")
            return self.publish(alive=False)
        finally:
            if self._worker is None or not self._worker.is_alive():
                self._daemon.close()


class _ProgressCallbacks:
    def __init__(self, runtime: CampaignRuntime, bridge: Any):
        self.runtime, self.bridge = runtime, bridge
        self.attempt_root, self.max_task_attempts = bridge.attempt_root, bridge.max_task_attempts

    def run_provider(self, attempt: Any) -> Any:
        self.runtime._provider_callback_entered = True
        self.runtime.capture("provider", attempt)
        return self.bridge.run_provider(attempt)

    def apply_effect(self, attempt: Any, result: Any) -> Any:
        self.runtime.capture("effect", attempt)
        return self.bridge.apply_effect(attempt, result)

    def validate_effect(self, attempt: Any, result: Any) -> Any:
        self.runtime.capture("validation", attempt)
        return self.bridge.validate_effect(attempt, result)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.bridge, name)


def validate_goal(config: CampaignConfig, goal_id: str) -> dict[str, Any]:
    plan = read_json(config.plan_path)
    goal = next((goal for goal in plan["goals"] if goal["id"] == goal_id), None)
    if goal is None:
        raise ValueError("unknown goal identity")
    status = read_json(config.state_root / "campaign-status.json")
    if status.get("run_id") != config.run_id or status.get("accepted_tree") != identity(observed_forest(config)):
        raise ValueError("goal status does not identify the current campaign repository forest")
    observed = float(status.get("observed_at", 0))
    if not 0 <= time.time() - observed <= 3600:
        raise ValueError("goal native completion observation is stale")
    required = set(goal["task_ids"])
    if not required <= set(status["verified_task_ids"]):
        raise ValueError("goal has tasks without current native completion evidence")
    receipts = status.get("native_completion_receipts_by_alias", {})
    if not status.get("native_completion_projection") or any(not receipts.get(alias) for alias in required):
        raise ValueError("native completion projection is missing")
    return {"schema": "worldwide-legal-search.goal-validation@1", "goal_id": goal_id,
        "native_projection": status["native_completion_projection"], "accepted_tree": status["accepted_tree"], "verified": True}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("command", choices=("bootstrap", "run", "status", "validate-goal", "repair-once"))
    result.add_argument("--config", type=Path)
    for option in ("site-root", "framework-root", "state-root", "plan-path", "implementation-command", "python-executable", "run-id"):
        result.add_argument("--" + option)
    result.add_argument("--goal-id")
    result.add_argument("--request", type=Path)
    result.add_argument("--once", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        config = config_from_args(args)
        if args.command == "bootstrap":
            output = bootstrap(config)
        elif args.command == "status":
            output = read_json(config.state_root / "campaign-status.json")
        elif args.command == "validate-goal":
            output = validate_goal(config, str(args.goal_id or ""))
        elif args.command == "repair-once":
            if not args.request:
                raise ValueError("repair-once requires a bound control request")
            output = repair_once(config, args.request)
        else:
            os.environ["PATH"] = str(Path(config.python_executable).parent) + os.pathsep + os.environ.get("PATH", "")
            with owner_lock(config):
                receipt = bootstrap(config, already_locked=True)
                runtime = CampaignRuntime(config, receipt)
                def request_stop(_signum: int, _frame: Any) -> None:
                    runtime.cancel.set()
                    runtime.stop.set()
                signal.signal(signal.SIGTERM, request_stop)
                signal.signal(signal.SIGINT, request_stop)
                output = runtime.run(once=args.once)
        print(json.dumps(output, sort_keys=True, default=str), flush=True)
        return 0
    except Exception as error:
        print(json.dumps({"schema": "wwl.native-runtime-error@1", "error_class": type(error).__name__, "error": str(error), "completion_authority": False}), file=sys.stderr)
        return 1


def repair_once(config: CampaignConfig, request_path: Path) -> dict[str, Any]:
    """Execute a separate operational repair task through all native gates."""
    request = read_json(request_path)
    action_id = str(request.get("action_id") or "")
    if request.get("schema") != "wwl.watchdog.action@1" or request.get("kind") != "repair" or request.get("run_id") != config.run_id or not action_id or not request.get("reproducer_receipt"):
        raise ValueError("repair task requires an admitted campaign action and reproducer")
    state = config.state_root / "repairs" / action_id
    task_id = "WWL-REPAIR-" + action_id[:16]
    source = read_json(config.plan_path)
    task = {"id": task_id, "title": "Repair the reproduced native supervisor defect", "status": "todo", "completion": "auto",
        "goal_id": "WWL-G000", "subgoal_id": "WWL-G000", "root_goal_id": "WWL-G000", "repository": "accelerate", "depends_on": [],
        "outputs": ["ipfs_accelerate_py/agent_supervisor/project_adapters/worldwide_legal_search_runtime.py", "test/api/test_worldwide_legal_search_runtime.py"],
        "evidence_output": "artifacts/worldwide-legal-search/repairs/" + action_id + ".json",
        "write_scope": ["ipfs_accelerate_py/agent_supervisor/project_adapters/", "test/api/"],
        "acceptance": "Reproduce and fix exactly the observed native defect, preserving routing, ownership, validation and completion gates. Original failure evidence: " + request["reproducer_receipt"] + "; reason: " + str(request.get("reason") or "") + ". Add a failing regression first and make it pass; do not alter criteria or fabricate receipts.",
        "validation_argv": [[config.python_executable, "-m", "pytest", "test/api/test_worldwide_legal_search_runtime.py", "test/api/test_worldwide_legal_search_provider.py", "test/api/test_worldwide_legal_search_goals.py", "-q"]],
        "bounds": {"max_attempts": 2, "wall_seconds_per_attempt": config.task_timeout_seconds, "validation_reserve_percent": 25}}
    plan = {**source, "namespace": config.run_id + "-repair-" + action_id[:16], "tasks": [task],
        "goals": [{"id": "WWL-G000", "title": "Repair the exact reproduced native defect", "parent_ids": [], "task_ids": [task_id], "status": "active", "acceptance": task["acceptance"]}]}
    plan_path, objectives, config_path = state / "repair-plan.json", state / "goals.objectives.md", state / "runtime-config.json"
    state.mkdir(mode=0o700, parents=True, exist_ok=True)
    atomic_json(plan_path, plan)
    objectives.write_text("# Operational repair\n\n## WWL-G000 Repair the reproduced native defect\n- Status: active\n- Parent goal ids json: []\n- Required task ids json: " + json.dumps([task_id]) + "\n")
    base = read_json(config.config_path)
    raw = {**base, **config.serializable(), "state_root": str(state), "plan_path": str(plan_path), "objectives_path": str(objectives), "run_id": plan["namespace"], "repair_command": [],
           "implementation_command": "", "allowed_workspace_roots": [str(state / "worktrees")]}
    atomic_json(config_path, raw)
    child = config_from_args(parser().parse_args(["run", "--config", str(config_path), "--once"]))
    with owner_lock(child):
        boot = bootstrap(child, already_locked=True)
        runtime = CampaignRuntime(child, boot)
        result = runtime.run(once=True)
    native = result.get("native_completion_receipts", {})
    validated = task_id in result.get("verified_task_ids", []) and bool(native)
    output = {"schema": "wwl.native-repair-result@1", "action_id": action_id, "validated": validated,
        "native_completion_receipt": next(iter(native.values()), "") if validated else "",
        "native_projection": result.get("native_completion_projection"), "framework_head": git(config.framework_root, "rev-parse", "HEAD"),
        "framework_tree": git(config.framework_root, "rev-parse", "HEAD^{tree}"), "completion_authority": False}
    atomic_json(state / "native-repair-result.json", output)
    return output


if __name__ == "__main__":
    raise SystemExit(main())
