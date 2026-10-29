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
import fnmatch
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
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

# This stdlib-only validator is embedded verbatim in the native immutable
# validation argv. Candidate imports cannot replace its scope or red/green
# checks. Every baseline module comes from the admitted Git object.
REPAIR_VALIDATOR = r'''
import hashlib, json, os, pathlib, subprocess, sys, tarfile, tempfile, xml.etree.ElementTree as ET
b = json.loads(sys.argv[1]); root = pathlib.Path.cwd()
def command(argv, *, cwd=root, env=None, timeout=900):
    return subprocess.run(argv, cwd=cwd, env=env, capture_output=True, timeout=timeout)
def require(ok, reason):
    if not ok: raise SystemExit(reason)
require(command(['git','merge-base','--is-ancestor',b['source_revision'],'HEAD']).returncode == 0, 'repair baseline is not an ancestor')
changed = command(['git','diff','--name-only',b['source_revision']]).stdout.decode().splitlines()
changed += command(['git','ls-files','--others','--exclude-standard']).stdout.decode().splitlines()
require(set(changed) <= set(b['allowed_paths']), 'repair changed an undeclared path')
for path, digest in b['protected_hashes'].items():
    target = root/path
    require(target.is_file() and not target.is_symlink() and hashlib.sha256(target.read_bytes()).hexdigest() == digest, 'repair changed a protected regression or criterion')
regression = root/b['regression_path']
require(regression.is_file() and not regression.is_symlink() and 0 < regression.stat().st_size <= 131072, 'repair regression is missing or unbounded')
with tempfile.TemporaryDirectory(prefix='wwl-repair-red-green-') as temporary:
    base = pathlib.Path(temporary); checkout = base/'baseline'; checkout.mkdir()
    listing = command(['git','ls-tree','-r','--name-only',b['source_revision']]).stdout.decode().splitlines()
    names = [p for p in listing if (p.startswith('ipfs_accelerate_py/') and p.endswith('.py')) or p in b['baseline_inputs']]
    require(len(names) <= 10000, 'repair baseline source exceeds bound')
    archive = subprocess.Popen(['git','archive',b['source_revision'],'--',*names], cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    total = 0
    with tarfile.open(fileobj=archive.stdout, mode='r|') as stream:
        for member in stream:
            total += member.size
            require(member.isdir() or member.isfile(), 'repair baseline contains a link')
            require(total <= 209715200 and member.size <= 16777216, 'repair baseline bytes exceed bound')
            stream.extract(member, checkout, filter='data')
    require(archive.wait(timeout=30) == 0, 'repair baseline archive failed')
    target = checkout/b['regression_path']; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(regression.read_bytes())
    def pytest(cwd, targets, name):
        environment = dict(os.environ); environment['PYTHONPATH'] = str(cwd)
        environment['PYTHONDONTWRITEBYTECODE'] = '1'
        environment['IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB'] = str(base/(name+'-seal.duckdb'))
        report = base/(name+'.xml')
        result = command([sys.executable,'-B','-m','pytest',*targets,'-q','-o','cache_dir='+str(base/(name+'-cache')),'--junitxml='+str(report)], cwd=cwd, env=environment)
        require(report.is_file(), name+' regression produced no independent test report')
        suites = ET.parse(report).getroot(); suites = [suites] if suites.tag == 'testsuite' else suites.findall('testsuite')
        counts = {key:sum(int(s.attrib.get(key,0)) for s in suites) for key in ('tests','errors','failures')}
        return result, counts
    red, counts = pytest(checkout,[b['regression_path']],'baseline')
    require(red.returncode == 1 and counts['tests'] >= 1 and counts['errors'] == 0 and counts['failures'] >= 1, 'new regression did not reproduce a test failure on the admitted baseline')
    green, counts = pytest(root,[b['regression_path'],*b['focused_tests']],'candidate')
    require(green.returncode == 0 and counts['tests'] >= 1 and counts['errors'] == counts['failures'] == 0, 'repair candidate failed independent regressions')
print(json.dumps({'schema':'wwl.native-repair-validation@1','action_id':b['action_id'],'baseline_revision':b['source_revision'],'baseline_failure_verified':True,'candidate_regressions_verified':True,'completion_authority':False},sort_keys=True))
'''


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
            "acceptance": [task["acceptance"]], "validations": [{"argv": list(argv)} for argv in task["validation_argv"]], "validation": " && ".join(validations),
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


def repair_pipeline_binding(config: CampaignConfig) -> dict[str, Any]:
    """Bind pipeline behavior independently from unrelated repository commits."""
    plan = read_json(config.plan_path)
    policy = read_json(config.plan_path.with_name("watchdog.policy.json"))
    paths = ["ipfs_accelerate_py/agent_supervisor/project_adapters/worldwide_legal_search_runtime.py",
        "ipfs_accelerate_py/agent_supervisor/project_adapters/worldwide_legal_search_provider.py",
        "ipfs_accelerate_py/agent_supervisor/project_adapters/worldwide_legal_search_continuation.py",
        "ipfs_accelerate_py/agent_supervisor/project_adapters/worldwide_legal_search_goals.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon_runner.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/database_portal_bridge.py",
        "ipfs_accelerate_py/agent_supervisor/task_sources/board_control_plane.py",
        "ipfs_accelerate_py/agent_supervisor/task_sources/database_task_source.py",
        "ipfs_accelerate_py/agent_supervisor/task_sources/intent_repository.py",
        "ipfs_accelerate_py/agent_supervisor/validation/validation_runtime.py",
        "ipfs_accelerate_py/agent_supervisor/validation/validation_commands.py",
        "ipfs_accelerate_py/agent_supervisor/validation/project_dependency_preflight.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/supervisor_runtime.py",
        "ipfs_accelerate_py/agent_supervisor/grok_cli_runner.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/grok_cli_runner.py",
        "ipfs_accelerate_py/agent_supervisor/provider_fallback_runner.py",
        "ipfs_accelerate_py/agent_supervisor/rescue/repair_dispatcher.py"]
    return {"schema": "wwl.native-repair-pipeline-binding@1", "interface": "wwl.native-repair-pipeline@1",
        "validator_sha256": hashlib.sha256(REPAIR_VALIDATOR.encode()).hexdigest(),
        "route_id": identity(plan.get("routing", {})), "repair_scope_id": identity(policy["repair_scope"]),
        "source_sha256": {path: hashlib.sha256((config.framework_root / path).read_bytes()).hexdigest() for path in paths}}


def verified_repair_canary(config: CampaignConfig) -> dict[str, Any] | None:
    from .worldwide_legal_search_watchdog import read_owned_json

    raw = read_json(config.config_path) if config.config_path.is_file() else {}
    path = Path(raw.get("repair_canary_receipt") or config.state_root / "repair-canary.json")
    try:
        receipt = read_owned_json(path)
        if receipt.get("schema") != "wwl.native-repair-canary@1" or receipt.get("receipt_id") != identity({key: value for key, value in receipt.items() if key != "receipt_id"}):
            return None
        if receipt.get("pipeline_binding") != repair_pipeline_binding(config):
            return None
        if not all(receipt.get(key) is True for key in ("baseline_failure_verified", "candidate_regressions_verified", "validated_promotion")):
            return None
        if not receipt.get("native_completion_receipt") or not receipt.get("native_projection") or not receipt.get("source_provenance"):
            return None
        return receipt
    except (OSError, ValueError, KeyError, TypeError):
        return None


def storage_observation(config: CampaignConfig) -> dict[str, Any]:
    from .worldwide_legal_search_watchdog import resource_observation
    raw = read_json(config.config_path) if config.config_path.is_file() else {}
    probe_root = config.state_root
    while not probe_root.exists() and probe_root.parent != probe_root:
        probe_root = probe_root.parent
    return resource_observation(repository=config.site_root, state_root=probe_root, framework_root=config.framework_root, config=raw)


def wait_for_storage(config: CampaignConfig, stop: threading.Event) -> dict[str, Any]:
    """Wait without native database opens, durable writes or provider work."""
    last_report = 0.0
    while True:
        observation = storage_observation(config)
        if not observation["blocked"] or stop.is_set():
            return observation
        if time.monotonic() - last_report >= 30:
            print(json.dumps({"schema": "wwl.native-runtime-resource-wait@1", "run_id": config.run_id,
                "blocked_kind": "resource_quota", "resource_pressure": observation, "new_provider_dispatch_admitted": False,
                "new_execution_pass_admitted": False, "completion_authority": False}, sort_keys=True), file=sys.stderr, flush=True)
            last_report = time.monotonic()
        stop.wait(min(5.0, max(0.1, config.poll_seconds)))


@contextlib.contextmanager
def owner_lock(config: CampaignConfig):
    config.state_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    # Establish explicit campaign custody before any native path resolver can
    # consider an unlocked lane database eligible for legacy relocation.
    from ..task_sources.board_control_plane import register_managed_state_root
    register_managed_state_root(config.state_root, owner_id=config.run_id)
    with (config.state_root / "native-owner.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("native campaign state already has a live owner") from exc
        yield


def bootstrap(config: CampaignConfig, *, already_locked: bool = False) -> dict[str, Any]:
    if storage_observation(config)["blocked"]:
        raise OSError(28, "native bootstrap is deferred by resource quota before state creation")
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
            with source.intent.read_session():
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
        repaired_observations = False
        for action in self._actions.values():
            observation = action.get("observation", {})
            if (action.get("request", {}).get("kind") == "reconcile"
                    and observation.get("status") == "blocked" and observation.get("effects_verified") is False):
                action["previous_unverified_terminal_observation"] = dict(observation)
                action["observation"] = {**observation, "status": "unknown"}
                repaired_observations = True
        if repaired_observations:
            atomic_json(ledger, self._actions)
        self._last_failure: dict[str, Any] = {}
        previous_status = config.state_root / "campaign-status.json"
        if previous_status.is_file():
            previous = read_json(previous_status)
            failure = previous.get("native_error", {})
            if previous.get("run_id") == config.run_id and isinstance(failure, dict) and failure.get("receipt_id") == identity({key: value for key, value in failure.items() if key != "receipt_id"}):
                self._last_failure = failure
        self._last_result: dict[str, Any] = {}
        self._worker: threading.Thread | None = None
        self._daemon: Any = None
        self._progress_at = time.time()
        self._progress_token = ""
        self._passes = 0
        self._loaded_framework_revision = git(config.framework_root, "rev-parse", "HEAD") if config.framework_root.is_dir() else ""
        self._last_successful_pass: dict[str, Any] = {}
        self._native_terminal_result: dict[str, Any] = {}
        self._candidate_request_path: Path | None = None
        self._candidate_request_sha256: str | None = None
        self._candidate_completed = False
        self._candidate_checkout_restart = False
        pending = config.state_root / "control/pending-candidate-continuation.json"
        if pending.is_file():
            from .worldwide_legal_search_watchdog import read_owned_json
            pointer = read_owned_json(pending)
            candidate = Path(pointer.get("request_path", ""))
            if candidate.is_absolute() and candidate.is_file():
                read_owned_json(candidate)  # Bound and authenticate bytes before hashing them.
            if (pointer.get("schema") != "wwl.pending-candidate-continuation@1"
                    or pointer.get("run_id") != config.run_id or not candidate.is_absolute()
                    or candidate.resolve(strict=True) != candidate or not candidate.is_relative_to(config.state_root)
                    or hashlib.sha256(candidate.read_bytes()).hexdigest() != pointer.get("request_sha256")):
                raise ValueError("pending candidate continuation custody differs")
            self._candidate_request_path = candidate
            self._candidate_request_sha256 = pointer["request_sha256"]
            self._read_candidate_request()
        self._provider_activity_binding: dict[str, Any] = {}
        self._provider_activity_sequence = 0
        self._provider_activity: dict[str, Any] = {}
        self._reload_request: dict[str, Any] = {}
        try:
            self._loaded_pipeline = repair_pipeline_binding(config)
        except (OSError, ValueError, KeyError):
            self._loaded_pipeline = {}
        self._provider_callback_entered = False
        self._retained_attempt_before_pass = False
        self._pending_controls: list[str] = [key for key, value in self._actions.items()
            if value.get("request", {}).get("kind") == "reconcile" and value.get("result", {}).get("status") == "accepted"
            and value.get("observation", {}).get("status") in {"running", "unknown"}]
        self._repair_thread: threading.Thread | None = None
        self._canary_refresh_lock = threading.Lock()
        self._canary_refresh_thread: threading.Thread | None = None
        self._canary_refresh_process: subprocess.Popen | None = None
        self._canary_refresh_next_at = 0.0
        self._canary_refresh_stopping = False

    def _maybe_refresh_repair_canary(self) -> None:
        """Refresh optional repair proof after reload, without blocking work."""
        if not self.config.repair_command or self.stop.is_set() or self.cancel.is_set():
            return
        with self._canary_refresh_lock:
            if self._canary_refresh_stopping or time.monotonic() < self._canary_refresh_next_at or (self._canary_refresh_thread is not None and self._canary_refresh_thread.is_alive()):
                return
            # A promotion must first fence and reload its old owner. A canary
            # cannot acknowledge that fence or delay the required handoff.
            try:
                if git(self.config.framework_root, "rev-parse", "HEAD") != self._loaded_framework_revision:
                    return
                current = verified_repair_canary(self.config)
            except (OSError, ValueError, KeyError, TypeError):
                self._canary_refresh_next_at = time.monotonic() + 300
                return
            if current is not None:
                self._canary_refresh_next_at = time.monotonic() + 30
                return
            self._canary_refresh_next_at = time.monotonic() + 300
            self._canary_refresh_thread = threading.Thread(target=self._refresh_repair_canary, name="wwl-repair-canary-refresh", daemon=True)
            self._canary_refresh_thread.start()

    def _refresh_repair_canary(self) -> None:
        from .worldwide_legal_search_canary import refresh_argv

        started = time.time()
        result: dict[str, Any] = {"schema": "wwl.native-repair-canary-refresh-observation@1", "started_at": started,
            "verified": False, "paid_provider_calls": 0, "completion_authority": False}
        path = self.config.state_root / "control/repair-canary-refresh.json"
        process = None
        try:
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with path.with_suffix(".log").open("wb") as log:
                with self._canary_refresh_lock:
                    if self._canary_refresh_stopping:
                        result["reason"] = "owner_shutdown_cancelled_optional_refresh"
                        return
                    process = subprocess.Popen(refresh_argv(self.config.python_executable, self.config.config_path), cwd=self.config.framework_root,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    self._canary_refresh_process = process
                try:
                    result["returncode"] = process.wait(timeout=900)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    result.update(returncode=process.wait(timeout=5), reason="bounded_refresh_deadline_elapsed")
            canary = verified_repair_canary(self.config)
            result["verified"] = result.get("returncode") == 0 and canary is not None
            result["receipt_id"] = canary["receipt_id"] if result["verified"] else ""
        except Exception as error:
            result.update(error_class=type(error).__name__, reason=str(error)[:500])
        finally:
            with self._canary_refresh_lock:
                self._canary_refresh_process = None
            result["completed_at"] = time.time()
            atomic_json(path, result)
            # Readiness publication only reads bounded filesystem receipts.
            # It neither opens the native DB nor acknowledges a reload fence.
            if not self.stop.is_set() and not self._canary_refresh_stopping:
                self.publish_readiness()

    def _stop_repair_canary_refresh(self) -> None:
        """Cancel optional proof work without joining it during owner reload."""
        with self._canary_refresh_lock:
            self._canary_refresh_stopping = True
            process = self._canary_refresh_process
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def repository_for_task(self, alias: str) -> Path:
        task = self.tasks.get(alias)
        if task is None:
            raise ValueError("unknown campaign task alias")
        if task["repository"] == "site":
            return self.config.site_root
        if task["repository"] == "accelerate":
            return self.config.framework_root
        raise ValueError("read-only reference cannot own an execution task")

    def _make_native_portal(self, **kwargs: Any) -> Any:
        from ..todo_daemon.implementation_daemon import PortalImplementationDaemon
        # Preserve exact native type identity for custody attestation.
        kwargs["implementation_cancelled"] = self.cancel.is_set
        kwargs["implementation_declared_outputs_only"] = bool(self.plan.get("operational_repair_binding"))
        return PortalImplementationDaemon(**kwargs)

    def open(self) -> None:
        from ..todo_daemon.implementation_daemon import DatabaseImplementationDaemon, PortalImplementationDaemon, parse_args
        from ..todo_daemon.implementation_daemon_runner import bind_database_portal_execution_from_args

        self._daemon = DatabaseImplementationDaemon(database_path=self.config.database,
            coordination_path=self.config.state_root / "coordination.duckdb", execution_path=self.config.state_root / "execution.duckdb",
            state_dir=self.config.state_root / "native", state_prefix="wwl", owner_session_id=self.config.run_id,
            authority_mode="embedded_exclusive", require_real_execution=True, max_task_attempts=self.config.max_task_attempts,
            task_prefix="WWL-", execution_slice_task_ids=(str(self._read_candidate_request()["task_id"]),)
                if self._candidate_request_path is not None else ()).open()
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
        for protected in ("config/worldwide_supervisor_host.json", "scripts/worldwide_supervisor_host.py",
                          "implementation_plan/worldwide_legal_search/watchdog_loop.py"):
            if (self.config.site_root / protected).is_file() and protected not in args.implementation_protected_path:
                args.implementation_protected_path.append(protected)
        captured: dict[str, Any] = {}
        proxy = SimpleNamespace(task_source=self._daemon.task_source,
            authorize_superseded_portal_attempt_binding=self._daemon.authorize_superseded_portal_attempt_binding,
            bind_execution_callbacks=lambda **callbacks: captured.update(callbacks))
        bridge = bind_database_portal_execution_from_args(proxy, args, repo_root=self.config.site_root, portal_daemon_class=self._make_native_portal,
            task_repo_root_resolver=self.repository_for_task, provider_handoff_binder=self._observe_provider_handoff,
            candidate_reconciliation_loader=self._load_candidate_admission if self._candidate_request_path is not None else None)
        self._bridge = bridge
        callbacks = _ProgressCallbacks(self, bridge)
        self._daemon.bind_execution_callbacks(provider_fn=bridge.run_provider, effect_fn=callbacks.apply_effect, validation_fn=callbacks.validate_effect)

    def capture(self, stage: str = "", attempt: Any = None) -> None:
        """Called only by the native DB owner worker, never the watcher."""
        source = self._daemon.task_source
        with source.intent.read_session():
            snapshot = source.snapshot().to_dict()
            records = source.list_tasks(limit=256).tasks
            evidence = dict(source.intent.completion_evidence_projection())
            receipt_cids = {item["task_cid"]: item["receipt_cid"] for item in evidence["completion_receipts"]}
            verified = [task.task_alias for task in records if task.status in COMPLETED and task.task_cid in receipt_cids]
            ready = [task.task_alias for task in source.ready_tasks(limit=256).tasks]
            plans = source.plan_projection()["plans"]
            native_active = [plan for plan in plans if plan["status"] == "active"]
        running = self._daemon.list_running_attempts()
        blockers = [self._native_blocker(record) for record in records if record.status == "blocked"]
        current = attempt or (running[0] if running else None)
        active = str(getattr(current, "task_alias", "") or "")
        phase = stage or str(getattr(current, "committed_phase", "") or "idle")
        signature = {"tasks": [(r.task_alias, r.status) for r in records], "verified": verified,
                     "active": active, "attempt_id": str(getattr(current, "attempt_id", "") or ""), "phase": phase, "receipts": receipt_cids}
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
                "source_snapshot": snapshot, "native_blocked_tasks": blockers}
            self._cached.update(native_active_plan_revision=native_active[0]["plan_cid"] if len(native_active) == 1 else "",
                native_active_plan_revision_receipt=native_active[0].get("body", {}).get("revision_cid", "") if len(native_active) == 1 else "")

    def _native_blocker(self, record: Any) -> dict[str, Any]:
        """Project canonical blocked state and bounded native diagnostics."""
        from .worldwide_legal_search_watchdog import read_owned_json
        receipt = dict(record.body.get("completion_receipt") or {})
        result = {"task_alias": record.task_alias, "task_cid": record.task_cid, "task_revision": record.revision,
            "status": "blocked", "attempt_id": str(receipt.get("attempt_id") or ""),
            "native_failure_receipt": str(receipt.get("settlement_id") or ""),
            "reason": "native_task_blocked", "effects_unknown": False, "completion_authority": False}
        attempt_id = result["attempt_id"]
        if not attempt_id or receipt.get("task_cid") != record.task_cid:
            return result
        root = self.config.state_root / "native/wwl_database_portal_attempts" / hashlib.sha256(attempt_id.encode()).hexdigest()[:24]
        try:
            binding = read_owned_json(root / "database-attempt-binding.json")
            if any(binding.get(key) != value for key, value in (("attempt_id", attempt_id), ("task_alias", record.task_alias), ("task_cid", record.task_cid))):
                return result
            events = root / "portal-events.jsonl"
            if events.is_symlink() or events.stat().st_uid != os.getuid() or events.stat().st_size > 1_048_576:
                return result
            raw = events.read_bytes()
            records = [json.loads(line) for line in raw.splitlines() if line.strip()]
            finished = [event for event in records if event.get("type") == "implementation_finished" and event.get("task_id") == record.task_alias]
            if not finished:
                started = [event for event in records if event.get("type") == "implementation_started" and event.get("task_id") == record.task_alias]
                inflight = [event for event in records if event.get("type") == "implementation_skipped" and event.get("reason") == "inflight_process" and event.get("task_id") == record.task_alias]
                if started and inflight:
                    result.update(reason="native_inflight_callback_terminalized", effects_unknown=True,
                        native_event_receipt=str(inflight[-1].get("event_id") or ""), native_events_sha256=hashlib.sha256(raw).hexdigest(),
                        diagnostic_source=str(events), native_binding_id=str(binding.get("binding_id") or ""))
                return result
            last = finished[-1]
            exception = last.get("exception_result") or {}
            cleanup_unknown = exception.get("exception_type") == "ProcessGroupCleanupUnverified"
            reason = "native_process_cleanup_unverified" if cleanup_unknown else "portal_provider_failed" if last.get("returncode") else "native_validation_or_acceptance_failed"
            result.update(reason=reason, exception_type=exception.get("exception_type", ""),
                effects_unknown=cleanup_unknown, provider_dispatched=last.get("provider_dispatched") is True,
                native_event_receipt=str(last.get("event_id") or ""), native_events_sha256=hashlib.sha256(raw).hexdigest(),
                diagnostic_source=str(events), native_binding_id=str(binding.get("binding_id") or ""))
        except (OSError, ValueError, TypeError):
            pass
        return result

    def _read_candidate_request(self) -> Mapping[str, Any]:
        from .worldwide_legal_search_watchdog import read_owned_json
        path = self._candidate_request_path
        if path is None or not path.is_absolute() or path.resolve(strict=True) != path or not path.is_relative_to(self.config.state_root):
            raise ValueError("candidate continuation request must be an exact owned campaign path")
        request = read_owned_json(path)
        expected = self._candidate_request_sha256
        if expected is not None and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("pending candidate continuation custody differs")
        if request.get("run_id") != self.config.run_id or request.get("task_id") not in self.tasks:
            raise ValueError("candidate continuation request campaign or task differs")
        return request

    def _load_candidate_admission(self, *, attempt: Any, paths: Any, binding: Mapping[str, Any], portal_daemon: Any) -> Any:
        from .worldwide_legal_search_continuation import CandidateContinuationAdmission
        request = self._read_candidate_request()
        if request["task_id"] != attempt.task_alias or request["database_task_cid"] != attempt.task_cid:
            raise ValueError("candidate continuation does not match the exact native selected task")
        prior_id = str(request.get("predecessor_attempt_id") or "")
        predecessor = self.config.state_root / "native/wwl_database_portal_attempts" / hashlib.sha256(prior_id.encode()).hexdigest()[:24] / "database-attempt-binding.json"
        bridge = self._bridge._for_attempt(attempt)
        previous = bridge._strict_binding(predecessor)
        return CandidateContinuationAdmission(request, previous, self._daemon.authorize_superseded_portal_attempt_binding)

    def _observe_provider_handoff(self, *, portal_daemon: Any, attempt: Any, paths: Any, binding: Mapping[str, Any]) -> None:
        """Observe the canonical native callback after its durable intent."""
        self._provider_callback_entered = True
        try:
            self.capture("provider", attempt)
            self._bind_provider_activity(attempt)
        except Exception:
            # Optional observation cannot replace or fail the native callback.
            pass

    def _bind_provider_activity(self, attempt: Any) -> None:
        """Reuse an exact attempt capability; the native bridge owns launch."""
        try:
            bridge = self._bridge._for_attempt(attempt)
            record = bridge._record_for_attempt(bridge.task_source, attempt)
            paths = bridge._paths(attempt)
            portal = bridge._projection_task_identity(bridge._render_projection(attempt, record), projection_path=paths.task_projection,
                task_alias=attempt.task_alias, board_namespace=bridge.board_namespace)
            attempt_id = str(attempt.attempt_id)
            binding = {"schema": "wwl.native-provider-activity-binding@1", "run_id": self.config.run_id,
                "task_id": attempt.task_alias, "task_cid": attempt.task_cid, "portal_task_cid": portal["canonical_task_cid"],
                "native_attempt_id": attempt_id, "portal_attempt_number": 1, "issued_at": time.time(),
                "capability_nonce": secrets.token_hex(32),
                "activity_path": str(self.config.state_root / "provider-activity" / (hashlib.sha256(attempt_id.encode()).hexdigest() + ".json"))}
            path = self.config.state_root / "provider-activity-binding.json"
            if path.exists():
                from .worldwide_legal_search_watchdog import read_owned_json
                previous = read_owned_json(path)
                if previous.get("native_attempt_id") == attempt_id:
                    keys = ("schema", "run_id", "task_id", "task_cid", "portal_task_cid", "native_attempt_id", "portal_attempt_number", "activity_path")
                    issued, nonce = previous.get("issued_at"), previous.get("capability_nonce")
                    valid = (not path.stat().st_mode & 0o077 and all(previous.get(key) == binding[key] for key in keys)
                        and type(issued) in (int, float) and math.isfinite(issued)
                        and int(attempt.started_at_ms) / 1000 <= issued <= time.time() + 5
                        and isinstance(nonce, str) and re.fullmatch(r"[0-9a-f]{64}", nonce)
                        and previous.get("binding_id") == identity({key: value for key, value in previous.items() if key != "binding_id"}))
                    if not valid:
                        return
                    if self._provider_activity_binding.get("binding_id") != previous["binding_id"]:
                        self._provider_activity_sequence, self._provider_activity = 0, {}
                    self._provider_activity_binding = previous
                    return
            binding["binding_id"] = identity(binding)
            atomic_json(path, binding)
            self._provider_activity_binding = binding
            self._provider_activity_sequence, self._provider_activity = 0, {}
        except Exception:
            # Missing optional telemetry never changes routing or custody.
            pass

    def _consume_provider_activity(self) -> None:
        from .worldwide_legal_search_watchdog import read_owned_json
        binding = self._provider_activity_binding
        if not binding or self._cached.get("stage") != "provider" or self._cached.get("active_attempt_id") != binding["native_attempt_id"]:
            return
        try:
            path = Path(binding["activity_path"])
            if path.stat().st_size > 4096 or path.stat().st_mode & 0o077:
                return
            value = read_owned_json(path)
            keys = ("run_id", "task_id", "task_cid", "portal_task_cid", "native_attempt_id", "portal_attempt_number", "binding_id", "capability_nonce")
            if value.get("schema") != "wwl.native-provider-activity@1" or any(value.get(key) != binding[key] for key in keys) or value.get("event_kind") != "tool_completed" or value.get("completion_authority") is not False:
                return
            sequence, observed = value.get("sequence"), value.get("observed_at")
            if type(sequence) is not int or not self._provider_activity_sequence < sequence <= 100000 or type(observed) not in (int, float) or not binding["issued_at"] <= observed <= time.time() + 5 or not 0 <= time.time() - observed <= 120:
                return
            with self._cache_lock:
                self._provider_activity_sequence = sequence
                self._progress_at = max(self._progress_at, observed)
                self._provider_activity = {key: value[key] for key in ("native_attempt_id", "sequence", "event_kind", "observed_at", "binding_id")}
                self._cached["progress_token"] = identity({"native": self._progress_token, "activity_attempt": binding["native_attempt_id"], "completed_tools": sequence})
        except (OSError, ValueError, TypeError, KeyError):
            pass

    def publish(self, *, alive: bool = True) -> dict[str, Any]:
        self._consume_provider_activity()
        now = time.time()
        with self._cache_lock:
            cached = dict(self._cached)
        failure = self._last_failure
        blockers = cached.get("native_blocked_tasks", [])
        native_unknown = any(item.get("effects_unknown") for item in blockers)
        repair_unknown = any(value.get("request", {}).get("kind") == "repair" and value.get("observation", {}).get("status") == "unknown" for value in self._actions.values())
        hold = any((self.config.state_root / name).exists() for name in ("HOLD", "STOP", "OPERATOR_STOP"))
        from ..rescue.maintenance_hold import hold_binding
        markers = [self.config.state_root / name for name in ("HOLD", "STOP", "OPERATOR_STOP")
                   if os.path.lexists(self.config.state_root / name)]
        maintenance_bindings = [hold_binding(path, now=now) for path in markers]
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
            "maintenance_hold_bindings": maintenance_bindings if markers and all(maintenance_bindings) else [],
            "completion_verified": completed, "completion_receipt": completion_receipt,
            "required_task_ids": required, "verified_task_ids": verified,
            "stage": str(cached.get("stage") or "startup"), "task_id": str(cached.get("active_task_id") or ""),
            "active_attempt_id": str(cached.get("active_attempt_id") or ""),
            "progress_token": str(cached.get("progress_token") or self.bootstrap_receipt["receipt_id"]),
            "heartbeat_at": now, "owner_alive": alive, "owner_identity_verified": alive,
            "runnable_count": len(cached.get("ready_task_ids", [])), "blocked_kind": "software_defect" if failure else ("permission" if native_unknown or self._native_terminal_result.get("reason") == "implementation_protected_path_mutated" else "none"),
            "dependency_running": False, "stage_deadline_at": cached.get("stage_deadline_at", 0),
            "effects_unknown": bool(failure.get("effects_unknown", False) or repair_unknown or native_unknown), "failure_key": failure.get("failure_key", ""),
            "reproducer_receipt": failure.get("receipt_id", ""),
            "action_observations": [value["observation"] for value in self._actions.values() if value.get("observation")],
            "owner_pid": os.getpid(), "meaningful_progress_at_ms": int(self._progress_at * 1000),
            "ready_task_ids": cached.get("ready_task_ids", []), "task_statuses": cached.get("task_statuses", {}),
            "native_completion_receipts": cached.get("native_completion_receipts", {}), "native_completion_projection": cached.get("native_completion_projection", ""),
            "native_completion_receipts_by_alias": cached.get("native_completion_receipts_by_alias", {}),
            "native_error": failure, "native_passes": self._passes, "completion_authority": False}
        snapshot.update(loaded_framework_revision=self._loaded_framework_revision, last_successful_native_pass=self._last_successful_pass)
        snapshot.update(native_active_plan_revision=cached.get("native_active_plan_revision", ""),
            native_active_plan_revision_receipt=cached.get("native_active_plan_revision_receipt", ""),
            original_admitted_plan_revision=self.bootstrap_receipt["admitted_plan_revision"],
            typed_goal_semantics_id=self.bootstrap_receipt.get("plan_definition_id", ""),
            last_native_pass_result=self._last_result, native_terminal_result=self._native_terminal_result)
        snapshot.update(native_blocked_tasks=blockers, provider_activity=dict(self._provider_activity), source_reload=dict(self._reload_request))
        for name in ("campaign-status.json", "status.json"):
            atomic_json(self.config.state_root / name, snapshot)
        return snapshot

    def publish_readiness(self) -> None:
        enabled_repair = bool(self.config.repair_command)
        canary = verified_repair_canary(self.config) if enabled_repair else None
        value = {"schema": "wwl.watchdog.ready@1", "run_id": self.config.run_id,
            "policy_id": "worldwide-legal-search-recovery-v1", "repository": str(self.config.site_root),
            "native_revision": git(self.config.framework_root, "rev-parse", "HEAD"),
            "admitted_plan_revision": self.bootstrap_receipt["admitted_plan_revision"],
            "readiness_receipt": self.bootstrap_receipt["receipt_id"], "required_task_ids": list(self.tasks),
            "bindings": {"task_source": True, "reconcile": True, "lifecycle_restart": False,
                "route_policy": True, "independent_validation": True, "validated_promotion": True,
                "incident_deduplication": True, "meaningful_progress": True, "repair_dispatch": enabled_repair,
                "scoped_repair_admission": enabled_repair, "exclusive_effect_ownership": True,
                "repair_canary_verified": bool(canary),
                "rescue_planner": False, "rescue_orchestrator": False},
            "control_canary_verified": False, "bootstrap_only": True}
        if canary:
            pipeline = {"schema": "worldwide-legal-search.repair-pipeline@1", "policy_id": value["policy_id"], "native_revision": value["native_revision"],
                "native_canary_receipt": canary["receipt_id"], "pipeline_binding": canary["pipeline_binding"],
                "native_completion_receipt": canary["native_completion_receipt"], "native_projection": canary["native_projection"]}
            pipeline["receipt_id"] = identity(pipeline)
            value["repair_pipeline"] = pipeline
        value.update(native_active_plan_revision=self._cached.get("native_active_plan_revision", ""),
            native_active_plan_revision_receipt=self._cached.get("native_active_plan_revision_receipt", ""),
            original_admitted_plan_revision=self.bootstrap_receipt["admitted_plan_revision"])
        atomic_json(self.config.state_root / "campaign-readiness.json", value)

    def _run_pass(self) -> None:
        self._provider_callback_entered = False
        try:
            self.capture()
            self._retained_attempt_before_pass = bool(self._daemon.list_running_attempts())
            self._reconcile_controls()
            self._last_result = dict(self._daemon.run_once())
            self.capture()
            implementation = self._last_result.get("implementation_result") or {}
            selected = str(self._last_result.get("active_task_id") or implementation.get("task_alias") or "")
            if implementation.get("status") in {"blocked", "failed"}:
                self._native_terminal_result = {**dict(implementation), "task_alias": selected}
            elif self._cached.get("native_blocked_tasks"):
                self._native_terminal_result = dict(self._cached["native_blocked_tasks"][0])
            elif self._native_terminal_result.get("task_alias") not in self._cached.get("task_statuses", {}) or self._cached["task_statuses"].get(self._native_terminal_result.get("task_alias")) != "blocked":
                self._native_terminal_result = {}
            if selected and selected in self._cached.get("verified_task_ids", []):
                self._last_successful_pass = {"owner_pid": os.getpid(), "framework_revision": self._loaded_framework_revision,
                    "completed_at": time.time(), "result": self._last_result, "authority_revision": self._cached.get("authority_revision"),
                    "progress_token": self._cached.get("progress_token"), "receipt_id": identity(self._last_result)}
                if (self._candidate_request_path is not None
                        and selected == self._read_candidate_request()["task_id"]):
                    native_receipt = self._cached.get("native_completion_receipts_by_alias", {}).get(selected)
                    if not native_receipt:
                        raise ValueError("candidate completion lacks its native receipt")
                    pointer = self.config.state_root / "control/pending-candidate-continuation.json"
                    if pointer.is_file():
                        saved = read_json(pointer)
                        if saved.get("request_path") != str(self._candidate_request_path) or saved.get("request_sha256") != hashlib.sha256(self._candidate_request_path.read_bytes()).hexdigest():
                            raise ValueError("candidate continuation pointer changed before completion")
                        atomic_json(self.config.state_root / "control/candidate-continuation-completed.json",
                            {"schema": "wwl.candidate-continuation-completed@1", "request": saved,
                             "task_id": selected, "native_completion_receipt": native_receipt,
                             "native_projection": self._cached.get("native_completion_projection"),
                             "paid_attempt_budget_reset": False, "completion_authority": False})
                        pointer.unlink()
                    self._last_failure = {}
                    self._candidate_completed = True
                    self.stop.set()  # Reopen the unsliced native task frontier.
            previous_task = str(self._last_failure.get("task_id") or "")
            if self._last_failure and (previous_task in self._cached.get("verified_task_ids", []) or (not previous_task and not self._cached.get("active_task_id"))):
                self._last_failure = {}
            self._check_accepted_source_reload(selected)
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
            command = getattr(error, "cmd", ())
            if (isinstance(error, subprocess.TimeoutExpired) and self._candidate_request_path is not None
                    and isinstance(command, (list, tuple)) and "worktree" in command
                    and "add" in command):
                # The service cgroup must drain before Native may fence the
                # dead PREPARING owner and admit its sole fresh-checkout retry.
                self._candidate_checkout_restart = True
                self.stop.set()
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
                unknown = self._last_failure.get("effects_unknown", False) or any(item.get("effects_unknown") for item in self._cached.get("native_blocked_tasks", []))
                evidence["native_completion_projection"] = self._cached.get("native_completion_projection", "")
                evidence["effects_unknown"] = bool(unknown)
                action["reconciliation_evidence"] = evidence
                outcome = "succeeded" if not unknown else "unknown"
                action["observation"] = {"schema": "wwl.watchdog.action-observation@1", "action_id": action_id,
                    "status": outcome, "receipt": identity(evidence), "effects_verified": not unknown}
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
            if re.fullmatch(r"[0-9a-f]{64}", action_id) is None or path.stem != action_id or request.get("run_id") != self.config.run_id or request.get("schema") != "wwl.watchdog.action@1":
                continue
            if action_id in self._actions:
                continue
            kind = request.get("kind")
            receipt = {"schema": "wwl.watchdog.action-result@1", "action_id": action_id, "status": "blocked", "receipt": identity(request)}
            reason = "native operation is not currently bound"
            policy_id = read_json(self.config.plan_path.with_name("watchdog.policy.json"))["policy_id"]
            stale = request.get("policy_id") != policy_id or request.get("authority_revision") != str(self._cached.get("authority_revision") or "bootstrap") or request.get("accepted_tree") != identity(observed_forest(self.config))
            held = any((self.config.state_root / name).exists() for name in ("HOLD", "STOP", "OPERATOR_STOP"))
            issued = request.get("issued_at")
            expired = type(issued) not in (int, float) or not 0 <= time.time() - issued <= 120
            if stale or held or expired:
                reason = "native action is stale, held or outside its owner admission window"
            elif kind == "reconcile":
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
                try:
                    admit_repair(self.config, path)
                except (OSError, ValueError) as error:
                    reason = "repair admission refused: " + str(error)[:500]
                else:
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
                if action["observation"].get("requires_fenced_reload"):
                    continue
                if action.get("repair_started"):
                    # A restarted owner cannot prove a prior dispatcher did
                    # not launch. Preserve the action's identity and uncertainty.
                    action["observation"] = {"schema": "wwl.watchdog.action-observation@1", "action_id": action_id,
                        "status": "unknown", "receipt": identity({"action": action_id, "prior_dispatch_requires_reconciliation": True}), "effects_verified": False}
                    continue
                # The host binds the native bounded dispatcher, which runs
                # repair-once below; its exit code is never repair evidence.
                action["repair_started"] = time.time()
                atomic_json(self.config.state_root / "control/action-ledger.json", self._actions)
                self._repair_thread = threading.Thread(target=self._execute_repair, args=(action_id, action), name="wwl-native-repair", daemon=False)
                self._repair_thread.start()
        atomic_json(self.config.state_root / "control/action-ledger.json", self._actions)

    def _execute_repair(self, action_id: str, action: dict[str, Any]) -> None:
        launched = False
        try:
            request_path = self.config.state_root / "control/requests" / (action_id + ".json")
            admit_repair(self.config, request_path)
            launched = True
            self._dispatch_repair(action_id, action, request_path)
        except Exception as error:
            evidence = {"schema": "wwl.native-repair-dispatch-failure@1", "action_id": action_id,
                "error_class": type(error).__name__, "reason": str(error)[:1500], "dispatch_may_have_started": launched}
            atomic_json(self.config.state_root / "repairs" / action_id / "dispatch-failure.json", evidence)
            action["observation"] = {"schema": "wwl.watchdog.action-observation@1", "action_id": action_id,
                "status": "unknown" if launched else "blocked", "receipt": identity(evidence), "effects_verified": not launched}
        finally:
            # Cancellation does not establish whether a provider/effect was
            # absent. The durable observation above retains that uncertainty.
            self.cancel.clear()
            atomic_json(self.config.state_root / "control/action-ledger.json", self._actions)

    def _dispatch_repair(self, action_id: str, action: dict[str, Any], request_path: Path) -> None:
        from ..rescue.repair_dispatcher import Config, run_worker

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
        passed = bool(evidence.get("native_completion_receipt") and evidence.get("native_projection") and evidence.get("validated") is True
            and evidence.get("action_id") == action_id and evidence.get("action_request_identity") == identity(action["request"])
            and evidence.get("framework_head") == git(self.config.framework_root, "rev-parse", "HEAD")
            and evidence.get("framework_tree") == git(self.config.framework_root, "rev-parse", "HEAD^{tree}"))
        action["observation"] = {"schema": "wwl.watchdog.action-observation@1", "action_id": action_id,
            "status": "running" if passed else "unknown", "receipt": identity({"dispatcher": result, "native_evidence": evidence}),
            "effects_verified": passed, "candidate_promotion_verified": passed, "recovery_verified": False,
            **({"requires_fenced_reload": True, "expected_framework_revision": evidence["framework_head"],
                "original_failure_receipt": action["request"]["reproducer_receipt"], "deadline_at": min(time.time() + 300, action["result"]["deadline_at"])} if passed else {})}
        if passed:
            self._last_failure = {}
            # The registered external lifecycle controller must fence/reload
            # this lane and prove its original fault cleared. No loaded old
            # module is allowed to resume tasks after source promotion.
            self.stop.set()
        atomic_json(self.config.state_root / "control/action-ledger.json", self._actions)

    def _repair_requires_reconciliation(self) -> bool:
        for action in self._actions.values():
            if action.get("request", {}).get("kind") != "repair" or not action.get("repair_started"):
                continue
            observation = action.get("observation", {})
            if observation.get("status") == "unknown":
                return True
            if observation.get("status") == "running" and (not observation.get("requires_fenced_reload") or observation.get("expected_framework_revision") != self._loaded_framework_revision):
                return True
        return False

    def _check_accepted_source_reload(self, task_alias: str) -> None:
        """Reload changed framework code only behind actual native acceptance."""
        task = self.tasks.get(task_alias)
        receipt = self._cached.get("native_completion_receipts_by_alias", {}).get(task_alias)
        if not task or task["repository"] != "accelerate" or not receipt or task_alias not in self._cached.get("verified_task_ids", []):
            return
        if self._last_failure.get("effects_unknown") or self._repair_requires_reconciliation():
            return
        current = repair_pipeline_binding(self.config)
        if not self._loaded_pipeline or current["source_sha256"] == self._loaded_pipeline["source_sha256"]:
            return
        request = {"schema": "wwl.native-accepted-source-reload@1", "run_id": self.config.run_id, "task_id": task_alias,
            "native_completion_receipt": receipt, "native_completion_projection": self._cached.get("native_completion_projection", ""),
            "loaded_framework_revision": self._loaded_framework_revision,
            "expected_framework_revision": git(self.config.framework_root, "rev-parse", "HEAD"),
            "expected_framework_tree": git(self.config.framework_root, "rev-parse", "HEAD^{tree}"),
            "old_pipeline_binding": self._loaded_pipeline, "new_pipeline_binding": current,
            "exit_code": 75, "native_history_preserved": True, "completion_authority": False}
        request["receipt_id"] = identity(request)
        atomic_json(self.config.state_root / "accepted-source-reload.json", request)
        self._reload_request = request
        self.stop.set()

    def run(self, *, once: bool = False) -> dict[str, Any]:
        observation = wait_for_storage(self.config, self.stop)
        if self.stop.is_set():
            return {"schema": "wwl.native-runtime-stopped@1", "resource_pressure": observation, "completion_authority": False}
        self.open()
        self.capture()
        self.publish_readiness()
        try:
            while not self.stop.is_set():
                if storage_observation(self.config)["blocked"]:
                    wait_for_storage(self.config, self.stop)
                    continue
                self._maybe_refresh_repair_canary()
                self.process_requests()
                self.settle_controls()
                hold = any((self.config.state_root / name).exists() for name in ("HOLD", "STOP", "OPERATOR_STOP"))
                repairing = self._repair_thread is not None and self._repair_thread.is_alive()
                repair_fence = self._repair_requires_reconciliation()
                if repair_fence and self._pending_controls and (self._worker is None or not self._worker.is_alive()):
                    self._worker = threading.Thread(target=self._reconcile_controls, name="wwl-native-reconciliation-owner", daemon=False)
                    self._worker.start()
                if not hold and not repairing and not repair_fence and (self._worker is None or not self._worker.is_alive()):
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
            if self._repair_thread is not None:
                self._repair_thread.join(timeout=60)
                if self._repair_thread.is_alive():
                    raise RuntimeError("native repair worker has not drained; refusing DB ownership release")
            return self.publish(alive=False)
        finally:
            self._stop_repair_canary_refresh()
            if self._worker is None or not self._worker.is_alive():
                self._daemon.close()


class _ProgressCallbacks:
    def __init__(self, runtime: CampaignRuntime, bridge: Any):
        self.runtime, self.bridge = runtime, bridge
        self.attempt_root, self.max_task_attempts = bridge.attempt_root, bridge.max_task_attempts

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
    result.add_argument("command", choices=("bootstrap", "run", "status", "validate-goal", "repair-once", "continue-candidate"))
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
            waiting_stop = threading.Event()
            signal.signal(signal.SIGTERM, lambda *_args: waiting_stop.set())
            signal.signal(signal.SIGINT, lambda *_args: waiting_stop.set())
            observation = wait_for_storage(config, waiting_stop)
            if waiting_stop.is_set():
                print(json.dumps({"schema": "wwl.native-runtime-stopped@1", "resource_pressure": observation, "completion_authority": False}), flush=True)
                return 0
            with owner_lock(config):
                receipt = bootstrap(config, already_locked=True)
                runtime = CampaignRuntime(config, receipt)
                if args.command == "continue-candidate":
                    if not args.request:
                        raise ValueError("continue-candidate requires an immutable native candidate request")
                    from .worldwide_legal_search_migration import _registered_stopped_fence
                    fence = _registered_stopped_fence(config)
                    if fence.get("owner_alive") is not False or fence.get("main_pid") != 0 or fence.get("cgroup_empty") is not True:
                        raise ValueError("candidate continuation requires the registered previous owner stopped and fenced")
                    runtime._candidate_request_path = args.request
                    runtime._candidate_request_sha256 = hashlib.sha256(args.request.read_bytes()).hexdigest()
                    runtime._read_candidate_request()
                    request_bytes = args.request.read_bytes()
                    pending = config.state_root / "control/pending-candidate-continuation.json"
                    pointer = {"schema": "wwl.pending-candidate-continuation@1", "run_id": config.run_id,
                        "request_path": str(args.request), "request_sha256": hashlib.sha256(request_bytes).hexdigest()}
                    if pending.exists() and read_json(pending) != pointer:
                        raise ValueError("another candidate continuation is already pending")
                    atomic_json(pending, pointer)
                def request_stop(_signum: int, _frame: Any) -> None:
                    runtime.cancel.set()
                    runtime.stop.set()
                signal.signal(signal.SIGTERM, request_stop)
                signal.signal(signal.SIGINT, request_stop)
                output = runtime.run(once=args.once or args.command == "continue-candidate")
                if runtime._candidate_checkout_restart or (runtime._candidate_completed and args.command == "run"):
                    print(json.dumps(output, sort_keys=True, default=str), flush=True)
                    return 75
                if args.command == "continue-candidate" and not runtime._candidate_completed:
                    print(json.dumps(output, sort_keys=True, default=str), flush=True)
                    return 1
                if runtime._reload_request:
                    print(json.dumps(output, sort_keys=True, default=str), flush=True)
                    return 75
        print(json.dumps(output, sort_keys=True, default=str), flush=True)
        return 0
    except Exception as error:
        print(json.dumps({"schema": "wwl.native-runtime-error@1", "error_class": type(error).__name__, "error": str(error), "completion_authority": False}), file=sys.stderr)
        return 1


def admit_repair(config: CampaignConfig, request_path: Path) -> dict[str, Any]:
    """Resolve a current owner failure and bind exact allowed repair paths."""
    from .worldwide_legal_search_watchdog import read_owned_json

    if storage_observation(config)["blocked"]:
        raise ValueError("repair admission is blocked by resource quota")

    request = read_owned_json(request_path)
    action_id = str(request.get("action_id") or "")
    if request.get("schema") != "wwl.watchdog.action@1" or request.get("kind") != "repair" or request.get("run_id") != config.run_id or re.fullmatch(r"[0-9a-f]{64}", action_id) is None or not request.get("reproducer_receipt"):
        raise ValueError("repair task requires an admitted campaign action and reproducer")
    if request_path.resolve() != config.state_root / "control/requests" / (action_id + ".json"):
        raise ValueError("repair request is outside the immutable campaign control queue")
    status = read_owned_json(config.state_root / "campaign-status.json")
    if status.get("run_id") != config.run_id or status.get("hold") or status.get("effects_unknown") is not False or status.get("authority_revision") != request.get("authority_revision"):
        raise ValueError("repair action no longer matches current canonical campaign authority")
    if status.get("native_error", {}).get("receipt_id") != request.get("reproducer_receipt"):
        raise ValueError("repair reproducer is an archived or resolved failure")
    issued = request.get("issued_at")
    if type(issued) not in (int, float) or not 0 <= time.time() - issued <= config.task_timeout_seconds + 120:
        raise ValueError("repair action is outside its bounded admission window")
    policy = read_owned_json(config.plan_path.with_name("watchdog.policy.json"))
    if request.get("policy_id") != policy["policy_id"] or request.get("accepted_tree") != identity(observed_forest(config)):
        raise ValueError("repair action policy or accepted repository forest is stale")
    failures = list((config.state_root / "failures").glob("*.json"))
    if len(failures) > 256:
        raise ValueError("repair failure lookup exceeds the admitted bound")
    matches = [value for path in failures if (value := read_owned_json(path)).get("receipt_id") == request["reproducer_receipt"]]
    if len(matches) != 1:
        raise ValueError("repair reproducer receipt is missing or ambiguous")
    failure = matches[0]
    body = {key: value for key, value in failure.items() if key != "receipt_id"}
    if identity(body) != failure["receipt_id"] or failure.get("schema") != "wwl.native-failure@1" or failure.get("run_id") != config.run_id:
        raise ValueError("repair reproducer integrity or campaign identity is invalid")
    if failure.get("effects_unknown") is not False or failure.get("source_revision") != git(config.framework_root, "rev-parse", "HEAD"):
        raise ValueError("repair requires reconciled effects and the current supervisor source")
    if failure != status.get("native_error"):
        raise ValueError("repair current error and durable reproducer differ")
    if not failure.get("traceback") or not failure.get("reproduce_argv") or not failure.get("source_frames"):
        raise ValueError("repair reproducer lacks a native traceback and source binding")
    patterns = policy["repair_scope"]["allowed_paths"]
    protected = set(read_json(config.plan_path)["execution"]["protected_acceptance_inputs"])
    owners = []
    for frame in failure["source_frames"]:
        relative = str(frame.get("path") or "")
        target = config.framework_root / relative
        if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts or not target.is_file() or target.is_symlink():
            raise ValueError("repair traceback contains an unsafe source path")
        if not any(fnmatch.fnmatchcase(relative, pattern) for pattern in patterns) or relative in protected or relative.startswith("test/"):
            # A wrapper on the allowed list cannot authorize changing an
            # unadmitted innermost cause or a protected test criterion.
            raise ValueError("repair traceback source lies outside the admitted production scope")
        expected = str(frame.get("source_sha256") or "")
        current = hashlib.sha256(target.read_bytes()).hexdigest()
        committed = subprocess.run(["git", "-C", str(config.framework_root), "show", "HEAD:" + relative], capture_output=True, timeout=30, check=False)
        if not expected or expected != current or committed.returncode or hashlib.sha256(committed.stdout).hexdigest() != expected:
            raise ValueError("repair traceback source changed after failure admission")
        owners.append(relative)
    owners = sorted(set(owners))
    regression = "test/api/test_worldwide_legal_search_repair_" + action_id[:16] + ".py"
    if (config.framework_root / regression).exists() or not any(fnmatch.fnmatchcase(regression, pattern) for pattern in patterns):
        raise ValueError("repair regression must be a new action-specific admitted path")
    focused = {"test/api/test_worldwide_legal_search_runtime.py", "test/api/test_worldwide_legal_search_provider.py", "test/api/test_worldwide_legal_search_goals.py"}
    for owner in owners:
        stem = Path(owner).stem
        for candidate in ("test/api/test_agent_supervisor_" + stem + ".py", "test/api/test_" + stem + ".py"):
            if (config.framework_root / candidate).is_file():
                focused.add(candidate)
    focused = sorted(path for path in focused if (config.framework_root / path).is_file())
    protected.update(focused)
    protected.update(path for path in ("pytest.ini", "test/pytest.ini", "test/conftest.py", "test/api/conftest.py") if (config.framework_root / path).is_file())
    evidence = "artifacts/worldwide-legal-search/repairs/" + action_id + ".json"
    protected_hashes = {path: hashlib.sha256((config.framework_root / path).read_bytes()).hexdigest() for path in sorted(protected) if (config.framework_root / path).is_file()}
    return {"schema": "wwl.native-repair-admission@1", "request": request, "failure": failure,
        "source_revision": failure["source_revision"], "source_paths": owners, "regression_path": regression,
        "focused_tests": focused, "protected_paths": sorted(protected), "evidence_path": evidence,
        "protected_hashes": protected_hashes, "allowed_paths": [*owners, regression, evidence]}


def repair_once(config: CampaignConfig, request_path: Path) -> dict[str, Any]:
    """Execute a separate operational repair task through all native gates."""
    admission = admit_repair(config, request_path)
    request, failure = admission["request"], admission["failure"]
    action_id = request["action_id"]
    state = config.state_root / "repairs" / action_id
    task_id = "WWL-REPAIR-" + action_id[:16]
    source = read_json(config.plan_path)
    validation_binding = {key: admission[key] for key in ("source_revision", "allowed_paths", "regression_path", "focused_tests", "protected_hashes")}
    validation_binding.update(action_id=action_id, baseline_inputs=[path for path in admission["protected_paths"] if path in admission["protected_hashes"]])
    task = {"id": task_id, "title": "Repair the reproduced native supervisor defect", "status": "todo", "completion": "auto",
        "goal_id": "WWL-G000", "subgoal_id": "WWL-G000", "root_goal_id": "WWL-G000", "repository": "accelerate", "depends_on": [],
        "outputs": [*admission["source_paths"], admission["regression_path"]],
        "evidence_output": admission["evidence_path"], "write_scope": admission["allowed_paths"],
        "acceptance": "Reproduce and fix exactly the observed native defect, preserving routing, ownership, validation and completion gates. Add the new action-specific regression and prove it fails on the bound baseline and passes after the fix. Existing tests, routing policies and acceptance criteria are read-only. Native failure evidence: " + json.dumps(failure, sort_keys=True),
        "validation_argv": [[config.python_executable, "-I", "-c", "exec(" + repr(REPAIR_VALIDATOR) + ")", json.dumps(validation_binding, sort_keys=True)]],
        "bounds": {"max_attempts": 2, "wall_seconds_per_attempt": config.task_timeout_seconds, "validation_reserve_percent": 25}}
    plan = {**source, "namespace": config.run_id + "-repair-" + action_id[:16], "tasks": [task], "operational_repair_binding": admission,
        "execution": {**source["execution"], "protected_acceptance_inputs": admission["protected_paths"]},
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
    native = result.get("native_completion_receipts_by_alias", {})
    validated = task_id in result.get("verified_task_ids", []) and bool(native.get(task_id)) and bool(result.get("native_completion_projection"))
    output = {"schema": "wwl.native-repair-result@1", "action_id": action_id, "validated": validated,
        "action_request_identity": identity(request), "original_failure_receipt": failure["receipt_id"], "baseline_revision": admission["source_revision"],
        "native_completion_receipt": native.get(task_id, "") if validated else "",
        "native_projection": result.get("native_completion_projection"), "framework_head": git(config.framework_root, "rev-parse", "HEAD"),
        "framework_tree": git(config.framework_root, "rev-parse", "HEAD^{tree}"), "completion_authority": False}
    atomic_json(state / "native-repair-result.json", output)
    return output


if __name__ == "__main__":
    raise SystemExit(main())
