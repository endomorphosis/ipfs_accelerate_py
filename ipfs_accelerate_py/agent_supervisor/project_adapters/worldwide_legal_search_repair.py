"""Isolated supervisor repair and validated promotion.

A reproduced native defect becomes one repair task. The task id is the durable
incident, so a new PID or attempt does not open a second writer. Grok is bound
through ``ipfs_accelerate_py.llm_router`` and may edit only a campaign-owned
repair worktree, and only the new reproducer plus the reproduced source frames.
Regressions and the canary run before a native merge. Promotion then reloads
and fences the lane. Rollback evidence stays with the incident. Nothing here
grants completion authority or launches a hosted model.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .worldwide_legal_search import (
    CANONICAL_ROUTING,
    PRIMARY_MODEL,
    PRIMARY_PROVIDER,
    ROUTE_POLICY,
    ROUTE_SCHEMA,
    ROUTER_NAME,
    WorldwideRouteError,
    bind_explicit_agent_route,
    require_canonical_routing,
)
from .worldwide_legal_search_progress import (
    DEFAULT_LIMITS,
    ProgressIdentityError,
    ProgressLedger,
    canonical_stage,
    digest,
    incident_identity,
)

SCHEMA = "wwl.repair.decision@1"
STATE_SCHEMA = "wwl.repair-state@1"
ROLLBACK_SCHEMA = "wwl.repair.rollback@1"
FAILURE_SCHEMA = "wwl.native-failure@1"
WORKTREE_SCHEMA = "wwl.repair.worktree@1"
REGRESSION_SCHEMA = "wwl.repair.regression@1"
CANARY_SCHEMA = "wwl.repair.canary@1"
REPAIR_LIMIT = DEFAULT_LIMITS["repair"]
PROVIDER_LIMIT = DEFAULT_LIMITS["provider"]
VALIDATION_LIMIT = DEFAULT_LIMITS["validation"]
MERGE_LIMIT = DEFAULT_LIMITS["merge"]
CAMPAIGN_REPAIR_CAP = 8
CANONICAL_PYTHON = "/usr/bin/python3.12"
HOLD_MARKERS = ("HOLD", "STOP", "OPERATOR_STOP")
BUDGET_NAMES = ("repair", "provider", "validation", "merge")
OWNER_MARKER = ".wwl-repair-owner.json"
DISPOSITIONS = (
    "reloaded",
    "duplicate_writer",
    "already_promoted",
    "effects_unknown",
    "held",
    "deadline_exceeded",
    "repair_budget_exhausted",
    "campaign_repair_cap",
    "scope_refused",
    "worktree_refused",
    "route_refused",
    "reproducer_required",
    "regressions_failed",
    "canary_failed",
    "promotion_refused",
    "reload_failed",
    "reload_not_pending",
    "validation_unbound",
    "promotion_unbound",
    "reload_unbound",
)
# Admitted isolated scope from the campaign watchdog policy. Exact paths still
# have to be named by the reproduced frames or be the new reproducer.
ALLOWED_PATTERNS = (
    "ipfs_accelerate_py/agent_supervisor/project_adapters/worldwide_legal_search*.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
    "ipfs_accelerate_py/agent_supervisor/rescue/repair_dispatcher.py",
    "ipfs_accelerate_py/agent_supervisor/rescue/supervisor_watchdog.py",
    "ipfs_accelerate_py/agent_supervisor/rescue/campaign_watchdog.py",
    "ipfs_accelerate_py/llm_router.py",
    "test/api/test_worldwide_legal_search*.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/database_portal_bridge.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon_runner.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/dependency_preflight_retry.py",
    "ipfs_accelerate_py/agent_supervisor/validation/project_dependency_preflight.py",
    "ipfs_accelerate_py/agent_supervisor/validation/validation_commands.py",
    "ipfs_accelerate_py/agent_supervisor/validation/validation_runtime.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/fenced_candidate_retirement.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/native_validation_failure_retry.py",
)
PROTECTED_EXACT = frozenset({
    "watchdog.policy.json",
    "docs/worldwide/product-acceptance.json",
})
PROTECTED_PREFIXES = (
    "implementation_plan/",
    "data/laws/",
    "docs/worldwide/laws/",
)
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_HEX_SHA = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_EDIT = re.compile(
    r"completion_authority[\"']?\s*[:=]\s*true|self_certified_completion[\"']?\s*[:=]\s*true|pytest\.(?:skip|xfail)|mark\.skip",
    re.IGNORECASE,
)
_REQUEST_FIELDS = frozenset({
    "action_id", "observed_at", "deadline_at", "phase", "hold", "hold_markers",
    "failure", "campaign_root", "live_checkout", "plan", "provider",
    "python_executable", "edits", "completion_authority",
})
_FAILURE_FIELDS = frozenset({
    "schema", "run_id", "task_id", "stage", "defect", "effects_unknown", "source_revision",
    "reproducer", "source_frames", "baseline_files", "receipt_id", "pid", "attempt",
})
_RESET_FIELDS = frozenset({
    "reset", "reset_budget", "reset_budgets", "reset_progress", "refund", "budgets_reset", "progress_reset",
})
_MAX_CHECKPOINT = 1_048_576
_MAX_FILE = 262_144
_MAX_FILES = 32
RouteBinder = Callable[..., Mapping[str, Any]]
StageRunner = Callable[[Mapping[str, Any]], Mapping[str, Any]]


class RepairContractError(ValueError):
    """Repair request or checkpoint violates the isolated repair contract."""


def repair_task_identity(*, run_id: str, task_id: str, stage: str, defect: Any = "") -> str:
    """Incident id for a repair. PID and attempt are not part of it."""
    return incident_identity(run_id=run_id, task_id=task_id, stage=stage, defect=defect)


def admitted_repair_task_id(incident_id: str) -> str:
    """Stable native repair task id. Sixteen hex digits match the provider grammar."""
    checked = _digest(incident_id, "incident_id")
    return "WWL-REPAIR-" + checked.split(":", 1)[1][:16]


def path_is_protected(path: str) -> bool:
    """Credentials, release criteria, policy and completion code stay outside repair."""
    if not _relative_ok(path):
        return True
    name = path.split("/")[-1].lower()
    if path in PROTECTED_EXACT or name in PROTECTED_EXACT or "completion" in name:
        return True
    if any(path.startswith(prefix) for prefix in PROTECTED_PREFIXES):
        return True
    for part in path.split("/"):
        lowered = part.lower()
        if lowered in {"secret", "secrets"} or lowered.startswith(".env") or "credential" in lowered:
            return True
    return False


def source_path_admitted(path: str) -> bool:
    """A reproduced frame is editable only when the campaign scope names it."""
    if not _relative_ok(path) or path_is_protected(path) or path.startswith("test/"):
        return False
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in ALLOWED_PATTERNS)


def reproducer_path_admitted(path: str) -> bool:
    """The new reproducer is the only test file a repair may add."""
    if not _relative_ok(path) or path_is_protected(path):
        return False
    return fnmatch.fnmatchcase(path, "test/api/test_worldwide_legal_search*.py")


def bind_repair_route(
    plan: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    workspace: Path | str,
    task_id: str,
    attempt: int,
    python_executable: str = CANONICAL_PYTHON,
) -> dict[str, Any]:
    """Bind the canonical Grok agent argv. This does not launch a provider."""
    binding = bind_explicit_agent_route(
        plan,
        config,
        workspace=workspace,
        task_id=task_id,
        attempt=attempt,
        python_executable=python_executable,
    )
    _assert_route(binding, workspace=Path(workspace), task_id=task_id, attempt=attempt)
    return dict(binding)


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise RepairContractError(f"{name} must be a bounded durable identifier")
    return value


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value.strip().lower()) is None:
        raise RepairContractError(f"{name} must be a sha256 digest")
    return value.strip().lower()


def _commit(value: Any, name: str) -> str:
    if not isinstance(value, str) or _COMMIT.fullmatch(value.strip().lower()) is None:
        raise RepairContractError(f"{name} must be a git commit")
    return value.strip().lower()


def _timestamp(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RepairContractError(f"{name} must be a finite timestamp")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")) or number < 0:
        raise RepairContractError(f"{name} must be a finite timestamp")
    return number


def _whole(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise RepairContractError(f"{name} must be a non-negative integer")
    return value


def _relative_ok(path: str) -> bool:
    if not isinstance(path, str) or not path or len(path) > 512:
        return False
    if path.startswith("/") or "\\" in path or "\x00" in path:
        return False
    return all(part not in {"", ".", ".."} for part in path.split("/"))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _content_sha(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _frame_sha(value: Any) -> str:
    if not isinstance(value, str):
        raise RepairContractError("source frame digest must be a sha256 digest")
    text = value.strip().lower()
    if text.startswith("sha256:"):
        text = text.split(":", 1)[1]
    if _HEX_SHA.fullmatch(text) is None:
        raise RepairContractError("source frame digest must be a sha256 digest")
    return text


def _directory(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RepairContractError(f"{name} must be an absolute directory")
    path = Path(value)
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise RepairContractError(f"{name} must be an absolute directory")
    return path


def _reject_completion(value: Mapping[str, Any]) -> None:
    if value.get("completion_authority") is True:
        raise RepairContractError("repair cannot grant completion authority")
    present = _RESET_FIELDS.intersection(value)
    if any(value.get(field) not in (False, None) for field in present):
        raise RepairContractError("repair cannot reset a durable budget")


def _markers(value: Any) -> tuple[str, ...]:
    if value in (None, []):
        return ()
    if not isinstance(value, list) or len(value) > 8:
        raise RepairContractError("hold marker is not recognized")
    markers = []
    for item in value:
        if item not in HOLD_MARKERS or item in markers:
            raise RepairContractError("hold marker is not recognized")
        markers.append(item)
    return tuple(markers)


def _assert_route(binding: Mapping[str, Any], *, workspace: Path, task_id: str, attempt: int) -> None:
    if not isinstance(binding, Mapping):
        raise RepairContractError("repair route is not the canonical grok agent binding")
    argv = binding.get("argv")
    if (
        binding.get("schema") != ROUTE_SCHEMA
        or binding.get("router") != ROUTER_NAME
        or binding.get("primary_provider") != PRIMARY_PROVIDER
        or binding.get("primary_model") != PRIMARY_MODEL
        or binding.get("fallback_policy") != ROUTE_POLICY
        or binding.get("launched") is not False
        or binding.get("paid_provider_calls") != 0
        or binding.get("hosted_model_invocations") != 0
        or binding.get("completion_authority") is not False
        or binding.get("task_id") != task_id
        or binding.get("attempt") != attempt
        or not isinstance(argv, list)
        or "--dangerously-bypass-approvals-and-sandbox" in argv
        or "--primary-unavailable-kind" in argv
    ):
        raise RepairContractError("repair route is not the canonical grok agent binding")
    try:
        primary = json.loads(argv[argv.index("--primary-command-json") + 1])
        fallback = json.loads(argv[argv.index("--fallback-command-json") + 1])
    except (ValueError, IndexError, TypeError) as error:
        raise RepairContractError("repair route is not the canonical grok agent binding") from error
    if not isinstance(primary, list) or not isinstance(fallback, list):
        raise RepairContractError("repair route is not the canonical grok agent binding")
    if "--dangerously-bypass-approvals-and-sandbox" in primary or "--dangerously-bypass-approvals-and-sandbox" in fallback:
        raise RepairContractError("repair route is not the canonical grok agent binding")
    if "danger-full-access" in fallback or "generate_text" in argv:
        raise RepairContractError("repair route is not the canonical grok agent binding")
    expected = workspace.resolve()
    workspaces = [_flag(argv, "--workspace"), _flag(primary, "--workspace"), _flag(fallback, "-C")]
    if any(path is None or Path(path).resolve() != expected for path in workspaces):
        raise RepairContractError("repair route workspace is not the campaign repair worktree")
    if primary[primary.index("--model") + 1] != PRIMARY_MODEL or primary[primary.index("--mode") + 1] != "agent":
        raise RepairContractError("repair route is not the canonical grok agent binding")
    if Path(str(argv[1])).name != "provider_fallback_runner.py":
        raise RepairContractError("repair route is not the canonical grok agent binding")


def _flag(argv: Sequence[Any], name: str) -> str | None:
    try:
        index = list(argv).index(name)
    except ValueError:
        return None
    if index + 1 >= len(argv):
        return None
    value = argv[index + 1]
    if not isinstance(value, str) or not value:
        return None
    return value


def _child(root: Path, relative: str) -> Path:
    if not _relative_ok(relative):
        raise RepairContractError("repair path escapes the campaign worktree")
    current = root
    parts = relative.split("/")
    for part in parts[:-1]:
        current = current / part
        if current.is_symlink() or (current.exists() and not current.is_dir()):
            raise RepairContractError("repair path escapes the campaign worktree")
    leaf = current / parts[-1]
    if leaf.is_symlink():
        raise RepairContractError("repair path escapes the campaign worktree")
    resolved_root = root.resolve()
    if not leaf.resolve().is_relative_to(resolved_root):
        raise RepairContractError("repair path escapes the campaign worktree")
    return leaf


def _write_bytes(root: Path, relative: str, payload: bytes) -> None:
    if len(payload) > _MAX_FILE:
        raise RepairContractError("repair file exceeds the admission bound")
    destination = _child(root, relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(destination, flags, 0o644)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _snapshot(root: Path) -> dict[str, str]:
    found: dict[str, str] = {}
    resolved_root = root.resolve()
    for path in sorted(resolved_root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            if path.is_symlink():
                raise RepairContractError("repair path escapes the campaign worktree")
            continue
        relative = path.relative_to(resolved_root).as_posix()
        if relative == OWNER_MARKER:
            continue
        found[relative] = _sha256_bytes(path.read_bytes())
        if len(found) > _MAX_FILES + 4:
            raise RepairContractError("repair worktree exceeds the admission bound")
    return found


class RepairSession:
    """One campaign run's isolated repair writers.

    The progress ledger, when bound, is the budget authority for repair,
    provider, validation and merge. A checkpoint reloads spent counts,
    fences and rollback records. It does not open a fresh budget.
    """

    def __init__(
        self,
        run_id: str,
        *,
        ledger: ProgressLedger | None = None,
        route_binder: RouteBinder | None = None,
        regression_runner: StageRunner | None = None,
        canary_runner: StageRunner | None = None,
        merger: StageRunner | None = None,
        reloader: StageRunner | None = None,
    ):
        self.run_id = _identifier(run_id, "run_id")
        if ledger is not None:
            if not isinstance(ledger, ProgressLedger):
                raise RepairContractError("repair ledger must be a progress ledger")
            if ledger.run_id != self.run_id:
                raise RepairContractError("repair run does not match the progress ledger")
            for name in BUDGET_NAMES:
                if ledger.limits.get(name) != DEFAULT_LIMITS[name]:
                    raise RepairContractError("durable repair bound differs from the campaign ledger")
        self.ledger = ledger
        self.route_binder = route_binder or bind_repair_route
        self.regression_runner = regression_runner
        self.canary_runner = canary_runner
        self.merger = merger
        self.reloader = reloader
        self.hold = {"held": False, "marker": ""}
        self.campaign_repairs = 0
        self.repair_charges: dict[str, list[str]] = {}
        self.spent: dict[str, dict[str, int]] = {name: {} for name in BUDGET_NAMES}
        self.tasks: dict[str, dict[str, Any]] = {}
        self._rollbacks: dict[str, list[dict[str, Any]]] = {}
        self._actions: dict[str, dict[str, Any]] = {}
        self._inflight: set[str] = set()
        self._charged_now: dict[str, set[str]] = {name: set() for name in BUDGET_NAMES}
        self.completion_authority = False

    def set_hold(self, held: bool, *, marker: str = "") -> dict[str, Any]:
        """Record or clear a durable hold. Repair does not override it."""
        if type(held) is not bool:
            raise RepairContractError("hold must be a boolean")
        if held:
            if marker not in HOLD_MARKERS:
                raise RepairContractError("hold marker is not recognized")
            self.hold = {"held": True, "marker": marker}
        else:
            if marker not in {"", *HOLD_MARKERS}:
                raise RepairContractError("hold marker is not recognized")
            self.hold = {"held": False, "marker": ""}
        return {
            "schema": SCHEMA,
            "disposition": "held",
            "held": self.hold["held"],
            "hold_marker": self.hold["marker"],
            "applied": False,
            "completion_authority": False,
        }

    def repair(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Admit at most one isolated repair writer for one reproduced defect."""
        normalized = self._normalize(request)
        action_id = normalized["action_id"]
        if action_id in self._actions:
            stored = self._actions[action_id]
            if stored["digest"] != normalized["digest"]:
                raise RepairContractError("action id reused with a different repair request")
            return {**stored["decision"], "deduplicated": True}
        previous_charges = self._charged_now
        self._charged_now = {name: set() for name in BUDGET_NAMES}
        try:
            decision = self._dispatch(normalized)
        finally:
            self._charged_now = previous_charges
        self._actions[action_id] = {"digest": normalized["digest"], "decision": decision}
        return dict(decision)

    def _dispatch(self, normalized: dict[str, Any]) -> dict[str, Any]:
        if normalized["incident_id"] in self._inflight:
            decision = self._refuse(normalized, "duplicate_writer", "a repair writer is already open for this incident")
        elif normalized["phase"] == "reload":
            decision = self._reload_only(normalized)
        else:
            decision = self._repair(normalized)
        return decision

    def rollback_evidence(self, repair_task_id: str) -> list[dict[str, Any]]:
        """Append-only rollback records for one repair task. Entries are not deleted."""
        task_id = _identifier(repair_task_id, "repair_task_id")
        return [dict(item) for item in self._rollbacks.get(task_id, [])]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": STATE_SCHEMA,
            "run_id": self.run_id,
            "repair_limit": REPAIR_LIMIT,
            "campaign_repair_cap": CAMPAIGN_REPAIR_CAP,
            "campaign_repairs": self.campaign_repairs,
            "repair_charges": {key: list(self.repair_charges[key]) for key in sorted(self.repair_charges)},
            "hold": dict(self.hold),
            "spent": {name: {key: self.spent[name][key] for key in sorted(self.spent[name])} for name in BUDGET_NAMES},
            "tasks": {key: _public_task(self.tasks[key]) for key in sorted(self.tasks)},
            "rollbacks": {key: [dict(item) for item in self._rollbacks[key]] for key in sorted(self._rollbacks)},
            "actions": {
                key: {"digest": self._actions[key]["digest"], "decision": dict(self._actions[key]["decision"])}
                for key in sorted(self._actions)
            },
            "ledger": None if self.ledger is None else self.ledger.to_dict(),
            "completion_authority": False,
        }

    def save(self, path: str | Path) -> None:
        """Atomically replace a checkpoint. A crash cannot widen a repair budget."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
        if len(payload) > _MAX_CHECKPOINT:
            raise RepairContractError("repair checkpoint exceeds the admission bound")
        descriptor, temporary = tempfile.mkstemp(prefix=".wwl-repair-", dir=target.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            directory = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except Exception:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise

    @classmethod
    def load(cls, path: str | Path, **runners: Any) -> RepairSession:
        raw = Path(path).read_bytes()
        if len(raw) > _MAX_CHECKPOINT:
            raise RepairContractError("repair checkpoint exceeds the admission bound")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise RepairContractError("repair checkpoint is not JSON") from error
        return cls.from_dict(value, **runners)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], **runners: Any) -> RepairSession:
        if not isinstance(value, Mapping) or value.get("schema") != STATE_SCHEMA:
            raise RepairContractError("repair checkpoint schema is not recognized")
        if value.get("completion_authority") is not False:
            raise RepairContractError("repair checkpoint cannot grant completion authority")
        if value.get("repair_limit") != REPAIR_LIMIT or value.get("campaign_repair_cap") != CAMPAIGN_REPAIR_CAP:
            raise RepairContractError("durable repair bound differs from the campaign ledger")
        ledger_value = value.get("ledger")
        ledger = ProgressLedger.from_dict(ledger_value) if ledger_value is not None else None
        session = cls(str(value.get("run_id") or ""), ledger=ledger, **runners)
        session.campaign_repairs = _whole(value.get("campaign_repairs"), "campaign repairs")
        if session.campaign_repairs > CAMPAIGN_REPAIR_CAP:
            raise RepairContractError("stored campaign repair count exceeds the durable cap")
        charges = value.get("repair_charges")
        if not isinstance(charges, Mapping):
            raise RepairContractError("stored repair budget was rewritten")
        for incident_id, actions in charges.items():
            _digest(incident_id, "incident_id")
            if not isinstance(actions, list) or not all(isinstance(item, str) and _ID.fullmatch(item) for item in actions):
                raise RepairContractError("stored repair budget was rewritten")
            session.repair_charges[incident_id] = list(actions)
        hold = value.get("hold")
        if not isinstance(hold, Mapping) or type(hold.get("held")) is not bool:
            raise RepairContractError("stored hold is malformed")
        marker = hold.get("marker") or ""
        if hold["held"] and marker not in HOLD_MARKERS:
            raise RepairContractError("stored hold marker is not recognized")
        if not hold["held"] and marker not in {"", *HOLD_MARKERS}:
            raise RepairContractError("stored hold marker is not recognized")
        session.hold = {"held": hold["held"], "marker": "" if not hold["held"] else marker}
        spent = value.get("spent")
        if not isinstance(spent, Mapping) or set(spent) != set(BUDGET_NAMES):
            raise RepairContractError("stored repair budget is malformed")
        for name in BUDGET_NAMES:
            rows = spent[name]
            if not isinstance(rows, Mapping):
                raise RepairContractError("stored repair budget is malformed")
            for incident_id, used in rows.items():
                _digest(incident_id, "incident_id")
                session.spent[name][incident_id] = _whole(used, name + " spent")
                if session.spent[name][incident_id] > DEFAULT_LIMITS[name]:
                    raise RepairContractError("stored repair budget exceeds the durable bound")
        tasks = value.get("tasks")
        if not isinstance(tasks, Mapping):
            raise RepairContractError("stored repair tasks are malformed")
        for incident_id, task in tasks.items():
            _digest(incident_id, "incident_id")
            session.tasks[incident_id] = _restore_task(incident_id, task)
        rollbacks = value.get("rollbacks")
        if not isinstance(rollbacks, Mapping):
            raise RepairContractError("stored rollback evidence is malformed")
        for task_id, records in rollbacks.items():
            _identifier(task_id, "repair_task_id")
            if not isinstance(records, list) or not records:
                raise RepairContractError("stored rollback evidence is malformed")
            session._rollbacks[task_id] = [_restore_rollback(item) for item in records]
        for task in session.tasks.values():
            if task["state"] != "open" and not session._rollbacks.get(task["repair_task_id"]):
                raise RepairContractError("stored repair dropped its rollback evidence")
        actions = value.get("actions")
        if not isinstance(actions, Mapping):
            raise RepairContractError("stored repair actions are malformed")
        for action_id, payload in actions.items():
            _identifier(action_id, "action_id")
            if not isinstance(payload, Mapping) or payload.get("decision", {}).get("completion_authority") is not False:
                raise RepairContractError("stored repair action cannot grant completion authority")
            session._actions[action_id] = {"digest": _digest(payload.get("digest"), "action digest"), "decision": dict(payload["decision"])}
        charged = sum(len(actions) for actions in session.repair_charges.values())
        if charged != session.campaign_repairs:
            raise RepairContractError("stored repair budget was rewritten")
        for incident_id, actions in session.repair_charges.items():
            if len(actions) != session.spent["repair"].get(incident_id, 0):
                raise RepairContractError("stored repair budget was rewritten")
        for incident_id, used in session.spent["repair"].items():
            if len(session.repair_charges.get(incident_id, [])) != used:
                raise RepairContractError("stored repair budget was rewritten")
        session._require_ledger_alignment()
        return session

    def _repair(self, normalized: dict[str, Any]) -> dict[str, Any]:
        refusal = self._precheck(normalized)
        if refusal is not None:
            return refusal
        self._charge(normalized["incident_id"], "repair")
        self.repair_charges.setdefault(normalized["incident_id"], []).append(normalized["action_id"])
        self.campaign_repairs += 1
        self._inflight.add(normalized["incident_id"])
        task = self._open_task(normalized)
        try:
            return self._execute(normalized, task)
        except RepairContractError as error:
            if task.get("closed"):
                return task["closed_decision"]
            return self._finish(normalized, task, "scope_refused", str(error), applied=bool(task.get("edits_applied")))
        finally:
            self._inflight.discard(normalized["incident_id"])

    def _precheck(self, normalized: dict[str, Any]) -> dict[str, Any] | None:
        if normalized["held"]:
            return self._refuse(normalized, "held", "hold blocks isolated repair")
        if normalized["observed_at"] > normalized["deadline_at"]:
            return self._refuse(normalized, "deadline_exceeded", "repair action is outside its deadline")
        if normalized["failure"]["effects_unknown"]:
            return self._refuse(normalized, "effects_unknown", "unknown effects block isolated repair")
        current = self.tasks.get(normalized["incident_id"])
        if current is not None and current["state"] == "reloaded":
            return self._refuse(normalized, "already_promoted", "this incident already has a reloaded promotion")
        if current is not None and current["state"] == "reload_failed":
            return self._refuse(normalized, "reload_failed", "reload must be retried before another repair")
        if self._remaining(normalized["incident_id"], "repair") <= 0:
            return self._refuse(normalized, "repair_budget_exhausted", "durable repair budget is exhausted")
        if self.campaign_repairs >= CAMPAIGN_REPAIR_CAP:
            return self._refuse(normalized, "campaign_repair_cap", "campaign repair cap is exhausted")
        if self.regression_runner is None or self.canary_runner is None:
            return self._refuse(normalized, "validation_unbound", "regressions and canary are not bound")
        if self.merger is None:
            return self._refuse(normalized, "promotion_unbound", "native merge is not bound")
        if self.reloader is None:
            return self._refuse(normalized, "reload_unbound", "reload and fence are not bound")
        try:
            require_canonical_routing(normalized["plan"])
        except WorldwideRouteError as error:
            return self._refuse(normalized, "route_refused", str(error))
        try:
            self._precheck_scope(normalized)
            self._precheck_worktree(normalized)
        except RepairContractError as error:
            reason = str(error)
            disposition = "reproducer_required" if reason.startswith("reproducer") else (
                "worktree_refused" if reason.startswith("worktree") or reason.startswith("campaign") else "scope_refused"
            )
            return self._refuse(normalized, disposition, reason)
        self._require_incident(normalized["incident_id"])
        return None

    def _execute(self, normalized: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
        worktree = Path(task["worktree"])
        self._materialize_baseline(worktree, normalized["failure"])
        attempt = self.spent["repair"][normalized["incident_id"]]
        try:
            binding = self.route_binder(
                normalized["plan"],
                self._route_config(normalized),
                workspace=worktree,
                task_id=task["repair_task_id"],
                attempt=attempt,
                python_executable=normalized["python_executable"],
            )
            _assert_route(binding, workspace=worktree, task_id=task["repair_task_id"], attempt=attempt)
        except (WorldwideRouteError, RepairContractError, OSError, ValueError, KeyError) as error:
            return self._finish(normalized, task, "route_refused", str(error), route=None)
        route = _route_view(binding, worktree)
        task["route"] = route
        self._charge(normalized["incident_id"], "provider")
        edit_order = self._apply_edits(worktree, normalized)
        task["edits_applied"] = True
        task["edit_order"] = edit_order
        sealed = _snapshot(worktree)
        self._charge(normalized["incident_id"], "validation")
        context = self._runner_context(normalized, task, route)
        try:
            regression = self.regression_runner(context) if self.regression_runner is not None else {}
        except Exception as error:  # noqa: BLE001 - a runner fault is a failed gate, not a second writer
            return self._finish(normalized, task, "regressions_failed", str(error), route=route, applied=True)
        regression_error = _regression_error(regression, normalized["failure"]["reproducer"]["argv"])
        if regression_error:
            return self._finish(normalized, task, "regressions_failed", regression_error, route=route, applied=True, regression=regression)
        if _snapshot(worktree) != sealed:
            return self._finish(normalized, task, "scope_refused", "validation changed the repair worktree", route=route, applied=True, regression=regression)
        try:
            canary = self.canary_runner(context) if self.canary_runner is not None else {}
        except Exception as error:  # noqa: BLE001 - a canary fault blocks promotion
            return self._finish(normalized, task, "canary_failed", str(error), route=route, applied=True, regression=regression)
        canary_error = _canary_error(canary)
        if canary_error:
            return self._finish(normalized, task, "canary_failed", canary_error, route=route, applied=True, regression=regression, canary=canary)
        if _snapshot(worktree) != sealed:
            return self._finish(normalized, task, "scope_refused", "canary changed the repair worktree", route=route, applied=True, regression=regression, canary=canary)
        self._charge(normalized["incident_id"], "merge")
        try:
            merge = self.merger(context) if self.merger is not None else {}
        except Exception as error:  # noqa: BLE001 - a merge fault keeps the baseline as rollback
            return self._finish(normalized, task, "promotion_refused", str(error), route=route, applied=True, regression=regression, canary=canary)
        merge_error = _merge_error(merge, normalized["failure"]["source_revision"])
        if merge_error:
            disposition = "effects_unknown" if merge.get("effects_unknown") is True else "promotion_refused"
            return self._finish(
                normalized, task, disposition, merge_error, route=route, applied=True,
                regression=regression, canary=canary, merge=merge,
            )
        try:
            reload = self.reloader({**context, "merge": _merge_view(merge)}) if self.reloader is not None else {}
        except Exception as error:  # noqa: BLE001 - reload failure retains the merge and the baseline
            return self._finish(
                normalized, task, "reload_failed", str(error), route=route, applied=True, promoted=True,
                regression=regression, canary=canary, merge=merge,
            )
        reload_error = _reload_error(reload, str(merge.get("merge_commit")))
        if reload_error:
            return self._finish(
                normalized, task, "reload_failed", reload_error, route=route, applied=True, promoted=True,
                regression=regression, canary=canary, merge=merge, reload=reload,
            )
        return self._finish(
            normalized, task, "reloaded", "promoted through native merge and reloaded", route=route,
            applied=True, promoted=True, reloaded=True, regression=regression, canary=canary, merge=merge, reload=reload,
        )

    def _reload_only(self, normalized: dict[str, Any]) -> dict[str, Any]:
        if normalized["held"]:
            return self._refuse(normalized, "held", "hold blocks reload")
        if normalized["observed_at"] > normalized["deadline_at"]:
            return self._refuse(normalized, "deadline_exceeded", "reload action is outside its deadline")
        current = self.tasks.get(normalized["incident_id"])
        if current is None or current["state"] != "reload_failed":
            return self._refuse(normalized, "reload_not_pending", "no promoted repair awaits reload")
        if self.reloader is None:
            return self._refuse(normalized, "reload_unbound", "reload and fence are not bound")
        # A later reload is a new action. The previous close only stops a
        # second finish of the action that already recorded reload_failed.
        current.pop("closed", None)
        current.pop("closed_decision", None)
        self._inflight.add(normalized["incident_id"])
        try:
            context = {
                "repair_task_id": current["repair_task_id"],
                "worktree": current["worktree"],
                "baseline_revision": current["baseline_revision"],
                "merge": dict(current["merge"]),
                "completion_authority": False,
            }
            try:
                reload = self.reloader(context)
            except Exception as error:  # noqa: BLE001 - reload stays pending for one later retry
                return self._finish(normalized, current, "reload_failed", str(error), promoted=True, applied=True, route=current.get("route"), merge=current.get("merge"))
            reload_error = _reload_error(reload, str(current["merge"].get("merge_commit")))
            if reload_error:
                return self._finish(
                    normalized, current, "reload_failed", reload_error, promoted=True, applied=True,
                    route=current.get("route"), merge=current.get("merge"), reload=reload,
                )
            return self._finish(
                normalized, current, "reloaded", "reloaded and fenced after the retained native merge",
                promoted=True, reloaded=True, applied=True, route=current.get("route"), merge=current.get("merge"), reload=reload,
            )
        finally:
            self._inflight.discard(normalized["incident_id"])

    def _normalize(self, request: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(request, Mapping):
            raise RepairContractError("repair request must be an object")
        _reject_completion(request)
        unknown = set(request) - _REQUEST_FIELDS
        if unknown:
            raise RepairContractError("repair request field is not recognized")
        action_id = _identifier(request.get("action_id"), "action_id")
        observed_at = _timestamp(request.get("observed_at"), "observed_at")
        deadline_at = _timestamp(request.get("deadline_at"), "deadline_at")
        phase = request.get("phase", "repair")
        if phase not in {"repair", "reload"}:
            raise RepairContractError("repair phase is not recognized")
        if request.get("hold") not in (None, True, False):
            raise RepairContractError("hold must be a boolean")
        markers = _markers(request.get("hold_markers"))
        held = request.get("hold") is True or bool(markers) or self.hold["held"]
        failure = self._failure(request.get("failure"))
        try:
            incident_id = repair_task_identity(
                run_id=self.run_id, task_id=failure["task_id"], stage=failure["stage"], defect=failure["defect"],
            )
        except ProgressIdentityError as error:
            raise RepairContractError(str(error)) from error
        repair_id = admitted_repair_task_id(incident_id)
        campaign_root = _directory(request.get("campaign_root"), "campaign root")
        live_checkout = _directory(request.get("live_checkout"), "live checkout")
        plan = request.get("plan")
        if not isinstance(plan, Mapping):
            raise RepairContractError("campaign plan must be an object")
        provider = request.get("provider")
        if not isinstance(provider, Mapping):
            raise RepairContractError("campaign provider configuration must be an object")
        python_executable = request.get("python_executable", CANONICAL_PYTHON)
        if python_executable != CANONICAL_PYTHON:
            raise RepairContractError("repair route must use the canonical python interpreter")
        edits = _edits(request.get("edits"), failure, repair_id)
        body = {
            "action_id": action_id,
            "observed_at": observed_at,
            "deadline_at": deadline_at,
            "phase": phase,
            "held": held,
            "incident_id": incident_id,
            "repair_task_id": repair_id,
            "failure": failure,
            "campaign_root": str(campaign_root),
            "live_checkout": str(live_checkout),
            "plan": _plain(plan),
            "provider": _plain(provider),
            "python_executable": python_executable,
            "edits": edits,
            "completion_authority": False,
        }
        return {**body, "digest": digest(body)}

    def _failure(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping) or set(value) - _FAILURE_FIELDS:
            raise RepairContractError("reproduced failure is malformed")
        if value.get("schema") != FAILURE_SCHEMA:
            raise RepairContractError("reproduced failure is malformed")
        if _identifier(value.get("run_id"), "run_id") != self.run_id:
            raise RepairContractError("reproduced failure belongs to another run")
        task_id = _identifier(value.get("task_id"), "task_id")
        try:
            stage = canonical_stage(value.get("stage"))
        except ProgressIdentityError as error:
            raise RepairContractError(str(error)) from error
        if type(value.get("effects_unknown")) is not bool:
            raise RepairContractError("reproduced failure must say whether effects are known")
        revision = _commit(value.get("source_revision"), "source revision")
        reproducer = value.get("reproducer")
        if not isinstance(reproducer, Mapping) or set(reproducer) != {"path", "argv"}:
            raise RepairContractError("reproducer is malformed")
        path = reproducer.get("path")
        if not isinstance(path, str) or not reproducer_path_admitted(path):
            raise RepairContractError("reproducer path is outside the admitted repair scope")
        argv = reproducer.get("argv")
        if not isinstance(argv, list) or not argv or len(argv) > 32 or not all(isinstance(item, str) and item and len(item) <= 512 for item in argv):
            raise RepairContractError("reproducer argv is malformed")
        if path not in argv:
            raise RepairContractError("reproducer argv does not name the reproducer")
        frames = value.get("source_frames")
        if not isinstance(frames, list) or not frames or len(frames) > 8:
            raise RepairContractError("reproduced source frames are required")
        source_frames = []
        seen = set()
        for frame in frames:
            if not isinstance(frame, Mapping) or set(frame) != {"path", "source_sha256"}:
                raise RepairContractError("reproduced source frame is malformed")
            frame_path = frame.get("path")
            if not isinstance(frame_path, str) or frame_path in seen:
                raise RepairContractError("reproduced source frame is malformed")
            seen.add(frame_path)
            source_frames.append({"path": frame_path, "source_sha256": _frame_sha(frame.get("source_sha256"))})
        baseline = value.get("baseline_files")
        if not isinstance(baseline, Mapping) or not baseline or len(baseline) > _MAX_FILES:
            raise RepairContractError("baseline files are required")
        files = {}
        for relative, text in baseline.items():
            if not isinstance(relative, str) or not isinstance(text, str) or len(text.encode("utf-8")) > _MAX_FILE:
                raise RepairContractError("baseline file is malformed")
            if not _relative_ok(relative):
                raise RepairContractError("baseline file escapes the campaign worktree")
            files[relative] = text
        receipt_id = ""
        if "receipt_id" in value and value.get("receipt_id") not in (None, ""):
            receipt_id = _digest(value.get("receipt_id"), "receipt_id")
        return {
            "schema": FAILURE_SCHEMA,
            "run_id": self.run_id,
            "task_id": task_id,
            "stage": stage,
            "defect": value.get("defect", ""),
            "effects_unknown": value["effects_unknown"],
            "source_revision": revision,
            "reproducer": {"path": path, "argv": list(argv)},
            "source_frames": source_frames,
            "baseline_files": files,
            "receipt_id": receipt_id,
        }

    def _precheck_scope(self, normalized: Mapping[str, Any]) -> None:
        failure = normalized["failure"]
        frames = {frame["path"]: frame["source_sha256"] for frame in failure["source_frames"]}
        baseline = failure["baseline_files"]
        reproducer = failure["reproducer"]["path"]
        if reproducer in baseline or reproducer in frames:
            raise RepairContractError("reproducer must be a new admitted test and not a reproduced source frame")
        for path, digest_hex in frames.items():
            if path_is_protected(path):
                raise RepairContractError("repair edit touches protected policy or completion code")
            if not source_path_admitted(path):
                raise RepairContractError("reproduced source frame is outside the admitted repair scope")
            if path not in baseline:
                raise RepairContractError("reproduced source frame is absent from the baseline")
            if _content_sha(baseline[path]) != digest_hex:
                raise RepairContractError("reproduced source frame does not match the baseline")
        edits = normalized["edits"]
        if not edits or edits[0]["role"] != "reproducer":
            raise RepairContractError("reproducer must be admitted before the defective behavior changes")
        roles = [item["role"] for item in edits]
        if roles.count("reproducer") != 1 or any(item["role"] == "reproducer" and item["path"] != reproducer for item in edits):
            raise RepairContractError("reproducer must be admitted before the defective behavior changes")
        source_edits = [item for item in edits if item["role"] == "source"]
        if not source_edits:
            raise RepairContractError("reproducer must be paired with the reproduced source frame")
        edited_sources = {item["path"] for item in source_edits}
        if edited_sources != set(frames):
            raise RepairContractError("repair edits must be exactly the reproduced source frames")
        failure_key = _failure_key(failure["defect"])
        for item in edits:
            if _FORBIDDEN_EDIT.search(item["text"]) is not None:
                raise RepairContractError("repair edit grants completion or skips the reproducer")
            if item["role"] == "reproducer" and failure_key and failure_key not in item["text"].lower():
                raise RepairContractError("reproducer does not name the reproduced defect")
            if item["role"] == "source" and item["text"] == baseline[item["path"]]:
                raise RepairContractError("reproduced source frame was not changed")

    def _precheck_worktree(self, normalized: Mapping[str, Any]) -> None:
        campaign = Path(normalized["campaign_root"]).resolve()
        live = Path(normalized["live_checkout"]).resolve()
        worktree = campaign / "repair-worktrees" / normalized["repair_task_id"]
        if campaign == live or campaign.is_relative_to(live) or live.is_relative_to(campaign):
            raise RepairContractError("worktree overlaps the live checkout")
        parent = campaign / "repair-worktrees"
        if parent.is_symlink() or worktree.is_symlink():
            raise RepairContractError("worktree must be a campaign-owned directory")
        if worktree.exists():
            marker = self._read_marker(worktree)
            if marker is None or marker.get("run_id") != self.run_id or marker.get("repair_task_id") != normalized["repair_task_id"]:
                raise RepairContractError("worktree belongs to another campaign writer")

    def _open_task(self, normalized: Mapping[str, Any]) -> dict[str, Any]:
        campaign = Path(normalized["campaign_root"])
        worktree = campaign / "repair-worktrees" / normalized["repair_task_id"]
        worktree.mkdir(parents=True, exist_ok=True)
        os.chmod(worktree, 0o700)
        previous = self.tasks.get(normalized["incident_id"])
        fence_generation = 0 if previous is None else previous["fence_generation"]
        task = {
            "incident_id": normalized["incident_id"],
            "repair_task_id": normalized["repair_task_id"],
            "state": "open",
            "worktree": str(worktree),
            "baseline_revision": normalized["failure"]["source_revision"],
            "fence_generation": fence_generation,
            "edit_order": [],
            "edits_applied": False,
            "route": None,
            "merge": None,
        }
        self.tasks[normalized["incident_id"]] = task
        self._write_marker(worktree, task, fenced=False)
        return task

    def _materialize_baseline(self, worktree: Path, failure: Mapping[str, Any]) -> None:
        for relative, text in failure["baseline_files"].items():
            _write_bytes(worktree, relative, text.encode("utf-8"))

    def _apply_edits(self, worktree: Path, normalized: Mapping[str, Any]) -> list[str]:
        order = []
        for item in normalized["edits"]:
            _write_bytes(worktree, item["path"], item["text"].encode("utf-8"))
            order.append(item["path"])
        return order

    def _route_config(self, normalized: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "state_root": normalized["campaign_root"],
            "allowed_workspace_roots": [normalized["campaign_root"]],
            "provider": dict(normalized["provider"]),
        }

    def _runner_context(self, normalized: Mapping[str, Any], task: Mapping[str, Any], route: Mapping[str, Any]) -> dict[str, Any]:
        failure = normalized["failure"]
        return {
            "schema": "wwl.repair.runner-context@1",
            "run_id": self.run_id,
            "repair_task_id": task["repair_task_id"],
            "incident_id": normalized["incident_id"],
            "worktree": task["worktree"],
            "live_checkout": normalized["live_checkout"],
            "baseline_revision": failure["source_revision"],
            "reproducer_path": failure["reproducer"]["path"],
            "reproducer_argv": list(failure["reproducer"]["argv"]),
            "source_paths": [frame["path"] for frame in failure["source_frames"]],
            "edit_order": list(task["edit_order"]),
            "route": dict(route),
            "completion_authority": False,
        }

    def _finish(
        self,
        normalized: Mapping[str, Any],
        task: dict[str, Any],
        disposition: str,
        reason: str,
        *,
        applied: bool = False,
        promoted: bool = False,
        reloaded: bool = False,
        route: Mapping[str, Any] | None = None,
        regression: Mapping[str, Any] | None = None,
        canary: Mapping[str, Any] | None = None,
        merge: Mapping[str, Any] | None = None,
        reload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if task.get("closed"):
            return task["closed_decision"]
        if disposition == "reloaded":
            state = "reloaded"
        elif disposition == "reload_failed" and promoted:
            state = "reload_failed"
        else:
            state = "fenced"
        task["state"] = state
        task["fence_generation"] = int(task.get("fence_generation") or 0) + 1
        merge_view = _merge_view(merge) if isinstance(merge, Mapping) and merge.get("merged") is True and merge.get("effects_unknown") is False else None
        if state == "reload_failed" and merge_view is None and isinstance(task.get("merge"), Mapping):
            merge_view = dict(task["merge"])
        if merge_view is not None:
            task["merge"] = merge_view
        worktree = Path(task["worktree"])
        if worktree.is_dir():
            self._write_marker(worktree, task, fenced=True)
        rollback = self._rollback(normalized, task, disposition, promoted=promoted, reloaded=reloaded, merge_view=merge_view)
        self._rollbacks.setdefault(task["repair_task_id"], []).append(rollback)
        if worktree.is_dir():
            _write_bytes(worktree, f"rollback-{task['fence_generation']}.json", _canonical_bytes(rollback))
            _write_bytes(worktree, "rollback.json", _canonical_bytes(rollback))
            evidence = self._evidence(normalized, task, disposition, rollback)
            _write_bytes(worktree, f"artifacts/worldwide-legal-search/repairs/{task['repair_task_id']}.json", _canonical_bytes(evidence))
        decision = self._decision(
            normalized, task, disposition, reason, applied=applied, promoted=promoted, reloaded=reloaded,
            route=route if route is not None else task.get("route"), regression=regression, canary=canary,
            merge=merge_view, reload=reload, rollback=rollback,
        )
        task["closed"] = True
        task["closed_decision"] = decision
        return decision

    def _refuse(self, normalized: Mapping[str, Any], disposition: str, reason: str) -> dict[str, Any]:
        task_id = normalized["repair_task_id"]
        current = self.tasks.get(normalized["incident_id"])
        baseline = normalized["failure"]["source_revision"]
        rollback = {
            "schema": ROLLBACK_SCHEMA,
            "run_id": self.run_id,
            "repair_task_id": task_id,
            "incident_id": normalized["incident_id"],
            "action_id": normalized["action_id"],
            "baseline_revision": baseline,
            "promoted_revision": current["merge"]["merge_commit"] if current and current.get("merge") else "",
            "merge_receipt": current["merge"]["merge_receipt"] if current and current.get("merge") else "",
            "worktree": current["worktree"] if current else "",
            "fenced": bool(current and current["state"] in {"fenced", "reload_failed", "reloaded"}),
            "fence_generation": 0 if current is None else current["fence_generation"],
            "invalidated_evidence": [],
            "retained": True,
            "disposition": disposition,
            "completion_authority": False,
        }
        self._rollbacks.setdefault(task_id, []).append(rollback)
        return self._decision(
            normalized,
            current or {"repair_task_id": task_id, "fence_generation": 0, "edit_order": [], "worktree": ""},
            disposition,
            reason,
            rollback=rollback,
        )

    def _decision(
        self,
        normalized: Mapping[str, Any],
        task: Mapping[str, Any],
        disposition: str,
        reason: str,
        *,
        applied: bool = False,
        promoted: bool = False,
        reloaded: bool = False,
        route: Mapping[str, Any] | None = None,
        regression: Mapping[str, Any] | None = None,
        canary: Mapping[str, Any] | None = None,
        merge: Mapping[str, Any] | None = None,
        reload: Mapping[str, Any] | None = None,
        rollback: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        incident_id = normalized["incident_id"]
        return {
            "schema": SCHEMA,
            "action_id": normalized["action_id"],
            "disposition": disposition,
            "reason": reason,
            "incident_id": incident_id,
            "repair_task_id": normalized["repair_task_id"],
            "deduplicated": False,
            "duplicate_writer": disposition == "duplicate_writer",
            "applied": applied,
            "promoted": promoted,
            "reloaded": reloaded,
            "fenced": bool(rollback and rollback.get("fenced")),
            "fence_generation": int(task.get("fence_generation") or 0),
            "edit_order": list(task.get("edit_order") or []),
            "worktree": task.get("worktree") or "",
            "route": None if route is None else dict(route),
            "regression": None if regression is None else _plain(regression),
            "canary": None if canary is None else _plain(canary),
            "merge": None if merge is None else dict(merge),
            "reload": None if reload is None else _plain(reload),
            "rollback": None if rollback is None else dict(rollback),
            "budget": self._budget(incident_id),
            "campaign_repairs": self.campaign_repairs,
            "completion_authority": False,
        }

    def _rollback(
        self,
        normalized: Mapping[str, Any],
        task: Mapping[str, Any],
        disposition: str,
        *,
        promoted: bool,
        reloaded: bool,
        merge_view: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        receipt = normalized["failure"]["receipt_id"]
        invalidated = []
        if reloaded:
            invalidated.append("baseline:" + normalized["failure"]["source_revision"])
            if receipt:
                invalidated.append(receipt)
        return {
            "schema": ROLLBACK_SCHEMA,
            "run_id": self.run_id,
            "repair_task_id": task["repair_task_id"],
            "incident_id": normalized["incident_id"],
            "action_id": normalized["action_id"],
            "baseline_revision": normalized["failure"]["source_revision"],
            "promoted_revision": "" if merge_view is None or not promoted else merge_view["merge_commit"],
            "merge_receipt": "" if merge_view is None or not promoted else merge_view["merge_receipt"],
            "worktree": task["worktree"],
            "fenced": True,
            "fence_generation": task["fence_generation"],
            "invalidated_evidence": invalidated,
            "retained": True,
            "disposition": disposition,
            "completion_authority": False,
        }

    def _evidence(self, normalized: Mapping[str, Any], task: Mapping[str, Any], disposition: str, rollback: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "schema": "wwl.repair.evidence@1",
            "repair_task_id": task["repair_task_id"],
            "incident_id": normalized["incident_id"],
            "disposition": disposition,
            "baseline_revision": rollback["baseline_revision"],
            "promoted_revision": rollback["promoted_revision"],
            "merge_receipt": rollback["merge_receipt"],
            "rollback_retained": True,
            "completion_authority": False,
        }

    def _charge(self, incident_id: str, budget: str) -> None:
        if self._remaining(incident_id, budget) <= 0:
            raise RepairContractError("durable repair budget is exhausted")
        if self.ledger is not None:
            ledger_spent = self.ledger.incidents[incident_id]["spent"][budget]
            if ledger_spent != self.spent[budget].get(incident_id, 0):
                raise RepairContractError("durable repair bound diverged from the progress ledger")
            charged = self.ledger.charge(incident_id, budget)
            if not charged["charged"]:
                raise RepairContractError("durable repair budget is exhausted")
            self.spent[budget][incident_id] = charged["spent"]
        else:
            self.spent[budget][incident_id] = self.spent[budget].get(incident_id, 0) + 1
        self._charged_now[budget].add(incident_id)

    def _remaining(self, incident_id: str, budget: str) -> int:
        return DEFAULT_LIMITS[budget] - self.spent[budget].get(incident_id, 0)

    def _budget(self, incident_id: str) -> dict[str, dict[str, Any]]:
        view = {}
        for name in BUDGET_NAMES:
            spent = self.spent[name].get(incident_id, 0)
            view[name] = {
                "spent": spent,
                "limit": DEFAULT_LIMITS[name],
                "remaining": DEFAULT_LIMITS[name] - spent,
                "charged": incident_id in self._charged_now[name],
                "budgets_reset": False,
                "completion_authority": False,
            }
        return view

    def _require_incident(self, incident_id: str) -> None:
        if self.ledger is None:
            return
        if incident_id not in self.ledger.incidents:
            raise RepairContractError("bound progress ledger has no such incident")
        for name in BUDGET_NAMES:
            if self.ledger.incidents[incident_id]["spent"][name] != self.spent[name].get(incident_id, 0):
                raise RepairContractError("durable repair bound diverged from the progress ledger")

    def _require_ledger_alignment(self) -> None:
        if self.ledger is None:
            return
        for name in BUDGET_NAMES:
            for incident_id, used in self.spent[name].items():
                if incident_id not in self.ledger.incidents:
                    raise RepairContractError("durable repair bound diverged from the progress ledger")
                if self.ledger.incidents[incident_id]["spent"][name] != used:
                    raise RepairContractError("durable repair bound diverged from the progress ledger")

    def _write_marker(self, worktree: Path, task: Mapping[str, Any], *, fenced: bool) -> None:
        payload = {
            "schema": WORKTREE_SCHEMA,
            "run_id": self.run_id,
            "repair_task_id": task["repair_task_id"],
            "fenced": fenced,
            "fence_generation": task["fence_generation"],
            "completion_authority": False,
        }
        _write_bytes(worktree, OWNER_MARKER, _canonical_bytes(payload))

    def _read_marker(self, worktree: Path) -> dict[str, Any] | None:
        path = worktree / OWNER_MARKER
        if not path.is_file() or path.is_symlink():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        if not isinstance(value, Mapping) or value.get("schema") != WORKTREE_SCHEMA:
            return None
        return dict(value)


def _edits(value: Any, failure: Mapping[str, Any], repair_task_id: str) -> list[dict[str, str]]:
    if not isinstance(value, list) or len(value) > 16:
        raise RepairContractError("repair edits are malformed")
    edits = []
    seen = set()
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {"path", "role", "text"}:
            raise RepairContractError("repair edits are malformed")
        path = item.get("path")
        role = item.get("role")
        text = item.get("text")
        if role not in {"reproducer", "source"} or not isinstance(path, str) or not isinstance(text, str):
            raise RepairContractError("repair edits are malformed")
        if path in seen or len(text.encode("utf-8")) > _MAX_FILE:
            raise RepairContractError("repair edits are malformed")
        if not _relative_ok(path):
            raise RepairContractError("repair path escapes the campaign worktree")
        if role == "reproducer" and path != failure["reproducer"]["path"]:
            raise RepairContractError("reproducer edit does not match the reproduced failure")
        if path_is_protected(path):
            raise RepairContractError("repair edit touches protected policy or completion code")
        if role == "source" and not source_path_admitted(path):
            raise RepairContractError("repair edit is outside the admitted repair scope")
        seen.add(path)
        edits.append({"path": path, "role": role, "text": text})
    if repair_task_id and any(item["path"].endswith(repair_task_id + ".json") for item in edits):
        raise RepairContractError("repair evidence is written by the session")
    return edits


def _failure_key(defect: Any) -> str:
    if isinstance(defect, Mapping):
        key = defect.get("failure_key")
        if isinstance(key, str) and key.strip():
            return " ".join(key.strip().lower().split())
    if isinstance(defect, str) and defect.strip():
        return " ".join(defect.strip().lower().split())
    return ""


def _route_view(binding: Mapping[str, Any], worktree: Path) -> dict[str, Any]:
    return {
        "schema": binding["schema"],
        "router": binding["router"],
        "primary_provider": binding["primary_provider"],
        "primary_model": binding["primary_model"],
        "fallback_provider": binding.get("fallback_provider"),
        "fallback_policy": binding["fallback_policy"],
        "task_id": binding["task_id"],
        "attempt": binding["attempt"],
        "workspace": str(worktree.resolve()),
        "argv": list(binding["argv"]),
        "launched": False,
        "paid_provider_calls": 0,
        "hosted_model_invocations": 0,
        "completion_authority": False,
    }


def _regression_error(value: Any, argv: Sequence[str]) -> str:
    if not isinstance(value, Mapping) or value.get("schema") != REGRESSION_SCHEMA:
        return "regression report is not the admitted schema"
    if value.get("completion_authority") is not False:
        return "regression report cannot grant completion authority"
    if value.get("baseline_failure_verified") is not True:
        return "reproducer did not fail on the baseline"
    if value.get("candidate_passed") is not True or value.get("focused_passed") is not True:
        return "candidate regressions failed"
    if value.get("protected_hashes_unchanged") is not True:
        return "protected files changed"
    if value.get("exit_status") != 0 or list(value.get("argv") or []) != list(argv):
        return "regression argv or exit status does not match the reproducer"
    return ""


def _canary_error(value: Any) -> str:
    if not isinstance(value, Mapping) or value.get("schema") != CANARY_SCHEMA:
        return "canary report is not the admitted schema"
    if value.get("completion_authority") is not False:
        return "canary report cannot grant completion authority"
    if value.get("validated_promotion") is not False:
        return "canary cannot promote itself"
    if value.get("paid_provider_calls") != 0 or value.get("hosted_model_invocations") != 0:
        return "canary cannot spend a hosted provider call"
    if value.get("baseline_failure_verified") is not True or value.get("candidate_regressions_verified") is not True:
        return "canary did not verify the baseline failure and the candidate"
    if value.get("repair_canary_verified") is not True:
        return "canary did not verify the isolated repair"
    return ""


def _merge_error(value: Any, baseline: str) -> str:
    if not isinstance(value, Mapping):
        return "native merge receipt is missing"
    if value.get("completion_authority") is not False:
        return "native merge cannot grant completion authority"
    if value.get("effects_unknown") is True:
        return "native merge effects are unknown"
    if value.get("effects_unknown") is not False or value.get("merged") is not True:
        return "native merge was not accepted"
    commit = value.get("merge_commit")
    receipt = value.get("merge_receipt")
    tree = value.get("tree")
    if not isinstance(commit, str) or _COMMIT.fullmatch(commit) is None or commit == baseline:
        return "native merge commit does not advance the baseline"
    if not isinstance(receipt, str) or _DIGEST.fullmatch(receipt) is None:
        return "native merge receipt is missing"
    if not isinstance(tree, str) or _DIGEST.fullmatch(tree) is None:
        return "native merge tree is missing"
    return ""


def _merge_view(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(value, Mapping) or value.get("merged") is not True:
        return None
    return {
        "merged": True,
        "merge_commit": value["merge_commit"],
        "merge_receipt": value["merge_receipt"],
        "tree": value["tree"],
        "effects_unknown": False,
        "completion_authority": False,
    }


def _reload_error(value: Any, merge_commit: str) -> str:
    if not isinstance(value, Mapping):
        return "reload report is missing"
    if value.get("completion_authority") is not False:
        return "reload cannot grant completion authority"
    if value.get("reloaded") is not True or value.get("fenced") is not True:
        return "lane was not reloaded and fenced"
    if value.get("lane_revision") != merge_commit:
        return "reloaded lane is not the promoted revision"
    return ""


def _plain(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(value, sort_keys=True))


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _public_task(task: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "incident_id": task["incident_id"],
        "repair_task_id": task["repair_task_id"],
        "state": task["state"],
        "worktree": task["worktree"],
        "baseline_revision": task["baseline_revision"],
        "fence_generation": task["fence_generation"],
        "edit_order": list(task.get("edit_order") or []),
        "merge": None if task.get("merge") is None else dict(task["merge"]),
    }


def _restore_task(incident_id: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RepairContractError("stored repair task is malformed")
    if value.get("incident_id") != incident_id:
        raise RepairContractError("stored repair task is malformed")
    state = value.get("state")
    if state not in {"open", "fenced", "reload_failed", "reloaded"}:
        raise RepairContractError("stored repair task is malformed")
    if state == "open":
        raise RepairContractError("stored repair task cannot resume an open writer")
    merge = value.get("merge")
    restored_merge = None
    if state in {"reload_failed", "reloaded"}:
        if not isinstance(merge, Mapping):
            raise RepairContractError("stored promoted repair lacks its merge receipt")
        probe = {
            "merged": True,
            "effects_unknown": False,
            "completion_authority": False,
            "merge_commit": merge.get("merge_commit"),
            "merge_receipt": merge.get("merge_receipt"),
            "tree": merge.get("tree"),
        }
        other = "f" * 40 if probe["merge_commit"] == "0" * 40 else "0" * 40
        if _merge_error(probe, other):
            raise RepairContractError("stored promoted repair lacks its merge receipt")
        restored_merge = dict(merge)
    elif merge not in (None, {}):
        raise RepairContractError("stored repair task is malformed")
    return {
        "incident_id": incident_id,
        "repair_task_id": _identifier(value.get("repair_task_id"), "repair_task_id"),
        "state": state,
        "worktree": str(value.get("worktree") or ""),
        "baseline_revision": _commit(value.get("baseline_revision"), "baseline revision"),
        "fence_generation": _whole(value.get("fence_generation"), "fence generation"),
        "edit_order": [item for item in value.get("edit_order") or [] if isinstance(item, str)],
        "edits_applied": True,
        "route": None,
        "merge": restored_merge,
    }


def _restore_rollback(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or value.get("schema") != ROLLBACK_SCHEMA:
        raise RepairContractError("stored rollback evidence is malformed")
    if value.get("retained") is not True or value.get("completion_authority") is not False:
        raise RepairContractError("stored rollback evidence is malformed")
    return dict(value)


# Routing identity used by callers that want the campaign object without a second copy.
REPAIR_ROUTING = {
    "router": ROUTER_NAME,
    "primary_provider": PRIMARY_PROVIDER,
    "primary_model": PRIMARY_MODEL,
    "policy": ROUTE_POLICY,
    "canonical": CANONICAL_ROUTING["router"] == ROUTER_NAME,
}
