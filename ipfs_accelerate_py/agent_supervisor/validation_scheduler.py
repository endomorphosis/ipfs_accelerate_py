"""Impact-selected, cached, and resource-bounded validation execution.

The scheduler is intentionally independent from the todo daemon.  It accepts
plain shell commands or classified command specifications, produces a JSON-safe
legacy-compatible report, and stores only successful results.  Persistent cache
keys bind a result to the target commit, candidate worktree content, command,
relevant environment, and dependency state.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import subprocess
import tempfile
import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .resource_scheduler import (
    AdmissionDecision,
    HostResourceSnapshot,
    LaneResourceRequirements,
    ProofResourceClass,
    ResourceAdmissionLease,
    ResourceLeaseBudget,
    ResourcePolicy,
    ResourceScheduler,
)
from .validation_commands import (
    DeclaredValidation,
    ValidationCommand,
    ValidationRequirementKind,
    ValidationSelection,
    ValidationStage,
    build_validation_commands,
    normalize_validation_command_text,
    select_validation_commands,
)


CACHE_SCHEMA = "ipfs_accelerate_py/agent-supervisor/validation-cache@1"
STAGED_REPORT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/staged-validation-report@1"
)
VALIDATION_DAG_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/validation-dag-receipt@2"
)
TRANSITIVE_IMPACT_EVIDENCE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/transitive-impact-validation-evidence@2"
)
TRANSITIVE_IMPACT_REQUIREMENT_ID = "266404049326363900535699811645710804440"
REQUIRED_AUTHORITY_GATES = (
    "semantic",
    "proof",
    "merge",
    "freshness",
    "completion",
)
STRICT_VALIDATION_STAGE_ORDER = (
    ValidationStage.CHEAP,
    ValidationStage.TARGETED,
    ValidationStage.BROAD,
    ValidationStage.TRANSLATION,
    ValidationStage.SOLVER,
    ValidationStage.KERNEL,
    ValidationStage.ATTESTATION,
)
VALIDATION_VERDICT_KINDS = (
    "deterministic",
    "translation",
    "solver",
    "kernel",
    "test",
    "attestation",
)
DEFAULT_RELEVANT_ENVIRONMENT = (
    "CI",
    "LANG",
    "LC_ALL",
    "NODE_ENV",
    "PATH",
    "PYTHONHASHSEED",
    "PYTHONPATH",
    "PYTHONWARNINGS",
    "RUSTFLAGS",
    "VIRTUAL_ENV",
)
DEPENDENCY_FILENAMES = frozenset(
    {
        ".gitmodules",
        "Cargo.lock",
        "Cargo.toml",
        "Pipfile",
        "Pipfile.lock",
        "go.mod",
        "go.sum",
        "package-lock.json",
        "package.json",
        "pnpm-lock.yaml",
        "poetry.lock",
        "pyproject.toml",
        "setup.cfg",
        "setup.py",
        "tox.ini",
        "uv.lock",
        "yarn.lock",
    }
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    return str(value)


@dataclass(frozen=True)
class ValidationCacheKey:
    """Canonical components and digest for one reusable validation result."""

    target_commit: str
    command: str
    environment: tuple[tuple[str, str], ...]
    dependency_state: str
    digest: str

    def to_dict(self) -> dict[str, object]:
        return {
            "target_commit": self.target_commit,
            "command": self.command,
            "environment": dict(self.environment),
            "dependency_state": self.dependency_state,
            "digest": self.digest,
        }


def relevant_environment(
    environment: Mapping[str, object] | None = None,
    extra_keys: Iterable[str] = (),
) -> dict[str, str]:
    """Return the stable, non-secret environment subset that affects tools."""

    source = os.environ if environment is None else environment
    keys = set(DEFAULT_RELEVANT_ENVIRONMENT)
    keys.update(str(key) for key in extra_keys if str(key))
    return {
        key: str(source[key])
        for key in sorted(keys)
        if key in source and source[key] is not None
    }


def build_validation_cache_key(
    *,
    target_commit: str,
    command: str | ValidationCommand,
    environment: Mapping[str, object] | None = None,
    dependency_state: Mapping[str, object] | Sequence[object] | str = "",
    relevant_environment_keys: Iterable[str] = (),
) -> ValidationCacheKey:
    """Build a content-addressed cache key from every validation input class."""

    if isinstance(command, ValidationCommand):
        command_text = command.command
        extra_keys = tuple(command.environment_keys) + tuple(relevant_environment_keys)
    else:
        command_text = str(command)
        extra_keys = tuple(relevant_environment_keys)
    normalized_command = normalize_validation_command_text(command_text)
    environment_subset = relevant_environment(environment, extra_keys)
    if isinstance(dependency_state, str):
        dependency_fingerprint = dependency_state
    else:
        dependency_fingerprint = _sha256_bytes(
            _canonical_json(_json_safe(dependency_state)).encode("utf-8")
        )
    payload = {
        "target_commit": str(target_commit or "unknown"),
        "command": normalized_command,
        "environment": environment_subset,
        "dependency_state": dependency_fingerprint,
    }
    return ValidationCacheKey(
        target_commit=payload["target_commit"],
        command=normalized_command,
        environment=tuple(environment_subset.items()),
        dependency_state=dependency_fingerprint,
        digest=_sha256_bytes(_canonical_json(payload).encode("utf-8")),
    )


class ValidationResultCache:
    """Process-safe-enough content-addressed cache using atomic entry files.

    Every key has its own immutable JSON file.  Concurrent writers of the same
    successful result converge through ``os.replace``; corrupt or incompatible
    entries are ignored.  Failures are never stored.
    """

    def __init__(self, cache_dir: Path | str, *, max_age_seconds: float | None = None) -> None:
        self.cache_dir = Path(cache_dir)
        self.max_age_seconds = None if max_age_seconds is None else max(0.0, float(max_age_seconds))
        self._lock = threading.Lock()

    def _path(self, key: ValidationCacheKey | str) -> Path:
        digest = key.digest if isinstance(key, ValidationCacheKey) else str(key)
        return self.cache_dir / digest[:2] / f"{digest}.json"

    def get(self, key: ValidationCacheKey | str) -> dict[str, Any] | None:
        path = self._path(key)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return None
        if not isinstance(payload, dict) or payload.get("schema") != CACHE_SCHEMA:
            return None
        expected = key.digest if isinstance(key, ValidationCacheKey) else str(key)
        if payload.get("cache_key") != expected:
            return None
        if self.max_age_seconds is not None:
            created_epoch = float(payload.get("created_epoch") or 0.0)
            if created_epoch <= 0 or time.time() - created_epoch > self.max_age_seconds:
                return None
        result = payload.get("result")
        if not isinstance(result, dict) or int(result.get("returncode", 1)) != 0:
            return None
        return dict(result)

    def put(self, key: ValidationCacheKey, result: Mapping[str, object]) -> bool:
        if int(result.get("returncode", 1)) != 0 or result.get("timed_out"):
            return False
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": CACHE_SCHEMA,
            "cache_key": key.digest,
            "key": key.to_dict(),
            "created_at": utc_now(),
            "created_epoch": time.time(),
            "result": _json_safe(dict(result)),
        }
        encoded = (_canonical_json(payload) + "\n").encode("utf-8")
        with self._lock:
            fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
            finally:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
        return True

    def clear(self) -> int:
        removed = 0
        if not self.cache_dir.exists():
            return removed
        for path in self.cache_dir.glob("*/*.json"):
            try:
                path.unlink()
                removed += 1
            except OSError:
                continue
        return removed


# Shorter public name used by some embedding callers.
ValidationCache = ValidationResultCache


def resolve_target_commit(workspace_path: Path | str) -> str:
    """Resolve the immutable HEAD commit, or an explicit non-git sentinel."""

    result = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD^{commit}"],
        cwd=Path(workspace_path),
        text=True,
        capture_output=True,
        check=False,
    )
    stdout = str(result.stdout or "").strip()
    return stdout if result.returncode == 0 and stdout else "uncommitted"


def discover_changed_files(workspace_path: Path | str) -> tuple[str, ...]:
    """Return tracked, staged, and untracked candidate paths."""

    cwd = Path(workspace_path)
    paths: set[str] = set()
    commands = (
        ["git", "diff", "--name-only", "-z", "HEAD"],
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
    )
    for command in commands:
        result = subprocess.run(command, cwd=cwd, capture_output=True, check=False)
        if result.returncode != 0:
            continue
        stdout = result.stdout if isinstance(result.stdout, bytes) else str(result.stdout or "").encode()
        paths.update(
            item.decode("utf-8", errors="surrogateescape").replace("\\", "/")
            for item in stdout.split(b"\0")
            if item
        )
    return tuple(sorted(paths))


def _dependency_file(path: Path) -> bool:
    name = path.name
    lower = name.lower()
    return (
        name in DEPENDENCY_FILENAMES
        or lower.startswith("requirements") and lower.endswith((".txt", ".in"))
        or lower.endswith((".lock", ".lock.json"))
    )


def collect_dependency_state(
    workspace_path: Path | str,
    *,
    changed_files: Iterable[str] = (),
) -> dict[str, object]:
    """Fingerprint manifests, gitlinks, and dirty candidate content.

    The dirty-content component is essential because daemon validation happens
    before the implementation commit is created; HEAD alone identifies only the
    baseline shared by many candidate worktrees.
    """

    root = Path(workspace_path)
    files: dict[str, str] = {}
    if root.exists():
        skipped_dirs = {".git", ".pytest_cache", ".mypy_cache", "__pycache__", "node_modules", "dist", "build"}
        for directory, dirnames, filenames in os.walk(root):
            dirnames[:] = [name for name in dirnames if name not in skipped_dirs]
            parent = Path(directory)
            for filename in filenames:
                path = parent / filename
                if path.is_symlink() or not _dependency_file(path):
                    continue
                try:
                    relative = path.relative_to(root).as_posix()
                    files[relative] = _sha256_bytes(path.read_bytes())
                except OSError:
                    files[path.name] = "unreadable"

    submodules = subprocess.run(
        ["git", "submodule", "status", "--recursive"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    gitlinks = str(submodules.stdout or "").splitlines() if submodules.returncode == 0 else []

    dirty_hasher = hashlib.sha256()
    normalized_changed = tuple(sorted({str(path).replace("\\", "/") for path in changed_files if str(path)}))
    for relative in normalized_changed:
        dirty_hasher.update(relative.encode("utf-8", errors="surrogateescape"))
        path = root / relative
        try:
            if path.is_file() and not path.is_symlink():
                dirty_hasher.update(path.read_bytes())
            elif path.is_symlink():
                dirty_hasher.update(os.readlink(path).encode("utf-8", errors="surrogateescape"))
            else:
                dirty_hasher.update(b"<deleted-or-non-file>")
        except OSError:
            dirty_hasher.update(b"<unreadable>")

    return {
        "manifest_hashes": files,
        "gitlinks": gitlinks,
        "changed_files": list(normalized_changed),
        "candidate_content_sha256": dirty_hasher.hexdigest(),
    }


ValidationRunner = Callable[..., Mapping[str, object]]


def run_validation_command(
    *,
    spec: ValidationCommand,
    workspace_path: Path,
    timeout_seconds: float,
    environment: Mapping[str, str],
) -> dict[str, object]:
    """Default non-interactive shell runner with captured combined output."""

    started_at = utc_now()
    try:
        completed = subprocess.run(
            ["/bin/bash", "-lc", spec.command],
            cwd=workspace_path,
            text=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            check=False,
            env=dict(environment),
        )
        return {
            "command": spec.command,
            "raw_command": spec.raw_command or spec.command,
            "started_at": started_at,
            "finished_at": utc_now(),
            "returncode": int(completed.returncode),
            "output": completed.stdout or "",
        }
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        return {
            "command": spec.command,
            "raw_command": spec.raw_command or spec.command,
            "started_at": started_at,
            "finished_at": utc_now(),
            "returncode": 124,
            "timed_out": True,
            "output": output,
        }


def _object_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if isinstance(result, Mapping):
            return dict(result)
    to_record = getattr(value, "to_record", None)
    if callable(to_record):
        result = to_record()
        if isinstance(result, Mapping):
            return dict(result)
    return {}


def _enum_text(value: Any) -> str:
    return str(getattr(value, "value", value) or "").strip().lower()


def _verdict_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    attempted: bool | None = None,
    passed: bool | None = None,
    omitted_reason: str = "",
) -> dict[str, Any]:
    safe_records = [_json_safe(dict(item)) for item in records]
    was_attempted = bool(records) if attempted is None else bool(attempted)
    if not was_attempted:
        verdict = "not_run"
        effective_passed: bool | None = None if passed is None else bool(passed)
    else:
        effective_passed = bool(passed)
        verdict = "passed" if effective_passed else "failed"
    result: dict[str, Any] = {
        "attempted": was_attempted,
        "passed": effective_passed,
        "verdict": verdict,
        "results": safe_records,
    }
    if omitted_reason:
        result["reason"] = omitted_reason
    return result


def _command_verdict_records(
    report: Mapping[str, Any] | None,
    *,
    phase: str,
) -> list[dict[str, Any]]:
    if not report:
        return []
    records: list[dict[str, Any]] = []
    for item in report.get("results", ()) or ():
        mapping = _object_mapping(item)
        if mapping:
            mapping["phase"] = phase
            mapping["source"] = "validation_command"
            mapping["passed"] = int(mapping.get("returncode", 1)) == 0
            records.append(mapping)
    return records


def _proof_records_by_verdict(
    proof_result: Any,
    proof_scheduler: Any,
) -> dict[str, list[dict[str, Any]]]:
    """Project proof nodes into stable trust-boundary verdict buckets."""

    grouped = {name: [] for name in VALIDATION_VERDICT_KINDS}
    if proof_result is None:
        return grouped
    report = _object_mapping(proof_result)
    snapshot = report
    nested = report.get("snapshot")
    if isinstance(nested, Mapping):
        snapshot = dict(nested)

    plan = getattr(proof_result, "plan", None) or getattr(
        proof_scheduler, "plan", None
    )
    steps = getattr(plan, "steps", ()) if plan is not None else ()
    stage_by_step: dict[str, str] = {}
    for step in steps:
        step_id = str(getattr(step, "step_id", "") or "")
        stage = _enum_text(getattr(step, "stage", ""))
        if step_id:
            stage_by_step[step_id] = stage
    if not stage_by_step:
        plan_mapping = _object_mapping(plan)
        for step in plan_mapping.get("steps", ()) or ():
            step_mapping = _object_mapping(step)
            step_id = str(step_mapping.get("step_id") or "")
            if step_id:
                stage_by_step[step_id] = _enum_text(
                    step_mapping.get("stage")
                )

    nodes = snapshot.get("nodes", ()) or ()
    for raw_node in nodes:
        node = _object_mapping(raw_node)
        step_id = str(node.get("step_id") or "")
        stage = stage_by_step.get(step_id, _enum_text(node.get("stage")))
        if stage in {"translate", "translation"}:
            kind = "translation"
        elif stage in {"model_draft", "solve", "solver"}:
            kind = "solver"
        elif stage in {"reconstruct", "kernel_verify", "kernel"}:
            kind = "kernel"
        elif stage in {"validate", "validation", "test"}:
            kind = "test"
        elif stage in {"attest", "attestation"}:
            kind = "attestation"
        else:
            # Persistence and extension stages are retained in the proof
            # report but do not blur one of the five validation verdicts.
            continue
        state = _enum_text(node.get("state") or node.get("status"))
        reason = str(node.get("reason_code") or "")
        accepted = state == "succeeded" or (
            state == "cancelled" and reason.startswith("portfolio_concluded:")
        )
        record = dict(node)
        record.update(
            {
                "source": "proof_scheduler",
                "stage": stage,
                "passed": accepted,
            }
        )
        grouped[kind].append(record)

    # Lightweight or embedding schedulers sometimes expose attempts but not
    # node snapshots.  Retain those results instead of dropping verdict data.
    if not any(grouped.values()):
        for raw_attempt in snapshot.get("attempts", ()) or ():
            attempt = _object_mapping(raw_attempt)
            stage = _enum_text(attempt.get("stage"))
            if stage in {"translate", "translation"}:
                kind = "translation"
            elif stage in {"model_draft", "solve", "solver"}:
                kind = "solver"
            elif stage in {"reconstruct", "kernel_verify", "kernel"}:
                kind = "kernel"
            elif stage in {"validate", "validation", "test"}:
                kind = "test"
            elif stage in {"attest", "attestation"}:
                kind = "attestation"
            else:
                continue
            status = _enum_text(attempt.get("status"))
            record = dict(attempt)
            record.update(
                {
                    "source": "proof_scheduler",
                    "stage": stage,
                    "passed": status in {"succeeded", "cancelled"},
                }
            )
            grouped[kind].append(record)
    return grouped


def _proof_phase_passed(
    proof_result: Any,
    proof_scheduler: Any,
    stages: Sequence[str],
) -> bool:
    """Return whether every node in selected proof stages terminated safely."""

    report = _object_mapping(proof_result)
    nested = report.get("snapshot")
    snapshot = dict(nested) if isinstance(nested, Mapping) else report
    plan = getattr(proof_result, "plan", None) or getattr(
        proof_scheduler, "plan", None
    )
    selected = {_enum_text(item) for item in stages}
    stage_by_step = {
        str(getattr(step, "step_id", "") or ""): _enum_text(
            getattr(step, "stage", "")
        )
        for step in getattr(plan, "steps", ()) if plan is not None
    }
    selected_nodes = []
    for raw_node in snapshot.get("nodes", ()) or ():
        node = _object_mapping(raw_node)
        stage = stage_by_step.get(
            str(node.get("step_id") or ""), _enum_text(node.get("stage"))
        )
        if stage in selected:
            selected_nodes.append(node)
    if selected_nodes:
        return all(
            _enum_text(node.get("state") or node.get("status")) == "succeeded"
            or (
                _enum_text(node.get("state") or node.get("status"))
                == "cancelled"
                and str(node.get("reason_code") or "").startswith(
                    "portfolio_concluded:"
                )
            )
            for node in selected_nodes
        )
    succeeded = getattr(proof_result, "succeeded", None)
    if succeeded is None:
        succeeded = report.get("succeeded", report.get("passed"))
    return bool(succeeded)


class ValidationDAGError(ValueError):
    """Raised when a persisted impact graph or DAG receipt is inconsistent."""


class ValidationNodeDisposition(str, Enum):
    SELECTED = "selected"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"
    OMITTED = "omitted"


def _normalize_impact_path(value: object) -> str:
    text = str(value or "").strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    if not text or text.startswith("/") or "\0" in text:
        return ""
    parts: list[str] = []
    for part in text.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if not parts:
                return ""
            parts.pop()
        else:
            parts.append(part)
    return "/".join(parts)


@dataclass(frozen=True)
class ImpactDependencyGraph:
    """Canonical file dependency graph used for validation impact closure.

    Each mapping key is a dependent path and its values are the paths it
    directly consumes.  Selection walks the reverse graph from a changed path
    to every affected consumer and test.
    """

    dependencies: Mapping[str, Sequence[str]]
    repository_tree_id: str
    validation_targets: Mapping[str, Sequence[str]] = field(default_factory=dict)
    graph_version: str = "impact-dependency-v2"
    graph_id: str = ""

    def __post_init__(self) -> None:
        tree = str(self.repository_tree_id or "").strip()
        if not tree:
            raise ValidationDAGError("impact graph requires repository_tree_id")
        object.__setattr__(self, "repository_tree_id", tree)
        normalized: dict[str, tuple[str, ...]] = {}
        for raw_dependent, raw_dependencies in dict(self.dependencies or {}).items():
            dependent = _normalize_impact_path(raw_dependent)
            if not dependent:
                raise ValidationDAGError("impact graph contains an unsafe dependent path")
            if isinstance(raw_dependencies, str):
                values: Iterable[object] = (raw_dependencies,)
            else:
                values = raw_dependencies
            normalized_values: set[str] = set()
            for value in values:
                path = _normalize_impact_path(value)
                if not path:
                    raise ValidationDAGError(
                        "impact graph contains an unsafe dependency path"
                    )
                normalized_values.add(path)
            direct = tuple(sorted(normalized_values))
            if dependent in direct:
                raise ValidationDAGError("impact graph contains a self dependency")
            normalized[dependent] = direct
        object.__setattr__(self, "dependencies", dict(sorted(normalized.items())))
        validation_targets: dict[str, tuple[str, ...]] = {}
        known_paths = set(self.reverse_dependencies)
        for raw_validation_id, raw_paths in dict(
            self.validation_targets or {}
        ).items():
            validation_id = str(raw_validation_id or "").strip()
            if not validation_id:
                raise ValidationDAGError(
                    "impact graph contains an empty validation identity"
                )
            values: Iterable[object] = (
                (raw_paths,) if isinstance(raw_paths, str) else raw_paths
            )
            normalized_paths: set[str] = set()
            for value in values:
                path = _normalize_impact_path(value)
                if not path:
                    raise ValidationDAGError(
                        "impact graph contains an unsafe validation target path"
                    )
                normalized_paths.add(path)
            paths = tuple(sorted(normalized_paths))
            if not paths:
                raise ValidationDAGError(
                    f"impact validation {validation_id!r} has no target paths"
                )
            unknown = tuple(path for path in paths if path not in known_paths)
            if unknown:
                raise ValidationDAGError(
                    "impact validation targets paths outside the dependency graph: "
                    + ", ".join(unknown)
                )
            validation_targets[validation_id] = paths
        object.__setattr__(
            self, "validation_targets", dict(sorted(validation_targets.items()))
        )
        version = str(self.graph_version or "").strip()
        if not version:
            raise ValidationDAGError("impact graph version is required")
        object.__setattr__(self, "graph_version", version)
        claimed = str(self.graph_id or "").strip()
        object.__setattr__(self, "graph_id", "")
        actual = _sha256_bytes(_canonical_json(self._identity_payload()).encode("utf-8"))
        if claimed and claimed != actual:
            raise ValidationDAGError("impact graph identity mismatch")
        object.__setattr__(self, "graph_id", actual)

    def _identity_payload(self) -> dict[str, object]:
        return {
            "repository_tree_id": self.repository_tree_id,
            "graph_version": self.graph_version,
            "dependencies": self.dependencies,
            "validation_targets": self.validation_targets,
        }

    @property
    def reverse_dependencies(self) -> Mapping[str, tuple[str, ...]]:
        reverse: dict[str, set[str]] = {}
        for dependent, dependencies in self.dependencies.items():
            reverse.setdefault(dependent, set())
            for dependency in dependencies:
                reverse.setdefault(dependency, set()).add(dependent)
        return {
            path: tuple(sorted(dependents))
            for path, dependents in sorted(reverse.items())
        }

    def affected_paths(self, changed_paths: Iterable[str]) -> tuple[str, ...]:
        roots = tuple(
            sorted(
                {
                    path
                    for value in changed_paths
                    if (path := _normalize_impact_path(value))
                }
            )
        )
        reverse = self.reverse_dependencies
        visited = set(roots)
        pending = deque(roots)
        while pending:
            current = pending.popleft()
            for dependent in reverse.get(current, ()):
                if dependent not in visited:
                    visited.add(dependent)
                    pending.append(dependent)
        return tuple(sorted(visited))

    def impact_path(self, source: str, target: str) -> tuple[str, ...]:
        start = _normalize_impact_path(source)
        goal = _normalize_impact_path(target)
        if not start or not goal:
            return ()
        reverse = self.reverse_dependencies
        pending = deque([(start, (start,))])
        visited = {start}
        while pending:
            current, path = pending.popleft()
            if current == goal:
                return path
            for dependent in reverse.get(current, ()):
                if dependent not in visited:
                    visited.add(dependent)
                    pending.append((dependent, (*path, dependent)))
        return ()

    def required_validations(
        self, affected_paths: Iterable[str]
    ) -> Mapping[str, tuple[str, ...]]:
        """Return every declared validation intersecting the impact closure."""

        affected = {
            path
            for value in affected_paths
            if (path := _normalize_impact_path(value))
        }
        return {
            validation_id: tuple(
                path for path in paths if path in affected
            )
            for validation_id, paths in self.validation_targets.items()
            if affected.intersection(paths)
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._identity_payload(), "graph_id": self.graph_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ImpactDependencyGraph":
        return cls(
            dependencies=payload.get("dependencies") or {},
            repository_tree_id=str(payload.get("repository_tree_id") or ""),
            validation_targets=payload.get("validation_targets") or {},
            graph_version=str(payload.get("graph_version") or "impact-dependency-v2"),
            graph_id=str(payload.get("graph_id") or ""),
        )


@dataclass(frozen=True)
class ValidationDAGNodeRecord:
    node_id: str
    command: str
    stage: str
    disposition: ValidationNodeDisposition
    reason: str
    impact_paths: tuple[str, ...] = ()
    returncode: int | None = None
    result_digest: str = ""
    validation_id: str = ""
    selected: bool = False
    mandatory: bool = False
    selection_reason: str = ""
    depends_on: tuple[str, ...] = ()
    observed_seeded_defect_id: str = ""

    def __post_init__(self) -> None:
        for name in (
            "node_id",
            "command",
            "stage",
            "reason",
            "result_digest",
            "validation_id",
            "selection_reason",
            "observed_seeded_defect_id",
        ):
            object.__setattr__(self, name, str(getattr(self, name) or "").strip())
        if not self.node_id or not self.command or not self.stage or not self.reason:
            raise ValidationDAGError("validation node record is incomplete")
        object.__setattr__(
            self, "disposition", ValidationNodeDisposition(self.disposition)
        )
        object.__setattr__(
            self,
            "impact_paths",
            tuple(
                sorted(
                    {
                        path
                        for value in self.impact_paths
                        if (path := _normalize_impact_path(value))
                    }
                )
            ),
        )
        if self.returncode is not None:
            if isinstance(self.returncode, bool):
                raise ValidationDAGError("validation returncode must be an integer")
            object.__setattr__(self, "returncode", int(self.returncode))
        if self.disposition in {
            ValidationNodeDisposition.SUCCEEDED,
            ValidationNodeDisposition.FAILED,
        }:
            if self.returncode is None or not self.result_digest:
                raise ValidationDAGError(
                    "executed validation node requires a bound result"
                )
        object.__setattr__(self, "selected", bool(self.selected))
        object.__setattr__(self, "mandatory", bool(self.mandatory))
        object.__setattr__(
            self,
            "depends_on",
            tuple(
                sorted(
                    {
                        str(value or "").strip()
                        for value in self.depends_on
                        if str(value or "").strip()
                    }
                )
            ),
        )
        if self.node_id in self.depends_on:
            raise ValidationDAGError("validation node cannot depend on itself")
        if self.mandatory and (not self.selected or not self.validation_id):
            raise ValidationDAGError(
                "mandatory validation node must be selected and identified"
            )
        if self.selected and self.disposition is ValidationNodeDisposition.OMITTED:
            raise ValidationDAGError("selected validation node cannot be omitted")
        if not self.selected and self.disposition is not ValidationNodeDisposition.OMITTED:
            raise ValidationDAGError(
                "unselected validation node must have omitted disposition"
            )
        if self.selected and not self.selection_reason:
            raise ValidationDAGError(
                "selected validation node requires its selection reason"
            )
        if (
            self.disposition
            not in {
                ValidationNodeDisposition.SUCCEEDED,
                ValidationNodeDisposition.FAILED,
            }
            and self.observed_seeded_defect_id
        ):
            raise ValidationDAGError(
                "unexecuted validation cannot observe a seeded defect"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "command": self.command,
            "stage": self.stage,
            "disposition": self.disposition.value,
            "reason": self.reason,
            "impact_paths": self.impact_paths,
            "returncode": self.returncode,
            "result_digest": self.result_digest,
            "validation_id": self.validation_id,
            "selected": self.selected,
            "mandatory": self.mandatory,
            "selection_reason": self.selection_reason,
            "depends_on": self.depends_on,
            "observed_seeded_defect_id": self.observed_seeded_defect_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ValidationDAGNodeRecord":
        return cls(
            node_id=str(payload.get("node_id") or ""),
            command=str(payload.get("command") or ""),
            stage=str(payload.get("stage") or ""),
            disposition=payload.get("disposition", ""),
            reason=str(payload.get("reason") or ""),
            impact_paths=tuple(payload.get("impact_paths") or ()),
            returncode=payload.get("returncode"),
            result_digest=str(payload.get("result_digest") or ""),
            validation_id=str(payload.get("validation_id") or ""),
            selected=payload.get("selected", False),
            mandatory=payload.get("mandatory", False),
            selection_reason=str(payload.get("selection_reason") or ""),
            depends_on=tuple(payload.get("depends_on") or ()),
            observed_seeded_defect_id=str(
                payload.get("observed_seeded_defect_id") or ""
            ),
        )


class ValidationAuthorityDisposition(str, Enum):
    PENDING = "pending"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class ValidationAuthorityGateRecord:
    """One downstream authority boundary affected by the validation result."""

    gate: str
    disposition: ValidationAuthorityDisposition
    reason: str
    depends_on: tuple[str, ...]

    def __post_init__(self) -> None:
        gate = str(self.gate or "").strip()
        reason = str(self.reason or "").strip()
        if gate not in REQUIRED_AUTHORITY_GATES:
            raise ValidationDAGError(f"unsupported validation authority gate: {gate}")
        if not reason:
            raise ValidationDAGError("validation authority gate requires a reason")
        object.__setattr__(self, "gate", gate)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(
            self,
            "disposition",
            ValidationAuthorityDisposition(self.disposition),
        )
        dependencies = tuple(
            sorted(
                {
                    str(value or "").strip()
                    for value in self.depends_on
                    if str(value or "").strip()
                }
            )
        )
        object.__setattr__(self, "depends_on", dependencies)

    def to_dict(self) -> dict[str, object]:
        return {
            "gate": self.gate,
            "disposition": self.disposition.value,
            "reason": self.reason,
            "depends_on": self.depends_on,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "ValidationAuthorityGateRecord":
        return cls(
            gate=str(payload.get("gate") or ""),
            disposition=payload.get("disposition", ""),
            reason=str(payload.get("reason") or ""),
            depends_on=tuple(payload.get("depends_on") or ()),
        )


@dataclass(frozen=True)
class TransitiveImpactValidationEvidence:
    requirement_id: str
    repository_tree_id: str
    objective_id: str
    policy_id: str
    graph_id: str
    seeded_defect_id: str
    seeded_defect_path: str
    impact_path: tuple[str, ...]
    failing_node_id: str
    failing_result_digest: str
    receipt_id: str
    evidence_id: str = ""

    def __post_init__(self) -> None:
        for name in (
            "requirement_id",
            "repository_tree_id",
            "objective_id",
            "policy_id",
            "graph_id",
            "seeded_defect_id",
            "failing_node_id",
            "failing_result_digest",
            "receipt_id",
        ):
            object.__setattr__(self, name, str(getattr(self, name) or "").strip())
            if not getattr(self, name):
                raise ValidationDAGError(f"{name} is required")
        if self.requirement_id != TRANSITIVE_IMPACT_REQUIREMENT_ID:
            raise ValidationDAGError("unsupported transitive-impact requirement")
        defect_path = _normalize_impact_path(self.seeded_defect_path)
        path = tuple(_normalize_impact_path(item) for item in self.impact_path)
        if not defect_path or any(not item for item in path):
            raise ValidationDAGError("transitive impact path is malformed")
        if len(path) < 3 or path[0] != defect_path:
            raise ValidationDAGError(
                "evidence requires a genuinely transitive impact path"
            )
        object.__setattr__(self, "seeded_defect_path", defect_path)
        object.__setattr__(self, "impact_path", path)
        claimed = str(self.evidence_id or "").strip()
        object.__setattr__(self, "evidence_id", "")
        actual = _sha256_bytes(_canonical_json(self._identity_payload()).encode("utf-8"))
        if claimed and claimed != actual:
            raise ValidationDAGError("transitive impact evidence identity mismatch")
        object.__setattr__(self, "evidence_id", actual)

    @property
    def proved_requirement_ids(self) -> tuple[str, ...]:
        return (self.requirement_id,)

    def _identity_payload(self) -> dict[str, object]:
        return {
            "schema": TRANSITIVE_IMPACT_EVIDENCE_SCHEMA,
            "requirement_id": self.requirement_id,
            "repository_tree_id": self.repository_tree_id,
            "objective_id": self.objective_id,
            "policy_id": self.policy_id,
            "graph_id": self.graph_id,
            "seeded_defect_id": self.seeded_defect_id,
            "seeded_defect_path": self.seeded_defect_path,
            "impact_path": self.impact_path,
            "failing_node_id": self.failing_node_id,
            "failing_result_digest": self.failing_result_digest,
            "receipt_id": self.receipt_id,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._identity_payload(), "evidence_id": self.evidence_id}

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "TransitiveImpactValidationEvidence":
        schema = str(payload.get("schema") or TRANSITIVE_IMPACT_EVIDENCE_SCHEMA)
        if schema != TRANSITIVE_IMPACT_EVIDENCE_SCHEMA:
            raise ValidationDAGError(
                f"unsupported transitive impact evidence schema: {schema}"
            )
        return cls(
            requirement_id=str(payload.get("requirement_id") or ""),
            repository_tree_id=str(payload.get("repository_tree_id") or ""),
            objective_id=str(payload.get("objective_id") or ""),
            policy_id=str(payload.get("policy_id") or ""),
            graph_id=str(payload.get("graph_id") or ""),
            seeded_defect_id=str(payload.get("seeded_defect_id") or ""),
            seeded_defect_path=str(payload.get("seeded_defect_path") or ""),
            impact_path=tuple(payload.get("impact_path") or ()),
            failing_node_id=str(payload.get("failing_node_id") or ""),
            failing_result_digest=str(payload.get("failing_result_digest") or ""),
            receipt_id=str(payload.get("receipt_id") or ""),
            evidence_id=str(payload.get("evidence_id") or ""),
        )


@dataclass(frozen=True)
class ValidationDAGReceipt:
    repository_tree_id: str
    objective_id: str
    policy_id: str
    proposal_receipt_id: str
    graph_id: str
    changed_paths: tuple[str, ...]
    affected_paths: tuple[str, ...]
    nodes: tuple[ValidationDAGNodeRecord, ...]
    passed: bool
    impact_graph: ImpactDependencyGraph | None = None
    required_validation_ids: tuple[str, ...] = ()
    selected_node_ids: tuple[str, ...] = ()
    coverage_complete: bool = False
    authority_gates: tuple[ValidationAuthorityGateRecord, ...] = ()
    seeded_defect_id: str = ""
    seeded_defect_path: str = ""
    uncovered_impact: bool = False
    transitive_evidence: TransitiveImpactValidationEvidence | None = None
    receipt_id: str = ""

    def __post_init__(self) -> None:
        for name in (
            "repository_tree_id",
            "objective_id",
            "policy_id",
            "proposal_receipt_id",
            "graph_id",
            "seeded_defect_id",
            "seeded_defect_path",
        ):
            object.__setattr__(self, name, str(getattr(self, name) or "").strip())
        for name in (
            "repository_tree_id",
            "objective_id",
            "policy_id",
            "proposal_receipt_id",
        ):
            if not getattr(self, name):
                raise ValidationDAGError(f"{name} is required")
        graph = self.impact_graph
        if graph is not None and not isinstance(graph, ImpactDependencyGraph):
            graph = ImpactDependencyGraph.from_dict(graph)
        if graph is not None:
            if graph.graph_id != self.graph_id:
                raise ValidationDAGError(
                    "validation DAG graph payload does not match graph identity"
                )
            if graph.repository_tree_id != self.repository_tree_id:
                raise ValidationDAGError(
                    "validation DAG graph is stale for its repository tree"
                )
        object.__setattr__(self, "impact_graph", graph)
        object.__setattr__(
            self,
            "changed_paths",
            tuple(sorted({_normalize_impact_path(item) for item in self.changed_paths if _normalize_impact_path(item)})),
        )
        object.__setattr__(
            self,
            "affected_paths",
            tuple(sorted({_normalize_impact_path(item) for item in self.affected_paths if _normalize_impact_path(item)})),
        )
        nodes = tuple(
            item
            if isinstance(item, ValidationDAGNodeRecord)
            else ValidationDAGNodeRecord.from_dict(item)
            for item in self.nodes
        )
        if len({node.node_id for node in nodes}) != len(nodes):
            raise ValidationDAGError("validation DAG contains duplicate nodes")
        object.__setattr__(
            self, "nodes", tuple(sorted(nodes, key=lambda node: node.node_id))
        )
        by_id = {node.node_id: node for node in nodes}
        for node in nodes:
            unknown = tuple(
                dependency
                for dependency in node.depends_on
                if dependency not in by_id
            )
            if unknown:
                raise ValidationDAGError(
                    "validation DAG node depends on an unknown node"
                )
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in visited:
                return
            if node_id in visiting:
                raise ValidationDAGError("validation DAG contains a dependency cycle")
            visiting.add(node_id)
            for dependency in by_id[node_id].depends_on:
                visit(dependency)
            visiting.remove(node_id)
            visited.add(node_id)

        for node_id in by_id:
            visit(node_id)
        required_validation_ids = tuple(
            sorted(
                {
                    str(value or "").strip()
                    for value in self.required_validation_ids
                    if str(value or "").strip()
                }
            )
        )
        selected_node_ids = tuple(
            sorted(
                {
                    str(value or "").strip()
                    for value in self.selected_node_ids
                    if str(value or "").strip()
                }
            )
        )
        actual_selected = tuple(
            sorted(node.node_id for node in nodes if node.selected)
        )
        if selected_node_ids != actual_selected:
            raise ValidationDAGError(
                "validation DAG selected-node population does not match nodes"
            )
        object.__setattr__(
            self, "required_validation_ids", required_validation_ids
        )
        object.__setattr__(self, "selected_node_ids", selected_node_ids)
        previous_stage_ids: tuple[str, ...] = ()
        for stage in STRICT_VALIDATION_STAGE_ORDER:
            stage_ids = tuple(
                sorted(
                    node.node_id
                    for node in nodes
                    if node.selected and node.stage == stage.label
                )
            )
            for node_id in stage_ids:
                if by_id[node_id].depends_on != previous_stage_ids:
                    raise ValidationDAGError(
                        "validation DAG dependency edges do not match strict "
                        "ready-node barriers"
                    )
            if stage_ids:
                previous_stage_ids = stage_ids
        known_stage_labels = {stage.label for stage in STRICT_VALIDATION_STAGE_ORDER}
        if any(node.selected and node.stage not in known_stage_labels for node in nodes):
            raise ValidationDAGError("validation DAG contains an unknown stage")
        required_counts = {
            validation_id: sum(
                1
                for node in nodes
                if node.validation_id == validation_id
                and node.selected
                and node.mandatory
            )
            for validation_id in required_validation_ids
        }
        required_nodes = {
            validation_id: tuple(
                node
                for node in nodes
                if node.validation_id == validation_id
                and node.selected
                and node.mandatory
            )
            for validation_id in required_validation_ids
        }
        graph_requirement_map = (
            graph.required_validations(self.affected_paths)
            if graph is not None
            else {}
        )
        graph_requirement_ids = tuple(sorted(graph_requirement_map))
        derived_coverage = bool(
            graph is not None
            and self.changed_paths
            and required_validation_ids
            and required_validation_ids == graph_requirement_ids
            and all(count == 1 for count in required_counts.values())
            and all(
                set(graph_requirement_map[validation_id]).issubset(
                    required_nodes[validation_id][0].impact_paths
                )
                for validation_id in required_validation_ids
                if required_nodes[validation_id]
            )
        )
        if bool(self.coverage_complete) != derived_coverage:
            raise ValidationDAGError(
                "validation DAG coverage verdict does not match graph declarations"
            )
        object.__setattr__(self, "coverage_complete", derived_coverage)
        object.__setattr__(self, "uncovered_impact", bool(self.uncovered_impact))
        actual_passed = bool(
            derived_coverage
            and not self.uncovered_impact
            and selected_node_ids
            and all(
                node.disposition is ValidationNodeDisposition.SUCCEEDED
                for node in nodes
                if node.selected
            )
        )
        if bool(self.passed) != actual_passed:
            raise ValidationDAGError("validation DAG verdict does not match nodes")
        object.__setattr__(self, "passed", actual_passed)
        gates = tuple(
            item
            if isinstance(item, ValidationAuthorityGateRecord)
            else ValidationAuthorityGateRecord.from_dict(item)
            for item in self.authority_gates
        )
        if tuple(sorted(gate.gate for gate in gates)) != tuple(
            sorted(REQUIRED_AUTHORITY_GATES)
        ):
            raise ValidationDAGError(
                "validation DAG must record every downstream authority gate exactly once"
            )
        expected_gate_disposition = (
            ValidationAuthorityDisposition.PENDING
            if actual_passed
            else ValidationAuthorityDisposition.BLOCKED
        )
        for gate in gates:
            if gate.disposition is not expected_gate_disposition:
                raise ValidationDAGError(
                    "validation authority disposition does not match DAG verdict"
                )
            if gate.depends_on != selected_node_ids:
                raise ValidationDAGError(
                    "validation authority gate must bind the complete selection"
                )
        object.__setattr__(
            self, "authority_gates", tuple(sorted(gates, key=lambda item: item.gate))
        )
        if bool(self.seeded_defect_id) != bool(self.seeded_defect_path):
            raise ValidationDAGError(
                "seeded defect identity and path must be provided together"
            )
        if self.seeded_defect_path:
            normalized_seed = _normalize_impact_path(self.seeded_defect_path)
            if not normalized_seed:
                raise ValidationDAGError("seeded defect path is malformed")
            object.__setattr__(self, "seeded_defect_path", normalized_seed)
        evidence = self.transitive_evidence
        if evidence is not None and not isinstance(
            evidence, TransitiveImpactValidationEvidence
        ):
            evidence = TransitiveImpactValidationEvidence.from_dict(evidence)
        object.__setattr__(self, "transitive_evidence", None)
        claimed = str(self.receipt_id or "").strip()
        object.__setattr__(self, "receipt_id", "")
        actual = _sha256_bytes(_canonical_json(self._identity_payload()).encode("utf-8"))
        if claimed and claimed != actual:
            raise ValidationDAGError("validation DAG receipt identity mismatch")
        object.__setattr__(self, "receipt_id", actual)
        if evidence is not None:
            failed = by_id.get(evidence.failing_node_id)
            expected_path = (
                graph.impact_path(
                    evidence.seeded_defect_path, evidence.impact_path[-1]
                )
                if graph is not None
                else ()
            )
            if (
                evidence.receipt_id != actual
                or evidence.repository_tree_id != self.repository_tree_id
                or evidence.objective_id != self.objective_id
                or evidence.policy_id != self.policy_id
                or evidence.graph_id != self.graph_id
                or evidence.seeded_defect_id != self.seeded_defect_id
                or evidence.seeded_defect_path != self.seeded_defect_path
                or failed is None
                or failed.disposition is not ValidationNodeDisposition.FAILED
                or failed.result_digest != evidence.failing_result_digest
                or failed.observed_seeded_defect_id
                != evidence.seeded_defect_id
                or failed.validation_id not in required_validation_ids
                or evidence.impact_path[-1] not in failed.impact_paths
                or evidence.impact_path != expected_path
                or evidence.seeded_defect_path not in self.changed_paths
                or any(
                    gate.disposition
                    is not ValidationAuthorityDisposition.BLOCKED
                    for gate in gates
                )
            ):
                raise ValidationDAGError(
                    "transitive evidence is detached from validation DAG receipt"
                )
            object.__setattr__(self, "transitive_evidence", evidence)

    def _identity_payload(self) -> dict[str, object]:
        return {
            "schema": VALIDATION_DAG_RECEIPT_SCHEMA,
            "repository_tree_id": self.repository_tree_id,
            "objective_id": self.objective_id,
            "policy_id": self.policy_id,
            "proposal_receipt_id": self.proposal_receipt_id,
            "graph_id": self.graph_id,
            "impact_graph": (
                self.impact_graph.to_dict()
                if self.impact_graph is not None
                else None
            ),
            "changed_paths": self.changed_paths,
            "affected_paths": self.affected_paths,
            "nodes": [node.to_dict() for node in self.nodes],
            "required_validation_ids": self.required_validation_ids,
            "selected_node_ids": self.selected_node_ids,
            "coverage_complete": self.coverage_complete,
            "authority_gates": [
                gate.to_dict() for gate in self.authority_gates
            ],
            "passed": self.passed,
            "seeded_defect_id": self.seeded_defect_id,
            "seeded_defect_path": self.seeded_defect_path,
            "uncovered_impact": self.uncovered_impact,
        }

    @property
    def proved_requirement_ids(self) -> tuple[str, ...]:
        return (
            self.transitive_evidence.proved_requirement_ids
            if self.transitive_evidence is not None
            else ()
        )

    @property
    def completion_authoritative(self) -> bool:
        return False

    @property
    def proof_authoritative(self) -> bool:
        """A passing validation DAG authorizes proof work, not proof claims."""

        return False

    def to_dict(self) -> dict[str, object]:
        return {
            **self._identity_payload(),
            "receipt_id": self.receipt_id,
            "transitive_evidence": (
                self.transitive_evidence.to_dict()
                if self.transitive_evidence is not None
                else None
            ),
            "proved_requirement_ids": self.proved_requirement_ids,
            "proof_authoritative": False,
            "completion_authoritative": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ValidationDAGReceipt":
        schema = str(payload.get("schema") or VALIDATION_DAG_RECEIPT_SCHEMA)
        if schema != VALIDATION_DAG_RECEIPT_SCHEMA:
            raise ValidationDAGError(f"unsupported validation DAG schema: {schema}")
        if payload.get("completion_authoritative") not in (None, False):
            raise ValidationDAGError("validation DAG cannot claim completion")
        if payload.get("proof_authoritative") not in (None, False):
            raise ValidationDAGError("validation DAG cannot claim code-proof authority")
        base = cls(
            repository_tree_id=str(payload.get("repository_tree_id") or ""),
            objective_id=str(payload.get("objective_id") or ""),
            policy_id=str(payload.get("policy_id") or ""),
            proposal_receipt_id=str(payload.get("proposal_receipt_id") or ""),
            graph_id=str(payload.get("graph_id") or ""),
            impact_graph=(
                ImpactDependencyGraph.from_dict(payload["impact_graph"])
                if payload.get("impact_graph")
                else None
            ),
            changed_paths=tuple(payload.get("changed_paths") or ()),
            affected_paths=tuple(payload.get("affected_paths") or ()),
            nodes=tuple(
                ValidationDAGNodeRecord.from_dict(item)
                for item in payload.get("nodes") or ()
            ),
            passed=payload.get("passed", False),
            required_validation_ids=tuple(
                payload.get("required_validation_ids") or ()
            ),
            selected_node_ids=tuple(payload.get("selected_node_ids") or ()),
            coverage_complete=payload.get("coverage_complete", False),
            authority_gates=tuple(
                ValidationAuthorityGateRecord.from_dict(item)
                for item in payload.get("authority_gates") or ()
            ),
            seeded_defect_id=str(payload.get("seeded_defect_id") or ""),
            seeded_defect_path=str(payload.get("seeded_defect_path") or ""),
            uncovered_impact=payload.get("uncovered_impact", False),
            receipt_id=str(payload.get("receipt_id") or ""),
        )
        evidence_payload = payload.get("transitive_evidence")
        if evidence_payload:
            base = cls(
                repository_tree_id=base.repository_tree_id,
                objective_id=base.objective_id,
                policy_id=base.policy_id,
                proposal_receipt_id=base.proposal_receipt_id,
                graph_id=base.graph_id,
                impact_graph=base.impact_graph,
                changed_paths=base.changed_paths,
                affected_paths=base.affected_paths,
                nodes=base.nodes,
                passed=base.passed,
                required_validation_ids=base.required_validation_ids,
                selected_node_ids=base.selected_node_ids,
                coverage_complete=base.coverage_complete,
                authority_gates=base.authority_gates,
                seeded_defect_id=base.seeded_defect_id,
                seeded_defect_path=base.seeded_defect_path,
                uncovered_impact=base.uncovered_impact,
                transitive_evidence=TransitiveImpactValidationEvidence.from_dict(
                    evidence_payload
                ),
                receipt_id=base.receipt_id,
            )
        claimed = tuple(payload.get("proved_requirement_ids") or ())
        if claimed and claimed != base.proved_requirement_ids:
            raise ValidationDAGError("validation DAG requirement claims mismatch")
        return base


def _authority_gate_records(
    selected_node_ids: Iterable[str], *, passed: bool
) -> tuple[ValidationAuthorityGateRecord, ...]:
    dependencies = tuple(sorted(set(selected_node_ids)))
    disposition = (
        ValidationAuthorityDisposition.PENDING
        if passed
        else ValidationAuthorityDisposition.BLOCKED
    )
    reason = (
        "validation_passed_requires_independent_authority"
        if passed
        else "validation_dag_failed"
    )
    return tuple(
        ValidationAuthorityGateRecord(
            gate=gate,
            disposition=disposition,
            reason=reason,
            depends_on=dependencies,
        )
        for gate in REQUIRED_AUTHORITY_GATES
    )


class ValidationScheduler:
    """Execute validation stages with bounded weighted parallelism and caching."""

    def __init__(
        self,
        *,
        cache: ValidationResultCache | None = None,
        cache_dir: Path | str | None = None,
        max_workers: int = 2,
        resource_budget: int | None = None,
        resource_scheduler: ResourceScheduler | None = None,
        resource_lease_budget: ResourceLeaseBudget | Mapping[str, Any] | None = None,
        resource_policy: ResourcePolicy | Mapping[str, Any] | None = None,
        host_resource_source: (
            Callable[..., Any] | HostResourceSnapshot | Mapping[str, Any] | None
        ) = None,
        provider_capacity_source: (
            Callable[..., Any] | Mapping[str, Any] | Sequence[Any] | None
        ) = None,
        resource_admission_timeout_seconds: float = 5.0,
        default_timeout_seconds: float = 1800.0,
        runner: ValidationRunner | None = None,
    ) -> None:
        if int(max_workers) <= 0:
            raise ValueError("max_workers must be positive")
        budget = int(resource_budget if resource_budget is not None else max_workers)
        if budget <= 0:
            raise ValueError("resource_budget must be positive")
        if cache is not None and cache_dir is not None:
            raise ValueError("provide cache or cache_dir, not both")
        if resource_scheduler is not None and resource_policy is not None:
            raise ValueError(
                "resource_scheduler cannot be combined with resource_policy"
            )
        self.cache = cache or (ValidationResultCache(cache_dir) if cache_dir is not None else None)
        self.max_workers = int(max_workers)
        self.resource_budget = budget
        self._implicit_resource_admission = (
            resource_scheduler is None
            and resource_policy is None
            and resource_lease_budget is None
            and host_resource_source is None
            and provider_capacity_source is None
        )
        if resource_scheduler is None:
            if isinstance(resource_policy, ResourcePolicy):
                policy = resource_policy
            else:
                policy_values = dict(resource_policy or {})
                policy_values.setdefault(
                    "max_lanes", max(self.max_workers, self.resource_budget)
                )
                policy_values.setdefault(
                    "max_cpu_proof_concurrency", self.resource_budget
                )
                policy_values.setdefault("require_provider_telemetry", False)
                policy = ResourcePolicy.from_mapping(policy_values)
            resource_scheduler = ResourceScheduler(policy)
        self.resource_scheduler = resource_scheduler
        if resource_lease_budget is None:
            self.resource_lease_budget = ResourceLeaseBudget.from_resource_budget(
                {},
                max_parallel=self.resource_budget,
                max_cpu_proof_concurrency=self.resource_budget,
                max_model_concurrency=self.resource_budget,
                max_artifact_concurrency=self.resource_budget,
                maximum_provider_latency_ms=(
                    self.resource_scheduler.policy.maximum_provider_latency_ms
                ),
            )
        elif isinstance(resource_lease_budget, ResourceLeaseBudget):
            self.resource_lease_budget = resource_lease_budget
        else:
            self.resource_lease_budget = ResourceLeaseBudget.from_mapping(
                resource_lease_budget
            )
        self._host_resource_source = host_resource_source
        self._provider_capacity_source = provider_capacity_source
        self.resource_admission_timeout_seconds = max(
            0.0, float(resource_admission_timeout_seconds)
        )
        self._resource_decisions: dict[str, AdmissionDecision] = {}
        self._resource_decision_lock = threading.Lock()
        self.default_timeout_seconds = max(0.001, float(default_timeout_seconds))
        self.runner = runner or run_validation_command

    @staticmethod
    def _read_capacity_source(source: Any, spec: ValidationCommand) -> Any:
        """Read a static or callable telemetry source without masking errors."""

        if not callable(source):
            return source
        try:
            signature = inspect.signature(source)
        except (TypeError, ValueError):
            return source()
        positional = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            and parameter.default is inspect.Parameter.empty
        ]
        return source(spec) if positional else source()

    def _resource_requirement(
        self, spec: ValidationCommand
    ) -> LaneResourceRequirements:
        command_id = _sha256_bytes(spec.command.encode("utf-8"))[:16]
        process_slots = min(
            self.resource_budget,
            self.resource_lease_budget.max_processes,
            max(1, int(spec.resource_cost)),
        )
        return LaneResourceRequirements(
            lane_id=f"validation:{spec.ordinal}:{command_id}",
            resource_class=ProofResourceClass.VALIDATION.value,
            process_slots=process_slots,
        )

    def _acquire_resource(
        self,
        spec: ValidationCommand,
        *,
        workspace_path: Path,
    ) -> tuple[AdmissionDecision, ResourceAdmissionLease | None]:
        transient_reasons = {
            "host_worker_capacity",
            "cpu_proof_concurrency",
            "resource_class_concurrency",
            "lease_process_capacity",
            "lease_cpu_proof_concurrency",
        }
        deadline = time.monotonic() + self.resource_admission_timeout_seconds
        while True:
            host = self._read_capacity_source(self._host_resource_source, spec)
            if host is None and self._implicit_resource_admission:
                # Legacy callers did not opt into live telemetry.  Use a stable
                # view while still reserving against the shared scheduler.
                host = HostResourceSnapshot(
                    worker_limit=max(self.max_workers, self.resource_budget),
                    available_worker_capacity=max(
                        self.max_workers, self.resource_budget
                    ),
                )
            providers = self._read_capacity_source(
                self._provider_capacity_source, spec
            )
            decision, lease = self.resource_scheduler.acquire(
                self._resource_requirement(spec),
                budget=self.resource_lease_budget,
                host=host,
                providers=providers,
                path=workspace_path,
            )
            with self._resource_decision_lock:
                self._resource_decisions[decision.lane_id] = decision
            if lease is not None:
                return decision, lease
            reasons = set(decision.reasons)
            if (
                not reasons.intersection(transient_reasons)
                or time.monotonic() >= deadline
            ):
                return decision, None
            time.sleep(0.01)

    @property
    def resource_decisions(self) -> Mapping[str, AdmissionDecision]:
        with self._resource_decision_lock:
            return dict(self._resource_decisions)

    def _execute(
        self,
        spec: ValidationCommand,
        *,
        workspace_path: Path,
        target_commit: str,
        environment: Mapping[str, str],
        dependency_state: Mapping[str, object] | Sequence[object] | str,
        runner: ValidationRunner,
    ) -> dict[str, object]:
        cache_key = build_validation_cache_key(
            target_commit=target_commit,
            command=spec,
            environment=environment,
            dependency_state=dependency_state,
        )
        if spec.cacheable and self.cache is not None:
            cached = self.cache.get(cache_key)
            if cached is not None:
                result = dict(cached)
                result.update(
                    {
                        "command": spec.command,
                        "raw_command": spec.raw_command or spec.command,
                        "cache_hit": True,
                        "cache_key": cache_key.digest,
                        "stage": spec.stage.label,
                        "resource_cost": spec.resource_cost,
                        "ordinal": spec.ordinal,
                    }
                )
                return result

        decision, resource_lease = self._acquire_resource(
            spec, workspace_path=workspace_path
        )
        if resource_lease is None:
            now = utc_now()
            return {
                "command": spec.command,
                "raw_command": spec.raw_command or spec.command,
                "started_at": now,
                "finished_at": now,
                "returncode": 75,
                "output": "",
                "error": "resource_admission_rejected",
                "cache_hit": False,
                "cache_key": cache_key.digest,
                "stage": spec.stage.label,
                "resource_cost": spec.resource_cost,
                "ordinal": spec.ordinal,
                "resource_admission": decision.to_dict(),
            }

        timeout = spec.timeout_seconds or self.default_timeout_seconds
        try:
            try:
                raw_result = runner(
                    spec=spec,
                    workspace_path=workspace_path,
                    timeout_seconds=timeout,
                    environment=environment,
                )
                result = dict(raw_result)
            except subprocess.TimeoutExpired:
                result = {"returncode": 124, "timed_out": True, "output": ""}
            except Exception as exc:  # runner is an execution isolation boundary
                result = {
                    "returncode": 1,
                    "output": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        finally:
            self.resource_scheduler.release(resource_lease)
        result.setdefault("command", spec.command)
        result.setdefault("raw_command", spec.raw_command or spec.command)
        result.setdefault("started_at", utc_now())
        result.setdefault("finished_at", utc_now())
        result["returncode"] = int(result.get("returncode", 1))
        result["cache_hit"] = False
        result["cache_key"] = cache_key.digest
        result["stage"] = spec.stage.label
        result["resource_cost"] = spec.resource_cost
        result["ordinal"] = spec.ordinal
        result["resource_admission"] = decision.to_dict()
        result["resource_lease"] = {
            "lease_id": resource_lease.lease_id,
            "resource_class": resource_lease.resource_class,
            "resource_pool": resource_lease.resource_pool,
            "child_limits": resource_lease.child_limits.to_dict(),
            "released": True,
        }
        if spec.cacheable and self.cache is not None and result["returncode"] == 0:
            # Output is useful in the current log but needlessly bloats durable cache.
            cache_result = {key: value for key, value in result.items() if key != "output"}
            self.cache.put(cache_key, cache_result)
        return result

    @staticmethod
    def _first_failure(results: Iterable[Mapping[str, object]]) -> Mapping[str, object] | None:
        return next((result for result in results if int(result.get("returncode", 1)) != 0), None)

    def _run_parallel_stage(
        self,
        specs: Sequence[ValidationCommand],
        execute: Callable[[ValidationCommand], dict[str, object]],
    ) -> list[dict[str, object]]:
        """Run a stage with worker and weighted-budget bounds.

        Submission is incremental.  Once a failure is observed no queued work
        is admitted, while already-running commands are drained safely.
        """

        pending = list(specs)
        active: dict[Future[dict[str, object]], tuple[ValidationCommand, int]] = {}
        completed: list[dict[str, object]] = []
        occupied = 0
        failed = False

        with ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="validation") as pool:
            while pending or active:
                admitted = False
                while pending and not failed and len(active) < self.max_workers:
                    spec = pending[0]
                    cost = min(self.resource_budget, max(1, int(spec.resource_cost)))
                    if active and occupied + cost > self.resource_budget:
                        break
                    pending.pop(0)
                    future = pool.submit(execute, spec)
                    active[future] = (spec, cost)
                    occupied += cost
                    admitted = True
                if not active:
                    break
                done, _ = wait(tuple(active), return_when=FIRST_COMPLETED)
                for future in done:
                    _spec, cost = active.pop(future)
                    occupied -= cost
                    result = future.result()
                    completed.append(result)
                    if int(result.get("returncode", 1)) != 0:
                        failed = True
                if not admitted and not done and active:
                    continue
        return completed

    @staticmethod
    def _validation_node_id(spec: ValidationCommand) -> str:
        return _sha256_bytes(
            _canonical_json(
                {
                    "command": spec.command,
                    "stage": spec.stage.label,
                    "validation_id": spec.validation_id,
                    "impact_paths": spec.impact_paths,
                    "ordinal": spec.ordinal,
                }
            ).encode("utf-8")
        )

    def run_validated(
        self,
        proposal_validation: Any,
        commands: Iterable[str | ValidationCommand] = (),
        *,
        workspace_path: Path | str,
        impact_graph: ImpactDependencyGraph | Mapping[str, Any] | None = None,
        validation_policy_id: str = "",
        objective_id: str = "",
        seeded_defect_id: str = "",
        seeded_defect_path: str = "",
        require_impact_graph: bool = True,
        target_commit: str | None = None,
        environment: Mapping[str, object] | None = None,
        dependency_state: (
            Mapping[str, object] | Sequence[object] | str | None
        ) = None,
        require_full_validation: bool = False,
        scope: str | None = None,
        runner: ValidationRunner | None = None,
    ) -> dict[str, Any]:
        """Run the strict proposal-first, impact-selected validation DAG.

        This is the authority-bearing entry point for implementation output.
        Legacy :meth:`run` remains available for administrative validation that
        has no implementation proposal.  A rejected proposal never calls the
        command runner, and its scheduler-bound receipt records every expensive
        node as undispatched.
        """

        from .proposal_validation import ProposalValidationResult

        proposal_result = (
            proposal_validation
            if isinstance(proposal_validation, ProposalValidationResult)
            else ProposalValidationResult.from_dict(proposal_validation)
        )
        specs = build_validation_commands(commands)
        expensive_specs = tuple(
            spec for spec in specs if spec.stage is not ValidationStage.CHEAP
        )
        expensive_node_ids = tuple(
            self._validation_node_id(spec) for spec in expensive_specs
        )
        if not proposal_result.accepted:
            bound = proposal_result.with_dispatch_outcome(
                expensive_node_ids=expensive_node_ids,
                expensive_checks_started=0,
            )
            blocked_nodes = [
                {
                    "node_id": self._validation_node_id(spec),
                    "command": spec.command,
                    "stage": spec.stage.label,
                    "disposition": ValidationNodeDisposition.BLOCKED.value,
                    "reason": "proposal_gate_failed",
                    "impact_paths": list(spec.impact_paths),
                }
                for spec in specs
            ]
            return {
                "attempted": False,
                "passed": False,
                "returncode": 78,
                "error": "proposal_validation_failed",
                "reason": "proposal_gate_failed",
                "results": [],
                "stages": [
                    {
                        "stage": "proposal",
                        "attempted": True,
                        "passed": False,
                        "planned_count": 1,
                        "executed_count": 1,
                    }
                ],
                "nodes": blocked_nodes,
                "proposal_validation": bound.to_dict(),
                "proposal_receipt": bound.receipt.to_dict(),
                "validation_dag_receipt": None,
                "proved_requirement_ids": bound.receipt.proved_requirement_ids,
                "proof_authoritative": False,
                "completion_authoritative": False,
                "merge_eligible": False,
            }

        # An accepted proposal cannot claim the rejection requirement even
        # though its descendant checks are about to run.
        bound = proposal_result.with_dispatch_outcome(
            expensive_node_ids=expensive_node_ids,
            expensive_checks_started=0,
        )
        graph: ImpactDependencyGraph | None
        if impact_graph is None:
            graph = None
        elif isinstance(impact_graph, ImpactDependencyGraph):
            graph = impact_graph
        else:
            graph = ImpactDependencyGraph.from_dict(impact_graph)
        if (
            graph is not None
            and graph.repository_tree_id
            != bound.proposal.repository_tree_id
        ):
            raise ValidationDAGError(
                "impact graph is stale for the proposal repository tree"
            )
        changed = bound.proposal.changed_paths
        if graph is None and require_impact_graph and expensive_specs:
            policy_id = str(validation_policy_id or "").strip() or _sha256_bytes(
                _canonical_json(
                    {
                        "kind": "strict-validation-dag-policy@2",
                        "proposal_policy_id": bound.policy.policy_id,
                        "commands": [spec.command for spec in specs],
                        "impact_graph_id": "missing-impact-graph",
                    }
                ).encode("utf-8")
            )
            missing_dependency_ids: dict[str, tuple[str, ...]] = {}
            previous_missing_stage: tuple[str, ...] = ()
            for stage in STRICT_VALIDATION_STAGE_ORDER:
                current_missing_stage = tuple(
                    self._validation_node_id(spec)
                    for spec in specs
                    if spec.stage is stage
                )
                for node_id in current_missing_stage:
                    missing_dependency_ids[node_id] = previous_missing_stage
                if current_missing_stage:
                    previous_missing_stage = tuple(
                        sorted(current_missing_stage)
                    )
            records = tuple(
                ValidationDAGNodeRecord(
                    node_id=self._validation_node_id(spec),
                    command=spec.command,
                    stage=spec.stage.label,
                    disposition=ValidationNodeDisposition.BLOCKED,
                    reason="impact_graph_missing",
                    impact_paths=spec.impact_paths,
                    validation_id=spec.validation_id,
                    selected=True,
                    mandatory=False,
                    selection_reason="impact_graph_missing_fail_closed",
                    depends_on=missing_dependency_ids.get(
                        self._validation_node_id(spec), ()
                    ),
                )
                for spec in specs
            )
            selected_node_ids = tuple(node.node_id for node in records)
            receipt = ValidationDAGReceipt(
                repository_tree_id=bound.proposal.repository_tree_id,
                objective_id=bound.proposal.objective_id,
                policy_id=policy_id,
                proposal_receipt_id=bound.receipt.receipt_id,
                graph_id="missing-impact-graph",
                changed_paths=changed,
                affected_paths=changed,
                nodes=records,
                passed=False,
                required_validation_ids=(),
                selected_node_ids=selected_node_ids,
                coverage_complete=False,
                authority_gates=_authority_gate_records(
                    selected_node_ids, passed=False
                ),
                uncovered_impact=True,
            )
            return {
                "attempted": False,
                "passed": False,
                "returncode": 78,
                "error": "impact_graph_missing",
                "reason": "impact_graph_missing",
                "results": [],
                "nodes": [node.to_dict() for node in records],
                "proposal_validation": bound.to_dict(),
                "proposal_receipt": bound.receipt.to_dict(),
                "validation_dag_receipt": receipt.to_dict(),
                "proved_requirement_ids": (),
                "proof_authoritative": False,
                "completion_authoritative": False,
                "merge_eligible": False,
                "impact_graph": None,
                "affected_paths": list(changed),
            }
        affected = graph.affected_paths(changed) if graph is not None else changed
        graph_id = graph.graph_id if graph is not None else "no-impact-graph"
        policy_id = str(validation_policy_id or "").strip() or _sha256_bytes(
            _canonical_json(
                {
                    "kind": "strict-validation-dag-policy@2",
                    "proposal_policy_id": bound.policy.policy_id,
                    "commands": [
                        {
                            "command": spec.command,
                            "stage": spec.stage.label,
                            "impact_paths": spec.impact_paths,
                            "validation_id": spec.validation_id,
                        }
                        for spec in specs
                    ],
                    "impact_graph_id": graph_id,
                    "require_full_validation": bool(require_full_validation),
                    "scope": str(scope or "impact"),
                    "stage_order": [
                        stage.label for stage in STRICT_VALIDATION_STAGE_ORDER
                    ],
                }
            ).encode("utf-8")
        )
        dag_objective = str(objective_id or bound.proposal.objective_id).strip()
        if dag_objective != bound.proposal.objective_id:
            raise ValidationDAGError(
                "validation objective does not match the accepted proposal"
            )

        selection = select_validation_commands(
            specs,
            affected,
            require_full_validation=require_full_validation,
            scope=scope,
        )
        selection_items = tuple(
            item for item in selection.items if item.spec is not None
        )
        effective_specs = tuple(item.spec for item in selection_items)
        decision_by_ordinal = {
            item.spec.ordinal: item for item in selection_items
        }
        required_validation_map = (
            graph.required_validations(affected) if graph is not None else {}
        )
        required_validation_ids = tuple(sorted(required_validation_map))
        coverage_errors: list[str] = []
        if not required_validation_ids:
            coverage_errors.append("no_required_validation_declared")
        for validation_id, target_paths in required_validation_map.items():
            matching = tuple(
                item
                for item in selection_items
                if item.spec.validation_id == validation_id
            )
            if len(matching) != 1:
                coverage_errors.append(
                    f"validation_population:{validation_id}:{len(matching)}"
                )
                continue
            item = matching[0]
            if not item.selected:
                coverage_errors.append(f"validation_omitted:{validation_id}")
            if not set(target_paths).issubset(item.spec.impact_paths):
                coverage_errors.append(
                    f"validation_target_mismatch:{validation_id}"
                )
        coverage_complete = not coverage_errors

        selected_specs = tuple(
            item.spec for item in selection_items if item.selected
        )
        dependency_ids: dict[str, tuple[str, ...]] = {}
        previous_stage_ids: tuple[str, ...] = ()
        for stage in STRICT_VALIDATION_STAGE_ORDER:
            current = tuple(
                spec for spec in selected_specs if spec.stage is stage
            )
            current_ids = tuple(
                self._validation_node_id(spec) for spec in current
            )
            for node_id in current_ids:
                dependency_ids[node_id] = previous_stage_ids
            if current_ids:
                previous_stage_ids = current_ids

        workspace = Path(workspace_path)
        commit = str(target_commit or bound.proposal.repository_tree_id)
        dependencies = (
            collect_dependency_state(workspace, changed_files=affected)
            if dependency_state is None
            else dependency_state
        )
        environment_source = os.environ if environment is None else environment
        execution_environment = {
            str(key): str(value) for key, value in environment_source.items()
        }
        command_runner = runner or self.runner
        results: list[dict[str, object]] = []
        stages: list[dict[str, object]] = []
        failed: Mapping[str, object] | None = None

        def execute(spec: ValidationCommand) -> dict[str, object]:
            return self._execute(
                spec,
                workspace_path=workspace,
                target_commit=commit,
                environment=execution_environment,
                dependency_state=dependencies,
                runner=command_runner,
            )

        if coverage_complete:
            for stage in STRICT_VALIDATION_STAGE_ORDER:
                stage_specs = tuple(
                    spec for spec in selected_specs if spec.stage is stage
                )
                if not stage_specs:
                    continue
                stage_started = utc_now()
                stage_results = self._run_parallel_stage(stage_specs, execute)
                results.extend(stage_results)
                failed = self._first_failure(stage_results)
                stages.append(
                    {
                        "stage": stage.label,
                        "started_at": stage_started,
                        "finished_at": utc_now(),
                        "planned_count": len(stage_specs),
                        "executed_count": len(stage_results),
                        "passed": failed is None,
                    }
                )
                if failed is not None:
                    break
        results.sort(key=lambda result: int(result.get("ordinal", len(specs))))
        report: dict[str, Any] = {
            "attempted": bool(results),
            "passed": coverage_complete and failed is None,
            "returncode": (
                0
                if coverage_complete and failed is None
                else int(failed.get("returncode", 1))
                if failed is not None
                else 78
            ),
            "results": results,
            "stages": stages,
            "selection": selection.to_dict(),
            "target_commit": commit,
            "dependency_state": _json_safe(dependencies),
            "coverage_errors": tuple(sorted(coverage_errors)),
            "cache_hits": sum(
                1 for result in results if result.get("cache_hit") is True
            ),
            "cache_misses": sum(
                1 for result in results if result.get("cache_hit") is not True
            ),
            "max_workers": self.max_workers,
            "resource_budget": self.resource_budget,
        }
        if failed is not None:
            report["failed_command"] = str(failed.get("command") or "")

        results_by_ordinal = {
            int(result.get("ordinal", -1)): result for result in results
        }
        records: list[ValidationDAGNodeRecord] = []
        for spec in effective_specs:
            decision = decision_by_ordinal[spec.ordinal]
            result = results_by_ordinal.get(spec.ordinal)
            if result is not None:
                returncode = int(result.get("returncode", 1))
                result_digest = _sha256_bytes(
                    _canonical_json(_json_safe(dict(result))).encode("utf-8")
                )
                disposition = (
                    ValidationNodeDisposition.SUCCEEDED
                    if returncode == 0
                    else ValidationNodeDisposition.FAILED
                )
                reason = (
                    "validation_passed"
                    if returncode == 0
                    else "validation_failed"
                )
            elif decision.selected:
                returncode = None
                result_digest = ""
                disposition = ValidationNodeDisposition.BLOCKED
                reason = (
                    "impact_coverage_incomplete"
                    if not coverage_complete
                    else "blocked_by_failed_dependency"
                )
            else:
                returncode = None
                result_digest = ""
                disposition = ValidationNodeDisposition.OMITTED
                reason = str(
                    decision.reason
                    or "not_selected_by_impact_analysis"
                )
            node_id = self._validation_node_id(spec)
            records.append(
                ValidationDAGNodeRecord(
                    node_id=node_id,
                    command=spec.command,
                    stage=spec.stage.label,
                    disposition=disposition,
                    reason=reason,
                    impact_paths=spec.impact_paths,
                    returncode=returncode,
                    result_digest=result_digest,
                    validation_id=spec.validation_id,
                    selected=decision.selected,
                    mandatory=(
                        decision.selected
                        and spec.validation_id in required_validation_ids
                    ),
                    selection_reason=decision.reason,
                    depends_on=dependency_ids.get(node_id, ()),
                    observed_seeded_defect_id=(
                        str(result.get("seeded_defect_id") or "")
                        if result is not None
                        else ""
                    ),
                )
            )

        selected_node_ids = tuple(
            sorted(node.node_id for node in records if node.selected)
        )
        uncovered_impact = not coverage_complete
        dag_passed = bool(report.get("passed", False)) and not uncovered_impact
        base_receipt = ValidationDAGReceipt(
            repository_tree_id=bound.proposal.repository_tree_id,
            objective_id=dag_objective,
            policy_id=policy_id,
            proposal_receipt_id=bound.receipt.receipt_id,
            graph_id=graph_id,
            impact_graph=graph,
            changed_paths=changed,
            affected_paths=affected,
            nodes=tuple(records),
            passed=dag_passed,
            required_validation_ids=required_validation_ids,
            selected_node_ids=selected_node_ids,
            coverage_complete=coverage_complete,
            authority_gates=_authority_gate_records(
                selected_node_ids, passed=dag_passed
            ),
            seeded_defect_id=str(seeded_defect_id or ""),
            seeded_defect_path=str(seeded_defect_path or ""),
            uncovered_impact=uncovered_impact,
        )
        evidence: TransitiveImpactValidationEvidence | None = None
        normalized_seed = _normalize_impact_path(seeded_defect_path)
        if graph is not None and seeded_defect_id and normalized_seed in changed:
            for spec, node in zip(effective_specs, records):
                if node.disposition is not ValidationNodeDisposition.FAILED:
                    continue
                if node.observed_seeded_defect_id != str(seeded_defect_id):
                    continue
                for target in spec.impact_paths:
                    path = graph.impact_path(normalized_seed, target)
                    if len(path) >= 3:
                        evidence = TransitiveImpactValidationEvidence(
                            requirement_id=TRANSITIVE_IMPACT_REQUIREMENT_ID,
                            repository_tree_id=base_receipt.repository_tree_id,
                            objective_id=base_receipt.objective_id,
                            policy_id=base_receipt.policy_id,
                            graph_id=base_receipt.graph_id,
                            seeded_defect_id=str(seeded_defect_id),
                            seeded_defect_path=normalized_seed,
                            impact_path=path,
                            failing_node_id=node.node_id,
                            failing_result_digest=node.result_digest,
                            receipt_id=base_receipt.receipt_id,
                        )
                        break
                if evidence is not None:
                    break
        receipt = (
            ValidationDAGReceipt(
                repository_tree_id=base_receipt.repository_tree_id,
                objective_id=base_receipt.objective_id,
                policy_id=base_receipt.policy_id,
                proposal_receipt_id=base_receipt.proposal_receipt_id,
                graph_id=base_receipt.graph_id,
                impact_graph=base_receipt.impact_graph,
                changed_paths=base_receipt.changed_paths,
                affected_paths=base_receipt.affected_paths,
                nodes=base_receipt.nodes,
                passed=base_receipt.passed,
                required_validation_ids=base_receipt.required_validation_ids,
                selected_node_ids=base_receipt.selected_node_ids,
                coverage_complete=base_receipt.coverage_complete,
                authority_gates=base_receipt.authority_gates,
                seeded_defect_id=base_receipt.seeded_defect_id,
                seeded_defect_path=base_receipt.seeded_defect_path,
                uncovered_impact=base_receipt.uncovered_impact,
                transitive_evidence=evidence,
                receipt_id=base_receipt.receipt_id,
            )
            if evidence is not None
            else base_receipt
        )
        report["proposal_validation"] = bound.to_dict()
        report["proposal_receipt"] = bound.receipt.to_dict()
        report["validation_dag_receipt"] = receipt.to_dict()
        report["nodes"] = [node.to_dict() for node in receipt.nodes]
        report["proved_requirement_ids"] = receipt.proved_requirement_ids
        report["proof_authoritative"] = False
        report["completion_authoritative"] = False
        report["merge_eligible"] = False
        report["freshness_authoritative"] = False
        report["impact_graph"] = graph.to_dict() if graph is not None else None
        report["affected_paths"] = list(affected)
        report["authority_gates"] = [
            gate.to_dict() for gate in receipt.authority_gates
        ]
        if receipt.uncovered_impact:
            report["passed"] = False
            report["returncode"] = 78
            report["error"] = "uncovered_validation_impact"
            report["reason"] = "impact_validation_population_incomplete"
        return report

    # Compatibility names used by orchestration callers.
    run_validation_dag = run_validated
    run_strict = run_validated

    def run(
        self,
        commands: Iterable[str | ValidationCommand] = (),
        *,
        workspace_path: Path | str,
        target_commit: str | None = None,
        changed_files: Iterable[str] | None = None,
        environment: Mapping[str, object] | None = None,
        dependency_state: Mapping[str, object] | Sequence[object] | str | None = None,
        require_full_validation: bool = False,
        scope: str | None = None,
        runner: ValidationRunner | None = None,
        proof_scheduler: Any = None,
        proof_plan: Any = None,
        proof_executor: Any = None,
        proof_executors: Mapping[Any, Callable[..., Any]] | None = None,
        proof_scheduler_options: Mapping[str, Any] | None = None,
        proof_timeout_seconds: float | None = None,
        fallback_plans: Iterable[Any] | Any = (),
        fallback_plan: Any | None = None,
    ) -> dict[str, Any]:
        """Schedule commands and return a legacy-compatible JSON report.

        Supplying proof or fallback inputs selects the additive staged
        pipeline.  Command-only callers retain the original report contract.
        """

        if (
            proof_scheduler is not None
            or proof_plan is not None
            or proof_executor is not None
            or proof_executors
            or fallback_plan is not None
            or bool(fallback_plans)
        ):
            return self.run_staged(
                commands,
                workspace_path=workspace_path,
                proof_scheduler=proof_scheduler,
                proof_plan=proof_plan,
                proof_executor=proof_executor,
                proof_executors=proof_executors,
                proof_scheduler_options=proof_scheduler_options,
                proof_timeout_seconds=proof_timeout_seconds,
                fallback_plans=fallback_plans,
                fallback_plan=fallback_plan,
                target_commit=target_commit,
                changed_files=changed_files,
                environment=environment,
                dependency_state=dependency_state,
                require_full_validation=require_full_validation,
                scope=scope,
                runner=runner,
            )

        specs = build_validation_commands(commands)
        if not specs:
            return {
                "attempted": False,
                "passed": True,
                "returncode": 0,
                "results": [],
                "reason": "no_commands",
            }

        workspace = Path(workspace_path)
        changed = discover_changed_files(workspace) if changed_files is None else tuple(changed_files)
        selection: ValidationSelection = select_validation_commands(
            specs,
            changed,
            require_full_validation=require_full_validation,
            scope=scope,
        )
        commit = str(target_commit or resolve_target_commit(workspace))
        dependencies = (
            collect_dependency_state(workspace, changed_files=changed)
            if dependency_state is None
            else dependency_state
        )
        environment_source = os.environ if environment is None else environment
        execution_environment = {
            str(key): str(value) for key, value in environment_source.items()
        }
        selected = tuple(selection.selected)
        command_runner = runner or self.runner
        results: list[dict[str, object]] = []
        stages: list[dict[str, object]] = []
        failed: Mapping[str, object] | None = None

        def execute(spec: ValidationCommand) -> dict[str, object]:
            return self._execute(
                spec,
                workspace_path=workspace,
                target_commit=commit,
                environment=execution_environment,
                dependency_state=dependencies,
                runner=command_runner,
            )

        for stage in ValidationStage:
            stage_specs = tuple(spec for spec in selected if spec.stage == stage)
            if not stage_specs:
                continue
            stage_started = utc_now()
            stage_results = self._run_parallel_stage(stage_specs, execute)
            results.extend(stage_results)
            failed = self._first_failure(stage_results)
            stages.append(
                {
                    "stage": stage.label,
                    "started_at": stage_started,
                    "finished_at": utc_now(),
                    "planned_count": len(stage_specs),
                    "executed_count": len(stage_results),
                    "passed": failed is None,
                }
            )
            if failed is not None:
                break

        # Parallel completion order is nondeterministic; reports are not.
        results.sort(key=lambda result: int(result.get("ordinal", len(specs))))

        cache_hits = sum(1 for result in results if result.get("cache_hit") is True)
        report: dict[str, Any] = {
            "attempted": bool(results),
            "passed": failed is None,
            "returncode": 0 if failed is None else int(failed.get("returncode", 1)),
            "results": results,
            "stages": stages,
            "selection": selection.to_dict(),
            "target_commit": commit,
            "dependency_state": _json_safe(dependencies),
            "cache_hits": cache_hits,
            "cache_misses": len(results) - cache_hits,
            "max_workers": self.max_workers,
            "resource_budget": self.resource_budget,
            "resource_lease_budget": self.resource_lease_budget.to_dict(),
            "resource_admission": [
                result["resource_admission"]
                for result in results
                if isinstance(result.get("resource_admission"), Mapping)
            ],
        }
        if failed is not None:
            report["failed_command"] = str(failed.get("command") or "")
            if failed.get("timed_out"):
                report["error"] = "timeout"
        return report

    @staticmethod
    def _fallback_values(
        fallback_plans: Iterable[Any] | Any,
        fallback_plan: Any | None,
    ) -> tuple[Any, ...]:
        values: list[Any] = []
        if fallback_plan is not None:
            values.append(fallback_plan)
        if fallback_plans is None:
            return tuple(values)
        if isinstance(fallback_plans, Mapping) or hasattr(
            fallback_plans, "validations"
        ):
            values.append(fallback_plans)
        else:
            values.extend(fallback_plans)
        return tuple(values)

    @staticmethod
    def _fallback_validations(plan: Any) -> tuple[DeclaredValidation, ...]:
        raw = getattr(plan, "validations", None)
        if raw is None and isinstance(plan, Mapping):
            raw = plan.get("validations", ())
        result: list[DeclaredValidation] = []
        for item in raw or ():
            if isinstance(item, DeclaredValidation):
                result.append(item)
            elif isinstance(item, Mapping):
                result.append(DeclaredValidation.from_dict(item))
        return tuple(result)

    @staticmethod
    def _fallback_field(plan: Any, name: str, default: Any = None) -> Any:
        if isinstance(plan, Mapping):
            return plan.get(name, default)
        return getattr(plan, name, default)

    def _build_proof_scheduler(
        self,
        *,
        proof_scheduler: Any,
        proof_plan: Any,
        proof_executor: Any,
        proof_executors: Mapping[Any, Callable[..., Any]] | None,
        proof_scheduler_options: Mapping[str, Any] | None,
    ) -> Any:
        if proof_scheduler is not None:
            if proof_plan is not None:
                raise ValueError(
                    "proof_scheduler cannot be combined with proof_plan"
                )
            return proof_scheduler
        if proof_plan is None:
            if proof_executor is not None or proof_executors:
                raise ValueError(
                    "proof_executor requires proof_plan or proof_scheduler"
                )
            return None
        if proof_executor is None and not proof_executors:
            raise ValueError("proof_plan requires a proof executor")

        # Kept local so proof_scheduler may expose validation adapters without
        # creating an import cycle at module-import time.
        from .proof_scheduler import ProofScheduler

        options = dict(proof_scheduler_options or {})
        supplied_resource_scheduler = options.get("resource_scheduler")
        if (
            supplied_resource_scheduler is not None
            and supplied_resource_scheduler is not self.resource_scheduler
        ):
            raise ValueError(
                "proof scheduler must use the validation scheduler's shared "
                "resource_scheduler"
            )
        supplied_budget = options.get("resource_lease_budget")
        if (
            supplied_budget is not None
            and supplied_budget is not self.resource_lease_budget
        ):
            raise ValueError(
                "proof scheduler must use the validation scheduler's shared "
                "resource_lease_budget"
            )
        options.setdefault("resource_scheduler", self.resource_scheduler)
        options.setdefault("resource_lease_budget", self.resource_lease_budget)
        options.setdefault("host_resource_source", self._host_resource_source)
        options.setdefault(
            "provider_capacity_source", self._provider_capacity_source
        )
        options.setdefault("staged_execution", True)
        if proof_executors:
            options["executors"] = proof_executors
        return ProofScheduler(proof_plan, proof_executor, **options)

    @staticmethod
    def _run_proof_scheduler(
        proof_scheduler: Any,
        timeout_seconds: float | None,
        stages: Sequence[str] | None = None,
    ) -> Any:
        callback = getattr(proof_scheduler, "run", None)
        if not callable(callback):
            if callable(proof_scheduler):
                callback = proof_scheduler
            else:
                raise TypeError("proof_scheduler must expose run()")
        try:
            signature = inspect.signature(callback)
        except (TypeError, ValueError):
            kwargs = {}
            if timeout_seconds is not None:
                kwargs["timeout_seconds"] = timeout_seconds
            if stages is not None:
                kwargs["stages"] = stages
            return callback(**kwargs)
        kwargs = {}
        if timeout_seconds is not None and "timeout_seconds" in signature.parameters:
            kwargs["timeout_seconds"] = timeout_seconds
        if stages is not None and "stages" in signature.parameters:
            kwargs["stages"] = stages
        return callback(**kwargs)

    def run_staged(
        self,
        commands: Iterable[str | ValidationCommand] = (),
        *,
        workspace_path: Path | str,
        proof_scheduler: Any = None,
        proof_plan: Any = None,
        proof_executor: Any = None,
        proof_executors: Mapping[Any, Callable[..., Any]] | None = None,
        proof_scheduler_options: Mapping[str, Any] | None = None,
        proof_timeout_seconds: float | None = None,
        fallback_plans: Iterable[Any] | Any = (),
        fallback_plan: Any | None = None,
        target_commit: str | None = None,
        changed_files: Iterable[str] | None = None,
        environment: Mapping[str, object] | None = None,
        dependency_state: (
            Mapping[str, object] | Sequence[object] | str | None
        ) = None,
        require_full_validation: bool = False,
        scope: str | None = None,
        runner: ValidationRunner | None = None,
    ) -> dict[str, Any]:
        """Run one fail-fast proof-and-validation pipeline.

        The method is additive to :meth:`run`: existing callers retain the
        CHEAP/TARGETED/BROAD command contract.  Staged callers receive explicit
        barriers:

        ``deterministic -> translation -> solver -> kernel -> focused tests
        -> broad tests -> attestation -> persistence``.

        A supplied proof plan is executed with the exact resource scheduler and
        lease budget used by shell validation.  A pre-built proof scheduler is
        accepted for durable resume; the report states whether it shares those
        same objects.
        """

        workspace = Path(workspace_path)
        changed = (
            discover_changed_files(workspace)
            if changed_files is None
            else tuple(changed_files)
        )
        commit = str(target_commit or resolve_target_commit(workspace))
        dependencies = (
            collect_dependency_state(workspace, changed_files=changed)
            if dependency_state is None
            else dependency_state
        )
        command_values = list(commands)
        plans = self._fallback_values(fallback_plans, fallback_plan)
        declarations_by_plan: list[tuple[Any, tuple[DeclaredValidation, ...]]] = []
        fallback_declarations: list[DeclaredValidation] = []
        for plan in plans:
            declarations = self._fallback_validations(plan)
            declarations_by_plan.append((plan, declarations))
            fallback_declarations.extend(declarations)

        specs = build_validation_commands(command_values)
        selection = select_validation_commands(
            specs,
            changed,
            require_full_validation=require_full_validation,
            scope=scope,
            fallback_validations=fallback_declarations,
        )
        selected = tuple(selection.selected)
        cheap_specs = tuple(
            spec for spec in selected if spec.stage is ValidationStage.CHEAP
        )
        focused_specs = tuple(
            spec for spec in selected if spec.stage is ValidationStage.TARGETED
        )
        broad_specs = tuple(
            spec for spec in selected if spec.stage is ValidationStage.BROAD
        )
        translation_specs = tuple(
            spec
            for spec in selected
            if spec.stage is ValidationStage.TRANSLATION
        )
        solver_specs = tuple(
            spec for spec in selected if spec.stage is ValidationStage.SOLVER
        )
        kernel_specs = tuple(
            spec for spec in selected if spec.stage is ValidationStage.KERNEL
        )
        attestation_specs = tuple(
            spec
            for spec in selected
            if spec.stage is ValidationStage.ATTESTATION
        )

        common = {
            "workspace_path": workspace,
            "target_commit": commit,
            # Empty means "all already-selected specs"; it avoids performing a
            # second, potentially different impact decision inside run().
            "changed_files": (),
            "environment": environment,
            "dependency_state": dependencies,
            "runner": runner,
        }
        deterministic_report = (
            self.run(cheap_specs, **common)
            if cheap_specs
            else {
                "attempted": False,
                "passed": True,
                "returncode": 0,
                "results": [],
                "reason": "no_deterministic_commands",
            }
        )
        deterministic_passed = bool(deterministic_report.get("passed", False))

        active_proof_scheduler = self._build_proof_scheduler(
            proof_scheduler=proof_scheduler,
            proof_plan=proof_plan,
            proof_executor=proof_executor,
            proof_executors=proof_executors,
            proof_scheduler_options=proof_scheduler_options,
        )
        proof_result: Any = None
        proof_error = ""
        proof_attempted = False
        proof_phase_reports: list[dict[str, Any]] = []
        proof_plan_value = getattr(active_proof_scheduler, "plan", None)
        proof_plan_stages = {
            _enum_text(getattr(step, "stage", ""))
            for step in getattr(proof_plan_value, "steps", ())
        }
        proof_run = (
            getattr(active_proof_scheduler, "run", None)
            if active_proof_scheduler is not None
            else None
        )
        try:
            proof_supports_partial = (
                callable(proof_run)
                and "stages" in inspect.signature(proof_run).parameters
            )
        except (TypeError, ValueError):
            proof_supports_partial = False
        proof_called_without_partial = False

        def empty_phase(reason: str) -> dict[str, Any]:
            return {
                "attempted": False,
                "passed": True,
                "returncode": 0,
                "results": [],
                "reason": reason,
            }

        proof_phase_ok = deterministic_passed

        def run_proof_phase(
            name: str,
            stages: Sequence[str],
            command_specs: Sequence[ValidationCommand] = (),
        ) -> bool:
            nonlocal proof_result
            nonlocal proof_error
            nonlocal proof_attempted
            nonlocal proof_called_without_partial

            gate_open = proof_phase_ok
            command_report = (
                self.run(command_specs, **common)
                if command_specs and gate_open
                else empty_phase(
                    "prior_stage_failed"
                    if not gate_open
                    else "no_phase_commands"
                )
            )
            command_ok = bool(command_report.get("passed", False))
            scheduler_attempted = False
            scheduler_ok = True
            error = ""
            relevant = (
                active_proof_scheduler is not None
                and (
                    not proof_plan_stages
                    or bool(proof_plan_stages.intersection(stages))
                )
            )
            if relevant and gate_open and command_ok:
                if not proof_supports_partial and proof_called_without_partial:
                    scheduler_ok = True
                else:
                    scheduler_attempted = True
                    proof_attempted = True
                    try:
                        proof_result = self._run_proof_scheduler(
                            active_proof_scheduler,
                            proof_timeout_seconds,
                            stages if proof_supports_partial else None,
                        )
                        scheduler_ok = (
                            _proof_phase_passed(
                                proof_result,
                                active_proof_scheduler,
                                stages,
                            )
                            if proof_supports_partial
                            else bool(
                                getattr(
                                    proof_result,
                                    "succeeded",
                                    _object_mapping(proof_result).get(
                                        "succeeded",
                                        _object_mapping(proof_result).get(
                                            "passed", False
                                        ),
                                    ),
                                )
                            )
                        )
                        proof_called_without_partial = not proof_supports_partial
                    except Exception as exc:
                        scheduler_ok = False
                        error = f"{type(exc).__name__}: {exc}"
                        proof_error = error
            phase_ok = gate_open and command_ok and scheduler_ok
            proof_phase_reports.append(
                {
                    "stage": name,
                    "attempted": bool(command_report.get("attempted"))
                    or scheduler_attempted,
                    "passed": phase_ok,
                    "command_report": command_report,
                    "proof_attempted": scheduler_attempted,
                    "proof_stages": list(stages),
                    "error": error,
                }
            )
            return phase_ok

        core_phases = (
            ("translation", ("translate",), translation_specs),
            ("solver", ("model_draft", "solve"), solver_specs),
            (
                "kernel",
                ("reconstruct", "kernel_verify"),
                kernel_specs,
            ),
            ("proof_validation", ("validate",), ()),
        )
        for phase_name, proof_stages, phase_commands in core_phases:
            phase_result = run_proof_phase(
                phase_name, proof_stages, phase_commands
            )
            proof_phase_ok = proof_phase_ok and phase_result
        proof_core_passed = proof_phase_ok
        proof_passed = (
            proof_core_passed
            if active_proof_scheduler is not None
            else proof_core_passed
        )

        fallback_can_continue = bool(plans) and all(
            bool(self._fallback_field(plan, "can_continue", False))
            and not bool(self._fallback_field(plan, "blocking", True))
            for plan in plans
        )
        # Fallback checks are still useful evidence for an enforcement-mode
        # block.  Without a declared fallback, proof failure remains fail-fast.
        may_run_focused = (
            deterministic_passed
            and (
                active_proof_scheduler is None
                or proof_core_passed
                or bool(plans)
            )
        )
        focused_report = (
            self.run(focused_specs, **common)
            if focused_specs and may_run_focused
            else {
                "attempted": False,
                "passed": not focused_specs,
                "returncode": 0,
                "results": [],
                "reason": (
                    "no_focused_commands"
                    if not focused_specs
                    else "proof_gate_failed"
                ),
            }
        )
        focused_passed = bool(focused_report.get("passed", False))
        proof_gate_passed = (
            fallback_can_continue
            if plans
            else active_proof_scheduler is None or proof_core_passed
        )
        may_run_broad = (
            deterministic_passed and focused_passed and proof_gate_passed
        )
        broad_report = (
            self.run(broad_specs, **common)
            if broad_specs and may_run_broad
            else {
                "attempted": False,
                "passed": not broad_specs,
                "returncode": 0,
                "results": [],
                "reason": (
                    "no_broad_commands"
                    if not broad_specs
                    else "prior_stage_failed"
                ),
            }
        )
        broad_passed = bool(broad_report.get("passed", False))

        post_proof_ok = True
        if (
            deterministic_passed
            and focused_passed
            and broad_passed
            and (
                active_proof_scheduler is None
                or proof_core_passed
            )
        ):
            proof_phase_ok = proof_core_passed
            post_proof_ok = run_proof_phase(
                "attestation", ("attest",), attestation_specs
            )
            proof_phase_ok = proof_phase_ok and post_proof_ok
            persist_ok = run_proof_phase("persist", ("persist",), ())
            post_proof_ok = post_proof_ok and persist_ok
        else:
            proof_phase_reports.extend(
                (
                    {
                        "stage": "attestation",
                        "attempted": False,
                        "passed": False,
                        "reason": "prior_stage_failed",
                    },
                    {
                        "stage": "persist",
                        "attempted": False,
                        "passed": False,
                        "reason": "prior_stage_failed",
                    },
                )
            )
        proof_passed = (
            (proof_core_passed and post_proof_ok)
            if active_proof_scheduler is not None
            else post_proof_ok
        )
        # A reviewed shadow/disabled fallback is allowed to continue after an
        # inconclusive proof without pretending that skipped attestation
        # passed.  Enforcement/canary fallbacks remain blocking.
        proof_gate_passed = (
            proof_gate_passed and post_proof_ok
        ) or fallback_can_continue
        proof_mapping = _object_mapping(proof_result)

        proof_grouped = _proof_records_by_verdict(
            proof_result, active_proof_scheduler
        )
        phase_command_reports = {
            str(item.get("stage") or ""): item.get("command_report")
            for item in proof_phase_reports
            if isinstance(item.get("command_report"), Mapping)
        }
        deterministic_records = _command_verdict_records(
            deterministic_report, phase="deterministic"
        ) + proof_grouped["deterministic"]
        translation_records = _command_verdict_records(
            phase_command_reports.get("translation"), phase="translation"
        ) + proof_grouped["translation"]
        solver_records = _command_verdict_records(
            phase_command_reports.get("solver"), phase="solver"
        ) + proof_grouped["solver"]
        kernel_records = _command_verdict_records(
            phase_command_reports.get("kernel"), phase="kernel"
        ) + proof_grouped["kernel"]
        attestation_records = _command_verdict_records(
            phase_command_reports.get("attestation"), phase="attestation"
        ) + proof_grouped["attestation"]
        focused_records = _command_verdict_records(
            focused_report, phase="focused"
        )
        broad_records = _command_verdict_records(broad_report, phase="broad")
        test_records = proof_grouped["test"] + focused_records + broad_records
        verdict_records = {
            "deterministic": deterministic_records,
            "translation": translation_records,
            "solver": solver_records,
            "kernel": kernel_records,
            "test": test_records,
            "attestation": attestation_records,
        }
        verdicts: dict[str, Any] = {}
        for kind in VALIDATION_VERDICT_KINDS:
            records = verdict_records[kind]
            verdicts[kind] = _verdict_summary(
                records,
                passed=all(bool(item.get("passed", True)) for item in records),
                omitted_reason=(
                    "proof_not_requested"
                    if kind
                    in {
                        "translation",
                        "solver",
                        "kernel",
                        "attestation",
                    }
                    and active_proof_scheduler is None
                    else ""
                ),
            )

        fallback_selection: list[dict[str, Any]] = []
        for plan, declarations in declarations_by_plan:
            validation_items: list[dict[str, Any]] = []
            for declaration in declarations:
                decision = next(
                    (
                        item
                        for item in selection.items
                        if (
                            item.declaration is not None
                            and item.declaration.validation_id
                            == declaration.validation_id
                        )
                        or (
                            item.spec is not None
                            and item.spec.validation_id
                            == declaration.validation_id
                        )
                    ),
                    None,
                )
                if decision is not None:
                    selected_flag = decision.selected
                    reason = f"fallback:{decision.reason}"
                    stage = (
                        decision.spec.stage.label
                        if decision.spec is not None
                        else ""
                    )
                    matched_paths = list(decision.matched_paths)
                elif declaration.kind is ValidationRequirementKind.MANUAL_REVIEW:
                    selected_flag = False
                    reason = "fallback:manual_review_required"
                    stage = ""
                    matched_paths = []
                else:
                    selected_flag = False
                    reason = "fallback:command_unresolved"
                    stage = ""
                    matched_paths = []
                item = declaration.to_dict()
                item.update(
                    {
                        "selected": selected_flag,
                        "selection_reason": reason,
                        "stage": stage,
                        "matched_paths": matched_paths,
                    }
                )
                validation_items.append(item)
            fallback_selection.append(
                {
                    "plan_id": str(
                        self._fallback_field(
                            plan,
                            "plan_id",
                            self._fallback_field(plan, "content_id", ""),
                        )
                        or ""
                    ),
                    "obligation_id": str(
                        self._fallback_field(plan, "obligation_id", "") or ""
                    ),
                    "can_continue": bool(
                        self._fallback_field(plan, "can_continue", False)
                    ),
                    "blocking": bool(
                        self._fallback_field(plan, "blocking", True)
                    ),
                    "validations": validation_items,
                }
            )

        executed_commands = {
            str(item.get("command") or "")
            for report in (
                deterministic_report,
                *tuple(
                    value
                    for value in phase_command_reports.values()
                    if isinstance(value, Mapping)
                ),
                focused_report,
                broad_report,
            )
            for item in report.get("results", ()) or ()
            if isinstance(item, Mapping)
        }
        execution_decisions = []
        for item in selection.items:
            spec = item.spec
            if spec is None:
                execution_reason = item.reason
            elif not item.selected:
                execution_reason = item.reason
            elif spec.command in executed_commands:
                execution_reason = "executed"
            elif not deterministic_passed:
                execution_reason = "deterministic_gate_failed"
            elif spec.stage is ValidationStage.BROAD and not may_run_broad:
                execution_reason = "prior_stage_failed"
            elif spec.stage is ValidationStage.TARGETED and not may_run_focused:
                execution_reason = "proof_gate_failed"
            else:
                execution_reason = "fail_fast_after_peer_failure"
            execution_decisions.append(
                {
                    **item.to_dict(),
                    "executed": (
                        spec is not None and spec.command in executed_commands
                    ),
                    "execution_reason": execution_reason,
                }
            )

        proof_checks: list[dict[str, Any]] = []
        plan = getattr(active_proof_scheduler, "plan", None)
        for step in getattr(plan, "steps", ()) if plan is not None else ():
            proof_checks.append(
                {
                    "step_id": str(getattr(step, "step_id", "") or ""),
                    "stage": _enum_text(getattr(step, "stage", "")),
                    "selected": deterministic_passed,
                    "reason": (
                        "opt_in_proof_plan"
                        if deterministic_passed
                        else "deterministic_gate_failed"
                    ),
                }
            )

        command_results = [
            *list(deterministic_report.get("results", ()) or ()),
            *[
                result
                for phase in proof_phase_reports
                for result in (
                    phase.get("command_report", {}).get("results", ())
                    if isinstance(phase.get("command_report"), Mapping)
                    else ()
                )
            ],
            *list(focused_report.get("results", ()) or ()),
            *list(broad_report.get("results", ()) or ()),
        ]
        first_command_failure = self._first_failure(command_results)
        passed = (
            deterministic_passed
            and focused_passed
            and broad_passed
            and proof_gate_passed
        )
        if first_command_failure is not None:
            returncode = int(first_command_failure.get("returncode", 1))
        elif not passed:
            returncode = 1
        else:
            returncode = 0

        selection_report = selection.to_dict()
        selection_report.update(
            {
                "decisions": execution_decisions,
                "fallback_checks": fallback_selection,
                "proof_checks": proof_checks,
            }
        )
        shared_scheduler = (
            active_proof_scheduler is None
            or getattr(active_proof_scheduler, "resource_scheduler", None)
            is self.resource_scheduler
        )
        shared_budget = (
            active_proof_scheduler is None
            or getattr(active_proof_scheduler, "resource_lease_budget", None)
            is self.resource_lease_budget
        )
        report: dict[str, Any] = {
            "schema": STAGED_REPORT_SCHEMA,
            "attempted": bool(command_results) or proof_attempted,
            "passed": passed,
            "returncode": returncode,
            "results": command_results,
            "stages": [
                {
                    "stage": "deterministic",
                    "attempted": bool(deterministic_report.get("attempted")),
                    "passed": deterministic_passed,
                    "reason": deterministic_report.get("reason", ""),
                },
                *proof_phase_reports[:4],
                {
                    "stage": "focused",
                    "attempted": bool(focused_report.get("attempted")),
                    "passed": focused_passed,
                    "reason": focused_report.get("reason", ""),
                },
                {
                    "stage": "broad",
                    "attempted": bool(broad_report.get("attempted")),
                    "passed": broad_passed,
                    "reason": broad_report.get("reason", ""),
                },
                *proof_phase_reports[4:],
            ],
            "selection": selection_report,
            "verdicts": verdicts,
            "proof": (
                _json_safe(proof_mapping)
                if proof_mapping
                else {
                    "attempted": proof_attempted,
                    "succeeded": proof_passed if proof_attempted else None,
                    "error": proof_error,
                }
            ),
            "fallbacks": fallback_selection,
            "target_commit": commit,
            "dependency_state": _json_safe(dependencies),
            "cache_hits": sum(
                item.get("cache_hit") is True for item in command_results
            ),
            "cache_misses": sum(
                item.get("cache_hit") is not True for item in command_results
            ),
            "max_workers": self.max_workers,
            "resource_budget": self.resource_budget,
            "resource_lease_budget": self.resource_lease_budget.to_dict(),
            "shared_resource_scheduler": shared_scheduler,
            "shared_resource_lease_budget": shared_budget,
            "resource_admission": [
                item["resource_admission"]
                for item in command_results
                if isinstance(item.get("resource_admission"), Mapping)
            ],
        }
        if first_command_failure is not None:
            report["failed_command"] = str(
                first_command_failure.get("command") or ""
            )
        elif proof_error:
            report["error"] = proof_error
        elif not proof_gate_passed:
            report["error"] = "proof_gate_failed"
        return report

    # Natural aliases used by different supervisor embeddings.
    schedule = run
    validate = run
    schedule_staged = run_staged
    validate_staged = run_staged


def schedule_validations(
    commands: Iterable[str | ValidationCommand],
    *,
    workspace_path: Path | str,
    **kwargs: object,
) -> dict[str, Any]:
    """Convenience wrapper for one uncached scheduler invocation."""

    scheduler_keys = {
        "cache",
        "cache_dir",
        "max_workers",
        "resource_budget",
        "resource_scheduler",
        "resource_lease_budget",
        "resource_policy",
        "host_resource_source",
        "provider_capacity_source",
        "resource_admission_timeout_seconds",
        "default_timeout_seconds",
        "runner",
    }
    scheduler_kwargs = {key: kwargs.pop(key) for key in tuple(kwargs) if key in scheduler_keys}
    return ValidationScheduler(**scheduler_kwargs).run(
        commands,
        workspace_path=workspace_path,
        **kwargs,
    )


def schedule_staged_validations(
    commands: Iterable[str | ValidationCommand] = (),
    *,
    workspace_path: Path | str,
    **kwargs: object,
) -> dict[str, Any]:
    """Convenience wrapper for proof-aware staged validation."""

    scheduler_keys = {
        "cache",
        "cache_dir",
        "max_workers",
        "resource_budget",
        "resource_scheduler",
        "resource_lease_budget",
        "resource_policy",
        "host_resource_source",
        "provider_capacity_source",
        "resource_admission_timeout_seconds",
        "default_timeout_seconds",
        "runner",
    }
    scheduler_kwargs = {
        key: kwargs.pop(key) for key in tuple(kwargs) if key in scheduler_keys
    }
    return ValidationScheduler(**scheduler_kwargs).run_staged(
        commands,
        workspace_path=workspace_path,
        **kwargs,
    )


def schedule_validated_proposal(
    proposal_validation: Any,
    commands: Iterable[str | ValidationCommand] = (),
    *,
    workspace_path: Path | str,
    **kwargs: object,
) -> dict[str, Any]:
    """Convenience wrapper for the strict proposal-first validation DAG."""

    scheduler_keys = {
        "cache",
        "cache_dir",
        "max_workers",
        "resource_budget",
        "resource_scheduler",
        "resource_lease_budget",
        "resource_policy",
        "host_resource_source",
        "provider_capacity_source",
        "resource_admission_timeout_seconds",
        "default_timeout_seconds",
        "runner",
    }
    scheduler_kwargs = {
        key: kwargs.pop(key) for key in tuple(kwargs) if key in scheduler_keys
    }
    return ValidationScheduler(**scheduler_kwargs).run_validated(
        proposal_validation,
        commands,
        workspace_path=workspace_path,
        **kwargs,
    )


# Compatibility spelling used by integrations that foreground proof work.
schedule_proof_validations = schedule_staged_validations
schedule_validation_dag = schedule_validated_proposal
