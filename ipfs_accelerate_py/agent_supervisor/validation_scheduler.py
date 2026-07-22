"""Impact-selected, cached, and resource-bounded validation execution.

The scheduler is intentionally independent from the todo daemon.  It accepts
plain shell commands or classified command specifications, produces a JSON-safe
legacy-compatible report, and stores only successful results.  Persistent cache
keys bind a result to the target commit, candidate worktree content, command,
relevant environment, and dependency state.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .validation_commands import (
    ValidationCommand,
    ValidationSelection,
    ValidationStage,
    build_validation_commands,
    normalize_validation_command_text,
    select_validation_commands,
)


CACHE_SCHEMA = "ipfs_accelerate_py/agent-supervisor/validation-cache@1"
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


class ValidationScheduler:
    """Execute validation stages with bounded weighted parallelism and caching."""

    def __init__(
        self,
        *,
        cache: ValidationResultCache | None = None,
        cache_dir: Path | str | None = None,
        max_workers: int = 2,
        resource_budget: int | None = None,
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
        self.cache = cache or (ValidationResultCache(cache_dir) if cache_dir is not None else None)
        self.max_workers = int(max_workers)
        self.resource_budget = budget
        self.default_timeout_seconds = max(0.001, float(default_timeout_seconds))
        self.runner = runner or run_validation_command

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

        timeout = spec.timeout_seconds or self.default_timeout_seconds
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

    def run(
        self,
        commands: Iterable[str | ValidationCommand],
        *,
        workspace_path: Path | str,
        target_commit: str | None = None,
        changed_files: Iterable[str] | None = None,
        environment: Mapping[str, object] | None = None,
        dependency_state: Mapping[str, object] | Sequence[object] | str | None = None,
        require_full_validation: bool = False,
        scope: str | None = None,
        runner: ValidationRunner | None = None,
    ) -> dict[str, Any]:
        """Schedule commands and return a legacy-compatible JSON report."""

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
            if stage == ValidationStage.CHEAP:
                stage_results: list[dict[str, object]] = []
                for spec in stage_specs:
                    result = execute(spec)
                    stage_results.append(result)
                    if int(result.get("returncode", 1)) != 0:
                        break
            else:
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
        }
        if failed is not None:
            report["failed_command"] = str(failed.get("command") or "")
            if failed.get("timed_out"):
                report["error"] = "timeout"
        return report

    # Natural aliases used by different supervisor embeddings.
    schedule = run
    validate = run


def schedule_validations(
    commands: Iterable[str | ValidationCommand],
    *,
    workspace_path: Path | str,
    **kwargs: object,
) -> dict[str, Any]:
    """Convenience wrapper for one uncached scheduler invocation."""

    scheduler_keys = {"cache", "cache_dir", "max_workers", "resource_budget", "default_timeout_seconds", "runner"}
    scheduler_kwargs = {key: kwargs.pop(key) for key in tuple(kwargs) if key in scheduler_keys}
    return ValidationScheduler(**scheduler_kwargs).run(
        commands,
        workspace_path=workspace_path,
        **kwargs,
    )
