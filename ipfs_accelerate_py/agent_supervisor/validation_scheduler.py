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
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
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


# Compatibility spelling used by integrations that foreground proof work.
schedule_proof_validations = schedule_staged_validations
