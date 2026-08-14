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
import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..analysis.code_evidence_graph import (
    ChangedASTSymbol,
    CodeImpactIndex,
    CodeImpactResult,
)
from ..analysis.cache_coordinator import (
    CacheAuthority,
    CacheNamespace,
    CacheQuotaPolicy,
    CacheRecordOutcome,
    NamespaceCacheCoordinator,
    SemanticCacheKey,
    build_namespace_semantic_key,
)
from ..runtime.resource_scheduler import (
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
from .validation_runtime import (
    VALIDATION_PYTHON_INTERPRETER_SHA256_ENV,
    VALIDATION_PYTHON_INTERPRETER_STAT_ENV,
    VALIDATION_PYTHON_LAUNCHER_MODE_ENV,
    VALIDATION_PYTHON_LAUNCHER_POLICY_SHA256_ENV,
    VALIDATION_PYTHON_LAUNCHER_SHA256_ENV,
    HermeticValidationRuntime,
    ValidationCancellationToken,
    ValidationResourceBounds,
    build_validation_environment,
    build_hermetic_validation_runtime,
    runner_requires_sealed_validation_python,
    run_hermetic_validation_process,
    validation_environment_for_runner,
    validation_shell_command,
)


CACHE_SCHEMA = "ipfs_accelerate_py/agent-supervisor/validation-cache@1"
STAGED_REPORT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/staged-validation-report@1"
)
VALIDATION_DAG_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/validation-dag-receipt@3"
)
TRANSITIVE_IMPACT_EVIDENCE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/transitive-impact-validation-evidence@2"
)
TRANSITIVE_IMPACT_REQUIREMENT_ID = "266404049326363900535699811645710804440"
TRANSITIVE_IMPACT_OBJECTIVE_ID = "ASI-G101"
TRANSITIVE_IMPACT_OBJECTIVE_REVISION = "ASI-G101@asi-075"
TRANSITIVE_IMPACT_COMPLETION_ANALYZER_VERSION = (
    "asi-g101-objective-validation@1"
)
TRANSITIVE_IMPACT_COMPLETION_CONFIGURATION_REVISION = (
    "strict-transitive-impact-completion@1"
)
STRICT_VALIDATION_PARENT_OBJECTIVE_ID = "ASI-G040"
STRICT_VALIDATION_DAG_COMPLETION_EVIDENCE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/"
    "strict-validation-dag-completion-evidence@1"
)
STRICT_VALIDATION_GATE_KINDS = (
    "schema",
    "authority",
    "patch",
    "path",
    "ast_interface",
    "impact_test",
    "semantic_proof",
    "merge",
    "freshness",
)
STRICT_VALIDATION_SCHEDULER_GATE_KINDS = (
    "impact_test",
    "semantic_proof",
    "merge",
    "freshness",
)
TRANSITIVE_IMPACT_ACCEPTANCE_CRITERIA = (
    (
        "The validation DAG is derived from the canonical changed-file and "
        "dependency/interface impact graph and validated declarations"
    ),
    (
        "the receipt contains the complete selected population and every "
        "mandatory direct and transitive validation exactly once"
    ),
    (
        "missing, stale, cyclic, inconsistent, or population-incomplete "
        "impact evidence fails closed before granting authority"
    ),
    (
        "a seeded upstream defect selects and executes its transitively "
        "affected consumer validation and records the real failure"
    ),
    (
        "semantic, proof, merge, freshness, and completion authority remain "
        "closed by explicit records bound to the failed validation"
    ),
    (
        "the exact transitive-impact requirement is emitted only by a "
        "tamper-evident current-tree witness"
    ),
)
VALIDATION_THROUGHPUT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/acceptance-throughput@1"
)
VALIDATION_THROUGHPUT_LANE = "validation"
IMPACT_SELECTED_VALIDATION_DAG_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/impact-selected-validation-dag@1"
)
IMPACT_SELECTED_VALIDATION_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/impact-selected-validation-receipt@1"
)
HERMETIC_VALIDATION_POLICY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/hermetic-validation-policy@1"
)
HERMETIC_VALIDATION_BENCHMARK_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/hermetic-validation-benchmark@1"
)
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
    "PYTHON",
    "PYTHONHASHSEED",
    "PYTHONNOUSERSITE",
    "PYTHONPATH",
    "PYTHONWARNINGS",
    VALIDATION_PYTHON_INTERPRETER_SHA256_ENV,
    VALIDATION_PYTHON_INTERPRETER_STAT_ENV,
    VALIDATION_PYTHON_LAUNCHER_MODE_ENV,
    VALIDATION_PYTHON_LAUNCHER_POLICY_SHA256_ENV,
    VALIDATION_PYTHON_LAUNCHER_SHA256_ENV,
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


@dataclass(frozen=True)
class ValidationStageBatch:
    """Results and monotonic benchmark counters for one parallel stage.

    Keeping these counters on the returned batch, instead of mutable scheduler
    state, makes one ``ValidationScheduler`` safe to use from multiple
    supervisor lanes.  ``serial_work_seconds`` is the measured sum of command
    execution time; comparing it with wall time exposes useful parallelism
    without relying on timestamp strings supplied by command adapters.
    """

    results: tuple[dict[str, object], ...]
    elapsed_seconds: float
    serial_work_seconds: float
    peak_parallelism: int

    @property
    def throughput_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return len(self.results) / self.elapsed_seconds

    @property
    def parallel_speedup(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.serial_work_seconds / self.elapsed_seconds

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": VALIDATION_THROUGHPUT_SCHEMA,
            "lane": VALIDATION_THROUGHPUT_LANE,
            "elapsed_seconds": self.elapsed_seconds,
            "serial_work_seconds": self.serial_work_seconds,
            "peak_parallelism": self.peak_parallelism,
            "completed_count": len(self.results),
            "throughput_per_second": self.throughput_per_second,
            "parallel_speedup": self.parallel_speedup,
        }


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


def _validation_result_digest(
    result: Mapping[str, object],
    *,
    cache_key: "ValidationCacheKey | None" = None,
    trust_stored_digest: bool = False,
) -> str:
    """Return the stable, authority-bearing digest for a command outcome.

    Execution reports contain useful operational fields (timestamps, resource
    lease identifiers, elapsed time, and ``cache_hit``), but those fields are
    deliberately not validation evidence.  Hashing the whole report made an
    exact cache replay produce a different validation DAG receipt from the
    original execution.  It also meant dropping bulky command output from the
    durable cache changed the receipt binding.

    The digest below binds the exact semantic cache key and command outcome.
    Output is retained by digest and byte length as well as in the bounded cache
    payload.  A stored digest is returned unchanged on replay; its cache
    envelope and semantic key are independently integrity checked by
    :class:`ValidationResultCache`.  Runner-provided digest fields are never
    trusted on first execution.
    """

    existing = str(result.get("validation_result_digest") or "").strip()
    if existing and trust_stored_digest:
        return existing
    output = str(result.get("output") or "")
    payload: dict[str, object] = {
        "schema": "ipfs_accelerate_py/agent-supervisor/validation-result@1",
        "cache_key": cache_key.digest if cache_key is not None else "",
        "target_commit": (
            cache_key.target_commit if cache_key is not None else ""
        ),
        "command": (
            cache_key.command
            if cache_key is not None
            else normalize_validation_command_text(
                str(result.get("command") or "")
            )
        ),
        "dependency_state": (
            cache_key.dependency_state if cache_key is not None else ""
        ),
        "environment": (
            dict(cache_key.environment) if cache_key is not None else {}
        ),
        "returncode": int(result.get("returncode", 1)),
        "timed_out": bool(result.get("timed_out", False)),
        "error": str(result.get("error") or ""),
        "outcome": str(result.get("outcome") or ""),
        "attempt_signatures": [
            str(item.get("diagnostic_signature") or "")
            for item in result.get("attempts", ())
            if isinstance(item, Mapping)
        ],
        "output_sha256": _sha256_bytes(output.encode("utf-8")),
        "output_bytes": len(output.encode("utf-8")),
        # Seeded-defect observations are validation evidence and must not be
        # detachable from the result that observed them.
        "seeded_defect_id": str(result.get("seeded_defect_id") or ""),
        # The nested-Python delivery receipt proves that execution used the
        # launcher policy already bound into the semantic cache environment.
        # Keep it authority-bearing rather than treating it as runtime
        # telemetry.
        "validation_python_launcher": _json_safe(
            result.get("validation_python_launcher") or {}
        ),
        "attempt_validation_python_launchers": [
            _json_safe(item.get("validation_python_launcher") or {})
            for item in result.get("attempts", ())
            if isinstance(item, Mapping)
        ],
        "runtime_id": str(result.get("runtime_id") or ""),
        "cancellation_id": str(result.get("cancellation_id") or ""),
        "hermetic_runtime": _json_safe(
            result.get("hermetic_runtime") or {}
        ),
        "attempt_runtime_receipts": [
            {
                "runtime_id": str(item.get("runtime_id") or ""),
                "cancellation_id": str(
                    item.get("cancellation_id") or ""
                ),
                "expected_runtime_id": str(
                    item.get("expected_runtime_id") or ""
                ),
                "expected_cancellation_id": str(
                    item.get("expected_cancellation_id") or ""
                ),
            }
            for item in result.get("attempts", ())
            if isinstance(item, Mapping)
        ],
    }
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


def _validation_python_launcher_receipt_matches_environment(
    result: Mapping[str, object],
    environment: Mapping[str, object],
) -> bool:
    """Require exact sealed-runner evidence for sealed cache identities."""

    mode = str(
        environment.get(VALIDATION_PYTHON_LAUNCHER_MODE_ENV) or ""
    )
    if not mode.endswith(":sealed-memfd"):
        return True
    receipt = result.get("validation_python_launcher")
    if not isinstance(receipt, Mapping):
        return False
    expected = {
        "content_sha256": str(
            environment.get(VALIDATION_PYTHON_LAUNCHER_SHA256_ENV) or ""
        ),
        "interpreter_sha256": str(
            environment.get(VALIDATION_PYTHON_INTERPRETER_SHA256_ENV) or ""
        ),
        "interpreter_stat": str(
            environment.get(VALIDATION_PYTHON_INTERPRETER_STAT_ENV) or ""
        ),
        "mode": mode,
        "policy_sha256": str(
            environment.get(
                VALIDATION_PYTHON_LAUNCHER_POLICY_SHA256_ENV
            )
            or ""
        ),
        "sealed": True,
    }
    if {str(key): value for key, value in receipt.items()} != expected:
        return False
    attempts = result.get("attempts")
    if isinstance(attempts, Sequence) and not isinstance(
        attempts, (str, bytes, bytearray)
    ):
        for attempt in attempts:
            if not isinstance(attempt, Mapping):
                return False
            attempt_receipt = attempt.get("validation_python_launcher")
            if not isinstance(attempt_receipt, Mapping):
                return False
            if (
                {str(key): value for key, value in attempt_receipt.items()}
                != expected
            ):
                return False
    return True


def _hermetic_runtime_receipts_match(
    result: Mapping[str, object],
    runtime: HermeticValidationRuntime,
    *,
    expected_attempts: int,
) -> bool:
    """Require cached authority to retain every exact runtime receipt."""

    if (
        str(result.get("runtime_id") or "") != runtime.runtime_id
        or str(result.get("cancellation_id") or "")
        != runtime.cancellation_id
        or _json_safe(result.get("hermetic_runtime") or {})
        != runtime.to_dict()
        or str(result.get("outcome") or "")
        != ValidationOutcome.PASSED.value
        or str(result.get("classification") or "")
        != ValidationOutcome.PASSED.value
        or result.get("authoritative") is not True
        or result.get("stable") is not True
    ):
        return False
    attempts = result.get("attempts")
    if not isinstance(attempts, Sequence) or isinstance(
        attempts, (str, bytes, bytearray)
    ):
        return False
    try:
        recorded_attempt_count = int(result.get("attempt_count", -1))
    except (TypeError, ValueError):
        return False
    if len(attempts) != expected_attempts or (
        recorded_attempt_count != expected_attempts
    ):
        return False
    for attempt in attempts:
        if not isinstance(attempt, Mapping):
            return False
        try:
            attempt_returncode = int(attempt.get("returncode", 1))
        except (TypeError, ValueError):
            return False
        if (
            str(attempt.get("runtime_id") or "") != runtime.runtime_id
            or str(attempt.get("cancellation_id") or "")
            != runtime.cancellation_id
            or attempt_returncode != 0
            or attempt.get("timed_out") is True
            or attempt.get("cancelled") is True
            or attempt.get("infrastructure_failure") is True
            or attempt.get("inconclusive") is True
        ):
            return False
    return True


@dataclass(frozen=True)
class ValidationCacheKey:
    """Canonical components and digest for one reusable validation result."""

    target_commit: str
    command: str
    environment: tuple[tuple[str, str], ...]
    dependency_state: str
    digest: str
    semantic_key: SemanticCacheKey | None = field(
        default=None, repr=False, compare=False
    )

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "target_commit": self.target_commit,
            "command": self.command,
            "environment": dict(self.environment),
            "dependency_state": self.dependency_state,
            "digest": self.digest,
        }
        if self.semantic_key is not None:
            result["semantic_key"] = self.semantic_key.to_dict()
        return result


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
    semantic_key = build_namespace_semantic_key(
        CacheNamespace.VALIDATION,
        target_commit=payload["target_commit"],
        # Candidate content is part of dependency_state even when HEAD is
        # unchanged.  Keep it separately named in the common contract so it
        # cannot be omitted by another validation adapter.
        candidate_tree=dependency_fingerprint or payload["target_commit"],
        command=normalized_command,
        environment=environment_subset,
        dependency_state=dependency_fingerprint or "none",
        toolchain={
            "python": environment_subset.get("VIRTUAL_ENV", "system"),
            "path": environment_subset.get("PATH", ""),
        },
        policy="successful-exact-result-only@1",
        schema_version=CACHE_SCHEMA,
    )
    return ValidationCacheKey(
        target_commit=payload["target_commit"],
        command=normalized_command,
        environment=tuple(environment_subset.items()),
        dependency_state=dependency_fingerprint,
        digest=_sha256_bytes(_canonical_json(payload).encode("utf-8")),
        semantic_key=semantic_key,
    )


class ValidationResultCache:
    """Integrity-checked validation namespace backed by the common envelope.

    Only successful, non-timeout command results are authoritative.  Native
    validation payload validation is rerun on every exact-key lookup; corrupt,
    stale, future-dated, or trust-poisoned entries are removed and repaired on
    the next production.  The coordinator supplies process-keyed leases and
    persistent count/byte/entry bounds without changing the scheduler's legacy
    report fields.
    """

    def __init__(
        self,
        cache_dir: Path | str,
        *,
        max_age_seconds: float | None = None,
        max_entries: int = 512,
        max_bytes: int = 32 * 1024 * 1024,
        max_entry_bytes: int = 256 * 1024,
        wait_timeout_seconds: float = 30.0,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.max_age_seconds = None if max_age_seconds is None else max(0.0, float(max_age_seconds))
        quota = CacheQuotaPolicy(
            max_entries=max_entries,
            max_bytes=max_bytes,
            max_entry_bytes=max_entry_bytes,
        )
        self.coordinator = NamespaceCacheCoordinator(
            self.cache_dir,
            quotas={CacheNamespace.VALIDATION: quota},
            wait_timeout_seconds=wait_timeout_seconds,
        )

    def _path(self, key: ValidationCacheKey | str) -> Path:
        if not isinstance(key, ValidationCacheKey) or key.semantic_key is None:
            digest = str(key)
            return self.cache_dir / "legacy" / digest[:2] / f"{digest}.json"
        return self.coordinator._entry_path(key.semantic_key)

    @staticmethod
    def _valid_result(result: Any) -> bool:
        if not isinstance(result, Mapping):
            return False
        try:
            return (
                int(result.get("returncode", 1)) == 0
                and result.get("timed_out") is not True
                and not result.get("error")
            )
        except (TypeError, ValueError):
            return False

    def _semantic_key(self, key: ValidationCacheKey | str) -> SemanticCacheKey | None:
        if isinstance(key, ValidationCacheKey):
            return key.semantic_key
        return None

    def get(self, key: ValidationCacheKey | str) -> dict[str, Any] | None:
        semantic_key = self._semantic_key(key)
        if semantic_key is None:
            return None
        lookup = self.coordinator.lookup(
            semantic_key,
            require_completion_evidence=True,
            payload_validator=self._valid_result,
        )
        if not lookup.hit or not isinstance(lookup.payload, Mapping):
            return None
        if (
            self.max_age_seconds is not None
            and lookup.entry is not None
            and (
                time.time() - lookup.entry.created_at_ms / 1000
                > self.max_age_seconds
            )
        ):
            try:
                self._path(key).unlink()
            except OSError:
                pass
            return None
        return dict(lookup.payload)

    def put(self, key: ValidationCacheKey, result: Mapping[str, object]) -> bool:
        if not self._valid_result(result) or key.semantic_key is None:
            return False
        entry = self.coordinator.put(
            key.semantic_key,
            _json_safe(dict(result)),
            outcome=CacheRecordOutcome.SUCCESSFUL,
            authority=CacheAuthority.AUTHORITATIVE,
            ttl_seconds=(
                max(1, int(self.max_age_seconds))
                if self.max_age_seconds is not None
                else None
            ),
            payload_schema=CACHE_SCHEMA,
            payload_validator=self._valid_result,
        )
        return entry is not None

    def get_diagnostic(
        self,
        key: ValidationCacheKey,
        *,
        max_age_seconds: float,
    ) -> dict[str, Any] | None:
        """Return an exact non-authoritative deterministic diagnostic."""

        path = (
            self.cache_dir
            / "diagnostics"
            / key.digest[:2]
            / f"{key.digest}.json"
        )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            created_at = float(payload["created_at"])
            result = payload["result"]
            digest = str(payload["diagnostic_digest"])
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return None
        if time.time() - created_at > max(0.0, float(max_age_seconds)):
            try:
                path.unlink()
            except OSError:
                pass
            return None
        expected = _sha256_bytes(
            _canonical_json(
                {
                    "cache_key": key.digest,
                    "created_at": created_at,
                    "result": result,
                }
            ).encode("utf-8")
        )
        if digest != expected or not isinstance(result, Mapping):
            try:
                path.unlink()
            except OSError:
                pass
            return None
        if str(result.get("outcome") or "") != (
            ValidationOutcome.DETERMINISTIC_FAILURE.value
        ):
            return None
        return dict(result)

    def put_diagnostic(
        self,
        key: ValidationCacheKey,
        result: Mapping[str, object],
    ) -> bool:
        """Persist a bounded diagnostic that can never satisfy authority."""

        if str(result.get("outcome") or "") != (
            ValidationOutcome.DETERMINISTIC_FAILURE.value
        ):
            return False
        path = (
            self.cache_dir
            / "diagnostics"
            / key.digest[:2]
            / f"{key.digest}.json"
        )
        created_at = time.time()
        safe_result = _json_safe(dict(result))
        unsigned = {
            "cache_key": key.digest,
            "created_at": created_at,
            "result": safe_result,
        }
        payload = {
            **unsigned,
            "diagnostic_digest": _sha256_bytes(
                _canonical_json(unsigned).encode("utf-8")
            ),
            "authoritative": False,
        }
        rendered = _canonical_json(payload)
        if len(rendered.encode("utf-8")) > 256 * 1024:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(
            f".{os.getpid()}.{threading.get_ident()}.tmp"
        )
        try:
            temporary.write_text(rendered, encoding="utf-8")
            os.replace(temporary, path)
        except OSError:
            try:
                temporary.unlink()
            except OSError:
                pass
            return False
        return True

    @contextmanager
    def single_flight(self, key: ValidationCacheKey) -> Iterable[None]:
        if key.semantic_key is None:
            yield
            return
        with self.coordinator.lease(key.semantic_key):
            yield

    def metrics(self):
        return self.coordinator.metrics(CacheNamespace.VALIDATION)

    stats = metrics

    def prune(self) -> dict[str, int]:
        return self.coordinator.gc(CacheNamespace.VALIDATION)

    def clear(self) -> int:
        return self.coordinator.clear(CacheNamespace.VALIDATION)


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
_HERMETIC_VALIDATION_RUNNER_ATTRIBUTE = (
    "__ipfs_accelerate_hermetic_validation_runner__"
)


def hermetic_validation_runner(runner: ValidationRunner) -> ValidationRunner:
    """Declare that a trusted runner executes the supplied runtime context."""

    setattr(runner, _HERMETIC_VALIDATION_RUNNER_ATTRIBUTE, True)
    return runner


def _runner_supports_hermetic_validation(runner: ValidationRunner) -> bool:
    target = getattr(runner, "__func__", runner)
    if (
        getattr(
            target,
            _HERMETIC_VALIDATION_RUNNER_ATTRIBUTE,
            False,
        )
        is not True
    ):
        return False
    try:
        signature = inspect.signature(runner)
    except (TypeError, ValueError):
        return False
    return "runtime_context" in signature.parameters


@hermetic_validation_runner
def run_validation_command(
    *,
    spec: ValidationCommand,
    workspace_path: Path,
    timeout_seconds: float,
    environment: Mapping[str, str],
    runtime_context: HermeticValidationRuntime | None = None,
    cancellation_token: ValidationCancellationToken | None = None,
    attempt_number: int = 1,
) -> dict[str, object]:
    """Default non-interactive shell runner with captured combined output."""

    started_at = utc_now()
    if runtime_context is not None:
        result = run_hermetic_validation_process(
            runtime_context,
            cancellation_token=cancellation_token,
        )
        return {
            "command": spec.command,
            "raw_command": spec.raw_command or spec.command,
            "started_at": started_at,
            "finished_at": utc_now(),
            "attempt_number": int(attempt_number),
            **result,
        }
    try:
        from .validation_runtime import apply_sealed_node_toolchain

        child_environment = apply_sealed_node_toolchain(
            dict(environment),
            workspace_path=workspace_path,
            command=spec.command,
        )
        completed = subprocess.run(
            validation_shell_command(spec.command),
            cwd=workspace_path,
            text=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            check=False,
            env=child_environment,
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


def _attempt_diagnostic_signature(result: Mapping[str, object]) -> str:
    output = str(result.get("output") or "")
    return _sha256_bytes(
        _canonical_json(
            {
                "returncode": int(result.get("returncode", 1)),
                "timed_out": bool(result.get("timed_out", False)),
                "cancelled": bool(result.get("cancelled", False)),
                "infrastructure_failure": bool(
                    result.get("infrastructure_failure", False)
                ),
                "inconclusive": bool(result.get("inconclusive", False)),
                "error": str(result.get("error") or ""),
                "seeded_defect_id": str(
                    result.get("seeded_defect_id") or ""
                ),
                "seeded_defect_ids": sorted(
                    str(value)
                    for value in (
                        (result.get("seeded_defect_ids"),)
                        if isinstance(
                            result.get("seeded_defect_ids"), str
                        )
                        else result.get("seeded_defect_ids") or ()
                    )
                ),
                "output_sha256": _sha256_bytes(output.encode("utf-8")),
            }
        ).encode("utf-8")
    )


def classify_validation_attempts(
    attempts: Sequence[Mapping[str, object]],
) -> ValidationOutcome:
    """Classify repeated observations without promoting intermittent passes."""

    if not attempts:
        return ValidationOutcome.INCONCLUSIVE
    if any(bool(item.get("cancelled", False)) for item in attempts):
        return ValidationOutcome.CANCELLED
    if any(bool(item.get("timed_out", False)) for item in attempts):
        return ValidationOutcome.TIMEOUT
    if any(
        bool(item.get("infrastructure_failure", False))
        or str(item.get("error") or "").startswith(
            ("resource_admission_", "hermetic_runtime_")
        )
        for item in attempts
    ):
        return ValidationOutcome.INFRASTRUCTURE_FAILURE
    if any(bool(item.get("inconclusive", False)) for item in attempts):
        return ValidationOutcome.INCONCLUSIVE
    pass_states = [
        int(item.get("returncode", 1)) == 0 for item in attempts
    ]
    signatures = {
        _attempt_diagnostic_signature(item) for item in attempts
    }
    if len(set(pass_states)) > 1:
        return ValidationOutcome.FLAKY
    if all(pass_states):
        # Successful command output is allowed to contain nondeterministic
        # timing text. Stability is about the verdict, not byte-identical logs.
        return ValidationOutcome.PASSED
    if len(signatures) > 1:
        return ValidationOutcome.FLAKY
    return ValidationOutcome.DETERMINISTIC_FAILURE


def validation_benchmark(
    *,
    baseline_seconds: Sequence[float],
    optimized_seconds: Sequence[float],
    minimum_reduction: float = 0.30,
) -> dict[str, object]:
    """Return a deterministic median time-to-useful-failure comparison."""

    import statistics

    baseline = tuple(float(value) for value in baseline_seconds)
    optimized = tuple(float(value) for value in optimized_seconds)
    if not baseline or not optimized or any(
        value < 0 for value in (*baseline, *optimized)
    ):
        raise ValidationDAGError(
            "validation benchmark requires non-negative baseline and optimized samples"
        )
    baseline_median = statistics.median(baseline)
    optimized_median = statistics.median(optimized)
    reduction = (
        (baseline_median - optimized_median) / baseline_median
        if baseline_median > 0
        else 0.0
    )
    threshold = float(minimum_reduction)
    return {
        "schema": HERMETIC_VALIDATION_BENCHMARK_SCHEMA,
        "baseline_samples_seconds": list(baseline),
        "optimized_samples_seconds": list(optimized),
        "baseline_median_seconds": baseline_median,
        "optimized_median_seconds": optimized_median,
        "reduction": reduction,
        "minimum_reduction": threshold,
        "passed": reduction >= threshold,
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


class ValidationTechnique(str, Enum):
    """Orthogonal test technique applied by an impact-selected check."""

    STANDARD = "standard"
    CONTRACT = "contract"
    DIFFERENTIAL = "differential"
    METAMORPHIC = "metamorphic"
    MUTATION = "mutation"


class ValidationOutcome(str, Enum):
    """Typed terminal outcome for a stabilized validation node."""

    PASSED = "passed"
    DETERMINISTIC_FAILURE = "deterministic_failure"
    FLAKY = "flaky"
    TIMEOUT = "timeout"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"
    INCONCLUSIVE = "inconclusive"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class HermeticValidationPolicy:
    """Execution and evidence policy for authority-bearing validation."""

    stability_runs: int = 2
    complete_selected_dag: bool = True
    strict_isolation: bool = True
    required_techniques: tuple[ValidationTechnique, ...] = (
        ValidationTechnique.CONTRACT,
        ValidationTechnique.DIFFERENTIAL,
        ValidationTechnique.METAMORPHIC,
        ValidationTechnique.MUTATION,
    )
    resource_bounds: ValidationResourceBounds = field(
        default_factory=ValidationResourceBounds
    )
    diagnostic_ttl_seconds: float = 3600.0
    minimum_time_to_failure_reduction: float = 0.30
    policy_id: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.stability_runs, bool) or int(self.stability_runs) < 2:
            raise ValidationDAGError(
                "hermetic validation requires at least two stability runs"
            )
        object.__setattr__(self, "stability_runs", int(self.stability_runs))
        object.__setattr__(
            self, "complete_selected_dag", bool(self.complete_selected_dag)
        )
        object.__setattr__(self, "strict_isolation", bool(self.strict_isolation))
        if not self.strict_isolation:
            raise ValidationDAGError(
                "hermetic validation cannot disable strict isolation"
            )
        raw_techniques = self.required_techniques
        techniques = tuple(
            sorted(
                {
                    ValidationTechnique(value)
                    for value in (
                        (raw_techniques,)
                        if isinstance(raw_techniques, str)
                        else raw_techniques
                    )
                },
                key=lambda value: list(ValidationTechnique).index(value),
            )
        )
        if ValidationTechnique.STANDARD in techniques:
            raise ValidationDAGError(
                "standard is not a hermetic coverage technique"
            )
        object.__setattr__(self, "required_techniques", techniques)
        if not isinstance(self.resource_bounds, ValidationResourceBounds):
            object.__setattr__(
                self,
                "resource_bounds",
                ValidationResourceBounds.from_dict(self.resource_bounds),
            )
        ttl = float(self.diagnostic_ttl_seconds)
        if ttl <= 0:
            raise ValidationDAGError(
                "hermetic diagnostic TTL must be positive"
            )
        object.__setattr__(self, "diagnostic_ttl_seconds", ttl)
        reduction = float(self.minimum_time_to_failure_reduction)
        if reduction < 0 or reduction >= 1:
            raise ValidationDAGError(
                "time-to-failure reduction must be in [0, 1)"
            )
        object.__setattr__(
            self, "minimum_time_to_failure_reduction", reduction
        )
        claimed = str(self.policy_id or "").strip()
        object.__setattr__(self, "policy_id", "")
        actual = _sha256_bytes(
            _canonical_json(self._identity_payload()).encode("utf-8")
        )
        if claimed and claimed != actual:
            raise ValidationDAGError(
                "hermetic validation policy identity mismatch"
            )
        object.__setattr__(self, "policy_id", actual)

    def _identity_payload(self) -> dict[str, object]:
        return {
            "schema": HERMETIC_VALIDATION_POLICY_SCHEMA,
            "stability_runs": self.stability_runs,
            "complete_selected_dag": self.complete_selected_dag,
            "strict_isolation": self.strict_isolation,
            "required_techniques": [
                value.value for value in self.required_techniques
            ],
            "resource_bounds": self.resource_bounds.to_dict(),
            "diagnostic_ttl_seconds": self.diagnostic_ttl_seconds,
            "minimum_time_to_failure_reduction": (
                self.minimum_time_to_failure_reduction
            ),
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._identity_payload(), "policy_id": self.policy_id}

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "HermeticValidationPolicy":
        return cls(
            stability_runs=int(value.get("stability_runs", 2)),
            complete_selected_dag=bool(
                value.get("complete_selected_dag", True)
            ),
            strict_isolation=bool(value.get("strict_isolation", True)),
            required_techniques=tuple(
                ValidationTechnique(item)
                for item in value.get(
                    "required_techniques",
                    (
                        "contract",
                        "differential",
                        "metamorphic",
                        "mutation",
                    ),
                )
            ),
            resource_bounds=ValidationResourceBounds.from_dict(
                value.get("resource_bounds") or {}
            ),
            diagnostic_ttl_seconds=float(
                value.get("diagnostic_ttl_seconds", 3600.0)
            ),
            minimum_time_to_failure_reduction=float(
                value.get("minimum_time_to_failure_reduction", 0.30)
            ),
            policy_id=str(value.get("policy_id") or ""),
        )


@dataclass(frozen=True)
class SeededValidationDefect:
    """Expected defect observation used to audit validation effectiveness."""

    defect_id: str
    path: str
    expected_check_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        defect_id = str(self.defect_id or "").strip()
        path = _normalize_impact_path(self.path)
        if not defect_id or not path:
            raise ValidationDAGError(
                "seeded validation defect requires identity and path"
            )
        object.__setattr__(self, "defect_id", defect_id)
        object.__setattr__(self, "path", path)
        object.__setattr__(
            self,
            "expected_check_ids",
            tuple(
                sorted(
                    {
                        str(value).strip()
                        for value in self.expected_check_ids
                        if str(value).strip()
                    }
                )
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "defect_id": self.defect_id,
            "path": self.path,
            "expected_check_ids": list(self.expected_check_ids),
        }


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


class ImpactValidationKind(str, Enum):
    """Mandatory implementation-output validation dimensions."""

    SYNTAX = "syntax"
    TYPE = "type"
    INTERFACE = "interface"
    UNIT = "unit"
    INTEGRATION = "integration"
    CONTRACT = "contract"
    RUNTIME = "runtime"


MANDATORY_VALIDATION_KINDS = tuple(ImpactValidationKind)
DEFAULT_VALIDATION_KIND_DEPENDENCIES: Mapping[
    ImpactValidationKind, tuple[ImpactValidationKind, ...]
] = {
    ImpactValidationKind.SYNTAX: (),
    ImpactValidationKind.TYPE: (ImpactValidationKind.SYNTAX,),
    ImpactValidationKind.INTERFACE: (ImpactValidationKind.TYPE,),
    ImpactValidationKind.UNIT: (ImpactValidationKind.TYPE,),
    ImpactValidationKind.INTEGRATION: (
        ImpactValidationKind.INTERFACE,
        ImpactValidationKind.UNIT,
    ),
    ImpactValidationKind.CONTRACT: (
        ImpactValidationKind.INTERFACE,
        ImpactValidationKind.UNIT,
    ),
    ImpactValidationKind.RUNTIME: (
        ImpactValidationKind.INTEGRATION,
        ImpactValidationKind.CONTRACT,
    ),
}
_IMPACT_KIND_STAGE: Mapping[ImpactValidationKind, ValidationStage] = {
    ImpactValidationKind.SYNTAX: ValidationStage.CHEAP,
    ImpactValidationKind.TYPE: ValidationStage.CHEAP,
    ImpactValidationKind.INTERFACE: ValidationStage.TARGETED,
    ImpactValidationKind.UNIT: ValidationStage.TARGETED,
    ImpactValidationKind.INTEGRATION: ValidationStage.BROAD,
    ImpactValidationKind.CONTRACT: ValidationStage.BROAD,
    ImpactValidationKind.RUNTIME: ValidationStage.ATTESTATION,
}


def _normalized_text(value: object) -> str:
    return " ".join(str(value or "").split())


@dataclass(frozen=True)
class ImpactValidationCheck:
    """One reviewed check available to the impact DAG planner.

    Empty ``targets`` means a repository-wide check.  Non-empty targets must
    name paths or symbols present in the exact :class:`CodeImpactIndex`.
    """

    check_id: str
    kind: ImpactValidationKind
    command: str
    technique: ValidationTechnique | None = None
    targets: tuple[str, ...] = ()
    acceptance_criteria: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    source: str = "repository_policy"
    resource_cost: int = 1
    cacheable: bool = True
    timeout_seconds: float | None = None
    environment_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        check_id = str(self.check_id or "").strip()
        command = normalize_validation_command_text(str(self.command or ""))
        if not check_id or not command:
            raise ValidationDAGError(
                "impact validation checks require check_id and command"
            )
        object.__setattr__(self, "check_id", check_id)
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "kind", ImpactValidationKind(self.kind))
        technique = self.technique
        if technique is None:
            technique = (
                ValidationTechnique.CONTRACT
                if self.kind is ImpactValidationKind.CONTRACT
                else ValidationTechnique.STANDARD
            )
        object.__setattr__(
            self, "technique", ValidationTechnique(technique)
        )
        for name in ("targets", "acceptance_criteria", "depends_on"):
            raw_values = getattr(self, name)
            values = (
                (raw_values,) if isinstance(raw_values, str) else raw_values
            )
            object.__setattr__(
                self,
                name,
                tuple(
                    sorted(
                        {
                            _normalized_text(value)
                            for value in values
                            if _normalized_text(value)
                        }
                    )
                ),
            )
        object.__setattr__(
            self,
            "environment_keys",
            tuple(
                sorted(
                    {
                        str(value).strip()
                        for value in (
                            (self.environment_keys,)
                            if isinstance(self.environment_keys, str)
                            else self.environment_keys
                        )
                        if str(value).strip()
                    }
                )
            ),
        )
        source = str(self.source or "repository_policy").strip()
        object.__setattr__(self, "source", source)
        if isinstance(self.resource_cost, bool) or int(self.resource_cost) < 1:
            raise ValidationDAGError(
                "impact validation resource_cost must be positive"
            )
        object.__setattr__(self, "resource_cost", int(self.resource_cost))
        if self.timeout_seconds is not None:
            timeout = float(self.timeout_seconds)
            if timeout <= 0:
                raise ValidationDAGError(
                    "impact validation timeout must be positive"
                )
            object.__setattr__(self, "timeout_seconds", timeout)

    def command_spec(self, *, ordinal: int) -> ValidationCommand:
        return ValidationCommand(
            command=self.command,
            raw_command=self.command,
            stage=_IMPACT_KIND_STAGE[self.kind],
            resource_cost=self.resource_cost,
            impact_paths=tuple(
                target for target in self.targets if "/" in target
            ),
            environment_keys=self.environment_keys,
            cacheable=self.cacheable,
            timeout_seconds=self.timeout_seconds,
            ordinal=ordinal,
            validation_id=self.check_id,
            source=self.source,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "check_id": self.check_id,
            "kind": self.kind.value,
            "technique": self.technique.value,
            "command": self.command,
            "targets": list(self.targets),
            "acceptance_criteria": list(self.acceptance_criteria),
            "depends_on": list(self.depends_on),
            "source": self.source,
            "resource_cost": self.resource_cost,
            "cacheable": self.cacheable,
            "timeout_seconds": self.timeout_seconds,
            "environment_keys": list(self.environment_keys),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ImpactValidationCheck":
        def values(name: str, *aliases: str) -> tuple[Any, ...]:
            raw: Any = value.get(name)
            for alias in aliases:
                if raw is None:
                    raw = value.get(alias)
            if raw is None:
                return ()
            return (raw,) if isinstance(raw, str) else tuple(raw)

        return cls(
            check_id=str(
                value.get("check_id")
                or value.get("validation_id")
                or value.get("id")
                or ""
            ),
            kind=ImpactValidationKind(
                str(value.get("kind") or value.get("check_kind") or "")
            ),
            technique=(
                ValidationTechnique(str(value.get("technique")))
                if value.get("technique")
                else None
            ),
            command=str(value.get("command") or ""),
            targets=values("targets", "impact_targets", "impact_paths"),
            acceptance_criteria=values("acceptance_criteria"),
            depends_on=values("depends_on"),
            source=str(value.get("source") or "repository_policy"),
            resource_cost=int(value.get("resource_cost", 1)),
            cacheable=bool(value.get("cacheable", True)),
            timeout_seconds=(
                float(value["timeout_seconds"])
                if value.get("timeout_seconds") is not None
                else None
            ),
            environment_keys=values("environment_keys"),
        )


@dataclass(frozen=True)
class RepositoryValidationPolicy:
    """Exact repository rules used to derive mandatory DAG nodes."""

    required_kinds: tuple[ImpactValidationKind, ...] = MANDATORY_VALIDATION_KINDS
    required_techniques: tuple[ValidationTechnique, ...] = ()
    required_check_ids: tuple[str, ...] = ()
    acceptance_bindings: Mapping[str, Sequence[str]] = field(default_factory=dict)
    kind_dependencies: Mapping[
        ImpactValidationKind | str,
        Sequence[ImpactValidationKind | str],
    ] = field(
        default_factory=lambda: dict(DEFAULT_VALIDATION_KIND_DEPENDENCIES)
    )
    require_acceptance_coverage: bool = True
    require_transitive_validation: bool = True
    policy_version: str = "repository-validation-policy-v1"
    policy_id: str = ""

    def __post_init__(self) -> None:
        raw_required_kinds = self.required_kinds
        required_kinds = tuple(
            sorted(
                {
                    ImpactValidationKind(value)
                    for value in (
                        (raw_required_kinds,)
                        if isinstance(raw_required_kinds, str)
                        else raw_required_kinds
                    )
                },
                key=lambda value: list(ImpactValidationKind).index(value),
            )
        )
        object.__setattr__(self, "required_kinds", required_kinds)
        raw_required_techniques = self.required_techniques
        object.__setattr__(
            self,
            "required_techniques",
            tuple(
                sorted(
                    {
                        ValidationTechnique(value)
                        for value in (
                            (raw_required_techniques,)
                            if isinstance(raw_required_techniques, str)
                            else raw_required_techniques
                        )
                    },
                    key=lambda value: list(ValidationTechnique).index(value),
                )
            ),
        )
        object.__setattr__(
            self,
            "required_check_ids",
            tuple(
                sorted(
                    {
                        str(value).strip()
                        for value in (
                            (self.required_check_ids,)
                            if isinstance(self.required_check_ids, str)
                            else self.required_check_ids
                        )
                        if str(value).strip()
                    }
                )
            ),
        )
        bindings: dict[str, tuple[str, ...]] = {}
        for criterion, check_ids in dict(self.acceptance_bindings or {}).items():
            normalized = _normalized_text(criterion)
            if not normalized:
                raise ValidationDAGError(
                    "repository policy has an empty acceptance binding"
                )
            values = (
                (check_ids,) if isinstance(check_ids, str) else check_ids
            )
            bindings[normalized.casefold()] = tuple(
                sorted(
                    {
                        str(value).strip()
                        for value in values
                        if str(value).strip()
                    }
                )
            )
        object.__setattr__(
            self, "acceptance_bindings", dict(sorted(bindings.items()))
        )
        dependencies: dict[
            ImpactValidationKind, tuple[ImpactValidationKind, ...]
        ] = {}
        for raw_kind, raw_dependencies in dict(
            self.kind_dependencies or {}
        ).items():
            kind = ImpactValidationKind(raw_kind)
            dependencies[kind] = tuple(
                sorted(
                    {ImpactValidationKind(value) for value in raw_dependencies},
                    key=lambda value: list(ImpactValidationKind).index(value),
                )
            )
            if kind in dependencies[kind]:
                raise ValidationDAGError(
                    "validation kind cannot depend on itself"
                )
        for kind in ImpactValidationKind:
            dependencies.setdefault(kind, ())
        object.__setattr__(
            self,
            "kind_dependencies",
            dict(
                sorted(
                    dependencies.items(),
                    key=lambda item: list(ImpactValidationKind).index(item[0]),
                )
            ),
        )
        version = str(self.policy_version or "").strip()
        if not version:
            raise ValidationDAGError(
                "repository validation policy version is required"
            )
        object.__setattr__(self, "policy_version", version)
        object.__setattr__(
            self,
            "require_acceptance_coverage",
            bool(self.require_acceptance_coverage),
        )
        object.__setattr__(
            self,
            "require_transitive_validation",
            bool(self.require_transitive_validation),
        )
        claimed = str(self.policy_id or "").strip()
        object.__setattr__(self, "policy_id", "")
        actual = _sha256_bytes(
            _canonical_json(self._identity_payload()).encode("utf-8")
        )
        if claimed and claimed != actual:
            raise ValidationDAGError(
                "repository validation policy identity mismatch"
            )
        object.__setattr__(self, "policy_id", actual)

    def _identity_payload(self) -> dict[str, object]:
        return {
            "policy_version": self.policy_version,
            "required_kinds": [value.value for value in self.required_kinds],
            "required_techniques": [
                value.value for value in self.required_techniques
            ],
            "required_check_ids": list(self.required_check_ids),
            "acceptance_bindings": self.acceptance_bindings,
            "kind_dependencies": {
                kind.value: [value.value for value in dependencies]
                for kind, dependencies in self.kind_dependencies.items()
            },
            "require_acceptance_coverage": self.require_acceptance_coverage,
            "require_transitive_validation": self.require_transitive_validation,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._identity_payload(), "policy_id": self.policy_id}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RepositoryValidationPolicy":
        raw_kinds = value.get(
            "required_kinds",
            [kind.value for kind in MANDATORY_VALIDATION_KINDS],
        )
        raw_check_ids = value.get("required_check_ids") or ()
        return cls(
            required_kinds=tuple(
                ImpactValidationKind(item)
                for item in (
                    (raw_kinds,)
                    if isinstance(raw_kinds, str)
                    else raw_kinds
                )
            ),
            required_techniques=tuple(
                ValidationTechnique(item)
                for item in value.get("required_techniques") or ()
            ),
            required_check_ids=(
                (raw_check_ids,)
                if isinstance(raw_check_ids, str)
                else tuple(raw_check_ids)
            ),
            acceptance_bindings=value.get("acceptance_bindings") or {},
            kind_dependencies=value.get("kind_dependencies")
            or dict(DEFAULT_VALIDATION_KIND_DEPENDENCIES),
            require_acceptance_coverage=bool(
                value.get("require_acceptance_coverage", True)
            ),
            require_transitive_validation=bool(
                value.get("require_transitive_validation", True)
            ),
            policy_version=str(
                value.get("policy_version")
                or "repository-validation-policy-v1"
            ),
            policy_id=str(value.get("policy_id") or ""),
        )


@dataclass(frozen=True)
class ImpactValidationPlanNode:
    check: ImpactValidationCheck
    selected: bool
    mandatory: bool
    selection_reasons: tuple[str, ...] = ()
    skipped_reason: str = ""
    depends_on: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.check, ImpactValidationCheck):
            object.__setattr__(
                self, "check", ImpactValidationCheck.from_dict(self.check)
            )
        object.__setattr__(self, "selected", bool(self.selected))
        object.__setattr__(self, "mandatory", bool(self.mandatory))
        object.__setattr__(
            self,
            "selection_reasons",
            tuple(
                sorted(
                    {
                        str(value).strip()
                        for value in self.selection_reasons
                        if str(value).strip()
                    }
                )
            ),
        )
        object.__setattr__(
            self,
            "depends_on",
            tuple(
                sorted(
                    {
                        str(value).strip()
                        for value in self.depends_on
                        if str(value).strip()
                    }
                )
            ),
        )
        object.__setattr__(
            self, "skipped_reason", str(self.skipped_reason or "").strip()
        )
        if self.selected and not self.selection_reasons:
            raise ValidationDAGError(
                "selected impact check requires selection reasons"
            )
        if self.selected and self.skipped_reason:
            raise ValidationDAGError(
                "selected impact check cannot have a skipped reason"
            )
        if not self.selected and not self.skipped_reason:
            raise ValidationDAGError(
                "skipped impact check requires a reason"
            )
        if self.mandatory and not self.selected:
            raise ValidationDAGError(
                "mandatory impact check must be selected"
            )

    @property
    def check_id(self) -> str:
        return self.check.check_id

    def to_dict(self) -> dict[str, object]:
        return {
            **self.check.to_dict(),
            "selected": self.selected,
            "mandatory": self.mandatory,
            "selection_reasons": list(self.selection_reasons),
            "selection_reason": ";".join(self.selection_reasons),
            "skipped_reason": self.skipped_reason,
            "depends_on": list(self.depends_on),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ImpactValidationPlanNode":
        return cls(
            check=ImpactValidationCheck.from_dict(value),
            selected=bool(value.get("selected", False)),
            mandatory=bool(value.get("mandatory", False)),
            selection_reasons=tuple(
                value.get("selection_reasons")
                or (
                    (value.get("selection_reason"),)
                    if value.get("selection_reason")
                    else ()
                )
            ),
            skipped_reason=str(value.get("skipped_reason") or ""),
            depends_on=tuple(value.get("depends_on") or ()),
        )


@dataclass(frozen=True)
class ImpactSelectedValidationDAG:
    """Immutable selection plan bound to tree, impact evidence, and policy."""

    repository_tree_id: str
    impact_index: CodeImpactIndex
    impact: CodeImpactResult
    policy: RepositoryValidationPolicy
    acceptance_criteria: tuple[str, ...]
    nodes: tuple[ImpactValidationPlanNode, ...]
    uncovered_impact: tuple[str, ...] = ()
    dag_id: str = ""

    def __post_init__(self) -> None:
        tree_id = str(self.repository_tree_id or "").strip()
        if (
            not tree_id
            or tree_id != self.impact.repository_tree_id
            or tree_id != self.impact_index.repository_tree_id
        ):
            raise ValidationDAGError(
                "impact validation DAG tree does not match impact evidence"
            )
        if self.impact.index_id != self.impact_index.index_id:
            raise ValidationDAGError(
                "impact validation DAG result is stale for its index"
            )
        recomputed = self.impact_index.impact(
            changed_symbols=self.impact.changed_symbols,
            changed_paths=self.impact.changed_paths,
        )
        if recomputed.to_dict() != self.impact.to_dict():
            raise ValidationDAGError(
                "impact validation DAG does not contain the complete graph closure"
            )
        object.__setattr__(self, "repository_tree_id", tree_id)
        object.__setattr__(
            self,
            "acceptance_criteria",
            tuple(
                sorted(
                    {
                        _normalized_text(value)
                        for value in self.acceptance_criteria
                        if _normalized_text(value)
                    }
                )
            ),
        )
        object.__setattr__(
            self,
            "nodes",
            tuple(sorted(self.nodes, key=lambda value: value.check_id)),
        )
        object.__setattr__(
            self,
            "uncovered_impact",
            tuple(
                sorted(
                    {
                        str(value).strip()
                        for value in self.uncovered_impact
                        if str(value).strip()
                    }
                )
            ),
        )
        ids = [node.check_id for node in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValidationDAGError(
                "impact validation DAG contains duplicate check IDs"
            )
        selected = {node.check_id for node in self.nodes if node.selected}
        for node in self.nodes:
            unknown = set(node.depends_on) - selected
            if unknown:
                raise ValidationDAGError(
                    f"check {node.check_id!r} has unselected dependencies: "
                    + ", ".join(sorted(unknown))
                )
        visiting: set[str] = set()
        visited: set[str] = set()
        by_id = {node.check_id: node for node in self.nodes}

        def visit(check_id: str) -> None:
            if check_id in visited:
                return
            if check_id in visiting:
                raise ValidationDAGError(
                    "impact validation DAG contains a dependency cycle"
                )
            visiting.add(check_id)
            for dependency in by_id[check_id].depends_on:
                visit(dependency)
            visiting.remove(check_id)
            visited.add(check_id)

        for check_id in sorted(selected):
            visit(check_id)
        claimed = str(self.dag_id or "").strip()
        object.__setattr__(self, "dag_id", "")
        actual = _sha256_bytes(
            _canonical_json(self._identity_payload()).encode("utf-8")
        )
        if claimed and claimed != actual:
            raise ValidationDAGError(
                "impact validation DAG identity mismatch"
            )
        object.__setattr__(self, "dag_id", actual)

    @property
    def selected_nodes(self) -> tuple[ImpactValidationPlanNode, ...]:
        return tuple(node for node in self.nodes if node.selected)

    @property
    def coverage_complete(self) -> bool:
        return not self.uncovered_impact

    def _identity_payload(self) -> dict[str, object]:
        return {
            "schema": IMPACT_SELECTED_VALIDATION_DAG_SCHEMA,
            "repository_tree_id": self.repository_tree_id,
            "impact_index": self.impact_index.to_dict(),
            "impact": self.impact.to_dict(),
            "policy": self.policy.to_dict(),
            "acceptance_criteria": list(self.acceptance_criteria),
            "nodes": [node.to_dict() for node in self.nodes],
            "uncovered_impact": list(self.uncovered_impact),
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._identity_payload(), "dag_id": self.dag_id}

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "ImpactSelectedValidationDAG":
        schema = str(
            value.get("schema") or IMPACT_SELECTED_VALIDATION_DAG_SCHEMA
        )
        if schema != IMPACT_SELECTED_VALIDATION_DAG_SCHEMA:
            raise ValidationDAGError(
                f"unsupported impact validation DAG schema: {schema}"
            )
        impact_value = value.get("impact")
        index_value = value.get("impact_index")
        policy_value = value.get("policy")
        if (
            not isinstance(impact_value, Mapping)
            or not isinstance(index_value, Mapping)
            or not isinstance(policy_value, Mapping)
        ):
            raise ValidationDAGError(
                "impact validation DAG is missing index, impact, or policy"
            )
        return cls(
            repository_tree_id=str(value.get("repository_tree_id") or ""),
            impact_index=CodeImpactIndex.from_dict(index_value),
            impact=CodeImpactResult.from_dict(impact_value),
            policy=RepositoryValidationPolicy.from_dict(policy_value),
            acceptance_criteria=tuple(
                value.get("acceptance_criteria") or ()
            ),
            nodes=tuple(
                ImpactValidationPlanNode.from_dict(item)
                for item in value.get("nodes") or ()
            ),
            uncovered_impact=tuple(value.get("uncovered_impact") or ()),
            dag_id=str(value.get("dag_id") or ""),
        )


def build_impact_selected_validation_dag(
    *,
    impact_index: CodeImpactIndex | Mapping[str, Any],
    checks: Iterable[ImpactValidationCheck | Mapping[str, Any]],
    changed_symbols: Iterable[
        str | ChangedASTSymbol | Mapping[str, Any]
    ] = (),
    changed_paths: Iterable[str] = (),
    acceptance_criteria: Iterable[str] = (),
    repository_policy: (
        RepositoryValidationPolicy | Mapping[str, Any] | None
    ) = None,
) -> ImpactSelectedValidationDAG:
    """Derive all mandatory and impacted nodes without executing commands."""

    index = (
        impact_index
        if isinstance(impact_index, CodeImpactIndex)
        else CodeImpactIndex.from_dict(impact_index)
    )
    policy = (
        repository_policy
        if isinstance(repository_policy, RepositoryValidationPolicy)
        else RepositoryValidationPolicy.from_dict(repository_policy)
        if repository_policy is not None
        else RepositoryValidationPolicy()
    )
    check_values = tuple(
        value
        if isinstance(value, ImpactValidationCheck)
        else ImpactValidationCheck.from_dict(value)
        for value in checks
    )
    check_by_id = {check.check_id: check for check in check_values}
    if len(check_by_id) != len(check_values):
        raise ValidationDAGError(
            "impact validation catalog contains duplicate check IDs"
        )
    unknown_dependencies = {
        dependency
        for check in check_values
        for dependency in check.depends_on
        if dependency not in check_by_id
    }
    if unknown_dependencies:
        raise ValidationDAGError(
            "impact validation catalog references unknown dependencies: "
            + ", ".join(sorted(unknown_dependencies))
        )

    symbol_changes = tuple(changed_symbols)
    path_changes = tuple(changed_paths)
    interface_changed = any(
        (
            value.interface_changed
            if isinstance(value, ChangedASTSymbol)
            else bool(value.get("interface_changed", False))
            if isinstance(value, Mapping)
            else False
        )
        for value in symbol_changes
    )
    impact = index.impact(
        changed_symbols=symbol_changes,
        changed_paths=path_changes,
    )
    criteria = tuple(
        sorted(
            {
                _normalized_text(value)
                for value in acceptance_criteria
                if _normalized_text(value)
            }
        )
    )
    direct_targets = set(impact.changed_symbols) | set(impact.changed_paths)
    affected_targets = set(impact.affected_symbols) | set(impact.affected_paths)
    selection_reasons: dict[str, set[str]] = {
        check.check_id: set() for check in check_values
    }
    mandatory: set[str] = set()
    uncovered: set[str] = {
        *(f"unknown_changed_symbol:{value}" for value in impact.uncovered_symbols),
        *(f"unknown_changed_path:{value}" for value in impact.uncovered_paths),
    }

    applicable: dict[str, bool] = {}
    for check in check_values:
        matched_direct = direct_targets.intersection(check.targets)
        matched_transitive = affected_targets.intersection(check.targets)
        applicable[check.check_id] = not check.targets or bool(
            matched_transitive
        )
        if not check.targets:
            selection_reasons[check.check_id].add(
                "repository_wide_policy_check"
            )
        if matched_direct:
            selection_reasons[check.check_id].add(
                "changed_ast_or_path:" + ",".join(sorted(matched_direct))
            )
        transitive_only = matched_transitive - direct_targets
        if transitive_only:
            selection_reasons[check.check_id].add(
                "transitive_dependency_impact:"
                + ",".join(sorted(transitive_only))
            )
        if (
            policy.require_transitive_validation
            and check.check_id in impact.required_validation_ids
        ):
            selection_reasons[check.check_id].add(
                "dependency_graph_validation_target"
            )
            mandatory.add(check.check_id)
        if interface_changed and check.kind in {
            ImpactValidationKind.INTERFACE,
            ImpactValidationKind.CONTRACT,
        }:
            selection_reasons[check.check_id].add(
                "changed_ast_interface"
            )

    for kind in policy.required_kinds:
        candidates = tuple(
            check
            for check in check_values
            if check.kind is kind and applicable[check.check_id]
        )
        if not candidates:
            uncovered.add(f"missing_mandatory_{kind.value}_check")
            continue
        for check in candidates:
            mandatory.add(check.check_id)
            selection_reasons[check.check_id].add(
                f"repository_policy_requires:{kind.value}"
            )

    for technique in policy.required_techniques:
        candidates = tuple(
            check
            for check in check_values
            if check.technique is technique and applicable[check.check_id]
        )
        if not candidates:
            uncovered.add(
                f"missing_mandatory_{technique.value}_technique"
            )
            continue
        for check in candidates:
            mandatory.add(check.check_id)
            selection_reasons[check.check_id].add(
                f"repository_policy_requires_technique:{technique.value}"
            )

    for check_id in policy.required_check_ids:
        check = check_by_id.get(check_id)
        if check is None:
            uncovered.add(f"missing_policy_check:{check_id}")
        elif not applicable[check_id]:
            uncovered.add(f"policy_check_does_not_cover_impact:{check_id}")
        else:
            mandatory.add(check_id)
            selection_reasons[check_id].add(
                "repository_policy_requires_check"
            )

    criterion_bindings = dict(policy.acceptance_bindings)
    for check in check_values:
        for criterion in check.acceptance_criteria:
            criterion_bindings.setdefault(
                _normalized_text(criterion).casefold(), ()
            )
            criterion_bindings[_normalized_text(criterion).casefold()] = tuple(
                sorted(
                    {
                        *criterion_bindings[
                            _normalized_text(criterion).casefold()
                        ],
                        check.check_id,
                    }
                )
            )
    for criterion in criteria:
        bound_ids = tuple(
            criterion_bindings.get(criterion.casefold(), ())
        )
        covered = False
        for check_id in bound_ids:
            check = check_by_id.get(check_id)
            if check is None:
                uncovered.add(
                    f"acceptance_references_missing_check:{check_id}"
                )
            elif applicable[check_id]:
                covered = True
                mandatory.add(check_id)
                selection_reasons[check_id].add(
                    f"task_acceptance:{criterion}"
                )
        if policy.require_acceptance_coverage and not covered:
            uncovered.add(f"uncovered_acceptance:{criterion}")

    if policy.require_transitive_validation:
        for validation_id in impact.required_validation_ids:
            check = check_by_id.get(validation_id)
            if check is None:
                uncovered.add(
                    f"missing_dependency_validation:{validation_id}"
                )
            elif not applicable[validation_id]:
                uncovered.add(
                    f"dependency_validation_target_mismatch:{validation_id}"
                )

    selected = {
        check_id
        for check_id, reasons in selection_reasons.items()
        if reasons and applicable[check_id]
    }
    selected.update(mandatory)
    # Explicit check dependencies are authoritative reviewed edges.
    pending = list(selected)
    while pending:
        check_id = pending.pop()
        for dependency in check_by_id[check_id].depends_on:
            if dependency not in selected:
                selected.add(dependency)
                selection_reasons[dependency].add(
                    f"prerequisite_for:{check_id}"
                )
                pending.append(dependency)

    # Repository kind dependencies bind each selected check to every selected
    # prerequisite check of the required kind.  Branches such as interface and
    # unit, or integration and contract, remain independent and can run in
    # parallel after their common prerequisites.
    dependencies_by_id: dict[str, set[str]] = {
        check_id: set(check_by_id[check_id].depends_on)
        for check_id in selected
    }
    for check_id in sorted(selected):
        check = check_by_id[check_id]
        for dependency_kind in policy.kind_dependencies[check.kind]:
            candidates = {
                candidate_id
                for candidate_id in selected
                if check_by_id[candidate_id].kind is dependency_kind
            }
            if not candidates:
                uncovered.add(
                    f"missing_{dependency_kind.value}_prerequisite_for:{check_id}"
                )
            dependencies_by_id[check_id].update(candidates)

    nodes = tuple(
        ImpactValidationPlanNode(
            check=check,
            selected=check.check_id in selected,
            mandatory=check.check_id in mandatory,
            selection_reasons=tuple(selection_reasons[check.check_id]),
            skipped_reason=(
                ""
                if check.check_id in selected
                else "no_changed_symbol_dependency_acceptance_or_policy_match"
            ),
            depends_on=tuple(
                sorted(dependencies_by_id.get(check.check_id, ()))
            ),
        )
        for check in check_values
    )
    return ImpactSelectedValidationDAG(
        repository_tree_id=index.repository_tree_id,
        impact_index=index,
        impact=impact,
        policy=policy,
        acceptance_criteria=criteria,
        nodes=nodes,
        uncovered_impact=tuple(uncovered),
    )


@dataclass(frozen=True)
class ImpactValidationNodeReceipt:
    """Complete outcome for one selected or skipped catalog entry."""

    check_id: str
    kind: ImpactValidationKind
    command: str
    disposition: ValidationNodeDisposition
    reason: str
    mandatory: bool
    technique: ValidationTechnique = ValidationTechnique.STANDARD
    selection_reasons: tuple[str, ...] = ()
    skipped_reason: str = ""
    depends_on: tuple[str, ...] = ()
    blocked_by: tuple[str, ...] = ()
    returncode: int | None = None
    result_digest: str = ""
    cache_hit: bool = False
    duration_seconds: float = 0.0
    observed_seeded_defect_id: str = ""

    def __post_init__(self) -> None:
        for name in (
            "check_id",
            "command",
            "reason",
            "skipped_reason",
            "result_digest",
            "observed_seeded_defect_id",
        ):
            object.__setattr__(
                self, name, str(getattr(self, name) or "").strip()
            )
        if not self.check_id or not self.command or not self.reason:
            raise ValidationDAGError(
                "impact validation node receipt is incomplete"
            )
        object.__setattr__(self, "kind", ImpactValidationKind(self.kind))
        object.__setattr__(
            self, "technique", ValidationTechnique(self.technique)
        )
        object.__setattr__(
            self, "disposition", ValidationNodeDisposition(self.disposition)
        )
        object.__setattr__(self, "mandatory", bool(self.mandatory))
        for name in ("selection_reasons", "depends_on", "blocked_by"):
            object.__setattr__(
                self,
                name,
                tuple(
                    sorted(
                        {
                            str(value).strip()
                            for value in getattr(self, name)
                            if str(value).strip()
                        }
                    )
                ),
            )
        if self.returncode is not None:
            object.__setattr__(self, "returncode", int(self.returncode))
        duration = float(self.duration_seconds)
        if duration < 0:
            raise ValidationDAGError(
                "impact validation duration cannot be negative"
            )
        object.__setattr__(self, "duration_seconds", duration)
        executed = self.disposition in {
            ValidationNodeDisposition.SUCCEEDED,
            ValidationNodeDisposition.FAILED,
        }
        if executed and (
            self.returncode is None or not self.result_digest
        ):
            raise ValidationDAGError(
                "executed impact validation requires a bound result"
            )
        if (
            self.disposition is ValidationNodeDisposition.SUCCEEDED
            and self.returncode != 0
        ):
            raise ValidationDAGError(
                "successful impact validation has a failing return code"
            )
        if (
            self.disposition is ValidationNodeDisposition.FAILED
            and self.returncode == 0
        ):
            raise ValidationDAGError(
                "failed impact validation has a successful return code"
            )
        if not executed and (
            self.returncode is not None
            or self.result_digest
            or self.observed_seeded_defect_id
        ):
            raise ValidationDAGError(
                "unexecuted impact validation cannot contain a result"
            )
        if (
            self.disposition is ValidationNodeDisposition.OMITTED
            and not self.skipped_reason
        ):
            raise ValidationDAGError(
                "omitted impact validation requires its skipped reason"
            )
        if (
            self.disposition is not ValidationNodeDisposition.OMITTED
            and self.skipped_reason
        ):
            raise ValidationDAGError(
                "selected impact validation cannot have a skipped reason"
            )
        if (
            self.disposition is not ValidationNodeDisposition.BLOCKED
            and self.blocked_by
        ):
            raise ValidationDAGError(
                "only blocked impact validation can name blockers"
            )
        allowed_reasons = {
            ValidationNodeDisposition.SUCCEEDED: {"validation_passed"},
            ValidationNodeDisposition.FAILED: {"validation_failed"},
            ValidationNodeDisposition.OMITTED: {"not_selected"},
            ValidationNodeDisposition.BLOCKED: {
                "uncovered_validation_impact",
                "blocked_by_failed_dependency",
                "fail_fast_after_failure",
                "scheduler_state_inconsistent",
            },
        }
        if self.reason not in allowed_reasons.get(self.disposition, set()):
            raise ValidationDAGError(
                "impact validation reason does not match its disposition"
            )
        if (
            self.reason == "blocked_by_failed_dependency"
        ) != bool(self.blocked_by):
            raise ValidationDAGError(
                "impact validation blockers do not match the blocked reason"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "check_id": self.check_id,
            "kind": self.kind.value,
            "technique": self.technique.value,
            "command": self.command,
            "disposition": self.disposition.value,
            "reason": self.reason,
            "mandatory": self.mandatory,
            "selection_reasons": list(self.selection_reasons),
            "selection_reason": ";".join(self.selection_reasons),
            "skipped_reason": self.skipped_reason,
            "depends_on": list(self.depends_on),
            "blocked_by": list(self.blocked_by),
            "returncode": self.returncode,
            "result_digest": self.result_digest,
            "cache_hit": self.cache_hit,
            "duration_seconds": self.duration_seconds,
            "observed_seeded_defect_id": self.observed_seeded_defect_id,
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "ImpactValidationNodeReceipt":
        return cls(
            check_id=str(value.get("check_id") or ""),
            kind=ImpactValidationKind(str(value.get("kind") or "")),
            technique=ValidationTechnique(
                str(
                    value.get("technique")
                    or (
                        "contract"
                        if str(value.get("kind") or "") == "contract"
                        else "standard"
                    )
                )
            ),
            command=str(value.get("command") or ""),
            disposition=ValidationNodeDisposition(
                str(value.get("disposition") or "")
            ),
            reason=str(value.get("reason") or ""),
            mandatory=bool(value.get("mandatory", False)),
            selection_reasons=tuple(
                value.get("selection_reasons") or ()
            ),
            skipped_reason=str(value.get("skipped_reason") or ""),
            depends_on=tuple(value.get("depends_on") or ()),
            blocked_by=tuple(value.get("blocked_by") or ()),
            returncode=(
                int(value["returncode"])
                if value.get("returncode") is not None
                else None
            ),
            result_digest=str(value.get("result_digest") or ""),
            cache_hit=bool(value.get("cache_hit", False)),
            duration_seconds=float(value.get("duration_seconds") or 0.0),
            observed_seeded_defect_id=str(
                value.get("observed_seeded_defect_id") or ""
            ),
        )


@dataclass(frozen=True)
class ImpactValidationDAGReceipt:
    """Tree-bound execution receipt retaining the entire check population."""

    dag: ImpactSelectedValidationDAG
    nodes: tuple[ImpactValidationNodeReceipt, ...]
    passed: bool
    started_at: str
    finished_at: str
    time_to_first_useful_failure_seconds: float | None = None
    receipt_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", tuple(self.nodes))
        if not str(self.started_at or "").strip() or not str(
            self.finished_at or ""
        ).strip():
            raise ValidationDAGError(
                "impact validation receipt requires timestamps"
            )
        plan_ids = {node.check_id for node in self.dag.nodes}
        receipt_ids = {node.check_id for node in self.nodes}
        if plan_ids != receipt_ids or len(receipt_ids) != len(self.nodes):
            raise ValidationDAGError(
                "impact validation receipt population is incomplete"
            )
        plan_by_id = {node.check_id: node for node in self.dag.nodes}
        for node in self.nodes:
            planned = plan_by_id[node.check_id]
            if (
                node.kind is not planned.check.kind
                or node.technique is not planned.check.technique
                or node.command != planned.check.command
                or node.mandatory != planned.mandatory
                or node.depends_on != planned.depends_on
                or node.selection_reasons != planned.selection_reasons
            ):
                raise ValidationDAGError(
                    f"impact validation result is not bound to plan node "
                    f"{node.check_id!r}"
                )
            if planned.selected == (
                node.disposition is ValidationNodeDisposition.OMITTED
            ):
                raise ValidationDAGError(
                    "impact validation selection disposition mismatch"
                )
            if not planned.selected and (
                node.skipped_reason != planned.skipped_reason
            ):
                raise ValidationDAGError(
                    "impact validation skipped reason mismatch"
                )
        actual_passed = (
            self.dag.coverage_complete
            and all(
                node.disposition is ValidationNodeDisposition.SUCCEEDED
                for node in self.nodes
                if node.check_id
                in {value.check_id for value in self.dag.selected_nodes}
            )
        )
        failure_present = bool(self.dag.uncovered_impact) or any(
            node.disposition is ValidationNodeDisposition.FAILED
            for node in self.nodes
        )
        if self.time_to_first_useful_failure_seconds is not None:
            elapsed = float(self.time_to_first_useful_failure_seconds)
            if elapsed < 0:
                raise ValidationDAGError(
                    "time to first useful failure cannot be negative"
                )
            object.__setattr__(
                self, "time_to_first_useful_failure_seconds", elapsed
            )
        if failure_present != (
            self.time_to_first_useful_failure_seconds is not None
        ):
            raise ValidationDAGError(
                "time to first useful failure does not match receipt outcome"
            )
        if bool(self.passed) != actual_passed:
            raise ValidationDAGError(
                "impact validation receipt pass state is inconsistent"
            )
        object.__setattr__(self, "passed", actual_passed)
        claimed = str(self.receipt_id or "").strip()
        object.__setattr__(self, "receipt_id", "")
        actual = _sha256_bytes(
            _canonical_json(self._identity_payload()).encode("utf-8")
        )
        if claimed and claimed != actual:
            raise ValidationDAGError(
                "impact validation receipt identity mismatch"
            )
        object.__setattr__(self, "receipt_id", actual)

    @property
    def uncovered_impact(self) -> tuple[str, ...]:
        return self.dag.uncovered_impact

    def _identity_payload(self) -> dict[str, object]:
        return {
            "schema": IMPACT_SELECTED_VALIDATION_RECEIPT_SCHEMA,
            "dag": self.dag.to_dict(),
            "nodes": [node.to_dict() for node in self.nodes],
            "passed": self.passed,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "time_to_first_useful_failure_seconds": (
                self.time_to_first_useful_failure_seconds
            ),
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._identity_payload(), "receipt_id": self.receipt_id}

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "ImpactValidationDAGReceipt":
        schema = str(
            value.get("schema")
            or IMPACT_SELECTED_VALIDATION_RECEIPT_SCHEMA
        )
        if schema != IMPACT_SELECTED_VALIDATION_RECEIPT_SCHEMA:
            raise ValidationDAGError(
                f"unsupported impact validation receipt schema: {schema}"
            )
        dag_value = value.get("dag")
        if not isinstance(dag_value, Mapping):
            raise ValidationDAGError(
                "impact validation receipt is missing its DAG"
            )
        return cls(
            dag=ImpactSelectedValidationDAG.from_dict(dag_value),
            nodes=tuple(
                ImpactValidationNodeReceipt.from_dict(item)
                for item in value.get("nodes") or ()
            ),
            passed=bool(value.get("passed", False)),
            started_at=str(value.get("started_at") or ""),
            finished_at=str(value.get("finished_at") or ""),
            time_to_first_useful_failure_seconds=(
                float(value["time_to_first_useful_failure_seconds"])
                if value.get("time_to_first_useful_failure_seconds")
                is not None
                else None
            ),
            receipt_id=str(value.get("receipt_id") or ""),
        )


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
    graph_version: str = "impact-dependency-v3"
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
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(path: str) -> None:
            if path in visited:
                return
            if path in visiting:
                raise ValidationDAGError(
                    "impact dependency graph contains a cycle"
                )
            visiting.add(path)
            for dependency in normalized.get(path, ()):
                visit(dependency)
            visiting.remove(path)
            visited.add(path)

        for path in sorted(self.reverse_dependencies):
            visit(path)
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
            graph_version=str(payload.get("graph_version") or "impact-dependency-v3"),
            graph_id=str(payload.get("graph_id") or ""),
        )


def build_declared_validation_plan_graph(
    commands: Iterable[str | ValidationCommand],
    *,
    repository_tree_id: str,
    changed_paths: Iterable[str],
) -> tuple[tuple[ValidationCommand, ...], ImpactDependencyGraph]:
    """Bind a reviewed task validation plan to an accepted proposal.

    Todo boards predate repository-wide impact indexes, but their validation
    commands are still reviewed inputs and are already bound into the accepted
    :class:`ImplementationProposal`.  This helper gives every declared command
    a deterministic identity and builds a conservative, proposal-local impact
    graph:

    * every changed path is a graph root;
    * every explicit validation target depends on every changed path unless it
      is itself changed; and
    * commands without an explicit target cover every changed path.

    The result authorizes only the declared validation population.  It does not
    emit transitive-impact proof evidence or weaken ``run_validated`` when its
    caller requires a repository-wide impact graph.
    """

    tree_id = str(repository_tree_id or "").strip()
    if not tree_id:
        raise ValidationDAGError(
            "declared validation plan requires repository_tree_id"
        )
    changed = tuple(
        sorted(
            {
                path
                for value in changed_paths
                if (path := _normalize_impact_path(value))
            }
        )
    )
    specs = build_validation_commands(commands)
    if not specs:
        raise ValidationDAGError(
            "declared validation plan requires at least one command"
        )
    # Clean / already-satisfied residual candidates have no patch. Still
    # authorize the declared validation population by seeding graph roots
    # from command impact targets (or a stable synthetic root) so empty-patch
    # revalidation can complete instead of thrashing on ValidationDAGError.
    if not changed:
        seed_paths: set[str] = set()
        for spec in specs:
            for value in spec.impact_paths:
                path = _normalize_impact_path(value)
                if path:
                    seed_paths.add(path)
        if not seed_paths:
            seed_paths.add("__no_change_candidate__")
        changed = tuple(sorted(seed_paths))

    dependencies: dict[str, tuple[str, ...]] = {
        path: () for path in changed
    }
    validation_targets: dict[str, tuple[str, ...]] = {}
    bound_specs: list[ValidationCommand] = []
    seen_ids: set[str] = set()
    for spec in specs:
        validation_id = str(spec.validation_id or "").strip()
        if not validation_id:
            digest = _sha256_bytes(
                _canonical_json(
                    {
                        "command": spec.command,
                        "ordinal": spec.ordinal,
                        "stage": spec.stage.label,
                    }
                ).encode("utf-8")
            )
            validation_id = f"declared:{digest}"
        if validation_id in seen_ids:
            raise ValidationDAGError(
                "declared validation plan contains duplicate validation IDs"
            )
        seen_ids.add(validation_id)

        targets = tuple(
            sorted(
                {
                    path
                    for value in spec.impact_paths
                    if (path := _normalize_impact_path(value))
                }
            )
        )
        if not targets:
            targets = changed
        for target in targets:
            dependencies.setdefault(
                target,
                tuple(path for path in changed if path != target),
            )
        validation_targets[validation_id] = targets
        bound_specs.append(
            replace(
                spec,
                validation_id=validation_id,
                impact_paths=targets,
            )
        )

    graph = ImpactDependencyGraph(
        repository_tree_id=tree_id,
        dependencies=dependencies,
        validation_targets=validation_targets,
        graph_version="declared-validation-plan-v1",
    )
    return tuple(bound_specs), graph


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
    blocked_by_failed_node_ids: tuple[str, ...] = ()
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
        object.__setattr__(
            self,
            "blocked_by_failed_node_ids",
            tuple(
                sorted(
                    {
                        str(value or "").strip()
                        for value in self.blocked_by_failed_node_ids
                        if str(value or "").strip()
                    }
                )
            ),
        )
        if self.node_id in self.depends_on:
            raise ValidationDAGError("validation node cannot depend on itself")
        if self.node_id in self.blocked_by_failed_node_ids:
            raise ValidationDAGError(
                "validation node cannot be blocked by itself"
            )
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
        if (
            self.disposition is not ValidationNodeDisposition.BLOCKED
            and self.blocked_by_failed_node_ids
        ):
            raise ValidationDAGError(
                "only a blocked validation may name failed prerequisites"
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
            "blocked_by_failed_node_ids": self.blocked_by_failed_node_ids,
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
            blocked_by_failed_node_ids=tuple(
                payload.get("blocked_by_failed_node_ids") or ()
            ),
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
        if (
            graph is not None
            and self.affected_paths != graph.affected_paths(self.changed_paths)
        ):
            raise ValidationDAGError(
                "validation DAG affected paths do not match the graph closure"
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

        def ancestors(node_id: str) -> set[str]:
            result: set[str] = set()
            pending = list(by_id[node_id].depends_on)
            while pending:
                dependency = pending.pop()
                if dependency in result:
                    continue
                result.add(dependency)
                pending.extend(by_id[dependency].depends_on)
            return result

        for node in nodes:
            failed_prerequisites = node.blocked_by_failed_node_ids
            if any(
                failed_id not in by_id
                or by_id[failed_id].disposition
                is not ValidationNodeDisposition.FAILED
                for failed_id in failed_prerequisites
            ):
                raise ValidationDAGError(
                    "blocked validation names a non-failed prerequisite"
                )
            if not set(failed_prerequisites).issubset(ancestors(node.node_id)):
                raise ValidationDAGError(
                    "blocked validation failure is not a dependency ancestor"
                )
            expected_failed = tuple(
                sorted(
                    dependency
                    for dependency in ancestors(node.node_id)
                    if by_id[dependency].disposition
                    is ValidationNodeDisposition.FAILED
                )
            )
            if (
                node.reason == "blocked_by_failed_dependency"
                and failed_prerequisites != expected_failed
            ):
                raise ValidationDAGError(
                    "blocked validation does not identify every failed prerequisite"
                )
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
        if bool(self.uncovered_impact) != (not derived_coverage):
            raise ValidationDAGError(
                "validation DAG uncovered-impact verdict does not match coverage"
            )
        object.__setattr__(self, "uncovered_impact", not derived_coverage)
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

    @property
    def code_proof_authoritative(self) -> bool:
        """Validation cannot substitute for fresh authoritative code proofs.

        A complete passing DAG is an input to implementation-obligation
        derivation.  It is deliberately not evidence that those obligations
        were discharged, even when solver, kernel, or attestation validation
        stages ran successfully.
        """

        return False

    def evaluate_objective_completion(
        self,
        *,
        proposal_validation: Any,
        current_state: Any = "active",
        evidence: Sequence[Any] = (),
        tasks_complete: bool = False,
        coverage: Any = None,
        analyzer_health: Any = None,
        exhaustion_quorum: Any = None,
        required_exhaustive_receipts: int = 2,
        child_goals: Sequence[Any] = (),
        now: Any = None,
        freshness_seconds: float | None = None,
        clock_skew_seconds: float | None = None,
        analysis_inconclusive: bool = False,
        blocked_reason: str = "",
    ) -> Any:
        """Evaluate ASI-G101 through its closed current-tree proof gate."""

        return _evaluate_transitive_impact_objective_completion(
            self,
            proposal_validation=proposal_validation,
            current_state=current_state,
            evidence=evidence,
            tasks_complete=tasks_complete,
            coverage=coverage,
            analyzer_health=analyzer_health,
            exhaustion_quorum=exhaustion_quorum,
            required_exhaustive_receipts=required_exhaustive_receipts,
            child_goals=child_goals,
            now=now,
            freshness_seconds=freshness_seconds,
            clock_skew_seconds=clock_skew_seconds,
            analysis_inconclusive=analysis_inconclusive,
            blocked_reason=blocked_reason,
        )

    def strict_validation_completion_evidence(
        self,
    ) -> "StrictValidationDAGCompletionEvidence":
        """Project the scheduler-owned part of the ASI-G040 proof packet.

        The projection is deliberately narrower than parent completion.  It
        authenticates impact-test selection and the downstream
        semantic/proof/merge/freshness authority boundaries, but cannot turn a
        validation result into completion authority.  The ASI-G040 evaluator
        must combine it with proposal and code-proof producers plus separate
        fresh passing completion validations.
        """

        return StrictValidationDAGCompletionEvidence(validation_dag=self)

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
            "code_proof_authoritative": False,
            "completion_authoritative": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ValidationDAGReceipt":
        schema = str(payload.get("schema") or VALIDATION_DAG_RECEIPT_SCHEMA)
        if schema != VALIDATION_DAG_RECEIPT_SCHEMA:
            raise ValidationDAGError(f"unsupported validation DAG schema: {schema}")
        for field_name in (
            "proof_authoritative",
            "code_proof_authoritative",
            "completion_authoritative",
        ):
            if payload.get(field_name) not in (None, False):
                raise ValidationDAGError(
                    f"validation DAG cannot claim {field_name}"
                )
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


@dataclass(frozen=True)
class StrictValidationDAGCompletionEvidence:
    """Tamper-evident scheduler projection for the ASI-G040 parent gate.

    ASI-G101's adversarial witness is intentionally a *failed* validation: the
    seeded transitive consumer must detect the defect and every downstream
    authority gate must close.  This record makes that counter-example usable
    by the parent objective without allowing a caller-authored summary to
    replace the complete validation population, graph, dependency trace, or
    authority-gate records.
    """

    validation_dag: ValidationDAGReceipt
    evidence_id: str = ""

    def __post_init__(self) -> None:
        receipt = self.validation_dag
        if not isinstance(receipt, ValidationDAGReceipt):
            if not isinstance(receipt, Mapping):
                raise ValidationDAGError(
                    "strict validation evidence requires a validation DAG receipt"
                )
            receipt = ValidationDAGReceipt.from_dict(receipt)
        object.__setattr__(self, "validation_dag", receipt)

        transitive = receipt.transitive_evidence
        gate_by_name = {gate.gate: gate for gate in receipt.authority_gates}
        impact_nodes = tuple(
            node
            for node in receipt.nodes
            if node.validation_id in receipt.required_validation_ids
            and node.selected
            and node.mandatory
        )
        if (
            receipt.objective_id != TRANSITIVE_IMPACT_OBJECTIVE_ID
            or transitive is None
            or transitive.requirement_id != TRANSITIVE_IMPACT_REQUIREMENT_ID
            or receipt.proved_requirement_ids
            != (TRANSITIVE_IMPACT_REQUIREMENT_ID,)
            or receipt.passed
            or not receipt.coverage_complete
            or receipt.uncovered_impact
            or not receipt.required_validation_ids
            or len(impact_nodes) != len(receipt.required_validation_ids)
            or {node.validation_id for node in impact_nodes}
            != set(receipt.required_validation_ids)
            or any(
                node.disposition is ValidationNodeDisposition.OMITTED
                for node in impact_nodes
            )
            or set(gate_by_name) != set(REQUIRED_AUTHORITY_GATES)
            or any(
                gate.disposition
                is not ValidationAuthorityDisposition.BLOCKED
                for gate in gate_by_name.values()
            )
            or any(
                gate.depends_on != receipt.selected_node_ids
                for gate in gate_by_name.values()
            )
        ):
            raise ValidationDAGError(
                "validation DAG does not qualify for the strict validation "
                "parent completion projection"
            )

        claimed = str(self.evidence_id or "").strip()
        object.__setattr__(self, "evidence_id", "")
        actual = _sha256_bytes(
            _canonical_json(self._identity_payload()).encode("utf-8")
        )
        if claimed and claimed != actual:
            raise ValidationDAGError(
                "strict validation completion evidence identity mismatch"
            )
        object.__setattr__(self, "evidence_id", actual)

    @property
    def parent_objective_id(self) -> str:
        return STRICT_VALIDATION_PARENT_OBJECTIVE_ID

    @property
    def objective_id(self) -> str:
        """The parent objective consuming this scheduler projection."""

        return self.parent_objective_id

    @property
    def child_objective_id(self) -> str:
        return self.validation_dag.objective_id

    @property
    def requirement_id(self) -> str:
        return TRANSITIVE_IMPACT_REQUIREMENT_ID

    @property
    def repository_tree_id(self) -> str:
        return self.validation_dag.repository_tree_id

    @property
    def policy_id(self) -> str:
        return self.validation_dag.policy_id

    @property
    def validation_policy_id(self) -> str:
        return self.policy_id

    @property
    def operational_receipt_id(self) -> str:
        return self.validation_dag.receipt_id

    @property
    def gate_kinds(self) -> tuple[str, ...]:
        """Closed ASI-G040 gate vocabulary shared by all three producers."""

        return STRICT_VALIDATION_GATE_KINDS

    @property
    def scheduler_gate_kinds(self) -> tuple[str, ...]:
        return STRICT_VALIDATION_SCHEDULER_GATE_KINDS

    @property
    def qualifies(self) -> bool:
        """Construction has re-derived every scheduler-owned invariant."""

        return True

    @property
    def proved_requirement_ids(self) -> tuple[str, ...]:
        return (self.requirement_id,)

    @property
    def impact_test_node_ids(self) -> tuple[str, ...]:
        required = set(self.validation_dag.required_validation_ids)
        return tuple(
            sorted(
                node.node_id
                for node in self.validation_dag.nodes
                if node.selected
                and node.mandatory
                and node.validation_id in required
            )
        )

    @property
    def completion_authoritative(self) -> bool:
        """The parent evaluator, never this projection, owns completion."""

        return False

    def evaluate_parent_completion(self, **kwargs: Any) -> Any:
        """Delegate ASI-G040 lifecycle evaluation with this bound projection."""

        if "validation_projection" in kwargs:
            raise TypeError(
                "validation_projection is supplied by the scheduler evidence"
            )
        from ..planning.formal_plan_conformance import (
            evaluate_strict_validation_completion,
        )

        return evaluate_strict_validation_completion(
            validation_projection=self,
            **kwargs,
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "schema": STRICT_VALIDATION_DAG_COMPLETION_EVIDENCE_SCHEMA,
            "objective_id": self.objective_id,
            "parent_objective_id": self.parent_objective_id,
            "child_objective_id": self.child_objective_id,
            "requirement_id": self.requirement_id,
            "proved_requirement_ids": self.proved_requirement_ids,
            "repository_tree_id": self.repository_tree_id,
            "policy_id": self.policy_id,
            "validation_policy_id": self.validation_policy_id,
            "receipt_id": self.operational_receipt_id,
            "operational_receipt_id": self.operational_receipt_id,
            "gate_kinds": self.gate_kinds,
            "scheduler_gate_kinds": self.scheduler_gate_kinds,
            "impact_test_node_ids": self.impact_test_node_ids,
            "validation_dag": self.validation_dag.to_dict(),
            "qualifies": self.qualifies,
            "completion_authoritative": False,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._identity_payload(), "evidence_id": self.evidence_id}

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "StrictValidationDAGCompletionEvidence":
        required_fields = (
            "schema",
            "objective_id",
            "parent_objective_id",
            "child_objective_id",
            "requirement_id",
            "proved_requirement_ids",
            "repository_tree_id",
            "policy_id",
            "validation_policy_id",
            "receipt_id",
            "operational_receipt_id",
            "gate_kinds",
            "scheduler_gate_kinds",
            "impact_test_node_ids",
            "validation_dag",
            "qualifies",
            "completion_authoritative",
            "evidence_id",
        )
        missing = tuple(name for name in required_fields if name not in payload)
        if missing:
            raise ValidationDAGError(
                "strict validation completion evidence is incomplete: "
                + ", ".join(missing)
            )
        unknown = tuple(
            sorted(str(name) for name in payload if name not in required_fields)
        )
        if unknown:
            raise ValidationDAGError(
                "strict validation completion evidence has unknown fields: "
                + ", ".join(unknown)
            )
        schema = str(payload.get("schema") or "")
        if schema != STRICT_VALIDATION_DAG_COMPLETION_EVIDENCE_SCHEMA:
            raise ValidationDAGError(
                f"unsupported strict validation completion schema: {schema}"
            )
        dag_payload = payload.get("validation_dag")
        if not isinstance(dag_payload, Mapping):
            raise ValidationDAGError(
                "strict validation evidence is missing its validation DAG"
            )
        evidence = cls(
            validation_dag=ValidationDAGReceipt.from_dict(dag_payload),
            evidence_id=str(payload.get("evidence_id") or ""),
        )
        expected = evidence._identity_payload()
        for name in (
            "objective_id",
            "parent_objective_id",
            "child_objective_id",
            "requirement_id",
            "proved_requirement_ids",
            "repository_tree_id",
            "policy_id",
            "validation_policy_id",
            "receipt_id",
            "operational_receipt_id",
            "gate_kinds",
            "scheduler_gate_kinds",
            "impact_test_node_ids",
            "qualifies",
            "completion_authoritative",
        ):
            if name in payload and _json_safe(payload[name]) != _json_safe(
                expected[name]
            ):
                raise ValidationDAGError(
                    "strict validation completion projection is inconsistent"
                )
        return evidence


def _evaluate_transitive_impact_objective_completion(
    receipt: ValidationDAGReceipt,
    *,
    proposal_validation: Any,
    current_state: Any,
    evidence: Sequence[Any],
    tasks_complete: bool,
    coverage: Any,
    analyzer_health: Any,
    exhaustion_quorum: Any,
    required_exhaustive_receipts: int,
    child_goals: Sequence[Any],
    now: Any,
    freshness_seconds: float | None,
    clock_skew_seconds: float | None,
    analysis_inconclusive: bool,
    blocked_reason: str,
) -> Any:
    """Closed ASI-G101 bridge kept outside the receipt serializer."""

    from ..planning.formal_plan_conformance import (
        evaluate_transitive_impact_admission_closure,
    )
    from ..objectives.goal_completion import evaluate_goal_completion
    from .proposal_validation import ProposalValidationResult

    proposal = (
        proposal_validation
        if isinstance(proposal_validation, ProposalValidationResult)
        else ProposalValidationResult.from_dict(proposal_validation)
    )
    proposal.require_admitted_binding(
        repository_tree_id=receipt.repository_tree_id,
        objective_id=receipt.objective_id,
        receipt_id=receipt.proposal_receipt_id,
    )
    closure = evaluate_transitive_impact_admission_closure(
        proposal_validation=proposal,
        validation_dag=receipt,
    )
    operational_complete = bool(
        receipt.objective_id == TRANSITIVE_IMPACT_OBJECTIVE_ID
        and receipt.transitive_evidence is not None
        and receipt.coverage_complete
        and not receipt.passed
        and not receipt.uncovered_impact
        and receipt.proved_requirement_ids
        == (TRANSITIVE_IMPACT_REQUIREMENT_ID,)
        and not closure.admitted
        and "validation_dag_failed" in closure.reason_codes
    )

    def payload(value: Any) -> dict[str, Any]:
        if isinstance(value, Mapping):
            return dict(value)
        converter = getattr(value, "to_dict", None)
        if callable(converter):
            converted = converter()
            if isinstance(converted, Mapping):
                return dict(converted)
        return {}

    expected_criteria = {
        " ".join(item.lower().split())
        for item in TRANSITIVE_IMPACT_ACCEPTANCE_CRITERIA
    }
    coverage_projection = getattr(coverage, "completion_gate_evidence", None)
    canonical_coverage = callable(coverage_projection)
    if canonical_coverage:
        try:
            projected_coverage = coverage_projection(
                TRANSITIVE_IMPACT_OBJECTIVE_ID
            )
        except (TypeError, ValueError):
            projected_coverage = {}
        coverage_value = (
            dict(projected_coverage)
            if isinstance(projected_coverage, Mapping)
            else {}
        )
    else:
        coverage_value = payload(coverage)
    rows_value = coverage_value.get("criteria")
    rows = rows_value if isinstance(rows_value, list) else []

    def criterion_key(value: Any) -> str:
        if isinstance(value, Mapping):
            value = value.get(
                "criterion",
                value.get(
                    "acceptance_criterion",
                    value.get("acceptance", ""),
                ),
            )
        return " ".join(str(value or "").strip().lower().split())

    def populated(row: Mapping[str, Any], *names: str) -> bool:
        for name in names:
            value = row.get(name)
            if isinstance(value, str) and value.strip():
                return True
            if (
                isinstance(value, Sequence)
                and not isinstance(value, (str, bytes, bytearray))
                and any(str(item or "").strip() for item in value)
            ):
                return True
        return False

    submitted_validation_ids: dict[str, set[str]] = {}
    evidence_records: list[dict[str, Any]] = []
    for item in evidence:
        record = (
            item.to_dict()
            if hasattr(item, "to_dict") and callable(item.to_dict)
            else dict(item)
            if isinstance(item, Mapping)
            else {}
        )
        if isinstance(record.get("evidence"), Mapping):
            record = dict(record["evidence"])
        evidence_records.append(record)
        criterion = criterion_key(record)
        identity = str(
            record.get(
                "provenance_cid",
                record.get("receipt_cid", ""),
            )
            or ""
        ).strip()
        if criterion and identity:
            submitted_validation_ids.setdefault(criterion, set()).add(identity)

    def validation_bound(row: Mapping[str, Any]) -> bool:
        if "validation_receipt_ids" in row:
            raw_ids = row.get("validation_receipt_ids")
            if not (
                isinstance(raw_ids, Sequence)
                and not isinstance(raw_ids, (str, bytes, bytearray))
            ):
                return False
            receipt_ids = {
                str(item or "").strip()
                for item in raw_ids
                if str(item or "").strip()
            }
            return bool(
                receipt_ids
                and receipt_ids.intersection(
                    submitted_validation_ids.get(criterion_key(row), set())
                )
            )
        # Compatibility for the mapping-backed completion records introduced
        # before GoalCoverageMap became the canonical coverage producer.
        return populated(row, "validation")

    normalized_rows = [
        criterion_key(row) if isinstance(row, Mapping) else ""
        for row in rows
    ]
    canonical_coverage_complete = True
    if canonical_coverage:
        freshness = coverage_value.get("freshness")
        freshness = freshness if isinstance(freshness, Mapping) else {}
        coverage_binding = coverage_value.get("binding")
        coverage_binding = (
            coverage_binding
            if isinstance(coverage_binding, Mapping)
            else {}
        )
        canonical_coverage_complete = bool(
            coverage_value.get("verified") is True
            and coverage_value.get("repository_tree")
            == receipt.repository_tree_id
            and freshness.get("all_receipts_fresh") is True
            and coverage_binding.get("all_receipts_bound") is True
            and coverage_binding.get("repository_tree")
            == receipt.repository_tree_id
        )
    coverage_complete = bool(
        operational_complete
        and canonical_coverage_complete
        and len(normalized_rows) == len(expected_criteria)
        and len(normalized_rows) == len(set(normalized_rows))
        and set(normalized_rows) == expected_criteria
        and all(
            isinstance(row, Mapping)
            and populated(
                row,
                "implementation",
                "changed_files",
                "predicted_files",
                "ast_symbols",
                "interfaces",
            )
            and validation_bound(row)
            for row in rows
        )
    )

    bound_criteria: list[str] = []
    evidence_bound = len(evidence_records) == len(expected_criteria)
    for record in evidence_records:
        validation = record.get("validation_receipt")
        validation = validation if isinstance(validation, Mapping) else {}
        bound_criteria.append(criterion_key(record))
        evidence_bound = bool(
            evidence_bound
            and validation.get("requirement_id")
            == TRANSITIVE_IMPACT_REQUIREMENT_ID
            and validation.get("objective_id")
            == TRANSITIVE_IMPACT_OBJECTIVE_ID
            and validation.get("operational_receipt_id") == receipt.receipt_id
            and validation.get("validation_policy_id") == receipt.policy_id
            and validation.get("tree_id") == receipt.repository_tree_id
        )
    evidence_bound = bool(
        evidence_bound
        and len(bound_criteria) == len(set(bound_criteria))
        and set(bound_criteria) == expected_criteria
    )
    if not coverage_complete or not evidence_bound:
        reasons = coverage_value.get("reason_codes")
        reasons = list(reasons) if isinstance(reasons, (list, tuple)) else []
        if not operational_complete:
            reasons.append("active_operational_evidence_missing")
        if not coverage_complete:
            reasons.append(
                "coverage_missing_implementation_validation_binding"
            )
        if not evidence_bound:
            reasons.append("validation_not_bound_to_operational_witness")
        coverage_value = {
            **coverage_value,
            "verified": False,
            "reason_codes": list(dict.fromkeys(reasons)),
        }

    from ..analysis.analyzer_health import AnalyzerHealthReport
    from ..objectives.scan_receipts import ExhaustionQuorumResult

    typed_health = isinstance(analyzer_health, AnalyzerHealthReport)
    evaluated_quorum = isinstance(exhaustion_quorum, ExhaustionQuorumResult)
    quorum_value = payload(exhaustion_quorum)
    binding = quorum_value.get("binding")
    binding = binding if isinstance(binding, Mapping) else {}

    health_value = payload(analyzer_health)
    analyzer_version = str(
        health_value.get("analyzer_version") or ""
    ).strip()
    if typed_health and not analyzer_version:
        analyzer_version = str(binding.get("analyzer_version") or "").strip()
    if not (
        str(health_value.get("status") or "").lower() == "healthy"
        and health_value.get("healthy") is True
        and health_value.get("safe_for_completion_reasoning") is True
        and analyzer_version
        == TRANSITIVE_IMPACT_COMPLETION_ANALYZER_VERSION
    ):
        health_value = {
            **health_value,
            "healthy": False,
            "safe_for_completion_reasoning": False,
        }

    canonical_binding = {
        "tree_id": receipt.repository_tree_id,
        "objective_revision": TRANSITIVE_IMPACT_OBJECTIVE_REVISION,
        "analyzer_version": TRANSITIVE_IMPACT_COMPLETION_ANALYZER_VERSION,
        "configuration_revision": (
            TRANSITIVE_IMPACT_COMPLETION_CONFIGURATION_REVISION
        ),
    }
    artifact_binding = {
        **canonical_binding,
        "objective_id": TRANSITIVE_IMPACT_OBJECTIVE_ID,
        "validation_policy_id": receipt.policy_id,
        "operational_receipt_id": receipt.receipt_id,
    }
    required_binding = (
        canonical_binding if evaluated_quorum else artifact_binding
    )
    members_value = quorum_value.get("members")
    members = members_value if isinstance(members_value, list) else []
    member_ids = [
        str(member.get("member_id") or "")
        for member in members
        if isinstance(member, Mapping)
    ]
    member_receipts = [
        str(member.get("receipt_cid") or "")
        for member in members
        if isinstance(member, Mapping)
    ]
    channels = [
        str(member.get("evidence_channel") or "")
        for member in members
        if isinstance(member, Mapping)
    ]
    evaluated_members_complete = bool(
        evaluated_quorum
        and quorum_value.get("satisfied") is True
        and all(
            isinstance(member, Mapping)
            and (
                "exhaustive"
                in str(member.get("scan_mode") or "").strip().lower()
                or str(member.get("scan_mode") or "").strip().lower()
                == "audit"
            )
            for member in members
        )
    )
    quorum_complete = bool(
        quorum_value.get("required_members") == required_exhaustive_receipts
        and quorum_value.get("member_count") == len(members)
        and len(members) >= required_exhaustive_receipts
        and quorum_value.get("satisfied") is True
        and quorum_value.get("quorum_met") is True
        and all(
            binding.get(key) == value
            for key, value in required_binding.items()
        )
        and len(member_ids) == len(members) == len(set(member_ids))
        and len(member_receipts) == len(members) == len(set(member_receipts))
        and len(channels) == len(set(channels))
        and all(member_ids)
        and all(member_receipts)
        and all(channels)
        and all(
            isinstance(member, Mapping)
            and isinstance(member.get("binding"), Mapping)
            and all(
                member["binding"].get(key) == value
                for key, value in required_binding.items()
            )
            for member in members
        )
        and (
            evaluated_members_complete
            or all(
                isinstance(member, Mapping)
                and member.get("healthy") is True
                and member.get("safe_for_completion_reasoning") is True
                and str(member.get("scan_mode") or "").lower()
                == "exhaustive"
                for member in members
            )
        )
    )
    if not quorum_complete:
        quorum_value = {
            **quorum_value,
            "satisfied": False,
            "quorum_met": False,
        }

    values: dict[str, Any] = {
        "current_state": current_state,
        "acceptance_criteria": TRANSITIVE_IMPACT_ACCEPTANCE_CRITERIA,
        "evidence": evidence,
        "tasks_complete": tasks_complete,
        "repository_tree": receipt.repository_tree_id,
        "now": now,
        "analysis_inconclusive": analysis_inconclusive,
        "blocked_reason": blocked_reason,
        "coverage": coverage_value,
        "analyzer_health": health_value,
        "exhaustion_quorum": quorum_value,
        "child_goals": child_goals,
        "analysis_result": None,
        "require_completion_gate": True,
    }
    if freshness_seconds is not None:
        values["freshness_seconds"] = freshness_seconds
    if clock_skew_seconds is not None:
        values["clock_skew_seconds"] = clock_skew_seconds
    return evaluate_goal_completion(**values)


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
        hermetic_policy: (
            HermeticValidationPolicy | Mapping[str, Any] | None
        ) = None,
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
        self.hermetic_policy = (
            hermetic_policy
            if isinstance(hermetic_policy, HermeticValidationPolicy)
            else HermeticValidationPolicy.from_dict(hermetic_policy)
            if hermetic_policy is not None
            else None
        )

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
        hermetic_policy: HermeticValidationPolicy | None = None,
        cancellation_token: ValidationCancellationToken | None = None,
        _cache_lease_held: bool = False,
    ) -> dict[str, object]:
        if hermetic_policy is None:
            hermetic_policy = self.hermetic_policy
        if (
            hermetic_policy is not None
            and not _runner_supports_hermetic_validation(runner)
        ):
            capability_error = (
                "hermetic_validation_runner_capability_missing"
            )
            capability_reason = (
                "hermetic_runner_does_not_consume_runtime_context"
            )
        elif (
            hermetic_policy is not None
            and runner_requires_sealed_validation_python(runner)
        ):
            capability_error = (
                "hermetic_sealed_runner_composition_unsupported"
            )
            capability_reason = (
                "sealed_nested_python_runner_cannot_claim_hermetic_"
                "execution"
            )
        else:
            capability_error = ""
            capability_reason = ""
        if capability_error:
            now = utc_now()
            result: dict[str, object] = {
                "command": spec.command,
                "raw_command": spec.raw_command or spec.command,
                "started_at": now,
                "finished_at": now,
                "returncode": 75,
                "output": "",
                "error": capability_error,
                "reason": capability_reason,
                "infrastructure_failure": True,
                "outcome": (
                    ValidationOutcome.INFRASTRUCTURE_FAILURE.value
                ),
                "classification": (
                    ValidationOutcome.INFRASTRUCTURE_FAILURE.value
                ),
                "authoritative": False,
                "stable": False,
                "cache_hit": False,
                "stage": spec.stage.label,
                "resource_cost": spec.resource_cost,
                "ordinal": spec.ordinal,
                "validation_id": spec.validation_id,
            }
            result["validation_result_digest"] = (
                _validation_result_digest(result)
            )
            return result
        timeout = spec.timeout_seconds or self.default_timeout_seconds
        runtime_context: HermeticValidationRuntime | None = None
        effective_dependencies = dependency_state
        if hermetic_policy is not None:
            cancellation_id = (
                cancellation_token.cancellation_id
                if cancellation_token is not None
                else "validation:"
                + _sha256_bytes(
                    _canonical_json(
                        {
                            "tree": target_commit,
                            "validation_id": spec.validation_id,
                            "command": spec.command,
                            "policy_id": hermetic_policy.policy_id,
                        }
                    ).encode("utf-8")
                )
            )
            try:
                runtime_context = build_hermetic_validation_runtime(
                    command=spec.command,
                    workspace_path=workspace_path,
                    repository_tree_id=target_commit,
                    environment=environment,
                    timeout_seconds=timeout,
                    cancellation_id=cancellation_id,
                    resource_bounds=hermetic_policy.resource_bounds,
                )
            except Exception as exc:
                now = utc_now()
                return {
                    "command": spec.command,
                    "raw_command": spec.raw_command or spec.command,
                    "started_at": now,
                    "finished_at": now,
                    "returncode": 75,
                    "output": "",
                    "error": (
                        f"hermetic_runtime_invalid:{type(exc).__name__}:{exc}"
                    ),
                    "infrastructure_failure": True,
                    "outcome": ValidationOutcome.INFRASTRUCTURE_FAILURE.value,
                    "classification": (
                        ValidationOutcome.INFRASTRUCTURE_FAILURE.value
                    ),
                    "authoritative": False,
                    "stable": False,
                    "cache_hit": False,
                    "stage": spec.stage.label,
                    "resource_cost": spec.resource_cost,
                    "ordinal": spec.ordinal,
                    "validation_id": spec.validation_id,
                }
            effective_dependencies = {
                "candidate": _json_safe(dependency_state),
                "hermetic_policy_id": hermetic_policy.policy_id,
                "hermetic_runtime_id": runtime_context.runtime_id,
            }
        cache_key = build_validation_cache_key(
            target_commit=target_commit,
            command=spec,
            environment=environment,
            dependency_state=effective_dependencies,
            relevant_environment_keys=environment,
        )
        if spec.cacheable and self.cache is not None:
            cached = self.cache.get(cache_key)
            if (
                cached is not None
                and not _validation_python_launcher_receipt_matches_environment(
                    cached,
                    environment,
                )
            ):
                cached = None
            if (
                cached is not None
                and runtime_context is not None
                and not _hermetic_runtime_receipts_match(
                    cached,
                    runtime_context,
                    expected_attempts=hermetic_policy.stability_runs,
                )
            ):
                cached = None
            if (
                cached is not None
                and runtime_context is not None
                and str(
                    cached.get("validation_result_digest") or ""
                )
                != _validation_result_digest(
                    cached,
                    cache_key=cache_key,
                )
            ):
                cached = None
            if cached is not None:
                result = dict(cached)
                # Entries produced before validation-result@1 remain readable.
                # They cannot manufacture an exact digest without the omitted
                # output, so bind such a replay to its legacy cached payload.
                result.setdefault(
                    "validation_result_digest",
                    _validation_result_digest(
                        result,
                        cache_key=cache_key,
                        trust_stored_digest=True,
                    ),
                )
                result.update(
                    {
                        "command": spec.command,
                        "raw_command": spec.raw_command or spec.command,
                        "cache_hit": True,
                        "cache_key": cache_key.digest,
                        "stage": spec.stage.label,
                        "resource_cost": spec.resource_cost,
                        "ordinal": spec.ordinal,
                        "validation_id": spec.validation_id,
                    }
                )
                if hermetic_policy is not None:
                    result.setdefault(
                        "outcome", ValidationOutcome.PASSED.value
                    )
                    result.setdefault(
                        "classification", ValidationOutcome.PASSED.value
                    )
                    result.setdefault("authoritative", True)
                    result.setdefault("stable", True)
                    result.setdefault(
                        "attempt_count", hermetic_policy.stability_runs
                    )
                return result
            if hermetic_policy is not None:
                diagnostic = self.cache.get_diagnostic(
                    cache_key,
                    max_age_seconds=(
                        hermetic_policy.diagnostic_ttl_seconds
                    ),
                )
                if diagnostic is not None:
                    result = dict(diagnostic)
                    result.update(
                        {
                            "command": spec.command,
                            "raw_command": spec.raw_command or spec.command,
                            "cache_hit": False,
                            "diagnostic_cache_hit": True,
                            "cache_key": cache_key.digest,
                            "stage": spec.stage.label,
                            "resource_cost": spec.resource_cost,
                            "ordinal": spec.ordinal,
                            "validation_id": spec.validation_id,
                            "authoritative": False,
                        }
                    )
                    return result
            if not _cache_lease_held:
                # Acquire the process-shared key lease before resource
                # admission.  The recursive call repeats the exact validated
                # lookup under the lease, closing the lookup-to-execution race.
                with self.cache.single_flight(cache_key):
                    return self._execute(
                        spec,
                        workspace_path=workspace_path,
                        target_commit=target_commit,
                        environment=environment,
                        dependency_state=dependency_state,
                        runner=runner,
                        hermetic_policy=hermetic_policy,
                        cancellation_token=cancellation_token,
                        _cache_lease_held=True,
                    )

        decision, resource_lease = self._acquire_resource(
            spec, workspace_path=workspace_path
        )
        if resource_lease is None:
            now = utc_now()
            rejected = {
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
                "validation_id": spec.validation_id,
                "resource_admission": decision.to_dict(),
                "infrastructure_failure": True,
                "outcome": ValidationOutcome.INFRASTRUCTURE_FAILURE.value,
                "classification": (
                    ValidationOutcome.INFRASTRUCTURE_FAILURE.value
                ),
                "authoritative": False,
                "stable": False,
            }
            rejected["validation_result_digest"] = _validation_result_digest(
                rejected, cache_key=cache_key
            )
            return rejected

        try:
            attempts: list[dict[str, object]] = []
            attempt_total = (
                hermetic_policy.stability_runs
                if hermetic_policy is not None
                else 1
            )
            for attempt_number in range(1, attempt_total + 1):
                if (
                    cancellation_token is not None
                    and cancellation_token.is_set()
                ):
                    attempt = {
                        "returncode": 130,
                        "output": "",
                        "cancelled": True,
                        "error": cancellation_token.reason or "cancelled",
                    }
                else:
                    try:
                        runner_kwargs: dict[str, object] = {
                            "spec": spec,
                            "workspace_path": workspace_path,
                            "timeout_seconds": timeout,
                            "environment": environment,
                        }
                        if hermetic_policy is not None:
                            try:
                                signature = inspect.signature(runner)
                                accepts_extra = any(
                                    parameter.kind
                                    is inspect.Parameter.VAR_KEYWORD
                                    for parameter in signature.parameters.values()
                                )
                            except (TypeError, ValueError):
                                accepts_extra = True
                                signature = None
                            optional = {
                                "runtime_context": runtime_context,
                                "cancellation_token": cancellation_token,
                                "attempt_number": attempt_number,
                            }
                            for key, value in optional.items():
                                if (
                                    accepts_extra
                                    or signature is not None
                                    and key in signature.parameters
                                ):
                                    runner_kwargs[key] = value
                        raw_result = runner(**runner_kwargs)
                        attempt = dict(raw_result)
                        if (
                            hermetic_policy is not None
                            and runtime_context is not None
                        ):
                            observed_runtime_id = str(
                                attempt.get("runtime_id") or ""
                            )
                            observed_cancellation_id = str(
                                attempt.get("cancellation_id") or ""
                            )
                            if (
                                observed_runtime_id
                                != runtime_context.runtime_id
                                or observed_cancellation_id
                                != runtime_context.cancellation_id
                            ):
                                attempt.update(
                                    {
                                        "returncode": 75,
                                        "error": (
                                            "hermetic_runtime_receipt_"
                                            "mismatch"
                                        ),
                                        "reason": (
                                            "runner_runtime_receipt_missing_"
                                            "or_mismatched"
                                        ),
                                        "infrastructure_failure": True,
                                        "expected_runtime_id": (
                                            runtime_context.runtime_id
                                        ),
                                        "expected_cancellation_id": (
                                            runtime_context.cancellation_id
                                        ),
                                        "observed_runtime_id": (
                                            observed_runtime_id
                                        ),
                                        "observed_cancellation_id": (
                                            observed_cancellation_id
                                        ),
                                    }
                                )
                        if (
                            cancellation_token is not None
                            and cancellation_token.is_set()
                        ):
                            attempt = {
                                **attempt,
                                "returncode": 130,
                                "cancelled": True,
                                "error": (
                                    cancellation_token.reason or "cancelled"
                                ),
                            }
                    except subprocess.TimeoutExpired:
                        attempt = {
                            "returncode": 124,
                            "timed_out": True,
                            "output": "",
                        }
                    except Exception as exc:
                        attempt = {
                            "returncode": 75,
                            "output": "",
                            "infrastructure_failure": True,
                            "error": (
                                f"runner_failed:{type(exc).__name__}:{exc}"
                            ),
                        }
                attempt["returncode"] = int(attempt.get("returncode", 1))
                attempt["attempt_number"] = attempt_number
                attempt["diagnostic_signature"] = (
                    _attempt_diagnostic_signature(attempt)
                )
                attempts.append(attempt)
                if (
                    hermetic_policy is not None
                    and (
                        attempt.get("timed_out")
                        or attempt.get("cancelled")
                        or attempt.get("infrastructure_failure")
                        or attempt.get("inconclusive")
                    )
                ):
                    break
            result = dict(attempts[-1])
            if hermetic_policy is not None:
                outcome = classify_validation_attempts(attempts)
                result["attempts"] = [_json_safe(item) for item in attempts]
                result["attempt_count"] = len(attempts)
                result["outcome"] = outcome.value
                result["classification"] = outcome.value
                result["stable"] = outcome in {
                    ValidationOutcome.PASSED,
                    ValidationOutcome.DETERMINISTIC_FAILURE,
                }
                result["intermittent_pass"] = (
                    outcome is ValidationOutcome.FLAKY
                    and any(
                        int(item.get("returncode", 1)) == 0
                        for item in attempts
                    )
                )
                result["authoritative"] = (
                    outcome is ValidationOutcome.PASSED
                )
                observed_seed_ids: set[str] = set()
                for item in attempts:
                    singular = str(
                        item.get("seeded_defect_id") or ""
                    ).strip()
                    if singular:
                        observed_seed_ids.add(singular)
                    multiple = item.get("seeded_defect_ids") or ()
                    if isinstance(multiple, str):
                        multiple = (multiple,)
                    observed_seed_ids.update(
                        str(value).strip()
                        for value in multiple
                        if str(value).strip()
                    )
                result["seeded_defect_ids"] = sorted(observed_seed_ids)
                if len(observed_seed_ids) == 1:
                    result["seeded_defect_id"] = next(
                        iter(observed_seed_ids)
                    )
                result["returncode"] = {
                    ValidationOutcome.PASSED: 0,
                    ValidationOutcome.DETERMINISTIC_FAILURE: int(
                        attempts[0].get("returncode", 1)
                    )
                    or 1,
                    ValidationOutcome.FLAKY: 86,
                    ValidationOutcome.TIMEOUT: 124,
                    ValidationOutcome.INFRASTRUCTURE_FAILURE: 75,
                    ValidationOutcome.INCONCLUSIVE: 79,
                    ValidationOutcome.CANCELLED: 130,
                }[outcome]
                result["hermetic_runtime"] = (
                    runtime_context.to_dict() if runtime_context else None
                )
        finally:
            self.resource_scheduler.release(resource_lease)
        result.setdefault("command", spec.command)
        result.setdefault("raw_command", spec.raw_command or spec.command)
        result.setdefault("started_at", utc_now())
        result.setdefault("finished_at", utc_now())
        result["returncode"] = int(result.get("returncode", 1))
        # Emit an explicit boolean in every freshly executed command receipt.
        # Security-sensitive consumers must be able to distinguish a proven
        # non-timeout from a legacy/partial record that simply omitted the
        # field.
        result["timed_out"] = bool(result.get("timed_out", False))
        result["cache_hit"] = False
        result["cache_key"] = cache_key.digest
        result["stage"] = spec.stage.label
        result["resource_cost"] = spec.resource_cost
        result["ordinal"] = spec.ordinal
        result["validation_id"] = spec.validation_id
        if (
            result["returncode"] == 0
            and not _validation_python_launcher_receipt_matches_environment(
                result,
                environment,
            )
        ):
            result.update(
                {
                    "returncode": 75,
                    "error": (
                        "validation_environment_python_launcher_"
                        "receipt_mismatch"
                    ),
                    "reason": (
                        "sealed_validation_python_launcher_"
                        "receipt_mismatch"
                    ),
                    "infrastructure_failure": True,
                    "outcome": (
                        ValidationOutcome.INFRASTRUCTURE_FAILURE.value
                    ),
                    "classification": (
                        ValidationOutcome.INFRASTRUCTURE_FAILURE.value
                    ),
                    "authoritative": False,
                    "stable": False,
                    "intermittent_pass": False,
                }
            )
        result["validation_result_digest"] = _validation_result_digest(
            result, cache_key=cache_key
        )
        if (
            result.get("outcome")
            == ValidationOutcome.DETERMINISTIC_FAILURE.value
        ):
            result["diagnostic_id"] = result["validation_result_digest"]
        result["resource_admission"] = decision.to_dict()
        result["resource_lease"] = {
            "lease_id": resource_lease.lease_id,
            "resource_class": resource_lease.resource_class,
            "resource_pool": resource_lease.resource_pool,
            "child_limits": resource_lease.child_limits.to_dict(),
            "released": True,
        }
        if (
            spec.cacheable
            and self.cache is not None
            and result["returncode"] == 0
            and (
                hermetic_policy is None
                or result.get("authoritative") is True
            )
        ):
            # The cache quota bounds disk use, so retain the complete result.
            # Exact receipt reuse means a replay must not silently substitute
            # an output-less approximation for the result that actually ran.
            self.cache.put(cache_key, result)
        elif (
            spec.cacheable
            and self.cache is not None
            and hermetic_policy is not None
            and result.get("outcome")
            == ValidationOutcome.DETERMINISTIC_FAILURE.value
        ):
            self.cache.put_diagnostic(cache_key, result)
        return result

    @staticmethod
    def _first_failure(results: Iterable[Mapping[str, object]]) -> Mapping[str, object] | None:
        return next((result for result in results if int(result.get("returncode", 1)) != 0), None)

    def _run_parallel_stage(
        self,
        specs: Sequence[ValidationCommand],
        execute: Callable[[ValidationCommand], dict[str, object]],
    ) -> ValidationStageBatch:
        """Run a stage with worker and weighted-budget bounds.

        Submission is incremental.  Once a failure is observed no queued work
        is admitted, while already-running commands are drained safely.
        """

        stage_started = time.monotonic()
        pending = list(specs)
        active: dict[Future[dict[str, object]], tuple[ValidationCommand, int]] = {}
        completed: list[dict[str, object]] = []
        occupied = 0
        peak_parallelism = 0
        failed = False

        def measured_execute(spec: ValidationCommand) -> dict[str, object]:
            command_started = time.monotonic()
            result = execute(spec)
            result.setdefault(
                "execution_elapsed_seconds",
                max(0.0, time.monotonic() - command_started),
            )
            return result

        with ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="validation") as pool:
            while pending or active:
                admitted = False
                while pending and not failed and len(active) < self.max_workers:
                    spec = pending[0]
                    cost = min(self.resource_budget, max(1, int(spec.resource_cost)))
                    if active and occupied + cost > self.resource_budget:
                        break
                    pending.pop(0)
                    future = pool.submit(measured_execute, spec)
                    active[future] = (spec, cost)
                    occupied += cost
                    peak_parallelism = max(peak_parallelism, len(active))
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
        elapsed = max(0.0, time.monotonic() - stage_started)
        serial_work = sum(
            max(0.0, float(result.get("execution_elapsed_seconds", 0.0) or 0.0))
            for result in completed
        )
        return ValidationStageBatch(
            results=tuple(completed),
            elapsed_seconds=elapsed,
            serial_work_seconds=serial_work,
            peak_parallelism=peak_parallelism,
        )

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

    def run_impact_selected(
        self,
        checks: Iterable[ImpactValidationCheck | Mapping[str, Any]],
        *,
        workspace_path: Path | str,
        impact_index: CodeImpactIndex | Mapping[str, Any],
        changed_symbols: Iterable[
            str | ChangedASTSymbol | Mapping[str, Any]
        ] = (),
        changed_paths: Iterable[str] = (),
        acceptance_criteria: Iterable[str] = (),
        repository_policy: (
            RepositoryValidationPolicy | Mapping[str, Any] | None
        ) = None,
        proposal_validation: Any | None = None,
        target_tree_id: str = "",
        environment: Mapping[str, object] | None = None,
        dependency_state: (
            Mapping[str, object] | Sequence[object] | str | None
        ) = None,
        runner: ValidationRunner | None = None,
        hermetic_policy: (
            HermeticValidationPolicy | Mapping[str, Any] | None
        ) = None,
        cancellation_token: ValidationCancellationToken | None = None,
        seeded_defects: Iterable[
            SeededValidationDefect | Mapping[str, Any]
        ] = (),
        baseline_time_to_first_failure_seconds: Sequence[float] = (),
        optimized_time_to_first_failure_seconds: Sequence[float] = (),
    ) -> dict[str, Any]:
        """Plan and execute the AST- and policy-derived validation DAG.

        This method shares command execution, exact-input caching, resource
        admission, and the worker pool implementation with the legacy and
        proposal-bound scheduler entry points.  It adds explicit check
        dependencies rather than creating a competing subprocess scheduler.
        """

        index = (
            impact_index
            if isinstance(impact_index, CodeImpactIndex)
            else CodeImpactIndex.from_dict(impact_index)
        )
        check_values = tuple(
            value
            if isinstance(value, ImpactValidationCheck)
            else ImpactValidationCheck.from_dict(value)
            for value in checks
        )
        symbol_changes = tuple(changed_symbols)
        path_changes = tuple(changed_paths)
        hermetic = (
            hermetic_policy
            if isinstance(hermetic_policy, HermeticValidationPolicy)
            else HermeticValidationPolicy.from_dict(hermetic_policy)
            if hermetic_policy is not None
            else self.hermetic_policy
        )
        defect_values = tuple(
            value
            if isinstance(value, SeededValidationDefect)
            else SeededValidationDefect(
                defect_id=str(
                    value.get("defect_id")
                    or value.get("seeded_defect_id")
                    or ""
                ),
                path=str(
                    value.get("path")
                    or value.get("seeded_defect_path")
                    or ""
                ),
                expected_check_ids=tuple(
                    value.get("expected_check_ids") or ()
                ),
            )
            for value in seeded_defects
        )
        proposal_receipt: dict[str, Any] | None = None
        if proposal_validation is not None:
            from .proposal_validation import ProposalValidationResult

            proposal_result = (
                proposal_validation
                if isinstance(proposal_validation, ProposalValidationResult)
                else ProposalValidationResult.from_dict(proposal_validation)
            )
            if not proposal_result.accepted:
                bound = proposal_result.with_dispatch_outcome(
                    expensive_node_ids=tuple(
                        check.check_id for check in check_values
                    ),
                    expensive_checks_started=0,
                )
                blocked_nodes = [
                    {
                        "check_id": check.check_id,
                        "kind": check.kind.value,
                        "command": check.command,
                        "disposition": (
                            ValidationNodeDisposition.BLOCKED.value
                        ),
                        "reason": "proposal_gate_failed",
                    }
                    for check in check_values
                ]
                return {
                    "attempted": False,
                    "passed": False,
                    "returncode": 78,
                    "error": "proposal_validation_failed",
                    "reason": "proposal_gate_failed",
                    "results": [],
                    "nodes": blocked_nodes,
                    "impact_validation_dag": None,
                    "impact_validation_receipt": None,
                    "proposal_validation": bound.to_dict(),
                    "proposal_receipt": bound.receipt.to_dict(),
                    "proved_requirement_ids": (
                        bound.receipt.proved_requirement_ids
                    ),
                    "proof_authoritative": False,
                    "code_proof_authoritative": False,
                    "completion_authoritative": False,
                    "freshness_authoritative": False,
                    "authoritative": False,
                    "merge_eligible": False,
                    "selection_reasons": {},
                    "skipped_reasons": {},
                    "uncovered_impact": ["proposal_validation_failed"],
                    "time_to_first_useful_failure_seconds": 0.0,
                    "time_to_first_useful_failure_ms": 0.0,
                }
            if (
                proposal_result.proposal.repository_tree_id
                != index.repository_tree_id
            ):
                raise ValidationDAGError(
                    "code impact index is stale for the accepted proposal"
                )
            path_changes = tuple(
                sorted(
                    {
                        *path_changes,
                        *proposal_result.proposal.changed_paths,
                    }
                )
            )
            bound = proposal_result.with_dispatch_outcome(
                expensive_node_ids=(),
                expensive_checks_started=0,
            )
            proposal_receipt = bound.receipt.to_dict()

        requested_tree = str(target_tree_id or index.repository_tree_id).strip()
        if requested_tree != index.repository_tree_id:
            raise ValidationDAGError(
                "target tree does not match the code impact index"
            )
        effective_repository_policy = repository_policy
        if hermetic is not None:
            policy_value = (
                repository_policy
                if isinstance(repository_policy, RepositoryValidationPolicy)
                else RepositoryValidationPolicy.from_dict(repository_policy)
                if repository_policy is not None
                else RepositoryValidationPolicy()
            )
            policy_payload = policy_value.to_dict()
            policy_payload["policy_id"] = ""
            policy_payload["required_techniques"] = [
                value.value
                for value in tuple(
                    dict.fromkeys(
                        (
                            *policy_value.required_techniques,
                            *hermetic.required_techniques,
                        )
                    )
                )
            ]
            effective_repository_policy = (
                RepositoryValidationPolicy.from_dict(policy_payload)
            )
        plan = build_impact_selected_validation_dag(
            impact_index=index,
            checks=check_values,
            changed_symbols=symbol_changes,
            changed_paths=path_changes,
            acceptance_criteria=acceptance_criteria,
            repository_policy=effective_repository_policy,
        )
        workspace = Path(workspace_path)
        command_runner = runner or self.runner
        execution_environment = build_validation_environment(environment)
        if hermetic is None:
            execution_environment = validation_environment_for_runner(
                execution_environment,
                command_runner,
            )
        dependencies = (
            collect_dependency_state(
                workspace, changed_files=plan.impact.affected_paths
            )
            if dependency_state is None
            else dependency_state
        )
        started_at = utc_now()
        started_monotonic = time.monotonic()
        selected_nodes = {node.check_id: node for node in plan.selected_nodes}
        ordered_selected = tuple(
            sorted(selected_nodes.values(), key=lambda node: node.check_id)
        )
        specs = {
            node.check_id: node.check.command_spec(ordinal=ordinal)
            for ordinal, node in enumerate(ordered_selected)
        }
        raw_results: dict[str, dict[str, object]] = {}
        outcomes: dict[str, ImpactValidationNodeReceipt] = {}
        first_failure_id = ""
        first_failure_elapsed: float | None = None

        for node in plan.nodes:
            if not node.selected:
                outcomes[node.check_id] = ImpactValidationNodeReceipt(
                    check_id=node.check_id,
                    kind=node.check.kind,
                    technique=node.check.technique,
                    command=node.check.command,
                    disposition=ValidationNodeDisposition.OMITTED,
                    reason="not_selected",
                    mandatory=False,
                    selection_reasons=(),
                    skipped_reason=node.skipped_reason,
                    depends_on=(),
                )

        if not plan.coverage_complete:
            first_failure_elapsed = 0.0
            for node in ordered_selected:
                outcomes[node.check_id] = ImpactValidationNodeReceipt(
                    check_id=node.check_id,
                    kind=node.check.kind,
                    technique=node.check.technique,
                    command=node.check.command,
                    disposition=ValidationNodeDisposition.BLOCKED,
                    reason="uncovered_validation_impact",
                    mandatory=node.mandatory,
                    selection_reasons=node.selection_reasons,
                    depends_on=node.depends_on,
                    blocked_by=(),
                )
        else:
            pending = set(selected_nodes)
            active: dict[
                Future[dict[str, object]], tuple[str, int, float]
            ] = {}
            successful: set[str] = set()
            completed_ids: set[str] = set()
            failed_ids: set[str] = set()
            occupied = 0
            fail_fast = False

            def execute(check_id: str) -> dict[str, object]:
                return self._execute(
                    specs[check_id],
                    workspace_path=workspace,
                    target_commit=index.repository_tree_id,
                    environment=execution_environment,
                    dependency_state={
                        "candidate": _json_safe(dependencies),
                        "impact_index_id": index.index_id,
                        "policy_id": plan.policy.policy_id,
                    },
                    runner=command_runner,
                    hermetic_policy=hermetic,
                    cancellation_token=cancellation_token,
                )

            def failed_ancestors(check_id: str) -> tuple[str, ...]:
                found: set[str] = set()
                queue = list(selected_nodes[check_id].depends_on)
                visited: set[str] = set()
                while queue:
                    dependency = queue.pop()
                    if dependency in visited:
                        continue
                    visited.add(dependency)
                    if dependency in failed_ids:
                        found.add(dependency)
                    queue.extend(selected_nodes[dependency].depends_on)
                return tuple(sorted(found))

            with ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix="impact-validation",
            ) as pool:
                while pending or active:
                    # A failed prerequisite blocks descendants.  Once a useful
                    # failure is known, unrelated queued work is also retained
                    # as a fail-fast receipt rather than silently disappearing.
                    for check_id in sorted(tuple(pending)):
                        blocked_by = failed_ancestors(check_id)
                        if (
                            blocked_by
                            and not (
                                hermetic is not None
                                and hermetic.complete_selected_dag
                            )
                        ) or (
                            fail_fast
                            and not (
                                hermetic is not None
                                and hermetic.complete_selected_dag
                            )
                        ):
                            node = selected_nodes[check_id]
                            outcomes[check_id] = ImpactValidationNodeReceipt(
                                check_id=check_id,
                                kind=node.check.kind,
                                technique=node.check.technique,
                                command=node.check.command,
                                disposition=ValidationNodeDisposition.BLOCKED,
                                reason=(
                                    "blocked_by_failed_dependency"
                                    if blocked_by
                                    else "fail_fast_after_failure"
                                ),
                                mandatory=node.mandatory,
                                selection_reasons=node.selection_reasons,
                                depends_on=node.depends_on,
                                blocked_by=blocked_by,
                            )
                            pending.remove(check_id)

                    ready = [
                        check_id
                        for check_id in sorted(pending)
                        if set(selected_nodes[check_id].depends_on).issubset(
                            completed_ids
                            if (
                                hermetic is not None
                                and hermetic.complete_selected_dag
                            )
                            else successful
                        )
                    ]
                    for check_id in ready:
                        if len(active) >= self.max_workers:
                            break
                        spec = specs[check_id]
                        cost = min(
                            self.resource_budget,
                            max(1, int(spec.resource_cost)),
                        )
                        if active and occupied + cost > self.resource_budget:
                            continue
                        pending.remove(check_id)
                        submitted = time.monotonic()
                        future = pool.submit(execute, check_id)
                        active[future] = (check_id, cost, submitted)
                        occupied += cost

                    if not active:
                        if pending:
                            # The plan constructor already rejects cycles.  A
                            # residual node here means an inconsistent runtime
                            # state and must fail closed with a complete row.
                            for check_id in sorted(tuple(pending)):
                                node = selected_nodes[check_id]
                                outcomes[check_id] = (
                                    ImpactValidationNodeReceipt(
                                        check_id=check_id,
                                        kind=node.check.kind,
                                        technique=node.check.technique,
                                        command=node.check.command,
                                        disposition=(
                                            ValidationNodeDisposition.BLOCKED
                                        ),
                                        reason=(
                                            "scheduler_state_inconsistent"
                                        ),
                                        mandatory=node.mandatory,
                                        selection_reasons=(
                                            node.selection_reasons
                                        ),
                                        depends_on=node.depends_on,
                                    )
                                )
                                pending.remove(check_id)
                            first_failure_elapsed = (
                                first_failure_elapsed
                                if first_failure_elapsed is not None
                                else time.monotonic() - started_monotonic
                            )
                        break

                    done, _ = wait(tuple(active), return_when=FIRST_COMPLETED)
                    for future in done:
                        check_id, cost, submitted = active.pop(future)
                        occupied -= cost
                        result = future.result()
                        raw_results[check_id] = result
                        completed_ids.add(check_id)
                        returncode = int(result.get("returncode", 1))
                        node = selected_nodes[check_id]
                        if returncode == 0:
                            successful.add(check_id)
                            disposition = ValidationNodeDisposition.SUCCEEDED
                        else:
                            failed_ids.add(check_id)
                            disposition = ValidationNodeDisposition.FAILED
                            fail_fast = not (
                                hermetic is not None
                                and hermetic.complete_selected_dag
                            )
                            if first_failure_elapsed is None:
                                first_failure_elapsed = (
                                    time.monotonic() - started_monotonic
                                )
                                first_failure_id = check_id
                        outcomes[check_id] = ImpactValidationNodeReceipt(
                            check_id=check_id,
                            kind=node.check.kind,
                            technique=node.check.technique,
                            command=node.check.command,
                            disposition=disposition,
                            reason=(
                                "validation_passed"
                                if returncode == 0
                                else "validation_failed"
                            ),
                            mandatory=node.mandatory,
                            selection_reasons=node.selection_reasons,
                            depends_on=node.depends_on,
                            returncode=returncode,
                            result_digest=_validation_result_digest(
                                result, trust_stored_digest=True
                            ),
                            cache_hit=bool(result.get("cache_hit", False)),
                            duration_seconds=max(
                                0.0, time.monotonic() - submitted
                            ),
                            observed_seeded_defect_id=str(
                                result.get("seeded_defect_id") or ""
                            ),
                        )

        finished_at = utc_now()
        receipt_nodes = tuple(
            outcomes[node.check_id] for node in plan.nodes
        )
        passed = plan.coverage_complete and all(
            node.disposition is ValidationNodeDisposition.SUCCEEDED
            for node in receipt_nodes
            if node.check_id in selected_nodes
        )
        receipt = ImpactValidationDAGReceipt(
            dag=plan,
            nodes=receipt_nodes,
            passed=passed,
            started_at=started_at,
            finished_at=finished_at,
            time_to_first_useful_failure_seconds=first_failure_elapsed,
        )
        results = [
            raw_results[node.check_id]
            for node in ordered_selected
            if node.check_id in raw_results
        ]
        failed_result = (
            raw_results.get(first_failure_id) if first_failure_id else None
        )
        report: dict[str, Any] = {
            "attempted": bool(results),
            "passed": passed,
            "returncode": (
                0
                if passed
                else int(failed_result.get("returncode", 1))
                if failed_result is not None
                else 78
            ),
            "error": (
                ""
                if passed
                else "uncovered_validation_impact"
                if not plan.coverage_complete
                else "validation_failed"
            ),
            "results": results,
            "nodes": [node.to_dict() for node in receipt_nodes],
            "impact_validation_dag": plan.to_dict(),
            "impact_validation_receipt": receipt.to_dict(),
            "selection_reasons": {
                node.check_id: list(node.selection_reasons)
                for node in plan.nodes
                if node.selected
            },
            "skipped_reasons": {
                node.check_id: node.skipped_reason
                for node in plan.nodes
                if not node.selected
            },
            "uncovered_impact": list(plan.uncovered_impact),
            "affected_symbols": list(plan.impact.affected_symbols),
            "affected_paths": list(plan.impact.affected_paths),
            "required_validation_ids": list(
                plan.impact.required_validation_ids
            ),
            "time_to_first_useful_failure_seconds": first_failure_elapsed,
            "time_to_first_useful_failure_ms": (
                first_failure_elapsed * 1000.0
                if first_failure_elapsed is not None
                else None
            ),
            "first_useful_failure_check_id": first_failure_id,
            "target_tree_id": index.repository_tree_id,
            "impact_index_id": index.index_id,
            "policy_id": plan.policy.policy_id,
            "cache_hits": sum(
                1 for result in results if result.get("cache_hit") is True
            ),
            "cache_misses": sum(
                1 for result in results if result.get("cache_hit") is not True
            ),
            "max_workers": self.max_workers,
            "resource_budget": self.resource_budget,
            "hermetic": hermetic is not None,
            "hermetic_policy": (
                hermetic.to_dict() if hermetic is not None else None
            ),
            "outcome_counts": {
                outcome.value: sum(
                    str(result.get("outcome") or "")
                    == outcome.value
                    for result in results
                )
                for outcome in ValidationOutcome
            },
        }
        observed_by_check: dict[str, set[str]] = {}
        for check_id, result in raw_results.items():
            observed = {
                str(result.get("seeded_defect_id") or "").strip()
            }
            raw_observed = result.get("seeded_defect_ids") or ()
            if isinstance(raw_observed, str):
                raw_observed = (raw_observed,)
            observed.update(
                str(value).strip() for value in raw_observed
            )
            observed_by_check[check_id] = {
                value for value in observed if value
            }
        defect_rows: list[dict[str, object]] = []
        escaped: list[str] = []
        for defect in defect_values:
            seeded_impact = plan.impact_index.impact(
                changed_paths=(defect.path,)
            )
            eligible_checks = (
                set(defect.expected_check_ids)
                if defect.expected_check_ids
                else set(observed_by_check)
            )
            transitive_chains: dict[str, list[str]] = {}
            for check_id in sorted(eligible_checks):
                planned_node = selected_nodes.get(check_id)
                if planned_node is None:
                    continue
                for target in planned_node.check.targets:
                    chain = seeded_impact.dependency_chains.get(target, ())
                    if len(chain) >= 3:
                        transitive_chains[check_id] = list(chain)
                        break
                    normalized_target = _normalize_impact_path(target)
                    path_chain = seeded_impact.dependency_chains.get(
                        normalized_target, ()
                    )
                    if len(path_chain) >= 3:
                        transitive_chains[check_id] = list(path_chain)
                        break
            observing_checks = tuple(
                sorted(
                    check_id
                    for check_id, observed in observed_by_check.items()
                    if check_id in eligible_checks
                    and defect.defect_id in observed
                    and int(
                        raw_results[check_id].get("returncode", 1)
                    )
                    != 0
                )
            )
            detected = bool(observing_checks)
            if not detected:
                escaped.append(defect.defect_id)
            defect_rows.append(
                {
                    **defect.to_dict(),
                    "detected": detected,
                    "transitive": bool(transitive_chains),
                    "transitive_impact_chains": transitive_chains,
                    "observing_check_ids": list(observing_checks),
                }
            )
        report["seeded_defects"] = defect_rows
        report["seeded_defect_summary"] = {
            "seeded_count": len(defect_rows),
            "detected_count": len(defect_rows) - len(escaped),
            "escaped_count": len(escaped),
            "zero_escaped": not escaped,
        }
        report["escaped_seeded_defect_ids"] = escaped
        if escaped:
            report["passed"] = False
            report["returncode"] = 87
            report["error"] = "seeded_defect_escaped"
        baseline_samples = tuple(
            float(value)
            for value in baseline_time_to_first_failure_seconds
        )
        optimized_samples = tuple(
            float(value)
            for value in optimized_time_to_first_failure_seconds
        )
        if baseline_samples:
            if not optimized_samples and first_failure_elapsed is not None:
                optimized_samples = (first_failure_elapsed,)
            if optimized_samples:
                report["time_to_first_failure_benchmark"] = (
                    validation_benchmark(
                        baseline_seconds=baseline_samples,
                        optimized_seconds=optimized_samples,
                        minimum_reduction=(
                            hermetic.minimum_time_to_failure_reduction
                            if hermetic is not None
                            else 0.30
                        ),
                    )
                )
        for non_authority_field in (
            "proof_authoritative",
            "code_proof_authoritative",
            "completion_authoritative",
            "freshness_authoritative",
            "authoritative",
            "merge_eligible",
        ):
            report[non_authority_field] = False
        if proposal_receipt is not None:
            report["proposal_receipt"] = proposal_receipt
        if failed_result is not None:
            report["failed_command"] = str(
                failed_result.get("command") or ""
            )
        return report

    def run_hermetic_impact_selected(
        self,
        checks: Iterable[ImpactValidationCheck | Mapping[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute the complete impact DAG under the strict v2 policy."""

        policy = kwargs.pop("hermetic_policy", None)
        if policy is None:
            policy = HermeticValidationPolicy()
        return self.run_impact_selected(
            checks,
            hermetic_policy=policy,
            **kwargs,
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
                "code_proof_authoritative": False,
                "completion_authoritative": False,
                "freshness_authoritative": False,
                "authoritative": False,
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
                        "kind": "strict-validation-dag-policy@3",
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
                "code_proof_authoritative": False,
                "completion_authoritative": False,
                "freshness_authoritative": False,
                "authoritative": False,
                "merge_eligible": False,
                "impact_graph": None,
                "affected_paths": list(changed),
            }
        affected = graph.affected_paths(changed) if graph is not None else changed
        graph_id = graph.graph_id if graph is not None else "no-impact-graph"
        policy_id = str(validation_policy_id or "").strip() or _sha256_bytes(
            _canonical_json(
                {
                    "kind": "strict-validation-dag-policy@3",
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
        command_runner = runner or self.runner
        execution_environment = build_validation_environment(environment)
        if self.hermetic_policy is None:
            execution_environment = validation_environment_for_runner(
                execution_environment,
                command_runner,
            )
        results: list[dict[str, object]] = []
        stages: list[dict[str, object]] = []
        stage_benchmarks: list[ValidationStageBatch] = []
        failed: Mapping[str, object] | None = None
        validation_started_monotonic = time.monotonic()
        first_failure_elapsed: float | None = (
            0.0 if not coverage_complete else None
        )

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
                stage_batch = self._run_parallel_stage(stage_specs, execute)
                stage_benchmarks.append(stage_batch)
                stage_results = list(stage_batch.results)
                results.extend(stage_results)
                failed = self._first_failure(stage_results)
                if failed is not None and first_failure_elapsed is None:
                    first_failure_elapsed = (
                        time.monotonic() - validation_started_monotonic
                    )
                stages.append(
                    {
                        "stage": stage.label,
                        "started_at": stage_started,
                        "finished_at": utc_now(),
                        "planned_count": len(stage_specs),
                        "executed_count": len(stage_results),
                        "passed": failed is None,
                        "throughput": stage_batch.to_dict(),
                    }
                )
                if failed is not None:
                    break
        results.sort(key=lambda result: int(result.get("ordinal", len(specs))))
        validation_elapsed = max(
            0.0, time.monotonic() - validation_started_monotonic
        )
        validation_serial_work = sum(
            batch.serial_work_seconds for batch in stage_benchmarks
        )
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
            "throughput": {
                "schema": VALIDATION_THROUGHPUT_SCHEMA,
                "lane": VALIDATION_THROUGHPUT_LANE,
                "elapsed_seconds": validation_elapsed,
                "serial_work_seconds": validation_serial_work,
                "peak_parallelism": max(
                    (batch.peak_parallelism for batch in stage_benchmarks),
                    default=0,
                ),
                "planned_count": len(selected_specs),
                "completed_count": len(results),
                "accepted_count": sum(
                    int(result.get("returncode", 1)) == 0 for result in results
                ),
                "throughput_per_second": (
                    len(results) / validation_elapsed
                    if validation_elapsed > 0
                    else 0.0
                ),
                "parallel_speedup": (
                    validation_serial_work / validation_elapsed
                    if validation_elapsed > 0
                    else 0.0
                ),
            },
            "time_to_first_useful_failure_seconds": first_failure_elapsed,
            "time_to_first_useful_failure_ms": (
                first_failure_elapsed * 1000.0
                if first_failure_elapsed is not None
                else None
            ),
        }
        if failed is not None:
            report["failed_command"] = str(failed.get("command") or "")

        results_by_ordinal = {
            int(result.get("ordinal", -1)): result for result in results
        }
        failed_node_ids = tuple(
            sorted(
                self._validation_node_id(spec)
                for spec in effective_specs
                if (
                    (result := results_by_ordinal.get(spec.ordinal)) is not None
                    and int(result.get("returncode", 1)) != 0
                )
            )
        )

        def dependency_ancestors(node_id: str) -> set[str]:
            ancestors: set[str] = set()
            pending = list(dependency_ids.get(node_id, ()))
            while pending:
                dependency_id = pending.pop()
                if dependency_id in ancestors:
                    continue
                ancestors.add(dependency_id)
                pending.extend(dependency_ids.get(dependency_id, ()))
            return ancestors

        records: list[ValidationDAGNodeRecord] = []
        for spec in effective_specs:
            decision = decision_by_ordinal[spec.ordinal]
            result = results_by_ordinal.get(spec.ordinal)
            if result is not None:
                returncode = int(result.get("returncode", 1))
                result_digest = _validation_result_digest(
                    result, trust_stored_digest=True
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
                if not coverage_complete:
                    reason = "impact_coverage_incomplete"
                else:
                    node_id = self._validation_node_id(spec)
                    failed_ancestors = tuple(
                        failed_id
                        for failed_id in failed_node_ids
                        if failed_id in dependency_ancestors(node_id)
                    )
                    reason = (
                        "blocked_by_failed_dependency"
                        if failed_ancestors
                        else "fail_fast_after_stage_failure"
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
                    blocked_by_failed_node_ids=(
                        tuple(
                            failed_id
                            for failed_id in failed_node_ids
                            if failed_id
                            in dependency_ancestors(node_id)
                        )
                        if (
                            disposition is ValidationNodeDisposition.BLOCKED
                            and reason == "blocked_by_failed_dependency"
                        )
                        else ()
                    ),
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
        report["code_proof_authoritative"] = False
        report["completion_authoritative"] = False
        report["merge_eligible"] = False
        report["freshness_authoritative"] = False
        report["authoritative"] = False
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
        selected = tuple(selection.selected)
        command_runner = runner or self.runner
        execution_environment = build_validation_environment(environment)
        if self.hermetic_policy is None:
            execution_environment = validation_environment_for_runner(
                execution_environment,
                command_runner,
            )
        results: list[dict[str, object]] = []
        stages: list[dict[str, object]] = []
        stage_benchmarks: list[ValidationStageBatch] = []
        failed: Mapping[str, object] | None = None
        validation_started_monotonic = time.monotonic()
        first_failure_elapsed: float | None = None

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
            stage_batch = self._run_parallel_stage(stage_specs, execute)
            stage_benchmarks.append(stage_batch)
            stage_results = list(stage_batch.results)
            results.extend(stage_results)
            failed = self._first_failure(stage_results)
            if failed is not None and first_failure_elapsed is None:
                first_failure_elapsed = (
                    time.monotonic() - validation_started_monotonic
                )
            stages.append(
                {
                    "stage": stage.label,
                    "started_at": stage_started,
                    "finished_at": utc_now(),
                    "planned_count": len(stage_specs),
                    "executed_count": len(stage_results),
                    "passed": failed is None,
                    "throughput": stage_batch.to_dict(),
                }
            )
            if failed is not None:
                break

        # Parallel completion order is nondeterministic; reports are not.
        results.sort(key=lambda result: int(result.get("ordinal", len(specs))))

        cache_hits = sum(1 for result in results if result.get("cache_hit") is True)
        validation_elapsed = max(
            0.0, time.monotonic() - validation_started_monotonic
        )
        validation_serial_work = sum(
            batch.serial_work_seconds for batch in stage_benchmarks
        )
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
            "throughput": {
                "schema": VALIDATION_THROUGHPUT_SCHEMA,
                "lane": VALIDATION_THROUGHPUT_LANE,
                "elapsed_seconds": validation_elapsed,
                "serial_work_seconds": validation_serial_work,
                "peak_parallelism": max(
                    (batch.peak_parallelism for batch in stage_benchmarks),
                    default=0,
                ),
                "planned_count": len(selected),
                "completed_count": len(results),
                "accepted_count": sum(
                    int(result.get("returncode", 1)) == 0 for result in results
                ),
                "throughput_per_second": (
                    len(results) / validation_elapsed
                    if validation_elapsed > 0
                    else 0.0
                ),
                "parallel_speedup": (
                    validation_serial_work / validation_elapsed
                    if validation_elapsed > 0
                    else 0.0
                ),
            },
            "time_to_first_useful_failure_seconds": first_failure_elapsed,
            "time_to_first_useful_failure_ms": (
                first_failure_elapsed * 1000.0
                if first_failure_elapsed is not None
                else None
            ),
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
        from ..proof.proof_scheduler import ProofScheduler

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
        "hermetic_policy",
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
        "hermetic_policy",
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
        "hermetic_policy",
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


def schedule_impact_selected_validations(
    checks: Iterable[ImpactValidationCheck | Mapping[str, Any]],
    *,
    workspace_path: Path | str,
    impact_index: CodeImpactIndex | Mapping[str, Any],
    **kwargs: object,
) -> dict[str, Any]:
    """Convenience wrapper for the AST- and policy-derived validation DAG."""

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
        "hermetic_policy",
    }
    scheduler_kwargs = {
        key: kwargs.pop(key) for key in tuple(kwargs) if key in scheduler_keys
    }
    return ValidationScheduler(**scheduler_kwargs).run_impact_selected(
        checks,
        workspace_path=workspace_path,
        impact_index=impact_index,
        **kwargs,
    )


# Compatibility spelling used by integrations that foreground proof work.
schedule_proof_validations = schedule_staged_validations
schedule_validation_dag = schedule_validated_proposal
schedule_impact_validation_dag = schedule_impact_selected_validations

# Descriptive compatibility names for policy and API clients.
ValidationCheckCategory = ImpactValidationKind
ValidationKind = ImpactValidationKind
ValidationCheck = ImpactValidationCheck
ValidationCheckSpec = ImpactValidationCheck
ValidationPolicy = RepositoryValidationPolicy
ValidationDAG = ImpactSelectedValidationDAG
ValidationDAGPlan = ImpactSelectedValidationDAG
ValidationDAGExecutionReceipt = ImpactValidationDAGReceipt
build_validation_dag = build_impact_selected_validation_dag
