"""LLM merge-conflict resolver payloads for autonomous agent supervisors."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import signal
import shlex
import sqlite3
import subprocess
import tempfile
import time
import tokenize
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from .event_log import read_jsonl_events


LLM_MERGE_RESOLVER_COMMAND_ENV = "IPFS_ACCELERATE_AGENT_LLM_MERGE_RESOLVER_COMMAND"
LLM_MERGE_RESOLVER_TIMEOUT_ENV = "IPFS_ACCELERATE_AGENT_LLM_MERGE_RESOLVER_TIMEOUT_SECONDS"
DEFAULT_LLM_MERGE_RESOLVER_TIMEOUT_SECONDS = 600.0
DEFAULT_PROMPT_HEADING = "Resolve the autonomous-agent supervisor merge conflict in this repository."
DEFAULT_COMPLETION_RULE = "Do not unblock the source task until validation passes."
MergePromptCallback = Callable[..., str]
MergeResolverPayloadCallback = Callable[..., dict[str, Any]]
MergeResolverInvoker = Callable[..., dict[str, Any]]

# Maximum number of times the resolver will attempt the same merge event
# before marking it as permanently failed. Prevents infinite retry loops.
_MAX_RESOLVE_ATTEMPTS_PER_EVENT = int(
    os.environ.get("IPFS_ACCELERATE_AGENT_MERGE_MAX_ATTEMPTS", "3")
)
_RESOLVED_EVENTS_FILENAME = ".agent-merge-resolved-events.json"


def conflict_fingerprint(event: Mapping[str, Any]) -> str:
    """Return a stable identity for the underlying conflict, not its log event.

    Timestamps and resolver-attempt numbers are intentionally excluded: those
    fields change every time a daemon observes the same conflicted merge.  File
    paths and commit identities are normalized so independent lanes converge on
    the same fingerprint.
    """

    nested = event.get("merge_result")
    merge_result = nested if isinstance(nested, Mapping) else event
    explicit = event.get("conflict_fingerprint") or merge_result.get("conflict_fingerprint")
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()

    def value(*keys: str) -> str:
        for key in keys:
            candidate = merge_result.get(key)
            if candidate in (None, ""):
                candidate = event.get(key)
            if candidate not in (None, ""):
                return str(candidate).strip()
        return ""

    paths_value = (
        merge_result.get("unmerged_paths")
        or merge_result.get("conflicted_paths")
        or merge_result.get("dirty_paths")
        or event.get("unmerged_paths")
        or event.get("conflicted_paths")
        or event.get("dirty_paths")
        or []
    )
    if isinstance(paths_value, str):
        paths = [item.strip() for item in paths_value.splitlines() if item.strip()]
    elif isinstance(paths_value, Sequence):
        paths = [str(item).strip() for item in paths_value if str(item).strip()]
    else:
        paths = []
    material = {
        "task": value("canonical_task_key", "canonical_task_id", "canonical_task_cid", "task_id").casefold(),
        "branch": value("branch", "branch_name", "source_branch").casefold(),
        "target_branch": value("target_branch").casefold(),
        "source_commit": value("source_commit", "commit_sha", "head_sha", "commit").casefold(),
        "target_commit": value("target_commit", "target_head", "base_commit").casefold(),
        "reason": " ".join(value("reason", "failure_reason").casefold().split()),
        "paths": sorted(set(paths)),
    }
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _event_fingerprint(event: dict[str, Any]) -> str:
    """Compatibility alias for the public full-length fingerprint."""

    return conflict_fingerprint(event)


@dataclass(frozen=True)
class ResolverClaim:
    """Fenced ownership of one active conflict-resolution attempt."""

    fingerprint: str
    owner_id: str
    token: str
    attempt: int
    acquired_at: float
    lease_expires_at: float
    event: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "owner_id": self.owner_id,
            "token": self.token,
            "attempt": self.attempt,
            "acquired_at": self.acquired_at,
            "lease_expires_at": self.lease_expires_at,
            "event": dict(self.event),
        }


class MergeResolverRegistry:
    """Durable, fenced registry for conflict resolver attempts.

    Acquisition is serialized by an immediate SQLite transaction.  Thus two
    daemon processes can observe the same event concurrently, but at most one
    receives a claim.  Abandoned claims can be recovered after their lease;
    once the configured attempt bound is reached the conflict is terminally
    quarantined with a JSON receipt rather than becoming a polling loop.
    """

    def __init__(
        self,
        state_dir: Path | str,
        *,
        max_attempts: int = _MAX_RESOLVE_ATTEMPTS_PER_EVENT,
        lease_timeout_seconds: float = DEFAULT_LLM_MERGE_RESOLVER_TIMEOUT_SECONDS,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.database_path = self.state_dir / "merge_resolver.sqlite3"
        self.quarantine_dir = self.state_dir / "merge_resolver_quarantine"
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        self.max_attempts = max(1, int(max_attempts))
        self.lease_timeout_seconds = max(1.0, float(lease_timeout_seconds))
        self._clock = clock or time.time
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conflict_resolutions (
                    fingerprint TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    owner_id TEXT NOT NULL DEFAULT '',
                    token TEXT NOT NULL DEFAULT '',
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    acquired_at REAL NOT NULL DEFAULT 0,
                    lease_expires_at REAL NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL,
                    last_error TEXT NOT NULL DEFAULT '',
                    event_json TEXT NOT NULL DEFAULT '{}',
                    outcome_json TEXT NOT NULL DEFAULT '{}',
                    receipt_path TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS conflict_resolutions_state
                  ON conflict_resolutions(state, lease_expires_at);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.database_path), timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def acquire(
        self,
        event: Mapping[str, Any],
        *,
        owner_id: str = "",
        lease_seconds: float | None = None,
    ) -> ResolverClaim | None:
        """Atomically acquire the event or return ``None`` when it is suppressed."""

        event_dict = dict(event)
        fingerprint = conflict_fingerprint(event_dict)
        owner = str(owner_id or f"resolver-{os.getpid()}")
        now = self._clock()
        lease_duration = max(
            1.0,
            float(self.lease_timeout_seconds if lease_seconds is None else lease_seconds),
        )
        token = uuid.uuid4().hex
        receipt: tuple[dict[str, Any], str, int] | None = None
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM conflict_resolutions WHERE fingerprint=?", (fingerprint,)
                ).fetchone()
                if row is not None:
                    state = str(row["state"])
                    attempts = int(row["attempt_count"])
                    if state in {"succeeded", "quarantined"}:
                        connection.commit()
                        return None
                    if state == "active" and float(row["lease_expires_at"]) > now:
                        connection.commit()
                        return None
                    if attempts >= self.max_attempts:
                        error = str(row["last_error"] or "resolver attempt bound reached")
                        connection.execute(
                            """UPDATE conflict_resolutions SET state='quarantined',
                               owner_id='', token='', lease_expires_at=0, updated_at=?
                               WHERE fingerprint=?""",
                            (now, fingerprint),
                        )
                        connection.commit()
                        receipt = (event_dict, error, attempts)
                    else:
                        attempt = attempts + 1
                        connection.execute(
                            """UPDATE conflict_resolutions SET state='active', owner_id=?,
                               token=?, attempt_count=?, acquired_at=?, lease_expires_at=?,
                               updated_at=?, event_json=?, last_error=''
                               WHERE fingerprint=?""",
                            (
                                owner,
                                token,
                                attempt,
                                now,
                                now + lease_duration,
                                now,
                                _json_text(event_dict),
                                fingerprint,
                            ),
                        )
                        connection.commit()
                        return ResolverClaim(
                            fingerprint=fingerprint,
                            owner_id=owner,
                            token=token,
                            attempt=attempt,
                            acquired_at=now,
                            lease_expires_at=now + lease_duration,
                            event=event_dict,
                        )
                else:
                    connection.execute(
                        """INSERT INTO conflict_resolutions (
                           fingerprint, state, owner_id, token, attempt_count,
                           acquired_at, lease_expires_at, updated_at, event_json
                           ) VALUES (?, 'active', ?, ?, 1, ?, ?, ?, ?)""",
                        (
                            fingerprint,
                            owner,
                            token,
                            now,
                            now + lease_duration,
                            now,
                            _json_text(event_dict),
                        ),
                    )
                    connection.commit()
                    return ResolverClaim(
                        fingerprint=fingerprint,
                        owner_id=owner,
                        token=token,
                        attempt=1,
                        acquired_at=now,
                        lease_expires_at=now + lease_duration,
                        event=event_dict,
                    )
            except Exception:
                connection.rollback()
                raise
        if receipt is not None:
            path = self._write_quarantine_receipt(fingerprint, *receipt)
            self._set_receipt_path(fingerprint, path)
        return None

    def heartbeat(
        self,
        claim: ResolverClaim,
        *,
        lease_seconds: float | None = None,
    ) -> bool:
        """Extend a current claim, rejecting stale fencing tokens."""

        now = self._clock()
        duration = max(
            1.0,
            float(self.lease_timeout_seconds if lease_seconds is None else lease_seconds),
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            result = connection.execute(
                """UPDATE conflict_resolutions SET lease_expires_at=?, updated_at=?
                   WHERE fingerprint=? AND state='active' AND owner_id=? AND token=?""",
                (now + duration, now, claim.fingerprint, claim.owner_id, claim.token),
            )
            connection.commit()
        return result.rowcount == 1

    def release(
        self,
        claim: ResolverClaim,
        *,
        succeeded: bool | None = None,
        outcome: str | Mapping[str, Any] | None = None,
        error: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> Path | None:
        """Release a fenced claim and return a receipt if it was quarantined."""

        if succeeded is None:
            if isinstance(outcome, Mapping):
                succeeded = bool(
                    outcome.get("succeeded")
                    or outcome.get("resolved")
                    or outcome.get("applied")
                    or outcome.get("merged")
                )
            else:
                succeeded = str(outcome or "").casefold() in {
                    "success",
                    "succeeded",
                    "resolved",
                    "completed",
                    "merged",
                }
        outcome_payload: dict[str, Any]
        if isinstance(outcome, Mapping):
            outcome_payload = dict(outcome)
        else:
            outcome_payload = {"outcome": str(outcome or ("succeeded" if succeeded else "failed"))}
        if metadata:
            outcome_payload["metadata"] = dict(metadata)
        now = self._clock()
        receipt_data: tuple[dict[str, Any], str, int] | None = None
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM conflict_resolutions WHERE fingerprint=?",
                (claim.fingerprint,),
            ).fetchone()
            if (
                row is None
                or str(row["state"]) != "active"
                or str(row["owner_id"]) != claim.owner_id
                or str(row["token"]) != claim.token
            ):
                connection.commit()
                return None
            attempts = int(row["attempt_count"])
            terminal = not succeeded and attempts >= self.max_attempts
            state = "succeeded" if succeeded else ("quarantined" if terminal else "failed")
            final_error = str(error or outcome_payload.get("error") or outcome_payload.get("apply_error") or "")
            connection.execute(
                """UPDATE conflict_resolutions SET state=?, owner_id='', token='',
                   lease_expires_at=0, updated_at=?, last_error=?, outcome_json=?
                   WHERE fingerprint=?""",
                (state, now, final_error, _json_text(outcome_payload), claim.fingerprint),
            )
            connection.commit()
            if terminal:
                receipt_data = (json.loads(row["event_json"] or "{}"), final_error, attempts)
        if receipt_data is None:
            return None
        path = self._write_quarantine_receipt(claim.fingerprint, *receipt_data)
        self._set_receipt_path(claim.fingerprint, path)
        return path

    def active_attempt(self, event_or_fingerprint: Mapping[str, Any] | str) -> ResolverClaim | None:
        """Return the current non-expired attempt, if any."""

        fingerprint = self._resolve_fingerprint(event_or_fingerprint)
        now = self._clock()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM conflict_resolutions WHERE fingerprint=?", (fingerprint,)
            ).fetchone()
        if row is None or str(row["state"]) != "active" or float(row["lease_expires_at"]) <= now:
            return None
        return ResolverClaim(
            fingerprint=fingerprint,
            owner_id=str(row["owner_id"]),
            token=str(row["token"]),
            attempt=int(row["attempt_count"]),
            acquired_at=float(row["acquired_at"]),
            lease_expires_at=float(row["lease_expires_at"]),
            event=json.loads(row["event_json"] or "{}"),
        )

    def status(self, event_or_fingerprint: Mapping[str, Any] | str) -> dict[str, Any]:
        """Return durable resolver state for one conflict."""

        fingerprint = self._resolve_fingerprint(event_or_fingerprint)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM conflict_resolutions WHERE fingerprint=?", (fingerprint,)
            ).fetchone()
        if row is None:
            return {"fingerprint": fingerprint, "state": "unseen", "attempt_count": 0}
        return {
            "fingerprint": fingerprint,
            "state": str(row["state"]),
            "owner_id": str(row["owner_id"]),
            "attempt_count": int(row["attempt_count"]),
            "acquired_at": float(row["acquired_at"]),
            "lease_expires_at": float(row["lease_expires_at"]),
            "last_error": str(row["last_error"]),
            "receipt_path": str(row["receipt_path"]),
        }

    @contextmanager
    def claim(
        self,
        event: Mapping[str, Any],
        *,
        owner_id: str = "",
        lease_seconds: float | None = None,
    ) -> Iterator[ResolverClaim | None]:
        """Context-manager form that records exceptions as failed attempts."""

        acquired = self.acquire(event, owner_id=owner_id, lease_seconds=lease_seconds)
        try:
            yield acquired
        except BaseException as exc:
            if acquired is not None:
                self.release(acquired, succeeded=False, error=f"{type(exc).__name__}: {exc}")
            raise
        else:
            if acquired is not None:
                self.release(acquired, succeeded=True, outcome="completed")

    @staticmethod
    def _resolve_fingerprint(event_or_fingerprint: Mapping[str, Any] | str) -> str:
        if isinstance(event_or_fingerprint, Mapping):
            return conflict_fingerprint(event_or_fingerprint)
        return str(event_or_fingerprint)

    def _write_quarantine_receipt(
        self,
        fingerprint: str,
        event: dict[str, Any],
        error: str,
        attempts: int,
    ) -> Path:
        path = self.quarantine_dir / f"{fingerprint}.json"
        _atomic_json_write(
            path,
            {
                "receipt_type": "merge_resolver_quarantine",
                "fingerprint": fingerprint,
                "attempt_count": attempts,
                "max_attempts": self.max_attempts,
                "reason": error or "resolver attempt bound reached",
                "quarantined_at": self._clock(),
                "event": event,
            },
        )
        return path

    def _set_receipt_path(self, fingerprint: str, path: Path) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "UPDATE conflict_resolutions SET receipt_path=?, updated_at=? WHERE fingerprint=?",
                (str(path), self._clock(), fingerprint),
            )
            connection.commit()


# Name the invariant explicitly for callers that do not need the LLM CLI layer.
ConflictResolverRegistry = MergeResolverRegistry
ResolverLease = ResolverClaim


def _json_text(value: Mapping[str, Any]) -> str:
    return json.dumps(dict(value), sort_keys=True, separators=(",", ":"), default=str)


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _load_resolved_events(state_dir: Path) -> dict[str, int]:
    """Load the set of already-resolved event fingerprints with attempt counts."""
    path = state_dir / _RESOLVED_EVENTS_FILENAME
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {k: int(v) for k, v in data.items()} if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


def _save_resolved_events(state_dir: Path, resolved: dict[str, int]) -> None:
    """Persist the resolved event fingerprint set."""
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / _RESOLVED_EVENTS_FILENAME
    try:
        _atomic_json_write(path, resolved)
    except OSError:
        pass


def check_event_idempotency(
    event: dict[str, Any],
    *,
    state_dir: Path,
) -> tuple[bool, str]:
    """Check if a merge event has already been attempted too many times.

    Returns (should_skip, reason).
    """
    fingerprint = _event_fingerprint(event)
    resolved = _load_resolved_events(state_dir)
    attempts = resolved.get(fingerprint, 0)
    if attempts >= _MAX_RESOLVE_ATTEMPTS_PER_EVENT:
        return True, f"event already attempted {attempts} times (max={_MAX_RESOLVE_ATTEMPTS_PER_EVENT})"
    return False, ""


def record_resolve_attempt(
    event: dict[str, Any],
    *,
    state_dir: Path,
) -> None:
    """Record that we attempted to resolve this event."""
    fingerprint = _event_fingerprint(event)
    resolved = _load_resolved_events(state_dir)
    resolved[fingerprint] = resolved.get(fingerprint, 0) + 1
    # Prune old entries (keep last 200)
    if len(resolved) > 200:
        sorted_items = sorted(resolved.items(), key=lambda x: x[1], reverse=True)
        resolved = dict(sorted_items[:200])
    _save_resolved_events(state_dir, resolved)


@dataclass(frozen=True)
class MergeResolverCliConfig:
    """Project-specific defaults for the reusable merge-resolver CLI."""

    default_events_path: Path
    default_repo_root: Path
    prompt_heading: str = DEFAULT_PROMPT_HEADING
    completion_rule: str = DEFAULT_COMPLETION_RULE
    extra_rules: Sequence[str] = field(default_factory=tuple)
    primary_command_env_var: str = ""
    fallback_command_env_var: str = LLM_MERGE_RESOLVER_COMMAND_ENV
    description: str = "Build or invoke an LLM merge resolver for agent-supervisor events"
    missing_event_exit_code: int = 0
    apply_failed_exit_code: int = 1


@dataclass(frozen=True)
class MergeResolverNamespaceSpec:
    """Namespace merge-resolver values without repo-root binding."""

    namespace: str
    prompt_heading: str = DEFAULT_PROMPT_HEADING
    completion_rule: str = DEFAULT_COMPLETION_RULE
    extra_rules: Sequence[str] = field(default_factory=tuple)
    state_prefix: str | None = None
    env_prefix: str = ""
    state_dir: Path | str | None = None
    description: str = "Build or invoke an LLM merge resolver for agent-supervisor events"
    missing_event_exit_code: int = 0
    apply_failed_exit_code: int = 1
    fallback_command_env_var: str = LLM_MERGE_RESOLVER_COMMAND_ENV


@dataclass(frozen=True)
class ConfiguredMergeResolverRunner:
    """Project-bound runner wiring for a merge-resolver CLI."""

    config: MergeResolverCliConfig

    def parse_args(self, argv: Sequence[str] | None = None) -> argparse.Namespace:
        """Parse merge-resolver CLI args using the bound config."""

        return build_configured_merge_resolver_arg_parser(self.config).parse_args(argv)

    def run(self, argv: Sequence[str] | None = None) -> int:
        """Run the configured merge-resolver CLI."""

        return run_configured_merge_resolver_cli(self.config, argv)

    def build_merge_prompt(self) -> MergePromptCallback:
        """Build a prompt callback using the bound project wording."""

        return build_merge_prompt_callback(
            prompt_heading=self.config.prompt_heading,
            completion_rule=self.config.completion_rule,
            extra_rules=self.config.extra_rules,
        )

    def resolver_payload(self) -> MergeResolverPayloadCallback:
        """Build a resolver payload callback using the bound project wording."""

        return build_resolver_payload_callback(
            prompt_heading=self.config.prompt_heading,
            completion_rule=self.config.completion_rule,
            extra_rules=self.config.extra_rules,
        )

    def llm_resolver_invoker(self) -> MergeResolverInvoker:
        """Build an LLM resolver invoker using the bound command env vars."""

        return build_llm_merge_resolver_invoker(
            primary_command_env_var=self.config.primary_command_env_var,
            fallback_command_env_var=self.config.fallback_command_env_var,
        )


def build_configured_merge_resolver_runner(
    *,
    default_events_path: Path | str,
    default_repo_root: Path | str,
    prompt_heading: str = DEFAULT_PROMPT_HEADING,
    completion_rule: str = DEFAULT_COMPLETION_RULE,
    extra_rules: Sequence[str] = (),
    primary_command_env_var: str = "",
    fallback_command_env_var: str = LLM_MERGE_RESOLVER_COMMAND_ENV,
    description: str = "Build or invoke an LLM merge resolver for agent-supervisor events",
    missing_event_exit_code: int = 0,
    apply_failed_exit_code: int = 1,
) -> ConfiguredMergeResolverRunner:
    """Build reusable merge-resolver runner wiring bound to project inputs."""

    return ConfiguredMergeResolverRunner(
        MergeResolverCliConfig(
            default_events_path=Path(default_events_path),
            default_repo_root=Path(default_repo_root),
            prompt_heading=prompt_heading,
            completion_rule=completion_rule,
            extra_rules=tuple(extra_rules),
            primary_command_env_var=primary_command_env_var,
            fallback_command_env_var=fallback_command_env_var,
            description=description,
            missing_event_exit_code=missing_event_exit_code,
            apply_failed_exit_code=apply_failed_exit_code,
        )
    )


def build_namespace_merge_resolver_runner(
    *,
    repo_root: Path | str,
    namespace: str,
    prompt_heading: str = DEFAULT_PROMPT_HEADING,
    completion_rule: str = DEFAULT_COMPLETION_RULE,
    extra_rules: Sequence[str] = (),
    state_prefix: str | None = None,
    env_prefix: str = "",
    state_dir: Path | str | None = None,
    description: str = "Build or invoke an LLM merge resolver for agent-supervisor events",
    missing_event_exit_code: int = 0,
    apply_failed_exit_code: int = 1,
    fallback_command_env_var: str = LLM_MERGE_RESOLVER_COMMAND_ENV,
) -> ConfiguredMergeResolverRunner:
    """Build a merge-resolver runner using the standard namespace state layout."""

    from .implementation_daemon_runner import namespace_implementation_state_artifact_paths
    from .wrapper_utils import agent_supervisor_namespace_paths, prefixed_env_var

    resolved_repo_root = Path(repo_root)
    namespace_paths = agent_supervisor_namespace_paths(resolved_repo_root, namespace)
    state_paths = namespace_implementation_state_artifact_paths(
        namespace_paths,
        state_prefix=state_prefix,
        state_dir=state_dir,
    )
    return build_configured_merge_resolver_runner(
        default_events_path=state_paths["events_path"],
        default_repo_root=resolved_repo_root,
        prompt_heading=prompt_heading,
        completion_rule=completion_rule,
        extra_rules=extra_rules,
        primary_command_env_var=(
            prefixed_env_var(env_prefix, "LLM_MERGE_RESOLVER_COMMAND") if env_prefix else ""
        ),
        fallback_command_env_var=fallback_command_env_var,
        description=description,
        missing_event_exit_code=missing_event_exit_code,
        apply_failed_exit_code=apply_failed_exit_code,
    )


def build_namespace_merge_resolver_runner_from_spec(
    *,
    repo_root: Path | str,
    resolver_spec: MergeResolverNamespaceSpec,
) -> ConfiguredMergeResolverRunner:
    """Build a namespace merge-resolver runner from a reusable namespace spec."""

    return build_namespace_merge_resolver_runner(
        repo_root=repo_root,
        namespace=resolver_spec.namespace,
        prompt_heading=resolver_spec.prompt_heading,
        completion_rule=resolver_spec.completion_rule,
        extra_rules=resolver_spec.extra_rules,
        state_prefix=resolver_spec.state_prefix,
        env_prefix=resolver_spec.env_prefix,
        state_dir=resolver_spec.state_dir,
        description=resolver_spec.description,
        missing_event_exit_code=resolver_spec.missing_event_exit_code,
        apply_failed_exit_code=resolver_spec.apply_failed_exit_code,
        fallback_command_env_var=resolver_spec.fallback_command_env_var,
    )


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    return read_jsonl_events(path, repair=True)


def latest_failed_merge_event(events: list[dict[str, Any]], *, task_id: str | None = None) -> dict[str, Any] | None:
    """Return the newest merge failure event, optionally filtered by task id."""

    resolved_task_ids: set[str] = set()
    resolved_branches: set[str] = set()
    for event in reversed(events):
        if str(event.get("type") or "") not in {"implementation_finished", "merge_finished", "merge_reconciled"}:
            continue
        merge_result = event.get("merge_result") if isinstance(event.get("merge_result"), dict) else event
        if not isinstance(merge_result, dict):
            continue
        event_task_id = str(event.get("task_id") or merge_result.get("task_id") or "")
        branch = str(
            merge_result.get("branch")
            or event.get("branch")
            or event.get("implementation_branch")
            or ""
        )
        if task_id and event_task_id != task_id:
            continue
        if merge_result.get("merged") or event.get("resolved") is True:
            if task_id:
                return None
            if event_task_id:
                resolved_task_ids.add(event_task_id)
            if branch:
                resolved_branches.add(branch)
            continue
        if not merge_result.get("attempted") or merge_result.get("merged"):
            continue
        if str(merge_result.get("reason") or "") == "not_attempted":
            continue
        if event_task_id and event_task_id in resolved_task_ids:
            continue
        if branch and branch in resolved_branches:
            continue
        return event
    return None


def unmerged_paths(repo_root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=U"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    return sorted(line.strip() for line in result.stdout.splitlines() if line.strip())


def merge_in_progress(repo_root: Path) -> bool:
    """Return whether ``repo_root`` currently owns an unfinished Git merge."""

    result = subprocess.run(
        ["git", "rev-parse", "--git-path", "MERGE_HEAD"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return False
    merge_head = Path(result.stdout.strip())
    if not merge_head.is_absolute():
        merge_head = repo_root / merge_head
    return merge_head.exists()


def active_merge_heads(repo_root: Path) -> tuple[str, ...]:
    """Return the commit identities currently recorded in ``MERGE_HEAD``."""

    result = subprocess.run(
        ["git", "rev-parse", "--git-path", "MERGE_HEAD"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return ()
    merge_head_path = Path(result.stdout.strip())
    if not merge_head_path.is_absolute():
        merge_head_path = repo_root / merge_head_path
    try:
        values = merge_head_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ()
    return tuple(value.strip().lower() for value in values if value.strip())


def _resolve_commit(repo_root: Path, value: Any) -> str:
    ref = str(value or "").strip()
    if not ref:
        return ""
    result = subprocess.run(
        ["git", "rev-parse", "--verify", f"{ref}^{{commit}}"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip().lower() if result.returncode == 0 else ""


def active_merge_matches_payload(payload: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    """Verify that a resolver payload describes the merge active in ``repo_root``.

    Event logs from independent lanes can contain old failures.  A live merge is
    eligible only when its ``MERGE_HEAD`` matches the recorded source branch or
    implementation commit; merely observing unmerged paths is not sufficient.
    """

    active_heads = set(active_merge_heads(repo_root))
    if not active_heads:
        return {
            "matches": False,
            "reason": "merge_not_active",
            "active_merge_heads": [],
            "expected_merge_heads": {},
        }

    nested = payload.get("merge_result")
    merge_result = nested if isinstance(nested, Mapping) else payload
    references: dict[str, Any] = {
        "branch": merge_result.get("branch") or payload.get("branch"),
        "implementation_commit": (
            merge_result.get("implementation_commit")
            or payload.get("implementation_commit")
        ),
        "source_commit": merge_result.get("source_commit") or payload.get("source_commit"),
        "commit_sha": merge_result.get("commit_sha") or payload.get("commit_sha"),
    }
    command = merge_result.get("command") or payload.get("command")
    if isinstance(command, Sequence) and not isinstance(command, (str, bytes)):
        command_parts = [str(item) for item in command]
        if "merge" in command_parts:
            merge_index = command_parts.index("merge")
            merge_operands = [
                item
                for item in command_parts[merge_index + 1 :]
                if item and not item.startswith("-")
            ]
            if merge_operands:
                references["command_source"] = merge_operands[-1]

    expected_heads = {
        label: commit
        for label, value in references.items()
        if (commit := _resolve_commit(repo_root, value))
    }
    matched = sorted(active_heads.intersection(expected_heads.values()))
    return {
        "matches": bool(matched),
        "reason": "matched" if matched else "active_merge_identity_mismatch",
        "active_merge_heads": sorted(active_heads),
        "expected_merge_heads": expected_heads,
        "matched_merge_heads": matched,
    }


def validate_resolved_paths(repo_root: Path, paths: Sequence[str]) -> dict[str, Any]:
    """Reject conflict markers and invalid Python in resolver-touched paths."""

    checked_paths: list[str] = []
    expanded_paths: list[str] = []
    invalid_paths: list[str] = []
    marker_findings: list[dict[str, Any]] = []
    syntax_errors: list[dict[str, Any]] = []
    root = repo_root.resolve()
    pending = list(dict.fromkeys(str(item).strip() for item in paths if str(item).strip()))
    visited: set[str] = set()

    while pending:
        raw_path = pending.pop(0)
        if raw_path in visited:
            continue
        visited.add(raw_path)
        relative = Path(raw_path)
        candidate = (root / relative).resolve()
        try:
            normalized = candidate.relative_to(root).as_posix()
        except ValueError:
            invalid_paths.append(raw_path)
            continue
        if relative.is_absolute() or not candidate.exists() or not candidate.is_file():
            if relative.is_absolute():
                invalid_paths.append(raw_path)
            elif candidate.is_dir():
                top_level_result = subprocess.run(
                    ["git", "rev-parse", "--show-toplevel"],
                    cwd=candidate,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                nested_repo_root = (
                    Path(top_level_result.stdout.strip()).resolve()
                    if top_level_result.returncode == 0 and top_level_result.stdout.strip()
                    else None
                )
                nested_paths: set[str] = set()
                for command in (
                    [
                        "git",
                        "diff-tree",
                        "--root",
                        "--no-commit-id",
                        "--name-only",
                        "-r",
                        "-m",
                        "HEAD",
                    ],
                    ["git", "diff", "--name-only", "HEAD"],
                    ["git", "ls-files", "--others", "--exclude-standard"],
                ):
                    result = subprocess.run(
                        command,
                        cwd=candidate,
                        text=True,
                        capture_output=True,
                        check=False,
                    )
                    if result.returncode == 0:
                        nested_paths.update(
                            line.strip()
                            for line in result.stdout.splitlines()
                            if line.strip()
                        )
                for nested_path in sorted(nested_paths)[:1000]:
                    if nested_repo_root == candidate:
                        expanded = (Path(normalized) / nested_path).as_posix()
                    else:
                        expanded = Path(nested_path).as_posix()
                        normalized_prefix = normalized.rstrip("/") + "/"
                        if normalized not in {"", "."} and not expanded.startswith(normalized_prefix):
                            continue
                    expanded_paths.append(expanded)
                    pending.append(expanded)
            continue
        checked_paths.append(normalized)
        try:
            content = candidate.read_bytes()
        except OSError as exc:
            syntax_errors.append({"path": normalized, "error": f"{type(exc).__name__}: {exc}"})
            continue
        for line_number, line in enumerate(content.splitlines(), start=1):
            stripped = line.lstrip()
            if (
                stripped == b"======="
                or stripped == b"<<<<<<<"
                or stripped.startswith(b"<<<<<<< ")
                or stripped == b">>>>>>>"
                or stripped.startswith(b">>>>>>> ")
            ):
                marker_findings.append(
                    {
                        "path": normalized,
                        "line": line_number,
                        "marker": stripped[:80].decode("utf-8", errors="replace"),
                    }
                )
        if candidate.suffix == ".py":
            try:
                with tokenize.open(candidate) as stream:
                    source = stream.read()
                ast.parse(source, filename=str(candidate))
            except (OSError, SyntaxError, UnicodeError) as exc:
                syntax_errors.append(
                    {
                        "path": normalized,
                        "line": int(getattr(exc, "lineno", 0) or 0),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    return {
        "valid": not invalid_paths and not marker_findings and not syntax_errors,
        "checked_paths": checked_paths,
        "expanded_paths": expanded_paths,
        "invalid_paths": invalid_paths,
        "marker_findings": marker_findings,
        "syntax_errors": syntax_errors,
    }


def merge_event_workspace(event: Mapping[str, Any], repo_root: Path) -> Path:
    """Resolve the checkout in which the recorded merge was attempted."""

    nested = event.get("merge_result")
    merge_result = nested if isinstance(nested, Mapping) else event
    for key in ("main_worktree_path", "merge_workspace_path", "workspace_path"):
        value = merge_result.get(key) or event.get(key)
        if not value:
            continue
        candidate = Path(str(value)).expanduser()
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        if candidate.is_dir():
            return candidate.resolve()
    return repo_root.resolve()


def compact_text(value: Any, *, limit: int = 2000) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def resolver_timeout_seconds(value: str | float | int | None = None) -> float | None:
    raw_value = os.environ.get(LLM_MERGE_RESOLVER_TIMEOUT_ENV, "") if value is None else value
    if raw_value in {None, ""}:
        return DEFAULT_LLM_MERGE_RESOLVER_TIMEOUT_SECONDS
    try:
        timeout_seconds = float(raw_value)
    except (TypeError, ValueError):
        return DEFAULT_LLM_MERGE_RESOLVER_TIMEOUT_SECONDS
    if timeout_seconds <= 0:
        return None
    return timeout_seconds


def _merge_result(event: dict[str, Any]) -> dict[str, Any]:
    value = event.get("merge_result")
    return value if isinstance(value, dict) else event


def build_merge_prompt(
    *,
    event: dict[str, Any],
    repo_root: Path,
    prompt_heading: str = DEFAULT_PROMPT_HEADING,
    completion_rule: str = DEFAULT_COMPLETION_RULE,
    extra_rules: Sequence[str] | None = None,
) -> str:
    """Build an LLM prompt that can resolve a semantic merge conflict."""

    merge_result = _merge_result(event)
    command = merge_result.get("command") or []
    if isinstance(command, list):
        command_text = shlex.join(str(part) for part in command)
    else:
        command_text = str(command)
    paths = unmerged_paths(repo_root)
    dirty_paths = merge_result.get("dirty_paths") or []
    rules = [
        "Inspect the conflicted files and implementation branch before editing.",
        "Preserve the semantic intent of both sides when possible.",
        "Keep changes scoped to the task and conflict resolution.",
        "Run the task validation after resolving the conflict.",
        "Commit the merge resolution in the owning repository or submodule.",
        "For submodule gitlink conflicts (mode 160000), prefer the newer commit reference; "
        "after resolving, run 'git submodule update --init --recursive' for affected paths.",
        "If a conflict involves recursive submodule references, resolve the innermost submodule first, "
        "then work outward to avoid circular dependency issues.",
        completion_rule,
    ]
    if extra_rules:
        rules.extend(str(rule) for rule in extra_rules if str(rule).strip())
    return "\n".join(
        [
            prompt_heading,
            "",
            f"Task id: {event.get('task_id') or merge_result.get('task_id')}",
            f"Attempt: {event.get('attempt') or merge_result.get('attempt')}",
            f"Implementation branch: {merge_result.get('branch')}",
            f"Target branch: {merge_result.get('target_branch')}",
            f"Merge reason: {merge_result.get('reason')}",
            f"Merge command: {command_text}",
            f"Repository: {repo_root}",
            f"Unmerged paths: {', '.join(paths) or 'none reported by git'}",
            f"Dirty paths: {', '.join(str(item) for item in dirty_paths) or 'none recorded'}",
            "",
            "Rules:",
            *(f"{index}. {rule}" for index, rule in enumerate(rules, start=1)),
            "",
            "Merge stdout excerpt:",
            compact_text(merge_result.get("stdout")),
            "",
            "Merge stderr excerpt:",
            compact_text(merge_result.get("stderr")),
        ]
    )


def build_merge_prompt_callback(
    *,
    prompt_heading: str = DEFAULT_PROMPT_HEADING,
    completion_rule: str = DEFAULT_COMPLETION_RULE,
    extra_rules: Sequence[str] | None = None,
) -> MergePromptCallback:
    """Build a prompt callback with project-specific merge-resolution wording."""

    configured_extra_rules = tuple(extra_rules or ())

    def callback(*, event: dict[str, Any], repo_root: Path) -> str:
        return build_merge_prompt(
            event=event,
            repo_root=repo_root,
            prompt_heading=prompt_heading,
            completion_rule=completion_rule,
            extra_rules=configured_extra_rules,
        )

    return callback


def resolver_payload(
    *,
    events_path: Path,
    repo_root: Path,
    task_id: str | None = None,
    prompt_heading: str = DEFAULT_PROMPT_HEADING,
    completion_rule: str = DEFAULT_COMPLETION_RULE,
    extra_rules: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return a dry-run JSON payload for the latest merge failure."""

    event = latest_failed_merge_event(iter_jsonl(events_path), task_id=task_id)
    if event is None:
        return {
            "found": False,
            "task_id": task_id,
            "events_path": str(events_path),
            "repo_root": str(repo_root),
            "prompt": "",
        }
    resolution_root = merge_event_workspace(event, repo_root)
    merge_result = _merge_result(event)
    return {
        "found": True,
        "task_id": str(event.get("task_id") or merge_result.get("task_id") or ""),
        "attempt": event.get("attempt") or merge_result.get("attempt"),
        "events_path": str(events_path),
        "repo_root": str(resolution_root),
        "conflict_fingerprint": conflict_fingerprint(event),
        "event_timestamp": str(event.get("timestamp") or ""),
        "branch": str(merge_result.get("branch") or ""),
        "implementation_commit": str(
            merge_result.get("implementation_commit")
            or event.get("implementation_commit")
            or ""
        ),
        "target_branch": str(merge_result.get("target_branch") or ""),
        "command": merge_result.get("command") or [],
        "reason": str(merge_result.get("reason") or ""),
        "dirty_paths": merge_result.get("dirty_paths") or [],
        "unmerged_paths": unmerged_paths(resolution_root),
        "merge_in_progress": merge_in_progress(resolution_root),
        "prompt": build_merge_prompt(
            event=event,
            repo_root=resolution_root,
            prompt_heading=prompt_heading,
            completion_rule=completion_rule,
            extra_rules=extra_rules,
        ),
    }


def build_resolver_payload_callback(
    *,
    prompt_heading: str = DEFAULT_PROMPT_HEADING,
    completion_rule: str = DEFAULT_COMPLETION_RULE,
    extra_rules: Sequence[str] | None = None,
) -> MergeResolverPayloadCallback:
    """Build a resolver-payload callback with project-specific prompt defaults."""

    configured_extra_rules = tuple(extra_rules or ())

    def callback(*, events_path: Path, repo_root: Path, task_id: str | None = None) -> dict[str, Any]:
        return resolver_payload(
            events_path=events_path,
            repo_root=repo_root,
            task_id=task_id,
            prompt_heading=prompt_heading,
            completion_rule=completion_rule,
            extra_rules=configured_extra_rules,
        )

    return callback


def invoke_llm_resolver(
    payload: dict[str, Any],
    *,
    command_template: str | None = None,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Invoke an external LLM resolver command with the prompt on stdin."""

    command_template = (command_template or os.environ.get(LLM_MERGE_RESOLVER_COMMAND_ENV, "")).strip()
    if not command_template:
        return {
            **payload,
            "applied": False,
            "apply_error": f"{LLM_MERGE_RESOLVER_COMMAND_ENV} is not set",
        }
    command = shlex.split(command_template)
    timeout = resolver_timeout_seconds(timeout_seconds)
    process = subprocess.Popen(
        command,
        cwd=payload.get("repo_root") or None,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(str(payload.get("prompt") or ""), timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate()
        return {
            **payload,
            "applied": False,
            "llm_command": command,
            "llm_timeout": True,
            "llm_timeout_seconds": timeout,
            "llm_returncode": None,
            "llm_stdout": compact_text(stdout or exc.stdout),
            "llm_stderr": compact_text(stderr or exc.stderr),
            "apply_error": f"LLM merge resolver timed out after {timeout} seconds",
        }
    return {
        **payload,
        "applied": process.returncode == 0,
        "llm_command": command,
        "llm_timeout": False,
        "llm_timeout_seconds": timeout,
        "llm_returncode": process.returncode,
        "llm_stdout": compact_text(stdout),
        "llm_stderr": compact_text(stderr),
    }


def _configured_command_template(primary_env_var: str, fallback_env_var: str) -> str | None:
    for env_var in (primary_env_var, fallback_env_var):
        if not env_var:
            continue
        configured = os.environ.get(env_var, "").strip()
        if configured:
            return configured
    return None


def _missing_command_error(primary_env_var: str, fallback_env_var: str) -> str:
    env_vars = [env_var for env_var in (primary_env_var, fallback_env_var) if env_var]
    if not env_vars:
        return "LLM merge resolver command is not configured"
    return f"{' or '.join(env_vars)} is not set"


def build_llm_merge_resolver_invoker(
    *,
    primary_command_env_var: str = "",
    fallback_command_env_var: str = LLM_MERGE_RESOLVER_COMMAND_ENV,
    missing_command_error: str = "",
) -> MergeResolverInvoker:
    """Build an invoker that resolves project and fallback command env vars."""

    def callback(payload: dict[str, Any], *, timeout_seconds: float | None = None) -> dict[str, Any]:
        command_template = _configured_command_template(primary_command_env_var, fallback_command_env_var)
        if command_template is None:
            return {
                **payload,
                "applied": False,
                "apply_error": missing_command_error
                or _missing_command_error(primary_command_env_var, fallback_command_env_var),
            }
        return invoke_llm_resolver(payload, command_template=command_template, timeout_seconds=timeout_seconds)

    return callback


def build_configured_merge_resolver_arg_parser(config: MergeResolverCliConfig) -> argparse.ArgumentParser:
    """Build a standard parser for a configured merge-resolver wrapper."""

    parser = argparse.ArgumentParser(description=config.description)
    parser.add_argument("--task-id", default=None, help="Resolve the latest merge failure for this task id.")
    parser.add_argument("--events-path", type=Path, default=config.default_events_path)
    parser.add_argument("--repo-root", type=Path, default=config.default_repo_root)
    parser.add_argument("--apply", action="store_true", help="Invoke the configured LLM resolver command.")
    parser.add_argument("--command", default=None, help="Resolver command template. Defaults to configured env vars.")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help=f"Resolver subprocess timeout. Defaults to {LLM_MERGE_RESOLVER_TIMEOUT_ENV} or 600 seconds; <=0 disables.",
    )
    return parser


def run_configured_merge_resolver_cli(
    config: MergeResolverCliConfig,
    argv: Sequence[str] | None = None,
) -> int:
    """Run a project-configured merge-resolver dry-run/apply CLI."""

    args = build_configured_merge_resolver_arg_parser(config).parse_args(argv)
    payload = resolver_payload(
        events_path=args.events_path,
        repo_root=args.repo_root,
        task_id=args.task_id,
        prompt_heading=config.prompt_heading,
        completion_rule=config.completion_rule,
        extra_rules=config.extra_rules,
    )
    if args.apply and payload.get("found"):
        if args.command:
            payload = invoke_llm_resolver(payload, command_template=args.command, timeout_seconds=args.timeout_seconds)
        else:
            invoker = build_llm_merge_resolver_invoker(
                primary_command_env_var=config.primary_command_env_var,
                fallback_command_env_var=config.fallback_command_env_var,
            )
            payload = invoker(payload, timeout_seconds=args.timeout_seconds)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload.get("found"):
        return config.missing_event_exit_code
    if args.apply and not payload.get("applied"):
        return config.apply_failed_exit_code
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build or invoke an LLM merge resolver for agent-supervisor events")
    parser.add_argument("--events-path", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--apply", action="store_true", help="Invoke the configured resolver command")
    parser.add_argument("--command", default=None, help="Resolver command template. Defaults to env var.")
    parser.add_argument("--prompt-heading", default=DEFAULT_PROMPT_HEADING)
    parser.add_argument("--completion-rule", default=DEFAULT_COMPLETION_RULE)
    parser.add_argument("--extra-rule", action="append", default=[])
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help=f"Resolver subprocess timeout. Defaults to {LLM_MERGE_RESOLVER_TIMEOUT_ENV} or 600 seconds; <=0 disables.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    payload = resolver_payload(
        events_path=args.events_path,
        repo_root=args.repo_root.resolve(),
        task_id=args.task_id,
        prompt_heading=args.prompt_heading,
        completion_rule=args.completion_rule,
        extra_rules=args.extra_rule,
    )
    if args.apply:
        payload = invoke_llm_resolver(payload, command_template=args.command, timeout_seconds=args.timeout_seconds)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.apply and payload.get("found") and not payload.get("applied"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
