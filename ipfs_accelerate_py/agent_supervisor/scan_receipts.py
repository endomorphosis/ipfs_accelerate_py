"""Typed receipts for supervisor backlog-refill scans.

Historically refill callbacks returned a list.  An empty list could mean that
the analyzer exhausted its search space, that a scan was skipped, that every
candidate was already represented, or that the analyzer failed.  Those states
are operationally different, and only one of them can contribute evidence to
goal completion.

This module defines the versioned result exchanged at refill boundaries.  The
legacy adapter is intentionally explicit: callers must name the reason for an
empty legacy result.  :class:`RefillScanResult` also rejects boolean coercion so
new completion code cannot accidentally restore the old ``if not findings``
ambiguity.
"""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Generic, Iterator, Mapping, Sequence, TypeVar, Union, overload


REFILL_SCAN_RESULT_SCHEMA_VERSION = 1
"""Current serialized :class:`RefillScanResult` schema version."""

# Short aliases are useful to consumers which persist mixed supervisor
# contracts in one envelope.
SCHEMA_VERSION = REFILL_SCAN_RESULT_SCHEMA_VERSION
CONTRACT_VERSION = REFILL_SCAN_RESULT_SCHEMA_VERSION

T = TypeVar("T")


class ScanTerminalReason(str, Enum):
    """Why a refill scan stopped.

    ``EXHAUSTED`` is deliberately distinct from all other empty outcomes.  It
    means the configured analyzer completed its eligible search space.  Even
    then, completion reasoning is opt-in through
    :attr:`RefillScanResult.safe_for_completion_reasoning`.
    """

    GENERATED = "generated"
    EXHAUSTED = "exhausted"
    DUPLICATE_ONLY = "duplicate_only"
    THRESHOLD_SATISFIED = "threshold_satisfied"
    COOLDOWN = "cooldown"
    DISABLED = "disabled"
    PARTIAL = "partial"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


class ScanMode(str, Enum):
    """Common scan modes.

    The receipt stores ``scan_mode`` as a validated string because analyzers
    may use more specific modes.  This enum supplies stable names for common
    modes and the modes currently emitted by the backlog threshold logic.
    """

    CODEBASE = "codebase"
    OBJECTIVE = "objective"
    INCREMENTAL = "incremental"
    EXHAUSTIVE = "exhaustive"
    AUDIT = "audit"
    FORCE = "force"
    LOW_BACKLOG = "low_backlog"
    DRAINED_EXHAUSTIVE = "drained_exhaustive"
    RUNNABLE_DRAINED_EXHAUSTIVE = "runnable_drained_exhaustive"
    RUNNABLE_DRAINED_LOW_BACKLOG = "runnable_drained_low_backlog"
    OPEN_TASK_THRESHOLD = "open_task_threshold"
    COOLDOWN = "cooldown"
    LEGACY = "legacy"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_datetime(value: Union[datetime, str], *, field_name: str) -> datetime:
    """Normalize an aware datetime or ISO-8601 string to UTC."""

    if isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            value = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from exc
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime or ISO-8601 string")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _nonempty(value: object, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} must not be empty")
    return text


def _json_value(value: Any) -> Any:
    """Project receipt values to JSON-safe primitives without mutating them."""

    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, datetime):
        return _utc_datetime(value, field_name="receipt value").isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return [_json_value(item) for item in sorted(value, key=repr)]
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _git_value(repo_root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def _git_bytes(repo_root: Path, *arguments: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return b""
    return completed.stdout if completed.returncode == 0 else b""


@dataclass(frozen=True)
class RepositoryTreeIdentity:
    """Repository and source-tree identity attached to a scan receipt."""

    repository_id: str
    tree_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "repository_id", _nonempty(self.repository_id, field_name="repository_id"))
        object.__setattr__(self, "tree_id", _nonempty(self.tree_id, field_name="tree_id"))

    def to_dict(self) -> dict[str, str]:
        return {"repository_id": self.repository_id, "tree_id": self.tree_id}


def scan_identity(repo_root: Union[Path, str]) -> RepositoryTreeIdentity:
    """Resolve stable best-effort git identity for ``repo_root``.

    ``repository_id`` identifies the git repository independently of its
    current commit when possible.  A clean checkout uses Git's tree object;
    dirty tracked and untracked content is folded into a ``sha256:`` identity
    so two materially different analyzed worktrees cannot share a receipt
    identity.  Non-git inputs remain identifiable using their canonical path
    and an explicit path-derived marker.
    """

    root = Path(repo_root).expanduser().resolve()
    top_level_text = _git_value(root, "rev-parse", "--show-toplevel")
    top_level = Path(top_level_text).resolve() if top_level_text else root

    common_dir_text = _git_value(root, "rev-parse", "--git-common-dir")
    if common_dir_text:
        common_dir = Path(common_dir_text)
        if not common_dir.is_absolute():
            common_dir = root / common_dir
        repository_id = str(common_dir.resolve())
    else:
        repository_id = str(top_level)

    head_tree = _git_value(root, "rev-parse", "HEAD^{tree}")
    if not head_tree:
        tree_id = f"unversioned:{sha256(str(root).encode('utf-8')).hexdigest()}"
    else:
        status = _git_bytes(
            top_level,
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        )
        if not status:
            tree_id = head_tree
        else:
            digest = sha256()
            digest.update(head_tree.encode("ascii", errors="replace"))
            digest.update(b"\0status\0")
            digest.update(status)
            digest.update(b"\0diff\0")
            digest.update(
                _git_bytes(top_level, "diff", "--binary", "--no-ext-diff", "HEAD", "--")
            )
            untracked = _git_bytes(
                top_level,
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
            ).split(b"\0")
            for raw_relative in sorted(path for path in untracked if path):
                digest.update(b"\0untracked\0")
                digest.update(raw_relative)
                try:
                    relative = raw_relative.decode("utf-8", errors="surrogateescape")
                    candidate = top_level / relative
                    if candidate.is_symlink():
                        digest.update(candidate.readlink().as_posix().encode("utf-8"))
                    elif candidate.is_file():
                        with candidate.open("rb") as stream:
                            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                                digest.update(chunk)
                except OSError:
                    # The status snapshot remains in the digest if a file is
                    # concurrently removed while identity is being resolved.
                    continue
            tree_id = f"sha256:{digest.hexdigest()}"
    return RepositoryTreeIdentity(repository_id=repository_id, tree_id=tree_id)


@dataclass(frozen=True)
class RefillScanResult(Generic[T]):
    """Versioned, unambiguous result of one backlog-refill scan.

    ``items`` contains records actually generated by the refill operation, not
    raw analyzer candidates.  Sequence-style access exists as a narrow
    migration aid for direct callers.  Boolean conversion is forbidden; use
    :attr:`generated_count` or inspect :attr:`terminal_reason` explicitly.
    """

    terminal_reason: ScanTerminalReason
    scan_mode: str
    analyzer_version: str
    repository_id: str
    tree_id: str
    started_at: datetime
    finished_at: datetime
    items: tuple[T, ...] = ()
    safe_for_completion_reasoning: bool = False
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = REFILL_SCAN_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        try:
            reason = (
                self.terminal_reason
                if isinstance(self.terminal_reason, ScanTerminalReason)
                else ScanTerminalReason(str(self.terminal_reason))
            )
        except ValueError as exc:
            raise ValueError(f"unknown scan terminal reason: {self.terminal_reason!r}") from exc
        object.__setattr__(self, "terminal_reason", reason)

        mode = self.scan_mode.value if isinstance(self.scan_mode, ScanMode) else self.scan_mode
        object.__setattr__(self, "scan_mode", _nonempty(mode, field_name="scan_mode"))
        object.__setattr__(
            self,
            "analyzer_version",
            _nonempty(self.analyzer_version, field_name="analyzer_version"),
        )
        object.__setattr__(
            self,
            "repository_id",
            _nonempty(self.repository_id, field_name="repository_id"),
        )
        object.__setattr__(self, "tree_id", _nonempty(self.tree_id, field_name="tree_id"))

        started_at = _utc_datetime(self.started_at, field_name="started_at")
        finished_at = _utc_datetime(self.finished_at, field_name="finished_at")
        if finished_at < started_at:
            raise ValueError("finished_at must not be earlier than started_at")
        object.__setattr__(self, "started_at", started_at)
        object.__setattr__(self, "finished_at", finished_at)

        if int(self.schema_version) != REFILL_SCAN_RESULT_SCHEMA_VERSION:
            raise ValueError(
                "unsupported refill scan result schema version: "
                f"{self.schema_version!r} (expected {REFILL_SCAN_RESULT_SCHEMA_VERSION})"
            )
        object.__setattr__(self, "schema_version", int(self.schema_version))

        if isinstance(self.items, (str, bytes, bytearray)):
            raise TypeError("items must be a sequence of generated records")
        object.__setattr__(self, "items", tuple(self.items))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

        item_count = len(self.items)
        if reason is ScanTerminalReason.GENERATED and item_count == 0:
            raise ValueError("generated scan results must contain at least one generated item")
        if reason not in {ScanTerminalReason.GENERATED, ScanTerminalReason.PARTIAL} and item_count:
            raise ValueError(
                f"{reason.value} scan results cannot contain generated items; use "
                "generated or partial"
            )
        if self.safe_for_completion_reasoning and reason is not ScanTerminalReason.EXHAUSTED:
            raise ValueError(
                "only an exhausted scan may be marked safe_for_completion_reasoning"
            )

        normalized_error = str(self.error or "").strip() or None
        if reason in {ScanTerminalReason.FAILED, ScanTerminalReason.TIMED_OUT} and not normalized_error:
            raise ValueError(f"{reason.value} scan results must include an error")
        object.__setattr__(self, "error", normalized_error)

    @property
    def reason(self) -> ScanTerminalReason:
        """Compatibility shorthand for :attr:`terminal_reason`."""

        return self.terminal_reason

    @property
    def findings(self) -> tuple[T, ...]:
        """Generated records, named for existing refill call sites."""

        return self.items

    @property
    def generated_count(self) -> int:
        return len(self.items)

    @property
    def contract_version(self) -> int:
        return self.schema_version

    @property
    def version(self) -> int:
        """Concise alias for consumers that call all contracts ``version``."""

        return self.schema_version

    @property
    def repository_identity(self) -> str:
        return self.repository_id

    @property
    def tree_identity(self) -> str:
        return self.tree_id

    @property
    def duration_seconds(self) -> float:
        return (self.finished_at - self.started_at).total_seconds()

    def __bool__(self) -> bool:
        raise TypeError(
            "RefillScanResult has no truth value; inspect terminal_reason, "
            "generated_count, or safe_for_completion_reasoning explicitly"
        )

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[T]:
        return iter(self.items)

    @overload
    def __getitem__(self, index: int) -> T:
        ...

    @overload
    def __getitem__(self, index: slice) -> tuple[T, ...]:
        ...

    def __getitem__(self, index: Union[int, slice]) -> Union[T, tuple[T, ...]]:
        return self.items[index]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable contract envelope."""

        return {
            "schema_version": self.schema_version,
            "contract_version": self.schema_version,
            "version": self.schema_version,
            "terminal_reason": self.terminal_reason.value,
            "scan_mode": self.scan_mode,
            "analyzer_version": self.analyzer_version,
            "repository_id": self.repository_id,
            "repository_identity": self.repository_id,
            "tree_id": self.tree_id,
            "tree_identity": self.tree_id,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
            "items": _json_value(self.items),
            "generated_count": self.generated_count,
            "safe_for_completion_reasoning": self.safe_for_completion_reasoning,
            "error": self.error,
            "metadata": _json_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RefillScanResult[Any]":
        """Validate and deserialize a receipt produced by :meth:`to_dict`."""

        schema_version = payload.get(
            "schema_version", payload.get("contract_version", payload.get("version"))
        )
        if schema_version is None:
            raise ValueError("refill scan result is missing schema_version")
        items = payload.get("items", payload.get("findings", ()))
        if items is None:
            items = ()
        if isinstance(items, (str, bytes, bytearray)) or not isinstance(items, Sequence):
            raise TypeError("refill scan result items must be a sequence")
        declared_count = payload.get("generated_count")
        if declared_count is not None and int(declared_count) != len(items):
            raise ValueError("generated_count does not match the number of items")
        return cls(
            schema_version=int(schema_version),
            terminal_reason=payload.get("terminal_reason", payload.get("reason")),
            scan_mode=payload.get("scan_mode", ""),
            analyzer_version=payload.get("analyzer_version", ""),
            repository_id=payload.get("repository_id", payload.get("repository_identity", "")),
            tree_id=payload.get("tree_id", payload.get("tree_identity", "")),
            started_at=payload.get("started_at", ""),
            finished_at=payload.get("finished_at", ""),
            items=tuple(items),
            safe_for_completion_reasoning=bool(
                payload.get("safe_for_completion_reasoning", False)
            ),
            error=payload.get("error"),
            metadata=payload.get("metadata") or {},
        )


def build_scan_result(
    terminal_reason: Union[ScanTerminalReason, str],
    scan_mode: Union[ScanMode, str],
    analyzer_version: str,
    repo_root: Union[Path, str],
    started_at: Union[datetime, str],
    findings: Sequence[T] = (),
    *,
    finished_at: Union[datetime, str, None] = None,
    safe_for_completion_reasoning: bool = False,
    error: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RefillScanResult[T]:
    """Build a scan receipt using the repository's current git identity."""

    identity = scan_identity(repo_root)
    return RefillScanResult(
        terminal_reason=terminal_reason,
        scan_mode=scan_mode,
        analyzer_version=analyzer_version,
        repository_id=identity.repository_id,
        tree_id=identity.tree_id,
        started_at=started_at,
        finished_at=finished_at or _utc_now(),
        items=tuple(findings),
        safe_for_completion_reasoning=safe_for_completion_reasoning,
        error=error,
        metadata=metadata or {},
    )


def adapt_legacy_scan_result(
    value: Union[RefillScanResult[T], Sequence[T]],
    *,
    empty_reason: Union[ScanTerminalReason, str],
    scan_mode: Union[ScanMode, str],
    analyzer_version: str,
    repository_id: str,
    tree_id: str,
    started_at: Union[datetime, str, None] = None,
    finished_at: Union[datetime, str, None] = None,
    safe_for_completion_reasoning: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> RefillScanResult[T]:
    """Convert a legacy list result without guessing what an empty list means.

    A non-empty list unambiguously maps to ``GENERATED``.  For an empty list,
    the caller must provide a typed terminal reason appropriate to the callback
    and may explicitly attest that an ``EXHAUSTED`` result is safe completion
    evidence.
    """

    if isinstance(value, RefillScanResult):
        return value
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError("legacy scan callbacks must return a sequence or RefillScanResult")
    started = started_at or _utc_now()
    finished = finished_at or _utc_now()
    items = tuple(value)
    reason: Union[ScanTerminalReason, str] = ScanTerminalReason.GENERATED if items else empty_reason
    return RefillScanResult(
        terminal_reason=reason,
        scan_mode=scan_mode,
        analyzer_version=analyzer_version,
        repository_id=repository_id,
        tree_id=tree_id,
        started_at=started,
        finished_at=finished,
        items=items,
        # Completion safety describes the empty terminal result, not the
        # adapter itself.  A callback which unexpectedly generates records is
        # still a valid GENERATED result and is never completion evidence.
        safe_for_completion_reasoning=(safe_for_completion_reasoning if not items else False),
        metadata=metadata or {"legacy_adapter": True},
    )


@dataclass(frozen=True)
class LegacyScanResultAdapter(Generic[T]):
    """Explicit callable adapter for one legacy list-returning callback."""

    callback: Callable[..., Union[RefillScanResult[T], Sequence[T]]]
    empty_reason: ScanTerminalReason
    scan_mode: str
    analyzer_version: str
    repository_id: str
    tree_id: str
    safe_for_completion_reasoning: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not callable(self.callback):
            raise TypeError("callback must be callable")
        # Validate static adapter configuration immediately, rather than after
        # the first potentially expensive callback execution.
        _nonempty(self.scan_mode, field_name="scan_mode")
        _nonempty(self.analyzer_version, field_name="analyzer_version")
        _nonempty(self.repository_id, field_name="repository_id")
        _nonempty(self.tree_id, field_name="tree_id")
        try:
            reason = (
                self.empty_reason
                if isinstance(self.empty_reason, ScanTerminalReason)
                else ScanTerminalReason(str(self.empty_reason))
            )
        except ValueError as exc:
            raise ValueError(f"unknown empty terminal reason: {self.empty_reason!r}") from exc
        if reason is ScanTerminalReason.GENERATED:
            raise ValueError("empty_reason cannot be generated")
        if self.safe_for_completion_reasoning and reason is not ScanTerminalReason.EXHAUSTED:
            raise ValueError("only exhausted legacy results may be completion-safe")
        object.__setattr__(self, "empty_reason", reason)

    def __call__(self, *args: Any, **kwargs: Any) -> RefillScanResult[T]:
        started_at = _utc_now()
        value = self.callback(*args, **kwargs)
        return adapt_legacy_scan_result(
            value,
            empty_reason=self.empty_reason,
            scan_mode=self.scan_mode,
            analyzer_version=self.analyzer_version,
            repository_id=self.repository_id,
            tree_id=self.tree_id,
            started_at=started_at,
            finished_at=_utc_now(),
            safe_for_completion_reasoning=self.safe_for_completion_reasoning,
            metadata=self.metadata or {"legacy_adapter": True},
        )


def adapt_legacy_scan_callback(
    callback: Callable[..., Union[RefillScanResult[T], Sequence[T]]],
    *,
    empty_reason: Union[ScanTerminalReason, str],
    scan_mode: Union[ScanMode, str],
    analyzer_version: str,
    repo_root: Union[Path, str, None] = None,
    repository_id: str | None = None,
    tree_id: str | None = None,
    safe_for_completion_reasoning: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> LegacyScanResultAdapter[T]:
    """Build an explicit adapter for a legacy list-returning callback.

    Callers may either provide ``repo_root`` or both resolved identity fields.
    Requiring ``empty_reason`` prevents an adapter from guessing that an empty
    legacy collection means exhaustion.
    """

    if repo_root is not None:
        identity = scan_identity(repo_root)
        resolved_repository_id = identity.repository_id
        resolved_tree_id = identity.tree_id
    else:
        resolved_repository_id = _nonempty(repository_id, field_name="repository_id")
        resolved_tree_id = _nonempty(tree_id, field_name="tree_id")
    return LegacyScanResultAdapter(
        callback=callback,
        empty_reason=(
            empty_reason
            if isinstance(empty_reason, ScanTerminalReason)
            else ScanTerminalReason(str(empty_reason))
        ),
        scan_mode=scan_mode.value if isinstance(scan_mode, ScanMode) else str(scan_mode),
        analyzer_version=analyzer_version,
        repository_id=resolved_repository_id,
        tree_id=resolved_tree_id,
        safe_for_completion_reasoning=safe_for_completion_reasoning,
        metadata=metadata or {"legacy_adapter": True},
    )


# Concise names retained for consumers that do not need the refill qualifier.
ScanResult = RefillScanResult
TerminalReason = ScanTerminalReason
RefillScanTerminalReason = ScanTerminalReason
create_scan_result = build_scan_result


__all__ = [
    "CONTRACT_VERSION",
    "LegacyScanResultAdapter",
    "REFILL_SCAN_RESULT_SCHEMA_VERSION",
    "RepositoryTreeIdentity",
    "SCHEMA_VERSION",
    "ScanMode",
    "ScanResult",
    "ScanTerminalReason",
    "TerminalReason",
    "RefillScanResult",
    "RefillScanTerminalReason",
    "adapt_legacy_scan_callback",
    "adapt_legacy_scan_result",
    "build_scan_result",
    "create_scan_result",
    "scan_identity",
]
