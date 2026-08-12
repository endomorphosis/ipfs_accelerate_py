"""Worker evidence factory for planning and doctor (WPD-012).

Interfaces: ``WorkerEvidenceFactory@1``, ``WorkerEvidenceView@1``

Builds one bounded, exact, content-addressed evidence view for the
pre-implementation kernel from live repository-forest bindings.  The view
records:

* forest binding (repository id, forest CID, tree, write alias, policy);
* dirty overlay digest;
* graph and index CIDs when independently available; and
* required-query coverage receipts.

Fail-closed rules:

* Path escapes relative to the bound forest descriptor are rejected.
* Incomplete required queries mark ``coverage_complete=False`` rather than
  inventing graph, index, or query-result facts.
* Source bodies never enter durable view payloads.
* Optional doctor-snapshot / graph / index adapters are nomination-only for
  identity binding; missing adapters leave slots empty and coverage incomplete.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final

from ..proof.formal_verification_contracts import content_identity
from .repository_forest import (
    AuthorityMode,
    ForestPolicy,
    ForestRootSpec,
    RepositoryAuthority,
    RepositoryDescriptor,
    RepositoryForest,
    RepositoryForestError,
    build_repository_forest,
    path_within_repository,
)


# ---------------------------------------------------------------------------
# Interface identity
# ---------------------------------------------------------------------------

WORKER_EVIDENCE_FACTORY_INTERFACE: Final[str] = "WorkerEvidenceFactory@1"
WORKER_EVIDENCE_VIEW_INTERFACE: Final[str] = "WorkerEvidenceView@1"
WORKER_EVIDENCE_FACTORY_VERSION: Final[int] = 1

WORKER_EVIDENCE_VIEW_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/worker-evidence-view@1"
)
WORKER_FOREST_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/worker-forest-binding@1"
)
WORKER_EVIDENCE_QUERY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/worker-evidence-query@1"
)
WORKER_QUERY_COVERAGE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/worker-query-coverage@1"
)

# Objective-heap evidence key for the WPD factories goal packet.
WORKER_EVIDENCE_FACTORY_EVIDENCE: Final[str] = "wpd/evidence-factory@1"

DEFAULT_WRITE_ALIAS: Final[str] = "worker"

# Closed required-query vocabulary for the pre-implementation evidence view.
class WorkerEvidenceQueryKind(str, Enum):
    """Closed query kinds that participate in worker evidence coverage."""

    FOREST_BINDING = "forest_binding"
    DIRTY_OVERLAY = "dirty_overlay"
    GRAPH_INDEX = "graph_index"
    PATH_SCOPE = "path_scope"
    DOCTOR_SNAPSHOT = "doctor_snapshot"


DEFAULT_REQUIRED_QUERIES: Final[tuple[WorkerEvidenceQueryKind, ...]] = (
    WorkerEvidenceQueryKind.FOREST_BINDING,
    WorkerEvidenceQueryKind.DIRTY_OVERLAY,
    WorkerEvidenceQueryKind.GRAPH_INDEX,
)

_BODY_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "ast",
        "ast_body",
        "body",
        "code",
        "content",
        "contents",
        "decoded",
        "file_text",
        "prompt",
        "raw",
        "raw_ast",
        "raw_text",
        "snippet",
        "source",
        "source_body",
        "source_code",
        "source_text",
        "text",
        "transcript",
    }
)

_MAX_PATHS: Final[int] = 4_096
_MAX_NOTES: Final[int] = 64
_MAX_TEXT: Final[int] = 4_096


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class WorkerEvidenceFactoryError(RuntimeError):
    """Fail-closed rejection for an unsafe or incomplete evidence factory run."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "worker_evidence_factory_error",
    ) -> None:
        super().__init__(message)
        self.reason_code = str(reason_code or "worker_evidence_factory_error")


class WorkerEvidencePathEscapeError(WorkerEvidenceFactoryError, ValueError):
    """A requested path escaped the bound forest descriptor root."""

    def __init__(self, message: str) -> None:
        super().__init__(message, reason_code="path_escape")


class WorkerEvidenceBoundsError(WorkerEvidenceFactoryError, ValueError):
    """A count or byte bound was exceeded."""

    def __init__(self, message: str) -> None:
        super().__init__(message, reason_code="bounds_exceeded")


class WorkerEvidenceAuthorityError(WorkerEvidenceFactoryError, ValueError):
    """Forest, tree, or identity bindings failed closed."""

    def __init__(self, message: str, *, reason_code: str = "authority_mismatch") -> None:
        super().__init__(message, reason_code=reason_code)


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class QueryCoverageStatus(str, Enum):
    """Closed coverage outcomes for one evidence query."""

    SATISFIED = "satisfied"
    INCOMPLETE = "incomplete"
    FAILED = "failed"
    NOT_REQUESTED = "not_requested"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text(value: Any, name: str, *, required: bool = True, limit: int = _MAX_TEXT) -> str:
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value
    else:
        raise WorkerEvidenceFactoryError(f"{name} must be a string")
    if "\x00" in text:
        raise WorkerEvidenceFactoryError(f"{name} must not contain NUL")
    stripped = text.strip()
    if required and not stripped:
        raise WorkerEvidenceFactoryError(f"{name} is required")
    if len(stripped.encode("utf-8")) > limit:
        raise WorkerEvidenceBoundsError(f"{name} exceeds its byte bound")
    return stripped


def _optional_text(value: Any, name: str, *, limit: int = _MAX_TEXT) -> str:
    return _text(value, name, required=False, limit=limit)


def _identifier(value: Any, name: str, *, required: bool = True) -> str:
    text = _text(value, name, required=required, limit=2048)
    if text and (text != text.strip() or " " in text or "\t" in text):
        raise WorkerEvidenceFactoryError(
            f"{name} must not contain surrounding whitespace or interior blanks"
        )
    return text


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise WorkerEvidenceFactoryError(f"{name} must be a boolean")
    return value


def _assert_body_free(value: Any, name: str = "record") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).casefold().replace("-", "_")
            if key_text in _BODY_MARKERS:
                raise WorkerEvidenceFactoryError(
                    f"{name} must not carry source bodies via {key!r}"
                )
            _assert_body_free(item, name)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _assert_body_free(item, name)


def _normalize_repo_path(value: Any, name: str = "path") -> str:
    raw = _text(value, name, required=True, limit=1024).replace("\\", "/")
    while raw.startswith("./"):
        raw = raw[2:]
    path = PurePosixPath(raw)
    if (
        path.is_absolute()
        or ".." in path.parts
        or raw in {".", ""}
        or raw.startswith("/")
    ):
        raise WorkerEvidencePathEscapeError(
            f"{name} must be a relative repository path without escape: {value!r}"
        )
    return path.as_posix()


def _query_kind(value: Any) -> WorkerEvidenceQueryKind:
    if isinstance(value, WorkerEvidenceQueryKind):
        return value
    key = str(getattr(value, "value", value) or "").strip().casefold()
    aliases = {
        "forest": WorkerEvidenceQueryKind.FOREST_BINDING,
        "forest_binding": WorkerEvidenceQueryKind.FOREST_BINDING,
        "dirty": WorkerEvidenceQueryKind.DIRTY_OVERLAY,
        "dirty_overlay": WorkerEvidenceQueryKind.DIRTY_OVERLAY,
        "overlay": WorkerEvidenceQueryKind.DIRTY_OVERLAY,
        "graph": WorkerEvidenceQueryKind.GRAPH_INDEX,
        "index": WorkerEvidenceQueryKind.GRAPH_INDEX,
        "graph_index": WorkerEvidenceQueryKind.GRAPH_INDEX,
        "graph/index": WorkerEvidenceQueryKind.GRAPH_INDEX,
        "path": WorkerEvidenceQueryKind.PATH_SCOPE,
        "paths": WorkerEvidenceQueryKind.PATH_SCOPE,
        "path_scope": WorkerEvidenceQueryKind.PATH_SCOPE,
        "doctor": WorkerEvidenceQueryKind.DOCTOR_SNAPSHOT,
        "doctor_snapshot": WorkerEvidenceQueryKind.DOCTOR_SNAPSHOT,
        "snapshot": WorkerEvidenceQueryKind.DOCTOR_SNAPSHOT,
    }
    try:
        return aliases[key]
    except KeyError as exc:
        raise WorkerEvidenceFactoryError(
            f"unsupported worker evidence query kind: {value!r}"
        ) from exc


def _cid_from_object(value: Any, *, attributes: Sequence[str]) -> str:
    """Extract a content identity from a mapping, object, or plain string.

    Never invents an identity: missing or empty values yield the empty string.
    """

    if value is None or value is False:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        for name in attributes:
            candidate = value.get(name)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        return ""
    for name in attributes:
        candidate = getattr(value, name, None)
        if callable(candidate):
            try:
                candidate = candidate()
            except Exception:
                continue
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            payload = to_dict()
        except Exception:
            payload = None
        if isinstance(payload, Mapping):
            return _cid_from_object(payload, attributes=attributes)
    return ""


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkerForestBinding:
    """Exact forest observation roots bound into a worker evidence view."""

    repository_id: str
    repository_forest_cid: str
    git_tree_id: str
    descriptor_cid: str
    write_alias: str
    policy_cid: str = ""
    dirty_overlay_cid: str = ""
    dirty: bool = False
    commit: str = ""
    schema: str = WORKER_FOREST_BINDING_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "repository_id", _identifier(self.repository_id, "repository_id")
        )
        object.__setattr__(
            self,
            "repository_forest_cid",
            _identifier(self.repository_forest_cid, "repository_forest_cid"),
        )
        object.__setattr__(
            self, "git_tree_id", _identifier(self.git_tree_id, "git_tree_id")
        )
        object.__setattr__(
            self, "descriptor_cid", _identifier(self.descriptor_cid, "descriptor_cid")
        )
        object.__setattr__(
            self, "write_alias", _identifier(self.write_alias, "write_alias")
        )
        object.__setattr__(
            self,
            "policy_cid",
            _identifier(self.policy_cid, "policy_cid", required=False),
        )
        object.__setattr__(
            self,
            "dirty_overlay_cid",
            _identifier(self.dirty_overlay_cid, "dirty_overlay_cid", required=False),
        )
        object.__setattr__(self, "dirty", _bool(self.dirty, "dirty"))
        object.__setattr__(
            self, "commit", _identifier(self.commit, "commit", required=False)
        )
        object.__setattr__(
            self,
            "schema",
            _text(self.schema, "schema", required=True),
        )
        if self.schema != WORKER_FOREST_BINDING_SCHEMA:
            raise WorkerEvidenceFactoryError("unsupported worker forest binding schema")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "repository_id": self.repository_id,
            "repository_forest_cid": self.repository_forest_cid,
            "git_tree_id": self.git_tree_id,
            "descriptor_cid": self.descriptor_cid,
            "write_alias": self.write_alias,
            "policy_cid": self.policy_cid,
            "dirty_overlay_cid": self.dirty_overlay_cid,
            "dirty": self.dirty,
            "commit": self.commit,
        }

    @property
    def binding_cid(self) -> str:
        return content_identity(self.to_dict())


@dataclass(frozen=True)
class WorkerEvidenceQueryRecord:
    """One required or optional query coverage receipt (body-free)."""

    kind: WorkerEvidenceQueryKind
    required: bool
    status: QueryCoverageStatus
    result_cid: str = ""
    reason_code: str = ""
    paths: tuple[str, ...] = ()
    schema: str = WORKER_EVIDENCE_QUERY_SCHEMA

    def __post_init__(self) -> None:
        kind = _query_kind(self.kind)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "required", _bool(self.required, "required"))
        status = self.status
        if not isinstance(status, QueryCoverageStatus):
            status = QueryCoverageStatus(str(status))
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "result_cid",
            _identifier(self.result_cid, "result_cid", required=False),
        )
        object.__setattr__(
            self,
            "reason_code",
            _optional_text(self.reason_code, "reason_code", limit=256),
        )
        paths = tuple(
            _normalize_repo_path(item, "paths")
            for item in (self.paths or ())
            if str(item).strip()
        )
        if len(paths) > _MAX_PATHS:
            raise WorkerEvidenceBoundsError("query paths exceed hard bound")
        object.__setattr__(self, "paths", paths)
        object.__setattr__(
            self, "schema", _text(self.schema, "schema", required=True)
        )
        if self.schema != WORKER_EVIDENCE_QUERY_SCHEMA:
            raise WorkerEvidenceFactoryError("unsupported worker evidence query schema")
        # Fail closed: incomplete/failed required queries must not claim a result.
        if (
            self.status
            in {QueryCoverageStatus.INCOMPLETE, QueryCoverageStatus.FAILED}
            and self.result_cid
        ):
            raise WorkerEvidenceFactoryError(
                "incomplete or failed queries must not invent result identities"
            )
        if self.status is QueryCoverageStatus.SATISFIED and not self.result_cid:
            raise WorkerEvidenceFactoryError(
                "satisfied queries require a non-empty result identity"
            )

    @property
    def query_id(self) -> str:
        return content_identity(
            {
                "schema": self.schema + "/identity",
                "kind": self.kind.value,
                "required": self.required,
                "status": self.status.value,
                "result_cid": self.result_cid,
                "reason_code": self.reason_code,
                "paths": list(self.paths),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "query_id": self.query_id,
            "kind": self.kind.value,
            "required": self.required,
            "status": self.status.value,
            "result_cid": self.result_cid,
            "reason_code": self.reason_code,
            "paths": list(self.paths),
        }


@dataclass(frozen=True)
class WorkerQueryCoverage:
    """Aggregate coverage receipt over the worker evidence query portfolio."""

    records: tuple[WorkerEvidenceQueryRecord, ...]
    coverage_complete: bool
    required_kinds: tuple[str, ...] = ()
    satisfied_kinds: tuple[str, ...] = ()
    incomplete_kinds: tuple[str, ...] = ()
    schema: str = WORKER_QUERY_COVERAGE_SCHEMA

    def __post_init__(self) -> None:
        records = tuple(self.records or ())
        object.__setattr__(self, "records", records)
        object.__setattr__(
            self, "coverage_complete", _bool(self.coverage_complete, "coverage_complete")
        )
        required = tuple(
            sorted({item.kind.value for item in records if item.required})
        )
        satisfied = tuple(
            sorted(
                {
                    item.kind.value
                    for item in records
                    if item.required and item.status is QueryCoverageStatus.SATISFIED
                }
            )
        )
        incomplete = tuple(
            sorted(
                {
                    item.kind.value
                    for item in records
                    if item.required
                    and item.status
                    in {
                        QueryCoverageStatus.INCOMPLETE,
                        QueryCoverageStatus.FAILED,
                        QueryCoverageStatus.NOT_REQUESTED,
                    }
                }
            )
        )
        object.__setattr__(self, "required_kinds", required)
        object.__setattr__(self, "satisfied_kinds", satisfied)
        object.__setattr__(self, "incomplete_kinds", incomplete)
        object.__setattr__(
            self, "schema", _text(self.schema, "schema", required=True)
        )
        if self.schema != WORKER_QUERY_COVERAGE_SCHEMA:
            raise WorkerEvidenceFactoryError("unsupported worker query coverage schema")
        # Structural invariant: coverage_complete iff no incomplete required kinds.
        expected_complete = not incomplete
        if self.coverage_complete != expected_complete:
            raise WorkerEvidenceFactoryError(
                "coverage_complete must reflect required-query satisfaction exactly"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "coverage_complete": self.coverage_complete,
            "required_kinds": list(self.required_kinds),
            "satisfied_kinds": list(self.satisfied_kinds),
            "incomplete_kinds": list(self.incomplete_kinds),
            "records": [item.to_dict() for item in self.records],
        }

    @property
    def coverage_cid(self) -> str:
        return content_identity(self.to_dict())


@dataclass(frozen=True)
class WorkerEvidenceView:
    """Bounded exact evidence view for planning and doctor pre-implementation use."""

    forest_binding: WorkerForestBinding
    query_coverage: WorkerQueryCoverage
    graph_cid: str = ""
    index_cid: str = ""
    doctor_snapshot_cid: str = ""
    admitted_paths: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    factory_interface: str = WORKER_EVIDENCE_FACTORY_INTERFACE
    view_interface: str = WORKER_EVIDENCE_VIEW_INTERFACE
    schema: str = WORKER_EVIDENCE_VIEW_SCHEMA
    extra_bindings: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.forest_binding, WorkerForestBinding):
            raise WorkerEvidenceFactoryError(
                "forest_binding must be WorkerForestBinding"
            )
        if not isinstance(self.query_coverage, WorkerQueryCoverage):
            raise WorkerEvidenceFactoryError(
                "query_coverage must be WorkerQueryCoverage"
            )
        object.__setattr__(
            self, "graph_cid", _identifier(self.graph_cid, "graph_cid", required=False)
        )
        object.__setattr__(
            self, "index_cid", _identifier(self.index_cid, "index_cid", required=False)
        )
        object.__setattr__(
            self,
            "doctor_snapshot_cid",
            _identifier(
                self.doctor_snapshot_cid, "doctor_snapshot_cid", required=False
            ),
        )
        paths = tuple(
            _normalize_repo_path(item, "admitted_paths")
            for item in (self.admitted_paths or ())
            if str(item).strip()
        )
        if len(paths) > _MAX_PATHS:
            raise WorkerEvidenceBoundsError("admitted_paths exceeds hard bound")
        object.__setattr__(self, "admitted_paths", paths)
        notes = tuple(
            _optional_text(item, "notes", limit=512)
            for item in (self.notes or ())
            if str(item).strip()
        )[:_MAX_NOTES]
        object.__setattr__(self, "notes", notes)
        object.__setattr__(
            self,
            "factory_interface",
            _text(self.factory_interface, "factory_interface", required=True),
        )
        object.__setattr__(
            self,
            "view_interface",
            _text(self.view_interface, "view_interface", required=True),
        )
        object.__setattr__(self, "schema", _text(self.schema, "schema", required=True))
        if self.schema != WORKER_EVIDENCE_VIEW_SCHEMA:
            raise WorkerEvidenceFactoryError("unsupported worker evidence view schema")
        if self.factory_interface != WORKER_EVIDENCE_FACTORY_INTERFACE:
            raise WorkerEvidenceFactoryError("unsupported worker evidence factory interface")
        if self.view_interface != WORKER_EVIDENCE_VIEW_INTERFACE:
            raise WorkerEvidenceFactoryError("unsupported worker evidence view interface")
        extras = {
            str(key): _identifier(value, f"extra_bindings[{key}]", required=False)
            for key, value in dict(self.extra_bindings or {}).items()
            if str(key).strip()
        }
        object.__setattr__(self, "extra_bindings", MappingProxyType(extras))
        # Incomplete graph_index coverage must not claim a complete CID pair.
        if (
            WorkerEvidenceQueryKind.GRAPH_INDEX.value
            in self.query_coverage.incomplete_kinds
            and self.graph_cid
            and self.index_cid
        ):
            raise WorkerEvidenceFactoryError(
                "incomplete graph_index coverage must not claim both graph and index CIDs"
            )
        payload = self.to_dict()
        _assert_body_free(payload, "worker_evidence_view")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "factory_interface": self.factory_interface,
            "view_interface": self.view_interface,
            "forest_binding": self.forest_binding.to_dict(),
            "dirty_overlay_cid": self.forest_binding.dirty_overlay_cid,
            "graph_cid": self.graph_cid,
            "index_cid": self.index_cid,
            "doctor_snapshot_cid": self.doctor_snapshot_cid,
            "admitted_paths": list(self.admitted_paths),
            "query_coverage": self.query_coverage.to_dict(),
            "coverage_complete": self.query_coverage.coverage_complete,
            "notes": list(self.notes),
            "extra_bindings": dict(self.extra_bindings),
        }

    @property
    def view_cid(self) -> str:
        """Content-addressed identity of this evidence view."""

        return content_identity(self.to_dict())

    @property
    def coverage_complete(self) -> bool:
        return self.query_coverage.coverage_complete

    @property
    def repository_forest_cid(self) -> str:
        return self.forest_binding.repository_forest_cid

    @property
    def dirty_overlay_cid(self) -> str:
        return self.forest_binding.dirty_overlay_cid

    def to_implementation_forest_roots(self) -> dict[str, str]:
        """Projection compatible with ``ImplementationForestRoots`` fields."""

        return {
            "repository_id": self.forest_binding.repository_id,
            "repository_forest_cid": self.forest_binding.repository_forest_cid,
            "git_tree_id": self.forest_binding.git_tree_id,
            "policy_root": self.forest_binding.policy_cid
            or self.forest_binding.repository_forest_cid,
            "dirty_overlay_cid": self.forest_binding.dirty_overlay_cid,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_single_repository_forest(
    checkout_root: str | Path,
    *,
    alias: str = DEFAULT_WRITE_ALIAS,
    sole_write_alias: str | None = None,
) -> RepositoryForest:
    """Build a one-root forest with the checkout as the sole write root."""

    write_alias = str(sole_write_alias or alias).strip() or DEFAULT_WRITE_ALIAS
    root = Path(checkout_root)
    if not root.exists() or not root.is_dir():
        raise WorkerEvidenceAuthorityError(
            f"checkout root is unavailable: {checkout_root}",
            reason_code="missing_root",
        )
    try:
        return build_repository_forest(
            ForestPolicy(
                roots=(
                    ForestRootSpec(
                        alias=write_alias,
                        root_path=root,
                        authority=RepositoryAuthority(
                            mode=AuthorityMode.READ_WRITE.value
                        ),
                        required=True,
                    ),
                ),
                sole_write_alias=write_alias,
            )
        )
    except RepositoryForestError as exc:
        raise WorkerEvidenceAuthorityError(
            f"repository forest construction failed: {exc.reason_code}",
            reason_code=exc.reason_code,
        ) from exc


def _write_descriptor(forest: RepositoryForest) -> RepositoryDescriptor:
    try:
        return forest.write_descriptor()
    except RepositoryForestError as exc:
        raise WorkerEvidenceAuthorityError(
            f"write descriptor unavailable: {exc.reason_code}",
            reason_code=exc.reason_code,
        ) from exc


def _binding_from_forest(forest: RepositoryForest) -> WorkerForestBinding:
    descriptor = _write_descriptor(forest)
    return WorkerForestBinding(
        repository_id=descriptor.repository_id,
        repository_forest_cid=forest.forest_id,
        git_tree_id=descriptor.tree,
        descriptor_cid=descriptor.descriptor_cid,
        write_alias=descriptor.alias,
        policy_cid=forest.policy_cid or forest.forest_id,
        dirty_overlay_cid=descriptor.dirty_overlay_digest,
        dirty=bool(descriptor.dirty),
        commit=descriptor.commit,
    )


def _admit_paths(
    descriptor: RepositoryDescriptor,
    paths: Sequence[str],
) -> tuple[str, ...]:
    admitted: list[str] = []
    for raw in paths:
        if not str(raw).strip():
            continue
        # Lexical escape first (fast fail for ``..`` / absolute forms).
        try:
            normalized = _normalize_repo_path(raw, "path")
        except WorkerEvidencePathEscapeError:
            raise
        try:
            path_within_repository(descriptor, normalized, require_existing=False)
        except RepositoryForestError as exc:
            if exc.reason_code in {"path_escape", "symlink_escape"}:
                raise WorkerEvidencePathEscapeError(
                    f"path escapes repository descriptor root: {raw!r}"
                ) from exc
            raise WorkerEvidenceAuthorityError(
                f"path could not be admitted under forest descriptor: {raw!r}",
                reason_code=exc.reason_code,
            ) from exc
        admitted.append(normalized)
    if len(admitted) > _MAX_PATHS:
        raise WorkerEvidenceBoundsError("admitted path count exceeds hard bound")
    # Stable unique order.
    return tuple(sorted(dict.fromkeys(admitted)))


def _record(
    kind: WorkerEvidenceQueryKind,
    *,
    required: bool,
    status: QueryCoverageStatus,
    result_cid: str = "",
    reason_code: str = "",
    paths: Sequence[str] = (),
) -> WorkerEvidenceQueryRecord:
    return WorkerEvidenceQueryRecord(
        kind=kind,
        required=required,
        status=status,
        result_cid=result_cid,
        reason_code=reason_code,
        paths=tuple(paths),
    )


class WorkerEvidenceFactory:
    """Production factory that binds live forest evidence for planner/doctor.

    Interface: ``WorkerEvidenceFactory@1``
    """

    INTERFACE: Final[str] = WORKER_EVIDENCE_FACTORY_INTERFACE
    VERSION: Final[int] = WORKER_EVIDENCE_FACTORY_VERSION

    def __init__(
        self,
        *,
        write_alias: str = DEFAULT_WRITE_ALIAS,
        default_required_queries: Sequence[WorkerEvidenceQueryKind | str]
        | None = None,
        graph_provider: Callable[[RepositoryForest, RepositoryDescriptor], Any]
        | None = None,
        index_provider: Callable[[RepositoryForest, RepositoryDescriptor], Any]
        | None = None,
        doctor_snapshot_provider: Callable[
            [RepositoryForest, RepositoryDescriptor], Any
        ]
        | None = None,
    ) -> None:
        self._write_alias = str(write_alias or DEFAULT_WRITE_ALIAS).strip()
        if not self._write_alias:
            raise WorkerEvidenceFactoryError("write_alias is required")
        queries = (
            DEFAULT_REQUIRED_QUERIES
            if default_required_queries is None
            else tuple(_query_kind(item) for item in default_required_queries)
        )
        if not queries:
            raise WorkerEvidenceFactoryError("default required queries must not be empty")
        self._default_required_queries = queries
        self._graph_provider = graph_provider
        self._index_provider = index_provider
        self._doctor_snapshot_provider = doctor_snapshot_provider
        self._last_view: WorkerEvidenceView | None = None

    @property
    def last_view(self) -> WorkerEvidenceView | None:
        return self._last_view

    def build(
        self,
        checkout_root: str | Path | None = None,
        *,
        forest: RepositoryForest | None = None,
        required_queries: Sequence[WorkerEvidenceQueryKind | str] | None = None,
        paths: Sequence[str] = (),
        graph_cid: str = "",
        index_cid: str = "",
        doctor_snapshot_cid: str = "",
        doctor_snapshot: Any = None,
        graph: Any = None,
        index: Any = None,
        notes: Sequence[str] = (),
        require_doctor_snapshot: bool = False,
    ) -> WorkerEvidenceView:
        """Build a content-addressed worker evidence view.

        Parameters
        ----------
        checkout_root:
            Live Git checkout used when ``forest`` is not supplied.
        forest:
            Pre-built :class:`RepositoryForest` observation authority.
        required_queries:
            Closed query kinds that must be satisfied for complete coverage.
            Defaults to forest binding, dirty overlay, and graph/index.
        paths:
            Repository-relative paths to admit under the write descriptor.
            Escapes raise :class:`WorkerEvidencePathEscapeError`.
        graph_cid / index_cid / doctor_snapshot_cid:
            Explicit content identities.  Empty values are never replaced with
            invented digests; missing required identities leave coverage false.
        graph / index / doctor_snapshot:
            Optional adapter objects or mappings whose known identity attributes
            may supply CIDs when the explicit parameters are empty.
        """

        bound_forest = self._resolve_forest(checkout_root=checkout_root, forest=forest)
        descriptor = _write_descriptor(bound_forest)
        binding = _binding_from_forest(bound_forest)
        admitted_paths = _admit_paths(descriptor, paths)

        required = tuple(
            dict.fromkeys(
                _query_kind(item)
                for item in (
                    self._default_required_queries
                    if required_queries is None
                    else required_queries
                )
            )
        )
        if require_doctor_snapshot and (
            WorkerEvidenceQueryKind.DOCTOR_SNAPSHOT not in required
        ):
            required = required + (WorkerEvidenceQueryKind.DOCTOR_SNAPSHOT,)

        resolved_graph_cid = _identifier(
            graph_cid
            or _cid_from_object(
                graph,
                attributes=("graph_cid", "content_id", "content_cid", "cid", "id"),
            )
            or self._invoke_provider(
                self._graph_provider,
                bound_forest,
                descriptor,
                attributes=("graph_cid", "content_id", "content_cid", "cid", "id"),
            ),
            "graph_cid",
            required=False,
        )
        resolved_index_cid = _identifier(
            index_cid
            or _cid_from_object(
                index,
                attributes=(
                    "index_id",
                    "index_cid",
                    "ast_index_id",
                    "content_id",
                    "content_cid",
                    "cid",
                    "id",
                ),
            )
            or self._invoke_provider(
                self._index_provider,
                bound_forest,
                descriptor,
                attributes=(
                    "index_id",
                    "index_cid",
                    "ast_index_id",
                    "content_id",
                    "content_cid",
                    "cid",
                    "id",
                ),
            ),
            "index_cid",
            required=False,
        )
        resolved_doctor_cid = _identifier(
            doctor_snapshot_cid
            or _cid_from_object(
                doctor_snapshot,
                attributes=(
                    "snapshot_cid",
                    "snapshot_id",
                    "content_id",
                    "content_cid",
                    "cid",
                    "id",
                ),
            )
            or self._invoke_provider(
                self._doctor_snapshot_provider,
                bound_forest,
                descriptor,
                attributes=(
                    "snapshot_cid",
                    "snapshot_id",
                    "content_id",
                    "content_cid",
                    "cid",
                    "id",
                ),
            ),
            "doctor_snapshot_cid",
            required=False,
        )

        records: list[WorkerEvidenceQueryRecord] = []
        view_notes: list[str] = [
            _optional_text(item, "notes", limit=512)
            for item in notes
            if str(item).strip()
        ]
        view_notes.append("target_code_not_imported")
        view_notes.append("source_bodies_excluded")

        for kind in required:
            if kind is WorkerEvidenceQueryKind.FOREST_BINDING:
                records.append(
                    _record(
                        kind,
                        required=True,
                        status=QueryCoverageStatus.SATISFIED,
                        result_cid=binding.binding_cid,
                        reason_code="forest_bound",
                    )
                )
            elif kind is WorkerEvidenceQueryKind.DIRTY_OVERLAY:
                if binding.dirty_overlay_cid:
                    records.append(
                        _record(
                            kind,
                            required=True,
                            status=QueryCoverageStatus.SATISFIED,
                            result_cid=binding.dirty_overlay_cid,
                            reason_code=(
                                "dirty_overlay_bound"
                                if binding.dirty
                                else "clean_overlay_bound"
                            ),
                        )
                    )
                else:
                    # Never invent an overlay digest.
                    records.append(
                        _record(
                            kind,
                            required=True,
                            status=QueryCoverageStatus.INCOMPLETE,
                            reason_code="dirty_overlay_unavailable",
                        )
                    )
                    view_notes.append("dirty_overlay_incomplete")
            elif kind is WorkerEvidenceQueryKind.GRAPH_INDEX:
                if resolved_graph_cid and resolved_index_cid:
                    combined = content_identity(
                        {
                            "schema": WORKER_EVIDENCE_QUERY_SCHEMA + "/graph-index",
                            "graph_cid": resolved_graph_cid,
                            "index_cid": resolved_index_cid,
                        }
                    )
                    records.append(
                        _record(
                            kind,
                            required=True,
                            status=QueryCoverageStatus.SATISFIED,
                            result_cid=combined,
                            reason_code="graph_and_index_bound",
                        )
                    )
                else:
                    # Incomplete: leave graph/index CIDs empty on the view when
                    # the required pair is not fully available — do not invent.
                    missing: list[str] = []
                    if not resolved_graph_cid:
                        missing.append("graph_cid")
                    if not resolved_index_cid:
                        missing.append("index_cid")
                    records.append(
                        _record(
                            kind,
                            required=True,
                            status=QueryCoverageStatus.INCOMPLETE,
                            reason_code="missing_" + "_and_".join(missing),
                        )
                    )
                    view_notes.append("graph_index_incomplete")
                    # Clear partial claims so incomplete coverage cannot be
                    # mistaken for complete graph/index facts.
                    resolved_graph_cid = ""
                    resolved_index_cid = ""
            elif kind is WorkerEvidenceQueryKind.PATH_SCOPE:
                if not paths:
                    records.append(
                        _record(
                            kind,
                            required=True,
                            status=QueryCoverageStatus.INCOMPLETE,
                            reason_code="no_paths_requested",
                        )
                    )
                    view_notes.append("path_scope_incomplete")
                else:
                    scope_cid = content_identity(
                        {
                            "schema": WORKER_EVIDENCE_QUERY_SCHEMA + "/path-scope",
                            "paths": list(admitted_paths),
                            "descriptor_cid": binding.descriptor_cid,
                        }
                    )
                    records.append(
                        _record(
                            kind,
                            required=True,
                            status=QueryCoverageStatus.SATISFIED,
                            result_cid=scope_cid,
                            reason_code="paths_admitted",
                            paths=admitted_paths,
                        )
                    )
            elif kind is WorkerEvidenceQueryKind.DOCTOR_SNAPSHOT:
                if resolved_doctor_cid:
                    records.append(
                        _record(
                            kind,
                            required=True,
                            status=QueryCoverageStatus.SATISFIED,
                            result_cid=resolved_doctor_cid,
                            reason_code="doctor_snapshot_bound",
                        )
                    )
                else:
                    records.append(
                        _record(
                            kind,
                            required=True,
                            status=QueryCoverageStatus.INCOMPLETE,
                            reason_code="doctor_snapshot_unavailable",
                        )
                    )
                    view_notes.append("doctor_snapshot_incomplete")
            else:  # pragma: no cover - closed enum
                raise WorkerEvidenceFactoryError(
                    f"unhandled required query kind: {kind!r}"
                )

        # Optional path admission record when paths were supplied but path_scope
        # was not required — still records admitted paths without inventing.
        if paths and WorkerEvidenceQueryKind.PATH_SCOPE not in required:
            scope_cid = content_identity(
                {
                    "schema": WORKER_EVIDENCE_QUERY_SCHEMA + "/path-scope",
                    "paths": list(admitted_paths),
                    "descriptor_cid": binding.descriptor_cid,
                }
            )
            records.append(
                _record(
                    WorkerEvidenceQueryKind.PATH_SCOPE,
                    required=False,
                    status=QueryCoverageStatus.SATISFIED,
                    result_cid=scope_cid,
                    reason_code="paths_admitted",
                    paths=admitted_paths,
                )
            )

        incomplete = any(
            item.required
            and item.status
            in {
                QueryCoverageStatus.INCOMPLETE,
                QueryCoverageStatus.FAILED,
                QueryCoverageStatus.NOT_REQUESTED,
            }
            for item in records
        )
        coverage = WorkerQueryCoverage(
            records=tuple(records),
            coverage_complete=not incomplete,
        )
        view = WorkerEvidenceView(
            forest_binding=binding,
            query_coverage=coverage,
            graph_cid=resolved_graph_cid,
            index_cid=resolved_index_cid,
            doctor_snapshot_cid=resolved_doctor_cid,
            admitted_paths=admitted_paths,
            notes=tuple(dict.fromkeys(item for item in view_notes if item)),
        )
        self._last_view = view
        return view

    # Back-compat / ergonomic alias used by planner-style call sites.
    def analyze(self, checkout_root: str | Path, **kwargs: Any) -> WorkerEvidenceView:
        return self.build(checkout_root, **kwargs)

    def resolve_path(
        self,
        path: str,
        *,
        forest: RepositoryForest | None = None,
        checkout_root: str | Path | None = None,
    ) -> Path:
        """Resolve a repository-relative path under the write descriptor."""

        bound = self._resolve_forest(checkout_root=checkout_root, forest=forest)
        descriptor = _write_descriptor(bound)
        normalized = _normalize_repo_path(path, "path")
        try:
            return path_within_repository(descriptor, normalized)
        except RepositoryForestError as exc:
            if exc.reason_code in {"path_escape", "symlink_escape"}:
                raise WorkerEvidencePathEscapeError(
                    f"path escapes repository descriptor root: {path!r}"
                ) from exc
            raise WorkerEvidenceAuthorityError(
                f"path resolution failed: {path!r}",
                reason_code=exc.reason_code,
            ) from exc

    def _resolve_forest(
        self,
        *,
        checkout_root: str | Path | None,
        forest: RepositoryForest | None,
    ) -> RepositoryForest:
        if forest is not None:
            if not isinstance(forest, RepositoryForest):
                raise WorkerEvidenceAuthorityError(
                    "forest must be a RepositoryForest",
                    reason_code="invalid_forest",
                )
            return forest
        if checkout_root is None:
            raise WorkerEvidenceAuthorityError(
                "checkout_root or forest is required",
                reason_code="missing_root",
            )
        return build_single_repository_forest(
            checkout_root,
            alias=self._write_alias,
        )

    def _invoke_provider(
        self,
        provider: Callable[[RepositoryForest, RepositoryDescriptor], Any] | None,
        forest: RepositoryForest,
        descriptor: RepositoryDescriptor,
        *,
        attributes: Sequence[str],
    ) -> str:
        if provider is None:
            return ""
        try:
            result = provider(forest, descriptor)
        except Exception as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            # Provider failure is incomplete coverage, not a hard factory abort,
            # unless the provider itself raised a path-escape / authority error.
            if isinstance(exc, WorkerEvidenceFactoryError):
                raise
            return ""
        return _cid_from_object(result, attributes=attributes)


def build_worker_evidence_factory(
    *,
    write_alias: str = DEFAULT_WRITE_ALIAS,
    default_required_queries: Sequence[WorkerEvidenceQueryKind | str] | None = None,
    graph_provider: Callable[[RepositoryForest, RepositoryDescriptor], Any]
    | None = None,
    index_provider: Callable[[RepositoryForest, RepositoryDescriptor], Any]
    | None = None,
    doctor_snapshot_provider: Callable[
        [RepositoryForest, RepositoryDescriptor], Any
    ]
    | None = None,
) -> WorkerEvidenceFactory:
    """Construct a production worker evidence factory."""

    return WorkerEvidenceFactory(
        write_alias=write_alias,
        default_required_queries=default_required_queries,
        graph_provider=graph_provider,
        index_provider=index_provider,
        doctor_snapshot_provider=doctor_snapshot_provider,
    )


def build_worker_evidence_view(
    checkout_root: str | Path | None = None,
    **kwargs: Any,
) -> WorkerEvidenceView:
    """One-shot helper that builds a factory and returns a view."""

    factory = build_worker_evidence_factory(
        write_alias=str(kwargs.pop("write_alias", DEFAULT_WRITE_ALIAS)),
    )
    return factory.build(checkout_root, **kwargs)


__all__ = [
    "DEFAULT_REQUIRED_QUERIES",
    "DEFAULT_WRITE_ALIAS",
    "QueryCoverageStatus",
    "WORKER_EVIDENCE_FACTORY_EVIDENCE",
    "WORKER_EVIDENCE_FACTORY_INTERFACE",
    "WORKER_EVIDENCE_FACTORY_VERSION",
    "WORKER_EVIDENCE_QUERY_SCHEMA",
    "WORKER_EVIDENCE_VIEW_INTERFACE",
    "WORKER_EVIDENCE_VIEW_SCHEMA",
    "WORKER_FOREST_BINDING_SCHEMA",
    "WORKER_QUERY_COVERAGE_SCHEMA",
    "WorkerEvidenceFactory",
    "WorkerEvidenceFactoryError",
    "WorkerEvidenceAuthorityError",
    "WorkerEvidenceBoundsError",
    "WorkerEvidencePathEscapeError",
    "WorkerEvidenceQueryKind",
    "WorkerEvidenceQueryRecord",
    "WorkerEvidenceView",
    "WorkerForestBinding",
    "WorkerQueryCoverage",
    "build_single_repository_forest",
    "build_worker_evidence_factory",
    "build_worker_evidence_view",
]
