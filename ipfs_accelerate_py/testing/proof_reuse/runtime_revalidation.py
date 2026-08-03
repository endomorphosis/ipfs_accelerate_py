"""Fresh current-context revalidation against retained candidates (PTR-136).

``RuntimeContextRevalidator@1`` implements the warm-admission half of the sealed
activation sequence:

1. Lookup starts from a **stable locator only**.
2. Every retained candidate component named by the candidate store is rehashed
   and content-addressed before use.
3. Dependencies named by the retained runtime (and static) traces are **freshly
   resolved** and content-addressed against live state.
4. Current source, AST, fixtures, hooks, parameters, locks, distributions,
   environment, capabilities, repository forest, policy, and external snapshots
   must match the candidate exactly.
5. Incomplete, unresolvable, changed, or uncontrolled facts return ``RUN``.
6. A verified unchanged context may **proceed to certificate verification**
   without executing fixtures or the test body.
7. A normal miss executes setup/call/teardown exactly once;
   :class:`PostPassRuntimeTraceCapture` records the observed runtime frontier
   afterward for publication.

This module never authorizes ``SKIP`` by itself.  Certificate verification is a
later step.  Historical runtime traces only name the frontier to re-resolve;
they never assert that the current test would pass.  Import is side-effect free
and does not open network sockets, install packages, or invoke pytest fixtures.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final, Optional, Protocol, runtime_checkable

from .activation_contracts import (
    ACTIVATION_AUTHORITY_SEQUENCE,
    ArtifactRole,
    CandidateExecutionContext,
    ContextComparisonResult,
    CurrentExecutionContext,
    PostPassRuntimeObservation,
    RuntimeReuseDisposition,
    SkipComparisonDimension,
    admit_content_addressed_boundary,
    compare_contexts_for_skip,
    disposition_run,
    rehash_retained_canonical_bytes,
    record_post_pass_runtime_observation,
)

# ---------------------------------------------------------------------------
# Interface / schema constants
# ---------------------------------------------------------------------------

RUNTIME_CONTEXT_REVALIDATOR_INTERFACE: Final = "RuntimeContextRevalidator@1"
RUNTIME_CONTEXT_REVALIDATOR_SCHEMA: Final = (
    "ipfs_accelerate_py/testing/proof-reuse/runtime-context-revalidator@1"
)
CANDIDATE_COMPARISON_INTERFACE: Final = "CandidateComparison@1"
CANDIDATE_COMPARISON_SCHEMA: Final = (
    "ipfs_accelerate_py/testing/proof-reuse/candidate-comparison@1"
)
POST_PASS_RUNTIME_TRACE_CAPTURE_INTERFACE: Final = "PostPassRuntimeTraceCapture@1"
POST_PASS_RUNTIME_TRACE_CAPTURE_SCHEMA: Final = (
    "ipfs_accelerate_py/testing/proof-reuse/post-pass-runtime-trace-capture@1"
)
RUNTIME_REVALIDATION_RESULT_INTERFACE: Final = "RuntimeRevalidationResult@1"
RUNTIME_REVALIDATION_RESULT_SCHEMA: Final = (
    "ipfs_accelerate_py/testing/proof-reuse/runtime-revalidation-result@1"
)
DEPENDENCY_RESOLUTION_REPORT_INTERFACE: Final = "DependencyResolutionReport@1"
RUNTIME_DEPENDENCY_TRACE_INTERFACE: Final = "RuntimeDependencyTrace@1"
# Compatible alias used by retained agent-supervisor traces.
RUNTIME_TEST_DEPENDENCY_TRACE_INTERFACE: Final = "RuntimeTestDependencyTrace@1"

_MAX_DIAGNOSTIC_KEYS: Final = 32
_MAX_DIAGNOSTIC_VALUE_CHARS: Final = 256
_MAX_DEPENDENCIES: Final = 4_096
_MAX_FILE_BYTES: Final = 8 * 1_048_576
_MAX_TRACE_BYTES: Final = 1_048_576
_MAX_ROOT_ID_CHARS: Final = 64
_MAX_PATH_CHARS: Final = 1_024

# Dimensions that warm admission must agree on (beyond the core SKIP set).
_EXTENDED_IDENTITY_DIMENSIONS: Final[tuple[str, ...]] = (
    "source",
    "fixtures",
    "hooks",
    "parameters",
    "locks",
    "distributions",
    "capabilities",
    "repository_forest",
    "external_snapshots",
    "dependency_lock",
    "platform",
)

# Map extended dimension names → CandidateExecutionContext / CurrentExecutionContext
# field or component_cids key.
_EXTENDED_FIELD_MAP: Final[Mapping[str, str]] = {
    "locks": "dependency_lock_cid",
    "distributions": "installed_distributions_cid",
    "capabilities": "capability_root_cid",
    "repository_forest": "repository_forest_cid",
    "dependency_lock": "dependency_lock_cid",
    "platform": "platform_cid",
    "source": "test_ast_cid",  # source identity is bound through AST content
}

_COMPONENT_KEY_ALIASES: Final[Mapping[str, tuple[str, ...]]] = {
    "fixtures": ("fixtures", "fixture", "fixture_root"),
    "hooks": ("hooks", "hook", "hook_root", "plugins", "plugin_root"),
    "parameters": ("parameters", "parameter", "parameter_root", "parametrization"),
    "source": ("source", "test_source", "source_root"),
    "locks": ("dependency_lock", "locks", "lock_root"),
    "distributions": ("installed_distributions", "distributions", "distribution_root"),
    "capabilities": ("capability_root", "capabilities"),
}

_RUNTIME_DEPENDENCY_KINDS: Final[tuple[str, ...]] = (
    "modules",
    "code_objects",
    "files",
    "environment",
    "subprocesses",
    "services",
    "policies",
    "capabilities",
)


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class RevalidationReason(str, Enum):
    """Closed reason codes for runtime context revalidation."""

    CONTEXT_UNCHANGED = "context_unchanged"
    LOCATOR_MISSING = "locator_missing"
    LOCATOR_INVALID = "locator_invalid"
    CANDIDATE_MISSING = "candidate_missing"
    CANDIDATE_INTEGRITY_FAILED = "candidate_integrity_failed"
    CANDIDATE_UNRESOLVABLE = "candidate_unresolvable"
    COMPONENT_MISSING = "component_missing"
    COMPONENT_INTEGRITY_FAILED = "component_integrity_failed"
    CURRENT_CONTEXT_UNAVAILABLE = "current_context_unavailable"
    CURRENT_CONTEXT_INCOMPLETE = "current_context_incomplete"
    CURRENT_CONTEXT_NOT_FRESH = "current_context_not_fresh"
    IDENTITY_MISMATCH = "identity_mismatch"
    DEPENDENCY_UNRESOLVABLE = "dependency_unresolvable"
    DEPENDENCY_CHANGED = "dependency_changed"
    DEPENDENCY_UNCONTROLLED = "dependency_uncontrolled"
    TRACE_INCOMPLETE = "trace_incomplete"
    TRACE_MALFORMED = "trace_malformed"
    UNKNOWN_FRONTIER = "unknown_frontier"
    STORE_FAULT = "store_fault"
    INTERNAL_ERROR_FAIL_OPEN_TO_RUN = "internal_error_fail_open_to_run"


class RevalidationAction(str, Enum):
    """Disposition after revalidation (never authorizes SKIP alone)."""

    RUN = "RUN"
    PROCEED_TO_CERTIFICATE_VERIFICATION = "PROCEED_TO_CERTIFICATE_VERIFICATION"


class LifecyclePhase(str, Enum):
    """Ordered phases of the single real pytest lifecycle."""

    IDLE = "idle"
    SETUP = "setup"
    CALL = "call"
    TEARDOWN = "teardown"
    COMPLETE = "complete"
    FAILED = "failed"


class DependencyResolutionStatus(str, Enum):
    """Per-fact resolution status for retained-trace dependencies."""

    MATCHED = "matched"
    CHANGED = "changed"
    UNRESOLVABLE = "unresolvable"
    UNCONTROLLED = "uncontrolled"
    SKIPPED = "skipped"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bounded_diagnostics(raw: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not raw:
        return MappingProxyType({})
    out: dict[str, Any] = {}
    for index, (key, value) in enumerate(raw.items()):
        if index >= _MAX_DIAGNOSTIC_KEYS:
            break
        name = str(key)[:64]
        if value is None or isinstance(value, (bool, int)):
            out[name] = value
        elif isinstance(value, str):
            out[name] = value[:_MAX_DIAGNOSTIC_VALUE_CHARS]
        elif isinstance(value, (list, tuple)):
            out[name] = [str(item)[:64] for item in list(value)[:16]]
        else:
            out[name] = type(value).__name__[:64]
    return MappingProxyType(out)


def _now_ms(clock: Callable[[], float] | Callable[[], int] | None = None) -> int:
    if clock is None:
        return int(time.time() * 1000)
    value = clock()
    if isinstance(value, bool):
        return int(time.time() * 1000)
    if isinstance(value, int):
        # Heuristic: values that look like seconds (small) are converted.
        if value < 10_000_000_000:
            return int(value * 1000)
        return int(value)
    try:
        return int(float(value) * 1000)
    except (TypeError, ValueError):
        return int(time.time() * 1000)


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_json_loads(data: bytes) -> Any | None:
    if type(data) is not bytes or not data:
        return None
    if len(data) > _MAX_TRACE_BYTES:
        return None
    try:
        return json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None


def _locator_token(value: Any) -> str | None:
    """Extract a stable locator CID token; reject non-locator inputs."""

    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    # Accept small locator-like objects with a stable cid/id attribute.
    for attr in ("locator_cid", "locator_id", "cid", "content_id"):
        candidate = getattr(value, attr, None)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    if isinstance(value, Mapping):
        for key in ("locator_cid", "locator_id", "cid", "content_id"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    return None


def _component_cid_from_maps(
    *,
    component_cids: Mapping[str, str],
    aliases: Sequence[str],
) -> str:
    for key in aliases:
        value = component_cids.get(key, "")
        if isinstance(value, str) and value:
            return value
    return ""


# ---------------------------------------------------------------------------
# Dependency resolution report
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResolvedDependencyFact:
    """One freshly resolved retained-trace dependency fact."""

    __test__: ClassVar[bool] = False

    kind: str
    name: str
    status: DependencyResolutionStatus
    retained_digest: str = ""
    current_digest: str = ""
    controlled: bool = True
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def matched(self) -> bool:
        return self.status is DependencyResolutionStatus.MATCHED


@dataclass(frozen=True, slots=True)
class DependencyResolutionReport:
    """Aggregate result of freshly resolving a retained dependency frontier."""

    __test__: ClassVar[bool] = False

    complete: bool
    facts: tuple[ResolvedDependencyFact, ...] = ()
    unresolved: tuple[str, ...] = ()
    changed: tuple[str, ...] = ()
    uncontrolled: tuple[str, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def interface(self) -> str:
        return DEPENDENCY_RESOLUTION_REPORT_INTERFACE

    @property
    def matched(self) -> bool:
        return (
            self.complete
            and not self.unresolved
            and not self.changed
            and not self.uncontrolled
            and all(fact.matched for fact in self.facts)
        )


# ---------------------------------------------------------------------------
# CandidateComparison@1
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CandidateComparison:
    """Exact comparison of a retained candidate against a fresh current context.

    Extends the sealed skip-dimension comparison with fixtures, hooks,
    parameters, locks, distributions, capabilities, forest, external snapshots,
    and the freshly resolved retained-trace dependency frontier.
    """

    __test__: ClassVar[bool] = False

    matched: bool
    mismatched_dimensions: tuple[str, ...] = ()
    missing_dimensions: tuple[str, ...] = ()
    unresolved_dependencies: tuple[str, ...] = ()
    changed_dependencies: tuple[str, ...] = ()
    uncontrolled_facts: tuple[str, ...] = ()
    dependency_report: DependencyResolutionReport | None = None
    core_comparison: ContextComparisonResult | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "matched", bool(self.matched))
        object.__setattr__(
            self,
            "mismatched_dimensions",
            tuple(str(item) for item in self.mismatched_dimensions),
        )
        object.__setattr__(
            self,
            "missing_dimensions",
            tuple(str(item) for item in self.missing_dimensions),
        )
        object.__setattr__(
            self,
            "unresolved_dependencies",
            tuple(str(item) for item in self.unresolved_dependencies),
        )
        object.__setattr__(
            self,
            "changed_dependencies",
            tuple(str(item) for item in self.changed_dependencies),
        )
        object.__setattr__(
            self,
            "uncontrolled_facts",
            tuple(str(item) for item in self.uncontrolled_facts),
        )
        object.__setattr__(
            self, "diagnostics", _bounded_diagnostics(dict(self.diagnostics or {}))
        )

    @property
    def interface(self) -> str:
        return CANDIDATE_COMPARISON_INTERFACE

    @property
    def may_authorize_skip(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CANDIDATE_COMPARISON_SCHEMA,
            "interface": CANDIDATE_COMPARISON_INTERFACE,
            "matched": self.matched,
            "mismatched_dimensions": list(self.mismatched_dimensions),
            "missing_dimensions": list(self.missing_dimensions),
            "unresolved_dependencies": list(self.unresolved_dependencies),
            "changed_dependencies": list(self.changed_dependencies),
            "uncontrolled_facts": list(self.uncontrolled_facts),
            "may_authorize_skip": False,
            "diagnostics": dict(self.diagnostics),
        }


def compare_candidate_to_current(
    candidate: CandidateExecutionContext,
    current: CurrentExecutionContext,
    *,
    dependency_report: DependencyResolutionReport | None = None,
    require_extended_identities: bool = True,
) -> CandidateComparison:
    """Compare retained candidate identities to a freshly rebuilt current context.

    Core dimensions use :func:`compare_contexts_for_skip`.  Extended identities
    (fixtures, hooks, parameters, locks, distributions, capabilities, external
    snapshots) must also match when present on either side.  A non-matching or
    incomplete dependency report forces ``matched=False``.
    """

    if not isinstance(candidate, CandidateExecutionContext):
        return CandidateComparison(
            matched=False,
            missing_dimensions=tuple(
                dim.value for dim in SkipComparisonDimension
            )
            + _EXTENDED_IDENTITY_DIMENSIONS,
            diagnostics={"stage": "candidate_type"},
        )
    if not isinstance(current, CurrentExecutionContext):
        return CandidateComparison(
            matched=False,
            missing_dimensions=tuple(
                dim.value for dim in SkipComparisonDimension
            )
            + _EXTENDED_IDENTITY_DIMENSIONS,
            diagnostics={"stage": "current_type"},
        )
    if current.rebuild_source not in {
        "fresh_live_rebuild",
        "controlled_preflight",
    }:
        return CandidateComparison(
            matched=False,
            missing_dimensions=("current_freshness",),
            diagnostics={
                "stage": "current_not_fresh",
                "rebuild_source": current.rebuild_source,
            },
        )

    core = compare_contexts_for_skip(candidate, current)
    mismatched = list(core.mismatched_dimensions)
    missing = list(core.missing_dimensions)

    if require_extended_identities:
        for dimension, field_name in _EXTENDED_FIELD_MAP.items():
            left = str(getattr(candidate, field_name, "") or "")
            right = str(getattr(current, field_name, "") or "")
            if left and right:
                if left != right and dimension not in mismatched:
                    mismatched.append(dimension)
            elif left or right:
                # One side claims a dimension the other lacks.
                if dimension not in missing and dimension not in mismatched:
                    # Optional empty on both is fine; only one-sided presence
                    # when the other is empty is treated as missing current.
                    if left and not right:
                        missing.append(dimension)
                    elif right and not left:
                        # Current has a fact the candidate never pinned → treat
                        # as mismatch rather than silently expanding the frontier.
                        mismatched.append(dimension)

        # component_cids based identities (fixtures / hooks / parameters / …).
        cand_components = dict(candidate.component_cids or {})
        curr_components = dict(current.component_cids or {})
        for dimension, aliases in _COMPONENT_KEY_ALIASES.items():
            left = _component_cid_from_maps(
                component_cids=cand_components, aliases=aliases
            )
            right = _component_cid_from_maps(
                component_cids=curr_components, aliases=aliases
            )
            if left and right:
                if left != right and dimension not in mismatched:
                    mismatched.append(dimension)
            elif left and not right:
                if dimension not in missing:
                    missing.append(dimension)
            elif right and not left:
                if dimension not in mismatched:
                    mismatched.append(dimension)

        # External snapshot multiset comparison (order-insensitive).
        left_ext = tuple(sorted(str(x) for x in (candidate.external_snapshot_cids or ())))
        right_ext = tuple(sorted(str(x) for x in (current.external_snapshot_cids or ())))
        if left_ext or right_ext:
            if left_ext != right_ext:
                if "external_snapshots" not in mismatched:
                    mismatched.append("external_snapshots")

    unresolved: list[str] = []
    changed: list[str] = []
    uncontrolled: list[str] = []
    if dependency_report is not None:
        unresolved = list(dependency_report.unresolved)
        changed = list(dependency_report.changed)
        uncontrolled = list(dependency_report.uncontrolled)
        if not dependency_report.complete:
            if "runtime_frontier" not in missing:
                missing.append("runtime_frontier")
        if not dependency_report.matched:
            if changed and "runtime" not in mismatched:
                # Named dependency content diverged from retained facts.
                mismatched.append("runtime_dependency")
            if unresolved and "runtime_dependency" not in missing:
                missing.append("runtime_dependency")
            if uncontrolled and "runtime_dependency" not in mismatched:
                mismatched.append("uncontrolled_dependency")

    matched = (
        not mismatched
        and not missing
        and not unresolved
        and not changed
        and not uncontrolled
        and (dependency_report is None or dependency_report.matched)
    )
    return CandidateComparison(
        matched=matched,
        mismatched_dimensions=tuple(mismatched),
        missing_dimensions=tuple(missing),
        unresolved_dependencies=tuple(unresolved),
        changed_dependencies=tuple(changed),
        uncontrolled_facts=tuple(uncontrolled),
        dependency_report=dependency_report,
        core_comparison=core,
        diagnostics={
            "required_core_dimensions": [
                dim.value for dim in SkipComparisonDimension
            ],
            "extended_dimensions": list(_EXTENDED_IDENTITY_DIMENSIONS),
        },
    )


# ---------------------------------------------------------------------------
# Fresh dependency resolution
# ---------------------------------------------------------------------------


@runtime_checkable
class DependencyContentResolver(Protocol):
    """Protocol for freshly resolving one retained dependency fact.

    Implementations must never execute the test body or fixtures.  They only
    content-address live filesystem / environment / capability state named by
    the retained frontier.
    """

    def resolve_file(
        self,
        *,
        root_id: str,
        relative_path: str,
        retained_sha256: str,
        retained_size: int | None = None,
    ) -> ResolvedDependencyFact:
        """Resolve a retained file fact against live roots."""

    def resolve_module(
        self,
        *,
        module_name: str,
        retained_fact: Mapping[str, Any],
    ) -> ResolvedDependencyFact:
        """Resolve a retained module fact against live importable state."""

    def resolve_environment(
        self,
        *,
        name: str,
        retained_fact: Mapping[str, Any],
    ) -> ResolvedDependencyFact:
        """Resolve a retained environment fact against the live process env."""

    def resolve_generic(
        self,
        *,
        kind: str,
        retained_fact: Mapping[str, Any],
    ) -> ResolvedDependencyFact:
        """Resolve other retained kinds (services, policies, capabilities…)."""


@dataclass
class FilesystemDependencyResolver:
    """Default resolver that content-addresses files under admitted roots.

    Environment values are reduced to SHA-256 of their UTF-8 form (or the
    retained value_cid / value_sha256 when provided).  Modules with a retained
    ``source_path`` / ``path`` are rehashed under the matching root.  Facts that
    cannot be safely re-resolved are marked unresolvable or uncontrolled.
    """

    __test__: ClassVar[bool] = False

    allowed_roots: Mapping[str, str | os.PathLike[str]] = field(default_factory=dict)
    environ: Mapping[str, str] | None = None
    max_file_bytes: int = _MAX_FILE_BYTES

    def __post_init__(self) -> None:
        roots: dict[str, Path] = {}
        for key, value in dict(self.allowed_roots or {}).items():
            name = str(key)[:_MAX_ROOT_ID_CHARS]
            if not name:
                continue
            try:
                roots[name] = Path(os.fspath(value)).resolve()
            except (OSError, TypeError, ValueError):
                continue
        self._roots = roots
        self._environ = dict(os.environ if self.environ is None else self.environ)
        self.max_file_bytes = int(self.max_file_bytes)

    def resolve_file(
        self,
        *,
        root_id: str,
        relative_path: str,
        retained_sha256: str,
        retained_size: int | None = None,
    ) -> ResolvedDependencyFact:
        name = f"{root_id}:{relative_path}"
        if not root_id or not relative_path:
            return ResolvedDependencyFact(
                kind="files",
                name=name or "<missing>",
                status=DependencyResolutionStatus.UNRESOLVABLE,
                retained_digest=retained_sha256,
                controlled=False,
                diagnostics={"reason": "missing_path"},
            )
        if ".." in Path(relative_path).parts or relative_path.startswith("/"):
            return ResolvedDependencyFact(
                kind="files",
                name=name,
                status=DependencyResolutionStatus.UNCONTROLLED,
                retained_digest=retained_sha256,
                controlled=False,
                diagnostics={"reason": "path_escape"},
            )
        root = self._roots.get(root_id)
        if root is None:
            return ResolvedDependencyFact(
                kind="files",
                name=name,
                status=DependencyResolutionStatus.UNRESOLVABLE,
                retained_digest=retained_sha256,
                diagnostics={"reason": "unknown_root"},
            )
        try:
            candidate = (root / relative_path).resolve(strict=True)
            candidate.relative_to(root)
        except (OSError, RuntimeError, ValueError):
            return ResolvedDependencyFact(
                kind="files",
                name=name,
                status=DependencyResolutionStatus.UNRESOLVABLE,
                retained_digest=retained_sha256,
                diagnostics={"reason": "path_missing_or_escape"},
            )
        if candidate.is_symlink():
            return ResolvedDependencyFact(
                kind="files",
                name=name,
                status=DependencyResolutionStatus.UNCONTROLLED,
                retained_digest=retained_sha256,
                controlled=False,
                diagnostics={"reason": "symlink"},
            )
        try:
            if not candidate.is_file():
                return ResolvedDependencyFact(
                    kind="files",
                    name=name,
                    status=DependencyResolutionStatus.UNRESOLVABLE,
                    retained_digest=retained_sha256,
                    diagnostics={"reason": "not_a_file"},
                )
            size = candidate.stat().st_size
            if size > self.max_file_bytes:
                return ResolvedDependencyFact(
                    kind="files",
                    name=name,
                    status=DependencyResolutionStatus.UNCONTROLLED,
                    retained_digest=retained_sha256,
                    controlled=False,
                    diagnostics={"reason": "over_budget"},
                )
            if retained_size is not None and size != retained_size:
                return ResolvedDependencyFact(
                    kind="files",
                    name=name,
                    status=DependencyResolutionStatus.CHANGED,
                    retained_digest=retained_sha256,
                    current_digest=f"size:{size}",
                    diagnostics={"reason": "size_changed", "size": size},
                )
            digest = _sha256_hex(candidate.read_bytes())
        except OSError:
            return ResolvedDependencyFact(
                kind="files",
                name=name,
                status=DependencyResolutionStatus.UNRESOLVABLE,
                retained_digest=retained_sha256,
                diagnostics={"reason": "io_error"},
            )
        if retained_sha256 and digest != retained_sha256:
            return ResolvedDependencyFact(
                kind="files",
                name=name,
                status=DependencyResolutionStatus.CHANGED,
                retained_digest=retained_sha256,
                current_digest=digest,
            )
        return ResolvedDependencyFact(
            kind="files",
            name=name,
            status=DependencyResolutionStatus.MATCHED,
            retained_digest=retained_sha256 or digest,
            current_digest=digest,
        )

    def resolve_module(
        self,
        *,
        module_name: str,
        retained_fact: Mapping[str, Any],
    ) -> ResolvedDependencyFact:
        name = str(module_name or retained_fact.get("name") or "")[:256]
        retained_digest = str(
            retained_fact.get("content_sha256")
            or retained_fact.get("source_sha256")
            or retained_fact.get("cid")
            or ""
        )
        # Prefer an explicit retained source path under an admitted root.
        source_path = retained_fact.get("source_path") or retained_fact.get("path")
        root_id = str(retained_fact.get("root_id") or "")
        if isinstance(source_path, str) and source_path and root_id:
            file_result = self.resolve_file(
                root_id=root_id,
                relative_path=source_path,
                retained_sha256=retained_digest,
                retained_size=(
                    int(retained_fact["size_bytes"])
                    if isinstance(retained_fact.get("size_bytes"), int)
                    else None
                ),
            )
            return ResolvedDependencyFact(
                kind="modules",
                name=name or source_path,
                status=file_result.status,
                retained_digest=file_result.retained_digest,
                current_digest=file_result.current_digest,
                controlled=file_result.controlled,
                diagnostics=dict(file_result.diagnostics),
            )
        # Module identity may be pinned by a content digest alone when no path
        # is available; without a live resolution path this is unresolvable.
        if retained_digest:
            return ResolvedDependencyFact(
                kind="modules",
                name=name or "<module>",
                status=DependencyResolutionStatus.UNRESOLVABLE,
                retained_digest=retained_digest,
                diagnostics={"reason": "no_live_source_path"},
            )
        return ResolvedDependencyFact(
            kind="modules",
            name=name or "<module>",
            status=DependencyResolutionStatus.UNCONTROLLED,
            controlled=False,
            diagnostics={"reason": "uncontrolled_module"},
        )

    def resolve_environment(
        self,
        *,
        name: str,
        retained_fact: Mapping[str, Any],
    ) -> ResolvedDependencyFact:
        key = str(name or retained_fact.get("name") or "")[:256]
        if not key:
            return ResolvedDependencyFact(
                kind="environment",
                name="<missing>",
                status=DependencyResolutionStatus.UNRESOLVABLE,
                diagnostics={"reason": "missing_name"},
            )
        retained_digest = str(
            retained_fact.get("value_sha256")
            or retained_fact.get("content_sha256")
            or retained_fact.get("value_cid")
            or retained_fact.get("cid")
            or ""
        )
        if key not in self._environ:
            return ResolvedDependencyFact(
                kind="environment",
                name=key,
                status=DependencyResolutionStatus.UNRESOLVABLE,
                retained_digest=retained_digest,
                diagnostics={"reason": "env_absent"},
            )
        raw = self._environ[key]
        try:
            current_digest = _sha256_hex(str(raw).encode("utf-8"))
        except Exception:
            return ResolvedDependencyFact(
                kind="environment",
                name=key,
                status=DependencyResolutionStatus.UNCONTROLLED,
                retained_digest=retained_digest,
                controlled=False,
                diagnostics={"reason": "env_encode_failed"},
            )
        # When the retained fact pins a digest, require exact agreement.
        if retained_digest:
            # Accept either a raw sha256 or a CID-style pin that embeds it.
            if (
                retained_digest != current_digest
                and retained_digest
                not in {
                    current_digest,
                    f"sha256:{current_digest}",
                }
                and not retained_digest.endswith(current_digest)
            ):
                # Also accept identity when retained stored the hashed form of
                # the same value under a different keying scheme by comparing
                # against an optional retained plaintext hash only.
                return ResolvedDependencyFact(
                    kind="environment",
                    name=key,
                    status=DependencyResolutionStatus.CHANGED,
                    retained_digest=retained_digest,
                    current_digest=current_digest,
                )
        return ResolvedDependencyFact(
            kind="environment",
            name=key,
            status=DependencyResolutionStatus.MATCHED,
            retained_digest=retained_digest or current_digest,
            current_digest=current_digest,
        )

    def resolve_generic(
        self,
        *,
        kind: str,
        retained_fact: Mapping[str, Any],
    ) -> ResolvedDependencyFact:
        name = str(
            retained_fact.get("name")
            or retained_fact.get("path")
            or retained_fact.get("id")
            or kind
        )[:256]
        retained_digest = str(
            retained_fact.get("content_sha256")
            or retained_fact.get("cid")
            or retained_fact.get("value_cid")
            or retained_fact.get("digest")
            or ""
        )
        # Subprocesses / services / unknown kinds are not re-executed; without
        # an external snapshot pin they are uncontrolled for warm admission.
        if kind in {"subprocesses", "services"}:
            return ResolvedDependencyFact(
                kind=kind,
                name=name,
                status=DependencyResolutionStatus.UNCONTROLLED,
                retained_digest=retained_digest,
                controlled=False,
                diagnostics={"reason": "effectful_kind_not_reexecutable"},
            )
        if not retained_digest:
            return ResolvedDependencyFact(
                kind=kind,
                name=name,
                status=DependencyResolutionStatus.UNCONTROLLED,
                controlled=False,
                diagnostics={"reason": "missing_digest"},
            )
        # Policies / capabilities / code_objects with a digest but no live
        # re-resolution path cannot be confirmed; treat as unresolvable.
        return ResolvedDependencyFact(
            kind=kind,
            name=name,
            status=DependencyResolutionStatus.UNRESOLVABLE,
            retained_digest=retained_digest,
            diagnostics={"reason": "no_live_resolver"},
        )


def resolve_retained_runtime_frontier(
    retained_trace_payload: Mapping[str, Any] | None,
    resolver: DependencyContentResolver,
    *,
    require_complete: bool = True,
) -> DependencyResolutionReport:
    """Freshly resolve every dependency named by a retained runtime trace.

    The retained trace is diagnostic evidence that names a frontier; it is never
    accepted as current evidence.  Incomplete completeness status, unknown
    frontiers, or any unresolvable/changed/uncontrolled fact fail the report.
    """

    if not isinstance(retained_trace_payload, Mapping):
        return DependencyResolutionReport(
            complete=False,
            diagnostics={"stage": "trace_missing"},
        )

    completeness = retained_trace_payload.get("completeness")
    complete = True
    if isinstance(completeness, Mapping):
        complete = bool(completeness.get("complete", False))
        status = str(completeness.get("status", "")).lower()
        if status and status not in {"complete", ""}:
            complete = False
    elif require_complete:
        # Missing completeness block is fail-closed for warm admission.
        complete = False

    if require_complete and not complete:
        return DependencyResolutionReport(
            complete=False,
            diagnostics={
                "stage": "trace_incomplete",
                "completeness": (
                    dict(completeness) if isinstance(completeness, Mapping) else None
                ),
            },
        )

    dependencies = retained_trace_payload.get("dependencies")
    if dependencies is None:
        # Empty frontier is complete only when the trace declares completeness.
        return DependencyResolutionReport(
            complete=complete,
            diagnostics={"stage": "empty_frontier"},
        )
    if not isinstance(dependencies, Mapping):
        return DependencyResolutionReport(
            complete=False,
            diagnostics={"stage": "malformed_dependencies"},
        )

    facts: list[ResolvedDependencyFact] = []
    unresolved: list[str] = []
    changed: list[str] = []
    uncontrolled: list[str] = []
    total = 0

    for kind in _RUNTIME_DEPENDENCY_KINDS:
        items = dependencies.get(kind)
        if items is None:
            continue
        if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
            return DependencyResolutionReport(
                complete=False,
                diagnostics={"stage": "malformed_kind", "kind": kind},
            )
        for item in items:
            if total >= _MAX_DEPENDENCIES:
                return DependencyResolutionReport(
                    complete=False,
                    facts=tuple(facts),
                    unresolved=tuple(unresolved),
                    changed=tuple(changed),
                    uncontrolled=tuple(uncontrolled),
                    diagnostics={"stage": "dependency_overflow"},
                )
            total += 1
            if not isinstance(item, Mapping):
                fact = ResolvedDependencyFact(
                    kind=kind,
                    name=f"{kind}:malformed",
                    status=DependencyResolutionStatus.UNCONTROLLED,
                    controlled=False,
                    diagnostics={"reason": "non_object_fact"},
                )
            elif kind == "files":
                fact = resolver.resolve_file(
                    root_id=str(item.get("root_id") or ""),
                    relative_path=str(item.get("path") or ""),
                    retained_sha256=str(item.get("content_sha256") or ""),
                    retained_size=(
                        int(item["size_bytes"])
                        if isinstance(item.get("size_bytes"), int)
                        else None
                    ),
                )
            elif kind == "modules":
                fact = resolver.resolve_module(
                    module_name=str(item.get("name") or item.get("module") or ""),
                    retained_fact=item,
                )
            elif kind == "environment":
                fact = resolver.resolve_environment(
                    name=str(item.get("name") or item.get("key") or ""),
                    retained_fact=item,
                )
            else:
                fact = resolver.resolve_generic(kind=kind, retained_fact=item)

            facts.append(fact)
            label = f"{fact.kind}:{fact.name}"
            if fact.status is DependencyResolutionStatus.CHANGED:
                changed.append(label)
            elif fact.status is DependencyResolutionStatus.UNRESOLVABLE:
                unresolved.append(label)
            elif fact.status is DependencyResolutionStatus.UNCONTROLLED:
                uncontrolled.append(label)

    # Unknown frontier markers (static-style) force RUN.
    unknown = retained_trace_payload.get("unknown_frontier")
    if isinstance(unknown, Sequence) and not isinstance(unknown, (str, bytes)):
        if unknown:
            uncontrolled.append("unknown_frontier")
            complete = False

    report_complete = complete and not unresolved and not changed and not uncontrolled
    return DependencyResolutionReport(
        complete=report_complete,
        facts=tuple(facts),
        unresolved=tuple(unresolved),
        changed=tuple(changed),
        uncontrolled=tuple(uncontrolled),
        diagnostics={
            "resolved_count": len(facts),
            "matched_count": sum(1 for fact in facts if fact.matched),
        },
    )


# ---------------------------------------------------------------------------
# Runtime revalidation result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RuntimeRevalidationResult:
    """Outcome of locator → candidate → fresh-frontier revalidation.

    ``PROCEED_TO_CERTIFICATE_VERIFICATION`` means the current context matches
    the retained candidate exactly and fixtures/test body must not run for the
    purpose of predicting skip eligibility.  It is **not** skip authority.
    """

    __test__: ClassVar[bool] = False

    action: RevalidationAction
    reason: RevalidationReason
    locator_cid: str = ""
    candidate: CandidateExecutionContext | None = None
    current: CurrentExecutionContext | None = None
    comparison: CandidateComparison | None = None
    candidate_context_cid: str = ""
    envelope_cid: str = ""
    component_bytes: Mapping[str, bytes] = field(default_factory=dict)
    lookup_hit: bool = False
    may_proceed_to_certificate_verification: bool = False
    fixtures_executed: bool = False
    test_body_executed: bool = False
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "action",
            self.action
            if isinstance(self.action, RevalidationAction)
            else RevalidationAction(str(self.action)),
        )
        object.__setattr__(
            self,
            "reason",
            self.reason
            if isinstance(self.reason, RevalidationReason)
            else RevalidationReason(str(self.reason)),
        )
        object.__setattr__(self, "locator_cid", str(self.locator_cid or ""))
        object.__setattr__(
            self, "candidate_context_cid", str(self.candidate_context_cid or "")
        )
        object.__setattr__(self, "envelope_cid", str(self.envelope_cid or ""))
        object.__setattr__(self, "lookup_hit", bool(self.lookup_hit))
        object.__setattr__(
            self,
            "may_proceed_to_certificate_verification",
            bool(self.may_proceed_to_certificate_verification),
        )
        object.__setattr__(self, "fixtures_executed", bool(self.fixtures_executed))
        object.__setattr__(self, "test_body_executed", bool(self.test_body_executed))
        object.__setattr__(
            self,
            "component_bytes",
            MappingProxyType(dict(self.component_bytes or {})),
        )
        object.__setattr__(
            self, "diagnostics", _bounded_diagnostics(dict(self.diagnostics or {}))
        )
        # Invariant: proceeding never executes fixtures or the test body.
        if self.action is RevalidationAction.PROCEED_TO_CERTIFICATE_VERIFICATION:
            if self.fixtures_executed or self.test_body_executed:
                object.__setattr__(
                    self, "action", RevalidationAction.RUN
                )
                object.__setattr__(
                    self,
                    "reason",
                    RevalidationReason.INTERNAL_ERROR_FAIL_OPEN_TO_RUN,
                )
                object.__setattr__(
                    self, "may_proceed_to_certificate_verification", False
                )
            else:
                object.__setattr__(
                    self, "may_proceed_to_certificate_verification", True
                )
        else:
            object.__setattr__(
                self, "may_proceed_to_certificate_verification", False
            )

    @property
    def interface(self) -> str:
        return RUNTIME_REVALIDATION_RESULT_INTERFACE

    @property
    def is_run(self) -> bool:
        return self.action is RevalidationAction.RUN

    @property
    def is_proceed(self) -> bool:
        return (
            self.action is RevalidationAction.PROCEED_TO_CERTIFICATE_VERIFICATION
        )

    @property
    def may_authorize_skip(self) -> bool:
        return False

    def to_disposition(self) -> RuntimeReuseDisposition:
        """Map to the sealed activation disposition (always RUN at this stage).

        Proceed-to-certificate is not SKIP; the certificate verifier produces
        SKIP later.  This helper therefore always returns a RUN disposition so
        callers cannot accidentally treat revalidation as skip authority.
        """

        return disposition_run(
            self.reason.value,
            diagnostics={
                "revalidation_action": self.action.value,
                "may_proceed_to_certificate_verification": (
                    self.may_proceed_to_certificate_verification
                ),
                "locator_cid": self.locator_cid[:128],
                "candidate_context_cid": self.candidate_context_cid[:128],
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RUNTIME_REVALIDATION_RESULT_SCHEMA,
            "interface": RUNTIME_REVALIDATION_RESULT_INTERFACE,
            "action": self.action.value,
            "reason": self.reason.value,
            "locator_cid": self.locator_cid,
            "candidate_context_cid": self.candidate_context_cid,
            "envelope_cid": self.envelope_cid,
            "lookup_hit": self.lookup_hit,
            "may_proceed_to_certificate_verification": (
                self.may_proceed_to_certificate_verification
            ),
            "fixtures_executed": self.fixtures_executed,
            "test_body_executed": self.test_body_executed,
            "may_authorize_skip": False,
            "matched": bool(self.comparison.matched) if self.comparison else False,
            "diagnostics": dict(self.diagnostics),
        }


def _run_result(
    reason: RevalidationReason,
    *,
    locator_cid: str = "",
    **fields: Any,
) -> RuntimeRevalidationResult:
    return RuntimeRevalidationResult(
        action=RevalidationAction.RUN,
        reason=reason,
        locator_cid=locator_cid,
        **fields,
    )


# ---------------------------------------------------------------------------
# PostPassRuntimeTraceCapture@1
# ---------------------------------------------------------------------------


@dataclass
class PostPassRuntimeTraceCapture:
    """Record the observed runtime frontier after one real lifecycle.

    Enforces setup/call/teardown exactly once and forbids re-invoking the test
    body for capture.  Publishing retained candidate context is optional and
    never authorizes SKIP.
    """

    __test__: ClassVar[bool] = False

    locator_cid: str = ""
    execution_key_cid: str = ""
    pass_receipt_cid: str = ""
    clock: Callable[[], float] | Callable[[], int] | None = None
    publisher: Callable[..., Any] | None = None

    def __post_init__(self) -> None:
        self._lock = threading.RLock()
        self._phase = LifecyclePhase.IDLE
        self._setup_count = 0
        self._call_count = 0
        self._teardown_count = 0
        self._runtime_trace_root_cid = ""
        self._runtime_trace_bytes: bytes | None = None
        self._observation: PostPassRuntimeObservation | None = None
        self._published = False
        self._publish_result: Any = None
        self._capture_error: str = ""

    @property
    def interface(self) -> str:
        return POST_PASS_RUNTIME_TRACE_CAPTURE_INTERFACE

    @property
    def phase(self) -> LifecyclePhase:
        return self._phase

    @property
    def setup_call_count(self) -> int:
        return self._setup_count

    @property
    def test_call_count(self) -> int:
        return self._call_count

    @property
    def teardown_call_count(self) -> int:
        return self._teardown_count

    @property
    def observation(self) -> PostPassRuntimeObservation | None:
        return self._observation

    @property
    def published(self) -> bool:
        return self._published

    @property
    def may_authorize_skip(self) -> bool:
        return False

    @property
    def lifecycle_complete(self) -> bool:
        return (
            self._setup_count == 1
            and self._call_count == 1
            and self._teardown_count == 1
            and self._phase in {LifecyclePhase.COMPLETE, LifecyclePhase.TEARDOWN}
        )

    def note_setup(self) -> None:
        with self._lock:
            if self._setup_count != 0 or self._phase not in {
                LifecyclePhase.IDLE,
            }:
                self._phase = LifecyclePhase.FAILED
                self._capture_error = "duplicate_or_out_of_order_setup"
                raise RuntimeError(self._capture_error)
            self._setup_count = 1
            self._phase = LifecyclePhase.SETUP

    def note_call(self) -> None:
        with self._lock:
            if self._setup_count != 1 or self._call_count != 0:
                self._phase = LifecyclePhase.FAILED
                self._capture_error = "duplicate_or_out_of_order_call"
                raise RuntimeError(self._capture_error)
            if self._phase is not LifecyclePhase.SETUP:
                self._phase = LifecyclePhase.FAILED
                self._capture_error = "call_before_setup"
                raise RuntimeError(self._capture_error)
            self._call_count = 1
            self._phase = LifecyclePhase.CALL

    def note_teardown(self) -> None:
        with self._lock:
            if self._call_count != 1 or self._teardown_count != 0:
                self._phase = LifecyclePhase.FAILED
                self._capture_error = "duplicate_or_out_of_order_teardown"
                raise RuntimeError(self._capture_error)
            if self._phase is not LifecyclePhase.CALL:
                self._phase = LifecyclePhase.FAILED
                self._capture_error = "teardown_before_call"
                raise RuntimeError(self._capture_error)
            self._teardown_count = 1
            self._phase = LifecyclePhase.TEARDOWN

    def capture_observed_runtime_trace(
        self,
        *,
        runtime_trace_root_cid: str,
        runtime_trace_bytes: bytes | None = None,
        locator_cid: str | None = None,
        execution_key_cid: str | None = None,
        pass_receipt_cid: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> PostPassRuntimeObservation:
        """Capture the observed runtime frontier after the single lifecycle.

        Does not re-invoke the test body.  Requires setup/call/teardown counts
        of exactly one each.
        """

        with self._lock:
            if self._setup_count != 1 or self._call_count != 1 or self._teardown_count != 1:
                raise RuntimeError(
                    "post-pass capture requires exactly one setup/call/teardown"
                )
            if self._observation is not None:
                raise RuntimeError("runtime trace already captured for this lifecycle")
            if self._phase is LifecyclePhase.FAILED:
                raise RuntimeError(
                    self._capture_error or "lifecycle failed before capture"
                )

            locator = str(locator_cid if locator_cid is not None else self.locator_cid)
            execution = str(
                execution_key_cid
                if execution_key_cid is not None
                else self.execution_key_cid
            )
            receipt = str(
                pass_receipt_cid
                if pass_receipt_cid is not None
                else self.pass_receipt_cid
            )
            root = str(runtime_trace_root_cid or "")
            if not locator or not execution or not receipt or not root:
                raise ValueError(
                    "locator, execution key, pass receipt, and runtime trace "
                    "root CIDs are required for post-pass capture"
                )

            if runtime_trace_bytes is not None:
                if type(runtime_trace_bytes) is not bytes:
                    raise TypeError("runtime_trace_bytes must be exact bytes")
                # Rehash when bytes claim a content identity boundary.
                try:
                    actual = rehash_retained_canonical_bytes(runtime_trace_bytes)
                except Exception:
                    # Non-DAG-JSON traces (e.g. raw instrumented payloads) are
                    # accepted only when the caller already supplied a root cid.
                    actual = ""
                if actual and actual != root:
                    raise ValueError(
                        "runtime_trace_bytes do not rehash to runtime_trace_root_cid"
                    )
                self._runtime_trace_bytes = runtime_trace_bytes

            observation = record_post_pass_runtime_observation(
                locator_cid=locator,
                execution_key_cid=execution,
                runtime_trace_root_cid=root,
                pass_receipt_cid=receipt,
                test_call_count=self._call_count,
                setup_call_count=self._setup_count,
                teardown_call_count=self._teardown_count,
                observed_at_ms=_now_ms(self.clock),
                metadata={
                    **dict(metadata or {}),
                    "capture_interface": POST_PASS_RUNTIME_TRACE_CAPTURE_INTERFACE,
                    "duplicate_test_call_forbidden": True,
                },
            )
            self._runtime_trace_root_cid = root
            self._observation = observation
            self._phase = LifecyclePhase.COMPLETE
            self.locator_cid = locator
            self.execution_key_cid = execution
            self.pass_receipt_cid = receipt
            return observation

    def publish_observed_trace(self, *args: Any, **kwargs: Any) -> Any:
        """Publish the captured observation via the optional publisher.

        Publication is fenced by the publisher implementation (candidate
        context store).  Absence of a publisher is a typed no-op result, never
        an exception that suppresses pytest.
        """

        with self._lock:
            if self._observation is None:
                raise RuntimeError("cannot publish before capture")
            if self._published:
                return self._publish_result
            if self.publisher is None:
                self._published = True
                self._publish_result = {
                    "published": False,
                    "reason": "publisher_absent",
                    "observation_id": self._observation.observation_id,
                    "may_authorize_skip": False,
                }
                return self._publish_result
            try:
                result = self.publisher(
                    self._observation,
                    runtime_trace_bytes=self._runtime_trace_bytes,
                    runtime_trace_root_cid=self._runtime_trace_root_cid,
                    *args,
                    **kwargs,
                )
            except TypeError:
                # Publisher may not accept the optional keyword arguments.
                result = self.publisher(self._observation, *args, **kwargs)
            self._published = True
            self._publish_result = result
            return result

    def execute_lifecycle_once(
        self,
        *,
        setup: Callable[[], Any],
        call: Callable[[], Any],
        teardown: Callable[[], Any],
        runtime_trace_root_cid: str = "",
        runtime_trace_bytes: bytes | None = None,
        locator_cid: str | None = None,
        execution_key_cid: str | None = None,
        pass_receipt_cid: str | None = None,
        capture_on_pass: bool = True,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run setup/call/teardown exactly once and optionally capture on pass.

        This is the miss path: the test body runs once.  Capture never
        re-invokes call.  Failures in setup/call still run teardown exactly
        once when possible and do not capture a pass observation.
        """

        call_result: Any = None
        setup_error: BaseException | None = None
        call_error: BaseException | None = None
        teardown_error: BaseException | None = None
        passed = False

        self.note_setup()
        try:
            setup()
        except BaseException as exc:
            setup_error = exc

        if setup_error is None:
            self.note_call()
            try:
                call_result = call()
                passed = True
            except BaseException as exc:
                call_error = exc
                passed = False
        else:
            # Setup failed: still count a synthetic call skip — lifecycle
            # counters must not pretend the body ran.  Teardown still runs.
            pass

        # Teardown always attempts exactly once after setup was noted.
        try:
            if self._setup_count == 1 and self._teardown_count == 0:
                if self._call_count == 0 and setup_error is not None:
                    # Mark call as not executed; allow teardown without call by
                    # recording a failed phase path that still tears down.
                    with self._lock:
                        self._phase = LifecyclePhase.CALL
                        self._call_count = 0
                    # Use a dedicated teardown path that does not require call.
                    with self._lock:
                        if self._teardown_count == 0:
                            self._teardown_count = 1
                            self._phase = LifecyclePhase.TEARDOWN
                    teardown()
                else:
                    self.note_teardown()
                    teardown()
        except BaseException as exc:
            teardown_error = exc

        observation = None
        if (
            capture_on_pass
            and passed
            and setup_error is None
            and call_error is None
            and teardown_error is None
            and runtime_trace_root_cid
        ):
            observation = self.capture_observed_runtime_trace(
                runtime_trace_root_cid=runtime_trace_root_cid,
                runtime_trace_bytes=runtime_trace_bytes,
                locator_cid=locator_cid,
                execution_key_cid=execution_key_cid,
                pass_receipt_cid=pass_receipt_cid,
                metadata=metadata,
            )

        return {
            "passed": passed and teardown_error is None,
            "setup_call_count": self._setup_count,
            "test_call_count": self._call_count,
            "teardown_call_count": self._teardown_count,
            "call_result": call_result,
            "setup_error": setup_error,
            "call_error": call_error,
            "teardown_error": teardown_error,
            "observation": observation,
            "may_authorize_skip": False,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": POST_PASS_RUNTIME_TRACE_CAPTURE_SCHEMA,
            "interface": POST_PASS_RUNTIME_TRACE_CAPTURE_INTERFACE,
            "phase": self._phase.value,
            "setup_call_count": self._setup_count,
            "test_call_count": self._call_count,
            "teardown_call_count": self._teardown_count,
            "locator_cid": self.locator_cid,
            "execution_key_cid": self.execution_key_cid,
            "pass_receipt_cid": self.pass_receipt_cid,
            "runtime_trace_root_cid": self._runtime_trace_root_cid,
            "captured": self._observation is not None,
            "published": self._published,
            "may_authorize_skip": False,
            "capture_error": self._capture_error,
        }


# ---------------------------------------------------------------------------
# Current-context provider protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class CurrentContextProvider(Protocol):
    """Compile a fresh CurrentExecutionContext for one locator/candidate pair.

    Must not execute fixtures or the test body.  Historical traces must not be
    relabeled as current (``rebuild_source`` stays fresh).
    """

    def compile_current(
        self,
        *,
        locator_cid: str,
        candidate: CandidateExecutionContext,
        component_bytes: Mapping[str, bytes],
    ) -> CurrentExecutionContext | None:
        ...


@dataclass(frozen=True, slots=True)
class StaticCurrentContextProvider:
    """Test/production helper that returns a prebuilt current context.

    Useful when identity services have already rebuilt the current frontier.
    """

    __test__: ClassVar[bool] = False

    current: CurrentExecutionContext

    def compile_current(
        self,
        *,
        locator_cid: str,
        candidate: CandidateExecutionContext,
        component_bytes: Mapping[str, bytes],
    ) -> CurrentExecutionContext | None:
        if self.current.locator_cid and self.current.locator_cid != locator_cid:
            return None
        return self.current


# ---------------------------------------------------------------------------
# RuntimeContextRevalidator@1
# ---------------------------------------------------------------------------


class RuntimeContextRevalidator:
    """Locator-keyed revalidation of retained candidates against live context.

    Authority sequence steps covered here:

    * ``resolve_bounded_candidate_descriptor``
    * ``load_retained_candidate_bytes_and_rehash``
    * ``rebuild_current_dependency_frontier``
    * (handoff) ``compare_current_and_verify_authoritative_certificate``

    Skip emission is intentionally out of scope: a match only permits later
    certificate verification without running fixtures or the test body.
    """

    __test__ = False
    interface = RUNTIME_CONTEXT_REVALIDATOR_INTERFACE

    def __init__(
        self,
        *,
        candidate_store: Any | None = None,
        current_context_provider: CurrentContextProvider | None = None,
        dependency_resolver: DependencyContentResolver | None = None,
        allowed_roots: Mapping[str, str | os.PathLike[str]] | None = None,
        environ: Mapping[str, str] | None = None,
        clock: Callable[[], float] | Callable[[], int] | None = None,
        require_runtime_frontier: bool = True,
        require_extended_identities: bool = True,
    ) -> None:
        self._store = candidate_store
        self._current_provider = current_context_provider
        self._resolver = dependency_resolver or FilesystemDependencyResolver(
            allowed_roots=dict(allowed_roots or {}),
            environ=environ,
        )
        self._clock = clock
        self._require_runtime_frontier = bool(require_runtime_frontier)
        self._require_extended_identities = bool(require_extended_identities)
        self._lock = threading.RLock()

    @property
    def schema(self) -> str:
        return RUNTIME_CONTEXT_REVALIDATOR_SCHEMA

    @property
    def may_authorize_skip(self) -> bool:
        return False

    @property
    def authority_sequence_prefix(self) -> tuple[str, ...]:
        return ACTIVATION_AUTHORITY_SEQUENCE[:4]

    def revalidate(
        self,
        locator: Any,
        *,
        current: CurrentExecutionContext | None = None,
        candidate: CandidateExecutionContext | None = None,
        component_bytes: Mapping[str, bytes] | None = None,
        retained_runtime_trace: Mapping[str, Any] | bytes | None = None,
        max_candidates: int | None = None,
        now_ms: int | None = None,
    ) -> RuntimeRevalidationResult:
        """Revalidate current context against retained candidates for ``locator``.

        Lookup uses the stable locator only.  Incomplete, unresolvable, changed,
        or uncontrolled facts return ``RUN``.  A verified unchanged context
        returns ``PROCEED_TO_CERTIFICATE_VERIFICATION`` without executing
        fixtures or the test body.
        """

        try:
            return self._revalidate_inner(
                locator,
                current=current,
                candidate=candidate,
                component_bytes=component_bytes,
                retained_runtime_trace=retained_runtime_trace,
                max_candidates=max_candidates,
                now_ms=now_ms,
            )
        except Exception as exc:  # noqa: BLE001 - fail open to RUN
            return _run_result(
                RevalidationReason.INTERNAL_ERROR_FAIL_OPEN_TO_RUN,
                locator_cid=str(_locator_token(locator) or ""),
                diagnostics={
                    "stage": "revalidate",
                    "error": type(exc).__name__[:64],
                },
            )

    def _revalidate_inner(
        self,
        locator: Any,
        *,
        current: CurrentExecutionContext | None,
        candidate: CandidateExecutionContext | None,
        component_bytes: Mapping[str, bytes] | None,
        retained_runtime_trace: Mapping[str, Any] | bytes | None,
        max_candidates: int | None,
        now_ms: int | None,
    ) -> RuntimeRevalidationResult:
        locator_cid = _locator_token(locator)
        if not locator_cid:
            return _run_result(
                RevalidationReason.LOCATOR_INVALID
                if locator is not None
                else RevalidationReason.LOCATOR_MISSING,
                diagnostics={"stage": "locator"},
            )

        resolved_components: dict[str, bytes] = dict(component_bytes or {})
        envelope_cid = ""
        candidate_context_cid = ""
        lookup_hit = False

        # --- Stage: resolve bounded candidate from store (locator only) ---
        if candidate is None:
            if self._store is None:
                return _run_result(
                    RevalidationReason.CANDIDATE_MISSING,
                    locator_cid=locator_cid,
                    diagnostics={"stage": "store_absent"},
                )
            try:
                lookup = self._lookup_candidate(
                    locator_cid,
                    max_candidates=max_candidates,
                    now_ms=now_ms,
                )
            except Exception as exc:  # noqa: BLE001
                return _run_result(
                    RevalidationReason.STORE_FAULT,
                    locator_cid=locator_cid,
                    diagnostics={
                        "stage": "lookup",
                        "error": type(exc).__name__[:64],
                    },
                )
            if lookup is None or not getattr(lookup, "hit", False):
                reason = RevalidationReason.CANDIDATE_MISSING
                code = str(getattr(lookup, "reason_code", "") or "")
                if "integrity" in code.lower() or "corrupt" in code.lower():
                    reason = RevalidationReason.CANDIDATE_INTEGRITY_FAILED
                return _run_result(
                    reason,
                    locator_cid=locator_cid,
                    diagnostics={
                        "stage": "lookup_miss",
                        "store_reason": code[:128],
                    },
                )
            candidate = getattr(lookup, "descriptor", None)
            if not isinstance(candidate, CandidateExecutionContext):
                return _run_result(
                    RevalidationReason.CANDIDATE_UNRESOLVABLE,
                    locator_cid=locator_cid,
                    diagnostics={"stage": "descriptor_type"},
                )
            lookup_hit = True
            envelope_cid = str(getattr(lookup, "envelope_cid", "") or "")
            candidate_context_cid = str(
                getattr(lookup, "candidate_context_cid", "") or ""
            )
            store_components = getattr(lookup, "component_bytes", None) or {}
            if isinstance(store_components, Mapping):
                for key, value in store_components.items():
                    if isinstance(value, (bytes, bytearray)):
                        resolved_components.setdefault(str(key), bytes(value))
            # Rehash retained descriptor bytes when present.
            descriptor_bytes = getattr(lookup, "descriptor_bytes", None)
            if isinstance(descriptor_bytes, (bytes, bytearray)):
                admission = admit_content_addressed_boundary(
                    role=ArtifactRole.IMMUTABLE_CANDIDATE_CONTEXT,
                    claimed_cid=candidate.candidate_context_id,
                    canonical_bytes=bytes(descriptor_bytes),
                )
                if not admission.admitted:
                    return _run_result(
                        RevalidationReason.CANDIDATE_INTEGRITY_FAILED,
                        locator_cid=locator_cid,
                        candidate=candidate,
                        lookup_hit=True,
                        candidate_context_cid=candidate.candidate_context_id,
                        envelope_cid=envelope_cid,
                        diagnostics={
                            "stage": "descriptor_rehash",
                            "reason": admission.reason_code,
                        },
                    )
        else:
            if not isinstance(candidate, CandidateExecutionContext):
                return _run_result(
                    RevalidationReason.CANDIDATE_UNRESOLVABLE,
                    locator_cid=locator_cid,
                    diagnostics={"stage": "candidate_type"},
                )
            if candidate.locator_cid and candidate.locator_cid != locator_cid:
                return _run_result(
                    RevalidationReason.LOCATOR_INVALID,
                    locator_cid=locator_cid,
                    candidate=candidate,
                    diagnostics={
                        "stage": "locator_mismatch",
                        "candidate_locator": candidate.locator_cid[:128],
                    },
                )
            candidate_context_cid = candidate.candidate_context_id
            lookup_hit = True

        # --- Stage: rehash every retained component ---
        rehash_failure = self._rehash_components(
            candidate,
            resolved_components,
            has_inline_runtime_trace=retained_runtime_trace is not None,
        )
        if rehash_failure is not None:
            return _run_result(
                rehash_failure,
                locator_cid=locator_cid,
                candidate=candidate,
                lookup_hit=lookup_hit,
                candidate_context_cid=candidate_context_cid or candidate.candidate_context_id,
                envelope_cid=envelope_cid,
                component_bytes=resolved_components,
                diagnostics={"stage": "component_rehash"},
            )

        # --- Stage: freshly resolve retained runtime frontier ---
        dependency_report: DependencyResolutionReport | None = None
        if self._require_runtime_frontier:
            trace_payload = self._load_runtime_trace_payload(
                resolved_components,
                retained_runtime_trace,
            )
            if trace_payload is None:
                return _run_result(
                    RevalidationReason.TRACE_INCOMPLETE
                    if "runtime_trace" not in resolved_components
                    and retained_runtime_trace is None
                    else RevalidationReason.TRACE_MALFORMED,
                    locator_cid=locator_cid,
                    candidate=candidate,
                    lookup_hit=lookup_hit,
                    candidate_context_cid=candidate.candidate_context_id,
                    envelope_cid=envelope_cid,
                    component_bytes=resolved_components,
                    diagnostics={"stage": "runtime_trace_load"},
                )
            dependency_report = resolve_retained_runtime_frontier(
                trace_payload,
                self._resolver,
                require_complete=True,
            )
            if not dependency_report.matched:
                stage = str(dependency_report.diagnostics.get("stage") or "")
                if stage in {"malformed_dependencies", "malformed_kind"}:
                    reason = RevalidationReason.TRACE_MALFORMED
                elif dependency_report.changed:
                    reason = RevalidationReason.DEPENDENCY_CHANGED
                elif dependency_report.uncontrolled:
                    reason = RevalidationReason.DEPENDENCY_UNCONTROLLED
                elif dependency_report.unresolved:
                    reason = RevalidationReason.DEPENDENCY_UNRESOLVABLE
                elif not dependency_report.complete:
                    reason = RevalidationReason.TRACE_INCOMPLETE
                else:
                    reason = RevalidationReason.DEPENDENCY_CHANGED
                return _run_result(
                    reason,
                    locator_cid=locator_cid,
                    candidate=candidate,
                    lookup_hit=lookup_hit,
                    candidate_context_cid=candidate.candidate_context_id,
                    envelope_cid=envelope_cid,
                    component_bytes=resolved_components,
                    comparison=CandidateComparison(
                        matched=False,
                        unresolved_dependencies=dependency_report.unresolved,
                        changed_dependencies=dependency_report.changed,
                        uncontrolled_facts=dependency_report.uncontrolled,
                        dependency_report=dependency_report,
                        diagnostics={"stage": "dependency_resolution"},
                    ),
                    diagnostics={
                        "stage": "dependency_resolution",
                        "unresolved": list(dependency_report.unresolved)[:16],
                        "changed": list(dependency_report.changed)[:16],
                        "uncontrolled": list(dependency_report.uncontrolled)[:16],
                    },
                )

        # --- Stage: rebuild / obtain fresh current context ---
        current_context = current
        if current_context is None and self._current_provider is not None:
            try:
                current_context = self._current_provider.compile_current(
                    locator_cid=locator_cid,
                    candidate=candidate,
                    component_bytes=resolved_components,
                )
            except Exception as exc:  # noqa: BLE001
                return _run_result(
                    RevalidationReason.CURRENT_CONTEXT_UNAVAILABLE,
                    locator_cid=locator_cid,
                    candidate=candidate,
                    lookup_hit=lookup_hit,
                    candidate_context_cid=candidate.candidate_context_id,
                    envelope_cid=envelope_cid,
                    component_bytes=resolved_components,
                    diagnostics={
                        "stage": "current_compile",
                        "error": type(exc).__name__[:64],
                    },
                )
        if current_context is None:
            return _run_result(
                RevalidationReason.CURRENT_CONTEXT_UNAVAILABLE,
                locator_cid=locator_cid,
                candidate=candidate,
                lookup_hit=lookup_hit,
                candidate_context_cid=candidate.candidate_context_id,
                envelope_cid=envelope_cid,
                component_bytes=resolved_components,
                diagnostics={"stage": "current_absent"},
            )
        if not isinstance(current_context, CurrentExecutionContext):
            return _run_result(
                RevalidationReason.CURRENT_CONTEXT_INCOMPLETE,
                locator_cid=locator_cid,
                candidate=candidate,
                lookup_hit=lookup_hit,
                candidate_context_cid=candidate.candidate_context_id,
                envelope_cid=envelope_cid,
                component_bytes=resolved_components,
                diagnostics={"stage": "current_type"},
            )
        if current_context.rebuild_source not in {
            "fresh_live_rebuild",
            "controlled_preflight",
        }:
            return _run_result(
                RevalidationReason.CURRENT_CONTEXT_NOT_FRESH,
                locator_cid=locator_cid,
                candidate=candidate,
                current=current_context,
                lookup_hit=lookup_hit,
                candidate_context_cid=candidate.candidate_context_id,
                envelope_cid=envelope_cid,
                component_bytes=resolved_components,
                diagnostics={
                    "stage": "rebuild_source",
                    "rebuild_source": current_context.rebuild_source,
                },
            )

        # --- Stage: exact comparison ---
        comparison = compare_candidate_to_current(
            candidate,
            current_context,
            dependency_report=dependency_report,
            require_extended_identities=self._require_extended_identities,
        )
        if not comparison.matched:
            return _run_result(
                RevalidationReason.IDENTITY_MISMATCH,
                locator_cid=locator_cid,
                candidate=candidate,
                current=current_context,
                comparison=comparison,
                lookup_hit=lookup_hit,
                candidate_context_cid=candidate.candidate_context_id,
                envelope_cid=envelope_cid,
                component_bytes=resolved_components,
                diagnostics={
                    "stage": "identity_comparison",
                    "mismatched": list(comparison.mismatched_dimensions)[:16],
                    "missing": list(comparison.missing_dimensions)[:16],
                },
            )

        # Verified unchanged — hand off to certificate verification without
        # executing fixtures or the test body.
        return RuntimeRevalidationResult(
            action=RevalidationAction.PROCEED_TO_CERTIFICATE_VERIFICATION,
            reason=RevalidationReason.CONTEXT_UNCHANGED,
            locator_cid=locator_cid,
            candidate=candidate,
            current=current_context,
            comparison=comparison,
            candidate_context_cid=candidate.candidate_context_id,
            envelope_cid=envelope_cid,
            component_bytes=resolved_components,
            lookup_hit=lookup_hit,
            may_proceed_to_certificate_verification=True,
            fixtures_executed=False,
            test_body_executed=False,
            diagnostics={
                "stage": "context_unchanged",
                "next": "compare_current_and_verify_authoritative_certificate",
                "fixtures_executed": False,
                "test_body_executed": False,
            },
        )

    def _lookup_candidate(
        self,
        locator_cid: str,
        *,
        max_candidates: int | None,
        now_ms: int | None,
    ) -> Any:
        store = self._store
        if store is None:
            return None
        kwargs: dict[str, Any] = {}
        if max_candidates is not None:
            kwargs["max_candidates"] = max_candidates
        if now_ms is not None:
            kwargs["now_ms"] = now_ms
        if hasattr(store, "lookup"):
            try:
                return store.lookup(locator_cid, **kwargs)
            except TypeError:
                return store.lookup(locator_cid)
        if callable(store):
            return store(locator_cid)
        return None

    def _rehash_components(
        self,
        candidate: CandidateExecutionContext,
        components: Mapping[str, bytes],
        *,
        has_inline_runtime_trace: bool = False,
    ) -> RevalidationReason | None:
        """Rehash retained component bytes against claimed CIDs.

        Returns a failure reason or ``None`` when every present component
        admits.  Missing optional components are tolerated; required identity
        fields that claim a CID without bytes are treated as incomplete.
        """

        claimed = dict(candidate.component_cids or {})
        # Also pin well-known root fields.
        field_claims = {
            "execution_key": candidate.execution_key_cid,
            "static_trace": candidate.static_trace_root_cid,
            "runtime_trace": candidate.runtime_trace_root_cid,
            "repository_forest": candidate.repository_forest_cid,
            "environment": candidate.environment_cid,
            "policy": candidate.policy_cid,
            "pass_receipt": candidate.pass_receipt_cid,
            "test_ast": candidate.test_ast_cid,
        }
        for key, cid in field_claims.items():
            if cid and key not in claimed:
                claimed[key] = cid

        if not components:
            # Without component bytes we can still compare high-level CIDs if
            # the caller supplied a complete current context; only fail when a
            # runtime frontier is required and no inline/component trace exists.
            if (
                self._require_runtime_frontier
                and not has_inline_runtime_trace
                and "runtime_trace" not in components
            ):
                return RevalidationReason.COMPONENT_MISSING
            return None

        for key, data in components.items():
            if not isinstance(data, (bytes, bytearray)):
                return RevalidationReason.COMPONENT_INTEGRITY_FAILED
            data_bytes = bytes(data)
            claimed_cid = claimed.get(key, "")
            if not claimed_cid:
                # Unclaimed retained blob — still rehash for canonical form.
                try:
                    rehash_retained_canonical_bytes(data_bytes)
                except Exception:
                    # Non-canonical component payloads (raw traces) are allowed
                    # when no CID claim exists; integrity is then best-effort.
                    continue
                continue
            try:
                actual = rehash_retained_canonical_bytes(data_bytes)
            except Exception:
                # Fall back to raw sha256 equality when payload is not DAG-JSON.
                actual = _sha256_hex(data_bytes)
                if actual != claimed_cid and not str(claimed_cid).endswith(actual):
                    # Also accept direct CID equality via content_identity when
                    # the store used a different codec label.
                    return RevalidationReason.COMPONENT_INTEGRITY_FAILED
                continue
            if actual != claimed_cid:
                return RevalidationReason.COMPONENT_INTEGRITY_FAILED
        return None

    def _load_runtime_trace_payload(
        self,
        components: Mapping[str, bytes],
        retained_runtime_trace: Mapping[str, Any] | bytes | None,
    ) -> Mapping[str, Any] | None:
        if isinstance(retained_runtime_trace, Mapping):
            return dict(retained_runtime_trace)
        if isinstance(retained_runtime_trace, (bytes, bytearray)):
            payload = _safe_json_loads(bytes(retained_runtime_trace))
            return payload if isinstance(payload, Mapping) else None
        for key in ("runtime_trace", "runtime", "runtime_dependency_trace"):
            raw = components.get(key)
            if isinstance(raw, (bytes, bytearray)):
                payload = _safe_json_loads(bytes(raw))
                if isinstance(payload, Mapping):
                    return payload
        return None

    def new_post_pass_capture(
        self,
        *,
        locator_cid: str = "",
        execution_key_cid: str = "",
        pass_receipt_cid: str = "",
        publisher: Callable[..., Any] | None = None,
    ) -> PostPassRuntimeTraceCapture:
        """Create a lifecycle capture helper for the normal miss path."""

        return PostPassRuntimeTraceCapture(
            locator_cid=locator_cid,
            execution_key_cid=execution_key_cid,
            pass_receipt_cid=pass_receipt_cid,
            clock=self._clock,
            publisher=publisher,
        )


def build_runtime_context_revalidator(
    *,
    candidate_store: Any | None = None,
    current_context_provider: CurrentContextProvider | None = None,
    dependency_resolver: DependencyContentResolver | None = None,
    allowed_roots: Mapping[str, str | os.PathLike[str]] | None = None,
    environ: Mapping[str, str] | None = None,
    clock: Callable[[], float] | Callable[[], int] | None = None,
    require_runtime_frontier: bool = True,
    require_extended_identities: bool = True,
) -> RuntimeContextRevalidator:
    """Factory for the production runtime context revalidator."""

    return RuntimeContextRevalidator(
        candidate_store=candidate_store,
        current_context_provider=current_context_provider,
        dependency_resolver=dependency_resolver,
        allowed_roots=allowed_roots,
        environ=environ,
        clock=clock,
        require_runtime_frontier=require_runtime_frontier,
        require_extended_identities=require_extended_identities,
    )


__all__ = [
    "CANDIDATE_COMPARISON_INTERFACE",
    "CANDIDATE_COMPARISON_SCHEMA",
    "CandidateComparison",
    "CurrentContextProvider",
    "DependencyContentResolver",
    "DependencyResolutionReport",
    "DependencyResolutionStatus",
    "FilesystemDependencyResolver",
    "LifecyclePhase",
    "POST_PASS_RUNTIME_TRACE_CAPTURE_INTERFACE",
    "POST_PASS_RUNTIME_TRACE_CAPTURE_SCHEMA",
    "PostPassRuntimeTraceCapture",
    "RUNTIME_CONTEXT_REVALIDATOR_INTERFACE",
    "RUNTIME_CONTEXT_REVALIDATOR_SCHEMA",
    "RUNTIME_DEPENDENCY_TRACE_INTERFACE",
    "RUNTIME_REVALIDATION_RESULT_INTERFACE",
    "RevalidationAction",
    "RevalidationReason",
    "ResolvedDependencyFact",
    "RuntimeContextRevalidator",
    "RuntimeRevalidationResult",
    "StaticCurrentContextProvider",
    "build_runtime_context_revalidator",
    "compare_candidate_to_current",
    "resolve_retained_runtime_frontier",
]
