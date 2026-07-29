"""Provider surface health backlog (SCA-609 / SCAEV179INDEXGRAPH).

Converts provider package MCP surface extraction outcomes into a typed,
content-addressed health ledger:

* unresolved registrations remain explicit evidence
* provider parse / root contradictions remain typed
* repair families are deduplicated (no per-file prompts)
* any mandatory unresolved surface blocks exhaustive parity
* rescans are zero-model and duplicate-free when inputs are unchanged

Graph authority and provider namespaces stay distinct.  This module never
synthesizes actual routes from expected descriptors.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Iterable

from ..proof.formal_verification_contracts import content_identity
from .python_mcp_surface_extractor import (
    PythonMcpPackageSurface,
    PythonMcpSurfaceExtractor,
    UnresolvedRegistration,
)
from .repository_indexer import MultiRootRepositoryIndex
from .repository_snapshot import (
    DEFAULT_PROVIDER_PACKAGE_SPECS,
    ProviderPackageSpec,
)


PROVIDER_SURFACE_HEALTH_INTERFACE: Final = "ProviderSurfaceHealth@1"
PROVIDER_SURFACE_HEALTH_VERSION: Final = "1"
PROVIDER_SURFACE_HEALTH_EVIDENCE: Final = "SCAEV179INDEXGRAPH"

PROVIDER_SURFACE_HEALTH_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/provider-surface-health@1"
)
PROVIDER_SURFACE_HEALTH_FAMILY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/provider-surface-health-family@1"
)
PROVIDER_SURFACE_HEALTH_ISSUE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/provider-surface-health-issue@1"
)
PROVIDER_SURFACE_HEALTH_BACKLOG_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/provider-surface-health-backlog@1"
)

DEFAULT_PROVIDER_SURFACE_HEALTH_BACKLOG_RELATIVE: Final = (
    "data/agent_supervisor/swissknife_contract_assurance/"
    "provider-surface-health/backlog.json"
)

# Hard envelope for published backlog artifacts.
DEFAULT_MAX_BACKLOG_BYTES: Final = 1_000_000


class ProviderSurfaceHealthError(ValueError):
    """Invalid provider surface health input or serialization failure."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "provider_surface_health_error",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.details = dict(details or {})


class ProviderSurfaceIssueKind(str, Enum):
    """Closed vocabulary for provider surface health issues."""

    UNRESOLVED_REGISTRATION = "unresolved_registration"
    PROVIDER_PARSE_FAILURE = "provider_parse_failure"
    PROVIDER_ROOT_CONTRADICTION = "provider_root_contradiction"
    MISSING_PACKAGE_SURFACE = "missing_package_surface"
    EMPTY_PACKAGE_SURFACE = "empty_package_surface"
    SYMBOL_EXTRACTION_INCOMPLETE = "symbol_extraction_incomplete"
    OPAQUE_GITLINK = "opaque_gitlink"


class ProviderSurfaceSeverity(str, Enum):
    BLOCKING = "blocking"
    ADVISORY = "advisory"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_hex(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Any) -> None:
    encoded = (_canonical_json(value).encode("utf-8")) + b"\n"
    _atomic_write_bytes(path, encoded)


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value
    else:
        raise ProviderSurfaceHealthError(f"{name} must be a string")
    text = text.strip()
    if required and not text:
        raise ProviderSurfaceHealthError(f"{name} is required")
    return text


@dataclass(frozen=True)
class ProviderSurfaceHealthIssue:
    """One typed unresolved registration or provider parse failure."""

    kind: ProviderSurfaceIssueKind
    package: str
    reason_code: str
    severity: ProviderSurfaceSeverity = ProviderSurfaceSeverity.BLOCKING
    path: str = ""
    symbol: str = ""
    detail: str = ""
    source_id: str = ""
    multi_root_id: str = ""
    repository_tree_id: str = ""
    metadata: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", ProviderSurfaceIssueKind(self.kind))
        object.__setattr__(
            self, "severity", ProviderSurfaceSeverity(self.severity)
        )
        object.__setattr__(self, "package", _text(self.package, "package"))
        object.__setattr__(
            self, "reason_code", _text(self.reason_code, "reason_code")
        )
        object.__setattr__(self, "path", _text(self.path, "path", required=False))
        object.__setattr__(
            self, "symbol", _text(self.symbol, "symbol", required=False)
        )
        object.__setattr__(
            self, "detail", _text(self.detail, "detail", required=False)
        )
        object.__setattr__(
            self, "source_id", _text(self.source_id, "source_id", required=False)
        )
        object.__setattr__(
            self,
            "multi_root_id",
            _text(self.multi_root_id, "multi_root_id", required=False),
        )
        object.__setattr__(
            self,
            "repository_tree_id",
            _text(
                self.repository_tree_id,
                "repository_tree_id",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata or {})),
        )

    @property
    def issue_id(self) -> str:
        return content_identity(self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": PROVIDER_SURFACE_HEALTH_ISSUE_SCHEMA,
            "kind": self.kind.value,
            "package": self.package,
            "reason_code": self.reason_code,
            "severity": self.severity.value,
            "path": self.path,
            "symbol": self.symbol,
            "detail": self.detail,
            "source_id": self.source_id,
            "multi_root_id": self.multi_root_id,
            "repository_tree_id": self.repository_tree_id,
            "metadata": dict(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._payload(),
            "issue_id": self.issue_id,
        }


@dataclass(frozen=True)
class ProviderSurfaceHealthFamily:
    """Deduplicated repair family (content-addressed expansion handle)."""

    family_key: str
    kind: ProviderSurfaceIssueKind
    package: str
    reason_code: str
    issue_ids: tuple[str, ...]
    predicted_files: tuple[str, ...]
    severity: ProviderSurfaceSeverity = ProviderSurfaceSeverity.BLOCKING
    member_count: int = 0
    detail: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", ProviderSurfaceIssueKind(self.kind))
        object.__setattr__(
            self, "severity", ProviderSurfaceSeverity(self.severity)
        )
        object.__setattr__(
            self, "family_key", _text(self.family_key, "family_key")
        )
        object.__setattr__(self, "package", _text(self.package, "package"))
        object.__setattr__(
            self, "reason_code", _text(self.reason_code, "reason_code")
        )
        ids = tuple(sorted({_text(item, "issue_id") for item in self.issue_ids}))
        object.__setattr__(self, "issue_ids", ids)
        files = tuple(
            sorted({_text(item, "predicted_file", required=False) for item in self.predicted_files if str(item or "").strip()})
        )
        object.__setattr__(self, "predicted_files", files)
        count = int(self.member_count or len(ids))
        if count < 0:
            raise ProviderSurfaceHealthError("member_count must be non-negative")
        object.__setattr__(self, "member_count", count)
        object.__setattr__(
            self, "detail", _text(self.detail, "detail", required=False)
        )

    @property
    def family_id(self) -> str:
        return content_identity(self._payload())

    @property
    def blocks_exhaustive_parity(self) -> bool:
        return self.severity is ProviderSurfaceSeverity.BLOCKING

    def expansion_handle(self) -> dict[str, Any]:
        """Bounded CodeEditPacket-compatible expansion handle (not a prompt)."""

        return {
            "interface": "CodeEditPacket@1",
            "kind": "provider_surface_health_expansion",
            "task_id": self.family_id,
            "package": self.package,
            "reason_code": self.reason_code,
            "predicted_files": list(self.predicted_files),
            "implementable": False,
            "non_implementable_reasons": (
                "provider_surface_health_requires_source_repair",
            ),
            "member_count": self.member_count,
            "llm_call_count": 0,
        }

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": PROVIDER_SURFACE_HEALTH_FAMILY_SCHEMA,
            "family_key": self.family_key,
            "kind": self.kind.value,
            "package": self.package,
            "reason_code": self.reason_code,
            "severity": self.severity.value,
            "issue_ids": list(self.issue_ids),
            "predicted_files": list(self.predicted_files),
            "member_count": self.member_count,
            "detail": self.detail,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._payload(),
            "family_id": self.family_id,
            "blocks_exhaustive_parity": self.blocks_exhaustive_parity,
            "expansion_handle": self.expansion_handle(),
        }


@dataclass(frozen=True)
class ProviderSurfaceHealthReport:
    """Content-addressed provider surface health receipt."""

    multi_root_id: str
    snapshot_id: str
    issues: tuple[ProviderSurfaceHealthIssue, ...]
    families: tuple[ProviderSurfaceHealthFamily, ...]
    package_surface_ids: tuple[str, ...] = ()
    packages_scanned: tuple[str, ...] = ()
    llm_call_count: int = 0
    evidence: str = PROVIDER_SURFACE_HEALTH_EVIDENCE

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "multi_root_id",
            _text(self.multi_root_id, "multi_root_id", required=False),
        )
        object.__setattr__(
            self,
            "snapshot_id",
            _text(self.snapshot_id, "snapshot_id", required=False),
        )
        issues = tuple(
            sorted(self.issues, key=lambda item: (item.package, item.issue_id))
        )
        families = tuple(
            sorted(self.families, key=lambda item: (item.package, item.family_id))
        )
        if not all(isinstance(item, ProviderSurfaceHealthIssue) for item in issues):
            raise ProviderSurfaceHealthError("issues must be ProviderSurfaceHealthIssue")
        if not all(
            isinstance(item, ProviderSurfaceHealthFamily) for item in families
        ):
            raise ProviderSurfaceHealthError(
                "families must be ProviderSurfaceHealthFamily"
            )
        object.__setattr__(self, "issues", issues)
        object.__setattr__(self, "families", families)
        object.__setattr__(
            self,
            "package_surface_ids",
            tuple(sorted({str(item) for item in self.package_surface_ids if item})),
        )
        object.__setattr__(
            self,
            "packages_scanned",
            tuple(sorted({str(item) for item in self.packages_scanned if item})),
        )
        if not isinstance(self.llm_call_count, int) or self.llm_call_count != 0:
            raise ProviderSurfaceHealthError(
                "provider surface health must record zero LLM calls"
            )
        object.__setattr__(
            self,
            "evidence",
            _text(self.evidence, "evidence") or PROVIDER_SURFACE_HEALTH_EVIDENCE,
        )

    @property
    def report_id(self) -> str:
        return content_identity(self._payload())

    @property
    def blocking_issue_count(self) -> int:
        return sum(
            1
            for item in self.issues
            if item.severity is ProviderSurfaceSeverity.BLOCKING
        )

    @property
    def blocking_family_count(self) -> int:
        return sum(1 for item in self.families if item.blocks_exhaustive_parity)

    @property
    def exhaustive_parity_allowed(self) -> bool:
        """Exhaustive parity requires zero blocking surface issues."""

        return self.blocking_issue_count == 0 and self.blocking_family_count == 0

    @property
    def unresolved_registration_count(self) -> int:
        return sum(
            1
            for item in self.issues
            if item.kind is ProviderSurfaceIssueKind.UNRESOLVED_REGISTRATION
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": PROVIDER_SURFACE_HEALTH_SCHEMA,
            "interface": PROVIDER_SURFACE_HEALTH_INTERFACE,
            "version": PROVIDER_SURFACE_HEALTH_VERSION,
            "evidence": self.evidence,
            "multi_root_id": self.multi_root_id,
            "snapshot_id": self.snapshot_id,
            "packages_scanned": list(self.packages_scanned),
            "package_surface_ids": list(self.package_surface_ids),
            "issues": [item.to_dict() for item in self.issues],
            "families": [item.to_dict() for item in self.families],
            "llm_call_count": 0,
            "exhaustive_parity_allowed": self.exhaustive_parity_allowed,
            "blocking_issue_count": self.blocking_issue_count,
            "blocking_family_count": self.blocking_family_count,
            "unresolved_registration_count": self.unresolved_registration_count,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._payload(),
            "report_id": self.report_id,
        }

    def to_backlog_dict(self) -> dict[str, Any]:
        """Compact backlog document for durable publication."""

        return {
            "schema": PROVIDER_SURFACE_HEALTH_BACKLOG_SCHEMA,
            "schema_version": 1,
            "interface": PROVIDER_SURFACE_HEALTH_INTERFACE,
            "evidence": self.evidence,
            "report_id": self.report_id,
            "multi_root_id": self.multi_root_id,
            "snapshot_id": self.snapshot_id,
            "llm_call_count": 0,
            "exhaustive_parity_allowed": self.exhaustive_parity_allowed,
            "blocking_issue_count": self.blocking_issue_count,
            "blocking_family_count": self.blocking_family_count,
            "unresolved_registration_count": self.unresolved_registration_count,
            "packages_scanned": list(self.packages_scanned),
            "package_surface_ids": list(self.package_surface_ids),
            "families": [item.to_dict() for item in self.families],
            "issues": [item.to_dict() for item in self.issues],
            # Explicit non-authoritative posture for repair projection.
            "per_file_prompts": False,
            "authority": "diagnostic",
            "actual_surfaces_complete": self.exhaustive_parity_allowed,
        }


def _family_key(
    *,
    kind: ProviderSurfaceIssueKind,
    package: str,
    reason_code: str,
    path_family: str = "",
) -> str:
    return _sha256_hex(
        {
            "kind": kind.value,
            "package": package,
            "reason_code": reason_code,
            "path_family": path_family,
        }
    )


def _path_family(path: str) -> str:
    text = str(path or "").replace("\\", "/").strip("/")
    if not text:
        return ""
    parts = text.split("/")
    if len(parts) <= 2:
        return text
    # Cluster by package-local directory so families stay coarse.
    return "/".join(parts[:2])


def _issue_from_unresolved(
    item: UnresolvedRegistration,
    *,
    multi_root_id: str,
    repository_tree_id: str,
) -> ProviderSurfaceHealthIssue:
    path = item.span.path if item.span is not None else ""
    return ProviderSurfaceHealthIssue(
        kind=ProviderSurfaceIssueKind.UNRESOLVED_REGISTRATION,
        package=item.provider,
        reason_code=item.reason.value,
        severity=ProviderSurfaceSeverity.BLOCKING,
        path=path,
        symbol=item.expression,
        detail=item.detail or item.registration_api,
        source_id=item.unresolved_id,
        multi_root_id=multi_root_id,
        repository_tree_id=repository_tree_id,
        metadata={
            "registration_api": item.registration_api,
            "expression": item.expression,
        },
    )


def _issues_from_multi_root(
    multi_root_index: MultiRootRepositoryIndex,
) -> list[ProviderSurfaceHealthIssue]:
    issues: list[ProviderSurfaceHealthIssue] = []
    multi_root_id = multi_root_index.multi_root_id
    for provider in multi_root_index.providers:
        package = provider.package
        tree_id = provider.observation.head_tree_id or provider.observation.index_tree_id
        if provider.opaque_gitlink or not provider.indexed:
            issues.append(
                ProviderSurfaceHealthIssue(
                    kind=ProviderSurfaceIssueKind.OPAQUE_GITLINK
                    if provider.opaque_gitlink
                    else ProviderSurfaceIssueKind.MISSING_PACKAGE_SURFACE,
                    package=package,
                    reason_code=(
                        provider.observation.reason_code
                        or (
                            "opaque_gitlink"
                            if provider.opaque_gitlink
                            else "provider_root_not_indexed"
                        )
                    ),
                    severity=ProviderSurfaceSeverity.BLOCKING,
                    path=provider.observation.scope_path,
                    detail="provider source was not indexed as an independent root",
                    source_id=provider.observation.observation_id,
                    multi_root_id=multi_root_id,
                    repository_tree_id=tree_id,
                )
            )
        if not provider.symbol_extraction_complete and provider.symbol_extraction_enabled:
            issues.append(
                ProviderSurfaceHealthIssue(
                    kind=ProviderSurfaceIssueKind.SYMBOL_EXTRACTION_INCOMPLETE,
                    package=package,
                    reason_code="symbol_extraction_incomplete",
                    severity=ProviderSurfaceSeverity.BLOCKING,
                    path=provider.observation.scope_path,
                    detail=",".join(provider.symbol_extraction_reason_codes)
                    or "symbol extraction incomplete",
                    multi_root_id=multi_root_id,
                    repository_tree_id=tree_id,
                    metadata={
                        "eligible": provider.symbol_eligible_file_count,
                        "extracted": provider.symbol_extracted_file_count,
                        "failed": provider.symbol_failed_file_count,
                        "reason_codes": list(provider.symbol_extraction_reason_codes),
                    },
                )
            )
        if provider.health is not None:
            from .analyzer_health import AnalyzerHealthStatus

            if provider.health.status is not AnalyzerHealthStatus.HEALTHY:
                status = provider.health.status.value
                for reason in provider.health.reasons[:16] or ("provider_unhealthy",):
                    issues.append(
                        ProviderSurfaceHealthIssue(
                            kind=ProviderSurfaceIssueKind.PROVIDER_PARSE_FAILURE,
                            package=package,
                            reason_code=str(reason),
                            severity=ProviderSurfaceSeverity.BLOCKING,
                            path=provider.observation.scope_path,
                            detail=f"provider analyzer health is {status}",
                            multi_root_id=multi_root_id,
                            repository_tree_id=tree_id,
                        )
                    )
        for contradiction in provider.observation.contradictions:
            issues.append(
                ProviderSurfaceHealthIssue(
                    kind=ProviderSurfaceIssueKind.PROVIDER_ROOT_CONTRADICTION,
                    package=package,
                    reason_code=contradiction.kind.value,
                    severity=ProviderSurfaceSeverity.BLOCKING,
                    path=contradiction.scope_path or provider.observation.scope_path,
                    detail=contradiction.detail,
                    source_id=contradiction.contradiction_id,
                    multi_root_id=multi_root_id,
                    repository_tree_id=tree_id,
                )
            )
    for contradiction in multi_root_index.contradictions:
        # Avoid double-counting observation-level contradictions already emitted.
        if any(
            item.source_id == contradiction.contradiction_id
            for item in issues
        ):
            continue
        issues.append(
            ProviderSurfaceHealthIssue(
                kind=ProviderSurfaceIssueKind.PROVIDER_ROOT_CONTRADICTION,
                package=contradiction.package,
                reason_code=contradiction.kind.value,
                severity=ProviderSurfaceSeverity.BLOCKING,
                path=contradiction.scope_path,
                detail=contradiction.detail,
                source_id=contradiction.contradiction_id,
                multi_root_id=multi_root_id,
            )
        )
    return issues


def _cluster_families(
    issues: Sequence[ProviderSurfaceHealthIssue],
) -> tuple[ProviderSurfaceHealthFamily, ...]:
    buckets: dict[str, list[ProviderSurfaceHealthIssue]] = {}
    for issue in issues:
        key = _family_key(
            kind=issue.kind,
            package=issue.package,
            reason_code=issue.reason_code,
            path_family=_path_family(issue.path),
        )
        buckets.setdefault(key, []).append(issue)

    families: list[ProviderSurfaceHealthFamily] = []
    for key, members in buckets.items():
        first = members[0]
        predicted = tuple(
            sorted({item.path for item in members if item.path})
        )
        families.append(
            ProviderSurfaceHealthFamily(
                family_key=key,
                kind=first.kind,
                package=first.package,
                reason_code=first.reason_code,
                issue_ids=tuple(item.issue_id for item in members),
                predicted_files=predicted,
                severity=first.severity,
                member_count=len(members),
                detail=first.detail,
            )
        )
    return tuple(sorted(families, key=lambda item: (item.package, item.family_id)))


def assess_provider_surface_health(
    *,
    package_surfaces: Sequence[PythonMcpPackageSurface] | None = None,
    multi_root_index: MultiRootRepositoryIndex | None = None,
    multi_root_id: str = "",
    snapshot_id: str = "",
    required_packages: Sequence[str] | None = None,
) -> ProviderSurfaceHealthReport:
    """Build a typed provider surface health report from cold-scan evidence.

    Unresolved registrations and provider root/parse failures remain blocking.
    Expected descriptors cannot clear unresolved surfaces.
    """

    surfaces = tuple(package_surfaces or ())
    multi_id = multi_root_id
    snap_id = snapshot_id
    if multi_root_index is not None:
        multi_id = multi_id or multi_root_index.multi_root_id
        snap_id = snap_id or multi_root_index.multi_root_snapshot.multi_root_id

    issues: list[ProviderSurfaceHealthIssue] = []
    packages_scanned: set[str] = set()
    surface_ids: list[str] = []

    if multi_root_index is not None:
        issues.extend(_issues_from_multi_root(multi_root_index))
        packages_scanned.update(item.package for item in multi_root_index.providers)

    seen_unresolved: set[str] = set()
    for surface in surfaces:
        packages_scanned.add(surface.provider)
        surface_ids.append(surface.surface_id)
        if not surface.tools and not surface.unresolved:
            issues.append(
                ProviderSurfaceHealthIssue(
                    kind=ProviderSurfaceIssueKind.EMPTY_PACKAGE_SURFACE,
                    package=surface.provider,
                    reason_code="empty_package_surface",
                    severity=ProviderSurfaceSeverity.BLOCKING,
                    detail="package surface extraction returned no tools",
                    source_id=surface.surface_id,
                    multi_root_id=multi_id,
                    repository_tree_id=surface.repository_tree_id,
                )
            )
        for unresolved in surface.unresolved:
            issue = _issue_from_unresolved(
                unresolved,
                multi_root_id=multi_id,
                repository_tree_id=surface.repository_tree_id,
            )
            if issue.issue_id in seen_unresolved:
                continue
            seen_unresolved.add(issue.issue_id)
            issues.append(issue)

    required = tuple(
        required_packages
        if required_packages is not None
        else tuple(item.package for item in DEFAULT_PROVIDER_PACKAGE_SPECS)
    )
    present = {surface.provider for surface in surfaces}
    if multi_root_index is not None:
        present |= {
            item.package
            for item in multi_root_index.providers
            if item.indexed and not item.opaque_gitlink
        }
    for package in required:
        if package not in present and package not in packages_scanned:
            issues.append(
                ProviderSurfaceHealthIssue(
                    kind=ProviderSurfaceIssueKind.MISSING_PACKAGE_SURFACE,
                    package=package,
                    reason_code="required_package_surface_missing",
                    severity=ProviderSurfaceSeverity.BLOCKING,
                    detail="required provider package surface was not extracted",
                    multi_root_id=multi_id,
                    repository_tree_id=snap_id,
                )
            )
            packages_scanned.add(package)

    # Deduplicate issues by content identity.
    unique: dict[str, ProviderSurfaceHealthIssue] = {}
    for issue in issues:
        unique[issue.issue_id] = issue
    issue_tuple = tuple(unique.values())
    families = _cluster_families(issue_tuple)

    return ProviderSurfaceHealthReport(
        multi_root_id=multi_id,
        snapshot_id=snap_id,
        issues=issue_tuple,
        families=families,
        package_surface_ids=tuple(surface_ids),
        packages_scanned=tuple(packages_scanned),
        llm_call_count=0,
        evidence=PROVIDER_SURFACE_HEALTH_EVIDENCE,
    )


def extract_actual_provider_package_surfaces(
    superproject_root: Path | str,
    *,
    multi_root_index: MultiRootRepositoryIndex | None = None,
    provider_packages: Sequence[ProviderPackageSpec | Mapping[str, Any]] | None = None,
    extractor: PythonMcpSurfaceExtractor | None = None,
    max_files: int = 4_000,
    max_file_bytes: int = 512 * 1024,
    max_total_bytes: int = 64 * 1024 * 1024,
) -> tuple[PythonMcpPackageSurface, ...]:
    """Cold-extract actual MCP surfaces from independent provider package roots.

    Never imports provider packages.  Opaque or missing roots are skipped (and
    must surface as typed health issues via :func:`assess_provider_surface_health`).
    """

    root = Path(superproject_root).expanduser().resolve()
    surface_extractor = extractor or PythonMcpSurfaceExtractor(
        max_files=max_files,
        max_file_bytes=max_file_bytes,
        max_total_bytes=max_total_bytes,
    )
    surfaces: list[PythonMcpPackageSurface] = []

    if multi_root_index is not None:
        for provider in multi_root_index.providers:
            package_root = str(provider.observation.package_root or "").strip()
            if not package_root or not Path(package_root).is_dir():
                continue
            if provider.opaque_gitlink or not provider.indexed:
                continue
            tree_id = (
                provider.observation.head_tree_id
                or provider.observation.index_tree_id
                or multi_root_index.multi_root_id
            )
            surfaces.append(
                surface_extractor.extract_package(
                    package_root,
                    provider=provider.package,
                    repository_tree_id=tree_id,
                )
            )
        return tuple(surfaces)

    specs: list[ProviderPackageSpec] = []
    if provider_packages is None:
        specs = list(DEFAULT_PROVIDER_PACKAGE_SPECS)
    else:
        for item in provider_packages:
            if isinstance(item, ProviderPackageSpec):
                specs.append(item)
            else:
                specs.append(
                    ProviderPackageSpec(
                        package=str(item.get("package") or ""),
                        scope_path=str(item.get("scope_path") or ""),
                        package_dirname=str(
                            item.get("package_dirname")
                            or item.get("package")
                            or ""
                        ),
                    )
                )

    for spec in specs:
        package_path = root.joinpath(*Path(spec.scope_path).parts, spec.package_dirname)
        if not package_path.is_dir():
            continue
        surfaces.append(
            surface_extractor.extract_package(
                package_path,
                provider=spec.package,
                repository_tree_id="",
            )
        )
    return tuple(surfaces)


def write_provider_surface_health_backlog(
    report: ProviderSurfaceHealthReport,
    destination: Path | str | None = None,
    *,
    max_bytes: int = DEFAULT_MAX_BACKLOG_BYTES,
) -> Path:
    """Atomically publish the provider surface health backlog document."""

    if not isinstance(report, ProviderSurfaceHealthReport):
        raise ProviderSurfaceHealthError("report must be ProviderSurfaceHealthReport")
    path = Path(
        destination
        if destination is not None
        else DEFAULT_PROVIDER_SURFACE_HEALTH_BACKLOG_RELATIVE
    )
    payload = report.to_backlog_dict()
    encoded = (_canonical_json(payload).encode("utf-8")) + b"\n"
    if len(encoded) > int(max_bytes):
        raise ProviderSurfaceHealthError(
            "provider surface health backlog exceeds max_bytes",
            reason_code="backlog_oversized",
            details={"byte_count": len(encoded), "max_bytes": int(max_bytes)},
        )
    _atomic_json(path, payload)
    return path


def provider_surface_health_blocks_exhaustive_parity(
    report: ProviderSurfaceHealthReport | Mapping[str, Any] | None,
) -> bool:
    """Return True when provider surface health must block exhaustive parity."""

    if report is None:
        return True
    if isinstance(report, ProviderSurfaceHealthReport):
        return not report.exhaustive_parity_allowed
    return not bool(report.get("exhaustive_parity_allowed"))


__all__ = [
    "DEFAULT_MAX_BACKLOG_BYTES",
    "DEFAULT_PROVIDER_SURFACE_HEALTH_BACKLOG_RELATIVE",
    "PROVIDER_SURFACE_HEALTH_BACKLOG_SCHEMA",
    "PROVIDER_SURFACE_HEALTH_EVIDENCE",
    "PROVIDER_SURFACE_HEALTH_FAMILY_SCHEMA",
    "PROVIDER_SURFACE_HEALTH_INTERFACE",
    "PROVIDER_SURFACE_HEALTH_ISSUE_SCHEMA",
    "PROVIDER_SURFACE_HEALTH_SCHEMA",
    "PROVIDER_SURFACE_HEALTH_VERSION",
    "ProviderSurfaceHealthError",
    "ProviderSurfaceHealthFamily",
    "ProviderSurfaceHealthIssue",
    "ProviderSurfaceHealthReport",
    "ProviderSurfaceIssueKind",
    "ProviderSurfaceSeverity",
    "assess_provider_surface_health",
    "extract_actual_provider_package_surfaces",
    "provider_surface_health_blocks_exhaustive_parity",
    "write_provider_surface_health_backlog",
]
