"""Repository, checkout, scope, and dirty-tree resolution (ASE-005).

This module is the leaf repository target resolver for prompt-only entrypoints.
It reuses :mod:`ipfs_accelerate_py.agent_supervisor.repository_forest` checkout
authority and snapshot helpers to select one allowlisted Git root/scope and
bind repository ID, checkout identity, HEAD/dirty overlay, and submodule
population.

Design rules enforced here:

- selection is deterministic under identical frozen evidence;
- only allowlisted roots may be selected (never widen from discovery);
- nearest enclosing Git toplevel wins for nested and submodule topologies;
- worktree checkouts bind a checkout-specific identity while sharing the
  portable repository ID derived from the common Git directory;
- dirty identity always observes staged, modified, deleted, and
  admitted-untracked overlay state (HEAD alone is never sufficient);
- symlink roots, parent-traversal scope, and equal-rank multi-root ambiguity
  fail closed or return a typed preview ambiguity without widening roots;
- prompt text and untrusted path labels cannot select a target outside the
  configured allowlist.
"""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar, Final, Iterable

from ipfs_accelerate_py.agent_supervisor.multiformats_identity import (
    cid_for_dag_json,
)
from ipfs_accelerate_py.agent_supervisor.repository_forest import (
    AuthorityMode,
    IgnorePolicy,
    RepositoryAuthority,
    RepositoryDescriptor,
    RepositoryForestError,
    build_repository_descriptor,
    empty_dirty_overlay_digest,
    path_within_repository,
    resolve_repository_root,
)

from .contracts import (
    DecisionEffect,
    EntrypointContractError,
    ResolutionDisposition,
    ResolutionSource,
    RevalidationRule,
    TargetCandidate,
    TargetInferenceDecision,
)

SCHEMA_PREFIX: Final = "ipfs_accelerate_py/agent-supervisor/entrypoints"
REPOSITORY_TARGET_EVIDENCE_SCHEMA: Final = (
    f"{SCHEMA_PREFIX}/repository-target-evidence@1"
)
REPOSITORY_TARGET_RESOLUTION_SCHEMA: Final = (
    f"{SCHEMA_PREFIX}/repository-target-resolution@1"
)
REPOSITORY_BINDING_SCHEMA: Final = f"{SCHEMA_PREFIX}/repository-binding@1"
TREE_IDENTITY_SCHEMA: Final = f"{SCHEMA_PREFIX}/worktree-tree-identity@1"
NESTED_POPULATION_SCHEMA: Final = f"{SCHEMA_PREFIX}/nested-repository-population@1"
CHECKOUT_IDENTITY_SCHEMA: Final = f"{SCHEMA_PREFIX}/checkout-identity@1"

REPOSITORY_TARGET_RESOLUTION_REQUIREMENT_ID: Final = (
    "target_resolver.REPOSITORY_TARGET_RESOLUTION_REQUIREMENT_ID"
)

REPOSITORY_FIELD_NAMES: Final[tuple[str, ...]] = (
    "repository_root",
    "repository_id",
    "checkout_id",
    "scope",
    "tree_id",
    "dirty_overlay",
    "submodules",
    "nested_repositories",
)

_SOURCE_PRECEDENCE: Final[Mapping[ResolutionSource, int]] = {
    ResolutionSource.CANONICAL_REQUEST: 10,
    ResolutionSource.EXPLICIT_OVERRIDE: 20,
    ResolutionSource.EXISTING_RUN: 30,
    ResolutionSource.AUTHENTICATED_TRANSPORT: 40,
    ResolutionSource.SIGNED_PROFILE: 50,
    ResolutionSource.REPOSITORY_HINT: 60,
    ResolutionSource.DISCOVERY: 80,
    ResolutionSource.BUILTIN_DEFAULT: 90,
}

_GIT_TIMEOUT_SECONDS: Final = 30
_MAX_ALLOWLIST: Final = 64
_ALIAS_RE = re.compile(r"^[A-Za-z][A-Za-z0-9._-]{0,127}\Z")
_LOGICAL_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9._-]{0,127}\Z")


class RepositoryTargetResolverError(EntrypointContractError):
    """Raised when repository target evidence is malformed or non-authoritative."""


def _cid(label: str, payload: Mapping[str, Any] | None = None) -> str:
    body: dict[str, Any] = {"label": label}
    if payload is not None:
        body["payload"] = dict(payload)
    return cid_for_dag_json(body)


def _require_absolute_path(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise RepositoryTargetResolverError(f"{name} is required")
    if not text.startswith("/") or "\\" in text:
        raise RepositoryTargetResolverError(
            f"{name} must be an absolute POSIX path"
        )
    normalized = os.path.normpath(text)
    if normalized != text or any(part == ".." for part in text.split("/")):
        raise RepositoryTargetResolverError(
            f"{name} must be lexically normalized without parent traversal"
        )
    return text


def _optional_text(value: Any, name: str, *, maximum: int = 512) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise RepositoryTargetResolverError(f"{name} must be text")
    text = value.strip()
    if len(text.encode("utf-8")) > maximum:
        raise RepositoryTargetResolverError(f"{name} exceeds {maximum} bytes")
    return text


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise RepositoryTargetResolverError(f"{name} must be a boolean")
    return value


def _logical_name(value: Any, *, default: str = "repository") -> str:
    text = str(value or "").strip() or default
    if not _LOGICAL_NAME_RE.fullmatch(text):
        raise RepositoryTargetResolverError(
            "logical_name must be a short alphanumeric identifier"
        )
    return text


def _alias(value: Any) -> str:
    text = str(value or "").strip()
    if not text or not _ALIAS_RE.fullmatch(text):
        raise RepositoryTargetResolverError(
            "alias must be a short alphanumeric identifier"
        )
    return text


def _sorted_unique_paths(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({str(item) for item in values if str(item)}))


def _git(
    cwd: Path,
    *arguments: str,
) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=cwd,
            text=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return 1, ""
    return completed.returncode, (completed.stdout or "").strip()


def _git_toplevel(path: Path) -> Path | None:
    """Return the Git toplevel for ``path`` without following symlink escapes."""

    try:
        if not path.exists():
            return None
    except OSError:
        return None
    probe = path if path.is_dir() else path.parent
    status, output = _git(probe, "rev-parse", "--show-toplevel")
    if status != 0 or not output:
        return None
    try:
        return Path(output).resolve(strict=True)
    except (OSError, RuntimeError):
        return None


def _is_symlink_path(path: Path) -> bool:
    try:
        return path.is_symlink()
    except OSError:
        return False


def _path_is_within(candidate: Path, root: Path) -> bool:
    try:
        candidate.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except (OSError, RuntimeError, ValueError):
        return False


def _normalize_allowlisted_root(
    raw: str | Path,
    *,
    follow_symlinks: bool,
) -> tuple[Path | None, str]:
    """Return ``(resolved_root, reason)`` for one allowlist entry."""

    text = str(raw or "").strip()
    if not text:
        return None, "empty_allowlist_entry"
    try:
        path = Path(text)
        if _is_symlink_path(path) and not follow_symlinks:
            return None, "symlink_root_rejected"
        resolved = resolve_repository_root(
            path,
            follow_symlinks=follow_symlinks,
        )
    except RepositoryForestError as exc:
        return None, exc.reason_code or "root_unresolvable"
    except (OSError, RuntimeError):
        return None, "root_unresolvable"
    toplevel = _git_toplevel(resolved)
    if toplevel is None:
        return None, "not_a_git_repository"
    if toplevel != resolved:
        # Allowlist entries must themselves be Git tops. Nested paths under a
        # parent checkout do not inherit authority.
        return None, "nested_path_not_repository_root"
    return toplevel, ""


def _scope_relative_from_cwd(root: Path, cwd: Path) -> str:
    """Return absolute scope path: cwd when under root, otherwise the root."""

    try:
        resolved_cwd = cwd.resolve(strict=False)
        resolved_root = root.resolve(strict=True)
        resolved_cwd.relative_to(resolved_root)
        if resolved_cwd.is_dir():
            return str(resolved_cwd)
        return str(resolved_cwd.parent)
    except (OSError, RuntimeError, ValueError):
        return str(root)


def _reject_parent_traversal(text: str) -> bool:
    if not text:
        return False
    normalized = text.replace("\\", "/")
    parts = PurePosixPath(normalized).parts
    return ".." in parts


def _candidate(
    *,
    field_name: str,
    value: str,
    source: ResolutionSource,
    evidence_cid: str,
    confidence_ppm: int = 1_000_000,
    rejection_reason: str = "",
) -> TargetCandidate:
    return TargetCandidate(
        field_name=field_name,
        value=value,
        source=source,
        source_precedence=_SOURCE_PRECEDENCE[source],
        evidence_cid=evidence_cid,
        confidence_ppm=confidence_ppm,
        rejection_reason=rejection_reason,
    )


def _decision(
    *,
    field_name: str,
    disposition: ResolutionDisposition,
    selected_value: str,
    selected_source: ResolutionSource,
    evidence_cid: str,
    candidates: Sequence[TargetCandidate],
    reason_codes: Sequence[str],
    effect: DecisionEffect = DecisionEffect.IDENTITY_ONLY,
    override_accepted: bool = False,
    revalidation_rule: RevalidationRule = RevalidationRule.BEFORE_MUTATION,
) -> TargetInferenceDecision:
    return TargetInferenceDecision(
        field_name=field_name,
        disposition=disposition,
        selected_value=selected_value,
        selected_source=selected_source,
        source_precedence=_SOURCE_PRECEDENCE[selected_source],
        evidence_cid=evidence_cid,
        candidates=tuple(candidates),
        reason_codes=tuple(reason_codes),
        effect=effect,
        override_accepted=override_accepted,
        fresh_until_ms=0,
        revalidation_rule=revalidation_rule,
    )


def _unavailable_decisions(
    *,
    evidence_cid: str,
    reason_codes: Sequence[str],
    root_candidates: Sequence[TargetCandidate] = (),
) -> tuple[TargetInferenceDecision, ...]:
    decisions: list[TargetInferenceDecision] = []
    reasons = tuple(reason_codes) or ("repository_target_unavailable",)
    for field_name in REPOSITORY_FIELD_NAMES:
        if field_name == "repository_root" and len(root_candidates) >= 2:
            decisions.append(
                _decision(
                    field_name=field_name,
                    disposition=ResolutionDisposition.AMBIGUOUS,
                    selected_value="",
                    selected_source=ResolutionSource.DISCOVERY,
                    evidence_cid=evidence_cid,
                    candidates=root_candidates,
                    reason_codes=reasons,
                    effect=DecisionEffect.IDENTITY_ONLY,
                )
            )
            continue
        if field_name == "repository_root" and root_candidates:
            # Denied / unavailable with a single rejected candidate.
            selected_source = root_candidates[0].source
            disposition = (
                ResolutionDisposition.DENIED
                if any(item.rejection_reason for item in root_candidates)
                and all(item.rejection_reason for item in root_candidates)
                else ResolutionDisposition.UNAVAILABLE
            )
            if disposition is ResolutionDisposition.DENIED:
                decisions.append(
                    _decision(
                        field_name=field_name,
                        disposition=disposition,
                        selected_value="",
                        selected_source=selected_source,
                        evidence_cid=evidence_cid,
                        candidates=root_candidates,
                        reason_codes=reasons,
                        effect=DecisionEffect.IDENTITY_ONLY,
                    )
                )
                continue
        decisions.append(
            _decision(
                field_name=field_name,
                disposition=ResolutionDisposition.UNAVAILABLE,
                selected_value="",
                selected_source=ResolutionSource.DISCOVERY,
                evidence_cid=evidence_cid,
                candidates=(),
                reason_codes=reasons,
                effect=DecisionEffect.IDENTITY_ONLY,
            )
        )
    return tuple(decisions)


@dataclass(frozen=True)
class RepositoryRootCandidate:
    """One allowlisted Git toplevel considered for selection."""

    root_path: str
    alias: str
    source: ResolutionSource
    evidence_cid: str
    depth_under_cwd: int = -1
    rejection_reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "root_path",
            _require_absolute_path(self.root_path, "root_path"),
        )
        object.__setattr__(self, "alias", _alias(self.alias))
        if not isinstance(self.source, ResolutionSource):
            try:
                object.__setattr__(
                    self, "source", ResolutionSource(str(self.source))
                )
            except ValueError as exc:
                raise RepositoryTargetResolverError(
                    f"unknown resolution source {self.source!r}"
                ) from exc
        object.__setattr__(
            self, "evidence_cid", str(self.evidence_cid or "").strip()
        )
        if not self.evidence_cid:
            raise RepositoryTargetResolverError("evidence_cid is required")
        if not isinstance(self.depth_under_cwd, int):
            raise RepositoryTargetResolverError("depth_under_cwd must be int")
        object.__setattr__(
            self,
            "rejection_reason",
            str(self.rejection_reason or "").strip(),
        )

    @property
    def viable(self) -> bool:
        return not self.rejection_reason

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_path": self.root_path,
            "alias": self.alias,
            "source": self.source.value,
            "evidence_cid": self.evidence_cid,
            "depth_under_cwd": self.depth_under_cwd,
            "rejection_reason": self.rejection_reason,
        }


@dataclass(frozen=True)
class RepositoryTargetEvidence:
    """Frozen inputs for repository/checkout/scope/dirty resolution.

    Prompt text is accepted only so callers can prove it is ignored for target
    selection.  Absolute allowlisted roots are the sole authority for which
    checkouts may be selected.
    """

    SCHEMA: ClassVar[str] = REPOSITORY_TARGET_EVIDENCE_SCHEMA

    cwd: str
    allowlisted_roots: tuple[str, ...]
    repository_hint: str = ""
    scope_hint: str = ""
    logical_name: str = "repository"
    follow_symlinks: bool = False
    allow_dirty_overlay: bool = True
    prompt_text: str = ""
    authority_mode: str = AuthorityMode.READ_WRITE.value

    def __post_init__(self) -> None:
        object.__setattr__(self, "cwd", _require_absolute_path(self.cwd, "cwd"))
        if isinstance(self.allowlisted_roots, (str, bytes)) or not isinstance(
            self.allowlisted_roots, Sequence
        ):
            raise RepositoryTargetResolverError(
                "allowlisted_roots must be a sequence of absolute paths"
            )
        roots = tuple(str(item).strip() for item in self.allowlisted_roots)
        if not roots:
            raise RepositoryTargetResolverError(
                "allowlisted_roots must not be empty"
            )
        if len(roots) > _MAX_ALLOWLIST:
            raise RepositoryTargetResolverError(
                f"allowlisted_roots exceeds {_MAX_ALLOWLIST} entries"
            )
        normalized_roots: list[str] = []
        for index, root in enumerate(roots):
            try:
                normalized_roots.append(
                    _require_absolute_path(root, f"allowlisted_roots[{index}]")
                )
            except RepositoryTargetResolverError:
                # Keep lexical form for fail-closed reporting; validation
                # during resolve rejects unsafe/unresolvable entries.
                if not root.startswith("/") or "\\" in root or ".." in root.split("/"):
                    raise
                normalized_roots.append(root)
        object.__setattr__(
            self, "allowlisted_roots", _sorted_unique_paths(normalized_roots)
        )
        object.__setattr__(
            self,
            "repository_hint",
            _optional_text(self.repository_hint, "repository_hint"),
        )
        object.__setattr__(
            self, "scope_hint", _optional_text(self.scope_hint, "scope_hint")
        )
        object.__setattr__(
            self, "logical_name", _logical_name(self.logical_name)
        )
        object.__setattr__(
            self,
            "follow_symlinks",
            _bool(self.follow_symlinks, "follow_symlinks"),
        )
        object.__setattr__(
            self,
            "allow_dirty_overlay",
            _bool(self.allow_dirty_overlay, "allow_dirty_overlay"),
        )
        # Prompt bodies are non-authoritative and excluded from content_id.
        object.__setattr__(
            self, "prompt_text", _optional_text(self.prompt_text, "prompt_text", maximum=8192)
        )
        mode = str(self.authority_mode or AuthorityMode.READ_WRITE.value).strip()
        allowed_modes = {item.value for item in AuthorityMode}
        if mode not in allowed_modes:
            raise RepositoryTargetResolverError(
                f"unsupported authority_mode {mode!r}"
            )
        object.__setattr__(self, "authority_mode", mode)

    def _payload(self) -> dict[str, Any]:
        # Prompt text is intentionally omitted so prompt injection cannot
        # perturb evidence identity.
        return {
            "schema": self.SCHEMA,
            "cwd": self.cwd,
            "allowlisted_roots": list(self.allowlisted_roots),
            "repository_hint": self.repository_hint,
            "scope_hint": self.scope_hint,
            "logical_name": self.logical_name,
            "follow_symlinks": self.follow_symlinks,
            "allow_dirty_overlay": self.allow_dirty_overlay,
            "authority_mode": self.authority_mode,
            "requirement_id": REPOSITORY_TARGET_RESOLUTION_REQUIREMENT_ID,
        }

    @property
    def content_id(self) -> str:
        return cid_for_dag_json(self._payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["content_id"] = self.content_id
        return payload


@dataclass(frozen=True)
class RepositoryTargetBinding:
    """Unique repository/checkout/scope/dirty binding."""

    SCHEMA: ClassVar[str] = REPOSITORY_BINDING_SCHEMA

    repository_root: str
    repository_id: str
    checkout_id: str
    scope_path: str
    tree_id: str
    dirty_overlay_cid: str
    submodule_population_cid: str
    nested_repository_population_cid: str
    head_commit: str
    head_tree: str
    dirty: bool
    descriptor_cid: str
    selected_source: ResolutionSource
    alias: str

    def __post_init__(self) -> None:
        for name in ("repository_root", "scope_path"):
            object.__setattr__(
                self,
                name,
                _require_absolute_path(getattr(self, name), name),
            )
        for name in (
            "repository_id",
            "checkout_id",
            "tree_id",
            "dirty_overlay_cid",
            "submodule_population_cid",
            "nested_repository_population_cid",
            "head_commit",
            "head_tree",
            "descriptor_cid",
            "alias",
        ):
            text = str(getattr(self, name) or "").strip()
            if not text:
                raise RepositoryTargetResolverError(f"{name} is required")
            object.__setattr__(self, name, text)
        object.__setattr__(self, "dirty", _bool(self.dirty, "dirty"))
        if not isinstance(self.selected_source, ResolutionSource):
            raise RepositoryTargetResolverError("selected_source is invalid")

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "repository_root": self.repository_root,
            "repository_id": self.repository_id,
            "checkout_id": self.checkout_id,
            "scope_path": self.scope_path,
            "tree_id": self.tree_id,
            "dirty_overlay_cid": self.dirty_overlay_cid,
            "submodule_population_cid": self.submodule_population_cid,
            "nested_repository_population_cid": (
                self.nested_repository_population_cid
            ),
            "head_commit": self.head_commit,
            "head_tree": self.head_tree,
            "dirty": self.dirty,
            "descriptor_cid": self.descriptor_cid,
            "selected_source": self.selected_source.value,
            "alias": self.alias,
        }

    @property
    def content_id(self) -> str:
        return cid_for_dag_json(self._payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["content_id"] = self.content_id
        return payload


@dataclass(frozen=True)
class RepositoryTargetResolution:
    """Complete repository target resolution result."""

    SCHEMA: ClassVar[str] = REPOSITORY_TARGET_RESOLUTION_SCHEMA

    decisions: tuple[TargetInferenceDecision, ...]
    evidence_cid: str
    binding: RepositoryTargetBinding | None
    unresolved_fields: tuple[str, ...]
    reason_codes: tuple[str, ...]
    candidates_considered: tuple[RepositoryRootCandidate, ...]
    prompt_target_ignored: bool = True
    roots_widened: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.decisions, (str, bytes)) or not isinstance(
            self.decisions, Sequence
        ):
            raise RepositoryTargetResolverError("decisions must be a sequence")
        decisions = tuple(
            item
            if isinstance(item, TargetInferenceDecision)
            else TargetInferenceDecision.from_dict(item)
            for item in self.decisions
        )
        names = tuple(item.field_name for item in decisions)
        if set(names) != set(REPOSITORY_FIELD_NAMES):
            missing = set(REPOSITORY_FIELD_NAMES).difference(names)
            extra = set(names).difference(REPOSITORY_FIELD_NAMES)
            raise RepositoryTargetResolverError(
                f"repository decisions have missing={sorted(missing)} "
                f"extra={sorted(extra)}"
            )
        if len(names) != len(set(names)):
            raise RepositoryTargetResolverError(
                "repository decisions contain duplicate fields"
            )
        decisions = tuple(sorted(decisions, key=lambda item: item.field_name))
        object.__setattr__(self, "decisions", decisions)
        object.__setattr__(
            self, "evidence_cid", str(self.evidence_cid or "").strip()
        )
        if not self.evidence_cid:
            raise RepositoryTargetResolverError("evidence_cid is required")
        if self.binding is not None and not isinstance(
            self.binding, RepositoryTargetBinding
        ):
            raise RepositoryTargetResolverError(
                "binding must be RepositoryTargetBinding or None"
            )
        expected_unresolved = tuple(
            sorted(item.field_name for item in decisions if item.unresolved)
        )
        unresolved = tuple(sorted({str(item) for item in self.unresolved_fields}))
        if unresolved != expected_unresolved:
            raise RepositoryTargetResolverError(
                "unresolved_fields must match unresolved decisions"
            )
        object.__setattr__(self, "unresolved_fields", unresolved)
        object.__setattr__(
            self,
            "reason_codes",
            tuple(sorted({str(item) for item in self.reason_codes if str(item)})),
        )
        object.__setattr__(
            self,
            "candidates_considered",
            tuple(self.candidates_considered),
        )
        object.__setattr__(
            self,
            "prompt_target_ignored",
            _bool(self.prompt_target_ignored, "prompt_target_ignored"),
        )
        object.__setattr__(
            self, "roots_widened", _bool(self.roots_widened, "roots_widened")
        )
        if self.roots_widened:
            raise RepositoryTargetResolverError(
                "repository resolution must never widen allowlisted roots"
            )
        if self.binding is not None and unresolved:
            raise RepositoryTargetResolverError(
                "unique binding cannot carry unresolved fields"
            )
        if self.binding is None and not unresolved:
            raise RepositoryTargetResolverError(
                "unresolved resolution requires unresolved fields"
            )

    @property
    def unique(self) -> bool:
        return self.binding is not None and not self.unresolved_fields

    @property
    def ambiguous(self) -> bool:
        return any(
            item.disposition is ResolutionDisposition.AMBIGUOUS
            for item in self.decisions
        )

    def decision(self, field_name: str) -> TargetInferenceDecision:
        for item in self.decisions:
            if item.field_name == field_name:
                return item
        raise KeyError(field_name)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "evidence_cid": self.evidence_cid,
            "binding": None if self.binding is None else self.binding.to_dict(),
            "decisions": [item.to_dict() for item in self.decisions],
            "unresolved_fields": list(self.unresolved_fields),
            "reason_codes": list(self.reason_codes),
            "candidates_considered": [
                item.to_dict() for item in self.candidates_considered
            ],
            "prompt_target_ignored": self.prompt_target_ignored,
            "roots_widened": self.roots_widened,
            "requirement_id": REPOSITORY_TARGET_RESOLUTION_REQUIREMENT_ID,
        }

    @property
    def content_id(self) -> str:
        return cid_for_dag_json(self._payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["content_id"] = self.content_id
        return payload


@dataclass(frozen=True)
class RepositoryTargetResolver:
    """Deterministic allowlisted repository/checkout/scope/dirty resolver."""

    def resolve(
        self, evidence: RepositoryTargetEvidence
    ) -> RepositoryTargetResolution:
        if not isinstance(evidence, RepositoryTargetEvidence):
            raise RepositoryTargetResolverError(
                "resolve requires frozen RepositoryTargetEvidence"
            )
        # Prompt text is accepted only to prove non-authority.
        _ = evidence.prompt_text
        evidence_cid = evidence.content_id
        cwd = Path(evidence.cwd)

        candidates, candidate_reasons = self._discover_candidates(evidence, cwd)
        selected, selection_reasons, root_source, override_accepted = (
            self._select_candidate(evidence, candidates)
        )

        if selected is None:
            root_field_candidates = tuple(
                _candidate(
                    field_name="repository_root",
                    value=item.root_path,
                    source=item.source,
                    evidence_cid=item.evidence_cid,
                    confidence_ppm=(
                        500_000 if item.viable else 0
                    ),
                    rejection_reason=item.rejection_reason
                    or (
                        "not_selected_equal_rank"
                        if any(
                            other.viable and other.root_path != item.root_path
                            for other in candidates
                        )
                        else "repository_root_unavailable"
                    ),
                )
                for item in candidates
                if item.viable
                or item.rejection_reason
                in {
                    "symlink_root_rejected",
                    "outside_allowlist",
                    "parent_traversal_rejected",
                    "explicit_hint_outside_allowlist",
                }
            )
            # For ambiguity, alternatives must not carry rejection before
            # selection logic marks them; rebuild equal-rank viable set.
            viable = [item for item in candidates if item.viable]
            reasons = tuple(
                dict.fromkeys([*candidate_reasons, *selection_reasons])
            ) or ("repository_target_unavailable",)
            if len(viable) >= 2 and "multiple_viable_repository_roots" in reasons:
                root_field_candidates = tuple(
                    _candidate(
                        field_name="repository_root",
                        value=item.root_path,
                        source=item.source,
                        evidence_cid=item.evidence_cid,
                        confidence_ppm=500_000,
                    )
                    for item in viable
                )
                decisions = _unavailable_decisions(
                    evidence_cid=evidence_cid,
                    reason_codes=reasons,
                    root_candidates=root_field_candidates,
                )
            elif root_field_candidates and all(
                item.rejection_reason for item in root_field_candidates
            ):
                decisions = _unavailable_decisions(
                    evidence_cid=evidence_cid,
                    reason_codes=reasons,
                    root_candidates=root_field_candidates,
                )
            else:
                decisions = _unavailable_decisions(
                    evidence_cid=evidence_cid,
                    reason_codes=reasons,
                    root_candidates=(),
                )
            return RepositoryTargetResolution(
                decisions=decisions,
                evidence_cid=evidence_cid,
                binding=None,
                unresolved_fields=tuple(
                    item.field_name for item in decisions if item.unresolved
                ),
                reason_codes=reasons,
                candidates_considered=tuple(candidates),
                prompt_target_ignored=True,
                roots_widened=False,
            )

        try:
            binding, bind_reasons = self._bind_selected(
                evidence,
                selected=selected,
                selected_source=root_source,
                candidates=candidates,
            )
        except RepositoryTargetResolverError as exc:
            decisions = _unavailable_decisions(
                evidence_cid=evidence_cid,
                reason_codes=(
                    *selection_reasons,
                    *candidate_reasons,
                    "binding_failed",
                    str(exc).split(":", 1)[0].replace(" ", "_").lower()
                    if str(exc)
                    else "binding_failed",
                ),
            )
            return RepositoryTargetResolution(
                decisions=decisions,
                evidence_cid=evidence_cid,
                binding=None,
                unresolved_fields=tuple(
                    item.field_name for item in decisions if item.unresolved
                ),
                reason_codes=tuple(
                    dict.fromkeys(
                        [*candidate_reasons, *selection_reasons, "binding_failed"]
                    )
                ),
                candidates_considered=tuple(candidates),
                prompt_target_ignored=True,
                roots_widened=False,
            )

        decisions = self._decisions_for_binding(
            binding=binding,
            evidence_cid=evidence_cid,
            selected_source=root_source,
            override_accepted=override_accepted,
            candidates=candidates,
            extra_reasons=tuple(
                dict.fromkeys([*selection_reasons, *bind_reasons])
            ),
        )
        return RepositoryTargetResolution(
            decisions=decisions,
            evidence_cid=evidence_cid,
            binding=binding,
            unresolved_fields=(),
            reason_codes=tuple(
                dict.fromkeys([*selection_reasons, *bind_reasons, *candidate_reasons])
            ),
            candidates_considered=tuple(candidates),
            prompt_target_ignored=True,
            roots_widened=False,
        )

    def _discover_candidates(
        self,
        evidence: RepositoryTargetEvidence,
        cwd: Path,
    ) -> tuple[list[RepositoryRootCandidate], list[str]]:
        reasons: list[str] = []
        candidates: list[RepositoryRootCandidate] = []
        seen: set[str] = set()
        try:
            cwd_resolved = cwd.resolve(strict=False)
        except (OSError, RuntimeError):
            cwd_resolved = cwd
            reasons.append("cwd_unresolvable")

        for index, raw in enumerate(evidence.allowlisted_roots):
            alias = f"{evidence.logical_name}-{index}" if index else evidence.logical_name
            # Stable alias: first allowlisted root uses logical_name; others get
            # a deterministic suffix.  Descriptor construction requires alias ==
            # logical_name, so bind uses the evidence logical_name for the
            # selected root only.
            resolved, reason = _normalize_allowlisted_root(
                raw,
                follow_symlinks=evidence.follow_symlinks,
            )
            evidence_cid = _cid(
                "allowlisted-root",
                {"path": raw, "reason": reason or "ok"},
            )
            if resolved is None:
                candidates.append(
                    RepositoryRootCandidate(
                        root_path=(
                            raw
                            if raw.startswith("/")
                            else f"/invalid/{index}"
                        ),
                        alias=_alias(f"rejected-{index}"),
                        source=ResolutionSource.DISCOVERY,
                        evidence_cid=evidence_cid,
                        rejection_reason=reason or "root_unresolvable",
                    )
                )
                reasons.append(reason or "root_unresolvable")
                continue
            key = str(resolved)
            if key in seen:
                continue
            seen.add(key)
            depth = -1
            if _path_is_within(cwd_resolved, resolved) or cwd_resolved == resolved:
                try:
                    depth = len(cwd_resolved.relative_to(resolved).parts)
                except ValueError:
                    depth = 0
            candidates.append(
                RepositoryRootCandidate(
                    root_path=str(resolved),
                    alias=_alias(alias if _ALIAS_RE.fullmatch(alias) else f"root{index}"),
                    source=ResolutionSource.DISCOVERY,
                    evidence_cid=evidence_cid,
                    depth_under_cwd=depth,
                )
            )
        return candidates, reasons

    def _select_candidate(
        self,
        evidence: RepositoryTargetEvidence,
        candidates: Sequence[RepositoryRootCandidate],
    ) -> tuple[
        RepositoryRootCandidate | None,
        list[str],
        ResolutionSource,
        bool,
    ]:
        reasons: list[str] = []
        viable = [item for item in candidates if item.viable]

        # Explicit repository hint (high-level override).  Must resolve to an
        # allowlisted root; never widen.
        hint = evidence.repository_hint
        if hint:
            if _reject_parent_traversal(hint):
                reasons.append("parent_traversal_rejected")
                return None, reasons, ResolutionSource.EXPLICIT_OVERRIDE, False
            hint_path = Path(hint)
            if not hint_path.is_absolute():
                hint_path = Path(evidence.cwd) / hint_path
            try:
                if _is_symlink_path(hint_path) and not evidence.follow_symlinks:
                    reasons.append("symlink_root_rejected")
                    return None, reasons, ResolutionSource.EXPLICIT_OVERRIDE, False
                if hint_path.exists():
                    hint_resolved = hint_path.resolve(strict=True)
                else:
                    reasons.append("repository_hint_missing")
                    return None, reasons, ResolutionSource.EXPLICIT_OVERRIDE, False
            except (OSError, RuntimeError):
                reasons.append("repository_hint_unresolvable")
                return None, reasons, ResolutionSource.EXPLICIT_OVERRIDE, False

            match: RepositoryRootCandidate | None = None
            for item in viable:
                root = Path(item.root_path)
                if hint_resolved == root or _path_is_within(hint_resolved, root):
                    # Prefer exact root match; if hint is nested under root,
                    # only accept when hint itself is that root's toplevel.
                    toplevel = _git_toplevel(hint_resolved)
                    if toplevel is not None and str(toplevel) == item.root_path:
                        match = item
                        break
                    if hint_resolved == root:
                        match = item
                        break
            if match is None:
                reasons.append("explicit_hint_outside_allowlist")
                return None, reasons, ResolutionSource.EXPLICIT_OVERRIDE, False
            reasons.append("explicit_repository_hint")
            return (
                match,
                reasons,
                ResolutionSource.EXPLICIT_OVERRIDE,
                True,
            )

        if not viable:
            reasons.append("no_viable_allowlisted_repository")
            return None, reasons, ResolutionSource.DISCOVERY, False

        under_cwd = [
            item for item in viable if item.depth_under_cwd >= 0
        ]
        if under_cwd:
            # Nearest enclosing root: maximum path depth under the candidate
            # means the candidate is deepest (most specific).
            # depth_under_cwd is parts from root to cwd, so smaller root that
            # still contains cwd has larger depth? Wait:
            # root=/a, cwd=/a/b/c → relative parts=2
            # root=/a/b, cwd=/a/b/c → relative parts=1
            # Nearest ancestor root is the one with *minimum* depth_under_cwd
            # among roots that contain cwd... No:
            # "nearest enclosing" means the deepest root that still encloses
            # cwd.  Deeper root has more path components, so relative depth
            # from that root to cwd is *smaller*.  Select min depth among
            # enclosing roots, and if ties, still unique if same path.
            min_depth = min(item.depth_under_cwd for item in under_cwd)
            nearest = [
                item for item in under_cwd if item.depth_under_cwd == min_depth
            ]
            # Prefer the longest root path among min-depth (should be unique).
            nearest.sort(key=lambda item: (-len(item.root_path), item.root_path))
            if len({item.root_path for item in nearest}) == 1:
                reasons.append("unique_nearest_ancestor")
                return nearest[0], reasons, ResolutionSource.DISCOVERY, False
            reasons.append("multiple_viable_repository_roots")
            reasons.append("nested_repository_ambiguity")
            return None, reasons, ResolutionSource.DISCOVERY, False

        if len(viable) == 1:
            reasons.append("unique_allowlisted_repository")
            return viable[0], reasons, ResolutionSource.DISCOVERY, False

        reasons.append("multiple_viable_repository_roots")
        return None, reasons, ResolutionSource.DISCOVERY, False

    def _bind_selected(
        self,
        evidence: RepositoryTargetEvidence,
        *,
        selected: RepositoryRootCandidate,
        selected_source: ResolutionSource,
        candidates: Sequence[RepositoryRootCandidate],
    ) -> tuple[RepositoryTargetBinding, list[str]]:
        reasons: list[str] = []
        root = Path(selected.root_path)
        ignore = IgnorePolicy(allow_dirty_overlay=evidence.allow_dirty_overlay)
        try:
            descriptor = build_repository_descriptor(
                root,
                alias=evidence.logical_name,
                authority=RepositoryAuthority(mode=evidence.authority_mode),
                ignore_policy=ignore,
                logical_name=evidence.logical_name,
                follow_symlinks=evidence.follow_symlinks,
            )
        except RepositoryForestError as exc:
            raise RepositoryTargetResolverError(
                f"descriptor_failed: {exc.reason_code}"
            ) from exc

        scope_path, scope_reasons = self._resolve_scope(
            evidence,
            descriptor=descriptor,
        )
        reasons.extend(scope_reasons)

        tree_id = cid_for_dag_json(
            {
                "schema": TREE_IDENTITY_SCHEMA,
                "head_commit": descriptor.commit,
                "head_tree": descriptor.tree,
                "dirty_overlay": descriptor.dirty_overlay_digest,
                "gitlink_closure": descriptor.portable_closure.gitlink_closure_cid,
            }
        )
        checkout_id = "checkout:" + cid_for_dag_json(
            {
                "schema": CHECKOUT_IDENTITY_SCHEMA,
                "resolved_root": str(descriptor.root_path),
                "local_binding": (
                    descriptor.local_locator.local_repository_binding_id
                ),
                "head_commit": descriptor.commit,
            }
        )

        nested_roots = sorted(
            {
                item.root_path
                for item in candidates
                if item.viable and item.root_path != selected.root_path
            }
        )
        nested_cid = cid_for_dag_json(
            {
                "schema": NESTED_POPULATION_SCHEMA,
                "roots": nested_roots,
            }
        )
        if descriptor.dirty:
            reasons.append("dirty_overlay_observed")
        if descriptor.portable_closure.gitlinks:
            reasons.append("submodule_gitlinks_bound")
        if nested_roots:
            reasons.append("nested_repository_population_recorded")

        binding = RepositoryTargetBinding(
            repository_root=str(descriptor.root_path),
            repository_id=descriptor.repository_id,
            checkout_id=checkout_id,
            scope_path=scope_path,
            tree_id=tree_id,
            dirty_overlay_cid=descriptor.dirty_overlay_digest,
            submodule_population_cid=(
                descriptor.portable_closure.gitlink_closure_cid
            ),
            nested_repository_population_cid=nested_cid,
            head_commit=descriptor.commit,
            head_tree=descriptor.tree,
            dirty=descriptor.dirty,
            descriptor_cid=descriptor.descriptor_cid,
            selected_source=selected_source,
            alias=evidence.logical_name,
        )
        return binding, reasons

    def _resolve_scope(
        self,
        evidence: RepositoryTargetEvidence,
        *,
        descriptor: RepositoryDescriptor,
    ) -> tuple[str, list[str]]:
        reasons: list[str] = []
        root = descriptor.root_path
        hint = evidence.scope_hint
        if hint:
            if _reject_parent_traversal(hint):
                raise RepositoryTargetResolverError(
                    "parent_traversal_rejected: scope_hint escapes repository root"
                )
            candidate = Path(hint)
            if not candidate.is_absolute():
                candidate = root / candidate
            try:
                if _is_symlink_path(candidate) and not evidence.follow_symlinks:
                    raise RepositoryTargetResolverError(
                        "symlink_scope_rejected: scope path is a symlink"
                    )
                resolved = path_within_repository(
                    descriptor,
                    candidate,
                    require_existing=False,
                )
            except RepositoryForestError as exc:
                raise RepositoryTargetResolverError(
                    f"scope_escape: {exc.reason_code}"
                ) from exc
            reasons.append("explicit_scope_hint")
            return str(resolved), reasons

        scope = _scope_relative_from_cwd(root, Path(evidence.cwd))
        if scope != str(root):
            reasons.append("scope_defaulted_to_cwd")
        else:
            reasons.append("scope_defaulted_to_repository_root")
        return scope, reasons

    def _decisions_for_binding(
        self,
        *,
        binding: RepositoryTargetBinding,
        evidence_cid: str,
        selected_source: ResolutionSource,
        override_accepted: bool,
        candidates: Sequence[RepositoryRootCandidate],
        extra_reasons: Sequence[str],
    ) -> tuple[TargetInferenceDecision, ...]:
        values = {
            "repository_root": binding.repository_root,
            "repository_id": binding.repository_id,
            "checkout_id": binding.checkout_id,
            "scope": binding.scope_path,
            "tree_id": binding.tree_id,
            "dirty_overlay": binding.dirty_overlay_cid,
            "submodules": binding.submodule_population_cid,
            "nested_repositories": binding.nested_repository_population_cid,
        }
        field_reasons: dict[str, tuple[str, ...]] = {
            "repository_root": tuple(
                code
                for code in extra_reasons
                if code
                in {
                    "unique_nearest_ancestor",
                    "unique_allowlisted_repository",
                    "explicit_repository_hint",
                    "submodule_gitlinks_bound",
                }
            )
            or ("repository_root_selected",),
            "repository_id": ("portable_repository_identity",),
            "checkout_id": ("checkout_specific_binding",),
            "scope": tuple(
                code
                for code in extra_reasons
                if code.startswith("scope_")
            )
            or ("scope_bound",),
            "tree_id": (
                ("dirty_overlay_observed",)
                if binding.dirty
                else ("clean_head_tree",)
            ),
            "dirty_overlay": (
                ("dirty_overlay_observed",)
                if binding.dirty
                else ("empty_dirty_overlay",)
            ),
            "submodules": (
                ("submodule_gitlinks_bound",)
                if binding.submodule_population_cid
                else ("no_submodules",)
            ),
            "nested_repositories": (
                ("nested_repository_population_recorded",)
                if "nested_repository_population_recorded" in extra_reasons
                else ("no_nested_repositories",)
            ),
        }

        decisions: list[TargetInferenceDecision] = []
        for field_name, value in values.items():
            field_candidates: list[TargetCandidate] = [
                _candidate(
                    field_name=field_name,
                    value=value,
                    source=selected_source,
                    evidence_cid=evidence_cid,
                )
            ]
            if field_name == "repository_root":
                for item in candidates:
                    if not item.viable or item.root_path == value:
                        continue
                    field_candidates.append(
                        _candidate(
                            field_name=field_name,
                            value=item.root_path,
                            source=item.source,
                            evidence_cid=item.evidence_cid,
                            confidence_ppm=250_000,
                            rejection_reason="not_nearest_or_not_selected",
                        )
                    )
            disposition = (
                ResolutionDisposition.DEFAULTED
                if selected_source is ResolutionSource.BUILTIN_DEFAULT
                else ResolutionDisposition.UNIQUE
            )
            if field_name == "scope" and "scope_defaulted_to_repository_root" in (
                field_reasons[field_name]
            ):
                disposition = ResolutionDisposition.DEFAULTED
                # Defaulted scope still uses discovery/default source, not
                # builtin_default unless no cwd context existed.
                scope_source = (
                    ResolutionSource.BUILTIN_DEFAULT
                    if selected_source is not ResolutionSource.EXPLICIT_OVERRIDE
                    else selected_source
                )
                if scope_source is ResolutionSource.BUILTIN_DEFAULT:
                    field_candidates = [
                        _candidate(
                            field_name=field_name,
                            value=value,
                            source=scope_source,
                            evidence_cid=evidence_cid,
                        )
                    ]
                    decisions.append(
                        _decision(
                            field_name=field_name,
                            disposition=disposition,
                            selected_value=value,
                            selected_source=scope_source,
                            evidence_cid=evidence_cid,
                            candidates=field_candidates,
                            reason_codes=field_reasons[field_name],
                            effect=DecisionEffect.IDENTITY_ONLY,
                            override_accepted=False,
                        )
                    )
                    continue
            decisions.append(
                _decision(
                    field_name=field_name,
                    disposition=(
                        ResolutionDisposition.UNIQUE
                        if field_name != "scope"
                        or "scope_defaulted_to_repository_root"
                        not in field_reasons[field_name]
                        else disposition
                    ),
                    selected_value=value,
                    selected_source=selected_source,
                    evidence_cid=evidence_cid,
                    candidates=field_candidates,
                    reason_codes=field_reasons[field_name],
                    effect=DecisionEffect.IDENTITY_ONLY,
                    override_accepted=(
                        override_accepted and field_name == "repository_root"
                    ),
                )
            )
        return tuple(decisions)


def resolve_repository_target(
    evidence: RepositoryTargetEvidence,
) -> RepositoryTargetResolution:
    """Module-level convenience wrapper around :class:`RepositoryTargetResolver`."""

    return RepositoryTargetResolver().resolve(evidence)


def empty_overlay_cid() -> str:
    """Return the canonical empty dirty-overlay digest."""

    return empty_dirty_overlay_digest()


__all__ = [
    "CHECKOUT_IDENTITY_SCHEMA",
    "NESTED_POPULATION_SCHEMA",
    "REPOSITORY_BINDING_SCHEMA",
    "REPOSITORY_FIELD_NAMES",
    "REPOSITORY_TARGET_EVIDENCE_SCHEMA",
    "REPOSITORY_TARGET_RESOLUTION_REQUIREMENT_ID",
    "REPOSITORY_TARGET_RESOLUTION_SCHEMA",
    "TREE_IDENTITY_SCHEMA",
    "RepositoryRootCandidate",
    "RepositoryTargetBinding",
    "RepositoryTargetEvidence",
    "RepositoryTargetResolution",
    "RepositoryTargetResolver",
    "RepositoryTargetResolverError",
    "empty_overlay_cid",
    "resolve_repository_target",
]
