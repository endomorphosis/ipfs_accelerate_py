"""Frozen four-repository forest manifest loader and replay validator.

A reviewed manifest names SwissKnife, the accelerator checkout, and the kit and
datasets roots with authority and policy expectations.  Live materialization
always derives fresh repository descriptors from Git; recorded commits in the
manifest are observational only and never trusted as identity.

Portable and local projections are persisted as separate documents so host
locators and credentials never fold into portable forest CIDs.  Replay
validation re-checks expected roots, sole-write authority, forest policy, and
analyzer profile without logging secrets or environment material.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .proof.formal_verification_contracts import content_identity
from .repository_forest import (
    ANALYZER_PROFILE_SCHEMA,
    AUTHORITY_SCHEMA,
    AnalyzerProfile,
    AuthorityMode,
    DEFAULT_ACCELERATOR_ALIAS,
    DEFAULT_DATASETS_ALIAS,
    DEFAULT_KIT_ALIAS,
    DEFAULT_SWISSKNIFE_ALIAS,
    DEFAULT_SWISSKNIFE_ROOT,
    FOREST_POLICY_SCHEMA,
    ForestPolicy,
    ForestRootSpec,
    IgnorePolicy,
    CaseUnicodePolicy,
    REPOSITORY_FOREST_SCHEMA,
    RepositoryAuthority,
    RepositoryForest,
    RepositoryForestError,
    build_repository_forest,
    forests_share_portable_identity,
    initial_vfs_assurance_forest_policy,
)


logger = logging.getLogger(__name__)

REPOSITORY_FOREST_MANIFEST_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.repository-forest-manifest@1"
)
MANIFEST_PORTABLE_PROJECTION_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.repository-forest-manifest-portable@1"
)
MANIFEST_LOCAL_PROJECTION_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.repository-forest-manifest-local@1"
)
MANIFEST_REPLAY_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.repository-forest-manifest-replay@1"
)

# Reviewed launch observations from the assurance plan.  These are historical
# notes only; materialization always re-derives commit/tree from live Git.
REVIEWED_OBSERVED_COMMITS: Mapping[str, str] = {
    DEFAULT_SWISSKNIFE_ALIAS: "df11f08fae17d35153e420fdcdc5b38d9f6b9a7f",
    DEFAULT_ACCELERATOR_ALIAS: "ff401f83b7e722e58af1696243b3aff9679a7002",
    DEFAULT_KIT_ALIAS: "f6a574375febbcf9a46fcd24bbc7bc5cfb551de5",
    DEFAULT_DATASETS_ALIAS: "6672d69242731f53b49f4f793ed3023b7ba36a0d",
}

FROZEN_FOUR_ROOT_ALIASES: tuple[str, ...] = (
    DEFAULT_SWISSKNIFE_ALIAS,
    DEFAULT_ACCELERATOR_ALIAS,
    DEFAULT_KIT_ALIAS,
    DEFAULT_DATASETS_ALIAS,
)

PORTABLE_PROJECTION_FILENAME = "forest_manifest.portable.json"
LOCAL_PROJECTION_FILENAME = "forest_manifest.local.json"

_ALIAS_RE = re.compile(r"^[A-Za-z][A-Za-z0-9._-]{0,127}\Z")
_GIT_OBJECT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_SECRET_FIELD_MARKERS = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "authorization",
        "cookie",
        "credential",
        "credentials",
        "password",
        "private_key",
        "refresh_token",
        "secret",
        "session_token",
        "token",
    }
)
_SECRET_VALUE_MARKERS = (
    "password=",
    "secret=",
    "api_key=",
    "apikey=",
    "token=",
    "authorization=",
    "bearer ",
)


class RepositoryForestManifestError(RepositoryForestError):
    """Fail-closed rejection for forest manifest load, persist, or replay."""


def _text(value: Any, *, field_name: str, required: bool = True) -> str:
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value.strip()
    else:
        raise RepositoryForestManifestError(
            "invalid_field_type",
            f"{field_name} must be a string",
        )
    if required and not text:
        raise RepositoryForestManifestError(
            "missing_required_field",
            f"{field_name} is required",
        )
    return text


def _normalize_alias(value: Any) -> str:
    text = _text(value, field_name="alias")
    if not _ALIAS_RE.fullmatch(text):
        raise RepositoryForestManifestError(
            "invalid_alias",
            "alias must be a short alphanumeric identifier",
        )
    return text


def _optional_git_object(value: Any, *, field_name: str) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    if not _GIT_OBJECT_RE.fullmatch(text):
        raise RepositoryForestManifestError(
            "invalid_git_object",
            f"{field_name} must be a full Git object identity when provided",
        )
    return text


def _reject_secret_keys(payload: Mapping[str, Any], *, path: str = "") -> None:
    for key, value in payload.items():
        key_text = str(key)
        lowered = key_text.lower().replace("-", "_")
        if lowered in _SECRET_FIELD_MARKERS or any(
            marker in lowered for marker in _SECRET_FIELD_MARKERS
        ):
            raise RepositoryForestManifestError(
                "secret_material_rejected",
                "manifest must not carry credential fields",
            )
        if isinstance(value, Mapping):
            next_path = f"{path}.{key_text}" if path else key_text
            _reject_secret_keys(value, path=next_path)
        elif isinstance(value, str):
            sample = value.lower()
            if any(marker in sample for marker in _SECRET_VALUE_MARKERS):
                raise RepositoryForestManifestError(
                    "secret_material_rejected",
                    "manifest must not carry credential-like values",
                )
        elif isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            for item in value:
                if isinstance(item, Mapping):
                    _reject_secret_keys(item, path=path)


def _safe_log(message: str, *args: Any) -> None:
    """Log only non-sensitive operational messages (never env or credentials)."""

    # Intentionally never interpolates os.environ or credential-bearing payloads.
    try:
        logger.debug(message, *args)
    except Exception:
        return


@dataclass(frozen=True)
class ReviewedManifestRoot:
    """One reviewed root entry in the frozen four-repository manifest."""

    alias: str
    authority_mode: str = AuthorityMode.READ_ONLY.value
    required: bool = True
    logical_name: str = ""
    remote_url: str = ""
    # Observational only — never used as the live descriptor commit.
    reviewed_commit: str = ""
    write_path_allowlist: tuple[str, ...] = ()
    ignore_policy: IgnorePolicy | None = None
    case_unicode_policy: CaseUnicodePolicy | None = None

    def __post_init__(self) -> None:
        alias = _normalize_alias(self.alias)
        object.__setattr__(self, "alias", alias)
        logical = _normalize_alias(self.logical_name or alias)
        object.__setattr__(self, "logical_name", logical)
        mode = _text(self.authority_mode, field_name="authority_mode")
        if mode not in {item.value for item in AuthorityMode}:
            raise RepositoryForestManifestError("unsupported_authority_mode")
        object.__setattr__(self, "authority_mode", mode)
        object.__setattr__(self, "required", bool(self.required))
        object.__setattr__(
            self,
            "reviewed_commit",
            _optional_git_object(
                self.reviewed_commit,
                field_name="reviewed_commit",
            ),
        )
        remote = str(self.remote_url or "").strip()
        # Strip URL credentials without logging them.
        if "://" in remote and "@" in remote.split("://", 1)[1]:
            scheme, remainder = remote.split("://", 1)
            remainder = remainder.rsplit("@", 1)[-1]
            remote = f"{scheme}://{remainder}".rstrip("/")
        object.__setattr__(self, "remote_url", remote)
        allowlist = tuple(
            sorted(
                {
                    str(item).strip().replace("\\", "/")
                    for item in (self.write_path_allowlist or ())
                    if str(item).strip()
                }
            )
        )
        if mode == AuthorityMode.READ_ONLY.value and allowlist:
            raise RepositoryForestManifestError(
                "read_only_write_allowlist",
                "read-only reviewed roots cannot carry write allowlists",
            )
        object.__setattr__(self, "write_path_allowlist", allowlist)
        ignore = self.ignore_policy or IgnorePolicy()
        if isinstance(ignore, Mapping):
            ignore = IgnorePolicy.from_dict(ignore)
        elif not isinstance(ignore, IgnorePolicy):
            raise RepositoryForestManifestError("invalid_ignore_policy")
        object.__setattr__(self, "ignore_policy", ignore)
        case_policy = self.case_unicode_policy or CaseUnicodePolicy()
        if isinstance(case_policy, Mapping):
            case_policy = CaseUnicodePolicy.from_dict(case_policy)
        elif not isinstance(case_policy, CaseUnicodePolicy):
            raise RepositoryForestManifestError("invalid_case_unicode_policy")
        object.__setattr__(self, "case_unicode_policy", case_policy)

    def authority(self) -> RepositoryAuthority:
        return RepositoryAuthority(
            mode=self.authority_mode,
            write_path_allowlist=self.write_path_allowlist,
        )

    def to_portable_dict(self) -> dict[str, Any]:
        assert isinstance(self.ignore_policy, IgnorePolicy)
        assert isinstance(self.case_unicode_policy, CaseUnicodePolicy)
        return {
            "alias": self.alias,
            "logical_name": self.logical_name,
            "remote_url": self.remote_url,
            "required": self.required,
            "authority_mode": self.authority_mode,
            "write_path_allowlist": list(self.write_path_allowlist),
            "reviewed_commit": self.reviewed_commit,
            "ignore_policy": self.ignore_policy.to_portable_dict(),
            "case_unicode_policy": self.case_unicode_policy.to_portable_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReviewedManifestRoot":
        if not isinstance(payload, Mapping):
            raise RepositoryForestManifestError("invalid_manifest_root")
        authority = payload.get("authority") or {}
        if isinstance(authority, Mapping) and "mode" in authority:
            mode = str(authority.get("mode") or AuthorityMode.READ_ONLY.value)
            allowlist = tuple(authority.get("write_path_allowlist") or ())
        else:
            mode = str(
                payload.get("authority_mode") or AuthorityMode.READ_ONLY.value
            )
            allowlist = tuple(payload.get("write_path_allowlist") or ())
        return cls(
            alias=str(payload.get("alias") or ""),
            logical_name=str(payload.get("logical_name") or ""),
            remote_url=str(payload.get("remote_url") or ""),
            required=bool(payload.get("required", True)),
            authority_mode=mode,
            write_path_allowlist=allowlist,
            reviewed_commit=str(payload.get("reviewed_commit") or ""),
            ignore_policy=IgnorePolicy.from_dict(
                payload.get("ignore_policy") or {}
            ),
            case_unicode_policy=CaseUnicodePolicy.from_dict(
                payload.get("case_unicode_policy") or {}
            ),
        )


@dataclass(frozen=True)
class ReviewedForestManifest:
    """Reviewed, host-independent four-repository forest manifest."""

    schema: str = REPOSITORY_FOREST_MANIFEST_SCHEMA
    roots: tuple[ReviewedManifestRoot, ...] = ()
    sole_write_alias: str = DEFAULT_ACCELERATOR_ALIAS
    analyzer_profile: AnalyzerProfile | None = None
    manifest_label: str = "vfs-assurance-four-repository-v1"
    # Local-only path hints; never included in portable identity.
    local_root_paths: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        schema = _text(self.schema, field_name="manifest.schema")
        if schema != REPOSITORY_FOREST_MANIFEST_SCHEMA:
            raise RepositoryForestManifestError(
                "unsupported_manifest_schema",
                f"unsupported schema: {schema}",
            )
        object.__setattr__(self, "schema", schema)
        write_alias = _normalize_alias(self.sole_write_alias)
        object.__setattr__(self, "sole_write_alias", write_alias)
        label = _text(self.manifest_label, field_name="manifest_label")
        object.__setattr__(self, "manifest_label", label)
        profile = self.analyzer_profile
        if profile is None:
            profile = AnalyzerProfile(profile_name="vfs-assurance-default")
        elif isinstance(profile, Mapping):
            profile = AnalyzerProfile.from_dict(profile)
        elif not isinstance(profile, AnalyzerProfile):
            raise RepositoryForestManifestError("invalid_analyzer_profile")
        object.__setattr__(self, "analyzer_profile", profile)

        normalized: list[ReviewedManifestRoot] = []
        aliases: set[str] = set()
        for raw in self.roots:
            root = (
                raw
                if isinstance(raw, ReviewedManifestRoot)
                else ReviewedManifestRoot.from_dict(raw)
            )
            if root.alias in aliases:
                raise RepositoryForestManifestError(
                    "duplicate_alias",
                    f"duplicate manifest alias: {root.alias}",
                )
            aliases.add(root.alias)
            normalized.append(root)
        normalized = sorted(normalized, key=lambda item: item.alias)
        if not normalized:
            raise RepositoryForestManifestError(
                "empty_manifest",
                "manifest must declare at least one root",
            )
        if write_alias not in aliases:
            raise RepositoryForestManifestError(
                "missing_write_root",
                "manifest must include the sole write alias",
            )
        writable = [
            item
            for item in normalized
            if item.authority_mode == AuthorityMode.READ_WRITE.value
        ]
        if len(writable) != 1 or writable[0].alias != write_alias:
            raise RepositoryForestManifestError(
                "write_root_cardinality",
                "exactly one root must match sole_write_alias with read/write authority",
            )
        object.__setattr__(self, "roots", tuple(normalized))

        paths: dict[str, str] = {}
        for key, value in dict(self.local_root_paths or {}).items():
            alias = _normalize_alias(key)
            if alias not in aliases:
                raise RepositoryForestManifestError(
                    "unknown_local_root_alias",
                    f"local root path for unknown alias: {alias}",
                )
            path_text = str(value or "").strip()
            if not path_text:
                continue
            paths[alias] = path_text
        object.__setattr__(self, "local_root_paths", paths)

    @property
    def manifest_cid(self) -> str:
        """Portable manifest identity (host paths excluded)."""

        return content_identity(self.to_portable_dict())

    def root_for_alias(self, alias: str) -> ReviewedManifestRoot:
        key = _normalize_alias(alias)
        for item in self.roots:
            if item.alias == key:
                return item
        raise RepositoryForestManifestError(
            "unknown_alias",
            f"no reviewed root for alias {key}",
        )

    def expected_aliases(self) -> tuple[str, ...]:
        return tuple(item.alias for item in self.roots)

    def to_portable_dict(self) -> dict[str, Any]:
        assert isinstance(self.analyzer_profile, AnalyzerProfile)
        return {
            "schema": self.schema,
            "manifest_label": self.manifest_label,
            "sole_write_alias": self.sole_write_alias,
            "analyzer_profile": self.analyzer_profile.to_portable_dict(),
            "analyzer_profile_cid": self.analyzer_profile.profile_cid,
            "roots": [item.to_portable_dict() for item in self.roots],
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.to_portable_dict()
        payload["manifest_cid"] = self.manifest_cid
        payload["local_root_paths"] = dict(self.local_root_paths or {})
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReviewedForestManifest":
        if not isinstance(payload, Mapping):
            raise RepositoryForestManifestError("invalid_manifest")
        _reject_secret_keys(payload)
        raw_roots = payload.get("roots") or ()
        if not isinstance(raw_roots, Sequence) or isinstance(
            raw_roots, (str, bytes, bytearray)
        ):
            raise RepositoryForestManifestError("invalid_manifest_roots")
        return cls(
            schema=str(
                payload.get("schema") or REPOSITORY_FOREST_MANIFEST_SCHEMA
            ),
            roots=tuple(
                ReviewedManifestRoot.from_dict(item) for item in raw_roots
            ),
            sole_write_alias=str(
                payload.get("sole_write_alias") or DEFAULT_ACCELERATOR_ALIAS
            ),
            analyzer_profile=AnalyzerProfile.from_dict(
                payload.get("analyzer_profile") or {
                    "profile_name": "vfs-assurance-default",
                }
            ),
            manifest_label=str(
                payload.get("manifest_label")
                or "vfs-assurance-four-repository-v1"
            ),
            local_root_paths=dict(payload.get("local_root_paths") or {}),
        )


def default_reviewed_four_repository_manifest(
    *,
    swissknife_root: str | Path = DEFAULT_SWISSKNIFE_ROOT,
    accelerator_root: str | Path | None = None,
    kit_root: str | Path | None = None,
    datasets_root: str | Path | None = None,
    analyzer_profile: AnalyzerProfile | Mapping[str, Any] | None = None,
    require_all_four: bool = True,
) -> ReviewedForestManifest:
    """Return the frozen four-repository reviewed manifest with local path hints.

    Reviewed commits are recorded as observations only.  Callers that materialize
    the forest always derive fresh descriptors from live checkouts.
    """

    accel = (
        Path(accelerator_root)
        if accelerator_root is not None
        else Path.cwd()
    )
    kit = Path(kit_root) if kit_root is not None else accel / DEFAULT_KIT_ALIAS
    datasets = (
        Path(datasets_root)
        if datasets_root is not None
        else accel / DEFAULT_DATASETS_ALIAS
    )
    roots = (
        ReviewedManifestRoot(
            alias=DEFAULT_SWISSKNIFE_ALIAS,
            authority_mode=AuthorityMode.READ_ONLY.value,
            required=True,
            reviewed_commit=REVIEWED_OBSERVED_COMMITS[DEFAULT_SWISSKNIFE_ALIAS],
        ),
        ReviewedManifestRoot(
            alias=DEFAULT_ACCELERATOR_ALIAS,
            authority_mode=AuthorityMode.READ_WRITE.value,
            required=True,
            reviewed_commit=REVIEWED_OBSERVED_COMMITS[
                DEFAULT_ACCELERATOR_ALIAS
            ],
        ),
        ReviewedManifestRoot(
            alias=DEFAULT_KIT_ALIAS,
            authority_mode=AuthorityMode.READ_ONLY.value,
            required=require_all_four,
            reviewed_commit=REVIEWED_OBSERVED_COMMITS[DEFAULT_KIT_ALIAS],
        ),
        ReviewedManifestRoot(
            alias=DEFAULT_DATASETS_ALIAS,
            authority_mode=AuthorityMode.READ_ONLY.value,
            required=require_all_four,
            reviewed_commit=REVIEWED_OBSERVED_COMMITS[DEFAULT_DATASETS_ALIAS],
        ),
    )
    if analyzer_profile is None:
        profile: AnalyzerProfile | Mapping[str, Any] | None = AnalyzerProfile(
            profile_name="vfs-assurance-default",
        )
    else:
        profile = analyzer_profile
    return ReviewedForestManifest(
        roots=roots,
        sole_write_alias=DEFAULT_ACCELERATOR_ALIAS,
        analyzer_profile=(
            AnalyzerProfile.from_dict(profile)
            if isinstance(profile, Mapping)
            else profile
        ),
        local_root_paths={
            DEFAULT_SWISSKNIFE_ALIAS: str(swissknife_root),
            DEFAULT_ACCELERATOR_ALIAS: str(accel),
            DEFAULT_KIT_ALIAS: str(kit),
            DEFAULT_DATASETS_ALIAS: str(datasets),
        },
    )


def load_reviewed_manifest(
    source: str | Path | Mapping[str, Any],
) -> ReviewedForestManifest:
    """Load a reviewed manifest from a mapping or JSON file path."""

    if isinstance(source, Mapping):
        return ReviewedForestManifest.from_dict(source)
    path = Path(source)
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RepositoryForestManifestError(
            "manifest_unreadable",
            "reviewed manifest file could not be read",
        ) from exc
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise RepositoryForestManifestError(
            "manifest_invalid_json",
            "reviewed manifest is not valid JSON",
        ) from exc
    if not isinstance(payload, Mapping):
        raise RepositoryForestManifestError(
            "invalid_manifest",
            "reviewed manifest root must be a JSON object",
        )
    return ReviewedForestManifest.from_dict(payload)


def forest_policy_from_manifest(
    manifest: ReviewedForestManifest,
    *,
    root_paths: Mapping[str, str | Path] | None = None,
) -> ForestPolicy:
    """Build a live forest policy from a reviewed manifest and host path map."""

    path_map: dict[str, str | Path] = dict(manifest.local_root_paths or {})
    if root_paths:
        for key, value in root_paths.items():
            path_map[_normalize_alias(key)] = value
    roots: list[ForestRootSpec] = []
    for reviewed in manifest.roots:
        if reviewed.alias not in path_map:
            if reviewed.required:
                raise RepositoryForestManifestError(
                    "missing_root_path",
                    f"no host path for required alias {reviewed.alias}",
                )
            continue
        roots.append(
            ForestRootSpec(
                alias=reviewed.alias,
                root_path=path_map[reviewed.alias],
                authority=reviewed.authority(),
                ignore_policy=reviewed.ignore_policy,
                case_unicode_policy=reviewed.case_unicode_policy,
                logical_name=reviewed.logical_name,
                remote_url=reviewed.remote_url,
                required=reviewed.required,
            )
        )
    assert isinstance(manifest.analyzer_profile, AnalyzerProfile)
    return ForestPolicy(
        roots=tuple(roots),
        sole_write_alias=manifest.sole_write_alias,
        analyzer_profile=manifest.analyzer_profile,
    )


@dataclass(frozen=True)
class ForestManifestMaterialization:
    """Fresh forest derived from a reviewed manifest plus dual projections."""

    schema: str = REPOSITORY_FOREST_MANIFEST_SCHEMA + "/materialization"
    manifest: ReviewedForestManifest = None  # type: ignore[assignment]
    forest: RepositoryForest = None  # type: ignore[assignment]
    observed_commit_mismatches: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, ReviewedForestManifest):
            raise RepositoryForestManifestError("missing_manifest")
        if not isinstance(self.forest, RepositoryForest):
            raise RepositoryForestManifestError("missing_forest")
        mismatches = tuple(
            dict.fromkeys(str(item) for item in self.observed_commit_mismatches)
        )
        reasons = tuple(dict.fromkeys(str(item) for item in self.reason_codes))
        object.__setattr__(self, "observed_commit_mismatches", mismatches)
        object.__setattr__(self, "reason_codes", reasons)

    @property
    def forest_id(self) -> str:
        return self.forest.forest_id

    @property
    def manifest_cid(self) -> str:
        return self.manifest.manifest_cid

    def to_portable_projection(self) -> dict[str, Any]:
        """Portable projection: no host paths, credentials, or local locators."""

        return {
            "schema": MANIFEST_PORTABLE_PROJECTION_SCHEMA,
            "manifest_cid": self.manifest.manifest_cid,
            "manifest": self.manifest.to_portable_dict(),
            "forest": self.forest.to_portable_dict(),
            "forest_id": self.forest.forest_id,
            "observed_commit_mismatches": list(self.observed_commit_mismatches),
            "reason_codes": list(self.reason_codes),
        }

    def to_local_projection(self) -> dict[str, Any]:
        """Local projection: includes host locators, excludes credentials."""

        portable = self.to_portable_projection()
        local_roots = {
            item.alias: {
                "root_path": item.local_locator.root_path,
                "resolved_root_path": item.local_locator.resolved_root_path,
                "local_repository_binding_id": (
                    item.local_locator.local_repository_binding_id
                ),
            }
            for item in self.forest.descriptors
        }
        return {
            "schema": MANIFEST_LOCAL_PROJECTION_SCHEMA,
            "manifest_cid": self.manifest.manifest_cid,
            "manifest": self.manifest.to_dict(),
            "forest": self.forest.to_dict(),
            "forest_id": self.forest.forest_id,
            "local_roots": local_roots,
            "portable_forest_id": portable["forest_id"],
            "observed_commit_mismatches": list(self.observed_commit_mismatches),
            "reason_codes": list(self.reason_codes),
        }


def materialize_forest_from_manifest(
    manifest: ReviewedForestManifest | Mapping[str, Any] | str | Path,
    *,
    root_paths: Mapping[str, str | Path] | None = None,
    follow_symlinks: bool = True,
    fail_on_missing_required: bool = True,
) -> ForestManifestMaterialization:
    """Load policy from a reviewed manifest and derive fresh descriptors.

    Recorded ``reviewed_commit`` values are compared for diagnostics only; the
    live Git HEAD/tree always author the resulting forest identity.
    """

    if not isinstance(manifest, ReviewedForestManifest):
        reviewed = load_reviewed_manifest(manifest)
    else:
        reviewed = manifest

    policy = forest_policy_from_manifest(reviewed, root_paths=root_paths)
    _safe_log(
        "materializing forest from reviewed manifest label=%s roots=%s",
        reviewed.manifest_label,
        len(policy.roots),
    )
    try:
        forest = build_repository_forest(
            policy,
            follow_symlinks=follow_symlinks,
            fail_on_missing_required=fail_on_missing_required,
        )
    except RepositoryForestError as exc:
        raise RepositoryForestManifestError(
            exc.reason_code,
            str(exc) or exc.reason_code,
        ) from exc

    mismatches: list[str] = []
    reasons: list[str] = list(forest.reason_codes)
    for descriptor in forest.descriptors:
        try:
            reviewed_root = reviewed.root_for_alias(descriptor.alias)
        except RepositoryForestManifestError:
            reasons.append(f"{descriptor.alias}:unexpected_alias")
            continue
        # Fresh derivation is authoritative; reviewed commits are observations.
        if (
            reviewed_root.reviewed_commit
            and descriptor.commit != reviewed_root.reviewed_commit
        ):
            mismatches.append(descriptor.alias)
            reasons.append(f"{descriptor.alias}:reviewed_commit_drift")
        if (
            reviewed_root.authority_mode
            and descriptor.authority.mode != reviewed_root.authority_mode
        ):
            raise RepositoryForestManifestError(
                "authority_mismatch",
                f"live authority for {descriptor.alias} does not match reviewed manifest",
            )
    return ForestManifestMaterialization(
        manifest=reviewed,
        forest=forest,
        observed_commit_mismatches=tuple(mismatches),
        reason_codes=tuple(dict.fromkeys(reasons)),
    )


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True)
    # Scrub accidental env expansion markers; never write os.environ values.
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def persist_manifest_projections(
    materialization: ForestManifestMaterialization,
    directory: str | Path,
    *,
    portable_filename: str = PORTABLE_PROJECTION_FILENAME,
    local_filename: str = LOCAL_PROJECTION_FILENAME,
) -> tuple[Path, Path]:
    """Persist portable and local projections as separate JSON documents."""

    target = Path(directory)
    portable_path = target / portable_filename
    local_path = target / local_filename
    portable = materialization.to_portable_projection()
    local = materialization.to_local_projection()
    # Guard rails: portable projection must not embed host locators.
    portable_text = json.dumps(portable, sort_keys=True)
    if "local_locator" in portable_text or "resolved_root_path" in portable_text:
        raise RepositoryForestManifestError(
            "portable_projection_host_leak",
            "portable projection must not include host locators",
        )
    for descriptor in materialization.forest.descriptors:
        host = descriptor.local_locator.resolved_root_path
        if host and not host.startswith("portable://") and host in portable_text:
            raise RepositoryForestManifestError(
                "portable_projection_host_leak",
                "portable projection must not include host paths",
            )
    _reject_secret_keys(portable)
    _reject_secret_keys(local)
    _atomic_write_json(portable_path, portable)
    _atomic_write_json(local_path, local)
    _safe_log(
        "persisted forest manifest projections portable=%s local=%s",
        portable_filename,
        local_filename,
    )
    return portable_path, local_path


def load_portable_projection(
    source: str | Path | Mapping[str, Any],
) -> dict[str, Any]:
    """Load a portable projection mapping from disk or memory."""

    if isinstance(source, Mapping):
        payload = dict(source)
    else:
        path = Path(source)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise RepositoryForestManifestError(
                "portable_projection_unreadable",
                "portable projection could not be read",
            ) from exc
        except json.JSONDecodeError as exc:
            raise RepositoryForestManifestError(
                "portable_projection_invalid_json",
                "portable projection is not valid JSON",
            ) from exc
    if not isinstance(payload, Mapping):
        raise RepositoryForestManifestError("invalid_portable_projection")
    _reject_secret_keys(payload)
    schema = str(payload.get("schema") or "")
    if schema and schema != MANIFEST_PORTABLE_PROJECTION_SCHEMA:
        raise RepositoryForestManifestError(
            "unsupported_portable_projection_schema"
        )
    return dict(payload)


def load_local_projection(
    source: str | Path | Mapping[str, Any],
) -> dict[str, Any]:
    """Load a local projection mapping from disk or memory."""

    if isinstance(source, Mapping):
        payload = dict(source)
    else:
        path = Path(source)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise RepositoryForestManifestError(
                "local_projection_unreadable",
                "local projection could not be read",
            ) from exc
        except json.JSONDecodeError as exc:
            raise RepositoryForestManifestError(
                "local_projection_invalid_json",
                "local projection is not valid JSON",
            ) from exc
    if not isinstance(payload, Mapping):
        raise RepositoryForestManifestError("invalid_local_projection")
    _reject_secret_keys(payload)
    schema = str(payload.get("schema") or "")
    if schema and schema != MANIFEST_LOCAL_PROJECTION_SCHEMA:
        raise RepositoryForestManifestError(
            "unsupported_local_projection_schema"
        )
    return dict(payload)


@dataclass(frozen=True)
class ManifestReplayValidation:
    """Result of validating a portable projection against reviewed expectations."""

    schema: str = MANIFEST_REPLAY_RECEIPT_SCHEMA
    valid: bool = False
    forest_id: str = ""
    manifest_cid: str = ""
    expected_aliases: tuple[str, ...] = ()
    observed_aliases: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "valid": self.valid,
            "forest_id": self.forest_id,
            "manifest_cid": self.manifest_cid,
            "expected_aliases": list(self.expected_aliases),
            "observed_aliases": list(self.observed_aliases),
            "reason_codes": list(self.reason_codes),
        }


def validate_manifest_replay(
    portable_projection: Mapping[str, Any] | str | Path,
    *,
    expected_manifest: ReviewedForestManifest | Mapping[str, Any] | None = None,
    require_aliases: Sequence[str] | None = None,
    require_sole_write_alias: str = DEFAULT_ACCELERATOR_ALIAS,
) -> ManifestReplayValidation:
    """Validate a portable projection's roots, authority, and policy on replay.

    Recomputes forest identity from the portable descriptors and checks that
    expected aliases, sole-write authority, policy CID, and analyzer profile
    still match the reviewed manifest expectations.
    """

    projection = load_portable_projection(portable_projection)
    reasons: list[str] = []

    if expected_manifest is None:
        raw_manifest = projection.get("manifest") or {}
        if not isinstance(raw_manifest, Mapping):
            raise RepositoryForestManifestError("missing_replay_manifest")
        reviewed = ReviewedForestManifest.from_dict(raw_manifest)
    elif isinstance(expected_manifest, ReviewedForestManifest):
        reviewed = expected_manifest
    else:
        reviewed = ReviewedForestManifest.from_dict(expected_manifest)

    embedded_manifest: ReviewedForestManifest | None = None
    embedded_raw = projection.get("manifest")
    if isinstance(embedded_raw, Mapping):
        try:
            embedded_manifest = ReviewedForestManifest.from_dict(embedded_raw)
        except RepositoryForestError as exc:
            reasons.append(f"embedded_manifest_invalid:{exc.reason_code}")

    claimed_manifest_cid = str(projection.get("manifest_cid") or "").strip()
    if embedded_manifest is not None:
        if (
            claimed_manifest_cid
            and claimed_manifest_cid != embedded_manifest.manifest_cid
        ):
            reasons.append("manifest_cid_mismatch")
        if expected_manifest is not None and (
            embedded_manifest.manifest_cid != reviewed.manifest_cid
        ):
            reasons.append("expected_manifest_mismatch")
        # When no explicit expected_manifest is supplied, trust the embedded
        # reviewed document as the expectation surface.
        if expected_manifest is None:
            reviewed = embedded_manifest
    elif claimed_manifest_cid and claimed_manifest_cid != reviewed.manifest_cid:
        reasons.append("manifest_cid_mismatch")

    forest_payload = projection.get("forest")
    if not isinstance(forest_payload, Mapping):
        raise RepositoryForestManifestError(
            "missing_portable_forest",
            "portable projection must include a forest object",
        )
    try:
        forest = RepositoryForest.from_portable_dict(forest_payload)
    except RepositoryForestError as exc:
        raise RepositoryForestManifestError(
            exc.reason_code,
            str(exc) or exc.reason_code,
        ) from exc

    expected_aliases = tuple(
        require_aliases
        if require_aliases is not None
        else reviewed.expected_aliases()
    )
    observed_aliases = tuple(item.alias for item in forest.descriptors)
    expected_set = set(expected_aliases)
    observed_set = set(observed_aliases)
    if expected_set - observed_set:
        reasons.append("missing_expected_roots")
    if observed_set - expected_set:
        reasons.append("unexpected_roots")

    write_alias = _normalize_alias(
        require_sole_write_alias or reviewed.sole_write_alias
    )
    if forest.sole_write_alias != write_alias:
        reasons.append("sole_write_alias_mismatch")
    try:
        write_desc = forest.write_descriptor()
        if write_desc.authority.mode != AuthorityMode.READ_WRITE.value:
            reasons.append("write_root_not_writable")
    except RepositoryForestError:
        reasons.append("missing_write_root")

    # Authority and policy checks use the caller's expected manifest when
    # provided; otherwise the embedded reviewed document.
    authority_source = (
        expected_manifest
        if isinstance(expected_manifest, ReviewedForestManifest)
        else reviewed
    )
    if isinstance(expected_manifest, Mapping):
        authority_source = ReviewedForestManifest.from_dict(expected_manifest)

    for reviewed_root in authority_source.roots:
        if reviewed_root.alias not in observed_set:
            if reviewed_root.required:
                reasons.append(f"{reviewed_root.alias}:missing_required_root")
            continue
        live = forest.descriptor_for_alias(reviewed_root.alias)
        if live.authority.mode != reviewed_root.authority_mode:
            reasons.append(f"{reviewed_root.alias}:authority_mismatch")
        assert isinstance(reviewed_root.ignore_policy, IgnorePolicy)
        if live.ignore_policy.policy_cid != reviewed_root.ignore_policy.policy_cid:
            reasons.append(f"{reviewed_root.alias}:ignore_policy_mismatch")
        assert isinstance(reviewed_root.case_unicode_policy, CaseUnicodePolicy)
        if (
            live.case_unicode_policy.policy_cid
            != reviewed_root.case_unicode_policy.policy_cid
        ):
            reasons.append(f"{reviewed_root.alias}:case_unicode_policy_mismatch")

    # When both expected and embedded manifests are present, also ensure the
    # embedded reviewed authority surface still matches live descriptors.
    if (
        embedded_manifest is not None
        and expected_manifest is not None
        and embedded_manifest.manifest_cid != authority_source.manifest_cid
    ):
        for embedded_root in embedded_manifest.roots:
            if embedded_root.alias not in observed_set:
                continue
            live = forest.descriptor_for_alias(embedded_root.alias)
            if live.authority.mode != embedded_root.authority_mode:
                reasons.append(
                    f"{embedded_root.alias}:embedded_authority_mismatch"
                )

    assert isinstance(authority_source.analyzer_profile, AnalyzerProfile)
    forest_profile = forest.analyzer_profile
    if not isinstance(forest_profile, AnalyzerProfile):
        forest_profile = AnalyzerProfile.from_dict(forest_profile or {})
    if forest_profile.profile_cid != authority_source.analyzer_profile.profile_cid:
        reasons.append("analyzer_profile_mismatch")

    # Policy CID on the forest should match a policy rebuilt from reviewed roots
    # (host-independent portable form).
    portable_policy = ForestPolicy(
        roots=tuple(
            ForestRootSpec(
                alias=item.alias,
                root_path=f"portable://{item.alias}",
                authority=item.authority(),
                ignore_policy=item.ignore_policy,
                case_unicode_policy=item.case_unicode_policy,
                logical_name=item.logical_name,
                remote_url=item.remote_url,
                required=item.required,
            )
            for item in authority_source.roots
            if item.alias in observed_set
        ),
        sole_write_alias=authority_source.sole_write_alias,
        analyzer_profile=authority_source.analyzer_profile,
    )
    if forest.policy_cid and forest.policy_cid != portable_policy.policy_cid:
        # Policy CID may differ when optional roots were skipped at materialize
        # time; compare against the subset actually present.
        if set(item.alias for item in portable_policy.roots) == observed_set:
            reasons.append("policy_cid_mismatch")

    claimed_forest_id = str(
        projection.get("forest_id") or forest_payload.get("forest_id") or ""
    ).strip()
    if claimed_forest_id and claimed_forest_id != forest.forest_id:
        reasons.append("forest_id_mismatch")

    valid = not reasons
    return ManifestReplayValidation(
        valid=valid,
        forest_id=forest.forest_id,
        manifest_cid=authority_source.manifest_cid,
        expected_aliases=expected_aliases,
        observed_aliases=observed_aliases,
        reason_codes=tuple(dict.fromkeys(reasons)),
    )


def materialize_initial_four_repository_forest(
    *,
    swissknife_root: str | Path = DEFAULT_SWISSKNIFE_ROOT,
    accelerator_root: str | Path,
    kit_root: str | Path | None = None,
    datasets_root: str | Path | None = None,
    analyzer_profile: AnalyzerProfile | Mapping[str, Any] | None = None,
    require_all_four: bool = True,
    fail_on_missing_required: bool = True,
) -> ForestManifestMaterialization:
    """Convenience: reviewed four-root manifest + fresh forest materialization."""

    manifest = default_reviewed_four_repository_manifest(
        swissknife_root=swissknife_root,
        accelerator_root=accelerator_root,
        kit_root=kit_root,
        datasets_root=datasets_root,
        analyzer_profile=analyzer_profile,
        require_all_four=require_all_four,
    )
    return materialize_forest_from_manifest(
        manifest,
        fail_on_missing_required=fail_on_missing_required,
    )


__all__ = [
    "ANALYZER_PROFILE_SCHEMA",
    "AUTHORITY_SCHEMA",
    "FROZEN_FOUR_ROOT_ALIASES",
    "FOREST_POLICY_SCHEMA",
    "ForestManifestMaterialization",
    "LOCAL_PROJECTION_FILENAME",
    "MANIFEST_LOCAL_PROJECTION_SCHEMA",
    "MANIFEST_PORTABLE_PROJECTION_SCHEMA",
    "MANIFEST_REPLAY_RECEIPT_SCHEMA",
    "ManifestReplayValidation",
    "PORTABLE_PROJECTION_FILENAME",
    "REPOSITORY_FOREST_MANIFEST_SCHEMA",
    "REPOSITORY_FOREST_SCHEMA",
    "REVIEWED_OBSERVED_COMMITS",
    "ReviewedForestManifest",
    "ReviewedManifestRoot",
    "RepositoryForestManifestError",
    "default_reviewed_four_repository_manifest",
    "forest_policy_from_manifest",
    "forests_share_portable_identity",
    "initial_vfs_assurance_forest_policy",
    "load_local_projection",
    "load_portable_projection",
    "load_reviewed_manifest",
    "materialize_forest_from_manifest",
    "materialize_initial_four_repository_forest",
    "persist_manifest_projections",
    "validate_manifest_replay",
]
