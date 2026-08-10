"""DCR-071: structural, implementable source-edit plans.

Interfaces
----------
* ``SourceEditPlan@1`` — one implementable (or explicitly non-implementable)
  plan that binds operator args, exact old-span hashes, and unique AST anchors.
* ``SourceEditSite@1`` — one path-local span with before-hash and rendered
  replacement bytes (not a catalog identity row).

Normative rules (fail-closed)
-----------------------------
* Catalog / identity bindings are evidence only.  A plan is implementable only
  when it carries a rendered replacement that changes source bytes and names a
  unique AST anchor under an exact old-span hash.
* Analysis-only, missing-surface, and IDL-gap dispositions are never
  implementable and never report mutation success.
* Body-free / catalog-only rows are non-implementable.
* Runtime model calls remain 0.

Predicted symbols: :class:`SourceEditPlan`, :class:`SourceEditSite`,
:func:`build_source_edit_plan`.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..proof.formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    content_identity,
)
from .contracts import RepairDisposition


# ---------------------------------------------------------------------------
# Interfaces / schemas / evidence
# ---------------------------------------------------------------------------

SOURCE_EDIT_PLAN_INTERFACE: Final[str] = "SourceEditPlan@1"
SOURCE_EDIT_SITE_INTERFACE: Final[str] = "SourceEditSite@1"
SOURCE_EDIT_PLAN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/source-edit-plan@1"
)
SOURCE_EDIT_SITE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/source-edit-site@1"
)
SOURCE_EDIT_PLAN_VERSION: Final[int] = 1
DCR_MATERIALIZATION_EVIDENCE: Final[str] = "dcr/materialization@1"

MAX_TEXT_BYTES: Final[int] = 4_096
MAX_PATH_BYTES: Final[int] = 1_024
MAX_SPAN_BYTES: Final[int] = 65_536
MAX_SITES: Final[int] = 64
MAX_OPERATOR_ARGS: Final[int] = 32

_PATH_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.[A-Za-z0-9_.-]+$"
)
_AST_ANCHOR_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*(?::[A-Za-z_][A-Za-z0-9_]*)?$"
)
_HASH_RE: Final[re.Pattern[str]] = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")

# Dispositions that may never claim a successful source mutation.
NON_MUTATING_DISPOSITIONS: Final[frozenset[str]] = frozenset(
    {
        RepairDisposition.ANALYSIS_ONLY.value,
        RepairDisposition.MISSING_SURFACE.value,
        RepairDisposition.IDL_GAP.value,
        RepairDisposition.BLOCKED.value,
        "catalog_only",
        "identity_catalog",
        "analysis_only",
        "missing_surface",
        "idl_gap",
        "blocked",
    }
)


class SourceEditPlanError(ContractValidationError):
    """Malformed or non-admissible structural source-edit plan input."""


class SourceEditPlanDisposition(str, Enum):  # noqa: UP042 - Python 3.8 support
    """Closed outcomes for one structural edit plan."""

    IMPLEMENTABLE = "implementable"
    NON_IMPLEMENTABLE = "non_implementable"
    CATALOG_EVIDENCE_ONLY = "catalog_evidence_only"
    ANALYSIS_ONLY = "analysis_only"
    MISSING_SURFACE = "missing_surface"
    IDL_GAP = "idl_gap"
    BLOCKED = "blocked"
    STALE_SPAN = "stale_span"
    AMBIGUOUS_ANCHOR = "ambiguous_anchor"
    EMPTY_REPLACEMENT = "empty_replacement"
    NO_BYTE_CHANGE = "no_byte_change"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _text(
    value: Any,
    name: str,
    *,
    required: bool = True,
    maximum: int = MAX_TEXT_BYTES,
) -> str:
    if value is None:
        if required:
            raise SourceEditPlanError(f"{name} is required")
        return ""
    if not isinstance(value, str):
        raise SourceEditPlanError(f"{name} must be a string")
    text = value.strip()
    if required and not text:
        raise SourceEditPlanError(f"{name} is required")
    if "\x00" in text:
        raise SourceEditPlanError(f"{name} must not contain NUL")
    if len(text.encode("utf-8")) > maximum:
        raise SourceEditPlanError(f"{name} exceeds its byte bound")
    return text


def _optional_text(value: Any, name: str, *, maximum: int = MAX_TEXT_BYTES) -> str:
    return _text(value, name, required=False, maximum=maximum)


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise SourceEditPlanError(f"{name} must be a boolean")
    return value


def _nonneg_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SourceEditPlanError(f"{name} must be a non-negative integer")
    return value


def _path(value: Any, name: str = "path") -> str:
    text = _text(value, name, maximum=MAX_PATH_BYTES)
    pure = PurePosixPath(text)
    if (
        pure.is_absolute()
        or text.startswith("~")
        or ".." in pure.parts
        or text != pure.as_posix()
    ):
        raise SourceEditPlanError(f"{name} must be a relative non-escaping path")
    if not _PATH_RE.fullmatch(text):
        raise SourceEditPlanError(f"{name} must be a closed relative source path")
    return text


def _content_hash(value: Any, name: str) -> str:
    text = _text(value, name)
    if not _HASH_RE.fullmatch(text):
        raise SourceEditPlanError(f"{name} must be a sha256 content hash")
    return text if text.startswith("sha256:") else f"sha256:{text}"


def _ast_anchor(value: Any, name: str = "ast_anchor") -> str:
    text = _text(value, name)
    if not _AST_ANCHOR_RE.fullmatch(text):
        raise SourceEditPlanError(
            f"{name} must be a unique module.attr or module.attr:symbol anchor"
        )
    return text


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise SourceEditPlanError(f"{name} must be a mapping")
    if len(value) > MAX_OPERATOR_ARGS:
        raise SourceEditPlanError(f"{name} exceeds its item bound")
    if not all(isinstance(key, str) for key in value):
        raise SourceEditPlanError(f"{name} keys must be strings")
    return MappingProxyType(dict(value))


def _normalize_disposition(value: Any) -> str:
    if isinstance(value, Enum):
        return str(value.value).strip().lower()
    return str(value or "").strip().lower()


def is_non_mutating_disposition(value: Any) -> bool:
    """Return True when the disposition cannot authorize a source mutation."""

    return _normalize_disposition(value) in NON_MUTATING_DISPOSITIONS


# ---------------------------------------------------------------------------
# SourceEditSite@1
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceEditSite(CanonicalContract):
    """Exact path-local span with rendered replacement (``SourceEditSite@1``)."""

    SCHEMA: ClassVar[str] = SOURCE_EDIT_SITE_SCHEMA
    INTERFACE: ClassVar[str] = SOURCE_EDIT_SITE_INTERFACE

    path: str
    start_offset: int
    end_offset: int
    before_hash: str
    old_span_text: str
    replacement_text: str
    ast_anchor: str
    operator_id: str = ""
    operator_args: Mapping[str, Any] = MappingProxyType({})
    unique_anchor: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _path(self.path, "path"))
        start = _nonneg_int(self.start_offset, "start_offset")
        end = _nonneg_int(self.end_offset, "end_offset")
        if end < start:
            raise SourceEditPlanError("end_offset must be >= start_offset")
        object.__setattr__(self, "start_offset", start)
        object.__setattr__(self, "end_offset", end)

        old_span = self.old_span_text
        if not isinstance(old_span, str):
            raise SourceEditPlanError("old_span_text must be a string")
        if "\x00" in old_span:
            raise SourceEditPlanError("old_span_text must not contain NUL")
        if len(old_span.encode("utf-8")) > MAX_SPAN_BYTES:
            raise SourceEditPlanError("old_span_text exceeds its byte bound")
        object.__setattr__(self, "old_span_text", old_span)

        replacement = self.replacement_text
        if not isinstance(replacement, str):
            raise SourceEditPlanError("replacement_text must be a string")
        if "\x00" in replacement:
            raise SourceEditPlanError("replacement_text must not contain NUL")
        if len(replacement.encode("utf-8")) > MAX_SPAN_BYTES:
            raise SourceEditPlanError("replacement_text exceeds its byte bound")
        object.__setattr__(self, "replacement_text", replacement)

        expected = _sha256_text(old_span)
        supplied = _content_hash(self.before_hash, "before_hash")
        if supplied != expected:
            raise SourceEditPlanError(
                f"before_hash must equal sha256(old_span_text); got {supplied}"
            )
        object.__setattr__(self, "before_hash", supplied)
        object.__setattr__(self, "ast_anchor", _ast_anchor(self.ast_anchor))
        object.__setattr__(
            self, "operator_id", _optional_text(self.operator_id, "operator_id")
        )
        object.__setattr__(
            self, "operator_args", _mapping(self.operator_args, "operator_args")
        )
        object.__setattr__(self, "unique_anchor", _bool(self.unique_anchor, "unique_anchor"))

        if end - start not in {0, len(old_span)} and old_span:
            # Allow absolute offsets that match the span length.
            if end - start != len(old_span):
                raise SourceEditPlanError(
                    "span offsets must match old_span_text length"
                )

    @property
    def after_hash(self) -> str:
        return _sha256_text(self.replacement_text)

    @property
    def changes_bytes(self) -> bool:
        return self.old_span_text != self.replacement_text

    @property
    def is_catalog_only(self) -> bool:
        """True when the site is an identity/catalog row without a real edit."""

        if not self.replacement_text and not self.old_span_text:
            return True
        # Explicit catalog markers in operator args.
        args = dict(self.operator_args)
        if args.get("catalog_only") is True or args.get("identity_catalog") is True:
            return True
        if args.get("mutation_kind") in {"catalog", "identity_catalog", "binding_only"}:
            return True
        return False

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": SOURCE_EDIT_PLAN_VERSION,
            "interface": self.INTERFACE,
            "path": self.path,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "before_hash": self.before_hash,
            "after_hash": self.after_hash,
            "old_span_text": self.old_span_text,
            "replacement_text": self.replacement_text,
            "ast_anchor": self.ast_anchor,
            "operator_id": self.operator_id,
            "operator_args": dict(self.operator_args),
            "unique_anchor": self.unique_anchor,
            "changes_bytes": self.changes_bytes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | Any) -> "SourceEditSite":
        if not isinstance(payload, Mapping):
            raise SourceEditPlanError("source edit site must be an object")
        return cls(
            path=str(payload.get("path") or ""),
            start_offset=int(payload.get("start_offset") or payload.get("start") or 0),
            end_offset=int(payload.get("end_offset") or payload.get("end") or 0),
            before_hash=str(payload.get("before_hash") or payload.get("old_span_hash") or ""),
            old_span_text=str(
                payload.get("old_span_text")
                or payload.get("span_text")
                or payload.get("before_text")
                or ""
            ),
            replacement_text=str(
                payload.get("replacement_text")
                or payload.get("replacement")
                or payload.get("after_text")
                or ""
            ),
            ast_anchor=str(payload.get("ast_anchor") or payload.get("anchor") or ""),
            operator_id=str(payload.get("operator_id") or ""),
            operator_args=dict(payload.get("operator_args") or {}),
            unique_anchor=bool(payload.get("unique_anchor", True)),
        )


# ---------------------------------------------------------------------------
# SourceEditPlan@1
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceEditPlan(CanonicalContract):
    """Implementable structural source-edit plan (``SourceEditPlan@1``)."""

    SCHEMA: ClassVar[str] = SOURCE_EDIT_PLAN_SCHEMA
    INTERFACE: ClassVar[str] = SOURCE_EDIT_PLAN_INTERFACE

    plan_id: str
    sites: tuple[SourceEditSite, ...]
    disposition: SourceEditPlanDisposition
    work_id: str = ""
    packet_cid: str = ""
    operator_cid: str = ""
    owner_root: str = ""
    worktree_root: str = ""
    admission_cid: str = ""
    reason_codes: tuple[str, ...] = ()
    catalog_evidence: Mapping[str, Any] = MappingProxyType({})
    runtime_model_calls: int = 0
    implementable: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.disposition, SourceEditPlanDisposition):
            disposition = self.disposition
        else:
            try:
                disposition = SourceEditPlanDisposition(str(self.disposition).strip())
            except ValueError as exc:
                raise SourceEditPlanError(
                    "disposition must be a SourceEditPlanDisposition member"
                ) from exc
        object.__setattr__(self, "disposition", disposition)

        if not isinstance(self.sites, Sequence) or isinstance(
            self.sites, (str, bytes, bytearray)
        ):
            raise SourceEditPlanError("sites must be a sequence")
        if len(self.sites) > MAX_SITES:
            raise SourceEditPlanError("sites exceed the closed bound")
        sites = tuple(
            item if isinstance(item, SourceEditSite) else SourceEditSite.from_dict(item)
            for item in self.sites
        )
        object.__setattr__(self, "sites", sites)

        reasons = tuple(
            _text(item, f"reason_codes[{index}]")
            for index, item in enumerate(self.reason_codes or ())
        )
        object.__setattr__(self, "reason_codes", reasons)

        object.__setattr__(self, "work_id", _optional_text(self.work_id, "work_id"))
        object.__setattr__(self, "packet_cid", _optional_text(self.packet_cid, "packet_cid"))
        object.__setattr__(
            self, "operator_cid", _optional_text(self.operator_cid, "operator_cid")
        )
        object.__setattr__(self, "owner_root", _optional_text(self.owner_root, "owner_root"))
        object.__setattr__(
            self, "worktree_root", _optional_text(self.worktree_root, "worktree_root")
        )
        object.__setattr__(
            self, "admission_cid", _optional_text(self.admission_cid, "admission_cid")
        )
        object.__setattr__(
            self, "catalog_evidence", _mapping(self.catalog_evidence, "catalog_evidence")
        )

        if (
            isinstance(self.runtime_model_calls, bool)
            or not isinstance(self.runtime_model_calls, int)
            or self.runtime_model_calls != 0
        ):
            raise SourceEditPlanError("runtime_model_calls must be exactly 0")
        object.__setattr__(self, "runtime_model_calls", 0)

        implementable, derived_reasons = self._derive_implementable(sites, disposition)
        merged_reasons = list(reasons)
        for code in derived_reasons:
            if code not in merged_reasons:
                merged_reasons.append(code)
        # Caller may only restrict implementability.
        final_implementable = bool(self.implementable) and implementable
        if disposition is not SourceEditPlanDisposition.IMPLEMENTABLE:
            final_implementable = False
        object.__setattr__(self, "implementable", final_implementable)
        object.__setattr__(self, "reason_codes", tuple(merged_reasons))

        plan_id = _optional_text(self.plan_id, "plan_id")
        if not plan_id:
            plan_id = content_identity(self._payload_without_plan_id())
        object.__setattr__(self, "plan_id", plan_id)

    @staticmethod
    def _derive_implementable(
        sites: Sequence[SourceEditSite],
        disposition: SourceEditPlanDisposition,
    ) -> tuple[bool, tuple[str, ...]]:
        reasons: list[str] = []
        if disposition in {
            SourceEditPlanDisposition.ANALYSIS_ONLY,
            SourceEditPlanDisposition.MISSING_SURFACE,
            SourceEditPlanDisposition.IDL_GAP,
            SourceEditPlanDisposition.BLOCKED,
            SourceEditPlanDisposition.CATALOG_EVIDENCE_ONLY,
            SourceEditPlanDisposition.NON_IMPLEMENTABLE,
        }:
            reasons.append(disposition.value)
            return False, tuple(reasons)
        if not sites:
            reasons.append("empty_sites")
            return False, tuple(reasons)

        anchors: list[str] = []
        any_change = False
        for site in sites:
            if not site.unique_anchor:
                reasons.append(SourceEditPlanDisposition.AMBIGUOUS_ANCHOR.value)
            anchors.append(site.ast_anchor)
            if site.is_catalog_only:
                reasons.append(SourceEditPlanDisposition.CATALOG_EVIDENCE_ONLY.value)
            if not site.changes_bytes:
                reasons.append(SourceEditPlanDisposition.NO_BYTE_CHANGE.value)
            else:
                any_change = True
            if not site.replacement_text and site.old_span_text:
                # Deletion is a real change; empty replacement with empty old is catalog.
                pass
            if site.replacement_text == "" and site.old_span_text == "":
                reasons.append(SourceEditPlanDisposition.EMPTY_REPLACEMENT.value)

        if len(anchors) != len(set(anchors)):
            reasons.append(SourceEditPlanDisposition.AMBIGUOUS_ANCHOR.value)
        if not any_change:
            if SourceEditPlanDisposition.NO_BYTE_CHANGE.value not in reasons:
                reasons.append(SourceEditPlanDisposition.NO_BYTE_CHANGE.value)

        # Stable unique reasons.
        ordered: list[str] = []
        seen: set[str] = set()
        for code in reasons:
            if code not in seen:
                seen.add(code)
                ordered.append(code)
        implementable = not ordered and any_change
        return implementable, tuple(ordered)

    def _payload_without_plan_id(self) -> dict[str, Any]:
        return {
            "contract_version": SOURCE_EDIT_PLAN_VERSION,
            "interface": self.INTERFACE,
            "sites": [site.to_dict() for site in self.sites],
            "disposition": self.disposition.value,
            "work_id": self.work_id,
            "packet_cid": self.packet_cid,
            "operator_cid": self.operator_cid,
            "owner_root": self.owner_root,
            "worktree_root": self.worktree_root,
            "admission_cid": self.admission_cid,
            "reason_codes": list(self.reason_codes),
            "catalog_evidence": dict(self.catalog_evidence),
            "runtime_model_calls": 0,
            "implementable": self.implementable,
            "grants_write_authority": False,
            "completion_authoritative": False,
        }

    def _payload(self) -> dict[str, Any]:
        return {**self._payload_without_plan_id(), "plan_id": self.plan_id}

    @property
    def claims_source_mutation(self) -> bool:
        """True only for implementable plans that change source bytes."""

        return self.implementable and any(site.changes_bytes for site in self.sites)

    def evidence_subset(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "implementable": self.implementable,
            "disposition": self.disposition.value,
            "reason_codes": list(self.reason_codes),
            "site_count": len(self.sites),
            "paths": sorted({site.path for site in self.sites}),
            "ast_anchors": [site.ast_anchor for site in self.sites],
            "before_hashes": [site.before_hash for site in self.sites],
            "after_hashes": [site.after_hash for site in self.sites],
            "operator_ids": sorted(
                {site.operator_id for site in self.sites if site.operator_id}
            ),
            "runtime_model_calls": 0,
            "catalog_evidence_only": bool(self.catalog_evidence)
            and not self.implementable,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | Any) -> "SourceEditPlan":
        if not isinstance(payload, Mapping):
            raise SourceEditPlanError("source edit plan must be an object")
        raw_sites = payload.get("sites") or ()
        if isinstance(raw_sites, (str, bytes, bytearray)) or not isinstance(
            raw_sites, Sequence
        ):
            raise SourceEditPlanError("sites must be a sequence")
        return cls(
            plan_id=str(payload.get("plan_id") or ""),
            sites=tuple(raw_sites),
            disposition=payload.get("disposition")
            or SourceEditPlanDisposition.NON_IMPLEMENTABLE,
            work_id=str(payload.get("work_id") or ""),
            packet_cid=str(payload.get("packet_cid") or ""),
            operator_cid=str(payload.get("operator_cid") or ""),
            owner_root=str(payload.get("owner_root") or ""),
            worktree_root=str(payload.get("worktree_root") or ""),
            admission_cid=str(payload.get("admission_cid") or ""),
            reason_codes=tuple(payload.get("reason_codes") or ()),
            catalog_evidence=dict(payload.get("catalog_evidence") or {}),
            runtime_model_calls=int(payload.get("runtime_model_calls") or 0),
            implementable=bool(payload.get("implementable", False)),
        )


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def make_source_edit_site(
    *,
    path: str,
    old_span_text: str,
    replacement_text: str,
    ast_anchor: str,
    start_offset: int = 0,
    operator_id: str = "",
    operator_args: Mapping[str, Any] | None = None,
    unique_anchor: bool = True,
) -> SourceEditSite:
    """Build one site with before_hash bound to ``old_span_text``."""

    text = old_span_text if isinstance(old_span_text, str) else ""
    return SourceEditSite(
        path=path,
        start_offset=start_offset,
        end_offset=start_offset + len(text),
        before_hash=_sha256_text(text),
        old_span_text=text,
        replacement_text=replacement_text if isinstance(replacement_text, str) else "",
        ast_anchor=ast_anchor,
        operator_id=operator_id,
        operator_args=dict(operator_args or {}),
        unique_anchor=unique_anchor,
    )


def disposition_for_row(row: Mapping[str, Any] | Any) -> SourceEditPlanDisposition:
    """Map a work-item / engine row disposition into a plan disposition."""

    if not isinstance(row, Mapping):
        return SourceEditPlanDisposition.NON_IMPLEMENTABLE
    raw = (
        row.get("disposition")
        or row.get("repair_disposition")
        or row.get("kind")
        or ""
    )
    normalized = _normalize_disposition(raw)
    if normalized in {
        RepairDisposition.ANALYSIS_ONLY.value,
        "analysis_only",
    }:
        return SourceEditPlanDisposition.ANALYSIS_ONLY
    if normalized in {
        RepairDisposition.MISSING_SURFACE.value,
        "missing_surface",
        "missing",
    }:
        return SourceEditPlanDisposition.MISSING_SURFACE
    if normalized in {RepairDisposition.IDL_GAP.value, "idl_gap", "idl"}:
        return SourceEditPlanDisposition.IDL_GAP
    if normalized in {RepairDisposition.BLOCKED.value, "blocked"}:
        return SourceEditPlanDisposition.BLOCKED
    if normalized in {"catalog_only", "identity_catalog", "catalog"}:
        return SourceEditPlanDisposition.CATALOG_EVIDENCE_ONLY
    if row.get("catalog_only") is True or row.get("identity_catalog") is True:
        return SourceEditPlanDisposition.CATALOG_EVIDENCE_ONLY
    if row.get("implementable") is True or normalized in {
        RepairDisposition.SINGLE_PATH_READY.value,
        "single_path_ready",
        "implementable",
        "ready",
    }:
        return SourceEditPlanDisposition.IMPLEMENTABLE
    return SourceEditPlanDisposition.NON_IMPLEMENTABLE


def build_source_edit_plan(
    *,
    sites: Sequence[SourceEditSite | Mapping[str, Any]] = (),
    disposition: SourceEditPlanDisposition | str | None = None,
    work_id: str = "",
    packet_cid: str = "",
    operator_cid: str = "",
    owner_root: str = "",
    worktree_root: str = "",
    admission_cid: str = "",
    reason_codes: Sequence[str] = (),
    catalog_evidence: Mapping[str, Any] | None = None,
    implementable: bool | None = None,
    row: Mapping[str, Any] | None = None,
) -> SourceEditPlan:
    """Build one fail-closed structural source-edit plan.

    When ``row`` is supplied, non-mutating dispositions are forced and any
    catalog-only payload is retained as evidence without becoming implementable.
    """

    resolved_disposition: SourceEditPlanDisposition
    if disposition is None and row is not None:
        resolved_disposition = disposition_for_row(row)
    elif disposition is None:
        resolved_disposition = SourceEditPlanDisposition.IMPLEMENTABLE
    elif isinstance(disposition, SourceEditPlanDisposition):
        resolved_disposition = disposition
    else:
        try:
            resolved_disposition = SourceEditPlanDisposition(str(disposition).strip())
        except ValueError as exc:
            raise SourceEditPlanError("unknown source-edit plan disposition") from exc

    if row is not None and is_non_mutating_disposition(
        row.get("disposition") or row.get("kind") or resolved_disposition.value
    ):
        # Force non-mutating plan regardless of supplied sites.
        if resolved_disposition is SourceEditPlanDisposition.IMPLEMENTABLE:
            resolved_disposition = disposition_for_row(row)

    catalog = dict(catalog_evidence or {})
    if row is not None:
        for key in ("catalog_binding", "catalog_id", "identity", "surface_id"):
            if key in row and key not in catalog:
                catalog[key] = row[key]

    plan = SourceEditPlan(
        plan_id="",
        sites=tuple(sites),
        disposition=resolved_disposition,
        work_id=work_id or str((row or {}).get("work_id") or ""),
        packet_cid=packet_cid or str((row or {}).get("packet_cid") or ""),
        operator_cid=operator_cid or str((row or {}).get("operator_cid") or ""),
        owner_root=owner_root or str((row or {}).get("owner_root") or ""),
        worktree_root=worktree_root or str((row or {}).get("worktree_root") or ""),
        admission_cid=admission_cid or str((row or {}).get("admission_cid") or ""),
        reason_codes=tuple(reason_codes),
        catalog_evidence=catalog,
        runtime_model_calls=0,
        implementable=True if implementable is None else bool(implementable),
    )
    return plan


def build_catalog_evidence_plan(
    *,
    work_id: str,
    catalog_evidence: Mapping[str, Any],
    disposition: SourceEditPlanDisposition = SourceEditPlanDisposition.CATALOG_EVIDENCE_ONLY,
    reason_codes: Sequence[str] = (),
) -> SourceEditPlan:
    """Build an evidence-only plan that never claims source mutation success."""

    return build_source_edit_plan(
        sites=(),
        disposition=disposition,
        work_id=work_id,
        catalog_evidence=catalog_evidence,
        reason_codes=reason_codes or (disposition.value,),
        implementable=False,
    )


__all__ = [
    "DCR_MATERIALIZATION_EVIDENCE",
    "NON_MUTATING_DISPOSITIONS",
    "SOURCE_EDIT_PLAN_INTERFACE",
    "SOURCE_EDIT_PLAN_SCHEMA",
    "SOURCE_EDIT_PLAN_VERSION",
    "SOURCE_EDIT_SITE_INTERFACE",
    "SOURCE_EDIT_SITE_SCHEMA",
    "SourceEditPlan",
    "SourceEditPlanDisposition",
    "SourceEditPlanError",
    "SourceEditSite",
    "build_catalog_evidence_plan",
    "build_source_edit_plan",
    "disposition_for_row",
    "is_non_mutating_disposition",
    "make_source_edit_site",
]
