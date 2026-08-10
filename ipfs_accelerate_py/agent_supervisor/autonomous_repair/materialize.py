"""DCR-071: structural source-edit materialization (not catalog identity rows).

Interfaces
----------
* ``StructuralRepairMaterializer@1`` — previews and applies operator-rendered
  source/structured-data mutations under an admitted owner worktree.
* ``AdmittedSourceEditOperator@1`` — one admitted, invertible source edit with
  exact old-span hash and unique AST anchor.
* Reuses ``CodeEditPacket@1`` as the supervisor-facing implementable packet
  projection when a tree/claim binding is present.

Normative rules (fail-closed)
-----------------------------
* Write only operator-rendered edits with exact old-span hash and unique AST
  anchor beneath the admitted owner worktree.
* Successful results must contain changed source bytes and a reversible diff.
* Catalog bindings remain evidence and never count as mutation success.
* Analysis-only / missing / IDL rows are nonpassing.
* Receipt-write failures are nonpassing.
* Runtime model calls remain 0.

Predicted symbols: :class:`StructuralRepairMaterializer`,
:class:`AdmittedSourceEditOperator`, :func:`apply_operator`,
:class:`CodeEditPacket` (imported).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..proof.code_edit_packet import (
    CODE_EDIT_PACKET_INTERFACE,
    CodeEditPacket,
    build_code_edit_packet,
)
from ..proof.formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
)
from .edit_plan import (
    DCR_MATERIALIZATION_EVIDENCE,
    SOURCE_EDIT_PLAN_INTERFACE,
    SourceEditPlan,
    SourceEditPlanDisposition,
    SourceEditSite,
    build_source_edit_plan,
    is_non_mutating_disposition,
    make_source_edit_site,
)


# ---------------------------------------------------------------------------
# Interfaces / schemas
# ---------------------------------------------------------------------------

STRUCTURAL_REPAIR_MATERIALIZER_INTERFACE: Final[str] = "StructuralRepairMaterializer@1"
ADMITTED_SOURCE_EDIT_OPERATOR_INTERFACE: Final[str] = "AdmittedSourceEditOperator@1"
SOURCE_WRITE_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/source-write-receipt@1"
)
MATERIALIZATION_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/structural-materialization-result@1"
)
MATERIALIZATION_VECTORS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/dcr-materialization-vectors@1"
)
DEFAULT_MATERIALIZATION_VECTORS_REL: Final[str] = (
    "data/agent_supervisor/deterministic_contract_repair/materialization-vectors.json"
)

MAX_FILE_BYTES: Final[int] = 1_048_576
MAX_PATH_BYTES: Final[int] = 1_024
MATERIALIZER_VERSION: Final[int] = 1


class StructuralMaterializeError(ContractValidationError):
    """Malformed materialization input or unsafe source write."""


class MaterializeDisposition(str, Enum):  # noqa: UP042 - Python 3.8 support
    """Closed outcomes for one structural materialization attempt."""

    APPLIED = "applied"
    PREVIEWED = "previewed"
    REJECTED = "rejected"
    NONPASSING = "nonpassing"
    RECEIPT_WRITE_FAILED = "receipt_write_failed"
    STALE_SPAN = "stale_span"
    PATH_ESCAPE = "path_escape"
    CATALOG_EVIDENCE_ONLY = "catalog_evidence_only"
    ANALYSIS_ONLY = "analysis_only"
    MISSING_SURFACE = "missing_surface"
    IDL_GAP = "idl_gap"
    NO_BYTE_CHANGE = "no_byte_change"
    NON_IMPLEMENTABLE = "non_implementable"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if value is None:
        if required:
            raise StructuralMaterializeError(f"{name} is required")
        return ""
    if not isinstance(value, str):
        raise StructuralMaterializeError(f"{name} must be a string")
    text = value.strip()
    if required and not text:
        raise StructuralMaterializeError(f"{name} is required")
    if "\x00" in text:
        raise StructuralMaterializeError(f"{name} must not contain NUL")
    return text


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise StructuralMaterializeError(f"{name} must be a boolean")
    return value


def _safe_relpath(value: Any, name: str = "path") -> str:
    text = _text(value, name, required=True)
    if len(text.encode("utf-8")) > MAX_PATH_BYTES:
        raise StructuralMaterializeError(f"{name} exceeds path byte bound")
    pure = PurePosixPath(text)
    if (
        pure.is_absolute()
        or text.startswith("~")
        or text.startswith("\\")
        or ".." in pure.parts
        or text != pure.as_posix()
        or pure.as_posix() in {"", "."}
    ):
        raise StructuralMaterializeError(f"{name} must be a relative in-worktree path")
    return pure.as_posix()


def _resolve_under(root: Path, relative: str) -> Path:
    """Resolve ``relative`` under ``root`` without allowing escape."""

    rel = _safe_relpath(relative, "path")
    root_resolved = root.resolve()
    target = (root_resolved / rel).resolve()
    try:
        target.relative_to(root_resolved)
    except ValueError as exc:
        raise StructuralMaterializeError(
            f"{MaterializeDisposition.PATH_ESCAPE.value}: path escapes worktree"
        ) from exc
    return target


def _reversible_patch(path: str, before: str, after: str) -> str:
    """Build a compact structured reversible patch document.

    The document embeds exact before/after UTF-8 payloads under content hashes
    so inverse reconstruction never depends on line-oriented diff parsing.
    """

    payload = {
        "format": "dcr-reversible-source-patch@1",
        "path": path,
        "before_hash": _sha256_text(before),
        "after_hash": _sha256_text(after),
        "before_text": before,
        "after_text": after,
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _inverse_patch_document(patch: str) -> str:
    """Swap before/after in a reversible patch document."""

    try:
        payload = json.loads(patch)
    except (TypeError, json.JSONDecodeError) as exc:
        raise StructuralMaterializeError("patch is not a reversible document") from exc
    if not isinstance(payload, Mapping):
        raise StructuralMaterializeError("patch is not a reversible document")
    before = payload.get("before_text")
    after = payload.get("after_text")
    if not isinstance(before, str) or not isinstance(after, str):
        raise StructuralMaterializeError("patch is missing before/after text")
    path = str(payload.get("path") or "")
    return _reversible_patch(path, before=after, after=before)


def _before_text_from_patch(patch: str) -> str:
    try:
        payload = json.loads(patch)
    except (TypeError, json.JSONDecodeError) as exc:
        raise StructuralMaterializeError("patch is not a reversible document") from exc
    if not isinstance(payload, Mapping) or not isinstance(payload.get("before_text"), str):
        raise StructuralMaterializeError("patch is missing before_text")
    before = str(payload["before_text"])
    claimed = str(payload.get("before_hash") or "")
    if claimed and claimed != _sha256_text(before):
        raise StructuralMaterializeError("patch before_hash mismatch")
    return before


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = text.encode("utf-8")
    if len(payload) > MAX_FILE_BYTES:
        raise StructuralMaterializeError("file exceeds single-file bound")
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        try:
            temporary_path.unlink(missing_ok=True)  # type: ignore[call-arg]
        except TypeError:
            # Python <3.8 missing_ok compatibility (not expected under 3.12).
            if temporary_path.exists():
                temporary_path.unlink()
        raise


def _apply_span_to_text(
    file_text: str,
    site: SourceEditSite,
) -> str:
    """Apply one site replacement, verifying the old-span hash."""

    start = site.start_offset
    end = site.end_offset
    if 0 <= start <= end <= len(file_text):
        region = file_text[start:end]
        if region == site.old_span_text or (start == end and not site.old_span_text):
            observed = _sha256_text(region)
            if site.old_span_text and observed != site.before_hash:
                raise StructuralMaterializeError(
                    f"{MaterializeDisposition.STALE_SPAN.value}: before_hash mismatch"
                )
            return file_text[:start] + site.replacement_text + file_text[end:]
    # Fallback: unique exact span search.
    if site.old_span_text:
        count = file_text.count(site.old_span_text)
        if count == 1:
            index = file_text.find(site.old_span_text)
            observed = _sha256_text(site.old_span_text)
            if observed != site.before_hash:
                raise StructuralMaterializeError(
                    f"{MaterializeDisposition.STALE_SPAN.value}: before_hash mismatch"
                )
            return (
                file_text[:index]
                + site.replacement_text
                + file_text[index + len(site.old_span_text) :]
            )
        if count == 0:
            raise StructuralMaterializeError(
                f"{MaterializeDisposition.STALE_SPAN.value}: old span missing"
            )
        raise StructuralMaterializeError(
            f"{MaterializeDisposition.STALE_SPAN.value}: old span not unique"
        )
    raise StructuralMaterializeError(
        f"{MaterializeDisposition.STALE_SPAN.value}: empty span with no offsets"
    )


# ---------------------------------------------------------------------------
# Write receipt / operator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceWriteReceipt(CanonicalContract):
    """Evidence for one structural source write (or failed write attempt)."""

    SCHEMA: ClassVar[str] = SOURCE_WRITE_RECEIPT_SCHEMA
    INTERFACE: ClassVar[str] = STRUCTURAL_REPAIR_MATERIALIZER_INTERFACE

    path: str
    before_hash: str
    after_hash: str
    before_bytes: int
    after_bytes: int
    ast_anchor: str
    operator_id: str
    operator_args: Mapping[str, Any]
    patch: str
    inverse_patch: str
    changed_source_bytes: bool
    reversible: bool
    written: bool
    receipt_written: bool
    disposition: MaterializeDisposition
    reason_codes: tuple[str, ...] = ()
    worktree_root: str = ""
    plan_id: str = ""
    runtime_model_calls: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _safe_relpath(self.path, "path"))
        object.__setattr__(self, "before_hash", _text(self.before_hash, "before_hash"))
        object.__setattr__(self, "after_hash", _text(self.after_hash, "after_hash"))
        for name in ("before_bytes", "after_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise StructuralMaterializeError(f"{name} must be a non-negative integer")
        object.__setattr__(self, "ast_anchor", _text(self.ast_anchor, "ast_anchor"))
        object.__setattr__(
            self, "operator_id", _text(self.operator_id, "operator_id", required=False)
        )
        if not isinstance(self.operator_args, Mapping):
            raise StructuralMaterializeError("operator_args must be a mapping")
        object.__setattr__(
            self, "operator_args", MappingProxyType(dict(self.operator_args))
        )
        object.__setattr__(self, "patch", self.patch if isinstance(self.patch, str) else "")
        object.__setattr__(
            self,
            "inverse_patch",
            self.inverse_patch if isinstance(self.inverse_patch, str) else "",
        )
        object.__setattr__(
            self, "changed_source_bytes", _bool(self.changed_source_bytes, "changed_source_bytes")
        )
        object.__setattr__(self, "reversible", _bool(self.reversible, "reversible"))
        object.__setattr__(self, "written", _bool(self.written, "written"))
        object.__setattr__(
            self, "receipt_written", _bool(self.receipt_written, "receipt_written")
        )
        if isinstance(self.disposition, MaterializeDisposition):
            disposition = self.disposition
        else:
            try:
                disposition = MaterializeDisposition(str(self.disposition).strip())
            except ValueError as exc:
                raise StructuralMaterializeError("unknown materialize disposition") from exc
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(
            self,
            "reason_codes",
            tuple(_text(item, "reason_codes") for item in (self.reason_codes or ())),
        )
        object.__setattr__(
            self, "worktree_root", _text(self.worktree_root, "worktree_root", required=False)
        )
        object.__setattr__(self, "plan_id", _text(self.plan_id, "plan_id", required=False))
        if (
            isinstance(self.runtime_model_calls, bool)
            or not isinstance(self.runtime_model_calls, int)
            or self.runtime_model_calls != 0
        ):
            raise StructuralMaterializeError("runtime_model_calls must be exactly 0")
        object.__setattr__(self, "runtime_model_calls", 0)

    @property
    def passed(self) -> bool:
        """Success requires real source change, reversible diff, and receipt write."""

        return (
            self.disposition is MaterializeDisposition.APPLIED
            and self.written
            and self.changed_source_bytes
            and self.reversible
            and self.receipt_written
            and self.before_hash != self.after_hash
            and bool(self.patch)
            and bool(self.inverse_patch)
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": MATERIALIZER_VERSION,
            "interface": self.INTERFACE,
            "schema": self.SCHEMA,
            "path": self.path,
            "before_hash": self.before_hash,
            "after_hash": self.after_hash,
            "before_bytes": self.before_bytes,
            "after_bytes": self.after_bytes,
            "ast_anchor": self.ast_anchor,
            "operator_id": self.operator_id,
            "operator_args": dict(self.operator_args),
            "patch": self.patch,
            "inverse_patch": self.inverse_patch,
            "changed_source_bytes": self.changed_source_bytes,
            "reversible": self.reversible,
            "written": self.written,
            "receipt_written": self.receipt_written,
            "disposition": self.disposition.value,
            "reason_codes": list(self.reason_codes),
            "worktree_root": self.worktree_root,
            "plan_id": self.plan_id,
            "runtime_model_calls": 0,
            "passed": self.passed,
            "completion_authoritative": False,
        }

    def evidence_subset(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "before_hash": self.before_hash,
            "after_hash": self.after_hash,
            "before_bytes": self.before_bytes,
            "after_bytes": self.after_bytes,
            "ast_anchor": self.ast_anchor,
            "operator_id": self.operator_id,
            "operator_args": dict(self.operator_args),
            "patch_present": bool(self.patch),
            "inverse_present": bool(self.inverse_patch),
            "changed_source_bytes": self.changed_source_bytes,
            "reversible": self.reversible,
            "receipt_written": self.receipt_written,
            "passed": self.passed,
            "disposition": self.disposition.value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SourceWriteReceipt":
        if not isinstance(payload, Mapping):
            raise StructuralMaterializeError("write receipt must be an object")
        return cls(
            path=str(payload.get("path") or ""),
            before_hash=str(payload.get("before_hash") or ""),
            after_hash=str(payload.get("after_hash") or ""),
            before_bytes=int(payload.get("before_bytes") or 0),
            after_bytes=int(payload.get("after_bytes") or 0),
            ast_anchor=str(payload.get("ast_anchor") or ""),
            operator_id=str(payload.get("operator_id") or ""),
            operator_args=dict(payload.get("operator_args") or {}),
            patch=str(payload.get("patch") or ""),
            inverse_patch=str(payload.get("inverse_patch") or ""),
            changed_source_bytes=bool(payload.get("changed_source_bytes", False)),
            reversible=bool(payload.get("reversible", False)),
            written=bool(payload.get("written", False)),
            receipt_written=bool(payload.get("receipt_written", False)),
            disposition=payload.get("disposition") or MaterializeDisposition.REJECTED,
            reason_codes=tuple(payload.get("reason_codes") or ()),
            worktree_root=str(payload.get("worktree_root") or ""),
            plan_id=str(payload.get("plan_id") or ""),
            runtime_model_calls=int(payload.get("runtime_model_calls") or 0),
        )


@dataclass(frozen=True)
class AdmittedSourceEditOperator(CanonicalContract):
    """One admitted, invertible source edit operator binding."""

    SCHEMA: ClassVar[str] = (
        "ipfs_accelerate_py/agent-supervisor/admitted-source-edit-operator@1"
    )
    INTERFACE: ClassVar[str] = ADMITTED_SOURCE_EDIT_OPERATOR_INTERFACE

    operator_id: str
    site: SourceEditSite
    admission_cid: str
    packet_cid: str = ""
    inverse_kind: str = "restore_old_span"
    grants_write_authority: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "operator_id", _text(self.operator_id, "operator_id"))
        if not isinstance(self.site, SourceEditSite):
            raise StructuralMaterializeError("site must be a SourceEditSite")
        object.__setattr__(
            self, "admission_cid", _text(self.admission_cid, "admission_cid")
        )
        object.__setattr__(
            self, "packet_cid", _text(self.packet_cid, "packet_cid", required=False)
        )
        object.__setattr__(
            self, "inverse_kind", _text(self.inverse_kind, "inverse_kind")
        )
        object.__setattr__(
            self,
            "grants_write_authority",
            _bool(self.grants_write_authority, "grants_write_authority"),
        )
        if not self.site.unique_anchor:
            raise StructuralMaterializeError("admitted operator requires a unique AST anchor")
        if not self.site.changes_bytes:
            raise StructuralMaterializeError(
                "admitted operator requires a byte-changing replacement"
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": MATERIALIZER_VERSION,
            "interface": self.INTERFACE,
            "operator_id": self.operator_id,
            "site": self.site.to_dict(),
            "admission_cid": self.admission_cid,
            "packet_cid": self.packet_cid,
            "inverse_kind": self.inverse_kind,
            "grants_write_authority": self.grants_write_authority,
            "runtime_model_calls": 0,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AdmittedSourceEditOperator":
        if not isinstance(payload, Mapping):
            raise StructuralMaterializeError("admitted operator must be an object")
        site_raw = payload.get("site")
        if not isinstance(site_raw, Mapping):
            raise StructuralMaterializeError("site must be an object")
        return cls(
            operator_id=str(payload.get("operator_id") or ""),
            site=SourceEditSite.from_dict(site_raw),
            admission_cid=str(payload.get("admission_cid") or ""),
            packet_cid=str(payload.get("packet_cid") or ""),
            inverse_kind=str(payload.get("inverse_kind") or "restore_old_span"),
            grants_write_authority=bool(payload.get("grants_write_authority", True)),
        )


@dataclass(frozen=True)
class MaterializationResult(CanonicalContract):
    """Aggregate result for one plan materialization."""

    SCHEMA: ClassVar[str] = MATERIALIZATION_RESULT_SCHEMA
    INTERFACE: ClassVar[str] = STRUCTURAL_REPAIR_MATERIALIZER_INTERFACE

    plan: SourceEditPlan
    receipts: tuple[SourceWriteReceipt, ...]
    disposition: MaterializeDisposition
    passed: bool
    code_edit_packet: CodeEditPacket | None = None
    notes: tuple[str, ...] = ()
    runtime_model_calls: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.plan, SourceEditPlan):
            raise StructuralMaterializeError("plan must be a SourceEditPlan")
        if not isinstance(self.receipts, Sequence) or isinstance(
            self.receipts, (str, bytes, bytearray)
        ):
            raise StructuralMaterializeError("receipts must be a sequence")
        if not all(isinstance(item, SourceWriteReceipt) for item in self.receipts):
            raise StructuralMaterializeError("receipts must contain SourceWriteReceipt")
        object.__setattr__(self, "receipts", tuple(self.receipts))
        if isinstance(self.disposition, MaterializeDisposition):
            disposition = self.disposition
        else:
            disposition = MaterializeDisposition(str(self.disposition))
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(self, "passed", _bool(self.passed, "passed"))
        if self.code_edit_packet is not None and not isinstance(
            self.code_edit_packet, CodeEditPacket
        ):
            raise StructuralMaterializeError("code_edit_packet must be CodeEditPacket")
        if (
            isinstance(self.runtime_model_calls, bool)
            or not isinstance(self.runtime_model_calls, int)
            or self.runtime_model_calls != 0
        ):
            raise StructuralMaterializeError("runtime_model_calls must be exactly 0")
        object.__setattr__(self, "runtime_model_calls", 0)
        # Enforce acceptance: pass only with real changed bytes + reversible diff.
        if self.passed:
            if not self.receipts or not all(item.passed for item in self.receipts):
                raise StructuralMaterializeError(
                    "passed materialization requires every write receipt to pass"
                )
            if not self.plan.implementable:
                raise StructuralMaterializeError(
                    "passed materialization requires an implementable plan"
                )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": MATERIALIZER_VERSION,
            "interface": self.INTERFACE,
            "schema": self.SCHEMA,
            "evidence_id": DCR_MATERIALIZATION_EVIDENCE,
            "plan": self.plan.to_dict(),
            "receipts": [item.to_dict() for item in self.receipts],
            "disposition": self.disposition.value,
            "passed": self.passed,
            "code_edit_packet": (
                self.code_edit_packet.to_dict() if self.code_edit_packet is not None else None
            ),
            "notes": list(self.notes),
            "runtime_model_calls": 0,
            "completion_authoritative": False,
        }

    def evidence_subset(self) -> dict[str, Any]:
        return {
            "evidence_id": DCR_MATERIALIZATION_EVIDENCE,
            "plan_id": self.plan.plan_id,
            "passed": self.passed,
            "disposition": self.disposition.value,
            "receipts": [item.evidence_subset() for item in self.receipts],
            "implementable": self.plan.implementable,
            "runtime_model_calls": 0,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MaterializationResult":
        if not isinstance(payload, Mapping):
            raise StructuralMaterializeError("materialization result must be an object")
        plan_raw = payload.get("plan")
        if not isinstance(plan_raw, Mapping):
            raise StructuralMaterializeError("plan must be an object")
        packet_raw = payload.get("code_edit_packet")
        packet = (
            CodeEditPacket.from_dict(packet_raw)
            if isinstance(packet_raw, Mapping)
            else None
        )
        return cls(
            plan=SourceEditPlan.from_dict(plan_raw),
            receipts=tuple(
                SourceWriteReceipt.from_dict(item)
                for item in (payload.get("receipts") or ())
            ),
            disposition=payload.get("disposition") or MaterializeDisposition.REJECTED,
            passed=bool(payload.get("passed", False)),
            code_edit_packet=packet,
            notes=tuple(payload.get("notes") or ()),
            runtime_model_calls=int(payload.get("runtime_model_calls") or 0),
        )


# ---------------------------------------------------------------------------
# apply_operator + materializer
# ---------------------------------------------------------------------------


def apply_operator(
    operator: AdmittedSourceEditOperator | SourceEditSite | Mapping[str, Any],
    *,
    worktree_root: str | Path,
    plan_id: str = "",
    write: bool = True,
    receipt_dir: str | Path | None = None,
    force_receipt_failure: bool = False,
) -> SourceWriteReceipt:
    """Apply one admitted source-edit operator under ``worktree_root``.

    When ``write`` is False, computes the would-be after bytes without mutating
    the filesystem (preview).  Receipt write failures yield a nonpassing
    receipt even if the source bytes were updated.
    """

    if isinstance(operator, AdmittedSourceEditOperator):
        site = operator.site
        operator_id = operator.operator_id
        if not operator.grants_write_authority and write:
            raise StructuralMaterializeError("operator does not grant write authority")
    elif isinstance(operator, SourceEditSite):
        site = operator
        operator_id = site.operator_id
    elif isinstance(operator, Mapping):
        if "site" in operator:
            admitted = AdmittedSourceEditOperator.from_dict(operator)
            site = admitted.site
            operator_id = admitted.operator_id
        else:
            site = SourceEditSite.from_dict(operator)
            operator_id = site.operator_id
    else:
        raise StructuralMaterializeError("operator must be admitted or a source site")

    if site.is_catalog_only or not site.changes_bytes:
        return SourceWriteReceipt(
            path=site.path,
            before_hash=site.before_hash,
            after_hash=site.after_hash,
            before_bytes=len(site.old_span_text.encode("utf-8")),
            after_bytes=len(site.replacement_text.encode("utf-8")),
            ast_anchor=site.ast_anchor,
            operator_id=operator_id,
            operator_args=dict(site.operator_args),
            patch="",
            inverse_patch="",
            changed_source_bytes=False,
            reversible=False,
            written=False,
            receipt_written=False,
            disposition=(
                MaterializeDisposition.CATALOG_EVIDENCE_ONLY
                if site.is_catalog_only
                else MaterializeDisposition.NO_BYTE_CHANGE
            ),
            reason_codes=(
                MaterializeDisposition.CATALOG_EVIDENCE_ONLY.value
                if site.is_catalog_only
                else MaterializeDisposition.NO_BYTE_CHANGE.value,
            ),
            worktree_root=str(worktree_root),
            plan_id=plan_id,
        )

    root = Path(worktree_root)
    if not root.is_dir() or root.is_symlink():
        raise StructuralMaterializeError("worktree_root must be an existing directory")

    target = _resolve_under(root, site.path)
    if not target.is_file():
        raise StructuralMaterializeError(
            f"{MaterializeDisposition.STALE_SPAN.value}: target file missing: {site.path}"
        )
    try:
        before_text = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise StructuralMaterializeError(f"unable to read target file: {site.path}") from exc
    if len(before_text.encode("utf-8")) > MAX_FILE_BYTES:
        raise StructuralMaterializeError("target file exceeds single-file bound")

    after_text = _apply_span_to_text(before_text, site)
    before_hash = _sha256_text(before_text)
    after_hash = _sha256_text(after_text)
    changed = before_text != after_text
    patch = _reversible_patch(site.path, before_text, after_text) if changed else ""
    inverse = _inverse_patch_document(patch) if changed else ""
    reversible = bool(changed and patch and inverse and after_text != before_text)

    written = False
    if write and changed:
        _atomic_write_text(target, after_text)
        # Re-read to confirm durable bytes.
        reread = target.read_text(encoding="utf-8")
        if reread != after_text:
            raise StructuralMaterializeError("post-write reread mismatch")
        written = True

    receipt_written = False
    disposition = (
        MaterializeDisposition.APPLIED
        if written and changed and reversible
        else (
            MaterializeDisposition.PREVIEWED
            if (not write) and changed and reversible
            else MaterializeDisposition.NO_BYTE_CHANGE
            if not changed
            else MaterializeDisposition.NONPASSING
        )
    )
    reason_codes: list[str] = []

    # Write receipt next to the worktree (or explicit receipt_dir).
    if write and changed and reversible:
        try:
            if force_receipt_failure:
                raise OSError("forced receipt write failure")
            dest_root = Path(receipt_dir) if receipt_dir is not None else root
            if not dest_root.is_dir():
                dest_root.mkdir(parents=True, exist_ok=True)
            receipt_name = (
                f".dcr-write-receipt-{_sha256_text(site.path + site.ast_anchor)[7:23]}.json"
            )
            receipt_path = dest_root / receipt_name
            provisional = {
                "schema": SOURCE_WRITE_RECEIPT_SCHEMA,
                "path": site.path,
                "before_hash": before_hash,
                "after_hash": after_hash,
                "ast_anchor": site.ast_anchor,
                "operator_id": operator_id,
                "operator_args": dict(site.operator_args),
                "patch": patch,
                "inverse_patch": inverse,
                "plan_id": plan_id,
            }
            _atomic_write_text(
                receipt_path,
                json.dumps(provisional, sort_keys=True, indent=2) + "\n",
            )
            receipt_written = True
        except OSError:
            receipt_written = False
            disposition = MaterializeDisposition.RECEIPT_WRITE_FAILED
            reason_codes.append(MaterializeDisposition.RECEIPT_WRITE_FAILED.value)

    if written and changed and reversible and not receipt_written:
        disposition = MaterializeDisposition.RECEIPT_WRITE_FAILED
        if MaterializeDisposition.RECEIPT_WRITE_FAILED.value not in reason_codes:
            reason_codes.append(MaterializeDisposition.RECEIPT_WRITE_FAILED.value)

    return SourceWriteReceipt(
        path=site.path,
        before_hash=before_hash,
        after_hash=after_hash,
        before_bytes=len(before_text.encode("utf-8")),
        after_bytes=len(after_text.encode("utf-8")),
        ast_anchor=site.ast_anchor,
        operator_id=operator_id,
        operator_args=dict(site.operator_args),
        patch=patch,
        inverse_patch=inverse,
        changed_source_bytes=changed,
        reversible=reversible,
        written=written,
        receipt_written=receipt_written if write else False,
        disposition=disposition,
        reason_codes=tuple(reason_codes),
        worktree_root=str(root),
        plan_id=plan_id,
    )


def invert_operator(
    receipt: SourceWriteReceipt,
    *,
    worktree_root: str | Path,
) -> SourceWriteReceipt:
    """Restore prior bytes using the receipt's before snapshot identity.

    Inverse re-applies the old file bytes from the inverse patch metadata by
    re-reading current after-hash and writing the inverse of the recorded
    replacement when the current file still matches ``after_hash``.
    """

    if not isinstance(receipt, SourceWriteReceipt):
        raise StructuralMaterializeError("receipt must be a SourceWriteReceipt")
    if not receipt.reversible or not receipt.patch:
        raise StructuralMaterializeError("receipt is not reversible")

    root = Path(worktree_root)
    target = _resolve_under(root, receipt.path)
    current = target.read_text(encoding="utf-8")
    if _sha256_text(current) != receipt.after_hash:
        raise StructuralMaterializeError("inverse requires the post-apply file hash")

    before_text = _before_text_from_patch(receipt.patch)
    if _sha256_text(before_text) != receipt.before_hash:
        raise StructuralMaterializeError("inverse reconstruction hash mismatch")

    site = make_source_edit_site(
        path=receipt.path,
        old_span_text=current,
        replacement_text=before_text,
        ast_anchor=receipt.ast_anchor,
        start_offset=0,
        operator_id=receipt.operator_id or "inverse:restore_old_span",
        operator_args={"inverse": True},
    )
    admitted = AdmittedSourceEditOperator(
        operator_id=site.operator_id,
        site=site,
        admission_cid="inverse",
        grants_write_authority=True,
    )
    return apply_operator(
        admitted,
        worktree_root=root,
        plan_id=receipt.plan_id,
        write=True,
    )


class StructuralRepairMaterializer:
    """Materialize structural source edits under an admitted owner worktree."""

    INTERFACE: ClassVar[str] = STRUCTURAL_REPAIR_MATERIALIZER_INTERFACE

    def __init__(
        self,
        *,
        worktree_root: str | Path,
        receipt_dir: str | Path | None = None,
    ) -> None:
        root = Path(worktree_root)
        if not root.is_dir() or root.is_symlink():
            raise StructuralMaterializeError(
                "worktree_root must be an existing non-symlink directory"
            )
        self.worktree_root = root
        self.receipt_dir = Path(receipt_dir) if receipt_dir is not None else None

    def preview(self, plan: SourceEditPlan) -> MaterializationResult:
        """Preview without writing; still nonpassing until applied with receipt."""

        return self._run(plan, write=False)

    def apply(
        self,
        plan: SourceEditPlan,
        *,
        force_receipt_failure: bool = False,
    ) -> MaterializationResult:
        """Apply all implementable sites; fail closed on non-mutating plans."""

        return self._run(
            plan,
            write=True,
            force_receipt_failure=force_receipt_failure,
        )

    def materialize(
        self,
        plan: SourceEditPlan | Mapping[str, Any],
        *,
        write: bool = True,
        force_receipt_failure: bool = False,
    ) -> MaterializationResult:
        if isinstance(plan, Mapping):
            plan = SourceEditPlan.from_dict(plan)
        if not isinstance(plan, SourceEditPlan):
            raise StructuralMaterializeError("plan must be a SourceEditPlan")
        return self._run(
            plan,
            write=write,
            force_receipt_failure=force_receipt_failure,
        )

    def _nonpassing_for_plan(
        self,
        plan: SourceEditPlan,
        *,
        disposition: MaterializeDisposition,
        notes: Sequence[str] = (),
    ) -> MaterializationResult:
        return MaterializationResult(
            plan=plan,
            receipts=(),
            disposition=disposition,
            passed=False,
            code_edit_packet=None,
            notes=tuple(notes) + tuple(plan.reason_codes),
            runtime_model_calls=0,
        )

    def _run(
        self,
        plan: SourceEditPlan,
        *,
        write: bool,
        force_receipt_failure: bool = False,
    ) -> MaterializationResult:
        # Gate non-mutating dispositions first.
        if plan.disposition is SourceEditPlanDisposition.ANALYSIS_ONLY:
            return self._nonpassing_for_plan(
                plan,
                disposition=MaterializeDisposition.ANALYSIS_ONLY,
                notes=("analysis_only_rows_are_nonpassing",),
            )
        if plan.disposition is SourceEditPlanDisposition.MISSING_SURFACE:
            return self._nonpassing_for_plan(
                plan,
                disposition=MaterializeDisposition.MISSING_SURFACE,
                notes=("missing_surface_rows_are_nonpassing",),
            )
        if plan.disposition is SourceEditPlanDisposition.IDL_GAP:
            return self._nonpassing_for_plan(
                plan,
                disposition=MaterializeDisposition.IDL_GAP,
                notes=("idl_gap_rows_are_nonpassing",),
            )
        if plan.disposition in {
            SourceEditPlanDisposition.CATALOG_EVIDENCE_ONLY,
            SourceEditPlanDisposition.NON_IMPLEMENTABLE,
            SourceEditPlanDisposition.BLOCKED,
        }:
            return self._nonpassing_for_plan(
                plan,
                disposition=MaterializeDisposition.CATALOG_EVIDENCE_ONLY
                if plan.disposition is SourceEditPlanDisposition.CATALOG_EVIDENCE_ONLY
                else MaterializeDisposition.NON_IMPLEMENTABLE,
                notes=("catalog_bindings_are_evidence_never_mutation_success",),
            )
        if not plan.implementable or not plan.sites:
            return self._nonpassing_for_plan(
                plan,
                disposition=MaterializeDisposition.NON_IMPLEMENTABLE,
                notes=("plan_not_implementable",),
            )

        receipts: list[SourceWriteReceipt] = []
        for site in plan.sites:
            if site.is_catalog_only:
                receipts.append(
                    SourceWriteReceipt(
                        path=site.path,
                        before_hash=site.before_hash,
                        after_hash=site.after_hash,
                        before_bytes=len(site.old_span_text.encode("utf-8")),
                        after_bytes=len(site.replacement_text.encode("utf-8")),
                        ast_anchor=site.ast_anchor,
                        operator_id=site.operator_id,
                        operator_args=dict(site.operator_args),
                        patch="",
                        inverse_patch="",
                        changed_source_bytes=False,
                        reversible=False,
                        written=False,
                        receipt_written=False,
                        disposition=MaterializeDisposition.CATALOG_EVIDENCE_ONLY,
                        reason_codes=(
                            MaterializeDisposition.CATALOG_EVIDENCE_ONLY.value,
                        ),
                        worktree_root=str(self.worktree_root),
                        plan_id=plan.plan_id,
                    )
                )
                continue
            admitted = AdmittedSourceEditOperator(
                operator_id=site.operator_id or "dcr-operator:structural_source_edit@1",
                site=site,
                admission_cid=plan.admission_cid or plan.plan_id,
                packet_cid=plan.packet_cid,
                grants_write_authority=True,
            )
            receipts.append(
                apply_operator(
                    admitted,
                    worktree_root=self.worktree_root,
                    plan_id=plan.plan_id,
                    write=write,
                    receipt_dir=self.receipt_dir,
                    force_receipt_failure=force_receipt_failure,
                )
            )

        all_passed = bool(receipts) and all(item.passed for item in receipts)
        if force_receipt_failure or any(
            item.disposition is MaterializeDisposition.RECEIPT_WRITE_FAILED
            for item in receipts
        ):
            disposition = MaterializeDisposition.RECEIPT_WRITE_FAILED
            all_passed = False
        elif all_passed:
            disposition = MaterializeDisposition.APPLIED
        elif write:
            disposition = MaterializeDisposition.NONPASSING
        else:
            # Preview of a valid plan is not a successful mutation.
            disposition = MaterializeDisposition.PREVIEWED
            all_passed = False

        packet: CodeEditPacket | None = None
        if plan.implementable:
            try:
                packet = build_code_edit_packet(
                    repository_tree_id=plan.worktree_root or str(self.worktree_root),
                    predicted_files=tuple(sorted({site.path for site in plan.sites})),
                    task_id=plan.work_id or plan.plan_id,
                    metadata={
                        "materializer": STRUCTURAL_REPAIR_MATERIALIZER_INTERFACE,
                        "plan_id": plan.plan_id,
                        "source_edit_plan": SOURCE_EDIT_PLAN_INTERFACE,
                    },
                )
            except Exception:
                packet = None

        return MaterializationResult(
            plan=plan,
            receipts=tuple(receipts),
            disposition=disposition,
            passed=all_passed,
            code_edit_packet=packet,
            notes=(),
            runtime_model_calls=0,
        )


def materialize_source_edit_plan(
    plan: SourceEditPlan | Mapping[str, Any],
    *,
    worktree_root: str | Path,
    write: bool = True,
    receipt_dir: str | Path | None = None,
    force_receipt_failure: bool = False,
) -> MaterializationResult:
    """Convenience wrapper around :class:`StructuralRepairMaterializer`."""

    materializer = StructuralRepairMaterializer(
        worktree_root=worktree_root,
        receipt_dir=receipt_dir,
    )
    return materializer.materialize(
        plan,
        write=write,
        force_receipt_failure=force_receipt_failure,
    )


def materialize_materialization_vectors(
    cases: Sequence[Mapping[str, Any]] | None = None,
    *,
    destination: str | Path | None = None,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build a compact materialization-vector catalog (evidence only)."""

    vectors: list[dict[str, Any]] = []
    for case in cases or ():
        case_id = str(case.get("case_id") or case.get("id") or "case")
        result_payload = case.get("result")
        if isinstance(result_payload, MaterializationResult):
            result = result_payload
        elif isinstance(result_payload, Mapping):
            result = MaterializationResult.from_dict(result_payload)
        else:
            # Encode expected disposition without requiring a full apply.
            expected_pass = bool(case.get("passed", False))
            disposition = str(
                case.get("disposition")
                or (
                    MaterializeDisposition.APPLIED.value
                    if expected_pass
                    else MaterializeDisposition.NONPASSING.value
                )
            )
            vectors.append(
                {
                    "case_id": case_id,
                    "passed": expected_pass,
                    "disposition": disposition,
                    "reason_codes": list(case.get("reason_codes") or ()),
                    "evidence_subset": dict(case.get("evidence_subset") or {}),
                }
            )
            continue
        vectors.append(
            {
                "case_id": case_id,
                "passed": result.passed,
                "disposition": result.disposition.value,
                "plan_id": result.plan.plan_id,
                "reason_codes": list(result.plan.reason_codes),
                "evidence_subset": result.evidence_subset(),
            }
        )

    catalog = {
        "schema": MATERIALIZATION_VECTORS_SCHEMA,
        "evidence_id": DCR_MATERIALIZATION_EVIDENCE,
        "interface": STRUCTURAL_REPAIR_MATERIALIZER_INTERFACE,
        "code_edit_packet_interface": CODE_EDIT_PACKET_INTERFACE,
        "version": MATERIALIZER_VERSION,
        "runtime_model_calls": 0,
        "vectors": vectors,
        "acceptance": {
            "success_requires_changed_source_bytes": True,
            "success_requires_reversible_diff": True,
            "analysis_only_nonpassing": True,
            "missing_surface_nonpassing": True,
            "idl_gap_nonpassing": True,
            "catalog_bindings_never_mutation_success": True,
            "receipt_write_failure_nonpassing": True,
        },
    }

    if destination is not None or repository_root is not None:
        root = Path(repository_root) if repository_root is not None else Path(".")
        path = (
            Path(destination)
            if destination is not None
            else root / DEFAULT_MATERIALIZATION_VECTORS_REL
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(catalog, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        catalog = {**catalog, "path": path.as_posix()}

    return catalog


__all__ = [
    "ADMITTED_SOURCE_EDIT_OPERATOR_INTERFACE",
    "CODE_EDIT_PACKET_INTERFACE",
    "DEFAULT_MATERIALIZATION_VECTORS_REL",
    "DCR_MATERIALIZATION_EVIDENCE",
    "MATERIALIZATION_RESULT_SCHEMA",
    "MATERIALIZATION_VECTORS_SCHEMA",
    "SOURCE_WRITE_RECEIPT_SCHEMA",
    "STRUCTURAL_REPAIR_MATERIALIZER_INTERFACE",
    "AdmittedSourceEditOperator",
    "CodeEditPacket",
    "MaterializationResult",
    "MaterializeDisposition",
    "SourceWriteReceipt",
    "StructuralMaterializeError",
    "StructuralRepairMaterializer",
    "apply_operator",
    "build_source_edit_plan",
    "invert_operator",
    "is_non_mutating_disposition",
    "make_source_edit_site",
    "materialize_materialization_vectors",
    "materialize_source_edit_plan",
]
