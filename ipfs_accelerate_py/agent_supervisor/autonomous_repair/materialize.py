"""Gated materializer for admitted body-free edit plans (no LLM).

Applies **only** plans with ``materialize_ready=true`` and an externally
admitted exact source-edit operator after re-validation:

1. Re-resolve MCP surfaces (must remain single-path resolved)
2. Verify preferred_path exists and contains registration/handler evidence
3. Verify exact old/new bytes, reversible diffs, and owner-root binding
4. Apply the admitted byte replacement with validation pending
5. Emit materialize receipts (never board-authoritative, never KERNEL_VERIFIED)

Does not invent source bodies or call doctor ``render_admitted`` without
``proof_admitted`` + exact span hashes.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from .interface_alias_registry import (
    InterfaceAliasRegistry,
    default_mcp_idl_alias_registry,
)
from .mcp_surface_resolution import resolve_mcp_surfaces

MATERIALIZE_INTERFACE: Final = "AutonomousRepairMaterializer@1"
MATERIALIZE_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/autonomous-repair-materialize-receipt@1"
)
MATERIALIZE_BATCH_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/autonomous-repair-materialize-batch@1"
)
SURFACE_BINDING_CATALOG_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/surface-identity-bindings@1"
)

# Default package-local catalog (created only when write_package_bindings=True)
DEFAULT_PACKAGE_BINDINGS_REL: Final = (
    "external/ipfs_accelerate/ipfs_accelerate_py/mcp_server/surface_identity_bindings.json"
)


class AdmittedSourceEditError(ValueError):
    """A proposed source edit lacks the evidence needed to mutate bytes."""


@dataclass(frozen=True)
class AdmittedSourceEditOperator:
    """One exact, reversible, owner-bound byte replacement.

    This narrow operator is intentionally the only mutable path in this early
    DCR-071/072 slice.  Catalog/IDL/analysis records cannot be coerced into it.
    """

    operator_id: str
    owner_root: str
    relative_path: str
    old_digest: str
    new_digest: str
    old_bytes_b64: str
    new_bytes_b64: str
    forward_diff: str
    inverse_diff: str
    disposition: str
    admitted: bool
    kind: str = "replace_exact_bytes"

    @classmethod
    def from_mapping(cls, raw: Any) -> AdmittedSourceEditOperator:
        if not isinstance(raw, Mapping):
            raise AdmittedSourceEditError("source_edit_operator_missing")
        fields = {
            "operator_id",
            "owner_root",
            "relative_path",
            "old_digest",
            "new_digest",
            "old_bytes_b64",
            "new_bytes_b64",
            "forward_diff",
            "inverse_diff",
            "disposition",
            "admitted",
            "kind",
        }
        unknown = set(raw) - fields
        if unknown:
            raise AdmittedSourceEditError("source_edit_operator_unknown_fields")
        try:
            operator = cls(
                operator_id=str(raw["operator_id"]),
                owner_root=str(raw["owner_root"]),
                relative_path=str(raw["relative_path"]),
                old_digest=str(raw["old_digest"]),
                new_digest=str(raw["new_digest"]),
                old_bytes_b64=str(raw["old_bytes_b64"]),
                new_bytes_b64=str(raw["new_bytes_b64"]),
                forward_diff=str(raw["forward_diff"]),
                inverse_diff=str(raw["inverse_diff"]),
                disposition=str(raw["disposition"]),
                admitted=raw["admitted"] is True,
                kind=str(raw.get("kind") or "replace_exact_bytes"),
            )
        except KeyError as exc:
            raise AdmittedSourceEditError("source_edit_operator_incomplete") from exc
        if (
            not operator.operator_id.strip()
            or not operator.owner_root.strip()
            or not operator.relative_path.strip()
            or operator.kind != "replace_exact_bytes"
            or not operator.admitted
            or operator.disposition != "validation_pending"
            or not operator.forward_diff.strip()
            or not operator.inverse_diff.strip()
        ):
            raise AdmittedSourceEditError("source_edit_operator_not_admitted")
        return operator

    @property
    def old_bytes(self) -> bytes:
        try:
            return base64.b64decode(self.old_bytes_b64, validate=True)
        except (ValueError, TypeError) as exc:
            raise AdmittedSourceEditError("source_edit_old_bytes_invalid") from exc

    @property
    def new_bytes(self) -> bytes:
        try:
            return base64.b64decode(self.new_bytes_b64, validate=True)
        except (ValueError, TypeError) as exc:
            raise AdmittedSourceEditError("source_edit_new_bytes_invalid") from exc

    def validate(self, *, repo_root: Path, preferred_path: str) -> tuple[Path, bytes, bytes]:
        """Validate exact ownership, bytes, and reversible diff binding; no write."""
        if self.owner_root != str(repo_root):
            raise AdmittedSourceEditError("source_edit_owner_root_mismatch")
        relative = Path(self.relative_path)
        if relative.is_absolute() or ".." in relative.parts or self.relative_path != preferred_path:
            raise AdmittedSourceEditError("source_edit_path_binding_mismatch")
        target = (repo_root / relative).resolve()
        if target.parent != repo_root and repo_root not in target.parents:
            raise AdmittedSourceEditError("source_edit_path_outside_owner_root")
        if not target.is_file():
            raise AdmittedSourceEditError("source_edit_target_missing")
        old_bytes, new_bytes = self.old_bytes, self.new_bytes
        old_digest = "sha256:" + hashlib.sha256(old_bytes).hexdigest()
        new_digest = "sha256:" + hashlib.sha256(new_bytes).hexdigest()
        if (
            old_digest != self.old_digest
            or new_digest != self.new_digest
            or old_bytes == new_bytes
            or target.read_bytes() != old_bytes
        ):
            raise AdmittedSourceEditError("source_edit_byte_digest_mismatch")
        if (
            self.old_digest not in self.forward_diff
            or self.new_digest not in self.forward_diff
            or self.old_digest not in self.inverse_diff
            or self.new_digest not in self.inverse_diff
        ):
            raise AdmittedSourceEditError("source_edit_diff_inverse_unbound")
        return target, old_bytes, new_bytes


@dataclass
class MaterializePolicy:
    """Fail-closed materialize gates."""

    require_materialize_ready: bool = True
    require_single_path: bool = True
    require_path_exists: bool = True
    require_handler_evidence: bool = True
    write_data_catalog: bool = True
    write_package_bindings: bool = False
    domain: str = "agent_supervisor"
    dry_run: bool = False

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> MaterializePolicy:
        raw = dict(raw or {})
        return cls(
            require_materialize_ready=bool(
                raw.get(
                    "requireMaterializeReady",
                    raw.get("require_materialize_ready", True),
                )
            ),
            require_single_path=bool(
                raw.get("requireSinglePath", raw.get("require_single_path", True))
            ),
            require_path_exists=bool(
                raw.get("requirePathExists", raw.get("require_path_exists", True))
            ),
            require_handler_evidence=bool(
                raw.get(
                    "requireHandlerEvidence",
                    raw.get("require_handler_evidence", True),
                )
            ),
            write_data_catalog=bool(
                raw.get("writeDataCatalog", raw.get("write_data_catalog", True))
            ),
            write_package_bindings=bool(
                raw.get(
                    "writePackageBindings",
                    raw.get("write_package_bindings", False),
                )
            ),
            domain=str(raw.get("domain") or "agent_supervisor"),
            dry_run=bool(raw.get("dryRun", raw.get("dry_run", False))),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MaterializeReceipt:
    plan_id: str
    work_id: str
    operation: str
    status: str  # applied | rejected | skipped | dry_run
    reasons: list[str] = field(default_factory=list)
    preferred_path: str = ""
    handler: str | None = None
    binding_id: str = ""
    files_written: list[str] = field(default_factory=list)
    revalidated_surface: dict[str, Any] = field(default_factory=dict)
    source_edit_disposition: str = "not_source_edit"
    source_edit_operator_id: str = ""
    old_digest: str = ""
    new_digest: str = ""
    mutation_applied: bool = False
    validation_pending: bool = False
    completion_authoritative: bool = False
    grants_execution_authority: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": MATERIALIZE_RECEIPT_SCHEMA,
            "interface": MATERIALIZE_INTERFACE,
            **asdict(self),
        }


def load_edit_plans(
    edit_plan_dir: str | Path,
    *,
    materialize_ready_only: bool = True,
) -> list[dict[str, Any]]:
    """Load admitted edit plan JSON objects from a directory."""
    root = Path(edit_plan_dir)
    plans: list[dict[str, Any]] = []
    index = root / "index.json"
    paths: list[Path] = []
    if index.is_file():
        doc = json.loads(index.read_text(encoding="utf-8"))
        for item in doc.get("plans") or []:
            if materialize_ready_only and not item.get("materialize_ready"):
                continue
            p = Path(str(item.get("path") or ""))
            if p.is_file():
                paths.append(p)
    else:
        paths = sorted(root.glob("edit-plan_*.json"))
    for path in paths:
        doc = json.loads(path.read_text(encoding="utf-8"))
        if materialize_ready_only and not doc.get("materialize_ready"):
            continue
        doc["_plan_file"] = str(path)
        plans.append(doc)
    return plans


def _verify_handler_evidence(
    path: Path,
    *,
    operation: str,
    handler: str | None,
) -> tuple[bool, list[str]]:
    """Confirm preferred path still mentions the op/handler registration."""
    reasons: list[str] = []
    if not path.is_file():
        return False, ["preferred_path_missing"]
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:  # noqa: BLE001
        return False, [f"path_read_error:{type(exc).__name__}"]

    op = operation or ""
    leaf = op.rsplit(".", 1)[-1] if op else ""
    tokens = [t for t in (op, leaf, handler or "") if t]
    # registration cues
    reg_cues = ("register_tool", "manager.register_tool", "server.register_tool")
    has_reg = any(c in text for c in reg_cues)
    if not has_reg:
        reasons.append("no_register_tool_evidence")

    found = []
    for tok in tokens:
        # word-ish match
        if tok in text or re.search(rf"['\"]{re.escape(tok)}['\"]", text):
            found.append(tok)
    if not found:
        reasons.append("operation_or_handler_not_found_in_path")
        return False, reasons
    if not has_reg:
        return False, reasons
    return True, [f"evidence_tokens:{','.join(found)}"]


def _binding_record(plan: Mapping[str, Any], surface: Mapping[str, Any]) -> dict[str, Any]:
    op = str(plan.get("operation") or "")
    path = str(plan.get("preferred_path") or "")
    handler = plan.get("handler")
    digest = hashlib.sha256(
        json.dumps(
            {
                "op": op,
                "path": path,
                "handler": handler,
                "aliases": list(plan.get("aliases") or []),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return {
        "binding_id": f"binding:{digest[:16]}",
        "operation": op,
        "aliases": list(plan.get("aliases") or []),
        "idl_methods": list(plan.get("idl_methods") or []),
        "preferred_path": path,
        "handler": handler,
        "registration_api": plan.get("registration_api"),
        "canonical_surface": surface.get("canonical"),
        "match_count": surface.get("match_count"),
        "doctor_operator": plan.get("doctor_operator"),
        "plan_id": plan.get("plan_id"),
        "work_id": plan.get("work_id"),
        "domain": plan.get("domain") or "agent_supervisor",
        "mediation": "package_mcp_interop|/mcp/tools/call|tools_dispatch",
        "authoritative": False,
        "body_free": True,
        "recorded_at": datetime.now(UTC).isoformat(),
    }


def _merge_catalog(path: Path, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    existing: dict[str, Any] = {"bindings": {}}
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            existing = {"bindings": {}}
    bindings = dict(existing.get("bindings") or {})
    for rec in records:
        op = str(rec.get("operation") or "")
        if not op:
            continue
        # Prefer newer binding_id by recorded_at
        prev = bindings.get(op)
        if prev and prev.get("binding_id") == rec.get("binding_id"):
            bindings[op] = dict(rec)
        else:
            bindings[op] = dict(rec)
    catalog = {
        "schema": SURFACE_BINDING_CATALOG_SCHEMA,
        "interface": MATERIALIZE_INTERFACE,
        "updated_at": datetime.now(UTC).isoformat(),
        "binding_count": len(bindings),
        "completion_authoritative": False,
        "bindings": bindings,
        "notes": [
            "Deterministic surface identity bindings from autonomous repair materializer.",
            "Not KERNEL_VERIFIED; consumers must re-proof before authoritative completion.",
            "Prefer preferred_path + handler for GUI/ORB/IDL mediation.",
        ],
    }
    return catalog


class AutonomousRepairMaterializer:
    """Apply materialize_ready edit plans under fail-closed gates."""

    def __init__(
        self,
        *,
        repo_root: str | Path,
        policy: MaterializePolicy | Mapping[str, Any] | None = None,
        alias_registry: InterfaceAliasRegistry | None = None,
        surface_files: Sequence[tuple[str, str | Path]] | None = None,
        data_catalog_path: str | Path | None = None,
        package_bindings_path: str | Path | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.policy = (
            MaterializePolicy.from_mapping(policy)
            if isinstance(policy, Mapping) or policy is None
            else policy
        )
        self.alias_registry = alias_registry or default_mcp_idl_alias_registry()
        self.surface_files = surface_files
        self.data_catalog_path = Path(
            data_catalog_path
            or (
                self.repo_root
                / "data"
                / "agent_supervisor"
                / "autonomous_repair"
                / "bindings"
                / "surface_identity_bindings.json"
            )
        )
        self.package_bindings_path = Path(
            package_bindings_path or (self.repo_root / DEFAULT_PACKAGE_BINDINGS_REL)
        )

    def materialize_plans(
        self,
        plans: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        policy = self.policy
        ops = [str(p.get("operation") or "") for p in plans if p.get("operation")]
        surfaces = resolve_mcp_surfaces(
            ops,
            repo_root=self.repo_root,
            surface_files=self.surface_files,
            alias_registry=self.alias_registry,
            prefer_mcp_server=True,
        )
        by_op = surfaces.by_operation()

        receipts: list[MaterializeReceipt] = []
        applied_bindings: list[dict[str, Any]] = []

        for plan in plans:
            op = str(plan.get("operation") or "")
            plan_id = str(plan.get("plan_id") or "")
            work_id = str(plan.get("work_id") or "")
            reasons: list[str] = []
            surface = by_op.get(op)
            surface_d = surface.to_dict() if surface else {}

            if policy.require_materialize_ready and not plan.get("materialize_ready"):
                receipts.append(
                    MaterializeReceipt(
                        plan_id=plan_id,
                        work_id=work_id,
                        operation=op,
                        status="rejected",
                        reasons=["not_materialize_ready"],
                        revalidated_surface=surface_d,
                    )
                )
                continue

            if policy.require_single_path:
                effective = getattr(surface, "effective_match_count", None) if surface else None
                if effective is None and surface is not None:
                    effective = surface.match_count
                collapsed = bool(getattr(surface, "collapsed", False)) if surface else False
                ok_single = bool(
                    surface and surface.status == "resolved" and (effective == 1 or collapsed)
                )
                if not ok_single:
                    receipts.append(
                        MaterializeReceipt(
                            plan_id=plan_id,
                            work_id=work_id,
                            operation=op,
                            status="rejected",
                            reasons=[
                                "revalidate_not_single_path",
                                f"status={getattr(surface, 'status', None)}",
                                f"match_count={getattr(surface, 'match_count', None)}",
                                f"effective={effective}",
                                f"collapsed={collapsed}",
                            ],
                            revalidated_surface=surface_d,
                        )
                    )
                    continue

            preferred = str(
                plan.get("preferred_path") or (surface.preferred_path if surface else "") or ""
            )
            # Prefer revalidated preferred_path if present
            if surface and surface.preferred_path:
                preferred = surface.preferred_path

            path_obj = (
                (
                    (self.repo_root / preferred).resolve()
                    if preferred and not Path(preferred).is_absolute()
                    else Path(preferred)
                )
                if preferred
                else None
            )

            if policy.require_path_exists:
                if path_obj is None or not path_obj.is_file():
                    receipts.append(
                        MaterializeReceipt(
                            plan_id=plan_id,
                            work_id=work_id,
                            operation=op,
                            status="rejected",
                            reasons=["preferred_path_missing"],
                            preferred_path=preferred,
                            revalidated_surface=surface_d,
                        )
                    )
                    continue

            handler = plan.get("handler") or (surface.handler if surface else None)
            if policy.require_handler_evidence and path_obj is not None:
                ok, ev = _verify_handler_evidence(path_obj, operation=op, handler=handler)
                if not ok:
                    receipts.append(
                        MaterializeReceipt(
                            plan_id=plan_id,
                            work_id=work_id,
                            operation=op,
                            status="rejected",
                            reasons=list(ev),
                            preferred_path=preferred,
                            handler=handler,
                            revalidated_surface=surface_d,
                        )
                    )
                    continue
                reasons.extend(ev)

            try:
                source_edit = AdmittedSourceEditOperator.from_mapping(
                    plan.get("source_edit_operator")
                )
                if path_obj is None:
                    raise AdmittedSourceEditError("source_edit_target_missing")
                target, _old_bytes, new_bytes = source_edit.validate(
                    repo_root=self.repo_root,
                    preferred_path=preferred,
                )
            except AdmittedSourceEditError as exc:
                receipts.append(
                    MaterializeReceipt(
                        plan_id=plan_id,
                        work_id=work_id,
                        operation=op,
                        status="rejected",
                        reasons=[str(exc)],
                        preferred_path=preferred,
                        handler=handler,
                        revalidated_surface=surface_d,
                        source_edit_disposition="not_admitted",
                    )
                )
                continue

            if not policy.dry_run:
                # The only mutation path is an exact admitted source operator.
                target.write_bytes(new_bytes)
            receipts.append(
                MaterializeReceipt(
                    plan_id=plan_id,
                    work_id=work_id,
                    operation=op,
                    status=(
                        "source_edit_validation_pending"
                        if not policy.dry_run
                        else "source_edit_dry_run_validation_pending"
                    ),
                    reasons=reasons or ["admitted_source_edit_applied"],
                    preferred_path=preferred,
                    handler=handler,
                    revalidated_surface=surface_d,
                    source_edit_disposition="validation_pending",
                    source_edit_operator_id=source_edit.operator_id,
                    old_digest=source_edit.old_digest,
                    new_digest=source_edit.new_digest,
                    mutation_applied=not policy.dry_run,
                    validation_pending=True,
                )
            )

        files_written_global: list[str] = []
        if applied_bindings and not policy.dry_run:
            if policy.write_data_catalog:
                catalog = _merge_catalog(self.data_catalog_path, applied_bindings)
                self.data_catalog_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                self.data_catalog_path.write_text(
                    json.dumps(catalog, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                files_written_global.append(str(self.data_catalog_path))
                # domain-scoped copy
                domain_path = (
                    self.data_catalog_path.parent
                    / f"surface_identity_bindings.{policy.domain}.json"
                )
                domain_path.write_text(
                    json.dumps(catalog, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                files_written_global.append(str(domain_path))

            if policy.write_package_bindings:
                catalog = _merge_catalog(self.package_bindings_path, applied_bindings)
                self.package_bindings_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                self.package_bindings_path.write_text(
                    json.dumps(catalog, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                files_written_global.append(str(self.package_bindings_path))

            # annotate receipts with written files
            for rec in receipts:
                if rec.status == "applied":
                    rec.files_written = list(files_written_global)

        applied = sum(1 for r in receipts if r.mutation_applied)
        rejected = sum(1 for r in receipts if r.status == "rejected")

        report = {
            "schema": MATERIALIZE_BATCH_SCHEMA,
            "interface": MATERIALIZE_INTERFACE,
            "recorded_at": datetime.now(UTC).isoformat(),
            # A byte mutation remains non-passing until independent validation.
            "passed": False,
            "llm_used": False,
            "model_call_count": 0,
            "completion_authoritative": False,
            "policy": policy.to_dict(),
            "summary": {
                "plan_count": len(plans),
                "applied": applied,
                "rejected": rejected,
                "bindings": 0,
                "validation_pending": sum(1 for r in receipts if r.validation_pending),
                "files_written": files_written_global,
            },
            "surface_revalidation": surfaces.to_dict(),
            "receipts": [r.to_dict() for r in receipts],
            "notes": [
                "Catalog/IDL/receipt-only records never count as applied source edits.",
                "Only an admitted exact byte replacement can mutate a source path.",
                "All source edits remain validation-pending and non-passing.",
                "Board completion and KERNEL_VERIFIED remain external re-proof concerns.",
            ],
        }
        return report


def materialize_edit_plan_dir(
    edit_plan_dir: str | Path,
    *,
    repo_root: str | Path,
    policy: MaterializePolicy | Mapping[str, Any] | None = None,
    alias_registry: InterfaceAliasRegistry | None = None,
    materialize_ready_only: bool = True,
) -> dict[str, Any]:
    """Load plans from disk and materialize under gates."""
    plans = load_edit_plans(edit_plan_dir, materialize_ready_only=materialize_ready_only)
    mat = AutonomousRepairMaterializer(
        repo_root=repo_root,
        policy=policy,
        alias_registry=alias_registry,
    )
    report = mat.materialize_plans(plans)
    # persist batch receipt next to plans
    out = Path(edit_plan_dir) / "materialize_receipt.json"
    if not (isinstance(policy, MaterializePolicy) and policy.dry_run) and not (
        isinstance(policy, Mapping) and policy.get("dry_run")
    ):
        try:
            pol = (
                policy
                if isinstance(policy, MaterializePolicy)
                else MaterializePolicy.from_mapping(policy)
            )
            if not pol.dry_run:
                out.write_text(
                    json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
                    encoding="utf-8",
                )
                report["summary"]["receipt_path"] = str(out)
        except Exception:  # noqa: BLE001
            pass
    return report


__all__ = [
    "DEFAULT_PACKAGE_BINDINGS_REL",
    "MATERIALIZE_BATCH_SCHEMA",
    "MATERIALIZE_INTERFACE",
    "MATERIALIZE_RECEIPT_SCHEMA",
    "SURFACE_BINDING_CATALOG_SCHEMA",
    "AdmittedSourceEditError",
    "AdmittedSourceEditOperator",
    "AutonomousRepairMaterializer",
    "MaterializePolicy",
    "MaterializeReceipt",
    "load_edit_plans",
    "materialize_edit_plan_dir",
]
