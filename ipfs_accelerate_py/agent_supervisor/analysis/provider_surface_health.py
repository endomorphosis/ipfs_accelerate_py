"""DCR-013 static provider registration and handler-surface reconciliation.

Only source bytes and Python AST are read.  Expected descriptors are compared
to observed registrations but can never create an observed registration.
"""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

PROVIDER_SURFACE_HEALTH_INTERFACE: Final[str] = "ProviderSurfaceHealth@1"
PROVIDER_SURFACE_HEALTH_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-repair-provider-surfaces@1"
)


class SurfaceStatus(StrEnum):
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    AMBIGUOUS = "ambiguous"
    PARSER_FAILURE = "parser_failure"


def _cid(value: Any, prefix: str) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return f"{prefix}:sha256:{hashlib.sha256(encoded).hexdigest()}"


def _literal(node: ast.AST) -> str:
    return (
        node.value.strip() if isinstance(node, ast.Constant) and isinstance(node.value, str) else ""
    )


def _name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_name(node.value)}.{node.attr}".lstrip(".")
    return ""


def _kwargs(node: ast.Call) -> dict[str, str]:
    return {
        item.arg: (_literal(item.value) or _name(item.value)) for item in node.keywords if item.arg
    }


@dataclass(frozen=True)
class ProviderSurfaceExpectation:
    operation: str
    mandatory: bool = True


@dataclass(frozen=True)
class ProviderSurfaceRow:
    package_root: str
    source_path: str
    line: int
    operation: str
    aliases: tuple[str, ...]
    dispatcher: str
    handler: str
    effect: str
    request_schema: str
    result_schema: str
    error_schema: str
    status: SurfaceStatus
    reason: str
    source_digest: str

    def anchor(self) -> tuple[str, str, str]:
        return self.operation, self.dispatcher, self.handler

    def to_dict(self) -> dict[str, Any]:
        return {
            "package_root": self.package_root,
            "source_path": self.source_path,
            "line": self.line,
            "operation": self.operation,
            "aliases": list(self.aliases),
            "dispatcher": self.dispatcher,
            "handler": self.handler,
            "effect": self.effect,
            "request_schema": self.request_schema,
            "result_schema": self.result_schema,
            "error_schema": self.error_schema,
            "status": self.status.value,
            "reason": self.reason,
            "source_digest": self.source_digest,
        }


@dataclass(frozen=True)
class ProviderSurfaceHealth:
    forest_identity: str
    index_identity: str
    rows: tuple[ProviderSurfaceRow, ...]
    mandatory_unresolved: tuple[str, ...]

    @property
    def receipt_id(self) -> str:
        return _cid(self.to_dict(include_receipt=False), "provider-surfaces")

    @property
    def parity_ready(self) -> bool:
        return not self.mandatory_unresolved and all(
            row.status is SurfaceStatus.RESOLVED for row in self.rows
        )

    def to_dict(self, *, include_receipt: bool = True) -> dict[str, Any]:
        payload = {
            "schema": PROVIDER_SURFACE_HEALTH_SCHEMA,
            "interface": PROVIDER_SURFACE_HEALTH_INTERFACE,
            "forest_identity": self.forest_identity,
            "index_identity": self.index_identity,
            "rows": [row.to_dict() for row in self.rows],
            "mandatory_unresolved": list(self.mandatory_unresolved),
            "parity_ready": self.parity_ready,
            "completion_authoritative": False,
            "provider_or_llm_invoked": False,
        }
        if include_receipt:
            payload["receipt_id"] = self.receipt_id
        return payload


class _Visitor(ast.NodeVisitor):
    def __init__(self, root: str, path: str, digest: str) -> None:
        self.root, self.path, self.digest, self.rows = root, path, digest, []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call) and _name(decorator.func).split(".")[-1] in {
                "register",
                "tool",
                "handler",
            }:
                values = _kwargs(decorator)
                op = (
                    values.get("operation")
                    or values.get("name")
                    or (_literal(decorator.args[0]) if decorator.args else "")
                )
                self._append(node.lineno, op, values, node.name)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        method = _name(node.func).split(".")[-1]
        if method in {"register", "register_tool", "add_handler", "declare_effect"}:
            values = _kwargs(node)
            op = (
                values.get("operation")
                or values.get("name")
                or (_literal(node.args[0]) if node.args else "")
            )
            handler = values.get("handler") or values.get("callback")
            if not handler and len(node.args) > 1:
                handler = _name(node.args[1])
            self._append(node.lineno, op, values, handler)
        self.generic_visit(node)

    def _append(self, line: int, operation: str, values: Mapping[str, str], handler: str) -> None:
        aliases = tuple(
            sorted(item.strip() for item in values.get("aliases", "").split(",") if item.strip())
        )
        dispatcher = values.get("dispatcher") or values.get("dispatch") or ""
        effect = values.get("effect") or ""
        schemas = (
            values.get("request_schema") or values.get("request") or "",
            values.get("result_schema") or values.get("result") or "",
            values.get("error_schema") or values.get("error") or "",
        )
        reason = ""
        if not operation:
            reason = "operation_missing"
        elif not handler:
            reason = "handler_unresolved"
        elif not dispatcher:
            reason = "dispatcher_unresolved"
        elif not effect:
            reason = "effect_unresolved"
        elif not all(schemas):
            reason = "schema_evidence_missing"
        self.rows.append(
            ProviderSurfaceRow(
                self.root,
                self.path,
                line,
                operation,
                aliases,
                dispatcher,
                handler,
                effect,
                *schemas,
                SurfaceStatus.RESOLVED if not reason else SurfaceStatus.UNRESOLVED,
                reason,
                self.digest,
            )
        )


def scan_provider_surfaces(
    package_roots: Mapping[str, str | Path],
    *,
    forest_identity: str,
    index_identity: str,
    expectations: Sequence[ProviderSurfaceExpectation] = (),
) -> ProviderSurfaceHealth:
    """Scan multi-root Python sources deterministically without importing them."""
    rows: list[ProviderSurfaceRow] = []
    for root_name, root_value in sorted(package_roots.items()):
        root = Path(root_value)
        for path in sorted(root.rglob("*.py")):
            relative = path.relative_to(root).as_posix()
            data = path.read_bytes()
            digest = "sha256:" + hashlib.sha256(data).hexdigest()
            try:
                tree = ast.parse(data.decode("utf-8"), filename=relative)
            except (SyntaxError, UnicodeDecodeError):
                rows.append(
                    ProviderSurfaceRow(
                        root_name,
                        relative,
                        0,
                        "",
                        (),
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        SurfaceStatus.PARSER_FAILURE,
                        "python_syntax_failure",
                        digest,
                    )
                )
                continue
            visitor = _Visitor(root_name, relative, digest)
            visitor.visit(tree)
            rows.extend(visitor.rows)
    anchors: dict[tuple[str, str, str], list[ProviderSurfaceRow]] = {}
    for row in rows:
        if row.status is SurfaceStatus.RESOLVED:
            anchors.setdefault(row.anchor(), []).append(row)
    normalized: list[ProviderSurfaceRow] = []
    for row in rows:
        if row.status is SurfaceStatus.RESOLVED and len(anchors[row.anchor()]) > 1:
            normalized.append(
                ProviderSurfaceRow(
                    **{
                        **row.__dict__,
                        "status": SurfaceStatus.AMBIGUOUS,
                        "reason": "duplicate_equivalent_anchor",
                    }
                )
            )
        else:
            normalized.append(row)
    observed = {row.operation for row in normalized if row.status is SurfaceStatus.RESOLVED}
    blockers = set()
    for expectation in expectations:
        if expectation.mandatory and expectation.operation not in observed:
            blockers.add(expectation.operation)
    blockers.update(
        row.operation or f"{row.package_root}:{row.source_path}"
        for row in normalized
        if row.status is not SurfaceStatus.RESOLVED
    )
    return ProviderSurfaceHealth(
        forest_identity,
        index_identity,
        tuple(
            sorted(
                normalized,
                key=lambda row: (row.package_root, row.source_path, row.line, row.operation),
            )
        ),
        tuple(sorted(blockers)),
    )


__all__ = [
    "PROVIDER_SURFACE_HEALTH_INTERFACE",
    "ProviderSurfaceExpectation",
    "ProviderSurfaceHealth",
    "ProviderSurfaceRow",
    "SurfaceStatus",
    "scan_provider_surfaces",
]
