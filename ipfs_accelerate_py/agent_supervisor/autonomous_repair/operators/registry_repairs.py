"""DCR-041 structural, non-writing preview and inverse builders.

Only exact AST identities are accepted.  This layer produces byte patches for
review, never applies them; activation stays pending on DCR-035 and the DCR-040
reviewed registry manifest.
"""

from __future__ import annotations

import ast
import base64
import copy
import difflib
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

from ...proof.formal_verification_contracts import content_identity
from .registry import OperatorRegistry

STRUCTURAL_REPAIR_PREVIEW_INTERFACE: Final[str] = "StructuralRepairPreview@1"
STRUCTURAL_REPAIR_PREVIEW_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/structural-repair-preview@1"
)


class StructuralPreviewStatus(StrEnum):
    PREVIEWED = "previewed"
    ABSTAINED = "abstained"
    REJECTED = "rejected"


class StructuralRepairPreviewError(ValueError):
    """A preview request is malformed rather than structurally inapplicable."""


def _digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _line_starts(data: bytes) -> list[int]:
    starts = [0]
    starts.extend(index + 1 for index, char in enumerate(data) if char == 10)
    return starts


def _span(data: bytes, node: ast.AST) -> tuple[int, int]:
    if not all(
        hasattr(node, item) for item in ("lineno", "col_offset", "end_lineno", "end_col_offset")
    ):
        raise StructuralRepairPreviewError("AST node has no exact source span")
    starts = _line_starts(data)
    try:
        return (
            starts[node.lineno - 1] + node.col_offset,  # type: ignore[attr-defined]
            starts[node.end_lineno - 1] + node.end_col_offset,  # type: ignore[attr-defined]
        )
    except (AttributeError, IndexError) as exc:
        raise StructuralRepairPreviewError("AST span is outside source bytes") from exc


def ast_span_identity(source: bytes, node: ast.AST) -> dict[str, Any]:
    """Return the exact byte-span identity a preview request must supply."""
    start, end = _span(source, node)
    return {
        "node_type": type(node).__name__,
        "start": start,
        "end": end,
        "sha256": _digest(source[start:end]),
    }


def _require_anchor(source: bytes, node: ast.AST, anchor: Any) -> None:
    if not isinstance(anchor, Mapping) or set(anchor) != {"node_type", "start", "end", "sha256"}:
        raise StructuralRepairPreviewError("anchor must be an exact AST/span identity")
    if dict(anchor) != ast_span_identity(source, node):
        raise StructuralRepairPreviewError(
            "anchor is stale or does not identify the selected AST node"
        )


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else ""
    return ""


def _literal_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or any(char in value for char in "\n\r;|&`$"):
        raise StructuralRepairPreviewError(f"{field} must be closed structural text")
    return value


def _parse_request(
    request: Mapping[str, Any],
) -> tuple[str, str, str, bytes, str, Mapping[str, Any], Mapping[str, Any], str]:
    required = {
        "operator_id",
        "action",
        "owner_root",
        "relative_path",
        "source_bytes",
        "source_digest",
        "anchor",
        "payload",
        "behavioral_postcondition",
    }
    if not isinstance(request, Mapping) or set(request) != required:
        raise StructuralRepairPreviewError("preview request fields are closed")
    source = request["source_bytes"]
    if not isinstance(source, bytes) or not source:
        raise StructuralRepairPreviewError("current source bytes are required")
    source_digest = _literal_text(request["source_digest"], "source_digest")
    if source_digest != _digest(source):
        raise StructuralRepairPreviewError("current source digest is stale")
    payload = request["payload"]
    behavior = request["behavioral_postcondition"]
    if not isinstance(payload, Mapping) or not isinstance(behavior, Mapping):
        raise StructuralRepairPreviewError("payload and behavioral postcondition must be mappings")
    return (
        _literal_text(request["operator_id"], "operator_id"),
        _literal_text(request["action"], "action"),
        _literal_text(request["owner_root"], "owner_root"),
        source,
        _literal_text(request["relative_path"], "relative_path"),
        request["anchor"],
        payload,
        content_identity(dict(behavior)),
    )


def _calls(tree: ast.AST, api: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _dotted_name(node.func) == api
    ]


def _abstain(reason: str, *, request_cid: str = "") -> StructuralRepairPreview:
    return StructuralRepairPreview(
        status=StructuralPreviewStatus.ABSTAINED,
        reason=reason,
        request_cid=request_cid,
    )


@dataclass(frozen=True)
class StructuralRepairPreview:
    status: StructuralPreviewStatus
    reason: str
    request_cid: str = ""
    operator_id: str = ""
    manifest_cid: str = ""
    old_digest: str = ""
    new_digest: str = ""
    forward_diff: str = ""
    inverse_diff: str = ""
    after_bytes: bytes = b""
    behavioral_postcondition_cid: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": STRUCTURAL_REPAIR_PREVIEW_SCHEMA,
            "interface": STRUCTURAL_REPAIR_PREVIEW_INTERFACE,
            "authoritative": False,
            "execution_authorized": False,
            "activation_status": "integration_pending_dcr035_dcr040",
            "status": self.status.value,
            "reason": self.reason,
            "request_cid": self.request_cid,
            "operator_id": self.operator_id,
            "manifest_cid": self.manifest_cid,
            "old_digest": self.old_digest,
            "new_digest": self.new_digest,
            "forward_diff": self.forward_diff,
            "inverse_diff": self.inverse_diff,
            "after_base64": base64.b64encode(self.after_bytes).decode("ascii"),
            "behavioral_postcondition_cid": self.behavioral_postcondition_cid,
            "model_call_count": 0,
        }


def _patch(
    source: bytes,
    start: int,
    end: int,
    replacement: bytes,
    *,
    request_cid: str,
    operator_id: str,
    manifest_cid: str,
    postcondition_cid: str,
) -> StructuralRepairPreview:
    after = source[:start] + replacement + source[end:]
    if after == source:
        return _abstain("non_idempotent_preview_required", request_cid=request_cid)
    before_text = source.decode("utf-8").splitlines(keepends=True)
    after_text = after.decode("utf-8").splitlines(keepends=True)
    forward = "".join(
        difflib.unified_diff(before_text, after_text, fromfile="before", tofile="after")
    )
    inverse = "".join(
        difflib.unified_diff(after_text, before_text, fromfile="after", tofile="before")
    )
    if not forward or not inverse:
        return _abstain("nonempty_inverse_required", request_cid=request_cid)
    return StructuralRepairPreview(
        status=StructuralPreviewStatus.PREVIEWED,
        reason="validation_pending",
        request_cid=request_cid,
        operator_id=operator_id,
        manifest_cid=manifest_cid,
        old_digest=_digest(source),
        new_digest=_digest(after),
        forward_diff=forward,
        inverse_diff=inverse,
        after_bytes=after,
        behavioral_postcondition_cid=postcondition_cid,
    )


def build_registry_repair_preview(
    request: Mapping[str, Any], *, registry: OperatorRegistry, manifest_cid: str
) -> StructuralRepairPreview:
    """Build one deterministic structural preview; no filesystem or code execution."""
    try:
        operator_id, action, owner, source, path, anchor, payload, postcondition_cid = (
            _parse_request(request)
        )
        registry_report = registry.report()
        if manifest_cid != registry_report["registry_cid"]:
            raise StructuralRepairPreviewError("reviewed manifest CID is stale")
        descriptors = {item.operator_id: item for item in registry.enumerate()}
        descriptor = descriptors.get(operator_id)
        if descriptor is None:
            raise StructuralRepairPreviewError("operator is absent from reviewed manifest")
        if owner != descriptor.owner_root or path not in descriptor.write_scope:
            raise StructuralRepairPreviewError("owner root or write path is not admitted")
        request_cid = content_identity(
            {
                key: (base64.b64encode(value).decode("ascii") if key == "source_bytes" else value)
                for key, value in request.items()
            }
        )
        try:
            tree = ast.parse(source, filename=path)
        except SyntaxError:
            return _abstain("source_ast_unavailable", request_cid=request_cid)
        if action == "add_missing_alias":
            if set(payload) != {"registry_symbol", "key", "alias"}:
                raise StructuralRepairPreviewError("alias payload fields are closed")
            symbol, key, alias = (
                _literal_text(payload[name], name) for name in ("registry_symbol", "key", "alias")
            )
            assignments = [
                node
                for node in tree.body
                if isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == symbol
            ]
            if len(assignments) != 1 or not isinstance(assignments[0].value, ast.Dict):
                return _abstain("literal_registry_not_uniquely_resolved", request_cid=request_cid)
            assignment = assignments[0]
            _require_anchor(source, assignment, anchor)
            entries = [
                (item_key, item_value)
                for item_key, item_value in zip(
                    assignment.value.keys, assignment.value.values, strict=True
                )
                if isinstance(item_key, ast.Constant) and item_key.value == key
            ]
            if len(entries) != 1 or not isinstance(entries[0][1], (ast.List, ast.Tuple, ast.Set)):
                return _abstain("dynamic_or_missing_registry_entry", request_cid=request_cid)
            values = entries[0][1]
            if any(
                not isinstance(item, ast.Constant) or not isinstance(item.value, str)
                for item in values.elts
            ):
                return _abstain("dynamic_registry_shape", request_cid=request_cid)
            if alias in {item.value for item in values.elts}:
                return _abstain("alias_already_present", request_cid=request_cid)
            replacement_node = copy.deepcopy(values)
            replacement_node.elts.append(ast.Constant(alias))
            start, end = _span(source, values)
            return _patch(
                source,
                start,
                end,
                ast.unparse(replacement_node).encode(),
                request_cid=request_cid,
                operator_id=operator_id,
                manifest_cid=manifest_cid,
                postcondition_cid=postcondition_cid,
            )
        if action == "add_missing_registration":
            if set(payload) != {"operation", "handler", "registration_api"}:
                raise StructuralRepairPreviewError("registration payload fields are closed")
            operation, handler, api = (
                _literal_text(payload[name], name)
                for name in ("operation", "handler", "registration_api")
            )
            handlers = [
                node
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == handler
            ]
            api_calls = _calls(tree, api)
            if len(handlers) != 1 or len(api_calls) != 1:
                return _abstain(
                    "handler_or_registration_api_not_uniquely_resolved", request_cid=request_cid
                )
            _require_anchor(source, api_calls[0], anchor)
            if any(
                len(call.args) >= 1
                and isinstance(call.args[0], ast.Constant)
                and call.args[0].value == operation
                for call in api_calls
            ):
                return _abstain("registration_already_present", request_cid=request_cid)
            addition = (
                b"" if source.endswith(b"\n") else b"\n"
            ) + f"{api}({operation!r}, {handler})\n".encode()
            return _patch(
                source,
                len(source),
                len(source),
                addition,
                request_cid=request_cid,
                operator_id=operator_id,
                manifest_cid=manifest_cid,
                postcondition_cid=postcondition_cid,
            )
        if action == "remove_duplicate_anchor":
            if set(payload) != {"operation", "handler", "registration_api"}:
                raise StructuralRepairPreviewError("duplicate payload fields are closed")
            operation, handler, api = (
                _literal_text(payload[name], name)
                for name in ("operation", "handler", "registration_api")
            )
            matches = [
                call
                for call in _calls(tree, api)
                if len(call.args) == 2
                and isinstance(call.args[0], ast.Constant)
                and call.args[0].value == operation
                and isinstance(call.args[1], ast.Name)
                and call.args[1].id == handler
            ]
            if len(matches) != 2:
                return _abstain("duplicate_anchor_not_provably_exact", request_cid=request_cid)
            selected = next(
                (call for call in matches if ast_span_identity(source, call) == dict(anchor)), None
            )
            if selected is None:
                raise StructuralRepairPreviewError("anchor is stale or not an exact duplicate call")
            parent = next(
                (
                    node
                    for node in tree.body
                    if isinstance(node, ast.Expr) and node.value is selected
                ),
                None,
            )
            if parent is None:
                return _abstain("duplicate_anchor_not_standalone", request_cid=request_cid)
            start, end = _span(source, parent)
            if source[end : end + 1] == b"\n":
                end += 1
            return _patch(
                source,
                start,
                end,
                b"",
                request_cid=request_cid,
                operator_id=operator_id,
                manifest_cid=manifest_cid,
                postcondition_cid=postcondition_cid,
            )
        raise StructuralRepairPreviewError("repair action is not closed")
    except StructuralRepairPreviewError as exc:
        return StructuralRepairPreview(status=StructuralPreviewStatus.REJECTED, reason=str(exc))


__all__ = [
    "STRUCTURAL_REPAIR_PREVIEW_INTERFACE",
    "STRUCTURAL_REPAIR_PREVIEW_SCHEMA",
    "StructuralPreviewStatus",
    "StructuralRepairPreview",
    "StructuralRepairPreviewError",
    "ast_span_identity",
    "build_registry_repair_preview",
]
