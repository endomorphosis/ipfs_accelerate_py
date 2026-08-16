"""DCR-041 structural preview/inverse fixtures remain hermetic and non-writing."""

from __future__ import annotations

import ast
import hashlib

from ipfs_accelerate_py.agent_supervisor.autonomous_repair.operators.registry import (
    OperatorDescriptor,
    OperatorRegistry,
)
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.operators.registry_repairs import (
    StructuralPreviewStatus,
    ast_span_identity,
    build_registry_repair_preview,
)


def _registry() -> OperatorRegistry:
    raw = {
        "operator_id": "operator:structural",
        "kind": "replace_exact_bytes",
        "input_schema": {
            "type": "object",
            "required": ["before_digest"],
            "properties": {"before_digest": "sha256"},
            "additional_properties": False,
        },
        "owner_root": "root:accelerate",
        "write_scope": ["catalog.py"],
        "before_predicates": ["predicate:before"],
        "after_predicates": ["predicate:after"],
        "applicability_proofs": ["cid:proof"],
        "preview": {"kind": "metadata_only", "fields": ["input_cid"]},
        "inverse": {"kind": "restore_exact_before_bytes", "binding": "before_digest"},
        "validation_commands": [["pytest", "-q", "test/api/test_catalog.py"]],
    }
    descriptor = OperatorDescriptor.from_mapping(raw)
    return OperatorRegistry(
        [descriptor], reviewed_manifest={descriptor.operator_id: descriptor.descriptor_id}
    )


def _request(
    source: bytes, anchor: dict[str, object], action: str, payload: dict[str, str]
) -> dict[str, object]:
    return {
        "operator_id": "operator:structural",
        "action": action,
        "owner_root": "root:accelerate",
        "relative_path": "catalog.py",
        "source_bytes": source,
        "source_digest": "sha256:" + hashlib.sha256(source).hexdigest(),
        "anchor": anchor,
        "payload": payload,
        "behavioral_postcondition": {
            "kind": "exact-structural-postcondition",
            "operation": payload.get("operation", payload.get("key", "catalog.read")),
        },
    }


def test_alias_registration_and_duplicate_previews_are_exact_reversible_and_idempotent() -> None:
    registry = _registry()
    manifest = registry.report()["registry_cid"]
    source = b'ALIASES = {"catalog.read": ("catalog_read",)}\n\ndef handle():\n    return 1\n\nserver.register_tool("other", handle)\n'
    tree = ast.parse(source)
    alias = build_registry_repair_preview(
        _request(
            source,
            ast_span_identity(source, tree.body[0]),
            "add_missing_alias",
            {"registry_symbol": "ALIASES", "key": "catalog.read", "alias": "read_catalog"},
        ),
        registry=registry,
        manifest_cid=manifest,
    )
    assert alias.status is StructuralPreviewStatus.PREVIEWED
    assert alias.forward_diff and alias.inverse_diff and alias.after_bytes != source
    alias_tree = ast.parse(alias.after_bytes)
    second = build_registry_repair_preview(
        _request(
            alias.after_bytes,
            ast_span_identity(alias.after_bytes, alias_tree.body[0]),
            "add_missing_alias",
            {"registry_symbol": "ALIASES", "key": "catalog.read", "alias": "read_catalog"},
        ),
        registry=registry,
        manifest_cid=manifest,
    )
    assert second.status is StructuralPreviewStatus.ABSTAINED

    call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call))
    registration = build_registry_repair_preview(
        _request(
            source,
            ast_span_identity(source, call),
            "add_missing_registration",
            {
                "operation": "catalog.read",
                "handler": "handle",
                "registration_api": "server.register_tool",
            },
        ),
        registry=registry,
        manifest_cid=manifest,
    )
    assert registration.status is StructuralPreviewStatus.PREVIEWED

    duplicate_source = source + b'server.register_tool("other", handle)\n'
    duplicate_tree = ast.parse(duplicate_source)
    duplicate_call = [
        node
        for node in ast.walk(duplicate_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "other"
    ][1]
    removal = build_registry_repair_preview(
        _request(
            duplicate_source,
            ast_span_identity(duplicate_source, duplicate_call),
            "remove_duplicate_anchor",
            {"operation": "other", "handler": "handle", "registration_api": "server.register_tool"},
        ),
        registry=registry,
        manifest_cid=manifest,
    )
    assert removal.status is StructuralPreviewStatus.PREVIEWED


def test_stale_ambiguous_dynamic_wrong_owner_and_manifest_are_nonpassing() -> None:
    registry = _registry()
    manifest = registry.report()["registry_cid"]
    source = b"ALIASES = build_aliases()\n"
    tree = ast.parse(source)
    dynamic = build_registry_repair_preview(
        _request(
            source,
            ast_span_identity(source, tree.body[0]),
            "add_missing_alias",
            {"registry_symbol": "ALIASES", "key": "catalog.read", "alias": "read_catalog"},
        ),
        registry=registry,
        manifest_cid=manifest,
    )
    assert dynamic.status is StructuralPreviewStatus.ABSTAINED

    literal = b'ALIASES = {"catalog.read": ("catalog_read",)}\n'
    literal_tree = ast.parse(literal)
    stale_anchor = ast_span_identity(literal, literal_tree.body[0])
    stale_anchor["sha256"] = "sha256:stale"
    assert (
        build_registry_repair_preview(
            _request(
                literal,
                stale_anchor,
                "add_missing_alias",
                {"registry_symbol": "ALIASES", "key": "catalog.read", "alias": "read_catalog"},
            ),
            registry=registry,
            manifest_cid=manifest,
        ).status
        is StructuralPreviewStatus.REJECTED
    )

    missing_handler = b'server.register_tool("other", missing_handler)\n'
    missing_tree = ast.parse(missing_handler)
    missing_call = next(node for node in ast.walk(missing_tree) if isinstance(node, ast.Call))
    assert (
        build_registry_repair_preview(
            _request(
                missing_handler,
                ast_span_identity(missing_handler, missing_call),
                "add_missing_registration",
                {
                    "operation": "catalog.read",
                    "handler": "missing_handler",
                    "registration_api": "server.register_tool",
                },
            ),
            registry=registry,
            manifest_cid=manifest,
        ).status
        is StructuralPreviewStatus.ABSTAINED
    )

    multiple = literal + literal
    multiple_tree = ast.parse(multiple)
    assert (
        build_registry_repair_preview(
            _request(
                multiple,
                ast_span_identity(multiple, multiple_tree.body[0]),
                "add_missing_alias",
                {"registry_symbol": "ALIASES", "key": "catalog.read", "alias": "read_catalog"},
            ),
            registry=registry,
            manifest_cid=manifest,
        ).status
        is StructuralPreviewStatus.ABSTAINED
    )

    wrong_owner = _request(
        source,
        ast_span_identity(source, tree.body[0]),
        "add_missing_alias",
        {"registry_symbol": "ALIASES", "key": "catalog.read", "alias": "read_catalog"},
    )
    wrong_owner["owner_root"] = "root:other"
    assert (
        build_registry_repair_preview(wrong_owner, registry=registry, manifest_cid=manifest).status
        is StructuralPreviewStatus.REJECTED
    )
    assert (
        build_registry_repair_preview(
            _request(
                source,
                ast_span_identity(source, tree.body[0]),
                "add_missing_alias",
                {"registry_symbol": "ALIASES", "key": "catalog.read", "alias": "read_catalog"},
            ),
            registry=registry,
            manifest_cid="bafy-stale",
        ).status
        is StructuralPreviewStatus.REJECTED
    )
