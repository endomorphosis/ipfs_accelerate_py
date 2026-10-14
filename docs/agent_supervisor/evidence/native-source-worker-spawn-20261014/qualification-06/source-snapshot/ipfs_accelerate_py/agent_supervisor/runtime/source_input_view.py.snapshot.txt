"""Bounded source presentation over a verified semantic input.

Canonical producer artifacts and identities stay unchanged. Optional factoring
replaces only normalized facet-AST slots in a versioned provider presentation;
retained original blocks restore the exact native transport. Source read views
confer neither capsule admission nor proof, edit, execution or task authority.
"""
from __future__ import annotations

from dataclasses import dataclass
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import platform
import re

from . import code_source_view as rendering
from . import semantic_router_translation as native

VIEW_SCHEMA = "supervisor-source-input-view@1"
BLOCK_SCHEMA = "supervisor-source-read-input@1"
RECEIPT_SCHEMA = "supervisor-source-input-projection@1"
AST_REF = "$source_input_ast"
MAX_VIEWS = 8
MAX_SOURCE_BYTES = 4 * 1024 * 1024
MAX_BLOCK_BYTES = 32768
MAX_PROVIDER_BYTES = 256000
MAX_RECEIPT_BYTES = 32768
INSTRUCTIONS = (
    "These are source read views of the same retained canonical files. Required raw sources, "
    "capsule admissions, assumptions, constraints and proof obligations remain unchanged. "
    "A compact view may omit comments and formatting; expand exact current source before editing "
    "or resolving uncertainty. AST references identify host-retained original facet evidence, "
    "not replacement producer capsules. Verify the current file SHA256 and use guarded symbol edits. "
    "No view grants proof, edit, execution or task-completion authority."
)


class SourceInputViewError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value: str | bytes) -> str:
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def _fail(code):
    raise SourceInputViewError(code)


def _bindings():
    return {"renderer_sha256": _sha(Path(rendering.__file__).read_bytes()),
            "projector_sha256": _sha(Path(__file__).read_bytes()),
            "python_version": platform.python_version()}


def _contains_marker(value):
    if isinstance(value, dict):
        return AST_REF in value or any(_contains_marker(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_marker(child) for child in value)
    return False


def _module_name(path):
    parts = list(PurePosixPath(path).parts)
    parts[-1] = parts[-1][:-3]
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts) or "__main__"


@dataclass(frozen=True)
class SourceInputView:
    native_provider_prompt: str
    provider_prompt: str
    block: str
    receipt_json: str

    @property
    def receipt(self):
        return json.loads(self.receipt_json)


_FALSE = {"semantic_equivalence_claimed", "proof_authority", "edit_authority",
          "execution_authority", "completion_authority"}
_FIELDS = {"schema", "format", "factor_ast", "source_transport_schema", "translation_cid", "task_id",
           "scope_cid", "semantic_root_cid", "source_manifest_sha256", "native_prompt_sha256",
           "native_provider_sha256", "native_provider_bytes", "provider_prompt_sha256", "provider_prompt_bytes",
           "block_sha256", "block_bytes", "views_included", "compact_views_included", "ast_slots_factored",
           "ast_bytes_factored", "ordered_selection_sha256", "views", "exclusions", "freshness_checked",
           "workspace_checked", "renderer_sha256", "projector_sha256", "python_version"} | _FALSE
_VIEW_FIELDS = {"id", "path", "symbol", "producer_qualified_name", "stable_symbol_id", "capsule_cid",
                "source_sha256", "source_cid", "content_sha256", "content_bytes", "format", "fallback_reason",
                "ast_sha256", "ast_bytes", "capsule_index", "facet_index", "kind", "raw_source_required",
                "class_child_bodies_included"}


def validate_source_input_view_receipt(receipt: dict) -> dict:
    if (type(receipt) is not dict or set(receipt) != _FIELDS or receipt.get("schema") != RECEIPT_SCHEMA
            or receipt.get("format") not in {"raw", "compact"}
            or type(receipt.get("factor_ast")) is not bool
            or type(receipt.get("freshness_checked")) is not bool
            or type(receipt.get("workspace_checked")) is not bool
            or any(receipt.get(key) is not False for key in _FALSE)):
        _fail("invalid_receipt")
    for key in ("source_manifest_sha256", "native_prompt_sha256", "native_provider_sha256",
                "provider_prompt_sha256", "block_sha256", "ordered_selection_sha256",
                "renderer_sha256", "projector_sha256"):
        if type(receipt[key]) is not str or re.fullmatch(r"[a-f0-9]{64}", receipt[key]) is None:
            _fail("invalid_receipt_digest")
    for key in ("source_transport_schema", "translation_cid", "task_id", "scope_cid", "semantic_root_cid", "python_version"):
        if type(receipt[key]) is not str or not 1 <= len(receipt[key]) <= 512:
            _fail("invalid_receipt_identity")
    for key, maximum in (("native_provider_bytes", native.MAX_BYTES), ("provider_prompt_bytes", MAX_PROVIDER_BYTES),
                         ("block_bytes", MAX_BLOCK_BYTES), ("views_included", MAX_VIEWS),
                         ("compact_views_included", MAX_VIEWS), ("ast_slots_factored", MAX_VIEWS),
                         ("ast_bytes_factored", native.MAX_BYTES)):
        if type(receipt[key]) is not int or not 0 <= receipt[key] <= maximum:
            _fail("invalid_receipt_count")
    rows = receipt["views"]
    if (type(rows) is not list or not 1 <= len(rows) <= MAX_VIEWS or receipt["views_included"] != len(rows)
            or type(receipt["exclusions"]) is not list or len(receipt["exclusions"]) > 32):
        _fail("invalid_receipt_views")
    ids = set()
    for index, row in enumerate(rows):
        if type(row) is not dict or set(row) != _VIEW_FIELDS or row.get("id") != f"v{index}":
            _fail("invalid_receipt_view")
        for key in ("path", "symbol", "producer_qualified_name", "stable_symbol_id", "capsule_cid", "source_cid", "kind"):
            if type(row[key]) is not str or not 1 <= len(row[key]) <= 1024:
                _fail("invalid_receipt_view")
        for key in ("source_sha256", "content_sha256", "ast_sha256"):
            if type(row[key]) is not str or re.fullmatch(r"[a-f0-9]{64}", row[key]) is None:
                _fail("invalid_receipt_view")
        for key, limit in (("content_bytes", MAX_BLOCK_BYTES), ("ast_bytes", native.MAX_BYTES),
                           ("capsule_index", 31), ("facet_index", 0)):
            if type(row[key]) is not int or not 0 <= row[key] <= limit:
                _fail("invalid_receipt_view")
        if (row["format"] not in {"raw", "compact"}
                or row["fallback_reason"] not in {None, "ast_mismatch", "unsupported_compact_rendering", "not_smaller"}
                or type(row["raw_source_required"]) is not bool
                or type(row["class_child_bodies_included"]) is not bool
                or row["stable_symbol_id"] in ids):
            _fail("invalid_receipt_view")
        ids.add(row["stable_symbol_id"])
    for row in receipt["exclusions"]:
        if (type(row) is not dict or set(row) != {"capsule_index", "reason"}
                or type(row["capsule_index"]) is not int or not 0 <= row["capsule_index"] <= 31
                or row["reason"] not in {"already_inline_raw", "unsupported_path", "multiple_or_missing_facets",
                    "unsupported_symbol", "view_limit", "source_limit", "ast_projection_mismatch", "view_count_limit"}):
            _fail("invalid_receipt_exclusion")
    if (receipt["compact_views_included"] != sum(row["format"] == "compact" for row in rows)
            or receipt["ast_slots_factored"] != (len(rows) if receipt["factor_ast"] else 0)
            or receipt["ast_bytes_factored"] != (sum(row["ast_bytes"] for row in rows) if receipt["factor_ast"] else 0)
            or len(_json(receipt).encode()) > MAX_RECEIPT_BYTES):
        _fail("invalid_receipt_totals")
    return receipt


def _verified(encoded, repository, require_current):
    if type(encoded) is not native.EncodedSemanticPrompt or type(require_current) is not bool:
        _fail("verified_semantic_input_required")
    root = Path(repository).absolute()
    if root.resolve(strict=True) != root:
        _fail("noncanonical_repository")
    try:
        producer = native.encode_semantic_router_prompt if require_current else native.replay_semantic_router_prompt_for_audit
        verified = producer(prompt=encoded.native_prompt, repository=root,
            transport_schema=native._provider_transport_schema(encoded.provider_prompt))
        if (verified.table != encoded.table or verified.provider_prompt != encoded.provider_prompt
                or verified.native_prompt != encoded.native_prompt):
            _fail("semantic_input_mismatch")
        parts = native._prompt_parts(encoded.native_prompt)
        payload = native._parse(parts[4])
        table = encoded.table.to_dict()
        if (payload.get("schema") != "supervisor-semantic-worker-context@2"
                or type(payload.get("worker_projection")) is not dict):
            _fail("explicit_projected_program_required")
        return root, payload, table
    except native.SemanticTranslationError:
        _fail("semantic_verification_failed")


def _assert_current(root, workspace, manifest):
    for path, binding in manifest.items():
        if _sha(native._read(root, path)) != binding["sha256"]:
            _fail("stale_source")
        if workspace is not None and _sha(native._read(workspace, path)) != binding["sha256"]:
            _fail("stale_workspace")


def project_source_input_view(*, encoded, repository: Path, format: str,
                              factor_ast: bool = False, workspace: Path | None = None,
                              require_current: bool = True) -> SourceInputView:
    if format not in {"raw", "compact"} or type(factor_ast) is not bool or type(require_current) is not bool:
        _fail("invalid_projection_policy")
    if not require_current and workspace is not None:
        _fail("historical_workspace_forbidden")
    if workspace is not None:
        workspace = Path(workspace).absolute()
        if workspace.resolve(strict=True) != workspace:
            _fail("noncanonical_workspace")
    root, payload, table = _verified(encoded, repository, require_current)
    manifest = payload["manifest"]
    if require_current:
        _assert_current(root, workspace, manifest)
    wire = native._parse(encoded.provider_prompt)
    if _contains_marker(wire):
        _fail("reserved_marker_collision")
    selected = payload["worker_projection"].get("selected_symbols")
    capsules, admissions = payload["capsules"], payload["admissions"]
    if (type(selected) is not list or not 1 <= len(selected) <= 32
            or len(selected) != len(capsules) or len(admissions) != len(capsules)):
        _fail("selected_population_mismatch")
    from ipfs_accelerate_py.mcp_server.mcplusplus.kubo_cid import cid_for_bytes
    from ipfs_datasets_py.logic.software_contracts.semantic_index.python_analysis import _projection
    blocks = PurePosixPath(table["semantic_artifact"]).parent / "blocks"
    sources, total_bytes, rows, view_records, exclusions = {}, 0, [], [], []
    raw_rows, compact_rows = [], []
    for index, (selected_row, capsule, admission) in enumerate(zip(selected, capsules, admissions)):
        def exclude(reason):
            exclusions.append({"capsule_index": index, "reason": reason})
        if len(rows) >= MAX_VIEWS:
            exclude("view_count_limit")
            continue
        if (type(selected_row) is not dict or set(selected_row) != {"path", "qualified_name", "stable_symbol_id"}
                or selected_row["stable_symbol_id"] != capsule["stable_symbol_id"]
                or selected_row["path"] != capsule["source_slice_path"]):
            _fail("selected_identity_mismatch")
        path, qualified = selected_row["path"], selected_row["qualified_name"]
        if path not in manifest or path not in payload["program_paths"]:
            _fail("source_scope_mismatch")
        if path in payload["raw_sources"] or path in payload["required_raw_paths"]:
            exclude("already_inline_raw")
            continue
        try:
            rendering._path(path)
        except rendering.SourceViewError:
            exclude("unsupported_path")
            continue
        facets = capsule.get("metadata", {}).get("facets")
        if (type(facets) is not list or len(facets) != 1 or type(facets[0]) is not dict
                or "version_evidence" not in facets[0] or type(facets[0]["version_evidence"]) is not dict):
            exclude("multiple_or_missing_facets")
            continue
        prefix = _module_name(path) + "."
        if qualified == _module_name(path):
            symbol = "module"
        elif type(qualified) is str and qualified.startswith(prefix):
            symbol = qualified[len(prefix):]
            # The helper reserves module for a whole-file read; a real binding
            # named module cannot reuse that namespace.
            if symbol == "module":
                exclude("unsupported_symbol")
                continue
        else:
            exclude("unsupported_symbol")
            continue
        binding = manifest[path]
        if capsule["source_cid"] != binding["source_cid"]:
            _fail("capsule_source_mismatch")
        if path not in sources:
            raw = native._read(root, str(blocks / binding["source_cid"]))
            if _sha(raw) != binding["sha256"] or cid_for_bytes(raw) != binding["source_cid"]:
                _fail("retained_source_mismatch")
            total_bytes += len(raw)
            if total_bytes > MAX_SOURCE_BYTES:
                _fail("source_budget_exceeded")
            sources[path] = raw
        raw = sources[path]
        try:
            source, _encoding = rendering._decode(raw)
            tree = ast.parse(source)
            nodes = [tree] if qualified == _module_name(path) else rendering._symbols(tree).get(symbol, [])
            assignment = False
            if not nodes and "." not in symbol:
                for node in tree.body:
                    target = (node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1
                              else node.target if isinstance(node, ast.AnnAssign) else None)
                    if isinstance(target, ast.Name) and target.id == symbol:
                        nodes.append(node)
                assignment = bool(nodes)
            raw_view = rendering.render_source_bytes(path, raw, symbol, "raw", read_only_assignment=assignment)
            compact_view = rendering.render_source_bytes(path, raw, symbol, "compact", read_only_assignment=assignment)
            if compact_view["format"] == "compact" and compact_view["content_bytes"] >= raw_view["content_bytes"]:
                compact_view = {**raw_view, "requested_format": "compact", "fallback_reason": "not_smaller"}
            if len(nodes) != 1:
                exclude("unsupported_symbol")
                continue
            if _projection(nodes[0]) != facets[0]["version_evidence"]:
                exclude("ast_projection_mismatch")
                continue
        except (rendering.SourceViewError, SyntaxError, RecursionError) as exc:
            code = getattr(exc, "code", "unsupported_symbol")
            exclude(code if code in {"view_limit", "source_limit"} else "unsupported_symbol")
            continue
        identity = f"v{len(rows)}"
        def public_row(view):
            return {"id": identity, "path": path, "symbol": symbol,
                    "source_sha256": binding["sha256"], "format": view["format"], "content": view["content"]}
        next_raw, next_compact = public_row(raw_view), public_row(compact_view)
        def block_value(values):
            return {"schema": BLOCK_SCHEMA, "views": values, "instructions": INSTRUCTIONS,
                    "edit_authority": False, "proof_authority": False, "completion_authority": False}
        if max(len(_json(block_value(raw_rows + [next_raw])).encode()),
               len(_json(block_value(compact_rows + [next_compact])).encode())) + 32 > MAX_BLOCK_BYTES:
            exclude("view_limit")
            continue
        chosen = raw_view if format == "raw" else compact_view
        row = next_raw if format == "raw" else next_compact
        raw_rows.append(next_raw)
        compact_rows.append(next_compact)
        rows.append(row)
        ast_text = _json(facets[0]["version_evidence"])
        view_records.append({"id": identity, "path": path, "symbol": symbol,
            "producer_qualified_name": qualified, "stable_symbol_id": capsule["stable_symbol_id"],
            "capsule_cid": capsule["capsule_cid"], "source_sha256": binding["sha256"],
            "source_cid": binding["source_cid"], "content_sha256": _sha(chosen["content"]),
            "content_bytes": chosen["content_bytes"], "format": chosen["format"],
            "fallback_reason": chosen["fallback_reason"], "ast_sha256": _sha(ast_text),
            "ast_bytes": len(ast_text.encode()), "capsule_index": index, "facet_index": 0,
            "kind": chosen["kind"], "raw_source_required": admission["ref"]["raw_source_required"],
            "class_child_bodies_included": chosen["kind"] == "ClassDef"})
        if factor_ast:
            target = wire["translated_semantic"]["capsules"][index]["metadata"]["facets"][0]
            if target.get("version_evidence") != facets[0]["version_evidence"]:
                _fail("facet_slot_mismatch")
            target["version_evidence"] = {AST_REF: identity}
    if not rows:
        _fail("no_eligible_source_views")
    if format == "compact" and not any(row["format"] == "compact" for row in rows):
        _fail("no_compact_source_views")
    read_block = {"schema": BLOCK_SCHEMA, "views": rows, "instructions": INSTRUCTIONS,
                  "edit_authority": False, "proof_authority": False, "completion_authority": False}
    block = "\n\nSource read views:\n" + _json(read_block)
    provider_prompt = (_json({"schema": VIEW_SCHEMA, "semantic_transport": wire, "source_read_view": read_block,
                             "factored_ast": True, "execution_authority": False, "completion_authority": False})
                       if factor_ast else encoded.provider_prompt + block)
    if len(provider_prompt.encode()) > MAX_PROVIDER_BYTES or len(block.encode()) > MAX_BLOCK_BYTES:
        _fail("projection_budget_exceeded")
    selection = [{key: row[key] for key in ("stable_symbol_id", "capsule_cid", "path", "symbol", "source_sha256", "ast_sha256")}
                 for row in view_records]
    receipt = {"schema": RECEIPT_SCHEMA, "format": format, "factor_ast": factor_ast,
        "source_transport_schema": native._provider_transport_schema(encoded.provider_prompt),
        "translation_cid": encoded.table.translation_cid, "task_id": table["task_id"],
        "scope_cid": table["scope_cid"], "semantic_root_cid": table["semantic_root_cid"],
        "source_manifest_sha256": _sha(_json(manifest)), "native_prompt_sha256": _sha(encoded.native_prompt),
        "native_provider_sha256": _sha(encoded.provider_prompt), "native_provider_bytes": len(encoded.provider_prompt.encode()),
        "provider_prompt_sha256": _sha(provider_prompt), "provider_prompt_bytes": len(provider_prompt.encode()),
        "block_sha256": _sha(block), "block_bytes": len(block.encode()), "views_included": len(rows),
        "compact_views_included": sum(row["format"] == "compact" for row in rows),
        "ast_slots_factored": len(rows) if factor_ast else 0,
        "ast_bytes_factored": sum(row["ast_bytes"] for row in view_records) if factor_ast else 0,
        "ordered_selection_sha256": _sha(_json(selection)), "views": view_records, "exclusions": exclusions,
        "freshness_checked": require_current, "workspace_checked": workspace is not None,
        **_bindings(), **{key: False for key in _FALSE}}
    validate_source_input_view_receipt(receipt)
    if require_current:
        _assert_current(root, workspace, manifest)
    return SourceInputView(encoded.provider_prompt, provider_prompt, block, _json(receipt))


def replay_source_input_view_for_audit(*, encoded, repository: Path, receipt: dict) -> SourceInputView:
    validate_source_input_view_receipt(receipt)
    result = project_source_input_view(encoded=encoded, repository=repository, format=receipt["format"],
                                      factor_ast=receipt["factor_ast"], require_current=False)
    ignored = {"freshness_checked", "workspace_checked"}
    if ({key: value for key, value in result.receipt.items() if key not in ignored}
            != {key: value for key, value in receipt.items() if key not in ignored}):
        _fail("historical_projection_mismatch")
    return result


def restore_source_input_view(*, provider_prompt: str, receipt: dict, encoded, repository: Path) -> str:
    """Restore exact pre-presentation transport from retained verified artifacts.

    Rebuilding and comparing the entire view prevents hidden field changes or
    fabricated replacement references; historical restoration grants no freshness.
    """
    result = replay_source_input_view_for_audit(encoded=encoded, repository=repository, receipt=receipt)
    if type(provider_prompt) is not str or provider_prompt != result.provider_prompt:
        _fail("projected_input_mismatch")
    return result.native_provider_prompt


def recheck_source_input_freshness(*, encoded, repository: Path, workspace: Path) -> None:
    """Recheck the already admitted complete manifest immediately before dispatch."""
    if type(encoded) is not native.EncodedSemanticPrompt or encoded.freshness_checked is not True:
        _fail("current_semantic_input_required")
    root, allocated = Path(repository).absolute(), Path(workspace).absolute()
    if root.resolve(strict=True) != root or allocated.resolve(strict=True) != allocated:
        _fail("noncanonical_repository")
    _assert_current(root, allocated, encoded.table.to_dict()["source_manifest"])


def assert_current_source_input_view(*, encoded, repository: Path, workspace: Path, receipt: dict) -> None:
    """Bind a projection receipt and recheck every captured source before dispatch."""
    validate_source_input_view_receipt(receipt)
    if (type(encoded) is not native.EncodedSemanticPrompt or not encoded.freshness_checked
            or receipt["freshness_checked"] is not True or receipt["workspace_checked"] is not True):
        _fail("current_workspace_projection_required")
    table = encoded.table.to_dict()
    if (receipt["native_prompt_sha256"] != _sha(encoded.native_prompt)
            or receipt["native_provider_sha256"] != _sha(encoded.provider_prompt)
            or receipt["translation_cid"] != encoded.table.translation_cid
            or receipt["source_manifest_sha256"] != _sha(_json(table["source_manifest"]))
            or any(receipt[key] != value for key, value in _bindings().items())
            or Path(repository).absolute() != Path(table["source_repository"])):
        _fail("projection_binding_mismatch")
    recheck_source_input_freshness(encoded=encoded, repository=repository, workspace=workspace)
