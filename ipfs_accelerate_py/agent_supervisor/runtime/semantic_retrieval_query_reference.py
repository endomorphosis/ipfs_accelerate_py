"""Reference one exact retrieval-query duplicate in required semantic evidence.

The owner holds the raw retrieval artifact pin independently. Current dispatch
uses the existing source-aware retrieval loader; historical replay uses pinned
native snapshot/result records and authenticated retained semantic sources.
The pure view is required evidence and grants no retrieval, source, task or
completion authority. It removes only ``query_text`` and has an exact inverse.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat

from ..context.context_compiler import build_text_context_references
from ..context.context_contracts import ContextReference
from . import semantic_router_translation as codec
from .semantic_context_projection import SemanticProjectionBindings, _encoded, _structure
from .semantic_decision_packet import SemanticDecisionPolicy, verify_semantic_decision_packet

SCHEMA = "supervisor-code-retrieval-query-reference@1"
PROJECTION_REVISION = "exact-required-semantic-query-reference@1"
MAX_CONTEXT_BYTES = 65_536
MAX_VIEW_BYTES = 131_072
MAX_ARTIFACT_BYTES = 2_000_000
MAX_SNAPSHOT_BYTES = 64_000_000
_BINDING_FIELDS = frozenset({"artifact", "expected_sha256", "task_id"})


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _path(value):
    _require(type(value) is str and value.strip() and 1 <= len(value.encode()) <= 4096
             and not PurePosixPath(value).is_absolute() and str(PurePosixPath(value)) == value
             and not any(part in {"", ".", ".."} for part in value.split("/"))
             and not any(char in value for char in ("\x00", "\n", "\r")),
             "canonical relative retrieval/source path required")
    return value


@dataclass(frozen=True)
class SemanticRetrievalQueryBinding:
    """Independently held raw artifact binding; never a derived chunk digest."""

    artifact: str
    expected_sha256: str
    task_id: str

    def __post_init__(self):
        _path(self.artifact)
        _require(type(self.expected_sha256) is str
                 and re.fullmatch(r"[0-9a-f]{64}", self.expected_sha256) is not None,
                 "independent raw retrieval SHA256 required")
        _require(type(self.task_id) is str and self.task_id == self.task_id.strip()
                 and 1 <= len(self.task_id.encode()) <= 512
                 and not any(char in self.task_id for char in ("\x00", "\n", "\r")),
                 "bounded independently held retrieval task identity required")

    def to_dict(self):
        return {key: getattr(self, key) for key in ("artifact", "expected_sha256", "task_id")}

    @classmethod
    def from_dict(cls, value):
        if type(value) is not dict or set(value) != _BINDING_FIELDS:
            raise ValueError("exact retrieval query binding fields required")
        return cls(**value)


def _root(repository):
    root = Path(repository).absolute()
    _require(root.resolve(strict=True) == root and root.is_dir(),
             "canonical existing retrieval repository required")
    return root


def _read(root, relative, maximum):
    """Read a bounded regular file through retained nonlink directory handles."""
    _path(relative)
    descriptors, edges, file_fd = [], [], None
    try:
        descriptors.append(os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC))
        root_stat = os.fstat(descriptors[0])
        for component in PurePosixPath(relative).parts[:-1]:
            parent = descriptors[-1]
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                            dir_fd=parent)
            descriptors.append(child)
            edges.append((parent, component, os.fstat(child)))
        parent, name = descriptors[-1], PurePosixPath(relative).name
        file_fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=parent)
        before = os.fstat(file_fd)
        _require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
                 and 0 < before.st_size <= maximum, "bounded single-link regular retrieval input required")
        with os.fdopen(file_fd, "rb", closefd=False) as stream:
            raw = stream.read(maximum + 1)
        after = os.fstat(file_fd)
        current = os.stat(name, dir_fd=parent, follow_symlinks=False)
        identity = lambda row: (row.st_dev, row.st_ino, row.st_size, row.st_mode, row.st_nlink,
                                row.st_mtime_ns, row.st_ctime_ns)
        _require(identity(before) == identity(after) == identity(current) and len(raw) == before.st_size,
                 "retrieval input changed during bounded read")
        for directory, component, original in edges:
            current = os.stat(component, dir_fd=directory, follow_symlinks=False)
            _require((current.st_dev, current.st_ino, current.st_mode) ==
                     (original.st_dev, original.st_ino, original.st_mode), "retrieval input directory changed")
        current = os.stat(root, follow_symlinks=False)
        _require((current.st_dev, current.st_ino, current.st_mode) ==
                 (root_stat.st_dev, root_stat.st_ino, root_stat.st_mode), "retrieval repository changed")
        return raw
    finally:
        if file_fd is not None:
            os.close(file_fd)
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _parse(raw, maximum):
    _require(type(raw) is bytes and 0 < len(raw) <= maximum, "bounded retrieval JSON bytes required")
    def unique(items):
        result = {}
        for key, value in items:
            _require(key not in result, "duplicate retrieval JSON key")
            result[key] = value
        return result
    def invalid(_):
        raise ValueError("finite retrieval JSON required")
    try:
        value = json.loads(raw.decode(), object_pairs_hook=unique, parse_constant=invalid)
    except (UnicodeError, RecursionError) as error:
        raise ValueError("bounded UTF-8 retrieval JSON required") from error
    pending, nodes = [(value, 0)], 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        _require(depth <= 40 and nodes <= len(raw), "retrieval JSON structure exceeds bounded depth or population")
        if type(item) is dict:
            pending.extend((child, depth + 1) for child in item.values())
        elif type(item) is list:
            pending.extend((child, depth + 1) for child in item)
        elif type(item) is float:
            _require(math.isfinite(item), "finite retrieval JSON number required")
    _require(type(value) is dict, "retrieval JSON object required")
    return value


def _semantic_parent(root, binding, semantic_original_text, semantic_bindings, *, current):
    _require(type(binding) is SemanticRetrievalQueryBinding
             and type(semantic_bindings) is SemanticProjectionBindings
             and binding.task_id == semantic_bindings.task_id
             and type(semantic_original_text) is str
             and _sha(semantic_original_text.encode()) == semantic_bindings.expected_sha256,
             "independent semantic and retrieval parent/task bindings required")
    parent, _ = codec._verified_semantic(root, artifact=semantic_bindings.artifact,
        text=semantic_original_text, task_id=binding.task_id, current=current)
    _require(parent.get("schema") == "supervisor-semantic-worker-context@1"
             and parent.get("semantic_root_cid") == semantic_bindings.semantic_root_cid
             and parent.get("scope_cid") == semantic_bindings.scope_cid,
             "retrieval reference requires its complete bound native semantic parent")
    return parent


def _historical_material(root, binding, semantic_parent, raw):
    """Authenticate native persistence without reading current source files."""
    from . import code_retrieval_context as retrieval
    _require(_sha(raw) == binding.expected_sha256, "raw retrieval artifact digest differs")
    payload = _parse(raw, MAX_ARTIFACT_BYTES)
    referenced = payload.get("schema") == retrieval.REFERENCED_SCHEMA
    _require(set(payload) == {"schema", "task_id", "query_text", "source_sha256", "result",
                              "snapshot_ref" if referenced else "snapshot"}
             and payload.get("schema") in {retrieval.SCHEMA, retrieval.REFERENCED_SCHEMA}
             and payload.get("task_id") == binding.task_id,
             "raw retrieval task or schema differs")
    query = payload["query_text"]
    _require(type(query) is str and query.strip() and len(query.encode()) <= 8192,
             "bounded literal retrieval query required")
    held = [(binding.artifact, raw, MAX_ARTIFACT_BYTES)]
    if referenced:
        reference = payload["snapshot_ref"]
        _require(type(reference) is dict and set(reference) == {"path", "sha256", "bytes", "index_id"}
                 and type(reference["bytes"]) is int and 0 < reference["bytes"] <= MAX_SNAPSHOT_BYTES
                 and type(reference["sha256"]) is str
                 and re.fullmatch(r"[0-9a-f]{64}", reference["sha256"]) is not None,
                 "exact bounded native retrieval snapshot reference required")
        snapshot_raw = _read(root, reference["path"], reference["bytes"])
        _require(len(snapshot_raw) == reference["bytes"] and _sha(snapshot_raw) == reference["sha256"],
                 "native retrieval snapshot digest or byte count differs")
        snapshot = retrieval.CodeVectorIndexSnapshot.from_dict(_parse(snapshot_raw, MAX_SNAPSHOT_BYTES))
        _require(snapshot.index_id == reference["index_id"], "native retrieval snapshot identity differs")
        held.append((reference["path"], snapshot_raw, reference["bytes"]))
    else:
        snapshot = retrieval.CodeVectorIndexSnapshot.from_dict(payload["snapshot"])
    result = retrieval.CodeVectorSearchResult.from_dict(payload["result"])
    retrieval._replay(snapshot, result)
    hashes = payload["source_sha256"]
    manifest = semantic_parent["manifest"]
    _require(type(hashes) is dict and 1 <= len(hashes) <= 64
             and set(hashes) == set(snapshot.included_paths) and set(hashes) <= set(manifest),
             "retrieval source scope escapes or differs from the authenticated semantic parent")
    sources = {}
    blocks = PurePosixPath(semantic_parent["_bound_artifact"]).parent / "blocks"
    for path, digest in hashes.items():
        _path(path)
        source = manifest[path]
        _require(type(digest) is str and re.fullmatch(r"[0-9a-f]{64}", digest) is not None
                 and type(source) is dict and source.get("sha256") == digest
                 and type(source.get("source_cid")) is str
                 and PurePosixPath(source["source_cid"]).name == source["source_cid"],
                 "native retrieval source hash differs from retained semantic source")
        source_raw = codec._read(root, (blocks / source["source_cid"]).as_posix())
        _require(_sha(source_raw) == digest, "retained retrieval source bytes differ")
        sources[path] = source_raw.decode()
    retrieval._verified_sources(snapshot, sources)
    native = {
        "schema": retrieval.SCHEMA, "task_id": binding.task_id, "status": "current", "stale_paths": [],
        "index_id": snapshot.index_id, "query_id": result.query.query_id, "result_id": result.result_id,
        "query_text": query, "nomination_only": True, "semantic_authority": False,
        "execution_authority": False, "completion_authority": False, "source_sha256": hashes,
        "hits": [{"path": hit.row.path, "symbol": hit.row.qualified_symbol,
                  "line_start": hit.row.line_start, "line_end": hit.row.line_end,
                  "rank": hit.rank, "score": hit.score, "row_id": hit.row.row_id,
                  "ast_record_id": hit.row.sidecar.ast_record_id,
                  "signature_refs": hit.row.sidecar.signature_refs, "call_refs": hit.row.sidecar.call_refs,
                  "effect_refs": hit.row.sidecar.effect_refs} for hit in result.hits],
    }
    text = retrieval._json(native)
    _require(len(text.encode()) <= MAX_CONTEXT_BYTES, "native retrieval worker projection exceeds bound")
    return text, held


def _load_context(*, repository, binding, semantic_original_text, semantic_bindings, current):
    root = _root(repository)
    parent = _semantic_parent(root, binding, semantic_original_text, semantic_bindings, current=current)
    material_parent = {**parent, "_bound_artifact": semantic_bindings.artifact}
    raw = _read(root, binding.artifact, MAX_ARTIFACT_BYTES)
    expected, held = _historical_material(root, binding, material_parent, raw)
    if current:
        from .code_retrieval_context import load_code_retrieval_context
        actual = load_code_retrieval_context(repository=root, artifact=binding.artifact,
            expected_sha256=binding.expected_sha256, task_id=binding.task_id)
        _require(actual == expected, "current retrieval differs from its authenticated original source projection")
    for path, original, maximum in held:
        _require(_read(root, path, maximum) == original, "retrieval parent or snapshot changed during verification")
    closing = _semantic_parent(root, binding, semantic_original_text, semantic_bindings, current=current)
    _require(_encoded(closing) == _encoded(parent), "semantic source parent changed during retrieval verification")
    return expected


def load_live_retrieval_query_context(*, repository, binding, semantic_original_text, semantic_bindings) -> str:
    """Return exact native worker text after both current parent/source checks."""
    return _load_context(repository=repository, binding=binding, semantic_original_text=semantic_original_text,
                         semantic_bindings=semantic_bindings, current=True)


def load_historical_retrieval_query_context(*, repository, binding, semantic_original_text, semantic_bindings) -> str:
    """Rebuild the original CURRENT envelope; no current-source claim is made.

    Status is preserved from the authenticated native baseline. This historical
    observation cannot admit future dispatch or claim freshness of host source.
    """
    return _load_context(repository=repository, binding=binding, semantic_original_text=semantic_original_text,
                         semantic_bindings=semantic_bindings, current=False)


def _native(native_text, binding):
    _require(type(binding) is SemanticRetrievalQueryBinding and type(native_text) is str,
             "typed retrieval binding and complete native worker text required")
    raw = native_text.encode()
    native = _parse(raw, MAX_CONTEXT_BYTES)
    _require(_encoded(native) == raw and native.get("schema") == "supervisor-code-retrieval-context@1"
             and native.get("task_id") == binding.task_id
             and native.get("status") == "current" and native.get("stale_paths") == []
             and native.get("nomination_only") is True
             and all(native.get(key) is False for key in
                     ("semantic_authority", "execution_authority", "completion_authority"))
             and type(native.get("hits")) is list and len(native["hits"]) <= 20,
             "exact current native retrieval nomination envelope required")
    for name in ("index_id", "query_id", "result_id"):
        _require(type(native.get(name)) is str and native[name].strip(), "native retrieval identities required")
    return native


def validate_native_retrieval_references(*, wire, binding, native_text):
    """Validate complete ordered original chunks without changing requiredness."""
    _native(native_text, binding)
    _require(type(wire) is dict and type(wire.get("evidence")) is list
             and wire.get("objective_id") == binding.task_id,
             "compiled retrieval task and evidence group required")
    for name in ("repository_id", "tree_id"):
        _require(type(wire.get(name)) is str and wire[name].strip(), "compiled retrieval repository/tree required")
    positions = tuple(index for index, record in enumerate(wire["evidence"])
        if type(record) is dict and record.get("kind") == "code-retrieval-context"
        and record.get("path") == binding.artifact)
    _require(1 <= len(positions) <= 1024, "complete bound native retrieval group required")
    references = tuple(ContextReference.from_dict(wire["evidence"][index]) for index in positions)
    _require(len({row.reference_id for row in references}) == len(references), "duplicate native retrieval reference identity")
    artifact_cid = "sha256:" + _sha(_encoded({"text": native_text}))
    for index, (position, row) in enumerate(zip(positions, references, strict=True)):
        _require(row.kind == "code-retrieval-context" and row.path == binding.artifact
                 and row.repository_id == wire["repository_id"] and row.tree_id == wire["tree_id"]
                 and row.required and row.metadata.get("required") is True
                 and type(row.metadata.get("chunk_index")) is int and row.metadata["chunk_index"] == index
                 and type(row.metadata.get("chunk_count")) is int and row.metadata["chunk_count"] == len(references)
                 and row.metadata.get("artifact_content_id") == artifact_cid
                 and row.referenced_content_id == "sha256:" + _sha(row.summary.encode())
                 and row.byte_count == len(row.summary.encode())
                 and _encoded(row.to_record()) == _encoded(wire["evidence"][position]),
                 "native retrieval chunk requiredness, scope, ordinal, content or metadata binding differs")
    _require("".join(row.summary for row in references) == native_text,
             "native retrieval group does not contain its exact authenticated whole envelope")
    return positions, references


def _target(*, native_text, binding, semantic_view, semantic_original_text, semantic_bindings, policy):
    _require(type(binding) is SemanticRetrievalQueryBinding
             and type(policy) is SemanticDecisionPolicy and policy.task_text_source_path is not None
             and type(semantic_bindings) is SemanticProjectionBindings
             and semantic_bindings.task_id == binding.task_id,
             "retrieval query reference requires a bound explicit @2 semantic text policy")
    semantic = verify_semantic_decision_packet(view=semantic_view, original_text=semantic_original_text,
        bindings=semantic_bindings, policy=policy)
    _require(semantic.get("schema") == "supervisor-semantic-decision-packet@2",
             "required literal retrieval target must remain in an exact @2 semantic packet")
    native = _native(native_text, binding)
    context = semantic["required_context"]
    path = policy.task_text_source_path
    literal = context["raw_sources"].get(path)
    source = context["manifest"].get(path)
    _require(path in context["required_raw_paths"] and type(literal) is str and literal.strip()
             and type(source) is dict and source.get("sha256") == _sha(literal.encode())
             and type(native.get("query_text")) is str and native["query_text"].encode() == literal.encode(),
             "retrieval query must exactly duplicate its immediately available required raw literal")
    hashes = native.get("source_sha256")
    _require(type(hashes) is dict and 1 <= len(hashes) <= 64 and set(hashes) <= set(context["manifest"]),
             "retrieval source scope escapes its required semantic source parent")
    for path, digest in hashes.items():
        _path(path)
        _require(type(digest) is str and re.fullmatch(r"[0-9a-f]{64}", digest) is not None
                 and type(context["manifest"][path]) is dict
                 and context["manifest"][path].get("sha256") == digest,
                 "retrieval source hash differs from the required semantic source parent")
    return native, semantic, literal


def build_retrieval_query_reference(*, native_text, binding, semantic_view,
                                   semantic_original_text, semantic_bindings, policy):
    """Build only after the caller independently authenticates both parents."""
    native, semantic, literal = _target(native_text=native_text, binding=binding,
        semantic_view=semantic_view, semantic_original_text=semantic_original_text,
        semantic_bindings=semantic_bindings, policy=policy)
    material = {
        "origin": binding.to_dict(), "semantic_origin": semantic_bindings.to_dict(),
        "native_context_sha256": _sha(native_text.encode()),
        "required_context": {key: value for key, value in native.items() if key != "query_text"},
        "required_query_text_reference": {
            "field_selector": ["query_text"],
            "target_selector": ["required_context", "raw_sources", policy.task_text_source_path],
            "target_packet_sha256": _sha(_encoded(semantic)), "text_sha256": _sha(literal.encode()),
            "text_bytes": len(literal.encode()), "required": True,
        },
    }
    view = {
        "schema": SCHEMA, "projection_revision": PROJECTION_REVISION, **material,
        "required_coverage_sha256": _sha(_encoded(material)),
        "required_query_text_rules": "The exact query string is already in the required semantic packet at target_selector; use that literal without expansion.",
        "semantic_authority": False, "execution_authority": False, "completion_authority": False,
        "semantic_equivalence_claimed": False,
    }
    _require(len(_encoded(view)) <= MAX_VIEW_BYTES, "required retrieval reference view exceeds bound")
    return view


def verify_retrieval_query_reference(*, view, native_text, binding, semantic_view,
                                    semantic_original_text, semantic_bindings, policy):
    _structure(view, MAX_VIEW_BYTES)
    _require(len(_encoded(view)) <= MAX_VIEW_BYTES, "required retrieval reference view exceeds bound")
    expected = build_retrieval_query_reference(native_text=native_text, binding=binding,
        semantic_view=semantic_view, semantic_original_text=semantic_original_text,
        semantic_bindings=semantic_bindings, policy=policy)
    _require(_encoded(view) == _encoded(expected), "retrieval reference differs from required parent, target or coverage")
    return expected


def restore_retrieval_query_reference(*, view, native_text, binding, semantic_view,
                                     semantic_original_text, semantic_bindings, policy) -> str:
    verified = verify_retrieval_query_reference(view=view, native_text=native_text, binding=binding,
        semantic_view=semantic_view, semantic_original_text=semantic_original_text,
        semantic_bindings=semantic_bindings, policy=policy)
    _, _, literal = _target(native_text=native_text, binding=binding, semantic_view=semantic_view,
        semantic_original_text=semantic_original_text, semantic_bindings=semantic_bindings, policy=policy)
    restored = _encoded({**verified["required_context"], "query_text": literal}).decode()
    _require(restored == native_text and _sha(restored.encode()) == verified["native_context_sha256"],
             "exact complete native retrieval envelope restoration failed")
    return restored


def retrieval_query_reference_references(*, view, native_text, binding, semantic_view,
                                        semantic_original_text, semantic_bindings, policy,
                                        coverage_ids=(), chunk_bytes=6144):
    _require(type(chunk_bytes) is int and 1 <= chunk_bytes <= 16384,
             "bounded retrieval view chunk byte budget required")
    _require(type(coverage_ids) in {tuple, list} and len(coverage_ids) <= 1024
             and all(type(item) is str and item.strip() and len(item.encode()) <= 4096 for item in coverage_ids),
             "bounded explicit original retrieval coverage identifiers required")
    verified = verify_retrieval_query_reference(view=view, native_text=native_text, binding=binding,
        semantic_view=semantic_view, semantic_original_text=semantic_original_text,
        semantic_bindings=semantic_bindings, policy=policy)
    return build_text_context_references(_encoded(verified).decode(), reference_prefix="code-retrieval-query-reference",
        kind="code-retrieval-context", path=binding.artifact, repository_id=semantic_bindings.repository_id,
        tree_id=semantic_bindings.tree_id, required=True, chunk_bytes=chunk_bytes,
        coverage_ids=(*coverage_ids, "code-retrieval-parent:" + binding.expected_sha256,
                      "code-retrieval-required:" + verified["required_coverage_sha256"]))
