"""Replay source-bound code retrieval into bounded, advisory worker context.

The producer supplies an actual native vector snapshot and query result. This
boundary proves their consistency and source freshness, not embedding quality,
plan admission, or completion. Changed source yields an explicit stale envelope
with no hits; it never silently reuses old recommendations as current evidence.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from ..analysis.code_symbol_vector_index import (
    CodeVectorIndexSnapshot, CodeVectorSearchResult,
    resolve_code_symbol_ast_facts, search_code_symbol_vector_index,
)
from ..analysis.program_ast_adapters import build_program_evidence_index


SCHEMA = "supervisor-code-retrieval-context@1"
REFERENCED_SCHEMA = "supervisor-code-retrieval-artifact@2"
MAX_ARTIFACT_BYTES = 2_000_000
MAX_SNAPSHOT_BYTES = 64_000_000


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _read(path, limit):
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("retrieval file exceeds byte bound")
    return raw


def _path(root: Path, name: str) -> Path:
    relative = Path(name)
    path = root / relative
    if (not isinstance(name, str) or not name or relative.as_posix() != name
            or relative.is_absolute() or ".." in relative.parts
            or path.is_symlink() or not path.resolve().is_relative_to(root)):
        raise ValueError("retrieval path escapes its canonical repository scope")
    return path


def _sources(root, paths, *, allow_missing=False):
    if not 1 <= len(paths) <= 64:
        raise ValueError("retrieval source scope must contain 1 to 64 paths")
    sources, hashes = {}, {}
    for name in paths:
        path = _path(root, name)
        if allow_missing and not path.exists():
            hashes[name] = None
            continue
        if path.stat().st_size > 2_000_000:
            raise ValueError("retrieval source exceeds byte bound")
        raw = _read(path, 2_000_000)
        sources[name] = raw.decode("utf-8")
        hashes[name] = hashlib.sha256(raw).hexdigest()
    return sources, hashes


def _verified_sources(snapshot, sources):
    evidence = build_program_evidence_index(sources)
    ast = evidence.ast_index
    if (not evidence.exhaustive or any(item.status != "success" for item in evidence.results)
            or {item.path for item in evidence.results} != set(sources)
            or set(ast.paths) != set(sources) or ast.index_id != snapshot.ast_index_id):
        raise ValueError("retrieval index differs from complete native source scan")
    records = {item.path: item for item in ast.path_records}
    expected = {(item.path, symbol) for item in ast.path_records
                for symbol in item.ast_record.qualified_symbols}
    if (len(snapshot.rows) != len(expected)
            or {(row.path, row.symbol) for row in snapshot.rows} != expected):
        raise ValueError("retrieval index omits or duplicates native symbols")
    for row in snapshot.rows:
        resolve_code_symbol_ast_facts(records[row.path], row)


def _replay(snapshot, result):
    if result.query.max_results > 20:
        raise ValueError("worker retrieval exceeds 20 results")
    replayed = search_code_symbol_vector_index(snapshot, result.query)
    if replayed != result:
        raise ValueError("retrieval result does not replay against its native index")


def prepare_code_retrieval_context(*, repository: Path, task_id: str, query_text: str,
                                   snapshot: CodeVectorIndexSnapshot,
                                   result: CodeVectorSearchResult, output: Path) -> dict:
    root = Path(repository).resolve(strict=True)
    if not isinstance(task_id, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,256}", task_id):
        raise ValueError("retrieval task identity is invalid")
    if not isinstance(query_text, str) or not query_text.strip() or len(query_text.encode()) > 8192:
        raise ValueError("retrieval query text is outside bounds")
    if not isinstance(snapshot, CodeVectorIndexSnapshot) or not isinstance(result, CodeVectorSearchResult):
        raise TypeError("native retrieval objects are required")
    _replay(snapshot, result)
    sources, hashes = _sources(root, snapshot.included_paths)
    _verified_sources(snapshot, sources)
    output = Path(output).absolute()
    if not output.is_relative_to(root) or output.resolve() != output or output.exists():
        raise ValueError("retrieval output must be a new repository-contained file")
    payload = {"schema": SCHEMA, "task_id": task_id, "query_text": query_text,
               "source_sha256": hashes, "snapshot": snapshot.to_dict(), "result": result.to_dict()}
    raw = _json(payload).encode()
    snapshot_path = None
    snapshot_raw = b""
    if len(raw) > MAX_ARTIFACT_BYTES:
        # Full indexes are persistence, not worker context. Keep the bounded
        # task envelope small while binding the complete immutable native index.
        snapshot_raw = _json(payload.pop("snapshot")).encode()
        if len(snapshot_raw) > MAX_SNAPSHOT_BYTES:
            raise ValueError("retrieval snapshot exceeds byte bound")
        snapshot_digest = hashlib.sha256(snapshot_raw).hexdigest()
        snapshot_path = output.with_name(output.name + ".index-" + snapshot_digest[:16] + ".json")
        if snapshot_path.exists():
            raise ValueError("retrieval snapshot output already exists")
        payload["schema"] = REFERENCED_SCHEMA
        payload["snapshot_ref"] = {
            "path": snapshot_path.relative_to(root).as_posix(), "sha256": snapshot_digest,
            "bytes": len(snapshot_raw), "index_id": snapshot.index_id,
        }
        raw = _json(payload).encode()
    if len(raw) > MAX_ARTIFACT_BYTES:
        raise ValueError("retrieval artifact exceeds byte bound")
    if _sources(root, snapshot.included_paths)[1] != hashes:
        raise ValueError("retrieval source changed during preparation")
    output.parent.mkdir(parents=True, exist_ok=True)
    if snapshot_path is not None:
        with snapshot_path.open("xb") as stream:
            stream.write(snapshot_raw)
    with output.open("xb") as stream:
        stream.write(raw)
    digest = hashlib.sha256(raw).hexdigest()
    return {"schema": SCHEMA, "sha256": digest, "index_id": snapshot.index_id,
            "query_id": result.query.query_id, "result_id": result.result_id,
            "metadata": {"Code retrieval artifact": output.relative_to(root).as_posix(),
                         "Code retrieval sha256": digest},
            "execution_authority": False, "completion_authority": False}


def load_code_retrieval_context(*, repository: Path, artifact: str, expected_sha256: str,
                                task_id: str) -> str:
    root = Path(repository).resolve(strict=True)
    path = _path(root, artifact)
    if path.stat().st_size > MAX_ARTIFACT_BYTES:
        raise ValueError("retrieval artifact exceeds byte bound")
    raw = _read(path, MAX_ARTIFACT_BYTES)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("retrieval artifact digest differs")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate retrieval artifact key")
            result[key] = value
        return result

    payload = json.loads(raw, object_pairs_hook=unique)
    referenced = isinstance(payload, dict) and payload.get("schema") == REFERENCED_SCHEMA
    if (not isinstance(payload, dict)
            or set(payload) != {"schema", "task_id", "query_text", "source_sha256", "result",
                                "snapshot_ref" if referenced else "snapshot"}
            or payload["schema"] not in {SCHEMA, REFERENCED_SCHEMA} or payload["task_id"] != task_id):
        raise ValueError("retrieval task or schema differs")
    if (not isinstance(payload["query_text"], str) or not payload["query_text"].strip()
            or len(payload["query_text"].encode()) > 8192):
        raise ValueError("retrieval query text is outside bounds")
    if referenced:
        ref = payload["snapshot_ref"]
        if (not isinstance(ref, dict) or set(ref) != {"path", "sha256", "bytes", "index_id"}
                or type(ref["bytes"]) is not int or not 0 < ref["bytes"] <= MAX_SNAPSHOT_BYTES
                or not isinstance(ref["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", ref["sha256"])):
            raise ValueError("retrieval snapshot reference differs")
        snapshot_raw = _read(_path(root, ref["path"]), ref["bytes"])
        if len(snapshot_raw) != ref["bytes"] or hashlib.sha256(snapshot_raw).hexdigest() != ref["sha256"]:
            raise ValueError("retrieval snapshot digest differs")
        snapshot = CodeVectorIndexSnapshot.from_dict(json.loads(snapshot_raw, object_pairs_hook=unique))
        if snapshot.index_id != ref["index_id"]:
            raise ValueError("retrieval snapshot identity differs")
    else:
        snapshot = CodeVectorIndexSnapshot.from_dict(payload["snapshot"])
    result = CodeVectorSearchResult.from_dict(payload["result"])
    _replay(snapshot, result)
    sources, hashes = _sources(root, snapshot.included_paths, allow_missing=True)
    original = payload["source_sha256"]
    if (not isinstance(original, dict) or set(original) != set(hashes)
            or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
                   for value in original.values())):
        raise ValueError("retrieval source manifest differs")
    stale = sorted(name for name in hashes if hashes[name] != original[name])
    if not stale:
        _verified_sources(snapshot, sources)
    if _sources(root, snapshot.included_paths, allow_missing=True)[1] != hashes:
        raise ValueError("retrieval source changed during dispatch observation")
    context = {"schema": SCHEMA, "task_id": task_id,
               "status": "stale" if stale else "current", "stale_paths": stale,
               "index_id": snapshot.index_id, "query_id": result.query.query_id,
               "result_id": result.result_id, "query_text": payload["query_text"],
               "nomination_only": True, "semantic_authority": False,
               "execution_authority": False, "completion_authority": False,
               "source_sha256": hashes, "hits": []}
    if not stale:
        context["hits"] = [{"path": hit.row.path, "symbol": hit.row.qualified_symbol,
                            "line_start": hit.row.line_start, "line_end": hit.row.line_end,
                            "rank": hit.rank, "score": hit.score, "row_id": hit.row.row_id,
                            "ast_record_id": hit.row.sidecar.ast_record_id,
                            "signature_refs": hit.row.sidecar.signature_refs,
                            "call_refs": hit.row.sidecar.call_refs,
                            "effect_refs": hit.row.sidecar.effect_refs} for hit in result.hits]
    text = _json(context)
    if len(text.encode()) > 65_536:
        raise ValueError("retrieval worker projection exceeds byte bound")
    return text
