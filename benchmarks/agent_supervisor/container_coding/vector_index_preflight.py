"""Build, hydrate and query native lexical vectors over complete permitted files."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import time


def qualify(repository: Path, output: Path, paths: list[str], query: str) -> dict:
    import duckdb
    from ipfs_accelerate_py.agent_supervisor.analysis.program_ast_adapters import build_program_evidence_index
    from ipfs_accelerate_py.agent_supervisor.analysis.code_symbol_vector_index import (
        CodeVectorIndexSnapshot, build_code_symbol_vector_index,
        resolve_code_symbol_ast_facts, search_code_symbol_vector_index,
    )
    from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import content_identity
    from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import SupervisorMetaIndex

    repository = repository.resolve(strict=True)
    if not paths or len(paths) > 64 or len(set(paths)) != len(paths):
        raise ValueError("provide 1 to 64 unique permitted files")
    if output.exists():
        raise ValueError("output must be a new directory")
    sources = {}
    inventory = {}
    for name in paths:
        path = repository / name
        if (Path(name).is_absolute() or ".." in Path(name).parts or path.is_symlink()
                or not path.resolve().is_relative_to(repository)):
            raise ValueError("input escapes repository")
        raw = path.read_bytes()
        if len(raw) > 2_000_000:
            raise ValueError("input exceeds source byte bound")
        sources[name] = raw.decode("utf-8")
        inventory[name] = hashlib.sha256(raw).hexdigest()
    started = time.monotonic()
    evidence = build_program_evidence_index(sources)
    ast = evidence.ast_index
    if (not evidence.exhaustive or len(evidence.results) != len(paths)
            or {item.path for item in evidence.results} != set(paths)
            or any(item.status != "success" for item in evidence.results)
            or {item.path for item in ast.path_records} != set(paths)):
        raise ValueError("native AST coverage is incomplete for permitted files")
    docs = {}
    for indexed in ast.path_records:
        parts = list(PurePosixPath(indexed.path.removesuffix(".py")).parts)
        if parts and parts[-1] == "__init__":
            parts.pop()
        module = ".".join(parts)
        for symbol in indexed.ast_record.qualified_symbols:
            qualified = module + "." + symbol if module else symbol
            docs[qualified] = Counter(re.findall("[a-z0-9]+", qualified.lower()))
    if not docs:
        raise ValueError("native AST contains no qualified symbols")
    vocabulary = sorted({term for doc in docs.values() for term in doc})
    weights = {term: 1 + math.log((1 + len(docs)) / (1 + sum(term in doc for doc in docs.values())))
               for term in vocabulary}

    def vector(text):
        words = Counter(re.findall("[a-z0-9]+", text.lower()))
        values = [words[term] * weights[term] for term in vocabulary]
        norm = math.sqrt(sum(value * value for value in values))
        if not norm:
            return [0.0 for _ in values]
        return [value / norm for value in values]

    scope = content_identity({"schema": "permitted-vector-inputs@1", "sources": inventory})
    config = content_identity({"vocabulary": vocabulary,
                               "weights": {term: value.hex() for term, value in weights.items()}})
    snapshot = build_code_symbol_vector_index(
        ast, forest_id=scope, tree_id=scope, coverage_id=ast.index_id,
        dimensions=len(vocabulary), model_id="lexical-tfidf-symbols@1",
        model_revision="1", configuration_id=config,
        vectors=lambda row: vector(row.qualified_symbol),
    )
    build_seconds = time.monotonic() - started
    output.mkdir(parents=True)
    database = output / "vectors.duckdb"
    with duckdb.connect(str(database), config={"threads": 1}) as connection:
        connection.execute("CREATE TABLE snapshots(id VARCHAR PRIMARY KEY, payload JSON)")
        connection.execute("INSERT INTO snapshots VALUES (?, ?)", [snapshot.index_id, json.dumps(snapshot.to_dict())])
    reopen = time.monotonic()
    with duckdb.connect(str(database), read_only=True, config={"threads": 1}) as connection:
        restored = CodeVectorIndexSnapshot.from_dict(json.loads(connection.execute(
            "SELECT payload FROM snapshots WHERE id=?", [snapshot.index_id]).fetchone()[0]))
    if restored != snapshot:
        raise RuntimeError("hydrated native index differs")
    records = {record.path: record for record in ast.path_records}
    replayed = {row.row_id: resolve_code_symbol_ast_facts(records[row.path], row) for row in restored.rows}
    hits = search_code_symbol_vector_index(restored, vector(query), max_results=5)
    meta = SupervisorMetaIndex(output / "metadata.duckdb", output / "metadata-lake")
    catalog = meta.register_catalog(kind="vector", locator_ref=str(database), tree_id=scope, project=False)
    for offset in range(0, len(restored.rows), 10_000):
        meta.link_identities([
            dict(subject_kind="path", subject_ref=row.path, catalog_id=catalog["catalog_id"],
                 record_kind="code_symbol", record_ref=row.row_id)
            for row in restored.rows[offset:offset + 10_000]
        ], project=False)
    projected = meta.project_ducklake()
    if projected["status"] != "projected":
        raise RuntimeError("DuckLake metadata projection failed")
    metadata_hits = {path: meta.compose_for_subject(subject_kind="path", subject_ref=path) for path in paths}
    for path in paths:
        if records.get(path) is not None and records[path].ast_record.qualified_symbols and not metadata_hits[path]["n"]:
            raise RuntimeError("hydrated metadata omitted an indexed path")
    if any(hashlib.sha256((repository / path).read_bytes()).hexdigest() != digest for path, digest in inventory.items()):
        raise RuntimeError("source changed during vector qualification")
    result = {
        "schema": "native-lexical-vector-qualification@1", "status": "qualified",
        "source_sha256": inventory, "index_id": snapshot.index_id, "symbols": len(snapshot.rows),
        "complete_permitted_scope": True, "native_fact_rows_replayed": len(replayed),
        "input_statuses": {item.path: item.status for item in evidence.results},
        "aggregated_native_fact_references": sum(ref.startswith("code-ast-facts:sha256:")
            for row in snapshot.rows for ref in (*row.sidecar.signature_refs, *row.sidecar.call_refs, *row.sidecar.effect_refs)),
        "model": "lexical TF-IDF over qualified symbol names; no learned embedding model",
        "query": query, "hits": hits.to_dict(), "ducklake": projected,
        "metadata_retrieval": metadata_hits,
        "seconds": {"build": build_seconds, "reopen_replay_query": time.monotonic()-reopen},
        "semantic_authority": False, "full_system_qualified": False,
        "provider_calls": 0, "token_savings": None,
    }
    (output / "ast.json").write_text(json.dumps(ast.to_dict(), indent=2) + "\n")
    (output / "evidence.json").write_text(json.dumps(evidence.to_dict(), indent=2) + "\n")
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--file", action="append", required=True, dest="paths")
    parser.add_argument("--query", required=True)
    args = parser.parse_args()
    result = qualify(args.repository, args.output, args.paths, args.query)
    print(json.dumps({key: result[key] for key in ("status", "symbols", "native_fact_rows_replayed", "aggregated_native_fact_references", "seconds")}))
