"""Strict, isolated native AST/meta-index/DuckLake qualification.

Input must be an explicit list of permitted task files, never a benchmark root.
This qualifies storage and retrieval only, not a full supervisor benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


def qualify(workspace: Path, output: Path, paths: list[str]) -> dict:
    import duckdb
    from ipfs_accelerate_py.agent_supervisor.analysis.database_repository_indexer import DatabaseRepositoryIndexer
    from ipfs_accelerate_py.agent_supervisor.analysis.duckdb_ast_index import SourceFileSpec
    from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import SupervisorMetaIndex

    workspace, output = workspace.resolve(), output.resolve()
    if output.exists():
        raise ValueError("qualification requires a fresh output directory")
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("provide a nonempty, unique permitted-file list")
    files, inventory = [], []
    for name in sorted(paths):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe input path: {name}")
        source = workspace / relative
        if source.is_symlink() or not source.resolve().is_relative_to(workspace):
            raise ValueError(f"input escapes workspace: {name}")
        body = source.read_bytes()
        if len(body) > 2_000_000:
            raise ValueError(f"input exceeds qualification bound: {name}")
        inventory.append({"path": name, "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)})
        files.append(SourceFileSpec(path=name, content=body.decode("utf-8")))
    identity = hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True)
    started = time.monotonic()
    ast_path = output / "ast.duckdb"
    db_path = output / "repository.duckdb"
    with DatabaseRepositoryIndexer(db_path, ast_database_path=ast_path) as indexer:
        scan = indexer.full_scan(worktree_id=f"worktree:{identity}", repository_id=f"repo:{identity}", tree_id=f"tree:{identity}", files=files)
        if not scan.complete:
            raise RuntimeError("native repository scan did not complete")
        symbols = indexer.ast_index.list_symbols(scan.snapshot_id)
    build_seconds = time.monotonic() - started
    if not symbols:
        raise RuntimeError("qualification requires at least one indexed Python symbol")
    meta = SupervisorMetaIndex(output / "metadata.duckdb", output / "lake")
    catalog = meta.register_catalog(kind="ast", locator_ref=str(ast_path), repository_id=f"repo:{identity}", tree_id=f"tree:{identity}", project=False)
    for symbol in symbols:
        meta.link_identity(subject_kind="path", subject_ref=symbol.path, catalog_id=catalog["catalog_id"], record_kind="symbol", record_ref=symbol.symbol_id, project=False)
    projected = meta.project_ducklake()
    if projected["status"] != "projected":
        raise RuntimeError(f"DuckLake projection failed: {projected}")
    reopened_at = time.monotonic()
    with DatabaseRepositoryIndexer(db_path, ast_database_path=ast_path) as indexer:
        reopened_symbols = indexer.ast_index.list_symbols(scan.snapshot_id)
    if reopened_symbols != symbols:
        raise RuntimeError("persisted AST query differs after reopening")
    query = meta.compose_for_subject(subject_kind="path", subject_ref=symbols[0].path)
    if not query["n"]:
        raise RuntimeError("metadata retrieval returned no rows")
    # Reopen the actual lake independently; no mocked backend or placeholder rows.
    with duckdb.connect(":memory:", config={"threads": 1}) as connection:
        connection.execute("SET autoinstall_known_extensions=false")
        connection.execute("SET autoload_known_extensions=false")
        connection.execute("LOAD ducklake")
        lake = str(output / "lake" / "metadata.ducklake").replace("'", "''")
        connection.execute(f"ATTACH 'ducklake:{lake}' AS lake (READ_ONLY)")
        lake_counts = {table: connection.execute(f"SELECT count(*) FROM lake.{table}").fetchone()[0] for table in ("catalogs", "identity_links", "capsule_bindings")}
        refs = {row[0] for row in connection.execute("SELECT record_ref FROM lake.identity_links").fetchall()}
    if refs != {symbol.symbol_id for symbol in symbols}:
        raise RuntimeError("DuckLake lost or introduced symbol links")
    result = {
        "status": "qualified", "scope": "native_ast_metadata_ducklake_storage_and_query_only",
        "full_system_benchmark": False, "supervisor_consumed_retrieval": False,
        "ducklake_authoritative": False, "completion_authority": False,
        "inventory": inventory, "snapshot_id": scan.snapshot_id,
        "scan": scan.to_dict(), "symbols": len(symbols), "projection": projected,
        "reopened_lake_counts": lake_counts, "retrieval": query,
        "timings_seconds": {"ast_build": build_seconds, "reopen_and_query": time.monotonic() - reopened_at, "total": time.monotonic() - started},
        "unqualified_components": ["bm25", "vector", "knowledge_graph", "world_model", "proof_cache", "prompt_goal_task_pipeline", "worker_retrieval_consumption"],
    }
    (output / "qualification.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--file", action="append", required=True, dest="paths")
    args = parser.parse_args()
    result = qualify(args.workspace, args.output, args.paths)
    print(json.dumps({"status": result["status"], "symbols": result["symbols"], "receipt": str(args.output / "qualification.json")}))


if __name__ == "__main__":
    main()
