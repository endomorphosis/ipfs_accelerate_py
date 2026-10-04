"""Qualify native code-vector retrieval using an existing, pinned local model.

This command performs local CPU inference through embeddings_router. It does not
download models, enable supervisor authority, or claim benchmark improvement.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path, PurePosixPath
import time


def _identity(payload: object) -> str:
    from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
        content_identity,
    )

    return content_identity(payload)


from ipfs_accelerate_py.agent_supervisor.runtime.local_learned_embedding import (
    _model_manifest, _LocalRouterModel, _PinnedRouterBackend,
)


def qualify(
    repository: Path,
    output: Path,
    paths: list[str],
    query: str,
    model_snapshot: Path,
    model_revision: str,
) -> dict:
    import duckdb
    from ipfs_accelerate_py import embeddings_router
    from ipfs_accelerate_py.agent_supervisor.analysis.code_symbol_vector_index import (
        CodeVectorIndexSnapshot,
        build_code_symbol_vector_index,
        resolve_code_symbol_ast_facts,
        search_code_symbol_vector_index,
    )
    from ipfs_accelerate_py.agent_supervisor.analysis.program_ast_adapters import (
        build_program_evidence_index,
    )
    from ipfs_accelerate_py.agent_supervisor.integrations.ipfs_datasets_embedding_provider import (
        IpfsDatasetsEmbeddingProvider,
        PinnedEmbeddingPolicy,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import (
        SupervisorMetaIndex,
    )

    repository = repository.resolve(strict=True)
    model_snapshot = model_snapshot.resolve(strict=True)
    if output.exists():
        raise ValueError("output must be a new directory")
    if not paths or len(paths) > 64 or len(set(paths)) != len(paths):
        raise ValueError("provide 1 to 64 unique permitted files")
    if not model_revision.strip() or model_snapshot.name != model_revision:
        raise ValueError("model revision must equal the exact local snapshot directory")
    if not query.strip() or len(query.encode()) > 8192:
        raise ValueError("query must contain 1 to 8192 UTF-8 bytes")
    sources, inventory = {}, {}
    for name in paths:
        path = repository / name
        if (
            Path(name).is_absolute()
            or ".." in Path(name).parts
            or path.is_symlink()
            or not path.resolve().is_relative_to(repository)
        ):
            raise ValueError("input escapes repository")
        raw = path.read_bytes()
        if len(raw) > 2_000_000:
            raise ValueError("input exceeds source byte bound")
        sources[name] = raw.decode("utf-8")
        inventory[name] = hashlib.sha256(raw).hexdigest()
    started = time.monotonic()
    evidence = build_program_evidence_index(sources)
    ast = evidence.ast_index
    if (
        not evidence.exhaustive
        or len(evidence.results) != len(paths)
        or {result.path for result in evidence.results} != set(paths)
        or any(result.status != "success" for result in evidence.results)
        or set(ast.paths) != set(paths)
    ):
        raise ValueError("native AST coverage is incomplete for permitted inputs")
    symbols = set()
    for indexed in ast.path_records:
        parts = list(PurePosixPath(indexed.path.removesuffix(".py")).parts)
        if parts and parts[-1] == "__init__":
            parts.pop()
        module = ".".join(parts)
        for symbol in indexed.ast_record.qualified_symbols:
            symbols.add(f"{module}.{symbol}" if module else symbol)
    symbols = sorted(symbols)
    if not symbols:
        raise ValueError("native AST contains no qualified symbols")
    manifest = _model_manifest(model_snapshot)
    artifact_id = _identity(manifest)
    scope = _identity({"schema": "permitted-vector-inputs@1", "sources": inventory})
    model = _LocalRouterModel(model_snapshot, artifact_id)
    dimensions = int(model.model.get_embedding_dimension())
    versions = {
        name: importlib.metadata.version(name)
        for name in ("sentence-transformers", "transformers", "torch")
    }
    configuration = {
        "model_artifact_id": artifact_id,
        "model_revision": model_revision,
        "versions": versions,
        "local_files_only": True,
        "trust_remote_code": False,
        "use_safetensors": True,
        "device": "cpu",
        "max_seq_length": int(model.model.max_seq_length),
        "post_fixed_point_normalization": "l2",
        "max_row_bytes": 32768,
    }
    config_id = _identity(configuration)
    policy = PinnedEmbeddingPolicy(
        provider_id="ipfs_accelerate_py.embeddings_router.huggingface-local-pinned",
        model_artifact_id=artifact_id,
        model_revision=model_revision,
        dimensions=dimensions,
        chunker_id="qualified-symbol-name@1",
        normalizer="l2",
        distance="cosine",
        corpus_root_id=scope,
        index_root_id=ast.index_id,
        forest_id=scope,
        tree_id=scope,
        config_id=config_id,
        allow_remote=False,
    )
    backend = _PinnedRouterBackend(policy, model)
    provider = IpfsDatasetsEmbeddingProvider(policy, backend=backend)
    if not provider.vector_lane_enabled:
        raise RuntimeError(
            "native embedding canary failed: " + str(provider.canary_receipt.to_dict())
        )
    receipts = []

    def embed(texts):
        result = provider.embed(texts)
        if result.status.value != "completed":
            raise RuntimeError("native embedding request failed: " + str(result.reasons))
        receipts.append({"request_id": result.request_id, "receipt_id": result.content_id})
        # Native receipts use fixed-point numbers; index cosine checks need unit vectors.
        normalized = []
        for vector in result.vectors:
            norm = math.sqrt(sum(value * value for value in vector))
            normalized.append([value / norm for value in vector])
        return normalized

    vector_map = {}
    for start in range(0, len(symbols), 32):
        batch = symbols[start : start + 32]
        vector_map.update(zip(batch, embed(batch), strict=True))
    snapshot = build_code_symbol_vector_index(
        ast,
        forest_id=scope,
        tree_id=scope,
        coverage_id=ast.index_id,
        dimensions=dimensions,
        model_id=artifact_id,
        model_revision=model_revision,
        chunker_id=policy.chunker_id,
        configuration_id=policy.policy_id,
        vectors=vector_map,
        max_row_bytes=32768,
    )
    build_seconds = time.monotonic() - started
    output.mkdir(parents=True)
    database = output / "vectors.duckdb"
    with duckdb.connect(str(database), config={"threads": 1}) as connection:
        connection.execute("CREATE TABLE snapshots(id VARCHAR PRIMARY KEY, payload JSON)")
        connection.execute(
            "INSERT INTO snapshots VALUES (?, ?)",
            [snapshot.index_id, json.dumps(snapshot.to_dict())],
        )
    hydration_started = time.monotonic()
    with duckdb.connect(str(database), read_only=True, config={"threads": 1}) as connection:
        restored = CodeVectorIndexSnapshot.from_dict(
            json.loads(
                connection.execute(
                    "SELECT payload FROM snapshots WHERE id=?", [snapshot.index_id]
                ).fetchone()[0]
            )
        )
    if restored != snapshot:
        raise RuntimeError("hydrated native index differs")
    records = {record.path: record for record in ast.path_records}
    for row in restored.rows:
        resolve_code_symbol_ast_facts(records[row.path], row)
    query_started = time.monotonic()
    hits = search_code_symbol_vector_index(restored, embed([query])[0], max_results=5)
    query_seconds = time.monotonic() - query_started
    meta = SupervisorMetaIndex(output / "metadata.duckdb", output / "metadata-lake")
    catalog = meta.register_catalog(
        kind="vector", locator_ref=str(database), tree_id=scope, project=False
    )
    for offset in range(0, len(restored.rows), 10_000):
        meta.link_identities([
            dict(subject_kind="path", subject_ref=row.path, catalog_id=catalog["catalog_id"],
                 record_kind="code_symbol", record_ref=row.row_id)
            for row in restored.rows[offset:offset + 10_000]
        ], project=False)
    projected = meta.project_ducklake()
    if projected["status"] != "projected":
        raise RuntimeError("DuckLake metadata projection failed")
    metadata_hits = {
        path: meta.compose_for_subject(subject_kind="path", subject_ref=path) for path in paths
    }
    if any(not metadata_hits[row.path]["n"] for row in restored.rows):
        raise RuntimeError("hydrated metadata omitted an indexed path")
    if _model_manifest(model_snapshot) != manifest:
        raise RuntimeError("model artifacts changed during qualification")
    if any(
        hashlib.sha256((repository / path).read_bytes()).hexdigest() != digest
        for path, digest in inventory.items()
    ):
        raise RuntimeError("source changed during qualification")
    result = {
        "schema": "native-local-learned-vector-qualification@1",
        "status": "qualified",
        "source_sha256": inventory,
        "model_snapshot": str(model_snapshot),
        "model_artifact_id": artifact_id,
        "model_revision": model_revision,
        "versions": versions,
        "configuration": configuration,
        "policy": policy.to_dict(),
        "canary": provider.canary_receipt.to_dict(),
        "router_discovery": embeddings_router.resolve_model(provider="huggingface").to_dict(),
        "router_traces": backend.traces,
        "embedding_receipts": receipts,
        "dimensions": dimensions,
        "max_row_bytes": restored.max_row_bytes,
        "symbols": len(restored.rows),
        "index_id": restored.index_id,
        "complete_permitted_scope": True,
        "input_statuses": {result.path: result.status for result in evidence.results},
        "native_fact_rows_replayed": len(restored.rows),
        "query": query,
        "hits": hits.to_dict(),
        "ducklake": projected,
        "metadata_retrieval": metadata_hits,
        "seconds": {
            "build": build_seconds,
            "query": query_seconds,
            "hydrate_replay_query_metadata": time.monotonic() - hydration_started,
        },
        "local_model_calls": model.calls,
        "local_model_texts": model.texts,
        "remote_provider_calls": 0,
        "embedding_input": "qualified symbol names only",
        "learned_embeddings": True,
        "nomination_only": True,
        "semantic_authority": False,
        "full_system_qualified": False,
        "token_savings": None,
        "quality_advantage": None,
    }
    for name, payload in (
        ("model-manifest.json", manifest),
        ("evidence.json", evidence.to_dict()),
        ("ast.json", ast.to_dict()),
        ("result.json", result),
    ):
        (output / name).write_text(json.dumps(payload, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--file", action="append", required=True, dest="paths")
    parser.add_argument("--query", required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    result = qualify(**vars(parser.parse_args()))
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "symbols",
                    "dimensions",
                    "index_id",
                    "model_artifact_id",
                    "local_model_calls",
                    "remote_provider_calls",
                    "seconds",
                )
            }
        )
    )
