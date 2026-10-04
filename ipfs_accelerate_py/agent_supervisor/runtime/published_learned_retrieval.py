"""Source-only rebinding of an explicitly pinned, offline learned index.

The launch signs the initial native index, canary, configuration and local
safetensors inventory. Publication may replace source roots, never the model,
runtime versions, query, scope or vector policy. No fallback or downloads run.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.metadata
from itertools import chain
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat

from ..analysis.code_symbol_vector_index import (
    CodeVectorIndexSnapshot, CodeVectorSearchResult, build_code_symbol_vector_index,
    resolve_code_symbol_ast_facts, search_code_symbol_vector_index,
)
from ..analysis.program_ast_adapters import build_program_evidence_index
from ..integrations.ipfs_datasets_embedding_provider import (
    EmbeddingCanaryReceipt, IpfsDatasetsEmbeddingProvider, PinnedEmbeddingPolicy,
)
from ..proof.formal_verification_contracts import content_identity
from .local_learned_embedding import _model_manifest, _LocalRouterModel, _PinnedRouterBackend
from .published_retrieval import RetrievalRefreshUnavailable, _historical_retrieval_metadata
from .semantic_router_translation import _read
from . import code_retrieval_context as retrieval
from .task_context_bundle import load_task_context_nomination

POLICY = "local-safetensors-symbols@1"
SCHEMA = "supervisor-published-learned-retrieval-policy@1"
RECEIPT_SCHEMA = "supervisor-published-learned-embedding-receipt@1"
PROVIDER = "ipfs_accelerate_py.embeddings_router.huggingface-local-pinned"
FIELDS = frozenset({"schema", "policy", "task_cid", "task_id", "artifact", "sha256",
    "index_id", "config_id", "query_id", "query_sha256", "scope_paths", "result_artifact",
    "result_sha256", "manifest_artifact", "manifest_sha256", "model_snapshot", "model_artifact_id",
    "model_revision", "pinned_policy_id", "configuration_id", "canary_receipt_id",
    "completion_authority", "execution_authority", "learned_embeddings"})


class LearnedRetrievalUnavailable(RetrievalRefreshUnavailable):
    def __init__(self, reason, receipt=None):
        if reason not in {"learned_pins_unavailable", "learned_model_assets_changed",
                          "learned_runtime_changed", "learned_canary_failed",
                          "learned_embedding_failed", "learned_source_scan_incomplete"}:
            raise ValueError("unknown learned retrieval availability reason")
        self.reason, self.receipt = reason, receipt
        ValueError.__init__(self, reason)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _owner_model(snapshot, repository):
    """Only trusted assets outside model-writable task source may be loaded."""
    snapshot = Path(snapshot).absolute()
    if (snapshot.resolve(strict=True) != snapshot or not snapshot.is_dir()
            or snapshot.is_relative_to(repository) or repository.is_relative_to(snapshot)):
        raise ValueError("model must be a separate canonical owner-controlled directory")
    # A writable parent could replace an otherwise read-only model directory.
    # A root-owned sticky temporary root is safe for the owner's private child.
    for parent in snapshot.parents:
        info = parent.stat()
        sticky_root = info.st_uid == 0 and bool(info.st_mode & stat.S_ISVTX)
        if info.st_uid not in {0, os.geteuid()} or (info.st_mode & 0o022 and not sticky_root):
            raise ValueError("model ancestors must prevent worker replacement")
    count, total, items = 0, 0, 0
    for path in chain((snapshot,), snapshot.rglob("*")):
        items += 1
        info = path.lstat()
        if (path.is_symlink() or info.st_uid not in {0, os.geteuid()} or info.st_mode & 0o022
                or not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode))):
            raise ValueError("model assets must be owner-controlled regular files/directories")
        if stat.S_ISREG(info.st_mode):
            count += 1
            total += info.st_size
        if count > 64 or items > 128 or total > 512_000_000:
            raise ValueError("local model inventory exceeds bounds")
    return snapshot


def _versions():
    return {name: importlib.metadata.version(name) for name in ("sentence-transformers", "transformers", "torch")}


def _sources(repository, paths):
    if not 1 <= len(paths) <= 64:
        raise ValueError("learned source scope exceeds bounds")
    raw = {path: _read(repository, path) for path in paths}
    if sum(map(len, raw.values())) > 8_000_000:
        raise ValueError("learned source bytes exceed bounds")
    return {path: value.decode() for path, value in raw.items()}, {path: _sha(value) for path, value in raw.items()}


def _snapshot_bytes(root, relative, limit):
    path = PurePosixPath(relative)
    if path.is_absolute() or str(path) != relative or ".." in path.parts or not path.parts:
        raise ValueError("noncanonical learned snapshot path")
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    leaf = None
    try:
        for name in path.parts[:-1]:
            child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        leaf = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        before = os.fstat(leaf)
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise ValueError("learned snapshot exceeds file bounds")
        with os.fdopen(leaf, "rb", closefd=False) as stream:
            raw = stream.read(limit + 1)
        after = os.fstat(leaf)
        if len(raw) > limit or any(getattr(before, key) != getattr(after, key)
                for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")):
            raise ValueError("learned snapshot changed during read")
        return raw
    finally:
        if leaf is not None:
            os.close(leaf)
        os.close(fd)


def _previous(repository, metadata, task_id):
    """Replay native objects using descriptor reads even for stale sources."""
    raw = _read(repository, metadata["code retrieval artifact"])
    if _sha(raw) != metadata["code retrieval sha256"]:
        raise ValueError("learned retrieval nomination changed")
    payload = json.loads(raw)
    referenced = payload.get("schema") == retrieval.REFERENCED_SCHEMA
    if (set(payload) != {"schema", "task_id", "query_text", "source_sha256", "result",
                        "snapshot_ref" if referenced else "snapshot"}
            or payload["schema"] not in {retrieval.SCHEMA, retrieval.REFERENCED_SCHEMA}
            or payload["task_id"] != task_id):
        raise ValueError("learned predecessor task/schema differs")
    if referenced:
        ref = payload["snapshot_ref"]
        if (set(ref) != {"path", "sha256", "bytes", "index_id"} or type(ref["bytes"]) is not int
                or not 0 < ref["bytes"] <= retrieval.MAX_SNAPSHOT_BYTES):
            raise ValueError("learned predecessor snapshot bound differs")
        snapshot_raw = _snapshot_bytes(repository, ref["path"], ref["bytes"])
        if len(snapshot_raw) != ref["bytes"] or _sha(snapshot_raw) != ref["sha256"]:
            raise ValueError("learned predecessor snapshot digest differs")
        snapshot = CodeVectorIndexSnapshot.from_dict(json.loads(snapshot_raw))
        if snapshot.index_id != ref["index_id"]:
            raise ValueError("learned predecessor snapshot identity differs")
    else:
        snapshot = CodeVectorIndexSnapshot.from_dict(payload["snapshot"])
    hits = CodeVectorSearchResult.from_dict(payload["result"])
    retrieval._replay(snapshot, hits)
    sources, hashes = _sources(repository, snapshot.included_paths)
    if set(payload["source_sha256"]) != set(hashes):
        raise ValueError("learned predecessor source inventory differs")
    stale = payload["source_sha256"] != hashes
    if not stale:
        retrieval._verified_sources(snapshot, sources)
    return {"status": "stale" if stale else "current", "query_text": payload["query_text"],
            "source_sha256": payload["source_sha256"]}, snapshot, hits


def _load_pins(repository, metadata, task_id, artifacts):
    from .published_task_context import _verify_retrieval_rebuild, PinnedRetrievalRebuild
    if not isinstance(artifacts, dict) or set(artifacts) != {"result", "manifest", "model_snapshot"}:
        raise ValueError("exact learned result, manifest and model snapshot nominations required")
    result_raw = _read(repository, artifacts["result"])
    manifest_raw = _read(repository, artifacts["manifest"])
    result, manifest = json.loads(result_raw), json.loads(manifest_raw)
    previous = _previous(repository, metadata, task_id)
    if previous is None:
        raise ValueError("learned predecessor retrieval is absent")
    context, snapshot, hits = previous
    policy = PinnedEmbeddingPolicy.from_dict(result["policy"])
    canary = EmbeddingCanaryReceipt.from_dict(result["canary"])
    configuration = result["configuration"]
    expected_configuration_keys = {"model_artifact_id", "model_revision", "versions", "local_files_only",
        "trust_remote_code", "use_safetensors", "device", "max_seq_length", "post_fixed_point_normalization", "max_row_bytes"}
    if (result.get("schema") != "native-local-learned-vector-qualification@1"
            or result.get("status") != "qualified" or result.get("complete_permitted_scope") is not True
            or result.get("learned_embeddings") is not True or result.get("nomination_only") is not True
            or result.get("semantic_authority") is not False or result.get("full_system_qualified") is not False
            or type(result.get("remote_provider_calls")) is not int or result["remote_provider_calls"] != 0
            or result.get("index_id") != snapshot.index_id
            or CodeVectorSearchResult.from_dict(result["hits"]) != hits
            or result.get("query") != context["query_text"]
            or result.get("source_sha256") != context["source_sha256"]
            or policy.provider_id != PROVIDER or policy.chunker_id != "qualified-symbol-name@1"
            or policy.normalizer != "l2" or policy.distance != "cosine"
            or policy.allow_remote or policy.remote_endpoint_id
            or manifest.get("schema") != "local-embedding-model-manifest@1"
            or content_identity(manifest) != policy.model_artifact_id
            or result.get("model_artifact_id") != policy.model_artifact_id
            or result.get("model_revision") != policy.model_revision
            or not isinstance(configuration, dict) or set(configuration) != expected_configuration_keys
            or content_identity(configuration) != policy.config_id
            or configuration["model_artifact_id"] != policy.model_artifact_id
            or configuration["model_revision"] != policy.model_revision
            or configuration["local_files_only"] is not True or configuration["trust_remote_code"] is not False
            or configuration["use_safetensors"] is not True or configuration["device"] != "cpu"
            or configuration["post_fixed_point_normalization"] != "l2"
            or configuration["max_row_bytes"] != snapshot.max_row_bytes
            or canary.policy_id != policy.policy_id or canary.disposition.value != "passed"
            or canary.vector_lane.value != "enabled" or canary.observed_dimensions != policy.dimensions
            or canary.backend_kind != _PinnedRouterBackend.kind or canary.sample_count != 3
            or not 1 <= len(snapshot.rows) <= 4096):
        raise ValueError("learned native policy/configuration/canary differs from predecessor")
    _verify_retrieval_rebuild(PinnedRetrievalRebuild(snapshot, hits, policy, policy), snapshot, hits)
    return context, snapshot, hits, policy, configuration, manifest, canary, result_raw, manifest_raw


def bind_published_learned_retrieval_policy(*, repository, bundle, task_cid, task_id, artifacts):
    root = Path(repository).resolve(strict=True)
    metadata = load_task_context_nomination(repository=root, artifact=bundle["artifact"],
        expected_sha256=bundle["sha256"], task_cid=task_cid, task_id=task_id)
    context, snapshot, hits, policy, config, manifest, canary, result_raw, manifest_raw = _load_pins(root, metadata, task_id, artifacts)
    model = _owner_model(artifacts["model_snapshot"], root)
    if context["status"] != "current" or model.name != policy.model_revision:
        raise ValueError("learned launch requires current sources and exact model revision")
    if _model_manifest(model) != manifest or _versions() != config["versions"]:
        raise ValueError("learned local model or runtime differs from retained pins")
    return {"schema": SCHEMA, "policy": POLICY, "task_cid": task_cid, "task_id": task_id,
        "artifact": metadata["code retrieval artifact"], "sha256": metadata["code retrieval sha256"],
        "index_id": snapshot.index_id, "config_id": snapshot.config.config_id,
        "query_id": hits.query.query_id, "query_sha256": _sha(context["query_text"].encode()),
        "scope_paths": list(snapshot.included_paths), "result_artifact": artifacts["result"],
        "result_sha256": _sha(result_raw), "manifest_artifact": artifacts["manifest"],
        "manifest_sha256": _sha(manifest_raw), "model_snapshot": str(model),
        "model_artifact_id": policy.model_artifact_id, "model_revision": policy.model_revision,
        "pinned_policy_id": policy.policy_id, "configuration_id": policy.config_id,
        "canary_receipt_id": canary.content_id, "execution_authority": False,
        "completion_authority": False, "learned_embeddings": True}


def published_learned_retrieval_rebuilder(*, repository, bundle, binding):
    from .published_task_context import PinnedRetrievalRebuild
    root = Path(repository).resolve(strict=True)
    if (not isinstance(binding, dict) or set(binding) != FIELDS or binding["schema"] != SCHEMA
            or binding["policy"] != POLICY or binding["learned_embeddings"] is not True
            or binding["execution_authority"] is not False or binding["completion_authority"] is not False):
        raise ValueError("invalid signed learned retrieval policy")
    # Copy only immutable serialized declarations into the callback closure.
    binding, bundle = json.loads(_json(binding)), json.loads(_json(bundle))

    def rebuild(*, repository, previous_snapshot, previous_result, query_text, output):
        if Path(repository).resolve(strict=True) != root:
            raise ValueError("learned rebuild belongs to a different repository")
        output = Path(output).absolute()
        if output.exists() or output.resolve() != output or not output.is_relative_to(root / ".runtime"):
            raise ValueError("learned refresh output must be a new repository runtime directory")
        model, backend, provider, current_policy = None, None, None, None
        receipts = []
        canary_observation = {"local_embedding_calls": 0, "local_embedding_texts": 0}

        def observation(status):
            return {"schema": RECEIPT_SCHEMA, "status": status,
                "previous_index_id": previous_snapshot.index_id,
                "previous_policy_id": binding["pinned_policy_id"],
                "current_policy_id": current_policy.policy_id if current_policy else None,
                "model_artifact_id": binding["model_artifact_id"],
                "model_revision": binding["model_revision"],
                "local_embedding_calls": model.calls if model is not None else 0,
                "local_embedding_texts": model.texts if model is not None else 0,
                "remote_embedding_calls": 0, "text_generation_calls": 0,
                "canary": provider.canary_receipt.to_dict() if provider else None,
                "canary_observation": dict(canary_observation),
                "embedding_receipts": list(receipts), "execution_authority": False,
                "semantic_authority": False, "completion_authority": False}

        reason = "learned_pins_unavailable"
        try:
            metadata = _historical_retrieval_metadata(repository=root, bundle=bundle,
                task_cid=binding["task_cid"], task_id=binding["task_id"])
            artifacts = {"result": binding["result_artifact"], "manifest": binding["manifest_artifact"],
                         "model_snapshot": binding["model_snapshot"]}
            context, original, original_hits, old_policy, config, manifest, canary, raw, manifest_raw = _load_pins(root, metadata, binding["task_id"], artifacts)
            if (original != previous_snapshot or original_hits != previous_result
                    or original.index_id != binding["index_id"] or original.config.config_id != binding["config_id"]
                    or metadata["code retrieval artifact"] != binding["artifact"]
                    or metadata["code retrieval sha256"] != binding["sha256"]
                    or _sha(raw) != binding["result_sha256"] or _sha(manifest_raw) != binding["manifest_sha256"]
                    or old_policy.policy_id != binding["pinned_policy_id"] or old_policy.config_id != binding["configuration_id"]
                    or old_policy.model_artifact_id != binding["model_artifact_id"]
                    or old_policy.model_revision != binding["model_revision"]
                    or canary.content_id != binding["canary_receipt_id"]
                    or original_hits.query.query_id != binding["query_id"]
                    or list(original.included_paths) != binding["scope_paths"]
                    or query_text != context["query_text"] or _sha(query_text.encode()) != binding["query_sha256"]):
                raise ValueError("learned rebuild differs from signed predecessor/query")
            reason = "learned_model_assets_changed"
            model_path = _owner_model(binding["model_snapshot"], root)
            if model_path.name != old_policy.model_revision or _model_manifest(model_path) != manifest:
                raise ValueError("learned model artifacts changed")
            reason = "learned_runtime_changed"
            if _versions() != config["versions"]:
                raise ValueError("learned runtime versions changed")
            reason = "learned_source_scan_incomplete"
            sources, hashes = _sources(root, original.included_paths)
            evidence = build_program_evidence_index(sources)
            ast = evidence.ast_index
            if (not evidence.exhaustive or len(evidence.results) != len(sources)
                    or {row.path for row in evidence.results} != set(sources)
                    or any(row.status != "success" for row in evidence.results) or set(ast.paths) != set(sources)):
                raise ValueError("learned source scan is incomplete")
            symbols = set()
            for indexed in ast.path_records:
                parts = list(PurePosixPath(indexed.path.removesuffix(".py")).parts)
                if parts and parts[-1] == "__init__":
                    parts.pop()
                module = ".".join(parts)
                symbols.update(f"{module}.{name}" if module else name for name in indexed.ast_record.qualified_symbols)
            if not 1 <= len(symbols) <= 4096:
                raise ValueError("learned symbol inventory exceeds bounds")
            scope = content_identity({"schema": "permitted-vector-inputs@1", "sources": hashes})
            current_policy = replace(old_policy, corpus_root_id=scope, index_root_id=ast.index_id,
                                     forest_id=scope, tree_id=scope)
            reason = "learned_embedding_failed"
            model = _LocalRouterModel(model_path, old_policy.model_artifact_id)
            if int(model.model.get_embedding_dimension()) != old_policy.dimensions or int(model.model.max_seq_length) != config["max_seq_length"]:
                raise ValueError("loaded learned model configuration changed")
            backend = _PinnedRouterBackend(current_policy, model)
            reason = "learned_canary_failed"
            before_calls, before_texts = model.calls, model.texts
            try:
                provider = IpfsDatasetsEmbeddingProvider(current_policy, backend=backend)
            finally:
                canary_observation.update(local_embedding_calls=model.calls - before_calls,
                                          local_embedding_texts=model.texts - before_texts)
            if not provider.vector_lane_enabled:
                raise ValueError("learned model canary failed")
            reason = "learned_embedding_failed"
            def embed(texts):
                before_calls, before_texts, value = model.calls, model.texts, None
                try:
                    value = provider.embed(texts)
                finally:
                    receipts.append({"request_id": value.request_id if value else None,
                        "receipt_id": value.content_id if value else None,
                        "status": value.status.value if value else "error", "text_count": len(texts),
                        "local_embedding_calls": model.calls - before_calls,
                        "local_embedding_texts": model.texts - before_texts})
                if value.status.value != "completed":
                    raise ValueError("native learned embedding failed")
                return [tuple(x / math.sqrt(sum(y * y for y in vector)) for x in vector) for vector in value.vectors]
            vectors = {}
            ordered = sorted(symbols)
            for offset in range(0, len(ordered), 32):
                batch = ordered[offset:offset + 32]
                vectors.update(zip(batch, embed(batch), strict=True))
            snapshot = build_code_symbol_vector_index(ast, forest_id=scope, tree_id=scope,
                coverage_id=ast.index_id, included_paths=original.included_paths, excluded_paths=original.excluded_paths,
                dimensions=current_policy.dimensions, model_id=current_policy.model_artifact_id,
                model_revision=current_policy.model_revision, producer_id=original.config.producer_id,
                chunker_id=current_policy.chunker_id, normalization=current_policy.normalizer,
                metric=current_policy.distance, configuration_id=current_policy.policy_id,
                vectors=vectors, previous=original, max_row_bytes=original.max_row_bytes)
            hits = search_code_symbol_vector_index(snapshot, embed([query_text])[0], max_results=original_hits.query.max_results)
            records = {item.path: item for item in ast.path_records}
            for row in snapshot.rows:
                resolve_code_symbol_ast_facts(records[row.path], row)
            if (_model_manifest(_owner_model(model_path, root)) != manifest or _versions() != config["versions"]
                    or _sources(root, original.included_paths)[1] != hashes):
                raise ValueError("learned source/model/runtime changed during reconstruction")
            output.mkdir(parents=True)
            snapshot_path = output / "snapshot.json"
            snapshot_path.write_text(_json(snapshot.to_dict()))
            restored = CodeVectorIndexSnapshot.from_dict(json.loads(snapshot_path.read_text()))
            if restored != snapshot or search_code_symbol_vector_index(restored, hits.query) != hits:
                raise ValueError("persisted learned index/query replay differs")
            receipt = {**observation("completed"), "index_id": snapshot.index_id,
                "result_id": hits.result_id, "source_sha256": hashes,
                "native_fact_rows_replayed": len(snapshot.rows)}
            (output / "embedding-receipt.json").write_text(_json(receipt))
            return PinnedRetrievalRebuild(restored, hits, old_policy, current_policy, embedding_receipt=receipt)
        except (OSError, ValueError, TypeError, KeyError, ImportError, RuntimeError, ArithmeticError) as error:
            raise LearnedRetrievalUnavailable(reason, observation("unavailable")) from error
    return rebuild
