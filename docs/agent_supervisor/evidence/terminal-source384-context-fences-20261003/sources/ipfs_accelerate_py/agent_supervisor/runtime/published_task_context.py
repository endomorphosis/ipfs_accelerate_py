"""Refresh advisory context after an actual owner-validated publication.

The immutable successor binds a completed native task, its retained receipt,
the owner's accepted source, and a fresh semantic/world capture. It does not
settle claims, alter tasks, admit a plan, or modify the launch nomination.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import re
import time

from ..proof.formal_verification_contracts import content_identity
from ..semantic_state.datasets_adapter import IpfsDatasetsSemanticStateProvider
from ..semantic_state.intent_world_snapshot import (
    capture_intent_world_snapshot, load_intent_world_context,
    persist_intent_world_snapshot,
)
from ..task_sources.intent_repository import IntentRepository
from . import code_retrieval_context as retrieval_context
from . import local_planning_admission as local
from .local_completion_bridge import _git, verify_owner_local_benchmark_observation
from .quack_state_server import QuackStateServer
from .semantic_context_runtime import (
    _load_semantic_payload, load_semantic_worker_context, prepare_semantic_context,
)
from .task_context_bundle import (
    load_task_context_nomination, read_task_context_historical_selection, write_task_context_bundle,
)


SCHEMA = "supervisor-published-task-context@1"


@dataclass(frozen=True)
class PinnedRetrievalRebuild:
    """Native objects plus policies proving only corpus/index roots changed."""

    snapshot: object
    result: object
    previous_policy: object
    current_policy: object
    embedding_receipt: dict | None = None


def _verify_embedding_observation(receipt, previous, *, completed=False):
    """Check producer-observed accounting; it conveys no execution authority."""
    counters = ("local_embedding_calls", "local_embedding_texts", "remote_embedding_calls", "text_generation_calls")
    if (not isinstance(receipt, dict)
            or receipt.get("schema") != "supervisor-published-learned-embedding-receipt@1"
            or receipt.get("status") != ("completed" if completed else "unavailable")
            or receipt.get("previous_index_id") != previous.index_id
            or receipt.get("previous_policy_id") != previous.config.configuration_id
            or receipt.get("model_artifact_id") != previous.config.model_id
            or receipt.get("model_revision") != previous.config.model_revision
            or any(receipt.get(key) is not False for key in ("semantic_authority", "execution_authority", "completion_authority"))
            or any(type(receipt.get(key)) is not int or receipt[key] < 0 for key in counters)
            or receipt["remote_embedding_calls"] or receipt["text_generation_calls"]):
        raise ValueError("learned embedding receipt observation identity differs")
    canary = receipt.get("canary_observation")
    batches = receipt.get("embedding_receipts")
    if (not isinstance(canary, dict) or set(canary) != set(counters[:2])
            or any(type(canary.get(key)) is not int or canary[key] < 0 for key in counters[:2])
            or canary["local_embedding_calls"] not in (0, 1)
            or canary["local_embedding_texts"] != 3 * canary["local_embedding_calls"]
            or not isinstance(batches, list) or len(batches) > 129
            or (completed and (not batches or canary["local_embedding_calls"] != 1))):
        raise ValueError("learned embedding receipt canary accounting differs")
    fields = {"request_id", "receipt_id", "status", "text_count", *counters[:2]}
    for batch in batches:
        if (not isinstance(batch, dict) or set(batch) != fields
                or type(batch.get("text_count")) is not int or not 1 <= batch["text_count"] <= 32
                or type(batch.get("local_embedding_calls")) is not int or batch["local_embedding_calls"] not in (0, 1)
                or type(batch.get("local_embedding_texts")) is not int
                or batch["local_embedding_texts"] != batch["text_count"] * batch["local_embedding_calls"]
                or (completed and batch.get("status") != "completed")):
            raise ValueError("learned embedding receipt batch accounting differs")
        if batch.get("status") == "completed" and any(
                not isinstance(batch.get(key), str) or re.fullmatch(r"baguqeera[a-z2-7]{52}", batch[key]) is None
                for key in ("request_id", "receipt_id")):
            raise ValueError("learned embedding receipt native handles differ")
    if any(receipt[key] != canary[key] + sum(batch[key] for batch in batches) for key in counters[:2]):
        raise ValueError("learned embedding receipt totals differ from observations")
    if receipt.get("canary") is not None:
        from ..integrations.ipfs_datasets_embedding_provider import EmbeddingCanaryReceipt
        native = EmbeddingCanaryReceipt.from_dict(receipt["canary"])
        if (native.policy_id != receipt.get("current_policy_id")
                or native.observed_dimensions not in (0, previous.config.dimensions)
                or (native.disposition.value == "passed" and canary["local_embedding_calls"] != 1)):
            raise ValueError("learned embedding receipt observed canary identity differs")
    return json.loads(json.dumps(receipt))


def _verify_retrieval_rebuild(value, previous, previous_result):
    from ..analysis.code_symbol_vector_index import CodeVectorIndexSnapshot, CodeVectorSearchResult
    from ..integrations.ipfs_datasets_embedding_provider import PinnedEmbeddingPolicy

    lineage = None
    if isinstance(value, PinnedRetrievalRebuild):
        rebuilt, hits = value.snapshot, value.result
        policies = (value.previous_policy, value.current_policy)
        if not all(isinstance(policy, PinnedEmbeddingPolicy) for policy in policies):
            raise ValueError("native pinned embedding policies required")
        changed_roots = {"corpus_root_id", "index_root_id", "forest_id", "tree_id"}
        if ({key: val for key, val in policies[0]._payload().items() if key not in changed_roots}
                != {key: val for key, val in policies[1]._payload().items() if key not in changed_roots}):
            raise ValueError("retrieval refresh changed the pinned embedding policy")
        if not all(isinstance(item, CodeVectorIndexSnapshot) for item in (previous, rebuilt)):
            raise ValueError("native retrieval snapshots required")
        for policy, snapshot in zip(policies, (previous, rebuilt), strict=True):
            if (policy.allow_remote or policy.remote_endpoint_id
                    or policy.policy_id != snapshot.config.configuration_id
                    or policy.model_artifact_id != snapshot.config.model_id
                    or policy.model_revision != snapshot.config.model_revision
                    or policy.dimensions != snapshot.config.dimensions
                    or policy.chunker_id != snapshot.config.chunker_id
                    or policy.normalizer != snapshot.config.normalization
                    or policy.distance != snapshot.config.metric
                    or policy.forest_id != snapshot.forest_id or policy.tree_id != snapshot.tree_id
                    or policy.index_root_id != snapshot.ast_index_id
                    or policy.corpus_root_id != snapshot.tree_id):
                raise ValueError("pinned retrieval policy does not bind its native snapshot")
        old_config, new_config = previous.config.to_dict(), rebuilt.config.to_dict()
        for name in ("configuration_id", "config_id"):
            old_config.pop(name, None)
            new_config.pop(name, None)
        if old_config != new_config:
            raise ValueError("retrieval refresh changed pinned index configuration")
        lineage = {"previous_policy": policies[0].to_dict(), "current_policy": policies[1].to_dict(),
                   "changed_fields": sorted(changed_roots), "model_configuration_preserved": True}
    else:
        rebuilt, hits = value
        if not isinstance(rebuilt, CodeVectorIndexSnapshot) or rebuilt.config != previous.config:
            raise ValueError("retrieval refresh changed the pinned model/configuration or scope")
    if (not isinstance(hits, CodeVectorSearchResult)
            or rebuilt.included_paths != previous.included_paths
            or rebuilt.excluded_paths != previous.excluded_paths
            or hits.query.max_results != previous_result.query.max_results):
        raise ValueError("retrieval refresh changed the pinned model/configuration or scope")
    if isinstance(value, PinnedRetrievalRebuild) and value.embedding_receipt is not None:
        from ..integrations.ipfs_datasets_embedding_provider import EmbeddingCanaryReceipt
        receipt = value.embedding_receipt
        _verify_embedding_observation(receipt, previous, completed=True)
        if (not isinstance(receipt, dict)
                or receipt.get("schema") != "supervisor-published-learned-embedding-receipt@1"
                or receipt.get("status") != "completed"
                or receipt.get("previous_index_id") != previous.index_id
                or receipt.get("index_id") != rebuilt.index_id or receipt.get("result_id") != hits.result_id
                or receipt.get("previous_policy_id") != policies[0].policy_id
                or receipt.get("current_policy_id") != policies[1].policy_id
                or receipt.get("model_artifact_id") != policies[1].model_artifact_id
                or receipt.get("model_revision") != policies[1].model_revision
                or any(receipt.get(key) is not False for key in (
                    "semantic_authority", "execution_authority", "completion_authority"))
                or any(type(receipt.get(key)) is not int or receipt[key] < 0 for key in (
                    "local_embedding_calls", "local_embedding_texts", "remote_embedding_calls", "text_generation_calls"))
                or receipt["remote_embedding_calls"] != 0 or receipt["text_generation_calls"] != 0
                or receipt.get("native_fact_rows_replayed") != len(rebuilt.rows)
                or not isinstance(receipt.get("source_sha256"), dict)
                or set(receipt["source_sha256"]) != set(rebuilt.included_paths)
                or any(not isinstance(sha, str) or re.fullmatch(r"[a-f0-9]{64}", sha) is None
                       for sha in receipt["source_sha256"].values())
                or content_identity({"schema": "permitted-vector-inputs@1", "sources": receipt["source_sha256"]}) != policies[1].corpus_root_id
                or sum(batch["text_count"] for batch in receipt["embedding_receipts"]) != len({row.qualified_symbol for row in rebuilt.rows}) + 1):
            raise ValueError("learned embedding receipt differs from pinned native rebuild")
        canary = EmbeddingCanaryReceipt.from_dict(receipt["canary"])
        if (canary.policy_id != policies[1].policy_id or canary.disposition.value != "passed"
                or canary.observed_dimensions != policies[1].dimensions or canary.sample_count != 3
                or canary.vector_lane.value != "enabled"
                or canary.backend_kind != "local-safetensors-via-embeddings-router"):
            raise ValueError("learned embedding receipt canary differs from pinned policy")
        lineage["embedding_receipt"] = json.loads(json.dumps(receipt))
    return rebuilt, hits, lineage


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate published context key")
        result[key] = value
    return result


def _verify_predecessor(*, root, result):
    bundle = result["predecessor_bundle"]
    metadata = load_task_context_nomination(repository=root, artifact=bundle["artifact"],
        expected_sha256=bundle["sha256"], task_cid=result["task_cid"], task_id=result["task_id"])
    _, semantic = _load_semantic_payload(repository=root,
        artifact=metadata["semantic context artifact"], expected_sha256=metadata["semantic context sha256"],
        task_id=result["task_id"], verify_sources=False)
    world = load_intent_world_context(artifact=root / metadata["world context artifact"],
        expected_sha256=metadata["world context sha256"], task_id=result["task_id"],
        repository_id=metadata["world context repository"])
    if (semantic["semantic_root_cid"] != result["predecessor_semantic_root_cid"]
            or world["semantic_root_cid"] != semantic["semantic_root_cid"]
            or world["world_snapshot_cid"] != result["predecessor_world_snapshot_cid"]):
        raise ValueError("predecessor context lineage differs")
    _previous_retrieval(root, metadata, result["task_id"])


def _owner_state(*, server, admission, task_cid):
    """Read actual owner authority; return only immutable observation bindings."""
    if type(server) is not QuackStateServer:
        raise TypeError("actual native owner required")
    with server._lock:
        identity = server.identity
        if identity is None or server.ready().get("ready") is not True:
            raise ValueError("native owner is not ready")
        verified = verify_owner_local_benchmark_observation(server=server, admission=admission)
        graph = verified["graph"]
        if len(graph.tasks) != 1 or graph.tasks[0].task_cid != task_cid:
            raise ValueError("publication refresh requires the exact single admitted task")
        root = Path(verified["manifest"]["repository"]).resolve(strict=True)
        head = _git(root, "rev-parse", "HEAD").decode().strip()
        if head == verified["profile"].baseline_commit:
            raise ValueError("publication refresh requires an accepted source transition")
        intent = IntentRepository(bound_connection=server._connection, install_schema=False)
        task = intent.get_task(task_cid)
        if task is None or task["status"] != "completed":
            raise ValueError("publication refresh requires the completed native task")
        contract = task["body"].get(local.CONTRACT_KEY, {}).get("payload", {})
        claim = task["body"].get("completion_receipt", {})
        if (contract.get("manifest") != admission["manifest"]
                or contract.get("graph_cid") != graph.content_id
                or not claim.get("attempt_id")):
            raise ValueError("completed task differs from the admitted publication")
        bindings = {
            "task_cid": task_cid, "task_id": task["task_alias"],
            "task_revision": task["revision"], "task_status": task["status"],
            "task_body_cid": content_identity(task["body"]),
            "completion_receipt_cid": content_identity(claim),
            "attempt_id": claim["attempt_id"],
            "local_contract_cid": content_identity(task["body"][local.CONTRACT_KEY]),
            "admission_cid": content_identity(admission),
            "published_commit": head,
            "accepted_source_tree_id": verified["current_source_tree_id"],
            "owner": {name: getattr(identity, name) for name in (
                "server_id", "store_id", "generation", "fence_epoch",
            )},
        }
        return root, bindings, task


def _source384_successor_unavailable(binding, bundle, receipt):
    return {"schema": "supervisor-source384-successor-observation@1", **binding,
        "status": "successor_unavailable", "reason": "source384_successor_not_prepared",
        "predecessor_bundle": dict(bundle), "source384_context": receipt,
        "source384_current": False, "needs_successor": True,
        "source384_scope": "sealed historical envelope; current source/model validity not established",
        "source_semantics_verified": False, "proof_authority": False,
        "execution_authority": False, "completion_authority": False,
        "canonical_task_mutated": False, "launch_nomination_mutated": False,
        "text_generation_calls": 0, "inference_calls": 0}


def observe_source384_successor_unavailable(*, server, admission, predecessor_bundle, task_cid):
    """Report the retained selection only after owner-verified completed publication."""
    root, binding, _task = _owner_state(server=server, admission=admission, task_cid=task_cid)
    if not isinstance(predecessor_bundle, dict) or set(predecessor_bundle) != {"artifact", "sha256"}:
        raise ValueError("exact predecessor bundle reference required")
    selected = read_task_context_historical_selection(repository=root,
        artifact=predecessor_bundle["artifact"], expected_sha256=predecessor_bundle["sha256"],
        task_cid=task_cid, task_id=binding["task_id"])
    if "source384_context" in selected:
        return _source384_successor_unavailable(binding, predecessor_bundle, selected["source384_context"])
    return None


def _block_reader(directory):
    def get_block(cid):
        if not isinstance(cid, str) or not cid or Path(cid).name != cid or cid in {".", ".."}:
            raise ValueError("invalid semantic block identity")
        path = directory / cid
        if path.is_symlink() or path.resolve() != path:
            raise ValueError("semantic block path must be canonical")
        return path.read_bytes()
    return get_block


def _previous_retrieval(root, metadata, task_id):
    """Use the native loader before exposing its authenticated snapshot."""
    from ..analysis.code_symbol_vector_index import CodeVectorIndexSnapshot, CodeVectorSearchResult

    artifact = metadata.get("code retrieval artifact")
    if artifact is None:
        return None
    context = json.loads(retrieval_context.load_code_retrieval_context(
        repository=root, artifact=artifact,
        expected_sha256=metadata["code retrieval sha256"], task_id=task_id,
    ))
    path = retrieval_context._path(root, artifact)
    raw = retrieval_context._read(path, retrieval_context.MAX_ARTIFACT_BYTES)
    if hashlib.sha256(raw).hexdigest() != metadata["code retrieval sha256"]:
        raise ValueError("retrieval nomination changed during refresh")
    payload = json.loads(raw)
    if "snapshot_ref" in payload:
        ref = payload["snapshot_ref"]
        raw = retrieval_context._read(retrieval_context._path(root, ref["path"]), ref["bytes"])
        if len(raw) != ref["bytes"] or hashlib.sha256(raw).hexdigest() != ref["sha256"]:
            raise ValueError("retrieval snapshot changed during refresh")
        snapshot = CodeVectorIndexSnapshot.from_dict(json.loads(raw))
    else:
        snapshot = CodeVectorIndexSnapshot.from_dict(payload["snapshot"])
    result = CodeVectorSearchResult.from_dict(payload["result"])
    retrieval_context._replay(snapshot, result)
    return context, snapshot, result


def refresh_published_task_context(*, server, admission, predecessor_bundle,
                                   task_cid: str, output: Path,
                                   retrieval_rebuilder=None) -> dict:
    """Build fresh source/intent evidence after actual canonical completion.

    An optional trusted local ``retrieval_rebuilder`` receives keyword args
    ``repository, previous_snapshot, previous_result, query_text, output`` and
    returns two native objects: (snapshot, result), or PinnedRetrievalRebuild
    with native old/new policies allowing only source-root rebinding. Model
    configuration and permitted path scope must remain identical. Without that capability, a
    source-stale retrieval dimension is explicitly unavailable, never reused
    or silently replaced with another embedding model.
    """
    started = time.monotonic()
    root, binding, task = _owner_state(server=server, admission=admission, task_cid=task_cid)
    if not isinstance(predecessor_bundle, dict) or set(predecessor_bundle) != {"artifact", "sha256"}:
        raise ValueError("exact predecessor bundle reference required")
    selected = read_task_context_historical_selection(repository=root, **{
        "artifact": predecessor_bundle["artifact"], "expected_sha256": predecessor_bundle["sha256"],
        "task_cid": task_cid, "task_id": binding["task_id"],
    })
    if "source384_context" in selected:
        return _source384_successor_unavailable(binding, predecessor_bundle, selected["source384_context"])
    metadata = selected["metadata"]
    _, previous = _load_semantic_payload(
        repository=root, artifact=metadata["semantic context artifact"],
        expected_sha256=metadata["semantic context sha256"], task_id=binding["task_id"],
        verify_sources=False,
    )
    previous_world = load_intent_world_context(
        artifact=root / metadata["world context artifact"],
        expected_sha256=metadata["world context sha256"], task_id=binding["task_id"],
        repository_id=metadata["world context repository"],
    )
    if previous_world["semantic_root_cid"] != previous["semantic_root_cid"]:
        raise ValueError("predecessor semantic and world roots differ")
    old_tasks = [row for row in previous_world["tasks"] if row["task_cid"] == task_cid]
    if (len(old_tasks) != 1 or content_identity(old_tasks[0]["body"].get(local.CONTRACT_KEY))
            != binding["local_contract_cid"]):
        raise ValueError("predecessor world belongs to a different admitted task")
    output = Path(output).absolute()
    if (output.exists() or output.resolve() != output
            or not output.is_relative_to(root / ".runtime")
            or any((root / name).is_relative_to(output) for name in previous["manifest"])):
        raise ValueError("refresh output must be a new separate repository runtime directory")
    permitted = set(task["body"][local.CONTRACT_KEY]["payload"]["task_spec"]["scope_paths"])
    if not set(previous["manifest"]) <= permitted:
        raise ValueError("predecessor semantic scope exceeds the signed task scope")
    required = previous.get("required_raw_paths", sorted(previous.get("raw_sources", {})))
    bounds = previous.get("preparation_bounds", {})
    projection = previous.get("worker_projection", {})
    semantic = prepare_semantic_context(
        repository=root, paths=tuple(previous["manifest"]), required_raw_paths=required,
        objective=previous["objective"], task_id=binding["task_id"], output=output / "semantic",
        max_symbols=bounds.get("max_symbols", 256),
        context_input_tokens=bounds.get("context_input_tokens", 8192),
        worker_query=projection.get("query", ""),
        worker_capsule_limit=projection.get("max_capsules", 8),
        worker_max_bytes=projection.get("max_bytes", 32768),
        _refresh_lineage={
            "schema": "supervisor-semantic-context-refresh@1",
            "previous_payload_sha256": metadata["semantic context sha256"],
            "previous_scope_cid": previous["scope_cid"],
            "previous_semantic_root_cid": previous["semantic_root_cid"],
            "previous_manifest": previous["manifest"],
            "attempt_id": binding["attempt_id"], "cause": "owner_validated_publication_completed",
        },
    )
    successor = {
        "Semantic context artifact": (output / "semantic/worker-context.json").relative_to(root).as_posix(),
        "Semantic context sha256": semantic["worker_payload_sha256"],
        "Semantic context refresh": "true",
    }
    old_retrieval = _previous_retrieval(root, metadata, binding["task_id"])
    retrieval = {"status": "unavailable", "reason": "predecessor_has_no_retrieval"}
    if old_retrieval is not None:
        context, snapshot, query_result = old_retrieval
        if not set(snapshot.included_paths) <= set(previous["manifest"]):
            raise ValueError("predecessor retrieval scope exceeds semantic scope")
        retrieval = {"status": "unavailable", "reason": "source_changed_and_pinned_rebuilder_unavailable",
                     "previous_index_id": snapshot.index_id, "previous_config_id": snapshot.config.config_id,
                     "stale_paths": context["stale_paths"]}
        if context["status"] == "current":
            successor.update({"Code retrieval artifact": metadata["code retrieval artifact"],
                              "Code retrieval sha256": metadata["code retrieval sha256"]})
            retrieval.update(status="current", reason="verified_unchanged_source_scope", index_id=snapshot.index_id)
        elif retrieval_rebuilder is not None:
            from .published_retrieval import RetrievalRefreshUnavailable
            try:
                rebuilt_value = retrieval_rebuilder(repository=root, previous_snapshot=snapshot,
                    previous_result=query_result, query_text=context["query_text"], output=output / "vectors")
            except RetrievalRefreshUnavailable as unavailable:
                retrieval["reason"] = unavailable.reason
                if getattr(unavailable, "receipt", None) is not None:
                    retrieval["embedding_receipt"] = _verify_embedding_observation(unavailable.receipt, snapshot)
            else:
                rebuilt, hits, policy_lineage = _verify_retrieval_rebuild(rebuilt_value, snapshot, query_result)
                produced = retrieval_context.prepare_code_retrieval_context(
                    repository=root, task_id=binding["task_id"], query_text=context["query_text"],
                    snapshot=rebuilt, result=hits, output=output / "code-retrieval.json")
                successor.update(produced["metadata"])
                retrieval.update(status="current", reason="rebuilt_with_pinned_configuration",
                                 index_id=rebuilt.index_id, result_id=hits.result_id,
                                 config_id=rebuilt.config.config_id, policy_lineage=policy_lineage)
                from .supervisor_meta_index import SupervisorMetaIndex
                meta = SupervisorMetaIndex(output / "retrieval-metadata.duckdb", output / "retrieval-metadata-lake")
                catalog = meta.register_catalog(kind="vector", locator_ref=str(output / "code-retrieval.json"),
                    tree_id=rebuilt.tree_id, project=False)
                for offset in range(0, len(rebuilt.rows), 10_000):
                    meta.link_identities([dict(subject_kind="path", subject_ref=row.path,
                        catalog_id=catalog["catalog_id"], record_kind="code_symbol", record_ref=row.row_id)
                        for row in rebuilt.rows[offset:offset + 10_000]], project=False)
                retrieval["ducklake"] = meta.project_ducklake()
                if retrieval["ducklake"]["status"] != "projected":
                    raise ValueError("refreshed vector metadata did not project to DuckLake")
    get_block = _block_reader(output / "semantic/blocks")
    view = IpfsDatasetsSemanticStateProvider().open_verified_view(semantic["semantic_root_cid"], get_block)
    with server._lock:
        if _owner_state(server=server, admission=admission, task_cid=task_cid)[1] != binding:
            raise ValueError("publication changed during source refresh")
        connection = server._connection
        connection.execute("BEGIN TRANSACTION")
        try:
            intent = IntentRepository(bound_connection=connection, install_schema=False)
            capture = capture_intent_world_snapshot(intent, repository_id=view.root.repository_id,
                task_cids=[task_cid], semantic_root_cid=semantic["semantic_root_cid"],
                get_semantic_block=get_block, transaction_owned_by_caller=True)
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
    world = persist_intent_world_snapshot(capture, output=output / "world", task_id=binding["task_id"])
    from ..semantic_state.intent_progress import project_intent_progress
    goal_progress = project_intent_progress(capture)
    successor.update({"World context artifact": Path(world["artifact"]).relative_to(root).as_posix(),
                      "World context sha256": world["artifact_sha256"],
                      "World context repository": view.root.repository_id})
    result = {
        "schema": "supervisor-task-context-preparation@1", "refresh_schema": SCHEMA,
        **binding, "metadata": successor, "predecessor_bundle": dict(predecessor_bundle),
        "predecessor_semantic_root_cid": previous["semantic_root_cid"],
        "predecessor_world_snapshot_cid": previous_world["world_snapshot_cid"],
        "semantic_root_cid": semantic["semantic_root_cid"],
        "world_snapshot_cid": capture["snapshot"]["snapshot_cid"],
        "plan_projection_cid": capture["plan_projection"]["projection_cid"],
        "event_watermark": capture["planning_context"]["event_watermark"],
        "repository_id": view.root.repository_id, "semantic": semantic, "world": world,
        "retrieval": retrieval, "unavailable_components": capture["planning_context"]["unavailable_components"],
        "goal_progress": goal_progress,
        "source_scope": {
            "preserved_paths": sorted(previous["manifest"]),
            "declared_outputs_outside_index_scope": sorted(
                item["path"] for item in task["body"][local.CONTRACT_KEY]["payload"]["task_spec"]["outputs"]
                if item["path"] not in previous["manifest"]),
            "scope_expanded": False,
            "accepted_source_tree_includes_declared_outputs": True,
        },
        "execution_authority": False, "completion_authority": False,
        "canonical_task_mutated": False, "launch_nomination_mutated": False,
        "text_generation_calls": 0, "seconds": time.monotonic() - started,
    }
    # Source and intent rechecks precede publication of the new nomination.
    _verify_predecessor(root=root, result=result)
    _verify_current(server=server, admission=admission, root=root, result=result)
    result["context_bundle"] = write_task_context_bundle(repository=root, prepared=[result],
        output=output / "context-bundle.json")
    raw = _json(result).encode()
    artifact = output / "result.json"
    with artifact.open("xb") as stream:
        stream.write(raw)
    return {**result, "refresh_artifact": artifact.relative_to(root).as_posix(),
            "refresh_sha256": hashlib.sha256(raw).hexdigest()}


def _verify_current(*, server, admission, root, result):
    current_root, binding, _ = _owner_state(server=server, admission=admission, task_cid=result["task_cid"])
    if current_root != root or any(result.get(key) != value for key, value in binding.items()):
        raise ValueError("published context no longer matches current owner/source/task")
    metadata = result["metadata"]
    semantic = json.loads(load_semantic_worker_context(repository=root,
        artifact=metadata["Semantic context artifact"], expected_sha256=metadata["Semantic context sha256"],
        task_id=binding["task_id"]))
    with server._lock:
        intent = IntentRepository(bound_connection=server._connection, install_schema=False)
        world = load_intent_world_context(artifact=root / metadata["World context artifact"],
            expected_sha256=metadata["World context sha256"], task_id=binding["task_id"],
            repository_id=metadata["World context repository"], intent=intent)
    if (semantic["semantic_root_cid"] != result["semantic_root_cid"]
            or world["semantic_root_cid"] != result["semantic_root_cid"]
            or world["world_snapshot_cid"] != result["world_snapshot_cid"]
            or world["plan_projection_cid"] != result["plan_projection_cid"]
            or world["event_watermark"] != result["event_watermark"]
            or world["repository_id"] != result["repository_id"]
            or world["unavailable_components"] != result["unavailable_components"]):
        raise ValueError("refreshed semantic/world identities differ")
    if "goal_progress" in result:
        from ..semantic_state.intent_progress import project_intent_progress
        raw_capture = retrieval_context._read(root / metadata["World context artifact"], 8_000_000)
        if hashlib.sha256(raw_capture).hexdigest() != metadata["World context sha256"]:
            raise ValueError("refreshed world changed during goal progress verification")
        capture = json.loads(raw_capture)
        if project_intent_progress(capture) != result["goal_progress"]:
            raise ValueError("refreshed goal progress differs from native world capture")
    if "Code retrieval artifact" in metadata:
        retrieved = json.loads(retrieval_context.load_code_retrieval_context(repository=root,
            artifact=metadata["Code retrieval artifact"], expected_sha256=metadata["Code retrieval sha256"],
            task_id=binding["task_id"]))
        if retrieved["status"] != "current" or retrieved["index_id"] != result["retrieval"]["index_id"]:
            raise ValueError("refreshed retrieval is not source-current")
    if _owner_state(server=server, admission=admission, task_cid=result["task_cid"])[1] != binding:
        raise ValueError("publication changed during refreshed context verification")
    return world


def load_published_task_context(*, server, admission, artifact: str, expected_sha256: str) -> dict:
    """Revalidate an immutable successor before planning or cache reuse."""
    verified = verify_owner_local_benchmark_observation(server=server, admission=admission)
    root = Path(verified["manifest"]["repository"]).resolve(strict=True)
    path = retrieval_context._path(root, artifact)
    raw = retrieval_context._read(path, 8_000_000)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("published context digest differs")
    result = json.loads(raw, object_pairs_hook=_unique)
    if (result.get("schema") != "supervisor-task-context-preparation@1"
            or result.get("refresh_schema") != SCHEMA
            or any(result.get(key) is not False for key in (
                "execution_authority", "completion_authority", "canonical_task_mutated", "launch_nomination_mutated"))):
        raise ValueError("published context schema/authority differs")
    _verify_predecessor(root=root, result=result)
    metadata = load_task_context_nomination(repository=root, artifact=result["context_bundle"]["artifact"],
        expected_sha256=result["context_bundle"]["sha256"], task_cid=result["task_cid"], task_id=result["task_id"])
    if metadata != {key.lower(): value for key, value in result["metadata"].items()}:
        raise ValueError("successor bundle differs from the refresh result")
    world = _verify_current(server=server, admission=admission, root=root, result=result)
    return {**result, "planning_context": world, "intent_freshness_checked": True,
            "refresh_artifact": artifact, "refresh_sha256": expected_sha256}
