"""Join ordinary world-input evidence to independently retained nominations.

Only immutable artifacts and retained source blocks are read. Native historical
reconstruction runs against private copies of the bounded, pinned artifacts;
neither current checkouts nor an Intent owner supply historical evidence.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory


SCHEMA = "terminal-world-nomination-audit@1"
MAX_FILE_BYTES = 2_000_000
MAX_TOTAL_BYTES = 8 * 1024 * 1024
_FAMILIES = ("source384_context", "numerical_formalization", "numerical_selection_384")
_NOMINATIONS = {
    "semantic-context": "semantic context",
    "code-retrieval-context": "code retrieval",
    "intent-world-context": "world context",
}


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def audit_world_nominations(*, packet, expected, prepared):
    """Replay exact selected projections without currentness or owner authority.

    An absent bundle preserves the preexisting capsule-only audit. Once a bundle
    is selected, all three nominations and their compiled projections must join.
    Refreshed semantic paths and empty retrieval require separate qualification.
    """
    if "context_bundle" not in expected:
        return None
    from . import terminal_context_audit as audit
    from .terminal_task_profile import SCOPED_SCHEMA
    from ipfs_accelerate_py.agent_supervisor.runtime import task_context_bundle as bundles
    from ipfs_accelerate_py.agent_supervisor.runtime.code_retrieval_context import (
        SCHEMA as RETRIEVAL_SCHEMA, REFERENCED_SCHEMA,
        replay_code_retrieval_context_for_audit,
    )
    from ipfs_accelerate_py.agent_supervisor.semantic_state.intent_world_snapshot import (
        load_intent_world_context, minify_intent_world_worker_context,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import (
        public_replay_without_metadata,
    )
    from ipfs_accelerate_py.mcp_server.mcplusplus.kubo_cid import cid_for_bytes

    binding = expected["context_bundle"]
    _need(type(binding) is dict and set(binding) == {"artifact", "sha256"},
          "exact ordinary context bundle binding required")
    profile = prepared.get("task_profile")
    _need(not any(key in expected for key in (*_FAMILIES, "terminal_semantic_scope"))
          and not (type(profile) is dict and profile.get("schema") == SCOPED_SCHEMA),
          "ordinary nomination replay excludes formalization and signed scope")
    root = Path(prepared["repository"])
    _need(root.is_absolute() and root.resolve(strict=True) == root,
          "canonical retained nomination repository required")
    task_id, task_cid = prepared["spec"]["task_key"], expected["task_cid"]
    root_fd = audit._open_directory(root)
    retained, total = {}, 0

    def read(relative, *, digest=None, size=None, limit=MAX_FILE_BYTES):
        nonlocal total
        _need(type(relative) is str and relative and len(relative.encode()) <= 4096
              and Path(relative).as_posix() == relative and "\0" not in relative,
              "bounded nomination artifact path required")
        if relative not in retained:
            remaining = MAX_TOTAL_BYTES - total
            _need(remaining > 0, "nomination retained artifact population exceeds byte bound")
            raw = audit._read_at(root_fd, relative, limit=min(limit, MAX_FILE_BYTES, remaining))
            retained[relative] = raw
            total += len(raw)
        raw = retained[relative]
        _need(len(raw) <= min(limit, MAX_FILE_BYTES), "nomination artifact exceeds byte bound")
        if digest is not None:
            _need(hashlib.sha256(raw).hexdigest() == audit._digest(digest),
                  "nomination artifact digest differs")
        if size is not None:
            _need(type(size) is int and 0 < size <= MAX_FILE_BYTES and len(raw) == size,
                  "nomination artifact byte count differs")
        return raw

    def stage(directory, relative, raw):
        # Every relative path has already passed the nofollow retained reader.
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(raw)
        return path

    try:
        bundle_raw = read(binding["artifact"], digest=binding["sha256"], limit=bundles.MAX_BYTES)
        bundle = audit._json(bundle_raw)
        _need(type(bundle) is dict and bundle.get("schema") == bundles.SCHEMA,
              "only ordinary nomination bundles support world replay")
        evidence = packet["capsule"]["evidence"]
        _need(not any(row.get("kind") == "source-formalization-context" for row in evidence),
              "ordinary nomination replay excludes formalization evidence")
        with TemporaryDirectory(prefix="terminal-world-nominations-") as temporary:
            staging = Path(temporary)
            selection_root = staging / "selection"
            selection_root.mkdir()
            stage(selection_root, "bundle.json", bundle_raw)
            with public_replay_without_metadata():
                selected = bundles.read_task_context_historical_selection(
                    repository=selection_root, artifact="bundle.json", expected_sha256=binding["sha256"],
                    task_cid=task_cid, task_id=task_id)
            _need(set(selected) == {"metadata"}, "ordinary nomination selected formalization evidence")
            metadata = selected["metadata"]
            texts, contexts, artifacts = {}, {}, {}
            for kind, prefix in _NOMINATIONS.items():
                name, digest = metadata[prefix + " artifact"], metadata[prefix + " sha256"]
                rows = sorted((row for row in evidence if row.get("kind") == kind),
                              key=lambda row: row["reference_id"])
                _need(bool(rows) and all(row.get("path") == name
                      and row.get("metadata", {}).get("required") is True for row in rows),
                      "compiled context differs from nominated artifact path")
                texts[kind] = "".join(row["summary"] for row in rows)
                contexts[kind] = audit._json(texts[kind].encode("utf-8"))
                artifacts[kind] = read(name, digest=digest)
            semantic = contexts["semantic-context"]
            _need(artifacts["semantic-context"].decode("utf-8") == texts["semantic-context"],
                  "compiled semantic context differs from nominated exact text")
            _need(type(semantic) is dict and semantic.get("task_id") == task_id
                  and type(semantic.get("manifest")) is dict
                  and 1 <= len(semantic["manifest"]) <= 64,
                  "bounded retained semantic source manifest required")

            retrieval = contexts["code-retrieval-context"]
            retrieval_artifact = audit._json(artifacts["code-retrieval-context"])
            _need(type(retrieval_artifact) is dict and retrieval_artifact.get("schema") in {
                RETRIEVAL_SCHEMA, REFERENCED_SCHEMA}, "empty or unknown retrieval replay is unsupported")
            _need(type(retrieval) is dict and retrieval.get("status") == "current"
                  and "historical_replay" not in retrieval and "freshness_checked" not in retrieval
                  and type(retrieval.get("source_sha256")) is dict
                  and 1 <= len(retrieval["source_sha256"]) <= 64,
                  "captured current retrieval source population required")
            sources = {}
            blocks = Path(metadata["semantic context artifact"]).parent / "blocks"
            for name, digest in retrieval["source_sha256"].items():
                source = semantic["manifest"][name]
                cid = source["source_cid"]
                _need(type(cid) is str and Path(cid).name == cid and cid not in {".", ".."}
                      and source["sha256"] == digest, "retrieval source differs from semantic binding")
                raw = read((blocks / cid).as_posix(), digest=digest)
                _need(cid_for_bytes(raw) == cid, "retained semantic source CID differs")
                sources[name] = raw
            snapshot = None
            if retrieval_artifact["schema"] == REFERENCED_SCHEMA:
                reference = retrieval_artifact["snapshot_ref"]
                _need(type(reference) is dict and set(reference) == {"path", "sha256", "bytes", "index_id"}
                      and type(reference["bytes"]) is int and 0 < reference["bytes"] <= MAX_FILE_BYTES,
                      "bounded retained retrieval snapshot reference required")
                snapshot = (reference["path"], read(reference["path"], digest=reference["sha256"],
                            size=reference["bytes"], limit=reference["bytes"]))

            # Native loaders only see these private copies of preflighted bytes.
            # In particular, a changed retained world file cannot race its
            # native stat/read into an unbounded read of substituted content.
            retrieval_root = staging / "retrieval"
            retrieval_root.mkdir()
            retrieval_name = metadata["code retrieval artifact"]
            stage(retrieval_root, retrieval_name, artifacts["code-retrieval-context"])
            if snapshot is not None:
                stage(retrieval_root, *snapshot)
            world_root = staging / "world"
            world_root.mkdir()
            world_path = stage(world_root, "capture.json", artifacts["intent-world-context"])
            with public_replay_without_metadata():
                replayed_retrieval = audit._json(replay_code_retrieval_context_for_audit(
                    repository=retrieval_root, artifact=retrieval_name,
                    expected_sha256=metadata["code retrieval sha256"], task_id=task_id,
                    retained_sources=sources).encode("utf-8"))
                _need(replayed_retrieval.pop("historical_replay", None) is True
                      and replayed_retrieval.pop("freshness_checked", None) is False
                      and replayed_retrieval.get("status") == "historical",
                      "native historical retrieval contract differs")
                replayed_retrieval["status"] = "current"
                _need(audit._same_json(replayed_retrieval, retrieval),
                      "compiled retrieval differs from nominated historical projection")
                world = load_intent_world_context(artifact=world_path,
                    expected_sha256=metadata["world context sha256"], task_id=task_id,
                    repository_id=metadata["world context repository"], intent=None)
                captured = contexts["intent-world-context"]
                _need(type(captured) is dict and type(captured.get("intent_freshness_checked")) is bool,
                      "explicit captured intent freshness observation required")
                # Preserve the declared dispatch observation to reproduce its
                # projection CID; no live Intent owner authenticates it here.
                world["intent_freshness_checked"] = captured["intent_freshness_checked"]
                if "semantic_coherence" in captured:
                    original_root = world.get("semantic_root_cid", "")
                    _need(not original_root or original_root == semantic["semantic_root_cid"],
                          "stale semantic world coherence is unsupported")
                    world["semantic_coherence"] = {
                        "status": "current" if original_root else "unbound",
                        "captured_semantic_root_cid": original_root,
                        "dispatch_semantic_root_cid": semantic["semantic_root_cid"],
                    }
                _need(audit._same_json(minify_intent_world_worker_context(world), captured),
                      "compiled world differs from nominated historical projection")
        return dict(schema=SCHEMA, context_bundle_sha256=binding["sha256"],
            retained_artifact_bytes=total, retained_artifact_count=len(retained),
            max_file_bytes=MAX_FILE_BYTES, max_total_bytes=MAX_TOTAL_BYTES,
            ordinary_bundle_task_selection_verified=True,
            semantic_nomination_exact_text_verified=True, semantic_producer_replayed=False,
            retrieval_nomination_historical_projection_verified=True,
            world_nomination_historical_projection_verified=True,
            recorded_intent_freshness_checked=captured["intent_freshness_checked"],
            intent_freshness_reverified=False, current_sources_checked=False,
            owner_signature_replayed=False, token_counts_replayed=False,
            proof_authority=False, execution_authority=False, completion_authority=False)
    finally:
        os.close(root_fd)
