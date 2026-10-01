"""Source-bound composition of datasets capsules, context packing and Doctor.

This preparation stage has no model, source-write, scheduling or completion
capability. Callers must supply the exact permitted input paths. Its worker
payload preserves required raw sources and keeps producer capsules intact.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile
from typing import Sequence
import uuid


def _encoded(value, *, pretty=False):
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
        allow_nan=False,
    ).encode()


def _scan_scoped_sources(sources, *, repository_id, max_symbols):
    """Cold-scan exactly captured bytes; neither target imports nor caches run."""
    from ipfs_datasets_py.logic.software_contracts.semantic_index.scanner import (
        RepositoryScanner,
    )
    from ipfs_datasets_py.logic.software_contracts.semantic_index.snapshot import (
        snapshot_repository,
    )

    with tempfile.TemporaryDirectory(prefix="supervisor-semantic-scope-") as temporary:
        snapshot = Path(temporary)
        for name, raw in sources.items():
            path = snapshot / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        acquired = snapshot_repository(
            snapshot,
            repository_id=repository_id,
            exclusions=(),
        )
        if {entry.path for entry in acquired.entries} != set(sources):
            raise ValueError("semantic scanner omitted a scoped source")
        state = RepositoryScanner(
            repository_id=repository_id,
            namespace="supervisor-scoped-context",
        ).scan_snapshot(acquired, sources)
        if len(state.symbols) > max_symbols:
            raise ValueError("semantic scope exceeds symbol bound")
        return state


def prepare_semantic_context(
    *,
    repository: Path,
    paths: Sequence[str],
    required_raw_paths: Sequence[str],
    objective: str,
    task_id: str,
    output: Path,
    max_source_bytes: int = 1_000_000,
    max_files: int = 64,
    max_symbols: int = 256,
    context_input_tokens: int = 8192,
    worker_query: str = "",
    worker_capsule_limit: int = 8,
    worker_max_bytes: int = 32768,
    max_payload_bytes: int = 2_000_000,
    _refresh_lineage: dict | None = None,
) -> dict:
    """Build a fresh, bounded worker context using the native producer APIs.

    A private source snapshot prevents the scanner from expanding a subdirectory
    to its containing Git repository. The manifest binds every copied byte.
    Compact JSON is lossless wire minification, not a semantic equivalence proof.
    """
    from ..semantic_state.datasets_adapter import IpfsDatasetsSemanticStateProvider
    from ..semantic_state.context_pack import pack_context
    from ..context.context_contracts import ContextBudget
    from ..semantic_state.program_world_database import ProgramWorldDatabase
    from .supervisor_meta_index import SupervisorMetaIndex
    from ..analysis.doctor_repository_diagnostics import (
        diagnose_repository,
        DoctorSourceUnit,
        DoctorAuthorityRoots,
    )
    from ..analysis.doctor_contract_adapters import materialize_runtime_diagnostics
    from ..proof.formal_verification_contracts import content_identity
    from ..semantic_state.wire import cid_for_payload, canonicalize_artifact
    from ...mcp_server.mcplusplus.kubo_cid import cid_for_bytes

    repository = repository.resolve(strict=True)
    if type(max_symbols) is not int or not 1 <= max_symbols <= 1024:
        raise ValueError("semantic symbol bound must be an integer from 1 to 1024")
    if type(context_input_tokens) is not int or not 1 <= context_input_tokens <= 128000:
        raise ValueError("semantic context token bound must be an integer from 1 to 128000")
    if (not isinstance(worker_query, str) or len(worker_query.encode()) > 8192
            or type(worker_capsule_limit) is not int or not 0 <= worker_capsule_limit <= 32
            or type(worker_max_bytes) is not int or not 8192 <= worker_max_bytes <= 65536):
        raise ValueError("invalid bounded semantic worker projection policy")
    selected = tuple(sorted(set(paths)))
    required = tuple(sorted(set(required_raw_paths)))
    if not objective.strip() or not task_id.strip():
        raise ValueError("objective and task identity are required")
    if (
        not selected
        or len(selected) > max_files
        or not required
        or not set(required) <= set(selected)
    ):
        raise ValueError("invalid input scope or required raw paths")
    sources = {}
    for name in selected:
        rel = PurePosixPath(name)
        if rel.is_absolute() or ".." in rel.parts or str(rel) != name:
            raise ValueError("source path must be canonical and repository-relative")
        path = repository / name
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(repository):
            raise ValueError("source path escapes repository or is a symlink")
        if path.stat().st_size > max_source_bytes:
            raise ValueError("source exceeds byte bound")
        raw = path.read_bytes()
        raw.decode("utf-8")
        sources[name] = raw
        if sum(map(len, sources.values())) > max_source_bytes:
            raise ValueError("source scope exceeds byte bound")
    manifest = {
        name: {"sha256": hashlib.sha256(raw).hexdigest(), "source_cid": cid_for_bytes(raw)}
        for name, raw in sources.items()
    }
    scope_payload = {"schema": "supervisor-source-scope@1", "sources": manifest}
    scope_cid = cid_for_payload(scope_payload)
    provider = IpfsDatasetsSemanticStateProvider()
    provider.capability.require_available("build_semantic_state")
    repository_id = cid_for_payload({"repository": str(repository)})
    state = _scan_scoped_sources(sources, repository_id=repository_id, max_symbols=max_symbols)
    bundle = provider.build_semantic_state(state)
    view = provider.view_semantic_state_bundle(bundle)
    root_cid = view.root.root_cid
    # A second producer scan starts from captured source, never from nominated
    # capsules or incremental state. This qualifies only this explicit scope.
    rebuilt = provider.build_semantic_state(
        _scan_scoped_sources(sources, repository_id=repository_id, max_symbols=max_symbols)
    )
    rebuilt_view = provider.view_semantic_state_bundle(rebuilt)
    if rebuilt_view.root.root_cid != root_cid:
        raise ValueError("cold scoped semantic reconstruction differs")
    reconstruction = {
        "schema": "supervisor-scoped-semantic-reconstruction@1",
        "scope_cid": scope_cid,
        "semantic_root_cid": root_cid,
        "nomination_matched": True,
        "reuse": "cold",
        "full_repository": False,
        "semantic_acceptance_authority": False,
        "completion_authority": False,
    }
    refresh_lineage = None
    if _refresh_lineage is not None:
        refresh_lineage = dict(_refresh_lineage)
        previous_manifest = refresh_lineage.pop("previous_manifest")
        if set(previous_manifest) != set(manifest):
            raise ValueError("refresh cannot expand or shrink source scope")
        refresh_lineage.update(
            {
                "scope_cid": scope_cid,
                "semantic_root_cid": root_cid,
                "source_delta": {
                    name: {"before": previous_manifest[name], "after": manifest[name]}
                    for name in manifest
                    if previous_manifest[name] != manifest[name]
                },
                "semantic_acceptance_authority": False,
                "completion_authority": False,
            }
        )
    from .semantic_capsule_selection import select_worker_capsules
    full_capsule_count = len(state.symbols)
    capsules, admissions, selected_symbols = select_worker_capsules(
        bundle=bundle, view=view, provider=provider, symbols=state.symbols,
        sources=sources, required=required, worker_query=worker_query,
        worker_capsule_limit=worker_capsule_limit, worker_max_bytes=worker_max_bytes,
    )
    projection = None
    if worker_query:
        # Ranking is a context nomination only. Retain exact producer capsules;
        # never summarize away raw-required caveats or promote heuristic scores.
        projection = {
            "schema": "supervisor-semantic-worker-projection@1",
            "selector": "lexical-query-symbol-name@1", "query": worker_query,
            "max_capsules": worker_capsule_limit, "max_bytes": worker_max_bytes,
            "full_capsule_count": full_capsule_count, "selected_capsule_count": len(capsules),
            "omitted_capsule_count": full_capsule_count - len(capsules),
            "selected_symbols": selected_symbols,
            "capsule_index_cid": view.root.capsule_index_cid,
            "raw_source_fetch_required": {name: manifest[name] for name in selected if name not in required},
            "omission_reason": "explicit bounded worker projection; full source and capsule index retained",
            "raw_required_caveat": "Read exact repository source before editing; omitted raw bytes are not substituted or proved equivalent by capsules.",
            "semantic_equivalence_claimed": False, "completion_authority": False,
        }
    raw_scope_payload = {
        "schema": "required-source-set@1",
        "sources": {p: manifest[p] for p in required},
    }
    raw_scope = cid_for_payload(raw_scope_payload)
    delta_payload = {"schema": "initial-context-scan@1", "scope_cid": scope_cid}
    packed = pack_context(
        objective=objective,
        target_source_cid=scope_cid if projection is not None else raw_scope,
        surrounding_source_cid=scope_cid,
        test_source_cid=raw_scope,
        delta_cid=cid_for_payload(delta_payload),
        dependency_admissions=admissions,
        budget=ContextBudget(max_input_tokens=context_input_tokens),
        assumptions=("Initial scoped context, not a full-repository world snapshot.",),
    )
    roots = DoctorAuthorityRoots(
        repository_id=state.repository_id,
        forest_id=scope_cid,
        tree_id=scope_cid,
        overlay_id=scope_cid,
        file_root_id=scope_cid,
        blob_root_id=scope_cid,
        config_id=content_identity({"paths": selected}),
        policy_id=content_identity({"mode": "context_preparation_only"}),
    )
    diagnostics = diagnose_repository(
        [
            DoctorSourceUnit(path=p, source_bytes=raw, blob_identity=manifest[p]["source_cid"])
            for p, raw in sources.items()
        ],
        authority_roots=roots,
    )
    doctor_snapshot, findings, bridge_cid = materialize_runtime_diagnostics(
        diagnostics, require_repository_id=state.repository_id
    )
    # Non-substitutable dependencies remain raw. Required inputs are never elided.
    raw_paths = set(required)
    if projection is None:
        for admission, symbol in zip(admissions, state.symbols):
            if admission.ref.raw_source_required:
                raw_paths.add(symbol.module_path)
    # Non-symbol-bearing files (instructions/configuration) cannot vanish.
    if projection is None:
        raw_paths.update(set(selected) - {s.module_path for s in state.symbols})
    payload = {
        "schema": "supervisor-semantic-worker-context@1",
        "task_id": task_id,
        "objective": objective,
        "scope_cid": scope_cid,
        "semantic_root_cid": root_cid,
        "manifest": manifest,
        "required_raw_paths": list(required),
        "preparation_bounds": {"max_symbols": max_symbols, "context_input_tokens": context_input_tokens},
        "capsules": capsules,
        "admissions": [a.to_dict() for a in admissions],
        "raw_sources": {p: sources[p].decode() for p in sorted(raw_paths)},
        "pack": packed.to_dict(),
        "doctor_snapshot_id": doctor_snapshot.snapshot_id,
        "doctor_manifest_cid": bridge_cid,
        "reconstruction": reconstruction,
        "refresh_lineage": refresh_lineage,
        "completion_authority": False,
    }
    if projection is not None:
        payload["worker_projection"] = projection
    compact = _encoded(payload)
    # The preliminary size estimate cannot account for every native pack field.
    # Repack the exact ranked prefix until it fits, retaining required raw input
    # verbatim. Omitted optional capsules remain in the complete producer index.
    while projection is not None and capsules and (
        len(compact) > worker_max_bytes or packed.budget_exceeded
    ):
        capsules.pop()
        admissions.pop()
        projection["selected_symbols"].pop()
        projection["selected_capsule_count"] = len(capsules)
        projection["omitted_capsule_count"] = full_capsule_count - len(capsules)
        projection["exact_byte_budget_repacked"] = True
        packed = pack_context(
            objective=objective, target_source_cid=scope_cid,
            surrounding_source_cid=scope_cid, test_source_cid=raw_scope,
            delta_cid=cid_for_payload(delta_payload), dependency_admissions=admissions,
            budget=ContextBudget(max_input_tokens=context_input_tokens),
            assumptions=("Initial scoped context, not a full-repository world snapshot.",),
        )
        payload["admissions"] = [admission.to_dict() for admission in admissions]
        payload["pack"] = packed.to_dict()
        compact = _encoded(payload)
    if (len(compact) > max_payload_bytes or packed.budget_exceeded
            or (projection is not None and len(compact) > worker_max_bytes)):
        raise ValueError("worker context exceeds configured budget")
    if any((repository / name).read_bytes() != raw for name, raw in sources.items()):
        raise ValueError("source changed during context preparation")
    output.mkdir(parents=True, exist_ok=False)
    (output / "worker-context.json").write_bytes(compact)
    blocks = output / "blocks"
    blocks.mkdir()
    for cid, raw in bundle.blocks.items():
        (blocks / cid).write_bytes(raw)
    for name, raw in sources.items():
        (blocks / manifest[name]["source_cid"]).write_bytes(raw)
    for artifact in (scope_payload, raw_scope_payload, delta_payload):
        (blocks / cid_for_payload(artifact)).write_bytes(canonicalize_artifact(artifact))
    (output / "doctor.json").write_bytes(
        _encoded(
            {
                "snapshot": doctor_snapshot.to_dict(),
                "findings": [f.to_dict() for f in findings],
                "manifest_cid": bridge_cid,
            }
        )
    )
    world = ProgramWorldDatabase(output / "world.duckdb", output / "world-lake")
    record = world.persist(
        {
            "task_id": task_id,
            "board": "semantic-context",
            "operation": "context_prepared",
            "scope_cid": scope_cid,
            "semantic_root_cid": root_cid,
            "pack_cid": packed.pack_cid,
            "doctor_manifest_cid": bridge_cid,
            "reconstruction_cid": cid_for_payload(reconstruction),
            "refresh_lineage": refresh_lineage,
            "proposal_only": True,
            "completion_authority": False,
        }
    )
    if world.records_for_decision(task_id=task_id)["n"] != 1:
        raise RuntimeError("world context record did not hydrate")
    meta = SupervisorMetaIndex(output / "metadata.duckdb", output / "metadata-lake")
    for kind, locator, ref in [
        ("capsule", blocks, view.root.capsule_index_cid),
        ("world_model", output / "world.duckdb", record["record_cid"]),
    ]:
        catalog = meta.register_catalog(
            kind=kind,
            locator_ref=str(locator),
            repository_id=state.repository_id,
            tree_id=scope_cid,
            project=False,
        )
        meta.link_identity(
            subject_kind="record_cid",
            subject_ref=scope_cid,
            catalog_id=catalog["catalog_id"],
            record_kind=kind,
            record_ref=ref,
            project=False,
        )
    ducklake_projection = meta.project_ducklake()
    if ducklake_projection["status"] != "projected":
        raise RuntimeError("semantic context metadata did not project to DuckLake")
    result = {
        "schema": "supervisor-semantic-context-preparation@1",
        "task_id": task_id,
        "scope_cid": scope_cid,
        "semantic_root_cid": root_cid,
        "capsules": full_capsule_count,
        "worker_capsules": len(capsules),
        "worker_projection": projection,
        "doctor_findings": len(findings),
        "pack_cid": packed.pack_cid,
        "worker_payload_sha256": hashlib.sha256(compact).hexdigest(),
        "compact_bytes": len(compact),
        "pretty_json_bytes": len(_encoded(payload, pretty=True)),
        "minification": "lossless JSON whitespace removal only",
        "reconstruction": reconstruction,
        "refresh_lineage": refresh_lineage,
        "provider_token_usage": None,
        "ducklake": ducklake_projection,
        "full_supervisor_qualified": False,
        "completion_authority": False,
    }
    (output / "result.json").write_bytes(_encoded(result, pretty=True))
    return result


class SemanticContextStale(ValueError):
    """A verified nomination is stale; only fresh source may replace it."""


def _load_semantic_payload(
    *,
    repository: Path,
    artifact: str,
    expected_sha256: str,
    task_id: str,
    max_payload_bytes: int = 2_000_000,
    verify_sources: bool = True,
) -> tuple[bytes, dict]:
    root = repository.resolve(strict=True)
    relative = PurePosixPath(artifact)
    if relative.is_absolute() or ".." in relative.parts or str(relative) != artifact:
        raise ValueError("semantic context artifact must be repository-relative")
    path = root / artifact
    if not path.resolve(strict=True).is_relative_to(root) or path.is_symlink():
        raise ValueError("semantic context artifact escapes repository")
    if path.stat().st_size > max_payload_bytes:
        raise ValueError("semantic context artifact exceeds byte bound")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("semantic context artifact digest mismatch")
    payload = json.loads(raw)
    if (
        payload.get("schema") != "supervisor-semantic-worker-context@1"
        or payload.get("task_id") != task_id
        or payload.get("completion_authority") is not False
    ):
        raise ValueError("semantic context task or schema mismatch")
    manifest = payload.get("manifest")
    bounds = payload.get("preparation_bounds", {"max_symbols": 256, "context_input_tokens": 8192})
    if (not isinstance(bounds, dict) or set(bounds) != {"max_symbols", "context_input_tokens"}
            or type(bounds["max_symbols"]) is not int
            or not 1 <= bounds["max_symbols"] <= 1024
            or type(bounds["context_input_tokens"]) is not int
            or not 1 <= bounds["context_input_tokens"] <= 128000):
        raise ValueError("invalid semantic preparation bounds")
    projection = payload.get("worker_projection")
    if projection is not None and (
        not isinstance(projection, dict)
        or projection.get("schema") != "supervisor-semantic-worker-projection@1"
        or projection.get("selector") != "lexical-query-symbol-name@1"
        or projection.get("semantic_equivalence_claimed") is not False
        or projection.get("completion_authority") is not False
        or not isinstance(projection.get("query"), str)
        or not projection["query"] or len(projection["query"].encode()) > 8192
        or type(projection.get("max_capsules")) is not int
        or not 0 <= projection["max_capsules"] <= 32
        or type(projection.get("max_bytes")) is not int
        or not 8192 <= projection["max_bytes"] <= 65536
        or len(raw) > projection["max_bytes"]
    ):
        raise ValueError("invalid bounded semantic worker projection")
    if not isinstance(manifest, dict) or not manifest or len(manifest) > 64:
        raise ValueError("invalid semantic context source manifest")
    for name, binding in manifest.items():
        if not isinstance(name, str) or not isinstance(binding, dict):
            raise ValueError("invalid semantic context source binding")
        rel = PurePosixPath(name)
        source = root / name
        if (
            rel.is_absolute()
            or ".." in rel.parts
            or str(rel) != name
            or source.is_symlink()
            or not source.resolve(strict=True).is_relative_to(root)
        ):
            raise ValueError("semantic source escapes repository")
        if source.stat().st_size > 1_000_000:
            raise ValueError("semantic source exceeds byte bound")
        if verify_sources and hashlib.sha256(source.read_bytes()).hexdigest() != binding.get(
            "sha256"
        ):
            raise SemanticContextStale("semantic context source is stale: " + name)
    return raw, payload


def load_semantic_worker_context(
    *,
    repository: Path,
    artifact: str,
    expected_sha256: str,
    task_id: str,
    max_payload_bytes: int = 2_000_000,
) -> str:
    """Recheck task, payload and source bindings immediately before dispatch."""
    raw, _ = _load_semantic_payload(
        repository=repository,
        artifact=artifact,
        expected_sha256=expected_sha256,
        task_id=task_id,
        max_payload_bytes=max_payload_bytes,
    )
    return raw.decode("utf-8")


def validate_semantic_worker_nomination(
    *,
    repository: Path,
    artifact: str,
    expected_sha256: str,
    task_id: str,
    max_payload_bytes: int = 2_000_000,
) -> None:
    """Validate immutable task nomination; its sources may need a fresh scan.

    This checks digest, schema, task and path bounds only. Dispatch must also
    strictly load a source-current payload, including when using a cached
    refresh. A cache never overrides changed or removed nominated artifacts.
    """
    _load_semantic_payload(
        repository=repository,
        artifact=artifact,
        expected_sha256=expected_sha256,
        task_id=task_id,
        max_payload_bytes=max_payload_bytes,
        verify_sources=False,
    )


def resolve_semantic_worker_context(
    *,
    repository: Path,
    artifact: str,
    expected_sha256: str,
    task_id: str,
    refresh_output: Path,
    attempt_id: str,
    max_payload_bytes: int = 2_000_000,
) -> dict:
    """Resolve a task nomination, cold-refreshing stale source before dispatch.

    Calling this explicitly enables refresh. Original artifacts are immutable;
    only the nominated scope is rescanned. Deleted files, changed nominations,
    escaped paths, and task mismatch fail closed. Refresh is observational and
    never turns a candidate or merge into an accepted canonical world root.
    """
    args = dict(
        repository=repository,
        artifact=artifact,
        expected_sha256=expected_sha256,
        task_id=task_id,
        max_payload_bytes=max_payload_bytes,
    )
    try:
        text = load_semantic_worker_context(**args)
        return {
            "text": text,
            "artifact": artifact,
            "sha256": expected_sha256,
            "refreshed": False,
            "refresh_lineage": None,
        }
    except SemanticContextStale:
        pass
    # Revalidate the original nomination and all path bounds even though its
    # source hashes are now stale. A digest/schema/task error is never repaired.
    _, previous = _load_semantic_payload(**args, verify_sources=False)
    required = previous.get("required_raw_paths", sorted(previous.get("raw_sources", {})))
    if (
        not isinstance(required, list)
        or not required
        or any(not isinstance(name, str) for name in required)
        or not set(required) <= set(previous["manifest"])
    ):
        raise ValueError("invalid semantic refresh required source policy")
    if not isinstance(attempt_id, str) or not attempt_id.strip() or len(attempt_id) > 256:
        raise ValueError("semantic refresh requires bounded attempt identity")
    root = repository.resolve(strict=True)
    parent = Path(refresh_output).absolute()
    if (
        not parent.is_relative_to(root)
        or parent.resolve() != parent
        or any((root / name).is_relative_to(parent) for name in previous["manifest"])
    ):
        raise ValueError(
            "semantic refresh output must be a separate repository-contained directory"
        )
    output = parent / uuid.uuid4().hex
    lineage = {
        "schema": "supervisor-semantic-context-refresh@1",
        "previous_payload_sha256": expected_sha256,
        "previous_scope_cid": previous["scope_cid"],
        "previous_semantic_root_cid": previous["semantic_root_cid"],
        "previous_manifest": previous["manifest"],
        "attempt_id": attempt_id,
        "cause": "source_changed_before_dispatch",
    }
    result = prepare_semantic_context(
        repository=root,
        paths=tuple(previous["manifest"]),
        required_raw_paths=required,
        objective=previous["objective"],
        task_id=task_id,
        output=output,
        max_payload_bytes=max_payload_bytes,
        _refresh_lineage=lineage,
        max_symbols=previous.get("preparation_bounds", {}).get("max_symbols", 256),
        context_input_tokens=previous.get("preparation_bounds", {}).get("context_input_tokens", 8192),
        worker_query=previous.get("worker_projection", {}).get("query", ""),
        worker_capsule_limit=previous.get("worker_projection", {}).get("max_capsules", 8),
        worker_max_bytes=previous.get("worker_projection", {}).get("max_bytes", 32768),
    )
    refreshed_artifact = (output / "worker-context.json").relative_to(root).as_posix()
    text = load_semantic_worker_context(
        repository=root,
        artifact=refreshed_artifact,
        expected_sha256=result["worker_payload_sha256"],
        task_id=task_id,
        max_payload_bytes=max_payload_bytes,
    )
    return {
        "text": text,
        "artifact": refreshed_artifact,
        "sha256": result["worker_payload_sha256"],
        "refreshed": True,
        "refresh_lineage": result["refresh_lineage"],
    }
