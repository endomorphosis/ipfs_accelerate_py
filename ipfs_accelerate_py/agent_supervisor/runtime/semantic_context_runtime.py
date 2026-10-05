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


def _program_paths(value, captured):
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("explicit semantic program paths must be a sequence")
    names = tuple(value)
    if (any(type(name) is not str or name in {"", "."}
            or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
            or str(PurePosixPath(name)) != name for name in names)
            or names != tuple(sorted(set(names))) or not set(names) <= set(captured)):
        raise ValueError("semantic program paths must be a canonical unique sorted captured subset")
    return names


def _source_scope_payload(manifest, program_paths=None):
    payload = {"schema": "supervisor-source-scope@1", "sources": manifest}
    if program_paths is not None:
        payload.update(schema="supervisor-source-scope@2", program_paths=list(program_paths))
    return payload


def _doctor_for_scope(*, sources, manifest, scope_cid, repository_id, program_paths):
    from ..analysis.doctor_repository_diagnostics import DoctorAuthorityRoots, DoctorSourceUnit, diagnose_repository
    from ..analysis.doctor_contract_adapters import materialize_runtime_diagnostics
    from ..proof.formal_verification_contracts import content_identity
    selected = tuple(sorted(sources))
    config = {"paths": selected}
    if program_paths is not None:
        config["program_paths"] = list(program_paths)
    roots = DoctorAuthorityRoots(repository_id=repository_id, forest_id=scope_cid,
        tree_id=scope_cid, overlay_id=scope_cid, file_root_id=scope_cid, blob_root_id=scope_cid,
        config_id=content_identity(config), policy_id=content_identity({"mode": "context_preparation_only"}))
    names = selected if program_paths is None else program_paths
    diagnostic = diagnose_repository([
        DoctorSourceUnit(path=name, source_bytes=sources[name], blob_identity=manifest[name]["source_cid"])
        for name in names], authority_roots=roots)
    return materialize_runtime_diagnostics(diagnostic, require_repository_id=repository_id)


def _worker_payload(*, base, sources, manifest, required, state, bundle, view, provider,
                    worker_query, worker_capsule_limit, worker_max_bytes, context_input_tokens):
    from .semantic_capsule_selection import select_worker_capsules
    from ..semantic_state.context_pack import pack_context
    from ..context.context_contracts import ContextBudget
    from ..semantic_state.wire import cid_for_payload
    selected, scope_cid = tuple(sorted(sources)), base["scope_cid"]
    full_capsule_count = len(state.symbols)
    capsules, admissions, selected_symbols = select_worker_capsules(
        bundle=bundle, view=view, provider=provider, symbols=state.symbols,
        sources=sources, required=required, worker_query=worker_query,
        worker_capsule_limit=worker_capsule_limit, worker_max_bytes=worker_max_bytes)
    projection = None
    if worker_query:
        projection = {"schema": "supervisor-semantic-worker-projection@1",
            "selector": "lexical-query-symbol-name@1", "query": worker_query,
            "max_capsules": worker_capsule_limit, "max_bytes": worker_max_bytes,
            "full_capsule_count": full_capsule_count, "selected_capsule_count": len(capsules),
            "omitted_capsule_count": full_capsule_count - len(capsules),
            "selected_symbols": selected_symbols, "capsule_index_cid": view.root.capsule_index_cid,
            "raw_source_fetch_required": {name: manifest[name] for name in selected
                if name not in required and ("program_paths" not in base or name in base["program_paths"])},
            "omission_reason": "explicit bounded worker projection; full source and capsule index retained",
            "raw_required_caveat": "Read exact repository source before editing; omitted raw bytes are not substituted or proved equivalent by capsules.",
            "semantic_equivalence_claimed": False, "completion_authority": False}
    raw_scope_payload = {"schema": "required-source-set@1", "sources": {p: manifest[p] for p in required}}
    raw_scope = cid_for_payload(raw_scope_payload)
    delta_payload = {"schema": "initial-context-scan@1", "scope_cid": scope_cid}
    def packed_context():
        return pack_context(objective=base["objective"],
            target_source_cid=scope_cid if projection is not None else raw_scope,
            surrounding_source_cid=scope_cid, test_source_cid=raw_scope,
            delta_cid=cid_for_payload(delta_payload), dependency_admissions=admissions,
            budget=ContextBudget(max_input_tokens=context_input_tokens),
            assumptions=("Initial scoped context, not a full-repository world snapshot.",))
    packed = packed_context()
    raw_paths = set(required)
    if "program_paths" in base:
        raw_paths.update(set(selected) - set(base["program_paths"]))
    if projection is None:
        for admission, symbol in zip(admissions, state.symbols):
            if admission.ref.raw_source_required:
                raw_paths.add(symbol.module_path)
        raw_paths.update(set(selected) - {symbol.module_path for symbol in state.symbols})
    payload = {**base, "capsules": capsules, "admissions": [a.to_dict() for a in admissions],
        "raw_sources": {p: sources[p].decode() for p in sorted(raw_paths)}, "pack": packed.to_dict()}
    if projection is not None:
        payload["worker_projection"] = projection
    compact = _encoded(payload)
    while projection is not None and capsules and (len(compact) > worker_max_bytes or packed.budget_exceeded):
        capsules.pop()
        admissions.pop()
        projection["selected_symbols"].pop()
        projection["selected_capsule_count"] = len(capsules)
        projection["omitted_capsule_count"] = full_capsule_count - len(capsules)
        projection["exact_byte_budget_repacked"] = True
        packed = packed_context()
        payload["admissions"] = [admission.to_dict() for admission in admissions]
        payload["pack"] = packed.to_dict()
        compact = _encoded(payload)
    return payload, compact, packed, projection, full_capsule_count, raw_scope_payload, delta_payload


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
    program_paths: Sequence[str] | None = None,
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
    program = _program_paths(program_paths, selected)
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
    scope_payload = _source_scope_payload(manifest, program)
    scope_cid = cid_for_payload(scope_payload)
    provider = IpfsDatasetsSemanticStateProvider()
    provider.capability.require_available("build_semantic_state")
    repository_id = cid_for_payload({"repository": str(repository)})
    program_sources = sources if program is None else {name: sources[name] for name in program}
    state = _scan_scoped_sources(program_sources, repository_id=repository_id, max_symbols=max_symbols)
    bundle = provider.build_semantic_state(state)
    view = provider.view_semantic_state_bundle(bundle)
    root_cid = view.root.root_cid
    # A second producer scan starts from captured source, never from nominated
    # capsules or incremental state. This qualifies only this explicit scope.
    rebuilt = provider.build_semantic_state(
        _scan_scoped_sources(program_sources, repository_id=repository_id, max_symbols=max_symbols)
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
    if program is not None:
        reconstruction.update(schema="supervisor-scoped-semantic-reconstruction@2",
            program_paths=list(program), semantic_symbol_count=len(state.symbols),
            captured_source_count=len(selected), program_source_count=len(program),
            support_source_count=len(selected) - len(program), doctor_source_paths=list(program))
    refresh_lineage = None
    if _refresh_lineage is not None:
        refresh_lineage = dict(_refresh_lineage)
        previous_manifest = refresh_lineage.pop("previous_manifest")
        if set(previous_manifest) != set(manifest):
            raise ValueError("refresh cannot expand or shrink source scope")
        if program is not None:
            previous_program = refresh_lineage.pop("previous_program_paths", None)
            if previous_program != list(program):
                raise ValueError("refresh cannot expand or shrink semantic program scope")
            refresh_lineage["program_paths"] = list(program)
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
    doctor_snapshot, findings, bridge_cid = _doctor_for_scope(
        sources=sources, manifest=manifest, scope_cid=scope_cid,
        repository_id=state.repository_id, program_paths=program)
    base = {
        "schema": "supervisor-semantic-worker-context@1",
        "task_id": task_id, "objective": objective, "scope_cid": scope_cid,
        "semantic_root_cid": root_cid, "manifest": manifest,
        "required_raw_paths": list(required),
        "preparation_bounds": {"max_symbols": max_symbols, "context_input_tokens": context_input_tokens},
        "doctor_snapshot_id": doctor_snapshot.snapshot_id, "doctor_manifest_cid": bridge_cid,
        "reconstruction": reconstruction, "refresh_lineage": refresh_lineage,
        "completion_authority": False,
    }
    if program is not None:
        base.update(schema="supervisor-semantic-worker-context@2", program_paths=list(program))
    payload, compact, packed, projection, full_capsule_count, raw_scope_payload, delta_payload = _worker_payload(
        base=base, sources=sources, manifest=manifest, required=required,
        state=state, bundle=bundle, view=view, provider=provider,
        worker_query=worker_query, worker_capsule_limit=worker_capsule_limit,
        worker_max_bytes=worker_max_bytes, context_input_tokens=context_input_tokens)
    capsules = payload["capsules"]
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
    if program is not None:
        result.update(schema="supervisor-semantic-context-preparation@2", program_paths=list(program))
    (output / "result.json").write_bytes(_encoded(result, pretty=True))
    return result


class SemanticContextStale(ValueError):
    """A verified nomination is stale; only fresh source may replace it."""


def _validate_program_payload(payload, *, sources=None, repository=None):
    from ..semantic_state.wire import cid_for_payload
    fields = {"schema", "task_id", "objective", "scope_cid", "semantic_root_cid", "manifest",
        "required_raw_paths", "preparation_bounds", "doctor_snapshot_id", "doctor_manifest_cid",
        "reconstruction", "refresh_lineage", "completion_authority", "program_paths", "capsules",
        "admissions", "raw_sources", "pack"}
    if set(payload) not in (fields, fields | {"worker_projection"}):
        raise ValueError("closed explicit semantic program payload required")
    bounds = payload["preparation_bounds"]
    if (type(bounds) is not dict or set(bounds) != {"max_symbols", "context_input_tokens"}
            or type(bounds["max_symbols"]) is not int or not 1 <= bounds["max_symbols"] <= 1024
            or type(bounds["context_input_tokens"]) is not int
            or not 1 <= bounds["context_input_tokens"] <= 128000
            or payload["completion_authority"] is not False):
        raise ValueError("invalid explicit semantic preparation bounds or authority")
    manifest = payload["manifest"]
    if type(payload["program_paths"]) is not list:
        raise ValueError("explicit semantic program paths required")
    program = _program_paths(payload["program_paths"], manifest)
    required = payload["required_raw_paths"]
    if type(required) is not list or not required or _program_paths(required, manifest) != tuple(required):
        raise ValueError("canonical required semantic sources required")
    if payload["scope_cid"] != cid_for_payload(_source_scope_payload(manifest, program)):
        raise ValueError("explicit semantic program scope identity differs")
    reconstruction = payload["reconstruction"]
    count = reconstruction.get("semantic_symbol_count") if type(reconstruction) is dict else None
    expected = {"schema": "supervisor-scoped-semantic-reconstruction@2",
        "scope_cid": payload["scope_cid"], "semantic_root_cid": payload["semantic_root_cid"],
        "nomination_matched": True, "reuse": "cold", "full_repository": False,
        "semantic_acceptance_authority": False, "completion_authority": False,
        "program_paths": list(program), "semantic_symbol_count": count,
        "captured_source_count": len(manifest), "program_source_count": len(program),
        "support_source_count": len(manifest) - len(program), "doctor_source_paths": list(program)}
    if (type(count) is not int or not 0 <= count <= bounds["max_symbols"]
            or reconstruction != expected or any(type(reconstruction[key]) is not bool
                for key in ("nomination_matched", "full_repository", "semantic_acceptance_authority", "completion_authority"))
            or any(type(reconstruction[key]) is not int
                for key in ("captured_source_count", "program_source_count", "support_source_count"))):
        raise ValueError("explicit semantic program reconstruction differs")
    capsules, admissions, raw_sources = payload["capsules"], payload["admissions"], payload["raw_sources"]
    projection = payload.get("worker_projection")
    if projection is not None and (
            type(projection) is not dict or projection.get("schema") != "supervisor-semantic-worker-projection@1"
            or projection.get("selector") != "lexical-query-symbol-name@1"
            or projection.get("semantic_equivalence_claimed") is not False
            or projection.get("completion_authority") is not False
            or type(projection.get("query")) is not str or not projection["query"]
            or len(projection["query"].encode()) > 8192
            or type(projection.get("max_capsules")) is not int or not 0 <= projection["max_capsules"] <= 32
            or type(projection.get("max_bytes")) is not int or not 8192 <= projection["max_bytes"] <= 65536
            or len(_encoded(payload)) > projection["max_bytes"]):
        raise ValueError("invalid explicit semantic worker projection")
    if (type(capsules) is not list or type(admissions) is not list or len(capsules) != len(admissions)
            or len(capsules) > count or (projection is None and len(capsules) != count)
            or type(raw_sources) is not dict or not set(raw_sources) <= set(manifest)
            or not (set(required) | (set(manifest) - set(program))) <= set(raw_sources)):
        raise ValueError("explicit semantic program projection differs")
    for name, text in raw_sources.items():
        if type(text) is not str or hashlib.sha256(text.encode()).hexdigest() != manifest[name]["sha256"]:
            raise ValueError("explicit semantic raw source binding differs")
    lineage = payload["refresh_lineage"]
    if lineage is not None and (type(lineage) is not dict or lineage.get("program_paths") != list(program)):
        raise ValueError("semantic refresh changed its program subset")
    if sources is None:
        return program
    from ..semantic_state.datasets_adapter import IpfsDatasetsSemanticStateProvider
    provider = IpfsDatasetsSemanticStateProvider()
    repository_id = cid_for_payload({"repository": str(repository)})
    state = _scan_scoped_sources({name: sources[name] for name in program},
        repository_id=repository_id, max_symbols=payload["preparation_bounds"]["max_symbols"])
    bundle = provider.build_semantic_state(state)
    view = provider.view_semantic_state_bundle(bundle)
    if view.root.root_cid != payload["semantic_root_cid"] or len(state.symbols) != count:
        raise ValueError("cold explicit semantic program reconstruction differs")
    doctor, _, bridge_cid = _doctor_for_scope(sources=sources, manifest=manifest,
        scope_cid=payload["scope_cid"], repository_id=repository_id, program_paths=program)
    if doctor.snapshot_id != payload["doctor_snapshot_id"] or bridge_cid != payload["doctor_manifest_cid"]:
        raise ValueError("explicit semantic Doctor source replay differs")
    base = {key: value for key, value in payload.items()
        if key not in {"capsules", "admissions", "raw_sources", "pack", "worker_projection"}}
    policy = projection or {}
    rebuilt, _, packed, _, _, _, _ = _worker_payload(base=base, sources=sources, manifest=manifest,
        required=required, state=state, bundle=bundle, view=view, provider=provider,
        worker_query=policy.get("query", ""), worker_capsule_limit=policy.get("max_capsules", 8),
        worker_max_bytes=policy.get("max_bytes", 32768),
        context_input_tokens=payload["preparation_bounds"]["context_input_tokens"])
    if _encoded(rebuilt) != _encoded(payload) or packed.budget_exceeded:
        raise ValueError("explicit semantic producer capsules or context packing differ")
    return program


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
        payload.get("schema") not in {"supervisor-semantic-worker-context@1", "supervisor-semantic-worker-context@2"}
        or payload.get("task_id") != task_id
        or payload.get("completion_authority") is not False
    ):
        raise ValueError("semantic context task or schema mismatch")
    explicit = payload["schema"] == "supervisor-semantic-worker-context@2"
    if explicit:
        def unique(pairs):
            value = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError("duplicate semantic program payload field")
                value[key] = item
            return value
        payload = json.loads(raw, object_pairs_hook=unique)
    elif ("program_paths" in payload or (payload.get("reconstruction") or {}).get("schema") ==
            "supervisor-scoped-semantic-reconstruction@2"):
        raise ValueError("explicit semantic program scope requires its versioned schema")
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
    sources = {}
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
        if explicit and (set(binding) != {"sha256", "source_cid"}
                or type(binding.get("sha256")) is not str or len(binding["sha256"]) != 64
                or any(char not in "0123456789abcdef" for char in binding["sha256"])
                or type(binding.get("source_cid")) is not str or not binding["source_cid"]):
            raise ValueError("invalid explicit semantic source identity")
        if verify_sources:
            original = source.read_bytes()
            if hashlib.sha256(original).hexdigest() != binding.get("sha256"):
                raise SemanticContextStale("semantic context source is stale: " + name)
            if explicit:
                from ...mcp_server.mcplusplus.kubo_cid import cid_for_bytes
                if cid_for_bytes(original) != binding["source_cid"]:
                    raise ValueError("explicit semantic source CID differs")
                sources[name] = original
                if sum(map(len, sources.values())) > 1_000_000:
                    raise ValueError("semantic source scope exceeds byte bound")
    if explicit:
        _validate_program_payload(payload, sources=sources if verify_sources else None, repository=root)
        if verify_sources and any((root / name).read_bytes() != original for name, original in sources.items()):
            raise SemanticContextStale("semantic source changed during explicit program replay")
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
    program_kwargs = {}
    if previous["schema"] == "supervisor-semantic-worker-context@2":
        lineage["previous_program_paths"] = previous["program_paths"]
        program_kwargs["program_paths"] = previous["program_paths"]
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
        **program_kwargs,
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
