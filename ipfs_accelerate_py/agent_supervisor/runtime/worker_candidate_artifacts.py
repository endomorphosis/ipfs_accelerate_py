"""Fresh advisory artifacts over an owner-captured working generation.

Datasets owns capsule facts and numerical grammar. This consumer only binds
their unchanged outputs to captured bytes; it publishes no accepted index,
reuses no proofs, and executes neither the target nor a model or prover.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path, PurePosixPath
import re

MAX_FILES = 64
MAX_SOURCE_BYTES = 1024 * 1024
MAX_SYMBOLS = 256
MAX_OUTPUT_BYTES = 16 * 1024 * 1024
AUTHORITY = dict(read_only=True, baseline_evidence_applicable=False,
    semantic_acceptance_authority=False, proof_authority=False,
    formalization_authority=False, execution_authority=False,
    completion_authority=False, omission_authority=False, mutation_authority=False,
    source_semantics_verified=False, whole_program_semantics_verified=False,
    source_obligations_discharged=0, prover_calls=0, solver_calls=0,
    provider_calls=0, model_inference_calls=0, training_steps=0)

_PRODUCERS = (
    __name__,
    "ipfs_accelerate_py.agent_supervisor.runtime.worker_candidate_selection",
    "ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime",
    "ipfs_accelerate_py.agent_supervisor.runtime.semantic_capsule_selection",
    "ipfs_accelerate_py.agent_supervisor.semantic_state.datasets_adapter",
    "ipfs_accelerate_py.agent_supervisor.semantic_state.wire",
    *('ipfs_datasets_py.logic.software_contracts.' + name for name in (
        'content', 'python_frontend', 'semantic_index.scanner', 'semantic_index.snapshot',
        'semantic_index.python_analysis', 'semantic_index.pytest_analysis',
        'semantic_index.symbol_graph', 'semantic_index.identity', 'semantic_index.models',
        'semantic_state.api', 'semantic_state.models', 'semantic_state.capsules',
        'semantic_state.bindings', 'semantic_state.merkle')),
    *('ipfs_datasets_py.logic.software_verification.' + name for name in (
        'numerical_logic', 'numerical_source', 'numerical_obligation_retrieval',
        'source_adapters', 'syntax_bridge', 'program')),
    'ipfs_datasets_py.logic.backends.smt.compiler',
    'ipfs_datasets_py.logic.families.registry',
    'ipfs_datasets_py.logic.syntax_core.ast',
    'ipfs_datasets_py.logic.syntax_core.signatures',
)


def _wire(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _producer_binding():
    return {name: _sha(Path(importlib.import_module(name).__file__).read_bytes())
        for name in sorted(_PRODUCERS)}


def _inputs(sources, program_paths, generation):
    _require(type(sources) is dict and 1 <= len(sources) <= MAX_FILES,
        "bounded captured candidate sources required")
    for name, raw in sources.items():
        _require(type(name) is str and 0 < len(name.encode()) <= 4096
            and name != "." and not PurePosixPath(name).is_absolute()
            and str(PurePosixPath(name)) == name and ".." not in PurePosixPath(name).parts
            and not any(value in name for value in ("\\", "\0", "\r", "\n"))
            and ".git" not in PurePosixPath(name).parts,
            "canonical candidate source path required")
        _require(type(raw) is bytes, "exact captured candidate bytes required")
    _require(sum(map(len, sources.values())) <= MAX_SOURCE_BYTES,
        "candidate source byte bound exceeded")
    _require(type(program_paths) is tuple and all(type(name) is str for name in program_paths)
        and program_paths == tuple(sorted(set(program_paths)))
        and set(program_paths) <= set(sources),
        "canonical captured candidate program subset required")
    _require(type(generation) is str and re.fullmatch(r"[0-9a-f]{64}", generation) is not None,
        "exact candidate workspace generation required")
    return dict(sorted(sources.items()))


def build_candidate_artifacts(*, sources: dict[str, bytes],
        program_paths: tuple[str, ...], workspace_generation: str) -> dict[str, bytes]:
    """Build a private, read-only generation cache without filesystem authority.

    Source capture, freshness checks and cache lifetime belong to the caller.
    The temporary scanner checkout contains only captured program bytes. The
    returned artifacts are advisory; native structural coverage is not a proof.
    Unsupported paths are accounted for explicitly rather than silently omitted.
    """
    sources = _inputs(sources, program_paths, workspace_generation)
    from .supervisor_meta_index import public_replay_without_metadata

    with public_replay_without_metadata():
        from .semantic_context_runtime import _scan_scoped_sources
        from .semantic_capsule_selection import immutable_bundle_capsule_reader
        from ..semantic_state.datasets_adapter import IpfsDatasetsSemanticStateProvider
        from ..semantic_state.wire import cid_for_payload
        from ipfs_datasets_py.logic.software_verification.numerical_logic import (
            formalize_numerical_source, describe_numerical_logic_profile)
        from ipfs_datasets_py.logic.software_verification.numerical_source import (
            MAX_SOURCE_BYTES as MAX_NUMERICAL_BYTES)

        producers = _producer_binding()
        hashes = {name: _sha(raw) for name, raw in sources.items()}
        scope = dict(schema="supervisor-worker-candidate-scope@1",
            workspace_generation=workspace_generation, source_hashes=hashes,
            program_paths=list(program_paths))
        scope_cid = cid_for_payload(scope)
        state = _scan_scoped_sources({name: sources[name] for name in program_paths},
            repository_id=scope_cid, max_symbols=MAX_SYMBOLS)
        provider = IpfsDatasetsSemanticStateProvider()
        bundle = provider.build_semantic_state(state)
        view = provider.view_semantic_state_bundle(bundle)
        reader = immutable_bundle_capsule_reader(bundle, root_cid=view.root.root_cid)
        capsules = [reader(symbol.stable_id).to_dict() for symbol in state.symbols]
        artifacts = [artifact.to_dict() for artifact in state.artifacts]
        reports, frontiers, coverage = {}, [], []
        for name, raw in sources.items():
            selected = name in program_paths
            native = [artifact for artifact in state.artifacts if artifact.path == name]
            count = sum(symbol.module_path == name for symbol in state.symbols)
            row = dict(path=name, source_sha256=hashes[name], source_bytes=len(raw),
                role="program" if selected else "support", capsule_count=count,
                semantic_status=("opaque" if any(str(item.confidence) == "opaque" for item in native)
                    else "scanned") if selected else "not_scanned_support",
                formal_status="not_selected_support", pending_obligation_count=0,
                families_with_artifacts=[])
            if selected:
                reason = None
                if not name.endswith(".py"):
                    reason = "numerical_grammar_requires_python_path"
                elif not raw:
                    reason = "numerical_grammar_requires_nonempty_source"
                elif len(raw) > MAX_NUMERICAL_BYTES:
                    reason = "numerical_grammar_source_byte_frontier"
                else:
                    try:
                        source = raw.decode("utf-8")
                    except UnicodeDecodeError:
                        reason = "numerical_grammar_requires_utf8"
                if reason is not None:
                    row.update(formal_status="unsupported", formal_frontier=reason)
                    frontiers.append(dict(path=name, source_sha256=hashes[name],
                        status="unsupported", reason=reason))
                else:
                    report = formalize_numerical_source(source, path=name)
                    _require(report["source_sha256"] == hashes[name],
                        "native candidate formal source binding differs")
                    reports[name] = report
                    row.update(formal_status=report["summary"]["structural_status"],
                        formal_report_sha256=report["report_sha256"],
                        pending_obligation_count=report["summary"]["pending_obligation_count"],
                        families_with_artifacts=report["summary"]["families_with_artifacts"],
                        functions=report["summary"]["functions"],
                        structurally_complete_functions=report["summary"]["structurally_complete_functions"],
                        partial_or_unsupported_functions=report["summary"]["partial_or_unsupported_functions"])
            coverage.append(row)
        _require(_producer_binding() == producers, "candidate producer sources changed during build")
        common = dict(workspace_generation=workspace_generation,
            source_state="working-candidate", source_hashes=hashes,
            program_paths=list(program_paths), candidate_scope_cid=scope_cid,
            semantic_root_cid=view.root.root_cid,
            producer_binding_sha256=_sha(_wire(producers)), **AUTHORITY)
        result = {
            "capsules": _wire(dict(schema="supervisor-worker-candidate-capsules@1", **common,
                capsule_index_cid=view.root.capsule_index_cid, capsules=capsules,
                analysis_artifacts=artifacts)),
            "formal": _wire(dict(schema="supervisor-worker-candidate-formal@1", **common,
                profile=describe_numerical_logic_profile(), reports=reports, frontiers=frontiers)),
        }
        result["summary"] = _wire(dict(schema="supervisor-worker-candidate-summary@1", **common,
            source_files=len(sources), program_files=len(program_paths),
            support_files=len(sources) - len(program_paths), capsule_count=len(capsules),
            formal_report_count=len(reports), formal_frontier_count=len(frontiers),
            pending_obligation_count=sum(row["pending_obligation_count"] for row in coverage),
            coverage=coverage, producer_sources=producers,
            producer_binding_scope="direct_native_producer_sources@1",
            artifacts={key: dict(sha256=_sha(raw), bytes=len(raw)) for key, raw in result.items()},
            interpretation="Fresh structural facts and pending obligations only; baseline proofs do not apply. "
                "Unsupported constructs and logic families retain native diagnostics. No lake build was run."))
        _require(sum(map(len, result.values())) <= MAX_OUTPUT_BYTES,
            "candidate artifact output byte bound exceeded")
        return result


__all__ = ["build_candidate_artifacts"]
