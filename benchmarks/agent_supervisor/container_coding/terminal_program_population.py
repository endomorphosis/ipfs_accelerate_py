"""Authenticate public program/support populations before selecting retrieval.

An empty population is a source observation, never a vector configuration or a
claim about code outside the independently signed public input scope.
"""
from __future__ import annotations

import json
from pathlib import Path

EMPTY_INDEX_SCHEMA = "terminal-empty-program-population@1"


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def terminal_program_partition(prepared):
    from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
    from ipfs_accelerate_py.agent_supervisor.runtime.terminal_source_partition import terminal_profile_partition

    manifest, _, _ = local._manifest(prepared["manifest"], initial=True)
    partition = terminal_profile_partition(repository=Path(prepared["repository"]), manifest=manifest)
    if partition is None:
        return None
    if set(prepared["worker_inputs"]) != set(manifest["sources"]):
        raise ValueError("public context population differs from signed input scope")
    return dict(program_paths=list(partition.program_paths),
        support_hashes={name: dict(role=role, sha256=digest)
            for name, role, digest in partition.support_hashes})


def _empty_index(observation, *, model_snapshot, model_revision):
    return dict(schema=EMPTY_INDEX_SCHEMA, qualified=True, index_id=None, symbols=0,
        native_fact_rows_replayed=0, embedding_calls=0, training_steps=0, download_calls=0,
        selected_model=dict(snapshot=str(Path(model_snapshot).resolve(strict=True)) if model_snapshot else None,
            revision=model_revision, consumed=False),
        observation=observation, source_sha256=observation["source_sha256"],
        support_hashes=observation["support_hashes"],
        source_population_cid=observation["source_population_cid"],
        disposition=observation["disposition"],
        proof_authority=False, execution_authority=False, completion_authority=False,
        source_semantics_verified=False, model_inference_performed=False)


def qualify_terminal_population(*, prepared, output, paths, model_snapshot, model_revision):
    partition = terminal_program_partition(prepared)
    if partition is not None:
        from ipfs_accelerate_py.agent_supervisor.runtime.empty_code_retrieval import observe_empty_program_population
        if sorted(paths) != sorted(partition["program_paths"]):
            raise ValueError("retrieval program scope differs from signed partition")
        observation = observe_empty_program_population(repository=Path(prepared["repository"]), **partition)
        if observation is not None:
            indexed = _empty_index(observation, model_snapshot=model_snapshot, model_revision=model_revision)
            root = Path(prepared["repository"]).resolve(strict=True)
            output = Path(output).absolute()
            if (output.resolve() != output or not output.is_relative_to(root)
                    or output.exists()
                    or any((root / name).is_relative_to(output) for name in prepared["worker_inputs"])):
                raise ValueError("empty index output must be a new separate canonical repository directory")
            output.mkdir(parents=True, exist_ok=False)
            with (output / "result.json").open("x") as stream:
                json.dump(indexed, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
            return indexed, partition
    if model_snapshot:
        from .learned_vector_preflight import qualify
        indexed = qualify(Path(prepared["repository"]), output, paths, prepared["query"], model_snapshot, model_revision)
    else:
        from .vector_index_preflight import qualify
        indexed = qualify(Path(prepared["repository"]), output, paths, prepared["query"])
    return indexed, partition


def verify_empty_terminal_population(*, prepared, indexed, model_snapshot, model_revision):
    from ipfs_accelerate_py.agent_supervisor.runtime.empty_code_retrieval import observe_empty_program_population

    partition = terminal_program_partition(prepared)
    if partition is None:
        raise ValueError("empty public retrieval requires an independently signed program/support partition")
    observation = observe_empty_program_population(repository=Path(prepared["repository"]), **partition)
    if observation is None or _wire(indexed) != _wire(_empty_index(observation,
            model_snapshot=model_snapshot, model_revision=model_revision)):
        raise ValueError("empty public population differs from complete current native program scan")
    return partition
