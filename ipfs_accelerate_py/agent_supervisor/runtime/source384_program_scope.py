"""Verified terminal program selection with immutable support provenance.

Current program hashes belong to the full Source384 ledger. Retained signed
manifest hashes establish the original support roles, not current program
semantics, source currentness, proof, planning or execution authority.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

from ..proof.formal_verification_contracts import content_identity
from . import local_planning_admission as local
from .terminal_source_partition import terminal_profile_partition
from .terminal_task_profile import PROFILE, INSTRUCTION, SMOKE

SCHEMA = "source384-terminal-program-scope@1"
ARTIFACT = "source-selection.json"
MAX_BYTES = 4 * 1024 * 1024


def _raw(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def verified_selection_manifest(envelope, *, repository, current=False):
    """Replay signatures without asserting historical program bytes are current."""
    if len(_raw(envelope)) > MAX_BYTES:
        raise ValueError("bounded Source384 selection manifest required")
    if current:
        manifest, _, _ = local._manifest(envelope, initial=True)
    else:
        if type(envelope) is not dict or type(envelope.get("payload")) is not dict:
            raise ValueError("signed Source384 selection manifest required")
        declared = envelope["payload"]
        local._validate_local_manifest_declarations(declared)
        profile = local.load_local_profile(repository_cid=declared["repository_cid"],
            profile_dir=Path(declared["profile_dir"]), lifecycle_dir=Path(declared["lifecycle_dir"]))
        manifest = local._verify_signature(envelope, profile)
        if manifest["profile_content_id"] != profile.content_id or any(not profile.allows(cap)
                for cap in ("read", "edit", "test", "isolated_worktree", "write_worktree")):
            raise ValueError("Source384 selection manifest owner changed")
    if "planning_inputs" in manifest:
        local._verify_planning_inputs(manifest)
    if manifest["schema"] == local.INTENT_MANIFEST_SCHEMA:
        local._verify_intent_requirements(manifest)
    root = Path(repository).absolute()
    if root.resolve(strict=True) != root or manifest["repository"] != str(root):
        raise ValueError("Source384 selection repository changed")
    return manifest


def program_scope(*, repository, source_hashes, envelope, current=False):
    """Keep every real input; exclude only exact independently signed support."""
    manifest = verified_selection_manifest(envelope, repository=repository, current=current)
    if set(source_hashes) != set(manifest["sources"]):
        raise ValueError("Source384 selection source population changed")
    if current and source_hashes != {name: row["sha256"] for name, row in manifest["sources"].items()}:
        raise ValueError("Source384 selection differs from initial signed source hashes")
    partition = terminal_profile_partition(repository=Path(repository), manifest=manifest)
    if partition is None:
        raise ValueError("Source384 scoped selection requires a canonical task profile")
    for name, _, digest in partition.support_hashes:
        if source_hashes[name] != digest:
            raise ValueError("Source384 immutable harness support changed")
    result = recorded_scope(source_hashes=source_hashes, envelope=envelope)
    if result["program_paths"] != list(partition.program_paths):
        raise ValueError("Source384 canonical profile program population differs")
    return result


def recorded_scope(*, source_hashes, envelope):
    """Reconstruct historical selection identity without source-current claims."""
    manifest = envelope["payload"]
    sources = manifest["sources"]
    support = ((INSTRUCTION, "instruction"), (PROFILE, "task_profile"), (SMOKE, "structural_smoke"))
    support_paths = {name for name, _ in support}
    if set(source_hashes) != set(sources) or not support_paths <= set(sources):
        raise ValueError("Source384 historical selection population differs")
    if any(source_hashes[name] != sources[name]["sha256"] for name in support_paths):
        raise ValueError("Source384 historical support hashes changed")
    return dict(schema=SCHEMA, manifest_cid=content_identity(envelope),
        selection_sha256=hashlib.sha256(_raw(envelope)).hexdigest(),
        source_population_sha256=hashlib.sha256(_raw(source_hashes)).hexdigest(),
        profile_sha256=sources[PROFILE]["sha256"], program_paths=sorted(set(sources) - support_paths),
        harness_support=[dict(path=name, role=role, sha256=sources[name]["sha256"])
                         for name, role in support],
        proof_authority=False, execution_authority=False, completion_authority=False,
        formalization_authority=False)


def require_selected_scope(source_hashes, scope):
    if (PROFILE in source_hashes) != (scope is not None):
        raise ValueError("Source384 canonical task profile requires verified program scope")
