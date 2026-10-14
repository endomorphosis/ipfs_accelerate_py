"""Durable owner-private evidence captured before a second coding child.

These exact historical bytes support reporting after the worker edits its
checkout and the live handoff is removed. This module provides no reload or
dispatch API. Retained advice is not a theorem, accepted index or completion
claim, and retaining its input does not establish native semantic equivalence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time
from types import MappingProxyType

from . import owner_candidate_round_checkpoint as storage
from . import source_formalization_context as guard
from .worker_source_views import canonical_json

SCHEMA = "supervisor-owner-candidate-dispatch-audit@1"
OBSERVATION_SCHEMA = "owner-candidate-dispatch-audit-observation@1"
MAX_MANIFEST_BYTES = 256 * 1024
MAX_RETAINED_BYTES = 24 * 1024 * 1024
MAX_SOURCE_BYTES = 1024 * 1024
ARTIFACT_FILENAMES = MappingProxyType(dict(
    summary="summary.json", capsules="capsules.json", formal="formal.json", context="context.json",
    candidate_input="candidate-input.json", native_prompt="native-prompt.txt",
    original_requirements="original-requirements.json",
    original_public_instruction="original-public-instruction.txt",
    original_public_artifact="original-public-artifact.bin", public_sources="public-sources.json",
    prompt_observation="prompt-observation.json", handoff="handoff.json", descriptor="descriptor.json",
    admission="admission.json", baseline_translation_table="baseline-translation-table.json"))
FILENAMES = ARTIFACT_FILENAMES
FLAGS = dict(audit_only=True, owner_private=True, worker_readable=False,
    retained_before_child_registration=True, candidate_input_retained=True,
    original_requirements_retained=True, dispatch_reload_allowed=False,
    proof_authority=False, formalization_authority=False, execution_authority=False,
    completion_authority=False, settlement_authority=False, retry_authority=False,
    accepted_index_published=False, source_semantics_verified=False,
    native_semantics_rebuilt=False, source_obligations_discharged=0)
_TOKEN = object()


def _need(value, reason):
    if not value:
        raise ValueError(reason)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def audit_directory_name(session_id):
    _need(type(session_id) is str and re.fullmatch(r"[0-9a-f]{32}", session_id) is not None,
          "exact candidate audit session required")
    return "candidate-audit-" + session_id


def audit_manifest_name(session_id):
    return audit_directory_name(session_id) + ".json"


def _runtime_bytes(path, expected, owner, *, digest):
    raw = guard._read(path, max(len(expected), 1))
    info = path.lstat()
    _need(raw == expected and _sha(raw) == digest and info.st_uid == info.st_gid == owner
        and stat.S_IMODE(info.st_mode) == 0o440,
        "candidate audit runtime publication changed")
    return raw


@dataclass(frozen=True, slots=True, init=False)
class CandidateDispatchAudit:
    """Immutable audit receipt. It is deliberately not a launch handle."""
    manifest_path: Path
    manifest_bytes: bytes = field(repr=False)

    def __init__(self, *, _token, manifest_path, manifest_bytes):
        _need(_token is _TOKEN, "candidate audit retention factory required")
        object.__setattr__(self, "manifest_path", manifest_path)
        object.__setattr__(self, "manifest_bytes", manifest_bytes)

    @property
    def manifest_sha256(self):
        return _sha(self.manifest_bytes)

    @property
    def observation(self):
        return {**json.loads(self.manifest_bytes)["observation"], "audit_manifest": dict(
            path=self.manifest_path.name, sha256=self.manifest_sha256, bytes=len(self.manifest_bytes))}


def persist_candidate_dispatch_audit(*, snapshot, prompt, source_broker, admission, handoff):
    """Retain one exact pre-dispatch package; fail without overwriting anything.

    The manifest is published last. Any failed publication rolls back only inodes
    created by this call. Successful packages intentionally survive launch failure,
    runtime cleanup and later source edits. The fixed name refuses duplicate writes.
    """
    from .owner_candidate_dispatch import CandidateDispatchAdmission
    from .owner_candidate_round_context import OwnerCandidateRoundContext
    from .owner_candidate_round_prompt import OwnerCandidateRoundPrompt
    from .worker_candidate_round_broker import WorkerCandidateRoundBroker
    from .candidate_dispatch_context import CandidateDispatchHandoff

    _need(type(snapshot) is OwnerCandidateRoundContext and type(prompt) is OwnerCandidateRoundPrompt
        and type(source_broker) is WorkerCandidateRoundBroker
        and type(admission) is CandidateDispatchAdmission and type(handoff) is CandidateDispatchHandoff,
        "exact live owner candidate handles required for audit retention")
    _need(prompt._snapshot is snapshot and source_broker._snapshot is snapshot,
          "candidate audit handles belong to different rounds")
    admitted = admission.revalidate(snapshot=snapshot, prompt=prompt, source_broker=source_broker)
    metadata, prompt_metadata = snapshot.metadata, prompt.metadata
    broker, owner = source_broker, source_broker.owner_uid
    _need(not broker._registered.is_set() and broker._process is None
        and handoff.provider_prompt == prompt.provider_prompt,
        "candidate audit must precede registration with exact prepared input")
    boundary, boundary_sha = guard._owner_boundary(broker.boundary)
    _need(owner == os.getuid() == os.geteuid() == os.getgid() == boundary["owner_uid"]
        and boundary["worker_uid"] == broker.worker_uid
        and boundary_sha == broker.descriptor["boundary_manifest_sha256"],
        "candidate audit owner boundary changed")
    parent = Path(broker.boundary).absolute().parent / "worker-observations"
    guard._protected_directory(parent, owner=owner, group=owner)
    _need(handoff.path.parent == broker.descriptor_path.parent == parent,
          "candidate audit runtime lineage changed")
    descriptor_raw = _runtime_bytes(broker.descriptor_path, canonical_json(broker.descriptor),
                                    owner, digest=broker.descriptor_sha256)
    handoff_raw = _runtime_bytes(handoff.path, handoff.metadata_bytes, owner, digest=handoff.sha256)
    handoff_value = json.loads(handoff_raw)
    _need(handoff_value["admission"] == admitted
        and handoff_value["provider_input_sha256"] == _sha(prompt.provider_prompt)
        and handoff_value["descriptor_sha256"] == broker.descriptor_sha256,
        "candidate audit handoff belongs to another invocation")
    deadline = min(admitted["deadline_monotonic_ns"], broker.descriptor["expires_monotonic_ns"],
                   handoff_value["expires_monotonic_ns"])
    def current_deadline():
        _need(time.monotonic_ns() < deadline, "candidate audit publication deadline exhausted")
    current_deadline()
    sources = broker.views.working_sources(expected_workspace_generation=metadata["workspace_generation"])
    _need({name: _sha(raw) for name, raw in sources.items()} == metadata["source_hashes"]
        and len(sources) <= 64 and sum(map(len, sources.values())) <= MAX_SOURCE_BYTES,
        "candidate audit captured source population differs")
    requirements = prompt._requirements
    payloads = dict(snapshot.artifacts, context=snapshot.metadata_bytes,
        candidate_input=prompt.provider_prompt, native_prompt=snapshot.native_prompt,
        original_requirements=requirements.metadata_bytes,
        original_public_instruction=requirements.public_instruction_block,
        original_public_artifact=requirements._public_artifact_bytes,
        public_sources=requirements._public_sources_bytes,
        prompt_observation=prompt.metadata_bytes, handoff=handoff_raw, descriptor=descriptor_raw,
        admission=canonical_json(admitted), baseline_translation_table=snapshot._source_broker._table_bytes)
    _need(set(payloads) == set(ARTIFACT_FILENAMES)
        and all(type(raw) is bytes for raw in payloads.values())
        and sum(map(len, payloads.values())) + sum(map(len, sources.values())) <= MAX_RETAINED_BYTES,
        "candidate audit payload exceeds retained byte bound")
    binding = {key: admitted[key] for key in ("round_number", "task_id", "native_prompt_sha256",
        "prior_session_id", "prior_descriptor_sha256", "session_id", "descriptor_sha256",
        "workspace_generation", "round_context_sha256", "candidate_scope_cid", "candidate_semantic_root_cid")}
    binding.update(producer_binding_sha256=metadata["producer_binding_sha256"],
        candidate_input_sha256=_sha(prompt.provider_prompt), handoff_sha256=handoff.sha256,
        original_requirements_sha256=_sha(requirements.metadata_bytes),
        public_instruction_block_sha256=_sha(requirements.public_instruction_block))
    directory = parent / audit_directory_name(binding["session_id"])
    manifest_path = parent / audit_manifest_name(binding["session_id"])
    parent_identity = storage._identity(parent)
    created, expected_files, directory_identity = {}, {}, None
    try:
        current_deadline()
        directory.mkdir(mode=0o700)
        directory_identity = storage._identity(directory)
        directory_info = directory.lstat()
        _need(directory_info.st_uid == directory_info.st_gid == owner
            and stat.S_IMODE(directory_info.st_mode) == 0o700,
            "exact owner-private candidate audit directory required")
        artifacts, retained_sources = {}, {}
        for key, raw in sorted(payloads.items()):
            current_deadline()
            path = directory / ARTIFACT_FILENAMES[key]
            storage._write(path, raw, created)
            expected_files[path] = raw
            artifacts[key] = dict(path=path.relative_to(parent).as_posix(), sha256=_sha(raw), bytes=len(raw))
        for index, (name, raw) in enumerate(sorted(sources.items())):
            current_deadline()
            path = directory / ("source-%04d.bin" % index)
            storage._write(path, raw, created)
            expected_files[path] = raw
            retained_sources[name] = dict(path=path.relative_to(parent).as_posix(), sha256=_sha(raw), bytes=len(raw))
        storage._sync_directory(directory)
        _need(admission.revalidate(snapshot=snapshot, prompt=prompt, source_broker=broker) == admitted
            and snapshot.metadata == metadata and prompt.metadata == prompt_metadata
            and broker.views.working_sources(expected_workspace_generation=metadata["workspace_generation"]) == sources,
            "candidate audit inputs changed before manifest publication")
        _runtime_bytes(broker.descriptor_path, descriptor_raw, owner, digest=broker.descriptor_sha256)
        _runtime_bytes(handoff.path, handoff_raw, owner, digest=handoff.sha256)
        guard._protected_directory(parent, owner=owner, group=owner)
        _need(storage._identity(parent) == parent_identity
            and storage._identity(directory) == directory_identity,
            "candidate audit directory changed before manifest publication")
        current_deadline()
        observation = dict(schema=OBSERVATION_SCHEMA, status="retained", **binding,
            retained_files=len(artifacts) + len(retained_sources),
            retained_bytes=sum(map(len, payloads.values())) + sum(map(len, sources.values())), **FLAGS)
        manifest = dict(schema=SCHEMA, binding=binding, artifacts=artifacts,
            candidate_sources=retained_sources, observation=observation, **FLAGS)
        raw = canonical_json(manifest)
        _need(len(raw) <= MAX_MANIFEST_BYTES, "candidate audit manifest exceeds byte bound")
        storage._write(manifest_path, raw, created)
        expected_files[manifest_path] = raw
        storage._sync_directory(parent)
        _need(admission.revalidate(snapshot=snapshot, prompt=prompt, source_broker=broker) == admitted,
            "candidate admission changed during audit publication")
        for path, identity in created.items():
            info = path.lstat()
            _need(storage._identity(path) == identity and info.st_uid == info.st_gid == owner
                and stat.S_IMODE(info.st_mode) == 0o400
                and guard._read(path, max(len(expected_files[path]), 1)) == expected_files[path],
                "candidate audit publication ownership changed")
        _runtime_bytes(broker.descriptor_path, descriptor_raw, owner, digest=broker.descriptor_sha256)
        _runtime_bytes(handoff.path, handoff_raw, owner, digest=handoff.sha256)
        _need(storage._identity(parent) == parent_identity
            and storage._identity(directory) == directory_identity,
            "candidate audit directory changed during publication")
        current_deadline()
        return CandidateDispatchAudit(_token=_TOKEN, manifest_path=manifest_path, manifest_bytes=raw)
    except BaseException:
        for path, identity in reversed(tuple(created.items())):
            try:
                if storage._identity(path) == identity:
                    path.unlink()
            except FileNotFoundError:
                pass
        if directory_identity is not None:
            try:
                if storage._identity(directory) == directory_identity:
                    directory.rmdir()
            except OSError:
                pass
        raise


__all__ = ["CandidateDispatchAudit", "persist_candidate_dispatch_audit", "audit_directory_name",
           "audit_manifest_name", "ARTIFACT_FILENAMES", "FILENAMES", "SCHEMA", "OBSERVATION_SCHEMA", "FLAGS"]
