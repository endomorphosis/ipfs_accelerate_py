"""Protected owner projection for one independently admitted candidate dispatch.

Publication needs live typed owner custody, not a serialized advisory checkpoint.
Workers verify the exact projection and current candidate sources immediately
before provider dispatch. Expected provider edits invalidate those sources;
this gate must never be used to accept or reject the resulting postimage.
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
import uuid

from . import source_formalization_context as guard
from . import worker_candidate_round_client as protocol
from .worker_source_views import canonical_json

SCHEMA = "supervisor-candidate-dispatch-owner-handoff@1"
OBSERVATION_SCHEMA = "supervisor-candidate-dispatch-observation@1"
DOCTOR_SCHEMA = "supervisor-candidate-dispatch-owner-handoff@2"
DOCTOR_OBSERVATION_SCHEMA = "supervisor-candidate-dispatch-observation@2"
HANDOFF_FLAGS = ("--candidate-dispatch-handoff", "--candidate-dispatch-handoff-sha256")
MAX_HANDOFF_BYTES = 256 * 1024
MAX_LIFETIME_NS = 90 * 1_000_000_000
# Doctor preparation/revalidation can exceed the ordinary publication window.
# Its explicit owner policy already bounds the entire original coding attempt;
# the descriptor and that unchanged deadline still constrain every dispatch.
MAX_DOCTOR_LIFETIME_NS = 1800 * 1_000_000_000
FLAGS = dict(proof_authority=False, formalization_authority=False,
    execution_authority=False, completion_authority=False, settlement_authority=False,
    baseline_evidence_applicable=False, source_semantics_verified=False,
    whole_program_semantics_verified=False, source_obligations_discharged=0)
_TOKEN = object()


def _need(value, reason):
    if not value:
        raise ValueError(reason)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _identity(path):
    value = path.lstat()
    return value.st_dev, value.st_ino


def _parse(raw):
    value = json.loads(raw, object_pairs_hook=guard._unique, parse_constant=guard._nonfinite)
    _need(type(value) is dict and canonical_json(value) == raw, "canonical candidate handoff required")
    return value


def _source_population(value, *, empty=False):
    _need(type(value) is dict and (0 if empty else 1) <= len(value) <= 768,
          "bounded candidate source population required")
    for name, digest in value.items():
        guard._relative(name)
        _need(".git" not in Path(name).parts and guard._digest(digest),
              "exact candidate source population required")
    return value


def _artifact_binding(snapshot, metadata):
    _need(set(snapshot.artifacts) == {"summary", "capsules", "formal"},
          "complete native candidate artifact set required")
    for name, raw in snapshot.artifacts.items():
        value = _parse(raw)
        _need(metadata["artifacts"][name] == dict(sha256=_sha(raw), bytes=len(raw))
            and all(value[key] == metadata[key] for key in ("workspace_generation", "source_hashes",
                "program_paths", "candidate_scope_cid", "semantic_root_cid", "producer_binding_sha256"))
            and all(value[key] is False for key in ("proof_authority", "formalization_authority",
                "execution_authority", "completion_authority", "baseline_evidence_applicable"))
            and value["source_obligations_discharged"] == 0,
            "candidate artifact differs from source-bound native snapshot")


@dataclass(frozen=True, slots=True, init=False)
class CandidateDispatchHandoff:
    path: Path
    sha256: str
    provider_prompt: bytes = field(repr=False)
    metadata_bytes: bytes = field(repr=False)
    _file_identity: tuple = field(repr=False)

    def __init__(self, *, _token, path, raw, provider_prompt):
        _need(_token is _TOKEN, "candidate handoff factory required")
        for key, value in (("path", path), ("sha256", _sha(raw)),
                ("provider_prompt", provider_prompt), ("metadata_bytes", raw),
                ("_file_identity", _identity(path))):
            object.__setattr__(self, key, value)

    @property
    def metadata(self):
        return json.loads(self.metadata_bytes)

    @property
    def extra_argv(self):
        return HANDOFF_FLAGS[0], str(self.path), HANDOFF_FLAGS[1], self.sha256

    def close(self):
        """Remove only the exact owner publication created by this handle."""
        try:
            if _identity(self.path) == self._file_identity:
                self.path.unlink()
        except FileNotFoundError:
            pass


def prepare_candidate_dispatch_handoff(*, snapshot, prompt, source_broker, admission):
    """Project exact typed owner admission into a protected worker-readable file."""
    from .owner_candidate_round_context import OwnerCandidateRoundContext
    from .owner_candidate_round_prompt import OwnerCandidateRoundPrompt
    from .worker_candidate_round_broker import WorkerCandidateRoundBroker
    from .owner_candidate_dispatch import CandidateDispatchAdmission

    _need(type(snapshot) is OwnerCandidateRoundContext and type(prompt) is OwnerCandidateRoundPrompt
        and type(source_broker) is WorkerCandidateRoundBroker
        and type(admission) is CandidateDispatchAdmission,
        "exact retained candidate snapshot, prompt, broker and dispatch admission required")
    _need(prompt._snapshot is snapshot and source_broker._snapshot is snapshot,
          "candidate dispatch objects belong to different rounds")
    admitted = admission.revalidate(snapshot=snapshot, prompt=prompt, source_broker=source_broker)
    metadata, prompt_metadata = snapshot.revalidate(), prompt.revalidate()
    source_broker._current_context()
    _artifact_binding(snapshot, metadata)
    _need(not source_broker._registered.is_set() and source_broker._process is None,
          "candidate handoff must precede child registration")
    boundary, boundary_sha = guard._owner_boundary(source_broker.boundary)
    owner = source_broker.owner_uid
    _need(boundary["owner_uid"] == owner == os.getuid() == os.geteuid() == os.getgid()
        and boundary["worker_uid"] == source_broker.worker_uid
        and boundary_sha == source_broker.descriptor["boundary_manifest_sha256"],
        "candidate dispatch owner boundary changed")
    parent = Path(source_broker.boundary).absolute().parent / "worker-observations"
    guard._protected_directory(parent, owner=owner, group=owner)
    parent_identity = _identity(parent)
    descriptor_path = source_broker.descriptor_path
    descriptor_raw = guard._read(descriptor_path, protocol.MAX_FRAME_BYTES)
    info = descriptor_path.lstat()
    _need(descriptor_path.parent == parent and info.st_uid == info.st_gid == owner
        and stat.S_IMODE(info.st_mode) == 0o440
        and descriptor_raw == canonical_json(source_broker.descriptor)
        and _sha(descriptor_raw) == source_broker.descriptor_sha256,
        "candidate source descriptor publication changed")
    doctor_selected = admitted["schema"] == "supervisor-owner-candidate-dispatch-admission@2"
    lifetime = MAX_DOCTOR_LIFETIME_NS if doctor_selected else MAX_LIFETIME_NS
    issued = time.monotonic_ns()
    expires = min(issued + lifetime, source_broker.descriptor["expires_monotonic_ns"],
                  source_broker._deadline_ns, admitted["deadline_monotonic_ns"])
    _need(type(expires) is int and expires - issued > 1_000_000_000,
          "candidate handoff has insufficient original deadline")
    prior = snapshot._source_broker
    baseline_manifest = json.loads(prior._table_bytes)["source_manifest"]
    baseline_sources = _source_population({name: row["sha256"] for name, row in baseline_manifest.items()})
    public_sources = json.loads(prompt._requirements._public_sources_bytes)
    body = dict(schema=SCHEMA, repository=str(source_broker.repository), workspace=str(source_broker.workspace),
        boundary_sha256=boundary_sha, descriptor_path=str(descriptor_path),
        descriptor_sha256=source_broker.descriptor_sha256,
        provider_input_sha256=_sha(prompt.provider_prompt), provider_input_bytes=len(prompt.provider_prompt),
        native_prompt_sha256=metadata["native_prompt_sha256"], native_prompt_bytes=len(snapshot.native_prompt),
        baseline_commit=prompt._requirements.metadata["baseline_commit"],
        baseline_binding=metadata["baseline_binding"], baseline_source_hashes=baseline_sources,
        baseline_source_manifest=baseline_manifest,
        public_source_manifest=public_sources, source_hashes=metadata["source_hashes"],
        program_paths=metadata["program_paths"], candidate_artifacts=metadata["artifacts"],
        producer_binding_sha256=metadata["producer_binding_sha256"], admission=admitted,
        prompt_observation=prompt_metadata, issued_monotonic_ns=issued, expires_monotonic_ns=expires,
        **FLAGS)
    if doctor_selected:
        from .candidate_doctor_dispatch import prepare_doctor_dispatch_projection
        body.update(schema=DOCTOR_SCHEMA,
            doctor=prepare_doctor_dispatch_projection(prompt=prompt, admission=admitted))
    raw = canonical_json(body)
    _need(len(raw) <= MAX_HANDOFF_BYTES, "candidate dispatch handoff exceeds bound")
    path = parent / ("candidate-dispatch-" + uuid.uuid4().hex + ".json")
    created = None
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        created_info = os.fstat(descriptor)
        created = created_info.st_dev, created_info.st_ino
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fchmod(stream.fileno(), 0o440)
            os.fsync(stream.fileno())
        _need(admission.revalidate(snapshot=snapshot, prompt=prompt, source_broker=source_broker) == admitted,
              "candidate admission changed during handoff publication")
        _need(prompt.revalidate() == prompt_metadata and snapshot.revalidate() == metadata,
              "candidate context changed during handoff publication")
        source_broker._current_context()
        guard._protected_directory(parent, owner=owner, group=owner)
        _need(_identity(parent) == parent_identity and _identity(path) == created
            and guard._read(path, MAX_HANDOFF_BYTES) == raw
            and time.monotonic_ns() < expires,
            "candidate handoff changed or expired during publication")
        fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        _need(time.monotonic_ns() < expires, "candidate handoff expired before publication")
        return CandidateDispatchHandoff(_token=_TOKEN, path=path, raw=raw, provider_prompt=prompt.provider_prompt)
    except BaseException:
        if created is not None:
            try:
                if _identity(path) == created:
                    path.unlink()
            except FileNotFoundError:
                pass
        raise


def validate_candidate_dispatch_context(*, prompt, repository, workspace, boundary,
        boundary_sha256, handoff, handoff_sha256, descriptor, descriptor_sha256,
        descriptor_path, launch_settings):
    """Worker-side validation before dispatch, without private owner replay."""
    _need(type(prompt) is str and len(prompt.encode()) <= 256_000,
          "bounded exact candidate provider input required")
    root, allocated = guard._workspace(repository, workspace)
    verified = guard._worker_boundary(Path(boundary).absolute(), boundary_sha256, allocated)
    owner = verified["owner_uid"]
    parent = Path(boundary).absolute().parent / "worker-observations"
    guard._protected_directory(parent, owner=owner, group=owner, worker=True)
    current = protocol.load_descriptor(Path(descriptor_path), descriptor_sha256, workspace=allocated)
    _need(current == descriptor and Path(descriptor_path).parent == parent
        and current["boundary_manifest_sha256"] == boundary_sha256
        and current["owner_uid"] == owner and current["worker_uid"] == verified["worker_uid"],
        "candidate descriptor differs from worker boundary")
    path = Path(handoff).absolute()
    _need(path.parent == parent and re.fullmatch(r"candidate-dispatch-[0-9a-f]{32}\.json", path.name)
        and guard._digest(handoff_sha256), "exact candidate handoff nomination required")
    raw = guard._read(path, MAX_HANDOFF_BYTES)
    info = path.lstat()
    _need(info.st_uid == info.st_gid == owner and stat.S_IMODE(info.st_mode) == 0o440
        and not os.access(path, os.W_OK) and _sha(raw) == handoff_sha256,
        "candidate handoff ownership, mode or digest differs")
    value = _parse(raw)
    fields = {"schema", "repository", "workspace", "boundary_sha256", "descriptor_path",
        "descriptor_sha256", "provider_input_sha256", "provider_input_bytes", "native_prompt_sha256",
        "native_prompt_bytes", "baseline_commit", "baseline_binding", "baseline_source_hashes", "baseline_source_manifest",
        "public_source_manifest", "source_hashes", "program_paths", "candidate_artifacts",
        "producer_binding_sha256", "admission", "prompt_observation", "issued_monotonic_ns",
        "expires_monotonic_ns", *FLAGS}
    doctor_selected = value.get("schema") == DOCTOR_SCHEMA
    _need(set(value) == fields | ({"doctor"} if doctor_selected else set())
        and value["schema"] in (SCHEMA, DOCTOR_SCHEMA)
        and all(type(value[key]) is type(expected) and value[key] == expected for key, expected in FLAGS.items())
        and value["repository"] == str(root) and value["workspace"] == str(allocated)
        and value["boundary_sha256"] == boundary_sha256
        and value["descriptor_path"] == str(descriptor_path)
        and value["descriptor_sha256"] == descriptor_sha256
        and value["provider_input_sha256"] == _sha(prompt.encode())
        and value["provider_input_bytes"] == len(prompt.encode()),
        "candidate handoff differs from exact worker invocation")
    issued, expires = value["issued_monotonic_ns"], value["expires_monotonic_ns"]
    admitted = value["admission"]
    admission_schema = "supervisor-owner-candidate-dispatch-admission@2" if doctor_selected else "supervisor-owner-candidate-dispatch-admission@1"
    _need(type(admitted) is dict and admitted.get("schema") == admission_schema
        and type(launch_settings) is dict and type(admitted.get("launch_settings")) is dict
        and admitted["launch_settings"] == launch_settings
        and all(type(launch_settings[key]) is type(expected)
            for key, expected in admitted["launch_settings"].items())
        and type(admitted.get("deadline_monotonic_ns")) is int,
        "candidate launch settings or admission differ")
    lifetime = MAX_DOCTOR_LIFETIME_NS if doctor_selected else MAX_LIFETIME_NS
    _need(type(issued) is int and type(expires) is int and 0 < expires-issued <= lifetime
        and issued <= time.monotonic_ns() < expires
        and expires <= min(current["expires_monotonic_ns"], admitted["deadline_monotonic_ns"]),
        "candidate handoff expired or exceeds original deadline")
    _need(admitted.get("descriptor_sha256") == descriptor_sha256
        and all(admitted.get(key) == current[key] for key in (
            "task_id", "native_prompt_sha256", "prior_session_id", "prior_descriptor_sha256",
            "session_id", "workspace_generation", "round_context_sha256", "round_number",
            "candidate_scope_cid", "candidate_semantic_root_cid"))
        and admitted.get("candidate_prompt_sha256") == value["provider_input_sha256"]
        and value["baseline_binding"] == {key: current[key] for key in protocol.BINDING_FIELDS},
        "candidate round binding differs from admitted descriptor")
    wire = _parse(prompt.encode())
    candidate = wire.get("candidate_binding", {})
    prompt_schema = "supervisor-owner-candidate-router-input@2" if doctor_selected else "supervisor-owner-candidate-router-input@1"
    _need(wire.get("schema") == prompt_schema
        and type(wire.get("native_prompt")) is str
        and _sha(wire["native_prompt"].encode()) == value["native_prompt_sha256"] == current["native_prompt_sha256"]
        and len(wire["native_prompt"].encode()) == value["native_prompt_bytes"]
        and wire.get("baseline_binding") == value["baseline_binding"]
        and wire.get("candidate_artifact_references") == value["candidate_artifacts"]
        and all(candidate.get(key) == value[key] for key in ("source_hashes", "program_paths", "producer_binding_sha256"))
        and candidate.get("workspace_generation") == current["workspace_generation"]
        and candidate.get("candidate_scope_cid") == current["candidate_scope_cid"]
        and candidate.get("semantic_root_cid") == current["candidate_semantic_root_cid"],
        "candidate prompt source or symbolic artifact binding differs")
    _need(doctor_selected or not any(key.startswith("doctor_") for key in wire)
        and not any(key.startswith("doctor_") for key in admitted),
        "ordinary candidate dispatch cannot acquire Doctor context or policy")
    baseline = _source_population(value["baseline_source_hashes"])
    sources = _source_population(value["source_hashes"])
    _need(type(value["baseline_source_manifest"]) is dict
        and _sha(canonical_json(value["baseline_source_manifest"])) == current["source_manifest_sha256"]
        and {name: row["sha256"] for name, row in value["baseline_source_manifest"].items()} == baseline
        and set(sources) == set(baseline) and type(value["program_paths"]) is list
        and value["program_paths"] == sorted(set(value["program_paths"]))
        and set(value["program_paths"]) <= set(sources), "candidate source population changed")
    _need(guard._git(root, "rev-parse", "HEAD").decode().strip() == value["baseline_commit"],
          "candidate canonical baseline changed")
    guard._current_sources(root, baseline)
    from .router_public_instruction import _current_sources as public_sources
    public_sources(root, value["public_source_manifest"])
    guard._current_sources(allocated, sources)
    doctor_observation = None
    if doctor_selected:
        from .candidate_doctor_dispatch import validate_doctor_dispatch_projection
        doctor_observation = validate_doctor_dispatch_projection(projection=value["doctor"], wire=wire,
            handoff=value, descriptor=current, repository=root, workspace=allocated)
    final_info = path.lstat()
    _need(guard._read(path, MAX_HANDOFF_BYTES) == raw
        and final_info.st_uid == final_info.st_gid == owner
        and stat.S_IMODE(final_info.st_mode) == 0o440 and not os.access(path, os.W_OK)
        and protocol.load_descriptor(Path(descriptor_path), descriptor_sha256, workspace=allocated) == current
        and time.monotonic_ns() < expires,
          "candidate handoff changed or expired during source validation")
    result = dict(schema=DOCTOR_OBSERVATION_SCHEMA if doctor_selected else OBSERVATION_SCHEMA,
        **{key: current[key] for key in (
        "task_id", "native_prompt_sha256", "session_id", "prior_session_id", "prior_descriptor_sha256",
        "workspace_generation", "round_context_sha256", "round_number", "candidate_scope_cid",
        "candidate_semantic_root_cid")}, descriptor_sha256=descriptor_sha256,
        handoff_sha256=handoff_sha256, provider_input_sha256=value["provider_input_sha256"],
        provider_input_bytes=value["provider_input_bytes"], native_prompt_bytes=value["native_prompt_bytes"],
        candidate_artifacts=value["candidate_artifacts"], producer_binding_sha256=value["producer_binding_sha256"],
        source_files=len(sources), expires_monotonic_ns=expires,
        dispatch_deadline_monotonic_ns=min(admitted["deadline_monotonic_ns"], current["expires_monotonic_ns"]),
        remaining_timeout_seconds=max(0, (min(admitted["deadline_monotonic_ns"],
            current["expires_monotonic_ns"])-time.monotonic_ns()) // 1_000_000_000),
        validation_phase="before_provider", owner_validation_inherited=True,
        producer_validation_inherited=True, neural_inference_replayed=False,
        owner_private_state_replayed_here=False, canonical_source_current=True,
        workspace_source_current=True, **FLAGS)
    if doctor_observation is not None:
        result["doctor_candidate_dispatch"] = doctor_observation
    return result


__all__ = ["CandidateDispatchHandoff", "prepare_candidate_dispatch_handoff",
           "validate_candidate_dispatch_context"]
