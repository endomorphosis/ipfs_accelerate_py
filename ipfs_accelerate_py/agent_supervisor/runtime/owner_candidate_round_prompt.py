"""Owner-local original requirements and advisory candidate prompt preparation.

Capture requirements before a child starts and before source edits. A later
failed-worker snapshot can retain those exact requirements alongside a fresh
candidate summary. These handles have no serialization loader and authorize no
dispatch, proof reuse, scope expansion, publication or completion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path

from .worker_source_views import canonical_json

REQUIREMENTS_SCHEMA = "supervisor-owner-original-requirements@1"
PROMPT_SCHEMA = "supervisor-owner-candidate-router-input@1"
DOCTOR_PROMPT_SCHEMA = "supervisor-owner-candidate-router-input@2"
OBSERVATION_SCHEMA = "supervisor-owner-candidate-prompt-observation@1"
MAX_PROMPT_BYTES = 256_000
MAX_SUMMARY_BYTES = 65_536
_OWNER_TOKEN = object()
FLAGS = dict(dispatch_usable=False, baseline_evidence_applicable=False,
    proof_authority=False, execution_authority=False, completion_authority=False,
    settlement_authority=False, retry_authority=False, automatic_retry_admitted=False,
    scope_expansion_authority=False, accepted_index_published=False,
    source_obligations_discharged=0, model_inference_calls=0, provider_calls=0)
_INSTRUCTIONS = (
    "The native task prompt and original public requirement block below are retained exactly. "
    "Their task requirements, constraints, declared output scope and acceptance requirements remain required. "
    "Semantic and formal evidence inside that original prompt describes the canonical baseline; "
    "it is historical for the changed candidate and cannot supply candidate proofs. "
    "The separate candidate binding identifies fresh structural artifacts for one working generation. "
    "Artifact references are digest-bound advisory bytes, not admitted retrieval endpoints. "
    "Pending obligations and unsupported coverage remain unresolved; no lake build, proof acceptance, "
    "candidate execution, another worker or task completion is authorized by this preparation."
)


def _need(condition, reason):
    if not condition:
        raise ValueError(reason)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _head(repository):
    from .doctor_candidate_runner import _git
    return _git(repository, "rev-parse", "HEAD").decode().strip()


def _initial_binding(broker):
    from .worker_source_broker import WorkerSourceBroker
    from . import worker_source_client as protocol
    from .supervisor_meta_index import public_replay_without_metadata

    _need(type(broker) is WorkerSourceBroker, "exact source broker required for original requirements")
    _need(getattr(broker, "original_requirements_unavailable_reason", None)
        != "doctor_residual_context_not_captured",
        "candidate preparation has no captured Doctor residual obligations or budget")
    _need(not broker._closed and not broker._registered.is_set()
        and broker._process is None and broker._child is None,
        "original requirements must be captured before child registration")
    with public_replay_without_metadata():
        broker._current_context()
    protocol.validate_descriptor(broker.descriptor)
    owner = protocol.process_identity(os.getpid())
    _need(broker.descriptor["owner_pid"] == owner["pid"]
        and broker.descriptor["owner_start_time_ticks"] == owner["start_time_ticks"]
        and _sha(canonical_json(broker.descriptor)) == broker.descriptor_sha256,
        "original requirements source owner or descriptor changed")
    return dict(repository=str(broker.repository), workspace=str(broker.workspace),
        baseline_commit=_head(broker.repository), prior_session_id=broker.descriptor["session_id"],
        descriptor_sha256=broker.descriptor_sha256, baseline_binding=dict(broker._binding),
        native_prompt_sha256=_sha(broker.prompt.encode("utf-8")), task_id=broker._binding["task_id"])


def _public_artifact(artifact, expected_sha256):
    from . import router_public_instruction as public
    artifact = Path(artifact).absolute()
    parent = public._directory(artifact.parent)
    try:
        raw, _ = public._read_public_artifact(parent, artifact.name, maximum=public.MAX_SUCCESSOR_BYTES)
    finally:
        os.close(parent)
    _need(_sha(raw) == expected_sha256, "original public instruction artifact changed")
    return raw


@dataclass(frozen=True, slots=True, init=False)
class OriginalRequirements:
    """Unchanged requirement bytes captured through native initial admission."""
    native_prompt: bytes = field(repr=False)
    public_instruction_block: bytes = field(repr=False)
    metadata_bytes: bytes = field(repr=False)
    _source_broker: object = field(repr=False, compare=False)
    _public_request: tuple | None = field(repr=False)
    _public_artifact_bytes: bytes = field(repr=False)
    _public_sources_bytes: bytes = field(repr=False)
    _doctor_requirements: object = field(repr=False, compare=False)

    def __init__(self, *, _owner_token, native_prompt, public_instruction_block,
            metadata_bytes, source_broker, public_request, public_artifact_bytes, public_sources_bytes,
            doctor_requirements=None):
        _need(_owner_token is _OWNER_TOKEN, "owner original requirements factory required")
        for name, value in (("native_prompt", native_prompt),
                ("public_instruction_block", public_instruction_block), ("metadata_bytes", metadata_bytes),
                ("_source_broker", source_broker), ("_public_request", public_request),
                ("_public_artifact_bytes", public_artifact_bytes), ("_public_sources_bytes", public_sources_bytes),
                ("_doctor_requirements", doctor_requirements)):
            object.__setattr__(self, name, value)

    @property
    def metadata(self):
        return json.loads(self.metadata_bytes)

    def revalidate(self, *, snapshot):
        """Recheck canonical requirements without re-admitting edited sources."""
        from .owner_candidate_round_context import OwnerCandidateRoundContext
        from . import source_formalization_context as guard
        _need(type(snapshot) is OwnerCandidateRoundContext and snapshot._source_broker is self._source_broker,
            "original requirements belong to another owner snapshot")
        candidate = snapshot.revalidate()
        metadata = self.metadata
        broker = self._source_broker
        root, allocated = guard._workspace(broker.repository, broker.workspace)
        _need(str(root) == metadata["repository"] and str(allocated) == metadata["workspace"]
            and _head(root) == metadata["baseline_commit"]
            and snapshot.native_prompt == self.native_prompt == broker.prompt.encode("utf-8")
            and _sha(self.native_prompt) == metadata["native_prompt_sha256"]
            and candidate["baseline_binding"] == metadata["baseline_binding"]
            and all(candidate[key] == metadata[key] for key in (
                "prior_session_id", "descriptor_sha256", "task_id", "native_prompt_sha256")),
            "original task, baseline or source binding changed")
        if self._public_request is not None:
            from . import router_public_instruction as public
            artifact, expected_sha256, task_cid = self._public_request
            _need(_public_artifact(artifact, expected_sha256) == self._public_artifact_bytes,
                "retained original instruction artifact changed")
            # This native branch reconstructs signed historical bytes only. It
            # must not be reused as an authorization to dispatch the candidate.
            block, receipt = public.load_public_instruction(artifact=Path(artifact),
                expected_sha256=expected_sha256, task_cid=task_cid,
                prompt=self.native_prompt.decode("utf-8"), repository=root,
                require_current_source=False)
            expected = dict(metadata["public_instruction_receipt"],
                source_freshness_verified=False, historical_replay=True)
            _need(block.encode("utf-8") == self.public_instruction_block and receipt == expected,
                "original public instruction bytes or native receipt changed")
            public._current_sources(root, json.loads(self._public_sources_bytes))
            _need(_public_artifact(artifact, expected_sha256) == self._public_artifact_bytes,
                "original instruction artifact changed during replay")
        else:
            _need(metadata["selection"] == "native-only" and self.public_instruction_block == b""
                and metadata["public_instruction_receipt"] is None,
                "native-only original requirement capture differs")
        _need(_sha(self.public_instruction_block) == metadata["public_instruction_block_sha256"]
            and len(self.public_instruction_block) == metadata["public_instruction_block_bytes"],
            "original public requirement block differs")
        snapshot.revalidate()
        _need(_head(root) == metadata["baseline_commit"], "original canonical baseline changed during replay")
        if self._doctor_requirements is not None:
            from .doctor_candidate_context import OriginalDoctorRequirements
            _need(type(self._doctor_requirements) is OriginalDoctorRequirements
                and self._doctor_requirements._source_broker is broker
                and self._doctor_requirements.revalidate(snapshot=snapshot) == metadata.get("doctor_requirements"),
                "original Doctor requirements differ")
        else:
            _need("doctor_requirements" not in metadata and broker._doctor_request is None,
                "original Doctor requirement capture missing")
        return metadata


def _capture(*, source_broker, public_request=None, doctor_request=None):
    selected_doctor = getattr(source_broker, "_doctor_request", None)
    doctor_nomination = None
    if doctor_request is not None:
        _need(type(doctor_request) is dict and set(doctor_request) ==
            {"artifact", "expected_sha256", "task_cid"}, "complete original Doctor nomination required")
        doctor_nomination = (str(Path(doctor_request["artifact"]).absolute()),
            doctor_request["expected_sha256"], doctor_request["task_cid"])
    _need(selected_doctor is None or selected_doctor == doctor_nomination,
        "original Doctor selection cannot be omitted or substituted")
    before = _initial_binding(source_broker)
    native = source_broker.prompt.encode("utf-8")
    block, receipt, artifact_bytes, sources = b"", None, b"", b"{}"
    if public_request is not None:
        from . import router_public_instruction as public
        artifact, expected_sha256, task_cid = public_request
        text, receipt = public.load_public_instruction(artifact=Path(artifact),
            expected_sha256=expected_sha256, task_cid=task_cid,
            prompt=source_broker.prompt, repository=source_broker.repository,
            workspace=source_broker.workspace, require_current_source=True)
        block = text.encode("utf-8")
        artifact_bytes = _public_artifact(artifact, expected_sha256)
        payload = json.loads(artifact_bytes, object_pairs_hook=public._unique)
        manifest = public._selection(payload)
        _need(manifest["baseline_commit"] == before["baseline_commit"]
            and receipt["task_id"] == before["task_id"], "original signed instruction baseline differs")
        sources = canonical_json(manifest["sources"])
        public._current_sources(source_broker.repository, manifest["sources"])
        public._current_sources(source_broker.workspace, manifest["sources"])
        _need(_public_artifact(artifact, expected_sha256) == artifact_bytes,
            "original public instruction artifact changed during capture")
    doctor = None
    if doctor_request is not None:
        from .doctor_candidate_context import capture_doctor_requirements
        doctor = capture_doctor_requirements(source_broker=source_broker, **doctor_request)
    _need(_initial_binding(source_broker) == before
        and source_broker.prompt.encode("utf-8") == native,
        "original requirements changed during capture")
    if doctor is not None:
        doctor.revalidate_initial()
    metadata = dict(schema=REQUIREMENTS_SCHEMA, **before,
        selection="public-instruction" if public_request is not None else "native-only",
        native_prompt_bytes=len(native), public_instruction_receipt=receipt,
        public_instruction_block_sha256=_sha(block), public_instruction_block_bytes=len(block),
        initial_source_current_verified=True, source_current_at_candidate=False,
        **FLAGS)
    if doctor is not None:
        metadata["doctor_requirements"] = doctor.metadata
    return OriginalRequirements(_owner_token=_OWNER_TOKEN, native_prompt=native,
        public_instruction_block=block, metadata_bytes=canonical_json(metadata), source_broker=source_broker,
        public_request=public_request, public_artifact_bytes=artifact_bytes, public_sources_bytes=sources,
        doctor_requirements=doctor)


def capture_public_requirements(*, source_broker, artifact: Path, expected_sha256: str,
                                task_cid: str, doctor_request=None) -> OriginalRequirements:
    """Capture the actual native verified public-instruction block before edits."""
    _need(type(expected_sha256) is str and type(task_cid) is str and bool(task_cid),
        "explicit original public instruction nomination required")
    return _capture(source_broker=source_broker,
        public_request=(str(Path(artifact).absolute()), expected_sha256, task_cid), doctor_request=doctor_request)


def capture_native_only_requirements(*, source_broker, doctor_request=None) -> OriginalRequirements:
    """Explicit owner selection for a launch with no external public instruction."""
    return _capture(source_broker=source_broker, doctor_request=doctor_request)


def _render(snapshot, requirements, max_prompt_bytes, doctor_context=None):
    _need(type(max_prompt_bytes) is int and 1 <= max_prompt_bytes <= MAX_PROMPT_BYTES,
        "bounded candidate prompt byte budget required")
    original = requirements.revalidate(snapshot=snapshot)
    if requirements._doctor_requirements is not None:
        from .doctor_candidate_context import DoctorCandidateContext
        _need(type(doctor_context) is DoctorCandidateContext
            and doctor_context._snapshot is snapshot and doctor_context._requirements is requirements,
            "Doctor candidate prompt requires explicit generation-bound advisory preparation")
        doctor_context.revalidate()
    else:
        _need(doctor_context is None, "Doctor advisory requires captured native constraints")
    context = snapshot.metadata
    summary = json.loads(snapshot.artifacts["summary"])
    formal = json.loads(snapshot.artifacts["formal"])
    selected = {key: summary[key] for key in ("source_files", "program_files", "support_files",
        "capsule_count", "formal_report_count", "formal_frontier_count", "pending_obligation_count", "coverage")}
    selected.update(logic_profile=formal["profile"], unsupported_frontiers=formal["frontiers"],
        complete_artifacts_deferred=True, pending_obligations_discharged=0)
    summary_bytes = canonical_json(selected)
    _need(len(summary_bytes) <= MAX_SUMMARY_BYTES, "complete candidate summary exceeds byte bound")
    candidate_binding = {key: context[key] for key in ("workspace_generation", "candidate_scope_cid",
        "semantic_root_cid", "producer_binding_sha256", "source_hashes", "program_paths")}
    payload = dict(schema=PROMPT_SCHEMA, instructions=_INSTRUCTIONS,
        native_prompt=requirements.native_prompt.decode("utf-8"),
        original_public_instruction_block=requirements.public_instruction_block.decode("utf-8"),
        original_requirements_binding=dict(task_id=original["task_id"],
            native_prompt_sha256=original["native_prompt_sha256"],
            selection=original["selection"], original_requirements_sha256=_sha(requirements.metadata_bytes),
            public_instruction_block_sha256=original["public_instruction_block_sha256"]),
        baseline_binding=dict(context["baseline_binding"]), candidate_binding=candidate_binding,
        candidate_summary=selected, candidate_artifact_references=context["artifacts"],
        prior_failure=dict(prior_session_id=context["prior_session_id"],
            descriptor_sha256=context["descriptor_sha256"], prior_returncode=context["prior_returncode"]),
        **FLAGS)
    if doctor_context is not None:
        payload.update(schema=DOCTOR_PROMPT_SCHEMA, doctor_context=dict(
            schema="supervisor-doctor-candidate-context-binding@1",
            sha256=_sha(doctor_context.metadata_bytes), bytes=len(doctor_context.metadata_bytes)),
            doctor_advisory=doctor_context.advisory_bytes.decode("utf-8"))
    body = canonical_json(payload)
    _need(len(body) <= max_prompt_bytes, "complete candidate prompt exceeds byte bound; requirements cannot be truncated")
    metadata = dict(schema=OBSERVATION_SCHEMA, status="prepared",
        task_id=original["task_id"], prior_session_id=context["prior_session_id"],
        descriptor_sha256=context["descriptor_sha256"], workspace_generation=context["workspace_generation"],
        native_prompt_sha256=original["native_prompt_sha256"], native_prompt_bytes=len(requirements.native_prompt),
        original_requirements_sha256=_sha(requirements.metadata_bytes),
        original_requirements_selection=original["selection"],
        public_instruction_block_sha256=original["public_instruction_block_sha256"],
        public_instruction_block_bytes=len(requirements.public_instruction_block),
        candidate_summary_sha256=_sha(summary_bytes), candidate_summary_bytes=len(summary_bytes),
        candidate_artifact_references=context["artifacts"],
        provider_prompt_sha256=_sha(body), provider_prompt_bytes=len(body),
        required_text_truncated=False, native_prompt_preserved=True, original_public_block_preserved=True,
        raw_prompt_exported=False, **FLAGS)
    if doctor_context is not None:
        metadata.update(schema="supervisor-owner-candidate-prompt-observation@2",
            doctor_context_sha256=_sha(doctor_context.metadata_bytes),
            doctor_advisory_sha256=_sha(doctor_context.advisory_bytes),
            doctor_advisory_bytes=len(doctor_context.advisory_bytes))
    requirements.revalidate(snapshot=snapshot)
    return body, canonical_json(metadata)


@dataclass(frozen=True, slots=True, init=False)
class OwnerCandidateRoundPrompt:
    """Fresh owner-local prompt bytes, never a dispatch or continuation grant."""
    provider_prompt: bytes = field(repr=False)
    metadata_bytes: bytes = field(repr=False)
    _snapshot: object = field(repr=False, compare=False)
    _requirements: OriginalRequirements = field(repr=False, compare=False)
    _max_prompt_bytes: int = field(repr=False)
    _doctor_context: object = field(repr=False, compare=False)

    def __init__(self, *, _owner_token, provider_prompt, metadata_bytes, snapshot, requirements, max_prompt_bytes,
                 doctor_context=None):
        _need(_owner_token is _OWNER_TOKEN, "owner candidate prompt factory required")
        for name, value in (("provider_prompt", provider_prompt), ("metadata_bytes", metadata_bytes),
                ("_snapshot", snapshot), ("_requirements", requirements), ("_max_prompt_bytes", max_prompt_bytes),
                ("_doctor_context", doctor_context)):
            object.__setattr__(self, name, value)

    @property
    def metadata(self):
        return json.loads(self.metadata_bytes)

    @property
    def dispatch_usable(self):
        return False

    def revalidate(self):
        body, metadata = _render(self._snapshot, self._requirements, self._max_prompt_bytes, self._doctor_context)
        _need(body == self.provider_prompt and metadata == self.metadata_bytes,
            "candidate prompt or original requirement binding changed")
        return self.metadata


def prepare_candidate_round_prompt(*, snapshot, requirements=None,
                                   max_prompt_bytes=MAX_PROMPT_BYTES, doctor_context=None) -> OwnerCandidateRoundPrompt:
    """Prepare complete original requirements plus a bounded advisory summary."""
    from .owner_candidate_round_context import OwnerCandidateRoundContext
    _need(type(snapshot) is OwnerCandidateRoundContext, "exact owner candidate snapshot required")
    if requirements is None:
        requirements = getattr(snapshot._source_broker, "original_requirements", None)
    _need(type(requirements) is OriginalRequirements, "typed owner original requirements capture required")
    body, metadata = _render(snapshot, requirements, max_prompt_bytes, doctor_context)
    return OwnerCandidateRoundPrompt(_owner_token=_OWNER_TOKEN, provider_prompt=body,
        metadata_bytes=metadata, snapshot=snapshot, requirements=requirements, max_prompt_bytes=max_prompt_bytes,
        doctor_context=doctor_context)


__all__ = ["OriginalRequirements", "OwnerCandidateRoundPrompt", "capture_public_requirements",
    "capture_native_only_requirements", "prepare_candidate_round_prompt"]
