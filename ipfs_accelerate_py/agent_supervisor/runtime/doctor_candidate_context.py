"""Retain Doctor constraints and bind advisory context to a changed candidate.

These owner-local preparations authorize no model call. Original evidence stays
historical; fresh candidate hashes establish only which source the unresolved
constraints accompany. Neither handle has a serialized reload API.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import stat

from .worker_source_views import canonical_json

CAPTURE_SCHEMA = "supervisor-owner-doctor-requirements@1"
SCHEMA = "supervisor-doctor-candidate-context@1"
_TOKEN = object()
FLAGS = dict(dispatch_usable=False, additional_call_admitted=False,
    residual_budget_enforced=False, proof_authority=False, completion_authority=False,
    scope_expansion_authority=False, baseline_evidence_applicable=False,
    source_obligations_discharged=0, provider_calls=0)


def _need(value, reason):
    if not value:
        raise ValueError(reason)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _artifact(path, digest):
    from . import doctor_residual_context as residual
    descriptor = residual._directory(path.parent)
    try:
        raw, info = residual._read(descriptor, path.name)
    finally:
        os.close(descriptor)
    _need(len(raw) <= residual.MAX_BYTES and _sha(raw) == digest
        and not stat.S_IMODE(info.st_mode) & 0o222,
        "captured Doctor residual artifact changed")
    return raw


@dataclass(frozen=True, slots=True, init=False)
class OriginalDoctorRequirements:
    """Exact native residual captured with fresh sources before the first child."""
    artifact_bytes: bytes = field(repr=False)
    advisory_bytes: bytes = field(repr=False)
    metadata_bytes: bytes = field(repr=False)
    _source_broker: object = field(repr=False, compare=False)
    _request: tuple = field(repr=False)
    _receipt_bytes: bytes = field(repr=False)

    def __init__(self, *, _token, artifact_bytes, advisory_bytes, metadata_bytes,
                 source_broker, request, receipt_bytes):
        _need(_token is _TOKEN, "native Doctor requirement capture factory required")
        for key, value in (("artifact_bytes", artifact_bytes), ("advisory_bytes", advisory_bytes),
                ("metadata_bytes", metadata_bytes), ("_source_broker", source_broker),
                ("_request", request), ("_receipt_bytes", receipt_bytes)):
            object.__setattr__(self, key, value)

    @property
    def metadata(self):
        return json.loads(self.metadata_bytes)

    def revalidate_initial(self):
        """Check complete initial Doctor sources while the child is still absent."""
        from .owner_candidate_round_prompt import _initial_binding
        from . import doctor_residual_context as residual
        from . import local_planning_admission as local
        broker = self._source_broker
        before = _initial_binding(broker)
        path, digest, task_cid = self._request
        text, receipt = residual.load_doctor_residual_advisory(artifact=Path(path),
            expected_sha256=digest, repository=broker.repository, task_cid=task_cid,
            prompt=broker.prompt, workspace=broker.workspace)
        baseline = json.loads(self.artifact_bytes)["source_manifest"]
        _need(text.encode() == self.advisory_bytes and canonical_json(receipt) == self._receipt_bytes
            and broker._doctor_request == self._request
            and _artifact(Path(path), digest) == self.artifact_bytes
            and _initial_binding(broker) == before
            and local._sources(broker.repository, sorted(baseline), max_files=256) == baseline
            and local._sources(broker.workspace, sorted(baseline), max_files=256) == baseline,
            "complete Doctor sources changed during initial capture")
        return self.metadata

    def revalidate(self, *, snapshot):
        """Check original canonical sources and the complete current Doctor inventory."""
        from .owner_candidate_round_context import OwnerCandidateRoundContext
        from . import doctor_residual_context as residual
        from . import local_planning_admission as local
        _need(type(snapshot) is OwnerCandidateRoundContext
            and snapshot._source_broker is self._source_broker,
            "Doctor requirements belong to another candidate snapshot")
        candidate = snapshot.revalidate()
        broker, metadata = self._source_broker, self.metadata
        path, digest, task_cid = self._request
        path = Path(path)
        _need(broker._doctor_request == self._request
            and _artifact(path, digest) == self.artifact_bytes,
            "captured Doctor residual artifact changed")
        # This is a fresh canonical-source check. Historical loader mode never
        # establishes candidate readiness, and the edited workspace is checked
        # independently against its generation below.
        text, receipt = residual.load_doctor_residual_advisory(artifact=path,
            expected_sha256=digest, repository=broker.repository, task_cid=task_cid,
            prompt=broker.prompt, require_current_source=True)
        _need(text.encode() == self.advisory_bytes and canonical_json(receipt) == self._receipt_bytes,
            "captured Doctor constraints or native receipt changed")
        payload = json.loads(self.artifact_bytes)
        _need(metadata["native_prompt_sha256"] == candidate["native_prompt_sha256"]
            and metadata["task_id"] == candidate["task_id"]
            and metadata["prior_session_id"] == candidate["prior_session_id"]
            and metadata["descriptor_sha256"] == candidate["descriptor_sha256"]
            and metadata["residual_budget"] == residual.validate_residual_budget(payload["capsule"]),
            "captured Doctor task or budget binding changed")
        baseline = payload["source_manifest"]
        current = local._sources(broker.workspace, sorted(baseline), max_files=256)
        for name, original in baseline.items():
            expected = dict(original)
            if name in candidate["source_hashes"]:
                expected["sha256"] = candidate["source_hashes"][name]
            _need(current[name] == expected,
                "Doctor source outside candidate inventory changed or candidate source drifted")
        _need(_artifact(path, digest) == self.artifact_bytes
            and snapshot.revalidate() == candidate,
            "Doctor candidate inputs changed during revalidation")
        _need(local._sources(broker.repository, sorted(baseline), max_files=256) == baseline
            and local._sources(broker.workspace, sorted(baseline), max_files=256) == current,
            "complete Doctor source inventory changed during revalidation")
        return metadata


def capture_doctor_requirements(*, source_broker, artifact, expected_sha256, task_cid):
    """Use the native current-source residual loader before any child registration."""
    from .owner_candidate_round_prompt import _initial_binding
    from . import doctor_residual_context as residual
    before = _initial_binding(source_broker)
    _need(type(expected_sha256) is str and type(task_cid) is str and bool(task_cid),
        "exact original Doctor nomination required")
    path = Path(artifact).absolute()
    nomination = (str(path), expected_sha256, task_cid)
    selected = source_broker._doctor_request
    _need(selected is None or selected == nomination,
        "original Doctor selection cannot be substituted")
    text, receipt = residual.load_doctor_residual_advisory(artifact=path,
        expected_sha256=expected_sha256, repository=source_broker.repository,
        task_cid=task_cid, prompt=source_broker.prompt, workspace=source_broker.workspace)
    raw = _artifact(path, expected_sha256)
    payload = json.loads(raw, object_pairs_hook=residual._unique)
    budget = residual.validate_residual_budget(payload["capsule"])
    _need(payload["base_commit"] == before["baseline_commit"]
        and payload["task_id"] == before["task_id"]
        and _initial_binding(source_broker) == before,
        "Doctor residual differs from original source-broker admission")
    final_text, final_receipt = residual.load_doctor_residual_advisory(artifact=path,
        expected_sha256=expected_sha256, repository=source_broker.repository,
        task_cid=task_cid, prompt=source_broker.prompt, workspace=source_broker.workspace)
    _need(final_text == text and final_receipt == receipt
        and _artifact(path, expected_sha256) == raw
        and _initial_binding(source_broker) == before and source_broker._doctor_request == selected,
        "Doctor requirements changed during original capture")
    advisory = text.encode()
    metadata = dict(schema=CAPTURE_SCHEMA, **before, task_cid=task_cid,
        artifact_sha256=expected_sha256, artifact_bytes=len(raw),
        context_cid=payload["context_cid"], native_capsule_id=payload["capsule"]["capsule_id"],
        advisory_sha256=_sha(advisory), advisory_bytes=len(advisory),
        source_manifest_sha256=_sha(canonical_json(payload["source_manifest"])),
        residual_budget=budget, initial_source_current_verified=True, **FLAGS)
    result = OriginalDoctorRequirements(_token=_TOKEN, artifact_bytes=raw, advisory_bytes=advisory,
        metadata_bytes=canonical_json(metadata), source_broker=source_broker,
        request=nomination, receipt_bytes=canonical_json(receipt))
    source_broker._doctor_request = nomination
    result.revalidate_initial()
    return result


def _projection(snapshot, requirements):
    from .owner_candidate_round_prompt import OriginalRequirements
    _need(type(requirements) is OriginalRequirements
        and type(requirements._doctor_requirements) is OriginalDoctorRequirements,
        "typed original requirements with captured Doctor constraints required")
    requirements.revalidate(snapshot=snapshot)
    doctor = requirements._doctor_requirements
    candidate, original = snapshot.metadata, doctor.metadata
    payload = json.loads(doctor.artifact_bytes)
    source_names = set(payload["source_manifest"])
    selected = source_names.intersection(candidate["source_hashes"])
    binding = {key: candidate[key] for key in ("task_id", "native_prompt_sha256",
        "prior_session_id", "descriptor_sha256", "workspace_generation", "candidate_scope_cid",
        "semantic_root_cid", "producer_binding_sha256", "source_hashes")}
    value = dict(schema=SCHEMA, original_doctor=original, candidate_binding=binding,
        original_requirements_sha256=_sha(requirements.metadata_bytes),
        original_advisory_sha256=_sha(doctor.advisory_bytes),
        original_advisory_bytes=len(doctor.advisory_bytes),
        candidate_inventory_paths=sorted(selected), unchanged_inventory_paths=sorted(source_names-selected),
        full_doctor_source_inventory_current=True, candidate_evidence_reproved=False,
        original_obligations_preserved=True, residual_budget=original["residual_budget"],
        continuation_status="requires_independent_budget_enforcement", **FLAGS)
    # Only the exact original advisory is retained as provider-facing text;
    # source manifests and the complete native artifact remain owner metadata.
    return canonical_json(value), doctor.advisory_bytes


@dataclass(frozen=True, slots=True, init=False)
class DoctorCandidateContext:
    metadata_bytes: bytes = field(repr=False)
    advisory_bytes: bytes = field(repr=False)
    _snapshot: object = field(repr=False, compare=False)
    _requirements: object = field(repr=False, compare=False)

    def __init__(self, *, _token, metadata_bytes, advisory_bytes, snapshot, requirements):
        _need(_token is _TOKEN, "owner Doctor candidate preparation factory required")
        for key, value in (("metadata_bytes", metadata_bytes), ("advisory_bytes", advisory_bytes),
                ("_snapshot", snapshot), ("_requirements", requirements)):
            object.__setattr__(self, key, value)

    @property
    def metadata(self):
        return json.loads(self.metadata_bytes)

    def revalidate(self):
        metadata, advisory = _projection(self._snapshot, self._requirements)
        _need(metadata == self.metadata_bytes and advisory == self.advisory_bytes,
            "Doctor candidate projection changed")
        return self.metadata


def prepare_doctor_candidate_context(*, snapshot, requirements):
    """Prepare a generation-bound advisory without admitting another coding call."""
    metadata, advisory = _projection(snapshot, requirements)
    return DoctorCandidateContext(_token=_TOKEN, metadata_bytes=metadata, advisory_bytes=advisory,
        snapshot=snapshot, requirements=requirements)


__all__ = ["OriginalDoctorRequirements", "DoctorCandidateContext",
    "capture_doctor_requirements", "prepare_doctor_candidate_context"]
