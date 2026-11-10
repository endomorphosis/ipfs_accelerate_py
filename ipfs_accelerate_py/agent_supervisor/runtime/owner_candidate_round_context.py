"""Owner-only preparation of advisory context after a failed coding child.

This does not admit another child, publish accepted state, or discharge a proof.
The caller must finish isolated-worker cleanup before calling this API. Existing
brokers record child wait and service shutdown, but no descendant-cleanup receipt.
Native generation is synchronous: the deadline is checked around generation,
not enforced by killing the native scanner halfway through its work.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from types import MappingProxyType

from .worker_source_views import canonical_json

SCHEMA = "supervisor-owner-candidate-round-context@1"
_OWNER_TOKEN = object()


def _need(condition, reason):
    if not condition:
        raise ValueError(reason)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _deadline(value):
    _need(type(value) in (int, float) and -sys.float_info.max <= value <= sys.float_info.max
          and math.isfinite(value),
          "finite absolute candidate deadline required")
    _need(time.monotonic() < value, "candidate preparation deadline exhausted")


def _producers():
    from .supervisor_meta_index import public_replay_without_metadata
    from .worker_candidate_artifacts import _producer_binding
    with public_replay_without_metadata():
        return canonical_json(_producer_binding())


def _closed_baseline(source_broker, prior_returncode, *, public_validation=None,
                     validation_launch=None, retired_worker=None):
    """Recheck the retained owner admission without adopting working bytes."""
    from .worker_source_broker import WorkerSourceBroker
    from .semantic_router_translation import encode_semantic_router_prompt, _parse, _read
    from . import source_formalization_context as guard
    from . import worker_source_client as protocol

    _need(type(source_broker) is WorkerSourceBroker, "exact owner source broker required")
    broker = source_broker
    if public_validation is not None:
        from .owner_candidate_dispatch import _continuation_reason, CapturedCodingLaunch, RetiredCodingWorker
        _need(type(validation_launch) is CapturedCodingLaunch and type(retired_worker) is RetiredCodingWorker
            and validation_launch.source_broker is broker and retired_worker.source_broker is broker
            and retired_worker.returncode == prior_returncode,
            "public validation belongs to another closed baseline")
        _continuation_reason(validation_launch, retired_worker, public_validation)
    _need(type(prior_returncode) is int and (1 <= prior_returncode <= 255
          or prior_returncode == 0 and public_validation is not None),
          "positive waited prior returncode in 1..255 required")
    _need(broker._closed and broker._stop.is_set()
          and (broker._thread is None or not broker._thread.is_alive()),
          "closed and stopped source broker required")
    _need(broker._registered.is_set() and type(broker._process) is subprocess.Popen
          and type(broker._process.returncode) is int
          and broker._process.returncode == prior_returncode
          and broker._process.poll() == prior_returncode
          and broker._child is not None and broker._child["pid"] == broker._process.pid,
          "registered failed child must already be waited")
    with broker._accounting_lock:
        _need(broker._inflight is None and broker._accepted == len(broker._records),
              "complete closed source accounting required")
    _need(not broker.socket_path.exists() and not broker.descriptor_path.exists(),
          "closed source transport still exists")
    protocol.validate_descriptor(broker.descriptor)
    owner = protocol.process_identity(os.getpid())
    _need(broker.descriptor["owner_pid"] == owner["pid"]
          and broker.descriptor["owner_start_time_ticks"] == owner["start_time_ticks"]
          and broker._child["parent_pid"] == owner["pid"],
          "candidate preparation belongs to another owner")
    _need(_sha(canonical_json(broker.descriptor)) == broker.descriptor_sha256,
          "closed source descriptor changed")
    boundary, boundary_sha = guard._owner_boundary(broker.boundary)
    _need(boundary_sha == broker.descriptor["boundary_manifest_sha256"]
          and boundary["owner_uid"] == os.getuid() == broker.owner_uid
          and boundary["worker_uid"] == broker.worker_uid,
          "closed source boundary changed")
    from .supervisor_meta_index import public_replay_without_metadata
    with public_replay_without_metadata():
        encoded = encode_semantic_router_prompt(prompt=broker.prompt, repository=broker.repository)
    table = encoded.table.to_dict()
    _need(canonical_json(table) == broker._table_bytes,
          "retained native semantic translation changed")
    binding = protocol.semantic_binding(encoded, broker._binding["profile"])
    _need(binding == broker._binding
          and all(broker.descriptor.get(key) == value for key, value in binding.items()),
          "retained native semantic binding changed")
    admission = broker.views.baseline_admission()
    _need(canonical_json(admission) == broker._view_admission,
          "retained baseline source admission changed")
    semantic = _parse(_read(broker.repository, table["semantic_artifact"]).decode("utf-8"))
    paths = tuple(sorted(semantic.get("program_paths", table["source_manifest"])))
    _need(paths == broker.candidate_context.program_paths,
          "retained candidate program population changed")
    return dict(baseline_binding=binding, baseline_scope_sha256=admission["scope_sha256"],
                prior_session_id=broker.descriptor["session_id"],
                prior_descriptor_sha256=broker.descriptor_sha256,
                prior_child=dict(broker._child), program_paths=list(paths))


@dataclass(frozen=True, slots=True, init=False)
class OwnerCandidateRoundContext:
    """An owner-local snapshot; bytes are immutable and metadata returns a copy.

    Keep this handle inside the owner. Serialized metadata has no admission,
    execution or proof authority. Call ``revalidate`` immediately before using
    its artifacts; successful revalidation brackets an observation, not a lock.
    """
    metadata_bytes: bytes
    native_prompt: bytes = field(repr=False)
    _artifact_items: tuple = field(repr=False)
    _source_broker: object = field(repr=False, compare=False)
    _baseline_bytes: bytes = field(repr=False)
    _absolute_deadline: float = field(repr=False)
    _public_validation: object = field(repr=False, compare=False)
    _validation_launch: object = field(repr=False, compare=False)
    _retired_worker: object = field(repr=False, compare=False)

    def __init__(self, *, _owner_token, metadata_bytes, native_prompt, artifact_items,
                 source_broker, baseline_bytes, absolute_deadline, public_validation=None,
                 validation_launch=None, retired_worker=None):
        _need(_owner_token is _OWNER_TOKEN, "owner preparation factory required")
        for name, value in (("metadata_bytes", metadata_bytes), ("native_prompt", native_prompt),
                ("_artifact_items", artifact_items), ("_source_broker", source_broker),
                ("_baseline_bytes", baseline_bytes), ("_absolute_deadline", absolute_deadline),
                ("_public_validation", public_validation), ("_validation_launch", validation_launch),
                ("_retired_worker", retired_worker)):
            object.__setattr__(self, name, value)

    @property
    def metadata(self):
        return json.loads(self.metadata_bytes)

    @property
    def artifacts(self):
        return MappingProxyType(dict(self._artifact_items))

    def revalidate(self):
        """Refuse stale baseline, candidate, producer or expired budget."""
        _deadline(self._absolute_deadline)
        metadata = self.metadata
        options = {} if self._public_validation is None else dict(public_validation=self._public_validation,
            validation_launch=self._validation_launch, retired_worker=self._retired_worker)
        state = _closed_baseline(self._source_broker, metadata["prior_returncode"], **options)
        _need(canonical_json(state) == self._baseline_bytes
              and self._source_broker.prompt.encode("utf-8") == self.native_prompt,
              "candidate round baseline changed")
        sources = self._source_broker.views.working_sources(
            expected_workspace_generation=metadata["workspace_generation"])
        _need({name: _sha(raw) for name, raw in sources.items()} == metadata["source_hashes"],
              "candidate round source hashes changed")
        _need(_sha(_producers()) == metadata["producer_binding_sha256"],
              "candidate round producer sources changed")
        _need(all(metadata["artifacts"][name] == dict(sha256=_sha(raw), bytes=len(raw))
                  for name, raw in self._artifact_items), "candidate round artifact pins changed")
        _need(canonical_json(_closed_baseline(self._source_broker, metadata["prior_returncode"], **options))
              == self._baseline_bytes, "candidate round baseline changed during validation")
        _need(self._source_broker.views.working_sources(
            expected_workspace_generation=metadata["workspace_generation"]) == sources,
            "candidate round source changed during validation")
        _need(_sha(_producers()) == metadata["producer_binding_sha256"],
              "candidate round producer sources changed during validation")
        _deadline(self._absolute_deadline)
        if self._public_validation is not None:
            observed = self._public_validation.revalidate(launch=self._validation_launch,
                retired_worker=self._retired_worker, snapshot=self)
            _need(metadata["public_validation_observation_sha256"] == _sha(canonical_json(observed)),
                "candidate public validation observation changed")
        return metadata


def prepare_candidate_round_context(*, source_broker, prior_returncode, round_number,
                                    absolute_deadline, public_validation=None,
                                    validation_launch=None, retired_worker=None):
    """Build fresh advisory artifacts after an already waited failed child.

    Scope is exactly the first round's admitted source population. Missing files
    refuse; newly added files remain outside coverage. No durable state is
    written here, no provider starts, and no next-round dispatch is admitted.
    """
    _need(type(round_number) is int and round_number > 0, "positive exact round number required")
    _deadline(absolute_deadline)
    options = {} if public_validation is None else dict(public_validation=public_validation,
        validation_launch=validation_launch, retired_worker=retired_worker)
    _need(public_validation is not None or validation_launch is None and retired_worker is None,
        "public validation handles must be selected together")
    baseline = _closed_baseline(source_broker, prior_returncode, **options)
    baseline_bytes = canonical_json(baseline)
    prompt = source_broker.prompt.encode("utf-8")
    paths = tuple(baseline["program_paths"])
    # The first path selects a page; working capture binds the entire admitted
    # population, including support files. No paging cursor is retained here.
    table = json.loads(source_broker._table_bytes)
    first = sorted(table["source_manifest"])[0]
    page = source_broker.views.working_page(path=first, view="original", max_bytes=8192)
    generation = page["workspace_generation"]
    sources = source_broker.views.working_sources(expected_workspace_generation=generation)
    producers = _producers()
    _deadline(absolute_deadline)
    from .worker_candidate_artifacts import build_candidate_artifacts
    artifacts = build_candidate_artifacts(sources=sources, program_paths=paths,
                                          workspace_generation=generation)
    summary = json.loads(artifacts["summary"])
    _need(canonical_json(summary["producer_sources"]) == producers,
          "candidate native producer binding differs")
    _need(summary["source_hashes"] == {name: _sha(raw) for name, raw in sources.items()}
          and summary["workspace_generation"] == generation,
          "candidate native source binding differs")
    metadata = dict(schema=SCHEMA, round_number=round_number, prior_returncode=prior_returncode,
        **baseline, workspace_generation=generation, source_state="working-candidate",
        descriptor_sha256=baseline["prior_descriptor_sha256"],
        task_id=baseline["baseline_binding"]["task_id"],
        native_prompt_sha256=baseline["baseline_binding"]["native_prompt_sha256"],
        source_hashes=summary["source_hashes"], source_files=len(sources),
        source_bytes=sum(map(len, sources.values())),
        producer_binding_sha256=_sha(producers), candidate_scope_cid=summary["candidate_scope_cid"],
        semantic_root_cid=summary["semantic_root_cid"],
        artifacts={name: dict(sha256=_sha(raw), bytes=len(raw)) for name, raw in artifacts.items()},
        coverage=summary["coverage"], pending_obligation_count=summary["pending_obligation_count"],
        baseline_evidence_applicable=False, proof_authority=False, execution_authority=False,
        completion_authority=False, next_round_admitted=False, accepted_index_published=False,
        formalization_authority=False, semantic_acceptance_authority=False,
        source_semantics_verified=False, whole_program_semantics_verified=False,
        source_obligations_discharged=0, provider_calls=0, prover_calls=0,
        model_inference_calls=0, training_steps=0, source_scope_expanded=False,
        child_wait_observed=True, source_service_stopped=True,
        descendant_cleanup_verified=False, hard_cancellation_deadline=False)
    if public_validation is not None:
        observed = public_validation.revalidate(launch=validation_launch, retired_worker=retired_worker)
        _need(observed["workspace_generation"] == generation and observed["source_hashes"] == metadata["source_hashes"],
            "candidate generation differs from public validation")
        metadata.update(schema="supervisor-owner-candidate-round-context@2",
            public_validation_observation_sha256=_sha(canonical_json(observed)))
    result = OwnerCandidateRoundContext(_owner_token=_OWNER_TOKEN,
        metadata_bytes=canonical_json(metadata), native_prompt=prompt,
        artifact_items=tuple(sorted(artifacts.items())), source_broker=source_broker,
        baseline_bytes=baseline_bytes, absolute_deadline=absolute_deadline, **options)
    result.revalidate()
    return result


__all__ = ["OwnerCandidateRoundContext", "prepare_candidate_round_context"]
