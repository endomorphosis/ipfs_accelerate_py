"""Signed public checks on retired candidates, without serialized retry grants.

Only the owner factories produce live handles. JSON is a bounded observation;
canonical completion must still rerun the signed checks after publication.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import threading
import time
from weakref import WeakKeyDictionary

from .worker_source_views import canonical_json
from . import terminal_task_profile as profiles

CONTRACT_SCHEMA = "supervisor-public-validation-contract@1"
OBSERVATION_SCHEMA = "supervisor-retired-candidate-validation@1"
FLAGS = dict(observation_only=True, completion_authority=False, proof_authority=False,
             execution_authority=False, continuation_authority=False)
_TOKEN = object()
_LOCK = threading.RLock()
_CONTRACTS = WeakKeyDictionary()
_OBSERVATIONS = WeakKeyDictionary()


def _need(value, reason):
    if not value:
        raise ValueError(reason)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _digest(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _copy(value):
    return json.loads(canonical_json(value))


def _file(root, name):
    from .router_public_instruction import _source
    raw, info = _source(root, name)
    _need(info.st_nlink == 1 and len(raw) <= 1_000_000, "bounded single-link validation input required")
    return raw, dict(sha256=_sha(raw), bytes=len(raw), mode=stat.S_IMODE(info.st_mode),
                     uid=info.st_uid, gid=info.st_gid)


def _states(root, names):
    values = {name: _file(root, name)[1] for name in sorted(names)}
    _need(sum(row["bytes"] for row in values.values()) <= 4_000_000, "validation source population exceeds bound")
    return values


def _validate_states(value, names):
    _need(type(value) is dict and set(value) == set(names), "exact validation source states required")
    for row in value.values():
        _need(type(row) is dict and set(row) == {"sha256", "bytes", "mode", "uid", "gid"}
              and _digest(row["sha256"]) and type(row["bytes"]) is int and 0 <= row["bytes"] <= 1_000_000
              and type(row["mode"]) is int and 0 <= row["mode"] <= 0o7777
              and all(type(row[k]) is int and 0 <= row[k] <= 2**32-1 for k in ("uid", "gid")),
              "closed validation source state required")


def _artifact(request):
    from .owner_candidate_round_prompt import _public_artifact
    from .router_public_instruction import _selection, _unique
    raw = _public_artifact(request["artifact"], request["sha256"])
    payload = json.loads(raw, object_pairs_hook=_unique)
    manifest = _selection(payload)
    _need(payload["task_cid"] == request["task_cid"], "public validation task selection differs")
    return raw, payload, manifest


def _spec(profile, task):
    from ..proof.formal_verification_contracts import content_identity
    from .local_planning_admission import LOCAL_POLICY
    expected = profiles.task_profile_spec(profile, policy_cid=content_identity(LOCAL_POLICY))
    actual = _copy(task)
    _need(type(actual.get("acceptance")) is list, "signed public acceptance required")
    for criterion in actual["acceptance"]:
        _need(type(criterion) is dict and type(criterion.get("evidence_cids")) is list,
              "closed signed acceptance required")
        criterion["evidence_cids"] = []
    _need(canonical_json(actual) == canonical_json(expected), "signed public validation commands or acceptance differ")
    return expected


def validate_public_validation_contract_metadata(value):
    """Check a closed historical shape; no owner execution is admitted here."""
    keys = {"schema", "profile", "task_id", "task_cid", "manifest_cid", "repository", "baseline_commit",
            "public_instruction_request", "original_requirements_sha256", "deadline_monotonic_ns",
            "boundary_sha256", "runner", "task_spec", "sources", "immutable_workspace_states", *FLAGS}
    _need(type(value) is dict and set(value) == keys and value["schema"] == CONTRACT_SCHEMA
          and all(value[k] is v for k, v in FLAGS.items()), "closed public validation contract required")
    profile = profiles.validate_task_profile(value["profile"])
    _need(profile["schema"] == profiles.PUBLIC_ACCEPTANCE_SCHEMA
          and canonical_json(value["profile"]) == canonical_json(profile), "explicit canonical public acceptance profile required")
    expected = _spec(profile, value["task_spec"])
    request = value["public_instruction_request"]
    _need(type(request) is dict and set(request) == {"artifact", "sha256", "task_cid"}
          and _digest(request["sha256"]) and type(request["artifact"]) is str
          and 0 < len(request["artifact"]) <= 4096 and Path(request["artifact"]).is_absolute()
          and request["task_cid"] == value["task_cid"], "exact public instruction request required")
    for key in ("task_id", "task_cid", "manifest_cid", "repository", "baseline_commit"):
        _need(type(value[key]) is str and 0 < len(value[key]) <= 4096, "bounded validation identity required")
    _need(value["task_id"] == expected["task_key"] and Path(value["repository"]).is_absolute()
          and _digest(value["original_requirements_sha256"]) and _digest(value["boundary_sha256"])
          and type(value["deadline_monotonic_ns"]) is int and value["deadline_monotonic_ns"] > 0,
          "public validation binding differs")
    sources = value["sources"]
    _need(type(sources) is dict and set(sources) == set(profiles.task_profile_worker_inputs(profile)),
          "exact signed public validation input population required")
    for row in sources.values():
        _need(type(row) is dict and set(row) == {"sha256", "executable"}
              and _digest(row["sha256"]) and type(row["executable"]) is bool,
              "closed signed public source binding required")
    immutable = set(sources) - {row["path"] for row in profile["outputs"]}
    _validate_states(value["immutable_workspace_states"], immutable)
    for name, row in value["immutable_workspace_states"].items():
        _need(row["sha256"] == sources[name]["sha256"]
              and bool(row["mode"] & 0o111) == sources[name]["executable"], "immutable validation source differs")
    from .candidate_execution import SCHEMA, ROOT
    runner = value["runner"]
    files = {str(ROOT / p) for p in ("bin/validation-worker", "bin/worker-entry", "container-boundary.json")}
    _need(type(runner) is dict and set(runner) == {"schema", "argv", "files", "owner_uid", "worker_uid", "namespaces", "completion_authority"}
          and runner["schema"] == SCHEMA and runner["argv"] == [str(ROOT / "bin/validation-worker")]
          and type(runner["files"]) is dict and set(runner["files"]) == files
          and all(_digest(h) for h in runner["files"].values())
          and type(runner["owner_uid"]) is int and runner["owner_uid"] >= 0 and runner["owner_uid"] != 1001
          and type(runner["worker_uid"]) is int and runner["worker_uid"] == 1001
          and runner["completion_authority"] is False and type(runner["namespaces"]) is dict
          and set(runner["namespaces"]) == {"pid", "mnt", "net"}
          and all(type(v) is str and len(v) <= 128 for v in runner["namespaces"].values())
          and runner["files"][str(ROOT / "container-boundary.json")] == value["boundary_sha256"],
          "closed isolated public validation runner required")
    return _copy(value)


def _replay_contract(value, repository, workspace):
    """Replay signature and immutable input joins without invoking a program."""
    from . import router_public_instruction as public
    value = validate_public_validation_contract_metadata(value)
    _, payload, manifest = _artifact(value["public_instruction_request"])
    _need(str(repository) == value["repository"] == manifest["repository"]
          and manifest["baseline_commit"] == value["baseline_commit"]
          and payload["manifest_cid"] == value["manifest_cid"] and payload["task_id"] == value["task_id"]
          and manifest["sources"] == value["sources"], "signed validation manifest changed")
    tasks = [task for task in manifest["tasks"] if task["task_key"] == value["task_id"]]
    _need(len(tasks) == 1 and _spec(value["profile"], tasks[0]) == value["task_spec"], "signed validation task changed")
    public._current_sources(repository, value["sources"])
    _need(_file(repository, profiles.PROFILE)[0] == profiles.task_profile_bytes(value["profile"])
          and _file(repository, profiles.SMOKE)[0] == profiles.task_profile_smoke(value["profile"]).encode(),
          "signed public validation support differs from reviewed profile")
    profiles.validate_task_profile_public_inputs(value["profile"], public_evaluator=_file(repository, "eval.py")[0])
    profiles.validate_task_profile(value["profile"], instruction=_file(repository, profiles.INSTRUCTION)[0].decode())
    _need(_states(workspace, value["immutable_workspace_states"]) == value["immutable_workspace_states"],
          "immutable public validation workspace inputs changed")
    return value


@dataclass(frozen=True, eq=False)
class CapturedPublicValidationContract:
    _token: object = field(repr=False)
    _source_broker: object = field(repr=False)
    _requirements: object = field(repr=False)
    metadata_bytes: bytes

    def __post_init__(self):
        _need(self._token is _TOKEN, "owner public validation capture required")

    @property
    def metadata(self):
        return json.loads(self.metadata_bytes)

    def revalidate(self):
        from .candidate_execution import verify_candidate_runner
        from .owner_candidate_round_prompt import _head
        with _LOCK:
            _need(_CONTRACTS.get(self._source_broker) is self, "registered public validation capture required")
        broker, value = self._source_broker, self.metadata
        _need(broker.original_requirements is self._requirements
              and _sha(self._requirements.metadata_bytes) == value["original_requirements_sha256"]
              and _head(broker.repository) == value["baseline_commit"]
              and time.monotonic_ns() < value["deadline_monotonic_ns"], "original validation custody or deadline changed")
        verify_candidate_runner(value["runner"])
        _replay_contract(value, broker.repository, broker.workspace)
        _need(time.monotonic_ns() < value["deadline_monotonic_ns"], "original validation deadline exhausted during replay")
        return value


def capture_public_validation_contract(*, source_broker, requirements, deadline_monotonic_ns):
    from .owner_candidate_round_prompt import OriginalRequirements
    from .worker_source_broker import WorkerSourceBroker
    from .candidate_execution import candidate_runner_from_environment
    _need(type(source_broker) is WorkerSourceBroker and type(requirements) is OriginalRequirements
          and requirements._source_broker is source_broker and source_broker.original_requirements is requirements
          and not source_broker._registered.is_set() and source_broker._process is None
          and requirements._public_request is not None, "original signed public requirements before coding required")
    artifact, digest, task_cid = requirements._public_request
    request = dict(artifact=str(artifact), sha256=digest, task_cid=task_cid)
    raw, payload, manifest = _artifact(request)
    _need(raw == requirements._public_artifact_bytes, "original public validation artifact differs")
    profile = profiles.validate_task_profile(json.loads(_file(source_broker.repository, profiles.PROFILE)[0]))
    task, = [task for task in manifest["tasks"] if task["task_key"] == payload["task_id"]]
    outputs = {row["path"] for row in profile["outputs"]}
    value = dict(schema=CONTRACT_SCHEMA, profile=profile, task_id=payload["task_id"], task_cid=task_cid,
        manifest_cid=payload["manifest_cid"], repository=str(source_broker.repository), baseline_commit=manifest["baseline_commit"],
        public_instruction_request=request, original_requirements_sha256=_sha(requirements.metadata_bytes),
        deadline_monotonic_ns=deadline_monotonic_ns, boundary_sha256=source_broker.descriptor["boundary_manifest_sha256"],
        runner=candidate_runner_from_environment(), task_spec=_spec(profile, task), sources=manifest["sources"],
        immutable_workspace_states=_states(source_broker.workspace, set(manifest["sources"])-outputs), **FLAGS)
    validate_public_validation_contract_metadata(value)
    result = CapturedPublicValidationContract(_TOKEN, source_broker, requirements, canonical_json(value))
    with _LOCK:
        _need(source_broker not in _CONTRACTS, "public validation contract already captured")
        _CONTRACTS[source_broker] = result
    result.revalidate()
    return result


def validate_public_validation_observation(value):
    keys = {"schema", "contract_sha256", "original_deadline_ns", "actual_returncode", "round_number", "session_id", "descriptor_sha256",
            "workspace_generation", "source_hashes", "workspace_states", "status", "failed_obligations", "checks", *FLAGS}
    _need(type(value) is dict and set(value) == keys and value["schema"] == OBSERVATION_SCHEMA
          and all(value[k] is v for k, v in FLAGS.items()) and _digest(value["contract_sha256"])
          and _digest(value["workspace_generation"]) and type(value["original_deadline_ns"]) is int
          and value["original_deadline_ns"] > 0 and type(value["actual_returncode"]) is int
          and -255 <= value["actual_returncode"] <= 255 and type(value["round_number"]) is int
          and value["round_number"] in (1, 2) and _digest(value["descriptor_sha256"])
          and type(value["session_id"]) is str and re.fullmatch(r"[0-9a-f]{32}", value["session_id"]) is not None,
          "closed retired validation observation required")
    names = {profiles.INSTRUCTION, profiles.PROFILE, profiles.SMOKE, "eval.py", "eigen.py"}
    _validate_states(value["workspace_states"], names)
    _need(type(value["source_hashes"]) is dict and 1 <= len(value["source_hashes"]) <= len(names)
          and set(value["source_hashes"]) <= names
          and all(_digest(h) and value["workspace_states"][p]["sha256"] == h for p, h in value["source_hashes"].items()),
          "exact observed candidate source hashes required")
    checks = value["checks"]
    _need(type(checks) is list and len(checks) == 3, "complete public validation observation required")
    statuses, failed = [], []
    for row, check, key in zip(checks, profiles.PUBLIC_CHECKS, profiles.PUBLIC_VALIDATION_KEYS):
        _need(type(row) is dict and set(row) == {"validation_key", "returncode", "result", "status", "reason"}
              and row["validation_key"] == key, "closed public check observation required")
        if row["result"] is not None:
            result = profiles.public_validation_result(row["result"], check=check)
            _need(row["status"] == result["status"] and row["reason"] == result["reason"]
                  and type(row["returncode"]) is int
                  and row["returncode"] == {"passed": 0, "failed": 1, "inconclusive": 2}[result["status"]],
                  "public check process and result disagree")
        else:
            _need(row["status"] == "inconclusive" and row["reason"] in {"invalid_result", "timeout", "output_limit", "not_run"}
                  and (row["returncode"] is None or type(row["returncode"]) is int and -255 <= row["returncode"] <= 255),
                  "unavailable public validation result must remain inconclusive")
        statuses.append(row["status"])
        if row["status"] == "failed":
            failed.append(key)
    status = "failed" if failed else "passed" if set(statuses) == {"passed"} else "inconclusive"
    _need(value["status"] == status and value["failed_obligations"] == failed,
          "public validation aggregate differs from checks")
    return _copy(value)


def public_validation_feedback(observation):
    value = validate_public_validation_observation(observation)
    checks = []
    for row in value["checks"]:
        measured = [] if row["result"] is None else row["result"]["measurements"]
        summary = dict(validation_key=row["validation_key"], status=row["status"], reason=row["reason"], measurements=[])
        if measured:
            # Retain all 15 measurements in owner evidence. The coding model
            # needs one scale estimate per size; this summary grants no proof.
            summary["measurement_aggregation"] = "median_of_three_batch_medians"
            for size in (2, 4, 6, 8, 10):
                group = [item for item in measured if item["size"] == size]
                summary["measurements"].append(dict(size=size, **{
                    key: sorted(item[key] for item in group)[1]
                    for key in ("candidate_median_seconds", "reference_median_seconds")}))
        checks.append(summary)
    return dict(status=value["status"], failed_obligations=value["failed_obligations"], checks=checks)


def _generation(broker, expected=None):
    if expected is None:
        first = sorted(json.loads(broker._table_bytes)["source_manifest"])[0]
        expected = broker.views.working_page(path=first, view="original", max_bytes=8192)["workspace_generation"]
    return expected, {p: _sha(raw) for p, raw in broker.views.working_sources(expected_workspace_generation=expected).items()}


@dataclass(frozen=True, eq=False)
class RetiredCandidateValidation:
    _token: object = field(repr=False)
    _launch: object = field(repr=False)
    _retired: object = field(repr=False)
    metadata_bytes: bytes

    def __post_init__(self):
        _need(self._token is _TOKEN, "live retired validation factory required")

    @property
    def metadata(self):
        return json.loads(self.metadata_bytes)

    def retained_metadata(self, *, launch, retired_worker):
        """Check historical live custody without claiming current generation validity."""
        from .owner_candidate_dispatch import CapturedCodingLaunch, RetiredCodingWorker, _LAUNCHES
        _need(type(launch) is CapturedCodingLaunch and type(retired_worker) is RetiredCodingWorker
              and self._launch is launch and self._retired is retired_worker
              and _LAUNCHES.get(launch.source_broker) is launch,
              "exact historical validation custody required")
        with _LOCK:
            _need(_OBSERVATIONS.get(launch.source_broker, {}).get(retired_worker.round_number) is self,
                  "registered live validation observation required")
        value = validate_public_validation_observation(self.metadata)
        registered = launch._state.initial_retirement if retired_worker.round_number == 1 else launch._state.second_retirement
        _need(registered is retired_worker and retired_worker._launch is launch
              and retired_worker.process.poll() == retired_worker.returncode == value["actual_returncode"]
              and value["session_id"] == retired_worker.source_broker.descriptor["session_id"]
              and value["descriptor_sha256"] == retired_worker.source_broker.descriptor_sha256
              and value["round_number"] == retired_worker.round_number,
              "historical validation retirement differs")
        return value

    def revalidate(self, *, launch, retired_worker, snapshot=None):
        from .owner_candidate_dispatch import CapturedCodingLaunch, RetiredCodingWorker
        _need(type(launch) is CapturedCodingLaunch and type(retired_worker) is RetiredCodingWorker
              and self._launch is launch and self._retired is retired_worker, "exact live validation custody required")
        with _LOCK:
            _need(_OBSERVATIONS.get(launch.source_broker, {}).get(retired_worker.round_number) is self,
                  "registered live validation observation required")
        value = self.retained_metadata(launch=launch, retired_worker=retired_worker)
        contract = launch.public_validation_contract
        _need(type(contract) is CapturedPublicValidationContract
              and value["contract_sha256"] == _sha(contract.metadata_bytes)
              and value["original_deadline_ns"] == contract.revalidate()["deadline_monotonic_ns"]
              and value["actual_returncode"] == retired_worker.revalidate()
              and value["round_number"] == retired_worker.round_number, "validation launch or retirement changed")
        generation, hashes = _generation(launch.source_broker, value["workspace_generation"])
        _need(hashes == value["source_hashes"]
              and _states(launch.source_broker.workspace, contract.metadata["sources"]) == value["workspace_states"],
              "validated candidate generation or modes changed")
        if snapshot is not None:
            meta = snapshot.metadata
            _need(snapshot._source_broker is launch.source_broker and meta["workspace_generation"] == generation
                  and meta["source_hashes"] == hashes and meta["prior_returncode"] == value["actual_returncode"],
                  "candidate snapshot differs from public validation generation")
        _need(time.monotonic_ns() < value["original_deadline_ns"], "original validation deadline exhausted during source checks")
        return value


def validate_retired_candidate(*, launch, retired_worker):
    from .owner_candidate_dispatch import CapturedCodingLaunch, RetiredCodingWorker
    from . import candidate_execution as execution
    _need(type(launch) is CapturedCodingLaunch and type(retired_worker) is RetiredCodingWorker
          and retired_worker._launch is launch, "actual owner launch and retirement required")
    retired_worker.revalidate()
    contract = launch.public_validation_contract
    _need(type(contract) is CapturedPublicValidationContract, "captured public validation contract required")
    metadata = contract.revalidate()
    with _LOCK:
        observations = _OBSERVATIONS.setdefault(launch.source_broker, {})
        _need(retired_worker.round_number not in observations, "retired candidate already validated")
        # Consume before execution; partial failures never permit replay.
        observations[retired_worker.round_number] = None
    generation, hashes = _generation(launch.source_broker)
    states = _states(launch.source_broker.workspace, metadata["sources"])
    rows = []
    for check, spec in zip(profiles.PUBLIC_CHECKS, metadata["task_spec"]["validations"]):
        row = dict(validation_key=spec["validation_key"], returncode=None, result=None,
                   status="inconclusive", reason="not_run")
        contract.revalidate()
        _need(_generation(launch.source_broker, generation)[1] == hashes
              and _states(launch.source_broker.workspace, metadata["sources"]) == states,
              "candidate changed before public validation")
        try:
            completed = execution.run_candidate_bounded(metadata["runner"], spec["argv"],
                cwd=launch.source_broker.workspace, timeout=min(300, launch.remaining_seconds))
            row["returncode"] = completed.returncode
            try:
                from .router_public_instruction import _unique
                result = profiles.public_validation_result(json.loads(completed.stdout, object_pairs_hook=_unique), check=check)
                _need(type(completed.returncode) is int
                      and completed.returncode == {"passed": 0, "failed": 1, "inconclusive": 2}[result["status"]],
                      "public check process and result disagree")
                row.update(result=result, status=result["status"], reason=result["reason"])
            except (ValueError, TypeError, UnicodeError):
                row["reason"] = "invalid_result"
        except execution.CandidateOutputLimitError:
            row["reason"] = "output_limit"
        except subprocess.TimeoutExpired:
            row["reason"] = "timeout"
        finally:
            # The launcher's ordinary exit alone cannot distinguish a failed
            # check from failed descendant cleanup. Observe the fixed owner
            # cleanup independently before retaining any check result.
            from .owner_candidate_dispatch import _isolated_cleanup
            cleanup = _isolated_cleanup()
            _need(type(cleanup.returncode) is int and cleanup.returncode == 0,
                  "public validation worker cleanup failed")
        contract.revalidate()
        _need(_generation(launch.source_broker, generation)[1] == hashes
              and _states(launch.source_broker.workspace, metadata["sources"]) == states,
              "candidate changed during public validation")
        rows.append(row)
    failed = [row["validation_key"] for row in rows if row["status"] == "failed"]
    value = dict(schema=OBSERVATION_SCHEMA, contract_sha256=_sha(contract.metadata_bytes),
        original_deadline_ns=metadata["deadline_monotonic_ns"], actual_returncode=retired_worker.returncode,
        round_number=retired_worker.round_number, session_id=retired_worker.source_broker.descriptor["session_id"],
        descriptor_sha256=retired_worker.source_broker.descriptor_sha256,
        workspace_generation=generation, source_hashes=hashes,
        workspace_states=states, status="failed" if failed else "passed" if all(r["status"] == "passed" for r in rows) else "inconclusive",
        failed_obligations=failed, checks=rows, **FLAGS)
    validate_public_validation_observation(value)
    result = RetiredCandidateValidation(_TOKEN, launch, retired_worker, canonical_json(value))
    with _LOCK:
        _OBSERVATIONS[launch.source_broker][retired_worker.round_number] = result
    result.revalidate(launch=launch, retired_worker=retired_worker)
    return result


def validate_public_validation_projection_binding(projection, *, handoff, wire, descriptor):
    """Closed historical joins; no signature replay, freshness or retry grant."""
    _need(type(projection) is dict and set(projection) == {"contract", "observation"}, "closed public validation projection required")
    contract = validate_public_validation_contract_metadata(projection["contract"])
    observation = validate_public_validation_observation(projection["observation"])
    admission = handoff["admission"]
    candidate = wire["candidate_binding"]
    _need(observation["contract_sha256"] == admission["public_validation_contract_sha256"] == _sha(canonical_json(contract))
          and admission["public_validation_observation_sha256"] == _sha(canonical_json(observation))
          and admission["public_validation_observation_bytes"] == len(canonical_json(observation))
          and canonical_json(wire.get("public_validation")) == canonical_json(dict(schema="supervisor-public-validation-feedback-binding@1",
              sha256=_sha(canonical_json(observation)), bytes=len(canonical_json(observation)),
              feedback=public_validation_feedback(observation)))
          and contract["deadline_monotonic_ns"] == observation["original_deadline_ns"] == admission["deadline_monotonic_ns"]
          and contract["boundary_sha256"] == descriptor["boundary_manifest_sha256"]
          and contract["original_requirements_sha256"] == wire["original_requirements_binding"]["original_requirements_sha256"]
          and observation["workspace_generation"] == candidate["workspace_generation"]
          and observation["source_hashes"] == candidate["source_hashes"]
          and observation["actual_returncode"] == wire["prior_failure"]["prior_returncode"]
          and observation["session_id"] == wire["prior_failure"]["prior_session_id"]
          and observation["descriptor_sha256"] == wire["prior_failure"]["descriptor_sha256"]
          and observation["round_number"] == 1
          and (1 <= observation["actual_returncode"] <= 255 or observation["actual_returncode"] == 0 and observation["status"] == "failed")
          and admission["continuation_reason"] == ("public_validation_failed" if observation["actual_returncode"] == 0 else "worker_exit_nonzero"),
          "public validation projection differs from candidate dispatch")
    return _copy(projection)


def validate_public_validation_projection(projection, *, handoff, wire, descriptor, repository, workspace):
    """Recheck a protected worker projection, never turn JSON into a live handle."""
    result = validate_public_validation_projection_binding(projection, handoff=handoff, wire=wire, descriptor=descriptor)
    contract, observation = result["contract"], result["observation"]
    _replay_contract(contract, Path(repository), Path(workspace))
    _need(_states(Path(workspace), contract["sources"]) == observation["workspace_states"],
          "public validation candidate states changed")
    return result
