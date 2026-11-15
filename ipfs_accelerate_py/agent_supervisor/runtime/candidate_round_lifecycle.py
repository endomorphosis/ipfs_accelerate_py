"""Private historical owner lifecycle; no admission or retry reload API."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import subprocess

from .worker_source_views import canonical_json

SCHEMA = "owner-candidate-rounds-retention@1"
DOCTOR_SCHEMA = "owner-candidate-rounds-retention@2"
PUBLIC_SCHEMA = "owner-candidate-rounds-retention@3"
MAX_BYTES = 65536
FLAGS = dict(audit_only=True, dispatch_reload_allowed=False, proof_authority=False,
    execution_authority=False, completion_authority=False)
ROW_FIELDS = frozenset(("round_number", "process_started", "session_id", "descriptor_sha256",
    "returncode", "wait_completed", "retirement_observed"))
OBSERVATION_FIELDS = frozenset(("schema", "rounds", "max_rounds", "elapsed_seconds", "outcome",
    "failure_type", "completion_authority", "proof_authority", "source_obligations_discharged",
    "accepted_index_published"))


def _need(value):
    if not value:
        raise ValueError("native owner lifecycle binding differs")


def _digest(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def lifecycle_name(session):
    _need(type(session) is str and re.fullmatch(r"[0-9a-f]{32}", session) is not None)
    return "candidate-rounds-" + session + ".json"


def _observation(value, *, normalized):
    public = type(value) is dict and value.get("schema") == "owner-candidate-rounds-observation@2"
    fields = OBSERVATION_FIELDS | ({"public_validation_observations", "owner_validation_outcome", "owner_returncode"}
        if public else set())
    _need(type(value) is dict and set(value) >= fields
          and not set(value)-fields-{"candidate_input_retention"}
          and value["schema"] in ("owner-candidate-rounds-observation@1", "owner-candidate-rounds-observation@2")
          and type(value["max_rounds"]) is int and value["max_rounds"] == 2
          and type(value["elapsed_seconds"]) in (int, float)
          and 0 <= value["elapsed_seconds"] <= 86400 and math.isfinite(value["elapsed_seconds"])
          and value["outcome"] in ("failed", "workers_returned")
          and (value["failure_type"] is None or type(value["failure_type"]) is str
               and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,95}", value["failure_type"]) is not None)
          and all(value[k] is False for k in ("completion_authority", "proof_authority", "accepted_index_published"))
          and type(value["source_obligations_discharged"]) is int and value["source_obligations_discharged"] == 0)
    rows = value["rounds"]
    _need(type(rows) is list and 1 <= len(rows) <= 2)
    sessions = set()
    for number, row in enumerate(rows, 1):
        _need(type(row) is dict and set(row) == ROW_FIELDS
              and type(row["round_number"]) is int and row["round_number"] == number
              and (type(row["process_started"]) is bool or normalized and row["process_started"] is None)
              and type(row["wait_completed"]) is bool and type(row["retirement_observed"]) is bool
              and (row["returncode"] is None or type(row["returncode"]) is int and -255 <= row["returncode"] <= 255)
              and _digest(row["descriptor_sha256"]))
        lifecycle_name(row["session_id"])
        _need(row["session_id"] not in sessions)
        sessions.add(row["session_id"])
        if normalized:
            _need(not row["wait_completed"] or row["process_started"] is True and row["returncode"] is not None)
            _need(not row["retirement_observed"] or row["process_started"] is True and row["returncode"] is not None)
    if public:
        from .candidate_public_validation import validate_public_validation_observation
        observations = value["public_validation_observations"]
        _need(type(observations) is list and len(observations) <= len(rows)
            and value["owner_validation_outcome"] in (None, "passed", "failed", "inconclusive")
            and (value["owner_returncode"] is None or type(value["owner_returncode"]) is int
                and -255 <= value["owner_returncode"] <= 255))
        for row, observed in zip(rows, observations):
            validate_public_validation_observation(observed)
            _need(row["retirement_observed"] is True
                and all(observed[key] == row[key] for key in ("round_number", "session_id", "descriptor_sha256"))
                and observed["actual_returncode"] == row["returncode"])
        _need(value["owner_validation_outcome"] == (observations[-1]["status"] if observations else None))
        if value["failure_type"] is None:
            actual = rows[-1]["returncode"]
            expected = 1 if actual == 0 and value["owner_validation_outcome"] != "passed" else actual
            _need(type(value["owner_returncode"]) is int and value["owner_returncode"] == expected)
        else:
            _need(value["owner_returncode"] is None)
    return value


def validate_retention(value):
    keys = {"schema", "prior_session_id", "prior_descriptor_sha256", "task_id", "native_prompt_sha256",
        "workspace_sha256", "owner_pid", "owner_start_time_ticks", "boundary_manifest_sha256",
        "observation", *FLAGS}
    doctor = type(value) is dict and value.get("schema") in (DOCTOR_SCHEMA, PUBLIC_SCHEMA)
    if doctor:
        keys.update(("doctor_coding_policy", "doctor_attempts_reserved"))
    _need(type(value) is dict and set(value) == keys and value["schema"] in (SCHEMA, DOCTOR_SCHEMA, PUBLIC_SCHEMA)
          and all(type(value[k]) is type(v) and value[k] == v for k, v in FLAGS.items()))
    lifecycle_name(value["prior_session_id"])
    _need(all(_digest(value[k]) for k in ("prior_descriptor_sha256", "native_prompt_sha256",
        "workspace_sha256", "boundary_manifest_sha256"))
        and type(value["task_id"]) is str and 0 < len(value["task_id"].encode()) <= 4096
        and all(type(value[k]) is int and value[k] > 0 for k in ("owner_pid", "owner_start_time_ticks")))
    observation = _observation(value["observation"], normalized=True)
    first = observation["rounds"][0]
    _need(first["session_id"] == value["prior_session_id"]
          and first["descriptor_sha256"] == value["prior_descriptor_sha256"])
    if doctor:
        from .candidate_round_policy import validate_doctor_coding_policy_shape
        policy = validate_doctor_coding_policy_shape(value["doctor_coding_policy"])
        from .candidate_round_policy import PUBLIC_VALIDATION_POLICY
        public = policy["policy_id"] == PUBLIC_VALIDATION_POLICY
        _need((value["schema"] == PUBLIC_SCHEMA) is public
            and (observation["schema"] == "owner-candidate-rounds-observation@2") is public)
        _need(all(policy[key] == value[key] for key in ("prior_session_id", "prior_descriptor_sha256",
            "task_id", "native_prompt_sha256")))
        reservations = value["doctor_attempts_reserved"]
        _need(type(reservations) is list and all(type(number) is int for number in reservations)
            and reservations in ([], [1], [1, 2]))
        for row in observation["rounds"]:
            _need(row["process_started"] is not True or row["round_number"] in reservations)
    return value


def parse_retention(raw):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            _need(key not in value)
            value[key] = item
        return value
    def nonfinite(_):
        raise ValueError("finite lifecycle JSON required")
    _need(type(raw) is bytes and len(raw) <= MAX_BYTES)
    value = json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)
    _need(canonical_json(value) == raw)
    return validate_retention(value)


def _native_rows(launch, observation):
    """Join caller diagnostics to live handles and fill omitted possible rounds.

    A candidate broker without registration may have failed after Popen. Null
    preserves that uncertainty instead of converting a missing row to zero.
    """
    from .owner_candidate_dispatch import RetiredCodingWorker, _closed
    from .worker_candidate_round_broker import WorkerCandidateRoundBroker
    supplied = {row["round_number"]: row for row in observation["rounds"]}
    state = launch._state
    brokers = [launch.source_broker]
    if state.candidate_broker is not None:
        _need(type(state.candidate_broker) is WorkerCandidateRoundBroker)
        brokers.append(state.candidate_broker)
    _need(set(supplied) <= set(range(1, len(brokers)+1)))
    result = []
    for number, broker in enumerate(brokers, 1):
        _closed(broker)
        broker.protocol.validate_descriptor(broker.descriptor)
        _need(hashlib.sha256(canonical_json(broker.descriptor)).hexdigest() == broker.descriptor_sha256)
        previous = supplied.get(number)
        if previous is not None:
            _need(previous["session_id"] == broker.descriptor["session_id"]
                  and previous["descriptor_sha256"] == broker.descriptor_sha256)
        process = broker._process
        if process is None:
            _need(not broker._registered.is_set())
            started, code = None, None
        else:
            _need(type(process) is subprocess.Popen and broker._registered.is_set()
                  and broker._child is not None and broker._child["pid"] == process.pid
                  and broker._child["parent_pid"] == os.getpid()
                  and type(process.returncode) is int and process.poll() == process.returncode)
            started, code = True, process.returncode
            if previous is not None:
                _need(previous["process_started"] is True and previous["returncode"] == code)
        retirement = state.initial_retirement if number == 1 else state.second_retirement
        retired = retirement is not None
        if retired:
            _need(type(retirement) is RetiredCodingWorker and retirement._launch is launch
                  and retirement.source_broker is broker and retirement.process is process
                  and retirement.returncode == code and retirement.round_number == number)
        if previous is not None:
            _need(not previous["retirement_observed"] or retired)
        result.append(dict(round_number=number, process_started=started,
            session_id=broker.descriptor["session_id"], descriptor_sha256=broker.descriptor_sha256,
            returncode=code, wait_completed=started is True and previous is not None and previous["wait_completed"],
            retirement_observed=retired))
    return result


def persist(*, source_broker, observation):
    from .worker_source_broker import WorkerSourceBroker
    from .source_formalization_context import _protected_directory, _owner_boundary
    from .owner_candidate_round_checkpoint import _write, _sync_directory, _identity
    from .owner_candidate_dispatch import CapturedCodingLaunch, _LAUNCHES, _identity as owner_identity

    _need(type(source_broker) is WorkerSourceBroker)
    launch = _LAUNCHES.get(source_broker)
    _need(type(launch) is CapturedCodingLaunch and launch.source_broker is source_broker
          and launch._owner_identity == owner_identity()
          and source_broker.owner_uid == os.getuid() == os.geteuid() == os.getgid() > 0)
    _observation(observation, normalized=False)
    descriptor = source_broker.descriptor
    boundary, boundary_sha = _owner_boundary(source_broker.boundary)
    _need(boundary_sha == descriptor["boundary_manifest_sha256"]
          and boundary["owner_uid"] == source_broker.owner_uid
          and boundary["worker_uid"] == source_broker.worker_uid
          and descriptor["owner_pid"] == os.getpid())
    normalized = dict(observation, rounds=_native_rows(launch, observation))
    if launch.public_validation_contract is not None:
        from .candidate_public_validation import RetiredCandidateValidation
        retained = []
        for number, observed in sorted(launch._state.public_validations.items()):
            _need(type(number) is int and number in (1, 2) and type(observed) is RetiredCandidateValidation)
            retired = launch._state.initial_retirement if number == 1 else launch._state.second_retirement
            retained.append(observed.retained_metadata(launch=launch, retired_worker=retired))
        _need(canonical_json(retained) == canonical_json(observation["public_validation_observations"]))
    payload = dict(schema=SCHEMA, prior_session_id=descriptor["session_id"],
        prior_descriptor_sha256=source_broker.descriptor_sha256,
        task_id=descriptor["task_id"], native_prompt_sha256=descriptor["native_prompt_sha256"],
        workspace_sha256=hashlib.sha256(str(source_broker.workspace).encode()).hexdigest(),
        owner_pid=descriptor["owner_pid"], owner_start_time_ticks=descriptor["owner_start_time_ticks"],
        boundary_manifest_sha256=boundary_sha, observation=normalized, **FLAGS)
    if launch.doctor_coding_policy is not None:
        from .candidate_round_policy import validate_doctor_coding_policy_binding
        policy = validate_doctor_coding_policy_binding(launch.doctor_coding_policy,
            doctor_requirements=launch._original_requirements._doctor_requirements.metadata,
            original_requirements_sha256=hashlib.sha256(launch._original_requirements.metadata_bytes).hexdigest(),
            deadline_monotonic_ns=int(launch.absolute_deadline*1_000_000_000),
            **({} if launch.public_validation_contract is None else dict(
                policy_id=launch.doctor_coding_policy["policy_id"],
                public_validation_contract=launch.public_validation_contract.metadata)))
        payload.update(schema=DOCTOR_SCHEMA if launch.public_validation_contract is None else PUBLIC_SCHEMA, doctor_coding_policy=policy,
            doctor_attempts_reserved=sorted(launch._state.reserved_attempts))
    validate_retention(payload)
    raw = canonical_json(payload)
    _need(len(raw) <= MAX_BYTES)
    parent = source_broker.descriptor_path.parent
    _need(parent == source_broker.boundary.parent / "worker-observations")
    _protected_directory(parent, owner=os.geteuid(), group=os.geteuid())
    parent_identity, created = _identity(parent), {}
    path = parent / lifecycle_name(descriptor["session_id"])
    try:
        _write(path, raw, created)
        _need(_identity(parent) == parent_identity and _identity(path) == created[path])
        info = path.lstat()
        _need(info.st_uid == info.st_gid == os.geteuid() and stat.S_IMODE(info.st_mode) == 0o400
              and info.st_nlink == 1 and path.read_bytes() == raw)
        _protected_directory(parent, owner=os.geteuid(), group=os.geteuid())
        _sync_directory(parent)
    except BaseException:
        if path in created:
            try:
                if _identity(path) == created[path]:
                    path.unlink()
            except FileNotFoundError:
                pass
        raise
    return dict(path=path.name, sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
