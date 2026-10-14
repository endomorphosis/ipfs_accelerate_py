"""One owner-captured continuation after an isolated failed coding worker.

The local handles here bind an actual wait and fixed container cleanup to one
additional child. Candidate snapshots and serialized observations alone never
grant that permission. The database attempt and its original deadline remain
owned by the enclosing supervisor invocation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
from subprocess import Popen as _NativePopen
import threading
import time
from weakref import WeakKeyDictionary

from .worker_source_views import canonical_json

_TOKEN = object()
_LOCK = threading.RLock()
_LAUNCHES = WeakKeyDictionary()
_PREFIX = ("/usr/bin/sudo", "-n", "-u", "benchmarkworker", "--",
           "/opt/ipfs-supervisor/bin/worker-entry")
_CLEANUP = (*_PREFIX, "--cleanup")
_VALUE_FLAGS = frozenset(("--provider", "--model", "--reasoning-effort", "--timeout",
    "--max-output-tokens", "--budget-profile", "--purpose", "--semantic-repository",
    "--grok-coding-max-turns", "--semantic-transport-schema", "--semantic-metadata-view",
    "--semantic-working-set", "--coding-reply-mode", "--source-view-profile",
    "--source-edit-mode", "--source-view-policy-json", "--public-instruction-artifact",
    "--public-instruction-sha256", "--public-instruction-task-cid",
    "--source-formalization-handoff", "--source-formalization-handoff-sha256"))


def _need(value, reason):
    if not value:
        raise ValueError(reason)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _identity():
    from .worker_source_client import process_identity
    return canonical_json(process_identity(os.getpid()))


def _arguments(argv):
    _need(type(argv) is list and all(type(value) is str for value in argv),
          "literal captured coding argv required")
    values, index = {}, 0
    while index < len(argv):
        flag, equals, value = argv[index].partition("=")
        _need(flag in _VALUE_FLAGS and flag not in values,
              "unsupported or duplicate continuation launch selection")
        if not equals:
            index += 1
            _need(index < len(argv), "complete continuation launch selection required")
            value = argv[index]
        _need(bool(value) and not value.startswith("--"), "literal launch value required")
        values[flag] = value
        index += 1
    return values


def _validate_initial_handoff(broker, formal, boundary_sha):
    """Inspect the owner publication without entering the worker-UID branch."""
    from . import source_formalization_context as guard
    _need(all(formal), "complete initial formalization handoff required")
    path = Path(formal[0]).absolute()
    parent = broker.boundary.parent / "worker-observations"
    guard._protected_directory(parent, owner=broker.owner_uid, group=broker.owner_uid)
    _need(path.parent == parent and re.fullmatch(r"formalization-[0-9a-f]{32}\.json", path.name)
          and guard._digest(formal[1]), "exact initial owner handoff path required")
    raw = guard._read(path, guard.MAX_HANDOFF_BYTES)
    info = path.lstat()
    _need(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o440
          and info.st_uid == info.st_gid == broker.owner_uid and info.st_nlink == 1
          and not path.is_symlink() and _sha(raw) == formal[1],
          "initial owner handoff publication changed")
    value = json.loads(raw, object_pairs_hook=guard._unique, parse_constant=guard._nonfinite)
    parsed = guard._parts(broker.prompt)
    _need(parsed is not None, "initial handoff requires exact native formalization evidence")
    payload, text, chunks = parsed
    expected_keys = {"schema", "repository", "workspace", "prompt_sha256", "worker_evidence_sha256",
        "task_id", "task_cid", "context_bundle", "source_hashes", "boundary_sha256",
        "issued_monotonic_ns", "expires_monotonic_ns", "owner_observation",
        "proof_authority", "execution_authority", "completion_authority"}
    _need(type(value) is dict and set(value) == expected_keys and guard._json(value).encode() == raw
          and value["schema"] == guard.HANDOFF_SCHEMA and value["repository"] == str(broker.repository)
          and value["workspace"] == str(broker.workspace)
          and value["prompt_sha256"] == _sha(broker.prompt.encode())
          and value["worker_evidence_sha256"] == _sha(text.encode())
          and value["boundary_sha256"] == boundary_sha
          and all(value[key] == payload[key] for key in ("task_id", "task_cid", "context_bundle"))
          and all(value[key] is False for key in ("proof_authority", "execution_authority", "completion_authority")),
          "initial handoff differs from captured owner invocation")
    sources = value["source_hashes"]
    _need(type(sources) is dict and len(sources) <= 768, "bounded initial formalization population required")
    guard._current_sources(broker.repository, sources)
    guard._current_sources(broker.workspace, sources)
    _need(value["owner_observation"] == guard._observation(payload, text, chunks, sources),
          "initial formalization owner observation changed")
    issued, expires = value["issued_monotonic_ns"], value["expires_monotonic_ns"]
    _need(type(issued) is int and type(expires) is int
          and expires-issued == guard.HANDOFF_LIFETIME_NS
          and issued <= time.monotonic_ns() <= expires,
          "initial formalization handoff expired")


@dataclass
class _State:
    phase: str = "captured"
    initial_retirement: object = None
    second_retirement: object = None
    candidate_broker: object = None
    admission: object = None
    retirement_started: set = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class CapturedCodingLaunch:
    _token: object = field(repr=False)
    source_broker: object = field(repr=False)
    _argv: tuple = field(repr=False)
    _settings_bytes: bytes = field(repr=False)
    _owner_identity: bytes = field(repr=False)
    _original_requirements: object = field(repr=False)
    _binding_bytes: bytes = field(repr=False)
    _descriptor_sha256: str
    absolute_deadline: float
    started_monotonic: float
    _state: _State = field(repr=False, compare=False)

    def __post_init__(self):
        _need(self._token is _TOKEN, "owner capture factory required")

    @property
    def launch_settings(self):
        return json.loads(self._settings_bytes)

    @property
    def remaining_seconds(self):
        remaining = self.absolute_deadline - time.monotonic()
        _need(remaining > 0, "original coding deadline exhausted")
        return remaining

    def revalidate(self):
        from . import source_formalization_context as guard
        self.remaining_seconds
        with _LOCK:
            _need(_LAUNCHES.get(self.source_broker) is self,
                  "unregistered owner launch handle")
        broker = self.source_broker
        _need(_identity() == self._owner_identity, "coding launch belongs to another owner")
        boundary, boundary_sha = guard._owner_boundary(broker.boundary)
        _need(boundary_sha == broker.descriptor["boundary_manifest_sha256"]
              and boundary["owner_uid"] == os.getuid() == os.geteuid() == broker.owner_uid
              and boundary["worker_uid"] == broker.worker_uid,
              "captured coding boundary changed")
        _need(broker.original_requirements is self._original_requirements
              and broker.prompt.encode() == self._original_requirements.native_prompt
              and canonical_json(broker._binding) == self._binding_bytes
              and broker.descriptor_sha256 == self._descriptor_sha256
              and _sha(canonical_json(broker.descriptor)) == self._descriptor_sha256,
              "original coding requirements or binding changed")
        return self.launch_settings

    def validate_first_command(self, command):
        from .worker_source_broker import worker_source_argv
        self.revalidate()
        _need(self._state.phase == "captured" and not self.source_broker._registered.is_set()
              and not self.source_broker._closed, "original launch is no longer fresh")
        expected = (*_PREFIX, *worker_source_argv(list(self._argv)),
                    *self.source_broker.extra_argv)
        _need(type(command) in (list, tuple) and tuple(command) == expected,
              "first child command differs from captured owner launch")


def capture_original_coding_launch(*, source_broker, argv, absolute_deadline, max_rounds=2):
    """Capture immutable initial choices before any worker is registered."""
    from .worker_source_broker import WorkerSourceBroker
    from .owner_candidate_round_prompt import OriginalRequirements
    from . import source_formalization_context as guard
    _need(type(max_rounds) is int and max_rounds == 2, "explicit two-round policy required")
    _need(type(source_broker) is WorkerSourceBroker, "exact initial source broker required")
    broker = source_broker
    _need(not broker._closed and not broker._registered.is_set() and broker._process is None,
          "launch choices must be captured before the original worker")
    values = _arguments(argv)
    _need(values.get("--purpose", "coding") == "coding", "ordinary coding continuation required")
    _need(values.get("--source-edit-mode", "disabled") == "disabled",
          "mapped edit continuation is not admitted")
    _need(values.get("--source-view-profile") == broker._binding["profile"]
          and values.get("--semantic-repository") == str(broker.repository),
          "original source selection differs from broker")
    requirements = broker.original_requirements
    _need(type(requirements) is OriginalRequirements and requirements._source_broker is broker,
          "complete original requirements capture required")
    _need(requirements._doctor_requirements is None,
          "Doctor continuation requires independent residual budget enforcement")
    _need(requirements.native_prompt == broker.prompt.encode(), "original native prompt changed")
    public = tuple(values.get(flag) for flag in ("--public-instruction-artifact",
        "--public-instruction-sha256", "--public-instruction-task-cid"))
    _need((not any(public) and requirements._public_request is None)
          or all(public) and tuple(map(str, requirements._public_request or ())) == public,
          "original public instruction selection differs")
    boundary, boundary_sha = guard._owner_boundary(broker.boundary)
    formal = tuple(values.get(flag) for flag in (
        "--source-formalization-handoff", "--source-formalization-handoff-sha256"))
    if any(formal):
        _validate_initial_handoff(broker, formal, boundary_sha)
    timeout = int(values.get("--timeout", "90"))
    _need(1 <= timeout <= 1800 and type(absolute_deadline) in (int, float)
          and 0 < absolute_deadline <= time.monotonic()+timeout
          and math.isfinite(absolute_deadline)
          and 0 < absolute_deadline - time.monotonic() <= timeout,
          "unchanged bounded original coding deadline required")
    settings = dict(provider=values.get("--provider", boundary.get("provider", "codex_cli")),
        model=values.get("--model", "gpt-6.1-sol"),
        reasoning_effort=values.get("--reasoning-effort", "high"),
        max_output_tokens=int(values.get("--max-output-tokens", "4096")),
        budget_profile=values.get("--budget-profile"),
        grok_coding_max_turns=int(values.get("--grok-coding-max-turns", "128")))
    _need(settings["provider"] in ("codex_cli", "grok_cli")
          and settings["reasoning_effort"] in ("low", "medium", "high", "xhigh", "max")
          and settings["max_output_tokens"] > 0
          and 1 <= settings["grok_coding_max_turns"] <= 128,
          "bounded original coding settings required")
    result = CapturedCodingLaunch(_TOKEN, broker, tuple(argv), canonical_json(settings),
        _identity(), requirements, canonical_json(broker._binding), broker.descriptor_sha256,
        float(absolute_deadline), float(absolute_deadline)-timeout, _State())
    with _LOCK:
        _need(broker not in _LAUNCHES, "original broker already captured a continuation policy")
        _LAUNCHES[broker] = result
    result.revalidate()
    return result


@dataclass(frozen=True, slots=True)
class RetiredCodingWorker:
    _token: object = field(repr=False)
    _launch: CapturedCodingLaunch = field(repr=False)
    source_broker: object = field(repr=False)
    process: object = field(repr=False)
    returncode: int
    round_number: int
    _child_bytes: bytes = field(repr=False)

    def __post_init__(self):
        _need(self._token is _TOKEN, "fixed owner cleanup factory required")

    def revalidate(self):
        self._launch.revalidate()
        state = self._launch._state
        registered = state.initial_retirement if self.round_number == 1 else state.second_retirement
        broker, process = self.source_broker, self.process
        _need(registered is self and type(process) is _NativePopen
              and broker._process is process and broker._registered.is_set()
              and type(process.returncode) is int and process.returncode == self.returncode
              and process.poll() == self.returncode
              and canonical_json(broker._child) == self._child_bytes,
              "retired worker custody changed")
        _closed(broker)
        return self.returncode


def _closed(broker):
    _need(broker._closed and broker._stop.is_set()
          and (broker._thread is None or not broker._thread.is_alive())
          and not broker.socket_path.exists() and not broker.descriptor_path.exists(),
          "retired coding source broker remains active")
    with broker._accounting_lock:
        _need(broker._inflight is None and broker._accepted == len(broker._records),
              "retired coding source accounting incomplete")


def _isolated_cleanup():
    return subprocess.run(list(_CLEANUP), cwd="/", stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, start_new_session=True, timeout=15)


def retire_coding_worker(*, launch, process, source_broker):
    """Always contain/close the selected round; mint only from actual custody.

    Call after wait (including reap after interruption). A missing/unregistered
    child still triggers cleanup but cannot create a retirement handle. Cleanup
    retains the fixed containment timeout; it never extends the coding lease.
    """
    _need(type(launch) is CapturedCodingLaunch, "captured owner coding launch required")
    with _LOCK:
        _need(_LAUNCHES.get(launch.source_broker) is launch,
              "unregistered owner launch handle")
        state = launch._state
        round_number = 1 if source_broker is launch.source_broker else 2
        _need(round_number == 1 or source_broker is state.candidate_broker,
              "retirement belongs to another source broker")
        retirement_name = "initial_retirement" if round_number == 1 else "second_retirement"
        _need(getattr(state, retirement_name) is None and round_number not in state.retirement_started,
              "coding worker already retired or retirement attempted")
        state.retirement_started.add(round_number)
    _need(_identity() == launch._owner_identity, "retirement belongs to another owner")
    numbers = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
    previous = {number: signal.getsignal(number) for number in numbers}
    try:
        for number in numbers:
            signal.signal(number, signal.SIG_IGN)
        try:
            cleanup = _isolated_cleanup()
            _need(type(cleanup.returncode) is int and cleanup.returncode == 0,
                  "isolated worker cleanup failed")
        finally:
            source_broker.close()
        _need(type(process) is _NativePopen and source_broker._process is process
              and source_broker._registered.is_set()
              and type(process.returncode) is int and process.poll() == process.returncode
              and source_broker._child is not None
              and source_broker._child["pid"] == process.pid
              and source_broker._child["parent_pid"] == os.getpid(),
              "actual registered waited coding worker required")
        _closed(source_broker)
        retirement = RetiredCodingWorker(_TOKEN, launch, source_broker, process,
            process.returncode, round_number, canonical_json(source_broker._child))
        with _LOCK:
            _need(getattr(state, retirement_name) is None, "coding worker already retired")
            setattr(state, retirement_name, retirement)
        return retirement
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


@dataclass(frozen=True, slots=True)
class CandidateDispatchAdmission:
    _token: object = field(repr=False)
    _launch: CapturedCodingLaunch = field(repr=False)
    _retired: RetiredCodingWorker = field(repr=False)
    _snapshot: object = field(repr=False)
    _prompt: object = field(repr=False)
    _source_broker: object = field(repr=False)
    _binding_bytes: bytes = field(repr=False)

    def __post_init__(self):
        _need(self._token is _TOKEN, "one-use owner admission factory required")

    def revalidate(self, *, snapshot, prompt, source_broker):
        launch = self._launch
        launch.revalidate()
        _need(launch._state.admission is self and launch._state.phase in ("preparing", "prepared", "starting")
              and snapshot is self._snapshot and prompt is self._prompt
              and source_broker is self._source_broker
              and launch._state.candidate_broker is source_broker,
              "candidate admission does not own these live handles")
        _need(self._retired.revalidate() in range(1, 256), "positive original coding failure required")
        metadata = snapshot.revalidate()
        prompt.revalidate()
        _need(snapshot._source_broker is launch.source_broker
              and snapshot._absolute_deadline == launch.absolute_deadline
              and prompt._snapshot is snapshot
              and prompt._requirements is launch._original_requirements
              and metadata["round_number"] == 2,
              "candidate admission requirements or fixed deadline changed")
        _need(not source_broker._closed and not source_broker._registered.is_set()
              and source_broker._process is None, "candidate service is not fresh")
        source_broker._current_context()
        binding = _admission_binding(launch, snapshot, prompt, source_broker)
        _need(canonical_json(binding) == self._binding_bytes, "candidate dispatch binding changed")
        return binding


def _admission_binding(launch, snapshot, prompt, source_broker):
    metadata = snapshot.metadata
    return dict(schema="supervisor-owner-candidate-dispatch-admission@1", round_number=2,
        original_argv_sha256=_sha(canonical_json(list(launch._argv))),
        deadline_monotonic_ns=int(launch.absolute_deadline * 1_000_000_000),
        launch_settings=launch.launch_settings, task_id=metadata["task_id"],
        native_prompt_sha256=metadata["native_prompt_sha256"],
        prior_session_id=metadata["prior_session_id"],
        prior_descriptor_sha256=metadata["prior_descriptor_sha256"],
        workspace_generation=metadata["workspace_generation"],
        round_context_sha256=_sha(snapshot.metadata_bytes),
        candidate_scope_cid=metadata["candidate_scope_cid"],
        candidate_semantic_root_cid=metadata["semantic_root_cid"],
        session_id=source_broker.descriptor["session_id"],
        descriptor_sha256=source_broker.descriptor_sha256,
        candidate_prompt_sha256=_sha(prompt.provider_prompt))


def _sealed_stdin(raw):
    from ..validation.validation_runtime import _validation_memfd_create
    seals = sum(getattr(fcntl, name, fallback) for name, fallback in (
        ("F_SEAL_SEAL", 1), ("F_SEAL_SHRINK", 2), ("F_SEAL_GROW", 4), ("F_SEAL_WRITE", 8)))
    fd = _validation_memfd_create("supervisor-candidate-prompt",
        getattr(os, "MFD_CLOEXEC", 1) | getattr(os, "MFD_ALLOW_SEALING", 2))
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(raw)
        os.lseek(fd, 0, os.SEEK_SET)
        fcntl.fcntl(fd, getattr(fcntl, "F_ADD_SEALS", 1033), seals)
        _need(fcntl.fcntl(fd, getattr(fcntl, "F_GET_SEALS", 1034)) & seals == seals
              and os.pread(fd, len(raw)+1, 0) == raw, "candidate stdin was not sealed")
        return fd
    except BaseException:
        os.close(fd)
        raise


def _spawn_worker(command, *, stdin, workspace):
    return subprocess.Popen(command, stdin=stdin, cwd=workspace)


def _terminate(process):
    if process is not None and process.poll() is None:
        try:
            process.terminate()
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


@dataclass(frozen=True, slots=True)
class RunningCandidateWorker:
    process: object
    source_broker: object
    _launch: CapturedCodingLaunch = field(repr=False)

    @property
    def remaining_seconds(self):
        return self._launch.remaining_seconds


@dataclass(frozen=True, slots=True)
class PreparedCandidateWorker:
    _token: object = field(repr=False)
    _launch: CapturedCodingLaunch = field(repr=False)
    snapshot: object = field(repr=False)
    prompt: object = field(repr=False)
    source_broker: object = field(repr=False)
    admission: CandidateDispatchAdmission = field(repr=False)
    handoff: object = field(repr=False)
    audit: object = field(repr=False)

    def __post_init__(self):
        _need(self._token is _TOKEN, "owner continuation preparation required")

    def start(self):
        launch = self._launch
        with _LOCK:
            _need(launch._state.phase == "prepared", "second coding worker already consumed")
            launch._state.phase = "starting"
        process, fd = None, None
        try:
            self.admission.revalidate(snapshot=self.snapshot, prompt=self.prompt,
                                      source_broker=self.source_broker)
            _need(self.handoff.provider_prompt == self.prompt.provider_prompt
                  and _sha(self.handoff.path.read_bytes()) == self.handoff.sha256,
                  "prepared candidate dispatch handoff changed")
            fd = _sealed_stdin(self.handoff.provider_prompt)
            timeout = math.floor(launch.remaining_seconds)
            _need(timeout >= 1, "insufficient remaining original coding deadline")
            settings = launch.launch_settings
            command = [*_PREFIX, "--semantic-repository", str(launch.source_broker.repository),
                "--purpose", "coding", "--timeout", str(timeout)]
            for key, value in settings.items():
                if value is not None:
                    command.extend(("--"+key.replace("_", "-"), str(value)))
            command.extend(self.source_broker.extra_argv)
            command.extend(self.handoff.extra_argv)
            self.admission.revalidate(snapshot=self.snapshot, prompt=self.prompt,
                                      source_broker=self.source_broker)
            process = _spawn_worker(command, stdin=fd, workspace=launch.source_broker.workspace)
            _need(type(process) is _NativePopen, "native candidate child required")
            self.source_broker.note_spawned(process)
            launch.remaining_seconds
            with _LOCK:
                launch._state.phase = "started"
            return RunningCandidateWorker(process, self.source_broker, launch)
        except BaseException:
            with _LOCK:
                launch._state.phase = "failed"
            numbers = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
            previous = {number: signal.getsignal(number) for number in numbers}
            try:
                for number in numbers:
                    signal.signal(number, signal.SIG_IGN)
                try:
                    _terminate(process)
                finally:
                    retire_coding_worker(launch=launch, process=process, source_broker=self.source_broker)
            except BaseException:
                # Cleanup errors cannot turn the original dispatch failure into
                # a success, or replace its exception with a secondary failure.
                pass
            finally:
                try:
                    for cleanup_action in (self.source_broker.close, self.handoff.close):
                        try:
                            cleanup_action()
                        except BaseException:
                            pass
                finally:
                    for number, handler in previous.items():
                        signal.signal(number, handler)
            raise
        finally:
            if fd is not None:
                os.close(fd)

    def close(self):
        """Retire an unused preparation; a running child needs actual cleanup."""
        with _LOCK:
            if self._launch._state.phase in ("preparing", "prepared"):
                self._launch._state.phase = "closed"
            process = self.source_broker._process
        if process is not None and process.poll() is None:
            raise ValueError("running candidate requires isolated owner retirement")
        if process is not None and self._launch._state.second_retirement is None:
            raise ValueError("waited candidate still requires isolated owner retirement")
        self.source_broker.close()
        self.handoff.close()


def prepare_second_coding_worker(*, launch, retired_worker):
    """Consume the only continuation claim and prepare a fresh candidate route."""
    from .owner_candidate_round_context import prepare_candidate_round_context
    from .owner_candidate_round_prompt import prepare_candidate_round_prompt
    from .worker_candidate_round_broker import WorkerCandidateRoundBroker
    from .candidate_dispatch_context import prepare_candidate_dispatch_handoff
    from .owner_candidate_dispatch_audit import persist_candidate_dispatch_audit
    _need(type(launch) is CapturedCodingLaunch and type(retired_worker) is RetiredCodingWorker,
          "live captured launch and fixed cleanup retirement required")
    launch.revalidate()
    _need(retired_worker._launch is launch and retired_worker.round_number == 1
          and retired_worker.revalidate() in range(1, 256),
          "positive first coding failure required")
    with _LOCK:
        _need(launch._state.phase == "captured", "original continuation claim already consumed")
        launch._state.phase = "preparing"
    broker = handoff = None
    try:
        snapshot = prepare_candidate_round_context(source_broker=launch.source_broker,
            prior_returncode=retired_worker.returncode, round_number=2,
            absolute_deadline=launch.absolute_deadline)
        prompt = prepare_candidate_round_prompt(snapshot=snapshot,
            requirements=launch._original_requirements)
        broker = WorkerCandidateRoundBroker(snapshot)
        launch._state.candidate_broker = broker
        admission = CandidateDispatchAdmission(_TOKEN, launch, retired_worker, snapshot, prompt,
            broker, canonical_json(_admission_binding(launch, snapshot, prompt, broker)))
        launch._state.admission = admission
        handoff = prepare_candidate_dispatch_handoff(snapshot=snapshot, prompt=prompt,
            source_broker=broker, admission=admission)
        admission.revalidate(snapshot=snapshot, prompt=prompt, source_broker=broker)
        audit = persist_candidate_dispatch_audit(snapshot=snapshot, prompt=prompt,
            source_broker=broker, admission=admission, handoff=handoff)
        admission.revalidate(snapshot=snapshot, prompt=prompt, source_broker=broker)
        launch._state.phase = "prepared"
        return PreparedCandidateWorker(_TOKEN, launch, snapshot, prompt, broker, admission, handoff, audit)
    except BaseException:
        launch._state.phase = "failed"
        if broker is not None:
            broker.close()
        if handoff is not None:
            handoff.close()
        raise


__all__ = ["CapturedCodingLaunch", "RetiredCodingWorker", "CandidateDispatchAdmission",
    "PreparedCandidateWorker", "RunningCandidateWorker", "capture_original_coding_launch",
    "retire_coding_worker", "prepare_second_coding_worker"]
