"""Run one host-owner candidate child through an independent native admission.

The caller supplies its already selected literal command and isolated cleanup
operation. This is not the deployed container-wrapper bridge or a router input
decoder. The original candidate envelope remains advisory outside that separate
route. Nothing here publishes a candidate, discharges an obligation or retries
the database provider callback.
"""
from __future__ import annotations

import subprocess
import time

from .native_candidate_round_admission import NativeCandidateRoundAdmission
from .worker_candidate_round_broker import WorkerCandidateRoundBroker


def _need(value, reason):
    if not value:
        raise ValueError(reason)


def _command(value):
    _need(type(value) in (list, tuple) and 1 <= len(value) <= 64
        and all(type(part) is str and 0 < len(part.encode()) <= 8192
            and "\0" not in part for part in value)
        and sum(len(part.encode()) for part in value) <= 65536,
        "bounded owner-selected literal candidate command required")
    return tuple(value)


def _environment(value):
    from .multi_supervisor_runner import DATABASE_PROGRAM_ENV_NAMES, STATE_CREDENTIAL_ENV_NAMES
    _need(type(value) is dict and len(value) <= 256
        and all(type(key) is str and type(item) is str and key
            and "=" not in key and "\0" not in key + item
            for key, item in value.items())
        and sum(len(key.encode()) + len(item.encode()) for key, item in value.items()) <= 131072,
        "bounded explicit candidate child environment required")
    forbidden = (*DATABASE_PROGRAM_ENV_NAMES, *STATE_CREDENTIAL_ENV_NAMES,
        "IPFS_ACCELERATE_AGENT_STATE_OWNER_TOKEN")
    _need(not any(name in value for name in forbidden),
        "candidate child environment contains native database authority")
    return dict(value)


def _stop_and_wait(process):
    """Reap this exact known Popen; isolated descendant cleanup is separate."""
    if subprocess.Popen.poll(process) is None:
        try:
            process.terminate()
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def run_native_candidate_round_worker(*, admission, source_broker, command,
        environment, cleanup, stdout=None, stderr=None):
    """Run once under the retained native claim and original absolute deadline.

    Stdin is an empty pipe until the task fence and birth registration finish.
    Only then are the exact candidate prompt bytes supplied. The caller must
    pass its existing descendant cleanup operation; an exact ``True`` return
    records its success. Output uses caller-selected streams so failed receipts
    can remain in the owner's ordinary logs, rather than an unbounded buffer.

    The native supervisor-run lease is independently checked by ``admission``;
    it is not a quantitative CPU/memory reservation or a container UID grant.
    """
    _need(type(admission) is NativeCandidateRoundAdmission,
        "exact native candidate admission required")
    admitted_broker = type(source_broker) is WorkerCandidateRoundBroker
    if not admitted_broker:
        from .candidate_dispatch_broker import CandidateDispatchBroker
        admitted_broker = (type(source_broker) is CandidateDispatchBroker
            and source_broker._admission is admission)
    _need(admitted_broker and source_broker._snapshot is admission.candidate_prompt._snapshot,
        "fresh candidate reader must belong to the admitted prompt")
    _need(admission.process is None and source_broker._process is None
        and not source_broker._registered.is_set() and not source_broker._closed,
        "unused candidate admission and reader required")
    _need(callable(cleanup), "explicit owner descendant cleanup operation required")
    _need(stdout != subprocess.PIPE and stderr != subprocess.PIPE,
        "candidate output requires owner streams rather than an unbounded pipe")
    literal = _command(command)
    environment = _environment(environment)
    prompt = admission.candidate_prompt
    attempt = admission.worker_attempt
    process = None
    waited = cleanup_succeeded = brokers_closed = False
    pending = None
    returncode = None
    try:
        admission.revalidate()
        source_broker._current_context()

        def spawn():
            nonlocal process
            process = subprocess.Popen(literal, cwd=source_broker.workspace,
                env=environment, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr,
                close_fds=True, start_new_session=True)
            return process

        process = admission.execute_spawn(spawn, source_broker.note_spawned)
        # The complete native fence has closed. The trusted command has not
        # received any task bytes; refuse late input drift before releasing it.
        source_broker._current_context()
        admission.revalidate()
        remaining = prompt._snapshot._absolute_deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("original candidate worker deadline exhausted")
        process.communicate(input=prompt.provider_prompt, timeout=remaining)
        waited = True
        returncode = process.returncode
    except BaseException as error:
        pending = error
    finally:
        # The local spawn callback retains its actual Popen before admission
        # postchecks can raise. Never clean another invocation's retained child.
        if process is not None:
            try:
                _stop_and_wait(process)
            except BaseException as error:
                if pending is None:
                    pending = error
            try:
                observed = cleanup(process)
                _need(type(observed) is bool, "exact boolean descendant cleanup result required")
                cleanup_succeeded = observed
            except BaseException as error:
                if pending is None:
                    pending = error
        if process is not None or admission.process is None:
            try:
                source_broker.close()
                _need(source_broker._closed and source_broker._stop.is_set()
                    and not source_broker._thread.is_alive()
                    and not any(path.exists() or path.is_symlink() for path in
                        (source_broker.descriptor_path, source_broker.socket_path)),
                    "candidate source reader did not fully retire")
                brokers_closed = True
            except BaseException as error:
                if pending is None:
                    pending = error
        if process is not None:
            try:
                attempt.finish(waited=waited, cleanup_succeeded=cleanup_succeeded,
                    brokers_closed=brokers_closed)
            except BaseException as error:
                if pending is None:
                    pending = error
        # With no retained Popen, preparation stays incomplete. A failed
        # process-creation call cannot prove that no child was ever created.
    if pending is not None:
        raise pending
    _need(type(returncode) is int, "candidate worker did not supply a waited outcome")
    return subprocess.CompletedProcess(literal, returncode if cleanup_succeeded else 1)


__all__ = ["run_native_candidate_round_worker"]
