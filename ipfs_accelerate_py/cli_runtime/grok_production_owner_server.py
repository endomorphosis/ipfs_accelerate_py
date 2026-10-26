"""Source-only synchronous live-owner bridge for a NEW production profile.

The actual-capability guard and exact-listener consumer are required. No root
lease, fixture authority, resource deserialization or operation on import.
This does not adopt the historical separate-owner launch topology.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import stat
import sys
import threading
import time
import types


class ProductionServerRefusal(RuntimeError):
    pass


def need(value, reason):
    if not value:
        raise ProductionServerRefusal(reason)


def _frame_deadline(work_deadline_monotonic):
    need(type(work_deadline_monotonic) in (int, float) and math.isfinite(work_deadline_monotonic),
         'production_owner_frame_deadline_invalid')
    now = time.monotonic()
    deadline = min(now+1.0, work_deadline_monotonic-3.0)
    need(deadline > now, 'production_owner_frame_absolute_deadline_expired')
    return deadline


def _remaining_frame_timeout(channel, deadline):
    remaining = deadline-time.monotonic()
    need(remaining > 0, 'production_owner_frame_absolute_deadline_expired')
    channel.settimeout(min(1.0, remaining))


def _receive_frame(core, channel, *, work_deadline_monotonic):
    """One bounded absolute frame, never an inactivity-renewed time grant."""
    deadline = _frame_deadline(work_deadline_monotonic)
    raw = bytearray()
    while b'\n' not in raw:
        _remaining_frame_timeout(channel, deadline)
        try:
            part = channel.recv(min(1024, core.MAX_FRAME+1-len(raw)))
        except TimeoutError:
            raise ProductionServerRefusal('production_owner_frame_absolute_deadline_expired') from None
        need(bool(part), 'production_owner_frame_unexpected_EOF')
        raw.extend(part)
        need(len(raw) <= core.MAX_FRAME, 'production_owner_frame_bound')
        need(time.monotonic() < deadline, 'production_owner_frame_absolute_deadline_expired')
    body, _, tail = raw.partition(b'\n')
    need(not tail, 'production_owner_multiple_frames_forbidden')
    value = core.decode(bytes(body))
    need(time.monotonic() < deadline, 'production_owner_frame_absolute_deadline_expired')
    return value


def _send_frame(core, channel, value, *, work_deadline_monotonic):
    # Python sendall uses the timeout for the complete call; recheck the exact
    # absolute phase boundary afterwards, including the original work margin.
    deadline = _frame_deadline(work_deadline_monotonic)
    _remaining_frame_timeout(channel, deadline)
    try:
        core.send(channel, value)
    except TimeoutError:
        raise ProductionServerRefusal('production_owner_frame_absolute_deadline_expired') from None
    need(time.monotonic() < deadline, 'production_owner_frame_absolute_deadline_expired')


def fresh_record(core, path, value, directory):
    need(type(path) is type(Path()) and path.is_absolute() and path.parent.resolve(strict=True) == path.parent and
         directory in path.parents and not os.path.lexists(path), 'production_owner_fresh_record_path_required')
    for parent in (path.parent, *path.parent.parents):
        need(stat.S_ISDIR(parent.lstat().st_mode), 'production_owner_record_parent_alias')
    metadata = path.parent.lstat()
    need(metadata.st_uid == os.getuid() and stat.S_IMODE(metadata.st_mode) == 0o700,
         'production_owner_private_record_parent_required')
    body = core.canonical(value)
    need(len(body) <= 8*1024**2, 'production_owner_record_bound')
    descriptor = os.open(path, os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW|os.O_CLOEXEC, 0o600)
    with os.fdopen(descriptor, 'wb') as stream:
        stream.write(body); stream.flush(); os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    metadata = path.lstat()
    need(stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1 and metadata.st_uid == os.getuid() and
         stat.S_IMODE(metadata.st_mode) == 0o600 and path.read_bytes() == body,
         'production_owner_durable_record_readback_failed')
    return {'path': str(path), 'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest()}


class ProductionOwnerServer:
    """Serve inside cancel_requested, on the same actual owner thread."""
    def __init__(self, *, core, factory_module, listener_module, owner, listener,
                 context_path, observation_directory):
        need(all(type(value) is types.ModuleType and sys.modules.get(value.__name__) is value and
                 value.__spec__.origin == value.__file__ for value in (core, factory_module, listener_module)),
             'production_owner_genuine_source_modules_required')
        need(type(owner) is factory_module.ExistingProductionOwner and
             type(listener) is listener_module.OwnedUnixListener and listener.disk is owner._disk and
             core is owner._core, 'production_owner_exact_existing_consumers_required')
        pins = dict(owner._pins)
        for path in (Path(__file__).absolute(), Path(core.__file__), Path(factory_module.__file__), Path(listener_module.__file__)):
            need(pins.get(str(path)) == factory_module.source_digest(path),
                 'production_owner_complete_server_source_pins_required')
        self.core, self.owner, self.listener = core, owner, listener
        self.context_path, self.observation_directory = context_path, observation_directory
        need(type(observation_directory) is type(Path()) and observation_directory.resolve(strict=True) == observation_directory and
             owner._disk.directory in observation_directory.parents and not list(observation_directory.iterdir()) and
             stat.S_IMODE(observation_directory.lstat().st_mode) == 0o700,
             'production_owner_fresh_private_observation_directory_required')
        need(type(context_path) is type(Path()) and context_path.parent.resolve(strict=True) == context_path.parent and
             owner._disk.directory in context_path.parents and not os.path.lexists(context_path),
             'production_owner_charged_private_context_path_required')
        self._pid, self._thread = os.getpid(), threading.get_ident()
        self._birth = core.process(self._pid)['birth']
        self._context_bytes = None
        self._sequence, self._rows = 0, []
        self._failure = None
        self.closed = False
        listener.check_usage()
        owner.current()

    def __reduce__(self):
        raise TypeError('production owner server cannot be serialized')

    def _lifetime(self):
        need(not self.closed and self._failure is None and os.getpid() == self._pid and
             threading.get_ident() == self._thread and self.core.process(self._pid)['birth'] == self._birth,
             'production_owner_server_lifetime_changed')

    def bind_child(self, process):
        """Call from existing on_started before the runner sends stdin."""
        self._lifetime()
        need(self._context_bytes is None, 'production_owner_one_actual_child_only')
        worker = self.owner.bind_child(process)
        self.listener.check_usage()
        self.owner.current()
        context = self.core.decode(self.owner._binding_bytes)
        context.update(request_sha256=self.owner._request_sha256,
            owner_pid=self._pid, owner_birth=self._birth,
            worker_pid=worker['pid'], worker_birth=worker['birth'], worker_process_group=worker['group_pid'],
            endpoint=str(self.listener.endpoint), socket=self.listener.witness(), nonce=os.urandom(24).hex())
        context['context_sha256'] = self.core.digest(context)
        context = self.core.validate_context(context)
        try:
            pin = fresh_record(self.core, self.context_path, context, self.owner._disk.directory)
            self.owner.current()
            self.listener.check_usage()
        except BaseException:
            self._failure = 'production_owner_context_publication_failed'
            raise
        self._context_bytes = self.core.canonical(context)
        self.context_pin = pin
        return dict(pin)

    def _serve_one(self):
        need(self._context_bytes is not None, 'production_owner_actual_child_not_bound')
        try:
            channel, _ = self.listener.listener.accept()
        except BlockingIOError:
            return False
        context = self.core.decode(self._context_bytes)
        with channel:
            peer = self.core.peer(channel)
            need(peer == self.owner._child and peer['parent_pid'] == self._pid,
                 'production_owner_actual_worker_kernel_peer_required')
            request = _receive_frame(self.core, channel, work_deadline_monotonic=context['work_deadline_monotonic'])
            need(type(request) is dict and set(request) == {'schema', 'sequence', 'stage', 'request_sha256',
                 'context_sha256', 'nonce', 'pid', 'birth', 'process_group'}, 'production_owner_request_fields')
            expected = {'schema': self.core.RPC_SCHEMA, 'sequence': self._sequence+1,
                'request_sha256': context['request_sha256'], 'context_sha256': context['context_sha256'],
                'nonce': context['nonce'], 'pid': peer['pid'], 'birth': peer['birth'], 'process_group': peer['group_pid']}
            need(self._sequence < self.core.MAX_REQUESTS and type(request['stage']) is str and
                 request['stage'] in self.core.STAGES and all(type(request[key]) is type(value) and
                 request[key] == value for key, value in expected.items()), 'production_owner_request_binding_or_replay')
            self.listener.check_usage()
            actual = self.owner.current()
            response = {'schema': self.core.RPC_SCHEMA, 'status': 'current', 'sequence': self._sequence+1,
                'stage': request['stage'], **{key: context[key] for key in ('request_sha256', 'context_sha256', 'nonce',
                    'owner_pid', 'owner_birth', 'worker_pid', 'worker_birth', 'worker_process_group',
                    'work_deadline_monotonic', 'native_deadline_monotonic')},
                'observed_monotonic': actual['observed_monotonic']}
            _send_frame(self.core, channel, response, work_deadline_monotonic=context['work_deadline_monotonic'])
            row = {'sequence': self._sequence+1, 'stage': request['stage'],
                'request_sha256': context['request_sha256'], 'context_sha256': context['context_sha256'],
                'response_sha256': self.core.digest(response), 'client_pid': peer['pid'], 'client_birth': peer['birth'],
                'client_process_group': peer['group_pid'], 'owner_pid': self._pid, 'owner_birth': self._birth,
                'observed_monotonic': actual['observed_monotonic'], 'source_bindings_sha256': actual['source_bindings_sha256'],
                'claim_binding': actual['claim_binding'], 'resource_binding': actual['resource_binding']}
            # Each successful ACK gets one distinct durable owner record. A
            # post-send persistence failure cancels and never claims success.
            fresh_record(self.core, self.observation_directory/f'{self._sequence+1:04d}.json', row, self.owner._disk.directory)
            self._rows.append(row)
            self._sequence += 1
            self.listener.check_usage()
            return True

    def poll_or_cancel(self):
        """Never attach this to observational on_progress or a background thread."""
        if self._failure is not None:
            return self._failure
        try:
            self._lifetime()
            self.owner.current()
            self.listener.check_usage()
            self._serve_one()
            return False
        except BaseException:
            self._failure = 'production_owner_currentness_refused'
            return self._failure

    def observations(self):
        expected = {f'{index:04d}.json' for index in range(1, len(self._rows)+1)}
        need({path.name for path in self.observation_directory.iterdir()} == expected,
             'production_owner_durable_observation_population_changed')
        for index, row in enumerate(self._rows, 1):
            path = self.observation_directory/f'{index:04d}.json'
            descriptor = os.open(path, os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC)
            try:
                metadata = os.fstat(descriptor)
                need(stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1 and
                     stat.S_IMODE(metadata.st_mode) == 0o600 and metadata.st_size <= self.core.MAX_FRAME,
                     'production_owner_durable_observation_custody_changed')
                body = os.read(descriptor, self.core.MAX_FRAME+1)
                need(body == self.core.canonical(row), 'production_owner_durable_observation_changed')
            finally:
                os.close(descriptor)
        return json.loads(self.core.canonical({'rows': self._rows}))['rows']

    def assert_result_eligible(self):
        """A late/native result cannot turn a latched service failure into success."""
        need(self._failure is None and self._context_bytes is not None and self._rows and
             self._rows[-1]['stage'] == 'worker_v4_response', 'production_owner_result_not_eligible')

    def close_after_group_drain(self):
        need(not self.closed and os.getpid() == self._pid and threading.get_ident() == self._thread,
             'production_owner_close_wrong_process_or_thread')
        # This genuine exact-listener consumer permits stopped cleanup after
        # revocation but requires actual absence of native consumers/group.
        self.listener.close_after_group_drain()
        need(not os.path.lexists(self.listener.endpoint), 'production_owner_socket_close_not_observed')
        self.closed = True
