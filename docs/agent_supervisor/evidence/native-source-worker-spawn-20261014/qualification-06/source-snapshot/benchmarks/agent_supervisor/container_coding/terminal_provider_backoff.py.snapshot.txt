"""Host-owned provider admission: defer new trials without replaying old work.

The default directory is shared by this OS user. Cooldowns are provider-wide,
not model-wide, so another model cannot bypass the observed provider stop.
One recovery probe holds an OS lock through its entire trial. No prompts,
credentials, token estimates, network calls, or provider retries live here.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field
import fcntl
import json
import math
import os
from pathlib import Path
import secrets
import stat
import time

SCHEMA = "terminal-provider-backoff@1"
MAX_HINT_SECONDS = 7 * 86400
PROVIDERS = ("codex_cli", "grok_cli")


class BackoffStateError(ValueError):
    """Unavailable state never grants dispatch permission."""


def default_directory() -> Path:
    base = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local/state")
    return base / "ipfs_accelerate_py/terminal-provider-backoff"


def _finite(value):
    return type(value) in (int, float) and 0 <= value < 10**15 and math.isfinite(value)


@dataclass
class Admission:
    allowed: bool
    provider: str
    reason: str
    generation: int
    retry_after_seconds: float | None = None
    recovery_probe: bool = False
    _probe_fd: int | None = field(default=None, repr=False)
    _settled: bool = field(default=False, repr=False)

    def observation(self):
        return {"schema": "terminal-provider-admission@1", "allowed": self.allowed,
                "provider": self.provider, "reason": self.reason,
                "generation": self.generation, "retry_after_seconds": self.retry_after_seconds,
                "recovery_probe": self.recovery_probe, "automatic_replay_allowed": False}

    def close(self):
        if self._probe_fd is not None:
            os.close(self._probe_fd)
            self._probe_fd = None


class ProviderBackoff:
    def __init__(self, directory: Path | None = None, *, wall_clock=time.time,
                 monotonic=time.monotonic, jitter=None, boot_id=None):
        self.directory = Path(directory if directory is not None else default_directory()).absolute()
        self.wall_clock, self.monotonic = wall_clock, monotonic
        self.jitter = jitter or (lambda: secrets.randbelow(1000000) / 1000000)
        self.boot_id = boot_id or Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        if not self.boot_id or len(self.boot_id) > 128:
            raise BackoffStateError("invalid_boot_identity")
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = self.directory.lstat()
        if (self.directory.resolve() != self.directory or not stat.S_ISDIR(info.st_mode)
                or info.st_uid != os.geteuid() or info.st_mode & 0o077):
            raise BackoffStateError("private_owned_state_directory_required")

    def _name(self, provider):
        if provider not in PROVIDERS:
            raise BackoffStateError("unsupported_provider")
        return self.directory / provider

    def _open(self, path, flags):
        fd = os.open(path, flags | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, 0o600)
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or info.st_uid != os.geteuid() or info.st_mode & 0o077):
            os.close(fd)
            raise BackoffStateError("private_owned_state_file_required")
        return fd

    @contextmanager
    def _locked(self, provider, *, wait=False, marker=False):
        suffix = '.marker-lock' if marker else '.lock'
        fd = self._open(self._name(provider).with_suffix(suffix), os.O_CREAT | os.O_RDWR)
        try:
            until = time.monotonic() + (1 if wait else 0)
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError as exc:
                    if time.monotonic() >= until:
                        raise BackoffStateError("state_busy") from exc
                    time.sleep(.01)
            yield
        finally:
            os.close(fd)

    def _empty(self, provider):
        return dict(schema=SCHEMA, provider=provider, generation=0, failures=0,
                    not_before_unix=0, not_before_monotonic=0, boot_id=self.boot_id,
                    recovery_required=False, probe_pending=False, blocked=False, reason="ready")

    def _read(self, provider):
        path = self._name(provider).with_suffix('.json')
        try:
            fd = self._open(path, os.O_RDONLY)
        except FileNotFoundError:
            return self._empty(provider)
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise BackoffStateError("duplicate_state_key")
                result[key] = value
            return result
        with os.fdopen(fd, 'rb') as stream:
            raw = stream.read(8193)
        try:
            if len(raw) > 8192:
                raise BackoffStateError("oversized_state")
            value = json.loads(raw, object_pairs_hook=unique)
            if (type(value) is not dict or set(value) != set(self._empty(provider))
                    or value['schema'] != SCHEMA or value['provider'] != provider
                    or type(value['generation']) is not int or not 0 <= value['generation'] < 2**63
                    or type(value['failures']) is not int or not 0 <= value['failures'] <= 63
                    or any(type(value[k]) is not bool for k in ('blocked', 'probe_pending', 'recovery_required'))
                    or any(not _finite(value[k]) for k in ('not_before_unix', 'not_before_monotonic'))
                    or type(value['boot_id']) is not str or not 0 < len(value['boot_id']) <= 128
                    or value['reason'] not in {'ready', 'rate_limit', 'billing', 'authentication', 'quota',
                        'invalid_provider_stop', 'provider_delay_unbounded', 'recovery_unverified',
                        'provider_outcome_unknown'}
                    or (not value['recovery_required'] and (value['blocked'] or value['probe_pending']))):
                raise BackoffStateError("invalid_state")
        except (TypeError, ValueError, KeyError, UnicodeError) as exc:
            raise BackoffStateError("invalid_state") from exc
        return value

    def _write(self, provider, value):
        target = self._name(provider).with_suffix('.json')
        temporary = self.directory / ('.state-' + secrets.token_hex(16))
        fd = self._open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        try:
            with os.fdopen(fd, 'w') as stream:
                json.dump(value, stream, sort_keys=True, allow_nan=False)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
            self._sync_directory()
        finally:
            temporary.unlink(missing_ok=True)

    def _sync_directory(self):
        fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _remaining(self, value):
        if value['boot_id'] == self.boot_id:
            return max(0., value['not_before_monotonic'] - self.monotonic())
        return max(0., value['not_before_unix'] - self.wall_clock())

    def acquire(self, provider) -> Admission:
        """Reserve at most one probe after cooldown; never wait or call a provider."""
        with self._locked(provider):
            state = self._read(provider)
            def denied(reason, delay=None):
                return Admission(False, provider, reason, state['generation'], delay)
            marker = self._name(provider).with_suffix('.unsettled')
            try:
                fd = self._open(marker, os.O_RDONLY)
            except FileNotFoundError:
                pass
            else:
                os.close(fd)
                return denied('settlement_unavailable')
            if state['blocked']:
                return denied(state['reason'])
            if not state['recovery_required']:
                return Admission(True, provider, 'ready', state['generation'])
            remaining = self._remaining(state)
            if remaining > 0:
                return denied('cooldown', remaining)
            fd = self._open(self._name(provider).with_suffix('.probe'), os.O_CREAT | os.O_RDWR)
            try:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return denied('recovery_probe_in_progress')
                if state['probe_pending']:
                    return denied('unsettled_recovery_probe')
                state['probe_pending'] = True
                state['generation'] += 1
                self._write(provider, state)
                permit = Admission(True, provider, 'recovery_probe', state['generation'],
                                   recovery_probe=True, _probe_fd=fd)
                fd = None
                return permit
            finally:
                if fd is not None:
                    os.close(fd)

    def acquire_with_wait(self, provider, *, max_wait_seconds=0, sleeper=time.sleep,
                          cancel_event=None):
        """Optionally wait before dispatch, bounded by one monotonic deadline."""
        if not _finite(max_wait_seconds) or max_wait_seconds > 60:
            raise ValueError('provider wait must be 0..60 seconds')
        deadline = self.monotonic() + max_wait_seconds
        while True:
            if cancel_event is not None and cancel_event.is_set():
                return Admission(False, provider, 'cancelled_before_dispatch', 0)
            permit = self.acquire(provider)
            if permit.allowed or permit.reason != 'cooldown':
                return permit
            remaining = deadline - self.monotonic()
            if remaining <= 0 or permit.retry_after_seconds > remaining:
                return permit
            # No state lock is held during a cancellable, bounded wait.
            sleeper(min(1., permit.retry_after_seconds, remaining))

    def settle(self, permit: Admission, *, stop=None, provider_succeeded=False, outcome_unknown=False):
        try:
            return self._settle(permit, stop=stop, provider_succeeded=provider_succeeded,
                                outcome_unknown=outcome_unknown)
        except (BackoffStateError, OSError, ValueError):
            # A failed persistence operation must not silently lose the stop.
            self.mark_unsettled(permit)
            raise

    def mark_unsettled(self, permit: Admission):
        """Persist uncertainty after dispatch or failed settlement; never retry."""
        # Independent of the state lock so a failed state commit still blocks.
        # Reset holds state then marker; this path never takes the state lock.
        with self._locked(permit.provider, wait=True, marker=True):
            marker = self._name(permit.provider).with_suffix('.unsettled')
            try:
                fd = self._open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            except FileExistsError:
                fd = self._open(marker, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            self._sync_directory()

    def _settle(self, permit: Admission, *, stop=None, provider_succeeded=False, outcome_unknown=False):
        """Settle an admitted trial once; stale success cannot erase a newer stop."""
        if not permit.allowed or permit._settled:
            raise BackoffStateError("unsettleable_admission")
        if permit.recovery_probe and permit._probe_fd is None:
            raise BackoffStateError("recovery_probe_custody_lost")
        with self._locked(permit.provider, wait=True):
            state = self._read(permit.provider)
            if stop is not None:
                reason = stop.get('reason_code') if type(stop) is dict else None
                hint = stop.get('retry_after_seconds') if type(stop) is dict else None
                valid = (type(stop) is dict and stop.get('observation_valid') is True
                         and stop.get('provider') == permit.provider
                         and reason in {'rate_limit', 'billing', 'authentication', 'quota'}
                         and (hint is None or (_finite(hint) and 0 < hint <= MAX_HINT_SECONDS))
                         and type(stop.get('retry_after_unbounded', False)) is bool)
                blocked = (not valid or reason != 'rate_limit' or stop.get('retry_after_unbounded') is True)
                state['failures'] = min(63, state['failures'] + 1)
                jitter = self.jitter()
                if not _finite(jitter) or jitter >= 1:
                    raise BackoffStateError("invalid_jitter")
                base = min(3600, 60 * 2**min(state['failures'] - 1, 6))
                delay = max(min(3600, base + min(60, base * .2) * jitter), hint or 0) if valid else 0
                # Concurrent stops may extend, never shorten, the current cooldown.
                delay = max(delay, self._remaining(state))
                # A later transient failure cannot erase an earlier manual block.
                previous_block_reason = state['reason'] if state['blocked'] else None
                state.update(generation=state['generation'] + 1, blocked=blocked,
                    recovery_required=True, probe_pending=False, boot_id=self.boot_id,
                    not_before_unix=self.wall_clock() + delay,
                    not_before_monotonic=self.monotonic() + delay,
                    reason=('invalid_provider_stop' if not valid else
                            'provider_delay_unbounded' if stop.get('retry_after_unbounded') is True else reason))
                if previous_block_reason is not None:
                    state.update(blocked=True, reason=previous_block_reason)
                self._write(permit.provider, state)
            elif outcome_unknown is True:
                state.update(generation=state['generation'] + 1, blocked=True,
                             recovery_required=True, probe_pending=False,
                             reason=state['reason'] if state['blocked'] else 'provider_outcome_unknown')
                self._write(permit.provider, state)
            elif permit.recovery_probe and permit.generation == state['generation']:
                if provider_succeeded is True:
                    cleared = self._empty(permit.provider)
                    cleared['generation'] = state['generation'] + 1
                    state = cleared
                else:
                    state.update(generation=state['generation'] + 1, blocked=True,
                                 probe_pending=False, reason='recovery_unverified')
                self._write(permit.provider, state)
            permit._settled = True
            return {"schema": SCHEMA, "provider": permit.provider, "reason": state['reason'],
                    "generation": state['generation'], "blocked": state['blocked'],
                    "recovery_required": state['recovery_required'],
                    "retry_after_seconds": None if state['blocked'] else self._remaining(state),
                    "automatic_replay_allowed": False}

    def abandon_before_dispatch(self, permit):
        """Release a validation-only probe; caller must have launched no work."""
        if not permit.allowed or permit._settled:
            raise BackoffStateError('unsettleable_admission')
        if permit.recovery_probe:
            if permit._probe_fd is None:
                raise BackoffStateError('recovery_probe_custody_lost')
            with self._locked(permit.provider, wait=True):
                state = self._read(permit.provider)
                if state['generation'] == permit.generation:
                    state.update(probe_pending=False, generation=state['generation'] + 1)
                    self._write(permit.provider, state)
        permit._settled = True

    def reset(self, provider):
        """Explicit operator action after availability repair; never replays trials."""
        with self._locked(provider, wait=True), self._locked(provider, wait=True, marker=True):
            current = self._read(provider)
            fd = self._open(self._name(provider).with_suffix('.probe'), os.O_CREAT | os.O_RDWR)
            try:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise BackoffStateError('recovery_probe_in_progress') from exc
                value = self._empty(provider)
                value['generation'] = current['generation'] + 1
                self._write(provider, value)
                self._name(provider).with_suffix('.unsettled').unlink(missing_ok=True)
                self._sync_directory()
            finally:
                os.close(fd)
        return {'schema': SCHEMA, 'provider': provider, 'reason': 'operator_reset',
                'generation': value['generation'], 'automatic_replay_allowed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('reset',))
    parser.add_argument('--provider', required=True, choices=PROVIDERS)
    parser.add_argument('--directory', type=Path)
    args = parser.parse_args()
    print(json.dumps(ProviderBackoff(args.directory).reset(args.provider), sort_keys=True))


if __name__ == '__main__':
    main()
