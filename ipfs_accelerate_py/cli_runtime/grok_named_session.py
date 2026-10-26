"""Finite, owner-bound ACP naming before a Grok planning invocation.

Source candidate, not production qualification. The caller must already own
source, native, credential, process and charged-storage custody. This module
does not acquire authority, reconstruct native session files, or retry a task.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import selectors
import stat
import subprocess
import sys
import threading
import time
from typing import Callable, Mapping
import uuid

MODEL = "grok-4.7"
TITLE = "Legal IR planning"
FRAME_LIMIT = 256 * 1024
STDOUT_LIMIT = 2 * 1024 * 1024
STDERR_LIMIT = 1024 * 1024
CONFIG_LIMIT = 1024 * 1024
METHODS = ("initialize", "session/new", "_x.ai/session/rename", "_x.ai/session/close")


class NamedSessionRefusal(RuntimeError):
    """Closed reason code; never include native output or credentials."""


class NamedSessionCaptureRefusal(NamedSessionRefusal):
    """A response exists, but durable allocation capture is unconfirmed."""
    side_effects_started = True
    retryable = False


def _refuse(code):
    raise NamedSessionRefusal(code)


def _signature(value):
    return (value.st_dev, value.st_ino, value.st_mode, value.st_uid,
            value.st_gid, value.st_nlink, value.st_size, value.st_mtime_ns,
            value.st_ctime_ns)


@dataclass(frozen=True)
class SourcePin:
    path: Path
    sha256: str
    maximum_bytes: int = 2 * 1024 * 1024

    def verify(self):
        """Stream a native binary pin without retaining its entire ELF."""
        if (type(self.path) is not type(Path()) or not self.path.is_absolute()
                or self.path.resolve(strict=True) != self.path
                or type(self.maximum_bytes) is not int or not 1 <= self.maximum_bytes <= 256*1024*1024
                or type(self.sha256) is not str or len(self.sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.sha256)):
            _refuse("source_pin_invalid")
        fd = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or before.st_size > self.maximum_bytes:
                _refuse("source_pin_size_or_type")
            digest, count = hashlib.sha256(), 0
            while True:
                raw = os.read(fd, 128*1024)
                if not raw:
                    break
                count += len(raw)
                if count > self.maximum_bytes:
                    _refuse("source_pin_size_or_type")
                digest.update(raw)
            if (count != before.st_size or digest.hexdigest() != self.sha256
                    or _signature(before) != _signature(os.fstat(fd))
                    or _signature(before) != _signature(self.path.stat(follow_symlinks=False))):
                _refuse("source_pin_changed")
            return _signature(before)
        finally:
            os.close(fd)

    def read(self):
        if (type(self.path) is not type(Path()) or not self.path.is_absolute()
                or self.path.resolve(strict=True) != self.path
                or type(self.sha256) is not str or len(self.sha256) != 64
                or type(self.maximum_bytes) is not int or not 1 <= self.maximum_bytes <= 2*1024*1024
                or any(c not in "0123456789abcdef" for c in self.sha256)):
            _refuse("source_pin_invalid")
        fd = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or before.st_size > self.maximum_bytes:
                _refuse("source_pin_size_or_type")
            raw = os.read(fd, self.maximum_bytes + 1)
            if (len(raw) != before.st_size or _signature(before) != _signature(os.fstat(fd))
                    or _signature(before) != _signature(self.path.stat(follow_symlinks=False))
                    or hashlib.sha256(raw).hexdigest() != self.sha256):
                _refuse("source_pin_changed")
            return raw, _signature(before)
        finally:
            os.close(fd)


def _owned_directory(path):
    if (type(path) is not type(Path()) or not path.is_absolute()
            or path.resolve(strict=True) != path):
        _refuse("owned_directory_origin")
    value = path.stat(follow_symlinks=False)
    if not stat.S_ISDIR(value.st_mode) or value.st_uid != os.getuid() or value.st_mode & 0o022:
        _refuse("owned_directory_permissions")
    return (value.st_dev, value.st_ino, value.st_uid, value.st_gid, value.st_mode)


def _process_identity(pid):
    raw = Path(f"/proc/{pid}/stat").read_bytes()
    end = raw.rfind(b")")
    fields = raw[end + 2:].split()
    if len(raw) > 65536 or end < 0 or len(fields) < 20:
        _refuse("process_identity_invalid")
    return int(fields[2]), int(fields[19])


def _birth(pid):
    return _process_identity(pid)[1]


def _group_births():
    group = os.getpgrp()
    members = {}
    for entry in os.scandir("/proc"):
        if not entry.name.isdecimal():
            continue
        pid = int(entry.name)
        try:
            if os.getpgid(pid) == group:
                observed_group, start_ticks = _process_identity(pid)
                if observed_group != group:
                    _refuse("process_group_identity_changed")
                members[str(pid)] = start_ticks
        except (ProcessLookupError, FileNotFoundError):
            continue
    if str(os.getpid()) not in members:
        _refuse("own_process_birth_missing")
    return {"process_group": group, "members_pid_start_ticks": members}


def _object(raw):
    if len(raw) > FRAME_LIMIT:
        _refuse("bootstrap_frame_byte_bound")
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                _refuse("bootstrap_duplicate_json_key")
            value[key] = item
        return value
    def constant(_value):
        _refuse("bootstrap_nonfinite_json")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique,
                           parse_constant=constant)
    except (ValueError, UnicodeError, RecursionError):
        _refuse("bootstrap_json_invalid")
    if type(value) is not dict:
        _refuse("bootstrap_json_object_required")
    return value


def _ack(frame, expected_id):
    if (frame.get("jsonrpc") != "2.0" or type(frame.get("id")) is not int
            or frame["id"] != expected_id or "method" in frame
            or "error" in frame or type(frame.get("result")) is not dict):
        _refuse("bootstrap_rpc_error_or_shape")
    return frame["result"]


def _validate_result(sequence, result, session_id):
    if sequence == 1 and (type(result.get("protocolVersion")) is not int or result["protocolVersion"] != 1):
        _refuse("bootstrap_initialize_version_mismatch")
    if sequence == 2 and result.get("sessionId") != session_id:
        _refuse("bootstrap_created_session_uuid_mismatch")
    if sequence == 3 and result.get("success") is not True:
        _refuse("bootstrap_rename_not_acknowledged")
    if sequence == 4 and (result.get("error") is not None
            or type(result.get("result")) is not dict
            or result["result"].get("success") is not True
            or result["result"].get("outcome") != "closed"):
        _refuse("bootstrap_close_not_acknowledged")


@dataclass(frozen=True)
class NativeHomePreparation:
    """Separate fresh initialization home; only normalized config may leave it."""
    home: Path
    seed_configuration: SourcePin


@dataclass(repr=False)
class NamedSessionRequest:
    """Nonserializable owner handoff. A callback is a witness, not a grant.

    The reviewed native owner supplies its live currentness/source/disk/child
    guard, pinned callback source and an absolute deadline in this process's
    monotonic clock. No dictionary, evidence span or CLI flag can select it.
    A future owner/source/acceptance generation must bind this exact contract.
    """
    workspace: Path
    native_home: Path
    audit_directory: Path
    binary: SourcePin
    configuration: SourcePin
    owner_sampling_policy: SourcePin
    preparation: NativeHomePreparation
    helper_source: SourcePin
    owner_source: SourcePin
    owner_guard: Callable[[str], None] = field(repr=False)
    source_generation: str
    deadline_monotonic: float
    allocation_session_id: str
    credential_slot: str
    credential_fingerprint: str
    system_requirements: SourcePin | None = None
    cleanup_reserve_seconds: float = 3.0
    bootstrap_max_seconds: float = 30.0
    _pid: int = field(init=False, repr=False)
    _birth: int = field(init=False, repr=False)
    _group: int = field(init=False, repr=False)
    _thread: int = field(init=False, repr=False)
    _phase: str = field(default="prepared", init=False)
    _session: str = field(default="", init=False, repr=False)
    _receipt: dict = field(default_factory=dict, init=False, repr=False)
    _directory_pins: tuple = field(init=False, repr=False)
    _binary_identity: tuple = field(init=False, repr=False)
    _target_configuration: SourcePin | None = field(default=None, init=False, repr=False)
    _preparation_receipt: dict = field(default_factory=dict, init=False, repr=False)
    _sealed: bool = field(default=False, init=False, repr=False)

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False) and not name.startswith("_"):
            raise AttributeError("owner handoff inputs are immutable")
        object.__setattr__(self, name, value)

    def __post_init__(self):
        self._pid, self._group, self._thread = os.getpid(), os.getpgrp(), threading.get_ident()
        self._birth = _birth(self._pid)
        if (any(type(pin) is not SourcePin for pin in (self.binary, self.configuration,
                self.owner_sampling_policy, self.helper_source, self.owner_source))
                or self.system_requirements is not None and type(self.system_requirements) is not SourcePin
                or self.configuration.maximum_bytes > CONFIG_LIMIT
                or self.owner_sampling_policy.maximum_bytes > CONFIG_LIMIT
                or type(self.preparation) is not NativeHomePreparation
                or self.preparation.seed_configuration != self.configuration):
            _refuse("owner_source_pin_types_or_bounds")
        if self.helper_source.path != Path(__file__).resolve(strict=True):
            _refuse("helper_source_origin")
        self.helper_source.read()
        if (type(self.deadline_monotonic) not in (int, float)
                or not math.isfinite(self.deadline_monotonic)
                or type(self.source_generation) is not str
                or not self.source_generation.startswith("grok-automatic-named-session/")
                or len(self.source_generation) > 128
                or type(self.credential_slot) is not str or not 1 <= len(self.credential_slot) <= 64
                or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for c in self.credential_slot)
                or type(self.credential_fingerprint) is not str or len(self.credential_fingerprint) != 16
                or any(c not in "0123456789abcdef" for c in self.credential_fingerprint)
                or not 1 <= self.cleanup_reserve_seconds <= 10
                or not 1 <= self.bootstrap_max_seconds <= 30):
            _refuse("owner_deadline_or_generation_invalid")
        if (type(self.allocation_session_id) is not str
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", self.allocation_session_id) is None):
            _refuse("owner_allocation_session_invalid")
        if (not inspect.isfunction(self.owner_guard)
                or Path(self.owner_guard.__code__.co_filename).resolve(strict=True) != self.owner_source.path):
            _refuse("owner_guard_source_origin")
        self.owner_source.read()
        self._binary_identity = self.binary.verify()
        if not os.access(self.binary.path, os.X_OK):
            _refuse("native_binary_not_executable")
        if any(pin.path.is_relative_to(home) for pin in
               (self.configuration, self.owner_sampling_policy) for home in
               (self.native_home, self.preparation.home)):
            _refuse("private_configuration_origin")
        for pin in (self.configuration, self.owner_sampling_policy):
            metadata = pin.path.stat(follow_symlinks=False)
            if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid()
                    or stat.S_IMODE(metadata.st_mode) != 0o600 or metadata.st_nlink != 1):
                _refuse("private_configuration_permissions")
        self._directory_pins = tuple(_owned_directory(p) for p in
            (self.workspace, self.native_home, self.audit_directory, self.preparation.home))
        if any(stat.S_IMODE(p.stat().st_mode) != 0o700 for p in
               (self.native_home, self.audit_directory, self.preparation.home)):
            _refuse("private_directory_permissions")
        homes = (self.workspace, self.native_home, self.audit_directory, self.preparation.home)
        if any(left == right or left.is_relative_to(right) or right.is_relative_to(left)
               for i, left in enumerate(homes) for right in homes[i+1:]):
            _refuse("owned_directories_overlap")
        if list(self.audit_directory.iterdir()):
            _refuse("bootstrap_audit_not_fresh")
        if list(self.native_home.iterdir()) or list(self.preparation.home.iterdir()):
            _refuse("native_home_not_fresh")
        self.assert_current("request_prepared")
        self._sealed = True

    def __repr__(self):
        return f"NamedSessionRequest(phase={self._phase!r}, credentials=<redacted>)"

    def __reduce__(self):
        raise TypeError("named session owner handoff cannot be serialized")

    def assert_current(self, stage):
        if (self._pid != os.getpid() or self._birth != _birth(os.getpid())
                or self._group != os.getpgrp() or self._thread != threading.get_ident()
                or self._phase in {"closed", "failed"}):
            _refuse("owner_lifetime_or_identity_changed")
        if time.monotonic() + self.cleanup_reserve_seconds >= self.deadline_monotonic:
            _refuse("owner_absolute_deadline")
        if self._directory_pins != tuple(_owned_directory(p) for p in
                (self.workspace, self.native_home, self.audit_directory, self.preparation.home)):
            _refuse("owned_directory_changed")
        self.owner_source.read()
        self.helper_source.read()
        if _signature(self.binary.path.stat(follow_symlinks=False)) != self._binary_identity:
            _refuse("native_binary_changed")
        self.configuration.read()
        self.owner_sampling_policy.read()
        if self._target_configuration is not None:
            self._target_configuration.read()
        elif list(self.native_home.iterdir()):
            _refuse("target_home_changed_before_preparation")
        _native_policy_inputs_current(self)
        if self.system_requirements is not None:
            self.system_requirements.read()
        if self.owner_guard(stage) is not None:
            _refuse("owner_guard_contract")

    def remaining_seconds(self, upper_bound):
        self.assert_current("remaining_deadline")
        value = min(float(upper_bound), self.deadline_monotonic - time.monotonic() - self.cleanup_reserve_seconds)
        if not math.isfinite(value) or value <= 0:
            _refuse("owner_absolute_deadline")
        return value

    @property
    def completed_session_id(self):
        return self._session if self._phase == "response_accepted" else ""

    def receipt(self):
        report = dict(self._receipt)
        report["allocation_session_id"] = self.allocation_session_id
        report["native_home_preparation"] = self._preparation_receipt
        return json.loads(json.dumps(report, allow_nan=False))

    def accept_response(self, value):
        if self._phase != "headless_closed" or type(value) is not str or not value.strip():
            _refuse("named_session_response_not_usable")
        self.assert_current("response_acceptance")
        self._phase = "response_accepted"
        self._receipt["response_accepted"] = True
        return value


def validate_preparation_option(provider, request, options):
    if type(request) is not NamedSessionRequest or provider != "grok_cli":
        _refuse("named_session_requires_exact_grok_owner_request")
    if (any(name in options and options[name] is not None and options[name] != ""
            for name in ("resume_session_id", "chat_session_id", "session_id"))
            or "continue_session" in options and options["continue_session"] is not False):
        _refuse("named_session_conflicting_native_selector")
    request.assert_current("allocation_binding")


def validate_allocation_session(request, allocation_session_id):
    """Check the owner identity exactly, without aliases or allocation writes."""
    if type(request) is not NamedSessionRequest:
        _refuse("named_session_requires_exact_grok_owner_request")
    if (type(allocation_session_id) is not str
            or allocation_session_id != request.allocation_session_id):
        _refuse("named_session_allocation_identity_mismatch")


def validate_public_dispatch(*, provider, model, request, options,
                             allow_local_fallback, allow_cross_provider_fallback,
                             allocation_session_id=None):
    """Fail before dependency, cache, allocation or provider selection work."""
    temperature = options.get("temperature")
    if (provider != "grok_cli" or model != MODEL
            or options.get("side_effecting") is not True
            or allow_local_fallback is not False or allow_cross_provider_fallback is not False
            or type(temperature) not in (int, float) or not math.isfinite(temperature)
            or temperature != 0):
        _refuse("named_session_exact_public_dispatch_policy")
    validate_preparation_option(provider, request, options)
    validate_allocation_session(request, allocation_session_id)
    if any(name in options for name in ("api_key", "_api_key_slot", "_api_key_fingerprint")):
        _refuse("named_session_inline_credential_override")


def _private_file(path):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    return os.fdopen(fd, "wb", buffering=0)


def _validate_sampling_profile(request, environment):
    """Require the explicit fixed profile, rather than interpreting overlays.

    Configuration is owner-prepared and hash-bound, never rewritten here. The
    existing selected-model-only helper is insufficient for ancillary calls.
    This new combined profile still requires a genuine automatic-path probe.
    """
    import toml
    for key in ("GROK_CONFIG", "GROK_CONFIG_PATH"):
        if environment.get(key):
            _refuse("named_session_config_overlay_not_qualified")
    if environment.get("GROK_TITLE_REFRESH") not in (None, "", "0", "false"):
        _refuse("named_session_title_environment_conflict")
    selected = request._target_configuration or request.configuration
    configuration = toml.loads(selected.read()[0].decode("utf-8"))
    # This is owner policy, not a native signed requirements cache.
    requirements = toml.loads(request.owner_sampling_policy.read()[0].decode("utf-8"))
    policies = [requirements]
    system = Path("/etc/grok/requirements.toml")
    if system.exists():
        if request.system_requirements is None or request.system_requirements.path != system:
            _refuse("named_session_system_requirements_not_bound")
        policies.append(toml.loads(request.system_requirements.read()[0].decode("utf-8")))
    elif request.system_requirements is not None:
        _refuse("named_session_system_requirements_disappeared")
    for document in (configuration, *policies):
        if any(document.get(key) for key in ("experiments", "version_overrides", "campaigns")):
            _refuse("named_session_config_patches_not_qualified")
    if (configuration.get("cli", {}).get("use_leader") is not False
            or configuration.get("features", {}).get("title_refresh") is not False
            or configuration.get("models", {}).get("session_summary") != MODEL
            or configuration.get("models", {}).get("temperature") != 0
            or type(configuration.get("models", {}).get("temperature")) not in (int, float)
            or configuration.get("model", {}).get(MODEL, {}).get("temperature") != 0
            or type(configuration.get("model", {}).get(MODEL, {}).get("temperature")) not in (int, float)
            or requirements.get("features", {}).get("title_refresh") is not False):
        _refuse("named_session_sampling_profile_missing")
    for vendor in ("claude", "cursor", "codex"):
        table = configuration.get("compat", {}).get(vendor, {})
        if any(table.get(name) is not False for name in
               ("skills", "rules", "agents", "mcps", "hooks", "sessions")):
            _refuse("named_session_startup_surfaces_not_disabled")
    # Configured credentials would add an unproved precedence relation with
    # the allocation-selected key. This opt-in profile uses that key only.
    if any("api_key" in value for value in configuration.get("model", {}).values() if type(value) is dict):
        _refuse("named_session_inline_credential_not_qualified")
    if configuration.get("api_key") or requirements.get("api_key"):
        _refuse("named_session_inline_credential_not_qualified")
    for policy in policies:
        allowed = {"features": {"title_refresh"}, "cli": {"use_leader"},
                   "models": {"temperature", "session_summary"}, "model": {MODEL}}
        if (set(policy) - set(allowed)
                or any(type(table) is not dict or set(table)-allowed[name]
                       for name, table in policy.items())
                or type(policy.get("model", {}).get(MODEL, {})) is not dict
                or set(policy.get("model", {}).get(MODEL, {})) - {"temperature"}):
            _refuse("named_session_requirements_profile_not_qualified")
        if (policy.get("cli", {}).get("use_leader", False) is not False
                or policy.get("features", {}).get("title_refresh", False) is not False
                or policy.get("models", {}).get("session_summary", MODEL) != MODEL
                or policy.get("models", {}).get("temperature", 0) != 0
                or type(policy.get("models", {}).get("temperature", 0)) not in (int, float)
                or policy.get("model", {}).get(MODEL, {}).get("temperature", 0) != 0
                or type(policy.get("model", {}).get(MODEL, {}).get("temperature", 0)) not in (int, float)):
            _refuse("named_session_requirements_conflict")
    # System requirements cannot be silently discarded or weakened. The
    # reviewed router checks their sampling constraints before this function;
    # this candidate additionally refuses an ambient managed config overlay.
    if Path("/etc/grok/managed_config.toml").exists():
        _refuse("named_session_managed_config_not_qualified")


def _native_policy_inputs_current(request):
    """Bind real system input or absence; never manufacture a signed cache."""
    system = Path("/etc/grok/requirements.toml")
    if os.path.lexists(system):
        if request.system_requirements is None or request.system_requirements.path != system:
            _refuse("named_session_system_requirements_not_bound")
        request.system_requirements.read()
    elif request.system_requirements is not None:
        _refuse("named_session_system_requirements_disappeared")
    if os.path.lexists(Path("/etc/grok/managed_config.toml")):
        _refuse("named_session_managed_config_not_qualified")
    # Native may own these outputs. This first strict profile cannot qualify
    # a subsequently active signed policy, so output presence refuses rather
    # than being parsed as owner policy or silently disregarded.
    for home in (request.native_home, request.preparation.home):
        for name in ("requirements.toml", "managed_config.toml"):
            if os.path.lexists(home / name):
                _refuse("named_session_native_policy_output_not_qualified")


def _normalized_configuration(seed, normalized):
    """Permit only the native initialization marker observed in actual CLI."""
    import toml
    before = toml.loads(seed.decode("utf-8"))
    after = toml.loads(normalized.decode("utf-8"))
    expected = json.loads(json.dumps(before, allow_nan=False))
    market = after.get("marketplace", {})
    if type(market) is not dict:
        _refuse("native_initialization_configuration_changed")
    marker = "default_skills_installs_purged"
    if marker in market:
        if market[marker] is not True:
            _refuse("native_initialization_configuration_changed")
        expected.setdefault("marketplace", {})[marker] = True
    if json.dumps(after, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        _refuse("native_initialization_configuration_changed")
    return normalized


def _read_normalized_configuration(path):
    """Read bounded private output after the native group has fully closed."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid()
                or stat.S_IMODE(before.st_mode) != 0o600 or before.st_nlink != 1
                or before.st_size > CONFIG_LIMIT):
            _refuse("native_initialization_configuration_permissions")
        raw = os.read(fd, CONFIG_LIMIT+1)
        if (len(raw) != before.st_size or _signature(before) != _signature(os.fstat(fd))
                or _signature(before) != _signature(path.stat(follow_symlinks=False))):
            _refuse("native_initialization_configuration_unstable")
        return raw
    finally:
        os.close(fd)


def _durable_configuration(path, raw):
    with _private_file(path) as target:
        if target.write(raw) != len(raw):
            _refuse("native_configuration_write_incomplete")
        os.fsync(target.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    selected = SourcePin(path, hashlib.sha256(raw).hexdigest(), CONFIG_LIMIT)
    selected.read()
    return selected


def prepare_native_home(lifetime):
    """Initialize installed CLI, close it, then copy config alone to fresh home.

    Every native step stays in the caller's genuine owner custody and absolute
    deadline. The returned receipt cannot attest HTTP: the independent fixture
    must separately establish zero inference POSTs. This is not native or Lean
    admission, and never manufactures or copies a native session database.
    """
    if type(lifetime) is not NamedSessionLifetime:
        _refuse("native_preparation_requires_exact_lifetime")
    try:
        return _prepare_native_home(lifetime)
    except BaseException:
        lifetime.request._phase = "failed"
        lifetime.environment.clear()
        raise


def _prepare_native_home(lifetime):
    request = lifetime.request
    if request._phase != "prepared" or request._target_configuration is not None:
        _refuse("native_preparation_already_used")
    request.assert_current("bootstrap_start")
    request._phase = "preparing_home"
    seed = request.configuration.read()[0]
    _durable_configuration(request.preparation.home / "config.toml", seed)
    # The initialization runner sends initialize only, then normal EOF. The
    # actual target's session/new is a later independent native operation.
    _run_acp(lifetime, preparation=True)
    request.assert_current("bootstrap_protocol")
    if _group_births() != lifetime.baseline:
        _refuse("initialization_process_group_not_restored")
    normalized = _normalized_configuration(seed,
        _read_normalized_configuration(request.preparation.home / "config.toml"))
    if list(request.native_home.iterdir()):
        _refuse("target_home_changed_before_preparation")
    request._target_configuration = _durable_configuration(request.native_home / "config.toml", normalized)
    request._target_configuration.read()
    request._phase = "home_prepared"
    request.assert_current("bootstrap_protocol")
    _validate_sampling_profile(request, lifetime.environment)
    request._preparation_receipt.update(configuration_validated=True,
        target_configuration_sha256=request._target_configuration.sha256,
        only_configuration_copied=True, native_state_copied=False,
        owner_policy_is_not_native_enforced_policy=True)
    with _private_file(request.audit_directory / "native-home-preparation.json") as target:
        target.write((json.dumps(request._preparation_receipt, sort_keys=True, allow_nan=False)+"\n").encode())
    return request._target_configuration


class NamedSessionLifetime:
    """Keep one private environment/state alive through both sequential calls."""
    def __init__(self, request, environment, *, model, base_command, selected_api_key,
                 selected_slot, selected_fingerprint):
        validate_preparation_option("grok_cli", request, {})
        if model != MODEL or list(base_command) != [str(request.binary.path)]:
            _refuse("named_session_model_or_base_command")
        if type(selected_api_key) is not str or not selected_api_key.strip():
            _refuse("named_session_selected_credential_missing")
        if (selected_slot != request.credential_slot
                or selected_fingerprint != request.credential_fingerprint
                or hashlib.sha256(selected_api_key.encode()).hexdigest()[:16] != selected_fingerprint):
            _refuse("named_session_selected_credential_binding_mismatch")
        self.request = request
        self.environment = dict(environment)
        self.environment.update(XAI_API_KEY=selected_api_key,
                                GROK_HOME=str(request.native_home), GROK_TITLE_REFRESH="0")
        for name in ("GROK_TELEMETRY_ENABLED", "GROK_FEEDBACK_ENABLED",
                     "GROK_TELEMETRY_TRACE_UPLOAD", "GROK_AUTO_UPDATE",
                     "GROK_MANAGED_MCPS_ENABLED", "GROK_MANAGED_MCP_GATEWAY_TOOLS_ENABLED",
                     "GROK_SUBAGENTS", "GROK_MEMORY", "GROK_WEB_FETCH"):
            self.environment[name] = "0"
        for vendor in ("CLAUDE", "CURSOR", "CODEX"):
            for component in ("SKILLS", "RULES", "AGENTS", "MCPS", "HOOKS", "SESSIONS"):
                self.environment[f"GROK_{vendor}_{component}_ENABLED"] = "0"
        for name in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME", "XDG_CACHE_HOME"):
            self.environment[name] = str(request.native_home / name.lower())
        # Preserve HOME and the native context/output settings. No global
        # environment change, credential-file read, or weight download.
        _validate_sampling_profile(request, self.environment)
        self.baseline = _group_births()
        self.session_id = ""

    def __repr__(self):
        return "NamedSessionLifetime(environment=<redacted>)"

    def __enter__(self):
        self.request.assert_current("bootstrap_start")
        if self.request._phase not in {"prepared", "home_prepared"}:
            _refuse("named_session_request_already_used")
        try:
            if self.request._phase == "prepared":
                prepare_native_home(self)
            if (self.request._phase != "home_prepared" or self.request._target_configuration is None
                    or self.request._preparation_receipt.get("initialization_successful") is not True
                    or self.request._preparation_receipt.get("only_configuration_copied") is not True
                    or self.request._preparation_receipt.get("native_state_copied") is not False):
                _refuse("native_preparation_incomplete")
            self.request._phase = "bootstrapping"
            self.session_id = _bootstrap(self)
        except BaseException:
            self.request._phase = "failed"
            self.environment.clear()
            raise
        self.request._session = self.session_id
        self.request._phase = "headless_ready"
        return self

    def headless_timeout(self, requested_timeout):
        if self.request._phase != "headless_ready":
            _refuse("headless_before_bootstrap_close")
        self.request.assert_current("headless_start")
        _validate_sampling_profile(self.request, self.environment)
        if _group_births() != self.baseline:
            _refuse("bootstrap_process_group_not_restored")
        self.request._phase = "headless_running"
        return self.request.remaining_seconds(requested_timeout)

    def __exit__(self, exc_type, exc, traceback):
        try:
            try:
                self.request.assert_current("headless_close")
                if _group_births() != self.baseline:
                    _refuse("headless_process_group_not_restored")
                if exc is None and self.request._phase != "headless_running":
                    _refuse("headless_not_executed")
            except BaseException:
                self.request._phase = "failed"
                self.request._receipt["lifetime_closed"] = False
                if exc is None:
                    raise
            else:
                self.request._phase = "headless_closed" if exc is None else "failed"
                self.request._receipt["lifetime_closed"] = True
        finally:
            # Retained files belong to the original charged owner. Clearing a
            # child environment is not a disk/source/native release claim.
            self.environment.clear()
        return False


def _bootstrap(lifetime):
    return _run_acp(lifetime, preparation=False)


def _run_acp(lifetime, *, preparation):
    request = lifetime.request
    session_id = None if preparation else str(uuid.uuid4())
    audit_directory = request.audit_directory
    environment = lifetime.environment
    if preparation:
        audit_directory = audit_directory / "initialization"
        audit_directory.mkdir(mode=0o700)
        environment = dict(environment)
        environment["GROK_HOME"] = str(request.preparation.home)
        for name in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME", "XDG_CACHE_HOME"):
            environment[name] = str(request.preparation.home / name.lower())
    command = [str(request.binary.path), "agent", "--no-leader", "--model", MODEL, "stdio"]
    stage_deadline = min(request.deadline_monotonic-request.cleanup_reserve_seconds,
                         time.monotonic()+request.bootstrap_max_seconds)
    selector = selectors.DefaultSelector()
    streams = {}
    process = None
    buffer = bytearray()
    sizes = {"stdout": 0, "stderr": 0}
    replies = []
    methods_attempted = []
    completed = False
    forced = False
    failure = None

    def check():
        request.assert_current("bootstrap_protocol")
        if time.monotonic() >= stage_deadline:
            _refuse("bootstrap_absolute_deadline")

    def drain():
        check()
        for key, _events in selector.select(min(.2, max(0, stage_deadline-time.monotonic()))):
            raw = os.read(key.fd, 65536)
            if not raw:
                selector.unregister(key.fileobj)
                continue
            kind = key.data
            sizes[kind] += len(raw)
            if sizes[kind] > (STDOUT_LIMIT if kind == "stdout" else STDERR_LIMIT):
                _refuse("bootstrap_stream_byte_bound")
            streams[kind].write(raw)
            if kind == "stdout":
                buffer.extend(raw)

    def call(sequence, method, params):
        check()
        raw = (json.dumps({"jsonrpc": "2.0", "id": sequence, "method": method,
                           "params": params}, separators=(",", ":"), allow_nan=False)+"\n").encode()
        if len(raw) > 16384:
            _refuse("bootstrap_request_byte_bound")
        streams["sent"].write(raw)
        methods_attempted.append(method)
        if process.stdin.write(raw) != len(raw):
            _refuse("bootstrap_request_short_write")
        while True:
            while b"\n" in buffer:
                raw, _, rest = buffer.partition(b"\n")
                buffer[:] = rest
                if len(raw) > FRAME_LIMIT:
                    _refuse("bootstrap_frame_byte_bound")
                if not raw.strip():
                    continue
                frame = _object(raw)
                if "id" not in frame:
                    if frame.get("jsonrpc") != "2.0" or type(frame.get("method")) is not str:
                        _refuse("bootstrap_notification_invalid")
                    continue
                result = _ack(frame, sequence)
                _validate_result(sequence, result, session_id)
                replies.append({"id": sequence, "method": method,
                    "response_sha256": hashlib.sha256(json.dumps(result, sort_keys=True,
                        separators=(",", ":"), allow_nan=False).encode()).hexdigest()})
                return
            if len(buffer) > FRAME_LIMIT:
                _refuse("bootstrap_frame_byte_bound")
            drain()
            if process.poll() is not None and not selector.get_map():
                _refuse("bootstrap_exited_before_reply")

    try:
        for name, filename in (("stdout", "bootstrap-stdout.jsonl"),
                               ("stderr", "bootstrap-stderr.bin"), ("sent", "bootstrap-sent.jsonl")):
            streams[name] = _private_file(audit_directory / filename)
        process = subprocess.Popen(command, cwd=request.workspace, env=environment,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=0, close_fds=True)
        for pipe, name in ((process.stdout, "stdout"), (process.stderr, "stderr")):
            os.set_blocking(pipe.fileno(), False)
            selector.register(pipe, selectors.EVENT_READ, name)
        call(1, METHODS[0], {"protocolVersion": 1, "clientCapabilities": {},
            "clientInfo": {"name": "ipfs-accelerate-named-planning", "version": "1"}})
        if not preparation:
            call(2, METHODS[1], {"cwd": str(request.workspace), "mcpServers": [],
                "_meta": {"sessionId": session_id, "modelId": MODEL}})
            call(3, METHODS[2], {"sessionId": session_id, "cwd": str(request.workspace), "title": TITLE})
            call(4, METHODS[3], {"sessionId": session_id})
        process.stdin.close()
        while selector.get_map() or process.poll() is None:
            drain()
        process.wait(timeout=max(.001, stage_deadline-time.monotonic()))
        if process.returncode != 0:
            _refuse("bootstrap_native_process_nonzero")
        while _group_births() != lifetime.baseline:
            check()
            time.sleep(min(.05, max(0, stage_deadline-time.monotonic())))
        completed = True
    except BaseException as error:
        # Native output, exception text and callback messages may be private.
        failure = type(error).__name__
        raise
    finally:
        primary = sys.exc_info()[1]
        cleanup_errors = []
        def cleanup(action):
            try:
                action()
            except BaseException as error:
                cleanup_errors.append(type(error).__name__)
        if process is not None:
            if process.stdin and not process.stdin.closed:
                cleanup(process.stdin.close)
            if process.poll() is None:
                forced = True
                cleanup(process.terminate)
                try:
                    process.wait(timeout=min(1, max(.001, request.deadline_monotonic-time.monotonic())))
                except subprocess.TimeoutExpired:
                    cleanup(process.kill)
                    cleanup(lambda: process.wait(timeout=min(1, max(.001,
                        request.deadline_monotonic-time.monotonic()))))
                except BaseException as error:
                    cleanup_errors.append(type(error).__name__)
            for pipe in (process.stdout, process.stderr):
                if pipe is not None:
                    cleanup(pipe.close)
        cleanup(selector.close)
        for stream in streams.values():
            cleanup(stream.close)
        report = {"schema": "grok-owned-acp-named-session@1",
            "report_scope": "home_initialize_only" if preparation else "bootstrap_close_only",
            "source_generation": request.source_generation,
            "session_id": session_id, "request_methods": [row["method"] for row in replies],
            "replies": replies, "no_session_prompt_submitted": True,
            "native_session_state_written_by_client": False,
            "zero_inference_posts": None, "http_observation": "not_instrumented",
            "native_process_pid": process.pid if process else None,
            "native_process_reaped": bool(process and process.poll() is not None),
            "native_process_returncode": process.returncode if process else None,
            "process_group_baseline_restored": completed,
            "stdout_bytes": sizes["stdout"], "stderr_bytes": sizes["stderr"],
            "forced_cleanup": forced, "bootstrap_successful": completed and not forced and not cleanup_errors,
            "cleanup_error_types": cleanup_errors,
            "failure_code": failure, "lifetime_closed": False, "response_accepted": False,
            "request_methods_attempted": methods_attempted,
            "session_new_attempted": METHODS[1] in methods_attempted,
            "production_configuration_qualified": False, "native_admission_granted_by_helper": False}
        if preparation:
            report["initialization_successful"] = report["bootstrap_successful"]
            request._preparation_receipt = report
        else:
            request._receipt = report
        try:
            with _private_file(audit_directory / "bootstrap-result.json") as target:
                target.write((json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n").encode())
        except BaseException:
            report["bootstrap_successful"] = False
            if preparation:
                report["initialization_successful"] = False
            report["report_written"] = False
            if primary is None:
                _refuse("bootstrap_report_write_failed")
        if cleanup_errors and primary is None:
            _refuse("bootstrap_cleanup_incomplete")
    if not completed or forced:
        _refuse("bootstrap_not_cleanly_closed")
    return session_id
