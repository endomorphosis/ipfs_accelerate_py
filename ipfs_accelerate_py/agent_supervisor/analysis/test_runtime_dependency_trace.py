"""Bounded, privacy-safe runtime dependency tracing for pytest (PTR-021).

``RuntimeTestDependencyTrace@1`` is diagnostic evidence, not skip authority.
It observes a deliberately small set of dependency/effect facts during a cold
test execution and binds the observation policy, instrumentation identity, and
all limits into one canonical ``ContentIdentity@1`` artifact.

The implementation is fail-closed for reuse and fail-open for test execution:

* unsupported, private, over-budget, concurrent, or unhealthy instrumentation
  makes the trace incomplete;
* audit/profile callbacks and every public recording method swallow their own
  failures, so tracer failure cannot replace the test's outcome;
* raw environment values, absolute/private paths, subprocess arguments,
  arbitrary output, and exception text are never retained;
* the process-wide audit hook is a small permanent dispatcher.  It is inactive
  outside a tracing context and never raises into the audited operation.

This module intentionally does not decide reuse eligibility.  PTR-022 composes
this typed evidence with the static trace and policy.
"""

from __future__ import annotations

import hashlib
import json
import marshal
import os
import re
import sys
import threading
import time
import weakref
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import CodeType
from typing import Any, ClassVar, Final

from ipfs_accelerate_py.agent_supervisor.analysis.test_execution_identity import (
    ContentIdentity,
    mint_content_identity,
)
from ipfs_accelerate_py.agent_supervisor.analysis.test_identity_components import (
    DEFAULT_ENVIRONMENT_ALLOWLIST,
    ENVIRONMENT_VALUE_POLICIES,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    canonical_json_bytes,
)
from multiformats import CID

RUNTIME_TEST_DEPENDENCY_TRACE_INTERFACE: Final = "RuntimeTestDependencyTrace@1"
RUNTIME_TEST_DEPENDENCY_TRACER_INTERFACE: Final = "RuntimeTestDependencyTracer@1"
RUNTIME_TRACE_INSTRUMENTATION_INTERFACE: Final = "RuntimeTraceInstrumentation@1"
RUNTIME_TRACE_LIMITS_INTERFACE: Final = "RuntimeTraceLimits@1"

RUNTIME_TEST_DEPENDENCY_TRACE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/runtime-test-dependency-trace@1"
)
RUNTIME_TRACE_INSTRUMENTATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/runtime-trace-instrumentation@1"
)
RUNTIME_TRACE_LIMITS_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/runtime-trace-limits@1"

_ELIGIBILITY_PROFILES: Final = frozenset({"pure", "snapshot_bound", "repository_forest_bound"})
_DEPENDENCY_KINDS: Final = (
    "modules",
    "code_objects",
    "files",
    "environment",
    "subprocesses",
    "services",
    "policies",
    "capabilities",
)
_IGNORED_NO_EFFECT_AUDIT_EVENTS: Final = frozenset({"object.__getattr__"})
_CID_RE: Final = re.compile(r"^b[a-z2-7]{20,}$")
_SAFE_NAME_RE: Final = re.compile(r"^[A-Za-z0-9_.:@/+ <>-]{1,256}$")
_MAX_COUNTER: Final = (1 << 63) - 1
_MISSING: Final = object()


class RuntimeTraceError(ValueError):
    """Invalid trace configuration or unavailable trace identity."""

    __test__ = False


class RuntimeTraceCompleteness(str, Enum):  # noqa: UP042 - project supports Python 3.8
    """Closed completeness state consumed by later eligibility policy."""

    COMPLETE = "complete"
    INCOMPLETE = "incomplete"

    @property
    def complete(self) -> bool:
        return self is RuntimeTraceCompleteness.COMPLETE


@dataclass(frozen=True)
class RuntimeTraceLimits:
    """Hard limits which are part of every runtime trace identity."""

    __test__: ClassVar[bool] = False

    max_events: int = 2_048
    max_modules: int = 256
    max_code_objects: int = 512
    max_files: int = 256
    max_environment: int = len(ENVIRONMENT_VALUE_POLICIES)
    max_subprocesses: int = 64
    max_services: int = 64
    max_policies: int = 32
    max_capabilities: int = 128
    max_file_bytes: int = 8 * 1_048_576
    max_trace_seconds: int = 900
    max_text_chars: int = 512

    _BOUNDS: ClassVar[Mapping[str, tuple[int, int]]] = {
        "max_events": (1, 65_536),
        "max_modules": (1, 4_096),
        "max_code_objects": (1, 8_192),
        "max_files": (1, 4_096),
        "max_environment": (1, len(ENVIRONMENT_VALUE_POLICIES)),
        "max_subprocesses": (1, 1_024),
        "max_services": (1, 1_024),
        "max_policies": (1, 256),
        "max_capabilities": (1, 2_048),
        "max_file_bytes": (1, 64 * 1_048_576),
        "max_trace_seconds": (1, 7_200),
        "max_text_chars": (32, 4_096),
    }

    def __post_init__(self) -> None:
        for name, (minimum, maximum) in self._BOUNDS.items():
            value = getattr(self, name)
            if type(value) is not int or not minimum <= value <= maximum:
                raise RuntimeTraceError(f"{name} must be an integer in [{minimum}, {maximum}]")

    @property
    def interface(self) -> str:
        return RUNTIME_TRACE_LIMITS_INTERFACE

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RUNTIME_TRACE_LIMITS_SCHEMA,
            "interface": RUNTIME_TRACE_LIMITS_INTERFACE,
            **{name: getattr(self, name) for name in sorted(self._BOUNDS)},
        }


@dataclass(frozen=True)
class RuntimeTestDependencyTrace:
    """Immutable canonical result of a bounded tracing session."""

    __test__: ClassVar[bool] = False

    content_identity: ContentIdentity | None
    retained_canonical_bytes: bytes
    completeness: RuntimeTraceCompleteness
    completeness_reasons: tuple[str, ...]
    observed_event_count: int
    recorded_fact_count: int
    dropped_event_count: int

    def __post_init__(self) -> None:
        if type(self.retained_canonical_bytes) is not bytes:
            raise RuntimeTraceError("retained_canonical_bytes must be exact bytes")
        if self.content_identity is not None:
            if self.retained_canonical_bytes != self.content_identity.canonical_bytes:
                raise RuntimeTraceError("trace bytes do not match ContentIdentity")
        if self.completeness.complete and self.completeness_reasons:
            raise RuntimeTraceError("a complete trace cannot carry incomplete reasons")
        for value in (
            self.observed_event_count,
            self.recorded_fact_count,
            self.dropped_event_count,
        ):
            if type(value) is not int or value < 0 or value > _MAX_COUNTER:
                raise RuntimeTraceError("trace counters must be bounded integers")

    @property
    def interface(self) -> str:
        return RUNTIME_TEST_DEPENDENCY_TRACE_INTERFACE

    @property
    def schema(self) -> str:
        return RUNTIME_TEST_DEPENDENCY_TRACE_SCHEMA

    @property
    def complete(self) -> bool:
        return self.completeness.complete

    @property
    def cid(self) -> str:
        return self.content_identity.cid if self.content_identity is not None else ""

    @property
    def trace_cid(self) -> str:
        return self.cid

    @property
    def root_cid(self) -> str:
        return self.cid

    @property
    def canonical_bytes(self) -> bytes:
        return self.retained_canonical_bytes

    def to_dict(self) -> dict[str, Any]:
        if not self.retained_canonical_bytes:
            return {}
        value = json.loads(self.retained_canonical_bytes.decode("utf-8"))
        if not isinstance(value, dict):  # pragma: no cover - constructor invariant
            raise RuntimeTraceError("runtime trace canonical bytes are not an object")
        return value

    def verify(self) -> RuntimeTestDependencyTrace:
        if self.content_identity is None:
            raise RuntimeTraceError("runtime trace has no CID-capable identity")
        self.content_identity.verify()
        if canonical_json_bytes(self.to_dict()) != self.retained_canonical_bytes:
            raise RuntimeTraceError("runtime trace bytes are not canonical")
        return self


_AUDIT_LOCK = threading.RLock()
_ACTIVE_TRACERS: weakref.WeakSet[RuntimeTestDependencyTracer] = weakref.WeakSet()
_AUDIT_HOOK_INSTALLED = False
_AUDIT_HOOK_INSTALL_ATTEMPTED = False
_AUDIT_INSTALL_PROBE_EVENT = "ipfs_accelerate_py.agent_supervisor.runtime_trace.audit_install_probe"
_AUDIT_INSTALL_PROBE_TOKEN = object()
_AUDIT_INSTALL_PROBE_ACKS = 0


def _audit_dispatch(event: str, arguments: tuple[Any, ...]) -> None:
    """Permanent process hook; must never raise into the audited operation."""

    global _AUDIT_INSTALL_PROBE_ACKS
    try:
        if (
            event == _AUDIT_INSTALL_PROBE_EVENT
            and len(arguments) == 1
            and arguments[0] is _AUDIT_INSTALL_PROBE_TOKEN
        ):
            with _AUDIT_LOCK:
                _AUDIT_INSTALL_PROBE_ACKS += 1
            return
        with _AUDIT_LOCK:
            active = tuple(_ACTIVE_TRACERS)
        for tracer in active:
            try:
                tracer._observe_audit_event(event, arguments, synthetic=False)
            except BaseException:
                # An audit hook exception can alter the operation under test.
                # This boundary therefore catches BaseException deliberately.
                try:
                    tracer._mark_internal_failure("audit_callback")
                except BaseException:
                    pass
    except BaseException:
        pass


def _install_audit_dispatch() -> bool:
    global _AUDIT_HOOK_INSTALLED, _AUDIT_HOOK_INSTALL_ATTEMPTED
    with _AUDIT_LOCK:
        if _AUDIT_HOOK_INSTALLED:
            return True
        if _AUDIT_HOOK_INSTALL_ATTEMPTED:
            return False
        _AUDIT_HOOK_INSTALL_ATTEMPTED = True
        acknowledgements_before = _AUDIT_INSTALL_PROBE_ACKS
        try:
            sys.addaudithook(_audit_dispatch)
            # CPython permits an existing hook to suppress ``addaudithook`` by
            # raising RuntimeError without propagating it to this caller.
            # Prove our dispatcher was actually registered before it can
            # contribute complete trace authority.
            sys.audit(
                _AUDIT_INSTALL_PROBE_EVENT,
                _AUDIT_INSTALL_PROBE_TOKEN,
            )
        except BaseException:
            return False
        if _AUDIT_INSTALL_PROBE_ACKS != acknowledgements_before + 1:
            return False
        _AUDIT_HOOK_INSTALLED = True
        return True


class RuntimeTestDependencyTracer:
    """Observe bounded runtime facts without acquiring outcome authority.

    ``allowed_roots`` maps public root identifiers to admitted filesystem roots.
    Only relative paths beneath those roots can enter evidence.  If omitted,
    filesystem observations are classified private and make the trace
    incomplete.

    Environment names must be a subset of the repository-wide reviewed
    allowlist.  ``subprocess_allowlist`` maps executable basenames to stable,
    public tool identities (normally CIDs).  Arguments are always reduced to a
    digest and count; their text is never retained.
    """

    __test__ = False

    def __init__(
        self,
        *,
        limits: RuntimeTraceLimits | None = None,
        allowed_roots: Mapping[str, os.PathLike[str] | str] | None = None,
        environment_allowlist: Iterable[str] = DEFAULT_ENVIRONMENT_ALLOWLIST,
        subprocess_allowlist: Mapping[str, str] | None = None,
        eligibility_profile: str = "pure",
        capture_code_objects: bool = True,
        identity_minter: Callable[[Any], ContentIdentity] = mint_content_identity,
    ) -> None:
        self.limits = limits or RuntimeTraceLimits()
        if eligibility_profile not in _ELIGIBILITY_PROFILES:
            raise RuntimeTraceError("eligibility_profile is not admitted")
        self.eligibility_profile = eligibility_profile
        self.capture_code_objects = bool(capture_code_objects)
        if not callable(identity_minter):
            raise RuntimeTraceError("identity_minter must be callable")
        self._identity_minter = identity_minter

        requested_env = tuple(environment_allowlist)
        if any(type(name) is not str for name in requested_env):
            raise RuntimeTraceError("environment allowlist contains a non-string")
        if any(name not in ENVIRONMENT_VALUE_POLICIES for name in requested_env):
            raise RuntimeTraceError("environment allowlist contains a non-reviewed variable")
        if len(set(requested_env)) > self.limits.max_environment:
            raise RuntimeTraceError("environment allowlist exceeds trace limit")
        self.environment_allowlist = frozenset(requested_env)

        roots: list[tuple[str, Path]] = []
        for root_id, raw_path in (allowed_roots or {}).items():
            safe_root_id = self._checked_name(root_id, field="root identifier")
            try:
                resolved = Path(os.fspath(raw_path)).resolve(strict=True)
            except (OSError, TypeError, ValueError) as exc:
                raise RuntimeTraceError("allowed root is unavailable") from exc
            if not resolved.is_dir():
                raise RuntimeTraceError("allowed root must be a directory")
            roots.append((safe_root_id, resolved))
        if len({root_id for root_id, _ in roots}) != len(roots):
            raise RuntimeTraceError("allowed root identifiers must be unique")
        # Longest path first gives deterministic behavior for nested roots.
        self._allowed_roots = tuple(sorted(roots, key=lambda item: (-len(item[1].parts), item[0])))

        tools: dict[str, str] = {}
        for executable, identity in (subprocess_allowlist or {}).items():
            name = self._executable_name(executable)
            tools[name] = self._checked_identity(identity, field="tool identity")
        self._subprocess_allowlist = tools

        self._lock = threading.RLock()
        self._facts: dict[str, dict[bytes, dict[str, Any]]] = {
            kind: {} for kind in _DEPENDENCY_KINDS
        }
        self._reasons: set[str] = set()
        self._unsupported_kinds: set[str] = set()
        self._private_kinds: set[str] = set()
        self._internal_failure_kinds: set[str] = set()
        self._observed_event_count = 0
        self._dropped_event_count = 0
        self._started = False
        self._stopped = False
        self._active = False
        self._start_monotonic = 0.0
        self._audit_hook_healthy = False
        self._profile_healthy = not self.capture_code_objects
        self._profile_callback = self._profile_dispatch
        self._previous_profile: Any = None
        self._previous_thread_profile: Any = None
        self._profile_installed = False
        self._thread_profile_installed = False
        self._inside_callback = threading.local()
        self._instrumentation: dict[str, Any] = {}
        self._instrumentation_cid = ""
        self._result: RuntimeTestDependencyTrace | None = None

    @staticmethod
    def _checked_name(value: Any, *, field: str) -> str:
        if type(value) is not str or not _SAFE_NAME_RE.fullmatch(value):
            raise RuntimeTraceError(f"{field} is not a bounded public name")
        return value

    @staticmethod
    def _checked_identity(value: Any, *, field: str) -> str:
        if type(value) is not str or not _CID_RE.fullmatch(value):
            raise RuntimeTraceError(f"{field} must be a canonical CIDv1/base32 identity")
        try:
            parsed = CID.decode(value)
        except Exception as exc:
            raise RuntimeTraceError(f"{field} must be a canonical CIDv1/base32 identity") from exc
        if (
            str(parsed) != value
            or parsed.version != 1
            or parsed.base.name != "base32"
            or parsed.codec.name not in {"dag-json", "raw"}
            or parsed.hashfun.name != "sha2-256"
            or len(parsed.raw_digest) != 32
        ):
            raise RuntimeTraceError(f"{field} must use CIDv1/base32/(dag-json|raw)/sha2-256")
        return value

    @staticmethod
    def _executable_name(value: Any) -> str:
        if type(value) is not str or not value or "\x00" in value:
            raise RuntimeTraceError("executable must be bounded text")
        name = value.replace("\\", "/").rsplit("/", 1)[-1]
        if not _SAFE_NAME_RE.fullmatch(name):
            raise RuntimeTraceError("executable basename is not a public name")
        return name

    @property
    def result(self) -> RuntimeTestDependencyTrace | None:
        return self._result

    @property
    def trace(self) -> RuntimeTestDependencyTrace | None:
        return self._result

    def __enter__(self) -> RuntimeTestDependencyTracer:
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        del exc_type, exc, traceback
        try:
            self.stop()
        except BaseException:
            # Never suppress or replace an exception from the test body.
            pass
        return False

    def start(self) -> RuntimeTestDependencyTracer:
        """Activate observation.  Instrumentation failure is evidence, not an error."""

        with self._lock:
            if self._started:
                self._mark_unsupported("invalid_lifecycle")
                return self
            self._started = True
            self._active = True
            self._start_monotonic = time.monotonic()
        try:
            self._build_instrumentation_identity()
        except BaseException:
            self._mark_internal_failure("instrumentation_identity")

        self._audit_hook_healthy = _install_audit_dispatch()
        if not self._audit_hook_healthy:
            self._mark_internal_failure("audit_hook_install")
        with _AUDIT_LOCK:
            existing = tuple(_ACTIVE_TRACERS)
            if existing:
                self._mark_unsupported("concurrent_trace")
                for tracer in existing:
                    tracer._mark_unsupported("concurrent_trace")
            _ACTIVE_TRACERS.add(self)

        if self.capture_code_objects:
            self._install_profile()
        return self

    def stop(self) -> RuntimeTestDependencyTrace:
        """Stop observation and return canonical evidence; this method never raises."""

        with self._lock:
            if self._result is not None:
                return self._result
            if not self._started:
                self._started = True
                self._mark_unsupported("not_started")
            self._active = False
            self._stopped = True
        with _AUDIT_LOCK:
            _ACTIVE_TRACERS.discard(self)
        self._restore_profile()
        try:
            if self._start_monotonic:
                elapsed = time.monotonic() - self._start_monotonic
                if elapsed > self.limits.max_trace_seconds:
                    self._mark_overflow("duration")
            result = self._build_result()
        except BaseException:
            self._mark_internal_failure("result_build")
            result = self._build_fallback_result()
        with self._lock:
            self._result = result
        return result

    finish = stop

    def _install_profile(self) -> None:
        try:
            self._previous_profile = sys.getprofile()
            get_thread_profile = getattr(threading, "getprofile", lambda: None)
            self._previous_thread_profile = get_thread_profile()
            if self._previous_profile is not None or self._previous_thread_profile is not None:
                self._mark_unsupported("preexisting_profiler")
                return
            self._inside_callback.active = True
            try:
                sys.setprofile(self._profile_callback)
                self._profile_installed = True
                threading.setprofile(self._profile_callback)
                self._thread_profile_installed = True
                self._profile_healthy = True
            finally:
                self._inside_callback.active = False
        except BaseException:
            self._profile_healthy = False
            self._mark_internal_failure("profile_install")

    def _restore_profile(self) -> None:
        try:
            self._inside_callback.active = True
            try:
                if self._profile_installed and sys.getprofile() is self._profile_callback:
                    sys.setprofile(self._previous_profile)
                if self._thread_profile_installed:
                    get_thread_profile = getattr(threading, "getprofile", lambda: None)
                    if get_thread_profile() is self._profile_callback:
                        threading.setprofile(self._previous_thread_profile)
            finally:
                self._inside_callback.active = False
        except BaseException:
            self._profile_healthy = False
            self._mark_internal_failure("profile_restore")

    def _profile_dispatch(self, frame: Any, event: str, argument: Any) -> None:
        del argument
        try:
            if event != "call" or not self._active:
                return
            if getattr(self._inside_callback, "active", False):
                return
            self._inside_callback.active = True
            try:
                code = frame.f_code
                if isinstance(code, CodeType) and Path(code.co_filename) != Path(__file__):
                    self._record_code_object(code, frame.f_globals.get("__name__", ""))
            finally:
                self._inside_callback.active = False
        except BaseException:
            try:
                self._mark_internal_failure("profile_callback")
            except BaseException:
                pass

    def _build_instrumentation_identity(self) -> None:
        source_digest = "unavailable"
        try:
            hasher = hashlib.sha256()
            with open(__file__, "rb") as source:
                while True:
                    chunk = source.read(128 * 1_024)
                    if not chunk:
                        break
                    hasher.update(chunk)
            source_digest = hasher.hexdigest()
        except (OSError, ValueError):
            self._mark_internal_failure("instrumentation_source")
        self._instrumentation = {
            "schema": RUNTIME_TRACE_INSTRUMENTATION_SCHEMA,
            "interface": RUNTIME_TRACE_INSTRUMENTATION_INTERFACE,
            "tracer_interface": RUNTIME_TEST_DEPENDENCY_TRACER_INTERFACE,
            "implementation": "cpython-audit-profile-explicit-adapters",
            "implementation_source_sha256": source_digest,
            "python_implementation": sys.implementation.name,
            "python_version": ".".join(str(value) for value in sys.version_info[:3]),
            "python_cache_tag": sys.implementation.cache_tag or "",
            "audit_schema": "cpython-audit-events@1",
            "ignored_no_effect_audit_events": sorted(_IGNORED_NO_EFFECT_AUDIT_EVENTS),
            "profile_schema": "python-call-code-objects@1",
            "adapter_schema": "ptr-runtime-explicit-adapters@1",
            "captures_code_objects": self.capture_code_objects,
        }
        self._instrumentation_cid = self._mint_internal_identity(self._instrumentation).cid

    def _mint_internal_identity(self, value: Any) -> ContentIdentity:
        """Mint tracer-owned evidence without observing the minter itself."""

        previous_guard = getattr(self._inside_callback, "active", False)
        self._inside_callback.active = True
        try:
            expected_canonical_bytes = canonical_json_bytes(value)
            identity = self._identity_minter(value)
            if not isinstance(identity, ContentIdentity):
                raise RuntimeTraceError("identity provider did not return ContentIdentity")
            identity.verify()
            if identity.canonical_bytes != expected_canonical_bytes:
                raise RuntimeTraceError("identity provider canonical bytes do not match input")
            return identity
        finally:
            self._inside_callback.active = previous_guard

    def _increment(self, attribute: str) -> None:
        with self._lock:
            value = getattr(self, attribute)
            setattr(self, attribute, min(_MAX_COUNTER, value + 1))

    def _mark_unsupported(self, kind: str) -> None:
        with self._lock:
            self._reasons.add("unsupported_event")
            self._unsupported_kinds.add(kind)

    def _mark_private(self, kind: str) -> None:
        with self._lock:
            self._reasons.add("private_event")
            self._private_kinds.add(kind)

    def _mark_overflow(self, kind: str) -> None:
        with self._lock:
            self._reasons.add("overflow")
            self._unsupported_kinds.add(f"overflow:{kind}")

    def _mark_internal_failure(self, kind: str) -> None:
        with self._lock:
            self._reasons.add("instrumentation_failure")
            self._internal_failure_kinds.add(kind)

    def _category_limit(self, category: str) -> int:
        return getattr(self.limits, f"max_{category}")

    def _accept_fact(self, category: str, fact: dict[str, Any]) -> bool:
        try:
            encoded = canonical_json_bytes(fact)
            if len(encoded) > self.limits.max_text_chars * 8:
                self._mark_overflow("fact_bytes")
                self._increment("_dropped_event_count")
                return False
            with self._lock:
                self._increment("_observed_event_count")
                if self._observed_event_count > self.limits.max_events:
                    self._mark_overflow("events")
                    self._increment("_dropped_event_count")
                    return False
                bucket = self._facts[category]
                if encoded in bucket:
                    return True
                if len(bucket) >= self._category_limit(category):
                    self._mark_overflow(category)
                    self._increment("_dropped_event_count")
                    return False
                bucket[encoded] = fact
            return True
        except BaseException:
            self._mark_internal_failure("record_fact")
            return False

    def _root_relative_path(
        self, raw_path: Any, *, mark_private: bool = True
    ) -> tuple[str, Path, str] | None:
        if not isinstance(raw_path, (str, bytes, os.PathLike)):
            if mark_private:
                self._mark_private("file_path")
            return None
        try:
            path = Path(os.fsdecode(os.fspath(raw_path)))
            lexical = Path(os.path.abspath(path))
        except (OSError, TypeError, ValueError):
            if mark_private:
                self._mark_private("file_path")
            return None
        for root_id, root in self._allowed_roots:
            try:
                relative = lexical.relative_to(root)
            except ValueError:
                continue
            if not relative.parts:
                self._mark_unsupported("directory_read")
                return None
            probe = root
            try:
                for part in relative.parts:
                    probe = probe / part
                    if probe.is_symlink():
                        self._mark_private("symlink")
                        return None
                resolved = lexical.resolve(strict=True)
                resolved.relative_to(root)
            except (OSError, RuntimeError, ValueError):
                self._mark_private("path_escape")
                return None
            relative_text = relative.as_posix()
            if len(relative_text) > self.limits.max_text_chars or any(
                ord(char) < 32 for char in relative_text
            ):
                self._mark_private("file_path")
                return None
            return root_id, resolved, relative_text
        if mark_private:
            self._mark_private("file_path")
        return None

    def _file_fact(self, raw_path: Any) -> dict[str, Any] | None:
        located = self._root_relative_path(raw_path)
        if located is None:
            return None
        root_id, resolved, relative = located
        try:
            stat_before = resolved.stat()
            if not resolved.is_file():
                self._mark_unsupported("non_regular_file")
                return None
            if stat_before.st_size > self.limits.max_file_bytes:
                self._mark_overflow("file_bytes")
                return None
            hasher = hashlib.sha256()
            count = 0
            previous_guard = getattr(self._inside_callback, "active", False)
            self._inside_callback.active = True
            try:
                with open(resolved, "rb") as stream:
                    while True:
                        chunk = stream.read(128 * 1_024)
                        if not chunk:
                            break
                        count += len(chunk)
                        if count > self.limits.max_file_bytes:
                            self._mark_overflow("file_bytes")
                            return None
                        hasher.update(chunk)
            finally:
                self._inside_callback.active = previous_guard
            stat_after = resolved.stat()
            if (
                count != stat_before.st_size
                or stat_before.st_size != stat_after.st_size
                or stat_before.st_mtime_ns != stat_after.st_mtime_ns
            ):
                self._mark_unsupported("file_changed_during_trace")
                return None
            return {
                "root_id": root_id,
                "path": relative,
                "size_bytes": count,
                "content_sha256": hasher.hexdigest(),
            }
        except (OSError, ValueError):
            self._mark_unsupported("file_unavailable")
            return None
        except BaseException:
            self._mark_internal_failure("file_identity")
            return None

    def record_file_read(self, path: os.PathLike[str] | str) -> bool:
        """Record an admitted file read without retaining an absolute path."""

        try:
            fact = self._file_fact(path)
            return fact is not None and self._accept_fact("files", fact)
        except BaseException:
            self._mark_internal_failure("record_file")
            return False

    def _record_code_object(self, code: CodeType, module: Any = "") -> bool:
        located = self._root_relative_path(code.co_filename, mark_private=False)
        if located is None:
            # Profiles see framework/stdlib code too.  Those paths are already
            # bound by interpreter/dependency identities and are ignored here;
            # only code claimed to be in-scope is recorded as a code object.
            return False
        root_id, _resolved, relative = located
        try:
            module_name = module if type(module) is str else ""
            if module_name and not _SAFE_NAME_RE.fullmatch(module_name):
                module_name = ""
            qualname = getattr(code, "co_qualname", code.co_name)
            if type(qualname) is not str or not _SAFE_NAME_RE.fullmatch(qualname):
                self._mark_private("code_name")
                return False
            # ``co_filename`` is often absolute.  Replace it before hashing so
            # the code identity is stable across checkout locations and cannot
            # act as a digest oracle for a private host path.  CPython audits
            # both code replacement and marshaling; suppress those events from
            # our own hook so instrumentation cannot observe itself.
            previous_guard = getattr(self._inside_callback, "active", False)
            self._inside_callback.active = True
            try:
                normalized_code = code.replace(co_filename=relative)
                digest = hashlib.sha256(marshal.dumps(normalized_code)).hexdigest()
            finally:
                self._inside_callback.active = previous_guard
            return self._accept_fact(
                "code_objects",
                {
                    "root_id": root_id,
                    "path": relative,
                    "module": module_name,
                    "qualname": qualname,
                    "first_line": max(0, int(code.co_firstlineno)),
                    "code_sha256": digest,
                },
            )
        except BaseException:
            self._mark_internal_failure("code_identity")
            return False

    def record_code_object(self, code: CodeType, module: str = "") -> bool:
        try:
            if not isinstance(code, CodeType):
                self._mark_unsupported("invalid_code_object")
                return False
            return self._record_code_object(code, module)
        except BaseException:
            self._mark_internal_failure("record_code")
            return False

    def record_module(
        self,
        name: str,
        *,
        source_path: os.PathLike[str] | str | None = None,
        native: bool = False,
    ) -> bool:
        try:
            safe_name = self._checked_name(name, field="module name")
            fact: dict[str, Any] = {"name": safe_name, "kind": "python"}
            if native:
                fact["kind"] = "native"
                self._mark_unsupported("native_module")
            if source_path is not None:
                source = self._file_fact(source_path)
                if source is None:
                    return False
                fact.update(
                    {
                        "root_id": source["root_id"],
                        "path": source["path"],
                        "source_sha256": source["content_sha256"],
                    }
                )
            return self._accept_fact("modules", fact)
        except RuntimeTraceError:
            self._mark_private("module_name")
            return False
        except BaseException:
            self._mark_internal_failure("record_module")
            return False

    def record_environment_read(self, name: str, value: Any = _MISSING) -> bool:
        """Bind a reviewed environment read while retaining no raw value."""

        try:
            if type(name) is not str or name not in self.environment_allowlist:
                self._mark_private("environment")
                return False
            raw = os.environ.get(name) if value is _MISSING else value
            if raw is not None and type(raw) is not str:
                self._mark_private("environment")
                return False
            value_identity = self._mint_internal_identity(
                {
                    "schema": "ipfs_accelerate_py/agent-supervisor/runtime-environment-value@1",
                    "name": name,
                    "present": raw is not None,
                    "value": raw if raw is not None else "",
                }
            ).cid
            return self._accept_fact("environment", {"name": name, "value_cid": value_identity})
        except BaseException:
            self._mark_internal_failure("record_environment")
            return False

    def getenv(self, name: str, default: Any = None) -> Any:
        """Outcome-transparent ``os.getenv`` adapter with observation."""

        value = os.getenv(name, default)
        try:
            self.record_environment_read(name, value)
        except BaseException:
            pass
        return value

    @staticmethod
    def _argument_digest(arguments: Any) -> tuple[int, str] | None:
        if isinstance(arguments, (str, bytes, os.PathLike)):
            items: Sequence[Any] = (arguments,)
        elif isinstance(arguments, (tuple, list)):
            items = arguments
        else:
            return None
        if len(items) > 4_096:
            return None
        hasher = hashlib.sha256()
        for item in items:
            try:
                raw = os.fspath(item) if isinstance(item, os.PathLike) else item
            except (TypeError, ValueError, OSError):
                return None
            if isinstance(raw, str):
                encoded = raw.encode("utf-8", "surrogatepass")
                tag = b"s"
            elif isinstance(raw, bytes):
                encoded = raw
                tag = b"b"
            else:
                return None
            hasher.update(tag)
            hasher.update(len(encoded).to_bytes(8, "big"))
            hasher.update(encoded)
        return len(items), hasher.hexdigest()

    def record_subprocess(
        self,
        executable: str,
        arguments: Any = (),
        *,
        tool_identity: str | None = None,
    ) -> bool:
        """Record a tool invocation, never raw arguments, cwd, env, or output."""

        try:
            name = self._executable_name(executable)
            argument_identity = self._argument_digest(arguments)
            if argument_identity is None:
                self._mark_private("subprocess_arguments")
                return False
            count, digest = argument_identity
            expected_identity = self._subprocess_allowlist.get(name)
            identity = (
                expected_identity
                if tool_identity is None
                else self._checked_identity(tool_identity, field="tool identity")
            )
            if expected_identity is None or identity != expected_identity:
                self._mark_unsupported("subprocess_tool")
            return self._accept_fact(
                "subprocesses",
                {
                    "executable": name,
                    "argument_count": count,
                    "arguments_sha256": digest,
                    "tool_identity": identity or "unadmitted",
                },
            )
        except RuntimeTraceError:
            self._mark_private("subprocess")
            return False
        except BaseException:
            self._mark_internal_failure("record_subprocess")
            return False

    def record_service(
        self, service: str, *, adapter_identity: str, snapshot_identity: str
    ) -> bool:
        try:
            return self._accept_fact(
                "services",
                {
                    "service": self._checked_name(service, field="service name"),
                    "adapter_identity": self._checked_identity(
                        adapter_identity, field="adapter identity"
                    ),
                    "snapshot_identity": self._checked_identity(
                        snapshot_identity, field="snapshot identity"
                    ),
                },
            )
        except RuntimeTraceError:
            self._mark_unsupported("service_adapter")
            return False
        except BaseException:
            self._mark_internal_failure("record_service")
            return False

    def record_policy(self, kind: str, policy_identity: str) -> bool:
        try:
            safe_kind = self._checked_name(kind, field="policy kind")
            if safe_kind not in {"clock", "randomness"}:
                self._mark_unsupported("policy_kind")
                return False
            return self._accept_fact(
                "policies",
                {
                    "kind": safe_kind,
                    "policy_identity": self._checked_identity(
                        policy_identity, field="policy identity"
                    ),
                },
            )
        except RuntimeTraceError:
            self._mark_unsupported("policy_identity")
            return False
        except BaseException:
            self._mark_internal_failure("record_policy")
            return False

    def record_randomness_policy(self, policy_identity: str) -> bool:
        return self.record_policy("randomness", policy_identity)

    def record_clock_policy(self, policy_identity: str) -> bool:
        return self.record_policy("clock", policy_identity)

    def record_capability(
        self, capability: str, *, adapter_identity: str, state_identity: str
    ) -> bool:
        try:
            return self._accept_fact(
                "capabilities",
                {
                    "capability": self._checked_name(capability, field="capability name"),
                    "adapter_identity": self._checked_identity(
                        adapter_identity, field="adapter identity"
                    ),
                    "state_identity": self._checked_identity(
                        state_identity, field="state identity"
                    ),
                },
            )
        except RuntimeTraceError:
            self._mark_unsupported("capability_adapter")
            return False
        except BaseException:
            self._mark_internal_failure("record_capability")
            return False

    record_hardware_capability = record_capability

    def record_unsupported_event(self, kind: str = "explicit") -> bool:
        del kind  # Attacker-controlled event labels are never retained.
        self._mark_unsupported("explicit")
        return False

    def record_private_event(self, kind: str = "explicit") -> bool:
        del kind
        self._mark_private("explicit")
        return False

    def observe_audit_event(self, event: str, arguments: tuple[Any, ...] = ()) -> None:
        """Test/adapter entry point with unsupported-event fail-closed behavior."""

        try:
            self._observe_audit_event(event, arguments, synthetic=True)
        except BaseException:
            self._mark_internal_failure("observe_audit")

    def _observe_audit_event(self, event: Any, arguments: Any, *, synthetic: bool) -> None:
        del synthetic
        if not self._active or getattr(self._inside_callback, "active", False):
            return
        self._inside_callback.active = True
        try:
            if type(event) is not str or type(arguments) is not tuple:
                self._mark_unsupported("malformed_audit_event")
                return
            if event == "open":
                path = arguments[0] if arguments else None
                mode = arguments[1] if len(arguments) > 1 else "r"
                flags = arguments[2] if len(arguments) > 2 else 0
                writes = False
                if isinstance(mode, str):
                    writes = any(marker in mode for marker in ("w", "a", "+", "x"))
                elif isinstance(flags, int):
                    writes = (flags & os.O_ACCMODE) != os.O_RDONLY
                if writes:
                    self._mark_unsupported("file_write")
                else:
                    self.record_file_read(path)
                return
            if event == "import":
                name = arguments[0] if arguments else ""
                filename = arguments[1] if len(arguments) > 1 else None
                native = isinstance(filename, str) and filename.lower().endswith(
                    (".so", ".pyd", ".dll", ".dylib")
                )
                if isinstance(filename, str) and filename:
                    self.record_module(name, source_path=filename, native=native)
                elif type(name) is str:
                    # The initial import-search audit event has no origin yet.
                    self.record_module(name)
                else:
                    self._mark_unsupported("malformed_import")
                return
            if event == "subprocess.Popen":
                executable = arguments[0] if arguments else ""
                argv = arguments[1] if len(arguments) > 1 else ()
                self.record_subprocess(executable, argv)
                return
            if event == "exec" and arguments and isinstance(arguments[0], CodeType):
                self._record_code_object(arguments[0])
                return

            if event.startswith("socket."):
                self._mark_unsupported("network")
                return
            if event.startswith("ctypes."):
                self._mark_unsupported("native_code")
                return
            if event in {"os.system", "os.posix_spawn", "os.posix_spawnp"} or event.startswith(
                "os.exec"
            ):
                self._mark_unsupported("process")
                return
            if event in {
                "os.listdir",
                "os.scandir",
                "os.remove",
                "os.rename",
                "os.rmdir",
                "os.mkdir",
                "os.chdir",
                "builtins.input",
            }:
                self._mark_unsupported("filesystem_effect")
                return
            if event in _IGNORED_NO_EFFECT_AUDIT_EVENTS:
                return
            # Completeness is closed: an audit event outside the explicitly
            # supported vocabulary cannot be silently treated as irrelevant.
            self._mark_unsupported("unknown_audit_event")
        finally:
            self._inside_callback.active = False

    def _dependencies_dict(self) -> dict[str, list[dict[str, Any]]]:
        with self._lock:
            return {
                kind: [self._facts[kind][key] for key in sorted(self._facts[kind])]
                for kind in _DEPENDENCY_KINDS
            }

    def _payload(self) -> dict[str, Any]:
        reasons = set(self._reasons)
        if not self._audit_hook_healthy:
            reasons.add("instrumentation_failure")
        if self.capture_code_objects and not self._profile_healthy:
            reasons.add("instrumentation_failure")
        if not self._instrumentation_cid:
            reasons.add("instrumentation_failure")
        dependencies = self._dependencies_dict()
        recorded = sum(len(items) for items in dependencies.values())
        completeness = (
            RuntimeTraceCompleteness.COMPLETE
            if not reasons
            else RuntimeTraceCompleteness.INCOMPLETE
        )
        return {
            "schema": RUNTIME_TEST_DEPENDENCY_TRACE_SCHEMA,
            "interface": RUNTIME_TEST_DEPENDENCY_TRACE_INTERFACE,
            "eligibility_profile": self.eligibility_profile,
            "completeness": {
                "status": completeness.value,
                "complete": completeness.complete,
                "reasons": sorted(reasons),
            },
            "instrumentation": self._instrumentation,
            "instrumentation_cid": self._instrumentation_cid,
            "limits": self.limits.to_dict(),
            "dependencies": dependencies,
            "health": {
                "audit_hook_healthy": self._audit_hook_healthy,
                "profile_healthy": self._profile_healthy,
                "started": self._started,
                "stopped": self._stopped,
                "observed_event_count": self._observed_event_count,
                "recorded_fact_count": recorded,
                "dropped_event_count": self._dropped_event_count,
                "unsupported_event_kinds": sorted(self._unsupported_kinds),
                "private_event_kinds": sorted(self._private_kinds),
                "internal_failure_kinds": sorted(self._internal_failure_kinds),
            },
        }

    def _build_result(self) -> RuntimeTestDependencyTrace:
        payload = self._payload()
        identity = self._mint_internal_identity(payload)
        canonical = identity.canonical_bytes
        status = RuntimeTraceCompleteness(payload["completeness"]["status"])
        return RuntimeTestDependencyTrace(
            content_identity=identity,
            retained_canonical_bytes=canonical,
            completeness=status,
            completeness_reasons=tuple(payload["completeness"]["reasons"]),
            observed_event_count=payload["health"]["observed_event_count"],
            recorded_fact_count=payload["health"]["recorded_fact_count"],
            dropped_event_count=payload["health"]["dropped_event_count"],
        )

    def _build_fallback_result(self) -> RuntimeTestDependencyTrace:
        try:
            payload = self._payload()
            payload["completeness"] = {
                "status": RuntimeTraceCompleteness.INCOMPLETE.value,
                "complete": False,
                "reasons": sorted(
                    set(payload["completeness"]["reasons"]) | {"instrumentation_failure"}
                ),
            }
            canonical = canonical_json_bytes(payload)
        except BaseException:
            payload = {
                "schema": RUNTIME_TEST_DEPENDENCY_TRACE_SCHEMA,
                "interface": RUNTIME_TEST_DEPENDENCY_TRACE_INTERFACE,
                "completeness": {
                    "status": "incomplete",
                    "complete": False,
                    "reasons": ["instrumentation_failure"],
                },
            }
            canonical = canonical_json_bytes(payload)
        return RuntimeTestDependencyTrace(
            content_identity=None,
            retained_canonical_bytes=canonical,
            completeness=RuntimeTraceCompleteness.INCOMPLETE,
            completeness_reasons=tuple(payload["completeness"]["reasons"]),
            observed_event_count=min(_MAX_COUNTER, self._observed_event_count),
            recorded_fact_count=0,
            dropped_event_count=min(_MAX_COUNTER, self._dropped_event_count),
        )


def trace_runtime_dependencies(
    operation: Callable[..., Any],
    *args: Any,
    tracer: RuntimeTestDependencyTracer | None = None,
    **kwargs: Any,
) -> tuple[Any, RuntimeTestDependencyTrace]:
    """Run ``operation`` and return its value plus trace.

    Exceptions from ``operation`` propagate unchanged after tracing is stopped;
    callers creating pass receipts use this helper only for successful runs.
    """

    active = tracer or RuntimeTestDependencyTracer()
    active.start()
    try:
        value = operation(*args, **kwargs)
    finally:
        result = active.stop()
    return value, result


__all__ = [
    "RUNTIME_TEST_DEPENDENCY_TRACE_INTERFACE",
    "RUNTIME_TEST_DEPENDENCY_TRACER_INTERFACE",
    "RUNTIME_TRACE_INSTRUMENTATION_INTERFACE",
    "RUNTIME_TRACE_LIMITS_INTERFACE",
    "RUNTIME_TEST_DEPENDENCY_TRACE_SCHEMA",
    "RUNTIME_TRACE_INSTRUMENTATION_SCHEMA",
    "RUNTIME_TRACE_LIMITS_SCHEMA",
    "RuntimeTraceCompleteness",
    "RuntimeTraceError",
    "RuntimeTraceLimits",
    "RuntimeTestDependencyTrace",
    "RuntimeTestDependencyTracer",
    "trace_runtime_dependencies",
]
