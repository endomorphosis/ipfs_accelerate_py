"""Runtime dependency tracer for controlled preflight evidence (PTR-146 surface)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType, TracebackType
from typing import Any, ClassVar, Final

from ipfs_accelerate_py.agent_supervisor.analysis.test_execution_identity import (
    mint_content_identity,
)

RUNTIME_TEST_DEPENDENCY_TRACE_INTERFACE: Final = "RuntimeTestDependencyTrace@1"
RUNTIME_TEST_DEPENDENCY_TRACER_INTERFACE: Final = "RuntimeTestDependencyTracer@1"
RUNTIME_TEST_DEPENDENCY_TRACE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/runtime-test-dependency-trace@1"
)


@dataclass(frozen=True, slots=True)
class RuntimeTestDependencyTrace:
    """Content-addressed runtime dependency evidence for one test invocation."""

    __test__: ClassVar[bool] = False
    interface: ClassVar[str] = RUNTIME_TEST_DEPENDENCY_TRACE_INTERFACE

    complete: bool
    trace_cid: str
    completeness_reasons: tuple[str, ...] = ()
    retained_canonical_bytes: bytes = b""
    eligibility_profile: str = "pure"
    dependencies: Mapping[str, Any] = field(default_factory=dict)

    def verify(self) -> None:
        if not self.trace_cid:
            raise ValueError("runtime trace is missing trace_cid")
        if self.retained_canonical_bytes:
            expected = mint_content_identity(
                # Re-parse retained bytes through JSON identity mint path.
                __import__("json").loads(
                    self.retained_canonical_bytes.decode("utf-8")
                )
            )
            if expected.cid != self.trace_cid:
                raise ValueError("runtime trace CID does not match retained bytes")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RUNTIME_TEST_DEPENDENCY_TRACE_SCHEMA,
            "interface": RUNTIME_TEST_DEPENDENCY_TRACE_INTERFACE,
            "complete": self.complete,
            "trace_cid": self.trace_cid,
            "completeness_reasons": list(self.completeness_reasons),
            "eligibility_profile": self.eligibility_profile,
            "dependencies": dict(self.dependencies),
        }


class RuntimeTestDependencyTracer:
    """Context-managed runtime observer used for controlled preflight traces.

    This production default records an empty pure profile when no impure
    activity is observed.  Callers that need deeper instrumentation inject a
    custom tracer factory through the lifecycle/plugin seam.
    """

    __test__: ClassVar[bool] = False
    interface: ClassVar[str] = RUNTIME_TEST_DEPENDENCY_TRACER_INTERFACE

    def __init__(
        self,
        *,
        allowed_roots: Mapping[str, Any] | None = None,
        capture_code_objects: bool = False,
    ) -> None:
        self.allowed_roots = {
            str(key): Path(value)
            for key, value in dict(allowed_roots or {}).items()
        }
        self.capture_code_objects = bool(capture_code_objects)
        self._active = False
        self._result: RuntimeTestDependencyTrace | None = None
        self._events: list[dict[str, Any]] = []

    @property
    def result(self) -> RuntimeTestDependencyTrace | None:
        return self._result

    def __enter__(self) -> "RuntimeTestDependencyTracer":
        self._active = True
        self._events = []
        self._result = None
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        del tb
        self._active = False
        reasons: list[str] = []
        if exc_type is not None:
            reasons.append(f"preflight_exception:{exc_type.__name__}")
        profile = "pure"
        deps = {
            "services": [],
            "capabilities": [],
            "subprocesses": [],
            "environment": [],
            "policies": [],
            "events": list(self._events),
            "allowed_roots": {
                key: str(path) for key, path in self.allowed_roots.items()
            },
            "capture_code_objects": self.capture_code_objects,
        }
        payload = {
            "schema": RUNTIME_TEST_DEPENDENCY_TRACE_SCHEMA,
            "interface": RUNTIME_TEST_DEPENDENCY_TRACE_INTERFACE,
            "eligibility_profile": profile,
            "completeness_reasons": reasons,
            "dependencies": deps,
        }
        try:
            identity = mint_content_identity(payload)
            complete = not reasons
            self._result = RuntimeTestDependencyTrace(
                complete=complete,
                trace_cid=identity.cid,
                completeness_reasons=tuple(reasons),
                retained_canonical_bytes=identity.canonical_bytes,
                eligibility_profile=profile,
                dependencies=MappingProxyType(deps),
            )
        except Exception as mint_exc:
            reasons.append(f"runtime_trace_mint_failed:{type(mint_exc).__name__}")
            self._result = RuntimeTestDependencyTrace(
                complete=False,
                trace_cid="",
                completeness_reasons=tuple(reasons),
                eligibility_profile="unknown",
                dependencies=MappingProxyType(deps),
            )
        # Never suppress the original exception.
        return False

    def note(self, event: Mapping[str, Any]) -> None:
        """Optional observer hook for injected instrumentation."""

        if not self._active:
            return
        if isinstance(event, Mapping) and len(self._events) < 256:
            self._events.append(dict(event))


__all__ = (
    "RUNTIME_TEST_DEPENDENCY_TRACE_INTERFACE",
    "RUNTIME_TEST_DEPENDENCY_TRACER_INTERFACE",
    "RuntimeTestDependencyTrace",
    "RuntimeTestDependencyTracer",
)
