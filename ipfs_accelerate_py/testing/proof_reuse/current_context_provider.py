"""Fresh current-context rebuild for warm candidate revalidation (PTR-145).

``CurrentContextProvider@1`` / ``DefaultCurrentContextProvider@1`` rebuild the
live identity used by locator-first warm admission **without** executing
fixtures or the test body.

Rebuilt surfaces (each content-addressed when present):

* AST / source identity
* static dependency trace
* fixtures, hooks, parameters
* repository forest
* dependency locks and installed distributions
* environment, capabilities, platform
* external snapshots
* policy

The retained historical runtime frontier is **not** relabeled as current.
Runtime identity on the rebuilt context is only admitted when a controlled
live rebuild (or an explicit live identity compiler) produces it.  This module
never authorizes ``SKIP``.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final, Optional, Protocol, runtime_checkable

from .activation_contracts import (
    CURRENT_EXECUTION_CONTEXT_INTERFACE,
    CandidateExecutionContext,
    CurrentExecutionContext,
)

# ---------------------------------------------------------------------------
# Interface / schema constants
# ---------------------------------------------------------------------------

CURRENT_CONTEXT_PROVIDER_INTERFACE: Final = "CurrentContextProvider@1"
DEFAULT_CURRENT_CONTEXT_PROVIDER_INTERFACE: Final = "DefaultCurrentContextProvider@1"
CURRENT_CONTEXT_PROVIDER_SCHEMA: Final = (
    "ipfs_accelerate_py/testing/proof-reuse/current-context-provider@1"
)
DEFAULT_CURRENT_CONTEXT_PROVIDER_SCHEMA: Final = (
    "ipfs_accelerate_py/testing/proof-reuse/default-current-context-provider@1"
)
CURRENT_CONTEXT_COMPILE_RESULT_INTERFACE: Final = "CurrentContextCompileResult@1"

_MAX_DIAGNOSTIC_KEYS: Final = 32
_MAX_DIAGNOSTIC_VALUE_CHARS: Final = 256
_MAX_PATH_CHARS: Final = 1_024
_MAX_FILE_BYTES: Final = 8 * 1_048_576

# Identity dimensions the warm path must rebuild from live state.
_REBUILD_DIMENSIONS: Final[tuple[str, ...]] = (
    "test_ast",
    "static_trace",
    "fixtures",
    "hooks",
    "parameters",
    "repository_forest",
    "dependency_lock",
    "installed_distributions",
    "environment",
    "capabilities",
    "platform",
    "external_snapshots",
    "policy",
    "runtime_trace",
    "execution_key",
)


class CurrentContextCompileReason(str, Enum):
    """Closed reason codes for current-context compilation."""

    COMPILED = "compiled"
    LOCATOR_MISSING = "locator_missing"
    LOCATOR_MISMATCH = "locator_mismatch"
    CANDIDATE_INVALID = "candidate_invalid"
    PROVIDER_ABSENT = "provider_absent"
    ITEM_UNAVAILABLE = "item_unavailable"
    IDENTITY_INCOMPLETE = "identity_incomplete"
    DIMENSION_UNAVAILABLE = "dimension_unavailable"
    FIXTURE_EXECUTION_FORBIDDEN = "fixture_execution_forbidden"
    TEST_BODY_EXECUTION_FORBIDDEN = "test_body_execution_forbidden"
    INTERNAL_ERROR = "internal_error"


class CurrentContextCompileResult:
    """Outcome of one current-context rebuild attempt (never skip authority)."""

    __test__: ClassVar[bool] = False

    def __init__(
        self,
        *,
        reason: CurrentContextCompileReason,
        context: CurrentExecutionContext | None = None,
        fixtures_executed: bool = False,
        test_body_executed: bool = False,
        rebuilt_dimensions: Sequence[str] = (),
        diagnostics: Mapping[str, Any] | None = None,
    ) -> None:
        self.reason = (
            reason
            if isinstance(reason, CurrentContextCompileReason)
            else CurrentContextCompileReason(str(reason))
        )
        self.context = context
        self.fixtures_executed = bool(fixtures_executed)
        self.test_body_executed = bool(test_body_executed)
        self.rebuilt_dimensions = tuple(str(item) for item in rebuilt_dimensions)
        self.diagnostics = _bounded_diagnostics(diagnostics)

    @property
    def interface(self) -> str:
        return CURRENT_CONTEXT_COMPILE_RESULT_INTERFACE

    @property
    def compiled(self) -> bool:
        return (
            self.reason is CurrentContextCompileReason.COMPILED
            and isinstance(self.context, CurrentExecutionContext)
            and not self.fixtures_executed
            and not self.test_body_executed
        )

    @property
    def may_authorize_skip(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CURRENT_CONTEXT_PROVIDER_SCHEMA,
            "interface": self.interface,
            "reason": self.reason.value,
            "compiled": self.compiled,
            "fixtures_executed": self.fixtures_executed,
            "test_body_executed": self.test_body_executed,
            "rebuilt_dimensions": list(self.rebuilt_dimensions),
            "may_authorize_skip": False,
            "diagnostics": dict(self.diagnostics),
        }


def _bounded_diagnostics(raw: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not raw:
        return MappingProxyType({})
    out: dict[str, Any] = {}
    for index, (key, value) in enumerate(raw.items()):
        if index >= _MAX_DIAGNOSTIC_KEYS:
            break
        name = str(key)[:64]
        if value is None or isinstance(value, (bool, int)):
            out[name] = value
        elif isinstance(value, str):
            out[name] = value[:_MAX_DIAGNOSTIC_VALUE_CHARS]
        elif isinstance(value, (list, tuple)):
            out[name] = [str(item)[:64] for item in list(value)[:16]]
        else:
            out[name] = type(value).__name__[:64]
    return MappingProxyType(out)


def _now_ms(clock: Callable[[], float] | Callable[[], int] | None = None) -> int:
    if clock is None:
        return int(time.time() * 1000)
    value = clock()
    if isinstance(value, bool):
        return int(time.time() * 1000)
    if isinstance(value, int):
        if value < 10_000_000_000:
            return int(value * 1000)
        return int(value)
    try:
        return int(float(value) * 1000)
    except (TypeError, ValueError):
        return int(time.time() * 1000)


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _locator_token(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    for attr in ("locator_cid", "locator_id", "cid", "content_id"):
        candidate = getattr(value, attr, None)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    if isinstance(value, Mapping):
        for key in ("locator_cid", "locator_id", "cid", "content_id"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    return None


@runtime_checkable
class CurrentContextProvider(Protocol):
    """Compile a fresh CurrentExecutionContext for one locator/candidate pair.

    Must not execute fixtures or the test body.  Historical traces must not be
    relabeled as current (``rebuild_source`` stays fresh).
    """

    def compile_current(
        self,
        *,
        locator_cid: str,
        candidate: CandidateExecutionContext,
        component_bytes: Mapping[str, bytes],
    ) -> CurrentExecutionContext | None:
        ...


# Type for injectable live-identity compilers.
LiveIdentityCompiler = Callable[
    ...,
    Mapping[str, Any] | CurrentExecutionContext | None,
]


@dataclass
class DefaultCurrentContextProvider:
    """Production current-context rebuild used by the two-stage warm path.

    Authority doctrine:

    * Lookup callers bind at most the **locator** and the **current collected
      item** before compilation.
    * Compilation never runs fixtures or the test body.
    * Live identity is preferred over retained candidate bytes; retained bytes
      only name what to compare against later.
    * Incomplete, unavailable, or exceptional rebuilds return ``None`` /
      a non-compiled result so the warm path fails open to ``RUN``.
    * ``may_authorize_skip`` is always false.
    """

    __test__: ClassVar[bool] = False

    identity_services: Any = None
    live_identity_compiler: LiveIdentityCompiler | None = None
    allowed_roots: Mapping[str, str | os.PathLike[str]] = field(default_factory=dict)
    environ: Mapping[str, str] | None = None
    clock: Callable[[], float] | Callable[[], int] | None = None
    rebuild_source: str = "fresh_live_rebuild"
    require_collected_item: bool = False
    # Optional dimension-level overrides: name → callable returning CID or value.
    dimension_rebuilders: Mapping[str, Callable[..., Any]] | None = None

    def __post_init__(self) -> None:
        self._lock = threading.RLock()
        self._collected_item: Any = None
        self._fixtures_executed = False
        self._test_body_executed = False
        self._compile_count = 0
        roots: dict[str, Path] = {}
        for key, value in dict(self.allowed_roots or {}).items():
            name = str(key)[:64]
            if not name:
                continue
            try:
                roots[name] = Path(os.fspath(value)).resolve()
            except (OSError, TypeError, ValueError):
                continue
        self._roots = roots
        self._environ = dict(os.environ if self.environ is None else self.environ)
        source = str(self.rebuild_source or "fresh_live_rebuild")
        if source not in {"fresh_live_rebuild", "controlled_preflight"}:
            source = "fresh_live_rebuild"
        self.rebuild_source = source
        self.dimension_rebuilders = dict(self.dimension_rebuilders or {})

    @property
    def interface(self) -> str:
        return DEFAULT_CURRENT_CONTEXT_PROVIDER_INTERFACE

    @property
    def schema(self) -> str:
        return DEFAULT_CURRENT_CONTEXT_PROVIDER_SCHEMA

    @property
    def may_authorize_skip(self) -> bool:
        return False

    @property
    def fixtures_executed(self) -> bool:
        return self._fixtures_executed

    @property
    def test_body_executed(self) -> bool:
        return self._test_body_executed

    @property
    def compile_count(self) -> int:
        return self._compile_count

    @property
    def collected_item(self) -> Any:
        return self._collected_item

    def bind_collected_item(self, item: Any) -> None:
        """Bind the current collected pytest item for the next compile.

        Warm lookup begins with locator + collected item only.  Binding does
        not execute fixtures or the test body.
        """

        with self._lock:
            self._collected_item = item

    def clear_collected_item(self) -> None:
        with self._lock:
            self._collected_item = None

    def note_fixture_execution_forbidden(self) -> None:
        """Diagnostic fence: providers must never call this on the warm path."""

        with self._lock:
            self._fixtures_executed = True

    def note_test_body_execution_forbidden(self) -> None:
        with self._lock:
            self._test_body_executed = True

    def compile_current(
        self,
        *,
        locator_cid: str,
        candidate: CandidateExecutionContext,
        component_bytes: Mapping[str, bytes],
        item: Any = None,
    ) -> CurrentExecutionContext | None:
        """Rebuild a fresh current context or return ``None`` (fail open to RUN).

        Never executes fixtures or the test body.  Accepts an optional ``item``
        override; otherwise uses the bound collected item.
        """

        result = self.compile_current_result(
            locator_cid=locator_cid,
            candidate=candidate,
            component_bytes=component_bytes,
            item=item,
        )
        return result.context if result.compiled else None

    def compile_current_result(
        self,
        *,
        locator_cid: str,
        candidate: CandidateExecutionContext,
        component_bytes: Mapping[str, bytes],
        item: Any = None,
    ) -> CurrentContextCompileResult:
        """Full typed compile outcome for diagnostics and tests."""

        with self._lock:
            self._compile_count += 1
            # Reset execution fences each compile; only live code can set them.
            self._fixtures_executed = False
            self._test_body_executed = False

        try:
            return self._compile_inner(
                locator_cid=locator_cid,
                candidate=candidate,
                component_bytes=component_bytes,
                item=item,
            )
        except Exception as exc:  # noqa: BLE001 - fail open
            return CurrentContextCompileResult(
                reason=CurrentContextCompileReason.INTERNAL_ERROR,
                diagnostics={
                    "stage": "compile",
                    "error": type(exc).__name__[:64],
                },
            )

    def _compile_inner(
        self,
        *,
        locator_cid: str,
        candidate: CandidateExecutionContext,
        component_bytes: Mapping[str, bytes],
        item: Any,
    ) -> CurrentContextCompileResult:
        locator = str(locator_cid or "").strip()
        if not locator:
            return CurrentContextCompileResult(
                reason=CurrentContextCompileReason.LOCATOR_MISSING,
                diagnostics={"stage": "locator"},
            )
        if not isinstance(candidate, CandidateExecutionContext):
            return CurrentContextCompileResult(
                reason=CurrentContextCompileReason.CANDIDATE_INVALID,
                diagnostics={"stage": "candidate_type"},
            )
        if candidate.locator_cid and candidate.locator_cid != locator:
            return CurrentContextCompileResult(
                reason=CurrentContextCompileReason.LOCATOR_MISMATCH,
                diagnostics={
                    "stage": "locator_mismatch",
                    "candidate_locator": candidate.locator_cid[:128],
                },
            )

        collected = item if item is not None else self._collected_item
        if self.require_collected_item and collected is None:
            return CurrentContextCompileResult(
                reason=CurrentContextCompileReason.ITEM_UNAVAILABLE,
                diagnostics={"stage": "item_required"},
            )

        # Primary path: explicit live identity compiler (session identity /
        # controlled preflight / test injection).  Must not execute fixtures.
        if self.live_identity_compiler is not None:
            return self._compile_from_live_compiler(
                locator_cid=locator,
                candidate=candidate,
                component_bytes=component_bytes,
                item=collected,
            )

        # Identity-services path: rebuild static surfaces from session defaults.
        if self.identity_services is not None:
            return self._compile_from_identity_services(
                locator_cid=locator,
                candidate=candidate,
                component_bytes=component_bytes,
                item=collected,
            )

        # Dimension rebuilder path: per-field live CID producers.
        if self.dimension_rebuilders:
            return self._compile_from_dimension_rebuilders(
                locator_cid=locator,
                candidate=candidate,
                component_bytes=component_bytes,
                item=collected,
            )

        # Live filesystem/env fingerprints under admitted roots when available —
        # still never runs fixtures/body.  Without a complete identity surface
        # this remains incomplete so warm admission fails open to RUN.
        return self._compile_from_live_roots(
            locator_cid=locator,
            candidate=candidate,
            component_bytes=component_bytes,
            item=collected,
        )

    def _compile_from_live_compiler(
        self,
        *,
        locator_cid: str,
        candidate: CandidateExecutionContext,
        component_bytes: Mapping[str, bytes],
        item: Any,
    ) -> CurrentContextCompileResult:
        compiler = self.live_identity_compiler
        assert compiler is not None
        try:
            produced = compiler(
                locator_cid=locator_cid,
                candidate=candidate,
                component_bytes=dict(component_bytes or {}),
                item=item,
                provider=self,
            )
        except TypeError:
            try:
                produced = compiler(
                    locator_cid=locator_cid,
                    candidate=candidate,
                    component_bytes=dict(component_bytes or {}),
                    item=item,
                )
            except TypeError:
                produced = compiler(candidate)  # type: ignore[misc]
        except Exception as exc:  # noqa: BLE001
            return CurrentContextCompileResult(
                reason=CurrentContextCompileReason.INTERNAL_ERROR,
                diagnostics={
                    "stage": "live_compiler",
                    "error": type(exc).__name__[:64],
                },
            )

        if self._fixtures_executed:
            return CurrentContextCompileResult(
                reason=CurrentContextCompileReason.FIXTURE_EXECUTION_FORBIDDEN,
                fixtures_executed=True,
                diagnostics={"stage": "fixtures_executed"},
            )
        if self._test_body_executed:
            return CurrentContextCompileResult(
                reason=CurrentContextCompileReason.TEST_BODY_EXECUTION_FORBIDDEN,
                test_body_executed=True,
                diagnostics={"stage": "test_body_executed"},
            )

        if isinstance(produced, CurrentExecutionContext):
            if produced.locator_cid and produced.locator_cid != locator_cid:
                return CurrentContextCompileResult(
                    reason=CurrentContextCompileReason.LOCATOR_MISMATCH,
                    diagnostics={"stage": "live_context_locator"},
                )
            if produced.rebuild_source not in {
                "fresh_live_rebuild",
                "controlled_preflight",
            }:
                return CurrentContextCompileResult(
                    reason=CurrentContextCompileReason.IDENTITY_INCOMPLETE,
                    diagnostics={
                        "stage": "rebuild_source",
                        "rebuild_source": produced.rebuild_source,
                    },
                )
            return CurrentContextCompileResult(
                reason=CurrentContextCompileReason.COMPILED,
                context=produced,
                rebuilt_dimensions=_REBUILD_DIMENSIONS,
                diagnostics={"stage": "live_compiler_context"},
            )

        if isinstance(produced, Mapping):
            return self._context_from_identity_map(
                locator_cid=locator_cid,
                candidate=candidate,
                identity_map=produced,
                stage="live_compiler_map",
            )

        return CurrentContextCompileResult(
            reason=CurrentContextCompileReason.IDENTITY_INCOMPLETE,
            diagnostics={"stage": "live_compiler_empty"},
        )

    def _compile_from_identity_services(
        self,
        *,
        locator_cid: str,
        candidate: CandidateExecutionContext,
        component_bytes: Mapping[str, bytes],
        item: Any,
    ) -> CurrentContextCompileResult:
        services = self.identity_services
        identity_map: dict[str, Any] = {}
        rebuilt: list[str] = []

        # Prefer a dedicated current-context compiler when the session provides one.
        for name in (
            "compile_current_context",
            "rebuild_current_context",
            "current_context",
        ):
            method = getattr(services, name, None)
            if not callable(method):
                continue
            try:
                produced = method(
                    item=item,
                    locator_cid=locator_cid,
                    candidate=candidate,
                    component_bytes=dict(component_bytes or {}),
                )
            except TypeError:
                try:
                    produced = method(item, locator_cid=locator_cid)
                except Exception as exc:  # noqa: BLE001
                    return CurrentContextCompileResult(
                        reason=CurrentContextCompileReason.INTERNAL_ERROR,
                        diagnostics={
                            "stage": f"identity_services.{name}",
                            "error": type(exc).__name__[:64],
                        },
                    )
            except Exception as exc:  # noqa: BLE001
                return CurrentContextCompileResult(
                    reason=CurrentContextCompileReason.INTERNAL_ERROR,
                    diagnostics={
                        "stage": f"identity_services.{name}",
                        "error": type(exc).__name__[:64],
                    },
                )
            if isinstance(produced, CurrentExecutionContext):
                return CurrentContextCompileResult(
                    reason=CurrentContextCompileReason.COMPILED,
                    context=produced,
                    rebuilt_dimensions=_REBUILD_DIMENSIONS,
                    diagnostics={"stage": f"identity_services.{name}"},
                )
            if isinstance(produced, Mapping):
                return self._context_from_identity_map(
                    locator_cid=locator_cid,
                    candidate=candidate,
                    identity_map=produced,
                    stage=f"identity_services.{name}",
                )

        # Assemble static surfaces from individual providers when available.
        provider_fields = {
            "repository_forest_cid": (
                "repository_forest_provider",
                "repository_forest_cid",
                "forest_id",
            ),
            "test_ast_cid": ("ast_provider", "test_ast_cid", "ast_cid"),
            "static_trace_root_cid": (
                "static_trace_provider",
                "static_trace_root_cid",
                "trace_cid",
            ),
            "environment_cid": ("environment_provider", "environment_cid"),
            "policy_cid": ("policy_provider", "policy_cid"),
            "dependency_lock_cid": ("lock_provider", "dependency_lock_cid"),
            "installed_distributions_cid": (
                "distribution_provider",
                "installed_distributions_cid",
            ),
            "capability_root_cid": ("capability_provider", "capability_root_cid"),
            "platform_cid": ("platform_provider", "platform_cid"),
            "runtime_trace_root_cid": (
                "runtime_preflight_provider",
                "runtime_trace_root_cid",
            ),
            "execution_key_cid": ("execution_key_provider", "execution_key_cid"),
        }
        for field_name, attrs in provider_fields.items():
            value = self._invoke_identity_provider(services, attrs, item=item)
            if value:
                identity_map[field_name] = value
                rebuilt.append(field_name.replace("_cid", "").replace("_root", ""))

        component_keys = (
            "fixtures",
            "hooks",
            "parameters",
            "source",
            "test_ast",
            "static_trace",
            "runtime_trace",
            "repository_forest",
            "environment",
            "policy",
        )
        components: dict[str, str] = {}
        for key in component_keys:
            provider_name = f"{key}_provider"
            value = self._invoke_identity_provider(
                services, (provider_name, f"{key}_cid", key), item=item
            )
            if value:
                components[key] = value
                if key not in rebuilt:
                    rebuilt.append(key)
        if components:
            identity_map["component_cids"] = components

        if not identity_map:
            return CurrentContextCompileResult(
                reason=CurrentContextCompileReason.PROVIDER_ABSENT
                if services is None
                else CurrentContextCompileReason.IDENTITY_INCOMPLETE,
                diagnostics={"stage": "identity_services_empty"},
            )
        return self._context_from_identity_map(
            locator_cid=locator_cid,
            candidate=candidate,
            identity_map=identity_map,
            stage="identity_services_fields",
            rebuilt=rebuilt,
        )

    def _invoke_identity_provider(
        self,
        services: Any,
        attrs: Sequence[str],
        *,
        item: Any,
    ) -> str:
        for attr in attrs:
            provider = getattr(services, attr, None)
            if provider is None:
                continue
            if callable(provider):
                try:
                    value = provider(item)
                except TypeError:
                    try:
                        value = provider()
                    except Exception:
                        continue
                except Exception:
                    continue
            else:
                value = provider
            if isinstance(value, str) and value.strip():
                return value.strip()
            token = _locator_token(value)
            if token:
                return token
            cid = getattr(value, "cid", None) or getattr(value, "content_id", None)
            if isinstance(cid, str) and cid.strip():
                return cid.strip()
        return ""

    def _compile_from_dimension_rebuilders(
        self,
        *,
        locator_cid: str,
        candidate: CandidateExecutionContext,
        component_bytes: Mapping[str, bytes],
        item: Any,
    ) -> CurrentContextCompileResult:
        identity_map: dict[str, Any] = {}
        components: dict[str, str] = {}
        rebuilt: list[str] = []
        rebuilders = dict(self.dimension_rebuilders or {})

        field_map = {
            "test_ast": "test_ast_cid",
            "static_trace": "static_trace_root_cid",
            "runtime_trace": "runtime_trace_root_cid",
            "repository_forest": "repository_forest_cid",
            "environment": "environment_cid",
            "policy": "policy_cid",
            "dependency_lock": "dependency_lock_cid",
            "installed_distributions": "installed_distributions_cid",
            "capabilities": "capability_root_cid",
            "platform": "platform_cid",
            "execution_key": "execution_key_cid",
        }
        for dimension, field_name in field_map.items():
            rebuilder = rebuilders.get(dimension)
            if rebuilder is None:
                continue
            try:
                value = rebuilder(
                    locator_cid=locator_cid,
                    candidate=candidate,
                    component_bytes=component_bytes,
                    item=item,
                )
            except TypeError:
                try:
                    value = rebuilder(candidate)
                except Exception as exc:  # noqa: BLE001
                    return CurrentContextCompileResult(
                        reason=CurrentContextCompileReason.INTERNAL_ERROR,
                        diagnostics={
                            "stage": f"rebuilder.{dimension}",
                            "error": type(exc).__name__[:64],
                        },
                    )
            except Exception as exc:  # noqa: BLE001
                return CurrentContextCompileResult(
                    reason=CurrentContextCompileReason.INTERNAL_ERROR,
                    diagnostics={
                        "stage": f"rebuilder.{dimension}",
                        "error": type(exc).__name__[:64],
                    },
                )
            token = value if isinstance(value, str) else _locator_token(value)
            if token:
                identity_map[field_name] = token
                rebuilt.append(dimension)

        for dimension in ("fixtures", "hooks", "parameters", "source"):
            rebuilder = rebuilders.get(dimension)
            if rebuilder is None:
                continue
            try:
                value = rebuilder(
                    locator_cid=locator_cid,
                    candidate=candidate,
                    component_bytes=component_bytes,
                    item=item,
                )
            except Exception:
                continue
            token = value if isinstance(value, str) else _locator_token(value)
            if token:
                components[dimension] = token
                rebuilt.append(dimension)

        snapshots_rebuilder = rebuilders.get("external_snapshots")
        if snapshots_rebuilder is not None:
            try:
                value = snapshots_rebuilder(
                    locator_cid=locator_cid,
                    candidate=candidate,
                    component_bytes=component_bytes,
                    item=item,
                )
            except Exception:
                value = None
            if isinstance(value, (list, tuple)):
                identity_map["external_snapshot_cids"] = tuple(
                    str(item) for item in value if item
                )
                rebuilt.append("external_snapshots")
            elif isinstance(value, str) and value:
                identity_map["external_snapshot_cids"] = (value,)
                rebuilt.append("external_snapshots")

        if components:
            identity_map["component_cids"] = components

        if not identity_map:
            return CurrentContextCompileResult(
                reason=CurrentContextCompileReason.DIMENSION_UNAVAILABLE,
                diagnostics={"stage": "dimension_rebuilders_empty"},
            )
        return self._context_from_identity_map(
            locator_cid=locator_cid,
            candidate=candidate,
            identity_map=identity_map,
            stage="dimension_rebuilders",
            rebuilt=rebuilt,
        )

    def _compile_from_live_roots(
        self,
        *,
        locator_cid: str,
        candidate: CandidateExecutionContext,
        component_bytes: Mapping[str, bytes],
        item: Any,
    ) -> CurrentContextCompileResult:
        """Best-effort live fingerprints when no identity compiler is wired.

        Without a complete identity surface the result is intentionally
        incomplete so warm admission returns RUN rather than inventing CIDs
        from retained candidate bytes.
        """

        del component_bytes  # retained bytes are never relabeled as current
        live_bits: dict[str, str] = {}
        if self._roots:
            digests: list[str] = []
            for root_id, root in sorted(self._roots.items()):
                try:
                    marker = f"{root_id}:{root.as_posix()}:{root.exists()}"
                    digests.append(_sha256_hex(marker.encode("utf-8")))
                except OSError:
                    continue
            if digests:
                live_bits["repository_forest_cid"] = _sha256_hex(
                    "|".join(digests).encode("utf-8")
                )
        if self._environ:
            # Only fingerprint the allowlist shape: sorted key names, not values
            # with secrets. Full env content-addressing is done by identity
            # services with an explicit allowlist.
            keys = sorted(str(key) for key in self._environ.keys())[:256]
            live_bits["environment_shape"] = _sha256_hex(
                "\0".join(keys).encode("utf-8")
            )

        # Item path fingerprint when available (source/AST surface).
        item_path = None
        if item is not None:
            raw = getattr(item, "path", None) or getattr(item, "fspath", None)
            if raw is not None:
                try:
                    item_path = Path(os.fspath(raw))
                    if item_path.is_file() and item_path.stat().st_size <= _MAX_FILE_BYTES:
                        live_bits["test_ast_cid"] = _sha256_hex(item_path.read_bytes())
                except OSError:
                    pass

        if not live_bits:
            return CurrentContextCompileResult(
                reason=CurrentContextCompileReason.PROVIDER_ABSENT,
                diagnostics={
                    "stage": "live_roots_empty",
                    "has_item": item is not None,
                    "root_count": len(self._roots),
                },
            )
        # Incomplete identity — do not fabricate a full CurrentExecutionContext
        # from partial fingerprints (would either mismatch or falsely match).
        return CurrentContextCompileResult(
            reason=CurrentContextCompileReason.IDENTITY_INCOMPLETE,
            rebuilt_dimensions=tuple(live_bits.keys()),
            diagnostics={
                "stage": "live_roots_partial",
                "partial_keys": list(live_bits.keys())[:16],
                "item_path": (
                    str(item_path)[:_MAX_PATH_CHARS] if item_path is not None else ""
                ),
            },
        )

    def _context_from_identity_map(
        self,
        *,
        locator_cid: str,
        candidate: CandidateExecutionContext,
        identity_map: Mapping[str, Any],
        stage: str,
        rebuilt: Sequence[str] | None = None,
    ) -> CurrentContextCompileResult:
        """Materialize CurrentExecutionContext from a live identity map.

        Missing dimensions fall back to empty (not candidate values) so a
        partial rebuild cannot silently inherit retained candidate identity.
        Callers that need matching CIDs must supply them from live compilers.
        """

        def _pick(key: str, *aliases: str) -> str:
            for name in (key, *aliases):
                value = identity_map.get(name)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            return ""

        components_raw = identity_map.get("component_cids") or identity_map.get(
            "components"
        )
        components: dict[str, str] = {}
        if isinstance(components_raw, Mapping):
            for key, value in components_raw.items():
                if isinstance(value, str) and value.strip():
                    components[str(key)] = value.strip()

        # Promote well-known field CIDs into component_cids for extended compare.
        for key, field_name in (
            ("test_ast", "test_ast_cid"),
            ("static_trace", "static_trace_root_cid"),
            ("runtime_trace", "runtime_trace_root_cid"),
            ("repository_forest", "repository_forest_cid"),
            ("environment", "environment_cid"),
            ("policy", "policy_cid"),
            ("fixtures", "fixtures"),
            ("hooks", "hooks"),
            ("parameters", "parameters"),
            ("source", "source"),
        ):
            if key not in components:
                token = _pick(field_name, key, f"{key}_cid")
                if token:
                    components[key] = token

        snapshots = identity_map.get("external_snapshot_cids") or identity_map.get(
            "external_snapshots"
        )
        if isinstance(snapshots, (list, tuple)):
            external = tuple(str(item) for item in snapshots if item)
        elif isinstance(snapshots, str) and snapshots:
            external = (snapshots,)
        else:
            external = ()

        # Required fields: use live values only. Empty required fields make
        # CurrentExecutionContext construction fail → incomplete.
        test_ast = _pick("test_ast_cid", "test_ast")
        static_trace = _pick("static_trace_root_cid", "static_trace")
        runtime_trace = _pick("runtime_trace_root_cid", "runtime_trace")
        forest = _pick("repository_forest_cid", "repository_forest")
        environment = _pick("environment_cid", "environment")
        policy = _pick("policy_cid", "policy")
        execution_key = _pick("execution_key_cid", "execution_key")

        required = {
            "test_ast_cid": test_ast,
            "static_trace_root_cid": static_trace,
            "runtime_trace_root_cid": runtime_trace,
            "repository_forest_cid": forest,
            "environment_cid": environment,
            "policy_cid": policy,
            "execution_key_cid": execution_key,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            return CurrentContextCompileResult(
                reason=CurrentContextCompileReason.IDENTITY_INCOMPLETE,
                rebuilt_dimensions=tuple(rebuilt or ()),
                diagnostics={
                    "stage": stage,
                    "missing_required": missing[:16],
                },
            )

        try:
            context = CurrentExecutionContext(
                locator_cid=locator_cid,
                execution_key_cid=execution_key,
                repository_forest_cid=forest,
                test_ast_cid=test_ast,
                static_trace_root_cid=static_trace,
                runtime_trace_root_cid=runtime_trace,
                environment_cid=environment,
                policy_cid=policy,
                dependency_lock_cid=_pick(
                    "dependency_lock_cid", "dependency_lock", "locks"
                ),
                installed_distributions_cid=_pick(
                    "installed_distributions_cid",
                    "installed_distributions",
                    "distributions",
                ),
                platform_cid=_pick("platform_cid", "platform"),
                capability_root_cid=_pick(
                    "capability_root_cid", "capabilities", "capability_root"
                ),
                component_cids=components,
                external_snapshot_cids=external,
                rebuild_source=self.rebuild_source,
                rebuilt_at_ms=_now_ms(self.clock),
                metadata={
                    "provider_interface": DEFAULT_CURRENT_CONTEXT_PROVIDER_INTERFACE,
                    "compile_stage": stage,
                    "fixtures_executed": False,
                    "test_body_executed": False,
                },
            )
        except Exception as exc:  # noqa: BLE001
            return CurrentContextCompileResult(
                reason=CurrentContextCompileReason.IDENTITY_INCOMPLETE,
                diagnostics={
                    "stage": stage,
                    "error": type(exc).__name__[:64],
                },
            )

        return CurrentContextCompileResult(
            reason=CurrentContextCompileReason.COMPILED,
            context=context,
            rebuilt_dimensions=tuple(rebuilt or _REBUILD_DIMENSIONS),
            diagnostics={"stage": stage},
        )


def build_default_current_context_provider(
    *,
    identity_services: Any = None,
    live_identity_compiler: LiveIdentityCompiler | None = None,
    allowed_roots: Mapping[str, str | os.PathLike[str]] | None = None,
    environ: Mapping[str, str] | None = None,
    clock: Callable[[], float] | Callable[[], int] | None = None,
    rebuild_source: str = "fresh_live_rebuild",
    require_collected_item: bool = False,
    dimension_rebuilders: Mapping[str, Callable[..., Any]] | None = None,
) -> DefaultCurrentContextProvider:
    """Factory for the production current-context provider."""

    return DefaultCurrentContextProvider(
        identity_services=identity_services,
        live_identity_compiler=live_identity_compiler,
        allowed_roots=dict(allowed_roots or {}),
        environ=environ,
        clock=clock,
        rebuild_source=rebuild_source,
        require_collected_item=require_collected_item,
        dimension_rebuilders=dimension_rebuilders,
    )


def current_context_from_candidate_identities(
    candidate: CandidateExecutionContext,
    *,
    locator_cid: str | None = None,
    rebuild_source: str = "fresh_live_rebuild",
    rebuilt_at_ms: int | None = None,
    component_cids: Mapping[str, str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> CurrentExecutionContext:
    """Build a CurrentExecutionContext that mirrors candidate identity fields.

    Intended for live compilers that have already verified every live dimension
    matches the candidate.  It is **not** a historical-trace relabel: callers
    must only invoke this after live resolution confirms agreement.
    """

    if not isinstance(candidate, CandidateExecutionContext):
        raise TypeError("candidate must be CandidateExecutionContext")
    fields: dict[str, Any] = {
        "locator_cid": locator_cid or candidate.locator_cid,
        "execution_key_cid": candidate.execution_key_cid,
        "repository_forest_cid": candidate.repository_forest_cid,
        "test_ast_cid": candidate.test_ast_cid,
        "static_trace_root_cid": candidate.static_trace_root_cid,
        "runtime_trace_root_cid": candidate.runtime_trace_root_cid,
        "environment_cid": candidate.environment_cid,
        "policy_cid": candidate.policy_cid,
        "dependency_lock_cid": candidate.dependency_lock_cid,
        "installed_distributions_cid": candidate.installed_distributions_cid,
        "platform_cid": candidate.platform_cid,
        "capability_root_cid": candidate.capability_root_cid,
        "component_cids": dict(component_cids or candidate.component_cids or {}),
        "external_snapshot_cids": tuple(candidate.external_snapshot_cids or ()),
        "rebuild_source": rebuild_source,
        "rebuilt_at_ms": (
            int(rebuilt_at_ms)
            if rebuilt_at_ms is not None
            else int(time.time() * 1000)
        ),
        "metadata": {
            "provider_interface": DEFAULT_CURRENT_CONTEXT_PROVIDER_INTERFACE,
            "live_match_confirmed": True,
            "fixtures_executed": False,
            "test_body_executed": False,
        },
    }
    if overrides:
        fields.update(dict(overrides))
    return CurrentExecutionContext(**fields)


__all__ = [
    "CURRENT_CONTEXT_COMPILE_RESULT_INTERFACE",
    "CURRENT_CONTEXT_PROVIDER_INTERFACE",
    "CURRENT_CONTEXT_PROVIDER_SCHEMA",
    "CURRENT_EXECUTION_CONTEXT_INTERFACE",
    "CurrentContextCompileReason",
    "CurrentContextCompileResult",
    "CurrentContextProvider",
    "DEFAULT_CURRENT_CONTEXT_PROVIDER_INTERFACE",
    "DEFAULT_CURRENT_CONTEXT_PROVIDER_SCHEMA",
    "DefaultCurrentContextProvider",
    "LiveIdentityCompiler",
    "build_default_current_context_provider",
    "current_context_from_candidate_identities",
]
