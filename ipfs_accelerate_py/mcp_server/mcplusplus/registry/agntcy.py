"""AgntcyAdapter@1 — isolated AGNTCY discovery adapter package.

This module is intentionally **isolated** from the rest of the MCP++ registry
stack. The official AGNTCY SDK is not a required dependency of
``ipfs_accelerate_py``; when the SDK (or a live AGNTCY endpoint) is unavailable
the adapter is marked **unsupported** and every mutator/lookup raises a
**typed reject** (:class:`AgntcyUnsupportedError`).

Live path (optional):
    Set ``MCPPLUSPLUS_AGNTCY_LIVE=1`` and ensure an AGNTCY client module is
    importable (``agntcy`` or ``agntcy_sdk``). When both are present,
    :func:`probe_agntcy_support` reports ``supported=True`` and
    :class:`AgntcyAdapter` may be constructed with ``allow_live=True``.
    Without that opt-in the adapter remains fail-closed unsupported so CI
    stays hermetic.

Blocker (documented):
    No usable official AGNTCY Python SDK is installed in the default runtime
    (``ModuleNotFoundError: agntcy`` / ``agntcy_sdk``). Until a reviewed SDK
    binding exists, this package exists only as an isolation boundary with a
    typed unsupported reject so callers cannot confuse "adapter present" with
    "AGNTCY discovery works".
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Optional, Sequence

from ipfs_accelerate_py.mcp_server.mcplusplus.registry.interface import (
    REGISTRY_INTERFACE,
    Registry,
    RegistryError,
)

AGNTCY_ADAPTER_INTERFACE = "AgntcyAdapter@1"
AGNTCY_PROVIDER_ID = "agntcy"

# Typed reject code (stable wire / log discriminator).
AGNTCY_UNSUPPORTED_CODE = "AGNTCY_UNSUPPORTED"

# Documented blocker identity (stable for tests and operator dashboards).
AGNTCY_BLOCKER_ID = "agntcy-sdk-unavailable"
AGNTCY_BLOCKER_SUMMARY = (
    "Official AGNTCY Python SDK is not installed or not usable in this runtime; "
    "AGNTCY discovery remains isolated and unsupported until a reviewed binding exists."
)

# Env flag to opt into a live AGNTCY probe/path (never on by default).
AGNTCY_LIVE_ENV = "MCPPLUSPLUS_AGNTCY_LIVE"

# Candidate import roots for an optional live SDK.
_AGNTCY_SDK_CANDIDATES = ("agntcy", "agntcy_sdk")


class AgntcyError(RegistryError):
    """Base error for AgntcyAdapter@1."""


class AgntcyUnsupportedError(AgntcyError):
    """Typed reject when AGNTCY discovery is unsupported or unavailable.

    Attributes
    ----------
    code:
        Stable discriminator (``AGNTCY_UNSUPPORTED``).
    reason:
        Human-readable explanation.
    blocker_id:
        Stable blocker id for dashboards / gap tracking.
    support:
        Snapshot of :class:`AgntcySupportStatus` at reject time (as dict).
    """

    def __init__(
        self,
        message: str | None = None,
        *,
        reason: str | None = None,
        blocker_id: str = AGNTCY_BLOCKER_ID,
        support: Mapping[str, Any] | None = None,
    ) -> None:
        self.code = AGNTCY_UNSUPPORTED_CODE
        self.reason = reason or AGNTCY_BLOCKER_SUMMARY
        self.blocker_id = blocker_id
        self.support = dict(support) if support is not None else {}
        text = message or f"{self.code}: {self.reason}"
        super().__init__(text)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the typed reject for logs / API envelopes."""

        return {
            "code": self.code,
            "reason": self.reason,
            "blocker_id": self.blocker_id,
            "provider": AGNTCY_PROVIDER_ID,
            "interface": AGNTCY_ADAPTER_INTERFACE,
            "supported": False,
            "support": dict(self.support),
        }


@dataclass(frozen=True)
class AgntcySupportStatus:
    """Result of probing whether a live AGNTCY path can be enabled."""

    supported: bool
    live_requested: bool
    sdk_available: bool
    sdk_module: Optional[str]
    blocker_id: Optional[str]
    blocker_summary: Optional[str]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "supported": self.supported,
            "live_requested": self.live_requested,
            "sdk_available": self.sdk_available,
            "sdk_module": self.sdk_module,
            "blocker_id": self.blocker_id,
            "blocker_summary": self.blocker_summary,
            "reason": self.reason,
            "provider": AGNTCY_PROVIDER_ID,
            "interface": AGNTCY_ADAPTER_INTERFACE,
        }


def _live_requested(env: Mapping[str, str] | None = None) -> bool:
    source = env if env is not None else os.environ
    raw = str(source.get(AGNTCY_LIVE_ENV, "") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _find_agntcy_sdk() -> tuple[bool, Optional[str]]:
    """Return ``(available, module_name)`` for the first importable SDK root."""

    for name in _AGNTCY_SDK_CANDIDATES:
        try:
            importlib.import_module(name)
        except Exception:
            continue
        return True, name
    return False, None


def probe_agntcy_support(
    *,
    env: Mapping[str, str] | None = None,
    force_sdk_available: Optional[bool] = None,
    force_sdk_module: Optional[str] = None,
) -> AgntcySupportStatus:
    """Probe AGNTCY support without constructing an adapter.

    Parameters
    ----------
    env:
        Optional environment mapping (defaults to ``os.environ``).
    force_sdk_available / force_sdk_module:
        Test-only overrides so hermetic suites can simulate a live SDK without
        installing one.
    """

    live = _live_requested(env)
    if force_sdk_available is not None:
        sdk_ok = bool(force_sdk_available)
        sdk_mod = force_sdk_module if sdk_ok else None
    else:
        sdk_ok, sdk_mod = _find_agntcy_sdk()

    if live and sdk_ok:
        return AgntcySupportStatus(
            supported=True,
            live_requested=True,
            sdk_available=True,
            sdk_module=sdk_mod,
            blocker_id=None,
            blocker_summary=None,
            reason="AGNTCY live path enabled (SDK importable and live flag set)",
        )

    if not sdk_ok:
        reason = AGNTCY_BLOCKER_SUMMARY
        blocker = AGNTCY_BLOCKER_ID
    else:
        reason = (
            f"AGNTCY SDK module {sdk_mod!r} is importable but live path is not "
            f"enabled; set {AGNTCY_LIVE_ENV}=1 to opt in"
        )
        blocker = "agntcy-live-not-enabled"

    return AgntcySupportStatus(
        supported=False,
        live_requested=live,
        sdk_available=sdk_ok,
        sdk_module=sdk_mod,
        blocker_id=blocker,
        blocker_summary=AGNTCY_BLOCKER_SUMMARY if not sdk_ok else reason,
        reason=reason,
    )


def is_agntcy_supported(
    *,
    env: Mapping[str, str] | None = None,
    force_sdk_available: Optional[bool] = None,
) -> bool:
    """Return True only when the optional live AGNTCY path is fully enabled."""

    return probe_agntcy_support(
        env=env, force_sdk_available=force_sdk_available
    ).supported


class AgntcyAdapter(Registry):
    """Isolated AGNTCY Registry@1 adapter.

    Default construction is always fail-closed: every Registry@1 method raises
    :class:`AgntcyUnsupportedError` with a typed reject payload. Callers that
    successfully probe live support may pass ``allow_live=True``; the live
    client path is still a stub until a reviewed SDK binding lands — live mode
    currently raises a typed reject explaining that the binding is incomplete
    (never silently pretends to discover agents).
    """

    def __init__(
        self,
        *,
        allow_live: bool = False,
        env: Mapping[str, str] | None = None,
        force_sdk_available: Optional[bool] = None,
        force_sdk_module: Optional[str] = None,
        support: Optional[AgntcySupportStatus] = None,
    ) -> None:
        self._support = support or probe_agntcy_support(
            env=env,
            force_sdk_available=force_sdk_available,
            force_sdk_module=force_sdk_module,
        )
        self._allow_live = bool(allow_live)
        self._reject_count = 0
        # Live binding is intentionally not implemented: even when support
        # probes green, operations reject with a distinct incomplete-binding
        # reason so we never fabricate discovery results.
        self._live_binding_complete = False

    @property
    def provider_id(self) -> str:
        return AGNTCY_PROVIDER_ID

    @property
    def interface(self) -> str:
        return AGNTCY_ADAPTER_INTERFACE

    @property
    def family_interface(self) -> str:
        return REGISTRY_INTERFACE

    @property
    def supported(self) -> bool:
        return bool(self._support.supported and self._allow_live and self._live_binding_complete)

    @property
    def support_status(self) -> AgntcySupportStatus:
        return self._support

    def _reject(self, operation: str) -> None:
        self._reject_count += 1
        support_dict = self._support.to_dict()
        if self._support.supported and self._allow_live and not self._live_binding_complete:
            raise AgntcyUnsupportedError(
                f"{AGNTCY_UNSUPPORTED_CODE}: operation {operation!r} rejected — "
                "AGNTCY live binding is incomplete",
                reason=(
                    "AGNTCY SDK probe succeeded and live mode was requested, but "
                    "no reviewed client binding is implemented yet; refusing to "
                    f"perform {operation!r}"
                ),
                blocker_id="agntcy-live-binding-incomplete",
                support=support_dict,
            )
        raise AgntcyUnsupportedError(
            f"{AGNTCY_UNSUPPORTED_CODE}: operation {operation!r} rejected — "
            f"{self._support.reason}",
            reason=self._support.reason,
            blocker_id=self._support.blocker_id or AGNTCY_BLOCKER_ID,
            support=support_dict,
        )

    # ------------------------------------------------------------------
    # Registry@1 — all paths typed-reject when unsupported / unbound
    # ------------------------------------------------------------------

    def publish(
        self,
        advertisement: Mapping[str, Any],
        *,
        now_ms: Optional[int] = None,
    ) -> dict[str, Any]:
        del advertisement, now_ms
        self._reject("publish")
        raise AssertionError("unreachable")  # pragma: no cover

    def refresh(
        self,
        advertisement: Mapping[str, Any],
        *,
        now_ms: Optional[int] = None,
    ) -> dict[str, Any]:
        del advertisement, now_ms
        self._reject("refresh")
        raise AssertionError("unreachable")  # pragma: no cover

    def withdraw(
        self,
        identity_did: str,
        *,
        now_ms: Optional[int] = None,
    ) -> bool:
        del identity_did, now_ms
        self._reject("withdraw")
        raise AssertionError("unreachable")  # pragma: no cover

    def lookup_by_identity(
        self,
        identity_did: str,
        *,
        now_ms: Optional[int] = None,
        include_stale: bool = False,
    ) -> Optional[dict[str, Any]]:
        del identity_did, now_ms, include_stale
        self._reject("lookup_by_identity")
        raise AssertionError("unreachable")  # pragma: no cover

    def lookup_by_interface_cid(
        self,
        interface_cid: str,
        *,
        now_ms: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        del interface_cid, now_ms
        self._reject("lookup_by_interface_cid")
        raise AssertionError("unreachable")  # pragma: no cover

    def lookup_by_semantic_capability(
        self,
        capability: str,
        *,
        now_ms: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        del capability, now_ms
        self._reject("lookup_by_semantic_capability")
        raise AssertionError("unreachable")  # pragma: no cover

    def lookup_by_policy(
        self,
        policy_language: str,
        *,
        now_ms: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        del policy_language, now_ms
        self._reject("lookup_by_policy")
        raise AssertionError("unreachable")  # pragma: no cover

    def lookup_by_proof(
        self,
        proof_system: str,
        *,
        now_ms: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        del proof_system, now_ms
        self._reject("lookup_by_proof")
        raise AssertionError("unreachable")  # pragma: no cover

    def select(
        self,
        *,
        interface_cid: Optional[str] = None,
        semantic_capability: Optional[str] = None,
        policy_language: Optional[str] = None,
        proof_system: Optional[str] = None,
        candidates: Optional[Sequence[Mapping[str, Any]]] = None,
        now_ms: Optional[int] = None,
    ) -> Optional[dict[str, Any]]:
        del (
            interface_cid,
            semantic_capability,
            policy_language,
            proof_system,
            candidates,
            now_ms,
        )
        self._reject("select")
        raise AssertionError("unreachable")  # pragma: no cover

    def list_all(
        self,
        *,
        now_ms: Optional[int] = None,
        include_stale: bool = False,
    ) -> list[dict[str, Any]]:
        del now_ms, include_stale
        self._reject("list_all")
        raise AssertionError("unreachable")  # pragma: no cover

    def stats(self) -> MutableMapping[str, Any]:
        """Diagnostics are always available (do not require live support)."""

        status = self._support.to_dict()
        return {
            "interface": self.interface,
            "family_interface": self.family_interface,
            "provider": self.provider_id,
            "execution_authority": False,
            "supported": self.supported,
            "allow_live": self._allow_live,
            "live_binding_complete": self._live_binding_complete,
            "reject_count": self._reject_count,
            "support": status,
            "blocker_id": self._support.blocker_id,
            "blocker_summary": self._support.blocker_summary,
        }


def create_agntcy_adapter(
    *,
    allow_live: bool = False,
    env: Mapping[str, str] | None = None,
    force_sdk_available: Optional[bool] = None,
    force_sdk_module: Optional[str] = None,
) -> AgntcyAdapter:
    """Factory for :class:`AgntcyAdapter` (always constructs; ops may reject)."""

    return AgntcyAdapter(
        allow_live=allow_live,
        env=env,
        force_sdk_available=force_sdk_available,
        force_sdk_module=force_sdk_module,
    )


def require_agntcy_supported(
    *,
    env: Mapping[str, str] | None = None,
    force_sdk_available: Optional[bool] = None,
) -> AgntcySupportStatus:
    """Return support status or raise the typed unsupported reject."""

    status = probe_agntcy_support(
        env=env, force_sdk_available=force_sdk_available
    )
    if not status.supported:
        raise AgntcyUnsupportedError(
            reason=status.reason,
            blocker_id=status.blocker_id or AGNTCY_BLOCKER_ID,
            support=status.to_dict(),
        )
    return status


__all__ = [
    "AGNTCY_ADAPTER_INTERFACE",
    "AGNTCY_BLOCKER_ID",
    "AGNTCY_BLOCKER_SUMMARY",
    "AGNTCY_LIVE_ENV",
    "AGNTCY_PROVIDER_ID",
    "AGNTCY_UNSUPPORTED_CODE",
    "AgntcyAdapter",
    "AgntcyError",
    "AgntcySupportStatus",
    "AgntcyUnsupportedError",
    "create_agntcy_adapter",
    "is_agntcy_supported",
    "probe_agntcy_support",
    "require_agntcy_supported",
]
