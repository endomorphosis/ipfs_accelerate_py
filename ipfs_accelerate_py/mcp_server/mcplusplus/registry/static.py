"""StaticRegistry@1 — in-process test / hermetic agent advertisement registry.

Provides a deterministic, thread-safe, clock-injectable implementation of
:class:`Registry` for unit tests and single-process deployments. It stores
validated AgentAdvertisement@1 records keyed by principal DID.

Selection is health-aware with a total order that breaks equal-health ties
by load and then lexicographic DID. Registry membership is never execution
authority.
"""

from __future__ import annotations

import copy
import threading
import time
from typing import Any, Callable, Mapping, MutableMapping, Optional, Sequence

from ipfs_accelerate_py.mcp_server.mcplusplus.registry.interface import (
    REGISTRY_INTERFACE,
    Registry,
    RegistryNotFoundError,
    RegistryStaleError,
    RegistryValidationError,
    identity_did,
    is_stale,
    policy_languages_of,
    proof_systems_of,
    select_advertisement,
    semantic_capabilities_of,
    sort_for_selection,
    validate_agent_advertisement,
)

STATIC_REGISTRY_INTERFACE = "StaticRegistry@1"
STATIC_PROVIDER_ID = "static"


def _wall_clock_ms() -> int:
    return int(time.time() * 1000)


class StaticRegistry(Registry):
    """In-memory Registry@1 adapter (StaticRegistry@1).

    Parameters
    ----------
    require_signed:
        When True, publish/refresh reject advertisements lacking a valid
        signature block shape.
    clock_ms:
        Callable returning the current epoch milliseconds. Defaults to wall
        clock. Inject a fixed or stepped clock in tests.
    """

    def __init__(
        self,
        *,
        require_signed: bool = False,
        clock_ms: Optional[Callable[[], int]] = None,
    ) -> None:
        self._lock = threading.RLock()
        self._records: dict[str, dict[str, Any]] = {}
        # Receipt time used when published_at_ms is omitted (for expiry calc).
        self._receipt_ms: dict[str, int] = {}
        self._require_signed = bool(require_signed)
        self._clock_ms = clock_ms if clock_ms is not None else _wall_clock_ms
        self._publish_count = 0
        self._refresh_count = 0
        self._withdraw_count = 0
        self._stale_rejects = 0
        self._unsigned_rejects = 0

    @property
    def provider_id(self) -> str:
        return STATIC_PROVIDER_ID

    @property
    def interface(self) -> str:
        # Advertise both the family contract and this adapter label.
        return STATIC_REGISTRY_INTERFACE

    @property
    def family_interface(self) -> str:
        """Parent Registry@1 family label."""

        return REGISTRY_INTERFACE

    def _now(self, now_ms: Optional[int]) -> int:
        if now_ms is not None:
            if not isinstance(now_ms, int) or isinstance(now_ms, bool) or now_ms < 0:
                raise RegistryValidationError("now_ms must be a non-negative integer")
            return now_ms
        value = self._clock_ms()
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise RegistryValidationError("clock_ms() must return a non-negative integer")
        return value

    def _normalize(
        self,
        advertisement: Mapping[str, Any],
        *,
        now_ms: int,
    ) -> dict[str, Any]:
        try:
            return validate_agent_advertisement(
                advertisement,
                require_signed=self._require_signed,
                now_ms=now_ms,
                reject_stale=True,
                receipt_ms=now_ms,
            )
        except RegistryStaleError:
            self._stale_rejects += 1
            raise
        except Exception as exc:
            # Count unsigned rejects without importing the error into a cycle.
            from ipfs_accelerate_py.mcp_server.mcplusplus.registry.interface import (
                RegistryUnsignedError,
            )

            if isinstance(exc, RegistryUnsignedError):
                self._unsigned_rejects += 1
            raise

    def _store(self, normalized: dict[str, Any], *, now_ms: int) -> dict[str, Any]:
        did = identity_did(normalized)
        stored = copy.deepcopy(normalized)
        # Materialize published_at when omitted so subsequent expiry is stable.
        if "published_at_ms" not in stored:
            stored["published_at_ms"] = now_ms
        self._records[did] = stored
        self._receipt_ms[did] = now_ms
        return copy.deepcopy(stored)

    def publish(
        self,
        advertisement: Mapping[str, Any],
        *,
        now_ms: Optional[int] = None,
    ) -> dict[str, Any]:
        """Validate and upsert an advertisement keyed by identity DID."""

        with self._lock:
            current = self._now(now_ms)
            normalized = self._normalize(advertisement, now_ms=current)
            result = self._store(normalized, now_ms=current)
            self._publish_count += 1
            return result

    def refresh(
        self,
        advertisement: Mapping[str, Any],
        *,
        now_ms: Optional[int] = None,
    ) -> dict[str, Any]:
        """Replace an existing advertisement; fail if the DID is unknown."""

        with self._lock:
            current = self._now(now_ms)
            normalized = self._normalize(advertisement, now_ms=current)
            did = identity_did(normalized)
            if did not in self._records:
                raise RegistryNotFoundError(
                    f"cannot refresh unknown identity {did!r}"
                )
            result = self._store(normalized, now_ms=current)
            self._refresh_count += 1
            return result

    def withdraw(
        self,
        identity_did_value: str,
        *,
        now_ms: Optional[int] = None,
    ) -> bool:
        """Remove the advertisement for ``identity_did_value`` if present."""

        del now_ms  # withdraw is immediate; clock unused but accepted for API parity
        if not isinstance(identity_did_value, str) or not identity_did_value.strip():
            raise RegistryValidationError("identity_did must be a non-empty string")
        did = identity_did_value.strip()
        with self._lock:
            existed = did in self._records
            self._records.pop(did, None)
            self._receipt_ms.pop(did, None)
            if existed:
                self._withdraw_count += 1
            return existed

    def _fresh_copy(
        self,
        did: str,
        record: Mapping[str, Any],
        *,
        now_ms: int,
        include_stale: bool,
    ) -> Optional[dict[str, Any]]:
        receipt = self._receipt_ms.get(did)
        if not include_stale and is_stale(record, now_ms=now_ms, receipt_ms=receipt):
            return None
        return copy.deepcopy(dict(record))

    def lookup_by_identity(
        self,
        identity_did_value: str,
        *,
        now_ms: Optional[int] = None,
        include_stale: bool = False,
    ) -> Optional[dict[str, Any]]:
        if not isinstance(identity_did_value, str) or not identity_did_value.strip():
            raise RegistryValidationError("identity_did must be a non-empty string")
        did = identity_did_value.strip()
        with self._lock:
            current = self._now(now_ms)
            record = self._records.get(did)
            if record is None:
                return None
            return self._fresh_copy(
                did, record, now_ms=current, include_stale=include_stale
            )

    def _iter_fresh(self, *, now_ms: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for did, record in self._records.items():
            copy_rec = self._fresh_copy(
                did, record, now_ms=now_ms, include_stale=False
            )
            if copy_rec is not None:
                out.append(copy_rec)
        return out

    def lookup_by_interface_cid(
        self,
        interface_cid: str,
        *,
        now_ms: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        if not isinstance(interface_cid, str) or not interface_cid.strip():
            raise RegistryValidationError("interface_cid must be a non-empty string")
        target = interface_cid.strip()
        with self._lock:
            current = self._now(now_ms)
            matches = [
                ad
                for ad in self._iter_fresh(now_ms=current)
                if target in list(ad.get("interface_cids") or [])
            ]
            return sort_for_selection(matches)

    def lookup_by_semantic_capability(
        self,
        capability: str,
        *,
        now_ms: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        if not isinstance(capability, str) or not capability.strip():
            raise RegistryValidationError("capability must be a non-empty string")
        target = capability.strip()
        with self._lock:
            current = self._now(now_ms)
            matches = [
                ad
                for ad in self._iter_fresh(now_ms=current)
                if target in semantic_capabilities_of(ad)
            ]
            return sort_for_selection(matches)

    def lookup_by_policy(
        self,
        policy_language: str,
        *,
        now_ms: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        if not isinstance(policy_language, str) or not policy_language.strip():
            raise RegistryValidationError("policy_language must be a non-empty string")
        target = policy_language.strip()
        with self._lock:
            current = self._now(now_ms)
            matches = [
                ad
                for ad in self._iter_fresh(now_ms=current)
                if target in policy_languages_of(ad)
            ]
            return sort_for_selection(matches)

    def lookup_by_proof(
        self,
        proof_system: str,
        *,
        now_ms: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        if not isinstance(proof_system, str) or not proof_system.strip():
            raise RegistryValidationError("proof_system must be a non-empty string")
        target = proof_system.strip()
        with self._lock:
            current = self._now(now_ms)
            matches = [
                ad
                for ad in self._iter_fresh(now_ms=current)
                if target in proof_systems_of(ad)
            ]
            return sort_for_selection(matches)

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
        """Select the best matching non-stale advertisement.

        Filters are AND-combined when multiple are provided. When ``candidates``
        is supplied, selection runs over that set (after optional filters) and
        does not consult the store except for clock resolution.
        """

        with self._lock:
            current = self._now(now_ms)
            if candidates is not None:
                pool = [dict(c) for c in candidates]
                # Drop stale candidates unless they lack enough fields to evaluate.
                filtered: list[dict[str, Any]] = []
                for ad in pool:
                    try:
                        if is_stale(ad, now_ms=current, receipt_ms=current):
                            continue
                    except RegistryValidationError:
                        # Incomplete ads in injected candidates are skipped.
                        continue
                    filtered.append(ad)
                pool = filtered
            else:
                pool = self._iter_fresh(now_ms=current)

            if interface_cid is not None:
                cid = interface_cid.strip()
                pool = [ad for ad in pool if cid in list(ad.get("interface_cids") or [])]
            if semantic_capability is not None:
                cap = semantic_capability.strip()
                pool = [ad for ad in pool if cap in semantic_capabilities_of(ad)]
            if policy_language is not None:
                pol = policy_language.strip()
                pool = [ad for ad in pool if pol in policy_languages_of(ad)]
            if proof_system is not None:
                proof = proof_system.strip()
                pool = [ad for ad in pool if proof in proof_systems_of(ad)]

            return select_advertisement(pool)

    def list_all(
        self,
        *,
        now_ms: Optional[int] = None,
        include_stale: bool = False,
    ) -> list[dict[str, Any]]:
        with self._lock:
            current = self._now(now_ms)
            out: list[dict[str, Any]] = []
            for did, record in self._records.items():
                copy_rec = self._fresh_copy(
                    did, record, now_ms=current, include_stale=include_stale
                )
                if copy_rec is not None:
                    out.append(copy_rec)
            # Stable order by DID for deterministic enumeration.
            out.sort(key=lambda ad: identity_did(ad))
            return out

    def stats(self) -> MutableMapping[str, Any]:
        with self._lock:
            return {
                "interface": self.interface,
                "family_interface": self.family_interface,
                "provider": self.provider_id,
                "execution_authority": False,
                "size": len(self._records),
                "publish_count": self._publish_count,
                "refresh_count": self._refresh_count,
                "withdraw_count": self._withdraw_count,
                "stale_rejects": self._stale_rejects,
                "unsigned_rejects": self._unsigned_rejects,
                "require_signed": self._require_signed,
            }


def create_static_registry(
    *,
    require_signed: bool = False,
    clock_ms: Optional[Callable[[], int]] = None,
) -> StaticRegistry:
    """Factory for :class:`StaticRegistry`."""

    return StaticRegistry(require_signed=require_signed, clock_ms=clock_ms)


__all__ = [
    "STATIC_PROVIDER_ID",
    "STATIC_REGISTRY_INTERFACE",
    "StaticRegistry",
    "create_static_registry",
]
