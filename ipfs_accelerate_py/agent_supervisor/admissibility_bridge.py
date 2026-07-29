"""Supervisor-side Intent admissibility bridge (SupervisorAdmissibilityBridge@1).

LIG-017 / LIG-G070: thin adapter that loads pinned proof-corpus artifacts and
returns :class:`AdmissibilityDecision` objects from the datasets gate without
duplicating join logic.

Design invariants
-----------------
* **Lazy imports** — ``ipfs_datasets_py`` admissibility / proof-corpus modules
  are imported only when a check runs.  Importing this module (or
  ``agent_supervisor`` more broadly) never requires optional heavy provers.
* **Fail closed** — missing datasets dependency, empty corpus, malformed
  inputs, and gate errors become structured reject / abstain results (or a
  raised :class:`AdmissibilityBridgeError` for contract violations).  Never
  silent allow.
* **No skill/prompt execution** — source bodies are never evaluated; the bridge
  only accepts pre-built Intent formal CIDs, envelopes, or artifact maps plus
  a corpus store / offline fixture snapshot.
* **Observation** — decision maps are serializable for decision-runtime
  receipts and audit logs.

Environment / flags (documented)
--------------------------------
* ``IPFS_ACCELERATE_ADMISSIBILITY_ENABLED`` — ``1``/``true`` enables default
  bridge construction from env (default: enabled when datasets is importable).
* ``IPFS_ACCELERATE_ADMISSIBILITY_STORE`` — filesystem root for
  :class:`~ipfs_datasets_py.logic.proof_corpus.store.ProofCorpusStore`.
* ``IPFS_ACCELERATE_ADMISSIBILITY_PROFILE`` — profile id (default
  ``legal-strict``).
* ``IPFS_ACCELERATE_ADMISSIBILITY_REQUIRE_DATASETS`` — when ``1``/``true``,
  construction fails hard if datasets gate cannot be imported; otherwise the
  bridge reports ``unavailable`` and fails closed on evaluate.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any, Final


# ---------------------------------------------------------------------------
# Interface / schema constants
# ---------------------------------------------------------------------------

SUPERVISOR_ADMISSIBILITY_BRIDGE_INTERFACE: Final = "SupervisorAdmissibilityBridge@1"
SUPERVISOR_ADMISSIBILITY_BRIDGE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/admissibility-bridge@1"
)
SUPERVISOR_ADMISSIBILITY_BRIDGE_VERSION: Final[int] = 1
SUPERVISOR_ADMISSIBILITY_OBSERVATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/admissibility-observation@1"
)

DEFAULT_PROFILE_ID: Final[str] = "legal-strict"

ENV_ADMISSIBILITY_ENABLED: Final = "IPFS_ACCELERATE_ADMISSIBILITY_ENABLED"
ENV_ADMISSIBILITY_STORE: Final = "IPFS_ACCELERATE_ADMISSIBILITY_STORE"
ENV_ADMISSIBILITY_PROFILE: Final = "IPFS_ACCELERATE_ADMISSIBILITY_PROFILE"
ENV_ADMISSIBILITY_REQUIRE_DATASETS: Final = (
    "IPFS_ACCELERATE_ADMISSIBILITY_REQUIRE_DATASETS"
)

# Lazy-import targets (never imported at module load).
_DATASETS_GATE_MODULE: Final = "ipfs_datasets_py.logic.admissibility.gate"
_DATASETS_PROFILES_MODULE: Final = "ipfs_datasets_py.logic.admissibility.profiles"
_DATASETS_REASONS_MODULE: Final = "ipfs_datasets_py.logic.admissibility.reasons"
_DATASETS_STORE_MODULE: Final = "ipfs_datasets_py.logic.proof_corpus.store"
_DATASETS_SCHEMAS_MODULE: Final = "ipfs_datasets_py.logic.proof_corpus.schemas"
_DATASETS_FORMAL_MODULE: Final = "ipfs_datasets_py.logic.formalization.compiler"


class AdmissibilityBridgeError(ValueError):
    """Raised when the bridge contract is violated (malformed config / args)."""


class AdmissibilityBridgeStatus(str, Enum):
    """Availability of the bridge relative to the optional datasets dependency."""

    READY = "ready"
    UNAVAILABLE = "unavailable"
    DISABLED = "disabled"
    MISCONFIGURED = "misconfigured"


class AdmissibilityDisposition(str, Enum):
    """Normalized disposition observed by the supervisor / decision-runtime."""

    ALLOW = "allow"
    REJECT = "reject"
    ABSTAIN = "abstain"
    ERROR = "error"
    UNAVAILABLE = "unavailable"


def _truthy_env(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if value is None:
        if required:
            raise AdmissibilityBridgeError(f"{name} is required; fail closed")
        return ""
    if not isinstance(value, str):
        raise AdmissibilityBridgeError(f"{name} must be a string; fail closed")
    text = value.strip()
    if required and not text:
        raise AdmissibilityBridgeError(f"{name} must be non-empty; fail closed")
    return text


# ---------------------------------------------------------------------------
# Lazy datasets dependency surface
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _DatasetsSurface:
    """Bound modules for one successful datasets import."""

    gate: ModuleType
    profiles: ModuleType
    reasons: ModuleType
    store: ModuleType
    schemas: ModuleType
    formal: ModuleType


_datasets_surface: _DatasetsSurface | None = None
_datasets_import_error: str | None = None


def datasets_available() -> bool:
    """Return True when the datasets gate package can be imported."""

    try:
        _load_datasets_surface()
        return True
    except AdmissibilityBridgeError:
        return False


def _load_datasets_surface() -> _DatasetsSurface:
    """Import datasets admissibility APIs once; fail closed on ImportError."""

    global _datasets_surface, _datasets_import_error
    if _datasets_surface is not None:
        return _datasets_surface
    if _datasets_import_error is not None:
        raise AdmissibilityBridgeError(
            f"ipfs_datasets_py admissibility gate unavailable: "
            f"{_datasets_import_error}; fail closed"
        )
    try:
        gate = importlib.import_module(_DATASETS_GATE_MODULE)
        profiles = importlib.import_module(_DATASETS_PROFILES_MODULE)
        reasons = importlib.import_module(_DATASETS_REASONS_MODULE)
        store = importlib.import_module(_DATASETS_STORE_MODULE)
        schemas = importlib.import_module(_DATASETS_SCHEMAS_MODULE)
        formal = importlib.import_module(_DATASETS_FORMAL_MODULE)
    except Exception as exc:  # noqa: BLE001 — surface any import failure closed
        _datasets_import_error = f"{type(exc).__name__}: {exc}"
        raise AdmissibilityBridgeError(
            f"ipfs_datasets_py admissibility gate unavailable: "
            f"{_datasets_import_error}; fail closed"
        ) from exc
    _datasets_surface = _DatasetsSurface(
        gate=gate,
        profiles=profiles,
        reasons=reasons,
        store=store,
        schemas=schemas,
        formal=formal,
    )
    return _datasets_surface


def reset_datasets_surface_cache() -> None:
    """Test helper: clear the lazy import cache."""

    global _datasets_surface, _datasets_import_error
    _datasets_surface = None
    _datasets_import_error = None


# ---------------------------------------------------------------------------
# Observation (decision-runtime friendly)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AdmissibilityObservation:
    """Serializable observation of one bridge evaluation for decision-runtime.

    Decision-runtime and receipts can attach this object without importing
    datasets types.  ``decision`` is the gate's ``to_dict()`` payload when
    available; otherwise ``None``.
    """

    disposition: AdmissibilityDisposition
    status: str
    profile_id: str
    intent_cid: str
    constraint_cids: tuple[str, ...]
    reason_codes: tuple[str, ...]
    config_digest: str
    bridge_status: AdmissibilityBridgeStatus
    decision: Mapping[str, Any] | None = None
    error: str = ""
    store_snapshot_digest: str = ""
    schema: str = SUPERVISOR_ADMISSIBILITY_OBSERVATION_SCHEMA
    interface: str = SUPERVISOR_ADMISSIBILITY_BRIDGE_INTERFACE

    def to_dict(self) -> dict[str, Any]:
        return {
            "bridge_status": self.bridge_status.value,
            "config_digest": self.config_digest,
            "constraint_cids": list(self.constraint_cids),
            "decision": dict(self.decision) if self.decision is not None else None,
            "disposition": self.disposition.value,
            "error": self.error,
            "intent_cid": self.intent_cid,
            "interface": self.interface,
            "profile_id": self.profile_id,
            "reason_codes": list(self.reason_codes),
            "schema": self.schema,
            "status": self.status,
            "store_snapshot_digest": self.store_snapshot_digest,
        }

    @property
    def is_allow(self) -> bool:
        return self.disposition is AdmissibilityDisposition.ALLOW

    @property
    def is_reject(self) -> bool:
        return self.disposition is AdmissibilityDisposition.REJECT

    @property
    def is_abstain(self) -> bool:
        return self.disposition is AdmissibilityDisposition.ABSTAIN


def _observation_from_decision(
    decision: Any,
    *,
    bridge_status: AdmissibilityBridgeStatus = AdmissibilityBridgeStatus.READY,
) -> AdmissibilityObservation:
    status_value = str(getattr(getattr(decision, "status", None), "value", decision.status))
    try:
        disposition = AdmissibilityDisposition(status_value)
    except ValueError:
        disposition = AdmissibilityDisposition.ERROR
    reason_codes = tuple(getattr(decision, "reason_codes", ()) or ())
    constraint_cids = tuple(getattr(decision, "constraint_cids", ()) or ())
    decision_map: Mapping[str, Any] | None
    if hasattr(decision, "to_dict"):
        decision_map = decision.to_dict()
    elif isinstance(decision, Mapping):
        decision_map = dict(decision)
    else:
        decision_map = None
    return AdmissibilityObservation(
        disposition=disposition,
        status=status_value,
        profile_id=str(getattr(decision, "profile_id", "") or ""),
        intent_cid=str(getattr(decision, "intent_cid", "") or ""),
        constraint_cids=constraint_cids,
        reason_codes=reason_codes,
        config_digest=str(getattr(decision, "config_digest", "") or ""),
        bridge_status=bridge_status,
        decision=decision_map,
        store_snapshot_digest=str(
            getattr(decision, "store_snapshot_digest", "") or ""
        ),
    )


def _closed_observation(
    *,
    disposition: AdmissibilityDisposition,
    bridge_status: AdmissibilityBridgeStatus,
    error: str,
    profile_id: str = "",
    intent_cid: str = "",
) -> AdmissibilityObservation:
    return AdmissibilityObservation(
        disposition=disposition,
        status=disposition.value,
        profile_id=profile_id,
        intent_cid=intent_cid,
        constraint_cids=(),
        reason_codes=(),
        config_digest="",
        bridge_status=bridge_status,
        decision=None,
        error=error,
    )


# ---------------------------------------------------------------------------
# Corpus loading (pinned artifacts / offline fixtures)
# ---------------------------------------------------------------------------


def open_proof_corpus_store(
    store_root: str | Path | None = None,
    envelopes: Sequence[Mapping[str, Any]] | None = None,
    *,
    store: Any | None = None,
) -> Any:
    """Open or accept a :class:`ProofCorpusStore` with optional seed envelopes.

    Parameters
    ----------
    store_root:
        Optional filesystem root for a durable corpus snapshot.
    envelopes:
        Optional sequence of envelope mappings (offline fixtures / mocks).
    store:
        Pre-built store instance; when provided, envelopes are still put when
        given.

    Returns
    -------
    ProofCorpusStore
        Store ready for gate evaluation.
    """

    surface = _load_datasets_surface()
    ProofCorpusStore = surface.store.ProofCorpusStore
    ArtifactEnvelope = surface.schemas.ArtifactEnvelope

    if store is not None:
        active = store
    else:
        root: Path | None = None
        if store_root is not None:
            if not isinstance(store_root, (str, Path)):
                raise AdmissibilityBridgeError(
                    "store_root must be a path string; fail closed"
                )
            root_path = Path(store_root)
            if not str(store_root).strip():
                raise AdmissibilityBridgeError(
                    "store_root must be non-empty; fail closed"
                )
            if not root_path.exists():
                raise AdmissibilityBridgeError(
                    f"store_root does not exist: {root_path}; fail closed"
                )
            root = root_path
        active = ProofCorpusStore(root=root)

    if envelopes is not None:
        if not isinstance(envelopes, Sequence) or isinstance(
            envelopes, (str, bytes, bytearray)
        ):
            raise AdmissibilityBridgeError(
                "envelopes must be a sequence of mappings; fail closed"
            )
        for index, item in enumerate(envelopes):
            if not isinstance(item, Mapping):
                raise AdmissibilityBridgeError(
                    f"envelopes[{index}] must be a mapping; fail closed"
                )
            # Accept either wire dicts or already-built envelopes.
            if hasattr(item, "content_cid") and hasattr(item, "to_dict"):
                active.put(item)
            else:
                active.put(ArtifactEnvelope.from_dict(item))
    return active


def load_pinned_intent(
    intent: str | Mapping[str, Any] | Any,
    *,
    store: Any | None = None,
) -> Any:
    """Normalize an intent reference for gate evaluation.

    Accepts a content CID string, envelope / artifact mapping, or a pre-built
    FormalizationArtifact / ArtifactEnvelope instance.  Does not formalize or
    execute source text.
    """

    if intent is None:
        raise AdmissibilityBridgeError("intent is required; fail closed")
    if isinstance(intent, str):
        text = intent.strip()
        if not text:
            raise AdmissibilityBridgeError(
                "intent content CID must be non-empty; fail closed"
            )
        return text
    if isinstance(intent, Mapping):
        return dict(intent)
    # Pass through typed objects (envelope / FormalizationArtifact).
    return intent


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------


@dataclass
class SupervisorAdmissibilityBridge:
    """SupervisorAdmissibilityBridge@1 — lazy, fail-closed gate adapter.

    Construct with an explicit store, a filesystem root, offline envelope
    fixtures, or via :meth:`from_env`.  Evaluation returns both the native
    gate decision (when available) and a provider-free
    :class:`AdmissibilityObservation` for decision-runtime.
    """

    store: Any | None = None
    store_root: str | Path | None = None
    envelopes: tuple[Mapping[str, Any], ...] | None = None
    profile_id: str = DEFAULT_PROFILE_ID
    enabled: bool = True
    require_datasets: bool = False
    _gate: Any | None = field(default=None, init=False, repr=False)
    _bridge_status: AdmissibilityBridgeStatus = field(
        default=AdmissibilityBridgeStatus.READY, init=False, repr=False
    )
    _status_detail: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not self.profile_id.strip():
            raise AdmissibilityBridgeError(
                "profile_id must be a non-empty string; fail closed"
            )
        object.__setattr__(self, "profile_id", self.profile_id.strip())
        if self.envelopes is not None and not isinstance(self.envelopes, tuple):
            object.__setattr__(self, "envelopes", tuple(self.envelopes))
        if not self.enabled:
            object.__setattr__(
                self, "_bridge_status", AdmissibilityBridgeStatus.DISABLED
            )
            object.__setattr__(
                self, "_status_detail", "admissibility bridge disabled by config"
            )
            return
        # Defer store/gate construction until first evaluate so import stays light.
        # Probe availability only when require_datasets is set.
        if self.require_datasets:
            try:
                _load_datasets_surface()
            except AdmissibilityBridgeError as exc:
                object.__setattr__(
                    self, "_bridge_status", AdmissibilityBridgeStatus.UNAVAILABLE
                )
                object.__setattr__(self, "_status_detail", str(exc))
                raise

    # -- interface -----------------------------------------------------------

    @property
    def interface(self) -> str:
        return SUPERVISOR_ADMISSIBILITY_BRIDGE_INTERFACE

    @property
    def schema(self) -> str:
        return SUPERVISOR_ADMISSIBILITY_BRIDGE_SCHEMA

    @property
    def version(self) -> int:
        return SUPERVISOR_ADMISSIBILITY_BRIDGE_VERSION

    @property
    def bridge_status(self) -> AdmissibilityBridgeStatus:
        return self._bridge_status

    @property
    def is_ready(self) -> bool:
        return (
            self.enabled
            and self._bridge_status is AdmissibilityBridgeStatus.READY
        )

    # -- factory -------------------------------------------------------------

    @classmethod
    def from_env(
        cls,
        *,
        store: Any | None = None,
        envelopes: Sequence[Mapping[str, Any]] | None = None,
    ) -> "SupervisorAdmissibilityBridge":
        """Build a bridge from documented environment variables."""

        enabled = _truthy_env(ENV_ADMISSIBILITY_ENABLED, default=True)
        require_datasets = _truthy_env(
            ENV_ADMISSIBILITY_REQUIRE_DATASETS, default=False
        )
        store_root = os.environ.get(ENV_ADMISSIBILITY_STORE) or None
        profile = os.environ.get(ENV_ADMISSIBILITY_PROFILE) or DEFAULT_PROFILE_ID
        return cls(
            store=store,
            store_root=store_root,
            envelopes=tuple(envelopes) if envelopes is not None else None,
            profile_id=profile,
            enabled=enabled,
            require_datasets=require_datasets,
        )

    @classmethod
    def from_offline_fixtures(
        cls,
        envelopes: Sequence[Mapping[str, Any]],
        *,
        profile_id: str = DEFAULT_PROFILE_ID,
    ) -> "SupervisorAdmissibilityBridge":
        """Build a bridge backed only by offline envelope fixtures / mocks."""

        if not isinstance(envelopes, Sequence) or isinstance(
            envelopes, (str, bytes, bytearray)
        ):
            raise AdmissibilityBridgeError(
                "envelopes must be a sequence; fail closed"
            )
        if not envelopes:
            raise AdmissibilityBridgeError(
                "offline fixtures require at least one envelope; fail closed"
            )
        return cls(
            envelopes=tuple(envelopes),
            profile_id=profile_id,
            enabled=True,
            require_datasets=True,
        )

    # -- store / gate resolution ---------------------------------------------

    def _resolve_store(self) -> Any:
        if self.store is not None and self.envelopes is None:
            return self.store
        if self.store is None and self.store_root is None and self.envelopes is None:
            raise AdmissibilityBridgeError(
                "store, store_root, or envelopes is required; fail closed"
            )
        return open_proof_corpus_store(
            self.store_root,
            self.envelopes,
            store=self.store,
        )

    def _resolve_gate(self) -> Any:
        if self._gate is not None:
            return self._gate
        surface = _load_datasets_surface()
        active_store = self._resolve_store()
        # Cache store when we opened it so repeated evaluates share the snapshot.
        if self.store is None:
            object.__setattr__(self, "store", active_store)
        gate = surface.gate.IntentAdmissibilityGate(store=active_store)
        object.__setattr__(self, "_gate", gate)
        object.__setattr__(self, "_bridge_status", AdmissibilityBridgeStatus.READY)
        return gate

    def ensure_ready(self) -> AdmissibilityBridgeStatus:
        """Probe datasets + store without evaluating an intent."""

        if not self.enabled:
            object.__setattr__(
                self, "_bridge_status", AdmissibilityBridgeStatus.DISABLED
            )
            return self._bridge_status
        try:
            self._resolve_gate()
            return AdmissibilityBridgeStatus.READY
        except AdmissibilityBridgeError as exc:
            object.__setattr__(
                self, "_bridge_status", AdmissibilityBridgeStatus.UNAVAILABLE
            )
            object.__setattr__(self, "_status_detail", str(exc))
            return self._bridge_status
        except Exception as exc:  # noqa: BLE001
            object.__setattr__(
                self, "_bridge_status", AdmissibilityBridgeStatus.MISCONFIGURED
            )
            object.__setattr__(
                self, "_status_detail", f"{type(exc).__name__}: {exc}"
            )
            return self._bridge_status

    # -- evaluation ----------------------------------------------------------

    def evaluate(
        self,
        intent: str | Mapping[str, Any] | Any,
        profile: str | None = None,
    ) -> Any:
        """Run the datasets gate and return the native AdmissibilityDecision.

        Raises
        ------
        AdmissibilityBridgeError
            When the bridge is disabled, datasets is unavailable, or inputs
            are malformed.  Policy outcomes (allow/reject/abstain) are returned
            as decisions, not raised.
        """

        if not self.enabled:
            raise AdmissibilityBridgeError(
                "admissibility bridge is disabled; fail closed"
            )
        gate = self._resolve_gate()
        normalized = load_pinned_intent(intent, store=self.store)
        active_profile = (
            profile.strip()
            if isinstance(profile, str) and profile.strip()
            else self.profile_id
        )
        return gate.evaluate(normalized, active_profile)

    def check(
        self,
        intent: str | Mapping[str, Any] | Any,
        profile: str | None = None,
    ) -> AdmissibilityObservation:
        """Evaluate and return a provider-free observation (never raises for policy).

        Contract violations and dependency failures become fail-closed
        observations with disposition ``error`` / ``unavailable``.
        """

        active_profile = (
            profile.strip()
            if isinstance(profile, str) and profile.strip()
            else self.profile_id
        )
        if not self.enabled:
            return _closed_observation(
                disposition=AdmissibilityDisposition.UNAVAILABLE,
                bridge_status=AdmissibilityBridgeStatus.DISABLED,
                error="admissibility bridge is disabled; fail closed",
                profile_id=active_profile,
            )
        try:
            decision = self.evaluate(intent, active_profile)
            return _observation_from_decision(
                decision, bridge_status=AdmissibilityBridgeStatus.READY
            )
        except AdmissibilityBridgeError as exc:
            status = self._bridge_status
            if status is AdmissibilityBridgeStatus.READY:
                status = AdmissibilityBridgeStatus.UNAVAILABLE
            return _closed_observation(
                disposition=AdmissibilityDisposition.UNAVAILABLE
                if status is AdmissibilityBridgeStatus.UNAVAILABLE
                else AdmissibilityDisposition.ERROR,
                bridge_status=status,
                error=str(exc),
                profile_id=active_profile,
            )
        except Exception as exc:  # noqa: BLE001 — fail closed
            return _closed_observation(
                disposition=AdmissibilityDisposition.ERROR,
                bridge_status=AdmissibilityBridgeStatus.MISCONFIGURED,
                error=f"admissibility evaluation failed closed: "
                f"{type(exc).__name__}: {exc}",
                profile_id=active_profile,
            )

    def check_intent_admissibility(
        self,
        intent: str | Mapping[str, Any] | Any,
        profile: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        """Wire-friendly check returning a JSON-serializable map.

        Mirrors the MCP tool surface shape closely enough for supervisor and
        decision-runtime consumers without requiring async.
        """

        observation = self.check(intent, profile)
        payload = observation.to_dict()
        payload["success"] = observation.is_allow
        payload["executed"] = False
        return payload

    def capabilities(self) -> dict[str, Any]:
        """Report bridge surface without evaluating or loading a corpus."""

        available = datasets_available()
        return {
            "interface": self.interface,
            "schema": self.schema,
            "version": self.version,
            "enabled": self.enabled,
            "bridge_status": self.bridge_status.value,
            "datasets_available": available,
            "default_profile_id": self.profile_id,
            "env_flags": {
                "enabled": ENV_ADMISSIBILITY_ENABLED,
                "store": ENV_ADMISSIBILITY_STORE,
                "profile": ENV_ADMISSIBILITY_PROFILE,
                "require_datasets": ENV_ADMISSIBILITY_REQUIRE_DATASETS,
            },
            "executed": False,
            "provers_imported": False,
        }


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def create_admissibility_bridge(
    *,
    store: Any | None = None,
    store_root: str | Path | None = None,
    envelopes: Sequence[Mapping[str, Any]] | None = None,
    profile_id: str = DEFAULT_PROFILE_ID,
    enabled: bool = True,
    require_datasets: bool = False,
) -> SupervisorAdmissibilityBridge:
    """Factory for an explicit bridge configuration."""

    return SupervisorAdmissibilityBridge(
        store=store,
        store_root=store_root,
        envelopes=tuple(envelopes) if envelopes is not None else None,
        profile_id=profile_id,
        enabled=enabled,
        require_datasets=require_datasets,
    )


def check_intent_admissibility(
    intent: str | Mapping[str, Any] | Any,
    profile: str | None = None,
    *,
    store: Any | None = None,
    store_root: str | Path | None = None,
    envelopes: Sequence[Mapping[str, Any]] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Module-level helper: one-shot admissibility check via a temporary bridge."""

    bridge = create_admissibility_bridge(
        store=store,
        store_root=store_root,
        envelopes=envelopes,
        profile_id=profile or DEFAULT_PROFILE_ID,
        enabled=True,
        require_datasets=False,
    )
    return bridge.check_intent_admissibility(intent, profile, **kwargs)


def observe_admissibility(
    intent: str | Mapping[str, Any] | Any,
    profile: str | None = None,
    *,
    store: Any | None = None,
    store_root: str | Path | None = None,
    envelopes: Sequence[Mapping[str, Any]] | None = None,
) -> AdmissibilityObservation:
    """Module-level helper returning an :class:`AdmissibilityObservation`."""

    bridge = create_admissibility_bridge(
        store=store,
        store_root=store_root,
        envelopes=envelopes,
        profile_id=profile or DEFAULT_PROFILE_ID,
    )
    return bridge.check(intent, profile)


__all__ = [
    "DEFAULT_PROFILE_ID",
    "ENV_ADMISSIBILITY_ENABLED",
    "ENV_ADMISSIBILITY_PROFILE",
    "ENV_ADMISSIBILITY_REQUIRE_DATASETS",
    "ENV_ADMISSIBILITY_STORE",
    "SUPERVISOR_ADMISSIBILITY_BRIDGE_INTERFACE",
    "SUPERVISOR_ADMISSIBILITY_BRIDGE_SCHEMA",
    "SUPERVISOR_ADMISSIBILITY_BRIDGE_VERSION",
    "SUPERVISOR_ADMISSIBILITY_OBSERVATION_SCHEMA",
    "AdmissibilityBridgeError",
    "AdmissibilityBridgeStatus",
    "AdmissibilityDisposition",
    "AdmissibilityObservation",
    "SupervisorAdmissibilityBridge",
    "check_intent_admissibility",
    "create_admissibility_bridge",
    "datasets_available",
    "load_pinned_intent",
    "observe_admissibility",
    "open_proof_corpus_store",
    "reset_datasets_surface_cache",
]
