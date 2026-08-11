"""Production default Deterministic Doctor factory (WPD-010).

Interface: ``DefaultDoctorFactory@1``

Binds live checkout stage backends onto
:class:`~ipfs_accelerate_py.agent_supervisor.control.deterministic_doctor_service.DeterministicDoctorService`
so CLI/API defaults can inspect and plan against an exact repository root.

Fail-closed rules:

* Construction and cold import never load LLM / remote model-provider surfaces.
* Optional stage backends are explicit; missing backends yield typed
  capability abstentions rather than free-form fallbacks.
* Live checkout composition is lazy: stage modules load only when the service
  runs an operation that needs them.
* Source bodies never enter factory identity records.

This module owns production default wiring only.  It does not re-implement
analysis, tactician, proof, or transaction engines — those remain in their
stage modules and are composed through the existing deterministic Doctor
runtime when a checkout root is supplied.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from .deterministic_doctor_service import (
    DeterministicDoctorService,
    DoctorStageBackends,
    assert_no_llm_surface_loaded,
    create_deterministic_doctor_service,
)


# ---------------------------------------------------------------------------
# Interface identity
# ---------------------------------------------------------------------------

DEFAULT_DOCTOR_FACTORY_INTERFACE: Final[str] = "DefaultDoctorFactory@1"
DEFAULT_DOCTOR_FACTORY_VERSION: Final[int] = 1
DEFAULT_DOCTOR_FACTORY_EVIDENCE: Final[str] = "wpd/default-doctor-factory@1"

DEFAULT_DOCTOR_FACTORY_DISCOVERY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/default-doctor-factory/discovery@1"
)
DEFAULT_DOCTOR_FACTORY_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/default-doctor-factory/binding@1"
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DefaultDoctorFactoryError(RuntimeError):
    """Fail-closed rejection for an unsafe or incomplete doctor factory run."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "default_doctor_factory_error",
    ) -> None:
        super().__init__(message)
        self.reason_code = str(reason_code or "default_doctor_factory_error")


class DefaultDoctorCheckoutError(DefaultDoctorFactoryError, ValueError):
    """Checkout root is missing, not a directory, or not allowlisted."""

    def __init__(self, message: str, *, reason_code: str = "checkout_unavailable") -> None:
        super().__init__(message, reason_code=reason_code)


# ---------------------------------------------------------------------------
# Binding record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DefaultDoctorBinding:
    """Body-free projection of how a default service was constructed."""

    checkout_root: str
    live_stages_bound: bool
    backends_available: tuple[str, ...]
    capability_gaps: tuple[str, ...]
    policy_id: str = ""
    notes: tuple[str, ...] = ()

    @property
    def binding_id(self) -> str:
        from ..proof.formal_verification_contracts import content_identity

        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DEFAULT_DOCTOR_FACTORY_BINDING_SCHEMA,
            "interface": DEFAULT_DOCTOR_FACTORY_INTERFACE,
            "checkout_root": self.checkout_root,
            "live_stages_bound": self.live_stages_bound,
            "backends_available": list(self.backends_available),
            "capability_gaps": list(self.capability_gaps),
            "policy_id": self.policy_id,
            "notes": list(self.notes),
        }


# Closed stage vocabulary mirrored from DoctorStageBackends field names.
_ALL_STAGE_SLOTS: Final[tuple[str, ...]] = (
    "diagnose",
    "plan",
    "synthesis",
    "impact",
    "transaction",
    "fixed_point",
    "explain",
    "retrieve",
    "tactician",
    "proof",
)


def _canonical_checkout(checkout_root: str | Path) -> Path:
    try:
        root = Path(checkout_root).expanduser().resolve(strict=True)
    except OSError as exc:
        raise DefaultDoctorCheckoutError(
            f"checkout root is unavailable: {checkout_root}",
            reason_code="checkout_unavailable",
        ) from exc
    if not root.is_dir():
        raise DefaultDoctorCheckoutError(
            "checkout root must be a directory",
            reason_code="checkout_not_directory",
        )
    return root


def _capability_gaps(available: Sequence[str]) -> tuple[str, ...]:
    present = set(available)
    return tuple(slot for slot in _ALL_STAGE_SLOTS if slot not in present)


def _policy_id(policy: Any) -> str:
    if policy is None:
        return ""
    if isinstance(policy, Mapping):
        return str(policy.get("policy_id") or "")
    return str(getattr(policy, "policy_id", "") or "")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class DefaultDoctorFactory:
    """Production factory that binds live checkout stages for Doctor service.

    Interface: ``DefaultDoctorFactory@1``
    """

    INTERFACE: Final[str] = DEFAULT_DOCTOR_FACTORY_INTERFACE
    VERSION: Final[int] = DEFAULT_DOCTOR_FACTORY_VERSION

    def __init__(
        self,
        *,
        deterministic: bool = True,
        repository_allowlist: Sequence[str | Path] | None = None,
        control_service: Any | None = None,
        receipt_store: Any | None = None,
        scope_policy: Any | None = None,
        index_root: str | Path | None = None,
    ) -> None:
        self._deterministic = bool(deterministic)
        self._repository_allowlist = (
            tuple(repository_allowlist) if repository_allowlist is not None else None
        )
        self._control_service = control_service
        self._receipt_store = receipt_store
        self._scope_policy = scope_policy
        self._index_root = index_root
        self._last_binding: DefaultDoctorBinding | None = None
        self._last_runtime: Any | None = None

    @property
    def last_binding(self) -> DefaultDoctorBinding | None:
        return self._last_binding

    @property
    def last_runtime(self) -> Any | None:
        """Runtime retained when live stages were bound (may be ``None``)."""

        return self._last_runtime

    @staticmethod
    def discovery() -> dict[str, Any]:
        """Cold static discovery; no checkout or optional provider is touched."""

        return {
            "schema": DEFAULT_DOCTOR_FACTORY_DISCOVERY_SCHEMA,
            "interface": DEFAULT_DOCTOR_FACTORY_INTERFACE,
            "version": DEFAULT_DOCTOR_FACTORY_VERSION,
            "evidence_key": DEFAULT_DOCTOR_FACTORY_EVIDENCE,
            "service_interface": DeterministicDoctorService.INTERFACE,
            "stage_slots": list(_ALL_STAGE_SLOTS),
            "live_checkout_composition": True,
            "llm_router_enabled": False,
            "remote_model_provider_calls_allowed": False,
            "network_access_allowed": False,
            "automatic_fallback": False,
            "processes_started": False,
            "database_opened": False,
            "optional_providers_loaded": False,
        }

    def build(
        self,
        checkout_root: str | Path | None = None,
        *,
        policy: Any | None = None,
        backends: DoctorStageBackends | None = None,
        bind_live_stages: bool | None = None,
        control_service: Any | None = None,
        receipt_store: Any | None = None,
        repository_allowlist: Sequence[str | Path] | None = None,
        scope_policy: Any | None = None,
        index_root: str | Path | None = None,
        cas: Any | None = None,
    ) -> DeterministicDoctorService:
        """Build a production :class:`DeterministicDoctorService`.

        Parameters
        ----------
        checkout_root:
            Exact repository checkout.  When supplied (and ``backends`` is not
            overridden, and ``bind_live_stages`` is not ``False``), live stage
            backends are bound through the deterministic Doctor runtime.
        policy:
            Optional deterministic-doctor policy mapping or object.
        backends:
            Explicit stage injectables.  When provided, they win over live
            checkout composition so callers can force empty slots or stubs.
        bind_live_stages:
            When ``False``, skip live composition even if ``checkout_root`` is
            set (empty slots remain explicit capability gaps).
        """

        import sys

        baseline_modules = frozenset(sys.modules)
        assert_no_llm_surface_loaded(baseline_modules=baseline_modules)

        use_live = (
            checkout_root is not None
            and backends is None
            and bind_live_stages is not False
        )
        if use_live:
            service = self._build_live_service(
                checkout_root,  # type: ignore[arg-type]
                policy=policy,
                control_service=control_service,
                receipt_store=receipt_store,
                repository_allowlist=repository_allowlist,
                scope_policy=scope_policy,
                index_root=index_root,
            )
        else:
            service = self._build_explicit_service(
                checkout_root=checkout_root,
                policy=policy,
                backends=backends,
                control_service=control_service,
                receipt_store=receipt_store,
                cas=cas,
            )

        assert_no_llm_surface_loaded(baseline_modules=baseline_modules)
        return service

    def build_service(
        self,
        checkout_root: str | Path | None = None,
        **kwargs: Any,
    ) -> DeterministicDoctorService:
        """Alias for :meth:`build`."""

        return self.build(checkout_root, **kwargs)

    def _build_live_service(
        self,
        checkout_root: str | Path,
        *,
        policy: Any | None,
        control_service: Any | None,
        receipt_store: Any | None,
        repository_allowlist: Sequence[str | Path] | None,
        scope_policy: Any | None,
        index_root: str | Path | None,
    ) -> DeterministicDoctorService:
        root = _canonical_checkout(checkout_root)
        # Lazy import keeps control→runtime out of cold module load and avoids
        # a package-DAG cycle at import time (runtime already depends on control).
        from ..runtime.deterministic_doctor_runtime import (  # noqa: WPS433
            DeterministicDoctorRuntimeError,
            create_deterministic_doctor_runtime,
        )

        allowlist = repository_allowlist
        if allowlist is None:
            allowlist = self._repository_allowlist
        # When no allowlist is configured, admit only the exact resolved root.
        effective_allowlist: Sequence[str | Path] = (
            allowlist if allowlist is not None else (root,)
        )
        resolved_index_root = index_root if index_root is not None else self._index_root
        try:
            runtime = create_deterministic_doctor_runtime(
                root,
                repository_allowlist=effective_allowlist,
                policy=policy,
                control_service=(
                    control_service
                    if control_service is not None
                    else self._control_service
                ),
                receipt_store=(
                    receipt_store if receipt_store is not None else self._receipt_store
                ),
                scope_policy=(
                    scope_policy if scope_policy is not None else self._scope_policy
                ),
                index_root=resolved_index_root,
                deterministic=self._deterministic,
            )
        except DeterministicDoctorRuntimeError as exc:
            reason = str(getattr(exc, "reason_code", "") or "checkout_unavailable")
            if reason in {
                "checkout_unavailable",
                "checkout_not_directory",
                "checkout_not_allowlisted",
                "allowlist_root_unavailable",
            }:
                raise DefaultDoctorCheckoutError(
                    str(exc) or reason,
                    reason_code=reason,
                ) from exc
            raise DefaultDoctorFactoryError(
                str(exc) or reason,
                reason_code=reason,
            ) from exc
        except OSError as exc:
            # Path resolution / allowlist materialization can surface as OSError
            # before the runtime wraps it on some platforms.
            raise DefaultDoctorCheckoutError(
                f"checkout root is unavailable: {root}",
                reason_code="checkout_unavailable",
            ) from exc
        service = runtime.service
        # Bound stage methods retain the runtime via __self__; keep an explicit
        # handle for operators/tests that need capability_graph / evidence.
        self._last_runtime = runtime
        available = service.backends_available
        self._last_binding = DefaultDoctorBinding(
            checkout_root=str(root),
            live_stages_bound=True,
            backends_available=available,
            capability_gaps=_capability_gaps(available),
            policy_id=_policy_id(policy) or _policy_id(service.policy),
            notes=(
                "live_checkout_stages_bound",
                "optional_providers_not_required",
                "llm_surface_not_loaded",
            ),
        )
        return service

    def _build_explicit_service(
        self,
        *,
        checkout_root: str | Path | None,
        policy: Any | None,
        backends: DoctorStageBackends | None,
        control_service: Any | None,
        receipt_store: Any | None,
        cas: Any | None,
    ) -> DeterministicDoctorService:
        resolved_backends = backends if backends is not None else DoctorStageBackends()
        service = create_deterministic_doctor_service(
            policy=policy,
            receipt_store=(
                receipt_store if receipt_store is not None else self._receipt_store
            ),
            backends=resolved_backends,
            control_service=(
                control_service
                if control_service is not None
                else self._control_service
            ),
            cas=cas,
        )
        self._last_runtime = None
        root_text = ""
        if checkout_root is not None:
            root_text = str(_canonical_checkout(checkout_root))
        available = service.backends_available
        notes = [
            "explicit_or_empty_backends",
            "capability_gaps_are_typed_abstentions",
            "llm_surface_not_loaded",
        ]
        if not available:
            notes.append("all_stage_slots_empty")
        self._last_binding = DefaultDoctorBinding(
            checkout_root=root_text,
            live_stages_bound=False,
            backends_available=available,
            capability_gaps=_capability_gaps(available),
            policy_id=_policy_id(policy) or _policy_id(service.policy),
            notes=tuple(notes),
        )
        return service


def build_default_doctor_factory(
    *,
    deterministic: bool = True,
    repository_allowlist: Sequence[str | Path] | None = None,
    control_service: Any | None = None,
    receipt_store: Any | None = None,
    scope_policy: Any | None = None,
    index_root: str | Path | None = None,
) -> DefaultDoctorFactory:
    """Construct a production default doctor factory."""

    return DefaultDoctorFactory(
        deterministic=deterministic,
        repository_allowlist=repository_allowlist,
        control_service=control_service,
        receipt_store=receipt_store,
        scope_policy=scope_policy,
        index_root=index_root,
    )


def build_default_doctor_service(
    checkout_root: str | Path | None = None,
    **kwargs: Any,
) -> DeterministicDoctorService:
    """Build the production default :class:`DeterministicDoctorService`.

    With ``checkout_root``, live inspect/plan stage backends are bound so
    fixture repositories can be diagnosed without caller-authored snapshot
    JSON.  Without backends (or with empty injectables), operations return
    typed capability abstentions.  Default construction never loads an LLM
    surface.
    """

    # Snapshot so we only fail closed if *this construction* loads LLM clients.
    # Ambient torch/transformers from host pytest/accelerate plugins must not
    # permanently stall default Doctor factory builds (WPD-010 unblock).
    import sys

    baseline_modules = frozenset(sys.modules)
    assert_no_llm_surface_loaded(baseline_modules=baseline_modules)

    factory_kwargs: dict[str, Any] = {
        "deterministic": bool(kwargs.pop("deterministic", True)),
    }
    # Factory-level defaults (also accepted on :meth:`DefaultDoctorFactory.build`
    # for per-call overrides).  When only construction-time values are needed
    # they land here so subsequent builds inherit them.
    for key in (
        "repository_allowlist",
        "control_service",
        "receipt_store",
        "scope_policy",
        "index_root",
    ):
        if key in kwargs and kwargs[key] is not None:
            # Leave the kwarg in place so build() can still prefer the explicit
            # call-site value over the factory default.
            factory_kwargs[key] = kwargs[key]

    factory = build_default_doctor_factory(**factory_kwargs)
    service = factory.build(checkout_root, **kwargs)
    assert_no_llm_surface_loaded(baseline_modules=baseline_modules)
    return service


__all__ = [
    "DEFAULT_DOCTOR_FACTORY_BINDING_SCHEMA",
    "DEFAULT_DOCTOR_FACTORY_DISCOVERY_SCHEMA",
    "DEFAULT_DOCTOR_FACTORY_EVIDENCE",
    "DEFAULT_DOCTOR_FACTORY_INTERFACE",
    "DEFAULT_DOCTOR_FACTORY_VERSION",
    "DefaultDoctorBinding",
    "DefaultDoctorCheckoutError",
    "DefaultDoctorFactory",
    "DefaultDoctorFactoryError",
    "assert_no_llm_surface_loaded",
    "build_default_doctor_factory",
    "build_default_doctor_service",
]
