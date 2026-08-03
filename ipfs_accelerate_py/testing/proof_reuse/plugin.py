"""Cold-import-safe pytest integration for proof-backed test reuse.

Optional cache, receipt, and xdist components are imported only after proof
reuse is enabled.  In xdist runs, workers remain read/execute-only and return
bounded publication intents to the single controller.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from .config import PROOF_REUSE_MODES, ProofReuseConfig, ProofReuseMode

PLUGIN_NAME = "ipfs-proof-reuse"
CONFIG_ATTRIBUTE = "_ipfs_proof_reuse_config"
ITEM_METADATA_ATTRIBUTE = "_ipfs_proof_reuse_metadata"
METRICS_ATTRIBUTE = "_ipfs_proof_reuse_metrics"
COORDINATOR_ATTRIBUTE = "_ipfs_proof_reuse_xdist_coordinator"
LOOKUP_SERVICE_ATTRIBUTE = "_ipfs_proof_reuse_lookup_service"
STORE_SERVICE_ATTRIBUTE = "_ipfs_proof_reuse_store_service"
CANDIDATE_STORE_SERVICE_ATTRIBUTE = "_ipfs_proof_reuse_candidate_store_service"
PROVIDER_SERVICE_ATTRIBUTE = "_ipfs_proof_reuse_provider_service"
ISSUER_SERVICE_ATTRIBUTE = "_ipfs_proof_reuse_issuer_service"
DEPENDENCY_INSTALLER_ATTRIBUTE = "_ipfs_proof_reuse_dependency_installer"
SERVICE_RESOLVER_ATTRIBUTE = "_ipfs_proof_reuse_service_resolver"
SERVICE_RESOLUTION_ATTRIBUTE = "_ipfs_proof_reuse_service_resolution"
IDENTITY_SERVICES_ATTRIBUTE = "_ipfs_proof_reuse_identity_services"
IDENTITY_FACTORY_ATTRIBUTE = "_ipfs_proof_reuse_identity_factory"
RUNTIME_PLUGIN_ATTRIBUTE = "_ipfs_proof_reuse_runtime_plugin"
RUNTIME_TRACE_ATTRIBUTE = "_ipfs_proof_reuse_runtime_trace"
RUNTIME_TRACE_CAPTURE_ATTRIBUTE = "_ipfs_proof_reuse_runtime_trace_capture"
RUNTIME_LIFECYCLE_ATTRIBUTE = "_ipfs_proof_reuse_runtime_trace_lifecycle"
CANDIDATE_PUBLICATION_ATTRIBUTE = "_ipfs_proof_reuse_candidate_publication"
DEFERRED_REQUEST_ATTRIBUTE = "_ipfs_proof_reuse_deferred_request"
EXECUTION_RECORDED_ATTRIBUTE = "_ipfs_proof_reuse_execution_recorded"
DEFAULT_SERVICES_ATTRIBUTE = "_ipfs_proof_reuse_default_services"
COMPOSITION_ATTRIBUTE = "_ipfs_proof_reuse_runtime_composition"
PROOF_REUSE_RUNTIME_COMPOSITION_INTERFACE = "ProofReuseRuntimeComposition@1"

MODE_OPTION = "--proof-reuse-mode"
REQUIRED_AUDIT_OPTION = "--proof-reuse-required-audit"
MODE_INI = "proof_reuse_mode"
REQUIRED_AUDIT_INI = "proof_reuse_required_audit"

DISABLED_MARKER = "proof_reuse_disabled"
EFFECTS_MARKER = "proof_reuse_effects"

_MARKER_DESCRIPTIONS = (
    (
        DISABLED_MARKER,
        "proof_reuse_disabled(reason=None): always execute this test; no "
        "proof-backed reuse lookup or write is permitted",
    ),
    (
        EFFECTS_MARKER,
        "proof_reuse_effects(*adapters): declare reviewed effect adapter names "
        "for proof-reuse dependency tracing",
    ),
)


@dataclass(frozen=True)
class ProofReuseItemMetadata:
    """Collection facts derived directly from one pytest item."""

    nodeid: str
    disabled: bool = False
    disabled_reason: str = ""
    effect_adapters: Tuple[str, ...] = ()


def pytest_addoption(parser: Any) -> None:
    """Register the cold shell's CLI and ini configuration."""

    group = parser.getgroup(
        "proof-reuse",
        "proof-backed reuse of exact pytest pass evidence",
    )
    group.addoption(
        MODE_OPTION,
        action="store",
        dest="proof_reuse_mode",
        choices=PROOF_REUSE_MODES,
        default=None,
        metavar="MODE",
        help=(
            "proof reuse mode: off, shadow, read, write, or readwrite "
            "(default: IPFS_TEST_PROOF_REUSE_MODE or off)"
        ),
    )
    group.addoption(
        REQUIRED_AUDIT_OPTION,
        action="store_true",
        dest="proof_reuse_required_audit",
        default=False,
        help=(
            "enable the separate CI required-audit policy; this is not a "
            "proof reuse mode"
        ),
    )
    parser.addini(
        MODE_INI,
        "proof reuse mode (off, shadow, read, write, or readwrite)",
        default="",
    )
    parser.addini(
        REQUIRED_AUDIT_INI,
        "enable the separate proof reuse required-audit CI policy",
        type="bool",
        default=False,
    )


def _getoption(config: Any, name: str, default: Any = None) -> Any:
    try:
        return config.getoption(name, default=default)
    except (AttributeError, TypeError, ValueError):
        return default


def _getini(config: Any, name: str, default: Any = None) -> Any:
    try:
        return config.getini(name)
    except (AttributeError, KeyError, TypeError, ValueError):
        return default


def get_proof_reuse_config(config: Any) -> ProofReuseConfig:
    """Return the resolved config, safely defaulting to off if unconfigured."""

    existing = getattr(config, CONFIG_ATTRIBUTE, None)
    if isinstance(existing, ProofReuseConfig):
        return existing
    return ProofReuseConfig()


def pytest_configure(config: Any) -> None:
    """Resolve configuration and install enabled runtime coordination."""

    for _marker_name, description in _MARKER_DESCRIPTIONS:
        config.addinivalue_line("markers", description)

    resolved = ProofReuseConfig.resolve(
        command_line_mode=_getoption(config, "proof_reuse_mode"),
        ini_mode=_getini(config, MODE_INI, ""),
        environ=os.environ,
        command_line_required_audit=_getoption(
            config,
            "proof_reuse_required_audit",
            False,
        ),
        ini_required_audit=_getini(config, REQUIRED_AUDIT_INI, False),
    )
    setattr(config, CONFIG_ATTRIBUTE, resolved)
    if not resolved.enabled:
        return

    from .reporting import ProofReuseSessionMetrics
    from .xdist import WORKER_INPUT_KEY, ProofReuseXdistCoordinator

    metrics = getattr(config, METRICS_ATTRIBUTE, None)
    if not isinstance(metrics, ProofReuseSessionMetrics):
        metrics = ProofReuseSessionMetrics()
        setattr(config, METRICS_ATTRIBUTE, metrics)

    worker_input = getattr(config, "workerinput", None)
    if isinstance(worker_input, Mapping):
        worker_id = str(worker_input.get("workerid", ""))
        coordinator = ProofReuseXdistCoordinator.from_worker_input(
            worker_input.get(WORKER_INPUT_KEY),
            metrics=metrics,
            worker_id=worker_id,
        )
    else:
        coordinator = ProofReuseXdistCoordinator.standalone(metrics=metrics)
    setattr(config, COORDINATOR_ATTRIBUTE, coordinator)
    _install_runtime_plugin(config)


def set_proof_reuse_services(
    config: Any,
    *,
    lookup: Any = None,
    store: Any = None,
    candidate_store: Any = None,
    provider: Any = None,
    issuer: Any = None,
) -> None:
    """Inject optional runtime services without probing providers at import."""

    setattr(config, LOOKUP_SERVICE_ATTRIBUTE, lookup)
    setattr(config, STORE_SERVICE_ATTRIBUTE, store)
    if candidate_store is not None:
        setattr(config, CANDIDATE_STORE_SERVICE_ATTRIBUTE, candidate_store)
    setattr(config, PROVIDER_SERVICE_ATTRIBUTE, provider)
    setattr(config, ISSUER_SERVICE_ATTRIBUTE, issuer)


def set_proof_reuse_dependency_installer(
    config: Any,
    installer: Any,
) -> None:
    """Inject a controlled lazy installer before ``pytest_configure``.

    The installer is consulted only for the closed dependency allowlist and
    only after an enabled proof-reuse mode observes that exact module missing.
    """

    if installer is not None:
        install = getattr(installer, "install", None)
        if not callable(installer) and not callable(install):
            raise TypeError("installer must be callable or expose install()")
    setattr(config, DEPENDENCY_INSTALLER_ATTRIBUTE, installer)


def set_proof_reuse_service_resolver(
    config: Any,
    resolver: Any,
) -> None:
    """Inject a managed service resolver for hermetic environments/tests."""

    if resolver is not None and not callable(getattr(resolver, "resolve", None)):
        raise TypeError("resolver must expose resolve()")
    setattr(config, SERVICE_RESOLVER_ATTRIBUTE, resolver)


def _proof_reuse_cache_root(config: Any) -> str:
    from .services import PROOF_REUSE_CACHE_DIR_ENV

    configured = os.environ.get(PROOF_REUSE_CACHE_DIR_ENV, "").strip()
    if configured:
        return os.path.abspath(os.path.expanduser(configured))
    root = getattr(config, "rootpath", None)
    if root is None:
        root = getattr(config, "rootdir", None)
    if root is None:
        root = os.getcwd()
    return os.path.join(
        os.path.abspath(os.fspath(root)),
        ".pytest_cache",
        "proof-reuse",
    )


def _config_root_path(config: Any) -> str | None:
    root = getattr(config, "rootpath", None)
    if root is None:
        root = getattr(config, "rootdir", None)
    if root is None:
        return None
    try:
        return os.path.abspath(os.fspath(root))
    except (TypeError, ValueError, OSError):
        return None


def _inject_default_services(config: Any) -> None:
    """Assemble enabled services once; every failure leaves tests runnable."""

    if all(
        getattr(config, attribute, None) is not None
        for attribute in (
            LOOKUP_SERVICE_ATTRIBUTE,
            STORE_SERVICE_ATTRIBUTE,
            PROVIDER_SERVICE_ATTRIBUTE,
        )
    ):
        return

    from .lazy_dependencies import (
        ProofReuseLazyDependencyInstaller,
        proof_reuse_install_permitted,
    )
    from .services import (
        DefaultProofReuseServices,
        LazyProofReuseServiceResolver,
        ProofReuseServiceResolution,
        compose_default_proof_reuse_services,
    )

    proof_config = get_proof_reuse_config(config)
    resolver = getattr(config, SERVICE_RESOLVER_ATTRIBUTE, None)
    installer = getattr(config, DEPENDENCY_INSTALLER_ATTRIBUTE, None)
    worker_input = getattr(config, "workerinput", None)
    if (
        installer is None
        and not isinstance(worker_input, Mapping)
        and proof_reuse_install_permitted(os.environ)
    ):
        try:
            installer = ProofReuseLazyDependencyInstaller()
        except Exception:
            installer = None
    if resolver is None:
        try:
            resolver = LazyProofReuseServiceResolver(installer=installer)
        except Exception:
            resolver = None
        if resolver is not None:
            setattr(config, SERVICE_RESOLVER_ATTRIBUTE, resolver)

    try:
        defaults = compose_default_proof_reuse_services(
            mode=proof_config.mode,
            root_path=_config_root_path(config),
            config=config,
            cache_root=_proof_reuse_cache_root(config),
            resolver=resolver,
            installer=installer,
            identity_services=getattr(config, IDENTITY_SERVICES_ATTRIBUTE, None),
            lookup=getattr(config, LOOKUP_SERVICE_ATTRIBUTE, None),
            store=getattr(config, STORE_SERVICE_ATTRIBUTE, None),
            candidate_store=getattr(config, CANDIDATE_STORE_SERVICE_ATTRIBUTE, None),
            provider=getattr(config, PROVIDER_SERVICE_ATTRIBUTE, None),
            issuer=getattr(config, ISSUER_SERVICE_ATTRIBUTE, None),
        )
    except Exception:
        defaults = DefaultProofReuseServices(
            degraded=True,
            reason_code="plugin_unavailable",
        )
    setattr(config, DEFAULT_SERVICES_ATTRIBUTE, defaults)

    resolution = defaults.resolution
    if resolution is None:
        try:
            resolution = (
                resolver.resolve(cache_root=_proof_reuse_cache_root(config))
                if resolver is not None
                else ProofReuseServiceResolution.unavailable("plugin_unavailable")
            )
        except Exception:
            resolution = ProofReuseServiceResolution.unavailable("plugin_unavailable")
    setattr(config, SERVICE_RESOLUTION_ATTRIBUTE, resolution)
    if defaults.degraded:
        metrics = getattr(config, METRICS_ATTRIBUTE, None)
        if metrics is not None and defaults.reason_code:
            metrics.degraded(reason_code=defaults.reason_code)

    if not isinstance(resolution, ProofReuseServiceResolution):
        return
    if not resolution.available:
        metrics = getattr(config, METRICS_ATTRIBUTE, None)
        if metrics is not None:
            metrics.degraded(reason_code=resolution.reason_code)
        # Fail closed for lookup/store/provider: never inject a partial
        # authority path when the optional provider resolution failed.
        # Non-authoritative helpers (lazy issuer, candidate store) may still
        # be retained on DEFAULT_SERVICES_ATTRIBUTE only.
        return

    for attribute, service in (
        # Prefer resolution handles for object-identity stability with the
        # memoized LazyProofReuseServiceResolver result (tests and production).
        (LOOKUP_SERVICE_ATTRIBUTE, resolution.lookup or defaults.lookup),
        (STORE_SERVICE_ATTRIBUTE, resolution.store or defaults.store),
        (
            CANDIDATE_STORE_SERVICE_ATTRIBUTE,
            getattr(defaults, "candidate_store", None),
        ),
        (PROVIDER_SERVICE_ATTRIBUTE, resolution.provider or defaults.provider),
        (ISSUER_SERVICE_ATTRIBUTE, defaults.issuer),
    ):
        if service is not None and getattr(config, attribute, None) is None:
            setattr(config, attribute, service)


@dataclass
class ProofReuseRuntimeComposition:
    """Compose lookup, revalidation, receipt capture, deferred issuance, xdist.

    Implements ``ProofReuseRuntimeComposition@1``.  The plugin owns
    orchestration but not trust: every optional-boundary failure degrades to
    normal pytest execution or a retained deferred receipt.
    """

    config: Any
    services: Any = None

    @property
    def interface(self) -> str:
        return PROOF_REUSE_RUNTIME_COMPOSITION_INTERFACE

    def ensure_identity_factory(self) -> Any:
        """Return the session-scoped default identity factory (PTR-143).

        The factory supplies :meth:`obtain_static_identity` for locator-first
        collection seeds.  Hermetic shells without a repository root keep
        ``None`` so unit tests observe typed fail-open diagnostics rather than
        a fabricated forest.
        """

        existing = getattr(self.config, IDENTITY_FACTORY_ATTRIBUTE, None)
        if existing is not None:
            return existing
        root = _config_root_path(self.config)
        if root is None:
            return None
        try:
            from .default_identity_services import DefaultIdentityServiceFactory

            proof_config = get_proof_reuse_config(self.config)
            factory = DefaultIdentityServiceFactory(
                mode=proof_config.mode,
                root_path=root,
                config=self.config,
            )
            setattr(self.config, IDENTITY_FACTORY_ATTRIBUTE, factory)
            # Also materialize the services bundle for full assembly upgrades.
            if getattr(self.config, IDENTITY_SERVICES_ATTRIBUTE, None) is None:
                setattr(
                    self.config,
                    IDENTITY_SERVICES_ATTRIBUTE,
                    factory.build_services(),
                )
            return factory
        except Exception:
            metrics = getattr(self.config, METRICS_ATTRIBUTE, None)
            if metrics is not None:
                metrics.degraded(reason_code="identity_factory_unavailable")
            return None

    def ensure_identity_services(self) -> Any:
        """Return session-scoped identity defaults without item registries."""

        existing = getattr(self.config, IDENTITY_SERVICES_ATTRIBUTE, None)
        if existing is not None:
            return existing
        factory = self.ensure_identity_factory()
        if factory is not None:
            try:
                services = factory.build_services()
                setattr(self.config, IDENTITY_SERVICES_ATTRIBUTE, services)
                return services
            except Exception:
                metrics = getattr(self.config, METRICS_ATTRIBUTE, None)
                if metrics is not None:
                    metrics.degraded(reason_code="identity_services_unavailable")
                return None
        root = _config_root_path(self.config)
        if root is None:
            # Hermetic shells without a root keep an empty provider bundle so
            # existing unit tests observe PROVIDER_UNAVAILABLE rather than a
            # fabricated root.
            return None
        try:
            from .default_identity_services import build_default_identity_services

            proof_config = get_proof_reuse_config(self.config)
            services = build_default_identity_services(
                mode=proof_config.mode,
                root_path=root,
                config=self.config,
            )
            setattr(self.config, IDENTITY_SERVICES_ATTRIBUTE, services)
            return services
        except Exception:
            metrics = getattr(self.config, METRICS_ATTRIBUTE, None)
            if metrics is not None:
                metrics.degraded(reason_code="identity_services_unavailable")
            return None

    def ensure_runtime_services(self) -> Any:
        try:
            _inject_default_services(self.config)
        except Exception:
            metrics = getattr(self.config, METRICS_ATTRIBUTE, None)
            if metrics is not None:
                metrics.degraded(reason_code="plugin_unavailable")
        return getattr(self.config, DEFAULT_SERVICES_ATTRIBUTE, None)

    def attach_post_pass_capture(self, item: Any) -> Any:
        """Attach a post-pass runtime observer that never re-invokes the body."""

        existing = getattr(item, RUNTIME_TRACE_CAPTURE_ATTRIBUTE, None)
        if existing is not None:
            return existing
        try:
            from .runtime_revalidation import PostPassRuntimeTraceCapture

            locator = getattr(item, "_ipfs_proof_reuse_locator", None)
            execution_key = getattr(item, "_ipfs_proof_reuse_execution_key", None)
            capture = PostPassRuntimeTraceCapture(
                locator_cid=str(
                    getattr(locator, "locator_id", None)
                    or getattr(locator, "content_id", None)
                    or ""
                ),
                execution_key_cid=str(
                    getattr(execution_key, "execution_key_id", None)
                    or getattr(execution_key, "content_id", None)
                    or ""
                ),
            )
            setattr(item, RUNTIME_TRACE_CAPTURE_ATTRIBUTE, capture)
            return capture
        except Exception:
            return None

    def attach_runtime_lifecycle(self, item: Any) -> Any:
        """Attach the cold-pass tracer lifecycle (PTR-146).

        The lifecycle starts immediately before setup and stops only after
        teardown.  It never re-invokes the test body.
        """

        existing = getattr(item, RUNTIME_LIFECYCLE_ATTRIBUTE, None)
        if existing is not None:
            return existing
        try:
            from .runtime_trace_lifecycle import attach_runtime_lifecycle

            root = _config_root_path(self.config)
            allowed_roots = {"repo": root} if root else None
            lifecycle = attach_runtime_lifecycle(
                item,
                allowed_roots=allowed_roots,
                capture_code_objects=False,
            )
            return lifecycle
        except Exception:
            metrics = getattr(self.config, METRICS_ATTRIBUTE, None)
            if metrics is not None:
                metrics.degraded(reason_code="runtime_trace_lifecycle_unavailable")
            return None

    def start_runtime_lifecycle(self, item: Any) -> Any:
        """Start observation immediately before setup. Never raises."""

        try:
            lifecycle = self.attach_runtime_lifecycle(item)
            if lifecycle is None:
                return None
            start = getattr(lifecycle, "start", None)
            if callable(start):
                start()
            return lifecycle
        except Exception:
            metrics = getattr(self.config, METRICS_ATTRIBUTE, None)
            if metrics is not None:
                metrics.degraded(reason_code="runtime_trace_start_failed")
            return None

    def stop_runtime_lifecycle(self, item: Any) -> Any:
        """Stop observation after teardown and attach the observed trace."""

        try:
            lifecycle = getattr(item, RUNTIME_LIFECYCLE_ATTRIBUTE, None)
            if lifecycle is None:
                return None
            stop = getattr(lifecycle, "stop", None)
            trace = stop() if callable(stop) else None
            if trace is not None:
                try:
                    setattr(item, RUNTIME_TRACE_ATTRIBUTE, trace)
                except Exception:
                    pass
            return trace
        except Exception:
            metrics = getattr(self.config, METRICS_ATTRIBUTE, None)
            if metrics is not None:
                metrics.degraded(reason_code="runtime_trace_stop_failed")
            return None

    def note_phase(self, item: Any, report: Any) -> None:
        # Cold-pass lifecycle (primary): record every phase outcome.
        lifecycle = getattr(item, RUNTIME_LIFECYCLE_ATTRIBUTE, None)
        if lifecycle is not None:
            try:
                note = getattr(lifecycle, "note_report", None)
                if callable(note):
                    note(report)
                else:
                    when = str(getattr(report, "when", ""))
                    outcome = str(getattr(report, "outcome", ""))
                    lifecycle.note_phase(when, outcome)
            except Exception:
                metrics = getattr(self.config, METRICS_ATTRIBUTE, None)
                if metrics is not None:
                    metrics.degraded(reason_code="runtime_trace_lifecycle_failed")

        capture = getattr(item, RUNTIME_TRACE_CAPTURE_ATTRIBUTE, None)
        if capture is None:
            return
        when = str(getattr(report, "when", ""))
        outcome = str(getattr(report, "outcome", ""))
        try:
            if when == "setup" and outcome == "passed":
                capture.note_setup()
            elif when == "call" and outcome == "passed":
                capture.note_call()
            elif when == "teardown":
                # Teardown is always noted so incomplete teardown disqualifies.
                capture.note_teardown()
        except Exception:
            # Capture faults never change pytest outcome.
            metrics = getattr(self.config, METRICS_ATTRIBUTE, None)
            if metrics is not None:
                metrics.degraded(reason_code="runtime_trace_capture_failed")

    def build_public_deferred_envelope(self, item: Any, receipt: Any) -> Any:
        try:
            from .receipt import DeferredIssuanceEnvelope

            existing = getattr(item, DEFERRED_REQUEST_ATTRIBUTE, None)
            if existing is not None:
                envelope = DeferredIssuanceEnvelope.from_mapping(existing)
                if envelope is not None:
                    return envelope.to_dict()
            envelope = DeferredIssuanceEnvelope.from_admitted_receipt(
                receipt,
                locator_cid=str(getattr(receipt, "locator_cid", "") or ""),
            )
            if envelope is None:
                return None
            public = envelope.to_dict()
            setattr(item, DEFERRED_REQUEST_ATTRIBUTE, public)
            return public
        except Exception:
            return None


def set_proof_reuse_identity_services(config: Any, services: Any) -> None:
    """Inject the validated automatic item-identity service bundle.

    The assembler module is cold-safe and imported only when this explicit
    setter is used or when enabled collection needs it.  Provider callbacks
    are never called here.
    """

    from .item_identity import ItemIdentityAssemblyServices

    if not isinstance(services, ItemIdentityAssemblyServices):
        raise TypeError("services must be ItemIdentityAssemblyServices")
    setattr(config, IDENTITY_SERVICES_ATTRIBUTE, services)


def _install_runtime_plugin(config: Any) -> None:
    if getattr(config, RUNTIME_PLUGIN_ATTRIBUTE, None) is not None:
        return
    composition = getattr(config, COMPOSITION_ATTRIBUTE, None)
    if not isinstance(composition, ProofReuseRuntimeComposition):
        composition = ProofReuseRuntimeComposition(config=config)
        setattr(config, COMPOSITION_ATTRIBUTE, composition)
    try:
        import pytest
    except Exception:
        return

    class _ProofReuseRuntimePlugin:
        @pytest.hookimpl(hookwrapper=True, tryfirst=True)
        def pytest_runtest_protocol(self, item: Any, nextitem: Any) -> Any:
            # PTR-146: start the production tracer immediately before setup
            # and stop only after teardown.  The body is invoked once by
            # pytest itself; this wrapper never re-enters it.
            try:
                composition = getattr(config, COMPOSITION_ATTRIBUTE, None)
                if not isinstance(composition, ProofReuseRuntimeComposition):
                    composition = ProofReuseRuntimeComposition(config=config)
                    setattr(config, COMPOSITION_ATTRIBUTE, composition)
                proof_config = get_proof_reuse_config(config)
                if proof_config.writes_receipts:
                    composition.start_runtime_lifecycle(item)
            except Exception:
                metrics = getattr(config, METRICS_ATTRIBUTE, None)
                if metrics is not None:
                    metrics.degraded(reason_code="runtime_trace_start_failed")
            outcome = yield
            try:
                composition = getattr(config, COMPOSITION_ATTRIBUTE, None)
                if isinstance(composition, ProofReuseRuntimeComposition):
                    composition.stop_runtime_lifecycle(item)
            except Exception:
                metrics = getattr(config, METRICS_ATTRIBUTE, None)
                if metrics is not None:
                    metrics.degraded(reason_code="runtime_trace_stop_failed")
            return outcome

        @pytest.hookimpl(hookwrapper=True, trylast=True)
        def pytest_runtest_makereport(self, item: Any, call: Any) -> Any:
            outcome = yield
            try:
                report = outcome.get_result()
                _record_runtime_report(config, item, report)
            except Exception:
                metrics = getattr(config, METRICS_ATTRIBUTE, None)
                if metrics is not None:
                    metrics.degraded(reason_code="runtime_hook_failed")

    runtime = _ProofReuseRuntimePlugin()
    try:
        config.pluginmanager.register(
            runtime,
            name=f"{PLUGIN_NAME}-runtime",
        )
    except Exception:
        metrics = getattr(config, METRICS_ATTRIBUTE, None)
        if metrics is not None:
            metrics.degraded(reason_code="runtime_registration_failed")
        return
    setattr(config, RUNTIME_PLUGIN_ATTRIBUTE, runtime)


def _bounded_marker_text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        return ""
    return value.strip()[:512]


def _marker_reason(marker: Any) -> str:
    if marker is None:
        return ""
    reason = getattr(marker, "kwargs", {}).get("reason")
    if reason is None:
        args = getattr(marker, "args", ())
        reason = args[0] if args else ""
    return _bounded_marker_text(reason)


def _effect_adapters(item: Any) -> Tuple[str, ...]:
    adapters = []
    seen = set()
    try:
        markers: Iterable[Any] = item.iter_markers(name=EFFECTS_MARKER)
    except (AttributeError, TypeError):
        marker = item.get_closest_marker(EFFECTS_MARKER)
        markers = () if marker is None else (marker,)
    for marker in markers:
        raw_values = list(getattr(marker, "args", ()))
        keyword_values = getattr(marker, "kwargs", {}).get("adapters", ())
        if isinstance(keyword_values, str):
            raw_values.append(keyword_values)
        else:
            try:
                raw_values.extend(keyword_values)
            except TypeError:
                # Malformed marker metadata must never disrupt collection.
                pass
        for raw_value in raw_values:
            adapter = _bounded_marker_text(raw_value)
            if adapter and adapter not in seen:
                seen.add(adapter)
                adapters.append(adapter)
    return tuple(adapters)


def collect_item_metadata(item: Any) -> ProofReuseItemMetadata:
    """Build metadata from a direct collected node, without a path registry."""

    disabled_marker = item.get_closest_marker(DISABLED_MARKER)
    return ProofReuseItemMetadata(
        nodeid=str(getattr(item, "nodeid", ""))[:2048],
        disabled=disabled_marker is not None,
        disabled_reason=_marker_reason(disabled_marker),
        effect_adapters=_effect_adapters(item),
    )


def get_item_metadata(item: Any) -> Optional[ProofReuseItemMetadata]:
    metadata = getattr(item, ITEM_METADATA_ATTRIBUTE, None)
    if isinstance(metadata, ProofReuseItemMetadata):
        return metadata
    return None


def pytest_collection_modifyitems(config: Any, items: Iterable[Any]) -> None:
    """Attach metadata, perform reads, and prepare controller-owned writes."""

    proof_config = get_proof_reuse_config(config)
    if proof_config.mode is ProofReuseMode.OFF:
        return
    collected = tuple(items)
    for item in collected:
        setattr(item, ITEM_METADATA_ATTRIBUTE, collect_item_metadata(item))

    from ...agent_supervisor.proof.test_execution_contracts import ReuseDecision
    from .lookup import (
        ITEM_LOOKUP_REQUEST_ATTRIBUTE,
        ProofReuseLookup,
        batch_lookup_reuse_decisions,
    )
    from .receipt import attach_collector
    from .xdist import ProofReuseXdistCoordinator, force_real_execution

    # Item identity is assembled through one session-scoped DI boundary.  An
    # absent bundle is represented by an empty bundle: each enabled item then
    # receives a typed RUN diagnostic and no lookup request.  Existing manual
    # identities are detected by the assembler and left untouched.  Explicit
    # injections remain authoritative; otherwise a root-scoped default factory
    # supplies session-memoized providers without per-test registries.
    #
    # PTR-143: collection first attaches a locator-first collection seed and
    # stable locator without runtime evidence, fixture calls, or a final
    # execution key.  Full assembly may later upgrade when runtime evidence is
    # available; it never fabricates that evidence at collection.
    from .collection_seed import assemble_and_attach_collection_seed
    from .item_identity import (
        ItemIdentityAssemblyServices,
        assemble_and_attach_item_identity,
    )

    composition = getattr(config, COMPOSITION_ATTRIBUTE, None)
    if not isinstance(composition, ProofReuseRuntimeComposition):
        composition = ProofReuseRuntimeComposition(config=config)
        setattr(config, COMPOSITION_ATTRIBUTE, composition)

    metrics = getattr(config, METRICS_ATTRIBUTE, None)
    identity_factory = getattr(config, IDENTITY_FACTORY_ATTRIBUTE, None)
    if identity_factory is None:
        identity_factory = composition.ensure_identity_factory()
    identity_services = getattr(config, IDENTITY_SERVICES_ATTRIBUTE, None)
    if not isinstance(identity_services, ItemIdentityAssemblyServices):
        defaults = composition.ensure_identity_services()
        if isinstance(defaults, ItemIdentityAssemblyServices):
            identity_services = defaults
        else:
            identity_services = ItemIdentityAssemblyServices()
    for item in collected:
        metadata = get_item_metadata(item)
        if metadata is None or metadata.disabled:
            continue
        try:
            # Locator-first static seed: no runtime trace, no execution key.
            assemble_and_attach_collection_seed(
                item,
                factory=identity_factory,
                services=identity_services,
                mode=proof_config.mode,
            )
        except Exception:
            if metrics is not None:
                metrics.degraded(reason_code="collection_seed_failed")
        try:
            # Full assembly remains for explicit runtime-evidence injection and
            # upgrades from an intermediate collection seed.  Without runtime
            # evidence it fails open to RUN and attaches no lookup request.
            assemble_and_attach_item_identity(item, identity_services)
        except Exception:
            if metrics is not None:
                metrics.degraded(reason_code="identity_assembly_failed")
            continue
        if proof_config.writes_receipts:
            composition.attach_post_pass_capture(item)
            # PTR-146: prepare the cold-pass lifecycle for one protocol run.
            composition.attach_runtime_lifecycle(item)

    coordinator = getattr(config, COORDINATOR_ATTRIBUTE, None)
    if not isinstance(coordinator, ProofReuseXdistCoordinator):
        if metrics is not None:
            metrics.degraded(reason_code="coordination_unavailable")
        for item in collected:
            force_real_execution(item)
        return
    if not coordinator.healthy:
        coordinator.mark_controller_unavailable(collected)
        return

    request_items = [
        item
        for item in collected
        if (
            getattr(item, ITEM_LOOKUP_REQUEST_ATTRIBUTE, None) is not None
            or (
                getattr(item, "_ipfs_proof_reuse_locator", None) is not None
                and getattr(
                    item,
                    "_ipfs_proof_reuse_execution_key",
                    None,
                )
                is not None
            )
        )
    ]
    lookup = getattr(config, LOOKUP_SERVICE_ATTRIBUTE, None)
    if (
        proof_config.reads_candidates
        and request_items
        and not isinstance(lookup, ProofReuseLookup)
    ):
        # Resolve optional CID/cache/verifier dependencies only when collection
        # produced an exact lookup identity. Ordinary fail-open execution never
        # imports or installs those providers.
        _inject_default_services(config)
        lookup = getattr(config, LOOKUP_SERVICE_ATTRIBUTE, None)
    if (
        proof_config.reads_candidates
        and request_items
        and isinstance(lookup, ProofReuseLookup)
    ):
        decisions = batch_lookup_reuse_decisions(
            lookup,
            request_items,
            apply_skips=proof_config.may_skip and coordinator.can_skip,
        )
        if metrics is not None:
            for decision in decisions:
                if not isinstance(decision, ReuseDecision):
                    metrics.degraded(reason_code="lookup_decision_invalid")
                    continue
                if decision.is_skip:
                    metrics.predicted(reason_code=decision.reason_code)
                    metrics.verified(reason_code=decision.reason_code)
                    if proof_config.may_skip and coordinator.can_skip:
                        metrics.skipped(reason_code=decision.reason_code)

    if proof_config.writes_receipts and coordinator.can_accept_publication:
        for item in collected:
            metadata = get_item_metadata(item)
            if metadata is None or metadata.disabled:
                continue
            attach_collector(item)


def _record_runtime_report(config: Any, item: Any, report: Any) -> None:
    """Record execution and queue a complete-pass intent after teardown."""

    from ...agent_supervisor.proof.test_execution_contracts import ReuseDecision
    from .lookup import ITEM_DECISION_ATTRIBUTE, ITEM_LOOKUP_REQUEST_ATTRIBUTE
    from .receipt import (
        ITEM_COLLECTOR_ATTRIBUTE,
        TestPassReceiptCollector,
        finalize_test_pass_receipt,
        public_deferred_mapping,
    )
    from .xdist import ProofReuseXdistCoordinator

    metrics = getattr(config, METRICS_ATTRIBUTE, None)
    coordinator = getattr(config, COORDINATOR_ATTRIBUTE, None)
    composition = getattr(config, COMPOSITION_ATTRIBUTE, None)
    if not isinstance(composition, ProofReuseRuntimeComposition):
        composition = ProofReuseRuntimeComposition(config=config)
        setattr(config, COMPOSITION_ATTRIBUTE, composition)

    collector = getattr(item, ITEM_COLLECTOR_ATTRIBUTE, None)
    if isinstance(collector, TestPassReceiptCollector):
        collector.record_report(report)

    try:
        composition.note_phase(item, report)
    except Exception:
        if metrics is not None:
            metrics.degraded(reason_code="runtime_trace_capture_failed")

    when = str(getattr(report, "when", ""))
    outcome = str(getattr(report, "outcome", ""))
    decision = getattr(item, ITEM_DECISION_ATTRIBUTE, None)
    proof_skipped = isinstance(decision, ReuseDecision) and decision.is_skip
    already_recorded = bool(getattr(item, EXECUTION_RECORDED_ATTRIBUTE, False))
    execution_terminal = (
        when == "call"
        or (when == "setup" and outcome != "passed")
        or when == "teardown"
    )
    if (
        metrics is not None
        and execution_terminal
        and not proof_skipped
        and not already_recorded
    ):
        duration_ms = max(
            0.0,
            float(getattr(report, "duration", 0.0) or 0.0) * 1000.0,
        )
        metrics.executed(latency_ms=duration_ms)
        setattr(item, EXECUTION_RECORDED_ATTRIBUTE, True)

    if when != "teardown" or not isinstance(collector, TestPassReceiptCollector):
        return
    proof_config = get_proof_reuse_config(config)
    if (
        not proof_config.writes_receipts
        or proof_skipped
        or not isinstance(coordinator, ProofReuseXdistCoordinator)
        or not coordinator.can_accept_publication
    ):
        return

    request = getattr(item, ITEM_LOOKUP_REQUEST_ATTRIBUTE, None)
    locator = getattr(request, "locator", None)
    execution_key = getattr(request, "execution_key", None)
    if locator is None:
        locator = getattr(item, "_ipfs_proof_reuse_locator", None)
    if execution_key is None:
        execution_key = getattr(
            item,
            "_ipfs_proof_reuse_execution_key",
            None,
        )
    # Prefer the cold-pass lifecycle stop result (PTR-146).  The protocol
    # wrapper stops after teardown; if a report arrives first, stop here.
    runtime_trace = getattr(item, RUNTIME_TRACE_ATTRIBUTE, None)
    lifecycle = getattr(item, RUNTIME_LIFECYCLE_ATTRIBUTE, None)
    if runtime_trace is None and lifecycle is not None:
        try:
            if not getattr(lifecycle, "stopped", False):
                runtime_trace = composition.stop_runtime_lifecycle(item)
            else:
                runtime_trace = getattr(lifecycle, "trace", None) or getattr(
                    lifecycle, "runtime_trace", None
                )
                if runtime_trace is not None:
                    setattr(item, RUNTIME_TRACE_ATTRIBUTE, runtime_trace)
        except Exception:
            if metrics is not None:
                metrics.degraded(reason_code="runtime_trace_stop_failed")
    capture = getattr(item, RUNTIME_TRACE_CAPTURE_ATTRIBUTE, None)
    if runtime_trace is None and capture is not None:
        # Prefer the complete observed post-pass capture when the item did not
        # attach an explicit runtime trace object.
        try:
            if getattr(capture, "lifecycle_complete", False):
                observation = getattr(capture, "observation", None)
                if observation is not None:
                    runtime_trace = observation
                else:
                    runtime_trace = capture
        except Exception:
            runtime_trace = capture

    # PTR-146: compile final execution key from the complete observed trace,
    # finalize the receipt bound to that key, and assemble the candidate
    # publication envelope.  Incomplete / skipped / failed paths publish
    # nothing authoritative.
    result = None
    try:
        from .candidate_publication import finalize_cold_pass_publication
        from .collection_seed import ITEM_COLLECTION_SEED_ATTRIBUTE

        seed = getattr(item, ITEM_COLLECTION_SEED_ATTRIBUTE, None)
        static_cid = ""
        forest_cid = ""
        if seed is not None:
            static_cid = str(getattr(seed, "static_trace_root_cid", "") or "")
            forest_cid = str(getattr(seed, "forest_id", "") or "")
        result, _completed, publication = finalize_cold_pass_publication(
            collector=collector,
            runtime_trace=runtime_trace,
            locator=locator,
            seed_execution_key=execution_key,
            repository_forest_cid=forest_cid,
            static_trace_root_cid=static_cid,
            require_runtime_trace=True,
            item=item,
        )
        if publication is not None:
            try:
                setattr(item, CANDIDATE_PUBLICATION_ATTRIBUTE, publication)
            except Exception:
                pass
    except Exception:
        if metrics is not None:
            metrics.degraded(reason_code="cold_pass_publication_failed")
        result = None

    if result is None:
        result = finalize_test_pass_receipt(
            collector,
            locator=locator,
            execution_key=execution_key,
            runtime_trace=runtime_trace,
            writes_receipts=False,
            require_runtime_trace=True,
            item=item,
        )
    if result.admitted and result.receipt is not None:
        deferred = composition.build_public_deferred_envelope(
            item,
            result.receipt,
        )
        if deferred is None:
            deferred = public_deferred_mapping(
                getattr(item, DEFERRED_REQUEST_ATTRIBUTE, None)
            )
        coordinator.queue_publication(
            result.receipt,
            deferred_request=deferred,
        )


def pytest_configure_node(node: Any) -> None:
    """Give each xdist worker a unique authenticated controller capability."""

    config = getattr(node, "config", None)
    if get_proof_reuse_config(config).mode is ProofReuseMode.OFF:
        return
    from .xdist import (
        COORDINATION_UNAVAILABLE,
        WORKER_INPUT_KEY,
        ProofReuseXdistCoordinator,
        ProofReuseXdistRole,
    )

    metrics = getattr(config, METRICS_ATTRIBUTE, None)
    coordinator = getattr(config, COORDINATOR_ATTRIBUTE, None)
    if not isinstance(coordinator, ProofReuseXdistCoordinator):
        coordinator = ProofReuseXdistCoordinator.controller(metrics=metrics)
        setattr(config, COORDINATOR_ATTRIBUTE, coordinator)
    elif coordinator.role is ProofReuseXdistRole.STANDALONE:
        coordinator = ProofReuseXdistCoordinator.controller(metrics=coordinator.metrics)
        setattr(config, COORDINATOR_ATTRIBUTE, coordinator)

    worker_input = getattr(node, "workerinput", None)
    gateway = getattr(node, "gateway", None)
    worker_id = str(
        getattr(gateway, "id", "")
        or (
            worker_input.get("workerid", "")
            if isinstance(worker_input, Mapping)
            else ""
        )
    )
    try:
        if not isinstance(worker_input, dict):
            raise TypeError("xdist worker input is unavailable")
        worker_input[WORKER_INPUT_KEY] = coordinator.configure_worker(worker_id)
    except Exception:
        coordinator.mark_controller_unavailable()
        if metrics is not None:
            metrics.degraded(reason_code=COORDINATION_UNAVAILABLE)


def pytest_testnodedown(node: Any, error: Any) -> None:
    """Merge a worker result once; malformed outputs carry no authority."""

    config = getattr(node, "config", None)
    if get_proof_reuse_config(config).mode is ProofReuseMode.OFF:
        return
    from .xdist import WORKER_OUTPUT_KEY, ProofReuseXdistCoordinator

    metrics = getattr(config, METRICS_ATTRIBUTE, None)
    coordinator = getattr(config, COORDINATOR_ATTRIBUTE, None)
    if not isinstance(coordinator, ProofReuseXdistCoordinator):
        if metrics is not None:
            metrics.degraded(reason_code="coordination_unavailable")
        return
    if error is not None:
        if metrics is not None:
            metrics.degraded(reason_code="worker_crash")
        coordinator.mark_controller_unavailable()
        return
    worker_output = getattr(node, "workeroutput", None)
    payload = (
        worker_output.get(WORKER_OUTPUT_KEY)
        if isinstance(worker_output, Mapping)
        else None
    )
    if payload is None:
        if metrics is not None:
            metrics.degraded(reason_code="worker_output_missing")
        coordinator.mark_controller_unavailable()
        return
    if not coordinator.accept_worker_output(payload):
        coordinator.mark_controller_unavailable()


# xdist owns these hook specifications.  Mark them optional without importing
# pytest during the module's cold-import path.
pytest_configure_node.pytest_impl = {"optionalhook": True}  # type: ignore[attr-defined]
pytest_testnodedown.pytest_impl = {"optionalhook": True}  # type: ignore[attr-defined]


def pytest_sessionfinish(session: Any, exitstatus: Any) -> None:
    """Return worker state or flush controller publications at session end."""

    config = getattr(session, "config", None)
    proof_config = get_proof_reuse_config(config)
    if proof_config.mode is ProofReuseMode.OFF:
        return
    from .receipt import clear_collectors
    from .xdist import (
        WORKER_OUTPUT_KEY,
        ProofReuseXdistCoordinator,
        ProofReuseXdistRole,
    )

    coordinator = getattr(config, COORDINATOR_ATTRIBUTE, None)
    if not isinstance(coordinator, ProofReuseXdistCoordinator):
        clear_collectors()
        return
    if coordinator.role is ProofReuseXdistRole.WORKER:
        worker_output = getattr(config, "workeroutput", None)
        if isinstance(worker_output, dict):
            worker_output[WORKER_OUTPUT_KEY] = coordinator.worker_output()
    elif (
        proof_config.writes_receipts
        and coordinator.healthy
        and coordinator.pending_publications
    ):
        if getattr(config, STORE_SERVICE_ATTRIBUTE, None) is None:
            # A store is needed only after at least one complete pass produced
            # a publication intent. This remains fail-open: an unavailable
            # cache/provider leaves the real test result untouched.
            _inject_default_services(config)
        coordinator.flush_publications(
            getattr(config, STORE_SERVICE_ATTRIBUTE, None),
            getattr(config, ISSUER_SERVICE_ATTRIBUTE, None),
            candidate_store=getattr(config, CANDIDATE_STORE_SERVICE_ATTRIBUTE, None),
        )
    clear_collectors()


def pytest_terminal_summary(
    terminalreporter: Any,
    exitstatus: Any,
    config: Any,
) -> None:
    """Emit one aggregate-only proof-reuse session line."""

    if get_proof_reuse_config(config).mode is ProofReuseMode.OFF:
        return
    metrics = getattr(config, METRICS_ATTRIBUTE, None)
    if metrics is None:
        return
    try:
        terminalreporter.write_line(metrics.summary_line())
    except Exception:
        return


__all__ = [
    "COMPOSITION_ATTRIBUTE",
    "CONFIG_ATTRIBUTE",
    "COORDINATOR_ATTRIBUTE",
    "CANDIDATE_STORE_SERVICE_ATTRIBUTE",
    "DEFAULT_SERVICES_ATTRIBUTE",
    "DEPENDENCY_INSTALLER_ATTRIBUTE",
    "DEFERRED_REQUEST_ATTRIBUTE",
    "DISABLED_MARKER",
    "EFFECTS_MARKER",
    "EXECUTION_RECORDED_ATTRIBUTE",
    "IDENTITY_FACTORY_ATTRIBUTE",
    "IDENTITY_SERVICES_ATTRIBUTE",
    "ISSUER_SERVICE_ATTRIBUTE",
    "ITEM_METADATA_ATTRIBUTE",
    "LOOKUP_SERVICE_ATTRIBUTE",
    "METRICS_ATTRIBUTE",
    "PLUGIN_NAME",
    "PROOF_REUSE_RUNTIME_COMPOSITION_INTERFACE",
    "PROVIDER_SERVICE_ATTRIBUTE",
    "ProofReuseItemMetadata",
    "ProofReuseRuntimeComposition",
    "CANDIDATE_PUBLICATION_ATTRIBUTE",
    "RUNTIME_LIFECYCLE_ATTRIBUTE",
    "RUNTIME_PLUGIN_ATTRIBUTE",
    "RUNTIME_TRACE_ATTRIBUTE",
    "RUNTIME_TRACE_CAPTURE_ATTRIBUTE",
    "SERVICE_RESOLUTION_ATTRIBUTE",
    "SERVICE_RESOLVER_ATTRIBUTE",
    "STORE_SERVICE_ATTRIBUTE",
    "collect_item_metadata",
    "get_item_metadata",
    "get_proof_reuse_config",
    "pytest_addoption",
    "pytest_collection_modifyitems",
    "pytest_configure",
    "pytest_configure_node",
    "pytest_sessionfinish",
    "pytest_terminal_summary",
    "pytest_testnodedown",
    "set_proof_reuse_dependency_installer",
    "set_proof_reuse_service_resolver",
    "set_proof_reuse_identity_services",
    "set_proof_reuse_services",
]
