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
ISSUER_SERVICE_ATTRIBUTE = "_ipfs_proof_reuse_issuer_service"
RUNTIME_PLUGIN_ATTRIBUTE = "_ipfs_proof_reuse_runtime_plugin"
RUNTIME_TRACE_ATTRIBUTE = "_ipfs_proof_reuse_runtime_trace"
DEFERRED_REQUEST_ATTRIBUTE = "_ipfs_proof_reuse_deferred_request"
EXECUTION_RECORDED_ATTRIBUTE = "_ipfs_proof_reuse_execution_recorded"

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
    issuer: Any = None,
) -> None:
    """Inject optional runtime services without probing providers at import."""

    setattr(config, LOOKUP_SERVICE_ATTRIBUTE, lookup)
    setattr(config, STORE_SERVICE_ATTRIBUTE, store)
    setattr(config, ISSUER_SERVICE_ATTRIBUTE, issuer)


def _install_runtime_plugin(config: Any) -> None:
    if getattr(config, RUNTIME_PLUGIN_ATTRIBUTE, None) is not None:
        return
    try:
        import pytest
    except Exception:
        return

    class _ProofReuseRuntimePlugin:
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

    metrics = getattr(config, METRICS_ATTRIBUTE, None)
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

    lookup = getattr(config, LOOKUP_SERVICE_ATTRIBUTE, None)
    if proof_config.reads_candidates and isinstance(lookup, ProofReuseLookup):
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
    )
    from .xdist import ProofReuseXdistCoordinator

    metrics = getattr(config, METRICS_ATTRIBUTE, None)
    coordinator = getattr(config, COORDINATOR_ATTRIBUTE, None)
    collector = getattr(item, ITEM_COLLECTOR_ATTRIBUTE, None)
    if isinstance(collector, TestPassReceiptCollector):
        collector.record_report(report)

    when = str(getattr(report, "when", ""))
    outcome = str(getattr(report, "outcome", ""))
    decision = getattr(item, ITEM_DECISION_ATTRIBUTE, None)
    proof_skipped = isinstance(decision, ReuseDecision) and decision.is_skip
    already_recorded = bool(
        getattr(item, EXECUTION_RECORDED_ATTRIBUTE, False)
    )
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

    if when != "teardown" or not isinstance(
        collector, TestPassReceiptCollector
    ):
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
    result = finalize_test_pass_receipt(
        collector,
        locator=locator,
        execution_key=execution_key,
        runtime_trace=getattr(item, RUNTIME_TRACE_ATTRIBUTE, None),
        writes_receipts=False,
        item=item,
    )
    if result.admitted and result.receipt is not None:
        coordinator.queue_publication(
            result.receipt,
            deferred_request=getattr(
                item,
                DEFERRED_REQUEST_ATTRIBUTE,
                None,
            ),
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
        coordinator = ProofReuseXdistCoordinator.controller(
            metrics=coordinator.metrics
        )
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
        worker_input[WORKER_INPUT_KEY] = coordinator.configure_worker(
            worker_id
        )
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
        coordinator.flush_publications(
            getattr(config, STORE_SERVICE_ATTRIBUTE, None),
            getattr(config, ISSUER_SERVICE_ATTRIBUTE, None),
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
    "CONFIG_ATTRIBUTE",
    "COORDINATOR_ATTRIBUTE",
    "DEFERRED_REQUEST_ATTRIBUTE",
    "DISABLED_MARKER",
    "EFFECTS_MARKER",
    "EXECUTION_RECORDED_ATTRIBUTE",
    "ISSUER_SERVICE_ATTRIBUTE",
    "ITEM_METADATA_ATTRIBUTE",
    "LOOKUP_SERVICE_ATTRIBUTE",
    "METRICS_ATTRIBUTE",
    "PLUGIN_NAME",
    "ProofReuseItemMetadata",
    "RUNTIME_PLUGIN_ATTRIBUTE",
    "RUNTIME_TRACE_ATTRIBUTE",
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
    "set_proof_reuse_services",
]
