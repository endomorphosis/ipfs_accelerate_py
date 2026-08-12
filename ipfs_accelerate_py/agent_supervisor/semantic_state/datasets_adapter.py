"""Pinned ``ipfs_datasets_py`` semantic-state adapter (SCH-002).

This module is the only harness import boundary to the sealed datasets ISI and
semantic-state surface.  It loads that surface lazily, checks the sealed
contract/schema/compiler versions, and exposes a read-only
``SemanticStateProvider`` that never grants put, CAS, WAL, provider, or network
authority to datasets.

Identity (symbol IDs, source CIDs, state/root CIDs, capsule CIDs) is always the
datasets producer's; this adapter never translates or re-derives them.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Protocol, runtime_checkable

from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import (
    UnavailableResult,
    validate_opaque_cid,
)

# ---------------------------------------------------------------------------
# Sealed contract pins (must match semantic_state_dependencies.seal.json)
# ---------------------------------------------------------------------------

ADAPTER_ID = "ipfs-datasets-semantic-state-adapter@1"
PROVIDER_CONTRACT = "SemanticStateProvider@1"

EXPECTED_SEMANTIC_STATE_SCHEMA = "ipfs-datasets.software-contracts.semantic-state@1"
EXPECTED_CAPSULE_SCHEMA = "ipfs-datasets.software-contracts.semantic-capsule@1"
EXPECTED_SELECTION_SCHEMA = "ipfs-datasets.software-contracts.semantic-test-selection@1"
EXPECTED_SEMANTIC_INDEX_SCHEMA = "ipfs-datasets.software-contracts.semantic-index@2"
EXPECTED_MERKLE_COMPILER_VERSION = "1"
EXPECTED_CAPSULE_COMPILER_VERSION = "1"
EXPECTED_SEMANTIC_STATE_API_SCHEMA = (
    "ipfs-datasets.software-contracts.semantic-state-api@1"
)
EXPECTED_STATE_VIEW_INTERFACE = "SemanticStateView@1"
EXPECTED_PRODUCER_INTERFACE = "SemanticStateProducer@1"
EXPECTED_BLOCK_READER_INTERFACE = "SemanticStateBlockReader@1"

CONFIDENCE_VALUES = frozenset({"exact", "conservative", "heuristic", "opaque"})

_REQUIRED_STATE_EXPORTS = (
    "SEMANTIC_STATE_API_SCHEMA",
    "SEMANTIC_STATE_VIEW_INTERFACE",
    "SEMANTIC_STATE_PRODUCER_INTERFACE",
    "SEMANTIC_STATE_BLOCK_READER_INTERFACE",
    "SemanticStateView",
    "build_semantic_state",
    "verify_semantic_state_bundle",
    "open_semantic_state",
    "view_semantic_state_bundle",
    "compile_semantic_capsule",
    "assess_capsule_freshness",
    "read_required_source",
    "extend_semantic_invalidation",
    "select_tests_and_proofs",
    "compare_test_selection_oracle",
)

_REQUIRED_INDEX_EXPORTS = (
    "scan_repository",
    "diff_repository_states",
    "calculate_invalidation",
    "explain_symbol",
    "explain_impact",
    "watch_repository",
)

_REQUIRED_MODEL_EXPORTS = (
    "SEMANTIC_STATE_SCHEMA",
    "SEMANTIC_CAPSULE_SCHEMA",
    "TEST_SELECTION_SCHEMA",
    "MERKLE_COMPILER_VERSION",
    "CAPSULE_COMPILER_VERSION",
)

_FORWARDED_OPS = frozenset(
    {
        "diff_repository_states",
        "calculate_invalidation",
        "build_semantic_state",
        "verify_semantic_state_bundle",
        "compile_semantic_capsule",
        "select_tests_and_proofs",
        "compare_test_selection_oracle",
    }
)

_CID_ATTR_NAMES = frozenset(
    {
        "state_cid",
        "root_cid",
        "capsule_cid",
        "node_cid",
        "version_cid",
        "source_cid",
        "selection_cid",
        "plan_cid",
        "delta_cid",
        "symbol_fact_cid",
        "assessment_cid",
        "comparison_cid",
        "index_cid",
    }
)


# ---------------------------------------------------------------------------
# Typed adapter failures
# ---------------------------------------------------------------------------


class SemanticStateUnavailable(RuntimeError):
    """Sealed datasets capability is missing, incompatible, or failed closed."""

    def __init__(
        self,
        operation: str,
        reason_code: str,
        diagnostic: str,
        *,
        retryable: bool = False,
        adapter_id: str = ADAPTER_ID,
    ) -> None:
        self.operation = operation
        self.reason_code = reason_code
        self.diagnostic = diagnostic[:512]
        self.retryable = bool(retryable)
        self.adapter_id = adapter_id
        super().__init__(
            f"{adapter_id}:{operation}:{reason_code}: {self.diagnostic}"
        )

    def to_unavailable_result(self) -> UnavailableResult:
        return UnavailableResult(
            operation=self.operation,
            adapter_id=self.adapter_id,
            reason_code=self.reason_code,
            retryable=self.retryable,
            diagnostic=self.diagnostic,
        )


class SourceBlobStale(RuntimeError):
    """Scanned source bytes no longer match the sealed producer binding.

    Callers must rescan rather than mix post-scan filesystem bytes into state.
    """

    requires_rescan: bool = True

    def __init__(self, diagnostic: str, *, kind: str = "source_blob_stale") -> None:
        self.kind = kind
        self.diagnostic = diagnostic[:512]
        super().__init__(self.diagnostic)


class SemanticStateAdapterError(ValueError):
    """Closed adapter validation failure (schema/CID/confidence/binding)."""


# ---------------------------------------------------------------------------
# Capability record
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SemanticStateCapability:
    """Closed capability witness for the pinned datasets semantic-state surface."""

    available: bool
    adapter_id: str
    contract_name: str
    semantic_state_schema: str
    capsule_schema: str
    selection_schema: str
    semantic_index_schema: str
    merkle_compiler_version: str
    capsule_compiler_version: str
    semantic_state_api_schema: str
    view_interface: str
    producer_interface: str
    block_reader_interface: str
    operations: tuple[str, ...]
    reason_code: str | None = None
    diagnostic: str | None = None

    def require_available(self, operation: str) -> None:
        if not self.available:
            raise SemanticStateUnavailable(
                operation,
                self.reason_code or "capability_unavailable",
                self.diagnostic or "datasets semantic-state surface unavailable",
                retryable=False,
            )


# ---------------------------------------------------------------------------
# Provider protocol (sealed pure-delegation surface names only)
# ---------------------------------------------------------------------------


@runtime_checkable
class SemanticStateProvider(Protocol):
    """Harness-facing provider marker for the pinned datasets adapter.

    Concrete operations live on :class:`IpfsDatasetsSemanticStateProvider`.
    The dependency-seal AST audit only admits ``open_semantic_state`` and
    ``scan_repository`` as pure-delegation methods on that concrete class;
    this Protocol therefore carries no producer-named method stubs.
    """

    @property
    def capability(self) -> SemanticStateCapability: ...


# ---------------------------------------------------------------------------
# Loading and version gates
# ---------------------------------------------------------------------------


def _attr(obj: Any, name: str) -> Any:
    return getattr(obj, name, None)


def _text_version(value: Any, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise SemanticStateAdapterError(f"{name} must be a nonempty trimmed string")
    return value


def _require_exports(module: Any, names: Sequence[str], label: str) -> None:
    missing = [name for name in names if not hasattr(module, name)]
    if missing:
        raise SemanticStateUnavailable(
            "load",
            "missing_exports",
            f"{label} missing required exports: {', '.join(missing)}",
            retryable=False,
        )


def _confidence_of(value: Any) -> str | None:
    if value is None:
        return None
    raw = getattr(value, "confidence", value)
    if hasattr(raw, "value"):
        raw = raw.value
    if type(raw) is not str:
        return None
    return raw


def _validate_confidence(value: Any, *, context: str) -> None:
    conf = _confidence_of(value)
    if conf is None:
        return
    if conf not in CONFIDENCE_VALUES:
        raise SemanticStateAdapterError(
            f"{context} confidence {conf!r} is outside the closed vocabulary"
        )


def _maybe_validate_cid_attr(obj: Any, name: str) -> None:
    if obj is None or not hasattr(obj, name):
        return
    value = getattr(obj, name)
    if value is None:
        return
    if type(value) is not str:
        raise SemanticStateAdapterError(f"{name} must be a CID string when present")
    validate_opaque_cid(value, name)


def _validate_identity_object(obj: Any, *, context: str) -> Any:
    """Fail closed on forged/unknown CIDs and illegal confidence; pass identity."""

    if obj is None:
        return None
    for name in _CID_ATTR_NAMES:
        try:
            _maybe_validate_cid_attr(obj, name)
        except Exception as exc:
            raise SemanticStateAdapterError(
                f"{context}: invalid {name}: {exc}"
            ) from exc
    _validate_confidence(obj, context=context)
    # Opaque/heuristic/exact/conservative all remain visible; no filtering.
    return obj


def _check_schema_pin(actual: Any, expected: str, name: str) -> None:
    text = _text_version(actual, name)
    if text != expected:
        raise SemanticStateUnavailable(
            "load",
            "schema_mismatch",
            f"{name} is {text!r}, expected sealed {expected!r}",
            retryable=False,
        )


def _load_pinned_modules() -> tuple[Any, Any, Any]:
    """Nested static imports — lazy, and not dynamic ``import_module`` calls."""

    try:
        from ipfs_datasets_py.logic.software_contracts import (
            semantic_index as index_mod,
        )
        from ipfs_datasets_py.logic.software_contracts.semantic_state import (
            models as models_mod,
        )
        from ipfs_datasets_py.logic.software_contracts import (
            semantic_state as state_mod,
        )
    except Exception as exc:  # ImportError and ambient layout failures
        raise SemanticStateUnavailable(
            "load",
            "import_failed",
            f"pinned datasets semantic-state surface import failed: {exc}",
            retryable=True,
        ) from exc
    return state_mod, index_mod, models_mod


def _build_surface(state_mod: Any, index_mod: Any, models_mod: Any) -> SimpleNamespace:
    _require_exports(state_mod, _REQUIRED_STATE_EXPORTS, "semantic_state")
    _require_exports(index_mod, _REQUIRED_INDEX_EXPORTS, "semantic_index")
    _require_exports(models_mod, _REQUIRED_MODEL_EXPORTS, "semantic_state.models")

    _check_schema_pin(
        _attr(state_mod, "SEMANTIC_STATE_API_SCHEMA"),
        EXPECTED_SEMANTIC_STATE_API_SCHEMA,
        "SEMANTIC_STATE_API_SCHEMA",
    )
    _check_schema_pin(
        _attr(state_mod, "SEMANTIC_STATE_VIEW_INTERFACE"),
        EXPECTED_STATE_VIEW_INTERFACE,
        "SEMANTIC_STATE_VIEW_INTERFACE",
    )
    _check_schema_pin(
        _attr(state_mod, "SEMANTIC_STATE_PRODUCER_INTERFACE"),
        EXPECTED_PRODUCER_INTERFACE,
        "SEMANTIC_STATE_PRODUCER_INTERFACE",
    )
    _check_schema_pin(
        _attr(state_mod, "SEMANTIC_STATE_BLOCK_READER_INTERFACE"),
        EXPECTED_BLOCK_READER_INTERFACE,
        "SEMANTIC_STATE_BLOCK_READER_INTERFACE",
    )
    _check_schema_pin(
        _attr(models_mod, "SEMANTIC_STATE_SCHEMA"),
        EXPECTED_SEMANTIC_STATE_SCHEMA,
        "SEMANTIC_STATE_SCHEMA",
    )
    _check_schema_pin(
        _attr(models_mod, "SEMANTIC_CAPSULE_SCHEMA"),
        EXPECTED_CAPSULE_SCHEMA,
        "SEMANTIC_CAPSULE_SCHEMA",
    )
    _check_schema_pin(
        _attr(models_mod, "TEST_SELECTION_SCHEMA"),
        EXPECTED_SELECTION_SCHEMA,
        "TEST_SELECTION_SCHEMA",
    )
    _check_schema_pin(
        str(_attr(models_mod, "MERKLE_COMPILER_VERSION")),
        EXPECTED_MERKLE_COMPILER_VERSION,
        "MERKLE_COMPILER_VERSION",
    )
    _check_schema_pin(
        str(_attr(models_mod, "CAPSULE_COMPILER_VERSION")),
        EXPECTED_CAPSULE_COMPILER_VERSION,
        "CAPSULE_COMPILER_VERSION",
    )

    index_schema = _attr(index_mod, "SEMANTIC_INDEX_SCHEMA")
    if index_schema is None:
        # Prefer models package constant when the index package re-exports models.
        try:
            from ipfs_datasets_py.logic.software_contracts.semantic_index import (
                models as index_models,
            )

            index_schema = _attr(index_models, "SEMANTIC_INDEX_SCHEMA")
        except Exception as exc:
            raise SemanticStateUnavailable(
                "load",
                "missing_index_schema",
                f"SEMANTIC_INDEX_SCHEMA unavailable: {exc}",
                retryable=False,
            ) from exc
    _check_schema_pin(
        index_schema,
        EXPECTED_SEMANTIC_INDEX_SCHEMA,
        "SEMANTIC_INDEX_SCHEMA",
    )

    operations = tuple(
        sorted(
            set(_REQUIRED_STATE_EXPORTS)
            | set(_REQUIRED_INDEX_EXPORTS)
            | {
                "SEMANTIC_STATE_SCHEMA",
                "SEMANTIC_CAPSULE_SCHEMA",
                "TEST_SELECTION_SCHEMA",
            }
        )
    )
    capability = SemanticStateCapability(
        available=True,
        adapter_id=ADAPTER_ID,
        contract_name=PROVIDER_CONTRACT,
        semantic_state_schema=EXPECTED_SEMANTIC_STATE_SCHEMA,
        capsule_schema=EXPECTED_CAPSULE_SCHEMA,
        selection_schema=EXPECTED_SELECTION_SCHEMA,
        semantic_index_schema=EXPECTED_SEMANTIC_INDEX_SCHEMA,
        merkle_compiler_version=EXPECTED_MERKLE_COMPILER_VERSION,
        capsule_compiler_version=EXPECTED_CAPSULE_COMPILER_VERSION,
        semantic_state_api_schema=EXPECTED_SEMANTIC_STATE_API_SCHEMA,
        view_interface=EXPECTED_STATE_VIEW_INTERFACE,
        producer_interface=EXPECTED_PRODUCER_INTERFACE,
        block_reader_interface=EXPECTED_BLOCK_READER_INTERFACE,
        operations=operations,
    )
    surface = SimpleNamespace(
        # Index / scan
        scan_repository=index_mod.scan_repository,
        diff_repository_states=index_mod.diff_repository_states,
        calculate_invalidation=index_mod.calculate_invalidation,
        explain_symbol=index_mod.explain_symbol,
        explain_impact=index_mod.explain_impact,
        watch_repository=index_mod.watch_repository,
        # State / capsules / selection
        build_semantic_state=state_mod.build_semantic_state,
        verify_semantic_state_bundle=state_mod.verify_semantic_state_bundle,
        open_semantic_state=state_mod.open_semantic_state,
        view_semantic_state_bundle=state_mod.view_semantic_state_bundle,
        compile_semantic_capsule=state_mod.compile_semantic_capsule,
        assess_capsule_freshness=state_mod.assess_capsule_freshness,
        read_required_source=state_mod.read_required_source,
        extend_semantic_invalidation=state_mod.extend_semantic_invalidation,
        select_tests_and_proofs=state_mod.select_tests_and_proofs,
        compare_test_selection_oracle=state_mod.compare_test_selection_oracle,
        # Types / constants (passthrough for callers that need them)
        SemanticStateView=state_mod.SemanticStateView,
        SEMANTIC_STATE_SCHEMA=models_mod.SEMANTIC_STATE_SCHEMA,
        SEMANTIC_CAPSULE_SCHEMA=models_mod.SEMANTIC_CAPSULE_SCHEMA,
        TEST_SELECTION_SCHEMA=models_mod.TEST_SELECTION_SCHEMA,
        SEMANTIC_INDEX_SCHEMA=index_schema,
        capability=capability,
        # Source failure types when present (for mapping)
        SourceAdmissionError=_attr(state_mod, "SourceAdmissionError"),
    )
    return surface


def _load_pinned_surface() -> SimpleNamespace:
    state_mod, index_mod, models_mod = _load_pinned_modules()
    return _build_surface(state_mod, index_mod, models_mod)


class _LazySurface:
    """Attribute proxy that loads the sealed surface on first use."""

    __slots__ = ("_loader", "_surface", "_error")

    def __init__(self, loader: Callable[[], SimpleNamespace]) -> None:
        object.__setattr__(self, "_loader", loader)
        object.__setattr__(self, "_surface", None)
        object.__setattr__(self, "_error", None)

    def _resolve(self) -> SimpleNamespace:
        existing = object.__getattribute__(self, "_surface")
        if existing is not None:
            return existing
        err = object.__getattribute__(self, "_error")
        if err is not None:
            raise err
        try:
            surface = object.__getattribute__(self, "_loader")()
        except SemanticStateUnavailable as exc:
            object.__setattr__(self, "_error", exc)
            raise
        except Exception as exc:  # pragma: no cover - defensive
            wrapped = SemanticStateUnavailable(
                "load",
                "load_failed",
                f"datasets surface failed to load: {exc}",
                retryable=True,
            )
            object.__setattr__(self, "_error", wrapped)
            raise wrapped from exc
        object.__setattr__(self, "_surface", surface)
        return surface

    def __getattr__(self, name: str) -> Any:
        return getattr(self._resolve(), name)


# ---------------------------------------------------------------------------
# Provider implementation
# ---------------------------------------------------------------------------


class IpfsDatasetsSemanticStateProvider:
    """Harness-facing provider over the sealed datasets semantic-state surface.

    ``open_semantic_state`` and ``scan_repository`` pure-delegate to the loaded
    surface (AST-seal requirement).  Remaining producer operations are exposed
    as validated forwarders that never rewrite identity.
    """

    __slots__ = ("_api", "_capability", "_forbid_filesystem_source")

    def __init__(
        self,
        surface: Any | None = None,
        *,
        capability: SemanticStateCapability | None = None,
        forbid_filesystem_source: bool = True,
    ) -> None:
        self._forbid_filesystem_source = bool(forbid_filesystem_source)
        if surface is not None:
            self._api = surface
            self._capability = capability or getattr(surface, "capability", None)
            if self._capability is None:
                self._capability = SemanticStateCapability(
                    available=True,
                    adapter_id=ADAPTER_ID,
                    contract_name=PROVIDER_CONTRACT,
                    semantic_state_schema=EXPECTED_SEMANTIC_STATE_SCHEMA,
                    capsule_schema=EXPECTED_CAPSULE_SCHEMA,
                    selection_schema=EXPECTED_SELECTION_SCHEMA,
                    semantic_index_schema=EXPECTED_SEMANTIC_INDEX_SCHEMA,
                    merkle_compiler_version=EXPECTED_MERKLE_COMPILER_VERSION,
                    capsule_compiler_version=EXPECTED_CAPSULE_COMPILER_VERSION,
                    semantic_state_api_schema=EXPECTED_SEMANTIC_STATE_API_SCHEMA,
                    view_interface=EXPECTED_STATE_VIEW_INTERFACE,
                    producer_interface=EXPECTED_PRODUCER_INTERFACE,
                    block_reader_interface=EXPECTED_BLOCK_READER_INTERFACE,
                    operations=tuple(sorted(_FORWARDED_OPS | {"open_semantic_state", "scan_repository"})),
                )
        else:
            self._capability = None
            self._api = _LazySurface(self._load_into_self)

    def _load_into_self(self) -> SimpleNamespace:
        surface = _load_pinned_surface()
        self._capability = surface.capability
        return surface

    @property
    def capability(self) -> SemanticStateCapability:
        if self._capability is None:
            # Force lazy load
            getattr(self._api, "capability")
        assert self._capability is not None
        return self._capability

    def _require(self, operation: str) -> Any:
        cap = self.capability
        cap.require_available(operation)
        return self._api

    # --- sealed pure-delegation methods (names fixed by AST audit) ----------

    def open_semantic_state(self, root_cid, get_block):
        return self._api.open_semantic_state(root_cid, get_block)

    def scan_repository(self, repo_path, previous_state=None):
        return self._api.scan_repository(repo_path, previous_state=previous_state)

    # --- validated forwarders (non-forbidden names) -------------------------

    def view_semantic_state_bundle(self, bundle: Any) -> Any:
        api = self._require("view_semantic_state_bundle")
        view = api.view_semantic_state_bundle(bundle)
        return _validate_view(view, context="view_semantic_state_bundle")

    def explain_symbol(self, repository_state: Any, symbol_id: str) -> Any:
        api = self._require("explain_symbol")
        result = api.explain_symbol(repository_state, symbol_id)
        return _validate_identity_object(result, context="explain_symbol")

    def explain_impact(
        self, repository_state: Any, changed_symbol_ids: Iterable[str]
    ) -> Any:
        api = self._require("explain_impact")
        result = api.explain_impact(repository_state, changed_symbol_ids)
        return _validate_identity_object(result, context="explain_impact")

    def assess_capsule_freshness(
        self, capsule: Any, *, current_state: Any, invalidation: Any = None
    ) -> Any:
        api = self._require("assess_capsule_freshness")
        result = api.assess_capsule_freshness(
            capsule, current_state=current_state, invalidation=invalidation
        )
        return _validate_identity_object(result, context="assess_capsule_freshness")

    def read_required_source(
        self,
        semantic_index: Any,
        symbol_id: str,
        *,
        expected_producer_state_cid: str,
        read_source_blob: Callable[[str], bytes] | None = None,
    ) -> Any:
        """Retrieve exact tree-bound source; map TOCTOU/mismatch to SourceBlobStale."""

        api = self._require("read_required_source")
        validate_opaque_cid(expected_producer_state_cid, "expected_producer_state_cid")
        if self._forbid_filesystem_source and read_source_blob is None:
            # Ambient Path fallback is forbidden by the producer and by this
            # adapter.  Require an explicit sealed blob reader or an index that
            # already exposes one.
            if not hasattr(semantic_index, "read_source_blob"):
                raise SourceBlobStale(
                    "exact source requires a sealed snapshot/blob reader; "
                    "ambient filesystem reads after scan are forbidden",
                    kind="source_unsafe_view",
                )
        try:
            if read_source_blob is None:
                materialization = api.read_required_source(
                    semantic_index,
                    symbol_id,
                    expected_producer_state_cid=expected_producer_state_cid,
                )
            else:
                materialization = api.read_required_source(
                    semantic_index,
                    symbol_id,
                    expected_producer_state_cid=expected_producer_state_cid,
                    read_source_blob=read_source_blob,
                )
        except SourceBlobStale:
            raise
        except Exception as exc:
            raise _map_source_failure(exc) from exc
        return materialization

    def extend_semantic_invalidation(
        self,
        previous_index: Any,
        current_index: Any,
        delta: Any,
        plan: Any,
        previous_state: Any,
        current_state: Any,
    ) -> Any:
        api = self._require("extend_semantic_invalidation")
        result = api.extend_semantic_invalidation(
            previous_index,
            current_index,
            delta,
            plan,
            previous_state,
            current_state,
        )
        return _validate_identity_object(result, context="extend_semantic_invalidation")

    def watch_repository(
        self,
        repo_path: Any,
        callback: Callable[[Any], Any],
        *,
        debounce_ms: int = 250,
    ) -> Any:
        """Start a datasets watcher.

        Watch notifications only wake a fresh canonical scan.  The adapter never
        promotes event paths or payloads into repository state; state always
        comes from ``scan_repository`` results carried on the notification.
        """

        api = self._require("watch_repository")
        if not callable(callback):
            raise SemanticStateAdapterError("callback must be callable")

        def _scan_only_callback(notification: Any) -> Any:
            # Deliberately do not inspect filesystem event metadata.  Only the
            # producer-scanned state on the notification is visible to callers.
            state = getattr(notification, "state", None)
            if state is None:
                raise SemanticStateAdapterError(
                    "watch notification missing scanned state; events cannot become state"
                )
            _maybe_validate_cid_attr(state, "state_cid")
            return callback(notification)

        return api.watch_repository(
            repo_path, _scan_only_callback, debounce_ms=debounce_ms
        )

    def open_verified_view(
        self, root_cid: str, get_block: Callable[[str], bytes]
    ) -> Any:
        """Open and adapter-validate a storage-neutral ``SemanticStateView``.

        Equivalent results for in-memory bundle readers and sealed durable
        ``get_block`` callables.  Never injects put/CAS/WAL/network handles.
        """

        if not callable(get_block):
            raise SemanticStateAdapterError("get_block must be callable")
        validate_opaque_cid(root_cid, "root_cid")
        self._require("open_semantic_state")
        try:
            view = self.open_semantic_state(root_cid, get_block)
        except SemanticStateUnavailable:
            raise
        except Exception as exc:
            raise SemanticStateAdapterError(
                f"open_semantic_state failed closed: {exc}"
            ) from exc
        return _validate_view(view, context="open_semantic_state", root_cid=root_cid)

    # --- generic forwarders for producer names sealed out of local defs -----

    def __getattr__(self, name: str) -> Any:
        if name in _FORWARDED_OPS:
            api = self._require(name)
            target = getattr(api, name)

            def _forward(*args: Any, **kwargs: Any) -> Any:
                if name == "select_tests_and_proofs":
                    return _forward_selection(target, args, kwargs)
                if name == "build_semantic_state":
                    result = target(*args, **kwargs)
                    return _validate_bundle(result, context="build_semantic_state")
                if name == "verify_semantic_state_bundle":
                    result = target(*args, **kwargs)
                    return _validate_identity_object(
                        result, context="verify_semantic_state_bundle"
                    )
                if name == "compile_semantic_capsule":
                    result = target(*args, **kwargs)
                    return _validate_capsule_result(
                        result, context="compile_semantic_capsule"
                    )
                if name == "compare_test_selection_oracle":
                    result = target(*args, **kwargs)
                    return _validate_identity_object(
                        result, context="compare_test_selection_oracle"
                    )
                result = target(*args, **kwargs)
                return _validate_identity_object(result, context=name)

            return _forward
        raise AttributeError(
            f"{type(self).__name__!s} has no attribute {name!r}"
        )


def _validate_view(
    view: Any, *, context: str, root_cid: str | None = None
) -> Any:
    if view is None:
        raise SemanticStateAdapterError(f"{context} returned no view")
    root = getattr(view, "root", None)
    if root is None:
        raise SemanticStateAdapterError(f"{context} view missing root")
    _validate_identity_object(root, context=f"{context}.root")
    actual_root = getattr(root, "root_cid", None)
    if root_cid is not None and actual_root is not None and actual_root != root_cid:
        raise SemanticStateAdapterError(
            f"{context}: root_cid binding mismatch "
            f"(requested {root_cid!r}, view {actual_root!r})"
        )
    if not callable(getattr(view, "get_block", None)):
        raise SemanticStateAdapterError(f"{context} view missing get_block")
    if not callable(getattr(view, "symbol_node", None)):
        raise SemanticStateAdapterError(f"{context} view missing symbol_node")
    if not callable(getattr(view, "capsule", None)):
        raise SemanticStateAdapterError(f"{context} view missing capsule")
    return view


def _validate_bundle(bundle: Any, *, context: str) -> Any:
    if bundle is None:
        raise SemanticStateAdapterError(f"{context} returned no bundle")
    root = getattr(bundle, "root", None)
    _validate_identity_object(root, context=f"{context}.root")
    if not callable(getattr(bundle, "get_block", None)):
        raise SemanticStateAdapterError(f"{context} bundle missing get_block")
    # Bundles are finite CID→bytes maps only — no put/CAS surface required.
    for forbidden in ("put", "put_block", "cas", "compare_and_swap", "wal", "network"):
        if callable(getattr(bundle, forbidden, None)):
            raise SemanticStateAdapterError(
                f"{context} bundle must not expose {forbidden} authority"
            )
    return bundle


def _validate_capsule_result(result: Any, *, context: str) -> Any:
    capsule = getattr(result, "capsule", result)
    _validate_identity_object(capsule, context=context)
    conf = _confidence_of(capsule)
    if conf in {"opaque", "heuristic"}:
        # Opaque/invalid-adjacent capsules remain visible and force raw source.
        # No filtering: callers must still see the capsule record.
        pass
    schema = getattr(capsule, "capsule_schema", None)
    if schema is not None and schema != EXPECTED_CAPSULE_SCHEMA:
        raise SemanticStateAdapterError(
            f"{context}: capsule_schema {schema!r} does not match sealed pin"
        )
    return result


def _forward_selection(target: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """Always preserve previous/current views for delete/rename evidence."""

    # Preferred signature:
    # select_tests_and_proofs(previous_state, current_state, invalidation, *, policy, explicit_rules=())
    if len(args) < 2 and "current_state" not in kwargs:
        raise SemanticStateAdapterError(
            "select_tests_and_proofs requires previous_state and current_state views"
        )
    if "previous_state" not in kwargs and len(args) == 0:
        raise SemanticStateAdapterError(
            "select_tests_and_proofs requires previous_state (may be None)"
        )
    result = target(*args, **kwargs)
    return _validate_selection(result)


def _validate_selection(selection: Any) -> Any:
    if selection is None:
        raise SemanticStateAdapterError("select_tests_and_proofs returned no selection")
    _validate_identity_object(selection, context="select_tests_and_proofs")
    schema = getattr(selection, "schema", None)
    if schema is None and hasattr(selection, "selection_schema"):
        schema = getattr(selection, "selection_schema")
    # Some producer models encode schema only in identity_payload.
    payload = getattr(selection, "identity_payload", None)
    if callable(payload):
        try:
            body = payload()
            if isinstance(body, Mapping):
                schema = body.get("schema", schema)
        except Exception:
            pass
    if schema is not None and schema != EXPECTED_SELECTION_SCHEMA:
        raise SemanticStateAdapterError(
            f"selection schema {schema!r} does not match sealed pin"
        )
    return selection


def _map_source_failure(exc: BaseException) -> BaseException:
    name = type(exc).__name__
    message = str(exc)
    kind = getattr(exc, "kind", None)
    requires = bool(getattr(exc, "requires_rescan", True))
    stale_names = {
        "SourceBindingMismatchError",
        "SourceCorruptError",
        "SourceWrongStateError",
        "SourceUnavailableError",
        "SourceAdmissionError",
    }
    if name in stale_names or requires or (
        isinstance(kind, str)
        and kind.startswith("source_")
    ):
        return SourceBlobStale(message, kind=str(kind or name))
    return SemanticStateAdapterError(f"read_required_source failed: {message}")


def inspect_semantic_state_capability(
    *, surface: Any | None = None
) -> SemanticStateCapability:
    """Return capability without raising when the surface is unavailable."""

    if surface is not None:
        cap = getattr(surface, "capability", None)
        if isinstance(cap, SemanticStateCapability):
            return cap
        return SemanticStateCapability(
            available=True,
            adapter_id=ADAPTER_ID,
            contract_name=PROVIDER_CONTRACT,
            semantic_state_schema=EXPECTED_SEMANTIC_STATE_SCHEMA,
            capsule_schema=EXPECTED_CAPSULE_SCHEMA,
            selection_schema=EXPECTED_SELECTION_SCHEMA,
            semantic_index_schema=EXPECTED_SEMANTIC_INDEX_SCHEMA,
            merkle_compiler_version=EXPECTED_MERKLE_COMPILER_VERSION,
            capsule_compiler_version=EXPECTED_CAPSULE_COMPILER_VERSION,
            semantic_state_api_schema=EXPECTED_SEMANTIC_STATE_API_SCHEMA,
            view_interface=EXPECTED_STATE_VIEW_INTERFACE,
            producer_interface=EXPECTED_PRODUCER_INTERFACE,
            block_reader_interface=EXPECTED_BLOCK_READER_INTERFACE,
            operations=(),
        )
    try:
        loaded = _load_pinned_surface()
        return loaded.capability
    except SemanticStateUnavailable as exc:
        return SemanticStateCapability(
            available=False,
            adapter_id=ADAPTER_ID,
            contract_name=PROVIDER_CONTRACT,
            semantic_state_schema=EXPECTED_SEMANTIC_STATE_SCHEMA,
            capsule_schema=EXPECTED_CAPSULE_SCHEMA,
            selection_schema=EXPECTED_SELECTION_SCHEMA,
            semantic_index_schema=EXPECTED_SEMANTIC_INDEX_SCHEMA,
            merkle_compiler_version=EXPECTED_MERKLE_COMPILER_VERSION,
            capsule_compiler_version=EXPECTED_CAPSULE_COMPILER_VERSION,
            semantic_state_api_schema=EXPECTED_SEMANTIC_STATE_API_SCHEMA,
            view_interface=EXPECTED_STATE_VIEW_INTERFACE,
            producer_interface=EXPECTED_PRODUCER_INTERFACE,
            block_reader_interface=EXPECTED_BLOCK_READER_INTERFACE,
            operations=(),
            reason_code=exc.reason_code,
            diagnostic=exc.diagnostic,
        )


def load_semantic_state_provider(
    surface: Any | None = None,
    *,
    forbid_filesystem_source: bool = True,
) -> IpfsDatasetsSemanticStateProvider:
    """Return a provider bound to the pinned datasets surface (or an inject)."""

    if surface is not None:
        return IpfsDatasetsSemanticStateProvider(
            surface,
            capability=getattr(surface, "capability", None),
            forbid_filesystem_source=forbid_filesystem_source,
        )
    provider = IpfsDatasetsSemanticStateProvider(
        forbid_filesystem_source=forbid_filesystem_source
    )
    # Eager capability check so import-time ambient failures surface at load.
    try:
        _ = provider.capability
        if not provider.capability.available:
            raise SemanticStateUnavailable(
                "load",
                provider.capability.reason_code or "capability_unavailable",
                provider.capability.diagnostic or "unavailable",
            )
    except SemanticStateUnavailable:
        raise
    return provider


__all__ = [
    "ADAPTER_ID",
    "PROVIDER_CONTRACT",
    "CONFIDENCE_VALUES",
    "EXPECTED_SEMANTIC_STATE_SCHEMA",
    "EXPECTED_CAPSULE_SCHEMA",
    "EXPECTED_SELECTION_SCHEMA",
    "EXPECTED_SEMANTIC_INDEX_SCHEMA",
    "SemanticStateAdapterError",
    "SemanticStateCapability",
    "SemanticStateProvider",
    "SemanticStateUnavailable",
    "SourceBlobStale",
    "IpfsDatasetsSemanticStateProvider",
    "inspect_semantic_state_capability",
    "load_semantic_state_provider",
]
