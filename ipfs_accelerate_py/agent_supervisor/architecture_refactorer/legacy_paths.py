"""Legacy, compatibility, fixture, and simulation path inventory (PCAR-015).

`LegacyPathInventory` binds mock workers, mock inference handlers, simulated
hardware, fake or compatibility CIDs, fixture providers, fallback success
paths, deprecated coordinators, legacy endpoint registries, and historical
provider routers to source identity and a closed reachability disposition.
Static and hermetic dynamic tracing never import inspected modules. Dynamic
loading uncertainty remains `unknown` and blocking, never `dead`. Origin taint
is preserved: compatibility, fixture, and simulation origins cannot satisfy
production capability, execution-success, proof, completion, or release
predicates. The inventory observes; it cannot delete, promote fake-to-live, or
grant production authority.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from ipfs_accelerate_py.utils.cid_utils import (
    canonical_dag_json_bytes,
    cid_for_dag_json,
    validate_cid,
)

from .architecture_ir import ArchitectureIR, ArchitectureNode
from .contracts import (
    ArchitectureContractError,
    Confidence,
    EdgeKind,
    NodeKind,
    SourceFactIdentity,
    SourceSpan,
    _closed_enum,
    _require_int,
    _require_mapping,
    _require_text,
    NON_PROBATIVE_CONFIDENCE,
)

LEGACY_INVENTORY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/legacy-simulation-inventory@1"
)
LEGACY_INVENTORY_VERSION = 1
LEGACY_INVENTORY_EVIDENCE = "pcar/legacy-simulation-inventory@1"
LEGACY_PATH_SCHEMA = "ipfs_accelerate_py/agent-supervisor/legacy-path@1"
LEGACY_PATH_VERSION = 1
DYNAMIC_RECORD_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/dynamic-reachability-record@1"
)
DYNAMIC_RECORD_VERSION = 1
TAINT_RECORD_SCHEMA = "ipfs_accelerate_py/agent-supervisor/origin-taint@1"
TAINT_RECORD_VERSION = 1
TRACE_SCHEMA = "ipfs_accelerate_py/agent-supervisor/reachability-trace@1"
TRACE_VERSION = 1
SCAN_SCHEMA = "ipfs_accelerate_py/agent-supervisor/side-effect-scan@1"
SCAN_SCHEMA_VERSION = 1
EXTRACTOR_IDENTITY = "pcar-015-legacy-path-inventory"
TASK_ID = "PCAR-015"
DEFAULT_FRESHNESS = "pcar-015-legacy-simulation"
EFFECT_CLASS = "read_only_analysis"
COMPACT_INVENTORY_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.architecture-refactorer"
    ".legacy-simulation-inventory@1"
)
SEALED_REPOSITORY_TREE = "a2d1529934197dc64fe18cfbaec9dc7daf438703^{tree}"
PRODUCTION_FLOW_INVARIANT = (
    "No compatibility, fixture, or simulation origin may satisfy a production "
    "capability, execution-success, proof, completion, or release predicate."
)
DEAD_CLASSIFICATION_POLICY = (
    "static_reachability_alone_never_proves_dead_where_dynamic_loading_is_possible"
)
INVENTORY_CAN_AUTHORIZE_DELETION = False
INVENTORY_CAN_PROMOTE_FAKE_TO_LIVE = False
INVENTORY_CAN_GRANT_PRODUCTION_AUTHORITY = False
INVENTORY_AUTHORITY = False
STATIC_REACHABILITY_PROVES_DEAD = False
DYNAMIC_UNCERTAINTY_BLOCKS_DEAD = True
CONTENT_IDENTITY_IS_NOT_AUTHORITY = True

_UNKNOWN_FIELD_MESSAGE = "unknown legacy-path field"
_MISSING_FIELD_MESSAGE = "missing legacy-path field"
_CID_PREFIXES = ("bagu", "bafy", "bafk", "sha256:")
_TEST_PATH_MARKERS = ("/test/", "test/", "tests/", "/tests/")
_COMPAT_PATH_MARKERS = (
    "compat",
    "legacy",
    "_legacy",
    "historical",
    "deprecated",
)
_SIMULATION_MARKERS = (
    "mock",
    "simulat",
    "fake",
    "fixture",
    "shadow",
    "stub",
)
_DYNAMIC_CALLEES = frozenset(
    {
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "eval",
        "exec",
        "compile",
        "__import__",
        "importlib.import_module",
        "import_module",
        "builtins.getattr",
        "builtins.eval",
        "builtins.exec",
        "builtins.__import__",
        "pkgutil.walk_packages",
        "importlib.util.find_spec",
    }
)
_EFFECTFUL_CALLEES = frozenset(
    {
        "print",
        "open",
        "exec",
        "eval",
        "compile",
        "__import__",
        "exit",
        "quit",
        "os.system",
        "os.popen",
        "os.remove",
        "os.unlink",
        "os.replace",
        "os.rename",
        "os.makedirs",
        "subprocess.run",
        "subprocess.call",
        "subprocess.Popen",
        "subprocess.check_call",
        "subprocess.check_output",
        "socket.socket",
        "urllib.request.urlopen",
        "requests.get",
        "requests.post",
        "pathlib.Path.write_text",
        "pathlib.Path.write_bytes",
        "sys.path.insert",
        "sys.path.append",
        "sys.exit",
    }
)
_FILESYSTEM_CALLEES = frozenset(
    {
        "open",
        "os.remove",
        "os.unlink",
        "os.replace",
        "os.rename",
        "os.makedirs",
        "pathlib.Path.write_text",
        "pathlib.Path.write_bytes",
    }
)
_PROCESS_CALLEES = frozenset(
    {
        "os.system",
        "os.popen",
        "subprocess.run",
        "subprocess.call",
        "subprocess.Popen",
        "subprocess.check_call",
        "subprocess.check_output",
        "sys.exit",
        "exit",
        "quit",
    }
)
_NETWORK_CALLEES = frozenset(
    {
        "socket.socket",
        "urllib.request.urlopen",
        "requests.get",
        "requests.post",
    }
)


class LegacyPathError(ArchitectureContractError):
    """Fail-closed legacy-path inventory contract violation."""


class LegacyPathAuthorityError(LegacyPathError):
    """Raised when the inventory is asked to delete, promote, or authorize."""


class PathKind(str, Enum):
    """Closed required legacy/simulation path classes (PCAR-PLAN-R1)."""

    MOCK_WORKERS = "mock workers"
    MOCK_INFERENCE_HANDLERS = "mock inference handlers"
    SIMULATED_HARDWARE = "simulated hardware"
    FAKE_OR_COMPATIBILITY_CIDS = "fake or compatibility CIDs"
    FIXTURE_PROVIDERS = "fixture providers"
    FALLBACK_SUCCESS_PATHS = "fallback success paths"
    DEPRECATED_COORDINATORS = "deprecated coordinators"
    LEGACY_ENDPOINT_REGISTRIES = "legacy endpoint registries"
    HISTORICAL_PROVIDER_ROUTERS = "historical provider routers"


REQUIRED_PATH_KINDS: tuple[PathKind, ...] = tuple(PathKind)
CLOSED_PATH_KINDS: frozenset[str] = frozenset(item.value for item in PathKind)


class ReachabilityDisposition(str, Enum):
    """Closed reachability vocabulary (PCAR-PLAN-R1)."""

    PRODUCTION_REACHABLE = "production_reachable"
    TEST_ONLY = "test_only"
    COMPATIBILITY_ONLY = "compatibility_only"
    DEAD = "dead"
    UNKNOWN = "unknown"


REQUIRED_REACHABILITY: tuple[ReachabilityDisposition, ...] = tuple(
    ReachabilityDisposition
)
CLOSED_REACHABILITY: frozenset[str] = frozenset(
    item.value for item in ReachabilityDisposition
)


class OriginTaint(str, Enum):
    """Closed origin-taint vocabulary preserved across reachability traces."""

    PRODUCTION = "production"
    COMPATIBILITY = "compatibility"
    FIXTURE = "fixture"
    SIMULATION = "simulation"
    MOCK = "mock"
    UNKNOWN = "unknown"


CLOSED_ORIGIN_TAINTS: frozenset[str] = frozenset(item.value for item in OriginTaint)
TAINTED_ORIGINS: frozenset[OriginTaint] = frozenset(
    {
        OriginTaint.COMPATIBILITY,
        OriginTaint.FIXTURE,
        OriginTaint.SIMULATION,
        OriginTaint.MOCK,
        OriginTaint.UNKNOWN,
    }
)
QUARANTINED_ORIGINS: frozenset[OriginTaint] = frozenset(
    {
        OriginTaint.COMPATIBILITY,
        OriginTaint.FIXTURE,
        OriginTaint.SIMULATION,
        OriginTaint.MOCK,
    }
)


class ProductionPredicate(str, Enum):
    """Closed production predicates that quarantined origins cannot satisfy."""

    CAPABILITY = "production_capability"
    EXECUTION_SUCCESS = "execution_success"
    PROOF = "proof"
    COMPLETION = "completion"
    RELEASE = "release"


REQUIRED_PRODUCTION_PREDICATES: tuple[ProductionPredicate, ...] = tuple(
    ProductionPredicate
)
CLOSED_PRODUCTION_PREDICATES: frozenset[str] = frozenset(
    item.value for item in ProductionPredicate
)


class DynamicMechanism(str, Enum):
    """Closed dynamic-loading mechanisms that block dead classification."""

    IMPORTLIB_IMPORT_MODULE = "importlib.import_module"
    META_PATH_FINDER = "importlib.abc.MetaPathFinder"
    GETATTR = "getattr"
    EVAL_EXEC = "eval_exec"
    PLUGIN_REGISTRY = "plugin_registry"
    UNKNOWN = "unknown"


CLOSED_DYNAMIC_MECHANISMS: frozenset[str] = frozenset(
    item.value for item in DynamicMechanism
)


class ReachabilityMethod(str, Enum):
    """How a reachability disposition was established."""

    STATIC_IMPORT = "static_import"
    STATIC_CALL = "static_call"
    ENTRYPOINT = "entrypoint"
    DECLARED_BINDING = "declared_binding"
    ARCHITECTURE_IR = "architecture_ir"
    DYNAMIC_UNKNOWN = "dynamic_unknown"
    UNREFERENCED_NO_DYNAMIC = "unreferenced_no_dynamic"
    UNKNOWN = "unknown"


CLOSED_REACHABILITY_METHODS: frozenset[str] = frozenset(
    item.value for item in ReachabilityMethod
)


class SideEffectKind(str, Enum):
    """Closed side-effect vocabulary for hermetic source scans."""

    NONE = "none"
    FILESYSTEM = "filesystem"
    PROCESS = "process"
    NETWORK = "network"
    MUTATION = "mutation"
    EXCEPTION = "exception"
    UNKNOWN = "unknown"


CLOSED_SIDE_EFFECTS: frozenset[str] = frozenset(item.value for item in SideEffectKind)


class LegacyConflictKind(str, Enum):
    """Closed hard-blocker vocabulary for the legacy inventory."""

    DEAD_WITH_DYNAMIC_LOADING = "dead_with_dynamic_loading"
    STATIC_ONLY_DEAD_CLAIM = "static_only_dead_claim"
    TAINTED_PRODUCTION_AUTHORITY = "tainted_production_authority"
    MISSING_SOURCE_IDENTITY = "missing_source_identity"
    MISSING_REQUIRED_PATH_KIND = "missing_required_path_kind"
    CONFLICTING_REACHABILITY = "conflicting_reachability"
    UNKNOWN_DYNAMIC_UNCLASSIFIED = "unknown_dynamic_unclassified"
    FAKE_TO_LIVE_PROMOTION = "fake_to_live_promotion"
    NON_PROBATIVE_DEAD = "non_probative_dead"


CLOSED_CONFLICT_KINDS: frozenset[str] = frozenset(
    item.value for item in LegacyConflictKind
)


_PATH_FIELDS = frozenset(
    {
        "content_identity",
        "dynamic_mechanisms",
        "kind",
        "origin_taint",
        "path",
        "path_id",
        "production_authority",
        "provenance",
        "reachability",
        "schema",
        "symbol",
        "uncertainty",
        "version",
    }
)
_DYNAMIC_FIELDS = frozenset(
    {
        "blocking",
        "content_identity",
        "mechanism",
        "path",
        "provenance",
        "schema",
        "symbol",
        "uncertainty",
        "version",
    }
)
_TAINT_FIELDS = frozenset(
    {
        "content_identity",
        "origin",
        "path",
        "predicates_blocked",
        "preserved",
        "schema",
        "symbol",
        "version",
    }
)
_TRACE_FIELDS = frozenset(
    {
        "content_identity",
        "disposition",
        "entrypoint",
        "hops",
        "method",
        "origin_taint",
        "schema",
        "target",
        "version",
    }
)
_SCAN_FIELDS = frozenset(
    {
        "content_identity",
        "effects",
        "imported_modules",
        "path",
        "schema",
        "side_effect_free",
        "version",
    }
)
_CONFLICT_FIELDS = frozenset(
    {
        "content_identity",
        "kind",
        "message",
        "path_ids",
        "schema",
        "version",
    }
)
_INVENTORY_FIELDS = frozenset(
    {
        "architecture_ir_identity",
        "authority",
        "can_authorize_deletion",
        "can_grant_production_authority",
        "can_promote_fake_to_live",
        "conflicts",
        "content_identity",
        "dynamic_records",
        "effect_class",
        "freshness",
        "paths",
        "repository_tree",
        "schema",
        "side_effect_scans",
        "taint_records",
        "traces",
        "version",
    }
)
_BINDING_FIELDS = frozenset(
    {
        "end_line",
        "kind",
        "nominated_symbol",
        "origin_taint",
        "path",
        "path_id",
        "reachability",
        "source_path",
        "start_line",
        "uncertainty",
    }
)
_INVENTORY_PATH_FIELDS = frozenset(
    {
        "kind",
        "nominated_symbol",
        "origin_taint",
        "path",
        "path_id",
        "present",
        "production_authority",
        "reachability",
        "source_span",
        "uncertainty",
    }
)
_INVENTORY_OPTIONAL_FIELDS = frozenset(
    {
        "dynamic_uncertainty",
        "mechanism",
    }
)


def _content_identity(payload: Mapping[str, Any]) -> str:
    return cid_for_dag_json(payload)


def _validate_dag_json_cid(value: str) -> str:
    try:
        return validate_cid(value, codecs=("dag-json",))
    except (TypeError, ValueError) as exc:
        raise LegacyPathError("content identity must be a dag-json CIDv1") from exc


def _reject_unknown(payload: Mapping[str, Any], allowed: Iterable[str]) -> None:
    extra = sorted(set(payload) - set(allowed))
    if extra:
        raise LegacyPathError(f"{_UNKNOWN_FIELD_MESSAGE}: {extra}")


def _require_fields(payload: Mapping[str, Any], allowed: Iterable[str]) -> None:
    allowed_fields = set(allowed)
    _reject_unknown(payload, allowed_fields)
    missing = sorted(allowed_fields - set(payload))
    if missing:
        raise LegacyPathError(f"{_MISSING_FIELD_MESSAGE}: {missing}")


def _require_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise LegacyPathError(f"{name} must be a boolean")
    return value


def _require_text_tuple(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise LegacyPathError(f"{name} must be a list of strings")
    items = tuple(
        _require_text(item, f"{name} item", error_type=LegacyPathError) for item in value
    )
    return tuple(sorted(set(items)))


def _require_ordered_text_tuple(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise LegacyPathError(f"{name} must be a list of strings")
    items: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = _require_text(item, f"{name} item", error_type=LegacyPathError)
        if text in seen:
            continue
        seen.add(text)
        items.append(text)
    return tuple(items)


def _looks_like_content_identity(value: str) -> bool:
    return value.startswith(_CID_PREFIXES)


def _wrap_contract(exc: ArchitectureContractError) -> LegacyPathError:
    if isinstance(exc, LegacyPathError):
        return exc
    return LegacyPathError(str(exc))


def _optional_text(value: Any, name: str) -> str:
    if value is None:
        return ""
    if type(value) is not str or "\x00" in value:
        raise LegacyPathError(f"{name} must be a string")
    return value


def _record_tuple(
    value: Any,
    record_type: type[Any],
    name: str,
) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise LegacyPathError(f"{name} must be a sequence")
    return tuple(
        item if isinstance(item, record_type) else record_type.from_mapping(item)
        for item in value
    )


def _fact(
    path: str,
    start: int,
    end: int,
    *,
    repository_tree: str,
    freshness: str,
    extractor_identity: str = EXTRACTOR_IDENTITY,
    confidence: Confidence = Confidence.EXACT,
) -> SourceFactIdentity:
    return SourceFactIdentity(
        extractor_identity=extractor_identity,
        span=SourceSpan(path, start, end),
        confidence=confidence,
        freshness=freshness,
        repository_tree=repository_tree,
    )


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/")


def _is_test_path(path: str) -> bool:
    lowered = f"/{_normalize_path(path).lower()}"
    return any(marker in lowered for marker in _TEST_PATH_MARKERS)


def _is_compat_path(path: str) -> bool:
    lowered = _normalize_path(path).lower()
    return any(marker in lowered for marker in _COMPAT_PATH_MARKERS)


def _text_suggests_simulation(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _SIMULATION_MARKERS)


def _default_taint_for_kind(kind: PathKind) -> OriginTaint:
    if kind is PathKind.FIXTURE_PROVIDERS:
        return OriginTaint.FIXTURE
    if kind in {
        PathKind.MOCK_WORKERS,
        PathKind.MOCK_INFERENCE_HANDLERS,
        PathKind.SIMULATED_HARDWARE,
    }:
        return OriginTaint.MOCK
    if kind in {
        PathKind.FAKE_OR_COMPATIBILITY_CIDS,
        PathKind.DEPRECATED_COORDINATORS,
        PathKind.LEGACY_ENDPOINT_REGISTRIES,
        PathKind.HISTORICAL_PROVIDER_ROUTERS,
    }:
        return OriginTaint.COMPATIBILITY
    if kind is PathKind.FALLBACK_SUCCESS_PATHS:
        return OriginTaint.SIMULATION
    return OriginTaint.UNKNOWN


def join_origin_taint(*origins: OriginTaint) -> OriginTaint:
    """Preserve the strongest (most quarantined) origin along a flow."""

    if not origins:
        return OriginTaint.UNKNOWN
    if any(item is OriginTaint.UNKNOWN for item in origins):
        return OriginTaint.UNKNOWN
    if any(item is OriginTaint.MOCK for item in origins):
        return OriginTaint.MOCK
    if any(item is OriginTaint.SIMULATION for item in origins):
        return OriginTaint.SIMULATION
    if any(item is OriginTaint.FIXTURE for item in origins):
        return OriginTaint.FIXTURE
    if any(item is OriginTaint.COMPATIBILITY for item in origins):
        return OriginTaint.COMPATIBILITY
    return OriginTaint.PRODUCTION


def origin_may_satisfy_production_predicate(
    origin: OriginTaint | str,
    predicate: ProductionPredicate | str | None = None,
) -> bool:
    """Quarantined origins cannot satisfy any production predicate."""

    parsed = _closed_enum(origin, OriginTaint, "origin taint", error_type=LegacyPathError)
    if predicate is not None:
        _closed_enum(
            predicate,
            ProductionPredicate,
            "production predicate",
            error_type=LegacyPathError,
        )
    return parsed is OriginTaint.PRODUCTION


def blocked_production_predicates(origin: OriginTaint) -> tuple[str, ...]:
    if origin not in TAINTED_ORIGINS:
        return ()
    return tuple(sorted(CLOSED_PRODUCTION_PREDICATES))


@dataclass(frozen=True)
class LegacySourceBinding:
    """Current-tree observational binding for one inventoried path."""

    path_id: str
    kind: PathKind
    path: str
    nominated_symbol: str
    origin_taint: OriginTaint
    reachability: ReachabilityDisposition
    source_path: str
    start_line: int
    end_line: int
    uncertainty: str = ""


CURRENT_LEGACY_BINDINGS: tuple[LegacySourceBinding, ...] = (
    LegacySourceBinding(
        path_id="mock-worker-accelerate",
        kind=PathKind.MOCK_WORKERS,
        path="ipfs_accelerate_py/ipfs_accelerate.py",
        nominated_symbol="MockWorker",
        origin_taint=OriginTaint.MOCK,
        reachability=ReachabilityDisposition.PRODUCTION_REACHABLE,
        source_path="ipfs_accelerate_py/ipfs_accelerate.py",
        start_line=141,
        end_line=141,
        uncertainty="production_constructor_substitutes_simulated_worker",
    ),
    LegacySourceBinding(
        path_id="mock-inference-handler",
        kind=PathKind.MOCK_INFERENCE_HANDLERS,
        path="ipfs_accelerate_py/ipfs_accelerate.py",
        nominated_symbol="_create_mock_handler",
        origin_taint=OriginTaint.MOCK,
        reachability=ReachabilityDisposition.PRODUCTION_REACHABLE,
        source_path="ipfs_accelerate_py/ipfs_accelerate.py",
        start_line=785,
        end_line=785,
        uncertainty="production_endpoint_registration_falls_back_to_mock_handler",
    ),
    LegacySourceBinding(
        path_id="mock-inference-mcp",
        kind=PathKind.MOCK_INFERENCE_HANDLERS,
        path="ipfs_accelerate_py/mcp/inference_tools.py",
        nominated_symbol="_mock_inference",
        origin_taint=OriginTaint.MOCK,
        reachability=ReachabilityDisposition.PRODUCTION_REACHABLE,
        source_path="ipfs_accelerate_py/mcp/inference_tools.py",
        start_line=109,
        end_line=109,
        uncertainty="mcp_inference_tools_return_mock_inference_results",
    ),
    LegacySourceBinding(
        path_id="mock-handler-legacy",
        kind=PathKind.MOCK_INFERENCE_HANDLERS,
        path="ipfs_accelerate_py/ipfs_accelerate_py_legacy.py",
        nominated_symbol="_create_mock_handler",
        origin_taint=OriginTaint.MOCK,
        reachability=ReachabilityDisposition.COMPATIBILITY_ONLY,
        source_path="ipfs_accelerate_py/ipfs_accelerate_py_legacy.py",
        start_line=327,
        end_line=327,
        uncertainty="legacy_compatibility_surface_must_not_be_canonical_authority",
    ),
    LegacySourceBinding(
        path_id="simulated-hw-legacy",
        kind=PathKind.SIMULATED_HARDWARE,
        path="ipfs_accelerate_py/ipfs_accelerate_py_legacy.py",
        nominated_symbol="_create_mock_hardware_detection",
        origin_taint=OriginTaint.SIMULATION,
        reachability=ReachabilityDisposition.COMPATIBILITY_ONLY,
        source_path="ipfs_accelerate_py/ipfs_accelerate_py_legacy.py",
        start_line=223,
        end_line=223,
        uncertainty="legacy_mock_hardware_detection_is_not_a_live_probe",
    ),
    LegacySourceBinding(
        path_id="cuda-mock-impl",
        kind=PathKind.SIMULATED_HARDWARE,
        path="ipfs_accelerate_py/worker/cuda_utils.py",
        nominated_symbol="create_cuda_mock_implementation",
        origin_taint=OriginTaint.SIMULATION,
        reachability=ReachabilityDisposition.PRODUCTION_REACHABLE,
        source_path="ipfs_accelerate_py/worker/cuda_utils.py",
        start_line=433,
        end_line=433,
        uncertainty="cuda_mock_implementation_must_not_prove_device_capability",
    ),
    LegacySourceBinding(
        path_id="fake-cid-hf-cache",
        kind=PathKind.FAKE_OR_COMPATIBILITY_CIDS,
        path="ipfs_accelerate_py/ipfs_backend_router.py",
        nominated_symbol="_generate_cid",
        origin_taint=OriginTaint.COMPATIBILITY,
        reachability=ReachabilityDisposition.PRODUCTION_REACHABLE,
        source_path="ipfs_accelerate_py/ipfs_backend_router.py",
        start_line=502,
        end_line=502,
        uncertainty="synthetic_bafy_cache_keys_are_not_multiformats_cids",
    ),
    LegacySourceBinding(
        path_id="mock-ipfs-random-cid",
        kind=PathKind.FAKE_OR_COMPATIBILITY_CIDS,
        path="ipfs_accelerate_py/mcp/tools/mock_ipfs.py",
        nominated_symbol="random_cid",
        origin_taint=OriginTaint.SIMULATION,
        reachability=ReachabilityDisposition.COMPATIBILITY_ONLY,
        source_path="ipfs_accelerate_py/mcp/tools/mock_ipfs.py",
        start_line=26,
        end_line=26,
        uncertainty="deprecated_mock_ipfs_client_emits_random_qm_strings",
    ),
    LegacySourceBinding(
        path_id="mock-ipfs-dynamic-loader",
        kind=PathKind.FAKE_OR_COMPATIBILITY_CIDS,
        path="ipfs_accelerate_py/mcp_server/tools/ipfs/__init__.py",
        nominated_symbol="_load_mock_ipfs_client",
        origin_taint=OriginTaint.UNKNOWN,
        reachability=ReachabilityDisposition.UNKNOWN,
        source_path="ipfs_accelerate_py/mcp_server/tools/ipfs/__init__.py",
        start_line=20,
        end_line=20,
        uncertainty="dynamic_import_of_legacy_or_stub_mock_client",
    ),
    LegacySourceBinding(
        path_id="fixture-mock-provider",
        kind=PathKind.FIXTURE_PROVIDERS,
        path="ipfs_accelerate_py/llm_router.py",
        nominated_symbol="_MockProvider",
        origin_taint=OriginTaint.FIXTURE,
        reachability=ReachabilityDisposition.PRODUCTION_REACHABLE,
        source_path="ipfs_accelerate_py/llm_router.py",
        start_line=7748,
        end_line=7748,
        uncertainty="offline_and_unit_test_provider_must_not_prove_live_capability",
    ),
    LegacySourceBinding(
        path_id="fixture-adversarial",
        kind=PathKind.FIXTURE_PROVIDERS,
        path="test/fixtures/adversarial_assurance/manifest.json",
        nominated_symbol="campaign_id",
        origin_taint=OriginTaint.FIXTURE,
        reachability=ReachabilityDisposition.TEST_ONLY,
        source_path="test/fixtures/adversarial_assurance/manifest.json",
        start_line=1,
        end_line=1,
    ),
    LegacySourceBinding(
        path_id="hermetic-model-inventory",
        kind=PathKind.FIXTURE_PROVIDERS,
        path="ipfs_accelerate_py/agent_supervisor/verification/model_route.py",
        nominated_symbol="default_inventory",
        origin_taint=OriginTaint.FIXTURE,
        reachability=ReachabilityDisposition.TEST_ONLY,
        source_path="ipfs_accelerate_py/agent_supervisor/verification/model_route.py",
        start_line=1461,
        end_line=1461,
        uncertainty="hermetic_provider_neutral_inventory_for_tests_and_local_runs",
    ),
    LegacySourceBinding(
        path_id="fallback-hardware-success",
        kind=PathKind.FALLBACK_SUCCESS_PATHS,
        path=(
            "ipfs_accelerate_py/mcp_server/tools/hardware_tools/"
            "native_hardware_tools.py"
        ),
        nominated_symbol="_test_hardware_fallback",
        origin_taint=OriginTaint.SIMULATION,
        reachability=ReachabilityDisposition.PRODUCTION_REACHABLE,
        source_path=(
            "ipfs_accelerate_py/mcp_server/tools/hardware_tools/"
            "native_hardware_tools.py"
        ),
        start_line=50,
        end_line=50,
        uncertainty="fallback_returns_overall_passed_true_without_a_live_probe",
    ),
    LegacySourceBinding(
        path_id="provider-fallback-runner",
        kind=PathKind.FALLBACK_SUCCESS_PATHS,
        path="ipfs_accelerate_py/agent_supervisor/provider_fallback_runner.py",
        nominated_symbol="main",
        origin_taint=OriginTaint.PRODUCTION,
        reachability=ReachabilityDisposition.PRODUCTION_REACHABLE,
        source_path="ipfs_accelerate_py/agent_supervisor/provider_fallback_runner.py",
        start_line=481,
        end_line=481,
        uncertainty="typed_quota_fallback_is_not_a_simulated_success",
    ),
    LegacySourceBinding(
        path_id="deprecated-workflow-coordinator",
        kind=PathKind.DEPRECATED_COORDINATORS,
        path="ipfs_accelerate_py/datasets_integration/workflow.py",
        nominated_symbol="WorkflowCoordinator",
        origin_taint=OriginTaint.COMPATIBILITY,
        reachability=ReachabilityDisposition.PRODUCTION_REACHABLE,
        source_path="ipfs_accelerate_py/datasets_integration/workflow.py",
        start_line=13,
        end_line=13,
        uncertainty="p2p_workflow_coordinator_falls_back_to_a_local_queue",
    ),
    LegacySourceBinding(
        path_id="legacy-mcp-endpoints",
        kind=PathKind.LEGACY_ENDPOINT_REGISTRIES,
        path="ipfs_accelerate_py/mcp/tools/endpoints.py",
        nominated_symbol="ENDPOINTS",
        origin_taint=OriginTaint.COMPATIBILITY,
        reachability=ReachabilityDisposition.COMPATIBILITY_ONLY,
        source_path="ipfs_accelerate_py/mcp/tools/endpoints.py",
        start_line=26,
        end_line=26,
        uncertainty="deprecated_in_memory_endpoint_registry_is_a_compatibility_shim",
    ),
    LegacySourceBinding(
        path_id="cli-endpoint-registry",
        kind=PathKind.LEGACY_ENDPOINT_REGISTRIES,
        path="ipfs_accelerate_py/cli_runtime/endpoints.py",
        nominated_symbol="CLIEndpointRegistry",
        origin_taint=OriginTaint.PRODUCTION,
        reachability=ReachabilityDisposition.PRODUCTION_REACHABLE,
        source_path="ipfs_accelerate_py/cli_runtime/endpoints.py",
        start_line=703,
        end_line=703,
        uncertainty="current_registry_retains_a_legacy_compatible_adapter_view",
    ),
    LegacySourceBinding(
        path_id="historical-embedding-router",
        kind=PathKind.HISTORICAL_PROVIDER_ROUTERS,
        path="ipfs_accelerate_py/embedding_router.py",
        nominated_symbol="embeddings_router",
        origin_taint=OriginTaint.COMPATIBILITY,
        reachability=ReachabilityDisposition.COMPATIBILITY_ONLY,
        source_path="ipfs_accelerate_py/embedding_router.py",
        start_line=7,
        end_line=7,
        uncertainty="compatibility_alias_must_not_be_treated_as_canonical_authority",
    ),
    LegacySourceBinding(
        path_id="historical-tts-router",
        kind=PathKind.HISTORICAL_PROVIDER_ROUTERS,
        path="ipfs_accelerate_py/tts_router.py",
        nominated_symbol="voice_router",
        origin_taint=OriginTaint.COMPATIBILITY,
        reachability=ReachabilityDisposition.COMPATIBILITY_ONLY,
        source_path="ipfs_accelerate_py/tts_router.py",
        start_line=3,
        end_line=3,
        uncertainty="historical_tts_router_is_a_backward_compatibility_shim",
    ),
    LegacySourceBinding(
        path_id="historical-ipfs-backend-router",
        kind=PathKind.HISTORICAL_PROVIDER_ROUTERS,
        path="ipfs_accelerate_py/ipfs_backend_router.py",
        nominated_symbol="IPFS_BACKEND",
        origin_taint=OriginTaint.COMPATIBILITY,
        reachability=ReachabilityDisposition.PRODUCTION_REACHABLE,
        source_path="ipfs_accelerate_py/ipfs_backend_router.py",
        start_line=19,
        end_line=19,
        uncertainty="pluggable_backend_router_must_not_treat_synthetic_cids_as_live",
    ),
    LegacySourceBinding(
        path_id="dynamic-landed-alias",
        kind=PathKind.HISTORICAL_PROVIDER_ROUTERS,
        path="ipfs_accelerate_py/agent_supervisor/__init__.py",
        nominated_symbol="_load_landed_module",
        origin_taint=OriginTaint.UNKNOWN,
        reachability=ReachabilityDisposition.UNKNOWN,
        source_path="ipfs_accelerate_py/agent_supervisor/__init__.py",
        start_line=632,
        end_line=632,
        uncertainty=(
            "landed_module_alias_membership_depends_on_"
            "AGENT_SUPERVISOR_LANDED_MODULE_TO_PACKAGE"
        ),
    ),
    LegacySourceBinding(
        path_id="dcr-shadow-simulation",
        kind=PathKind.FIXTURE_PROVIDERS,
        path="ipfs_accelerate_py/agent_supervisor/evaluation/dcr_shadow.py",
        nominated_symbol="ShadowProposal",
        origin_taint=OriginTaint.SIMULATION,
        reachability=ReachabilityDisposition.TEST_ONLY,
        source_path="ipfs_accelerate_py/agent_supervisor/evaluation/dcr_shadow.py",
        start_line=97,
        end_line=97,
        uncertainty="static_reachability_does_not_prove_production_isolation",
    ),
)


def _path_sort_key(item: "LegacyPathRecord") -> tuple[str, str, str]:
    return (item.kind.value, item.path_id, item.path)


def _dynamic_sort_key(item: "DynamicReachabilityRecord") -> tuple[str, str, str]:
    return (item.path, item.symbol, item.mechanism.value)


def _taint_sort_key(item: "OriginTaintRecord") -> tuple[str, str, str]:
    return (item.path, item.symbol, item.origin.value)


def _trace_sort_key(item: "ReachabilityTrace") -> tuple[str, str, str]:
    return (item.entrypoint, item.target, item.disposition.value)


def _scan_sort_key(item: "SideEffectScan") -> str:
    return item.path


def _conflict_sort_key(item: "LegacyConflict") -> tuple[str, str, tuple[str, ...]]:
    return (item.kind.value, item.message, item.path_ids)


@dataclass(frozen=True)
class DynamicReachabilityRecord:
    """One dynamic-loading site that blocks dead classification."""

    path: str
    symbol: str
    mechanism: DynamicMechanism
    provenance: SourceFactIdentity
    uncertainty: str
    blocking: bool = True
    schema: str = DYNAMIC_RECORD_SCHEMA
    version: int = DYNAMIC_RECORD_VERSION
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=LegacyPathError)
        if schema != DYNAMIC_RECORD_SCHEMA:
            raise LegacyPathError("unexpected dynamic-reachability schema")
        version = _require_int(self.version, "version", error_type=LegacyPathError)
        if version != DYNAMIC_RECORD_VERSION:
            raise LegacyPathError("unexpected dynamic-reachability version")
        path = _require_text(self.path, "path", error_type=LegacyPathError)
        symbol = _require_text(self.symbol, "symbol", error_type=LegacyPathError)
        mechanism = _closed_enum(
            self.mechanism,
            DynamicMechanism,
            "dynamic mechanism",
            error_type=LegacyPathError,
        )
        provenance = (
            self.provenance
            if isinstance(self.provenance, SourceFactIdentity)
            else SourceFactIdentity.from_mapping(self.provenance)
        )
        uncertainty = _require_text(
            self.uncertainty, "uncertainty", error_type=LegacyPathError
        )
        blocking = _require_bool(self.blocking, "blocking")
        if not blocking:
            raise LegacyPathError(
                "dynamic-loading uncertainty must remain blocking, not dead"
            )
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "mechanism", mechanism)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "uncertainty", uncertainty)
        object.__setattr__(self, "blocking", True)
        identity = _content_identity(self._identity_payload())
        if self.content_identity:
            claimed = _validate_dag_json_cid(
                _require_text(
                    self.content_identity,
                    "content_identity",
                    error_type=LegacyPathError,
                )
            )
            if claimed != identity:
                raise LegacyPathError("dynamic-reachability content identity mismatch")
        object.__setattr__(self, "content_identity", identity)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "blocking": True,
            "mechanism": self.mechanism.value,
            "path": self.path,
            "provenance": self.provenance.to_dict(),
            "schema": self.schema,
            "symbol": self.symbol,
            "uncertainty": self.uncertainty,
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise LegacyPathError("dynamic-reachability content identity mismatch")
        return {**payload, "content_identity": identity}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "DynamicReachabilityRecord":
        mapping = _require_mapping(payload, error_type=LegacyPathError)
        _require_fields(mapping, _DYNAMIC_FIELDS)
        try:
            record = cls(
                path=mapping["path"],
                symbol=mapping["symbol"],
                mechanism=mapping["mechanism"],
                provenance=mapping["provenance"],
                uncertainty=mapping["uncertainty"],
                blocking=mapping["blocking"],
                schema=mapping["schema"],
                version=mapping["version"],
            )
        except ArchitectureContractError as exc:
            raise _wrap_contract(exc) from exc
        if mapping["content_identity"] != record.content_identity:
            raise LegacyPathError("dynamic-reachability content identity mismatch")
        return record

    from_dict = from_mapping


@dataclass(frozen=True)
class OriginTaintRecord:
    """Preserved origin taint for one inventoried symbol."""

    path: str
    symbol: str
    origin: OriginTaint
    predicates_blocked: tuple[str, ...]
    preserved: bool = True
    schema: str = TAINT_RECORD_SCHEMA
    version: int = TAINT_RECORD_VERSION
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=LegacyPathError)
        if schema != TAINT_RECORD_SCHEMA:
            raise LegacyPathError("unexpected origin-taint schema")
        version = _require_int(self.version, "version", error_type=LegacyPathError)
        if version != TAINT_RECORD_VERSION:
            raise LegacyPathError("unexpected origin-taint version")
        path = _require_text(self.path, "path", error_type=LegacyPathError)
        symbol = _require_text(self.symbol, "symbol", error_type=LegacyPathError)
        origin = _closed_enum(
            self.origin, OriginTaint, "origin taint", error_type=LegacyPathError
        )
        predicates = _require_text_tuple(self.predicates_blocked, "predicates_blocked")
        expected = blocked_production_predicates(origin)
        if predicates != expected:
            raise LegacyPathError(
                "origin taint must block exactly the closed production predicates"
            )
        preserved = _require_bool(self.preserved, "preserved")
        if not preserved:
            raise LegacyPathError("origin taint must be preserved")
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "predicates_blocked", predicates)
        object.__setattr__(self, "preserved", True)
        identity = _content_identity(self._identity_payload())
        if self.content_identity:
            claimed = _validate_dag_json_cid(
                _require_text(
                    self.content_identity,
                    "content_identity",
                    error_type=LegacyPathError,
                )
            )
            if claimed != identity:
                raise LegacyPathError("origin-taint content identity mismatch")
        object.__setattr__(self, "content_identity", identity)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "origin": self.origin.value,
            "path": self.path,
            "predicates_blocked": list(self.predicates_blocked),
            "preserved": True,
            "schema": self.schema,
            "symbol": self.symbol,
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise LegacyPathError("origin-taint content identity mismatch")
        return {**payload, "content_identity": identity}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "OriginTaintRecord":
        mapping = _require_mapping(payload, error_type=LegacyPathError)
        _require_fields(mapping, _TAINT_FIELDS)
        record = cls(
            path=mapping["path"],
            symbol=mapping["symbol"],
            origin=mapping["origin"],
            predicates_blocked=mapping["predicates_blocked"],
            preserved=mapping["preserved"],
            schema=mapping["schema"],
            version=mapping["version"],
        )
        if mapping["content_identity"] != record.content_identity:
            raise LegacyPathError("origin-taint content identity mismatch")
        return record

    from_dict = from_mapping


@dataclass(frozen=True)
class ReachabilityTrace:
    """One entrypoint-to-target reachability observation."""

    entrypoint: str
    target: str
    disposition: ReachabilityDisposition
    origin_taint: OriginTaint
    method: ReachabilityMethod
    hops: tuple[str, ...] = ()
    schema: str = TRACE_SCHEMA
    version: int = TRACE_VERSION
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=LegacyPathError)
        if schema != TRACE_SCHEMA:
            raise LegacyPathError("unexpected reachability-trace schema")
        version = _require_int(self.version, "version", error_type=LegacyPathError)
        if version != TRACE_VERSION:
            raise LegacyPathError("unexpected reachability-trace version")
        entrypoint = _require_text(
            self.entrypoint, "entrypoint", error_type=LegacyPathError
        )
        target = _require_text(self.target, "target", error_type=LegacyPathError)
        disposition = _closed_enum(
            self.disposition,
            ReachabilityDisposition,
            "reachability",
            error_type=LegacyPathError,
        )
        origin = _closed_enum(
            self.origin_taint,
            OriginTaint,
            "origin taint",
            error_type=LegacyPathError,
        )
        method = _closed_enum(
            self.method,
            ReachabilityMethod,
            "reachability method",
            error_type=LegacyPathError,
        )
        hops = _require_ordered_text_tuple(self.hops, "hops")
        if (
            disposition is ReachabilityDisposition.DEAD
            and method is ReachabilityMethod.STATIC_IMPORT
        ):
            raise LegacyPathError(
                "static reachability alone never proves dead code"
            )
        if (
            disposition is ReachabilityDisposition.DEAD
            and method is ReachabilityMethod.DYNAMIC_UNKNOWN
        ):
            raise LegacyPathError(
                "dynamic-loading uncertainty must remain unknown, not dead"
            )
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "entrypoint", entrypoint)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(self, "origin_taint", origin)
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "hops", hops)
        identity = _content_identity(self._identity_payload())
        if self.content_identity:
            claimed = _validate_dag_json_cid(
                _require_text(
                    self.content_identity,
                    "content_identity",
                    error_type=LegacyPathError,
                )
            )
            if claimed != identity:
                raise LegacyPathError("reachability-trace content identity mismatch")
        object.__setattr__(self, "content_identity", identity)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition.value,
            "entrypoint": self.entrypoint,
            "hops": list(self.hops),
            "method": self.method.value,
            "origin_taint": self.origin_taint.value,
            "schema": self.schema,
            "target": self.target,
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise LegacyPathError("reachability-trace content identity mismatch")
        return {**payload, "content_identity": identity}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ReachabilityTrace":
        mapping = _require_mapping(payload, error_type=LegacyPathError)
        _require_fields(mapping, _TRACE_FIELDS)
        record = cls(
            entrypoint=mapping["entrypoint"],
            target=mapping["target"],
            disposition=mapping["disposition"],
            origin_taint=mapping["origin_taint"],
            method=mapping["method"],
            hops=mapping["hops"],
            schema=mapping["schema"],
            version=mapping["version"],
        )
        if mapping["content_identity"] != record.content_identity:
            raise LegacyPathError("reachability-trace content identity mismatch")
        return record

    from_dict = from_mapping


@dataclass(frozen=True)
class SideEffectScan:
    """Hermetic AST scan of one source; never executes the module."""

    path: str
    effects: tuple[SideEffectKind, ...]
    imported_modules: tuple[str, ...]
    side_effect_free: bool
    schema: str = SCAN_SCHEMA
    version: int = SCAN_SCHEMA_VERSION
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=LegacyPathError)
        if schema != SCAN_SCHEMA:
            raise LegacyPathError("unexpected side-effect-scan schema")
        version = _require_int(self.version, "version", error_type=LegacyPathError)
        if version != SCAN_SCHEMA_VERSION:
            raise LegacyPathError("unexpected side-effect-scan version")
        path = _require_text(self.path, "path", error_type=LegacyPathError)
        if isinstance(self.effects, (str, bytes, bytearray)) or not isinstance(
            self.effects, Sequence
        ):
            raise LegacyPathError("effects must be a list of closed side-effect kinds")
        effects = tuple(
            _closed_enum(
                item, SideEffectKind, "side effect", error_type=LegacyPathError
            )
            for item in self.effects
        )
        imported = _require_text_tuple(self.imported_modules, "imported_modules")
        side_effect_free = _require_bool(self.side_effect_free, "side_effect_free")
        expected_free = effects == (SideEffectKind.NONE,) or effects == ()
        if side_effect_free != expected_free:
            raise LegacyPathError("side_effect_free must match recorded effects")
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "effects", effects)
        object.__setattr__(self, "imported_modules", imported)
        object.__setattr__(self, "side_effect_free", side_effect_free)
        identity = _content_identity(self._identity_payload())
        if self.content_identity:
            claimed = _validate_dag_json_cid(
                _require_text(
                    self.content_identity,
                    "content_identity",
                    error_type=LegacyPathError,
                )
            )
            if claimed != identity:
                raise LegacyPathError("side-effect-scan content identity mismatch")
        object.__setattr__(self, "content_identity", identity)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "effects": [item.value for item in self.effects],
            "imported_modules": list(self.imported_modules),
            "path": self.path,
            "schema": self.schema,
            "side_effect_free": self.side_effect_free,
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise LegacyPathError("side-effect-scan content identity mismatch")
        return {**payload, "content_identity": identity}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SideEffectScan":
        mapping = _require_mapping(payload, error_type=LegacyPathError)
        _require_fields(mapping, _SCAN_FIELDS)
        record = cls(
            path=mapping["path"],
            effects=mapping["effects"],
            imported_modules=mapping["imported_modules"],
            side_effect_free=mapping["side_effect_free"],
            schema=mapping["schema"],
            version=mapping["version"],
        )
        if mapping["content_identity"] != record.content_identity:
            raise LegacyPathError("side-effect-scan content identity mismatch")
        return record

    from_dict = from_mapping


@dataclass(frozen=True)
class LegacyConflict:
    """Typed hard blocker that prevents inventory closure."""

    kind: LegacyConflictKind
    message: str
    path_ids: tuple[str, ...] = ()
    schema: str = "ipfs_accelerate_py/agent-supervisor/legacy-conflict@1"
    version: int = 1
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=LegacyPathError)
        version = _require_int(self.version, "version", error_type=LegacyPathError)
        kind = _closed_enum(
            self.kind,
            LegacyConflictKind,
            "legacy conflict kind",
            error_type=LegacyPathError,
        )
        message = _require_text(self.message, "message", error_type=LegacyPathError)
        path_ids = _require_text_tuple(self.path_ids, "path_ids")
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "path_ids", path_ids)
        identity = _content_identity(self._identity_payload())
        if self.content_identity:
            claimed = _validate_dag_json_cid(
                _require_text(
                    self.content_identity,
                    "content_identity",
                    error_type=LegacyPathError,
                )
            )
            if claimed != identity:
                raise LegacyPathError("legacy-conflict content identity mismatch")
        object.__setattr__(self, "content_identity", identity)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "message": self.message,
            "path_ids": list(self.path_ids),
            "schema": self.schema,
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise LegacyPathError("legacy-conflict content identity mismatch")
        return {**payload, "content_identity": identity}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "LegacyConflict":
        mapping = _require_mapping(payload, error_type=LegacyPathError)
        _require_fields(mapping, _CONFLICT_FIELDS)
        record = cls(
            kind=mapping["kind"],
            message=mapping["message"],
            path_ids=mapping["path_ids"],
            schema=mapping["schema"],
            version=mapping["version"],
        )
        if mapping["content_identity"] != record.content_identity:
            raise LegacyPathError("legacy-conflict content identity mismatch")
        return record

    from_dict = from_mapping


def _validate_path_invariants(
    *,
    path_id: str,
    kind: PathKind,
    origin_taint: OriginTaint,
    reachability: ReachabilityDisposition,
    production_authority: bool,
    provenance: SourceFactIdentity,
    dynamic_mechanisms: tuple[DynamicMechanism, ...],
) -> None:
    if _looks_like_content_identity(path_id):
        raise LegacyPathError("content identity is not inferred to be authority")
    if production_authority:
        raise LegacyPathAuthorityError(
            "legacy-path inventory cannot grant production authority"
        )
    if origin_taint in QUARANTINED_ORIGINS and production_authority:
        raise LegacyPathError("tainted origin cannot be production authority")
    if (
        reachability is ReachabilityDisposition.DEAD
        and dynamic_mechanisms
    ):
        raise LegacyPathError(
            "dynamic-loading uncertainty must remain unknown, not dead"
        )
    if (
        reachability is ReachabilityDisposition.DEAD
        and provenance.confidence in NON_PROBATIVE_CONFIDENCE
    ):
        raise LegacyPathError("heuristic or opaque facts cannot prove dead code")
    if (
        reachability is ReachabilityDisposition.DEAD
        and STATIC_REACHABILITY_PROVES_DEAD
    ):
        raise LegacyPathError("static reachability alone never proves dead code")
    if kind in {
        PathKind.MOCK_WORKERS,
        PathKind.MOCK_INFERENCE_HANDLERS,
        PathKind.SIMULATED_HARDWARE,
        PathKind.FIXTURE_PROVIDERS,
    } and origin_taint is OriginTaint.PRODUCTION:
        raise LegacyPathError(
            "mock, simulation, and fixture paths cannot have production origin"
        )


@dataclass(frozen=True)
class LegacyPathRecord:
    """One inventoried legacy/simulation/fixture/compatibility path."""

    path_id: str
    kind: PathKind
    path: str
    symbol: str
    origin_taint: OriginTaint
    reachability: ReachabilityDisposition
    provenance: SourceFactIdentity
    production_authority: bool = False
    dynamic_mechanisms: tuple[DynamicMechanism, ...] = ()
    uncertainty: str = ""
    schema: str = LEGACY_PATH_SCHEMA
    version: int = LEGACY_PATH_VERSION
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=LegacyPathError)
        if schema != LEGACY_PATH_SCHEMA:
            raise LegacyPathError("unexpected legacy-path schema")
        version = _require_int(self.version, "version", error_type=LegacyPathError)
        if version != LEGACY_PATH_VERSION:
            raise LegacyPathError("unexpected legacy-path version")
        path_id = _require_text(self.path_id, "path_id", error_type=LegacyPathError)
        kind = _closed_enum(
            self.kind, PathKind, "path kind", error_type=LegacyPathError
        )
        path = _require_text(self.path, "path", error_type=LegacyPathError)
        symbol = _require_text(self.symbol, "symbol", error_type=LegacyPathError)
        origin = _closed_enum(
            self.origin_taint,
            OriginTaint,
            "origin taint",
            error_type=LegacyPathError,
        )
        reachability = _closed_enum(
            self.reachability,
            ReachabilityDisposition,
            "reachability",
            error_type=LegacyPathError,
        )
        provenance = (
            self.provenance
            if isinstance(self.provenance, SourceFactIdentity)
            else SourceFactIdentity.from_mapping(self.provenance)
        )
        production_authority = _require_bool(
            self.production_authority, "production_authority"
        )
        if isinstance(self.dynamic_mechanisms, (str, bytes, bytearray)) or not isinstance(
            self.dynamic_mechanisms, Sequence
        ):
            raise LegacyPathError("dynamic_mechanisms must be a sequence")
        mechanisms = tuple(
            sorted(
                {
                    _closed_enum(
                        item,
                        DynamicMechanism,
                        "dynamic mechanism",
                        error_type=LegacyPathError,
                    )
                    for item in self.dynamic_mechanisms
                },
                key=lambda item: item.value,
            )
        )
        uncertainty = _optional_text(self.uncertainty, "uncertainty")
        if reachability is ReachabilityDisposition.UNKNOWN and not uncertainty:
            raise LegacyPathError("unknown reachability must declare uncertainty")
        if mechanisms and reachability is ReachabilityDisposition.DEAD:
            raise LegacyPathError(
                "dynamic-loading uncertainty must remain unknown, not dead"
            )
        _validate_path_invariants(
            path_id=path_id,
            kind=kind,
            origin_taint=origin,
            reachability=reachability,
            production_authority=production_authority,
            provenance=provenance,
            dynamic_mechanisms=mechanisms,
        )
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "path_id", path_id)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "origin_taint", origin)
        object.__setattr__(self, "reachability", reachability)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "production_authority", False)
        object.__setattr__(self, "dynamic_mechanisms", mechanisms)
        object.__setattr__(self, "uncertainty", uncertainty)
        identity = _content_identity(self._identity_payload())
        if self.content_identity:
            claimed = _validate_dag_json_cid(
                _require_text(
                    self.content_identity,
                    "content_identity",
                    error_type=LegacyPathError,
                )
            )
            if claimed != identity:
                raise LegacyPathError("legacy-path content identity mismatch")
        object.__setattr__(self, "content_identity", identity)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "dynamic_mechanisms": [item.value for item in self.dynamic_mechanisms],
            "kind": self.kind.value,
            "origin_taint": self.origin_taint.value,
            "path": self.path,
            "path_id": self.path_id,
            "production_authority": False,
            "provenance": self.provenance.to_dict(),
            "reachability": self.reachability.value,
            "schema": self.schema,
            "symbol": self.symbol,
            "uncertainty": self.uncertainty,
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise LegacyPathError("legacy-path content identity mismatch")
        return {**payload, "content_identity": identity}

    @property
    def is_quarantined(self) -> bool:
        return self.origin_taint in QUARANTINED_ORIGINS

    @property
    def may_satisfy_production_predicate(self) -> bool:
        return origin_may_satisfy_production_predicate(self.origin_taint)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "LegacyPathRecord":
        mapping = _require_mapping(payload, error_type=LegacyPathError)
        _require_fields(mapping, _PATH_FIELDS)
        try:
            record = cls(
                path_id=mapping["path_id"],
                kind=mapping["kind"],
                path=mapping["path"],
                symbol=mapping["symbol"],
                origin_taint=mapping["origin_taint"],
                reachability=mapping["reachability"],
                provenance=mapping["provenance"],
                production_authority=mapping["production_authority"],
                dynamic_mechanisms=mapping["dynamic_mechanisms"],
                uncertainty=mapping["uncertainty"],
                schema=mapping["schema"],
                version=mapping["version"],
            )
        except ArchitectureContractError as exc:
            raise _wrap_contract(exc) from exc
        if mapping["content_identity"] != record.content_identity:
            raise LegacyPathError("legacy-path content identity mismatch")
        return record

    from_dict = from_mapping


def classify_legacy_path(
    record: LegacyPathRecord | Mapping[str, Any],
    *,
    repository_tree: str,
    freshness: str = DEFAULT_FRESHNESS,
    extractor_identity: str = EXTRACTOR_IDENTITY,
) -> LegacyPathRecord:
    """Re-bind one path record onto the inventory tree and freshness."""

    parsed = (
        record
        if isinstance(record, LegacyPathRecord)
        else LegacyPathRecord.from_mapping(record)
    )
    provenance = parsed.provenance
    if (
        provenance.repository_tree != repository_tree
        or provenance.freshness != freshness
        or provenance.extractor_identity != extractor_identity
    ):
        provenance = SourceFactIdentity(
            extractor_identity=extractor_identity,
            span=provenance.span,
            confidence=provenance.confidence,
            freshness=freshness,
            repository_tree=repository_tree,
        )
    return LegacyPathRecord(
        path_id=parsed.path_id,
        kind=parsed.kind,
        path=parsed.path,
        symbol=parsed.symbol,
        origin_taint=parsed.origin_taint,
        reachability=parsed.reachability,
        provenance=provenance,
        production_authority=False,
        dynamic_mechanisms=parsed.dynamic_mechanisms,
        uncertainty=parsed.uncertainty,
    )


def _path_from_binding(
    binding: LegacySourceBinding,
    *,
    repository_tree: str,
    freshness: str,
    extractor_identity: str,
) -> LegacyPathRecord:
    mechanisms: tuple[DynamicMechanism, ...] = ()
    if binding.reachability is ReachabilityDisposition.UNKNOWN:
        if "meta_path" in binding.uncertainty or "finder" in binding.uncertainty:
            mechanisms = (DynamicMechanism.META_PATH_FINDER,)
        elif "import" in binding.uncertainty or "dynamic" in binding.uncertainty:
            mechanisms = (DynamicMechanism.IMPORTLIB_IMPORT_MODULE,)
        else:
            mechanisms = (DynamicMechanism.UNKNOWN,)
    confidence = (
        Confidence.OPAQUE
        if binding.reachability is ReachabilityDisposition.UNKNOWN
        else Confidence.EXACT
    )
    return LegacyPathRecord(
        path_id=binding.path_id,
        kind=binding.kind,
        path=binding.path,
        symbol=binding.nominated_symbol,
        origin_taint=binding.origin_taint,
        reachability=binding.reachability,
        provenance=_fact(
            binding.source_path,
            binding.start_line,
            binding.end_line,
            repository_tree=repository_tree,
            freshness=freshness,
            extractor_identity=extractor_identity,
            confidence=confidence,
        ),
        dynamic_mechanisms=mechanisms,
        uncertainty=binding.uncertainty,
    )


def current_legacy_paths(
    *,
    repository_tree: str,
    freshness: str = DEFAULT_FRESHNESS,
    extractor_identity: str = EXTRACTOR_IDENTITY,
) -> tuple[LegacyPathRecord, ...]:
    """Classify the current-tree legacy/simulation bindings."""

    return tuple(
        sorted(
            (
                _path_from_binding(
                    binding,
                    repository_tree=repository_tree,
                    freshness=freshness,
                    extractor_identity=extractor_identity,
                )
                for binding in CURRENT_LEGACY_BINDINGS
            ),
            key=_path_sort_key,
        )
    )


def current_dynamic_records(
    *,
    repository_tree: str,
    freshness: str = DEFAULT_FRESHNESS,
    extractor_identity: str = EXTRACTOR_IDENTITY,
) -> tuple[DynamicReachabilityRecord, ...]:
    """Dynamic-loading sites from current-tree bindings that remain unknown."""

    records: list[DynamicReachabilityRecord] = []
    for binding in CURRENT_LEGACY_BINDINGS:
        if binding.reachability is not ReachabilityDisposition.UNKNOWN:
            continue
        mechanism = DynamicMechanism.IMPORTLIB_IMPORT_MODULE
        if "finder" in binding.uncertainty or "meta_path" in binding.uncertainty:
            mechanism = DynamicMechanism.META_PATH_FINDER
        records.append(
            DynamicReachabilityRecord(
                path=binding.path,
                symbol=binding.nominated_symbol,
                mechanism=mechanism,
                provenance=_fact(
                    binding.source_path,
                    binding.start_line,
                    binding.end_line,
                    repository_tree=repository_tree,
                    freshness=freshness,
                    extractor_identity=extractor_identity,
                    confidence=Confidence.OPAQUE,
                ),
                uncertainty=binding.uncertainty or "dynamic_loading_membership",
            )
        )
    return tuple(sorted(records, key=_dynamic_sort_key))


def taint_record_for(path: LegacyPathRecord) -> OriginTaintRecord:
    """Preserve origin taint and the blocked production predicates."""

    return OriginTaintRecord(
        path=path.path,
        symbol=path.symbol,
        origin=path.origin_taint,
        predicates_blocked=blocked_production_predicates(path.origin_taint),
    )


def classify_reachability(
    *,
    static_from_production: bool,
    static_from_test: bool,
    static_from_compatibility: bool,
    dynamic_records: Sequence[DynamicReachabilityRecord] = (),
    unreferenced: bool = False,
    dead_when_unreferenced: bool = False,
) -> ReachabilityDisposition:
    """Closed reachability classifier. Dynamic sites stay unknown, never dead."""

    if dynamic_records:
        return ReachabilityDisposition.UNKNOWN
    if static_from_production:
        return ReachabilityDisposition.PRODUCTION_REACHABLE
    if static_from_compatibility and not static_from_test:
        return ReachabilityDisposition.COMPATIBILITY_ONLY
    if static_from_test and not static_from_production:
        return ReachabilityDisposition.TEST_ONLY
    if static_from_compatibility:
        return ReachabilityDisposition.COMPATIBILITY_ONLY
    if unreferenced and dead_when_unreferenced:
        return ReachabilityDisposition.DEAD
    return ReachabilityDisposition.UNKNOWN


def refuse_deletion(path_id: str) -> None:
    """Reject attempts to treat the inventory as a deletion authority."""

    _require_text(path_id, "path_id", error_type=LegacyPathError)
    raise LegacyPathAuthorityError(
        "legacy-path inventory cannot authorize deletion"
    )


def refuse_fake_to_live_promotion(path_id: str) -> None:
    """Reject attempts to promote a fake, fixture, or simulation origin."""

    _require_text(path_id, "path_id", error_type=LegacyPathError)
    raise LegacyPathAuthorityError(
        "legacy-path inventory cannot promote fake-to-live"
    )


def refuse_production_authority(path_id: str) -> None:
    """Reject attempts to grant production authority from this inventory."""

    _require_text(path_id, "path_id", error_type=LegacyPathError)
    raise LegacyPathAuthorityError(
        "legacy-path inventory cannot grant production authority"
    )


def refuse_dead_from_static_only(path_id: str) -> None:
    """Static reachability alone never proves dead code."""

    _require_text(path_id, "path_id", error_type=LegacyPathError)
    raise LegacyPathError("static reachability alone never proves dead code")


def _mechanism_for_callee(callee: str) -> DynamicMechanism | None:
    if callee in {"importlib.import_module", "import_module", "__import__", "builtins.__import__"}:
        return DynamicMechanism.IMPORTLIB_IMPORT_MODULE
    if callee in {"eval", "exec", "compile", "builtins.eval", "builtins.exec"}:
        return DynamicMechanism.EVAL_EXEC
    if callee in {"getattr", "setattr", "delattr", "globals", "locals", "builtins.getattr"}:
        return DynamicMechanism.GETATTR
    if callee in {"pkgutil.walk_packages", "importlib.util.find_spec"}:
        return DynamicMechanism.PLUGIN_REGISTRY
    if callee in _DYNAMIC_CALLEES:
        return DynamicMechanism.UNKNOWN
    return None


def _effect_kinds_for(callee: str) -> tuple[SideEffectKind, ...]:
    kinds: list[SideEffectKind] = []
    if callee in _FILESYSTEM_CALLEES or callee.endswith(".write_text") or callee.endswith(
        ".write_bytes"
    ):
        kinds.append(SideEffectKind.FILESYSTEM)
    if callee in _PROCESS_CALLEES:
        kinds.append(SideEffectKind.PROCESS)
    if callee in _NETWORK_CALLEES:
        kinds.append(SideEffectKind.NETWORK)
    if callee in {"sys.path.insert", "sys.path.append", "setattr", "delattr", "globals"}:
        kinds.append(SideEffectKind.MUTATION)
    if callee in {"exec", "eval", "compile", "__import__"}:
        kinds.append(SideEffectKind.UNKNOWN)
    if not kinds and callee in _EFFECTFUL_CALLEES:
        kinds.append(SideEffectKind.MUTATION)
    return tuple(kinds)


def _module_effects(tree: ast.Module) -> tuple[SideEffectKind, ...]:
    kinds: set[SideEffectKind] = set()
    for node in tree.body:
        if isinstance(node, ast.Raise):
            kinds.add(SideEffectKind.EXCEPTION)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            callee = _call_name(node.value.func)
            kinds.update(_effect_kinds_for(callee))
        elif isinstance(node, ast.Call):
            kinds.update(_effect_kinds_for(_call_name(node.func)))
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            kinds.update(_effect_kinds_for(_call_name(node.value.func)))
    if not kinds:
        return (SideEffectKind.NONE,)
    return tuple(sorted(kinds, key=lambda item: item.value))


def scan_sources_without_import(
    sources: Mapping[str, str],
    *,
    repository_tree: str,
    freshness: str = DEFAULT_FRESHNESS,
    extractor_identity: str = EXTRACTOR_IDENTITY,
) -> tuple[
    tuple[DynamicReachabilityRecord, ...],
    tuple[SideEffectScan, ...],
    dict[str, set[str]],
]:
    """Parse declared text. Never import or execute inspected modules."""

    if not isinstance(sources, Mapping) or isinstance(sources, (str, bytes, bytearray)):
        raise LegacyPathError("sources must be an object mapping paths to text")
    dynamic: list[DynamicReachabilityRecord] = []
    scans: list[SideEffectScan] = []
    imports: dict[str, set[str]] = {}
    for raw_path, raw_text in sources.items():
        path = _require_text(raw_path, "source path", error_type=LegacyPathError)
        if type(raw_text) is not str:
            raise LegacyPathError(f"source text must be a string: {path}")
        if not path.endswith(".py"):
            continue
        try:
            tree = ast.parse(raw_text)
        except SyntaxError as exc:
            raise LegacyPathError(f"source is not parseable: {path}") from exc
        effects = _module_effects(tree)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module:
                    imported.add(module)
                for alias in node.names:
                    imported.add(
                        f"{module}.{alias.name}" if module else alias.name
                    )
            elif isinstance(node, ast.Call):
                callee = _call_name(node.func)
                mechanism = _mechanism_for_callee(callee)
                if mechanism is None:
                    continue
                start = int(getattr(node, "lineno", 1) or 1)
                end = int(getattr(node, "end_lineno", start) or start)
                dynamic.append(
                    DynamicReachabilityRecord(
                        path=path,
                        symbol=callee,
                        mechanism=mechanism,
                        provenance=_fact(
                            path,
                            start,
                            end,
                            repository_tree=repository_tree,
                            freshness=freshness,
                            extractor_identity=extractor_identity,
                            confidence=Confidence.OPAQUE,
                        ),
                        uncertainty="dynamic_loading_target_not_statically_bound",
                    )
                )
            elif isinstance(node, ast.ClassDef) and any(
                base_name.endswith("MetaPathFinder") or base_name.endswith("Finder")
                for base in node.bases
                for base_name in (_call_name(base),)
            ):
                start = int(getattr(node, "lineno", 1) or 1)
                end = int(getattr(node, "end_lineno", start) or start)
                dynamic.append(
                    DynamicReachabilityRecord(
                        path=path,
                        symbol=node.name,
                        mechanism=DynamicMechanism.META_PATH_FINDER,
                        provenance=_fact(
                            path,
                            start,
                            end,
                            repository_tree=repository_tree,
                            freshness=freshness,
                            extractor_identity=extractor_identity,
                            confidence=Confidence.OPAQUE,
                        ),
                        uncertainty="meta_path_finder_resolves_names_at_import_time",
                    )
                )
        scans.append(
            SideEffectScan(
                path=path,
                effects=effects,
                imported_modules=tuple(imported),
                side_effect_free=effects == (SideEffectKind.NONE,) or effects == (),
            )
        )
        imports[path] = imported
    return (
        tuple(sorted(dynamic, key=_dynamic_sort_key)),
        tuple(sorted(scans, key=_scan_sort_key)),
        imports,
    )


def _entrypoint_kind(path: str, declared: Mapping[str, str] | None) -> str:
    if declared and path in declared:
        return declared[path]
    if _is_test_path(path):
        return "test"
    if _is_compat_path(path):
        return "compatibility"
    return "production"


def _module_name_from_path(path: str) -> str:
    text = _normalize_path(path)
    if text.endswith(".py"):
        text = text[:-3]
    parts = [part for part in text.split("/") if part and part != "__init__"]
    return ".".join(parts) if parts else path


def _defined_symbols(text: str) -> set[str]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


def trace_entrypoint_reachability(
    sources: Mapping[str, str],
    *,
    repository_tree: str,
    freshness: str = DEFAULT_FRESHNESS,
    entrypoints: Sequence[str] | None = None,
    entrypoint_kinds: Mapping[str, str] | None = None,
    origin_by_symbol: Mapping[str, OriginTaint] | None = None,
    dead_when_unreferenced: bool = False,
    extractor_identity: str = EXTRACTOR_IDENTITY,
) -> tuple[ReachabilityTrace, ...]:
    """Statically trace entrypoints through imports. Dynamic loaders stay unknown."""

    dynamic, _scans, imports = scan_sources_without_import(
        sources,
        repository_tree=repository_tree,
        freshness=freshness,
        extractor_identity=extractor_identity,
    )
    dynamic_paths = {item.path for item in dynamic}
    roots = tuple(entrypoints or ())
    if not roots:
        roots = tuple(
            path
            for path in sources
            if path.endswith(".py")
            and (
                path.endswith("cli.py")
                or path.endswith("__main__.py")
                or "main" in _defined_symbols(sources[path])
            )
        )
    reachable: dict[str, list[tuple[str, tuple[str, ...]]]] = {}
    queue: list[tuple[str, str, tuple[str, ...]]] = []
    seen_edges: set[tuple[str, str]] = set()
    for root in roots:
        queue.append((root, root, (root,)))
        reachable.setdefault(root, []).append((root, (root,)))
    while queue:
        current, root, hops = queue.pop()
        for imported in imports.get(current, ()):
            for candidate, text in sources.items():
                module = _module_name_from_path(candidate)
                symbols = _defined_symbols(text)
                leaf = imported.rsplit(".", 1)[-1]
                if not (
                    imported == module
                    or imported.startswith(f"{module}.")
                    or leaf in symbols
                    or candidate.replace("/", ".").endswith(f"{imported}.py")
                    or candidate.replace("/", ".").endswith(imported)
                ):
                    continue
                edge = (root, candidate)
                if edge in seen_edges:
                    continue
                seen_edges.add(edge)
                next_hops = hops + (candidate,)
                reachable.setdefault(candidate, []).append((root, next_hops))
                queue.append((candidate, root, next_hops))
    traces: list[ReachabilityTrace] = []
    origins = origin_by_symbol or {}
    for path, text in sources.items():
        if not path.endswith(".py"):
            continue
        symbols = _defined_symbols(text) or {_module_name_from_path(path)}
        arrivals = reachable.get(path, [])
        static_hit = bool(arrivals)
        kinds = {
            _entrypoint_kind(root, entrypoint_kinds) for root, _hops in arrivals
        }
        production_root = next(
            (
                (root, hops)
                for root, hops in arrivals
                if _entrypoint_kind(root, entrypoint_kinds) == "production"
            ),
            arrivals[0] if arrivals else ("", ()),
        )
        root, hops = production_root
        dynamic_here = tuple(item for item in dynamic if item.path == path)
        disposition = classify_reachability(
            static_from_production=static_hit and "production" in kinds,
            static_from_test=static_hit and "test" in kinds,
            static_from_compatibility=static_hit and "compatibility" in kinds,
            dynamic_records=dynamic_here if path in dynamic_paths else (),
            unreferenced=not static_hit,
            dead_when_unreferenced=dead_when_unreferenced and path not in dynamic_paths,
        )
        if path in dynamic_paths:
            disposition = ReachabilityDisposition.UNKNOWN
            method = ReachabilityMethod.DYNAMIC_UNKNOWN
        elif static_hit:
            method = (
                ReachabilityMethod.ENTRYPOINT
                if path == root
                else ReachabilityMethod.STATIC_IMPORT
            )
        elif disposition is ReachabilityDisposition.DEAD:
            method = ReachabilityMethod.UNREFERENCED_NO_DYNAMIC
        else:
            method = ReachabilityMethod.UNKNOWN
        for symbol in sorted(symbols):
            origin = origins.get(symbol) or origins.get(f"{path}:{symbol}")
            if origin is None:
                if _text_suggests_simulation(symbol) or _text_suggests_simulation(text[:400]):
                    origin = OriginTaint.SIMULATION
                elif _is_compat_path(path):
                    origin = OriginTaint.COMPATIBILITY
                elif _is_test_path(path):
                    origin = OriginTaint.FIXTURE
                else:
                    origin = OriginTaint.PRODUCTION
            traces.append(
                ReachabilityTrace(
                    entrypoint=root or path,
                    target=f"{_module_name_from_path(path)}.{symbol}",
                    disposition=disposition,
                    origin_taint=origin,
                    method=method,
                    hops=hops,
                )
            )
    return tuple(sorted(traces, key=_trace_sort_key))


def preserve_origin_taint(
    source: OriginTaint | str,
    *downstream: OriginTaint | str,
) -> OriginTaint:
    """Join origin taints so quarantined origins never become production."""

    parsed = [
        _closed_enum(item, OriginTaint, "origin taint", error_type=LegacyPathError)
        for item in (source, *downstream)
    ]
    joined = join_origin_taint(*parsed)
    if parsed[0] in QUARANTINED_ORIGINS and joined is OriginTaint.PRODUCTION:
        raise LegacyPathError("origin taint must be preserved")
    return joined


def detect_legacy_conflicts(
    paths: Sequence[LegacyPathRecord],
    dynamic_records: Sequence[DynamicReachabilityRecord] = (),
) -> tuple[LegacyConflict, ...]:
    """Fail closed on dead+dynamic, tainted authority, and missing kinds."""

    conflicts: list[LegacyConflict] = []
    covered = {item.kind for item in paths}
    missing = [item.value for item in REQUIRED_PATH_KINDS if item not in covered]
    if missing:
        conflicts.append(
            LegacyConflict(
                kind=LegacyConflictKind.MISSING_REQUIRED_PATH_KIND,
                message=f"missing required path kinds: {missing}",
            )
        )
    dynamic_paths = {item.path for item in dynamic_records}
    seen: dict[str, LegacyPathRecord] = {}
    for item in paths:
        if not item.provenance.span.path:
            conflicts.append(
                LegacyConflict(
                    kind=LegacyConflictKind.MISSING_SOURCE_IDENTITY,
                    message=f"{item.path_id} is missing source identity",
                    path_ids=(item.path_id,),
                )
            )
        if item.production_authority:
            conflicts.append(
                LegacyConflict(
                    kind=LegacyConflictKind.TAINTED_PRODUCTION_AUTHORITY,
                    message=f"{item.path_id} cannot grant production authority",
                    path_ids=(item.path_id,),
                )
            )
        if item.origin_taint in QUARANTINED_ORIGINS and item.may_satisfy_production_predicate:
            conflicts.append(
                LegacyConflict(
                    kind=LegacyConflictKind.TAINTED_PRODUCTION_AUTHORITY,
                    message=f"{item.path_id} tainted origin cannot satisfy production",
                    path_ids=(item.path_id,),
                )
            )
        if item.reachability is ReachabilityDisposition.DEAD and (
            item.dynamic_mechanisms or item.path in dynamic_paths
        ):
            conflicts.append(
                LegacyConflict(
                    kind=LegacyConflictKind.DEAD_WITH_DYNAMIC_LOADING,
                    message=f"{item.path_id} cannot be dead under dynamic loading",
                    path_ids=(item.path_id,),
                )
            )
        if (
            item.reachability is ReachabilityDisposition.DEAD
            and item.provenance.confidence in NON_PROBATIVE_CONFIDENCE
        ):
            conflicts.append(
                LegacyConflict(
                    kind=LegacyConflictKind.NON_PROBATIVE_DEAD,
                    message=f"{item.path_id} heuristic facts cannot prove dead code",
                    path_ids=(item.path_id,),
                )
            )
        if (
            item.reachability is ReachabilityDisposition.UNKNOWN
            and item.dynamic_mechanisms
            and not item.uncertainty
        ):
            conflicts.append(
                LegacyConflict(
                    kind=LegacyConflictKind.UNKNOWN_DYNAMIC_UNCLASSIFIED,
                    message=f"{item.path_id} unknown dynamic path must declare uncertainty",
                    path_ids=(item.path_id,),
                )
            )
        existing = seen.get(item.path_id)
        if existing is not None and existing.reachability is not item.reachability:
            conflicts.append(
                LegacyConflict(
                    kind=LegacyConflictKind.CONFLICTING_REACHABILITY,
                    message=f"{item.path_id} has conflicting reachability",
                    path_ids=(item.path_id,),
                )
            )
        seen[item.path_id] = item
    return tuple(sorted(conflicts, key=_conflict_sort_key))


def classify_architecture_legacy(
    graph: ArchitectureIR | None,
) -> tuple[LegacyPathRecord, ...]:
    """Select COMPATIBILITY and SIMULATION nodes as inventoried paths."""

    if graph is None:
        return ()
    records: list[LegacyPathRecord] = []
    for index, node in enumerate(graph.nodes):
        if node.kind is NodeKind.SIMULATION:
            kind = PathKind.SIMULATED_HARDWARE
            origin = OriginTaint.SIMULATION
            reachability = ReachabilityDisposition.UNKNOWN
            uncertainty = "architecture_ir_simulation_node_is_not_production_authority"
        elif node.kind is NodeKind.COMPATIBILITY:
            kind = PathKind.HISTORICAL_PROVIDER_ROUTERS
            origin = OriginTaint.COMPATIBILITY
            reachability = ReachabilityDisposition.COMPATIBILITY_ONLY
            uncertainty = "architecture_ir_compatibility_node_is_not_canonical_authority"
        else:
            continue
        identity = node.node_id
        symbol = identity.rsplit(":", 1)[-1]
        records.append(
            LegacyPathRecord(
                path_id=f"ir-{index}-{symbol}",
                kind=kind,
                path=node.provenance.span.path,
                symbol=symbol,
                origin_taint=origin,
                reachability=reachability,
                provenance=node.provenance,
                uncertainty=uncertainty,
                dynamic_mechanisms=(
                    (DynamicMechanism.UNKNOWN,)
                    if reachability is ReachabilityDisposition.UNKNOWN
                    else ()
                ),
            )
        )
    fallback_targets = {
        edge.target
        for edge in graph.edges
        if edge.kind is EdgeKind.FALLBACKS_TO
    }
    for node in graph.nodes:
        if node.node_id not in fallback_targets:
            continue
        if node.kind in {NodeKind.SIMULATION, NodeKind.COMPATIBILITY}:
            continue
        records.append(
            LegacyPathRecord(
                path_id=f"ir-fallback-{node.node_id}",
                kind=PathKind.FALLBACK_SUCCESS_PATHS,
                path=node.provenance.span.path,
                symbol=node.node_id.rsplit(":", 1)[-1],
                origin_taint=OriginTaint.SIMULATION,
                reachability=ReachabilityDisposition.PRODUCTION_REACHABLE,
                provenance=node.provenance,
                uncertainty="fallbacks_to_edge_preserves_simulation_origin",
            )
        )
    return tuple(sorted(records, key=_path_sort_key))


def paths_from_inventory(
    inventory: Mapping[str, Any],
    *,
    repository_tree: str,
    freshness: str = DEFAULT_FRESHNESS,
    extractor_identity: str = EXTRACTOR_IDENTITY,
) -> tuple[LegacyPathRecord, ...]:
    """Load compact inventory path rows without treating them as authority."""

    mapping = _require_mapping(inventory, error_type=LegacyPathError)
    paths_payload = mapping.get("paths")
    if isinstance(paths_payload, (str, bytes, bytearray)) or not isinstance(
        paths_payload, Sequence
    ):
        raise LegacyPathError("inventory paths must be a sequence")
    records: list[LegacyPathRecord] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(paths_payload):
        row = _require_mapping(raw, error_type=LegacyPathError)
        extra = set(row) - _INVENTORY_PATH_FIELDS - _INVENTORY_OPTIONAL_FIELDS
        if extra:
            raise LegacyPathError(f"{_UNKNOWN_FIELD_MESSAGE}: {sorted(extra)}")
        missing = sorted(_INVENTORY_PATH_FIELDS - set(row))
        if missing:
            raise LegacyPathError(f"{_MISSING_FIELD_MESSAGE}: {missing}")
        path_id = _require_text(row["path_id"], "path_id", error_type=LegacyPathError)
        if path_id in seen_ids:
            raise LegacyPathError("inventory path ids must be unique")
        seen_ids.add(path_id)
        kind = _closed_enum(
            row["kind"], PathKind, "path kind", error_type=LegacyPathError
        )
        span_payload = row["source_span"]
        if not isinstance(span_payload, Mapping):
            raise LegacyPathError("source_span must be an object")
        span = SourceSpan.from_mapping(span_payload)
        origin = _closed_enum(
            row["origin_taint"],
            OriginTaint,
            "origin taint",
            error_type=LegacyPathError,
        )
        reachability = _closed_enum(
            row["reachability"],
            ReachabilityDisposition,
            "reachability",
            error_type=LegacyPathError,
        )
        uncertainty = row.get("uncertainty")
        mechanisms: tuple[DynamicMechanism, ...] = ()
        if reachability is ReachabilityDisposition.UNKNOWN or row.get(
            "dynamic_uncertainty"
        ):
            mechanism_value = row.get("mechanism")
            if mechanism_value:
                mechanisms = (
                    _closed_enum(
                        mechanism_value,
                        DynamicMechanism,
                        "dynamic mechanism",
                        error_type=LegacyPathError,
                    ),
                )
            else:
                mechanisms = (DynamicMechanism.UNKNOWN,)
        records.append(
            LegacyPathRecord(
                path_id=path_id,
                kind=kind,
                path=row["path"],
                symbol=row["nominated_symbol"],
                origin_taint=origin,
                reachability=reachability,
                provenance=SourceFactIdentity(
                    extractor_identity=extractor_identity,
                    span=span,
                    confidence=(
                        Confidence.OPAQUE
                        if reachability is ReachabilityDisposition.UNKNOWN
                        else Confidence.EXACT
                    ),
                    freshness=freshness,
                    repository_tree=repository_tree,
                ),
                production_authority=False,
                dynamic_mechanisms=mechanisms,
                uncertainty="" if uncertainty is None else str(uncertainty),
            )
        )
        if index >= 0 and row["production_authority"] is True:
            raise LegacyPathAuthorityError(
                "legacy-path inventory cannot grant production authority"
            )
    return tuple(sorted(records, key=_path_sort_key))


def compact_inventory_payload(
    paths: Sequence[LegacyPathRecord],
    *,
    repository_tree: str = SEALED_REPOSITORY_TREE,
) -> dict[str, Any]:
    """Compact observational inventory. This payload is not authority."""

    rows = []
    for item in sorted(paths, key=_path_sort_key):
        uncertainty: str | None = item.uncertainty or None
        rows.append(
            {
                "dynamic_uncertainty": bool(item.dynamic_mechanisms)
                or item.reachability is ReachabilityDisposition.UNKNOWN,
                "kind": item.kind.value,
                "nominated_symbol": item.symbol,
                "origin_taint": item.origin_taint.value,
                "path": item.path,
                "path_id": item.path_id,
                "present": True,
                "production_authority": False,
                "reachability": item.reachability.value,
                "source_span": item.provenance.span.to_dict(),
                "uncertainty": uncertainty,
            }
        )
    return {
        "authority": False,
        "closed_reachability": [item.value for item in REQUIRED_REACHABILITY],
        "dead_classification_policy": DEAD_CLASSIFICATION_POLICY,
        "inspection": {
            "method": "static_and_hermetic_dynamic_reachability",
            "nonclaim": (
                "static reachability alone never proves dead code where "
                "dynamic loading is possible"
            ),
        },
        "production_flow_invariant": PRODUCTION_FLOW_INVARIANT,
        "repository_tree": repository_tree,
        "required_categories": [item.value for item in REQUIRED_PATH_KINDS],
        "schema": COMPACT_INVENTORY_SCHEMA,
        "task_id": TASK_ID,
        "paths": rows,
    }


@dataclass(frozen=True)
class LegacyPathInventory:
    """Canonical inventory of legacy/simulation/fixture/compatibility paths."""

    repository_tree: str
    freshness: str
    paths: tuple[LegacyPathRecord, ...]
    dynamic_records: tuple[DynamicReachabilityRecord, ...] = ()
    taint_records: tuple[OriginTaintRecord, ...] = ()
    traces: tuple[ReachabilityTrace, ...] = ()
    side_effect_scans: tuple[SideEffectScan, ...] = ()
    conflicts: tuple[LegacyConflict, ...] = ()
    schema: str = LEGACY_INVENTORY_SCHEMA
    version: int = LEGACY_INVENTORY_VERSION
    effect_class: str = EFFECT_CLASS
    authority: bool = INVENTORY_AUTHORITY
    can_authorize_deletion: bool = INVENTORY_CAN_AUTHORIZE_DELETION
    can_promote_fake_to_live: bool = INVENTORY_CAN_PROMOTE_FAKE_TO_LIVE
    can_grant_production_authority: bool = INVENTORY_CAN_GRANT_PRODUCTION_AUTHORITY
    architecture_ir_identity: str = ""
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=LegacyPathError)
        if schema != LEGACY_INVENTORY_SCHEMA:
            raise LegacyPathError("unexpected legacy-inventory schema")
        version = _require_int(self.version, "version", error_type=LegacyPathError)
        if version != LEGACY_INVENTORY_VERSION:
            raise LegacyPathError("unexpected legacy-inventory version")
        repository_tree = _require_text(
            self.repository_tree, "repository_tree", error_type=LegacyPathError
        )
        freshness = _require_text(
            self.freshness, "freshness", error_type=LegacyPathError
        )
        effect_class = _require_text(
            self.effect_class, "effect_class", error_type=LegacyPathError
        )
        if effect_class != EFFECT_CLASS:
            raise LegacyPathError(
                "legacy-path inventory effect class is read_only_analysis"
            )
        if self.authority is not False:
            raise LegacyPathAuthorityError(
                "legacy-path inventory cannot grant production authority"
            )
        if self.can_authorize_deletion is not False:
            raise LegacyPathAuthorityError(
                "legacy-path inventory cannot authorize deletion"
            )
        if self.can_promote_fake_to_live is not False:
            raise LegacyPathAuthorityError(
                "legacy-path inventory cannot promote fake-to-live"
            )
        if self.can_grant_production_authority is not False:
            raise LegacyPathAuthorityError(
                "legacy-path inventory cannot grant production authority"
            )
        architecture_ir_identity = self.architecture_ir_identity
        if architecture_ir_identity:
            architecture_ir_identity = _validate_dag_json_cid(
                _require_text(
                    architecture_ir_identity,
                    "architecture_ir_identity",
                    error_type=LegacyPathError,
                )
            )
        else:
            architecture_ir_identity = ""
        paths = tuple(
            sorted(
                _record_tuple(self.paths, LegacyPathRecord, "paths"),
                key=_path_sort_key,
            )
        )
        path_ids = tuple(item.path_id for item in paths)
        if len(path_ids) != len(set(path_ids)):
            raise LegacyPathError("legacy path ids must be unique")
        for item in paths:
            if item.provenance.repository_tree != repository_tree:
                raise LegacyPathError(
                    "path provenance repository_tree must match the inventory"
                )
            if item.provenance.freshness != freshness:
                raise LegacyPathError(
                    "path provenance freshness must match the inventory"
                )
        dynamic_records = tuple(
            sorted(
                _record_tuple(
                    self.dynamic_records, DynamicReachabilityRecord, "dynamic_records"
                ),
                key=_dynamic_sort_key,
            )
        )
        taint_records = tuple(
            sorted(
                _record_tuple(self.taint_records, OriginTaintRecord, "taint_records"),
                key=_taint_sort_key,
            )
        )
        if not taint_records:
            taint_records = tuple(
                sorted((taint_record_for(item) for item in paths), key=_taint_sort_key)
            )
        traces = tuple(
            sorted(
                _record_tuple(self.traces, ReachabilityTrace, "traces"),
                key=_trace_sort_key,
            )
        )
        scans = tuple(
            sorted(
                _record_tuple(
                    self.side_effect_scans, SideEffectScan, "side_effect_scans"
                ),
                key=_scan_sort_key,
            )
        )
        declared = tuple(
            sorted(
                _record_tuple(self.conflicts, LegacyConflict, "conflicts"),
                key=_conflict_sort_key,
            )
        )
        detected = detect_legacy_conflicts(paths, dynamic_records)
        declared_keys = {
            (item.kind, item.message, item.path_ids) for item in declared
        }
        missing = [
            item
            for item in detected
            if (item.kind, item.message, item.path_ids) not in declared_keys
        ]
        if missing:
            raise LegacyPathError(
                "legacy-path conflicts must include detected hard blockers"
            )
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "repository_tree", repository_tree)
        object.__setattr__(self, "freshness", freshness)
        object.__setattr__(self, "effect_class", effect_class)
        object.__setattr__(self, "authority", False)
        object.__setattr__(self, "can_authorize_deletion", False)
        object.__setattr__(self, "can_promote_fake_to_live", False)
        object.__setattr__(self, "can_grant_production_authority", False)
        object.__setattr__(self, "architecture_ir_identity", architecture_ir_identity)
        object.__setattr__(self, "paths", paths)
        object.__setattr__(self, "dynamic_records", dynamic_records)
        object.__setattr__(self, "taint_records", taint_records)
        object.__setattr__(self, "traces", traces)
        object.__setattr__(self, "side_effect_scans", scans)
        object.__setattr__(self, "conflicts", declared)
        identity = _content_identity(self._identity_payload())
        if self.content_identity:
            claimed = _validate_dag_json_cid(
                _require_text(
                    self.content_identity,
                    "content_identity",
                    error_type=LegacyPathError,
                )
            )
            if claimed != identity:
                raise LegacyPathError("legacy-inventory content identity mismatch")
        object.__setattr__(self, "content_identity", identity)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "architecture_ir_identity": self.architecture_ir_identity,
            "authority": False,
            "can_authorize_deletion": False,
            "can_grant_production_authority": False,
            "can_promote_fake_to_live": False,
            "conflicts": [item.to_dict() for item in self.conflicts],
            "dynamic_records": [item.to_dict() for item in self.dynamic_records],
            "effect_class": self.effect_class,
            "freshness": self.freshness,
            "paths": [item.to_dict() for item in self.paths],
            "repository_tree": self.repository_tree,
            "schema": self.schema,
            "side_effect_scans": [item.to_dict() for item in self.side_effect_scans],
            "taint_records": [item.to_dict() for item in self.taint_records],
            "traces": [item.to_dict() for item in self.traces],
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise LegacyPathError("legacy-inventory content identity mismatch")
        return {**payload, "content_identity": identity}

    def to_json(self) -> str:
        return canonical_dag_json_bytes(self.to_dict()).decode("utf-8")

    def compact_dict(self) -> dict[str, Any]:
        return compact_inventory_payload(
            self.paths, repository_tree=self.repository_tree
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "LegacyPathInventory":
        mapping = _require_mapping(payload, error_type=LegacyPathError)
        _require_fields(mapping, _INVENTORY_FIELDS)
        try:
            inventory = cls(
                repository_tree=mapping["repository_tree"],
                freshness=mapping["freshness"],
                paths=mapping["paths"],
                dynamic_records=mapping["dynamic_records"],
                taint_records=mapping["taint_records"],
                traces=mapping["traces"],
                side_effect_scans=mapping["side_effect_scans"],
                conflicts=mapping["conflicts"],
                schema=mapping["schema"],
                version=mapping["version"],
                effect_class=mapping["effect_class"],
                authority=mapping["authority"],
                can_authorize_deletion=mapping["can_authorize_deletion"],
                can_promote_fake_to_live=mapping["can_promote_fake_to_live"],
                can_grant_production_authority=mapping[
                    "can_grant_production_authority"
                ],
                architecture_ir_identity=mapping["architecture_ir_identity"],
            )
        except ArchitectureContractError as exc:
            raise _wrap_contract(exc) from exc
        if mapping["content_identity"] != inventory.content_identity:
            raise LegacyPathError("legacy-inventory content identity mismatch")
        return inventory

    from_dict = from_mapping

    @classmethod
    def from_json(cls, text: str) -> "LegacyPathInventory":
        payload = json.loads(text)
        if not isinstance(payload, Mapping):
            raise LegacyPathError("legacy-inventory JSON must be an object")
        return cls.from_mapping(payload)

    @property
    def covers_required_path_kinds(self) -> bool:
        return {item.kind for item in self.paths} >= set(REQUIRED_PATH_KINDS)

    @property
    def covers_required_reachability(self) -> bool:
        observed = {item.reachability for item in self.paths}
        observed.update(item.disposition for item in self.traces)
        return observed >= {
            ReachabilityDisposition.PRODUCTION_REACHABLE,
            ReachabilityDisposition.TEST_ONLY,
            ReachabilityDisposition.COMPATIBILITY_ONLY,
            ReachabilityDisposition.UNKNOWN,
        }

    @property
    def unknown_dynamic_count(self) -> int:
        return sum(
            1
            for item in self.paths
            if item.reachability is ReachabilityDisposition.UNKNOWN
        )

    @property
    def fails_closed(self) -> bool:
        return bool(self.conflicts)

    def paths_for(self, kind: PathKind | str) -> tuple[LegacyPathRecord, ...]:
        parsed = _closed_enum(
            kind, PathKind, "path kind", error_type=LegacyPathError
        )
        return tuple(item for item in self.paths if item.kind is parsed)

    def path_for(self, path_id: str) -> LegacyPathRecord:
        ident = _require_text(path_id, "path_id", error_type=LegacyPathError)
        for item in self.paths:
            if item.path_id == ident:
                return item
        raise LegacyPathError(f"unknown legacy path id: {ident}")

    def authorize_deletion(self, path_id: str) -> None:
        refuse_deletion(path_id)

    def promote_fake_to_live(self, path_id: str) -> None:
        refuse_fake_to_live_promotion(path_id)

    def grant_production_authority(self, path_id: str) -> None:
        refuse_production_authority(path_id)


def build_legacy_path_inventory(
    paths: Sequence[LegacyPathRecord | Mapping[str, Any]] | None = None,
    *,
    inventory: Mapping[str, Any] | None = None,
    sources: Mapping[str, str] | None = None,
    architecture: ArchitectureIR | Mapping[str, Any] | None = None,
    entrypoints: Sequence[str] | None = None,
    entrypoint_kinds: Mapping[str, str] | None = None,
    origin_by_symbol: Mapping[str, OriginTaint] | None = None,
    dead_when_unreferenced: bool = False,
    extra_conflicts: Sequence[LegacyConflict | Mapping[str, Any]] = (),
    repository_tree: str,
    freshness: str = DEFAULT_FRESHNESS,
    extractor_identity: str = EXTRACTOR_IDENTITY,
) -> LegacyPathInventory:
    """Inventory required path types with explicit reachability and origin taint."""

    graph: ArchitectureIR | None
    if architecture is None:
        graph = None
    elif isinstance(architecture, ArchitectureIR):
        graph = architecture
    else:
        try:
            graph = ArchitectureIR.from_mapping(architecture)
        except ArchitectureContractError as exc:
            raise _wrap_contract(exc) from exc
    if graph is not None:
        if graph.repository_tree != repository_tree:
            raise LegacyPathError(
                "ArchitectureIR repository_tree must match the legacy inventory"
            )
        if graph.freshness != freshness:
            raise LegacyPathError(
                "ArchitectureIR freshness must match the legacy inventory"
            )
    dynamic: tuple[DynamicReachabilityRecord, ...] = ()
    scans: tuple[SideEffectScan, ...] = ()
    traces: tuple[ReachabilityTrace, ...] = ()
    if sources is not None:
        dynamic, scans, _imports = scan_sources_without_import(
            sources,
            repository_tree=repository_tree,
            freshness=freshness,
            extractor_identity=extractor_identity,
        )
        traces = trace_entrypoint_reachability(
            sources,
            repository_tree=repository_tree,
            freshness=freshness,
            entrypoints=entrypoints,
            entrypoint_kinds=entrypoint_kinds,
            origin_by_symbol=origin_by_symbol,
            dead_when_unreferenced=dead_when_unreferenced,
            extractor_identity=extractor_identity,
        )
    if inventory is not None:
        parsed_paths = paths_from_inventory(
            inventory,
            repository_tree=repository_tree,
            freshness=freshness,
            extractor_identity=extractor_identity,
        )
    elif paths is None:
        parsed_paths = current_legacy_paths(
            repository_tree=repository_tree,
            freshness=freshness,
            extractor_identity=extractor_identity,
        )
        if not dynamic:
            dynamic = current_dynamic_records(
                repository_tree=repository_tree,
                freshness=freshness,
                extractor_identity=extractor_identity,
            )
    else:
        parsed_paths = tuple(
            classify_legacy_path(
                item,
                repository_tree=repository_tree,
                freshness=freshness,
                extractor_identity=extractor_identity,
            )
            for item in paths
        )
    ir_paths = classify_architecture_legacy(graph)
    merged_paths = tuple(sorted((*parsed_paths, *ir_paths), key=_path_sort_key))
    extra = tuple(
        item if isinstance(item, LegacyConflict) else LegacyConflict.from_mapping(item)
        for item in extra_conflicts
    )
    detected = detect_legacy_conflicts(merged_paths, dynamic)
    merged_conflicts: dict[tuple[Any, ...], LegacyConflict] = {}
    for item in (*detected, *extra):
        merged_conflicts[(item.kind, item.message, item.path_ids)] = item
    taint_records = tuple(
        sorted((taint_record_for(item) for item in merged_paths), key=_taint_sort_key)
    )
    return LegacyPathInventory(
        repository_tree=repository_tree,
        freshness=freshness,
        paths=merged_paths,
        dynamic_records=dynamic,
        taint_records=taint_records,
        traces=traces,
        side_effect_scans=scans,
        conflicts=tuple(sorted(merged_conflicts.values(), key=_conflict_sort_key)),
        architecture_ir_identity="" if graph is None else graph.content_identity,
    )


build_current_legacy_path_inventory = build_legacy_path_inventory


__all__ = [
    "CLOSED_CONFLICT_KINDS",
    "CLOSED_DYNAMIC_MECHANISMS",
    "CLOSED_ORIGIN_TAINTS",
    "CLOSED_PATH_KINDS",
    "CLOSED_PRODUCTION_PREDICATES",
    "CLOSED_REACHABILITY",
    "CLOSED_REACHABILITY_METHODS",
    "CLOSED_SIDE_EFFECTS",
    "COMPACT_INVENTORY_SCHEMA",
    "CONTENT_IDENTITY_IS_NOT_AUTHORITY",
    "CURRENT_LEGACY_BINDINGS",
    "DEAD_CLASSIFICATION_POLICY",
    "DEFAULT_FRESHNESS",
    "DYNAMIC_RECORD_SCHEMA",
    "DYNAMIC_RECORD_VERSION",
    "DYNAMIC_UNCERTAINTY_BLOCKS_DEAD",
    "EFFECT_CLASS",
    "EXTRACTOR_IDENTITY",
    "INVENTORY_AUTHORITY",
    "INVENTORY_CAN_AUTHORIZE_DELETION",
    "INVENTORY_CAN_GRANT_PRODUCTION_AUTHORITY",
    "INVENTORY_CAN_PROMOTE_FAKE_TO_LIVE",
    "LEGACY_INVENTORY_EVIDENCE",
    "LEGACY_INVENTORY_SCHEMA",
    "LEGACY_INVENTORY_VERSION",
    "LEGACY_PATH_SCHEMA",
    "LEGACY_PATH_VERSION",
    "PRODUCTION_FLOW_INVARIANT",
    "QUARANTINED_ORIGINS",
    "REQUIRED_PATH_KINDS",
    "REQUIRED_PRODUCTION_PREDICATES",
    "REQUIRED_REACHABILITY",
    "SEALED_REPOSITORY_TREE",
    "STATIC_REACHABILITY_PROVES_DEAD",
    "TAINTED_ORIGINS",
    "TAINT_RECORD_SCHEMA",
    "TASK_ID",
    "TRACE_SCHEMA",
    "DynamicMechanism",
    "DynamicReachabilityRecord",
    "LegacyConflict",
    "LegacyConflictKind",
    "LegacyPathAuthorityError",
    "LegacyPathError",
    "LegacyPathInventory",
    "LegacyPathRecord",
    "LegacySourceBinding",
    "OriginTaint",
    "OriginTaintRecord",
    "PathKind",
    "ProductionPredicate",
    "ReachabilityDisposition",
    "ReachabilityMethod",
    "ReachabilityTrace",
    "SideEffectKind",
    "SideEffectScan",
    "blocked_production_predicates",
    "build_current_legacy_path_inventory",
    "build_legacy_path_inventory",
    "classify_architecture_legacy",
    "classify_legacy_path",
    "classify_reachability",
    "compact_inventory_payload",
    "current_dynamic_records",
    "current_legacy_paths",
    "detect_legacy_conflicts",
    "join_origin_taint",
    "origin_may_satisfy_production_predicate",
    "paths_from_inventory",
    "preserve_origin_taint",
    "refuse_dead_from_static_only",
    "refuse_deletion",
    "refuse_fake_to_live_promotion",
    "refuse_production_authority",
    "scan_sources_without_import",
    "taint_record_for",
    "trace_entrypoint_reachability",
]
