"""PCTDD-031: fixture- and proof-aware xdist scheduling.

Accelerate-owned placement over the existing controller/worker xdist
coordination.  Session/module/package/class fixture affinity and integer
proof cost improve worker assignment.  Placement never omits tests, never
merges resource pools, never authorizes pytest skip, and never lets a
worker publish.

Unknown or incomplete fixture semantics drop only the affinity hint and
still place every test.  Production ZK, key ceremony, and direct-execution
profiles remain typed unavailable.

Import is cold-safe: no pytest, network, package installer, or prover.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final

from .xdist_reuse_coordination import XDIST_REUSE_COORDINATION_INTERFACE

# This module is accelerate integration, not a pytest test module.
__test__ = False

FIXTURE_PROOF_AWARE_XDIST_INTERFACE: Final = "FixtureProofAwareXdistScheduling@1"
FIXTURE_PROOF_AWARE_XDIST_RESULT_INTERFACE: Final = (
    "FixtureProofAwareXdistPlacement@1"
)
FIXTURE_PROOF_AWARE_XDIST_POLICY_INTERFACE: Final = (
    "FixtureProofAwareXdistPolicy@1"
)
FIXTURE_PROOF_AWARE_XDIST_SCHEMA: Final = (
    "ipfs_accelerate_py/testing/proof-reuse/fixture-proof-aware-xdist@1"
)
CLAIM_CLASS: Final = "IntegrityCommitment"
SCHEDULING_AUTHORITY: Final = (
    "ipfs_accelerate_py.testing.proof_reuse.fixture_proof_aware_xdist"
)
PREDECESSOR_INTERFACE: Final = XDIST_REUSE_COORDINATION_INTERFACE
ITEM_SCHEDULING_UNIT_ATTRIBUTE: Final = "_ipfs_proof_reuse_scheduling_unit"
ITEM_PLACEMENT_WORKER_ATTRIBUTE: Final = "_ipfs_proof_reuse_xdist_worker"
RESOURCE_POOL_MARKER: Final = "proof_reuse_resource_pool"
PROOF_COST_MARKER: Final = "proof_reuse_proof_cost"
ITEM_RESOURCE_POOL_ATTRIBUTE: Final = "_ipfs_proof_reuse_resource_pool"
ITEM_PROOF_COST_ATTRIBUTE: Final = "_ipfs_proof_reuse_proof_cost"
ITEM_AFFINITY_FIXTURES_ATTRIBUTE: Final = "_ipfs_proof_reuse_affinity_fixtures"
DEFAULT_RESOURCE_POOL: Final = "cpu"
UNASSIGNED_WORKER_PREFIX: Final = "unassigned:"
MAX_TEXT_CHARS: Final = 4_096
MAX_NODE_ID_CHARS: Final = 2_048
MAX_PROOF_COST: Final = 1_000_000_000
DEFAULT_PROOF_COST: Final = 1
_DIGEST_PREFIX: Final = "sha256:"

CLOSED_RESOURCE_POOLS: Final[tuple[str, ...]] = (
    "cpu",
    "gpu",
    "prover",
    "hash",
    "store",
)
AFFINITY_SCOPES: Final[frozenset[str]] = frozenset(
    {"session", "package", "module", "class"}
)
FUNCTION_SCOPE: Final = "function"

SCHEDULING_ESTABLISHES: Final = (
    "fixture affinity and proof cost improve placement without omitting "
    "tests or merging resource pools"
)
SCHEDULING_DOES_NOT: Final = (
    "execution or semantics; skip; omitting tests; merging resource pools; "
    "worker publication; current-root publication; production ZK; "
    "self-approval"
)

_PRIVATE_SUBSTRINGS: Final[tuple[str, ...]] = (
    "api_key",
    "authorization",
    "cookie",
    "credential",
    "password",
    "private",
    "proving_key",
    "secret",
    "session",
    "signing_key",
    "token",
    "witness",
)

_TYPED_UNAVAILABLE: Final[tuple[tuple[str, str, str], ...]] = (
    (
        "aggregate_selected_test_zk",
        "aggregate_selected_test_zk_missing",
        "aggregate selected-test ZK remains a versioned successor; fixture-"
        "proof-aware placement cannot upgrade leaf TestPassStatementV1 claims",
    ),
    (
        "production_zk",
        "production_zk_key_ceremony_unavailable",
        "production ZK proving remains typed unavailable; fixture-proof-aware "
        "xdist scheduling cannot admit simulated, structural, or self-verified "
        "proofs",
    ),
    (
        "key_ceremony",
        "production_zk_key_ceremony_unavailable",
        "no production-eligible key ceremony is admitted by fixture-proof-"
        "aware xdist scheduling",
    ),
    (
        "direct_execution_profile",
        "direct_execution_profile_optional",
        "direct CPython execution profiles remain optional and unadmitted; "
        "they cannot upgrade fixture-proof-aware placement integrity "
        "commitments",
    ),
)


class FixtureProofAwareXdistError(ValueError):
    """Raised when fixture-proof-aware xdist scheduling is unsafe."""

    __test__ = False


def _is_private_key(key: str) -> bool:
    lowered = key.lower().replace("-", "_")
    return any(marker in lowered for marker in _PRIVATE_SUBSTRINGS)


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        raise FixtureProofAwareXdistError(
            "floating-point values are not JSON-safe for xdist scheduling"
        )
    if isinstance(value, Mapping):
        ready: dict[str, Any] = {}
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            name = str(key)
            if _is_private_key(name):
                raise FixtureProofAwareXdistError(
                    f"xdist scheduling rejects private material key {name!r}"
                )
            ready[name] = _json_ready(item)
        return ready
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise FixtureProofAwareXdistError(
            "xdist scheduling rejects secret or raw bytes"
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_ready(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_ready(to_dict())
    raise FixtureProofAwareXdistError(
        f"value of type {type(value).__name__} is not JSON-serializable"
    )


def canonical_public_bytes(value: Any) -> bytes:
    """Return canonical JSON bytes for a privacy-safe public payload."""

    return json.dumps(
        _json_ready(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def public_digest(value: Any) -> str:
    """Return ``sha256:<hex>`` of canonical public bytes."""

    return _DIGEST_PREFIX + hashlib.sha256(canonical_public_bytes(value)).hexdigest()


SCHEDULING_POLICY: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "interface": FIXTURE_PROOF_AWARE_XDIST_POLICY_INTERFACE,
        "scheduling_interface": FIXTURE_PROOF_AWARE_XDIST_INTERFACE,
        "predecessor_interface": PREDECESSOR_INTERFACE,
        "may_authorize_skip": False,
        "may_omit_tests": False,
        "may_merge_resource_pools": False,
        "workers_may_publish": False,
        "controller_owns_writes": True,
        "production_admitted": False,
        "self_approved": False,
        "normal_execution_fallback": True,
        "closed_resource_pools": list(CLOSED_RESOURCE_POOLS),
        "affinity_scopes": sorted(AFFINITY_SCOPES),
        "schema_authority": SCHEDULING_AUTHORITY,
    }
)
DEFAULT_POLICY_CID: Final = public_digest(dict(SCHEDULING_POLICY))


def record_typed_unavailable(
    *,
    capability: str,
    reason_code: str,
    message: str,
) -> dict[str, Any]:
    """Record a typed unavailable case without changing claim meaning."""

    record = {
        "capability": capability,
        "reason_code": reason_code,
        "message": message,
        "status": "typed_unavailable",
        "production_admitted": False,
        "claim_unchanged": True,
        "self_approved": False,
    }
    if record["production_admitted"] or record["self_approved"] or not record["claim_unchanged"]:
        raise FixtureProofAwareXdistError(
            "typed unavailable cases cannot admit, self-approve, or change claims"
        )
    return record


def pytest_xdist_runtime_available() -> bool:
    """Return whether pytest-xdist imports in this process.

    Availability is judged by a sealed import, never by a provider-side PATH
    probe.  Missing xdist does not omit tests or weaken placement invariants.
    """

    try:
        import importlib.util

        return importlib.util.find_spec("xdist") is not None
    except Exception:
        return False


def typed_unavailable_records() -> tuple[dict[str, Any], ...]:
    """Closed set of PCTDD-031 typed unavailable capabilities."""

    records = [
        record_typed_unavailable(
            capability=capability,
            reason_code=reason_code,
            message=message,
        )
        for capability, reason_code, message in _TYPED_UNAVAILABLE
    ]
    if not pytest_xdist_runtime_available():
        records.append(
            record_typed_unavailable(
                capability="pytest_xdist_plugin_runtime",
                reason_code="pytest_xdist_runtime_unresolved",
                message=(
                    "pytest-xdist is not importable in this sealed PATH; "
                    "fixture-affinity and proof-cost placement still run as "
                    "controller-owned scheduling and are not admitted as "
                    "production"
                ),
            )
        )
    return tuple(records)


def authority_descriptor() -> dict[str, Any]:
    """Return the scheduling authority without widening others."""

    return {
        "canonical_semantic_and_statement_authority": "ipfs_datasets_py",
        "execution_scheduling_admission_authority": "ipfs_accelerate_py",
        "verified_storage_wal_cas_authority": "ipfs_kit_py",
        "scheduling": SCHEDULING_AUTHORITY,
        "predecessor_interface": PREDECESSOR_INTERFACE,
        "does_not": SCHEDULING_DOES_NOT,
        "establishes": SCHEDULING_ESTABLISHES,
        "claim_class": CLAIM_CLASS,
        "may_authorize_skip": False,
        "may_omit_tests": False,
        "may_merge_resource_pools": False,
        "production_admitted": False,
        "self_approved": False,
        "worker_authored_test_is_sufficient_alone": False,
        "workers_may_publish": False,
        "controller_owns_writes": True,
        "normal_execution_fallback": True,
        "scheduling_interface": FIXTURE_PROOF_AWARE_XDIST_INTERFACE,
        "closed_resource_pools": list(CLOSED_RESOURCE_POOLS),
        "affinity_scopes": sorted(AFFINITY_SCOPES),
    }


def workers_may_publish() -> bool:
    """Workers never publish accepted reuse evidence."""

    return False


def _bounded_text(value: Any, *, max_chars: int = MAX_TEXT_CHARS) -> str:
    text = "" if value is None else str(value)
    if len(text) > max_chars:
        return text[:max_chars]
    return text


def _bounded_nodeid(value: Any) -> str:
    text = _bounded_text(value, max_chars=MAX_NODE_ID_CHARS).strip()
    if any(ord(character) < 32 for character in text):
        raise FixtureProofAwareXdistError("nodeid contains control characters")
    return text


def _normalize_resource_pool(value: Any) -> str:
    text = _bounded_text(value, max_chars=32).strip().lower().replace(" ", "-")
    if not text:
        return DEFAULT_RESOURCE_POOL
    if text in CLOSED_RESOURCE_POOLS:
        return text
    if text.startswith("cpu-"):
        return "cpu"
    if (
        len(text) <= 32
        and text[0].isalpha()
        and all(character.isalnum() or character in "-_" for character in text)
    ):
        # Distinct unknown pools stay distinct so they cannot merge with cpu.
        return text
    raise FixtureProofAwareXdistError(
        f"resource pool {text!r} is not a public pool token"
    )


def _normalize_proof_cost(value: Any) -> int:
    if value is None or value is False:
        return DEFAULT_PROOF_COST
    if value is True or isinstance(value, bool):
        raise FixtureProofAwareXdistError("proof cost must be a non-negative integer")
    if isinstance(value, float):
        raise FixtureProofAwareXdistError(
            "floating-point proof cost is not JSON-safe for xdist scheduling"
        )
    try:
        cost = int(value)
    except (TypeError, ValueError) as exc:
        raise FixtureProofAwareXdistError("proof cost must be a non-negative integer") from exc
    if cost < 0:
        raise FixtureProofAwareXdistError("proof cost must be a non-negative integer")
    if cost > MAX_PROOF_COST:
        raise FixtureProofAwareXdistError("proof cost is over budget")
    return cost


def _unassigned_worker_id(resource_pool: str) -> str:
    return f"{UNASSIGNED_WORKER_PREFIX}{resource_pool}"


@dataclass(frozen=True)
class SchedulingUnit:
    """One collected test as a public scheduling fact.

    Not a pytest test class.
    """

    __test__ = False

    nodeid: str
    resource_pool: str = DEFAULT_RESOURCE_POOL
    proof_cost: int = DEFAULT_PROOF_COST
    affinity_key: str = ""
    affinity_fixtures: tuple[str, ...] = ()
    unknown_fixture_semantics: bool = False

    def __post_init__(self) -> None:
        nodeid = _bounded_nodeid(self.nodeid)
        if not nodeid:
            raise FixtureProofAwareXdistError("scheduling unit nodeid is required")
        object.__setattr__(self, "nodeid", nodeid)
        object.__setattr__(self, "resource_pool", _normalize_resource_pool(self.resource_pool))
        object.__setattr__(self, "proof_cost", _normalize_proof_cost(self.proof_cost))
        affinity_key = _bounded_text(self.affinity_key, max_chars=80)
        if affinity_key and not affinity_key.startswith(_DIGEST_PREFIX):
            raise FixtureProofAwareXdistError("affinity key must be a public digest")
        if self.unknown_fixture_semantics:
            affinity_key = ""
            object.__setattr__(self, "affinity_fixtures", ())
        object.__setattr__(self, "affinity_key", affinity_key)
        fixtures = tuple(
            _bounded_text(name, max_chars=256)
            for name in self.affinity_fixtures
            if _bounded_text(name, max_chars=256)
        )
        object.__setattr__(self, "affinity_fixtures", fixtures)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodeid": self.nodeid,
            "resource_pool": self.resource_pool,
            "proof_cost": self.proof_cost,
            "affinity_key": self.affinity_key,
            "affinity_fixtures": list(self.affinity_fixtures),
            "unknown_fixture_semantics": self.unknown_fixture_semantics,
        }


@dataclass(frozen=True)
class WorkerSpec:
    """One xdist worker bound to exactly one resource pool."""

    __test__ = False

    worker_id: str
    resource_pool: str = DEFAULT_RESOURCE_POOL

    def __post_init__(self) -> None:
        worker_id = _bounded_text(self.worker_id, max_chars=256).strip()
        if not worker_id:
            raise FixtureProofAwareXdistError("worker_id is required")
        if worker_id.startswith(UNASSIGNED_WORKER_PREFIX):
            raise FixtureProofAwareXdistError(
                "worker_id must not use the unassigned reservation prefix"
            )
        object.__setattr__(self, "worker_id", worker_id)
        object.__setattr__(self, "resource_pool", _normalize_resource_pool(self.resource_pool))

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "resource_pool": self.resource_pool,
        }


def _affinity_key_for(
    resource_pool: str,
    fixtures: Sequence[tuple[str, str, str]],
) -> str:
    if not fixtures:
        return ""
    return public_digest(
        {
            "resource_pool": resource_pool,
            "fixtures": [
                {"name": name, "scope": scope, "identity": identity}
                for name, scope, identity in fixtures
            ],
        }
    )


def _definition_scope(definition: Any) -> str:
    if isinstance(definition, Mapping):
        scope = definition.get("fixture_scope") or definition.get("scope") or FUNCTION_SCOPE
    else:
        scope = (
            getattr(definition, "fixture_scope", None)
            or getattr(definition, "scope", None)
            or FUNCTION_SCOPE
        )
    text = str(scope or FUNCTION_SCOPE)
    if text not in AFFINITY_SCOPES and text != FUNCTION_SCOPE:
        return FUNCTION_SCOPE
    return text


def _definition_name(definition: Any) -> str:
    if isinstance(definition, Mapping):
        return _bounded_text(definition.get("name"), max_chars=256)
    return _bounded_text(getattr(definition, "name", ""), max_chars=256)


def _definition_identity(definition: Any) -> str:
    if isinstance(definition, Mapping):
        identity = (
            definition.get("definition_cid")
            or definition.get("origin_path")
            or ""
        )
    else:
        identity = (
            getattr(definition, "definition_cid", None)
            or getattr(definition, "origin_path", None)
            or ""
        )
    return _bounded_text(identity, max_chars=256)


def affinity_fixtures_from_definitions(
    definitions: Iterable[Any] | None,
) -> tuple[tuple[str, str, str], ...]:
    """Return public (name, scope, identity) tuples for affinity scopes."""

    found: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for definition in definitions or ():
        scope = _definition_scope(definition)
        if scope not in AFFINITY_SCOPES:
            continue
        name = _definition_name(definition)
        if not name:
            continue
        identity = _definition_identity(definition)
        key = (name, scope, identity)
        if key in seen:
            continue
        seen.add(key)
        found.append(key)
    found.sort()
    return tuple(found)


def _closure_definitions(closure: Any) -> tuple[Any, ...]:
    if closure is None:
        return ()
    if isinstance(closure, Mapping):
        definitions = closure.get("definitions") or ()
        return tuple(definitions)
    definitions = getattr(closure, "definitions", None)
    if definitions is None and hasattr(closure, "to_dict"):
        try:
            payload = closure.to_dict()
        except Exception:
            payload = None
        if isinstance(payload, Mapping):
            definitions = payload.get("definitions") or ()
    if definitions is None:
        return ()
    return tuple(definitions)


def _unknown_fixture_semantics(item: Any, closure: Any = None) -> bool:
    explicit = getattr(item, "_ipfs_proof_reuse_unknown_fixture_semantics", None)
    if explicit is True:
        return True
    extraction = getattr(item, "_ipfs_proof_reuse_fixture_definition_extraction", None)
    if extraction is not None:
        if bool(getattr(extraction, "requires_full_execution", False)):
            return True
        reasons = getattr(extraction, "full_execution_reasons", ()) or ()
        if reasons:
            return True
        extracted_closure = getattr(extraction, "closure", None)
        if extracted_closure is not None:
            closure = extracted_closure
    if closure is None:
        closure = getattr(item, "_ipfs_proof_reuse_fixture_definition_closure", None)
    if closure is None:
        return False
    completeness = str(
        getattr(closure, "completeness", None)
        or (closure.get("completeness") if isinstance(closure, Mapping) else "")
        or ""
    ).lower()
    if completeness in {"unknown", "incomplete"}:
        return True
    if bool(getattr(closure, "requires_full_execution", False)):
        return True
    if bool(getattr(closure, "truncated", False)):
        return True
    return False


def _marker_values(item: Any, name: str) -> tuple[Any, ...]:
    getter = getattr(item, "iter_markers", None)
    if not callable(getter):
        closest = getattr(item, "get_closest_marker", None)
        if not callable(closest):
            return ()
        marker = closest(name)
        if marker is None:
            return ()
        args = tuple(getattr(marker, "args", ()) or ())
        kwargs = getattr(marker, "kwargs", None) or {}
        if args:
            return args
        if "value" in kwargs:
            return (kwargs["value"],)
        return ()
    values: list[Any] = []
    try:
        markers = getter(name)
    except Exception:
        return ()
    for marker in markers or ():
        args = tuple(getattr(marker, "args", ()) or ())
        if args:
            values.extend(args)
            continue
        kwargs = getattr(marker, "kwargs", None) or {}
        if "value" in kwargs:
            values.append(kwargs["value"])
    return tuple(values)


def resource_pool_from_item(item: Any) -> str:
    """Return the closed resource pool for *item* without merging others."""

    raw = getattr(item, ITEM_RESOURCE_POOL_ATTRIBUTE, None)
    if raw:
        return _normalize_resource_pool(raw)
    values = _marker_values(item, RESOURCE_POOL_MARKER)
    if values:
        return _normalize_resource_pool(values[0])
    keywords = getattr(item, "keywords", None)
    if isinstance(keywords, Mapping) and RESOURCE_POOL_MARKER in keywords:
        return _normalize_resource_pool(keywords.get(RESOURCE_POOL_MARKER))
    return DEFAULT_RESOURCE_POOL


def proof_cost_from_item(item: Any) -> int:
    """Return the integer proof cost for *item*."""

    raw = getattr(item, ITEM_PROOF_COST_ATTRIBUTE, None)
    if raw is not None and raw is not False:
        return _normalize_proof_cost(raw)
    values = _marker_values(item, PROOF_COST_MARKER)
    if values:
        return _normalize_proof_cost(values[0])
    return DEFAULT_PROOF_COST


def scheduling_unit_from_mapping(payload: Mapping[str, Any]) -> SchedulingUnit:
    """Build one unit from a public mapping."""

    if not isinstance(payload, Mapping):
        raise FixtureProofAwareXdistError("scheduling unit must be a mapping")
    fixtures = payload.get("affinity_fixtures") or ()
    definitions = payload.get("definitions") or ()
    affinity_pairs = affinity_fixtures_from_definitions(definitions)
    names = tuple(name for name, _scope, _identity in affinity_pairs) or tuple(
        _bounded_text(name, max_chars=256) for name in fixtures if name
    )
    unknown = bool(payload.get("unknown_fixture_semantics"))
    pool = _normalize_resource_pool(payload.get("resource_pool"))
    affinity_key = "" if unknown else (
        _bounded_text(payload.get("affinity_key"), max_chars=80)
        or _affinity_key_for(pool, affinity_pairs)
    )
    return SchedulingUnit(
        nodeid=str(payload.get("nodeid") or ""),
        resource_pool=pool,
        proof_cost=payload.get("proof_cost", DEFAULT_PROOF_COST),
        affinity_key=affinity_key,
        affinity_fixtures=names,
        unknown_fixture_semantics=unknown,
    )


def scheduling_unit_from_item(item: Any) -> SchedulingUnit:
    """Derive one public scheduling unit from a collected item or mapping."""

    if isinstance(item, SchedulingUnit):
        return item
    if isinstance(item, Mapping):
        return scheduling_unit_from_mapping(item)
    existing = getattr(item, ITEM_SCHEDULING_UNIT_ATTRIBUTE, None)
    if isinstance(existing, SchedulingUnit):
        return existing
    nodeid = _bounded_nodeid(getattr(item, "nodeid", "") or "")
    if not nodeid:
        raise FixtureProofAwareXdistError("scheduling unit nodeid is required")
    explicit = getattr(item, ITEM_AFFINITY_FIXTURES_ATTRIBUTE, None)
    closure = getattr(item, "_ipfs_proof_reuse_fixture_definition_closure", None)
    definitions: Iterable[Any] = ()
    if explicit:
        names = tuple(_bounded_text(name, max_chars=256) for name in explicit if name)
        affinity_pairs = tuple((name, "session", name) for name in names)
    else:
        definitions = _closure_definitions(closure)
        if not definitions:
            snapshot = getattr(item, "_ipfs_proof_reuse_collection_snapshot", None)
            definitions = getattr(snapshot, "definitions", ()) or ()
        affinity_pairs = affinity_fixtures_from_definitions(definitions)
        names = tuple(name for name, _scope, _identity in affinity_pairs)
    unknown = _unknown_fixture_semantics(item, closure)
    pool = resource_pool_from_item(item)
    return SchedulingUnit(
        nodeid=nodeid,
        resource_pool=pool,
        proof_cost=proof_cost_from_item(item),
        affinity_key="" if unknown else _affinity_key_for(pool, affinity_pairs),
        affinity_fixtures=names,
        unknown_fixture_semantics=unknown,
    )


def attach_scheduling_descriptors(items: Iterable[Any]) -> tuple[SchedulingUnit, ...]:
    """Attach scheduling units to collected items without executing them."""

    attached: list[SchedulingUnit] = []
    for index, item in enumerate(items):
        if isinstance(item, SchedulingUnit):
            attached.append(item)
            continue
        try:
            unit = scheduling_unit_from_item(item)
        except FixtureProofAwareXdistError:
            nodeid = _bounded_nodeid(getattr(item, "nodeid", "") or "") or f"item:{index}"
            unit = SchedulingUnit(
                nodeid=nodeid,
                unknown_fixture_semantics=True,
            )
        if not isinstance(item, Mapping):
            setattr(item, ITEM_SCHEDULING_UNIT_ATTRIBUTE, unit)
        attached.append(unit)
    return tuple(attached)


def _units_from(items: Iterable[Any]) -> tuple[SchedulingUnit, ...]:
    units: list[SchedulingUnit] = []
    seen: set[str] = set()
    for index, item in enumerate(items):
        if isinstance(item, SchedulingUnit):
            unit = item
        else:
            try:
                unit = scheduling_unit_from_item(item)
            except FixtureProofAwareXdistError:
                nodeid = ""
                if isinstance(item, Mapping):
                    nodeid = _bounded_nodeid(item.get("nodeid") or "")
                else:
                    nodeid = _bounded_nodeid(getattr(item, "nodeid", "") or "")
                unit = SchedulingUnit(
                    nodeid=nodeid or f"item:{index}",
                    unknown_fixture_semantics=True,
                )
        if unit.nodeid in seen:
            raise FixtureProofAwareXdistError(
                f"duplicate scheduling nodeid {unit.nodeid!r}"
            )
        seen.add(unit.nodeid)
        units.append(unit)
    return tuple(units)


def _workers_from(workers: Iterable[Any]) -> tuple[WorkerSpec, ...]:
    specs: list[WorkerSpec] = []
    seen: set[str] = set()
    for worker in workers:
        if isinstance(worker, WorkerSpec):
            spec = worker
        elif isinstance(worker, Mapping):
            spec = WorkerSpec(
                worker_id=str(worker.get("worker_id") or worker.get("id") or ""),
                resource_pool=worker.get("resource_pool") or DEFAULT_RESOURCE_POOL,
            )
        else:
            spec = WorkerSpec(
                worker_id=str(
                    getattr(worker, "worker_id", None)
                    or getattr(worker, "id", None)
                    or ""
                ),
                resource_pool=getattr(worker, "resource_pool", DEFAULT_RESOURCE_POOL),
            )
        if spec.worker_id in seen:
            raise FixtureProofAwareXdistError(
                f"duplicate worker_id {spec.worker_id!r}"
            )
        seen.add(spec.worker_id)
        specs.append(spec)
    return tuple(specs)


@dataclass(frozen=True)
class _AffinityGroup:
    key: str
    resource_pool: str
    units: tuple[SchedulingUnit, ...]

    @property
    def proof_cost(self) -> int:
        return sum(unit.proof_cost for unit in self.units)

    @property
    def sort_key(self) -> tuple[int, str, str]:
        first = self.units[0].nodeid if self.units else ""
        return (-self.proof_cost, self.key or first, first)


def _group_units(units: Sequence[SchedulingUnit]) -> tuple[_AffinityGroup, ...]:
    grouped: dict[tuple[str, str], list[SchedulingUnit]] = {}
    singles: list[_AffinityGroup] = []
    for unit in units:
        if unit.affinity_key and not unit.unknown_fixture_semantics:
            grouped.setdefault((unit.resource_pool, unit.affinity_key), []).append(unit)
            continue
        singles.append(
            _AffinityGroup(key="", resource_pool=unit.resource_pool, units=(unit,))
        )
    groups = [
        _AffinityGroup(key=key, resource_pool=pool, units=tuple(members))
        for (pool, key), members in grouped.items()
    ]
    groups.extend(singles)
    groups.sort(key=lambda group: group.sort_key)
    return tuple(groups)


def _assign_groups(
    groups: Sequence[_AffinityGroup],
    workers: Sequence[WorkerSpec],
) -> dict[str, list[str]]:
    loads = {spec.worker_id: 0 for spec in workers}
    assigned: dict[str, list[str]] = {spec.worker_id: [] for spec in workers}
    ordered_workers = sorted(workers, key=lambda spec: spec.worker_id)
    for group in groups:
        chosen = min(
            ordered_workers,
            key=lambda spec: (loads[spec.worker_id], spec.worker_id),
        )
        assigned[chosen.worker_id].extend(unit.nodeid for unit in group.units)
        loads[chosen.worker_id] += group.proof_cost
    for nodeids in assigned.values():
        # Preserve group locality but keep a stable per-worker order.
        nodeids.sort()
    return assigned


def _makespan(
    assignments: Mapping[str, Sequence[str]],
    costs: Mapping[str, int],
) -> int:
    if not assignments:
        return 0
    return max(
        (sum(costs.get(nodeid, 0) for nodeid in nodeids) for nodeids in assignments.values()),
        default=0,
    )


def _affinity_spread(
    assignments: Mapping[str, Sequence[str]],
    units: Sequence[SchedulingUnit],
) -> int:
    worker_by_node = {
        nodeid: worker_id
        for worker_id, nodeids in assignments.items()
        for nodeid in nodeids
    }
    by_key: dict[str, set[str]] = {}
    for unit in units:
        if not unit.affinity_key:
            continue
        worker_id = worker_by_node.get(unit.nodeid)
        if not worker_id:
            continue
        by_key.setdefault(unit.affinity_key, set()).add(worker_id)
    return sum(max(0, len(workers) - 1) for workers in by_key.values())


def round_robin_within_pools(
    units: Sequence[SchedulingUnit],
    workers: Sequence[WorkerSpec],
) -> dict[str, list[str]]:
    """Naive per-pool round-robin that still refuses to merge pools."""

    assigned: dict[str, list[str]] = {spec.worker_id: [] for spec in workers}
    by_pool: dict[str, list[WorkerSpec]] = {}
    for spec in workers:
        by_pool.setdefault(spec.resource_pool, []).append(spec)
    for pool in by_pool:
        by_pool[pool] = sorted(by_pool[pool], key=lambda spec: spec.worker_id)
    cursors: dict[str, int] = {pool: 0 for pool in by_pool}
    for unit in units:
        pool_workers = by_pool.get(unit.resource_pool) or ()
        if not pool_workers:
            unassigned = _unassigned_worker_id(unit.resource_pool)
            assigned.setdefault(unassigned, []).append(unit.nodeid)
            continue
        index = cursors[unit.resource_pool] % len(pool_workers)
        assigned[pool_workers[index].worker_id].append(unit.nodeid)
        cursors[unit.resource_pool] += 1
    return assigned


@dataclass(frozen=True)
class FixtureProofAwarePlacement:
    """Complete placement of every collected test.

    Not a pytest test class.
    """

    __test__ = False

    assignments: Mapping[str, tuple[str, ...]]
    unit_worker: Mapping[str, str]
    units: tuple[SchedulingUnit, ...]
    workers: tuple[WorkerSpec, ...]
    makespan: int
    naive_makespan: int
    affinity_spread: int
    naive_affinity_spread: int
    omitted_nodeids: tuple[str, ...] = ()
    pools_merged: bool = False
    may_authorize_skip: bool = False
    production_admitted: bool = False
    self_approved: bool = False
    claim_unchanged: bool = True
    workers_may_publish: bool = False
    controller_owns_writes: bool = True
    claim_class: str = CLAIM_CLASS
    interface: str = FIXTURE_PROOF_AWARE_XDIST_RESULT_INTERFACE
    scheduling_interface: str = FIXTURE_PROOF_AWARE_XDIST_INTERFACE

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "assignments",
            MappingProxyType(
                {
                    str(worker_id): tuple(nodeids)
                    for worker_id, nodeids in self.assignments.items()
                }
            ),
        )
        object.__setattr__(self, "unit_worker", MappingProxyType(dict(self.unit_worker)))
        if self.omitted_nodeids:
            raise FixtureProofAwareXdistError(
                "fixture-proof-aware xdist scheduling must not omit tests"
            )
        if self.pools_merged:
            raise FixtureProofAwareXdistError(
                "fixture-proof-aware xdist scheduling must not merge resource pools"
            )
        if self.may_authorize_skip:
            raise FixtureProofAwareXdistError(
                "fixture-proof-aware xdist scheduling must not authorize skip"
            )
        if self.production_admitted or self.self_approved or not self.claim_unchanged:
            raise FixtureProofAwareXdistError(
                "fixture-proof-aware xdist scheduling cannot admit production, "
                "self-approve, or change claims"
            )
        if self.workers_may_publish:
            raise FixtureProofAwareXdistError("workers must not publish")
        placed = [nodeid for nodeids in self.assignments.values() for nodeid in nodeids]
        expected = [unit.nodeid for unit in self.units]
        if sorted(placed) != sorted(expected):
            raise FixtureProofAwareXdistError(
                "fixture-proof-aware xdist scheduling must not omit tests"
            )
        if len(placed) != len(set(placed)):
            raise FixtureProofAwareXdistError("placement contains duplicate nodeids")
        worker_pool = {spec.worker_id: spec.resource_pool for spec in self.workers}
        unit_pool = {unit.nodeid: unit.resource_pool for unit in self.units}
        for worker_id, nodeids in self.assignments.items():
            if worker_id.startswith(UNASSIGNED_WORKER_PREFIX):
                reserved = worker_id[len(UNASSIGNED_WORKER_PREFIX) :]
                for nodeid in nodeids:
                    if unit_pool[nodeid] != reserved:
                        raise FixtureProofAwareXdistError(
                            "unassigned reservation merged resource pools"
                        )
                continue
            pool = worker_pool.get(worker_id)
            if pool is None:
                raise FixtureProofAwareXdistError(
                    f"placement worker {worker_id!r} is not a declared worker"
                )
            for nodeid in nodeids:
                if unit_pool[nodeid] != pool:
                    raise FixtureProofAwareXdistError(
                        "fixture-proof-aware xdist scheduling must not merge resource pools"
                    )

    @property
    def affinity_improved(self) -> bool:
        return self.affinity_spread < self.naive_affinity_spread

    @property
    def cost_improved(self) -> bool:
        return self.makespan < self.naive_makespan

    @property
    def placement_improved(self) -> bool:
        return (
            self.affinity_improved
            or self.cost_improved
            or (
                self.affinity_spread <= self.naive_affinity_spread
                and self.makespan <= self.naive_makespan
                and bool(self.units)
            )
        )

    @property
    def omitted(self) -> bool:
        return False

    def worker_for(self, nodeid: str) -> str:
        return self.unit_worker[nodeid]

    def nodeids_for(self, worker_id: str) -> tuple[str, ...]:
        return tuple(self.assignments.get(worker_id, ()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface": self.interface,
            "scheduling_interface": self.scheduling_interface,
            "claim_class": self.claim_class,
            "assignments": {
                worker_id: list(nodeids)
                for worker_id, nodeids in self.assignments.items()
            },
            "unit_worker": dict(self.unit_worker),
            "makespan": self.makespan,
            "naive_makespan": self.naive_makespan,
            "affinity_spread": self.affinity_spread,
            "naive_affinity_spread": self.naive_affinity_spread,
            "affinity_improved": self.affinity_improved,
            "cost_improved": self.cost_improved,
            "placement_improved": self.placement_improved,
            "omitted_nodeids": [],
            "pools_merged": False,
            "may_authorize_skip": False,
            "production_admitted": False,
            "self_approved": False,
            "claim_unchanged": True,
            "workers_may_publish": False,
            "controller_owns_writes": True,
        }


def place_items(
    items: Iterable[Any],
    workers: Iterable[Any],
) -> FixtureProofAwarePlacement:
    """Place every collected test using fixture affinity and proof cost."""

    units = _units_from(items)
    workers_spec = _workers_from(workers)
    by_pool_units: dict[str, list[SchedulingUnit]] = {}
    for unit in units:
        by_pool_units.setdefault(unit.resource_pool, []).append(unit)
    by_pool_workers: dict[str, list[WorkerSpec]] = {}
    for spec in workers_spec:
        by_pool_workers.setdefault(spec.resource_pool, []).append(spec)

    assignments: dict[str, list[str]] = {spec.worker_id: [] for spec in workers_spec}
    for pool, pool_units in sorted(by_pool_units.items()):
        pool_workers = tuple(by_pool_workers.get(pool) or ())
        if not pool_workers:
            unassigned = _unassigned_worker_id(pool)
            assignments.setdefault(unassigned, []).extend(
                unit.nodeid for unit in pool_units
            )
            continue
        groups = _group_units(pool_units)
        placed = _assign_groups(groups, pool_workers)
        for worker_id, nodeids in placed.items():
            assignments.setdefault(worker_id, []).extend(nodeids)

    for worker_id, nodeids in assignments.items():
        # Keep deterministic order per worker while preserving affinity clumps
        # already appended as contiguous group batches.
        assignments[worker_id] = list(nodeids)

    costs = {unit.nodeid: unit.proof_cost for unit in units}
    naive = round_robin_within_pools(units, workers_spec)
    unit_worker = {
        nodeid: worker_id
        for worker_id, nodeids in assignments.items()
        for nodeid in nodeids
    }
    frozen_assignments = {
        worker_id: tuple(nodeids) for worker_id, nodeids in assignments.items()
    }
    return FixtureProofAwarePlacement(
        assignments=frozen_assignments,
        unit_worker=unit_worker,
        units=units,
        workers=workers_spec,
        makespan=_makespan(frozen_assignments, costs),
        naive_makespan=_makespan(naive, costs),
        affinity_spread=_affinity_spread(frozen_assignments, units),
        naive_affinity_spread=_affinity_spread(naive, units),
    )


def attach_placement(items: Iterable[Any], placement: FixtureProofAwarePlacement) -> None:
    """Record the assigned worker on each collected item."""

    for item in items:
        if isinstance(item, (SchedulingUnit, Mapping)):
            continue
        nodeid = _bounded_nodeid(getattr(item, "nodeid", "") or "")
        worker_id = placement.unit_worker.get(nodeid)
        if worker_id:
            setattr(item, ITEM_PLACEMENT_WORKER_ATTRIBUTE, worker_id)


def _worker_pool_from_node(node: Any) -> str:
    worker_input = getattr(node, "workerinput", None)
    if isinstance(worker_input, Mapping):
        raw = worker_input.get("ipfs_proof_reuse_resource_pool")
        if raw:
            return _normalize_resource_pool(raw)
    spec = getattr(getattr(node, "gateway", None), "spec", None)
    if isinstance(spec, Mapping) and spec.get("resource_pool"):
        return _normalize_resource_pool(spec.get("resource_pool"))
    pool = getattr(node, "resource_pool", None)
    if pool:
        return _normalize_resource_pool(pool)
    return DEFAULT_RESOURCE_POOL


def _worker_id_from_node(node: Any, fallback: str) -> str:
    gateway = getattr(node, "gateway", None)
    worker_id = str(getattr(gateway, "id", "") or "")
    if worker_id:
        return worker_id
    worker_input = getattr(node, "workerinput", None)
    if isinstance(worker_input, Mapping):
        raw = str(worker_input.get("workerid") or worker_input.get("worker_id") or "")
        if raw:
            return raw
    return fallback


class FixtureProofAwareXdistScheduler:
    """Controller-owned xdist scheduler using fixture/proof placement.

    Duck-typed to the pytest-xdist scheduler protocol.  Import stays cold:
    this class never imports pytest or xdist.
    """

    interface = FIXTURE_PROOF_AWARE_XDIST_INTERFACE

    def __init__(self, config: Any, log: Any = None) -> None:
        self.config = config
        self.log = log
        self.numnodes = 0
        self.node2pending: dict[Any, list[int]] = {}
        self.node2collection: dict[Any, list[str]] = {}
        self.node2pool: dict[Any, str] = {}
        self.node2id: dict[Any, str] = {}
        self.collection: list[str] | None = None
        self.placement: FixtureProofAwarePlacement | None = None
        self._scheduled = False

    @property
    def collection_is_completed(self) -> bool:
        return self.numnodes > 0 and len(self.node2collection) >= self.numnodes

    def has_pending(self) -> bool:
        if not self._scheduled:
            return True
        return any(self.node2pending.values())

    def add_node(self, node: Any) -> None:
        if node in self.node2pending:
            return
        self.node2pending[node] = []
        self.numnodes += 1
        worker_id = _worker_id_from_node(node, f"gw{self.numnodes - 1}")
        self.node2id[node] = worker_id
        self.node2pool[node] = _worker_pool_from_node(node)

    def add_node_collection(self, node: Any, collection: Iterable[str]) -> None:
        if node not in self.node2pending:
            self.add_node(node)
        self.node2collection[node] = [str(nodeid) for nodeid in collection]

    def mark_test_complete(
        self,
        node: Any,
        item_index: int,
        duration: Any = 0,
    ) -> None:
        del duration
        pending = self.node2pending.get(node)
        if pending is None:
            return
        try:
            pending.remove(item_index)
        except ValueError:
            return

    def remove_node(self, node: Any) -> None:
        self.node2pending.pop(node, None)
        self.node2collection.pop(node, None)
        self.node2pool.pop(node, None)
        self.node2id.pop(node, None)
        self.numnodes = len(self.node2pending)

    def schedule(self) -> None:
        if not self.collection_is_completed:
            return
        if not self._scheduled:
            self._initial_distribute()
            self._scheduled = True
        for node, pending in list(self.node2pending.items()):
            if pending:
                continue
            shutdown = getattr(node, "shutdown", None)
            if callable(shutdown):
                try:
                    shutdown()
                except Exception:
                    continue

    def _canonical_collection(self) -> list[str]:
        collections = list(self.node2collection.values())
        if not collections:
            return []
        first = list(collections[0])
        for other in collections[1:]:
            if list(other) != first:
                # Disagreeing worker collections cannot drop tests: keep the
                # longest unique sequence in first-seen order.
                seen = list(first)
                known = set(first)
                for nodeid in other:
                    if nodeid not in known:
                        known.add(nodeid)
                        seen.append(nodeid)
                first = seen
        return first

    def _units_for_collection(self, collection: Sequence[str]) -> tuple[SchedulingUnit, ...]:
        config = self.config
        attached: dict[str, SchedulingUnit] = {}
        items = getattr(config, "ipfs_proof_reuse_scheduling_items", None)
        if items is None:
            session = getattr(config, "_ipfs_proof_reuse_session", None)
            items = getattr(session, "items", None)
        if items is not None:
            for item in items:
                try:
                    unit = scheduling_unit_from_item(item)
                except Exception:
                    continue
                attached[unit.nodeid] = unit
        units: list[SchedulingUnit] = []
        for nodeid in collection:
            existing = attached.get(nodeid)
            if existing is not None:
                units.append(existing)
                continue
            units.append(SchedulingUnit(nodeid=nodeid))
        return tuple(units)

    def _initial_distribute(self) -> None:
        collection = self._canonical_collection()
        self.collection = list(collection)
        index_by_nodeid = {nodeid: index for index, nodeid in enumerate(collection)}
        workers = [
            WorkerSpec(worker_id=self.node2id[node], resource_pool=self.node2pool[node])
            for node in self.node2pending
        ]
        units = self._units_for_collection(collection)
        try:
            placement = place_items(units, workers)
        except FixtureProofAwareXdistError:
            # Fail closed to per-pool round-robin rather than omit tests.
            naive = round_robin_within_pools(units, workers)
            assignments = {key: tuple(value) for key, value in naive.items()}
            costs = {unit.nodeid: unit.proof_cost for unit in units}
            unit_worker = {
                nodeid: worker_id
                for worker_id, nodeids in assignments.items()
                for nodeid in nodeids
            }
            placement = FixtureProofAwarePlacement(
                assignments=assignments,
                unit_worker=unit_worker,
                units=units,
                workers=tuple(workers),
                makespan=_makespan(assignments, costs),
                naive_makespan=_makespan(assignments, costs),
                affinity_spread=_affinity_spread(assignments, units),
                naive_affinity_spread=_affinity_spread(assignments, units),
            )
        self.placement = placement
        node_by_id = {worker_id: node for node, worker_id in self.node2id.items()}
        leftover: list[int] = []
        for worker_id, nodeids in placement.assignments.items():
            indices = [
                index_by_nodeid[nodeid]
                for nodeid in nodeids
                if nodeid in index_by_nodeid
            ]
            node = node_by_id.get(worker_id)
            if node is None:
                leftover.extend(indices)
                continue
            self.node2pending[node] = list(indices)
            send = getattr(node, "send_runtest_some", None)
            if callable(send) and indices:
                send(indices)
        # Unassigned-pool tests remain in the placement without merging pools.
        # pytest-xdist still requires every index to run, so leftover indices
        # drain through the first node as normal execution fallback.
        if leftover:
            drain = next(iter(self.node2pending))
            self.node2pending[drain].extend(leftover)
            send = getattr(drain, "send_runtest_some", None)
            if callable(send):
                send(leftover)


def make_xdist_scheduler(config: Any, log: Any = None) -> FixtureProofAwareXdistScheduler | None:
    """Return the controller-owned scheduler, or None when unused."""

    return FixtureProofAwareXdistScheduler(config, log)


__all__ = [
    "AFFINITY_SCOPES",
    "CLAIM_CLASS",
    "CLOSED_RESOURCE_POOLS",
    "DEFAULT_POLICY_CID",
    "DEFAULT_RESOURCE_POOL",
    "FIXTURE_PROOF_AWARE_XDIST_INTERFACE",
    "FIXTURE_PROOF_AWARE_XDIST_POLICY_INTERFACE",
    "FIXTURE_PROOF_AWARE_XDIST_RESULT_INTERFACE",
    "FIXTURE_PROOF_AWARE_XDIST_SCHEMA",
    "ITEM_AFFINITY_FIXTURES_ATTRIBUTE",
    "ITEM_PLACEMENT_WORKER_ATTRIBUTE",
    "ITEM_PROOF_COST_ATTRIBUTE",
    "ITEM_RESOURCE_POOL_ATTRIBUTE",
    "ITEM_SCHEDULING_UNIT_ATTRIBUTE",
    "PREDECESSOR_INTERFACE",
    "PROOF_COST_MARKER",
    "RESOURCE_POOL_MARKER",
    "SCHEDULING_DOES_NOT",
    "SCHEDULING_ESTABLISHES",
    "SCHEDULING_POLICY",
    "FixtureProofAwarePlacement",
    "FixtureProofAwareXdistError",
    "FixtureProofAwareXdistScheduler",
    "SchedulingUnit",
    "WorkerSpec",
    "affinity_fixtures_from_definitions",
    "attach_placement",
    "attach_scheduling_descriptors",
    "authority_descriptor",
    "make_xdist_scheduler",
    "place_items",
    "proof_cost_from_item",
    "public_digest",
    "pytest_xdist_runtime_available",
    "record_typed_unavailable",
    "resource_pool_from_item",
    "round_robin_within_pools",
    "scheduling_unit_from_item",
    "typed_unavailable_records",
    "workers_may_publish",
]
