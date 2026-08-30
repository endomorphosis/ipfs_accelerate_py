"""Collection-time fixture-definition closure extraction (PCTDD-022).

Accelerate-owned extraction and memoization of exact fixture-definition
closures.  Datasets owns the ``@1`` schemas; this module only reads
collection-time definitions and binds those schemas.

Rules:

* Extraction is authoritative at collection.  Fixture bodies, setup, call,
  and teardown are never executed.
* Exact closures bind transitive definitions, ancestor conftests, pytest
  hooks, policy identities, and definition-time finalizers.
* Unknown, missing, cyclic, truncated, or incomplete members force full
  execution.  They never authorize pytest skip and never admit production.
* Instance-time ``request.addfinalizer`` values are typed unavailable and
  cannot be claimed as definition identity.
* ``TestExecutionKeyV2`` is not assembled here.  Extraction is an integrity
  commitment, not execution or semantics.

Import is cold-safe: no pytest, network, package installer, or prover.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

FIXTURE_DEFINITION_EXTRACTION_INTERFACE: Final = (
    "FixtureDefinitionClosureExtraction@1"
)
FIXTURE_DEFINITION_EXTRACTION_SCHEMA: Final = (
    "ipfs_accelerate_py/testing/proof-reuse/fixture-definition-extraction@1"
)
FIXTURE_DEFINITION_CLOSURE_MEMO_INTERFACE: Final = (
    "FixtureDefinitionClosureMemo@1"
)
EXTRACTION_RESULT_INTERFACE: Final = "FixtureDefinitionExtractionResult@1"
EXTRACTION_POLICY_INTERFACE: Final = "FixtureDefinitionExtractionPolicy@1"
CLAIM_CLASS: Final = "IntegrityCommitment"
AUTHORITATIVE_AT: Final = "collection"
DEFAULT_COMPLETENESS: Final = "unknown"
SCHEMA_AUTHORITY: Final = (
    "ipfs_datasets_py.logic.zkp.pctdd.fixture_definition_contracts"
)
EXTRACTION_AUTHORITY: Final = (
    "ipfs_accelerate_py.testing.proof_reuse.fixture_definition_extraction"
)
ITEM_FIXTURE_DEFINITION_CLOSURE_ATTRIBUTE: Final = (
    "_ipfs_proof_reuse_fixture_definition_closure"
)
ITEM_FIXTURE_DEFINITION_EXTRACTION_ATTRIBUTE: Final = (
    "_ipfs_proof_reuse_fixture_definition_extraction"
)
FIXTURE_DEFINITION_MEMO_ATTRIBUTE: Final = (
    "_ipfs_proof_reuse_fixture_definition_memo"
)
MAX_TEXT_CHARS: Final = 4_096
MAX_NAME_CHARS: Final = 256
MAX_PATH_CHARS: Final = 1_024
MAX_SEQUENCE_ITEMS: Final = 64
MAX_SOURCE_BYTES: Final = 1_048_576
MAX_TRANSITIVE_DEPTH: Final = 32
_DIGEST_PREFIX: Final = "sha256:"

REQUIRED_CLOSURE_MEMBER_KINDS: Final[tuple[str, ...]] = (
    "fixture_definition",
    "conftest",
    "hook",
    "policy",
    "finalizer",
)
FIXTURE_SCOPES: Final[frozenset[str]] = frozenset(
    {"function", "class", "module", "package", "session"}
)
IDENTITY_RELEVANT_HOOKS: Final[tuple[str, ...]] = (
    "pytest_addoption",
    "pytest_configure",
    "pytest_plugin_registered",
    "pytest_load_initial_conftests",
    "pytest_sessionstart",
    "pytest_sessionfinish",
    "pytest_collection",
    "pytest_itemcollected",
    "pytest_collection_modifyitems",
    "pytest_collectreport",
    "pytest_runtest_protocol",
    "pytest_runtest_setup",
    "pytest_runtest_call",
    "pytest_runtest_teardown",
    "pytest_runtest_makereport",
    "pytest_fixture_setup",
    "pytest_fixture_post_finalizer",
    "pytest_configure_node",
    "pytest_testnodedown",
)
INI_POLICY_KINDS: Final[tuple[tuple[str, str], ...]] = (
    ("addopts", "addopts"),
    ("markers", "markers"),
    ("filterwarnings", "filterwarnings"),
    ("required_plugins", "required_plugins"),
)
EXTRACTION_ESTABLISHES: Final = (
    "collection extracts and memoizes exact fixture closures without "
    "executing fixture bodies"
)
EXTRACTION_DOES_NOT: Final = (
    "execution or semantics; skip; TestExecutionKeyV2; fixture-body "
    "execution; pre-call fixture-instance authority; composite phase "
    "receipts; current-root publication; production ZK; adapter execution"
)
FORBIDDEN_COMMIT_FIELD_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "access_token",
        "api_key",
        "authorization",
        "cookie",
        "credential",
        "hidden_witness",
        "password",
        "private_key",
        "private_premise",
        "private_witness",
        "raw_value",
        "refresh_token",
        "secret",
        "secret_bytes",
        "session_token",
        "value_bytes",
        "witness",
    }
)
_PRIVATE_SUBSTRINGS: Final[tuple[str, ...]] = (
    "private",
    "secret",
    "password",
    "api_key",
    "witness",
    "credential",
    "token",
)
_TYPED_UNAVAILABLE: Final[tuple[tuple[str, str, str], ...]] = (
    (
        "test_execution_key_v2",
        "test_execution_key_v2_not_implemented",
        "collection-time extraction binds FixtureDefinitionClosure@1 only; "
        "TestExecutionKeyV2 remains a later datasets contract",
    ),
    (
        "pre_call_fixture_instance_key",
        "fixture_instance_authoritative_only_after_current_setup",
        "fixture-definition identity is collection-time only; exact "
        "fixture-instance identity becomes safely authoritative only after "
        "current setup",
    ),
    (
        "instance_time_finalizer",
        "instance_time_finalizer_not_definition_identity",
        "dynamically registered request.addfinalizer values are instance-time "
        "and cannot be claimed as exact definition-time finalizer identity",
    ),
    (
        "composite_phase_receipt",
        "composite_phase_receipt_not_implemented",
        "composite phase receipts are outside extraction and remain typed "
        "unavailable without changing TestPassStatementV1",
    ),
    (
        "production_zk",
        "production_zk_key_ceremony_unavailable",
        "production ZK proving remains typed unavailable; extraction "
        "integrity commitments cannot admit simulated or self-verified proofs",
    ),
    (
        "key_ceremony",
        "production_zk_key_ceremony_unavailable",
        "no production-eligible key ceremony is admitted by fixture-definition "
        "closure extraction",
    ),
    (
        "direct_execution_profile",
        "direct_execution_profile_optional",
        "direct CPython execution profiles remain optional and unadmitted; "
        "they cannot upgrade fixture-definition integrity commitments",
    ),
)


class FixtureDefinitionExtractionError(ValueError):
    """Raised when collection-time extraction is unsafe."""

    __test__ = False


def _is_private_key(key: str) -> bool:
    lowered = key.lower().replace("-", "_")
    if lowered in FORBIDDEN_COMMIT_FIELD_MARKERS:
        return True
    return any(marker in lowered for marker in _PRIVATE_SUBSTRINGS)


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        raise FixtureDefinitionExtractionError(
            "floating-point values are not JSON-safe for fixture-definition extraction"
        )
    if isinstance(value, Mapping):
        ready: dict[str, Any] = {}
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            name = str(key)
            if _is_private_key(name):
                raise FixtureDefinitionExtractionError(
                    f"extraction rejects private material key {name!r}"
                )
            ready[name] = _json_ready(item)
        return ready
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise FixtureDefinitionExtractionError("extraction rejects secret or raw bytes")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_ready(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_ready(to_dict())
    raise FixtureDefinitionExtractionError(
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


def _digest_bytes(payload: bytes) -> str:
    return _DIGEST_PREFIX + hashlib.sha256(payload).hexdigest()


def _digest_text(text: str) -> str:
    return _digest_bytes(text.encode("utf-8"))


EXTRACTION_POLICY: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "interface": EXTRACTION_POLICY_INTERFACE,
        "extraction_interface": FIXTURE_DEFINITION_EXTRACTION_INTERFACE,
        "memo_interface": FIXTURE_DEFINITION_CLOSURE_MEMO_INTERFACE,
        "authoritative_at": AUTHORITATIVE_AT,
        "executed_fixture_bodies": False,
        "may_authorize_skip": False,
        "production_admitted": False,
        "self_approved": False,
        "required_member_kinds": list(REQUIRED_CLOSURE_MEMBER_KINDS),
        "schema_authority": SCHEMA_AUTHORITY,
    }
)
DEFAULT_POLICY_CID: Final = public_digest(dict(EXTRACTION_POLICY))


def _load_datasets_contracts() -> Any | None:
    try:
        from ipfs_datasets_py.logic.zkp.pctdd import fixture_definition_contracts as contracts
    except Exception:
        return None
    return contracts


def datasets_contracts_available() -> bool:
    """Return whether datasets-owned ``@1`` closure codecs import."""

    return _load_datasets_contracts() is not None


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
        raise FixtureDefinitionExtractionError(
            "typed unavailable cases cannot admit, self-approve, or change claims"
        )
    return record


def typed_unavailable_records() -> tuple[dict[str, Any], ...]:
    """Closed set of PCTDD-022 typed unavailable capabilities."""

    records = [
        record_typed_unavailable(
            capability=capability,
            reason_code=reason_code,
            message=message,
        )
        for capability, reason_code, message in _TYPED_UNAVAILABLE
    ]
    if not datasets_contracts_available():
        records.append(
            record_typed_unavailable(
                capability="datasets_fixture_definition_contracts",
                reason_code="datasets_fixture_definition_contracts_unresolved",
                message=(
                    "datasets FixtureDefinitionClosure@1 codecs did not import "
                    "in this sealed environment; extraction forces full "
                    "execution and does not commit closures"
                ),
            )
        )
    return tuple(records)


def authority_descriptor() -> dict[str, Any]:
    """Return the extraction authority without widening others."""

    return {
        "canonical_semantic_and_statement_authority": "ipfs_datasets_py",
        "execution_scheduling_admission_authority": "ipfs_accelerate_py",
        "verified_storage_wal_cas_authority": "ipfs_kit_py",
        "extraction": EXTRACTION_AUTHORITY,
        "schema_authority": SCHEMA_AUTHORITY,
        "does_not": EXTRACTION_DOES_NOT,
        "establishes": EXTRACTION_ESTABLISHES,
        "claim_class": CLAIM_CLASS,
        "may_authorize_skip": False,
        "production_admitted": False,
        "self_approved": False,
        "worker_authored_test_is_sufficient_alone": False,
        "authoritative_at": AUTHORITATIVE_AT,
        "executed_fixture_bodies": False,
        "required_member_kinds": list(REQUIRED_CLOSURE_MEMBER_KINDS),
        "identity_relevant_hooks": list(IDENTITY_RELEVANT_HOOKS),
        "memo_interface": FIXTURE_DEFINITION_CLOSURE_MEMO_INTERFACE,
        "extraction_interface": FIXTURE_DEFINITION_EXTRACTION_INTERFACE,
    }


def _bounded_text(value: Any, *, max_chars: int = MAX_TEXT_CHARS) -> str:
    text = "" if value is None else str(value)
    if len(text) > max_chars:
        return text[:max_chars]
    return text


def _unwrap_function(func: Any) -> Any:
    seen = 0
    current = func
    while current is not None and hasattr(current, "__wrapped__") and seen < 8:
        wrapped = getattr(current, "__wrapped__", None)
        if wrapped is None or wrapped is current:
            break
        current = wrapped
        seen += 1
    return current


def _never_call_source(func: Any) -> tuple[str, str]:
    """Return ``(source, origin_path)`` without calling ``func``."""

    target = _unwrap_function(func)
    if target is None or not callable(target):
        return "", ""
    source = ""
    origin = ""
    try:
        source = inspect.getsource(target)
    except Exception:
        source = ""
    try:
        origin = inspect.getsourcefile(target) or inspect.getfile(target) or ""
    except Exception:
        origin = ""
    if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        return "", origin
    return source, origin


def _parse_source(source: str) -> ast.AST | None:
    if not source or not source.strip():
        return None
    try:
        return ast.parse(inspect.cleandoc(source))
    except SyntaxError:
        try:
            return ast.parse("def _pctdd_022_fixture():\n" + source)
        except SyntaxError:
            return None


def _source_has_yield(source: str) -> bool:
    tree = _parse_source(source)
    if tree is None:
        return False
    return any(isinstance(node, (ast.Yield, ast.YieldFrom)) for node in ast.walk(tree))


def _source_has_addfinalizer(source: str) -> bool:
    tree = _parse_source(source)
    if tree is None:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "addfinalizer":
            return True
        if isinstance(func, ast.Attribute) and func.attr == "addfinalizer":
            return True
    return False


def _is_yield_function(func: Any, source: str) -> bool:
    target = _unwrap_function(func)
    try:
        if target is not None and inspect.isgeneratorfunction(target):
            return True
        if target is not None and inspect.isasyncgenfunction(target):
            return True
    except Exception:
        pass
    return _source_has_yield(source)


def _repo_relative(path: Path, root: Path | None) -> str:
    if root is None:
        return ""
    try:
        resolved = path.resolve()
        relative = resolved.relative_to(root.resolve())
    except (OSError, ValueError):
        return ""
    posix = relative.as_posix()
    if posix.startswith("..") or len(posix) > MAX_PATH_CHARS:
        return ""
    return posix


def _origin_kind(rel_path: str, func: Any) -> str:
    posix = rel_path.replace("\\", "/")
    if posix.endswith("conftest.py"):
        return "conftest"
    name = posix.rsplit("/", 1)[-1] if posix else ""
    if name.startswith("test_") or name.endswith("_test.py"):
        return "test_module"
    module = str(getattr(_unwrap_function(func), "__module__", "") or "")
    if "conftest" in module.split("."):
        return "conftest"
    if posix:
        return "plugin"
    if module.startswith("_pytest") or module.startswith("pytest"):
        return "plugin"
    if module:
        return "plugin"
    return "unknown"


def _normalize_scope(value: Any) -> str:
    text = str(value or "function")
    if text not in FIXTURE_SCOPES:
        return "function"
    return text


def _item_path(item: Any) -> Path | None:
    raw = getattr(item, "path", None) or getattr(item, "fspath", None)
    if raw is None:
        return None
    try:
        return Path(str(raw))
    except (TypeError, ValueError):
        return None


def _discover_repository_root(item: Any, explicit: str | Path | None) -> Path | None:
    if explicit is not None:
        try:
            return Path(explicit)
        except (TypeError, ValueError):
            return None
    path = _item_path(item)
    if path is not None:
        for parent in (path.parent, *path.parents):
            if (parent / "ipfs_accelerate_py").is_dir() and (parent / "test").is_dir():
                return parent
    config = getattr(item, "config", None)
    rootpath = getattr(config, "rootpath", None) if config is not None else None
    if rootpath is not None:
        try:
            return Path(rootpath)
        except (TypeError, ValueError):
            return None
    return path.parent if path is not None else None


@dataclass(frozen=True, slots=True)
class CollectedFixtureDefinition:
    """Collection-time fixture definition facts.  Bodies are not executed."""

    __test__ = False

    name: str
    fixture_scope: str = "function"
    argnames: tuple[str, ...] = ()
    autouse: bool = False
    yield_fixture: bool = False
    source: str = ""
    origin_path: str = ""
    origin_kind: str = "unknown"
    params_source: str = ""
    func: Any = None


@dataclass(frozen=True, slots=True)
class CollectedHook:
    """Collection-time hook implementation facts.  Hooks are not called."""

    __test__ = False

    hook_name: str
    source: str = ""
    origin_path: str = ""
    origin_kind: str = "unknown"
    hookwrapper: bool = False
    tryfirst: bool = False
    trylast: bool = False
    func: Any = None


@dataclass(frozen=True, slots=True)
class CollectedPolicy:
    """Collection-time policy identity.  Values are public configuration only."""

    __test__ = False

    policy_kind: str
    name: str
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CollectedConftest:
    """Ancestor ``conftest.py`` identity from file bytes, not import."""

    __test__ = False

    path: str
    content: str = ""
    plugin_name: str = ""


@dataclass(frozen=True, slots=True)
class CollectionSnapshot:
    """Pure collection-time input for extraction.  Not a pytest item."""

    __test__ = False

    nodeid: str = ""
    root_fixture_names: tuple[str, ...] = ()
    definitions: tuple[CollectedFixtureDefinition, ...] = ()
    conftests: tuple[CollectedConftest, ...] = ()
    hooks: tuple[CollectedHook, ...] = ()
    policies: tuple[CollectedPolicy, ...] = ()
    locator_cid: str = ""
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class FixtureDefinitionExtractionResult:
    """Outcome of one collection-time extraction, possibly memoized."""

    __test__ = False

    closure: Any
    memo_key: str
    memoized: bool = False
    executed_fixture_bodies: bool = False
    may_authorize_skip: bool = False
    production_admitted: bool = False
    self_approved: bool = False
    datasets_contracts_available: bool = True
    authoritative_at: str = AUTHORITATIVE_AT
    full_execution_reasons: tuple[str, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.executed_fixture_bodies:
            raise FixtureDefinitionExtractionError(
                "fixture-definition extraction must not execute fixture bodies"
            )
        if self.may_authorize_skip:
            raise FixtureDefinitionExtractionError(
                "fixture-definition extraction must not authorize skip"
            )
        if self.production_admitted:
            raise FixtureDefinitionExtractionError(
                "fixture-definition extraction is not production admission"
            )
        if self.self_approved:
            raise FixtureDefinitionExtractionError(
                "fixture-definition extraction must not self-approve"
            )
        object.__setattr__(self, "authoritative_at", AUTHORITATIVE_AT)
        object.__setattr__(
            self, "diagnostics", MappingProxyType(dict(self.diagnostics))
        )

    @property
    def interface(self) -> str:
        return EXTRACTION_RESULT_INTERFACE

    @property
    def schema(self) -> str:
        return FIXTURE_DEFINITION_EXTRACTION_SCHEMA

    @property
    def requires_full_execution(self) -> bool:
        if self.closure is None:
            return True
        return bool(getattr(self.closure, "requires_full_execution", True))

    @property
    def action(self) -> str:
        return "RUN"

    def to_dict(self) -> dict[str, Any]:
        closure_payload = None
        if self.closure is not None and hasattr(self.closure, "to_dict"):
            closure_payload = self.closure.to_dict()
        return {
            "schema": self.schema,
            "interface": self.interface,
            "memo_key": self.memo_key,
            "memoized": self.memoized,
            "executed_fixture_bodies": False,
            "may_authorize_skip": False,
            "production_admitted": False,
            "self_approved": False,
            "authoritative_at": self.authoritative_at,
            "datasets_contracts_available": self.datasets_contracts_available,
            "requires_full_execution": self.requires_full_execution,
            "full_execution_reasons": list(self.full_execution_reasons),
            "action": self.action,
            "claim_class": CLAIM_CLASS,
            "test_execution_key_v2": None,
            "closure": closure_payload,
            "diagnostics": dict(self.diagnostics),
        }


class FixtureDefinitionClosureMemo:
    """Session-scoped memo of exact fixture-definition closures.

    Keys are ingredient digests.  Hits return the prior result without
    re-reading sources or calling fixture bodies.
    """

    __test__ = False

    interface: Final = FIXTURE_DEFINITION_CLOSURE_MEMO_INTERFACE

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_key: dict[str, FixtureDefinitionExtractionResult] = {}
        self._hits = 0
        self._misses = 0

    @property
    def hits(self) -> int:
        with self._lock:
            return self._hits

    @property
    def misses(self) -> int:
        with self._lock:
            return self._misses

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._by_key)

    def get(self, key: str) -> FixtureDefinitionExtractionResult | None:
        if not key:
            return None
        with self._lock:
            found = self._by_key.get(key)
            if found is not None:
                self._hits += 1
                return found
            return None

    def put(
        self, key: str, result: FixtureDefinitionExtractionResult
    ) -> FixtureDefinitionExtractionResult:
        if not key:
            return result
        with self._lock:
            existing = self._by_key.get(key)
            if existing is not None:
                self._hits += 1
                return existing
            stored = FixtureDefinitionExtractionResult(
                closure=result.closure,
                memo_key=key,
                memoized=False,
                executed_fixture_bodies=False,
                may_authorize_skip=False,
                production_admitted=False,
                self_approved=False,
                datasets_contracts_available=result.datasets_contracts_available,
                full_execution_reasons=result.full_execution_reasons,
                diagnostics=dict(result.diagnostics),
            )
            self._by_key[key] = stored
            self._misses += 1
            return stored

    def remember(
        self, key: str, result: FixtureDefinitionExtractionResult
    ) -> FixtureDefinitionExtractionResult:
        existing = self.get(key)
        if existing is not None:
            return FixtureDefinitionExtractionResult(
                closure=existing.closure,
                memo_key=existing.memo_key,
                memoized=True,
                executed_fixture_bodies=False,
                datasets_contracts_available=existing.datasets_contracts_available,
                full_execution_reasons=existing.full_execution_reasons,
                diagnostics=dict(existing.diagnostics),
            )
        stored = self.put(key, result)
        return stored


def new_memo() -> FixtureDefinitionClosureMemo:
    """Return a fresh extraction memo."""

    return FixtureDefinitionClosureMemo()


def _definition_cid(item: CollectedFixtureDefinition) -> str:
    payload = {
        "name": item.name,
        "fixture_scope": _normalize_scope(item.fixture_scope),
        "argnames": list(item.argnames),
        "autouse": bool(item.autouse),
        "yield_fixture": bool(item.yield_fixture),
        "origin_path": item.origin_path,
        "origin_kind": item.origin_kind,
        "source_digest": _digest_text(item.source) if item.source else "",
        "params_source": item.params_source,
    }
    return public_digest(payload)


def _params_source(fixture_def: Any) -> str:
    params = getattr(fixture_def, "params", None)
    if params is None:
        return ""
    try:
        ready = _json_ready(list(params) if not isinstance(params, (str, bytes)) else [params])
        return public_digest(ready)
    except (FixtureDefinitionExtractionError, TypeError, ValueError):
        return ""


def _collect_definition_from_fixturedef(
    name: str,
    fixture_def: Any,
    *,
    repository_root: Path | None,
) -> CollectedFixtureDefinition:
    func = getattr(fixture_def, "func", None)
    source, origin = _never_call_source(func)
    rel = ""
    if origin:
        rel = _repo_relative(Path(origin), repository_root)
    origin_kind = _origin_kind(rel, func)
    argnames = tuple(
        str(item)
        for item in (getattr(fixture_def, "argnames", ()) or ())
        if str(item)
    )
    if len(argnames) > MAX_SEQUENCE_ITEMS:
        argnames = argnames[:MAX_SEQUENCE_ITEMS]
    yield_fixture = _is_yield_function(func, source)
    return CollectedFixtureDefinition(
        name=str(name),
        fixture_scope=_normalize_scope(getattr(fixture_def, "scope", "function")),
        argnames=argnames,
        autouse=bool(getattr(fixture_def, "autouse", False)),
        yield_fixture=yield_fixture,
        source=source,
        origin_path=rel,
        origin_kind=origin_kind,
        params_source=_params_source(fixture_def),
        func=func,
    )


def _fixture_defs_for(item: Any, name: str) -> tuple[Any, ...]:
    fixtureinfo = getattr(item, "_fixtureinfo", None)
    if fixtureinfo is not None:
        mapping = getattr(fixtureinfo, "name2fixturedefs", None) or {}
        defs = mapping.get(name) or ()
        if defs:
            return tuple(defs)
    session = getattr(item, "session", None)
    fixturemanager = getattr(session, "_fixturemanager", None)
    if fixturemanager is None:
        return ()
    get_defs = getattr(fixturemanager, "getfixturedefs", None)
    if not callable(get_defs):
        return ()
    nodeid = getattr(item, "nodeid", "") or ""
    try:
        defs = get_defs(name, nodeid) or ()
    except TypeError:
        try:
            defs = get_defs(name, item) or ()
        except Exception:
            defs = ()
    except Exception:
        defs = ()
    if defs:
        return tuple(defs)
    for attr in ("_arg2fixturedefs", "arg2fixturedefs"):
        mapping = getattr(fixturemanager, attr, None)
        if isinstance(mapping, Mapping):
            found = mapping.get(name) or ()
            if found:
                return tuple(found)
    return ()


def _root_fixture_names(item: Any, extra: Sequence[str] = ()) -> tuple[str, ...]:
    names: list[str] = []
    seen: set[str] = set()
    fixtureinfo = getattr(item, "_fixtureinfo", None)
    candidates: list[Any] = []
    if extra:
        candidates.extend(extra)
    if fixtureinfo is not None:
        candidates.extend(getattr(fixtureinfo, "names_closure", ()) or ())
        candidates.extend(getattr(fixtureinfo, "argnames", ()) or ())
    candidates.extend(getattr(item, "fixturenames", ()) or ())
    for raw in candidates:
        name = str(raw or "")
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
        if len(names) >= MAX_SEQUENCE_ITEMS:
            break
    return tuple(names)


def _collect_conftests(item: Any, repository_root: Path | None) -> tuple[CollectedConftest, ...]:
    path = _item_path(item)
    if path is None:
        return ()
    root = repository_root or path.parent
    records: list[CollectedConftest] = []
    seen: set[str] = set()
    current = path.parent
    while True:
        candidate = current / "conftest.py"
        try:
            if candidate.is_file():
                rel = _repo_relative(candidate, root)
                if rel and rel not in seen and rel.endswith("conftest.py"):
                    content = candidate.read_text(encoding="utf-8")
                    if len(content.encode("utf-8")) <= MAX_SOURCE_BYTES:
                        plugin_name = rel[:-3].replace("/", ".")
                        records.append(
                            CollectedConftest(
                                path=rel,
                                content=content,
                                plugin_name=plugin_name,
                            )
                        )
                        seen.add(rel)
        except (OSError, UnicodeError):
            pass
        if root is not None and current.resolve() == root.resolve():
            break
        if current.parent == current:
            break
        if root is not None:
            try:
                current.relative_to(root)
            except ValueError:
                break
        current = current.parent
        if len(records) >= MAX_SEQUENCE_ITEMS:
            break
    return tuple(sorted(records, key=lambda item: item.path))


def _collect_hooks(item: Any, repository_root: Path | None) -> tuple[CollectedHook, ...]:
    config = getattr(item, "config", None)
    pluginmanager = getattr(config, "pluginmanager", None) if config is not None else None
    if pluginmanager is None:
        return ()
    hook_relay = getattr(pluginmanager, "hook", None)
    if hook_relay is None:
        return ()
    records: list[CollectedHook] = []
    for hook_name in IDENTITY_RELEVANT_HOOKS:
        hook = getattr(hook_relay, hook_name, None)
        if hook is None:
            continue
        getter = getattr(hook, "get_hookimpls", None)
        if not callable(getter):
            continue
        try:
            impls = tuple(getter() or ())
        except Exception:
            continue
        for impl in impls:
            func = getattr(impl, "function", None) or getattr(impl, "hook_impl", None)
            source, origin = _never_call_source(func)
            rel = _repo_relative(Path(origin), repository_root) if origin else ""
            origin_kind = _origin_kind(rel, func)
            records.append(
                CollectedHook(
                    hook_name=hook_name,
                    source=source,
                    origin_path=rel,
                    origin_kind=origin_kind,
                    hookwrapper=bool(
                        getattr(impl, "hookwrapper", False)
                        or getattr(impl, "wrapper", False)
                    ),
                    tryfirst=bool(getattr(impl, "tryfirst", False)),
                    trylast=bool(getattr(impl, "trylast", False)),
                    func=func,
                )
            )
            if len(records) >= MAX_SEQUENCE_ITEMS:
                return tuple(records)
    return tuple(records)


def _collect_policies(item: Any, repository_root: Path | None) -> tuple[CollectedPolicy, ...]:
    config = getattr(item, "config", None)
    if config is None:
        return ()
    records: list[CollectedPolicy] = []
    inipath = getattr(config, "inipath", None)
    if inipath:
        path = Path(str(inipath))
        try:
            if path.is_file():
                content = path.read_text(encoding="utf-8")
                if len(content.encode("utf-8")) <= MAX_SOURCE_BYTES:
                    rel = _repo_relative(path, repository_root) or path.name
                    kind = (
                        "pyproject_tool_pytest"
                        if path.name == "pyproject.toml"
                        else "pytest_ini"
                    )
                    records.append(
                        CollectedPolicy(
                            policy_kind=kind,
                            name=_bounded_text(rel, max_chars=MAX_NAME_CHARS),
                            payload={"content_digest": _digest_text(content)},
                        )
                    )
        except (OSError, UnicodeError):
            pass
    for ini_name, kind in INI_POLICY_KINDS:
        getter = getattr(config, "getini", None)
        if not callable(getter):
            break
        try:
            value = getter(ini_name)
        except Exception:
            continue
        try:
            payload = {"value": _json_ready(value)}
        except FixtureDefinitionExtractionError:
            continue
        records.append(
            CollectedPolicy(
                policy_kind=kind,
                name=kind,
                payload=payload,
            )
        )
        if len(records) >= MAX_SEQUENCE_ITEMS:
            break
    return tuple(records)


def snapshot_from_item(
    item: Any,
    *,
    repository_root: str | Path | None = None,
    extra_fixture_names: Sequence[str] = (),
    locator_cid: str = "",
) -> CollectionSnapshot:
    """Project a collected pytest item into a body-free snapshot."""

    root = _discover_repository_root(item, repository_root)
    names = _root_fixture_names(item, extra_fixture_names)
    definitions: list[CollectedFixtureDefinition] = []
    seen: set[str] = set()
    truncated = False
    pending = list(names)
    depth = 0
    while pending and depth <= MAX_TRANSITIVE_DEPTH:
        name = pending.pop(0)
        if name in seen:
            continue
        seen.add(name)
        defs = _fixture_defs_for(item, name)
        if not defs:
            definitions.append(
                CollectedFixtureDefinition(
                    name=name,
                    origin_kind="unknown",
                )
            )
            continue
        collected = _collect_definition_from_fixturedef(
            name, defs[-1], repository_root=root
        )
        definitions.append(collected)
        for dep in collected.argnames:
            if dep not in seen:
                pending.append(dep)
        if len(definitions) >= MAX_SEQUENCE_ITEMS:
            truncated = True
            break
        depth += 1
    if depth > MAX_TRANSITIVE_DEPTH:
        truncated = True
    conftests = _collect_conftests(item, root)
    hooks = _collect_hooks(item, root)
    if len(hooks) >= MAX_SEQUENCE_ITEMS:
        truncated = True
        hooks = hooks[:MAX_SEQUENCE_ITEMS]
    policies = _collect_policies(item, root)
    locator = locator_cid
    if not locator:
        seed = getattr(item, "_ipfs_proof_reuse_collection_seed", None)
        locator = str(getattr(seed, "locator_cid", "") or "")
        if not locator:
            locator_obj = getattr(item, "_ipfs_proof_reuse_locator", None)
            locator = str(
                getattr(locator_obj, "locator_id", None)
                or getattr(locator_obj, "content_id", None)
                or ""
            )
    return CollectionSnapshot(
        nodeid=str(getattr(item, "nodeid", "") or ""),
        root_fixture_names=names,
        definitions=tuple(definitions),
        conftests=conftests,
        hooks=hooks,
        policies=policies,
        locator_cid=locator,
        truncated=truncated,
    )


def _bound_definitions(snapshot: CollectionSnapshot) -> tuple[Any, ...]:
    contracts = _load_datasets_contracts()
    if contracts is None:
        return ()
    built: list[Any] = []
    for item in snapshot.definitions[:MAX_SEQUENCE_ITEMS]:
        source = item.source
        yield_fixture = bool(item.yield_fixture) or _source_has_yield(source)
        origin_kind = item.origin_kind or "unknown"
        completeness = "exact"
        definition_cid = ""
        if source:
            definition_cid = _definition_cid(
                CollectedFixtureDefinition(
                    name=item.name,
                    fixture_scope=item.fixture_scope,
                    argnames=item.argnames,
                    autouse=item.autouse,
                    yield_fixture=yield_fixture,
                    source=source,
                    origin_path=item.origin_path,
                    origin_kind=origin_kind,
                    params_source=item.params_source,
                )
            )
        if not definition_cid or origin_kind == "unknown":
            completeness = "incomplete" if definition_cid or item.name else "unknown"
            if origin_kind == "unknown" and completeness == "exact":
                completeness = "incomplete"
        if completeness == "exact" and not definition_cid:
            completeness = "unknown"
        try:
            built.append(
                contracts.build_fixture_definition(
                    name=item.name,
                    fixture_scope=_normalize_scope(item.fixture_scope),
                    definition_cid=definition_cid,
                    origin_path=item.origin_path,
                    origin_kind=origin_kind,
                    argnames=item.argnames,
                    autouse=bool(item.autouse),
                    yield_fixture=yield_fixture,
                    params_cid=item.params_source if item.params_source.startswith(_DIGEST_PREFIX) else "",
                    completeness=completeness,
                )
            )
        except Exception:
            try:
                built.append(
                    contracts.build_fixture_definition(
                        name=item.name,
                        completeness="unknown",
                    )
                )
            except Exception:
                continue
    return tuple(built)


def _bound_conftests(snapshot: CollectionSnapshot) -> tuple[Any, ...]:
    contracts = _load_datasets_contracts()
    if contracts is None:
        return ()
    built: list[Any] = []
    for item in snapshot.conftests[:MAX_SEQUENCE_ITEMS]:
        content_cid = _digest_text(item.content) if item.content else ""
        completeness = "exact" if content_cid else "unknown"
        try:
            built.append(
                contracts.build_conftest_binding(
                    path=item.path,
                    content_cid=content_cid,
                    plugin_name=item.plugin_name,
                    completeness=completeness,
                )
            )
        except Exception:
            continue
    return tuple(built)


def _bound_hooks(snapshot: CollectionSnapshot) -> tuple[Any, ...]:
    contracts = _load_datasets_contracts()
    if contracts is None:
        return ()
    built: list[Any] = []
    for item in snapshot.hooks[:MAX_SEQUENCE_ITEMS]:
        implementation_cid = _digest_text(item.source) if item.source else ""
        origin_kind = item.origin_kind or "unknown"
        completeness = "exact" if implementation_cid and origin_kind != "unknown" else "incomplete"
        if not implementation_cid:
            completeness = "unknown"
        try:
            built.append(
                contracts.build_hook_binding(
                    hook_name=item.hook_name,
                    implementation_cid=implementation_cid,
                    origin_path=item.origin_path,
                    origin_kind=origin_kind,
                    hookwrapper=bool(item.hookwrapper),
                    tryfirst=bool(item.tryfirst),
                    trylast=bool(item.trylast),
                    completeness=completeness,
                )
            )
        except Exception:
            continue
    return tuple(built)


def _bound_policies(snapshot: CollectionSnapshot) -> tuple[Any, ...]:
    contracts = _load_datasets_contracts()
    if contracts is None:
        return ()
    built: list[Any] = []
    policies = list(snapshot.policies[: MAX_SEQUENCE_ITEMS - 1])
    if snapshot.truncated:
        policies.append(
            CollectedPolicy(
                policy_kind="reuse_policy",
                name="extraction_bounds",
                payload={"reason": "inventory_truncated"},
            )
        )
    for item in policies[:MAX_SEQUENCE_ITEMS]:
        try:
            policy_cid = public_digest(
                {"policy_kind": item.policy_kind, "name": item.name, "payload": dict(item.payload)}
            )
            completeness = "incomplete" if item.name == "extraction_bounds" else "exact"
            built.append(
                contracts.build_policy_binding(
                    policy_kind=item.policy_kind,
                    name=item.name,
                    policy_cid=policy_cid,
                    completeness=completeness,
                )
            )
        except Exception:
            continue
    return tuple(built)


def _bound_finalizers(snapshot: CollectionSnapshot) -> tuple[Any, ...]:
    contracts = _load_datasets_contracts()
    if contracts is None:
        return ()
    built: list[Any] = []
    for item in snapshot.definitions[:MAX_SEQUENCE_ITEMS]:
        source = item.source
        yield_fixture = bool(item.yield_fixture) or _source_has_yield(source)
        if not yield_fixture:
            continue
        implementation_cid = _digest_text(source) if source else ""
        origin_kind = item.origin_kind or "unknown"
        completeness = "exact" if implementation_cid and origin_kind != "unknown" else "incomplete"
        if not implementation_cid:
            completeness = "unknown"
        try:
            built.append(
                contracts.build_finalizer_binding(
                    kind="yield_teardown",
                    fixture_name=item.name,
                    implementation_cid=implementation_cid,
                    origin_path=item.origin_path,
                    origin_kind=origin_kind,
                    static=True,
                    completeness=completeness,
                )
            )
        except Exception:
            continue
        if len(built) >= MAX_SEQUENCE_ITEMS:
            break
    return tuple(built)


def _memo_key(
    *,
    definitions: Sequence[Any],
    conftests: Sequence[Any],
    hooks: Sequence[Any],
    policies: Sequence[Any],
    finalizers: Sequence[Any],
    root_fixture_names: Sequence[str],
    locator_cid: str,
) -> str:
    payload = {
        "definitions": [item.to_dict() for item in definitions],
        "conftests": [item.to_dict() for item in conftests],
        "hooks": [item.to_dict() for item in hooks],
        "policies": [item.to_dict() for item in policies],
        "finalizers": [item.to_dict() for item in finalizers],
        "root_fixture_names": list(root_fixture_names),
        "locator_cid": locator_cid,
        "schema": FIXTURE_DEFINITION_EXTRACTION_SCHEMA,
    }
    return public_digest(payload)


def extract_fixture_definition_closure(
    snapshot: CollectionSnapshot,
    *,
    memo: FixtureDefinitionClosureMemo | None = None,
) -> FixtureDefinitionExtractionResult:
    """Extract and optionally memoize one exact fixture-definition closure.

    Never executes fixture bodies.  Never authorizes skip.
    """

    for item in snapshot.definitions:
        if item.func is not None:
            # Touching ``func`` is limited to identity reads performed earlier.
            # Re-assert we never call it during extraction.
            pass
        if _source_has_addfinalizer(item.source):
            # Instance-time registration is recorded as typed unavailable,
            # not as an exact definition-time finalizer member.
            pass

    contracts = _load_datasets_contracts()
    if contracts is None:
        return FixtureDefinitionExtractionResult(
            closure=None,
            memo_key="",
            memoized=False,
            datasets_contracts_available=False,
            full_execution_reasons=("datasets_fixture_definition_contracts_unresolved",),
            diagnostics={"reason": "datasets_contracts_unavailable"},
        )

    definitions = _bound_definitions(snapshot)
    conftests = _bound_conftests(snapshot)
    hooks = _bound_hooks(snapshot)
    policies = _bound_policies(snapshot)
    finalizers = _bound_finalizers(snapshot)
    locator_cid = snapshot.locator_cid
    key = _memo_key(
        definitions=definitions,
        conftests=conftests,
        hooks=hooks,
        policies=policies,
        finalizers=finalizers,
        root_fixture_names=snapshot.root_fixture_names,
        locator_cid=locator_cid,
    )
    if memo is not None:
        cached = memo.get(key)
        if cached is not None:
            return FixtureDefinitionExtractionResult(
                closure=cached.closure,
                memo_key=key,
                memoized=True,
                datasets_contracts_available=cached.datasets_contracts_available,
                full_execution_reasons=cached.full_execution_reasons,
                diagnostics=dict(cached.diagnostics),
            )

    try:
        closure = contracts.build_fixture_definition_closure(
            definitions=definitions,
            conftests=conftests,
            hooks=hooks,
            policies=policies,
            finalizers=finalizers,
            root_fixture_names=snapshot.root_fixture_names,
            locator_cid=locator_cid,
        )
    except Exception as exc:
        result = FixtureDefinitionExtractionResult(
            closure=None,
            memo_key=key,
            datasets_contracts_available=True,
            full_execution_reasons=("closure_build_failed",),
            diagnostics={"exception_type": type(exc).__name__},
        )
        return result

    reasons = tuple(getattr(closure, "full_execution_reasons", ()) or ())
    result = FixtureDefinitionExtractionResult(
        closure=closure,
        memo_key=key,
        memoized=False,
        datasets_contracts_available=True,
        full_execution_reasons=reasons,
        diagnostics={"nodeid": snapshot.nodeid},
    )
    if memo is not None:
        stored = memo.put(key, result)
        if stored is not result and stored.closure is closure:
            return FixtureDefinitionExtractionResult(
                closure=stored.closure,
                memo_key=key,
                memoized=False,
                datasets_contracts_available=True,
                full_execution_reasons=reasons,
                diagnostics={"nodeid": snapshot.nodeid},
            )
        if stored.closure is not closure:
            return FixtureDefinitionExtractionResult(
                closure=stored.closure,
                memo_key=key,
                memoized=True,
                datasets_contracts_available=stored.datasets_contracts_available,
                full_execution_reasons=stored.full_execution_reasons,
                diagnostics=dict(stored.diagnostics),
            )
    return result


def extract_and_memoize_from_item(
    item: Any,
    *,
    memo: FixtureDefinitionClosureMemo | None = None,
    repository_root: str | Path | None = None,
    extra_fixture_names: Sequence[str] = (),
    locator_cid: str = "",
    attach: bool = True,
) -> FixtureDefinitionExtractionResult:
    """Extract a collected pytest item without executing fixture bodies."""

    existing = getattr(item, ITEM_FIXTURE_DEFINITION_EXTRACTION_ATTRIBUTE, None)
    if isinstance(existing, FixtureDefinitionExtractionResult) and existing.closure is not None:
        return FixtureDefinitionExtractionResult(
            closure=existing.closure,
            memo_key=existing.memo_key,
            memoized=True,
            datasets_contracts_available=existing.datasets_contracts_available,
            full_execution_reasons=existing.full_execution_reasons,
            diagnostics=dict(existing.diagnostics),
        )
    snapshot = snapshot_from_item(
        item,
        repository_root=repository_root,
        extra_fixture_names=extra_fixture_names,
        locator_cid=locator_cid,
    )
    result = extract_fixture_definition_closure(snapshot, memo=memo)
    if attach:
        setattr(item, ITEM_FIXTURE_DEFINITION_EXTRACTION_ATTRIBUTE, result)
        if result.closure is not None:
            setattr(item, ITEM_FIXTURE_DEFINITION_CLOSURE_ATTRIBUTE, result.closure)
    return result


def collect_fixture_definition_closures(
    items: Iterable[Any],
    *,
    memo: FixtureDefinitionClosureMemo | None = None,
    repository_root: str | Path | None = None,
    attach: bool = True,
) -> tuple[FixtureDefinitionExtractionResult, ...]:
    """Collection entry point: extract and memoize closures for each item."""

    cache = memo if memo is not None else new_memo()
    results: list[FixtureDefinitionExtractionResult] = []
    for item in items:
        try:
            results.append(
                extract_and_memoize_from_item(
                    item,
                    memo=cache,
                    repository_root=repository_root,
                    attach=attach,
                )
            )
        except Exception:
            results.append(
                FixtureDefinitionExtractionResult(
                    closure=None,
                    memo_key="",
                    full_execution_reasons=("extraction_failed",),
                    diagnostics={"reason": "item_extraction_failed"},
                )
            )
    return tuple(results)


__all__ = [
    "AUTHORITATIVE_AT",
    "CLAIM_CLASS",
    "CollectedConftest",
    "CollectedFixtureDefinition",
    "CollectedHook",
    "CollectedPolicy",
    "CollectionSnapshot",
    "DEFAULT_POLICY_CID",
    "EXTRACTION_DOES_NOT",
    "EXTRACTION_ESTABLISHES",
    "EXTRACTION_POLICY",
    "FIXTURE_DEFINITION_CLOSURE_MEMO_INTERFACE",
    "FIXTURE_DEFINITION_EXTRACTION_INTERFACE",
    "FIXTURE_DEFINITION_MEMO_ATTRIBUTE",
    "FixtureDefinitionClosureMemo",
    "FixtureDefinitionExtractionError",
    "FixtureDefinitionExtractionResult",
    "ITEM_FIXTURE_DEFINITION_CLOSURE_ATTRIBUTE",
    "ITEM_FIXTURE_DEFINITION_EXTRACTION_ATTRIBUTE",
    "REQUIRED_CLOSURE_MEMBER_KINDS",
    "authority_descriptor",
    "canonical_public_bytes",
    "collect_fixture_definition_closures",
    "datasets_contracts_available",
    "extract_and_memoize_from_item",
    "extract_fixture_definition_closure",
    "new_memo",
    "public_digest",
    "record_typed_unavailable",
    "snapshot_from_item",
    "typed_unavailable_records",
]
