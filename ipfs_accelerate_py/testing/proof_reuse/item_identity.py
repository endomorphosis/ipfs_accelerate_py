"""Fail-closed automatic pytest item identity assembly.

This module is the collection-time glue between a direct pytest item and the
existing proof-reuse contracts.  It deliberately has a standard-library-only
import surface; AST, repository, trace, CID, eligibility, and lookup modules
are loaded only when :func:`assemble_and_attach_item_identity` is called.

There is no per-test path registry.  The assembler derives the repository
relative node, exact current AST identities, and static dependency trace from
the collected item.  Session-scoped providers supply the current repository
forest, complete fixture/dependency inventory, policy inputs, and (if one
exists) fresh controlled-preflight runtime evidence.

Important admission limitation
------------------------------

Normal pytest fixture values do not exist until setup, and
``RuntimeTestDependencyTrace@1`` is produced while a test executes.  Therefore
an authoritative execution key cannot normally be constructed before the
collection-time cache lookup.  A retained trace from a prior run is not current
evidence and is never accepted here.  Unless an injected provider returns a
``CurrentRuntimeTraceEvidence@1`` binding produced by a fresh controlled
preflight for the exact node + forest + static trace + component root + runtime
policy, assembly returns a typed non-reusable result and the test executes.

Identity admission is not skip authority.  Even a successful result has action
``RUN`` and merely attaches a lookup request for the existing local verifier.
Only that verifier can later return ``SKIP`` for an exact authoritative
certificate.
"""

from __future__ import annotations

import ast
import hashlib
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar, Final, Optional


AUTOMATIC_ITEM_IDENTITY_INTERFACE: Final = "AutomaticItemIdentityAssembler@1"
AUTOMATIC_ITEM_IDENTITY_RESULT_INTERFACE: Final = (
    "AutomaticItemIdentityAssembly@1"
)
CURRENT_RUNTIME_TRACE_EVIDENCE_INTERFACE: Final = (
    "CurrentRuntimeTraceEvidence@1"
)
CURRENT_ITEM_INPUTS_INTERFACE: Final = "CurrentItemIdentityInputs@1"

AUTOMATIC_ITEM_IDENTITY_SCHEMA: Final = (
    "ipfs_accelerate_py/testing/proof-reuse/automatic-item-identity@1"
)
CURRENT_RUNTIME_TRACE_EVIDENCE_SCHEMA: Final = (
    "ipfs_accelerate_py/testing/proof-reuse/current-runtime-trace-evidence@1"
)

ITEM_IDENTITY_RESULT_ATTRIBUTE: Final = "_ipfs_proof_reuse_identity_assembly"
ITEM_LOCATOR_ATTRIBUTE: Final = "_ipfs_proof_reuse_locator"
ITEM_EXECUTION_KEY_ATTRIBUTE: Final = "_ipfs_proof_reuse_execution_key"
ITEM_ELIGIBILITY_ATTRIBUTE: Final = "_ipfs_proof_reuse_eligibility"
ITEM_POLICY_ATTRIBUTE: Final = "_ipfs_proof_reuse_policy"

_MAX_SOURCE_BYTES: Final = 8 * 1_048_576
_MAX_FIXTURES: Final = 512
_MAX_MARKERS: Final = 256
_SAFE_VERSION_CHARS: Final = 128


class ItemIdentityAssemblyReason(str, Enum):
    """Closed, non-secret collection-time admission result."""

    ADMITTED_FOR_LOOKUP = "admitted_for_lookup"
    ITEM_DISABLED = "item_disabled"
    ITEM_UNSUPPORTED = "item_unsupported"
    ITEM_PATH_UNAVAILABLE = "item_path_unavailable"
    ITEM_PATH_OUTSIDE_FOREST = "item_path_outside_forest"
    EXISTING_IDENTITY_CONFLICT = "existing_identity_conflict"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    REPOSITORY_FOREST_UNAVAILABLE = "repository_forest_unavailable"
    REPOSITORY_FOREST_INCOMPLETE = "repository_forest_incomplete"
    AST_INDEX_UNAVAILABLE = "ast_index_unavailable"
    STATIC_TRACE_INCOMPLETE = "static_trace_incomplete"
    AST_IDENTITY_UNAVAILABLE = "ast_identity_unavailable"
    COMPONENT_INPUT_UNAVAILABLE = "component_input_unavailable"
    COMPONENT_INVENTORY_MISMATCH = "component_inventory_mismatch"
    COMPONENTS_NON_REUSABLE = "components_non_reusable"
    POLICY_INPUT_UNAVAILABLE = "policy_input_unavailable"
    RUNTIME_EVIDENCE_UNAVAILABLE = "runtime_evidence_unavailable"
    RUNTIME_EVIDENCE_NOT_CURRENT = "runtime_evidence_not_current"
    RUNTIME_TRACE_INCOMPLETE = "runtime_trace_incomplete"
    ELIGIBILITY_DENIED = "eligibility_denied"
    IDENTITY_COMPILER_REJECTED = "identity_compiler_rejected"
    ATTACHMENT_FAILED = "attachment_failed"
    INTERNAL_ERROR_FAIL_OPEN_TO_RUN = "internal_error_fail_open_to_run"


class CurrentInputCompleteness(str, Enum):
    """Provider assertion accepted only at the explicit DI boundary."""

    EXACT_CURRENT = "exact_current"


class RuntimeEvidenceProvenance(str, Enum):
    """Runtime evidence sources admitted before a lookup.

    A historical/cache provenance is intentionally absent from this enum.
    """

    FRESH_CONTROLLED_PREFLIGHT = "fresh_controlled_preflight"


@dataclass(frozen=True)
class CurrentItemComponentInputs:
    """Complete current component inventory returned by one session provider.

    ``fixtures`` must contain exactly the fixture names active for the item.
    ``conftests`` must contain exactly the existing ancestor ``conftest.py``
    files.  ``plugins`` and ``installed_distributions`` are required because an
    empty ambient inventory is not a safe default.
    """

    __test__: ClassVar[bool] = False

    completeness: CurrentInputCompleteness
    fixtures: tuple[Mapping[str, Any], ...] = ()
    conftests: tuple[Mapping[str, Any], ...] = ()
    hooks: tuple[Mapping[str, Any], ...] = ()
    plugins: tuple[Mapping[str, Any], ...] = ()
    lock_files: tuple[Mapping[str, Any], ...] = ()
    installed_distributions: tuple[tuple[str, str], ...] = ()
    environment: Optional[Mapping[str, str]] = None
    environment_allowlist: tuple[str, ...] = ()
    interpreter_facts: Optional[Mapping[str, Any]] = None
    platform_facts: Optional[Mapping[str, Any]] = None
    hardware_facts: Optional[Mapping[str, Any]] = None
    capability_facts: Optional[Mapping[str, Any]] = None
    capability_allowlist: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.completeness is not CurrentInputCompleteness.EXACT_CURRENT:
            raise ValueError("component inventory is not exact-current")
        for name in (
            "fixtures",
            "conftests",
            "hooks",
            "plugins",
            "lock_files",
            "installed_distributions",
        ):
            values = tuple(getattr(self, name) or ())
            if len(values) > _MAX_FIXTURES:
                raise ValueError("%s exceeds its bounded inventory" % name)
            object.__setattr__(self, name, values)
        if not self.plugins:
            raise ValueError("active pytest plugin inventory is required")
        if not self.installed_distributions:
            raise ValueError("installed distribution inventory is required")
        if self.environment is None:
            raise ValueError("current allowlisted environment snapshot is required")
        if not self.environment_allowlist:
            raise ValueError("reviewed environment allowlist is required")
        for name in ("interpreter_facts", "platform_facts", "hardware_facts"):
            if getattr(self, name) is None:
                raise ValueError("%s is required" % name)


@dataclass(frozen=True)
class CurrentItemPolicyInputs:
    """Exact current policy/configuration inputs used by lookup and key assembly."""

    __test__: ClassVar[bool] = False

    completeness: CurrentInputCompleteness
    policy_identity: Any
    verification_policy: Mapping[str, Any]
    reuse_policy: Any
    command_semantics: Mapping[str, Any]
    pytest_config: Mapping[str, Any]
    plugin_versions: Mapping[str, Any]
    runtime_completeness_policy: Mapping[str, Any]
    canonicalization_schema: Mapping[str, Any]
    tracer_schema: Mapping[str, Any]
    certificate_schema: Mapping[str, Any]
    snapshot_adapters: Mapping[str, str] = field(default_factory=dict)
    collection_schema_version: str = "1"

    def __post_init__(self) -> None:
        if self.completeness is not CurrentInputCompleteness.EXACT_CURRENT:
            raise ValueError("policy inventory is not exact-current")
        for name in (
            "verification_policy",
            "command_semantics",
            "pytest_config",
            "plugin_versions",
            "runtime_completeness_policy",
            "canonicalization_schema",
            "tracer_schema",
            "certificate_schema",
        ):
            value = getattr(self, name)
            if not isinstance(value, Mapping) or not value:
                raise ValueError("%s must be a nonempty mapping" % name)
        version = str(self.collection_schema_version or "").strip()
        if not version or len(version) > _SAFE_VERSION_CHARS:
            raise ValueError("collection_schema_version is invalid")
        object.__setattr__(self, "collection_schema_version", version)
        object.__setattr__(
            self, "snapshot_adapters", dict(self.snapshot_adapters or {})
        )

    def verified_identities(self) -> Mapping[str, Any]:
        """Verify retained policy bytes and mint every collection context CID."""

        from ...agent_supervisor.analysis.test_execution_identity import (
            ContentIdentity,
            mint_content_identity,
            reject_pseudo_cid,
        )
        from ...agent_supervisor.analysis.test_reuse_eligibility import (
            TestReuseEligibilityPolicy,
        )

        if not isinstance(self.policy_identity, ContentIdentity):
            raise ValueError("policy_identity must retain canonical bytes")
        self.policy_identity.verify()
        if not isinstance(self.reuse_policy, TestReuseEligibilityPolicy):
            raise ValueError("reuse_policy must be TestReuseEligibilityPolicy")
        verification = dict(self.verification_policy)
        if verification.get("policy_cid") != self.policy_identity.cid:
            raise ValueError("verification policy does not bind policy_identity")
        for key in (
            "policy_cid",
            "statement_cid",
            "circuit_cid",
            "verifying_key_cid",
        ):
            reject_pseudo_cid(str(verification.get(key) or ""), field_name=key)
        return {
            "policy": self.policy_identity,
            "command": mint_content_identity(dict(self.command_semantics)),
            "config": mint_content_identity(dict(self.pytest_config)),
            "plugins": mint_content_identity(dict(self.plugin_versions)),
            "runtime_policy": mint_content_identity(
                dict(self.runtime_completeness_policy)
            ),
            "canonicalization": mint_content_identity(
                dict(self.canonicalization_schema)
            ),
            "tracer": mint_content_identity(dict(self.tracer_schema)),
            "certificate": mint_content_identity(dict(self.certificate_schema)),
        }


@dataclass(frozen=True)
class CurrentRuntimeTraceEvidence:
    """Fresh-preflight binding for one exact current collection context.

    Construction verifies a domain-separated retained identity.  It does not
    infer freshness: the injected provider is responsible for performing the
    controlled preflight in the current session.  Raw
    ``RuntimeTestDependencyTrace`` values are rejected by the assembler.
    """

    __test__: ClassVar[bool] = False

    trace: Any
    node_id: str
    repository_forest_cid: str
    static_trace_root_cid: str
    identity_components_cid: str
    runtime_completeness_policy_cid: str
    binding_identity: Any
    provenance: RuntimeEvidenceProvenance = (
        RuntimeEvidenceProvenance.FRESH_CONTROLLED_PREFLIGHT
    )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": CURRENT_RUNTIME_TRACE_EVIDENCE_SCHEMA,
            "interface": CURRENT_RUNTIME_TRACE_EVIDENCE_INTERFACE,
            "provenance": self.provenance.value,
            "node_id": self.node_id,
            "repository_forest_cid": self.repository_forest_cid,
            "static_trace_root_cid": self.static_trace_root_cid,
            "identity_components_cid": self.identity_components_cid,
            "runtime_completeness_policy_cid": (
                self.runtime_completeness_policy_cid
            ),
            "runtime_trace_root_cid": str(getattr(self.trace, "trace_cid", "")),
        }

    def __post_init__(self) -> None:
        from ...agent_supervisor.analysis.test_execution_identity import (
            ContentIdentity,
            mint_content_identity,
            normalize_pytest_node_id,
        )
        from ...agent_supervisor.analysis.test_runtime_dependency_trace import (
            RuntimeTestDependencyTrace,
        )

        if self.provenance is not RuntimeEvidenceProvenance.FRESH_CONTROLLED_PREFLIGHT:
            raise ValueError("historical runtime trace provenance is not admitted")
        object.__setattr__(self, "node_id", normalize_pytest_node_id(self.node_id))
        if not isinstance(self.trace, RuntimeTestDependencyTrace):
            raise ValueError("trace must be RuntimeTestDependencyTrace")
        self.trace.verify()
        if not self.trace.complete:
            raise ValueError("runtime trace is incomplete")
        if not isinstance(self.binding_identity, ContentIdentity):
            raise ValueError("runtime binding must retain canonical bytes")
        self.binding_identity.verify()
        expected = mint_content_identity(self._payload())
        if (
            expected.cid != self.binding_identity.cid
            or expected.canonical_bytes != self.binding_identity.canonical_bytes
        ):
            raise ValueError("runtime evidence binding does not match its inputs")

    @classmethod
    def bind_fresh_preflight(
        cls,
        *,
        trace: Any,
        node_id: str,
        repository_forest_cid: str,
        static_trace_root_cid: str,
        identity_components_cid: str,
        runtime_completeness_policy_cid: str,
    ) -> "CurrentRuntimeTraceEvidence":
        """Bind a provider's just-completed controlled preflight.

        This method intentionally does not accept a provenance argument.
        """

        from ...agent_supervisor.analysis.test_execution_identity import (
            mint_content_identity,
            normalize_pytest_node_id,
        )

        normalized = normalize_pytest_node_id(node_id)
        payload = {
            "schema": CURRENT_RUNTIME_TRACE_EVIDENCE_SCHEMA,
            "interface": CURRENT_RUNTIME_TRACE_EVIDENCE_INTERFACE,
            "provenance": RuntimeEvidenceProvenance.FRESH_CONTROLLED_PREFLIGHT.value,
            "node_id": normalized,
            "repository_forest_cid": repository_forest_cid,
            "static_trace_root_cid": static_trace_root_cid,
            "identity_components_cid": identity_components_cid,
            "runtime_completeness_policy_cid": runtime_completeness_policy_cid,
            "runtime_trace_root_cid": str(getattr(trace, "trace_cid", "")),
        }
        return cls(
            trace=trace,
            node_id=normalized,
            repository_forest_cid=repository_forest_cid,
            static_trace_root_cid=static_trace_root_cid,
            identity_components_cid=identity_components_cid,
            runtime_completeness_policy_cid=runtime_completeness_policy_cid,
            binding_identity=mint_content_identity(payload),
        )

    def verify_current(
        self,
        *,
        node_id: str,
        repository_forest_cid: str,
        static_trace_root_cid: str,
        identity_components_cid: str,
        runtime_completeness_policy_cid: str,
    ) -> None:
        """Reject substitution across any exact current input."""

        self.__post_init__()
        expected = (
            node_id,
            repository_forest_cid,
            static_trace_root_cid,
            identity_components_cid,
            runtime_completeness_policy_cid,
        )
        actual = (
            self.node_id,
            self.repository_forest_cid,
            self.static_trace_root_cid,
            self.identity_components_cid,
            self.runtime_completeness_policy_cid,
        )
        if actual != expected:
            raise ValueError("runtime evidence does not bind current item inputs")


@dataclass(frozen=True)
class ItemIdentityAssemblyServices:
    """Session-scoped providers; none is called during module import."""

    __test__: ClassVar[bool] = False

    repository_forest_provider: Optional[Callable[[Any], Any]] = None
    analysis_index_provider: Optional[Callable[[Any, Any], Any]] = None
    component_inputs_provider: Optional[
        Callable[[Any, Any, Any, Any], Any]
    ] = None
    policy_inputs_provider: Optional[
        Callable[[Any, Any, Any, Any, Any], Any]
    ] = None
    runtime_evidence_provider: Optional[
        Callable[[Any, Any, Any, Any, Any, Any], Any]
    ] = None
    identity_compiler: Any = None


@dataclass(frozen=True)
class AutomaticItemIdentityAssembly:
    """Result of automatic assembly.  It never itself authorizes ``SKIP``."""

    __test__: ClassVar[bool] = False

    reason: ItemIdentityAssemblyReason
    stage: str
    locator_artifact: Any = None
    execution_artifact: Any = None
    eligibility: Any = None
    lookup_request: Any = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def interface(self) -> str:
        return AUTOMATIC_ITEM_IDENTITY_RESULT_INTERFACE

    @property
    def reusable(self) -> bool:
        return (
            self.reason is ItemIdentityAssemblyReason.ADMITTED_FOR_LOOKUP
            and self.lookup_request is not None
        )

    @property
    def admitted_for_lookup(self) -> bool:
        return self.reusable

    @property
    def action(self) -> str:
        # Identity assembly has no skip authority.
        return "RUN"

    @property
    def authorizes_skip(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface": self.interface,
            "reason": self.reason.value,
            "stage": self.stage,
            "reusable": self.reusable,
            "action": self.action,
            "authorizes_skip": False,
            "has_locator": self.locator_artifact is not None,
            "has_execution_key": self.execution_artifact is not None,
            "has_lookup_request": self.lookup_request is not None,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class _ItemFacts:
    path: Path
    relative_path: str
    node_id: str
    function_name: str
    class_name: str
    parameter_id: str
    parameter_value: Any
    parameterized: bool
    fixture_names: tuple[str, ...]
    markers: tuple[str, ...]
    effect_adapters: tuple[str, ...]


@dataclass(frozen=True)
class _AstIdentities:
    module_cid: str
    class_cid: str
    function_cid: str
    decorator_cids: tuple[str, ...]
    test_ast_cid: str


def _failure(
    reason: ItemIdentityAssemblyReason,
    stage: str,
    **diagnostics: Any,
) -> AutomaticItemIdentityAssembly:
    bounded: dict[str, Any] = {}
    for key, value in list(diagnostics.items())[:16]:
        if value is None or isinstance(value, (bool, int)):
            bounded[str(key)[:64]] = value
        elif isinstance(value, str):
            bounded[str(key)[:64]] = value[:128]
        elif isinstance(value, (tuple, list)):
            bounded[str(key)[:64]] = [
                str(item)[:64] for item in list(value)[:16]
            ]
        else:
            bounded[str(key)[:64]] = type(value).__name__[:64]
    return AutomaticItemIdentityAssembly(
        reason=reason,
        stage=str(stage)[:64],
        diagnostics=bounded,
    )


def _call_provider(
    provider: Any,
    *arguments: Any,
    reason: ItemIdentityAssemblyReason,
    stage: str,
) -> tuple[Any, Optional[AutomaticItemIdentityAssembly]]:
    if not callable(provider):
        return None, _failure(
            ItemIdentityAssemblyReason.PROVIDER_UNAVAILABLE,
            stage,
            provider=stage,
        )
    try:
        return provider(*arguments), None
    except BaseException as exc:
        return None, _failure(
            reason,
            stage,
            exception_type=type(exc).__name__,
        )


def _path_under(path: Path, root: Path) -> Optional[str]:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return None


def _item_path(item: Any) -> Path:
    raw = getattr(item, "path", None)
    if raw is None:
        raw = getattr(item, "fspath", None)
    if raw is None:
        raise ValueError("collected item has no path")
    path = Path(os.fspath(raw)).resolve(strict=True)
    if not path.is_file() or path.suffix != ".py":
        raise ValueError("collected item path is not a Python file")
    return path


def _select_descriptor(forest: Any, path: Path) -> Any:
    candidates: list[tuple[int, Any]] = []
    for descriptor in forest.descriptors:
        try:
            root = descriptor.root_path.resolve(strict=True)
        except (OSError, ValueError):
            continue
        if _path_under(path, root) is not None:
            candidates.append((len(root.parts), descriptor))
    if not candidates:
        raise ValueError("item is outside repository forest")
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        raise ValueError("item matches ambiguous forest roots")
    return candidates[0][1]


def _marker_names(item: Any) -> tuple[str, ...]:
    try:
        markers = tuple(item.iter_markers())
    except (AttributeError, TypeError):
        markers = ()
    names = []
    for marker in markers:
        name = str(getattr(marker, "name", "") or "").strip()
        if name and name not in names:
            names.append(name[:128])
    return tuple(sorted(names[:_MAX_MARKERS]))


def _effect_adapters(item: Any) -> tuple[str, ...]:
    try:
        markers = tuple(item.iter_markers(name="proof_reuse_effects"))
    except (AttributeError, TypeError):
        markers = ()
    adapters: set[str] = set()
    for marker in markers:
        raw = list(getattr(marker, "args", ()) or ())
        keywords = getattr(marker, "kwargs", {}) or {}
        configured = keywords.get("adapters", ())
        raw.extend((configured,) if isinstance(configured, str) else configured)
        for value in raw:
            if isinstance(value, str):
                text = value.strip()
                if text and len(text) <= 128:
                    adapters.add(text)
    return tuple(sorted(adapters))


def _item_facts(item: Any, descriptor: Any) -> _ItemFacts:
    from ...agent_supervisor.analysis.test_execution_identity import (
        normalize_pytest_node_id,
    )

    path = _item_path(item)
    root = descriptor.root_path.resolve(strict=True)
    relative = _path_under(path, root)
    if relative is None:
        raise ValueError("item path is outside selected descriptor")
    raw_node = str(getattr(item, "nodeid", "") or "")
    selectors = raw_node.split("::")[1:] if "::" in raw_node else []
    canonical_node = relative + (
        "::" + "::".join(selectors) if selectors else ""
    )
    canonical_node = normalize_pytest_node_id(canonical_node)
    function_name = str(
        getattr(item, "originalname", "")
        or getattr(item, "name", "")
        or (selectors[-1] if selectors else "")
    ).split("[", 1)[0]
    if not function_name:
        raise ValueError("test function name is unavailable")
    class_object = getattr(item, "cls", None)
    class_name = str(getattr(class_object, "__qualname__", "") or "")
    if not class_name and len(selectors) > 1:
        class_name = selectors[-2].split("[", 1)[0]

    callspec = getattr(item, "callspec", None)
    parameterized = callspec is not None
    parameter_id = ""
    parameter_value: Any = None
    if parameterized:
        parameter_id = str(getattr(callspec, "id", "") or "")
        raw_params = getattr(callspec, "params", None)
        if not isinstance(raw_params, Mapping):
            raise ValueError("parameterized item has no parameter mapping")
        parameter_value = dict(raw_params)

    raw_fixtures = getattr(item, "fixturenames", ())
    if isinstance(raw_fixtures, (str, bytes)) or not isinstance(
        raw_fixtures, Sequence
    ):
        raise ValueError("fixture inventory is unavailable")
    fixture_names = tuple(sorted({str(name) for name in raw_fixtures if str(name)}))
    if len(fixture_names) > _MAX_FIXTURES:
        raise ValueError("fixture inventory exceeds bound")
    return _ItemFacts(
        path=path,
        relative_path=relative,
        node_id=canonical_node,
        function_name=function_name,
        class_name=class_name,
        parameter_id=parameter_id,
        parameter_value=parameter_value,
        parameterized=parameterized,
        fixture_names=fixture_names,
        markers=_marker_names(item),
        effect_adapters=_effect_adapters(item),
    )


def _source_hash_matches(record: Any, source: str) -> bool:
    claimed = str(getattr(record, "source_sha256", "") or "")
    if claimed.startswith("sha256:"):
        claimed = claimed[7:]
    return claimed == hashlib.sha256(source.encode("utf-8")).hexdigest()


def _ast_identities(facts: _ItemFacts, index: Any) -> _AstIdentities:
    from ...agent_supervisor.analysis.test_execution_identity import (
        mint_content_identity,
    )

    indexed = index.record_for_path(facts.relative_path)
    if indexed is None or indexed.ast_record.parse_error:
        raise ValueError("current test AST record is unavailable")
    if facts.path.stat().st_size > _MAX_SOURCE_BYTES:
        raise ValueError("test source exceeds AST identity bound")
    source = facts.path.read_text(encoding="utf-8")
    if not _source_hash_matches(indexed.ast_record, source):
        raise ValueError("AST index is stale for current source")
    tree = ast.parse(source, filename=facts.relative_path)

    functions: list[tuple[tuple[str, ...], ast.AST]] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.stack: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def _function(self, node: Any) -> None:
            functions.append((tuple(self.stack), node))
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._function(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._function(node)

    Visitor().visit(tree)
    candidates = [
        (owners, node)
        for owners, node in functions
        if getattr(node, "name", "") == facts.function_name
    ]
    if facts.class_name:
        class_parts = tuple(
            part for part in facts.class_name.split(".") if part != "<locals>"
        )
        narrowed = [
            pair
            for pair in candidates
            if pair[0][-len(class_parts) :] == class_parts
        ]
        candidates = narrowed
    if len(candidates) != 1:
        raise ValueError("test AST symbol is missing or ambiguous")
    owners, function = candidates[0]

    def ast_cid(kind: str, node: ast.AST) -> str:
        return mint_content_identity(
            {
                "schema": AUTOMATIC_ITEM_IDENTITY_SCHEMA + "/ast",
                "kind": kind,
                "path": facts.relative_path,
                "ast": ast.dump(
                    node,
                    annotate_fields=True,
                    include_attributes=False,
                ),
            }
        ).cid

    module_cid = ast_cid("module", tree)
    function_cid = ast_cid("function", function)
    class_cid = ""
    if owners:
        wanted = owners[-1]
        class_nodes = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name == wanted
        ]
        if len(class_nodes) != 1:
            raise ValueError("test class AST is missing or ambiguous")
        class_cid = ast_cid("class", class_nodes[0])
    decorator_cids = tuple(
        sorted(ast_cid("decorator", node) for node in function.decorator_list)
    )
    test_ast_cid = mint_content_identity(
        {
            "schema": AUTOMATIC_ITEM_IDENTITY_SCHEMA + "/test-ast",
            "module_cid": module_cid,
            "class_cid": class_cid,
            "function_cid": function_cid,
            "decorator_cids": list(decorator_cids),
        }
    ).cid
    return _AstIdentities(
        module_cid=module_cid,
        class_cid=class_cid,
        function_cid=function_cid,
        decorator_cids=decorator_cids,
        test_ast_cid=test_ast_cid,
    )


def _ancestor_conftest_paths(root: Path, test_path: Path) -> tuple[str, ...]:
    paths: list[str] = []
    current = test_path.parent
    while True:
        candidate = current / "conftest.py"
        if candidate.is_file():
            relative = _path_under(candidate.resolve(strict=True), root)
            if relative is None:
                raise ValueError("ancestor conftest escapes repository root")
            paths.append(relative)
        if current == root:
            break
        if current.parent == current or _path_under(current.parent, root) is None:
            raise ValueError("test ancestry does not reach repository root")
        current = current.parent
    return tuple(sorted(paths))


def _record_names(
    records: Sequence[Mapping[str, Any]], key: str
) -> tuple[str, ...]:
    return tuple(sorted(str(record.get(key) or "") for record in records))


def _validate_component_inventory(
    inputs: CurrentItemComponentInputs,
    facts: _ItemFacts,
    root: Path,
) -> None:
    fixture_names = _record_names(inputs.fixtures, "name")
    if fixture_names != facts.fixture_names:
        raise ValueError("fixture inventory does not exactly match pytest item")
    conftest_paths = _record_names(inputs.conftests, "path")
    if conftest_paths != _ancestor_conftest_paths(root, facts.path):
        raise ValueError("conftest inventory is not exact-current")
    installed_names = {
        str(pair[0]).strip().lower().replace("_", "-")
        for pair in inputs.installed_distributions
        if isinstance(pair, (tuple, list)) and len(pair) == 2
    }
    if "pytest" not in installed_names:
        raise ValueError("installed distribution inventory omits pytest")


def _compile_components(
    inputs: CurrentItemComponentInputs,
    facts: _ItemFacts,
) -> Any:
    from ...agent_supervisor.analysis.test_identity_components import (
        TestIdentityComponents,
    )

    values = {
        "parameter_id": facts.parameter_id,
        "fixtures": inputs.fixtures,
        "conftests": inputs.conftests,
        "hooks": inputs.hooks,
        "plugins": inputs.plugins,
        "lock_files": inputs.lock_files,
        "installed_distributions": inputs.installed_distributions,
        "environment": inputs.environment,
        "environment_allowlist": inputs.environment_allowlist,
        "interpreter_facts": inputs.interpreter_facts,
        "platform_facts": inputs.platform_facts,
        "hardware_facts": inputs.hardware_facts,
        "capability_facts": inputs.capability_facts,
        "capability_allowlist": inputs.capability_allowlist,
    }
    if facts.parameterized:
        values["parameter_value"] = facts.parameter_value
    return TestIdentityComponents.compile(**values)


def _disabled(item: Any) -> bool:
    try:
        return item.get_closest_marker("proof_reuse_disabled") is not None
    except (AttributeError, TypeError):
        return False


def _existing_identity(item: Any) -> bool:
    return any(
        getattr(item, name, None) is not None
        for name in (
            ITEM_LOCATOR_ATTRIBUTE,
            ITEM_EXECUTION_KEY_ATTRIBUTE,
            "_ipfs_proof_reuse_lookup_request",
        )
    )


def _attach(item: Any, result: AutomaticItemIdentityAssembly) -> bool:
    request = result.lookup_request
    execution = getattr(result.execution_artifact, "execution_key", None)
    locator = getattr(result.locator_artifact, "locator", None)
    if request is None or execution is None or locator is None:
        return False
    values = (
        (ITEM_LOCATOR_ATTRIBUTE, locator),
        (ITEM_EXECUTION_KEY_ATTRIBUTE, execution),
        (ITEM_ELIGIBILITY_ATTRIBUTE, result.eligibility),
        (ITEM_POLICY_ATTRIBUTE, request.current_policy),
        ("_ipfs_proof_reuse_lookup_request", request),
    )
    written: list[str] = []
    try:
        for name, value in values:
            setattr(item, name, value)
            written.append(name)
        setattr(item, ITEM_IDENTITY_RESULT_ATTRIBUTE, result)
        return True
    except BaseException:
        for name in written:
            try:
                delattr(item, name)
            except BaseException:
                pass
        return False


class AutomaticItemIdentityAssembler:
    """Assemble exact current identity inputs for a collected pytest item."""

    __test__ = False

    def __init__(self, services: ItemIdentityAssemblyServices) -> None:
        if not isinstance(services, ItemIdentityAssemblyServices):
            raise TypeError("services must be ItemIdentityAssemblyServices")
        self.services = services

    @property
    def interface(self) -> str:
        return AUTOMATIC_ITEM_IDENTITY_INTERFACE

    def assemble(self, item: Any) -> AutomaticItemIdentityAssembly:
        """Return a lookup-admission result; every uncertainty returns ``RUN``."""

        try:
            return self._assemble(item)
        except BaseException as exc:
            return _failure(
                ItemIdentityAssemblyReason.INTERNAL_ERROR_FAIL_OPEN_TO_RUN,
                "assembler",
                exception_type=type(exc).__name__,
            )

    def _assemble(self, item: Any) -> AutomaticItemIdentityAssembly:
        if _disabled(item):
            return _failure(ItemIdentityAssemblyReason.ITEM_DISABLED, "item")
        if _existing_identity(item):
            return _failure(
                ItemIdentityAssemblyReason.EXISTING_IDENTITY_CONFLICT, "item"
            )
        try:
            path = _item_path(item)
        except (OSError, TypeError, ValueError) as exc:
            return _failure(
                ItemIdentityAssemblyReason.ITEM_PATH_UNAVAILABLE,
                "item",
                exception_type=type(exc).__name__,
            )

        forest, failed = _call_provider(
            self.services.repository_forest_provider,
            item,
            reason=ItemIdentityAssemblyReason.REPOSITORY_FOREST_UNAVAILABLE,
            stage="repository_forest",
        )
        if failed is not None:
            return failed
        from ...agent_supervisor.repository_forest import (
            RepositoryForest,
            descriptor_satisfies_repository_descriptor,
        )

        if not isinstance(forest, RepositoryForest):
            return _failure(
                ItemIdentityAssemblyReason.REPOSITORY_FOREST_UNAVAILABLE,
                "repository_forest",
            )
        if forest.reason_codes or not forest.descriptors or any(
            not descriptor_satisfies_repository_descriptor(descriptor)
            for descriptor in forest.descriptors
        ):
            return _failure(
                ItemIdentityAssemblyReason.REPOSITORY_FOREST_INCOMPLETE,
                "repository_forest",
                reason_count=len(forest.reason_codes),
            )
        try:
            descriptor = _select_descriptor(forest, path)
            facts = _item_facts(item, descriptor)
        except (OSError, TypeError, ValueError) as exc:
            return _failure(
                ItemIdentityAssemblyReason.ITEM_PATH_OUTSIDE_FOREST,
                "item",
                exception_type=type(exc).__name__,
            )

        index, failed = _call_provider(
            self.services.analysis_index_provider,
            item,
            descriptor,
            reason=ItemIdentityAssemblyReason.AST_INDEX_UNAVAILABLE,
            stage="ast_index",
        )
        if failed is not None:
            return failed
        from ...agent_supervisor.analysis.analysis_ast_index import AnalysisASTIndex
        from ...agent_supervisor.analysis.test_static_dependency_trace import (
            StaticTestDependencyTracer,
        )

        if not isinstance(index, AnalysisASTIndex):
            return _failure(
                ItemIdentityAssemblyReason.AST_INDEX_UNAVAILABLE, "ast_index"
            )
        try:
            static_trace = StaticTestDependencyTracer(
                index, descriptor.root_path
            ).trace(
                facts.relative_path,
                test_symbol=facts.function_name,
                node_id=facts.node_id,
            )
            static_trace.verify()
        except BaseException as exc:
            return _failure(
                ItemIdentityAssemblyReason.STATIC_TRACE_INCOMPLETE,
                "static_trace",
                exception_type=type(exc).__name__,
            )
        frontier_kinds = {entry.kind for entry in static_trace.unknown_frontier}
        effect_only = bool(frontier_kinds) and frontier_kinds <= {
            "uncontrolled_effect"
        }
        if not static_trace.complete and not (
            effect_only and facts.effect_adapters
        ):
            return _failure(
                ItemIdentityAssemblyReason.STATIC_TRACE_INCOMPLETE,
                "static_trace",
                frontier_count=len(static_trace.unknown_frontier),
            )
        try:
            ast_identities = _ast_identities(facts, index)
        except BaseException as exc:
            return _failure(
                ItemIdentityAssemblyReason.AST_IDENTITY_UNAVAILABLE,
                "ast_identity",
                exception_type=type(exc).__name__,
            )

        component_inputs, failed = _call_provider(
            self.services.component_inputs_provider,
            item,
            facts,
            descriptor,
            static_trace,
            reason=ItemIdentityAssemblyReason.COMPONENT_INPUT_UNAVAILABLE,
            stage="components",
        )
        if failed is not None:
            return failed
        if not isinstance(component_inputs, CurrentItemComponentInputs):
            return _failure(
                ItemIdentityAssemblyReason.COMPONENT_INPUT_UNAVAILABLE,
                "components",
            )
        try:
            _validate_component_inventory(
                component_inputs,
                facts,
                descriptor.root_path.resolve(strict=True),
            )
            components = _compile_components(component_inputs, facts)
        except BaseException as exc:
            return _failure(
                ItemIdentityAssemblyReason.COMPONENT_INVENTORY_MISMATCH,
                "components",
                exception_type=type(exc).__name__,
            )
        if not components.reusable:
            return _failure(
                ItemIdentityAssemblyReason.COMPONENTS_NON_REUSABLE,
                "components",
                reason_count=len(components.non_reusable_reasons),
            )

        policy_inputs, failed = _call_provider(
            self.services.policy_inputs_provider,
            item,
            facts,
            descriptor,
            static_trace,
            components,
            reason=ItemIdentityAssemblyReason.POLICY_INPUT_UNAVAILABLE,
            stage="policy",
        )
        if failed is not None:
            return failed
        if not isinstance(policy_inputs, CurrentItemPolicyInputs):
            return _failure(
                ItemIdentityAssemblyReason.POLICY_INPUT_UNAVAILABLE, "policy"
            )
        try:
            identities = policy_inputs.verified_identities()
        except BaseException as exc:
            return _failure(
                ItemIdentityAssemblyReason.POLICY_INPUT_UNAVAILABLE,
                "policy",
                exception_type=type(exc).__name__,
            )

        runtime_evidence, failed = _call_provider(
            self.services.runtime_evidence_provider,
            item,
            facts,
            descriptor,
            static_trace,
            components,
            policy_inputs,
            reason=ItemIdentityAssemblyReason.RUNTIME_EVIDENCE_UNAVAILABLE,
            stage="runtime_evidence",
        )
        if failed is not None:
            return failed
        if not isinstance(runtime_evidence, CurrentRuntimeTraceEvidence):
            return _failure(
                ItemIdentityAssemblyReason.RUNTIME_EVIDENCE_NOT_CURRENT,
                "runtime_evidence",
            )
        try:
            runtime_evidence.verify_current(
                node_id=facts.node_id,
                repository_forest_cid=forest.forest_id,
                static_trace_root_cid=static_trace.trace_cid,
                identity_components_cid=components.component_root_cid,
                runtime_completeness_policy_cid=identities[
                    "runtime_policy"
                ].cid,
            )
        except BaseException as exc:
            return _failure(
                ItemIdentityAssemblyReason.RUNTIME_EVIDENCE_NOT_CURRENT,
                "runtime_evidence",
                exception_type=type(exc).__name__,
            )
        if not runtime_evidence.trace.complete:
            return _failure(
                ItemIdentityAssemblyReason.RUNTIME_TRACE_INCOMPLETE,
                "runtime_evidence",
            )

        from ...agent_supervisor.analysis.test_execution_identity import (
            TestExecutionIdentityCompiler,
            mint_content_identity,
        )
        from ...agent_supervisor.analysis.test_reuse_eligibility import (
            DirtyStateEvidence,
            evaluate_reuse_eligibility,
        )

        compiler = self.services.identity_compiler
        if compiler is None:
            compiler = TestExecutionIdentityCompiler()
        if not isinstance(compiler, TestExecutionIdentityCompiler):
            return _failure(
                ItemIdentityAssemblyReason.IDENTITY_COMPILER_REJECTED,
                "compiler",
            )
        dirty_identity = mint_content_identity(
            {
                "schema": AUTOMATIC_ITEM_IDENTITY_SCHEMA + "/dirty-overlay",
                "dirty": descriptor.dirty,
                "overlay_digest": descriptor.dirty_overlay_digest,
                "reason_codes": list(descriptor.reason_codes),
            }
        )
        eligibility = evaluate_reuse_eligibility(
            static_trace=static_trace,
            runtime_trace=runtime_evidence.trace,
            repository_forest=forest,
            effect_adapters=facts.effect_adapters,
            snapshot_adapters=policy_inputs.snapshot_adapters,
            parameters_supported=components.reusable,
            parameter_non_reusable_reason=(
                components.non_reusable_reasons[0]
                if components.non_reusable_reasons
                else ""
            ),
            dirty_state=DirtyStateEvidence(
                dirty=descriptor.dirty,
                dirty_overlay_cid=dirty_identity.cid,
                dirty_accounted=not descriptor.reason_codes,
                reason_codes=descriptor.reason_codes,
            ),
            policy=policy_inputs.reuse_policy,
        )
        if not eligibility.reusable:
            return _failure(
                ItemIdentityAssemblyReason.ELIGIBILITY_DENIED,
                "eligibility",
                reason_count=len(eligibility.reason_codes),
            )

        locator_artifact = compiler.compile_locator(
            repository_id=descriptor.repository_id,
            package_identity=descriptor.descriptor_cid,
            root_identity=descriptor.descriptor_cid,
            node_id=facts.node_id,
            collection_schema_version=(
                policy_inputs.collection_schema_version
            ),
            parameter_id=facts.parameter_id,
            parameter_values_cid=(
                components.parameter_cid if facts.parameterized else ""
            ),
            selection_semantics="exact_node",
            metadata={
                "assembler_interface": AUTOMATIC_ITEM_IDENTITY_INTERFACE,
                "repository_alias": descriptor.alias,
            },
        )
        if not locator_artifact.reusable or locator_artifact.locator is None:
            return _failure(
                ItemIdentityAssemblyReason.IDENTITY_COMPILER_REJECTED,
                "locator",
                compiler_reason=locator_artifact.reason_code,
            )

        import pytest

        component_fields = components.execution_key_fields()
        component_map = dict(component_fields.pop("components"))
        component_map.update(
            {
                "automatic_item_identity": mint_content_identity(
                    {
                        "schema": AUTOMATIC_ITEM_IDENTITY_SCHEMA,
                        "interface": AUTOMATIC_ITEM_IDENTITY_INTERFACE,
                    }
                ).cid,
                "current_runtime_evidence": (
                    runtime_evidence.binding_identity.cid
                ),
                "repository_descriptor": descriptor.descriptor_cid,
            }
        )
        execution_artifact = compiler.compile_execution_key(
            locator_cid=locator_artifact.locator_cid,
            repository_forest_cid=forest.forest_id,
            git_commit_id=descriptor.commit,
            git_tree_id=descriptor.tree,
            gitlink_state_cid=(
                descriptor.portable_closure.gitlink_closure_cid
            ),
            dirty_overlay_cid=dirty_identity.cid,
            test_module_cid=ast_identities.module_cid,
            test_class_cid=ast_identities.class_cid,
            test_function_cid=ast_identities.function_cid,
            decorator_cids=ast_identities.decorator_cids,
            test_ast_cid=ast_identities.test_ast_cid,
            static_trace_root_cid=static_trace.trace_cid,
            static_unknown_frontier=tuple(
                entry.frontier_id for entry in static_trace.unknown_frontier
            ),
            runtime_trace_root_cid=runtime_evidence.trace.trace_cid,
            runtime_completeness_policy=identities["runtime_policy"].cid,
            pytest_version=str(pytest.__version__)[:_SAFE_VERSION_CHARS],
            python_version=(
                "%d.%d.%d"
                % (
                    os.sys.version_info.major,
                    os.sys.version_info.minor,
                    os.sys.version_info.micro,
                )
            ),
            plugin_versions_cid=identities["plugins"].cid,
            command_semantics_cid=identities["command"].cid,
            config_cid=identities["config"].cid,
            markers=facts.markers,
            external_snapshot_cids=tuple(
                sorted(policy_inputs.snapshot_adapters.values())
            ),
            policy_cid=identities["policy"].cid,
            canonicalization_schema_cid=identities[
                "canonicalization"
            ].cid,
            tracer_schema_cid=identities["tracer"].cid,
            certificate_schema_cid=identities["certificate"].cid,
            eligibility_class=eligibility.eligibility_class,
            components=component_map,
            **component_fields,
        )
        if (
            not execution_artifact.reusable
            or execution_artifact.execution_key is None
        ):
            return _failure(
                ItemIdentityAssemblyReason.IDENTITY_COMPILER_REJECTED,
                "execution_key",
                compiler_reason=execution_artifact.reason_code,
            )

        from .lookup import ProofReuseLookupRequest

        request = ProofReuseLookupRequest(
            item=item,
            locator=locator_artifact.locator,
            execution_key=execution_artifact.execution_key,
            eligibility=eligibility,
            current_policy=dict(policy_inputs.verification_policy),
        )
        return AutomaticItemIdentityAssembly(
            reason=ItemIdentityAssemblyReason.ADMITTED_FOR_LOOKUP,
            stage="complete",
            locator_artifact=locator_artifact,
            execution_artifact=execution_artifact,
            eligibility=eligibility,
            lookup_request=request,
            diagnostics={"runtime_provenance": runtime_evidence.provenance.value},
        )


def assemble_and_attach_item_identity(
    item: Any,
    services: ItemIdentityAssemblyServices,
) -> AutomaticItemIdentityAssembly:
    """One call for the pytest plugin: assemble, attach if safe, otherwise RUN."""

    result = AutomaticItemIdentityAssembler(services).assemble(item)
    if result.admitted_for_lookup and not _attach(item, result):
        result = _failure(
            ItemIdentityAssemblyReason.ATTACHMENT_FAILED, "attachment"
        )
    try:
        setattr(item, ITEM_IDENTITY_RESULT_ATTRIBUTE, result)
    except BaseException:
        # A pytest item that rejects diagnostics must still execute.
        if result.admitted_for_lookup:
            return _failure(
                ItemIdentityAssemblyReason.ATTACHMENT_FAILED, "attachment"
            )
    return result


__all__ = (
    "AUTOMATIC_ITEM_IDENTITY_INTERFACE",
    "AUTOMATIC_ITEM_IDENTITY_RESULT_INTERFACE",
    "AUTOMATIC_ITEM_IDENTITY_SCHEMA",
    "CURRENT_ITEM_INPUTS_INTERFACE",
    "CURRENT_RUNTIME_TRACE_EVIDENCE_INTERFACE",
    "CURRENT_RUNTIME_TRACE_EVIDENCE_SCHEMA",
    "ITEM_IDENTITY_RESULT_ATTRIBUTE",
    "AutomaticItemIdentityAssembler",
    "AutomaticItemIdentityAssembly",
    "CurrentInputCompleteness",
    "CurrentItemComponentInputs",
    "CurrentItemPolicyInputs",
    "CurrentRuntimeTraceEvidence",
    "ItemIdentityAssemblyReason",
    "ItemIdentityAssemblyServices",
    "RuntimeEvidenceProvenance",
    "assemble_and_attach_item_identity",
)
