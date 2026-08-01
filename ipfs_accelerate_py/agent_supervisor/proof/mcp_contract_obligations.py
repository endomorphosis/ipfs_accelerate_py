"""Compile reviewed MCP++ claims into canonical, solver-neutral obligations.

Interface: ``McpContractObligations@1``

This module is an adapter, not a theorem generator.  It accepts only the
closed :class:`McpClaimFamily` vocabulary and exact, identifier-only premises.
The theorem statement is a canonical structured expression selected from the
reviewed family table below; callers cannot provide prose or source text.

The resulting record keeps three existing representations bound together:

* a domain-separated shared logic-IR view from ``ipfs_datasets_py``;
* the accelerator's existing :class:`CodeProofObligation`; and
* the accelerator's existing ``CodeClaimRecord@1`` lifecycle record.

No representation establishes the theorem.  Analysis results are observations,
and supported obligations remain open until independently discharged at the
required assurance level.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from ..analysis.content_identity_bridge import (
    LOGIC_IR_PROFILE,
    identify_logic_ir,
)
from ..analysis.mcp_contract_analysis import (
    ContractParityClaim,
    ParityState,
)
from ..analysis.mcp_contract_catalog import (
    McpClaimFamily,
    McpContractCatalog,
    ContractRecord,
    ReviewState,
)
from .code_claim_contracts import (
    ClaimFamily,
    ClaimStatus,
    CodeClaimRecord,
    claim_from_obligation,
)
from .formal_verification_contracts import (
    AssuranceLevel,
    CodeProofObligation,
    canonical_json,
    content_identity,
)


MCP_CONTRACT_OBLIGATIONS_INTERFACE: Final = "McpContractObligations@1"
MCP_CONTRACT_OBLIGATION_INTERFACE: Final = MCP_CONTRACT_OBLIGATIONS_INTERFACE
MCP_CONTRACT_OBLIGATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/mcp-contract-obligation@1"
)
MCP_LOGIC_VIEW_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/mcp-contract-logic-view@1"
)
MCP_LOGIC_EXPRESSION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/mcp-contract-logic-expression@1"
)
MCP_CONTRACT_OBLIGATION_VERSION: Final = "1"
MCP_LOGIC_IR_DOMAIN: Final = "ipfs-accelerate/mcp-contract-obligation"
MCP_LOGIC_IR_SCHEMA_VERSION: Final = "mcp-contract-logic/v1"
MCP_OBLIGATION_COMPILER_ID: Final = "mcp-contract-obligation-compiler@1"

_MAX_IDENTIFIER_BYTES: Final = 2_048
_CLAIM_FIELDS: Final = frozenset(
    {
        "schema",
        "claim_id",
        "family",
        "state",
        "operation_id",
        "premise_ids",
        "reason_codes",
        "counterexamples",
    }
)


class McpContractObligationError(ValueError):
    """A claim cannot safely be compiled into a reviewed obligation."""


class LogicFragment(str, Enum):
    """Closed solver-neutral fragments used by MCP contract obligations."""

    GRAPH = "graph"
    RELATION = "relation"
    SCHEMA = "schema"
    DEONTIC = "deontic"
    TEMPORAL = "temporal"
    UNSUPPORTED = "unsupported"

    # Descriptive aliases retained for callers that name the proof technique.
    GRAPH_REACHABILITY = "graph"
    RELATIONAL = "relation"
    JSON_SCHEMA = "schema"
    DEONTIC_ORDER = "deontic"
    BOUNDED_TEMPORAL = "temporal"


class LogicOperator(str, Enum):
    """Reviewed operators; there is deliberately no free-form predicate."""

    DECLARED_TOOL_EXISTS = "declared_tool_exists"
    DESCRIPTOR_SCHEMA_MATCHES = "descriptor_schema_matches"
    INVOCATION_REACHABLE = "invocation_reachable"
    ARGUMENTS_PRESERVED = "arguments_preserved"
    RESULT_ENVELOPE_PRESERVED = "result_envelope_preserved"
    POLICY_BEFORE_EFFECT = "policy_before_effect"
    NO_COMPATIBILITY_BYPASS = "no_compatibility_bypass"
    TRANSPORT_PARITY = "transport_parity"
    DISCOVERY_EXECUTION_PARITY = "discovery_execution_parity"
    FAILURE_PARITY = "failure_parity"
    SNAPSHOT_FRESHNESS = "snapshot_freshness"
    NO_DYNAMIC_AUTHORITY = "no_dynamic_authority"
    UNSUPPORTED = "unsupported"


_FAMILY_LOGIC: Final[
    Mapping[McpClaimFamily, tuple[LogicFragment, LogicOperator]]
] = MappingProxyType(
    {
        McpClaimFamily.DECLARED_TOOL_EXISTS: (
            LogicFragment.GRAPH,
            LogicOperator.DECLARED_TOOL_EXISTS,
        ),
        McpClaimFamily.DESCRIPTOR_SCHEMA_MATCHES: (
            LogicFragment.SCHEMA,
            LogicOperator.DESCRIPTOR_SCHEMA_MATCHES,
        ),
        McpClaimFamily.INVOCATION_REACHABLE: (
            LogicFragment.GRAPH,
            LogicOperator.INVOCATION_REACHABLE,
        ),
        McpClaimFamily.ARGUMENTS_PRESERVED: (
            LogicFragment.SCHEMA,
            LogicOperator.ARGUMENTS_PRESERVED,
        ),
        McpClaimFamily.RESULT_ENVELOPE_PRESERVED: (
            LogicFragment.SCHEMA,
            LogicOperator.RESULT_ENVELOPE_PRESERVED,
        ),
        McpClaimFamily.POLICY_BEFORE_EFFECT: (
            LogicFragment.DEONTIC,
            LogicOperator.POLICY_BEFORE_EFFECT,
        ),
        McpClaimFamily.NO_COMPATIBILITY_BYPASS: (
            LogicFragment.DEONTIC,
            LogicOperator.NO_COMPATIBILITY_BYPASS,
        ),
        McpClaimFamily.TRANSPORT_PARITY: (
            LogicFragment.RELATION,
            LogicOperator.TRANSPORT_PARITY,
        ),
        McpClaimFamily.DISCOVERY_EXECUTION_PARITY: (
            LogicFragment.RELATION,
            LogicOperator.DISCOVERY_EXECUTION_PARITY,
        ),
        McpClaimFamily.FAILURE_PARITY: (
            LogicFragment.RELATION,
            LogicOperator.FAILURE_PARITY,
        ),
        McpClaimFamily.SNAPSHOT_FRESHNESS: (
            LogicFragment.TEMPORAL,
            LogicOperator.SNAPSHOT_FRESHNESS,
        ),
        McpClaimFamily.NO_DYNAMIC_AUTHORITY: (
            LogicFragment.GRAPH,
            LogicOperator.NO_DYNAMIC_AUTHORITY,
        ),
    }
)

_CODE_FAMILY: Final[Mapping[McpClaimFamily, ClaimFamily]] = MappingProxyType(
    {
        McpClaimFamily.DECLARED_TOOL_EXISTS: ClaimFamily.API_CONTRACT,
        McpClaimFamily.DESCRIPTOR_SCHEMA_MATCHES: ClaimFamily.API_CONTRACT,
        McpClaimFamily.INVOCATION_REACHABLE: ClaimFamily.DEPENDENCY_REACHABILITY,
        McpClaimFamily.ARGUMENTS_PRESERVED: ClaimFamily.API_CONTRACT,
        McpClaimFamily.RESULT_ENVELOPE_PRESERVED: ClaimFamily.API_CONTRACT,
        McpClaimFamily.POLICY_BEFORE_EFFECT: ClaimFamily.SECURITY_PROPERTY,
        McpClaimFamily.NO_COMPATIBILITY_BYPASS: ClaimFamily.SECURITY_PROPERTY,
        McpClaimFamily.TRANSPORT_PARITY: ClaimFamily.SEMANTIC_EQUIVALENCE,
        McpClaimFamily.DISCOVERY_EXECUTION_PARITY: (
            ClaimFamily.SEMANTIC_EQUIVALENCE
        ),
        McpClaimFamily.FAILURE_PARITY: ClaimFamily.SEMANTIC_EQUIVALENCE,
        McpClaimFamily.SNAPSHOT_FRESHNESS: ClaimFamily.BEHAVIORAL_INVARIANT,
        McpClaimFamily.NO_DYNAMIC_AUTHORITY: ClaimFamily.SECURITY_PROPERTY,
    }
)


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise McpContractObligationError(f"{name} must be a string")
    if value != value.strip() or "\x00" in value:
        raise McpContractObligationError(
            f"{name} must not contain surrounding whitespace or NUL"
        )
    if required and not value:
        raise McpContractObligationError(f"{name} is required")
    if len(value.encode("utf-8")) > _MAX_IDENTIFIER_BYTES:
        raise McpContractObligationError(f"{name} is oversized")
    return value


def _identifier(value: Any, name: str) -> str:
    """Validate an identifier and reject inline source/graph/corpus material."""

    result = _text(value, name)
    lowered = result.lower()
    if (
        any(character.isspace() for character in result)
        or result.startswith(("{", "["))
        or lowered.startswith(
            (
                "def ",
                "class ",
                "function ",
                "import ",
                "from ",
                "<graph",
                "digraph ",
                "source:",
            )
        )
    ):
        raise McpContractObligationError(
            f"{name} must be a compact identifier, not source or graph data"
        )
    return result


def _ids(
    values: Any,
    name: str,
    *,
    required: bool = False,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(
        values, Sequence
    ):
        raise McpContractObligationError(
            f"{name} must be a sequence of compact identifiers"
        )
    result = tuple(sorted({_identifier(item, name) for item in values}))
    if required and not result:
        raise McpContractObligationError(f"{name} must not be empty")
    return result


def _assurance(value: AssuranceLevel | str) -> AssuranceLevel:
    if isinstance(value, AssuranceLevel):
        return value
    try:
        return AssuranceLevel(str(value))
    except (TypeError, ValueError) as exc:
        raise McpContractObligationError(
            f"unknown required_assurance: {value!r}"
        ) from exc


def _claim(value: ContractParityClaim | Mapping[str, Any]) -> ContractParityClaim:
    if isinstance(value, ContractParityClaim):
        return value
    if not isinstance(value, Mapping):
        raise McpContractObligationError(
            "claim must be a ContractParityClaim or canonical mapping"
        )
    if set(value).difference(_CLAIM_FIELDS):
        raise McpContractObligationError(
            "claim contains unsupported fields; source/theorem payloads are not accepted"
        )
    try:
        return ContractParityClaim.from_dict(value)
    except (TypeError, ValueError) as exc:
        raise McpContractObligationError(f"invalid contract claim: {exc}") from exc


def _logic_for_claim(
    claim: ContractParityClaim,
) -> tuple[LogicFragment, LogicOperator, str]:
    if claim.state is ParityState.UNSUPPORTED:
        reason = claim.reason_codes[0] if claim.reason_codes else "unsupported_claim"
        return LogicFragment.UNSUPPORTED, LogicOperator.UNSUPPORTED, reason
    try:
        fragment, operator = _FAMILY_LOGIC[claim.family]
    except KeyError:
        return (
            LogicFragment.UNSUPPORTED,
            LogicOperator.UNSUPPORTED,
            "unsupported_claim_family",
        )
    return fragment, operator, ""


def _resolve_contract(
    catalog: McpContractCatalog,
    claim: ContractParityClaim,
    contract: ContractRecord | str | None,
) -> ContractRecord:
    if not isinstance(catalog, McpContractCatalog):
        raise McpContractObligationError(
            "catalog must be a validated McpContractCatalog"
        )
    if isinstance(contract, ContractRecord):
        stored = catalog.get_contract(contract.contract_id)
        if stored is None or stored != contract:
            raise McpContractObligationError(
                "contract is not bound to the supplied catalog"
            )
        result = stored
    elif isinstance(contract, str) and contract:
        result = catalog.get_contract(_identifier(contract, "contract_id"))
        if result is None:
            raise McpContractObligationError(
                "contract_id is not present in the supplied catalog"
            )
    elif contract is None or contract == "":
        candidates = tuple(
            item
            for item in catalog.contracts
            if item.claim_family is claim.family
            and (
                item.subject == claim.operation_id
                or item.tool_name == claim.operation_id
            )
        )
        if len(candidates) != 1:
            raise McpContractObligationError(
                "contract must identify exactly one reviewed catalog property"
            )
        result = candidates[0]
    else:
        raise McpContractObligationError(
            "contract must be a ContractRecord, contract_id, or None"
        )
    if result.review_state is not ReviewState.REVIEWED:
        raise McpContractObligationError(
            "contract property is not in reviewed state"
        )
    if result.claim_family is not claim.family:
        raise McpContractObligationError(
            "claim family does not match the reviewed contract property"
        )
    if (
        result.subject != claim.operation_id
        and result.tool_name != claim.operation_id
    ):
        raise McpContractObligationError(
            "claim operation does not match the reviewed contract property"
        )
    return result


@dataclass(frozen=True, slots=True)
class McpLogicView:
    """Canonical structured theorem view under the shared logic-IR profile."""

    family: McpClaimFamily
    fragment: LogicFragment
    operator: LogicOperator
    operation_id: str
    property_id: str
    claim_id: str
    premise_ids: tuple[str, ...]
    assumption_ids: tuple[str, ...]
    supported: bool = True
    unsupported_reason: str = ""

    def __post_init__(self) -> None:
        try:
            family = (
                self.family
                if isinstance(self.family, McpClaimFamily)
                else McpClaimFamily(str(self.family))
            )
            fragment = (
                self.fragment
                if isinstance(self.fragment, LogicFragment)
                else LogicFragment(str(self.fragment))
            )
            operator = (
                self.operator
                if isinstance(self.operator, LogicOperator)
                else LogicOperator(str(self.operator))
            )
        except (TypeError, ValueError) as exc:
            raise McpContractObligationError(
                "logic view uses an unknown family, fragment, or operator"
            ) from exc
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "fragment", fragment)
        object.__setattr__(self, "operator", operator)
        for name in ("operation_id", "property_id", "claim_id"):
            object.__setattr__(
                self, name, _identifier(getattr(self, name), name)
            )
        object.__setattr__(
            self, "premise_ids", _ids(self.premise_ids, "premise_ids")
        )
        object.__setattr__(
            self, "assumption_ids", _ids(self.assumption_ids, "assumption_ids")
        )
        reason = _identifier(
            self.unsupported_reason,
            "unsupported_reason",
        ) if self.unsupported_reason else ""
        object.__setattr__(self, "unsupported_reason", reason)
        expected_supported = fragment is not LogicFragment.UNSUPPORTED
        if bool(self.supported) != expected_supported:
            raise McpContractObligationError(
                "supported flag must agree with the logic fragment"
            )
        if expected_supported:
            expected = _FAMILY_LOGIC.get(family)
            if expected != (fragment, operator):
                raise McpContractObligationError(
                    "family is not bound to its reviewed logic operator"
                )
            if reason:
                raise McpContractObligationError(
                    "supported logic view cannot carry unsupported_reason"
                )
        elif operator is not LogicOperator.UNSUPPORTED or not reason:
            raise McpContractObligationError(
                "unsupported logic view requires an explicit reason"
            )

    def expression_dict(self) -> dict[str, Any]:
        """Return the closed theorem expression (never caller-authored prose)."""

        return {
            "schema": MCP_LOGIC_EXPRESSION_SCHEMA,
            "operator": self.operator.value,
            "terms": {
                "claim_id": self.claim_id,
                "operation_id": self.operation_id,
                "property_id": self.property_id,
            },
        }

    @property
    def statement(self) -> str:
        return canonical_json(self.expression_dict())

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": MCP_LOGIC_VIEW_SCHEMA,
            "version": MCP_CONTRACT_OBLIGATION_VERSION,
            "family": self.family.value,
            "fragment": self.fragment.value,
            "expression": self.expression_dict(),
            "premise_ids": list(self.premise_ids),
            "assumption_ids": list(self.assumption_ids),
            "supported": self.supported,
            "unsupported_reason": self.unsupported_reason,
        }

    @property
    def identity(self) -> Any:
        return identify_logic_ir(
            self._identity_payload(),
            domain=MCP_LOGIC_IR_DOMAIN,
            schema_version=MCP_LOGIC_IR_SCHEMA_VERSION,
        )

    @property
    def logic_id(self) -> str:
        return self.identity.cid

    @property
    def content_id(self) -> str:
        return self.logic_id

    @property
    def identity_profile(self) -> str:
        return LOGIC_IR_PROFILE

    def to_dict(self) -> dict[str, Any]:
        identity = self.identity
        return {
            **self._identity_payload(),
            "logic_id": identity.cid,
            "identity": identity.to_dict(),
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    def canonical_bytes(self) -> bytes:
        return self.to_json().encode("utf-8")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "McpLogicView":
        if not isinstance(value, Mapping):
            raise McpContractObligationError("logic view must be an object")
        allowed = {
            "schema",
            "version",
            "family",
            "fragment",
            "expression",
            "premise_ids",
            "assumption_ids",
            "supported",
            "unsupported_reason",
            "logic_id",
            "identity",
        }
        if set(value).difference(allowed):
            raise McpContractObligationError(
                "logic view contains unsupported fields"
            )
        if value.get("schema") not in (None, MCP_LOGIC_VIEW_SCHEMA):
            raise McpContractObligationError("unsupported logic-view schema")
        if value.get("version") not in (
            None,
            MCP_CONTRACT_OBLIGATION_VERSION,
        ):
            raise McpContractObligationError("unsupported logic-view version")
        expression = value.get("expression")
        if not isinstance(expression, Mapping):
            raise McpContractObligationError(
                "logic view requires a structured expression"
            )
        if set(expression) != {"schema", "operator", "terms"}:
            raise McpContractObligationError(
                "logic expression must use the reviewed closed shape"
            )
        if expression.get("schema") != MCP_LOGIC_EXPRESSION_SCHEMA:
            raise McpContractObligationError(
                "unsupported logic-expression schema"
            )
        terms = expression.get("terms")
        if not isinstance(terms, Mapping):
            raise McpContractObligationError(
                "logic expression requires compact terms"
            )
        if set(terms) != {"claim_id", "operation_id", "property_id"}:
            raise McpContractObligationError(
                "logic expression contains unsupported terms"
            )
        result = cls(
            family=value.get("family", ""),
            fragment=value.get("fragment", ""),
            operator=expression.get("operator", ""),
            operation_id=terms.get("operation_id", ""),
            property_id=terms.get("property_id", ""),
            claim_id=terms.get("claim_id", ""),
            premise_ids=tuple(value.get("premise_ids") or ()),
            assumption_ids=tuple(value.get("assumption_ids") or ()),
            supported=bool(value.get("supported", False)),
            unsupported_reason=str(value.get("unsupported_reason") or ""),
        )
        claimed_id = value.get("logic_id")
        if claimed_id is not None and claimed_id != result.logic_id:
            raise McpContractObligationError(
                "logic-view identity does not match canonical content"
            )
        identity = value.get("identity")
        if identity is not None:
            if not isinstance(identity, Mapping):
                raise McpContractObligationError("identity must be an object")
            expected = result.identity
            for key, expected_value in (
                ("profile", expected.profile),
                ("cid", expected.cid),
                ("digest", expected.digest),
                ("domain", expected.domain),
            ):
                if identity.get(key) != expected_value:
                    raise McpContractObligationError(
                        "logic-view identity metadata mismatch"
                    )
        return result

    @classmethod
    def from_json(cls, value: str) -> "McpLogicView":
        try:
            payload = json.loads(value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise McpContractObligationError("logic-view JSON is malformed") from exc
        return cls.from_dict(payload)


def _load_shared_ir_claim(value: Mapping[str, Any]) -> Any:
    try:
        from ipfs_datasets_py.logic.ir_core.claims import IRClaim
    except ImportError as exc:
        raise McpContractObligationError(
            "ipfs_datasets_py shared logic IR is unavailable"
        ) from exc
    try:
        return IRClaim.from_dict(value)
    except (TypeError, ValueError) as exc:
        raise McpContractObligationError(
            f"invalid shared logic IR claim: {exc}"
        ) from exc


def _build_shared_ir(
    logic_view: McpLogicView,
    *,
    catalog_id: str,
    catalog_version: str,
    repository_id: str,
    snapshot_id: str,
    scope_ids: tuple[str, ...],
    toolchain_id: str,
    policy_id: str,
    required_assurance: AssuranceLevel,
) -> Any:
    try:
        from ipfs_datasets_py.logic.ir_core.claims import (
            Assumption,
            FrozenMap,
            IRClaim,
            ProofObligation,
        )
    except ImportError as exc:
        raise McpContractObligationError(
            "ipfs_datasets_py shared logic IR is unavailable"
        ) from exc

    assumptions = tuple(
        Assumption(
            assumption_id=assumption_id,
            statement=canonical_json(
                {
                    "operator": "assumption_ref",
                    "terms": [assumption_id],
                }
            ),
            source_refs=(assumption_id,),
        )
        for assumption_id in logic_view.assumption_ids
    )
    obligation = ProofObligation(
        obligation_id=logic_view.logic_id,
        statement=logic_view.statement,
        assumption_ids=logic_view.assumption_ids,
        logic_family=logic_view.fragment.value,
        source_refs=logic_view.premise_ids,
        metadata=FrozenMap(
            {
                "catalog_id": catalog_id,
                "catalog_version": catalog_version,
                "identity_profile": LOGIC_IR_PROFILE,
                "policy_id": policy_id,
                "property_id": logic_view.property_id,
                "repository_id": repository_id,
                "required_assurance": required_assurance.value,
                "scope_ids": list(scope_ids),
                "snapshot_id": snapshot_id,
                "supported": logic_view.supported,
                "toolchain_id": toolchain_id,
                "unsupported_reason": logic_view.unsupported_reason,
            }
        ),
    )
    return IRClaim(
        claim_id=logic_view.claim_id,
        declaration_id=logic_view.property_id,
        statement=logic_view.statement,
        assumptions=assumptions,
        obligations=(obligation,),
        domain=MCP_LOGIC_IR_DOMAIN,
        source_refs=tuple(
            sorted(set(logic_view.premise_ids) | {logic_view.property_id})
        ),
        metadata=FrozenMap(
            {
                "logic_id": logic_view.logic_id,
                "operation_id": logic_view.operation_id,
                "schema": MCP_LOGIC_VIEW_SCHEMA,
            }
        ),
    )


def _claim_status(state: ParityState, supported: bool) -> ClaimStatus:
    if not supported:
        return ClaimStatus.UNSUPPORTED
    if state is ParityState.REFUTED:
        return ClaimStatus.REFUTED
    if state is ParityState.NOT_MEASURED:
        return ClaimStatus.NOT_MEASURED
    # A passing/partial analysis is observation, not a proof.
    return ClaimStatus.OPEN


@dataclass(frozen=True, slots=True)
class McpContractObligation:
    """Bound canonical logic, code-proof, and claim-lifecycle projections."""

    logic_view: McpLogicView
    code_obligation: CodeProofObligation
    code_claim: CodeClaimRecord
    shared_ir_claim: Any
    catalog_id: str
    catalog_version: str
    contract_id: str
    toolchain_id: str
    policy_id: str
    invalidators: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.logic_view, McpLogicView):
            raise McpContractObligationError(
                "logic_view must be an McpLogicView"
            )
        if not isinstance(self.code_obligation, CodeProofObligation):
            raise McpContractObligationError(
                "code_obligation must be a CodeProofObligation"
            )
        if not isinstance(self.code_claim, CodeClaimRecord):
            raise McpContractObligationError(
                "code_claim must be a CodeClaimRecord"
            )
        for name in (
            "catalog_id",
            "catalog_version",
            "contract_id",
            "toolchain_id",
            "policy_id",
        ):
            object.__setattr__(
                self, name, _identifier(getattr(self, name), name)
            )
        shared = self.shared_ir_claim
        if isinstance(shared, Mapping):
            shared = _load_shared_ir_claim(shared)
            object.__setattr__(self, "shared_ir_claim", shared)
        if not callable(getattr(shared, "to_dict", None)):
            raise McpContractObligationError(
                "shared_ir_claim must be an ipfs_datasets IRClaim"
            )
        invalidators = tuple(
            MappingProxyType(
                {
                    "kind": _identifier(item.get("kind", ""), "invalidator.kind"),
                    "reason_code": _text(
                        item.get("reason_code", ""),
                        "invalidator.reason_code",
                        required=False,
                    ),
                    "source": _identifier(
                        item.get("source", ""), "invalidator.source"
                    ),
                    "value": _identifier(
                        item.get("value", ""), "invalidator.value"
                    ),
                }
            )
            for item in self.invalidators
        )
        if not invalidators:
            raise McpContractObligationError(
                "compiled obligation requires invalidators"
            )
        compiler_kinds = {
            item["kind"]
            for item in invalidators
            if item["source"] == "compiler"
        }
        required_compiler_kinds = {
            "assumption_set",
            "catalog",
            "policy",
            "premise_set",
            "required_assurance",
            "scope_set",
            "snapshot",
            "toolchain",
        }
        if not required_compiler_kinds.issubset(compiler_kinds):
            raise McpContractObligationError(
                "compiled obligation has incomplete binding invalidators"
            )
        object.__setattr__(
            self,
            "invalidators",
            tuple(
                sorted(
                    invalidators,
                    key=lambda item: (
                        item["source"],
                        item["kind"],
                        item["value"],
                        item["reason_code"],
                    ),
                )
            ),
        )
        self._validate_bindings()

    def _validate_bindings(self) -> None:
        logic = self.logic_view
        code = self.code_obligation
        claim = self.code_claim
        shared = self.shared_ir_claim
        metadata = code.metadata
        if (
            code.statement != logic.statement
            or code.premise_ids != logic.premise_ids
            or claim.premise_ids != logic.premise_ids
            or claim.assumption_ids != logic.assumption_ids
            or claim.property_id != logic.property_id
            or claim.obligation_id != code.obligation_id
            or claim.repository_id != code.repository_id
            or claim.repository_tree_id != code.repository_tree_id
            or claim.scope_ids != code.ast_scope_ids
            or claim.required_assurance is not code.required_assurance
            or claim.toolchain_id != self.toolchain_id
            or claim.policy_id != self.policy_id
            or claim.catalog_version != self.catalog_version
            or self.contract_id != logic.property_id
        ):
            raise McpContractObligationError(
                "logic, code obligation, and claim bindings disagree"
            )
        required_metadata = {
            "assumption_ids": list(logic.assumption_ids),
            "catalog_id": self.catalog_id,
            "catalog_version": self.catalog_version,
            "contract_id": self.contract_id,
            "logic_id": logic.logic_id,
            "logic_fragment": logic.fragment.value,
            "policy_id": self.policy_id,
            "snapshot_id": code.repository_tree_id,
            "supported": logic.supported,
            "toolchain_id": self.toolchain_id,
        }
        if any(metadata.get(key) != value for key, value in required_metadata.items()):
            raise McpContractObligationError(
                "code obligation omits or changes a mandatory binding"
            )
        shared_obligation = shared.obligations[0]
        shared_metadata = shared_obligation.metadata.to_dict()
        required_shared_metadata = {
            "catalog_id": self.catalog_id,
            "catalog_version": self.catalog_version,
            "identity_profile": LOGIC_IR_PROFILE,
            "policy_id": self.policy_id,
            "property_id": self.contract_id,
            "repository_id": code.repository_id,
            "required_assurance": code.required_assurance.value,
            "scope_ids": list(code.ast_scope_ids),
            "snapshot_id": code.repository_tree_id,
            "supported": logic.supported,
            "toolchain_id": self.toolchain_id,
            "unsupported_reason": logic.unsupported_reason,
        }
        if (
            shared.claim_id != logic.claim_id
            or shared_obligation.obligation_id != logic.logic_id
            or tuple(shared_obligation.assumption_ids) != logic.assumption_ids
            or tuple(shared_obligation.source_refs) != logic.premise_ids
            or shared_obligation.statement != logic.statement
            or any(
                shared_metadata.get(key) != value
                for key, value in required_shared_metadata.items()
            )
        ):
            raise McpContractObligationError(
                "shared logic IR is detached from the canonical logic view"
            )

    @property
    def obligation_id(self) -> str:
        return self.code_obligation.obligation_id

    @property
    def property_id(self) -> str:
        return self.contract_id

    @property
    def premise_ids(self) -> tuple[str, ...]:
        return self.logic_view.premise_ids

    @property
    def assumption_ids(self) -> tuple[str, ...]:
        return self.logic_view.assumption_ids

    @property
    def snapshot_id(self) -> str:
        return self.code_obligation.repository_tree_id

    @property
    def scope_ids(self) -> tuple[str, ...]:
        return self.code_obligation.ast_scope_ids

    @property
    def required_assurance(self) -> AssuranceLevel:
        return self.code_obligation.required_assurance

    @property
    def logic_fragment(self) -> LogicFragment:
        return self.logic_view.fragment

    @property
    def supported(self) -> bool:
        return self.logic_view.supported

    @property
    def code_proof_obligation(self) -> CodeProofObligation:
        """Descriptive alias for callers using the full interface name."""

        return self.code_obligation

    @property
    def claim_record(self) -> CodeClaimRecord:
        return self.code_claim

    @property
    def logic_ir(self) -> Any:
        return self.shared_ir_claim

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": MCP_CONTRACT_OBLIGATION_SCHEMA,
            "interface": MCP_CONTRACT_OBLIGATIONS_INTERFACE,
            "version": MCP_CONTRACT_OBLIGATION_VERSION,
            "logic_view": self.logic_view.to_dict(),
            "code_obligation": self.code_obligation.to_dict(),
            "code_claim": self.code_claim.to_dict(),
            "shared_ir_claim": self.shared_ir_claim.to_dict(),
            "catalog_id": self.catalog_id,
            "catalog_version": self.catalog_version,
            "contract_id": self.contract_id,
            "toolchain_id": self.toolchain_id,
            "policy_id": self.policy_id,
            "invalidators": [dict(item) for item in self.invalidators],
        }

    @property
    def content_id(self) -> str:
        return content_identity(self._identity_payload())

    @property
    def compiled_obligation_id(self) -> str:
        return self.content_id

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "compiled_obligation_id": self.compiled_obligation_id,
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    def canonical_bytes(self) -> bytes:
        return self.to_json().encode("utf-8")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "McpContractObligation":
        if not isinstance(value, Mapping):
            raise McpContractObligationError(
                "compiled contract obligation must be an object"
            )
        allowed = {
            "schema",
            "interface",
            "version",
            "logic_view",
            "code_obligation",
            "code_claim",
            "shared_ir_claim",
            "catalog_id",
            "catalog_version",
            "contract_id",
            "toolchain_id",
            "policy_id",
            "invalidators",
            "compiled_obligation_id",
        }
        if set(value).difference(allowed):
            raise McpContractObligationError(
                "compiled obligation contains unsupported fields"
            )
        if value.get("schema") not in (None, MCP_CONTRACT_OBLIGATION_SCHEMA):
            raise McpContractObligationError(
                "unsupported compiled-obligation schema"
            )
        if value.get("interface") not in (
            None,
            MCP_CONTRACT_OBLIGATIONS_INTERFACE,
        ):
            raise McpContractObligationError(
                "unsupported compiled-obligation interface"
            )
        if value.get("version") not in (
            None,
            MCP_CONTRACT_OBLIGATION_VERSION,
        ):
            raise McpContractObligationError(
                "unsupported compiled-obligation version"
            )
        try:
            result = cls(
                logic_view=McpLogicView.from_dict(value.get("logic_view") or {}),
                code_obligation=CodeProofObligation.from_dict(
                    value.get("code_obligation") or {}
                ),
                code_claim=CodeClaimRecord.from_dict(
                    value.get("code_claim") or {}
                ),
                shared_ir_claim=_load_shared_ir_claim(
                    value.get("shared_ir_claim") or {}
                ),
                catalog_id=str(value.get("catalog_id") or ""),
                catalog_version=str(value.get("catalog_version") or ""),
                contract_id=str(value.get("contract_id") or ""),
                toolchain_id=str(value.get("toolchain_id") or ""),
                policy_id=str(value.get("policy_id") or ""),
                invalidators=tuple(value.get("invalidators") or ()),
            )
        except McpContractObligationError:
            raise
        except (TypeError, ValueError) as exc:
            raise McpContractObligationError(
                f"invalid compiled obligation: {exc}"
            ) from exc
        claimed_id = value.get("compiled_obligation_id")
        if claimed_id is not None and claimed_id != result.compiled_obligation_id:
            raise McpContractObligationError(
                "compiled-obligation identity does not match canonical content"
            )
        return result

    @classmethod
    def from_json(cls, value: str) -> "McpContractObligation":
        try:
            payload = json.loads(value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise McpContractObligationError(
                "compiled-obligation JSON is malformed"
            ) from exc
        return cls.from_dict(payload)


def compile_contract_claim(
    claim: ContractParityClaim | Mapping[str, Any],
    *,
    catalog: McpContractCatalog,
    contract: ContractRecord | str | None = None,
    contract_id: str = "",
    repository_id: str,
    snapshot_id: str = "",
    repository_tree_id: str = "",
    tree_id: str = "",
    scope_ids: Sequence[str] = (),
    ast_scope_ids: Sequence[str] = (),
    assumption_ids: Sequence[str] = (),
    toolchain_id: str,
    policy_id: str,
    required_assurance: AssuranceLevel | str = AssuranceLevel.KERNEL_VERIFIED,
) -> McpContractObligation:
    """Compile one reviewed claim using only closed, structured templates.

    ``snapshot_id`` and ``repository_tree_id`` are aliases.  If both are
    supplied they must be identical.  Premise order has set semantics and is
    normalized by :class:`ContractParityClaim`; the resulting identity is
    therefore invariant to input premise order.
    """

    normalized_claim = _claim(claim)
    if contract_id:
        normalized_contract_id = _identifier(contract_id, "contract_id")
        if isinstance(contract, ContractRecord):
            supplied_contract_id = contract.contract_id
        else:
            supplied_contract_id = str(contract or "")
        if supplied_contract_id and supplied_contract_id != normalized_contract_id:
            raise McpContractObligationError(
                "contract and contract_id disagree"
            )
        contract = normalized_contract_id
    property_record = _resolve_contract(catalog, normalized_claim, contract)
    repository = _identifier(repository_id, "repository_id")
    snapshot_values = tuple(
        value for value in (snapshot_id, repository_tree_id, tree_id) if value
    )
    if len(set(snapshot_values)) > 1:
        raise McpContractObligationError(
            "snapshot_id, repository_tree_id, and tree_id disagree"
        )
    snapshot = snapshot_values[0] if snapshot_values else ""
    snapshot = _identifier(snapshot, "snapshot_id")
    if scope_ids and ast_scope_ids and tuple(scope_ids) != tuple(ast_scope_ids):
        raise McpContractObligationError(
            "scope_ids and ast_scope_ids disagree"
        )
    scopes = _ids(scope_ids or ast_scope_ids, "scope_ids", required=True)
    assumptions = _ids(assumption_ids, "assumption_ids")
    toolchain = _identifier(toolchain_id, "toolchain_id")
    policy = _identifier(policy_id, "policy_id")
    required = _assurance(required_assurance)
    premises = _ids(normalized_claim.premise_ids, "premise_ids", required=True)

    fragment, operator, unsupported_reason = _logic_for_claim(normalized_claim)
    logic_view = McpLogicView(
        family=normalized_claim.family,
        fragment=fragment,
        operator=operator,
        operation_id=normalized_claim.operation_id,
        property_id=property_record.contract_id,
        claim_id=normalized_claim.claim_id,
        premise_ids=premises,
        assumption_ids=assumptions,
        supported=fragment is not LogicFragment.UNSUPPORTED,
        unsupported_reason=unsupported_reason,
    )

    catalog_invalidators = tuple(
        {
            "source": "catalog",
            "kind": item.kind.value,
            "value": item.value,
            "reason_code": item.reason_code,
        }
        for item in property_record.invalidators
    )
    compiler_invalidators = (
        {
            "source": "compiler",
            "kind": "snapshot",
            "value": snapshot,
            "reason_code": "snapshot_changed",
        },
        {
            "source": "compiler",
            "kind": "scope_set",
            "value": content_identity({"scope_ids": list(scopes)}),
            "reason_code": "scope_changed",
        },
        {
            "source": "compiler",
            "kind": "premise_set",
            "value": content_identity({"premise_ids": list(premises)}),
            "reason_code": "premises_changed",
        },
        {
            "source": "compiler",
            "kind": "assumption_set",
            "value": content_identity({"assumption_ids": list(assumptions)}),
            "reason_code": "assumptions_changed",
        },
        {
            "source": "compiler",
            "kind": "catalog",
            "value": catalog.catalog_id,
            "reason_code": "catalog_changed",
        },
        {
            "source": "compiler",
            "kind": "toolchain",
            "value": toolchain,
            "reason_code": "toolchain_changed",
        },
        {
            "source": "compiler",
            "kind": "policy",
            "value": policy,
            "reason_code": "policy_changed",
        },
        {
            "source": "compiler",
            "kind": "required_assurance",
            "value": required.value,
            "reason_code": "required_assurance_changed",
        },
    )
    invalidators = catalog_invalidators + compiler_invalidators
    metadata = {
        "interface": MCP_CONTRACT_OBLIGATIONS_INTERFACE,
        "assumption_ids": list(assumptions),
        "catalog_id": catalog.catalog_id,
        "catalog_version": catalog.catalog_version,
        "claim_id": normalized_claim.claim_id,
        "contract_id": property_record.contract_id,
        "identity_profile": LOGIC_IR_PROFILE,
        "invalidators": [dict(item) for item in invalidators],
        "logic_fragment": fragment.value,
        "logic_id": logic_view.logic_id,
        "operation_id": normalized_claim.operation_id,
        "policy_id": policy,
        "snapshot_id": snapshot,
        "supported": logic_view.supported,
        "toolchain_id": toolchain,
        "unsupported_reason": unsupported_reason,
    }
    code_obligation = CodeProofObligation(
        repository_id=repository,
        repository_tree_id=snapshot,
        ast_scope_ids=scopes,
        statement=logic_view.statement,
        premise_ids=premises,
        template_id=f"mcp-contract/{fragment.value}",
        template_version=MCP_CONTRACT_OBLIGATION_VERSION,
        template_semantic_hash=logic_view.logic_id,
        invariant_class=f"mcp_contract:{normalized_claim.family.value}",
        task_id=normalized_claim.operation_id,
        required_assurance=required,
        fallback_checks=(
            ("mcp-contract:unsupported-fragment",)
            if not logic_view.supported
            else ()
        ),
        metadata=metadata,
    )
    code_claim = claim_from_obligation(
        code_obligation,
        property_id=property_record.contract_id,
        claim_family=(
            ClaimFamily.UNSUPPORTED
            if not logic_view.supported
            else _CODE_FAMILY[normalized_claim.family]
        ),
        assumption_ids=assumptions,
        producer_id=MCP_OBLIGATION_COMPILER_ID,
        toolchain_id=toolchain,
        policy_id=policy,
        catalog_version=catalog.catalog_version,
        status=_claim_status(normalized_claim.state, logic_view.supported),
        metadata={
            "catalog_id": catalog.catalog_id,
            "contract_claim_id": normalized_claim.claim_id,
            "logic_fragment": fragment.value,
            "logic_id": logic_view.logic_id,
            "observation_state": normalized_claim.state.value,
            "supported": logic_view.supported,
            "unsupported_reason": unsupported_reason,
        },
    )
    # ``claim_from_obligation`` forces unsupported templates to unsupported;
    # for supported templates preserve only the analysis-derived lifecycle.
    if logic_view.supported:
        code_claim = code_claim.with_updates(
            status=_claim_status(normalized_claim.state, True)
        )
    shared_ir = _build_shared_ir(
        logic_view,
        catalog_id=catalog.catalog_id,
        catalog_version=catalog.catalog_version,
        repository_id=repository,
        snapshot_id=snapshot,
        scope_ids=scopes,
        toolchain_id=toolchain,
        policy_id=policy,
        required_assurance=required,
    )
    return McpContractObligation(
        logic_view=logic_view,
        code_obligation=code_obligation,
        code_claim=code_claim,
        shared_ir_claim=shared_ir,
        catalog_id=catalog.catalog_id,
        catalog_version=catalog.catalog_version,
        contract_id=property_record.contract_id,
        toolchain_id=toolchain,
        policy_id=policy,
        invalidators=invalidators,
    )


def compile_contract_claims(
    claims: Sequence[ContractParityClaim | Mapping[str, Any]],
    **bindings: Any,
) -> tuple[McpContractObligation, ...]:
    """Compile a deterministic set of claims with shared exact bindings."""

    if isinstance(claims, (str, bytes, bytearray)) or not isinstance(
        claims, Sequence
    ):
        raise McpContractObligationError("claims must be a sequence")
    results = tuple(compile_contract_claim(item, **bindings) for item in claims)
    by_id = {item.compiled_obligation_id: item for item in results}
    if len(by_id) != len(results):
        raise McpContractObligationError("claims compile to duplicate obligations")
    return tuple(by_id[key] for key in sorted(by_id))


# Compatibility spellings for callers emphasizing MCP or code-proof output.
compile_mcp_contract_claim = compile_contract_claim
compile_mcp_contract_obligation = compile_contract_claim
CompiledMcpContractObligation = McpContractObligation
CanonicalMcpLogicView = McpLogicView


def compile_code_proof_obligation(
    claim: ContractParityClaim | Mapping[str, Any],
    **bindings: Any,
) -> CodeProofObligation:
    """Compile and return the existing ``CodeProofObligation`` projection."""

    return compile_contract_claim(claim, **bindings).code_obligation


__all__ = [
    "MCP_CONTRACT_OBLIGATIONS_INTERFACE",
    "MCP_CONTRACT_OBLIGATION_INTERFACE",
    "MCP_CONTRACT_OBLIGATION_SCHEMA",
    "MCP_LOGIC_VIEW_SCHEMA",
    "MCP_LOGIC_EXPRESSION_SCHEMA",
    "MCP_CONTRACT_OBLIGATION_VERSION",
    "MCP_LOGIC_IR_DOMAIN",
    "MCP_LOGIC_IR_SCHEMA_VERSION",
    "MCP_OBLIGATION_COMPILER_ID",
    "McpContractObligationError",
    "LogicFragment",
    "LogicOperator",
    "McpLogicView",
    "CanonicalMcpLogicView",
    "McpContractObligation",
    "CompiledMcpContractObligation",
    "compile_contract_claim",
    "compile_mcp_contract_claim",
    "compile_mcp_contract_obligation",
    "compile_code_proof_obligation",
    "compile_contract_claims",
]
