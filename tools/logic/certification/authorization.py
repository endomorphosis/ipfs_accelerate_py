#!/usr/bin/env python3
"""Semantic certification for reference Datalog and SecPAL authorization.

``AuthorizationSemanticCertification@1`` / FVT-G102 (FVT-038).

Promotes the already-usable in-process Datalog and SecPAL-style engines only
after full authorization semantics are certified. This lane owns the reference
authorization corpus and focused checks; it never installs external shadows
and never edits the central multi-prover certificate.

Acceptance covered
------------------
* both engines exercise allow, deny, unknown, conflict, scoped delegation,
  revocation, negative, and malformed inputs;
* rule, principal, scope, and delegation mutations change or quarantine the
  verdict;
* counterexamples replay deterministically;
* receipts bind the exact policy digest and engine identity;
* certification grants authorization-decision authority only — never theorem
  authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Final, Iterable, Mapping, Sequence

# Allow running as a script from a worktree without an installed package.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DATASETS_ROOT = _REPO_ROOT / "ipfs_datasets_py"
for candidate in (_REPO_ROOT, _DATASETS_ROOT):
    text = str(candidate)
    if text not in sys.path:
        sys.path.insert(0, text)

from ipfs_datasets_py.logic.backends.datalog.adapters import (  # noqa: E402
    DATALOG_AUTHORIZATION_BACKEND_VERSION,
    DEFAULT_AUTHORIZATION_FIXTURES,
    SECPAL_AUTHORIZATION_BACKEND_VERSION,
    AuthorizationBackendError,
    AuthorizationFixture,
    DatalogAuthorizationBackend,
    EngineKind,
    EvaluationReceipt,
    ReferenceAuthorizationEvaluator,
    SecPALAuthorizationBackend,
)
from ipfs_datasets_py.logic.backends.results import (  # noqa: E402
    ResultAuthority,
    ResultStatus,
)
from ipfs_datasets_py.logic.backends.toolchain_roles import (  # noqa: E402
    ToolchainAuthorityCeiling,
    ToolRole,
    get_tool_role,
)
from ipfs_datasets_py.logic.ir_core.claims import FrozenMap  # noqa: E402
from ipfs_datasets_py.logic.ir_core.protocols import (  # noqa: E402
    BackendRequest,
    ExecutionBounds,
    QueryKind,
)
from ipfs_datasets_py.logic.software_verification.authorization import (  # noqa: E402
    AuthorizationAtom,
    AuthorizationEvidenceAuthority,
    AuthorizationIR,
    AuthorizationRule,
    AuthorizationTerm,
    AuthorizationValidationError,
    DecisionOutcome,
    DecisionQuery,
    EffectKind,
    GeneratedCodeCorrectness,
    RuleKind,
)

# Optional lane binding helpers (present after FVT-037).
try:  # pragma: no cover - import surface varies by worktree packaging
    from tools.logic.certification.roles import (  # type: ignore
        bind_lane_handler as _bind_lane_handler,
        build_role_aware_policy as _build_role_aware_policy,
    )
except Exception:  # pragma: no cover
    _bind_lane_handler = None  # type: ignore[assignment]
    _build_role_aware_policy = None  # type: ignore[assignment]


INTERFACE: Final = "AuthorizationSemanticCertification@1"
SCHEMA_VERSION: Final = "authorization-semantic-certification/v1"
MANIFEST_SCHEMA: Final = "authorization-semantic-certification-manifest/v1"
GOAL_ID: Final = "FVT-G102"
TASK_ID: Final = "FVT-038"
PROGRAM: Final = "formal-verification-tactician/authorization-certification"
LANE_ID: Final = "datalog_secpal"
HANDLER_ID: Final = "authorization_semantic_certification@1"
CERTIFICATION_SURFACE: Final = "tools.logic.certification.authorization"
AUTHORITY_CEILING: Final = ToolchainAuthorityCeiling.AUTHORIZATION.value

ENGINE_DATALOG: Final = "datalog-authorization"
ENGINE_SECPAL: Final = "secpal-authorization"
REFERENCE_ENGINES: Final = (ENGINE_DATALOG, ENGINE_SECPAL)

DEFAULT_MANIFEST_RELATIVE: Final = Path(
    "test/fixtures/formal_verification/toolchains/authorization/manifest.json"
)

# Closed categories required by FVT-G102 acceptance.
REQUIRED_CATEGORIES: Final = frozenset(
    {
        "allow",
        "deny",
        "unknown",
        "conflict",
        "delegation",
        "revocation",
        "negative",
        "malformed",
    }
)
REQUIRED_MUTATION_KINDS: Final = frozenset(
    {"rule", "principal", "scope", "delegation"}
)

CHECK_KINDS: Final = frozenset({"positive", "negative", "mutation", "replay", "malformed"})


class AuthorizationSemanticCertificationError(ValueError):
    """Raised when semantic certification inputs or results are invalid."""


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One hermetic semantic check outcome."""

    check_id: str
    kind: str
    status: str
    expected: str
    observed: str
    detail: str = ""
    policy_digest: str = ""
    engine_id: str = ""
    authority: str = AUTHORITY_CEILING
    is_theorem_authority: bool = False

    def __post_init__(self) -> None:
        if self.kind not in CHECK_KINDS and self.kind not in {
            "positive",
            "negative",
            "mutation",
            "replay",
            "malformed",
            "authority",
        }:
            # Allow authority as a closed extra kind used by receipts.
            if self.kind != "authority":
                raise AuthorizationSemanticCertificationError(
                    f"unknown check kind {self.kind!r}"
                )
        if self.status not in {"passed", "failed", "quarantined", "error", "skipped"}:
            raise AuthorizationSemanticCertificationError(
                f"unknown check status {self.status!r}"
            )
        if self.is_theorem_authority:
            raise AuthorizationSemanticCertificationError(
                "authorization semantic checks cannot claim theorem authority"
            )
        if self.authority not in {AUTHORITY_CEILING, "authorization"}:
            raise AuthorizationSemanticCertificationError(
                "authorization semantic checks may only claim authorization authority"
            )

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "authority": self.authority,
            "check_id": self.check_id,
            "detail": self.detail,
            "engine_id": self.engine_id,
            "expected": self.expected,
            "is_theorem_authority": False,
            "kind": self.kind,
            "observed": self.observed,
            "policy_digest": self.policy_digest,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class CaseSpec:
    """Compact recipe for one semantic corpus case (no bulk IR dump)."""

    case_id: str
    category: str
    expected_outcome: str
    recipe: str
    base_fixture_id: str = ""
    mutation_kind: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_fixture_id": self.base_fixture_id,
            "case_id": self.case_id,
            "category": self.category,
            "expected_outcome": self.expected_outcome,
            "mutation_kind": self.mutation_kind,
            "notes": self.notes,
            "recipe": self.recipe,
        }


@dataclass
class EngineRunRecord:
    """One engine evaluation used for binding and replay."""

    engine_id: str
    case_id: str
    outcome: str
    status: str
    policy_digest: str
    request_digest: str
    receipt_id: str
    authority: str
    is_theorem_authority: bool
    bound_rule_ids: tuple[str, ...] = ()
    explanation_digest: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "authority": self.authority,
            "bound_rule_ids": list(self.bound_rule_ids),
            "case_id": self.case_id,
            "engine_id": self.engine_id,
            "error": self.error,
            "explanation_digest": self.explanation_digest,
            "is_theorem_authority": self.is_theorem_authority,
            "outcome": self.outcome,
            "policy_digest": self.policy_digest,
            "receipt_id": self.receipt_id,
            "request_digest": self.request_digest,
            "status": self.status,
        }


@dataclass
class EngineCertification:
    """Per-engine semantic certification summary."""

    engine_id: str
    interface_version: str
    usable: bool
    certified: bool
    authority_ceiling: str
    checks: list[CheckResult] = field(default_factory=list)
    case_results: list[EngineRunRecord] = field(default_factory=list)
    block_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "authority_ceiling": self.authority_ceiling,
            "block_reasons": list(self.block_reasons),
            "case_results": [item.to_dict() for item in self.case_results],
            "certified": self.certified,
            "checks": [item.to_dict() for item in self.checks],
            "engine_id": self.engine_id,
            "interface_version": self.interface_version,
            "usable": self.usable,
        }


# ---------------------------------------------------------------------------
# Corpus recipes (generators — not bulk golden IR dumps)
# ---------------------------------------------------------------------------


def _fixture_by_id(fixture_id: str) -> AuthorizationFixture:
    for item in DEFAULT_AUTHORIZATION_FIXTURES:
        if item.fixture_id == fixture_id:
            return item
    raise AuthorizationSemanticCertificationError(
        f"unknown base fixture {fixture_id!r}"
    )


def _fixture_by_category(category: str) -> AuthorizationFixture:
    for item in DEFAULT_AUTHORIZATION_FIXTURES:
        if item.category == category:
            return item
    raise AuthorizationSemanticCertificationError(
        f"no default fixture for category {category!r}"
    )


def _const(value: str, sort: str = "principal") -> AuthorizationTerm:
    return AuthorizationTerm.constant(value, sort)


def _var(value: str, sort: str = "principal") -> AuthorizationTerm:
    return AuthorizationTerm.variable(value, sort)


def _mapped_from(document: AuthorizationIR) -> dict[str, tuple[str, ...]]:
    source_ids = tuple(item.ref_id for item in document.sources) or (
        "source:authz-fixtures",
    )
    span_ids = tuple(item.span_id for item in document.spans) or (
        "span:authz-fixtures",
    )
    return {"source_ref_ids": source_ids, "span_ids": span_ids}


def build_revocation_case() -> tuple[AuthorizationIR, DecisionQuery, DecisionOutcome]:
    """Allow baseline after revoking Alice's admin membership must deny."""

    allow = _fixture_by_category("allow")
    roles = tuple(
        replace(role, member_principal_ids=())
        if role.role_id == "role:admin"
        else role
        for role in allow.document.roles
    )
    facts = tuple(
        fact
        for fact in allow.document.facts
        if fact.fact_id != "fact:alice-admin"
    )
    document = replace(
        allow.document,
        roles=roles,
        facts=facts,
        document_id="",
        metadata=FrozenMap(
            {
                "fixture_set": "authorization-semantic-certification",
                "recipe": "revoke_admin_membership",
            }
        ),
    )
    return document, allow.query, DecisionOutcome.DENY


def build_negative_case() -> tuple[AuthorizationIR, DecisionQuery, DecisionOutcome]:
    """Negative polarity / out-of-policy request must not allow."""

    allow = _fixture_by_category("allow")
    query = DecisionQuery(
        "query:negative-non-admin-write",
        principal_id="principal:bob",
        action="write",
        resource="docs/payroll",
        source_ref_ids=allow.query.source_ref_ids,
        span_ids=allow.query.span_ids,
    )
    return allow.document, query, DecisionOutcome.UNKNOWN


def apply_rule_mutation(
    document: AuthorizationIR,
) -> AuthorizationIR:
    """Change the allow rule action from read to write (must drop allow)."""

    mapped = _mapped_from(document)
    rules: list[AuthorizationRule] = []
    for rule in document.rules:
        if rule.rule_id == "rule:admin-may-read":
            rules.append(
                AuthorizationRule(
                    "rule:admin-may-write",
                    head=AuthorizationAtom(
                        "pred:may",
                        (
                            _var("P"),
                            _const("write", "action"),
                            _var("R", "resource"),
                        ),
                    ),
                    body=rule.body,
                    kind=RuleKind.DATALOG,
                    effect=EffectKind.ALLOW,
                    stratum=rule.stratum,
                    **mapped,
                )
            )
        else:
            rules.append(rule)
    return replace(document, rules=tuple(rules), document_id="")


def apply_principal_mutation(
    document: AuthorizationIR,
) -> AuthorizationIR:
    """Reassign admin membership from Alice to Carol (Alice must lose allow)."""

    roles = tuple(
        replace(role, member_principal_ids=("principal:carol",))
        if role.role_id == "role:admin"
        else role
        for role in document.roles
    )
    facts = []
    for fact in document.facts:
        if fact.fact_id == "fact:alice-admin":
            facts.append(
                replace(
                    fact,
                    atom=AuthorizationAtom(
                        "pred:role",
                        (
                            _const("principal:carol", "principal"),
                            _const("role:admin", "role"),
                        ),
                    ),
                )
            )
        else:
            facts.append(fact)
    return replace(document, roles=roles, facts=tuple(facts), document_id="")


def apply_scope_mutation(
    document: AuthorizationIR,
) -> AuthorizationIR:
    """Narrow delegated resource scope away from the query resource."""

    if not document.delegations:
        raise AuthorizationSemanticCertificationError(
            "scope mutation requires a document with delegations"
        )
    delegations = tuple(
        replace(item, resource_scope=("docs/other/",))
        if item.delegation_id == "delegation:alice-bob"
        else item
        for item in document.delegations
    )
    return replace(document, delegations=delegations, document_id="")


def apply_delegation_mutation(
    document: AuthorizationIR,
) -> AuthorizationIR:
    """Drop all delegations (delegated allow must disappear)."""

    if not document.delegations:
        raise AuthorizationSemanticCertificationError(
            "delegation mutation requires a document with delegations"
        )
    return replace(document, delegations=(), document_id="")


MUTATION_APPLIERS: Final[Mapping[str, Callable[[AuthorizationIR], AuthorizationIR]]] = {
    "rule": apply_rule_mutation,
    "principal": apply_principal_mutation,
    "scope": apply_scope_mutation,
    "delegation": apply_delegation_mutation,
}


def default_case_specs() -> tuple[CaseSpec, ...]:
    """Compact recipe list for the authorization semantic corpus."""

    return (
        CaseSpec(
            case_id="case:allow",
            category="allow",
            expected_outcome=DecisionOutcome.ALLOW.value,
            recipe="default_fixture",
            base_fixture_id="fixture:allow",
            notes="Admin Alice may read sensitive payroll.",
        ),
        CaseSpec(
            case_id="case:deny",
            category="deny",
            expected_outcome=DecisionOutcome.DENY.value,
            recipe="default_fixture",
            base_fixture_id="fixture:deny",
            notes="Non-admin Bob is denied on sensitive payroll.",
        ),
        CaseSpec(
            case_id="case:unknown",
            category="unknown",
            expected_outcome=DecisionOutcome.UNKNOWN.value,
            recipe="default_fixture",
            base_fixture_id="fixture:unknown",
            notes="No allow or deny evidence for delete.",
        ),
        CaseSpec(
            case_id="case:conflict",
            category="conflict",
            expected_outcome=DecisionOutcome.CONFLICT.value,
            recipe="default_fixture",
            base_fixture_id="fixture:conflict",
            notes="Explicit allow and deny evidence retained as conflict.",
        ),
        CaseSpec(
            case_id="case:delegation",
            category="delegation",
            expected_outcome=DecisionOutcome.ALLOW.value,
            recipe="default_fixture",
            base_fixture_id="fixture:delegation",
            notes="Scoped delegation permits Bob under docs/public/.",
        ),
        CaseSpec(
            case_id="case:revocation",
            category="revocation",
            expected_outcome=DecisionOutcome.DENY.value,
            recipe="revoke_admin_membership",
            base_fixture_id="fixture:allow",
            notes="Revoking Alice admin membership flips allow to deny.",
        ),
        CaseSpec(
            case_id="case:negative",
            category="negative",
            expected_outcome=DecisionOutcome.UNKNOWN.value,
            recipe="non_admin_write_sensitive",
            base_fixture_id="fixture:allow",
            notes="Negative/out-of-policy write must not allow.",
        ),
        CaseSpec(
            case_id="case:malformed",
            category="malformed",
            expected_outcome="error",
            recipe="invalid_authorization_ir_payload",
            notes="Malformed IR must fail closed (error/quarantine), never allow.",
        ),
        CaseSpec(
            case_id="case:mutation.rule",
            category="mutation",
            expected_outcome=DecisionOutcome.UNKNOWN.value,
            recipe="mutate_rule_action_read_to_write",
            base_fixture_id="fixture:allow",
            mutation_kind="rule",
            notes="Rule action mutation must change the allow verdict.",
        ),
        CaseSpec(
            case_id="case:mutation.principal",
            category="mutation",
            expected_outcome=DecisionOutcome.DENY.value,
            recipe="mutate_admin_principal_alice_to_carol",
            base_fixture_id="fixture:allow",
            mutation_kind="principal",
            notes="Principal mutation must change the allow verdict.",
        ),
        CaseSpec(
            case_id="case:mutation.scope",
            category="mutation",
            expected_outcome=DecisionOutcome.UNKNOWN.value,
            recipe="mutate_delegation_resource_scope",
            base_fixture_id="fixture:delegation",
            mutation_kind="scope",
            notes="Scope mutation must drop delegated allow.",
        ),
        CaseSpec(
            case_id="case:mutation.delegation",
            category="mutation",
            expected_outcome=DecisionOutcome.UNKNOWN.value,
            recipe="drop_delegations",
            base_fixture_id="fixture:delegation",
            mutation_kind="delegation",
            notes="Delegation removal must drop delegated allow.",
        ),
    )


def build_default_manifest() -> dict[str, Any]:
    """Machine-readable compact corpus manifest (recipes only)."""

    specs = default_case_specs()
    return {
        "schema_version": MANIFEST_SCHEMA,
        "interface": INTERFACE,
        "goal_id": GOAL_ID,
        "task_id": TASK_ID,
        "program": PROGRAM,
        "lane_id": LANE_ID,
        "handler_id": HANDLER_ID,
        "certification_surface": CERTIFICATION_SURFACE,
        "authority_ceiling": AUTHORITY_CEILING,
        "forbids_theorem_authority": True,
        "engines": list(REFERENCE_ENGINES),
        "engine_interfaces": {
            ENGINE_DATALOG: DATALOG_AUTHORIZATION_BACKEND_VERSION,
            ENGINE_SECPAL: SECPAL_AUTHORIZATION_BACKEND_VERSION,
        },
        "required_categories": sorted(REQUIRED_CATEGORIES),
        "required_mutation_kinds": sorted(REQUIRED_MUTATION_KINDS),
        "check_kinds": sorted(CHECK_KINDS),
        "case_recipes": [item.to_dict() for item in specs],
        "policy": {
            "in_process_only": True,
            "no_external_shadow_install": True,
            "no_central_certificate_edit": True,
            "receipts_bind_policy_and_engine": True,
            "authorization_decision_authority_only": True,
            "counterexamples_replay_deterministically": True,
            "mutations_must_change_or_quarantine": True,
        },
    }


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    """Load the checked-in manifest or fall back to the default recipe set."""

    target = path or (_REPO_ROOT / DEFAULT_MANIFEST_RELATIVE)
    if target.is_file():
        payload = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise AuthorizationSemanticCertificationError(
                "authorization manifest must be a JSON object"
            )
        if payload.get("interface") != INTERFACE:
            raise AuthorizationSemanticCertificationError(
                f"manifest interface must be {INTERFACE}"
            )
        return payload
    return build_default_manifest()


def write_manifest(path: Path | None = None) -> Path:
    """Write the compact default manifest atomically-friendly (overwrite)."""

    target = path or (_REPO_ROOT / DEFAULT_MANIFEST_RELATIVE)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = build_default_manifest()
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def materialize_case(
    spec: CaseSpec,
) -> tuple[AuthorizationIR | None, DecisionQuery | None, str]:
    """Expand a compact recipe into (document, query, expected_token).

    Malformed cases return ``(None, None, "error")``.
    """

    if spec.category == "malformed" or spec.recipe == "invalid_authorization_ir_payload":
        return None, None, "error"

    if spec.recipe == "default_fixture":
        fixture = _fixture_by_id(spec.base_fixture_id)
        return fixture.document, fixture.query, fixture.expected_outcome.value

    if spec.recipe == "revoke_admin_membership":
        document, query, expected = build_revocation_case()
        return document, query, expected.value

    if spec.recipe == "non_admin_write_sensitive":
        document, query, expected = build_negative_case()
        return document, query, expected.value

    if spec.category == "mutation" or spec.mutation_kind:
        kind = spec.mutation_kind or spec.recipe
        if kind not in MUTATION_APPLIERS and kind in {
            "mutate_rule_action_read_to_write": "rule",
            "mutate_admin_principal_alice_to_carol": "principal",
            "mutate_delegation_resource_scope": "scope",
            "drop_delegations": "delegation",
        }:
            # recipe aliases map below
            pass
        recipe_to_kind = {
            "mutate_rule_action_read_to_write": "rule",
            "mutate_admin_principal_alice_to_carol": "principal",
            "mutate_delegation_resource_scope": "scope",
            "drop_delegations": "delegation",
            "rule": "rule",
            "principal": "principal",
            "scope": "scope",
            "delegation": "delegation",
        }
        mutation_kind = recipe_to_kind.get(spec.mutation_kind) or recipe_to_kind.get(
            spec.recipe
        )
        if mutation_kind is None:
            raise AuthorizationSemanticCertificationError(
                f"unknown mutation recipe {spec.recipe!r}"
            )
        base = _fixture_by_id(spec.base_fixture_id)
        applier = MUTATION_APPLIERS[mutation_kind]
        document = applier(base.document)
        return document, base.query, spec.expected_outcome

    raise AuthorizationSemanticCertificationError(
        f"unable to materialize case {spec.case_id!r} recipe={spec.recipe!r}"
    )


# ---------------------------------------------------------------------------
# Engine execution
# ---------------------------------------------------------------------------


def _backend_for(engine_id: str):
    if engine_id == ENGINE_DATALOG:
        return DatalogAuthorizationBackend(), DATALOG_AUTHORIZATION_BACKEND_VERSION
    if engine_id == ENGINE_SECPAL:
        return SecPALAuthorizationBackend(), SECPAL_AUTHORIZATION_BACKEND_VERSION
    raise AuthorizationSemanticCertificationError(f"unknown engine {engine_id!r}")


def _document_with_query(
    document: AuthorizationIR, query: DecisionQuery
) -> AuthorizationIR:
    """Ensure the evaluated document carries the query under test."""

    existing = {item.query_id: item for item in document.queries}
    if existing.get(query.query_id) == query:
        return document
    existing[query.query_id] = query
    return replace(document, queries=tuple(existing.values()), document_id="")


def _make_request(
    engine_id: str,
    document: AuthorizationIR,
    query: DecisionQuery,
    *,
    request_id: str,
) -> BackendRequest:
    encoding = "secpal" if engine_id == ENGINE_SECPAL else "authorization-ir"
    family = "secpal" if engine_id == ENGINE_SECPAL else "authorization"
    bound_document = _document_with_query(document, query)
    return BackendRequest(
        request_id=request_id,
        claim_id=f"claim:{request_id}",
        declaration_id=f"declaration:{request_id}",
        claim_digest="a" * 64,
        obligation_id=f"obligation:{request_id}",
        obligation_digest="b" * 64,
        assumption_ids=("assumption:reviewed-policy",),
        logic_family=family,
        query_kind=QueryKind.POLICY_APPROVAL,
        bounds=ExecutionBounds(timeout_ms=500, max_steps=256),
        payload=FrozenMap(
            {
                "encoding": encoding,
                "authorization_ir": bound_document.to_dict(),
                "query_id": query.query_id,
                "query": query.to_dict(),
            }
        ),
        requested_backend_id=engine_id,
    )


def _stable_json_digest(payload: Mapping[str, Any] | Sequence[Any] | str) -> str:
    if isinstance(payload, str):
        raw = payload.encode("utf-8")
    else:
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _bound_rule_ids(receipt: EvaluationReceipt) -> tuple[str, ...]:
    if receipt.explanation is None:
        return ()
    return tuple(
        step.reference_id
        for step in receipt.explanation.steps
        if step.kind.value == "rule" and step.reference_id
    )


def run_engine_case(
    engine_id: str,
    case_id: str,
    document: AuthorizationIR | None,
    query: DecisionQuery | None,
    *,
    expect_error: bool = False,
) -> EngineRunRecord:
    """Run one case on one in-process engine and return a bound record."""

    backend, _version = _backend_for(engine_id)
    if expect_error or document is None or query is None:
        # Force a malformed payload path.
        request = BackendRequest(
            request_id=f"request:{case_id}:{engine_id}:malformed",
            claim_id=f"claim:{case_id}",
            declaration_id=f"declaration:{case_id}",
            claim_digest="c" * 64,
            obligation_id=f"obligation:{case_id}",
            obligation_digest="d" * 64,
            assumption_ids=(),
            logic_family="authorization",
            query_kind=QueryKind.POLICY_APPROVAL,
            bounds=ExecutionBounds(timeout_ms=250, max_steps=64),
            payload=FrozenMap(
                {
                    "encoding": "authorization-ir",
                    "authorization_ir": {"bogus": True, "not": "an ir"},
                    "query_id": "query:missing",
                }
            ),
            requested_backend_id=engine_id,
        )
        try:
            backend.run(request)
            return EngineRunRecord(
                engine_id=engine_id,
                case_id=case_id,
                outcome="allow",
                status="unexpected_success",
                policy_digest="",
                request_digest=request.digest,
                receipt_id="",
                authority=AUTHORITY_CEILING,
                is_theorem_authority=False,
                error="malformed input was accepted",
            )
        except (AuthorizationBackendError, AuthorizationValidationError, ValueError) as exc:
            return EngineRunRecord(
                engine_id=engine_id,
                case_id=case_id,
                outcome="error",
                status="error",
                policy_digest="",
                request_digest=request.digest,
                receipt_id="",
                authority=AUTHORITY_CEILING,
                is_theorem_authority=False,
                error=str(exc)[:240],
            )

    request = _make_request(
        engine_id,
        document,
        query,
        request_id=f"request:{case_id}:{engine_id}",
    )
    outcome = backend.run(request)
    receipt = outcome.receipt
    if receipt.authority is not AuthorizationEvidenceAuthority.AUTHORIZATION:
        raise AuthorizationSemanticCertificationError(
            f"{engine_id} emitted non-authorization authority"
        )
    if receipt.is_theorem_authority:
        raise AuthorizationSemanticCertificationError(
            f"{engine_id} claimed theorem authority"
        )
    if outcome.result.authority is not ResultAuthority.AUTHORIZATION:
        raise AuthorizationSemanticCertificationError(
            f"{engine_id} result authority is not authorization"
        )
    if (
        receipt.generated_code_correctness
        is not GeneratedCodeCorrectness.NOT_ESTABLISHED
    ):
        raise AuthorizationSemanticCertificationError(
            f"{engine_id} established generated-code correctness"
        )

    explanation_digest = ""
    if receipt.explanation is not None:
        explanation_digest = _stable_json_digest(receipt.explanation.to_dict())

    return EngineRunRecord(
        engine_id=engine_id,
        case_id=case_id,
        outcome=receipt.outcome.value,
        status=outcome.result.status.value,
        policy_digest=receipt.source_binding.document_digest,
        request_digest=receipt.request_digest,
        receipt_id=receipt.receipt_id,
        authority=receipt.authority.value,
        is_theorem_authority=False,
        bound_rule_ids=_bound_rule_ids(receipt),
        explanation_digest=explanation_digest,
    )


def _reference_outcome(
    document: AuthorizationIR, query: DecisionQuery
) -> DecisionOutcome:
    decision, _, _ = ReferenceAuthorizationEvaluator().evaluate(document, query)
    return decision.outcome


# ---------------------------------------------------------------------------
# Certification
# ---------------------------------------------------------------------------


def _case_specs_from_manifest(manifest: Mapping[str, Any]) -> tuple[CaseSpec, ...]:
    raw = manifest.get("case_recipes")
    if not isinstance(raw, list) or not raw:
        return default_case_specs()
    specs: list[CaseSpec] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise AuthorizationSemanticCertificationError(
                "case_recipes entries must be objects"
            )
        specs.append(
            CaseSpec(
                case_id=str(item["case_id"]),
                category=str(item["category"]),
                expected_outcome=str(item["expected_outcome"]),
                recipe=str(item["recipe"]),
                base_fixture_id=str(item.get("base_fixture_id") or ""),
                mutation_kind=str(item.get("mutation_kind") or ""),
                notes=str(item.get("notes") or ""),
            )
        )
    return tuple(specs)


def certify_engine(
    engine_id: str,
    *,
    specs: Sequence[CaseSpec] | None = None,
) -> EngineCertification:
    """Run the full semantic matrix for one in-process authorization engine."""

    backend, interface_version = _backend_for(engine_id)
    usable = backend.is_available()
    selected = tuple(specs or default_case_specs())
    checks: list[CheckResult] = []
    records: list[EngineRunRecord] = []
    block_reasons: list[str] = []

    # Role ceiling binding (fail closed if roles module demotes the tool).
    try:
        role = get_tool_role(engine_id)
        if role.role is not ToolRole.AUTHORITY:
            block_reasons.append("tool_role_is_not_authority")
        if role.authority_ceiling is not ToolchainAuthorityCeiling.AUTHORIZATION:
            block_reasons.append("authority_ceiling_is_not_authorization")
    except Exception as exc:  # pragma: no cover - roles always present post FVT-037
        block_reasons.append(f"role_lookup_failed:{type(exc).__name__}")

    # ---- category positives (allow/deny/unknown/conflict/delegation/revocation/negative)
    category_seen: set[str] = set()
    baseline_by_case: dict[str, EngineRunRecord] = {}

    for spec in selected:
        if spec.category == "mutation":
            continue
        document, query, expected = materialize_case(spec)
        expect_error = expected == "error" or spec.category == "malformed"
        record = run_engine_case(
            engine_id,
            spec.case_id,
            document,
            query,
            expect_error=expect_error,
        )
        records.append(record)
        baseline_by_case[spec.case_id] = record
        category_seen.add(spec.category)

        if expect_error:
            ok = record.outcome == "error" and record.status in {
                "error",
                "quarantined",
            }
            # Must never silently allow malformed input.
            if record.outcome == DecisionOutcome.ALLOW.value:
                ok = False
            checks.append(
                CheckResult(
                    check_id=f"{engine_id}.{spec.case_id}.malformed",
                    kind="malformed",
                    status="passed" if ok else "failed",
                    expected="error",
                    observed=record.outcome,
                    detail=record.error or "malformed input handling",
                    policy_digest=record.policy_digest,
                    engine_id=engine_id,
                )
            )
            if not ok:
                block_reasons.append(f"malformed_not_fail_closed:{spec.case_id}")
            continue

        kind = "negative" if spec.category == "negative" else "positive"
        ok = record.outcome == expected and not record.is_theorem_authority
        if document is not None and query is not None:
            # Cross-check reference semantics for non-error cases.
            ref = _reference_outcome(document, query)
            if ref.value != expected:
                ok = False
                block_reasons.append(
                    f"reference_expected_mismatch:{spec.case_id}:{ref.value}"
                )
        checks.append(
            CheckResult(
                check_id=f"{engine_id}.{spec.case_id}.{kind}",
                kind=kind,
                status="passed" if ok else "failed",
                expected=expected,
                observed=record.outcome,
                detail=spec.notes or spec.recipe,
                policy_digest=record.policy_digest,
                engine_id=engine_id,
            )
        )
        if not ok:
            block_reasons.append(f"case_failed:{spec.case_id}")

        # Authority ceiling check per case.
        authority_ok = (
            record.authority == AUTHORITY_CEILING
            and record.is_theorem_authority is False
        )
        checks.append(
            CheckResult(
                check_id=f"{engine_id}.{spec.case_id}.authority",
                kind="authority",
                status="passed" if authority_ok else "failed",
                expected=AUTHORITY_CEILING,
                observed=record.authority,
                detail="authorization-decision authority only",
                policy_digest=record.policy_digest,
                engine_id=engine_id,
            )
        )
        if not authority_ok:
            block_reasons.append(f"authority_breach:{spec.case_id}")

        # Deterministic replay for counterexample-bearing categories.
        if spec.category in {"deny", "conflict", "unknown", "revocation", "negative"}:
            replay = run_engine_case(engine_id, f"{spec.case_id}:replay", document, query)
            records.append(replay)
            replay_ok = (
                replay.outcome == record.outcome
                and replay.policy_digest == record.policy_digest
                and replay.explanation_digest == record.explanation_digest
                and replay.authority == record.authority
            )
            checks.append(
                CheckResult(
                    check_id=f"{engine_id}.{spec.case_id}.replay",
                    kind="replay",
                    status="passed" if replay_ok else "failed",
                    expected=record.outcome,
                    observed=replay.outcome,
                    detail="counterexample/decision replay must be deterministic",
                    policy_digest=replay.policy_digest,
                    engine_id=engine_id,
                )
            )
            if not replay_ok:
                block_reasons.append(f"replay_unstable:{spec.case_id}")

    missing_categories = sorted(REQUIRED_CATEGORIES - category_seen)
    if missing_categories:
        block_reasons.append(f"missing_categories:{','.join(missing_categories)}")
        checks.append(
            CheckResult(
                check_id=f"{engine_id}.corpus.categories",
                kind="positive",
                status="failed",
                expected=",".join(sorted(REQUIRED_CATEGORIES)),
                observed=",".join(sorted(category_seen)),
                detail="required semantic categories incomplete",
                engine_id=engine_id,
            )
        )

    # ---- mutations
    mutation_seen: set[str] = set()
    for spec in selected:
        if spec.category != "mutation":
            continue
        base = _fixture_by_id(spec.base_fixture_id)
        base_record = run_engine_case(
            engine_id,
            f"{spec.case_id}:baseline",
            base.document,
            base.query,
        )
        records.append(base_record)
        document, query, expected = materialize_case(spec)
        mutated = run_engine_case(engine_id, spec.case_id, document, query)
        records.append(mutated)
        mutation_seen.add(spec.mutation_kind or spec.recipe)

        changed = mutated.outcome != base_record.outcome
        quarantined = mutated.outcome in {
            DecisionOutcome.UNKNOWN.value,
            DecisionOutcome.CONFLICT.value,
            "error",
        } and base_record.outcome == DecisionOutcome.ALLOW.value
        matches_expected = mutated.outcome == expected
        ok = (changed or quarantined) and matches_expected and not mutated.is_theorem_authority
        # Policy digest must change when the document mutates.
        policy_changed = mutated.policy_digest != base_record.policy_digest
        ok = ok and policy_changed

        checks.append(
            CheckResult(
                check_id=f"{engine_id}.{spec.case_id}.mutation",
                kind="mutation",
                status="passed" if ok else "failed",
                expected=f"{expected} (changed from {base_record.outcome})",
                observed=mutated.outcome,
                detail=(
                    f"mutation_kind={spec.mutation_kind or spec.recipe}; "
                    f"policy_digest_changed={policy_changed}"
                ),
                policy_digest=mutated.policy_digest,
                engine_id=engine_id,
            )
        )
        if not ok:
            block_reasons.append(f"mutation_failed:{spec.case_id}")

    # Normalize mutation kind names for coverage.
    normalized_mutations = set()
    for item in mutation_seen:
        if item in REQUIRED_MUTATION_KINDS:
            normalized_mutations.add(item)
        elif "rule" in item:
            normalized_mutations.add("rule")
        elif "principal" in item:
            normalized_mutations.add("principal")
        elif "scope" in item:
            normalized_mutations.add("scope")
        elif "delegation" in item:
            normalized_mutations.add("delegation")
    missing_mutations = sorted(REQUIRED_MUTATION_KINDS - normalized_mutations)
    if missing_mutations:
        block_reasons.append(f"missing_mutations:{','.join(missing_mutations)}")

    # Receipt binding: every successful evaluation binds policy digest + engine.
    for record in records:
        if record.outcome == "error":
            continue
        bound = bool(record.policy_digest) and record.engine_id == engine_id
        if not bound:
            block_reasons.append(f"unbound_receipt:{record.case_id}")

    all_passed = all(item.passed for item in checks) and not block_reasons and usable
    certified = all_passed

    return EngineCertification(
        engine_id=engine_id,
        interface_version=interface_version,
        usable=usable,
        certified=certified,
        authority_ceiling=AUTHORITY_CEILING,
        checks=checks,
        case_results=records,
        block_reasons=sorted(set(block_reasons)),
    )


def certify_authorization_semantics(
    *,
    engines: Sequence[str] | None = None,
    manifest: Mapping[str, Any] | None = None,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Run semantic certification for all reference authorization engines."""

    loaded = dict(manifest) if manifest is not None else load_manifest(manifest_path)
    specs = _case_specs_from_manifest(loaded)
    selected_engines = tuple(engines or loaded.get("engines") or REFERENCE_ENGINES)

    engine_results: list[EngineCertification] = []
    for engine_id in selected_engines:
        engine_results.append(certify_engine(engine_id, specs=specs))

    all_certified = bool(engine_results) and all(item.certified for item in engine_results)
    any_theorem = any(
        check.is_theorem_authority
        for engine in engine_results
        for check in engine.checks
    )
    if any_theorem:
        all_certified = False

    categories = sorted(
        {
            str(item.get("category"))
            for item in loaded.get("case_recipes", [])
            if isinstance(item, Mapping)
        }
        or {spec.category for spec in specs}
    )

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "interface": INTERFACE,
        "goal_id": GOAL_ID,
        "task_id": TASK_ID,
        "program": PROGRAM,
        "lane_id": LANE_ID,
        "handler_id": HANDLER_ID,
        "certification_surface": CERTIFICATION_SURFACE,
        "authority_ceiling": AUTHORITY_CEILING,
        "forbids_theorem_authority": True,
        "certified": all_certified,
        "engines": [item.to_dict() for item in engine_results],
        "engine_ids": [item.engine_id for item in engine_results],
        "categories_exercised": categories,
        "mutation_kinds": sorted(REQUIRED_MUTATION_KINDS),
        "manifest": {
            "schema_version": loaded.get("schema_version", MANIFEST_SCHEMA),
            "interface": loaded.get("interface", INTERFACE),
            "case_count": len(specs),
            "path": str(
                manifest_path or (_REPO_ROOT / DEFAULT_MANIFEST_RELATIVE)
            ),
        },
        "policy": {
            "in_process_only": True,
            "no_external_shadow_install": True,
            "no_central_certificate_edit": True,
            "receipts_bind_policy_and_engine": True,
            "authorization_decision_authority_only": True,
            "counterexamples_replay_deterministically": True,
            "mutations_must_change_or_quarantine": True,
            "grants_authorization_decision_authority": True,
            "grants_theorem_authority": False,
        },
        "summary": {
            "engines_certified": sum(1 for item in engine_results if item.certified),
            "engines_total": len(engine_results),
            "checks_passed": sum(
                1 for engine in engine_results for check in engine.checks if check.passed
            ),
            "checks_total": sum(len(engine.checks) for engine in engine_results),
            "block_reasons": sorted(
                {
                    reason
                    for engine in engine_results
                    for reason in engine.block_reasons
                }
            ),
        },
    }
    payload["certificate_digest_sha256"] = _stable_json_digest(
        {
            key: value
            for key, value in payload.items()
            if key != "certificate_digest_sha256"
        }
    )
    return payload


def authorization_lane_handler(
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """Lane handler for ``datalog_secpal`` / role-aware promotion binding."""

    result = certify_authorization_semantics(
        engines=kwargs.get("engines"),
        manifest_path=kwargs.get("manifest_path"),
    )
    return {
        "lane_id": LANE_ID,
        "owner_module": CERTIFICATION_SURFACE,
        "handler_id": HANDLER_ID,
        "status": "certified" if result["certified"] else "failed",
        "certified": bool(result["certified"]),
        "authority_ceiling": AUTHORITY_CEILING,
        "reason_codes": list(result["summary"].get("block_reasons") or []),
        "certificate_digest_sha256": result["certificate_digest_sha256"],
        "engine_ids": list(result.get("engine_ids") or []),
        "args_received": bool(args) or bool(kwargs),
        "interface": INTERFACE,
        "goal_id": GOAL_ID,
        "task_id": TASK_ID,
        "grants_theorem_authority": False,
    }


def bind_authorization_lane(
    policy: Any | None = None,
    *,
    replace: bool = True,
) -> Any:
    """Bind this certifier into a role-aware promotion policy when available."""

    if _bind_lane_handler is None or _build_role_aware_policy is None:
        return policy
    target = policy if policy is not None else _build_role_aware_policy()
    return _bind_lane_handler(
        LANE_ID,
        authorization_lane_handler,
        policy=target,
        replace=replace,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Semantically certify in-process Datalog and SecPAL authorization "
            f"({INTERFACE} / {GOAL_ID})."
        )
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full certification receipt as JSON",
    )
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help=f"Write the compact corpus manifest to {DEFAULT_MANIFEST_RELATIVE}",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional path to an authorization corpus manifest",
    )
    parser.add_argument(
        "--engine",
        action="append",
        dest="engines",
        default=None,
        help="Limit certification to one engine id (repeatable)",
    )
    args = parser.parse_args(argv)

    if args.write_manifest:
        path = write_manifest(args.manifest)
        if not args.json:
            print(f"wrote {path}")
            return 0

    receipt = certify_authorization_semantics(
        engines=args.engines,
        manifest_path=args.manifest,
    )
    if args.json or args.write_manifest:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    else:
        status = "CERTIFIED" if receipt["certified"] else "FAILED"
        print(f"{INTERFACE} {status}")
        print(
            f"goal={GOAL_ID} task={TASK_ID} lane={LANE_ID} "
            f"engines={','.join(receipt['engine_ids'])}"
        )
        summary = receipt["summary"]
        print(
            f"checks={summary['checks_passed']}/{summary['checks_total']} "
            f"engines_certified={summary['engines_certified']}/{summary['engines_total']}"
        )
        if summary["block_reasons"]:
            print("block_reasons:")
            for reason in summary["block_reasons"]:
                print(f"  - {reason}")
        print(f"digest={receipt['certificate_digest_sha256']}")
    return 0 if receipt["certified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "INTERFACE",
    "SCHEMA_VERSION",
    "MANIFEST_SCHEMA",
    "GOAL_ID",
    "TASK_ID",
    "PROGRAM",
    "LANE_ID",
    "HANDLER_ID",
    "CERTIFICATION_SURFACE",
    "AUTHORITY_CEILING",
    "ENGINE_DATALOG",
    "ENGINE_SECPAL",
    "REFERENCE_ENGINES",
    "REQUIRED_CATEGORIES",
    "REQUIRED_MUTATION_KINDS",
    "AuthorizationSemanticCertificationError",
    "CaseSpec",
    "CheckResult",
    "EngineCertification",
    "EngineRunRecord",
    "apply_delegation_mutation",
    "apply_principal_mutation",
    "apply_rule_mutation",
    "apply_scope_mutation",
    "authorization_lane_handler",
    "bind_authorization_lane",
    "build_default_manifest",
    "build_negative_case",
    "build_revocation_case",
    "certify_authorization_semantics",
    "certify_engine",
    "default_case_specs",
    "load_manifest",
    "main",
    "materialize_case",
    "run_engine_case",
    "write_manifest",
]
