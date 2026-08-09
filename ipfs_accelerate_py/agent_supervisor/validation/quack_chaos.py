"""Quack security, concurrency, conflict, and restart chaos harness (DQP-034).

Interfaces: ``QuackChaosReport@1``, ``QuackSecurityPolicy@1``

Proves the control-plane security and resilience boundary without granting
providers or LLM processes a Quack token or arbitrary SQL surface:

* unauthorized / cross-root / file / extension statements fail *before effect*
* same-row conflicts are bounded by the typed retry policy
* stale clients cannot write after restart or store-generation rotation
* live scenarios cannot silently skip when the capability profile claims
  ``compatible``

Hermetic embedded DuckDB paths cover the full chaos population. When the
pinned profile reports ``compatible``, live Quack transport scenarios are
mandatory; otherwise hermetic mode is recorded explicitly (never a silent
skip of a required gate).
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import random
import re
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..runtime.control_plane_backup import (
    StoreGenerationRotation,
    build_control_plane_backup,
)
from ..runtime.quack_state_server import (
    DEFAULT_LOOPBACK_HOST,
    FakeQuackTransport,
    QuackStateServerBindError,
    QuackStateServerConfig,
    TokenVault,
    assert_bind_admitted,
    build_server,
    provider_safe_environment,
    sanitize_for_export,
)
from ..task_sources.control_plane_contracts import (
    CommandKind,
    CommandOutcome,
    StateAuthorityClass,
    StateCommand,
    StoreGeneration,
)
from ..task_sources.control_plane_migrations import duckdb_available
from ..task_sources.control_plane_schema import install_control_plane_schema
from ..task_sources.control_plane_transactions import (
    RetryPolicy,
    StaleGenerationError,
    TransactionConflictKind,
    default_retry_policy,
)
from ..task_sources.duckdb_state import open_duckdb_connection
from ..task_sources.quack_capabilities import (
    DEFAULT_QUACK_BETA_LIMITATIONS,
    QuackCapabilityReport,
    QuackCapabilityStatus,
    default_compatibility_profile,
    probe_quack_capabilities,
)
from ..task_sources.quack_state_client import (
    QuackClientSQLError,
    QuackStateClient,
    StatementTemplate,
    TransportMode,
    open_embedded_client,
)

# ---------------------------------------------------------------------------
# Interface / schema identities
# ---------------------------------------------------------------------------

QUACK_CHAOS_REPORT_INTERFACE: Final = "QuackChaosReport@1"
QUACK_SECURITY_POLICY_INTERFACE: Final = "QuackSecurityPolicy@1"
QUACK_CHAOS_REPORT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/quack-chaos-report@1"
)
QUACK_SECURITY_POLICY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/quack-security-policy@1"
)
CHAOS_SCENARIO_RESULT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/chaos-scenario-result@1"
)
AUTHORIZATION_DECISION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/statement-authorization-decision@1"
)
QUACK_CHAOS_VERSION: Final[int] = 1
TASK_ID: Final = "DQP-034"
GOAL_ID: Final = "DQP-G070"
EVIDENCE: Final = "dqp/quack-chaos@1"

DEFAULT_CLIENT_COUNT: Final = 4
DEFAULT_MAX_CONFLICT_ATTEMPTS: Final = 8
DEFAULT_LATENCY_BOUND_MS: Final = 30_000
DEFAULT_SECRET_SCAN_MAX_BYTES: Final = 1_048_576

# Closed statement authorization vocabulary (parsed verbs, not prefix regex).
_STATEMENT_VERB_RE: Final = re.compile(
    r"^\s*(?P<verb>WITH|SELECT|INSERT|UPDATE|DELETE|ATTACH|DETACH|COPY|"
    r"INSTALL|LOAD|PRAGMA|CALL|EXPORT|IMPORT|DROP|ALTER|CREATE|TRUNCATE|"
    r"VACUUM|FORCE|CHECKPOINT|BEGIN|COMMIT|ROLLBACK|SET|USE|EXPLAIN|"
    r"DESCRIBE|SHOW|VALUES)\b",
    re.IGNORECASE | re.DOTALL,
)
_MULTI_STATEMENT_RE: Final = re.compile(r";\s*\S", re.DOTALL)
_COMMENT_RE: Final = re.compile(r"--|/\*")
_PATH_LITERAL_RE: Final = re.compile(
    r"(?:read_csv_auto|read_csv|read_parquet|read_json_auto|read_json|"
    r"read_text|read_blob|copy|export|import|attach)\s*\(\s*['\"]|"
    r"['\"][^'\"]*(?:/|\.\.)[^'\"]*['\"]",
    re.IGNORECASE,
)
_CROSS_ROOT_RE: Final = re.compile(
    r"(?:\.\./|/etc/|/proc/|/sys/|file://|~[/\\]|[A-Za-z]:\\)",
    re.IGNORECASE,
)
_EXTENSION_SURFACE_RE: Final = re.compile(
    r"\b(?:INSTALL|LOAD)\b|\.duckdb_extension\b|community\.duckdb\.org",
    re.IGNORECASE,
)
_PYTHON_UDF_RE: Final = re.compile(
    r"\b(?:CREATE\s+(?:OR\s+REPLACE\s+)?(?:MACRO|FUNCTION)|"
    r"CREATE\s+TYPE|python_eval|pyarrow)\b",
    re.IGNORECASE,
)

_TOKEN_LIKE_RE: Final = re.compile(
    r"(?:token|secret|password|api[_-]?key|bearer|authorization|"
    r"quack[_-]?auth|credential)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class QuackChaosError(RuntimeError):
    """Base fail-closed error for the chaos harness."""


class QuackChaosLiveGateError(QuackChaosError):
    """Live scenario was skipped while the profile claims compatible."""


class QuackChaosScenarioError(QuackChaosError):
    """A required chaos scenario failed its acceptance predicate."""


class QuackChaosAuthorizationError(QuackChaosError):
    """Statement authorization policy denied a query before effect."""


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class ChaosScenarioId(str, Enum):
    """Closed set of chaos / security scenarios exercised by DQP-034."""

    TOKEN_ISOLATION = "token_isolation"
    PROVIDER_ENV_ISOLATION = "provider_env_isolation"
    RAW_SQL_REJECTION = "raw_sql_rejection"
    FORBIDDEN_SURFACE = "forbidden_surface"
    LOOPBACK_BIND = "loopback_bind"
    STATEMENT_AUTHORIZATION = "statement_authorization"
    FOUR_CLIENT_CONCURRENCY = "four_client_concurrency"
    HOT_ROW_CONFLICT = "hot_row_conflict"
    LOST_REPLY_IDEMPOTENCY = "lost_reply_idempotency"
    RETRY_JITTER = "retry_jitter"
    STALE_AFTER_ROTATION = "stale_after_rotation"
    SERVER_RESTART_STALE = "server_restart_stale"
    CREDENTIAL_ROTATION = "credential_rotation"
    SECRET_SCAN = "secret_scan"
    LATENCY_BOUND = "latency_bound"
    DENIAL_LOGGING = "denial_logging"
    LIVE_GATE_POLICY = "live_gate_policy"
    TLS_BOUNDARY = "tls_boundary"
    PYTHON_UDF_LIMITATION = "python_udf_limitation"
    SPLIT_BRAIN_OWNERSHIP = "split_brain_ownership"


class ScenarioMode(str, Enum):
    """How a scenario was executed."""

    HERMETIC = "hermetic"
    LIVE = "live"
    SKIPPED = "skipped"


class ScenarioOutcome(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class AuthorizationAction(str, Enum):
    ALLOW = "allow"
    DENY = "deny"


class AuthorizationReason(str, Enum):
    ADMITTED_TEMPLATE = "admitted_template"
    ADMITTED_META = "admitted_meta"
    EMPTY_STATEMENT = "empty_statement"
    MULTI_STATEMENT = "multi_statement"
    COMMENT_SMUGGLING = "comment_smuggling"
    UNKNOWN_VERB = "unknown_verb"
    FORBIDDEN_VERB = "forbidden_verb"
    FILE_PATH = "file_path"
    CROSS_ROOT = "cross_root"
    EXTENSION_SURFACE = "extension_surface"
    PYTHON_UDF = "python_udf"
    ARBITRARY_SQL = "arbitrary_sql"
    RAW_IDENTIFIER = "raw_identifier"


class ChaosVerdict(str, Enum):
    """Suite conclusion — not a promotion decision."""

    PASSED = "passed"
    FAILED = "failed"
    INCOMPLETE = "incomplete"


REQUIRED_SCENARIOS: Final[tuple[ChaosScenarioId, ...]] = tuple(ChaosScenarioId)

# Statement verbs admitted only via closed client templates (not free SQL).
ADMITTED_MUTATION_VERBS: Final[frozenset[str]] = frozenset(
    {"INSERT", "UPDATE", "DELETE"}
)
ADMITTED_QUERY_VERBS: Final[frozenset[str]] = frozenset(
    {
        "SELECT",
        "WITH",
        "VALUES",
        "EXPLAIN",
        "DESCRIBE",
        "SHOW",
        "BEGIN",
        "COMMIT",
        "ROLLBACK",
        "CHECKPOINT",
        "SET",
        "USE",
    }
)
FORBIDDEN_VERBS: Final[frozenset[str]] = frozenset(
    {
        "ATTACH",
        "DETACH",
        "COPY",
        "INSTALL",
        "LOAD",
        "PRAGMA",
        "CALL",
        "EXPORT",
        "IMPORT",
        "DROP",
        "ALTER",
        "CREATE",
        "TRUNCATE",
        "VACUUM",
        "FORCE",
    }
)


# ---------------------------------------------------------------------------
# Security policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuackSecurityPolicy:
    """Least-privilege statement and transport policy for Quack clients.

    Interface: ``QuackSecurityPolicy@1``.

    Authorization is a *parsed* verb / surface decision. A prefix regex is
    explicitly not the security boundary (see threat model).
    """

    INTERFACE: ClassVar[str] = QUACK_SECURITY_POLICY_INTERFACE
    SCHEMA: ClassVar[str] = QUACK_SECURITY_POLICY_SCHEMA

    policy_id: str = "agent-supervisor-quack-security-1"
    policy_version: int = QUACK_CHAOS_VERSION
    require_loopback: bool = True
    allow_remote_without_reviewed_policy: bool = False
    tls_required_for_remote: bool = True
    provider_receives_token: bool = False
    provider_receives_arbitrary_sql: bool = False
    default_authz_permits_all_authenticated: bool = True
    # Quack beta: default server authz is open once authenticated; the typed
    # client is the primary SQL boundary.
    client_template_only: bool = True
    deny_file_paths: bool = True
    deny_cross_root: bool = True
    deny_extension_install: bool = True
    deny_python_udfs: bool = True
    max_retry_attempts: int = DEFAULT_MAX_CONFLICT_ATTEMPTS
    max_latency_ms: int = DEFAULT_LATENCY_BOUND_MS
    client_count: int = DEFAULT_CLIENT_COUNT
    beta_limitations: tuple[str, ...] = DEFAULT_QUACK_BETA_LIMITATIONS
    admitted_query_verbs: frozenset[str] = field(
        default_factory=lambda: ADMITTED_QUERY_VERBS
    )
    admitted_mutation_verbs: frozenset[str] = field(
        default_factory=lambda: ADMITTED_MUTATION_VERBS
    )
    forbidden_verbs: frozenset[str] = field(default_factory=lambda: FORBIDDEN_VERBS)
    tls_boundary_statement: str = (
        "Loopback Quack binds do not terminate TLS. Any future remote "
        "deployment requires explicit review, TLS termination, OS isolation, "
        "credential rotation, and a parsed statement authorization policy "
        "before bind admission."
    )
    python_udf_limitation_statement: str = (
        "Python UDFs, macros that escape to host code, and arbitrary "
        "CREATE FUNCTION surfaces are denied at the client authorization "
        "boundary. Model-supplied code must never register server-side "
        "callables against control.duckdb."
    )

    def __post_init__(self) -> None:
        if not str(self.policy_id or "").strip():
            raise ValueError("policy_id must not be empty")
        if int(self.policy_version) != QUACK_CHAOS_VERSION:
            raise ValueError("unsupported security policy version")
        if int(self.max_retry_attempts) < 1:
            raise ValueError("max_retry_attempts must be >= 1")
        if int(self.client_count) < 2:
            raise ValueError("client_count must be >= 2")
        if self.provider_receives_token or self.provider_receives_arbitrary_sql:
            raise ValueError(
                "security policy must never admit provider token or SQL access"
            )
        object.__setattr__(
            self,
            "beta_limitations",
            tuple(str(item) for item in self.beta_limitations),
        )
        object.__setattr__(
            self,
            "admitted_query_verbs",
            frozenset(str(v).upper() for v in self.admitted_query_verbs),
        )
        object.__setattr__(
            self,
            "admitted_mutation_verbs",
            frozenset(str(v).upper() for v in self.admitted_mutation_verbs),
        )
        object.__setattr__(
            self,
            "forbidden_verbs",
            frozenset(str(v).upper() for v in self.forbidden_verbs),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "require_loopback": self.require_loopback,
            "allow_remote_without_reviewed_policy": (
                self.allow_remote_without_reviewed_policy
            ),
            "tls_required_for_remote": self.tls_required_for_remote,
            "provider_receives_token": self.provider_receives_token,
            "provider_receives_arbitrary_sql": self.provider_receives_arbitrary_sql,
            "default_authz_permits_all_authenticated": (
                self.default_authz_permits_all_authenticated
            ),
            "client_template_only": self.client_template_only,
            "deny_file_paths": self.deny_file_paths,
            "deny_cross_root": self.deny_cross_root,
            "deny_extension_install": self.deny_extension_install,
            "deny_python_udfs": self.deny_python_udfs,
            "max_retry_attempts": int(self.max_retry_attempts),
            "max_latency_ms": int(self.max_latency_ms),
            "client_count": int(self.client_count),
            "beta_limitations": list(self.beta_limitations),
            "admitted_query_verbs": sorted(self.admitted_query_verbs),
            "admitted_mutation_verbs": sorted(self.admitted_mutation_verbs),
            "forbidden_verbs": sorted(self.forbidden_verbs),
            "tls_boundary_statement": self.tls_boundary_statement,
            "python_udf_limitation_statement": self.python_udf_limitation_statement,
        }


def default_security_policy() -> QuackSecurityPolicy:
    """Return the program's sealed least-privilege Quack security policy."""

    return QuackSecurityPolicy()


@dataclass(frozen=True)
class AuthorizationDecision:
    """Result of evaluating one candidate statement against the policy."""

    SCHEMA: ClassVar[str] = AUTHORIZATION_DECISION_SCHEMA

    action: AuthorizationAction
    reason: AuthorizationReason
    verb: str = ""
    detail: str = ""
    effect_attempted: bool = False

    def __post_init__(self) -> None:
        action = self.action
        if not isinstance(action, AuthorizationAction):
            action = AuthorizationAction(str(action))
        reason = self.reason
        if not isinstance(reason, AuthorizationReason):
            reason = AuthorizationReason(str(reason))
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "verb", str(self.verb or "").upper())
        object.__setattr__(self, "detail", str(self.detail or ""))
        object.__setattr__(self, "effect_attempted", bool(self.effect_attempted))

    @property
    def allowed(self) -> bool:
        return self.action is AuthorizationAction.ALLOW

    @property
    def denied(self) -> bool:
        return self.action is AuthorizationAction.DENY

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "action": self.action.value,
            "reason": self.reason.value,
            "verb": self.verb,
            "detail": self.detail,
            "effect_attempted": self.effect_attempted,
        }


def authorize_statement(
    sql: str,
    *,
    policy: QuackSecurityPolicy | None = None,
    via_template: bool = False,
) -> AuthorizationDecision:
    """Authorize a candidate SQL string *before* any engine effect.

    Fail-closed: empty, multi-statement, comment-smuggled, file, cross-root,
    extension, Python-UDF, and forbidden-verb statements are denied with
    ``effect_attempted=False``.
    """

    sealed = policy or default_security_policy()
    text = str(sql or "")
    stripped = text.strip()
    if not stripped:
        return AuthorizationDecision(
            action=AuthorizationAction.DENY,
            reason=AuthorizationReason.EMPTY_STATEMENT,
            detail="empty statement",
            effect_attempted=False,
        )
    if _MULTI_STATEMENT_RE.search(stripped):
        return AuthorizationDecision(
            action=AuthorizationAction.DENY,
            reason=AuthorizationReason.MULTI_STATEMENT,
            detail="multi-statement SQL is forbidden",
            effect_attempted=False,
        )
    if _COMMENT_RE.search(stripped):
        return AuthorizationDecision(
            action=AuthorizationAction.DENY,
            reason=AuthorizationReason.COMMENT_SMUGGLING,
            detail="SQL comments are forbidden in authorized statements",
            effect_attempted=False,
        )
    if sealed.deny_python_udfs and _PYTHON_UDF_RE.search(stripped):
        return AuthorizationDecision(
            action=AuthorizationAction.DENY,
            reason=AuthorizationReason.PYTHON_UDF,
            detail="Python UDF / host-escape surface denied",
            effect_attempted=False,
        )
    if sealed.deny_extension_install and _EXTENSION_SURFACE_RE.search(stripped):
        return AuthorizationDecision(
            action=AuthorizationAction.DENY,
            reason=AuthorizationReason.EXTENSION_SURFACE,
            detail="extension INSTALL/LOAD surface denied",
            effect_attempted=False,
        )
    if sealed.deny_cross_root and _CROSS_ROOT_RE.search(stripped):
        return AuthorizationDecision(
            action=AuthorizationAction.DENY,
            reason=AuthorizationReason.CROSS_ROOT,
            detail="cross-root or absolute filesystem path denied",
            effect_attempted=False,
        )
    if sealed.deny_file_paths and _PATH_LITERAL_RE.search(stripped):
        return AuthorizationDecision(
            action=AuthorizationAction.DENY,
            reason=AuthorizationReason.FILE_PATH,
            detail="file-path or external-reader surface denied",
            effect_attempted=False,
        )

    match = _STATEMENT_VERB_RE.match(stripped)
    if match is None:
        return AuthorizationDecision(
            action=AuthorizationAction.DENY,
            reason=AuthorizationReason.UNKNOWN_VERB,
            detail="unrecognized statement verb",
            effect_attempted=False,
        )
    verb = str(match.group("verb") or "").upper()
    if verb in sealed.forbidden_verbs:
        return AuthorizationDecision(
            action=AuthorizationAction.DENY,
            reason=AuthorizationReason.FORBIDDEN_VERB,
            verb=verb,
            detail=f"verb {verb} is forbidden at the client boundary",
            effect_attempted=False,
        )
    if sealed.client_template_only and not via_template:
        return AuthorizationDecision(
            action=AuthorizationAction.DENY,
            reason=AuthorizationReason.ARBITRARY_SQL,
            verb=verb,
            detail="arbitrary SQL requires a closed named template",
            effect_attempted=False,
        )
    if verb in sealed.admitted_query_verbs or verb in sealed.admitted_mutation_verbs:
        return AuthorizationDecision(
            action=AuthorizationAction.ALLOW,
            reason=(
                AuthorizationReason.ADMITTED_TEMPLATE
                if via_template
                else AuthorizationReason.ADMITTED_META
            ),
            verb=verb,
            detail="admitted by sealed policy",
            effect_attempted=False,
        )
    return AuthorizationDecision(
        action=AuthorizationAction.DENY,
        reason=AuthorizationReason.UNKNOWN_VERB,
        verb=verb,
        detail=f"verb {verb} is not admitted",
        effect_attempted=False,
    )


def assert_authorized(
    sql: str,
    *,
    policy: QuackSecurityPolicy | None = None,
    via_template: bool = False,
) -> AuthorizationDecision:
    """Raise :class:`QuackChaosAuthorizationError` when the statement is denied."""

    decision = authorize_statement(sql, policy=policy, via_template=via_template)
    if decision.denied:
        raise QuackChaosAuthorizationError(
            f"statement denied ({decision.reason.value}): {decision.detail}"
        )
    return decision


# ---------------------------------------------------------------------------
# Scenario results / report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChaosScenarioResult:
    """One scenario observation inside a chaos suite run."""

    SCHEMA: ClassVar[str] = CHAOS_SCENARIO_RESULT_SCHEMA

    scenario_id: ChaosScenarioId
    outcome: ScenarioOutcome
    mode: ScenarioMode
    detail: str = ""
    metrics: Mapping[str, Any] = field(default_factory=dict)
    denials: tuple[Mapping[str, Any], ...] = ()
    duration_ms: float = 0.0

    def __post_init__(self) -> None:
        scenario = self.scenario_id
        if not isinstance(scenario, ChaosScenarioId):
            scenario = ChaosScenarioId(str(scenario))
        outcome = self.outcome
        if not isinstance(outcome, ScenarioOutcome):
            outcome = ScenarioOutcome(str(outcome))
        mode = self.mode
        if not isinstance(mode, ScenarioMode):
            mode = ScenarioMode(str(mode))
        object.__setattr__(self, "scenario_id", scenario)
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "detail", str(self.detail or ""))
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics or {})))
        object.__setattr__(
            self,
            "denials",
            tuple(dict(item) for item in self.denials),
        )
        object.__setattr__(self, "duration_ms", float(self.duration_ms or 0.0))

    @property
    def passed(self) -> bool:
        return self.outcome is ScenarioOutcome.PASSED

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "scenario_id": self.scenario_id.value,
            "outcome": self.outcome.value,
            "mode": self.mode.value,
            "detail": self.detail,
            "metrics": dict(self.metrics),
            "denials": [dict(item) for item in self.denials],
            "duration_ms": self.duration_ms,
        }


@dataclass(frozen=True)
class QuackChaosReport:
    """Immutable suite report for security / concurrency / restart chaos.

    Interface: ``QuackChaosReport@1``.
    """

    INTERFACE: ClassVar[str] = QUACK_CHAOS_REPORT_INTERFACE
    SCHEMA: ClassVar[str] = QUACK_CHAOS_REPORT_SCHEMA

    verdict: ChaosVerdict
    policy: QuackSecurityPolicy
    scenarios: tuple[ChaosScenarioResult, ...]
    profile_status: str
    profile_claims_compatible: bool
    live_gate_enforced: bool
    duckdb_available: bool
    workspace: str = ""
    task_id: str = TASK_ID
    goal_id: str = GOAL_ID
    evidence: str = EVIDENCE
    report_version: int = QUACK_CHAOS_VERSION
    beta_limitations: tuple[str, ...] = DEFAULT_QUACK_BETA_LIMITATIONS
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        verdict = self.verdict
        if not isinstance(verdict, ChaosVerdict):
            verdict = ChaosVerdict(str(verdict))
        if not isinstance(self.policy, QuackSecurityPolicy):
            raise TypeError("policy must be QuackSecurityPolicy")
        scenarios = tuple(self.scenarios)
        for item in scenarios:
            if not isinstance(item, ChaosScenarioResult):
                raise TypeError("scenarios must contain ChaosScenarioResult")
        object.__setattr__(self, "verdict", verdict)
        object.__setattr__(self, "scenarios", scenarios)
        object.__setattr__(self, "profile_status", str(self.profile_status or ""))
        object.__setattr__(
            self,
            "profile_claims_compatible",
            bool(self.profile_claims_compatible),
        )
        object.__setattr__(self, "live_gate_enforced", bool(self.live_gate_enforced))
        object.__setattr__(self, "duckdb_available", bool(self.duckdb_available))
        object.__setattr__(
            self,
            "beta_limitations",
            tuple(str(item) for item in self.beta_limitations),
        )
        object.__setattr__(self, "details", MappingProxyType(dict(self.details or {})))

    @property
    def passed(self) -> bool:
        return self.verdict is ChaosVerdict.PASSED

    @property
    def failed_scenarios(self) -> tuple[ChaosScenarioResult, ...]:
        return tuple(
            item
            for item in self.scenarios
            if item.outcome in {ScenarioOutcome.FAILED, ScenarioOutcome.ERROR}
        )

    @property
    def skipped_scenarios(self) -> tuple[ChaosScenarioResult, ...]:
        return tuple(
            item for item in self.scenarios if item.outcome is ScenarioOutcome.SKIPPED
        )

    def scenario(self, scenario_id: ChaosScenarioId | str) -> ChaosScenarioResult | None:
        key = (
            scenario_id
            if isinstance(scenario_id, ChaosScenarioId)
            else ChaosScenarioId(str(scenario_id))
        )
        for item in self.scenarios:
            if item.scenario_id is key:
                return item
        return None

    def content_id(self) -> str:
        payload = canonical_chaos_bytes(self.to_dict())
        return f"sha256:{hashlib.sha256(payload).hexdigest()}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "report_version": self.report_version,
            "task_id": self.task_id,
            "goal_id": self.goal_id,
            "evidence": self.evidence,
            "verdict": self.verdict.value,
            "policy": self.policy.to_dict(),
            "scenarios": [item.to_dict() for item in self.scenarios],
            "profile_status": self.profile_status,
            "profile_claims_compatible": self.profile_claims_compatible,
            "live_gate_enforced": self.live_gate_enforced,
            "duckdb_available": self.duckdb_available,
            "workspace": self.workspace,
            "beta_limitations": list(self.beta_limitations),
            "details": dict(self.details),
            "failed_count": len(self.failed_scenarios),
            "skipped_count": len(self.skipped_scenarios),
            "passed_count": sum(1 for item in self.scenarios if item.passed),
        }


def canonical_chaos_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")


# ---------------------------------------------------------------------------
# Live gate
# ---------------------------------------------------------------------------


def profile_claims_compatible(
    report: QuackCapabilityReport | Mapping[str, Any] | str | None,
) -> bool:
    """True when the capability profile asserts full pin compatibility."""

    if report is None:
        return False
    if isinstance(report, QuackCapabilityReport):
        return report.status is QuackCapabilityStatus.COMPATIBLE
    if isinstance(report, Mapping):
        status = str(report.get("status") or "").strip().lower()
        return status == QuackCapabilityStatus.COMPATIBLE.value
    return str(report).strip().lower() == QuackCapabilityStatus.COMPATIBLE.value


def enforce_live_gate(
    *,
    profile_compatible: bool,
    scenario_id: ChaosScenarioId | str,
    mode: ScenarioMode | str,
    outcome: ScenarioOutcome | str,
) -> None:
    """Fail closed when a live-required scenario is skipped under a compatible pin.

    Acceptance: live tests cannot silently skip when the profile claims
    compatible. Hermetic execution under a non-compatible profile is allowed
    and must be recorded as ``hermetic``, not omitted.
    """

    mode_value = mode if isinstance(mode, ScenarioMode) else ScenarioMode(str(mode))
    outcome_value = (
        outcome if isinstance(outcome, ScenarioOutcome) else ScenarioOutcome(str(outcome))
    )
    if not profile_compatible:
        return
    if mode_value is ScenarioMode.SKIPPED or outcome_value is ScenarioOutcome.SKIPPED:
        raise QuackChaosLiveGateError(
            f"scenario {scenario_id!s} cannot skip while capability profile "
            "claims compatible"
        )


def require_duckdb_or_raise(*, context: str = "quack chaos") -> None:
    if not duckdb_available():
        raise QuackChaosError(f"DuckDB is required for {context}")


# ---------------------------------------------------------------------------
# Secret scan
# ---------------------------------------------------------------------------


def scan_for_secrets(
    surface: Any,
    *,
    known_secrets: Sequence[str] = (),
    max_bytes: int = DEFAULT_SECRET_SCAN_MAX_BYTES,
) -> list[str]:
    """Return findings if raw secrets appear in a published surface."""

    findings: list[str] = []
    try:
        if isinstance(surface, (bytes, bytearray)):
            text = bytes(surface[:max_bytes]).decode("utf-8", errors="replace")
        elif isinstance(surface, Mapping):
            text = json.dumps(surface, sort_keys=True, default=str)
        elif isinstance(surface, Sequence) and not isinstance(surface, (str, bytes)):
            text = json.dumps(list(surface), sort_keys=True, default=str)
        else:
            text = str(surface)
    except Exception as exc:  # pragma: no cover - defensive
        return [f"surface_serialization_failed:{type(exc).__name__}"]

    if len(text.encode("utf-8")) > max_bytes:
        text = text.encode("utf-8")[:max_bytes].decode("utf-8", errors="replace")

    for secret in known_secrets:
        value = str(secret or "")
        if value and value in text:
            findings.append("known_secret_present")
            break

    # Token-bearing keys with non-handle values are findings.
    if isinstance(surface, Mapping):
        for key, value in surface.items():
            key_text = str(key)
            if _TOKEN_LIKE_RE.search(key_text):
                value_text = str(value)
                if value_text and not value_text.startswith("handle:"):
                    if value_text not in {"[REDACTED]", "redacted", "***"}:
                        findings.append(f"token_bearing_key:{key_text}")
    return findings


# ---------------------------------------------------------------------------
# Fixtures / workers
# ---------------------------------------------------------------------------


def _utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def prepare_chaos_database(
    workspace: Path | str,
    *,
    task_count: int = 8,
    owner_id: str = "quack-chaos",
) -> Path:
    """Install schema, seed generation and tasks under ``workspace``."""

    require_duckdb_or_raise(context="chaos database prepare")
    root = Path(workspace)
    root.mkdir(parents=True, exist_ok=True)
    db = root / "control.duckdb"
    install_control_plane_schema(
        db,
        application_version="0.0.45",
        tool_version="1.5.2",
        owner_id=owner_id,
    )
    with open_duckdb_connection(db) as connection:
        connection.execute("DELETE FROM store_generations")
        connection.execute(
            """
            INSERT INTO store_generations (
                generation, schema_revision, fence_epoch, revision,
                database_uuid, birth_id, created_at
            ) VALUES (1, 1, 1, 0, ?, ?, ?)
            """,
            [
                "123e4567-e89b-12d3-a456-426614174000",
                f"birth:{owner_id}",
                "1970-01-01T00:00:00Z",
            ],
        )
        connection.execute(
            """
            INSERT INTO goals (
                goal_cid, goal_alias, objective_id, parent_goal_cid, ordinal,
                title, status, created_at, updated_at, revision, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                "goal:chaos",
                "G-CHAOS",
                "objective:chaos",
                "",
                1,
                "Chaos",
                "open",
                "1970-01-01T00:00:00Z",
                "1970-01-01T00:00:00Z",
                0,
                "{}",
            ],
        )
        for index in range(int(task_count)):
            task_cid = f"task:chaos:{index + 1:03d}"
            connection.execute(
                """
                INSERT INTO tasks (
                    task_cid, task_alias, goal_cid, plan_cid, objective_id,
                    ordinal, status, revision, priority, created_at, updated_at,
                    identity_json, body_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    task_cid,
                    f"T-CHAOS-{index + 1:03d}",
                    "goal:chaos",
                    "",
                    "objective:chaos",
                    index + 1,
                    "ready",
                    0,
                    "P0",
                    "1970-01-01T00:00:00Z",
                    "1970-01-01T00:00:00Z",
                    "{}",
                    "{}",
                ],
            )
    return db


def _worker_cas(payload: dict[str, Any], queue: Any) -> None:
    """Multiprocess worker used by concurrency / conflict scenarios."""

    db = Path(payload["db"])
    task_cid = str(payload["task_cid"])
    owner_id = str(payload["owner_id"])
    status = str(payload["status"])
    expected_revision = int(payload.get("expected_revision", 0))
    max_attempts = int(payload.get("max_attempts", DEFAULT_MAX_CONFLICT_ATTEMPTS))
    try:
        client = open_embedded_client(
            db,
            owner_id=owner_id,
            seed_generation=False,
            connect_timeout_seconds=60.0,
            retry_policy=RetryPolicy(
                max_attempts=max_attempts,
                base_delay_seconds=0.01,
                max_delay_seconds=0.1,
                jitter_ratio=0.5,
                seed=hash(owner_id) & 0xFFFF,
            ),
        )
        try:
            result = client.cas_task_status(
                task_cid=task_cid,
                expected_task_revision=expected_revision,
                new_status=status,
                idempotency_key=f"idem:{owner_id}:{task_cid}",
                command_id=f"cmd:{owner_id}:{task_cid}",
            )
            queue.put(
                {
                    "ok": True,
                    "outcome": result.outcome.value,
                    "changed": result.changed,
                    "task_cid": task_cid,
                    "attempts": result.attempts,
                    "conflict_kind": (
                        None
                        if result.conflict_kind is None
                        else result.conflict_kind.value
                    ),
                    "owner_id": owner_id,
                }
            )
        finally:
            client.close()
    except Exception as exc:  # pragma: no cover - surfaced via queue
        queue.put(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "task_cid": task_cid,
                "owner_id": owner_id,
            }
        )


def _run_process_workers(
    payloads: Sequence[Mapping[str, Any]],
    *,
    timeout_seconds: float = 90.0,
) -> list[dict[str, Any]]:
    ctx = mp.get_context("spawn")
    queue: Any = ctx.Queue()
    processes = []
    for payload in payloads:
        process = ctx.Process(
            target=_worker_cas,
            args=(dict(payload), queue),
        )
        processes.append(process)
        process.start()
    results: list[dict[str, Any]] = []
    for _ in processes:
        results.append(queue.get(timeout=timeout_seconds))
    for process in processes:
        process.join(timeout=timeout_seconds)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            raise QuackChaosScenarioError("chaos worker process did not exit")
        if process.exitcode not in (0, None):
            # Workers report errors via the queue; non-zero is still a signal.
            pass
    return results


def _compatible_capability_report() -> QuackCapabilityReport:
    profile = default_compatibility_profile()
    return QuackCapabilityReport(
        status=QuackCapabilityStatus.COMPATIBLE,
        profile=profile,
        duckdb_importable=True,
        duckdb_version="1.5.2",
        extension_fingerprint="sha256:" + ("ab" * 32),
        observed_functions=tuple(profile.required_functions),
        observed_surfaces=tuple(profile.required_surfaces),
        beta_limitations=DEFAULT_QUACK_BETA_LIMITATIONS,
    )


# ---------------------------------------------------------------------------
# Individual scenarios
# ---------------------------------------------------------------------------


def _scenario_timer() -> Callable[[], float]:
    started = time.perf_counter()

    def elapsed_ms() -> float:
        return (time.perf_counter() - started) * 1000.0

    return elapsed_ms


def scenario_token_isolation(
    workspace: Path,
    *,
    policy: QuackSecurityPolicy,
) -> ChaosScenarioResult:
    del policy
    elapsed = _scenario_timer()
    state_dir = workspace / "server-token"
    state_dir.mkdir(parents=True, exist_ok=True)
    db = state_dir / "control.duckdb"
    transport = FakeQuackTransport()
    server = build_server(
        database_path=db,
        state_dir=state_dir,
        transport=transport,
        capability_probe=lambda **_kwargs: _compatible_capability_report(),
    )
    identity = server.start()
    try:
        status = server.status()
        logs = server.logs()
        token = None
        if server._vault is not None:  # noqa: SLF001 - chaos inspects vault
            token = server._vault.resolve(identity.secret_handle)
        findings = scan_for_secrets(status, known_secrets=[token or ""])
        findings.extend(scan_for_secrets(logs, known_secrets=[token or ""]))
        findings.extend(
            scan_for_secrets(identity.to_dict(), known_secrets=[token or ""])
        )
        # Provider env must never inherit the raw token.
        provider_env = provider_safe_environment(
            {
                "PATH": "/usr/bin",
                "QUACK_TOKEN": token or "should-not-pass",
                "HOME": str(state_dir / "provider-home"),
                "TASK_ID": TASK_ID,
            }
        )
        if "QUACK_TOKEN" in provider_env:
            findings.append("provider_env_has_quack_token")
        if token and token in provider_env.values():
            findings.append("provider_env_has_raw_token_value")
        status_text = json.dumps(status, default=str)
        logs_text = json.dumps(list(logs), default=str)
        if token and token in status_text:
            findings.append("status_contains_raw_token")
        if token and token in logs_text:
            findings.append("logs_contain_raw_token")
        ok = (
            not findings
            and token is not None
            and bool(token)
            and identity.secret_handle.startswith("handle:")
        )
        return ChaosScenarioResult(
            scenario_id=ChaosScenarioId.TOKEN_ISOLATION,
            outcome=ScenarioOutcome.PASSED if ok else ScenarioOutcome.FAILED,
            mode=ScenarioMode.HERMETIC,
            detail="token absent from status/logs/provider env" if ok else ";".join(findings),
            metrics={"findings": findings, "secret_handle": identity.secret_handle},
            duration_ms=elapsed(),
        )
    finally:
        server.stop()


def scenario_provider_env_isolation(
    *,
    policy: QuackSecurityPolicy,
    raw_token: str = "super-secret-quack-token-value",
) -> ChaosScenarioResult:
    del policy
    elapsed = _scenario_timer()
    env = provider_safe_environment(
        {
            "PATH": "/usr/bin",
            "HOME": "/tmp/provider-home",
            "LANG": "C",
            "QUACK_TOKEN": raw_token,
            "QUACK_AUTH_TOKEN": raw_token,
            "API_KEY": "provider-must-not-see",
            "OPENAI_API_KEY": "sk-test",
            "PASSWORD": "nope",
            "AUTHORIZATION": f"Bearer {raw_token}",
            "TASK_ID": TASK_ID,
            "PYTHONPATH": "/opt/app",
        },
        extra={"BEARER_TOKEN": raw_token, "SAFE_FLAG": "1"},
    )
    denied_present = any(
        key in env
        for key in (
            "QUACK_TOKEN",
            "QUACK_AUTH_TOKEN",
            "API_KEY",
            "OPENAI_API_KEY",
            "PASSWORD",
            "AUTHORIZATION",
            "BEARER_TOKEN",
        )
    )
    token_leaked = raw_token in env.values()
    ok = (
        not denied_present
        and not token_leaked
        and env.get("TASK_ID") == TASK_ID
        and env.get("SAFE_FLAG") == "1"
    )
    return ChaosScenarioResult(
        scenario_id=ChaosScenarioId.PROVIDER_ENV_ISOLATION,
        outcome=ScenarioOutcome.PASSED if ok else ScenarioOutcome.FAILED,
        mode=ScenarioMode.HERMETIC,
        detail=(
            "provider environment excludes token/SQL credentials"
            if ok
            else "provider environment leaked credentials"
        ),
        metrics={"env_keys": sorted(env.keys()), "denied_present": denied_present},
        duration_ms=elapsed(),
    )


def scenario_raw_sql_rejection(db: Path) -> ChaosScenarioResult:
    elapsed = _scenario_timer()
    denials: list[dict[str, Any]] = []
    client = open_embedded_client(db, owner_id="chaos:raw-sql", seed_generation=False)
    try:
        attempts = [
            "SELECT 1",
            "DROP TABLE tasks",
            "ATTACH '/tmp/x.duckdb' AS evil",
            "INSTALL quack",
            "COPY tasks TO '/tmp/out.csv'",
        ]
        for sql in attempts:
            decision = authorize_statement(sql, via_template=False)
            denials.append(decision.to_dict())
            if decision.allowed:
                return ChaosScenarioResult(
                    scenario_id=ChaosScenarioId.RAW_SQL_REJECTION,
                    outcome=ScenarioOutcome.FAILED,
                    mode=ScenarioMode.HERMETIC,
                    detail=f"policy allowed free SQL: {sql!r}",
                    denials=tuple(denials),
                    duration_ms=elapsed(),
                )
            try:
                client.execute_sql(sql)
            except QuackClientSQLError as exc:
                denials.append({"client_error": str(exc), "sql": sql})
            else:
                return ChaosScenarioResult(
                    scenario_id=ChaosScenarioId.RAW_SQL_REJECTION,
                    outcome=ScenarioOutcome.FAILED,
                    mode=ScenarioMode.HERMETIC,
                    detail=f"client executed free SQL: {sql!r}",
                    denials=tuple(denials),
                    duration_ms=elapsed(),
                )
        return ChaosScenarioResult(
            scenario_id=ChaosScenarioId.RAW_SQL_REJECTION,
            outcome=ScenarioOutcome.PASSED,
            mode=ScenarioMode.HERMETIC,
            detail="arbitrary SQL rejected by policy and client before effect",
            denials=tuple(denials),
            duration_ms=elapsed(),
        )
    finally:
        client.close()


def scenario_forbidden_surface() -> ChaosScenarioResult:
    elapsed = _scenario_timer()
    samples = {
        "attach": "ATTACH '/var/lib/evil.duckdb' AS other (READ_WRITE)",
        "install": "INSTALL quack FROM 'http://community.duckdb.org'",
        "load": "LOAD '/tmp/evil.duckdb_extension'",
        "file_read": "SELECT * FROM read_csv_auto('/etc/passwd')",
        "cross_root": "COPY tasks TO '../../../tmp/exfil.csv'",
        "python_udf": "CREATE FUNCTION evil AS (x) -> python_eval(x)",
    }
    denials: list[dict[str, Any]] = []
    for name, sql in samples.items():
        decision = authorize_statement(sql)
        denials.append({"name": name, **decision.to_dict()})
        if decision.allowed or decision.effect_attempted:
            return ChaosScenarioResult(
                scenario_id=ChaosScenarioId.FORBIDDEN_SURFACE,
                outcome=ScenarioOutcome.FAILED,
                mode=ScenarioMode.HERMETIC,
                detail=f"surface {name} was not denied before effect",
                denials=tuple(denials),
                duration_ms=elapsed(),
            )
        # Template registry must also refuse these SQL shapes.
        try:
            StatementTemplate(name=f"evil_{name}", sql=sql, parameter_names=())
        except QuackClientSQLError as exc:
            denials.append({"name": name, "template_error": str(exc)})
        else:
            return ChaosScenarioResult(
                scenario_id=ChaosScenarioId.FORBIDDEN_SURFACE,
                outcome=ScenarioOutcome.FAILED,
                mode=ScenarioMode.HERMETIC,
                detail=f"template accepted forbidden surface {name}",
                denials=tuple(denials),
                duration_ms=elapsed(),
            )
    return ChaosScenarioResult(
        scenario_id=ChaosScenarioId.FORBIDDEN_SURFACE,
        outcome=ScenarioOutcome.PASSED,
        mode=ScenarioMode.HERMETIC,
        detail="file/attach/install/extension/python-udf denied before effect",
        denials=tuple(denials),
        duration_ms=elapsed(),
    )


def scenario_loopback_bind() -> ChaosScenarioResult:
    elapsed = _scenario_timer()
    denials: list[dict[str, Any]] = []
    try:
        assert_bind_admitted(DEFAULT_LOOPBACK_HOST)
    except Exception as exc:  # pragma: no cover - loopback must always pass
        return ChaosScenarioResult(
            scenario_id=ChaosScenarioId.LOOPBACK_BIND,
            outcome=ScenarioOutcome.FAILED,
            mode=ScenarioMode.HERMETIC,
            detail=f"loopback rejected: {exc}",
            duration_ms=elapsed(),
        )
    try:
        assert_bind_admitted("0.0.0.0")
        denials.append({"host": "0.0.0.0", "admitted": True})
        return ChaosScenarioResult(
            scenario_id=ChaosScenarioId.LOOPBACK_BIND,
            outcome=ScenarioOutcome.FAILED,
            mode=ScenarioMode.HERMETIC,
            detail="non-loopback bind admitted without remote policy",
            denials=tuple(denials),
            duration_ms=elapsed(),
        )
    except QuackStateServerBindError as exc:
        denials.append({"host": "0.0.0.0", "error": str(exc)})
    try:
        QuackStateServerConfig(
            database_path=Path("/tmp/control.duckdb"),
            state_dir=Path("/tmp/state"),
            host="8.8.8.8",
        )
        return ChaosScenarioResult(
            scenario_id=ChaosScenarioId.LOOPBACK_BIND,
            outcome=ScenarioOutcome.FAILED,
            mode=ScenarioMode.HERMETIC,
            detail="config admitted non-loopback host",
            duration_ms=elapsed(),
        )
    except QuackStateServerBindError as exc:
        denials.append({"host": "8.8.8.8", "error": str(exc)})
    client = QuackStateClient(owner_id="chaos:loopback")
    try:
        client.attach("quack:8.8.8.8:42100", mode=TransportMode.QUACK)
        return ChaosScenarioResult(
            scenario_id=ChaosScenarioId.LOOPBACK_BIND,
            outcome=ScenarioOutcome.FAILED,
            mode=ScenarioMode.HERMETIC,
            detail="client attached to non-loopback quack URI",
            duration_ms=elapsed(),
        )
    except Exception as exc:
        denials.append({"client_uri": "quack:8.8.8.8:42100", "error": str(exc)})
    finally:
        client.close()
    return ChaosScenarioResult(
        scenario_id=ChaosScenarioId.LOOPBACK_BIND,
        outcome=ScenarioOutcome.PASSED,
        mode=ScenarioMode.HERMETIC,
        detail="non-loopback binds fail closed without reviewed policy",
        denials=tuple(denials),
        duration_ms=elapsed(),
    )


def scenario_statement_authorization(policy: QuackSecurityPolicy) -> ChaosScenarioResult:
    elapsed = _scenario_timer()
    admitted = authorize_statement(
        "SELECT task_cid, status FROM tasks WHERE task_cid = ?",
        policy=policy,
        via_template=True,
    )
    denied = authorize_statement(
        "SELECT task_cid FROM tasks; DROP TABLE tasks",
        policy=policy,
        via_template=True,
    )
    ok = admitted.allowed and denied.denied and not denied.effect_attempted
    return ChaosScenarioResult(
        scenario_id=ChaosScenarioId.STATEMENT_AUTHORIZATION,
        outcome=ScenarioOutcome.PASSED if ok else ScenarioOutcome.FAILED,
        mode=ScenarioMode.HERMETIC,
        detail="parsed verb policy admits templates and denies multi-statement",
        denials=(admitted.to_dict(), denied.to_dict()),
        duration_ms=elapsed(),
    )


def scenario_four_client_concurrency(
    db: Path,
    *,
    policy: QuackSecurityPolicy,
) -> ChaosScenarioResult:
    elapsed = _scenario_timer()
    count = int(policy.client_count)
    payloads = [
        {
            "db": str(db),
            "task_cid": f"task:chaos:{index + 1:03d}",
            "owner_id": f"chaos:client-{index}",
            "status": "claimed",
            "expected_revision": 0,
            "max_attempts": policy.max_retry_attempts,
        }
        for index in range(count)
    ]
    results = _run_process_workers(payloads)
    accepted = [
        item
        for item in results
        if item.get("ok") and item.get("outcome") == CommandOutcome.ACCEPTED.value
    ]
    ok = len(accepted) == count
    # Verify distinct rows advanced once each.
    if ok:
        client = open_embedded_client(
            db, owner_id="chaos:verify-four", seed_generation=False
        )
        try:
            for index in range(count):
                task_cid = f"task:chaos:{index + 1:03d}"
                rows = client.execute("select_task_by_cid", {"task_cid": task_cid})
                if not rows or rows[0]["status"] != "claimed" or int(rows[0]["revision"]) != 1:
                    ok = False
                    break
        finally:
            client.close()
    return ChaosScenarioResult(
        scenario_id=ChaosScenarioId.FOUR_CLIENT_CONCURRENCY,
        outcome=ScenarioOutcome.PASSED if ok else ScenarioOutcome.FAILED,
        mode=ScenarioMode.HERMETIC,
        detail=(
            f"{count} independent clients committed non-conflicting work"
            if ok
            else "four-client concurrency failed"
        ),
        metrics={"results": results, "accepted": len(accepted), "client_count": count},
        duration_ms=elapsed(),
    )


def scenario_hot_row_conflict(
    db: Path,
    *,
    policy: QuackSecurityPolicy,
) -> ChaosScenarioResult:
    elapsed = _scenario_timer()
    # Use a dedicated task not claimed by the four-client scenario when possible.
    task_cid = "task:chaos:007"
    payloads = [
        {
            "db": str(db),
            "task_cid": task_cid,
            "owner_id": f"chaos:hot-{index}",
            "status": f"hot-{index}",
            "expected_revision": 0,
            "max_attempts": policy.max_retry_attempts,
        }
        for index in range(2)
    ]
    results = _run_process_workers(payloads)
    accepted = [
        item
        for item in results
        if item.get("ok") and item.get("outcome") == CommandOutcome.ACCEPTED.value
    ]
    # Exactly one writer wins the same-row CAS at expected_revision=0.
    ok = len(accepted) == 1
    attempts = [int(item.get("attempts") or 0) for item in results if item.get("ok")]
    if attempts and max(attempts) > int(policy.max_retry_attempts):
        ok = False
    final_status = ""
    client = open_embedded_client(db, owner_id="chaos:verify-hot", seed_generation=False)
    try:
        rows = client.execute("select_task_by_cid", {"task_cid": task_cid})
        if not rows or int(rows[0]["revision"]) != 1:
            ok = False
        final_status = str(rows[0]["status"]) if rows else ""
    finally:
        client.close()
    return ChaosScenarioResult(
        scenario_id=ChaosScenarioId.HOT_ROW_CONFLICT,
        outcome=ScenarioOutcome.PASSED if ok else ScenarioOutcome.FAILED,
        mode=ScenarioMode.HERMETIC,
        detail=(
            "same-row conflict bounded to one accepted writer"
            if ok
            else "hot-row conflict not bounded"
        ),
        metrics={
            "results": results,
            "accepted": len(accepted),
            "max_attempts_observed": max(attempts) if attempts else 0,
            "max_attempts_policy": policy.max_retry_attempts,
            "final_status": final_status,
        },
        duration_ms=elapsed(),
    )


def scenario_lost_reply_idempotency(db: Path) -> ChaosScenarioResult:
    elapsed = _scenario_timer()
    task_cid = "task:chaos:008"
    client = open_embedded_client(
        db, owner_id="chaos:lost-reply", seed_generation=False
    )
    try:
        # Ensure row is at a known revision for CAS.
        rows = client.execute("select_task_by_cid", {"task_cid": task_cid})
        revision = int(rows[0]["revision"]) if rows else 0
        first = client.cas_task_status(
            task_cid=task_cid,
            expected_task_revision=revision,
            new_status="idempotent",
            idempotency_key="idem:chaos-lost-reply",
            command_id="cmd:chaos-lost-reply",
        )
        if first.outcome is not CommandOutcome.ACCEPTED:
            return ChaosScenarioResult(
                scenario_id=ChaosScenarioId.LOST_REPLY_IDEMPOTENCY,
                outcome=ScenarioOutcome.FAILED,
                mode=ScenarioMode.HERMETIC,
                detail=f"initial CAS not accepted: {first.outcome.value}",
                duration_ms=elapsed(),
            )
        digest = first.result_digest
        store_rev = client.load_generation().revision
        replay = client.cas_task_status(
            task_cid=task_cid,
            expected_task_revision=revision,
            new_status="idempotent",
            idempotency_key="idem:chaos-lost-reply",
            command_id="cmd:chaos-lost-reply",
        )
        ok = (
            replay.outcome is CommandOutcome.IDEMPOTENT_REPLAY
            and replay.changed is False
            and replay.result_digest == digest
            and client.load_generation().revision == store_rev
        )
        return ChaosScenarioResult(
            scenario_id=ChaosScenarioId.LOST_REPLY_IDEMPOTENCY,
            outcome=ScenarioOutcome.PASSED if ok else ScenarioOutcome.FAILED,
            mode=ScenarioMode.HERMETIC,
            detail=(
                "lost-reply replay returned the one committed result"
                if ok
                else "idempotent replay diverged"
            ),
            metrics={
                "first": first.outcome.value,
                "replay": replay.outcome.value,
                "digest": digest,
            },
            duration_ms=elapsed(),
        )
    finally:
        client.close()


def scenario_retry_jitter(policy: QuackSecurityPolicy) -> ChaosScenarioResult:
    elapsed = _scenario_timer()
    retry = RetryPolicy(
        max_attempts=policy.max_retry_attempts,
        base_delay_seconds=0.05,
        max_delay_seconds=0.4,
        jitter_ratio=0.25,
        seed=42,
    )
    rng = random.Random(42)
    delays = [retry.delay_for_attempt(i, rng=rng) for i in range(0, 5)]
    bounded = all(0.0 <= delay <= retry.max_delay_seconds for delay in delays)
    rng2 = random.Random(42)
    delays2 = [retry.delay_for_attempt(i, rng=rng2) for i in range(0, 5)]
    deterministic = delays == delays2
    default = default_retry_policy()
    ok = (
        bounded
        and deterministic
        and default.max_attempts >= 1
        and retry.max_attempts <= 128
        and retry.max_attempts == policy.max_retry_attempts
    )
    return ChaosScenarioResult(
        scenario_id=ChaosScenarioId.RETRY_JITTER,
        outcome=ScenarioOutcome.PASSED if ok else ScenarioOutcome.FAILED,
        mode=ScenarioMode.HERMETIC,
        detail="retry delays are bounded and jittered deterministically under seed",
        metrics={"delays": delays, "max_attempts": retry.max_attempts},
        duration_ms=elapsed(),
    )


def scenario_stale_after_rotation(
    db: Path,
    workspace: Path,
    *,
    policy: QuackSecurityPolicy,
) -> ChaosScenarioResult:
    del policy
    elapsed = _scenario_timer()
    backup_root = workspace / "backups-rotation"
    backup_root.mkdir(parents=True, exist_ok=True)
    state_dir = workspace / "state-rotation"
    state_dir.mkdir(parents=True, exist_ok=True)

    client = open_embedded_client(
        db, owner_id="chaos:pre-rotation", seed_generation=False
    )
    try:
        pre = client.load_generation()
        pre_generation = pre.generation
        pre_fence = pre.fence_epoch
        session = client.session
        assert session is not None
        # Hold stale command material, then rotate under maintenance-free path.
    finally:
        client.close()

    service = build_control_plane_backup(
        database_path=db,
        backup_root=backup_root,
        state_dir=state_dir,
    )
    rotation = service.rotate_generation(
        reason="chaos-harness-rotation",
        require_maintenance_lease=False,
        birth_id="birth:chaos-rotated",
    )
    if not isinstance(rotation, StoreGenerationRotation):
        return ChaosScenarioResult(
            scenario_id=ChaosScenarioId.STALE_AFTER_ROTATION,
            outcome=ScenarioOutcome.FAILED,
            mode=ScenarioMode.HERMETIC,
            detail="rotation did not return StoreGenerationRotation",
            duration_ms=elapsed(),
        )
    if not rotation.invalidates(pre_generation, pre_fence):
        return ChaosScenarioResult(
            scenario_id=ChaosScenarioId.STALE_AFTER_ROTATION,
            outcome=ScenarioOutcome.FAILED,
            mode=ScenarioMode.HERMETIC,
            detail="rotation did not invalidate pre-rotation writer",
            metrics=rotation.to_dict(),
            duration_ms=elapsed(),
        )

    stale = open_embedded_client(
        db, owner_id="chaos:stale-writer", seed_generation=False
    )
    try:
        live = stale.load_generation()
        if live.generation <= pre_generation:
            return ChaosScenarioResult(
                scenario_id=ChaosScenarioId.STALE_AFTER_ROTATION,
                outcome=ScenarioOutcome.FAILED,
                mode=ScenarioMode.HERMETIC,
                detail="store generation did not advance",
                metrics={"live": live.to_dict(), "rotation": rotation.to_dict()},
                duration_ms=elapsed(),
            )
        session = stale.session
        assert session is not None
        command = StateCommand(
            command_id="cmd:stale-after-rotation",
            command_kind=CommandKind.CLAIM,
            store_id="control.duckdb",
            session_id=session.session_id,
            expected_generation=pre_generation,
            expected_revision=0,
            fence_epoch=pre_fence,
            idempotency_key="idem:stale-after-rotation",
            authority_class=StateAuthorityClass.AUTHORITATIVE,
            parameters={
                "task_cid": "task:chaos:006",
                "expected_task_revision": 0,
                "status": "should-fail",
            },
        )
        result = stale.submit_command(command, refresh_on_conflict=False)
        ok = (
            result.outcome is CommandOutcome.STALE
            or result.conflict_kind
            in {
                TransactionConflictKind.STALE_GENERATION,
                TransactionConflictKind.FENCE_MISMATCH,
            }
        )
        if result.outcome is CommandOutcome.ACCEPTED:
            ok = False
        return ChaosScenarioResult(
            scenario_id=ChaosScenarioId.STALE_AFTER_ROTATION,
            outcome=ScenarioOutcome.PASSED if ok else ScenarioOutcome.FAILED,
            mode=ScenarioMode.HERMETIC,
            detail=(
                "stale client cannot write after generation rotation"
                if ok
                else f"stale write outcome={result.outcome.value}"
            ),
            metrics={
                "pre_generation": pre_generation,
                "live_generation": live.generation,
                "outcome": result.outcome.value,
                "conflict_kind": (
                    None if result.conflict_kind is None else result.conflict_kind.value
                ),
                "rotation": rotation.to_dict(),
            },
            duration_ms=elapsed(),
        )
    finally:
        stale.close()


def scenario_server_restart_stale(workspace: Path) -> ChaosScenarioResult:
    elapsed = _scenario_timer()
    state_dir = workspace / "server-restart"
    state_dir.mkdir(parents=True, exist_ok=True)
    db = state_dir / "control.duckdb"
    transport = FakeQuackTransport()
    server = build_server(
        database_path=db,
        state_dir=state_dir,
        transport=transport,
        capability_probe=lambda **_kwargs: _compatible_capability_report(),
    )
    identity = server.start()
    first_generation = identity.generation
    first_uuid = identity.database_uuid
    first_schema = identity.schema_revision
    server.stop()
    # Simulate restart: new server generation on same store.
    transport2 = FakeQuackTransport()
    server2 = build_server(
        database_path=db,
        state_dir=state_dir,
        transport=transport2,
        capability_probe=lambda **_kwargs: _compatible_capability_report(),
    )
    identity2 = server2.start()
    second_generation = identity2.generation
    # Release exclusive ownership before embedded clients attach to the file.
    server2.stop()

    ok = second_generation > first_generation
    client = open_embedded_client(
        db, owner_id="chaos:restart-stale", seed_generation=False
    )
    try:
        live = client.load_generation()
        session = client.session
        assert session is not None
        command = StateCommand(
            command_id="cmd:restart-stale",
            command_kind=CommandKind.CLAIM,
            store_id="control.duckdb",
            session_id=session.session_id,
            expected_generation=first_generation,
            expected_revision=0,
            fence_epoch=first_generation,
            idempotency_key="idem:restart-stale",
            parameters={
                "task_cid": "task:missing",
                "expected_task_revision": 0,
                "status": "x",
            },
        )
        result = client.submit_command(command, refresh_on_conflict=False)
        stale_ok = result.outcome is CommandOutcome.STALE or (
            result.conflict_kind is TransactionConflictKind.STALE_GENERATION
        )
        # Direct generation assert also fails closed.
        txn = client.transaction(
            expected_generation=StoreGeneration(
                store_id="control.duckdb",
                generation=first_generation,
                schema_revision=first_schema,
                fence_epoch=first_generation,
                revision=0,
                database_uuid=first_uuid,
                birth_id="birth:stale",
            )
        )
        direct_stale = False
        try:
            txn.begin()
            try:
                txn.assert_expected_generation()
            finally:
                txn.rollback()
        except StaleGenerationError:
            direct_stale = True
        ok = (
            ok
            and stale_ok
            and direct_stale
            and live.generation == second_generation
        )
        return ChaosScenarioResult(
            scenario_id=ChaosScenarioId.SERVER_RESTART_STALE,
            outcome=ScenarioOutcome.PASSED if ok else ScenarioOutcome.FAILED,
            mode=ScenarioMode.HERMETIC,
            detail=(
                "stale clients cannot write after server restart"
                if ok
                else "restart did not fence stale clients"
            ),
            metrics={
                "first_generation": first_generation,
                "second_generation": second_generation,
                "submit_outcome": result.outcome.value,
                "direct_stale": direct_stale,
            },
            duration_ms=elapsed(),
        )
    finally:
        client.close()


def scenario_credential_rotation(workspace: Path) -> ChaosScenarioResult:
    elapsed = _scenario_timer()
    state_dir = workspace / "cred-rotation"
    state_dir.mkdir(parents=True, exist_ok=True)
    vault = TokenVault(state_dir)
    handle_v1 = "handle:quack-token:server-chaos:g1"
    handle_v2 = "handle:quack-token:server-chaos:g2"
    secret_v1 = vault.mint(secret_handle=handle_v1, generation=1)
    token_v1 = vault.resolve(handle_v1)
    vault.destroy()
    # Rotated vault must not serve the old handle/token.
    vault2 = TokenVault(state_dir)
    secret_v2 = vault2.mint(secret_handle=handle_v2, generation=2)
    token_v2 = vault2.resolve(handle_v2)
    old_resolve_failed = False
    try:
        vault2.resolve(handle_v1)
    except Exception:
        old_resolve_failed = True
    ok = (
        secret_v1.handle == handle_v1
        and secret_v2.handle == handle_v2
        and token_v1 != token_v2
        and old_resolve_failed
        and vault2.generation == 2
    )
    # Export surfaces must redact token material.
    export = sanitize_for_export(
        {"token": token_v2, "secret_handle": handle_v2, "ok": True},
        token=token_v2,
    )
    if token_v2 in json.dumps(export, default=str):
        ok = False
    return ChaosScenarioResult(
        scenario_id=ChaosScenarioId.CREDENTIAL_ROTATION,
        outcome=ScenarioOutcome.PASSED if ok else ScenarioOutcome.FAILED,
        mode=ScenarioMode.HERMETIC,
        detail=(
            "credential rotation invalidates prior token handles"
            if ok
            else "credential rotation failed"
        ),
        metrics={
            "generation": vault2.generation,
            "old_resolve_failed": old_resolve_failed,
            "export_keys": sorted(export.keys()),
        },
        duration_ms=elapsed(),
    )


def scenario_secret_scan(workspace: Path) -> ChaosScenarioResult:
    elapsed = _scenario_timer()
    state_dir = workspace / "secret-scan"
    state_dir.mkdir(parents=True, exist_ok=True)
    db = state_dir / "control.duckdb"
    transport = FakeQuackTransport()
    server = build_server(
        database_path=db,
        state_dir=state_dir,
        transport=transport,
        capability_probe=lambda **_kwargs: _compatible_capability_report(),
    )
    identity = server.start()
    try:
        token = server._vault.resolve(identity.secret_handle) if server._vault else ""  # noqa: SLF001
        surfaces = {
            "status": server.status(),
            "logs": server.logs(),
            "identity": identity.to_dict(),
            "provider_env": provider_safe_environment(
                {"PATH": "/usr/bin", "QUACK_TOKEN": token, "HOME": str(state_dir)}
            ),
        }
        findings: list[str] = []
        for name, surface in surfaces.items():
            hits = scan_for_secrets(surface, known_secrets=[token])
            findings.extend(f"{name}:{hit}" for hit in hits)
        ok = not findings
        return ChaosScenarioResult(
            scenario_id=ChaosScenarioId.SECRET_SCAN,
            outcome=ScenarioOutcome.PASSED if ok else ScenarioOutcome.FAILED,
            mode=ScenarioMode.HERMETIC,
            detail="published surfaces contain no raw auth token" if ok else ";".join(findings),
            metrics={"findings": findings},
            duration_ms=elapsed(),
        )
    finally:
        server.stop()


def scenario_latency_bound(
    db: Path,
    *,
    policy: QuackSecurityPolicy,
) -> ChaosScenarioResult:
    elapsed = _scenario_timer()
    client = open_embedded_client(
        db, owner_id="chaos:latency", seed_generation=False
    )
    try:
        started = time.perf_counter()
        # Bounded batch of read + one CAS against a free-ish row.
        for _ in range(20):
            client.execute("count_tasks")
        rows = client.execute("select_task_by_cid", {"task_cid": "task:chaos:005"})
        revision = int(rows[0]["revision"]) if rows else 0
        client.cas_task_status(
            task_cid="task:chaos:005",
            expected_task_revision=revision,
            new_status="latency-ok",
            idempotency_key=f"idem:latency:{uuid.uuid4()}",
            command_id=f"cmd:latency:{uuid.uuid4()}",
        )
        duration_ms = (time.perf_counter() - started) * 1000.0
        ok = duration_ms <= float(policy.max_latency_ms)
        return ChaosScenarioResult(
            scenario_id=ChaosScenarioId.LATENCY_BOUND,
            outcome=ScenarioOutcome.PASSED if ok else ScenarioOutcome.FAILED,
            mode=ScenarioMode.HERMETIC,
            detail=(
                f"batch completed in {duration_ms:.1f}ms <= {policy.max_latency_ms}ms"
                if ok
                else f"batch exceeded latency bound: {duration_ms:.1f}ms"
            ),
            metrics={
                "duration_ms": duration_ms,
                "max_latency_ms": policy.max_latency_ms,
            },
            duration_ms=elapsed(),
        )
    finally:
        client.close()


def scenario_denial_logging(policy: QuackSecurityPolicy) -> ChaosScenarioResult:
    elapsed = _scenario_timer()
    denials: list[dict[str, Any]] = []
    samples = [
        "INSTALL quack",
        "ATTACH '/tmp/x' AS z",
        "SELECT * FROM read_csv_auto('/etc/passwd')",
        "CREATE FUNCTION x AS (a) -> a",
    ]
    for sql in samples:
        decision = authorize_statement(sql, policy=policy)
        # Denial logs carry digests + closed reason codes only — never raw SQL.
        record = {
            "sql_digest": f"sha256:{hashlib.sha256(sql.encode()).hexdigest()}",
            "action": decision.action.value,
            "reason": decision.reason.value,
            "verb": decision.verb,
            "effect_attempted": decision.effect_attempted,
            "logged_at": _utc_iso(),
        }
        denials.append(record)
        if decision.allowed or decision.effect_attempted:
            return ChaosScenarioResult(
                scenario_id=ChaosScenarioId.DENIAL_LOGGING,
                outcome=ScenarioOutcome.FAILED,
                mode=ScenarioMode.HERMETIC,
                detail="denial logging observed an allow/effect",
                denials=tuple(denials),
                duration_ms=elapsed(),
            )
    serialized = json.dumps(denials)
    leaked = any(
        fragment in serialized
        for fragment in ("INSTALL quack", "/etc/passwd", "/tmp/x", "python_eval")
    )
    ok = not leaked and all(item["action"] == "deny" for item in denials)
    return ChaosScenarioResult(
        scenario_id=ChaosScenarioId.DENIAL_LOGGING,
        outcome=ScenarioOutcome.PASSED if ok else ScenarioOutcome.FAILED,
        mode=ScenarioMode.HERMETIC,
        detail=(
            "denials recorded with digests and reasons before effect"
            if ok
            else "denial log leaked raw SQL"
        ),
        denials=tuple(denials),
        duration_ms=elapsed(),
    )


def scenario_tls_boundary(policy: QuackSecurityPolicy) -> ChaosScenarioResult:
    elapsed = _scenario_timer()
    ok = (
        policy.tls_required_for_remote
        and "TLS" in policy.tls_boundary_statement
        and policy.require_loopback
        and not policy.allow_remote_without_reviewed_policy
    )
    return ChaosScenarioResult(
        scenario_id=ChaosScenarioId.TLS_BOUNDARY,
        outcome=ScenarioOutcome.PASSED if ok else ScenarioOutcome.FAILED,
        mode=ScenarioMode.HERMETIC,
        detail=policy.tls_boundary_statement if ok else "TLS boundary policy incomplete",
        metrics={"tls_required_for_remote": policy.tls_required_for_remote},
        duration_ms=elapsed(),
    )


def scenario_python_udf_limitation(policy: QuackSecurityPolicy) -> ChaosScenarioResult:
    elapsed = _scenario_timer()
    decision = authorize_statement(
        "CREATE OR REPLACE FUNCTION model_escape(x) AS python_eval(x)",
        policy=policy,
    )
    ok = (
        policy.deny_python_udfs
        and decision.denied
        and decision.reason is AuthorizationReason.PYTHON_UDF
        and not decision.effect_attempted
        and "Python UDF" in policy.python_udf_limitation_statement
    )
    return ChaosScenarioResult(
        scenario_id=ChaosScenarioId.PYTHON_UDF_LIMITATION,
        outcome=ScenarioOutcome.PASSED if ok else ScenarioOutcome.FAILED,
        mode=ScenarioMode.HERMETIC,
        detail=policy.python_udf_limitation_statement if ok else "UDF limitation not enforced",
        denials=(decision.to_dict(),),
        duration_ms=elapsed(),
    )


def scenario_split_brain_ownership(workspace: Path) -> ChaosScenarioResult:
    elapsed = _scenario_timer()
    state_dir = workspace / "split-brain"
    state_dir.mkdir(parents=True, exist_ok=True)
    db = state_dir / "control.duckdb"
    transport_a = FakeQuackTransport()
    server_a = build_server(
        database_path=db,
        state_dir=state_dir,
        transport=transport_a,
        capability_probe=lambda **_kwargs: _compatible_capability_report(),
    )
    identity_a = server_a.start()
    transport_b = FakeQuackTransport()
    server_b = build_server(
        database_path=db,
        state_dir=state_dir,
        transport=transport_b,
        capability_probe=lambda **_kwargs: _compatible_capability_report(),
    )
    second_failed = False
    second_error = ""
    try:
        server_b.start()
    except Exception as exc:
        second_failed = True
        second_error = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            server_b.stop()
        except Exception:
            pass
        server_a.stop()
    ok = second_failed and identity_a.server_id
    return ChaosScenarioResult(
        scenario_id=ChaosScenarioId.SPLIT_BRAIN_OWNERSHIP,
        outcome=ScenarioOutcome.PASSED if ok else ScenarioOutcome.FAILED,
        mode=ScenarioMode.HERMETIC,
        detail=(
            "second concurrent owner fails closed"
            if ok
            else "split-brain ownership was admitted"
        ),
        metrics={
            "first_server_id": identity_a.server_id,
            "second_error": second_error,
        },
        duration_ms=elapsed(),
    )


def scenario_live_gate_policy(
    *,
    profile_compatible: bool,
    observed_modes: Mapping[str, str],
) -> ChaosScenarioResult:
    elapsed = _scenario_timer()
    try:
        for scenario_id, mode_text in observed_modes.items():
            enforce_live_gate(
                profile_compatible=profile_compatible,
                scenario_id=scenario_id,
                mode=mode_text,
                outcome=(
                    ScenarioOutcome.SKIPPED
                    if mode_text == ScenarioMode.SKIPPED.value
                    else ScenarioOutcome.PASSED
                ),
            )
        # Positive control: compatible + skip must raise.
        raised = False
        try:
            enforce_live_gate(
                profile_compatible=True,
                scenario_id=ChaosScenarioId.FOUR_CLIENT_CONCURRENCY,
                mode=ScenarioMode.SKIPPED,
                outcome=ScenarioOutcome.SKIPPED,
            )
        except QuackChaosLiveGateError:
            raised = True
        ok = raised
        # Negative control: non-compatible skip is allowed.
        enforce_live_gate(
            profile_compatible=False,
            scenario_id=ChaosScenarioId.FOUR_CLIENT_CONCURRENCY,
            mode=ScenarioMode.SKIPPED,
            outcome=ScenarioOutcome.SKIPPED,
        )
        return ChaosScenarioResult(
            scenario_id=ChaosScenarioId.LIVE_GATE_POLICY,
            outcome=ScenarioOutcome.PASSED if ok else ScenarioOutcome.FAILED,
            mode=ScenarioMode.HERMETIC,
            detail=(
                "live gate rejects silent skips under compatible profile"
                if ok
                else "live gate did not reject compatible skip"
            ),
            metrics={
                "profile_compatible": profile_compatible,
                "observed_modes": dict(observed_modes),
            },
            duration_ms=elapsed(),
        )
    except QuackChaosLiveGateError as exc:
        return ChaosScenarioResult(
            scenario_id=ChaosScenarioId.LIVE_GATE_POLICY,
            outcome=ScenarioOutcome.FAILED,
            mode=ScenarioMode.HERMETIC,
            detail=str(exc),
            duration_ms=elapsed(),
        )


# ---------------------------------------------------------------------------
# Suite runner
# ---------------------------------------------------------------------------


@dataclass
class QuackChaosHarness:
    """Executable chaos suite bound to one workspace and security policy."""

    workspace: Path
    policy: QuackSecurityPolicy = field(default_factory=default_security_policy)
    capability_probe: Callable[..., QuackCapabilityReport] | None = None
    _results: list[ChaosScenarioResult] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.workspace = Path(self.workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        if not isinstance(self.policy, QuackSecurityPolicy):
            raise TypeError("policy must be QuackSecurityPolicy")

    def _probe(self) -> QuackCapabilityReport:
        if self.capability_probe is not None:
            return self.capability_probe()
        return probe_quack_capabilities()

    def run(self) -> QuackChaosReport:
        """Execute the full required scenario population and seal a report."""

        require_duckdb_or_raise(context="quack chaos suite")
        capability = self._probe()
        compatible = profile_claims_compatible(capability)
        db = prepare_chaos_database(
            self.workspace / "db",
            task_count=max(8, self.policy.client_count + 4),
        )
        results: list[ChaosScenarioResult] = []

        def record(result: ChaosScenarioResult) -> None:
            enforce_live_gate(
                profile_compatible=compatible,
                scenario_id=result.scenario_id,
                mode=result.mode,
                outcome=result.outcome,
            )
            results.append(result)

        def run_scenario(
            scenario_id: ChaosScenarioId,
            fn: Callable[[], ChaosScenarioResult],
        ) -> None:
            try:
                record(fn())
            except QuackChaosLiveGateError:
                raise
            except Exception as exc:
                record(
                    ChaosScenarioResult(
                        scenario_id=scenario_id,
                        outcome=ScenarioOutcome.ERROR,
                        mode=ScenarioMode.HERMETIC,
                        detail=f"{type(exc).__name__}: {exc}",
                    )
                )

        # Security boundary scenarios (always hermetic-executable).
        run_scenario(
            ChaosScenarioId.PROVIDER_ENV_ISOLATION,
            lambda: scenario_provider_env_isolation(policy=self.policy),
        )
        run_scenario(ChaosScenarioId.FORBIDDEN_SURFACE, scenario_forbidden_surface)
        run_scenario(ChaosScenarioId.LOOPBACK_BIND, scenario_loopback_bind)
        run_scenario(
            ChaosScenarioId.STATEMENT_AUTHORIZATION,
            lambda: scenario_statement_authorization(self.policy),
        )
        run_scenario(
            ChaosScenarioId.RETRY_JITTER,
            lambda: scenario_retry_jitter(self.policy),
        )
        run_scenario(
            ChaosScenarioId.TLS_BOUNDARY,
            lambda: scenario_tls_boundary(self.policy),
        )
        run_scenario(
            ChaosScenarioId.PYTHON_UDF_LIMITATION,
            lambda: scenario_python_udf_limitation(self.policy),
        )
        run_scenario(
            ChaosScenarioId.DENIAL_LOGGING,
            lambda: scenario_denial_logging(self.policy),
        )
        run_scenario(
            ChaosScenarioId.CREDENTIAL_ROTATION,
            lambda: scenario_credential_rotation(self.workspace),
        )
        run_scenario(
            ChaosScenarioId.TOKEN_ISOLATION,
            lambda: scenario_token_isolation(self.workspace, policy=self.policy),
        )
        run_scenario(
            ChaosScenarioId.SECRET_SCAN,
            lambda: scenario_secret_scan(self.workspace),
        )
        run_scenario(
            ChaosScenarioId.SPLIT_BRAIN_OWNERSHIP,
            lambda: scenario_split_brain_ownership(self.workspace),
        )
        run_scenario(
            ChaosScenarioId.RAW_SQL_REJECTION,
            lambda: scenario_raw_sql_rejection(db),
        )
        run_scenario(
            ChaosScenarioId.FOUR_CLIENT_CONCURRENCY,
            lambda: scenario_four_client_concurrency(db, policy=self.policy),
        )
        run_scenario(
            ChaosScenarioId.HOT_ROW_CONFLICT,
            lambda: scenario_hot_row_conflict(db, policy=self.policy),
        )
        run_scenario(
            ChaosScenarioId.LOST_REPLY_IDEMPOTENCY,
            lambda: scenario_lost_reply_idempotency(db),
        )
        run_scenario(
            ChaosScenarioId.LATENCY_BOUND,
            lambda: scenario_latency_bound(db, policy=self.policy),
        )
        run_scenario(
            ChaosScenarioId.STALE_AFTER_ROTATION,
            lambda: scenario_stale_after_rotation(
                db, self.workspace, policy=self.policy
            ),
        )
        run_scenario(
            ChaosScenarioId.SERVER_RESTART_STALE,
            lambda: scenario_server_restart_stale(self.workspace),
        )

        observed_modes = {
            item.scenario_id.value: item.mode.value for item in results
        }
        run_scenario(
            ChaosScenarioId.LIVE_GATE_POLICY,
            lambda: scenario_live_gate_policy(
                profile_compatible=compatible,
                observed_modes=observed_modes,
            ),
        )

        # Ensure every required scenario id is present.
        present = {item.scenario_id for item in results}
        for required in REQUIRED_SCENARIOS:
            if required not in present:
                results.append(
                    ChaosScenarioResult(
                        scenario_id=required,
                        outcome=ScenarioOutcome.FAILED,
                        mode=ScenarioMode.HERMETIC,
                        detail="required scenario missing from suite",
                    )
                )

        failed = [
            item
            for item in results
            if item.outcome in {ScenarioOutcome.FAILED, ScenarioOutcome.ERROR}
        ]
        skipped = [
            item for item in results if item.outcome is ScenarioOutcome.SKIPPED
        ]
        if compatible and skipped:
            verdict = ChaosVerdict.FAILED
        elif failed:
            verdict = ChaosVerdict.FAILED
        elif len(results) < len(REQUIRED_SCENARIOS):
            verdict = ChaosVerdict.INCOMPLETE
        else:
            verdict = ChaosVerdict.PASSED

        report = QuackChaosReport(
            verdict=verdict,
            policy=self.policy,
            scenarios=tuple(results),
            profile_status=capability.status.value,
            profile_claims_compatible=compatible,
            live_gate_enforced=True,
            duckdb_available=True,
            workspace=str(self.workspace),
            beta_limitations=tuple(
                capability.beta_limitations or DEFAULT_QUACK_BETA_LIMITATIONS
            ),
            details={
                "extension_fingerprint": capability.extension_fingerprint or "",
                "required_scenario_count": len(REQUIRED_SCENARIOS),
                "executed_scenario_count": len(results),
            },
        )
        self._results = list(results)
        return report


def run_quack_chaos_suite(
    workspace: Path | str,
    *,
    policy: QuackSecurityPolicy | None = None,
    capability_probe: Callable[..., QuackCapabilityReport] | None = None,
) -> QuackChaosReport:
    """Run the sealed DQP-034 chaos suite and return a content-addressed report."""

    harness = QuackChaosHarness(
        workspace=Path(workspace),
        policy=policy or default_security_policy(),
        capability_probe=capability_probe,
    )
    return harness.run()


def assert_chaos_passed(report: QuackChaosReport) -> None:
    """Raise if the chaos suite did not fully pass."""

    if not isinstance(report, QuackChaosReport):
        raise TypeError("report must be QuackChaosReport")
    if report.verdict is not ChaosVerdict.PASSED:
        failed = [
            f"{item.scenario_id.value}:{item.outcome.value}:{item.detail}"
            for item in report.failed_scenarios
        ]
        raise QuackChaosScenarioError(
            "quack chaos suite did not pass: " + "; ".join(failed[:12])
        )


__all__ = (
    "ADMITTED_MUTATION_VERBS",
    "ADMITTED_QUERY_VERBS",
    "AUTHORIZATION_DECISION_SCHEMA",
    "AuthorizationAction",
    "AuthorizationDecision",
    "AuthorizationReason",
    "CHAOS_SCENARIO_RESULT_SCHEMA",
    "ChaosScenarioId",
    "ChaosScenarioResult",
    "ChaosVerdict",
    "DEFAULT_CLIENT_COUNT",
    "DEFAULT_LATENCY_BOUND_MS",
    "DEFAULT_MAX_CONFLICT_ATTEMPTS",
    "EVIDENCE",
    "FORBIDDEN_VERBS",
    "GOAL_ID",
    "QUACK_CHAOS_REPORT_INTERFACE",
    "QUACK_CHAOS_REPORT_SCHEMA",
    "QUACK_CHAOS_VERSION",
    "QUACK_SECURITY_POLICY_INTERFACE",
    "QUACK_SECURITY_POLICY_SCHEMA",
    "QuackChaosError",
    "QuackChaosHarness",
    "QuackChaosLiveGateError",
    "QuackChaosAuthorizationError",
    "QuackChaosReport",
    "QuackChaosScenarioError",
    "QuackSecurityPolicy",
    "REQUIRED_SCENARIOS",
    "ScenarioMode",
    "ScenarioOutcome",
    "TASK_ID",
    "assert_authorized",
    "assert_chaos_passed",
    "authorize_statement",
    "canonical_chaos_bytes",
    "default_security_policy",
    "enforce_live_gate",
    "prepare_chaos_database",
    "profile_claims_compatible",
    "require_duckdb_or_raise",
    "run_quack_chaos_suite",
    "scan_for_secrets",
)
