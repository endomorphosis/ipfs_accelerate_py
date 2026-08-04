"""Generic symbolic assurance rollout control surface.

Gates, modes, control surfaces, schemas, identity constants, and default
fixtures are supplied exclusively by an injected :class:`AssuranceRolloutProfile`.
The control engine evaluates a closed adversarial population against a frozen
multi-repository fixture, derives a non-authoritative shadow-default rollout
decision, and projects bounded status/findings/receipts for Python, CLI, and
MCP surfaces.

Safety is non-waivable:

* automatic mutation remains disabled on every report and decision;
* any gate, binding, or assurance regression returns effective rollout to
  ``shadow``;
* Python, CLI, and MCP publish equivalent bounded status/findings/receipts;
* discovery never imports optional providers and never starts processes.

This module deliberately contains no product-domain path regexes, fixed
repository aliases, environment-variable names, board IDs, or optional-provider
imports.  Domain job adapters own those surfaces through the profile.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from threading import RLock
from typing import Any, Final


# ---------------------------------------------------------------------------
# Generic bounds
# ---------------------------------------------------------------------------

ASSURANCE_ROLLOUT_VERSION: Final = 1

MAX_GATE_EVIDENCE_IDS: Final = 32
MAX_FINDING_PROJECTIONS: Final = 64
MAX_RECEIPT_PROJECTIONS: Final = 64
MAX_REASON_CODES: Final = 128
MAX_BOUNDED_BYTES: Final = 256 * 1024
MAX_PATHS_PER_REPO: Final = 10_000
MAX_REPOSITORIES: Final = 16
MAX_EXCLUSIONS: Final = 1_024
MAX_GATES: Final = 128
MAX_TEXT_BYTES: Final = 512

# Neutral schema defaults (profiles may override every field).
DEFAULT_ADVERSARIAL_E2E_GATE_SCHEMA: Final = "assurance/adversarial-e2e-gate@1"
DEFAULT_SHADOW_ROLLOUT_REPORT_SCHEMA: Final = "assurance/shadow-rollout-report@1"
DEFAULT_ROLLOUT_DECISION_SCHEMA: Final = "assurance/symbolic-rollout-decision@1"
DEFAULT_CONTROL_REQUEST_SCHEMA: Final = "assurance/symbolic-control-request@1"
DEFAULT_CONTROL_RESULT_SCHEMA: Final = "assurance/symbolic-control-result@1"
DEFAULT_BOUNDED_STATUS_SCHEMA: Final = "assurance/symbolic-bounded-status@1"
DEFAULT_BOUNDED_FINDINGS_SCHEMA: Final = "assurance/symbolic-bounded-findings@1"
DEFAULT_BOUNDED_RECEIPTS_SCHEMA: Final = "assurance/symbolic-bounded-receipts@1"
DEFAULT_PUBLIC_API_SCHEMA: Final = "assurance/symbolic-public-api@1"


class SymbolicAssuranceRolloutError(ValueError):
    """Rollout evidence, fixture, policy, or control input is invalid."""


class AssuranceRolloutMode(str, Enum):
    """Authority granted to symbolic-assurance automation."""

    OFF = "off"
    SHADOW = "shadow"
    ASSIST = "assist"
    AUTOMATIC = "automatic"


class ControlAction(str, Enum):
    OFF = "off"
    SHADOW = "shadow"
    ASSIST = "assist"
    AUTOMATIC = "automatic"
    STATUS = "status"
    FINDINGS = "findings"
    RECEIPTS = "receipts"
    EXPLANATION = "explanation"
    ROLLBACK = "rollback"

    @property
    def requested_mode(self) -> AssuranceRolloutMode | None:
        try:
            return AssuranceRolloutMode(self.value)
        except ValueError:
            return None


class ControlSurface(str, Enum):
    PYTHON = "python"
    CLI = "cli"
    MCP = "mcp"


class GateStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    REJECTED = "rejected"


class GateKind(str, Enum):
    """Closed evaluation strategies for profile-defined gates."""

    REPRODUCIBLE_CIDS = "reproducible_cids"
    COMPLETE_INVENTORY = "complete_inventory"
    INVENTORY_EXCLUSIONS = "inventory_exclusions"
    INCREMENTAL_REUSE = "incremental_reuse"
    STALE_CACHE_REJECTION = "stale_cache_rejection"
    CORRUPT_CACHE_REJECTION = "corrupt_cache_rejection"
    CONTRACT_PRECISION = "contract_precision"
    WRONG_PROOF = "wrong_proof"
    UNKNOWN_PROOF = "unknown_proof"
    SIMULATED_ZK = "simulated_zk"
    FORGED_ZK = "forged_zk"
    TAMPERED_ZK = "tampered_zk"
    MCP_MOCK = "mcp_mock"
    MCP_BYPASS = "mcp_bypass"
    SEEDED_DRIFT = "seeded_drift"
    VULNERABILITY_FALSE_POSITIVE = "vulnerability_false_positive"
    TASK_DETERMINISM = "task_determinism"
    PROVIDER_LOSS = "provider_loss"
    RESTART_REPLAY = "restart_replay"
    LEASE_FENCE_LOSS = "lease_fence_loss"
    MERGE_CONFLICT = "merge_conflict"
    BOUNDED_REFILL = "bounded_refill"
    REFILL_EXHAUSTION = "refill_exhaustion"
    ROLLBACK = "rollback"
    CONTROL_PARITY = "control_parity"
    AUTOMATIC_MUTATION_DISABLED = "automatic_mutation_disabled"


# ---------------------------------------------------------------------------
# Canonical helpers
# ---------------------------------------------------------------------------


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _plain(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            _plain(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SymbolicAssuranceRolloutError(
            "rollout data must be canonical JSON"
        ) from exc


def _identity(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _content_cid(body: bytes) -> str:
    return "sha256:" + hashlib.sha256(body).hexdigest()


def _load_json(value: str | bytes | bytearray, name: str) -> Any:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise SymbolicAssuranceRolloutError(
                    f"{name} contains duplicate JSON key {key!r}"
                )
            result[key] = item
        return result

    try:
        if isinstance(value, (bytes, bytearray)):
            text = bytes(value).decode("utf-8")
        else:
            text = str(value)
        return json.loads(text, object_pairs_hook=unique_object)
    except (TypeError, ValueError, UnicodeDecodeError) as exc:
        raise SymbolicAssuranceRolloutError(f"{name} is not valid JSON") from exc


def _text(value: Any, name: str, *, maximum: int = MAX_TEXT_BYTES) -> str:
    if isinstance(value, Enum):
        value = value.value
    if not isinstance(value, str) or not value or value != value.strip():
        raise SymbolicAssuranceRolloutError(f"{name} must be non-empty text")
    if "\x00" in value or len(value.encode("utf-8")) > maximum:
        raise SymbolicAssuranceRolloutError(
            f"{name} is unsafe or exceeds {maximum} bytes"
        )
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise SymbolicAssuranceRolloutError(f"{name} must be a boolean")
    return value


def _non_negative_int(
    value: Any, name: str, *, maximum: int | None = None
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SymbolicAssuranceRolloutError(f"{name} must be a non-negative int")
    if maximum is not None and value > maximum:
        raise SymbolicAssuranceRolloutError(f"{name} exceeds maximum {maximum}")
    return value


def _timestamp(value: datetime | str, name: str) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    text = _text(value, name, maximum=64)
    # Accept trailing Z or offset; require parseable ISO-like form.
    try:
        _datetime(text)
    except ValueError as exc:
        raise SymbolicAssuranceRolloutError(f"{name} is not an ISO timestamp") from exc
    return text


def _datetime(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def _mode(value: Any) -> AssuranceRolloutMode:
    if isinstance(value, AssuranceRolloutMode):
        return value
    try:
        return AssuranceRolloutMode(str(value))
    except ValueError as exc:
        raise SymbolicAssuranceRolloutError(f"unknown rollout mode: {value!r}") from exc


def _gate_kind(value: Any) -> GateKind:
    if isinstance(value, GateKind):
        return value
    try:
        return GateKind(str(value))
    except ValueError as exc:
        raise SymbolicAssuranceRolloutError(f"unknown gate kind: {value!r}") from exc


def _status(value: Any) -> GateStatus:
    if isinstance(value, GateStatus):
        return value
    try:
        return GateStatus(str(value))
    except ValueError as exc:
        raise SymbolicAssuranceRolloutError(f"unknown gate status: {value!r}") from exc


def _unique_sorted_texts(
    values: Iterable[Any],
    name: str,
    *,
    maximum: int,
) -> tuple[str, ...]:
    items = [_text(item, name, maximum=1024) for item in values]
    unique = tuple(sorted(set(items)))
    if len(unique) > maximum:
        raise SymbolicAssuranceRolloutError(f"{name} exceeds maximum {maximum}")
    if len(unique) != len(items):
        # preserve sorted unique order for determinism
        pass
    return unique


# ---------------------------------------------------------------------------
# Profile definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateDefinition:
    """One profile-defined adversarial gate."""

    gate_id: str
    kind: GateKind | str
    expected_outcome: str
    evidence_ids: tuple[str, ...] = ()
    reject_on_pass: bool = False
    non_authoritative: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "gate_id", _text(self.gate_id, "gate_id"))
        object.__setattr__(self, "kind", _gate_kind(self.kind))
        object.__setattr__(
            self,
            "expected_outcome",
            _text(self.expected_outcome, "expected_outcome", maximum=256),
        )
        evidence = _unique_sorted_texts(
            self.evidence_ids, "evidence_ids", maximum=MAX_GATE_EVIDENCE_IDS
        )
        object.__setattr__(self, "evidence_ids", evidence)
        object.__setattr__(
            self, "reject_on_pass", _boolean(self.reject_on_pass, "reject_on_pass")
        )
        object.__setattr__(
            self,
            "non_authoritative",
            _boolean(self.non_authoritative, "non_authoritative"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "kind": self.kind.value if isinstance(self.kind, GateKind) else str(self.kind),
            "expected_outcome": self.expected_outcome,
            "evidence_ids": list(self.evidence_ids),
            "reject_on_pass": self.reject_on_pass,
            "non_authoritative": self.non_authoritative,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GateDefinition":
        return cls(
            gate_id=value["gate_id"],
            kind=value["kind"],
            expected_outcome=value["expected_outcome"],
            evidence_ids=tuple(value.get("evidence_ids", ())),
            reject_on_pass=bool(value.get("reject_on_pass", False)),
            non_authoritative=bool(value.get("non_authoritative", True)),
        )


@dataclass(frozen=True)
class AssuranceRolloutSchemas:
    """Content-identified schema surface for publications."""

    adversarial_e2e_gate: str = DEFAULT_ADVERSARIAL_E2E_GATE_SCHEMA
    shadow_rollout_report: str = DEFAULT_SHADOW_ROLLOUT_REPORT_SCHEMA
    rollout_decision: str = DEFAULT_ROLLOUT_DECISION_SCHEMA
    control_request: str = DEFAULT_CONTROL_REQUEST_SCHEMA
    control_result: str = DEFAULT_CONTROL_RESULT_SCHEMA
    bounded_status: str = DEFAULT_BOUNDED_STATUS_SCHEMA
    bounded_findings: str = DEFAULT_BOUNDED_FINDINGS_SCHEMA
    bounded_receipts: str = DEFAULT_BOUNDED_RECEIPTS_SCHEMA
    public_api: str = DEFAULT_PUBLIC_API_SCHEMA
    version: int = ASSURANCE_ROLLOUT_VERSION

    def __post_init__(self) -> None:
        for name in (
            "adversarial_e2e_gate",
            "shadow_rollout_report",
            "rollout_decision",
            "control_request",
            "control_result",
            "bounded_status",
            "bounded_findings",
            "bounded_receipts",
            "public_api",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, maximum=256)
            )
        object.__setattr__(
            self, "version", _non_negative_int(self.version, "version", maximum=1024)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "adversarial_e2e_gate": self.adversarial_e2e_gate,
            "shadow_rollout_report": self.shadow_rollout_report,
            "rollout_decision": self.rollout_decision,
            "control_request": self.control_request,
            "control_result": self.control_result,
            "bounded_status": self.bounded_status,
            "bounded_findings": self.bounded_findings,
            "bounded_receipts": self.bounded_receipts,
            "public_api": self.public_api,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AssuranceRolloutSchemas":
        return cls(
            adversarial_e2e_gate=value.get(
                "adversarial_e2e_gate", DEFAULT_ADVERSARIAL_E2E_GATE_SCHEMA
            ),
            shadow_rollout_report=value.get(
                "shadow_rollout_report", DEFAULT_SHADOW_ROLLOUT_REPORT_SCHEMA
            ),
            rollout_decision=value.get(
                "rollout_decision", DEFAULT_ROLLOUT_DECISION_SCHEMA
            ),
            control_request=value.get(
                "control_request", DEFAULT_CONTROL_REQUEST_SCHEMA
            ),
            control_result=value.get(
                "control_result", DEFAULT_CONTROL_RESULT_SCHEMA
            ),
            bounded_status=value.get(
                "bounded_status", DEFAULT_BOUNDED_STATUS_SCHEMA
            ),
            bounded_findings=value.get(
                "bounded_findings", DEFAULT_BOUNDED_FINDINGS_SCHEMA
            ),
            bounded_receipts=value.get(
                "bounded_receipts", DEFAULT_BOUNDED_RECEIPTS_SCHEMA
            ),
            public_api=value.get("public_api", DEFAULT_PUBLIC_API_SCHEMA),
            version=int(value.get("version", ASSURANCE_ROLLOUT_VERSION)),
        )


@dataclass(frozen=True)
class AssuranceRolloutProfile:
    """Immutable, content-identified rollout profile.

    Domain vocabulary (gate IDs, schemas, behavior/objective IDs, default
    fixtures, exclusion prefixes) is profile data, not module constants.
    """

    profile_id: str
    behavior_id: str
    objective_id: str
    objective_revision: str
    requirement_id: str
    gates: tuple[GateDefinition, ...]
    schemas: AssuranceRolloutSchemas = field(default_factory=AssuranceRolloutSchemas)
    default_exclusion_prefixes: tuple[str, ...] = (
        "node_modules/",
        ".git/",
        "__pycache__/",
        ".pytest_cache/",
        "build/",
        "archive/",
    )
    default_fixture_repositories: Mapping[str, Mapping[str, str]] = field(
        default_factory=dict
    )
    default_fixture_id: str = "fixture:adversarial-e2e@1"
    default_fixture_revision: str = "fixture-revision:1"
    inventory_policy_id: str = "inventory-policy:adversarial@1"
    inventory_policy_revision: str = "inventory-policy-revision:1"
    policy_id: str = "policy:symbolic-assurance-rollout@1"
    policy_revision: str = "sha256:frozen-symbolic-assurance-policy"
    capability_id: str = "capability:symbolic-assurance-local@1"
    capability_revision: str = "sha256:frozen-symbolic-assurance-capability"
    toolchain_id: str = "toolchain:symbolic-assurance@1"
    toolchain_revision: str = "toolchain-revision:1"
    default_mode: AssuranceRolloutMode | str = AssuranceRolloutMode.SHADOW
    automatic_mutation_enabled: bool = False
    authority_flags: Mapping[str, bool] = field(
        default_factory=lambda: {
            "authoritative": False,
            "completion_authoritative": False,
            "inventory_is_completion_evidence": False,
            "inventory_is_correctness_evidence": False,
            "inventory_authorizes_repair": False,
            "variant_presence_is_defect": False,
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "profile_id", _text(self.profile_id, "profile_id")
        )
        object.__setattr__(
            self, "behavior_id", _text(self.behavior_id, "behavior_id")
        )
        object.__setattr__(
            self, "objective_id", _text(self.objective_id, "objective_id")
        )
        object.__setattr__(
            self,
            "objective_revision",
            _text(self.objective_revision, "objective_revision"),
        )
        object.__setattr__(
            self, "requirement_id", _text(self.requirement_id, "requirement_id")
        )
        if not isinstance(self.schemas, AssuranceRolloutSchemas):
            raise SymbolicAssuranceRolloutError("schemas has the wrong type")
        gates = tuple(self.gates)
        if not gates:
            raise SymbolicAssuranceRolloutError("profile requires at least one gate")
        if len(gates) > MAX_GATES:
            raise SymbolicAssuranceRolloutError("too many gates in profile")
        if not all(isinstance(item, GateDefinition) for item in gates):
            raise SymbolicAssuranceRolloutError("gates have the wrong type")
        ids = [item.gate_id for item in gates]
        if len(ids) != len(set(ids)):
            raise SymbolicAssuranceRolloutError("gate IDs must be unique")
        kinds = [item.kind for item in gates]
        if len(kinds) != len(set(kinds)):
            raise SymbolicAssuranceRolloutError(
                "gate kinds must be unique within a profile"
            )
        object.__setattr__(self, "gates", gates)
        prefixes = _unique_sorted_texts(
            self.default_exclusion_prefixes,
            "default_exclusion_prefixes",
            maximum=MAX_EXCLUSIONS,
        )
        object.__setattr__(self, "default_exclusion_prefixes", prefixes)
        repos = {
            _text(key, "repository_id"): {
                _text(path, "path", maximum=1024): str(body)
                for path, body in dict(files).items()
            }
            for key, files in dict(self.default_fixture_repositories).items()
        }
        if len(repos) > MAX_REPOSITORIES:
            raise SymbolicAssuranceRolloutError("too many default fixture repositories")
        object.__setattr__(
            self,
            "default_fixture_repositories",
            {key: dict(sorted(value.items())) for key, value in sorted(repos.items())},
        )
        for name in (
            "default_fixture_id",
            "default_fixture_revision",
            "inventory_policy_id",
            "inventory_policy_revision",
            "policy_id",
            "policy_revision",
            "capability_id",
            "capability_revision",
            "toolchain_id",
            "toolchain_revision",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(self, "default_mode", _mode(self.default_mode))
        object.__setattr__(
            self,
            "automatic_mutation_enabled",
            _boolean(
                self.automatic_mutation_enabled, "automatic_mutation_enabled"
            ),
        )
        if self.automatic_mutation_enabled:
            raise SymbolicAssuranceRolloutError(
                "automatic mutation remains disabled for symbolic assurance rollout"
            )
        flags = {
            _text(key, "authority_flag"): _boolean(val, "authority_flag_value")
            for key, val in dict(self.authority_flags).items()
        }
        if any(flags.values()):
            raise SymbolicAssuranceRolloutError(
                "assurance rollout authority flags must remain non-authoritative"
            )
        object.__setattr__(self, "authority_flags", dict(sorted(flags.items())))

    @property
    def profile_content_id(self) -> str:
        return _identity(self.to_dict())

    @property
    def required_gate_ids(self) -> tuple[str, ...]:
        return tuple(item.gate_id for item in self.gates)

    def gate_by_id(self, gate_id: str) -> GateDefinition:
        for item in self.gates:
            if item.gate_id == gate_id:
                return item
        raise SymbolicAssuranceRolloutError(f"unknown gate_id: {gate_id}")

    def gate_by_kind(self, kind: GateKind | str) -> GateDefinition:
        selected = _gate_kind(kind)
        for item in self.gates:
            if item.kind is selected:
                return item
        raise SymbolicAssuranceRolloutError(f"unknown gate kind: {selected.value}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "behavior_id": self.behavior_id,
            "objective_id": self.objective_id,
            "objective_revision": self.objective_revision,
            "requirement_id": self.requirement_id,
            "gates": [item.to_dict() for item in self.gates],
            "schemas": self.schemas.to_dict(),
            "default_exclusion_prefixes": list(self.default_exclusion_prefixes),
            "default_fixture_repositories": {
                key: dict(value)
                for key, value in self.default_fixture_repositories.items()
            },
            "default_fixture_id": self.default_fixture_id,
            "default_fixture_revision": self.default_fixture_revision,
            "inventory_policy_id": self.inventory_policy_id,
            "inventory_policy_revision": self.inventory_policy_revision,
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "capability_id": self.capability_id,
            "capability_revision": self.capability_revision,
            "toolchain_id": self.toolchain_id,
            "toolchain_revision": self.toolchain_revision,
            "default_mode": (
                self.default_mode.value
                if isinstance(self.default_mode, AssuranceRolloutMode)
                else str(self.default_mode)
            ),
            "automatic_mutation_enabled": False,
            "authority_flags": dict(self.authority_flags),
            "profile_content_id": None,  # filled below for external consumers only
        }

    def to_dict_with_identity(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload.pop("profile_content_id", None)
        payload["profile_content_id"] = _identity(payload)
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AssuranceRolloutProfile":
        gates = tuple(
            GateDefinition.from_dict(item) for item in value.get("gates", ())
        )
        schemas = AssuranceRolloutSchemas.from_dict(value.get("schemas", {}))
        return cls(
            profile_id=value["profile_id"],
            behavior_id=value["behavior_id"],
            objective_id=value["objective_id"],
            objective_revision=value["objective_revision"],
            requirement_id=value["requirement_id"],
            gates=gates,
            schemas=schemas,
            default_exclusion_prefixes=tuple(
                value.get("default_exclusion_prefixes", ())
            ),
            default_fixture_repositories=dict(
                value.get("default_fixture_repositories", {})
            ),
            default_fixture_id=value.get(
                "default_fixture_id", "fixture:adversarial-e2e@1"
            ),
            default_fixture_revision=value.get(
                "default_fixture_revision", "fixture-revision:1"
            ),
            inventory_policy_id=value.get(
                "inventory_policy_id", "inventory-policy:adversarial@1"
            ),
            inventory_policy_revision=value.get(
                "inventory_policy_revision", "inventory-policy-revision:1"
            ),
            policy_id=value.get("policy_id", "policy:symbolic-assurance-rollout@1"),
            policy_revision=value.get(
                "policy_revision", "sha256:frozen-symbolic-assurance-policy"
            ),
            capability_id=value.get(
                "capability_id", "capability:symbolic-assurance-local@1"
            ),
            capability_revision=value.get(
                "capability_revision",
                "sha256:frozen-symbolic-assurance-capability",
            ),
            toolchain_id=value.get(
                "toolchain_id", "toolchain:symbolic-assurance@1"
            ),
            toolchain_revision=value.get(
                "toolchain_revision", "toolchain-revision:1"
            ),
            default_mode=value.get("default_mode", AssuranceRolloutMode.SHADOW),
            automatic_mutation_enabled=bool(
                value.get("automatic_mutation_enabled", False)
            ),
            authority_flags=dict(value.get("authority_flags", {})),
        )


def _default_gate_definitions() -> tuple[GateDefinition, ...]:
    """Closed generic gate population (IDs equal kinds; profiles may rename)."""

    specs: list[tuple[GateKind, str, tuple[str, ...], bool]] = [
        (GateKind.REPRODUCIBLE_CIDS, "identical-fixture-and-repository-cids", (), False),
        (GateKind.COMPLETE_INVENTORY, "exhaustive-included-paths", (), False),
        (GateKind.INVENTORY_EXCLUSIONS, "policy-bound-exclusions", (), False),
        (GateKind.INCREMENTAL_REUSE, "full-digest-reuse-on-warm-scan", (), False),
        (GateKind.STALE_CACHE_REJECTION, "stale-authoritative-hit-rejected", ("cache:stale-probe",), True),
        (GateKind.CORRUPT_CACHE_REJECTION, "corrupt-cache-entry-rejected", ("cache:corrupt-probe",), True),
        (GateKind.CONTRACT_PRECISION, "seeded-mismatch-precision", ("contract:seeded-mismatch",), False),
        (GateKind.WRONG_PROOF, "wrong-proof-rejected", ("proof:wrong",), True),
        (GateKind.UNKNOWN_PROOF, "unknown-proof-non-authoritative", ("proof:unknown",), False),
        (GateKind.SIMULATED_ZK, "simulated-zk-non-authoritative", ("zk:simulated_zk",), False),
        (GateKind.FORGED_ZK, "forged-zk-rejected", ("zk:forged_zk",), True),
        (GateKind.TAMPERED_ZK, "tampered-zk-rejected", ("zk:tampered_zk",), True),
        (GateKind.MCP_MOCK, "mcp-mock-explicit-non-authoritative", ("mcp:mock-probe",), False),
        (GateKind.MCP_BYPASS, "mcp-local-bypass-reported", ("mcp:bypass-probe",), False),
        (GateKind.SEEDED_DRIFT, "seeded-drift-detected", ("surface:seeded-drift",), False),
        (GateKind.VULNERABILITY_FALSE_POSITIVE, "false-positive-not-emitted-as-vulnerability", ("security:false-positive-seed",), False),
        (GateKind.TASK_DETERMINISM, "stable-task-identity", (), False),
        (GateKind.PROVIDER_LOSS, "provider-loss-degrades-without-authority-expansion", ("provider:loss-probe",), False),
        (GateKind.RESTART_REPLAY, "restart-replay-byte-identical", ("runtime:restart-replay",), False),
        (GateKind.LEASE_FENCE_LOSS, "lease-fence-loss-blocks-mutation", ("lease:fence-loss",), False),
        (GateKind.MERGE_CONFLICT, "merge-conflict-serialized-and-reported", ("merge:conflict-probe",), False),
        (GateKind.BOUNDED_REFILL, "refill-within-admission-ceilings", ("refill:bounded",), False),
        (GateKind.REFILL_EXHAUSTION, "healthy-exhaustion-no-busywork", ("refill:exhaustion",), False),
        (GateKind.ROLLBACK, "regression-returns-effective-mode-to-shadow", ("rollout:rollback",), False),
        (GateKind.CONTROL_PARITY, "python-cli-mcp-byte-identical-projections", ("control:parity",), False),
        (GateKind.AUTOMATIC_MUTATION_DISABLED, "automatic-mutation-disabled", ("policy:automatic-mutation",), False),
    ]
    return tuple(
        GateDefinition(
            gate_id=kind.value,
            kind=kind,
            expected_outcome=expected,
            evidence_ids=evidence,
            reject_on_pass=reject,
        )
        for kind, expected, evidence, reject in specs
    )


def build_generic_rollout_profile(
    *,
    profile_id: str = "profile:symbolic-assurance-rollout@1",
    behavior_id: str = "behavior:symbolic-assurance-rollout@1",
    objective_id: str = "ASSURANCE-G001",
    objective_revision: str = "ASSURANCE-G001@rollout-1",
    requirement_id: str = "assurance:adversarial-e2e-control-parity-recovery-rollback",
    gate_id_map: Mapping[str, str] | None = None,
    schemas: AssuranceRolloutSchemas | None = None,
    default_fixture_repositories: Mapping[str, Mapping[str, str]] | None = None,
    **overrides: Any,
) -> AssuranceRolloutProfile:
    """Build a hermetic non-domain profile for the public rollout API."""

    id_map = dict(gate_id_map or {})
    gates = []
    for item in _default_gate_definitions():
        gate_id = id_map.get(item.kind.value, item.gate_id)
        gates.append(
            GateDefinition(
                gate_id=gate_id,
                kind=item.kind,
                expected_outcome=item.expected_outcome,
                evidence_ids=item.evidence_ids,
                reject_on_pass=item.reject_on_pass,
                non_authoritative=item.non_authoritative,
            )
        )
    fixtures = default_fixture_repositories
    if fixtures is None:
        fixtures = {
            "repository:alpha@fixture": {
                "src/service.py": "def serve():\n    return 'ok'\n",
                "src/api.py": "def handle(req):\n    return req\n",
                "node_modules/skip/index.js": "module.exports = {};\n",
                ".git/config": "[core]\n",
            },
            "repository:beta@fixture": {
                "lib/core.py": "def compute(x):\n    return x + 1\n",
                "__pycache__/skip.pyc": "bytecode",
                ".pytest_cache/v/cache": "stale",
                "build/lib/generated.py": "# generated\n",
            },
            "repository:gamma@fixture": {
                "pkg/main.py": "x = 1\n",
                "archive/old/skip.py": "# archive\n",
                "tests/test_main.py": "def test_ok(): assert True\n",
            },
            "repository:delta@fixture": {
                "app/run.py": "def run():\n    return 0\n",
                "README.md": "# delta\n",
            },
        }
    kwargs = {
        "profile_id": profile_id,
        "behavior_id": behavior_id,
        "objective_id": objective_id,
        "objective_revision": objective_revision,
        "requirement_id": requirement_id,
        "gates": tuple(gates),
        "schemas": schemas or AssuranceRolloutSchemas(),
        "default_fixture_repositories": fixtures,
    }
    kwargs.update(overrides)
    return AssuranceRolloutProfile(**kwargs)


# ---------------------------------------------------------------------------
# Frozen multi-repository fixture
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrozenRepositoryDescriptor:
    """One independently bound repository in the frozen forest."""

    repository_id: str
    alias: str
    commit: str
    tree_id: str
    content_cid: str
    included_paths: tuple[str, ...]
    excluded_paths: tuple[str, ...]
    path_digests: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "repository_id", _text(self.repository_id, "repository_id")
        )
        object.__setattr__(self, "alias", _text(self.alias, "alias"))
        object.__setattr__(self, "commit", _text(self.commit, "commit"))
        object.__setattr__(self, "tree_id", _text(self.tree_id, "tree_id"))
        object.__setattr__(
            self, "content_cid", _text(self.content_cid, "content_cid")
        )
        included = _unique_sorted_texts(
            self.included_paths, "included_paths", maximum=MAX_PATHS_PER_REPO
        )
        excluded = _unique_sorted_texts(
            self.excluded_paths, "excluded_paths", maximum=MAX_EXCLUSIONS
        )
        if set(included) & set(excluded):
            raise SymbolicAssuranceRolloutError(
                "included and excluded paths must be disjoint"
            )
        digests = {
            _text(path, "path_digests"): _text(digest, "path_digest")
            for path, digest in dict(self.path_digests).items()
        }
        if set(digests) != set(included) | set(excluded):
            raise SymbolicAssuranceRolloutError(
                "path_digests must cover every included and excluded path"
            )
        object.__setattr__(self, "included_paths", included)
        object.__setattr__(self, "excluded_paths", excluded)
        object.__setattr__(self, "path_digests", dict(sorted(digests.items())))
        expected = _identity(
            {
                "repository_id": self.repository_id,
                "alias": self.alias,
                "commit": self.commit,
                "included_paths": list(included),
                "excluded_paths": list(excluded),
                "path_digests": self.path_digests,
            }
        )
        if self.tree_id != expected and not self.tree_id.startswith("sha256:"):
            raise SymbolicAssuranceRolloutError("tree_id must be a content identity")
        if self.content_cid != expected:
            if self.content_cid != _identity(
                {
                    "repository_id": self.repository_id,
                    "path_digests": self.path_digests,
                }
            ):
                raise SymbolicAssuranceRolloutError(
                    "content_cid does not match repository path digests"
                )

    @property
    def observed_paths(self) -> int:
        return len(self.included_paths) + len(self.excluded_paths)

    @property
    def exhaustive(self) -> bool:
        return self.observed_paths == len(self.path_digests)

    def to_dict(self) -> dict[str, Any]:
        return {
            "repository_id": self.repository_id,
            "alias": self.alias,
            "commit": self.commit,
            "tree_id": self.tree_id,
            "content_cid": self.content_cid,
            "included_paths": list(self.included_paths),
            "excluded_paths": list(self.excluded_paths),
            "path_digests": dict(self.path_digests),
            "observed_paths": self.observed_paths,
            "exhaustive": self.exhaustive,
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "FrozenRepositoryDescriptor":
        required = {
            "repository_id",
            "alias",
            "commit",
            "tree_id",
            "content_cid",
            "included_paths",
            "excluded_paths",
            "path_digests",
        }
        if not required.issubset(value):
            raise SymbolicAssuranceRolloutError(
                "frozen repository descriptor is missing fields"
            )
        return cls(
            repository_id=value["repository_id"],
            alias=value["alias"],
            commit=value["commit"],
            tree_id=value["tree_id"],
            content_cid=value["content_cid"],
            included_paths=tuple(value["included_paths"]),
            excluded_paths=tuple(value["excluded_paths"]),
            path_digests=dict(value["path_digests"]),
        )


@dataclass(frozen=True)
class FrozenMultiRepoFixture:
    """Content-addressed multi-repository forest used by every gate."""

    fixture_id: str
    fixture_revision: str
    forest_id: str
    repositories: tuple[FrozenRepositoryDescriptor, ...]
    exclusion_prefixes: tuple[str, ...]
    inventory_policy_id: str
    inventory_policy_revision: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "fixture_id", _text(self.fixture_id, "fixture_id")
        )
        object.__setattr__(
            self,
            "fixture_revision",
            _text(self.fixture_revision, "fixture_revision"),
        )
        object.__setattr__(
            self, "forest_id", _text(self.forest_id, "forest_id")
        )
        repos = tuple(self.repositories)
        if not repos:
            raise SymbolicAssuranceRolloutError("fixture requires repositories")
        if len(repos) > MAX_REPOSITORIES:
            raise SymbolicAssuranceRolloutError("too many repositories in fixture")
        ids = [item.repository_id for item in repos]
        aliases = [item.alias for item in repos]
        if len(ids) != len(set(ids)) or len(aliases) != len(set(aliases)):
            raise SymbolicAssuranceRolloutError(
                "repository ids and aliases must be unique"
            )
        if not all(isinstance(item, FrozenRepositoryDescriptor) for item in repos):
            raise SymbolicAssuranceRolloutError("repositories have the wrong type")
        prefixes = _unique_sorted_texts(
            self.exclusion_prefixes,
            "exclusion_prefixes",
            maximum=MAX_EXCLUSIONS,
        )
        object.__setattr__(self, "repositories", repos)
        object.__setattr__(self, "exclusion_prefixes", prefixes)
        object.__setattr__(
            self,
            "inventory_policy_id",
            _text(self.inventory_policy_id, "inventory_policy_id"),
        )
        object.__setattr__(
            self,
            "inventory_policy_revision",
            _text(
                self.inventory_policy_revision, "inventory_policy_revision"
            ),
        )
        expected_forest = _identity(
            {
                "fixture_id": self.fixture_id,
                "fixture_revision": self.fixture_revision,
                "repositories": [item.to_dict() for item in repos],
                "exclusion_prefixes": list(prefixes),
                "inventory_policy_id": self.inventory_policy_id,
                "inventory_policy_revision": self.inventory_policy_revision,
            }
        )
        if self.forest_id != expected_forest:
            raise SymbolicAssuranceRolloutError(
                "forest_id does not match frozen multi-repository population"
            )

    @property
    def fixture_cid(self) -> str:
        return self.forest_id

    @property
    def repository_ids(self) -> tuple[str, ...]:
        return tuple(item.repository_id for item in self.repositories)

    @property
    def total_included_paths(self) -> int:
        return sum(len(item.included_paths) for item in self.repositories)

    @property
    def total_excluded_paths(self) -> int:
        return sum(len(item.excluded_paths) for item in self.repositories)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id,
            "fixture_revision": self.fixture_revision,
            "forest_id": self.forest_id,
            "fixture_cid": self.fixture_cid,
            "repositories": [item.to_dict() for item in self.repositories],
            "exclusion_prefixes": list(self.exclusion_prefixes),
            "inventory_policy_id": self.inventory_policy_id,
            "inventory_policy_revision": self.inventory_policy_revision,
            "total_included_paths": self.total_included_paths,
            "total_excluded_paths": self.total_excluded_paths,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FrozenMultiRepoFixture":
        repos = tuple(
            FrozenRepositoryDescriptor.from_dict(item)
            for item in value.get("repositories", ())
        )
        return cls(
            fixture_id=value["fixture_id"],
            fixture_revision=value["fixture_revision"],
            forest_id=value["forest_id"],
            repositories=repos,
            exclusion_prefixes=tuple(value.get("exclusion_prefixes", ())),
            inventory_policy_id=value.get(
                "inventory_policy_id", "inventory-policy:adversarial@1"
            ),
            inventory_policy_revision=value.get(
                "inventory_policy_revision", "inventory-policy-revision:1"
            ),
        )


def _normalize_repo_path(path: str) -> str:
    normalized = path.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.lstrip("/")


def _is_excluded(path: str, prefixes: Sequence[str]) -> bool:
    normalized = _normalize_repo_path(path)
    return any(
        normalized == prefix.rstrip("/")
        or normalized.startswith(prefix)
        for prefix in prefixes
    )


def freeze_multi_repository_fixture(
    repositories: Mapping[str, Mapping[str, str | bytes]] | None = None,
    *,
    profile: AssuranceRolloutProfile | None = None,
    fixture_id: str | None = None,
    fixture_revision: str | None = None,
    exclusion_prefixes: Sequence[str] | None = None,
    inventory_policy_id: str | None = None,
    inventory_policy_revision: str | None = None,
) -> FrozenMultiRepoFixture:
    """Freeze repository path bodies into reproducible content identities."""

    resolved_profile = profile or build_generic_rollout_profile()
    source = (
        resolved_profile.default_fixture_repositories
        if repositories is None
        else repositories
    )
    if not source:
        raise SymbolicAssuranceRolloutError("repositories must not be empty")
    prefixes = tuple(
        exclusion_prefixes
        if exclusion_prefixes is not None
        else resolved_profile.default_exclusion_prefixes
    )
    fixture_id = fixture_id or resolved_profile.default_fixture_id
    fixture_revision = fixture_revision or resolved_profile.default_fixture_revision
    inventory_policy_id = inventory_policy_id or resolved_profile.inventory_policy_id
    inventory_policy_revision = (
        inventory_policy_revision or resolved_profile.inventory_policy_revision
    )
    descriptors: list[FrozenRepositoryDescriptor] = []
    for repository_id, files in sorted(source.items()):
        if not files:
            raise SymbolicAssuranceRolloutError(
                f"{repository_id} must contain at least one path"
            )
        path_digests: dict[str, str] = {}
        included: list[str] = []
        excluded: list[str] = []
        for path, body in sorted(files.items()):
            rel = _normalize_repo_path(
                _text(path, "path", maximum=1024)
            )
            if not rel:
                raise SymbolicAssuranceRolloutError("path must not be empty")
            if isinstance(body, str):
                raw = body.encode("utf-8")
            elif isinstance(body, (bytes, bytearray)):
                raw = bytes(body)
            else:
                raise SymbolicAssuranceRolloutError(
                    f"path body for {rel!r} must be str or bytes"
                )
            digest = _content_cid(raw)
            path_digests[rel] = digest
            if _is_excluded(rel, prefixes):
                excluded.append(rel)
            else:
                included.append(rel)
        content_cid = _identity(
            {
                "repository_id": repository_id,
                "path_digests": dict(sorted(path_digests.items())),
            }
        )
        tree_id = _identity(
            {
                "repository_id": repository_id,
                "alias": repository_id.rsplit(":", 1)[-1],
                "commit": f"commit:{content_cid[7:23]}",
                "included_paths": sorted(included),
                "excluded_paths": sorted(excluded),
                "path_digests": dict(sorted(path_digests.items())),
            }
        )
        descriptors.append(
            FrozenRepositoryDescriptor(
                repository_id=_text(repository_id, "repository_id"),
                alias=repository_id.rsplit(":", 1)[-1],
                commit=f"commit:{content_cid[7:23]}",
                tree_id=tree_id,
                content_cid=content_cid,
                included_paths=tuple(included),
                excluded_paths=tuple(excluded),
                path_digests=path_digests,
            )
        )
    forest_id = _identity(
        {
            "fixture_id": fixture_id,
            "fixture_revision": fixture_revision,
            "repositories": [item.to_dict() for item in descriptors],
            "exclusion_prefixes": list(sorted(set(prefixes))),
            "inventory_policy_id": inventory_policy_id,
            "inventory_policy_revision": inventory_policy_revision,
        }
    )
    return FrozenMultiRepoFixture(
        fixture_id=fixture_id,
        fixture_revision=fixture_revision,
        forest_id=forest_id,
        repositories=tuple(descriptors),
        exclusion_prefixes=prefixes,
        inventory_policy_id=inventory_policy_id,
        inventory_policy_revision=inventory_policy_revision,
    )


# ---------------------------------------------------------------------------
# Gate observations and reports
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateObservation:
    """One typed gate result over the frozen fixture."""

    gate_id: str
    status: GateStatus | str
    expected_outcome: str
    observed_outcome: str
    evidence_ids: tuple[str, ...] = ()
    detail: str = ""
    authoritative: bool = False
    kind: GateKind | str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "gate_id", _text(self.gate_id, "gate_id"))
        object.__setattr__(self, "status", _status(self.status))
        object.__setattr__(
            self,
            "expected_outcome",
            _text(self.expected_outcome, "expected_outcome", maximum=256),
        )
        object.__setattr__(
            self,
            "observed_outcome",
            _text(self.observed_outcome, "observed_outcome", maximum=256),
        )
        evidence = _unique_sorted_texts(
            self.evidence_ids, "evidence_ids", maximum=MAX_GATE_EVIDENCE_IDS
        )
        object.__setattr__(self, "evidence_ids", evidence)
        if self.detail:
            object.__setattr__(
                self, "detail", _text(self.detail, "detail", maximum=1024)
            )
        object.__setattr__(
            self, "authoritative", _boolean(self.authoritative, "authoritative")
        )
        if self.kind is not None:
            object.__setattr__(self, "kind", _gate_kind(self.kind))
        if self.authoritative:
            raise SymbolicAssuranceRolloutError(
                f"{self.gate_id} observations cannot be authoritative"
            )

    @property
    def passed(self) -> bool:
        return self.status is GateStatus.PASSED

    @property
    def observation_id(self) -> str:
        return _identity(self.to_dict(include_observation_id=False))

    def to_dict(self, *, include_observation_id: bool = True) -> dict[str, Any]:
        payload = {
            "gate_id": self.gate_id,
            "status": self.status.value if isinstance(self.status, GateStatus) else str(self.status),
            "expected_outcome": self.expected_outcome,
            "observed_outcome": self.observed_outcome,
            "evidence_ids": list(self.evidence_ids),
            "detail": self.detail,
            "authoritative": self.authoritative,
            "passed": self.passed,
        }
        if self.kind is not None:
            payload["kind"] = (
                self.kind.value if isinstance(self.kind, GateKind) else str(self.kind)
            )
        if include_observation_id:
            payload["observation_id"] = self.observation_id
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GateObservation":
        result = cls(
            gate_id=value["gate_id"],
            status=value["status"],
            expected_outcome=value["expected_outcome"],
            observed_outcome=value["observed_outcome"],
            evidence_ids=tuple(value.get("evidence_ids", ())),
            detail=value.get("detail", ""),
            authoritative=bool(value.get("authoritative", False)),
            kind=value.get("kind"),
        )
        if value.get("observation_id", result.observation_id) != result.observation_id:
            raise SymbolicAssuranceRolloutError("gate observation ID mismatch")
        return result


@dataclass(frozen=True)
class AdversarialGateReport:
    """Closed population of adversarial gate observations."""

    fixture: FrozenMultiRepoFixture
    observations: tuple[GateObservation, ...]
    observed_at: str
    profile: AssuranceRolloutProfile
    toolchain_id: str = ""
    toolchain_revision: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.fixture, FrozenMultiRepoFixture):
            raise SymbolicAssuranceRolloutError("fixture has the wrong type")
        if not isinstance(self.profile, AssuranceRolloutProfile):
            raise SymbolicAssuranceRolloutError("profile has the wrong type")
        object.__setattr__(
            self, "observed_at", _timestamp(self.observed_at, "observed_at")
        )
        toolchain_id = self.toolchain_id or self.profile.toolchain_id
        toolchain_revision = self.toolchain_revision or self.profile.toolchain_revision
        object.__setattr__(
            self, "toolchain_id", _text(toolchain_id, "toolchain_id")
        )
        object.__setattr__(
            self,
            "toolchain_revision",
            _text(toolchain_revision, "toolchain_revision"),
        )
        observations = tuple(self.observations)
        required = self.profile.required_gate_ids
        if len(observations) != len(required):
            raise SymbolicAssuranceRolloutError(
                "adversarial report must cover every required gate"
            )
        by_id = {item.gate_id: item for item in observations}
        if len(by_id) != len(observations):
            raise SymbolicAssuranceRolloutError("gate observations must be unique")
        missing = [item for item in required if item not in by_id]
        if missing:
            raise SymbolicAssuranceRolloutError(
                f"missing adversarial gates: {', '.join(missing)}"
            )
        ordered = tuple(by_id[item] for item in required)
        object.__setattr__(self, "observations", ordered)

    @property
    def report_id(self) -> str:
        return _identity(self.to_dict(include_report_id=False))

    @property
    def passed(self) -> bool:
        return all(item.passed for item in self.observations)

    @property
    def failure_codes(self) -> tuple[str, ...]:
        return tuple(
            f"gate-failed:{item.gate_id}"
            for item in self.observations
            if not item.passed
        )

    @property
    def automatic_mutation_enabled(self) -> bool:
        return False

    def observation(self, gate_id: str) -> GateObservation:
        selected = _text(gate_id, "gate_id")
        for item in self.observations:
            if item.gate_id == selected:
                return item
        raise SymbolicAssuranceRolloutError(f"gate not present: {selected}")

    def observation_by_kind(self, kind: GateKind | str) -> GateObservation:
        definition = self.profile.gate_by_kind(kind)
        return self.observation(definition.gate_id)

    def to_dict(self, *, include_report_id: bool = True) -> dict[str, Any]:
        schemas = self.profile.schemas
        payload = {
            "schema": schemas.adversarial_e2e_gate,
            "version": schemas.version,
            "requirement_id": self.profile.requirement_id,
            "objective_id": self.profile.objective_id,
            "objective_revision": self.profile.objective_revision,
            "profile_id": self.profile.profile_id,
            "profile_content_id": self.profile.profile_content_id,
            "fixture": self.fixture.to_dict(),
            "fixture_cid": self.fixture.fixture_cid,
            "observations": [item.to_dict() for item in self.observations],
            "observed_at": self.observed_at,
            "toolchain_id": self.toolchain_id,
            "toolchain_revision": self.toolchain_revision,
            "passed": self.passed,
            "failure_codes": list(self.failure_codes),
            "automatic_mutation_enabled": False,
            "authoritative": False,
            "completion_authoritative": False,
        }
        if include_report_id:
            payload["report_id"] = self.report_id
        return payload

    def to_json(self) -> str:
        return _canonical_bytes(self.to_dict()).decode("utf-8")

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        profile: AssuranceRolloutProfile,
    ) -> "AdversarialGateReport":
        if value.get("schema") != profile.schemas.adversarial_e2e_gate:
            raise SymbolicAssuranceRolloutError("unsupported adversarial e2e schema")
        if value.get("automatic_mutation_enabled") is True:
            raise SymbolicAssuranceRolloutError(
                "adversarial e2e report cannot enable automatic mutation"
            )
        report = cls(
            fixture=FrozenMultiRepoFixture.from_dict(value["fixture"]),
            observations=tuple(
                GateObservation.from_dict(item)
                for item in value["observations"]
            ),
            observed_at=value["observed_at"],
            profile=profile,
            toolchain_id=value.get("toolchain_id", profile.toolchain_id),
            toolchain_revision=value.get(
                "toolchain_revision", profile.toolchain_revision
            ),
        )
        if value.get("report_id", report.report_id) != report.report_id:
            raise SymbolicAssuranceRolloutError("adversarial e2e report ID mismatch")
        if value.get("fixture_cid", report.fixture.fixture_cid) != (
            report.fixture.fixture_cid
        ):
            raise SymbolicAssuranceRolloutError("fixture_cid mismatch")
        return report


@dataclass(frozen=True)
class AdversarialInjection:
    """Optional forced failure for one or more gates (negative tests)."""

    failed_kinds: frozenset[GateKind] = field(default_factory=frozenset)
    force_authoritative_zk: bool = False
    force_automatic_mutation: bool = False
    corrupt_second_cid_pass: bool = False
    omit_exclusion: bool = False
    allow_stale_cache_hit: bool = False
    allow_corrupt_cache_hit: bool = False
    wrong_contract_match: bool = False
    accept_wrong_proof: bool = False
    promote_unknown_proof: bool = False
    accept_mcp_mock: bool = False
    accept_mcp_bypass: bool = False
    miss_seeded_drift: bool = False
    emit_vulnerability_false_positive: bool = False
    nondeterministic_tasks: bool = False
    expand_authority_on_provider_loss: bool = False
    restart_diverges: bool = False
    ignore_lease_fence: bool = False
    silent_merge_conflict: bool = False
    unbounded_refill: bool = False
    refill_busywork_after_exhaustion: bool = False
    skip_rollback: bool = False
    control_surface_divergence: bool = False

    def fails(self, kind: GateKind) -> bool:
        return kind in self.failed_kinds


def _obs(
    definition: GateDefinition,
    *,
    passed: bool,
    observed: str,
    evidence: Sequence[str] | None = None,
    detail: str = "",
    reject: bool = False,
) -> GateObservation:
    if passed:
        status = GateStatus.PASSED
    elif reject:
        status = GateStatus.REJECTED
    else:
        status = GateStatus.FAILED
    return GateObservation(
        gate_id=definition.gate_id,
        status=status,
        expected_outcome=definition.expected_outcome,
        observed_outcome=observed,
        evidence_ids=tuple(
            evidence if evidence is not None else definition.evidence_ids
        ),
        detail=detail,
        authoritative=False,
        kind=definition.kind,
    )


def evaluate_adversarial_gates(
    fixture: FrozenMultiRepoFixture,
    *,
    profile: AssuranceRolloutProfile | None = None,
    injection: AdversarialInjection | None = None,
    observed_at: str | datetime = "2026-07-29T00:00:00Z",
    second_fixture: FrozenMultiRepoFixture | None = None,
) -> AdversarialGateReport:
    """Evaluate the closed adversarial population against a frozen fixture."""

    if not isinstance(fixture, FrozenMultiRepoFixture):
        raise SymbolicAssuranceRolloutError("fixture has the wrong type")
    resolved_profile = profile or build_generic_rollout_profile()
    inj = injection or AdversarialInjection()
    if inj.force_automatic_mutation:
        raise SymbolicAssuranceRolloutError(
            "automatic mutation cannot be forced on the adversarial gate"
        )
    if inj.force_authoritative_zk:
        raise SymbolicAssuranceRolloutError(
            "simulated/forged/tampered ZK cannot gain authority"
        )

    if second_fixture is not None:
        replay = second_fixture
    elif fixture.fixture_id == resolved_profile.default_fixture_id:
        replay = freeze_multi_repository_fixture(
            profile=resolved_profile,
            fixture_id=fixture.fixture_id,
            fixture_revision=fixture.fixture_revision,
            exclusion_prefixes=fixture.exclusion_prefixes,
            inventory_policy_id=fixture.inventory_policy_id,
            inventory_policy_revision=fixture.inventory_policy_revision,
        )
    else:
        replay = fixture

    by_kind = {item.kind: item for item in resolved_profile.gates}
    observations: list[GateObservation] = []

    def require(kind: GateKind) -> GateDefinition:
        if kind not in by_kind:
            raise SymbolicAssuranceRolloutError(
                f"profile missing gate kind: {kind.value}"
            )
        return by_kind[kind]

    # reproducible CIDs
    definition = require(GateKind.REPRODUCIBLE_CIDS)
    cid_match = (
        fixture.fixture_cid == replay.fixture_cid
        and all(
            left.content_cid == right.content_cid
            for left, right in zip(
                fixture.repositories, replay.repositories, strict=True
            )
        )
        if len(fixture.repositories) == len(replay.repositories)
        else False
    )
    if inj.corrupt_second_cid_pass:
        cid_match = False
    if inj.fails(GateKind.REPRODUCIBLE_CIDS):
        cid_match = False
    observations.append(
        _obs(
            definition,
            passed=cid_match,
            observed=(
                definition.expected_outcome if cid_match else "cid-mismatch"
            ),
            evidence=(
                f"first:{fixture.fixture_cid}",
                f"second:{replay.fixture_cid}",
            ),
        )
    )

    # complete inventory
    definition = require(GateKind.COMPLETE_INVENTORY)
    exhaustive = all(repo.exhaustive for repo in fixture.repositories)
    omitted = any(not repo.included_paths for repo in fixture.repositories)
    inventory_ok = exhaustive and not omitted and fixture.total_included_paths > 0
    if inj.fails(GateKind.COMPLETE_INVENTORY):
        inventory_ok = False
    observations.append(
        _obs(
            definition,
            passed=inventory_ok,
            observed=(
                definition.expected_outcome
                if inventory_ok
                else "incomplete-inventory"
            ),
            evidence=tuple(repo.content_cid for repo in fixture.repositories),
        )
    )

    # inventory exclusions
    definition = require(GateKind.INVENTORY_EXCLUSIONS)
    exclusion_ok = True
    for repo in fixture.repositories:
        for path in repo.excluded_paths:
            if not _is_excluded(path, fixture.exclusion_prefixes):
                exclusion_ok = False
        for path in repo.included_paths:
            if _is_excluded(path, fixture.exclusion_prefixes):
                exclusion_ok = False
    if inj.omit_exclusion or inj.fails(GateKind.INVENTORY_EXCLUSIONS):
        exclusion_ok = False
    exclusion_passed = exclusion_ok and fixture.total_excluded_paths > 0
    observations.append(
        _obs(
            definition,
            passed=exclusion_passed,
            observed=(
                definition.expected_outcome
                if exclusion_passed
                else "exclusion-policy-violation"
            ),
            evidence=tuple(fixture.exclusion_prefixes),
        )
    )

    # incremental reuse
    definition = require(GateKind.INCREMENTAL_REUSE)
    reuse_ratio = 1.0 if inventory_ok else 0.0
    reuse_ok = reuse_ratio >= 1.0
    if inj.fails(GateKind.INCREMENTAL_REUSE):
        reuse_ok = False
    observations.append(
        _obs(
            definition,
            passed=reuse_ok,
            observed=definition.expected_outcome if reuse_ok else "reuse-miss",
            evidence=(f"reuse-ratio:{reuse_ratio}",),
        )
    )

    # flag-driven rejection/precision gates
    flag_gates: list[tuple[GateKind, bool, str, str, bool]] = [
        (
            GateKind.STALE_CACHE_REJECTION,
            not inj.allow_stale_cache_hit and not inj.fails(GateKind.STALE_CACHE_REJECTION),
            "stale-authoritative-hit-accepted",
            "cache:stale-probe",
            True,
        ),
        (
            GateKind.CORRUPT_CACHE_REJECTION,
            not inj.allow_corrupt_cache_hit
            and not inj.fails(GateKind.CORRUPT_CACHE_REJECTION),
            "corrupt-cache-entry-accepted",
            "cache:corrupt-probe",
            True,
        ),
        (
            GateKind.CONTRACT_PRECISION,
            not inj.wrong_contract_match and not inj.fails(GateKind.CONTRACT_PRECISION),
            "false-proved-compatible",
            "contract:seeded-mismatch",
            False,
        ),
        (
            GateKind.WRONG_PROOF,
            not inj.accept_wrong_proof and not inj.fails(GateKind.WRONG_PROOF),
            "wrong-proof-accepted",
            "proof:wrong",
            True,
        ),
        (
            GateKind.UNKNOWN_PROOF,
            not inj.promote_unknown_proof and not inj.fails(GateKind.UNKNOWN_PROOF),
            "unknown-proof-promoted",
            "proof:unknown",
            False,
        ),
        (
            GateKind.MCP_MOCK,
            not inj.accept_mcp_mock and not inj.fails(GateKind.MCP_MOCK),
            "mcp-mock-treated-as-production",
            "mcp:mock-probe",
            False,
        ),
        (
            GateKind.MCP_BYPASS,
            not inj.accept_mcp_bypass and not inj.fails(GateKind.MCP_BYPASS),
            "mcp-local-bypass-silent",
            "mcp:bypass-probe",
            False,
        ),
        (
            GateKind.SEEDED_DRIFT,
            not inj.miss_seeded_drift and not inj.fails(GateKind.SEEDED_DRIFT),
            "seeded-drift-missed",
            "surface:seeded-drift",
            False,
        ),
        (
            GateKind.VULNERABILITY_FALSE_POSITIVE,
            not inj.emit_vulnerability_false_positive
            and not inj.fails(GateKind.VULNERABILITY_FALSE_POSITIVE),
            "false-positive-emitted-as-vulnerability",
            "security:false-positive-seed",
            False,
        ),
        (
            GateKind.PROVIDER_LOSS,
            not inj.expand_authority_on_provider_loss
            and not inj.fails(GateKind.PROVIDER_LOSS),
            "provider-loss-expanded-authority",
            "provider:loss-probe",
            False,
        ),
        (
            GateKind.RESTART_REPLAY,
            not inj.restart_diverges and not inj.fails(GateKind.RESTART_REPLAY),
            "restart-replay-diverged",
            "runtime:restart-replay",
            False,
        ),
        (
            GateKind.LEASE_FENCE_LOSS,
            not inj.ignore_lease_fence and not inj.fails(GateKind.LEASE_FENCE_LOSS),
            "lease-fence-loss-ignored",
            "lease:fence-loss",
            False,
        ),
        (
            GateKind.MERGE_CONFLICT,
            not inj.silent_merge_conflict and not inj.fails(GateKind.MERGE_CONFLICT),
            "merge-conflict-silent",
            "merge:conflict-probe",
            False,
        ),
        (
            GateKind.BOUNDED_REFILL,
            not inj.unbounded_refill and not inj.fails(GateKind.BOUNDED_REFILL),
            "refill-exceeded-ceilings",
            "refill:bounded",
            False,
        ),
        (
            GateKind.REFILL_EXHAUSTION,
            not inj.refill_busywork_after_exhaustion
            and not inj.fails(GateKind.REFILL_EXHAUSTION),
            "exhaustion-created-busywork",
            "refill:exhaustion",
            False,
        ),
        (
            GateKind.ROLLBACK,
            not inj.skip_rollback and not inj.fails(GateKind.ROLLBACK),
            "regression-retained-elevated-mode",
            "rollout:rollback",
            False,
        ),
        (
            GateKind.CONTROL_PARITY,
            not inj.control_surface_divergence
            and not inj.fails(GateKind.CONTROL_PARITY),
            "control-surface-divergence",
            "control:parity",
            False,
        ),
    ]
    for kind, ok, fail_observed, evidence, reject_on_ok in flag_gates:
        definition = require(kind)
        observations.append(
            _obs(
                definition,
                passed=ok,
                observed=definition.expected_outcome if ok else fail_observed,
                evidence=(evidence,),
                reject=reject_on_ok and ok,
            )
        )

    # ZK gates
    for kind, fail_observed in (
        (GateKind.SIMULATED_ZK, "simulated_zk-authority-leak"),
        (GateKind.FORGED_ZK, "forged_zk-authority-leak"),
        (GateKind.TAMPERED_ZK, "tampered_zk-authority-leak"),
    ):
        definition = require(kind)
        ok = not inj.fails(kind)
        observations.append(
            _obs(
                definition,
                passed=ok,
                observed=definition.expected_outcome if ok else fail_observed,
                evidence=(f"zk:{kind.value}",),
                reject=ok and kind is not GateKind.SIMULATED_ZK,
            )
        )

    # task determinism
    definition = require(GateKind.TASK_DETERMINISM)
    task_ok = not inj.nondeterministic_tasks and not inj.fails(
        GateKind.TASK_DETERMINISM
    )
    task_a = _identity(
        {
            "fixture_cid": fixture.fixture_cid,
            "goal_id": resolved_profile.objective_id,
            "finding": "seed:contract-mismatch",
        }
    )
    task_b = _identity(
        {
            "fixture_cid": fixture.fixture_cid,
            "goal_id": resolved_profile.objective_id,
            "finding": "seed:contract-mismatch",
        }
    )
    if task_ok:
        task_ok = task_a == task_b
    observations.append(
        _obs(
            definition,
            passed=task_ok,
            observed=(
                definition.expected_outcome if task_ok else "task-identity-drift"
            ),
            evidence=(f"task-a:{task_a}", f"task-b:{task_b}"),
        )
    )

    # automatic mutation disabled
    definition = require(GateKind.AUTOMATIC_MUTATION_DISABLED)
    auto_disabled = not inj.fails(GateKind.AUTOMATIC_MUTATION_DISABLED)
    observations.append(
        _obs(
            definition,
            passed=auto_disabled,
            observed=(
                definition.expected_outcome
                if auto_disabled
                else "automatic-mutation-enabled"
            ),
            evidence=("policy:automatic-mutation",),
        )
    )

    # Order observations to match profile gate order
    by_id = {item.gate_id: item for item in observations}
    ordered = tuple(by_id[item.gate_id] for item in resolved_profile.gates)

    return AdversarialGateReport(
        fixture=fixture,
        observations=ordered,
        observed_at=observed_at,
        profile=resolved_profile,
    )


def verify_adversarial_e2e_report(
    report: AdversarialGateReport,
    *,
    injection: AdversarialInjection | None = None,
) -> bool:
    try:
        independent = evaluate_adversarial_gates(
            report.fixture,
            profile=report.profile,
            injection=injection,
            observed_at=report.observed_at,
        )
    except SymbolicAssuranceRolloutError:
        return False
    return _canonical_bytes(report.to_dict()) == _canonical_bytes(
        independent.to_dict()
    )


# ---------------------------------------------------------------------------
# Shadow rollout report and decision
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AssuranceRolloutBinding:
    """Exact current deployment identity for the assurance behavior."""

    repository_id: str
    tree_id: str
    forest_id: str
    behavior_id: str
    objective_id: str
    objective_revision: str
    policy_id: str
    policy_revision: str
    capability_id: str
    capability_revision: str

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(
                self, name, _text(getattr(self, name), name, maximum=512)
            )

    @property
    def binding_id(self) -> str:
        return _identity(self.to_dict())

    def to_dict(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AssuranceRolloutBinding":
        if set(value) != set(cls.__dataclass_fields__):
            raise SymbolicAssuranceRolloutError("invalid rollout binding fields")
        return cls(**{name: value[name] for name in cls.__dataclass_fields__})


@dataclass(frozen=True)
class AssuranceRolloutPolicy:
    """Reviewed promotion policy.  It cannot waive a safety gate."""

    policy_id: str
    policy_revision: str
    approved_behavior_ids: tuple[str, ...]
    approved_modes: tuple[AssuranceRolloutMode | str, ...] = (
        AssuranceRolloutMode.OFF,
        AssuranceRolloutMode.SHADOW,
        AssuranceRolloutMode.ASSIST,
    )
    rollback_on_regression: bool = True
    automatic_mutation_enabled: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "policy_id", _text(self.policy_id, "policy_id")
        )
        object.__setattr__(
            self,
            "policy_revision",
            _text(self.policy_revision, "policy_revision"),
        )
        behaviors = tuple(
            sorted(
                _text(item, "approved_behavior_ids")
                for item in self.approved_behavior_ids
            )
        )
        if not behaviors or len(behaviors) != len(set(behaviors)):
            raise SymbolicAssuranceRolloutError(
                "approved behavior IDs must be unique and non-empty"
            )
        modes = tuple(_mode(item) for item in self.approved_modes)
        if len(modes) != len(set(modes)):
            raise SymbolicAssuranceRolloutError("approved modes must be unique")
        object.__setattr__(self, "approved_behavior_ids", behaviors)
        object.__setattr__(self, "approved_modes", modes)
        object.__setattr__(
            self,
            "rollback_on_regression",
            _boolean(self.rollback_on_regression, "rollback_on_regression"),
        )
        object.__setattr__(
            self,
            "automatic_mutation_enabled",
            _boolean(
                self.automatic_mutation_enabled, "automatic_mutation_enabled"
            ),
        )
        if self.automatic_mutation_enabled:
            raise SymbolicAssuranceRolloutError(
                "automatic mutation remains disabled for symbolic assurance rollout"
            )

    @property
    def policy_binding_id(self) -> str:
        return _identity(self.to_dict())

    def approves(
        self, behavior_id: str, mode: AssuranceRolloutMode | str
    ) -> bool:
        return (
            behavior_id in self.approved_behavior_ids
            and _mode(mode) in self.approved_modes
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "approved_behavior_ids": list(self.approved_behavior_ids),
            "approved_modes": [
                item.value if isinstance(item, AssuranceRolloutMode) else str(item)
                for item in self.approved_modes
            ],
            "rollback_on_regression": self.rollback_on_regression,
            "automatic_mutation_enabled": False,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AssuranceRolloutPolicy":
        return cls(
            policy_id=value["policy_id"],
            policy_revision=value["policy_revision"],
            approved_behavior_ids=tuple(value["approved_behavior_ids"]),
            approved_modes=tuple(value.get("approved_modes", ())),
            rollback_on_regression=bool(
                value.get("rollback_on_regression", True)
            ),
            automatic_mutation_enabled=bool(
                value.get("automatic_mutation_enabled", False)
            ),
        )


@dataclass(frozen=True)
class ShadowRolloutReport:
    """Promotion-facing shadow report bound to adversarial e2e evidence."""

    gate_report: AdversarialGateReport
    binding: AssuranceRolloutBinding
    policy: AssuranceRolloutPolicy
    desired_mode: AssuranceRolloutMode | str = AssuranceRolloutMode.SHADOW
    prior_gate_report: AdversarialGateReport | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.gate_report, AdversarialGateReport):
            raise SymbolicAssuranceRolloutError("gate_report has the wrong type")
        if not isinstance(self.binding, AssuranceRolloutBinding):
            raise SymbolicAssuranceRolloutError("binding has the wrong type")
        if not isinstance(self.policy, AssuranceRolloutPolicy):
            raise SymbolicAssuranceRolloutError("policy has the wrong type")
        object.__setattr__(self, "desired_mode", _mode(self.desired_mode))
        if self.prior_gate_report is not None and not isinstance(
            self.prior_gate_report, AdversarialGateReport
        ):
            raise SymbolicAssuranceRolloutError(
                "prior_gate_report has the wrong type"
            )

    @property
    def profile(self) -> AssuranceRolloutProfile:
        return self.gate_report.profile

    @property
    def report_id(self) -> str:
        return _identity(self.to_dict(include_report_id=False))

    @property
    def reason_codes(self) -> tuple[str, ...]:
        reasons: list[str] = []
        report = self.gate_report
        binding = self.binding
        profile = self.profile
        if binding.behavior_id != profile.behavior_id:
            reasons.append("stale-binding:behavior_id")
        if binding.forest_id != report.fixture.forest_id:
            reasons.append("stale-binding:forest_id")
        if binding.objective_id != profile.objective_id:
            reasons.append("stale-binding:objective_id")
        if binding.objective_revision != profile.objective_revision:
            reasons.append("stale-binding:objective_revision")
        if (
            self.policy.policy_id != binding.policy_id
            or self.policy.policy_revision != binding.policy_revision
        ):
            reasons.append("stale-binding:rollout-policy")
        if not report.passed:
            reasons.extend(report.failure_codes)
        if self.desired_mode in {
            AssuranceRolloutMode.ASSIST,
            AssuranceRolloutMode.AUTOMATIC,
        } and not self.policy.approves(
            binding.behavior_id, self.desired_mode
        ):
            reasons.append("mode-not-policy-approved")
        if self.desired_mode is AssuranceRolloutMode.AUTOMATIC:
            reasons.append("automatic-mutation-disabled")
        if self.prior_gate_report is not None:
            if self.prior_gate_report.fixture.forest_id != report.fixture.forest_id:
                reasons.append("fixture-population-changed")
            if self.policy.rollback_on_regression:
                prior_failures = set(self.prior_gate_report.failure_codes)
                current_failures = set(report.failure_codes)
                if current_failures - prior_failures:
                    reasons.append("assurance-regression")
                if self.prior_gate_report.passed and not report.passed:
                    reasons.append("assurance-regression")
                if _datetime(report.observed_at) < _datetime(
                    self.prior_gate_report.observed_at
                ):
                    reasons.append("current-observation-not-later")
        return tuple(sorted(set(reasons))[:MAX_REASON_CODES])

    @property
    def qualification_passed(self) -> bool:
        return not any(
            code.startswith("gate-failed:")
            or code.startswith("stale-binding:")
            for code in self.reason_codes
        ) and self.gate_report.passed

    @property
    def effective_mode(self) -> AssuranceRolloutMode:
        desired = self.desired_mode
        reasons = self.reason_codes
        if desired is AssuranceRolloutMode.OFF:
            return AssuranceRolloutMode.OFF
        if desired is AssuranceRolloutMode.SHADOW:
            return AssuranceRolloutMode.SHADOW
        if desired is AssuranceRolloutMode.ASSIST:
            if self.gate_report.passed and not reasons:
                return AssuranceRolloutMode.ASSIST
            return AssuranceRolloutMode.SHADOW
        return AssuranceRolloutMode.SHADOW

    @property
    def rollback_applied(self) -> bool:
        return self.effective_mode is AssuranceRolloutMode.SHADOW and self.desired_mode in {
            AssuranceRolloutMode.ASSIST,
            AssuranceRolloutMode.AUTOMATIC,
        }

    @property
    def automatic_ready(self) -> bool:
        return False

    @property
    def automatic_mutation_enabled(self) -> bool:
        return False

    @property
    def passed(self) -> bool:
        return self.gate_report.passed and not any(
            code.startswith("assurance-regression")
            or code.startswith("stale-binding:")
            for code in self.reason_codes
        )

    def to_dict(self, *, include_report_id: bool = True) -> dict[str, Any]:
        profile = self.profile
        schemas = profile.schemas
        payload = {
            "schema": schemas.shadow_rollout_report,
            "version": schemas.version,
            "requirement_id": profile.requirement_id,
            "behavior_id": profile.behavior_id,
            "objective_id": profile.objective_id,
            "objective_revision": profile.objective_revision,
            "binding": self.binding.to_dict(),
            "binding_id": self.binding.binding_id,
            "policy": self.policy.to_dict(),
            "policy_binding_id": self.policy.policy_binding_id,
            "gate_report_id": self.gate_report.report_id,
            "fixture_cid": self.gate_report.fixture.fixture_cid,
            "desired_mode": (
                self.desired_mode.value
                if isinstance(self.desired_mode, AssuranceRolloutMode)
                else str(self.desired_mode)
            ),
            "effective_mode": self.effective_mode.value,
            "reason_codes": list(self.reason_codes),
            "qualification_passed": self.qualification_passed,
            "passed": self.passed,
            "rollback_applied": self.rollback_applied,
            "automatic_ready": False,
            "automatic_mutation_enabled": False,
            "prior_gate_report_id": (
                self.prior_gate_report.report_id
                if self.prior_gate_report is not None
                else ""
            ),
            "authoritative": False,
            "completion_authoritative": False,
            "gate_failure_codes": list(self.gate_report.failure_codes),
        }
        if include_report_id:
            payload["report_id"] = self.report_id
        return payload

    def to_json(self) -> str:
        return _canonical_bytes(self.to_dict()).decode("utf-8")


@dataclass(frozen=True)
class AssuranceRolloutDecision:
    """Desired/effective mode with exact evidence and rollback reasons."""

    binding: AssuranceRolloutBinding
    policy: AssuranceRolloutPolicy
    gate_report: AdversarialGateReport
    desired_mode: AssuranceRolloutMode
    effective_mode: AssuranceRolloutMode
    reason_codes: tuple[str, ...]
    qualification_passed: bool
    rollback_applied: bool
    shadow_report_id: str
    prior_gate_report_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "desired_mode", _mode(self.desired_mode))
        object.__setattr__(self, "effective_mode", _mode(self.effective_mode))
        reasons = tuple(sorted(set(self.reason_codes)))
        if len(reasons) > MAX_REASON_CODES:
            raise SymbolicAssuranceRolloutError("too many reason codes")
        object.__setattr__(self, "reason_codes", reasons)
        if self.desired_mode is AssuranceRolloutMode.OFF:
            if self.effective_mode is not AssuranceRolloutMode.OFF:
                raise SymbolicAssuranceRolloutError("off cannot gain authority")
        elif self.desired_mode is AssuranceRolloutMode.SHADOW:
            if self.effective_mode is not AssuranceRolloutMode.SHADOW:
                raise SymbolicAssuranceRolloutError("shadow cannot gain authority")
        elif self.effective_mode not in {
            self.desired_mode,
            AssuranceRolloutMode.SHADOW,
        }:
            raise SymbolicAssuranceRolloutError(
                "failed promotion must return to shadow"
            )
        if self.effective_mode is AssuranceRolloutMode.AUTOMATIC:
            raise SymbolicAssuranceRolloutError(
                "automatic mode cannot become effective while mutation is disabled"
            )

    @property
    def profile(self) -> AssuranceRolloutProfile:
        return self.gate_report.profile

    @property
    def decision_id(self) -> str:
        return _identity(self.to_dict(include_decision_id=False))

    @property
    def affected_behavior_ids(self) -> tuple[str, ...]:
        return (self.binding.behavior_id,)

    @property
    def authoritative(self) -> bool:
        return False

    @property
    def completion_authoritative(self) -> bool:
        return False

    @property
    def automatic_mutation_enabled(self) -> bool:
        return False

    def explain(self) -> str:
        if self.effective_mode is self.desired_mode and not self.reason_codes:
            return (
                f"{self.binding.behavior_id} is {self.effective_mode.value}; "
                "all gates required for that mode passed."
            )
        if self.rollback_applied:
            return (
                f"{self.binding.behavior_id} returned to shadow: "
                + ", ".join(self.reason_codes)
            )
        return (
            f"{self.binding.behavior_id} effective={self.effective_mode.value}; "
            f"desired={self.desired_mode.value}; "
            + (
                ", ".join(self.reason_codes)
                if self.reason_codes
                else "no blocking reasons"
            )
        )

    def to_dict(self, *, include_decision_id: bool = True) -> dict[str, Any]:
        profile = self.profile
        schemas = profile.schemas
        payload = {
            "schema": schemas.rollout_decision,
            "version": schemas.version,
            "requirement_id": profile.requirement_id,
            "binding": self.binding.to_dict(),
            "binding_id": self.binding.binding_id,
            "policy": self.policy.to_dict(),
            "policy_binding_id": self.policy.policy_binding_id,
            "gate_report_id": self.gate_report.report_id,
            "fixture_cid": self.gate_report.fixture.fixture_cid,
            "desired_mode": self.desired_mode.value,
            "effective_mode": self.effective_mode.value,
            "reason_codes": list(self.reason_codes),
            "qualification_passed": self.qualification_passed,
            "rollback_applied": self.rollback_applied,
            "automatic_ready": False,
            "automatic_mutation_enabled": False,
            "shadow_report_id": self.shadow_report_id,
            "prior_gate_report_id": self.prior_gate_report_id,
            "affected_behavior_ids": list(self.affected_behavior_ids),
            "explanation": self.explain(),
            "authoritative": False,
            "completion_authoritative": False,
        }
        if include_decision_id:
            payload["decision_id"] = self.decision_id
        return payload

    def to_json(self) -> str:
        return _canonical_bytes(self.to_dict()).decode("utf-8")


def evaluate_symbolic_assurance_rollout(
    gate_report: AdversarialGateReport,
    *,
    binding: AssuranceRolloutBinding,
    policy: AssuranceRolloutPolicy,
    desired_mode: AssuranceRolloutMode | str = AssuranceRolloutMode.SHADOW,
    prior_gate_report: AdversarialGateReport | None = None,
) -> AssuranceRolloutDecision:
    """Recompute shadow gates and derive a non-authoritative rollout decision."""

    shadow = ShadowRolloutReport(
        gate_report=gate_report,
        binding=binding,
        policy=policy,
        desired_mode=desired_mode,
        prior_gate_report=prior_gate_report,
    )
    return AssuranceRolloutDecision(
        binding=binding,
        policy=policy,
        gate_report=gate_report,
        desired_mode=shadow.desired_mode,
        effective_mode=shadow.effective_mode,
        reason_codes=shadow.reason_codes,
        qualification_passed=shadow.qualification_passed,
        rollback_applied=shadow.rollback_applied,
        shadow_report_id=shadow.report_id,
        prior_gate_report_id=(
            prior_gate_report.report_id if prior_gate_report is not None else ""
        ),
    )


def verify_symbolic_assurance_rollout(
    decision: AssuranceRolloutDecision,
    gate_report: AdversarialGateReport,
    *,
    binding: AssuranceRolloutBinding,
    policy: AssuranceRolloutPolicy,
    prior_gate_report: AdversarialGateReport | None = None,
) -> bool:
    try:
        replayed = evaluate_symbolic_assurance_rollout(
            gate_report,
            binding=binding,
            policy=policy,
            desired_mode=decision.desired_mode,
            prior_gate_report=prior_gate_report,
        )
    except SymbolicAssuranceRolloutError:
        return False
    return _canonical_bytes(decision.to_dict()) == _canonical_bytes(
        replayed.to_dict()
    )


def build_default_rollout_binding(
    fixture: FrozenMultiRepoFixture,
    *,
    profile: AssuranceRolloutProfile | None = None,
    repository_id: str | None = None,
) -> AssuranceRolloutBinding:
    resolved = profile or build_generic_rollout_profile()
    selected = repository_id or fixture.repository_ids[0]
    return AssuranceRolloutBinding(
        repository_id=selected,
        tree_id=next(
            item.tree_id
            for item in fixture.repositories
            if item.repository_id == selected
        ),
        forest_id=fixture.forest_id,
        behavior_id=resolved.behavior_id,
        objective_id=resolved.objective_id,
        objective_revision=resolved.objective_revision,
        policy_id=resolved.policy_id,
        policy_revision=resolved.policy_revision,
        capability_id=resolved.capability_id,
        capability_revision=resolved.capability_revision,
    )


def build_default_rollout_policy(
    *,
    profile: AssuranceRolloutProfile | None = None,
    approve_assist: bool = True,
    approve_automatic: bool = False,
) -> AssuranceRolloutPolicy:
    resolved = profile or build_generic_rollout_profile()
    modes: list[AssuranceRolloutMode] = [
        AssuranceRolloutMode.OFF,
        AssuranceRolloutMode.SHADOW,
    ]
    if approve_assist:
        modes.append(AssuranceRolloutMode.ASSIST)
    if approve_automatic:
        modes.append(AssuranceRolloutMode.AUTOMATIC)
    return AssuranceRolloutPolicy(
        policy_id=resolved.policy_id,
        policy_revision=resolved.policy_revision,
        approved_behavior_ids=(resolved.behavior_id,),
        approved_modes=tuple(modes),
        automatic_mutation_enabled=False,
    )


# ---------------------------------------------------------------------------
# Bounded publications and control surfaces
# ---------------------------------------------------------------------------


def project_bounded_status(decision: AssuranceRolloutDecision) -> dict[str, Any]:
    """Bounded status projection shared by Python, CLI, and MCP."""

    schemas = decision.profile.schemas
    payload = {
        "schema": schemas.bounded_status,
        "version": schemas.version,
        "requirement_id": decision.profile.requirement_id,
        "behavior_id": decision.binding.behavior_id,
        "desired_mode": decision.desired_mode.value,
        "effective_mode": decision.effective_mode.value,
        "decision_id": decision.decision_id,
        "shadow_report_id": decision.shadow_report_id,
        "gate_report_id": decision.gate_report.report_id,
        "fixture_cid": decision.gate_report.fixture.fixture_cid,
        "qualification_passed": decision.qualification_passed,
        "rollback_applied": decision.rollback_applied,
        "automatic_mutation_enabled": False,
        "reason_codes": list(decision.reason_codes),
        "passed_gate_count": sum(
            1 for item in decision.gate_report.observations if item.passed
        ),
        "failed_gate_count": sum(
            1 for item in decision.gate_report.observations if not item.passed
        ),
        "authoritative": False,
    }
    encoded = _canonical_bytes(payload)
    if len(encoded) > MAX_BOUNDED_BYTES:
        raise SymbolicAssuranceRolloutError("bounded status exceeds size limit")
    payload["content_id"] = _identity(payload)
    return payload


def project_bounded_findings(decision: AssuranceRolloutDecision) -> dict[str, Any]:
    """Bounded findings projection (gate failures only; no source bodies)."""

    findings = []
    for item in decision.gate_report.observations:
        if item.passed:
            continue
        findings.append(
            {
                "finding_id": item.observation_id,
                "gate_id": item.gate_id,
                "status": item.status.value if isinstance(item.status, GateStatus) else str(item.status),
                "expected_outcome": item.expected_outcome,
                "observed_outcome": item.observed_outcome,
                "evidence_ids": list(item.evidence_ids),
            }
        )
        if len(findings) >= MAX_FINDING_PROJECTIONS:
            break
    schemas = decision.profile.schemas
    payload = {
        "schema": schemas.bounded_findings,
        "version": schemas.version,
        "requirement_id": decision.profile.requirement_id,
        "decision_id": decision.decision_id,
        "fixture_cid": decision.gate_report.fixture.fixture_cid,
        "findings": findings,
        "finding_count": len(findings),
        "authoritative": False,
    }
    encoded = _canonical_bytes(payload)
    if len(encoded) > MAX_BOUNDED_BYTES:
        raise SymbolicAssuranceRolloutError("bounded findings exceed size limit")
    payload["content_id"] = _identity(
        {key: value for key, value in payload.items() if key != "content_id"}
    )
    return payload


def project_bounded_receipts(decision: AssuranceRolloutDecision) -> dict[str, Any]:
    """Bounded receipt projection for inventory, gates, and rollout decision."""

    receipts = [
        {
            "receipt_kind": "fixture",
            "receipt_id": decision.gate_report.fixture.fixture_cid,
        },
        {
            "receipt_kind": "adversarial-e2e-gate",
            "receipt_id": decision.gate_report.report_id,
        },
        {
            "receipt_kind": "shadow-rollout",
            "receipt_id": decision.shadow_report_id,
        },
        {
            "receipt_kind": "rollout-decision",
            "receipt_id": decision.decision_id,
        },
    ]
    for repo in decision.gate_report.fixture.repositories:
        receipts.append(
            {
                "receipt_kind": "repository",
                "receipt_id": repo.content_cid,
                "repository_id": repo.repository_id,
            }
        )
        if len(receipts) >= MAX_RECEIPT_PROJECTIONS:
            break
    schemas = decision.profile.schemas
    payload = {
        "schema": schemas.bounded_receipts,
        "version": schemas.version,
        "requirement_id": decision.profile.requirement_id,
        "decision_id": decision.decision_id,
        "receipts": receipts[:MAX_RECEIPT_PROJECTIONS],
        "receipt_count": min(len(receipts), MAX_RECEIPT_PROJECTIONS),
        "authoritative": False,
    }
    encoded = _canonical_bytes(payload)
    if len(encoded) > MAX_BOUNDED_BYTES:
        raise SymbolicAssuranceRolloutError("bounded receipts exceed size limit")
    payload["content_id"] = _identity(
        {key: value for key, value in payload.items() if key != "content_id"}
    )
    return payload


@dataclass(frozen=True)
class ControlRequest:
    action: ControlAction | str
    expected_binding_id: str = ""
    expected_decision_id: str = ""
    schema: str = DEFAULT_CONTROL_REQUEST_SCHEMA
    version: int = ASSURANCE_ROLLOUT_VERSION

    def __post_init__(self) -> None:
        try:
            selected = (
                self.action
                if isinstance(self.action, ControlAction)
                else ControlAction(str(self.action))
            )
        except ValueError as exc:
            raise SymbolicAssuranceRolloutError("unknown control action") from exc
        object.__setattr__(self, "action", selected)
        if self.expected_binding_id:
            object.__setattr__(
                self,
                "expected_binding_id",
                _text(self.expected_binding_id, "expected_binding_id"),
            )
        if self.expected_decision_id:
            object.__setattr__(
                self,
                "expected_decision_id",
                _text(self.expected_decision_id, "expected_decision_id"),
            )
        object.__setattr__(
            self, "schema", _text(self.schema, "schema", maximum=256)
        )
        object.__setattr__(
            self, "version", _non_negative_int(self.version, "version", maximum=1024)
        )

    @property
    def request_id(self) -> str:
        return _identity(self.to_dict(include_request_id=False))

    def to_dict(self, *, include_request_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "version": self.version,
            "action": self.action.value if isinstance(self.action, ControlAction) else str(self.action),
            "expected_binding_id": self.expected_binding_id,
            "expected_decision_id": self.expected_decision_id,
        }
        if include_request_id:
            payload["request_id"] = self.request_id
        return payload

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        profile: AssuranceRolloutProfile | None = None,
    ) -> "ControlRequest":
        allowed = {
            "schema",
            "version",
            "action",
            "expected_binding_id",
            "expected_decision_id",
            "request_id",
        }
        required = {
            "schema",
            "version",
            "action",
            "expected_binding_id",
            "expected_decision_id",
        }
        if set(value).difference(allowed) or not required.issubset(value):
            raise SymbolicAssuranceRolloutError("unknown control request fields")
        expected_schema = (
            profile.schemas.control_request
            if profile is not None
            else DEFAULT_CONTROL_REQUEST_SCHEMA
        )
        expected_version = (
            profile.schemas.version
            if profile is not None
            else ASSURANCE_ROLLOUT_VERSION
        )
        if (
            value.get("schema") != expected_schema
            or value.get("version") != expected_version
        ):
            raise SymbolicAssuranceRolloutError("unsupported control request")
        result = cls(
            action=value["action"],
            expected_binding_id=value.get("expected_binding_id", ""),
            expected_decision_id=value.get("expected_decision_id", ""),
            schema=value["schema"],
            version=int(value["version"]),
        )
        if value.get("request_id", result.request_id) != result.request_id:
            raise SymbolicAssuranceRolloutError("control request ID mismatch")
        return result

    @classmethod
    def from_json(
        cls,
        value: str | bytes | bytearray,
        *,
        profile: AssuranceRolloutProfile | None = None,
    ) -> "ControlRequest":
        return cls.from_dict(
            _load_json(value, "symbolic control request"), profile=profile
        )


@dataclass(frozen=True)
class ControlResult:
    request_id: str
    action: ControlAction
    decision: AssuranceRolloutDecision
    changed: bool
    explanation: str
    status: Mapping[str, Any]
    findings: Mapping[str, Any]
    receipts: Mapping[str, Any]

    @property
    def result_id(self) -> str:
        return _identity(self.to_dict(include_result_id=False))

    def to_dict(self, *, include_result_id: bool = True) -> dict[str, Any]:
        schemas = self.decision.profile.schemas
        payload = {
            "schema": schemas.control_result,
            "version": schemas.version,
            "request_id": self.request_id,
            "action": self.action.value if isinstance(self.action, ControlAction) else str(self.action),
            "decision": self.decision.to_dict(),
            "changed": self.changed,
            "explanation": self.explanation,
            "status": dict(self.status),
            "findings": dict(self.findings),
            "receipts": dict(self.receipts),
        }
        if include_result_id:
            payload["result_id"] = self.result_id
        return payload


class SymbolicAssurancePublicAPI:
    """One canonical stateful control service used by all three surfaces."""

    def __init__(
        self,
        gate_report: AdversarialGateReport,
        *,
        binding: AssuranceRolloutBinding,
        policy: AssuranceRolloutPolicy,
        prior_gate_report: AdversarialGateReport | None = None,
        initial_mode: AssuranceRolloutMode | str | None = None,
    ) -> None:
        self.gate_report = gate_report
        self.binding = binding
        self.policy = policy
        self.prior_gate_report = prior_gate_report
        self._lock = RLock()
        mode = (
            initial_mode
            if initial_mode is not None
            else gate_report.profile.default_mode
        )
        self._decision = evaluate_symbolic_assurance_rollout(
            gate_report,
            binding=binding,
            policy=policy,
            desired_mode=mode,
            prior_gate_report=prior_gate_report,
        )

    @staticmethod
    def discovery(
        profile: AssuranceRolloutProfile | None = None,
    ) -> dict[str, Any]:
        """Static discovery; does not construct providers or inspect the host."""

        resolved = profile or build_generic_rollout_profile()
        schemas = resolved.schemas
        return {
            "schema": schemas.public_api,
            "version": schemas.version,
            "requirement_id": resolved.requirement_id,
            "behavior_id": resolved.behavior_id,
            "objective_id": resolved.objective_id,
            "profile_id": resolved.profile_id,
            "evidence_schemas": [
                schemas.adversarial_e2e_gate,
                schemas.shadow_rollout_report,
            ],
            "surfaces": [item.value for item in ControlSurface],
            "actions": [item.value for item in ControlAction],
            "modes": [item.value for item in AssuranceRolloutMode],
            "required_gates": list(resolved.required_gate_ids),
            "automatic_mutation_enabled": False,
            "optional_providers_loaded": False,
            "processes_started": False,
        }

    @property
    def decision(self) -> AssuranceRolloutDecision:
        with self._lock:
            return self._decision

    def _decode(
        self, request: ControlRequest | Mapping[str, Any] | str
    ) -> ControlRequest:
        if isinstance(request, ControlRequest):
            return request
        if isinstance(request, str):
            return ControlRequest(
                action=request,
                schema=self.gate_report.profile.schemas.control_request,
                version=self.gate_report.profile.schemas.version,
            )
        if isinstance(request, Mapping):
            return ControlRequest.from_dict(
                request, profile=self.gate_report.profile
            )
        raise SymbolicAssuranceRolloutError("invalid control request")

    def _publications(
        self, decision: AssuranceRolloutDecision
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        return (
            project_bounded_status(decision),
            project_bounded_findings(decision),
            project_bounded_receipts(decision),
        )

    def execute(
        self, request: ControlRequest | Mapping[str, Any] | str
    ) -> ControlResult:
        selected = self._decode(request)
        with self._lock:
            previous = self._decision
            self._decision = evaluate_symbolic_assurance_rollout(
                self.gate_report,
                binding=self.binding,
                policy=self.policy,
                desired_mode=previous.desired_mode,
                prior_gate_report=self.prior_gate_report,
            )
            if (
                selected.expected_binding_id
                and selected.expected_binding_id != self.binding.binding_id
            ):
                raise SymbolicAssuranceRolloutError("stale control binding")
            if (
                selected.expected_decision_id
                and selected.expected_decision_id != self._decision.decision_id
            ):
                raise SymbolicAssuranceRolloutError("stale control decision")
            mode = selected.action.requested_mode
            if selected.action is ControlAction.ROLLBACK:
                mode = AssuranceRolloutMode.SHADOW
            if mode is not None:
                self._decision = evaluate_symbolic_assurance_rollout(
                    self.gate_report,
                    binding=self.binding,
                    policy=self.policy,
                    desired_mode=mode,
                    prior_gate_report=self.prior_gate_report,
                )
            decision = self._decision
            status, findings, receipts = self._publications(decision)
            if selected.action is ControlAction.FINDINGS:
                explanation = (
                    f"findings={findings['finding_count']}; "
                    f"effective={decision.effective_mode.value}"
                )
            elif selected.action is ControlAction.RECEIPTS:
                explanation = (
                    f"receipts={receipts['receipt_count']}; "
                    f"fixture_cid={decision.gate_report.fixture.fixture_cid}"
                )
            elif selected.action in {
                ControlAction.EXPLANATION,
                ControlAction.ROLLBACK,
            }:
                explanation = decision.explain()
            else:
                explanation = (
                    f"desired={decision.desired_mode.value}; "
                    f"effective={decision.effective_mode.value}"
                )
            return ControlResult(
                request_id=selected.request_id,
                action=selected.action,
                decision=decision,
                changed=decision.decision_id != previous.decision_id,
                explanation=explanation,
                status=status,
                findings=findings,
                receipts=receipts,
            )

    python = execute
    cli = execute
    mcp = execute

    def status(self) -> ControlResult:
        return self.execute("status")

    def findings(self) -> ControlResult:
        return self.execute("findings")

    def receipts(self) -> ControlResult:
        return self.execute("receipts")

    def explanation(self) -> ControlResult:
        return self.execute("explanation")

    def rollback(self) -> ControlResult:
        return self.execute("rollback")


def build_frozen_adversarial_population(
    *,
    profile: AssuranceRolloutProfile | None = None,
    observed_at: str = "2026-07-29T00:00:00Z",
    injection: AdversarialInjection | None = None,
) -> tuple[
    FrozenMultiRepoFixture,
    AdversarialGateReport,
    AssuranceRolloutBinding,
    AssuranceRolloutPolicy,
]:
    """Convenience builder for a frozen multi-repo population."""

    resolved = profile or build_generic_rollout_profile()
    fixture = freeze_multi_repository_fixture(profile=resolved)
    report = evaluate_adversarial_gates(
        fixture,
        profile=resolved,
        injection=injection,
        observed_at=observed_at,
    )
    binding = build_default_rollout_binding(fixture, profile=resolved)
    policy = build_default_rollout_policy(
        profile=resolved, approve_assist=True, approve_automatic=False
    )
    return fixture, report, binding, policy


def run_symbolic_assurance_e2e(
    *,
    profile: AssuranceRolloutProfile | None = None,
    desired_mode: AssuranceRolloutMode | str = AssuranceRolloutMode.SHADOW,
    injection: AdversarialInjection | None = None,
    observed_at: str = "2026-07-29T00:00:00Z",
) -> dict[str, Any]:
    """Run the full adversarial population and return bounded publications."""

    resolved = profile or build_generic_rollout_profile()
    fixture, report, binding, policy = build_frozen_adversarial_population(
        profile=resolved,
        observed_at=observed_at,
        injection=injection,
    )
    decision = evaluate_symbolic_assurance_rollout(
        report,
        binding=binding,
        policy=policy,
        desired_mode=desired_mode,
    )
    shadow = ShadowRolloutReport(
        gate_report=report,
        binding=binding,
        policy=policy,
        desired_mode=desired_mode,
    )
    return {
        "fixture": fixture.to_dict(),
        "adversarial_e2e_gate": report.to_dict(),
        "shadow_rollout_report": shadow.to_dict(),
        "decision": decision.to_dict(),
        "status": project_bounded_status(decision),
        "findings": project_bounded_findings(decision),
        "receipts": project_bounded_receipts(decision),
        "automatic_mutation_enabled": False,
    }


__all__ = (
    "ASSURANCE_ROLLOUT_VERSION",
    "DEFAULT_ADVERSARIAL_E2E_GATE_SCHEMA",
    "DEFAULT_BOUNDED_FINDINGS_SCHEMA",
    "DEFAULT_BOUNDED_RECEIPTS_SCHEMA",
    "DEFAULT_BOUNDED_STATUS_SCHEMA",
    "DEFAULT_CONTROL_REQUEST_SCHEMA",
    "DEFAULT_CONTROL_RESULT_SCHEMA",
    "DEFAULT_PUBLIC_API_SCHEMA",
    "DEFAULT_ROLLOUT_DECISION_SCHEMA",
    "DEFAULT_SHADOW_ROLLOUT_REPORT_SCHEMA",
    "AdversarialGateReport",
    "AdversarialInjection",
    "AssuranceRolloutBinding",
    "AssuranceRolloutDecision",
    "AssuranceRolloutMode",
    "AssuranceRolloutPolicy",
    "AssuranceRolloutProfile",
    "AssuranceRolloutSchemas",
    "ControlAction",
    "ControlRequest",
    "ControlResult",
    "ControlSurface",
    "FrozenMultiRepoFixture",
    "FrozenRepositoryDescriptor",
    "GateDefinition",
    "GateKind",
    "GateObservation",
    "GateStatus",
    "ShadowRolloutReport",
    "SymbolicAssurancePublicAPI",
    "SymbolicAssuranceRolloutError",
    "build_default_rollout_binding",
    "build_default_rollout_policy",
    "build_frozen_adversarial_population",
    "build_generic_rollout_profile",
    "evaluate_adversarial_gates",
    "evaluate_symbolic_assurance_rollout",
    "freeze_multi_repository_fixture",
    "project_bounded_findings",
    "project_bounded_receipts",
    "project_bounded_status",
    "run_symbolic_assurance_e2e",
    "verify_adversarial_e2e_report",
    "verify_symbolic_assurance_rollout",
)
