"""Closed deterministic harness records for the semantic-compression loop.

These types are admission/reference metadata and MCP++ payload bodies. They
never recompute datasets symbol, capsule, Merkle, or selection facts. Durable
collections are sorted; unknown fields and enums fail closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

HARNESS_CONTRACTS_SCHEMA = "semantic-state-harness@1"
HARNESS_ROOT_MANIFEST_SCHEMA = "ipfs-accelerate.semantic-state-root-manifest@1"
BOARD_NAMESPACE = "semantic-compression-harness-v1"

_CID_ALPHABET = frozenset("abcdefghijklmnopqrstuvwxyz234567")


class HarnessError(ValueError):
    """Closed-record or wire-contract violation."""


class HarnessMode(str, Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"


class WorkKind(str, Enum):
    TASK_PARSING = "task_parsing"
    SCAN = "scan"
    CAPSULE_COMPILATION = "capsule_compilation"
    TEST_SELECTION = "test_selection"
    CONTEXT_PACKING = "context_packing"
    MODEL_INVOCATION = "model_invocation"
    STATIC_CHECK = "static_check"
    PYTEST = "pytest"
    PROVER = "prover"
    PERSISTENCE = "persistence"


class Availability(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class ModelRoute(str, Enum):
    DETERMINISTIC_ONLY = "deterministic_only"
    SMALL_LOCAL_MODEL = "small_local_model"
    MEDIUM_MODEL = "medium_model"
    FRONTIER_MODEL = "frontier_model"
    HUMAN_REVIEW_REQUIRED = "human_review_required"


class HarnessDisposition(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    UNAVAILABLE = "unavailable"


class AcceptanceDisposition(str, Enum):
    BOOTSTRAP = "bootstrap"
    ACCEPTED = "accepted"
    CANDIDATE = "candidate"
    REJECTED = "rejected"


def _text(value: Any, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise HarnessError(f"{name} must be a nonempty trimmed string")
    if any(not char.isprintable() for char in value):
        raise HarnessError(f"{name} contains non-printable characters")
    return value


def _optional_text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name)


def _enum(value: Any, enum_type: type[Enum], name: str) -> str:
    try:
        return enum_type(value).value
    except (TypeError, ValueError) as exc:
        raise HarnessError(f"{name} has unsupported value {value!r}") from exc


def validate_opaque_cid(value: Any, name: str) -> str:
    """Accept a Kubo-compatible CIDv1 base32 string as an opaque reference."""

    text = _text(value, name)
    if not text.startswith("b") or len(text) < 50 or len(text) > 128:
        raise HarnessError(f"{name} is not a CIDv1 reference")
    if any(char not in _CID_ALPHABET for char in text):
        raise HarnessError(f"{name} is not a lowercase base32 CID")
    if text.startswith("cidv1-sha256-") or text.startswith("sim:") or text.startswith("degraded:"):
        raise HarnessError(f"{name} is a forged or non-Kubo CID")
    return text


def _optional_cid(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return validate_opaque_cid(value, name)


def _closed(data: Mapping[str, Any], fields: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(data, Mapping):
        raise HarnessError(f"{name} must be an object")
    actual = set(data)
    if actual != fields:
        raise HarnessError(
            f"{name} fields must be exactly {sorted(fields)}, got {sorted(actual)}"
        )
    return dict(data)


def _unique_sorted_cids(values: Any, name: str) -> tuple[str, ...]:
    if not isinstance(values, list):
        raise HarnessError(f"{name} must be a list")
    ordered = tuple(sorted(validate_opaque_cid(item, name) for item in values))
    if len(ordered) != len(set(ordered)):
        raise HarnessError(f"{name} must not contain duplicates")
    return ordered


def _unique_sorted_texts(values: Any, name: str) -> tuple[str, ...]:
    if not isinstance(values, list):
        raise HarnessError(f"{name} must be a list")
    ordered = tuple(sorted(_text(item, name) for item in values))
    if len(ordered) != len(set(ordered)):
        raise HarnessError(f"{name} must not contain duplicates")
    return ordered


def _nonneg_int(value: Any, name: str) -> int:
    if type(value) is not int or isinstance(value, bool) or value < 0:
        raise HarnessError(f"{name} must be a nonnegative integer")
    return value


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise HarnessError(f"{name} must be a boolean")
    return value


def _string_int_map(value: Any, name: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise HarnessError(f"{name} must be an object")
    result: dict[str, int] = {}
    for key, item in value.items():
        result[_text(key, f"{name} key")] = _nonneg_int(item, f"{name}.{key}")
    return {key: result[key] for key in sorted(result)}


@dataclass(frozen=True)
class UnavailableResult:
    operation: str
    adapter_id: str
    reason_code: str
    retryable: bool
    diagnostic: str

    _FIELDS = frozenset(
        {"operation", "adapter_id", "reason_code", "retryable", "diagnostic"}
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "adapter_id": self.adapter_id,
            "reason_code": self.reason_code,
            "retryable": self.retryable,
            "diagnostic": self.diagnostic,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "UnavailableResult":
        payload = _closed(data, cls._FIELDS, "UnavailableResult")
        diagnostic = _text(payload["diagnostic"], "diagnostic")
        if len(diagnostic) > 512:
            raise HarnessError("diagnostic must be at most 512 characters")
        return cls(
            operation=_text(payload["operation"], "operation"),
            adapter_id=_text(payload["adapter_id"], "adapter_id"),
            reason_code=_text(payload["reason_code"], "reason_code"),
            retryable=_bool(payload["retryable"], "retryable"),
            diagnostic=diagnostic,
        )


@dataclass(frozen=True)
class SemanticCapsuleRef:
    """Admission-only reference. Authoritative capsule facts stay in datasets."""

    capsule_cid: str
    semantic_state_root_cid: str
    stable_symbol_id: str
    version_cid: str
    source_cid: str
    confidence: str
    validity_bindings: tuple[str, ...]
    raw_source_required: bool

    _FIELDS = frozenset(
        {
            "capsule_cid",
            "semantic_state_root_cid",
            "stable_symbol_id",
            "version_cid",
            "source_cid",
            "confidence",
            "validity_bindings",
            "raw_source_required",
        }
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "capsule_cid": self.capsule_cid,
            "semantic_state_root_cid": self.semantic_state_root_cid,
            "stable_symbol_id": self.stable_symbol_id,
            "version_cid": self.version_cid,
            "source_cid": self.source_cid,
            "confidence": self.confidence,
            "validity_bindings": list(self.validity_bindings),
            "raw_source_required": self.raw_source_required,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SemanticCapsuleRef":
        payload = _closed(data, cls._FIELDS, "SemanticCapsuleRef")
        return cls(
            capsule_cid=validate_opaque_cid(payload["capsule_cid"], "capsule_cid"),
            semantic_state_root_cid=validate_opaque_cid(
                payload["semantic_state_root_cid"], "semantic_state_root_cid"
            ),
            stable_symbol_id=_text(payload["stable_symbol_id"], "stable_symbol_id"),
            version_cid=validate_opaque_cid(payload["version_cid"], "version_cid"),
            source_cid=validate_opaque_cid(payload["source_cid"], "source_cid"),
            confidence=_text(payload["confidence"], "confidence"),
            validity_bindings=_unique_sorted_cids(
                payload["validity_bindings"], "validity_bindings"
            ),
            raw_source_required=_bool(
                payload["raw_source_required"], "raw_source_required"
            ),
        )


@dataclass(frozen=True)
class ContextPack:
    objective: str
    target_source_cid: str
    surrounding_source_cid: str
    test_source_cid: str
    dependency_capsule_cids: tuple[str, ...]
    obligation_cids: tuple[str, ...]
    counterexample_cids: tuple[str, ...]
    delta_cid: str
    interface_cids: tuple[str, ...]
    assumptions: tuple[str, ...]
    exclusions: tuple[str, ...]
    token_totals: Mapping[str, int]
    estimator_version: str
    risk: str
    route: str
    escalation_recommendation: str

    _FIELDS = frozenset(
        {
            "objective",
            "target_source_cid",
            "surrounding_source_cid",
            "test_source_cid",
            "dependency_capsule_cids",
            "obligation_cids",
            "counterexample_cids",
            "delta_cid",
            "interface_cids",
            "assumptions",
            "exclusions",
            "token_totals",
            "estimator_version",
            "risk",
            "route",
            "escalation_recommendation",
        }
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "target_source_cid": self.target_source_cid,
            "surrounding_source_cid": self.surrounding_source_cid,
            "test_source_cid": self.test_source_cid,
            "dependency_capsule_cids": list(self.dependency_capsule_cids),
            "obligation_cids": list(self.obligation_cids),
            "counterexample_cids": list(self.counterexample_cids),
            "delta_cid": self.delta_cid,
            "interface_cids": list(self.interface_cids),
            "assumptions": list(self.assumptions),
            "exclusions": list(self.exclusions),
            "token_totals": dict(self.token_totals),
            "estimator_version": self.estimator_version,
            "risk": self.risk,
            "route": self.route,
            "escalation_recommendation": self.escalation_recommendation,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ContextPack":
        payload = _closed(data, cls._FIELDS, "ContextPack")
        return cls(
            objective=_text(payload["objective"], "objective"),
            target_source_cid=validate_opaque_cid(
                payload["target_source_cid"], "target_source_cid"
            ),
            surrounding_source_cid=validate_opaque_cid(
                payload["surrounding_source_cid"], "surrounding_source_cid"
            ),
            test_source_cid=validate_opaque_cid(
                payload["test_source_cid"], "test_source_cid"
            ),
            dependency_capsule_cids=_unique_sorted_cids(
                payload["dependency_capsule_cids"], "dependency_capsule_cids"
            ),
            obligation_cids=_unique_sorted_cids(
                payload["obligation_cids"], "obligation_cids"
            ),
            counterexample_cids=_unique_sorted_cids(
                payload["counterexample_cids"], "counterexample_cids"
            ),
            delta_cid=validate_opaque_cid(payload["delta_cid"], "delta_cid"),
            interface_cids=_unique_sorted_cids(payload["interface_cids"], "interface_cids"),
            assumptions=_unique_sorted_texts(payload["assumptions"], "assumptions"),
            exclusions=_unique_sorted_texts(payload["exclusions"], "exclusions"),
            token_totals=_string_int_map(payload["token_totals"], "token_totals"),
            estimator_version=_text(payload["estimator_version"], "estimator_version"),
            risk=_text(payload["risk"], "risk"),
            route=_enum(payload["route"], ModelRoute, "route"),
            escalation_recommendation=_text(
                payload["escalation_recommendation"], "escalation_recommendation"
            ),
        )


@dataclass(frozen=True)
class PatchProposal:
    provider_id: str
    mode: str
    base_tree_cid: str
    base_root_cid: str
    unified_diff_cid: str
    declared_paths: tuple[str, ...]
    generation: int

    _FIELDS = frozenset(
        {
            "provider_id",
            "mode",
            "base_tree_cid",
            "base_root_cid",
            "unified_diff_cid",
            "declared_paths",
            "generation",
        }
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "mode": self.mode,
            "base_tree_cid": self.base_tree_cid,
            "base_root_cid": self.base_root_cid,
            "unified_diff_cid": self.unified_diff_cid,
            "declared_paths": list(self.declared_paths),
            "generation": self.generation,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PatchProposal":
        payload = _closed(data, cls._FIELDS, "PatchProposal")
        return cls(
            provider_id=_text(payload["provider_id"], "provider_id"),
            mode=_enum(payload["mode"], HarnessMode, "mode"),
            base_tree_cid=validate_opaque_cid(payload["base_tree_cid"], "base_tree_cid"),
            base_root_cid=validate_opaque_cid(payload["base_root_cid"], "base_root_cid"),
            unified_diff_cid=validate_opaque_cid(
                payload["unified_diff_cid"], "unified_diff_cid"
            ),
            declared_paths=_unique_sorted_texts(
                payload["declared_paths"], "declared_paths"
            ),
            generation=_nonneg_int(payload["generation"], "generation"),
        )


@dataclass(frozen=True)
class TestSelectionRef:
    """Opaque handle. Selected nodes and reason paths stay in datasets."""

    selection_cid: str
    previous_semantic_state_root_cid: str | None
    current_semantic_state_root_cid: str

    _FIELDS = frozenset(
        {
            "selection_cid",
            "previous_semantic_state_root_cid",
            "current_semantic_state_root_cid",
        }
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "selection_cid": self.selection_cid,
            "previous_semantic_state_root_cid": self.previous_semantic_state_root_cid,
            "current_semantic_state_root_cid": self.current_semantic_state_root_cid,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TestSelectionRef":
        payload = _closed(data, cls._FIELDS, "TestSelectionRef")
        return cls(
            selection_cid=validate_opaque_cid(payload["selection_cid"], "selection_cid"),
            previous_semantic_state_root_cid=_optional_cid(
                payload["previous_semantic_state_root_cid"],
                "previous_semantic_state_root_cid",
            ),
            current_semantic_state_root_cid=validate_opaque_cid(
                payload["current_semantic_state_root_cid"],
                "current_semantic_state_root_cid",
            ),
        )


@dataclass(frozen=True)
class VerificationReceipt:
    tree_cid: str
    config_cid: str
    dependency_lock_cid: str
    policy_cid: str
    interface_cid: str
    root_cid: str
    command_identity: str
    selection_ref: TestSelectionRef
    exit_code: int
    output_artifact_cids: tuple[str, ...]
    simulated: bool
    fresh: bool
    acceptance_eligible: bool

    _FIELDS = frozenset(
        {
            "tree_cid",
            "config_cid",
            "dependency_lock_cid",
            "policy_cid",
            "interface_cid",
            "root_cid",
            "command_identity",
            "selection_ref",
            "exit_code",
            "output_artifact_cids",
            "simulated",
            "fresh",
            "acceptance_eligible",
        }
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tree_cid": self.tree_cid,
            "config_cid": self.config_cid,
            "dependency_lock_cid": self.dependency_lock_cid,
            "policy_cid": self.policy_cid,
            "interface_cid": self.interface_cid,
            "root_cid": self.root_cid,
            "command_identity": self.command_identity,
            "selection_ref": self.selection_ref.to_dict(),
            "exit_code": self.exit_code,
            "output_artifact_cids": list(self.output_artifact_cids),
            "simulated": self.simulated,
            "fresh": self.fresh,
            "acceptance_eligible": self.acceptance_eligible,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VerificationReceipt":
        payload = _closed(data, cls._FIELDS, "VerificationReceipt")
        if not isinstance(payload["selection_ref"], Mapping):
            raise HarnessError("selection_ref must be an object")
        exit_code = payload["exit_code"]
        if type(exit_code) is not int or isinstance(exit_code, bool):
            raise HarnessError("exit_code must be an integer")
        return cls(
            tree_cid=validate_opaque_cid(payload["tree_cid"], "tree_cid"),
            config_cid=validate_opaque_cid(payload["config_cid"], "config_cid"),
            dependency_lock_cid=validate_opaque_cid(
                payload["dependency_lock_cid"], "dependency_lock_cid"
            ),
            policy_cid=validate_opaque_cid(payload["policy_cid"], "policy_cid"),
            interface_cid=validate_opaque_cid(payload["interface_cid"], "interface_cid"),
            root_cid=validate_opaque_cid(payload["root_cid"], "root_cid"),
            command_identity=_text(payload["command_identity"], "command_identity"),
            selection_ref=TestSelectionRef.from_dict(payload["selection_ref"]),
            exit_code=exit_code,
            output_artifact_cids=_unique_sorted_cids(
                payload["output_artifact_cids"], "output_artifact_cids"
            ),
            simulated=_bool(payload["simulated"], "simulated"),
            fresh=_bool(payload["fresh"], "fresh"),
            acceptance_eligible=_bool(
                payload["acceptance_eligible"], "acceptance_eligible"
            ),
        )


@dataclass(frozen=True)
class RootRef:
    root_cid: str
    generation: int

    _FIELDS = frozenset({"root_cid", "generation"})

    def to_dict(self) -> dict[str, Any]:
        return {"root_cid": self.root_cid, "generation": self.generation}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RootRef":
        payload = _closed(data, cls._FIELDS, "RootRef")
        return cls(
            root_cid=validate_opaque_cid(payload["root_cid"], "root_cid"),
            generation=_nonneg_int(payload["generation"], "generation"),
        )


@dataclass(frozen=True)
class SemanticStateRootManifest:
    repository_id: str
    base_tree_cid: str
    candidate_tree_cid: str
    datasets_state_cid: str
    datasets_semantic_state_root_cid: str
    capsule_index_cid: str
    delta_cid: str
    invalidation_cid: str
    obligation_set_cid: str
    test_selection_cid: str
    receipt_index_cid: str
    environment_binding_cids: tuple[str, ...]
    event_head_cid: str
    versions: Mapping[str, str]
    acceptance_disposition: str

    _FIELDS = frozenset(
        {
            "repository_id",
            "base_tree_cid",
            "candidate_tree_cid",
            "datasets_state_cid",
            "datasets_semantic_state_root_cid",
            "capsule_index_cid",
            "delta_cid",
            "invalidation_cid",
            "obligation_set_cid",
            "test_selection_cid",
            "receipt_index_cid",
            "environment_binding_cids",
            "event_head_cid",
            "versions",
            "acceptance_disposition",
        }
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "repository_id": self.repository_id,
            "base_tree_cid": self.base_tree_cid,
            "candidate_tree_cid": self.candidate_tree_cid,
            "datasets_state_cid": self.datasets_state_cid,
            "datasets_semantic_state_root_cid": self.datasets_semantic_state_root_cid,
            "capsule_index_cid": self.capsule_index_cid,
            "delta_cid": self.delta_cid,
            "invalidation_cid": self.invalidation_cid,
            "obligation_set_cid": self.obligation_set_cid,
            "test_selection_cid": self.test_selection_cid,
            "receipt_index_cid": self.receipt_index_cid,
            "environment_binding_cids": list(self.environment_binding_cids),
            "event_head_cid": self.event_head_cid,
            "versions": dict(self.versions),
            "acceptance_disposition": self.acceptance_disposition,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SemanticStateRootManifest":
        payload = _closed(data, cls._FIELDS, "SemanticStateRootManifest")
        if not isinstance(payload["versions"], Mapping):
            raise HarnessError("versions must be an object")
        versions = {
            _text(key, "versions key"): _text(item, f"versions.{key}")
            for key, item in payload["versions"].items()
        }
        required_versions = {
            "semantic_index_schema",
            "semantic_state_schema",
            "capsule_schema",
            "selection_schema",
        }
        if set(versions) != required_versions:
            raise HarnessError(
                "versions must bind semantic_index_schema, semantic_state_schema, "
                "capsule_schema, and selection_schema"
            )
        return cls(
            repository_id=_text(payload["repository_id"], "repository_id"),
            base_tree_cid=validate_opaque_cid(payload["base_tree_cid"], "base_tree_cid"),
            candidate_tree_cid=validate_opaque_cid(
                payload["candidate_tree_cid"], "candidate_tree_cid"
            ),
            datasets_state_cid=validate_opaque_cid(
                payload["datasets_state_cid"], "datasets_state_cid"
            ),
            datasets_semantic_state_root_cid=validate_opaque_cid(
                payload["datasets_semantic_state_root_cid"],
                "datasets_semantic_state_root_cid",
            ),
            capsule_index_cid=validate_opaque_cid(
                payload["capsule_index_cid"], "capsule_index_cid"
            ),
            delta_cid=validate_opaque_cid(payload["delta_cid"], "delta_cid"),
            invalidation_cid=validate_opaque_cid(
                payload["invalidation_cid"], "invalidation_cid"
            ),
            obligation_set_cid=validate_opaque_cid(
                payload["obligation_set_cid"], "obligation_set_cid"
            ),
            test_selection_cid=validate_opaque_cid(
                payload["test_selection_cid"], "test_selection_cid"
            ),
            receipt_index_cid=validate_opaque_cid(
                payload["receipt_index_cid"], "receipt_index_cid"
            ),
            environment_binding_cids=_unique_sorted_cids(
                payload["environment_binding_cids"], "environment_binding_cids"
            ),
            event_head_cid=validate_opaque_cid(
                payload["event_head_cid"], "event_head_cid"
            ),
            versions={key: versions[key] for key in sorted(versions)},
            acceptance_disposition=_enum(
                payload["acceptance_disposition"],
                AcceptanceDisposition,
                "acceptance_disposition",
            ),
        )


@dataclass(frozen=True)
class HarnessResult:
    disposition: str
    previous_root: RootRef | None
    current_root: RootRef
    patch: PatchProposal | None
    receipt_cids: tuple[str, ...]
    obligation_cids: tuple[str, ...]
    event_head_cid: str
    reasons: tuple[str, ...]

    _FIELDS = frozenset(
        {
            "disposition",
            "previous_root",
            "current_root",
            "patch",
            "receipt_cids",
            "obligation_cids",
            "event_head_cid",
            "reasons",
        }
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition,
            "previous_root": None if self.previous_root is None else self.previous_root.to_dict(),
            "current_root": self.current_root.to_dict(),
            "patch": None if self.patch is None else self.patch.to_dict(),
            "receipt_cids": list(self.receipt_cids),
            "obligation_cids": list(self.obligation_cids),
            "event_head_cid": self.event_head_cid,
            "reasons": list(self.reasons),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HarnessResult":
        payload = _closed(data, cls._FIELDS, "HarnessResult")
        previous = payload["previous_root"]
        patch = payload["patch"]
        return cls(
            disposition=_enum(payload["disposition"], HarnessDisposition, "disposition"),
            previous_root=None if previous is None else RootRef.from_dict(previous),
            current_root=RootRef.from_dict(payload["current_root"]),
            patch=None if patch is None else PatchProposal.from_dict(patch),
            receipt_cids=_unique_sorted_cids(payload["receipt_cids"], "receipt_cids"),
            obligation_cids=_unique_sorted_cids(
                payload["obligation_cids"], "obligation_cids"
            ),
            event_head_cid=validate_opaque_cid(
                payload["event_head_cid"], "event_head_cid"
            ),
            reasons=_unique_sorted_texts(payload["reasons"], "reasons"),
        )
