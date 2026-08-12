"""Emit MCP++ Profile B verification receipts and enforce freshness admission.

Interface: ``SemanticVerificationReceipt@1``, ``ReceiptFreshnessAdmission@1``

Receipts bind pre/post trees, datasets Merkle/state roots, capsule index, delta,
selection, command/toolchain/dependency/config/policy/interface identities,
provider mode, proof outcomes, and output blocks. Identity is exclusively
``canonicalize_artifact`` plus ``cid_for_bytes`` (real CIDv1).

A receipt is admissible for verification or state-root promotion only when:
- every binding matches the current world,
- required stages passed,
- all referenced artifacts rehash,
- the event parent is current,
- proof unavailability is explicit (never coerced to passed),
- ``simulation == false``,
- freshness is ``fresh``.

Stale, simulated, incomplete, or mismatched receipts remain inspectable but
never satisfy acceptance. Operational/scheduler/provider receipts do not prove
correctness.

SCH-009 / sch/receipt@1 / sch/freshness@1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import (
    BOARD_NAMESPACE,
    HarnessError,
    VerificationReceipt,
    _bool,
    _closed,
    _text,
    _unique_sorted_cids,
    _unique_sorted_texts,
    validate_opaque_cid,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.wire import (
    SemanticStateWireCodec,
    cid_for_payload,
)

# ---------------------------------------------------------------------------
# Interfaces / schemas / closed vocabularies
# ---------------------------------------------------------------------------

RECEIPT_INTERFACE = "SemanticVerificationReceipt@1"
RECEIPT_SCHEMA = "ipfs-accelerate.semantic-verification-receipt@1"
FRESHNESS_ADMISSION_INTERFACE = "ReceiptFreshnessAdmission@1"
FRESHNESS_ADMISSION_SCHEMA = "ipfs-accelerate.receipt-freshness-admission@1"
RECEIPT_INDEX_SCHEMA = "ipfs-accelerate.semantic-receipt-index@1"
ADAPTER_ID = "ipfs-accelerate.semantic-state.receipts"

FRESHNESS_FRESH = "fresh"
FRESHNESS_STALE = "stale"
FRESHNESS_UNKNOWN = "unknown"
FRESHNESS_VALUES = frozenset({FRESHNESS_FRESH, FRESHNESS_STALE, FRESHNESS_UNKNOWN})

ADMISSION_ADMITTED = "admitted"
ADMISSION_REJECTED = "rejected"
ADMISSION_STALE = "stale"
ADMISSION_SIMULATED = "simulated"
ADMISSION_UNAVAILABLE = "unavailable"
ADMISSION_INCOMPLETE = "incomplete"
ADMISSION_DECISIONS = frozenset(
    {
        ADMISSION_ADMITTED,
        ADMISSION_REJECTED,
        ADMISSION_STALE,
        ADMISSION_SIMULATED,
        ADMISSION_UNAVAILABLE,
        ADMISSION_INCOMPLETE,
    }
)

PROVIDER_MODE_PRODUCTION = "production"
PROVIDER_MODE_DEVELOPMENT = "development"
PROVIDER_MODE_SIMULATED = "simulated"
PROVIDER_MODES = frozenset(
    {PROVIDER_MODE_PRODUCTION, PROVIDER_MODE_DEVELOPMENT, PROVIDER_MODE_SIMULATED}
)

PROOF_STATUS_PASSED = "passed"
PROOF_STATUS_FAILED = "failed"
PROOF_STATUS_UNAVAILABLE = "unavailable"
PROOF_STATUS_SKIPPED = "skipped"
PROOF_STATUSES = frozenset(
    {
        PROOF_STATUS_PASSED,
        PROOF_STATUS_FAILED,
        PROOF_STATUS_UNAVAILABLE,
        PROOF_STATUS_SKIPPED,
    }
)

# Binding keys that participate in freshness comparison (CID or identity).
_BINDING_CID_FIELDS = (
    "pre_tree_cid",
    "post_tree_cid",
    "datasets_state_cid",
    "datasets_semantic_state_root_cid",
    "capsule_index_cid",
    "delta_cid",
    "selection_cid",
    "current_semantic_state_root_cid",
    "toolchain_cid",
    "dependency_lock_cid",
    "config_cid",
    "policy_cid",
    "interface_cid",
)

# Policy / interface drift produces additional typed obligations.
_POLICY_STALE_OBLIGATIONS = (
    "obligation:policy_decision",
    "obligation:security_admission",
)
_INTERFACE_STALE_OBLIGATIONS = (
    "obligation:interface_description",
    "obligation:client_adapter",
)


class ReceiptError(HarnessError):
    """Closed receipt or freshness-admission contract violation."""


class StaleReceiptError(ReceiptError):
    """Raised when a receipt cannot be admitted because bindings are stale."""

    def __init__(
        self,
        message: str,
        *,
        stale_obligations: Sequence[str] = (),
        reason_codes: Sequence[str] = (),
    ) -> None:
        super().__init__(message)
        self.stale_obligations = tuple(sorted(set(stale_obligations)))
        self.reason_codes = tuple(sorted(set(reason_codes)))


class SimulatedReceiptError(ReceiptError):
    """Raised when a simulated receipt is offered for verification or promotion."""


class UnavailableProofError(ReceiptError):
    """Raised when unavailable proof evidence is treated as a passed proof."""


@runtime_checkable
class _DurablePort(Protocol):
    def put(
        self,
        artifact: Mapping[str, Any],
        *,
        expected_cid: str,
        codec: str = "dag-json",
    ) -> Mapping[str, Any]: ...

    def get(self, cid: str) -> Mapping[str, Any]: ...

    def has(self, cid: str) -> bool: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _freshness_value(value: Any) -> str:
    text = str(getattr(value, "value", value)).strip()
    if text not in FRESHNESS_VALUES:
        raise ReceiptError(f"freshness must be one of {sorted(FRESHNESS_VALUES)}")
    return text


def _admission_value(value: Any) -> str:
    text = str(getattr(value, "value", value)).strip()
    if text not in ADMISSION_DECISIONS:
        raise ReceiptError(f"admission must be one of {sorted(ADMISSION_DECISIONS)}")
    return text


def _provider_mode(value: Any) -> str:
    text = _text(value, "provider_mode")
    if text not in PROVIDER_MODES:
        raise ReceiptError(f"provider_mode must be one of {sorted(PROVIDER_MODES)}")
    return text


def _proof_status(value: Any) -> str:
    text = _text(value, "proof_status")
    if text not in PROOF_STATUSES:
        raise ReceiptError(f"proof status must be one of {sorted(PROOF_STATUSES)}")
    return text


def _exit_code(value: Any) -> int:
    if type(value) is not int or isinstance(value, bool):
        raise ReceiptError("exit_code must be an integer")
    return value


def _attr(obj: Any, *names: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        for name in names:
            if name in obj:
                return obj[name]
        return default
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _optional_text_cid(value: Any, name: str) -> str | None:
    if value is None or value == "":
        return None
    return validate_opaque_cid(value, name)


# ---------------------------------------------------------------------------
# Binding surface
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReceiptBindings:
    """Exact content-addressed world a verification receipt is bound to."""

    pre_tree_cid: str
    post_tree_cid: str
    datasets_state_cid: str
    datasets_semantic_state_root_cid: str
    capsule_index_cid: str
    delta_cid: str
    selection_cid: str
    previous_semantic_state_root_cid: str | None
    current_semantic_state_root_cid: str
    command_identity: str
    toolchain_cid: str
    dependency_lock_cid: str
    config_cid: str
    policy_cid: str
    interface_cid: str
    provider_mode: str
    proof_outcomes: tuple[tuple[str, str], ...]
    output_artifact_cids: tuple[str, ...]
    event_parent_cid: str | None

    _FIELDS = frozenset(
        {
            "pre_tree_cid",
            "post_tree_cid",
            "datasets_state_cid",
            "datasets_semantic_state_root_cid",
            "capsule_index_cid",
            "delta_cid",
            "selection_cid",
            "previous_semantic_state_root_cid",
            "current_semantic_state_root_cid",
            "command_identity",
            "toolchain_cid",
            "dependency_lock_cid",
            "config_cid",
            "policy_cid",
            "interface_cid",
            "provider_mode",
            "proof_outcomes",
            "output_artifact_cids",
            "event_parent_cid",
        }
    )

    def __post_init__(self) -> None:
        for name in _BINDING_CID_FIELDS:
            object.__setattr__(
                self, name, validate_opaque_cid(getattr(self, name), name)
            )
        object.__setattr__(
            self,
            "previous_semantic_state_root_cid",
            _optional_text_cid(
                self.previous_semantic_state_root_cid,
                "previous_semantic_state_root_cid",
            ),
        )
        object.__setattr__(
            self,
            "event_parent_cid",
            _optional_text_cid(self.event_parent_cid, "event_parent_cid"),
        )
        object.__setattr__(
            self, "command_identity", _text(self.command_identity, "command_identity")
        )
        object.__setattr__(self, "provider_mode", _provider_mode(self.provider_mode))
        outcomes = _normalize_proof_outcomes(self.proof_outcomes)
        object.__setattr__(self, "proof_outcomes", outcomes)
        object.__setattr__(
            self,
            "output_artifact_cids",
            _unique_sorted_cids(
                list(self.output_artifact_cids), "output_artifact_cids"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pre_tree_cid": self.pre_tree_cid,
            "post_tree_cid": self.post_tree_cid,
            "datasets_state_cid": self.datasets_state_cid,
            "datasets_semantic_state_root_cid": self.datasets_semantic_state_root_cid,
            "capsule_index_cid": self.capsule_index_cid,
            "delta_cid": self.delta_cid,
            "selection_cid": self.selection_cid,
            "previous_semantic_state_root_cid": self.previous_semantic_state_root_cid,
            "current_semantic_state_root_cid": self.current_semantic_state_root_cid,
            "command_identity": self.command_identity,
            "toolchain_cid": self.toolchain_cid,
            "dependency_lock_cid": self.dependency_lock_cid,
            "config_cid": self.config_cid,
            "policy_cid": self.policy_cid,
            "interface_cid": self.interface_cid,
            "provider_mode": self.provider_mode,
            "proof_outcomes": [
                {"proof_id": pid, "status": status}
                for pid, status in self.proof_outcomes
            ],
            "output_artifact_cids": list(self.output_artifact_cids),
            "event_parent_cid": self.event_parent_cid,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReceiptBindings":
        payload = _closed(data, cls._FIELDS, "ReceiptBindings")
        outcomes_raw = payload["proof_outcomes"]
        if not isinstance(outcomes_raw, list):
            raise ReceiptError("proof_outcomes must be a list")
        outcomes: list[tuple[str, str]] = []
        for item in outcomes_raw:
            if not isinstance(item, Mapping):
                raise ReceiptError("proof_outcomes items must be objects")
            item_closed = _closed(
                item, frozenset({"proof_id", "status"}), "proof_outcome"
            )
            outcomes.append(
                (
                    _text(item_closed["proof_id"], "proof_id"),
                    _proof_status(item_closed["status"]),
                )
            )
        return cls(
            pre_tree_cid=payload["pre_tree_cid"],
            post_tree_cid=payload["post_tree_cid"],
            datasets_state_cid=payload["datasets_state_cid"],
            datasets_semantic_state_root_cid=payload[
                "datasets_semantic_state_root_cid"
            ],
            capsule_index_cid=payload["capsule_index_cid"],
            delta_cid=payload["delta_cid"],
            selection_cid=payload["selection_cid"],
            previous_semantic_state_root_cid=payload[
                "previous_semantic_state_root_cid"
            ],
            current_semantic_state_root_cid=payload[
                "current_semantic_state_root_cid"
            ],
            command_identity=payload["command_identity"],
            toolchain_cid=payload["toolchain_cid"],
            dependency_lock_cid=payload["dependency_lock_cid"],
            config_cid=payload["config_cid"],
            policy_cid=payload["policy_cid"],
            interface_cid=payload["interface_cid"],
            provider_mode=payload["provider_mode"],
            proof_outcomes=tuple(outcomes),
            output_artifact_cids=tuple(payload["output_artifact_cids"]),
            event_parent_cid=payload["event_parent_cid"],
        )

    @property
    def bindings_cid(self) -> str:
        return cid_for_payload(self.to_dict())

    def has_unavailable_proof(self) -> bool:
        return any(status == PROOF_STATUS_UNAVAILABLE for _, status in self.proof_outcomes)

    def all_required_proofs_passed(self) -> bool:
        """Return True when every bound proof is passed (none unavailable/failed)."""

        if not self.proof_outcomes:
            return True
        return all(status == PROOF_STATUS_PASSED for _, status in self.proof_outcomes)


def _normalize_proof_outcomes(raw: Any) -> tuple[tuple[str, str], ...]:
    if raw is None:
        return ()
    if not isinstance(raw, (list, tuple)):
        raise ReceiptError("proof_outcomes must be a sequence")
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for item in raw:
        if isinstance(item, Mapping):
            pid = _text(item.get("proof_id"), "proof_id")
            status = _proof_status(item.get("status"))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            pid = _text(item[0], "proof_id")
            status = _proof_status(item[1])
        else:
            raise ReceiptError("proof_outcomes items must be (proof_id, status) pairs")
        if pid in seen:
            raise ReceiptError(f"duplicate proof_id {pid!r}")
        seen.add(pid)
        pairs.append((pid, status))
    return tuple(sorted(pairs, key=lambda pair: pair[0]))


# ---------------------------------------------------------------------------
# Compiled receipt
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompiledReceipt:
    """Closed, content-addressed verification receipt (MCP++ Profile B body)."""

    schema: str
    interface: str
    bindings: ReceiptBindings
    exit_code: int
    stages_passed: bool
    simulated: bool
    fresh: bool
    acceptance_eligible: bool
    freshness: str
    unavailable_proof: bool
    reason_codes: tuple[str, ...]
    receipt_cid: str
    output_cid: str

    _BODY_FIELDS = frozenset(
        {
            "schema",
            "interface",
            "bindings",
            "exit_code",
            "stages_passed",
            "simulated",
            "fresh",
            "acceptance_eligible",
            "freshness",
            "unavailable_proof",
            "reason_codes",
        }
    )

    def __post_init__(self) -> None:
        if self.schema != RECEIPT_SCHEMA:
            raise ReceiptError(f"schema must be {RECEIPT_SCHEMA}")
        if self.interface != RECEIPT_INTERFACE:
            raise ReceiptError(f"interface must be {RECEIPT_INTERFACE}")
        if not isinstance(self.bindings, ReceiptBindings):
            raise ReceiptError("bindings must be ReceiptBindings")
        object.__setattr__(self, "exit_code", _exit_code(self.exit_code))
        for name in (
            "stages_passed",
            "simulated",
            "fresh",
            "acceptance_eligible",
            "unavailable_proof",
        ):
            if type(getattr(self, name)) is not bool:
                raise ReceiptError(f"{name} must be a boolean")
        object.__setattr__(self, "freshness", _freshness_value(self.freshness))
        object.__setattr__(
            self,
            "reason_codes",
            _unique_sorted_texts(list(self.reason_codes), "reason_codes"),
        )
        validate_opaque_cid(self.receipt_cid, "receipt_cid")
        validate_opaque_cid(self.output_cid, "output_cid")
        # Hard invariants.
        if self.simulated and self.acceptance_eligible:
            raise ReceiptError(
                "simulated receipts can never be acceptance_eligible"
            )
        if self.freshness != FRESHNESS_FRESH and self.acceptance_eligible:
            raise ReceiptError(
                "stale or unknown receipts can never be acceptance_eligible"
            )
        if not self.fresh and self.acceptance_eligible:
            raise ReceiptError("non-fresh receipts can never be acceptance_eligible")
        if self.unavailable_proof and self.acceptance_eligible:
            raise ReceiptError(
                "unavailable proof cannot make a receipt acceptance_eligible"
            )
        if self.bindings.has_unavailable_proof() != self.unavailable_proof:
            raise ReceiptError("unavailable_proof flag must match proof_outcomes")
        if (
            self.bindings.provider_mode == PROVIDER_MODE_SIMULATED
            and not self.simulated
        ):
            raise ReceiptError("provider_mode simulated requires simulated=true")

    def body_dict(self) -> dict[str, Any]:
        """Return the closed receipt body (excludes derived CIDs)."""

        return {
            "schema": self.schema,
            "interface": self.interface,
            "bindings": self.bindings.to_dict(),
            "exit_code": self.exit_code,
            "stages_passed": self.stages_passed,
            "simulated": self.simulated,
            "fresh": self.fresh,
            "acceptance_eligible": self.acceptance_eligible,
            "freshness": self.freshness,
            "unavailable_proof": self.unavailable_proof,
            "reason_codes": list(self.reason_codes),
        }

    def to_dict(self) -> dict[str, Any]:
        body = self.body_dict()
        body["output_cid"] = self.output_cid
        body["receipt_cid"] = self.receipt_cid
        return body

    def to_verification_receipt(self) -> VerificationReceipt:
        """Project into the closed harness ``VerificationReceipt`` contract."""

        return VerificationReceipt.from_dict(
            {
                "tree_cid": self.bindings.post_tree_cid,
                "config_cid": self.bindings.config_cid,
                "dependency_lock_cid": self.bindings.dependency_lock_cid,
                "policy_cid": self.bindings.policy_cid,
                "interface_cid": self.bindings.interface_cid,
                "root_cid": self.bindings.current_semantic_state_root_cid,
                "command_identity": self.bindings.command_identity,
                "selection_ref": {
                    "selection_cid": self.bindings.selection_cid,
                    "previous_semantic_state_root_cid": (
                        self.bindings.previous_semantic_state_root_cid
                    ),
                    "current_semantic_state_root_cid": (
                        self.bindings.current_semantic_state_root_cid
                    ),
                },
                "exit_code": self.exit_code,
                "output_artifact_cids": list(self.bindings.output_artifact_cids),
                "simulated": self.simulated,
                "fresh": self.fresh,
                "acceptance_eligible": self.acceptance_eligible,
            }
        )

    def as_mcp_execution_receipt(self) -> dict[str, Any]:
        """Encode as MCP++ Profile B execution receipt (output_cid/receipt_cid)."""

        codec = SemanticStateWireCodec()
        return codec.encode_execution_receipt(self.body_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CompiledReceipt":
        if not isinstance(data, Mapping):
            raise ReceiptError("CompiledReceipt must be an object")
        required = cls._BODY_FIELDS | frozenset({"output_cid", "receipt_cid"})
        # Allow either body-only or full (with CIDs); rehash always.
        if "output_cid" in data or "receipt_cid" in data:
            payload = _closed(data, required, "CompiledReceipt")
        else:
            payload = _closed(data, cls._BODY_FIELDS, "CompiledReceipt")
            payload = dict(payload)
            payload["output_cid"] = ""
            payload["receipt_cid"] = ""
        if not isinstance(payload["bindings"], Mapping):
            raise ReceiptError("bindings must be an object")
        bindings = ReceiptBindings.from_dict(payload["bindings"])
        body = {
            "schema": _text(payload["schema"], "schema"),
            "interface": _text(payload["interface"], "interface"),
            "bindings": bindings.to_dict(),
            "exit_code": payload["exit_code"],
            "stages_passed": payload["stages_passed"],
            "simulated": payload["simulated"],
            "fresh": payload["fresh"],
            "acceptance_eligible": payload["acceptance_eligible"],
            "freshness": payload["freshness"],
            "unavailable_proof": payload["unavailable_proof"],
            "reason_codes": list(payload["reason_codes"]),
        }
        expected_output = cid_for_payload(body)
        expected_receipt = cid_for_payload(
            {"output_cid": expected_output, "result": body}
        )
        stored_output = payload.get("output_cid") or expected_output
        stored_receipt = payload.get("receipt_cid") or expected_receipt
        if stored_output != expected_output:
            raise ReceiptError("output_cid does not match rehashed body")
        if stored_receipt != expected_receipt:
            raise ReceiptError("receipt_cid does not match rehashed content")
        return cls(
            schema=body["schema"],
            interface=body["interface"],
            bindings=bindings,
            exit_code=_exit_code(body["exit_code"]),
            stages_passed=_bool(body["stages_passed"], "stages_passed"),
            simulated=_bool(body["simulated"], "simulated"),
            fresh=_bool(body["fresh"], "fresh"),
            acceptance_eligible=_bool(
                body["acceptance_eligible"], "acceptance_eligible"
            ),
            freshness=_freshness_value(body["freshness"]),
            unavailable_proof=_bool(body["unavailable_proof"], "unavailable_proof"),
            reason_codes=tuple(body["reason_codes"]),
            receipt_cid=expected_receipt,
            output_cid=expected_output,
        )

    def rehash(self) -> "CompiledReceipt":
        """Return a receipt whose CIDs match canonical body bytes."""

        return CompiledReceipt.from_dict(self.body_dict())


# ---------------------------------------------------------------------------
# Freshness policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReceiptFreshnessPolicy:
    """Compare receipt bindings to the current world and emit stale obligations.

    Any bound input change stales the receipt. Policy and interface drift also
    invalidate policy decisions / security admission and interface descriptions
    / client adapters respectively.
    """

    def assess(
        self,
        bindings: ReceiptBindings,
        *,
        current: Mapping[str, Any],
        event_parent_current: bool = True,
        output_artifacts_present: bool = True,
    ) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
        """Return ``(freshness, stale_obligations, reason_codes)``."""

        if not isinstance(current, Mapping):
            raise ReceiptError("current bindings must be an object")
        obligations: list[str] = []
        reasons: list[str] = []

        for field in _BINDING_CID_FIELDS:
            expected = getattr(bindings, field)
            if field not in current:
                obligations.append(f"stale:{field}")
                reasons.append(f"missing_current:{field}")
                continue
            actual = current[field]
            if actual is None or actual == "":
                obligations.append(f"stale:{field}")
                reasons.append(f"missing_current:{field}")
                continue
            actual_cid = validate_opaque_cid(actual, f"current.{field}")
            if actual_cid != expected:
                obligations.append(f"stale:{field}")
                reasons.append(f"mismatch:{field}")

        # Optional previous root: only compare when both sides provide it.
        prev_current = current.get("previous_semantic_state_root_cid", None)
        if bindings.previous_semantic_state_root_cid is not None:
            if prev_current is None or prev_current == "":
                obligations.append("stale:previous_semantic_state_root_cid")
                reasons.append("missing_current:previous_semantic_state_root_cid")
            else:
                actual_prev = validate_opaque_cid(
                    prev_current, "current.previous_semantic_state_root_cid"
                )
                if actual_prev != bindings.previous_semantic_state_root_cid:
                    obligations.append("stale:previous_semantic_state_root_cid")
                    reasons.append("mismatch:previous_semantic_state_root_cid")

        # Command identity is a non-CID binding that must still match.
        if "command_identity" in current:
            actual_cmd = _text(current["command_identity"], "current.command_identity")
            if actual_cmd != bindings.command_identity:
                obligations.append("stale:command_identity")
                reasons.append("mismatch:command_identity")

        # Provider mode drift stales the receipt.
        if "provider_mode" in current:
            actual_mode = _provider_mode(current["provider_mode"])
            if actual_mode != bindings.provider_mode:
                obligations.append("stale:provider_mode")
                reasons.append("mismatch:provider_mode")

        # Policy / interface changes carry additional invalidation obligations.
        if any(item == "stale:policy_cid" for item in obligations):
            obligations.extend(_POLICY_STALE_OBLIGATIONS)
            reasons.append("policy_invalidates_decisions")
        if any(item == "stale:interface_cid" for item in obligations):
            obligations.extend(_INTERFACE_STALE_OBLIGATIONS)
            reasons.append("interface_invalidates_adapters")

        # Dependency/lock and config changes stale verification receipts.
        if any(item == "stale:dependency_lock_cid" for item in obligations):
            obligations.append("obligation:dependent_summary")
            obligations.append("obligation:verification_receipt")
        if any(item == "stale:config_cid" for item in obligations):
            obligations.append("obligation:bound_test_receipt")

        if not event_parent_current:
            obligations.append("stale:event_parent")
            reasons.append("event_parent_not_current")

        if not output_artifacts_present:
            obligations.append("stale:output_artifacts")
            reasons.append("output_artifacts_missing")

        # Explicit unavailable proofs never silently pass.
        if bindings.has_unavailable_proof():
            reasons.append("unavailable_proof_explicit")

        unique_obs = tuple(sorted(set(obligations)))
        unique_reasons = tuple(sorted(set(reasons)))
        if unique_obs:
            return FRESHNESS_STALE, unique_obs, unique_reasons
        return FRESHNESS_FRESH, (), unique_reasons


# ---------------------------------------------------------------------------
# Admission record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReceiptAdmission:
    """Freshness-admission decision for one compiled receipt."""

    schema: str
    interface: str
    receipt_cid: str
    admission: str
    freshness: str
    can_verify: bool
    can_promote_root: bool
    stale_obligations: tuple[str, ...]
    reason_codes: tuple[str, ...]
    simulated: bool
    unavailable_proof: bool

    def __post_init__(self) -> None:
        if self.schema != FRESHNESS_ADMISSION_SCHEMA:
            raise ReceiptError(f"schema must be {FRESHNESS_ADMISSION_SCHEMA}")
        if self.interface != FRESHNESS_ADMISSION_INTERFACE:
            raise ReceiptError(f"interface must be {FRESHNESS_ADMISSION_INTERFACE}")
        validate_opaque_cid(self.receipt_cid, "receipt_cid")
        object.__setattr__(self, "admission", _admission_value(self.admission))
        object.__setattr__(self, "freshness", _freshness_value(self.freshness))
        for name in (
            "can_verify",
            "can_promote_root",
            "simulated",
            "unavailable_proof",
        ):
            if type(getattr(self, name)) is not bool:
                raise ReceiptError(f"{name} must be a boolean")
        object.__setattr__(
            self,
            "stale_obligations",
            _unique_sorted_texts(list(self.stale_obligations), "stale_obligations"),
        )
        object.__setattr__(
            self,
            "reason_codes",
            _unique_sorted_texts(list(self.reason_codes), "reason_codes"),
        )
        if self.admission != ADMISSION_ADMITTED and (
            self.can_verify or self.can_promote_root
        ):
            raise ReceiptError(
                "non-admitted receipts cannot verify or promote a state root"
            )
        if self.simulated and (self.can_verify or self.can_promote_root):
            raise ReceiptError("simulated receipts cannot verify or promote")
        if self.freshness != FRESHNESS_FRESH and (
            self.can_verify or self.can_promote_root
        ):
            raise ReceiptError("stale/unknown receipts cannot verify or promote")
        if self.unavailable_proof and self.can_promote_root:
            raise ReceiptError(
                "unavailable proof cannot promote a state root"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.interface,
            "receipt_cid": self.receipt_cid,
            "admission": self.admission,
            "freshness": self.freshness,
            "can_verify": self.can_verify,
            "can_promote_root": self.can_promote_root,
            "stale_obligations": list(self.stale_obligations),
            "reason_codes": list(self.reason_codes),
            "simulated": self.simulated,
            "unavailable_proof": self.unavailable_proof,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReceiptAdmission":
        payload = _closed(
            data,
            frozenset(
                {
                    "schema",
                    "interface",
                    "receipt_cid",
                    "admission",
                    "freshness",
                    "can_verify",
                    "can_promote_root",
                    "stale_obligations",
                    "reason_codes",
                    "simulated",
                    "unavailable_proof",
                }
            ),
            "ReceiptAdmission",
        )
        return cls(
            schema=_text(payload["schema"], "schema"),
            interface=_text(payload["interface"], "interface"),
            receipt_cid=payload["receipt_cid"],
            admission=payload["admission"],
            freshness=payload["freshness"],
            can_verify=_bool(payload["can_verify"], "can_verify"),
            can_promote_root=_bool(payload["can_promote_root"], "can_promote_root"),
            stale_obligations=tuple(payload["stale_obligations"]),
            reason_codes=tuple(payload["reason_codes"]),
            simulated=_bool(payload["simulated"], "simulated"),
            unavailable_proof=_bool(
                payload["unavailable_proof"], "unavailable_proof"
            ),
        )


# ---------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------


@dataclass
class ReceiptCompiler:
    """Compile, content-address, optionally store, and rehash verification receipts."""

    durable: _DurablePort | None = None
    wire: SemanticStateWireCodec | None = None

    def __post_init__(self) -> None:
        if self.wire is None:
            self.wire = SemanticStateWireCodec()

    def compile(
        self,
        bindings: ReceiptBindings | Mapping[str, Any],
        *,
        exit_code: int,
        stages_passed: bool,
        simulated: bool = False,
        reason_codes: Sequence[str] = (),
        store: bool = True,
    ) -> CompiledReceipt:
        """Compile a closed content-addressed receipt and optionally store it first."""

        if isinstance(bindings, Mapping):
            bindings = ReceiptBindings.from_dict(bindings)
        if not isinstance(bindings, ReceiptBindings):
            raise ReceiptError("bindings must be ReceiptBindings or a closed mapping")

        exit_code = _exit_code(exit_code)
        if type(stages_passed) is not bool:
            raise ReceiptError("stages_passed must be a boolean")
        if type(simulated) is not bool:
            raise ReceiptError("simulated must be a boolean")

        provider_simulated = bindings.provider_mode == PROVIDER_MODE_SIMULATED
        simulated = bool(simulated or provider_simulated)
        unavailable = bindings.has_unavailable_proof()

        reasons = set(reason_codes or ())
        if simulated:
            reasons.add("simulation")
        if unavailable:
            reasons.add("unavailable_proof_explicit")
        if not stages_passed:
            reasons.add("stages_not_passed")
        if exit_code != 0:
            reasons.add("nonzero_exit")

        # Freshness at compile time is fresh relative to the provided bindings;
        # later admit_receipt reassesses against the current world.
        freshness = FRESHNESS_FRESH
        fresh = True
        acceptance_eligible = (
            stages_passed
            and exit_code == 0
            and not simulated
            and not unavailable
            and bindings.all_required_proofs_passed()
            and fresh
        )
        if not acceptance_eligible and not stages_passed:
            reasons.add("not_acceptance_eligible")

        body = {
            "schema": RECEIPT_SCHEMA,
            "interface": RECEIPT_INTERFACE,
            "bindings": bindings.to_dict(),
            "exit_code": exit_code,
            "stages_passed": stages_passed,
            "simulated": simulated,
            "fresh": fresh,
            "acceptance_eligible": acceptance_eligible,
            "freshness": freshness,
            "unavailable_proof": unavailable,
            "reason_codes": sorted(reasons),
        }
        output_cid = cid_for_payload(body)
        receipt_cid = cid_for_payload({"output_cid": output_cid, "result": body})
        receipt = CompiledReceipt(
            schema=RECEIPT_SCHEMA,
            interface=RECEIPT_INTERFACE,
            bindings=bindings,
            exit_code=exit_code,
            stages_passed=stages_passed,
            simulated=simulated,
            fresh=fresh,
            acceptance_eligible=acceptance_eligible,
            freshness=freshness,
            unavailable_proof=unavailable,
            reason_codes=tuple(sorted(reasons)),
            receipt_cid=receipt_cid,
            output_cid=output_cid,
        )

        if store:
            self.store(receipt)
        return receipt

    def store(self, receipt: CompiledReceipt) -> str:
        """Store the receipt body before any external reference (store-before-ref)."""

        if self.durable is None:
            raise ReceiptError(
                "durable port is required to store receipts before reference"
            )
        body = receipt.body_dict()
        expected = receipt.output_cid
        recomputed = cid_for_payload(body)
        if recomputed != expected:
            raise ReceiptError("receipt body does not rehash to output_cid")
        if not self.durable.has(expected):
            self.durable.put(body, expected_cid=expected, codec="dag-json")
        # Also store the MCP++ envelope so receipt_cid is resolvable.
        envelope = {
            "output_cid": receipt.output_cid,
            "result": body,
        }
        if not self.durable.has(receipt.receipt_cid):
            self.durable.put(
                envelope, expected_cid=receipt.receipt_cid, codec="dag-json"
            )
        return receipt.receipt_cid

    def load(self, receipt_cid: str) -> CompiledReceipt:
        """Load and rehash a previously stored receipt."""

        if self.durable is None:
            raise ReceiptError("durable port is required to load receipts")
        cid = validate_opaque_cid(receipt_cid, "receipt_cid")
        if not self.durable.has(cid):
            raise ReceiptError(f"receipt {cid} is not stored")
        stored = self.durable.get(cid)
        if not isinstance(stored, Mapping):
            raise ReceiptError("stored receipt must be an object")
        # Envelope form or bare body.
        if "result" in stored and "output_cid" in stored:
            result = stored["result"]
            if not isinstance(result, Mapping):
                raise ReceiptError("stored receipt result must be an object")
            body = dict(result)
            body["output_cid"] = stored["output_cid"]
            body["receipt_cid"] = cid
            return CompiledReceipt.from_dict(body)
        body = dict(stored)
        return CompiledReceipt.from_dict(body)

    def rehash_and_validate(self, receipt: CompiledReceipt | Mapping[str, Any]) -> CompiledReceipt:
        """Closed-schema validate and rehash a receipt payload."""

        if isinstance(receipt, CompiledReceipt):
            data = receipt.to_dict()
        elif isinstance(receipt, Mapping):
            data = dict(receipt)
        else:
            raise ReceiptError("receipt must be CompiledReceipt or mapping")
        return CompiledReceipt.from_dict(data)


def compile_verification_receipt(
    bindings: ReceiptBindings | Mapping[str, Any],
    *,
    exit_code: int,
    stages_passed: bool,
    simulated: bool = False,
    reason_codes: Sequence[str] = (),
    durable: _DurablePort | None = None,
    store: bool | None = None,
) -> CompiledReceipt:
    """Module-level compile helper (store only when a durable port is provided)."""

    compiler = ReceiptCompiler(durable=durable)
    if store is None:
        store = durable is not None
    return compiler.compile(
        bindings,
        exit_code=exit_code,
        stages_passed=stages_passed,
        simulated=simulated,
        reason_codes=reason_codes,
        store=store,
    )


# ---------------------------------------------------------------------------
# Admission
# ---------------------------------------------------------------------------


def admit_receipt(
    receipt: CompiledReceipt | Mapping[str, Any],
    *,
    current: Mapping[str, Any],
    event_parent_current: bool = True,
    output_artifacts_present: bool = True,
    require_stored: bool = False,
    durable: _DurablePort | None = None,
    policy: ReceiptFreshnessPolicy | None = None,
    raise_on_reject: bool = False,
) -> ReceiptAdmission:
    """Admit a receipt for verification / state-root promotion.

    Fail-closed: stale, simulated, incomplete, or unavailable-proof receipts
    never set ``can_verify`` / ``can_promote_root``.
    """

    if isinstance(receipt, Mapping):
        compiled = CompiledReceipt.from_dict(receipt)
    elif isinstance(receipt, CompiledReceipt):
        compiled = CompiledReceipt.from_dict(receipt.to_dict())
    else:
        raise ReceiptError("receipt must be CompiledReceipt or mapping")

    # Rehash gate: body must match declared CIDs.
    body = compiled.body_dict()
    if cid_for_payload(body) != compiled.output_cid:
        raise ReceiptError("receipt output_cid does not rehash")
    expected_receipt_cid = cid_for_payload(
        {"output_cid": compiled.output_cid, "result": body}
    )
    if expected_receipt_cid != compiled.receipt_cid:
        raise ReceiptError("receipt_cid does not rehash")

    if require_stored:
        if durable is None:
            raise ReceiptError("durable port required when require_stored is true")
        if not durable.has(compiled.receipt_cid) and not durable.has(
            compiled.output_cid
        ):
            raise ReceiptError("receipt must be stored before admission reference")

    freshness_policy = policy or ReceiptFreshnessPolicy()
    freshness, stale_obs, assess_reasons = freshness_policy.assess(
        compiled.bindings,
        current=current,
        event_parent_current=event_parent_current,
        output_artifacts_present=output_artifacts_present,
    )

    reasons: list[str] = list(assess_reasons)
    reasons.extend(compiled.reason_codes)
    admission = ADMISSION_ADMITTED
    can_verify = False
    can_promote = False

    if compiled.simulated or compiled.bindings.provider_mode == PROVIDER_MODE_SIMULATED:
        admission = ADMISSION_SIMULATED
        reasons.append("simulation_never_verifies_or_promotes")
        if raise_on_reject:
            raise SimulatedReceiptError(
                "simulated receipt cannot satisfy verification or state-root promotion"
            )
    elif freshness != FRESHNESS_FRESH or stale_obs:
        admission = ADMISSION_STALE
        reasons.append("stale_bindings")
        if raise_on_reject:
            raise StaleReceiptError(
                "stale receipt cannot satisfy verification or state-root promotion",
                stale_obligations=stale_obs,
                reason_codes=reasons,
            )
    elif compiled.unavailable_proof or compiled.bindings.has_unavailable_proof():
        admission = ADMISSION_UNAVAILABLE
        reasons.append("unavailable_proof_explicit")
        if raise_on_reject:
            raise UnavailableProofError(
                "unavailable proof is explicit and cannot promote a state root"
            )
    elif not compiled.stages_passed or compiled.exit_code != 0:
        admission = ADMISSION_INCOMPLETE
        reasons.append("stages_incomplete")
    elif not compiled.fresh or compiled.freshness != FRESHNESS_FRESH:
        admission = ADMISSION_STALE
        reasons.append("receipt_marked_stale")
        if raise_on_reject:
            raise StaleReceiptError(
                "receipt marked non-fresh cannot satisfy verification",
                stale_obligations=stale_obs,
                reason_codes=reasons,
            )
    elif not compiled.acceptance_eligible:
        admission = ADMISSION_REJECTED
        reasons.append("not_acceptance_eligible")
    else:
        admission = ADMISSION_ADMITTED
        can_verify = True
        can_promote = True

    # Hard fail-closed: never admit simulate/stale for promotion.
    if admission != ADMISSION_ADMITTED:
        can_verify = False
        can_promote = False

    return ReceiptAdmission(
        schema=FRESHNESS_ADMISSION_SCHEMA,
        interface=FRESHNESS_ADMISSION_INTERFACE,
        receipt_cid=compiled.receipt_cid,
        admission=admission,
        freshness=freshness if admission != ADMISSION_SIMULATED else FRESHNESS_FRESH,
        can_verify=can_verify,
        can_promote_root=can_promote,
        stale_obligations=stale_obs,
        reason_codes=tuple(sorted(set(reasons))),
        simulated=compiled.simulated,
        unavailable_proof=compiled.unavailable_proof,
    )


def receipt_may_verify(admission: ReceiptAdmission) -> bool:
    return (
        admission.admission == ADMISSION_ADMITTED
        and admission.can_verify
        and not admission.simulated
        and admission.freshness == FRESHNESS_FRESH
    )


def receipt_may_promote_root(admission: ReceiptAdmission) -> bool:
    return (
        receipt_may_verify(admission)
        and admission.can_promote_root
        and not admission.unavailable_proof
    )


def build_receipt_index(receipt_cids: Sequence[str]) -> dict[str, Any]:
    """Build a closed, sorted receipt index artifact for root manifests."""

    if not isinstance(receipt_cids, (list, tuple)):
        raise ReceiptError("receipt_cids must be a sequence")
    # Accept input duplicates; emit a unique sorted list.
    unique = sorted({validate_opaque_cid(item, "receipt_cids") for item in receipt_cids})
    cids = _unique_sorted_cids(list(unique), "receipt_cids")
    body = {
        "schema": RECEIPT_INDEX_SCHEMA,
        "receipt_cids": list(cids),
    }
    body["index_cid"] = cid_for_payload(
        {"schema": RECEIPT_INDEX_SCHEMA, "receipt_cids": list(cids)}
    )
    return body


def receipts_descriptor() -> dict[str, Any]:
    return {
        "interface": RECEIPT_INTERFACE,
        "freshness_interface": FRESHNESS_ADMISSION_INTERFACE,
        "schema": RECEIPT_SCHEMA,
        "freshness_schema": FRESHNESS_ADMISSION_SCHEMA,
        "adapter_id": ADAPTER_ID,
        "board_namespace": BOARD_NAMESPACE,
        "operations": (
            "compile_verification_receipt",
            "admit_receipt",
            "rehash_and_validate",
            "store",
            "load",
            "build_receipt_index",
        ),
        "forbids": (
            "stale_acceptance",
            "simulation_acceptance",
            "unavailable_proof_as_passed",
            "store_after_reference",
            "pseudo_cid",
            "operational_receipt_as_correctness",
        ),
    }


__all__ = [
    "ADAPTER_ID",
    "ADMISSION_ADMITTED",
    "ADMISSION_INCOMPLETE",
    "ADMISSION_REJECTED",
    "ADMISSION_SIMULATED",
    "ADMISSION_STALE",
    "ADMISSION_UNAVAILABLE",
    "CompiledReceipt",
    "FRESHNESS_ADMISSION_INTERFACE",
    "FRESHNESS_ADMISSION_SCHEMA",
    "FRESHNESS_FRESH",
    "FRESHNESS_STALE",
    "FRESHNESS_UNKNOWN",
    "PROOF_STATUS_FAILED",
    "PROOF_STATUS_PASSED",
    "PROOF_STATUS_SKIPPED",
    "PROOF_STATUS_UNAVAILABLE",
    "PROVIDER_MODE_DEVELOPMENT",
    "PROVIDER_MODE_PRODUCTION",
    "PROVIDER_MODE_SIMULATED",
    "RECEIPT_INTERFACE",
    "RECEIPT_SCHEMA",
    "ReceiptAdmission",
    "ReceiptBindings",
    "ReceiptCompiler",
    "ReceiptError",
    "ReceiptFreshnessPolicy",
    "SimulatedReceiptError",
    "StaleReceiptError",
    "UnavailableProofError",
    "admit_receipt",
    "build_receipt_index",
    "compile_verification_receipt",
    "receipt_may_promote_root",
    "receipt_may_verify",
    "receipts_descriptor",
]
