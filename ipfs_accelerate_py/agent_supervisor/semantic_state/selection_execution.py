"""Project a sealed datasets ``TestSelection`` into bounded execute commands.

Interface: ``SelectionExecutionAdapter@1``

Datasets remains the only semantic selection authority.  This module:

* verifies a harness ``TestSelectionRef`` against a producer ``TestSelection``
  (selection CID and previous/current semantic-state root bindings);
* maps already-selected pytest node IDs and proof IDs into explicit
  ``ValidationCommand`` / proof obligations;
* enforces the producer ``none`` / ``full_pytest`` / ``full_proofs`` / ``both``
  fallback without weakening it, then may only escalate via harness assurance;
* binds every command to exact tree / config / dependency-lock / toolchain /
  policy / interface CIDs and retains producer reason-path CIDs in provenance.

It never traverses semantic edges, chooses a second affected set, calls
``run_impact_selected``, imports or collects target tests, guesses node IDs,
or invents a second selection.  Cold import is side-effect free.
"""

from __future__ import annotations

import hashlib
import json
import shlex
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final

from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import (
    BOARD_NAMESPACE,
    HarnessError,
    TestSelectionRef,
    UnavailableResult,
    validate_opaque_cid,
    _closed,
    _nonneg_int,
    _optional_cid,
    _text,
    _unique_sorted_texts,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.scheduling_contracts import (
    CancellationToken,
)
from ipfs_accelerate_py.agent_supervisor.validation.validation_commands import (
    ValidationCommand,
    ValidationRequirementKind,
    ValidationStage,
    ValidationVerdictKind,
)

# ---------------------------------------------------------------------------
# Interface pins
# ---------------------------------------------------------------------------

SELECTION_EXECUTION_INTERFACE: Final[str] = "SelectionExecutionAdapter@1"
SELECTION_EXECUTION_SCHEMA: Final[str] = "semantic-state-selection-execution@1"
ADAPTER_ID: Final[str] = "semantic-selection-execution-adapter"
COMMAND_PROVENANCE_SCHEMA: Final[str] = "semantic-state-command-provenance@1"
COMMAND_BINDING_SCHEMA: Final[str] = "semantic-state-command-binding@1"
ASSURANCE_POLICY_SCHEMA: Final[str] = "semantic-state-harness-assurance@1"

# Producer fallback vocabulary (sealed datasets SelectionFallback).
FALLBACK_NONE: Final[str] = "none"
FALLBACK_FULL_PYTEST: Final[str] = "full_pytest"
FALLBACK_FULL_PROOFS: Final[str] = "full_proofs"
FALLBACK_BOTH: Final[str] = "both"
_PRODUCER_FALLBACKS: Final[frozenset[str]] = frozenset(
    {
        FALLBACK_NONE,
        FALLBACK_FULL_PYTEST,
        FALLBACK_FULL_PROOFS,
        FALLBACK_BOTH,
    }
)

_DEFAULT_PYTEST_TIMEOUT_SECONDS: Final[float] = 1800.0
_DEFAULT_PROOF_TIMEOUT_SECONDS: Final[float] = 1800.0
_DEFAULT_STATIC_TIMEOUT_SECONDS: Final[float] = 300.0
_MAX_DIAGNOSTIC: Final[int] = 512
_MAX_COMMANDS: Final[int] = 10_000


# ---------------------------------------------------------------------------
# Errors and typed outcomes
# ---------------------------------------------------------------------------


class SelectionExecutionError(HarnessError):
    """Sealed selection projection failed closed."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "selection_execution_error",
    ) -> None:
        super().__init__(message)
        self.reason_code = str(reason_code or "selection_execution_error")


class SelectionBindingError(SelectionExecutionError):
    """``TestSelectionRef`` does not match the producer selection block."""

    def __init__(self, message: str) -> None:
        super().__init__(message, reason_code="selection_binding_mismatch")


class FallbackWeakeningError(SelectionExecutionError):
    """Attempt to weaken a producer ambiguity/opaque/config/dependency fallback."""

    def __init__(self, message: str) -> None:
        super().__init__(message, reason_code="fallback_weakening_forbidden")


class SelectionTimeout(SelectionExecutionError):
    """A typed execution timeout (never a bare float or string)."""

    def __init__(
        self,
        message: str,
        *,
        timeout: "TypedTimeout",
        command_identity: str = "",
    ) -> None:
        super().__init__(message, reason_code="execution_timeout")
        self.timeout = timeout
        self.command_identity = str(command_identity or "")


class SelectionCancelled(SelectionExecutionError):
    """A typed cooperative cancellation outcome."""

    def __init__(
        self,
        message: str,
        *,
        cancellation_id: str,
        reason: str = "cancelled",
        command_identity: str = "",
    ) -> None:
        super().__init__(message, reason_code="execution_cancelled")
        self.cancellation_id = str(cancellation_id or "")
        self.cancel_reason = str(reason or "cancelled")
        self.command_identity = str(command_identity or "")


class CommandKind(str, Enum):
    """Executable projection kind for one materialized command."""

    STATIC_CHECK = "static_check"
    PYTEST_NODE = "pytest_node"
    FULL_PYTEST = "full_pytest"
    PROOF = "proof"
    FULL_PROOFS = "full_proofs"


# ---------------------------------------------------------------------------
# Closed records
# ---------------------------------------------------------------------------


def _clip(text: str) -> str:
    value = str(text or "").strip() or "unspecified"
    if len(value) > _MAX_DIAGNOSTIC:
        return value[: _MAX_DIAGNOSTIC - 3] + "..."
    return value


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SelectionExecutionError(f"{name} must be a positive number")
    number = float(value)
    if number <= 0.0:
        raise SelectionExecutionError(f"{name} must be positive")
    return number


def _optional_positive_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    return _positive_float(value, name)


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest_identity(prefix: str, payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return f"{prefix}:{digest}"


def _fallback_value(value: Any) -> str:
    if value is None:
        return FALLBACK_NONE
    raw = getattr(value, "value", value)
    text = str(raw).strip()
    if text not in _PRODUCER_FALLBACKS:
        raise SelectionExecutionError(
            f"unsupported producer fallback {text!r}",
            reason_code="unsupported_fallback",
        )
    return text


def _needs_full_pytest(fallback: str) -> bool:
    return fallback in {FALLBACK_FULL_PYTEST, FALLBACK_BOTH}


def _needs_full_proofs(fallback: str) -> bool:
    return fallback in {FALLBACK_FULL_PROOFS, FALLBACK_BOTH}


def combine_fallbacks(*fallbacks: str) -> str:
    """Return the least upper bound of producer/harness fallbacks.

    Escalation is allowed; weakening is never performed by this function.
    """

    need_pytest = any(_needs_full_pytest(_fallback_value(item)) for item in fallbacks)
    need_proofs = any(_needs_full_proofs(_fallback_value(item)) for item in fallbacks)
    if need_pytest and need_proofs:
        return FALLBACK_BOTH
    if need_pytest:
        return FALLBACK_FULL_PYTEST
    if need_proofs:
        return FALLBACK_FULL_PROOFS
    return FALLBACK_NONE


def assert_fallback_not_weakened(*, producer: str, effective: str) -> None:
    """Fail closed when an effective fallback drops producer pytest/proof force."""

    producer_fb = _fallback_value(producer)
    effective_fb = _fallback_value(effective)
    if _needs_full_pytest(producer_fb) and not _needs_full_pytest(effective_fb):
        raise FallbackWeakeningError(
            f"cannot weaken producer fallback {producer_fb!r} to {effective_fb!r}"
        )
    if _needs_full_proofs(producer_fb) and not _needs_full_proofs(effective_fb):
        raise FallbackWeakeningError(
            f"cannot weaken producer fallback {producer_fb!r} to {effective_fb!r}"
        )


@dataclass(frozen=True)
class TypedTimeout:
    """Positive wall-clock budget for one command or stage."""

    seconds: float
    stage: str = "command"

    _FIELDS = frozenset({"seconds", "stage"})

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "seconds", _positive_float(self.seconds, "timeout.seconds")
        )
        object.__setattr__(self, "stage", _text(self.stage, "timeout.stage"))

    def to_dict(self) -> dict[str, Any]:
        return {"seconds": self.seconds, "stage": self.stage}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TypedTimeout":
        payload = _closed(data, cls._FIELDS, "TypedTimeout")
        return cls(seconds=payload["seconds"], stage=payload["stage"])


@dataclass(frozen=True)
class CommandBinding:
    """Exact content-addressed environment a command may execute against."""

    tree_cid: str
    config_cid: str
    dependency_lock_cid: str
    toolchain_cid: str
    policy_cid: str
    interface_cid: str

    _FIELDS = frozenset(
        {
            "tree_cid",
            "config_cid",
            "dependency_lock_cid",
            "toolchain_cid",
            "policy_cid",
            "interface_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "tree_cid", validate_opaque_cid(self.tree_cid, "tree_cid")
        )
        object.__setattr__(
            self, "config_cid", validate_opaque_cid(self.config_cid, "config_cid")
        )
        object.__setattr__(
            self,
            "dependency_lock_cid",
            validate_opaque_cid(self.dependency_lock_cid, "dependency_lock_cid"),
        )
        object.__setattr__(
            self,
            "toolchain_cid",
            validate_opaque_cid(self.toolchain_cid, "toolchain_cid"),
        )
        object.__setattr__(
            self, "policy_cid", validate_opaque_cid(self.policy_cid, "policy_cid")
        )
        object.__setattr__(
            self,
            "interface_cid",
            validate_opaque_cid(self.interface_cid, "interface_cid"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tree_cid": self.tree_cid,
            "config_cid": self.config_cid,
            "dependency_lock_cid": self.dependency_lock_cid,
            "toolchain_cid": self.toolchain_cid,
            "policy_cid": self.policy_cid,
            "interface_cid": self.interface_cid,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CommandBinding":
        payload = _closed(data, cls._FIELDS, "CommandBinding")
        return cls(**payload)

    @property
    def binding_cid(self) -> str:
        return _digest_identity(
            "sch-bind",
            {"schema": COMMAND_BINDING_SCHEMA, **self.to_dict()},
        )


@dataclass(frozen=True)
class CommandProvenance:
    """Auditable origin of one materialized command.

    Retains producer reason-path CIDs and the sealed selection/root CIDs.  It
    does not re-derive graph edges or invent a second selection.
    """

    selection_cid: str
    previous_semantic_state_root_cid: str | None
    current_semantic_state_root_cid: str
    producer_fallback: str
    effective_fallback: str
    fallback_reasons: tuple[str, ...]
    reason_path_cids: tuple[str, ...]
    binding: CommandBinding
    covered_seed_obligation_ids: tuple[str, ...] = ()
    unresolved_obligation_ids: tuple[str, ...] = ()
    known_test_universe_cid: str | None = None
    known_test_universe_count: int = 0
    policy_cid: str | None = None

    _FIELDS = frozenset(
        {
            "selection_cid",
            "previous_semantic_state_root_cid",
            "current_semantic_state_root_cid",
            "producer_fallback",
            "effective_fallback",
            "fallback_reasons",
            "reason_path_cids",
            "binding",
            "covered_seed_obligation_ids",
            "unresolved_obligation_ids",
            "known_test_universe_cid",
            "known_test_universe_count",
            "policy_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "selection_cid",
            validate_opaque_cid(self.selection_cid, "selection_cid"),
        )
        object.__setattr__(
            self,
            "previous_semantic_state_root_cid",
            _optional_cid(
                self.previous_semantic_state_root_cid,
                "previous_semantic_state_root_cid",
            ),
        )
        object.__setattr__(
            self,
            "current_semantic_state_root_cid",
            validate_opaque_cid(
                self.current_semantic_state_root_cid,
                "current_semantic_state_root_cid",
            ),
        )
        producer = _fallback_value(self.producer_fallback)
        effective = _fallback_value(self.effective_fallback)
        assert_fallback_not_weakened(producer=producer, effective=effective)
        object.__setattr__(self, "producer_fallback", producer)
        object.__setattr__(self, "effective_fallback", effective)
        object.__setattr__(
            self,
            "fallback_reasons",
            _unique_sorted_texts(list(self.fallback_reasons), "fallback_reasons"),
        )
        # Reason path CIDs are opaque producer identities; accept CID alphabet.
        paths: list[str] = []
        for item in self.reason_path_cids:
            paths.append(validate_opaque_cid(item, "reason_path_cid"))
        object.__setattr__(self, "reason_path_cids", tuple(sorted(set(paths))))
        if not isinstance(self.binding, CommandBinding):
            raise SelectionExecutionError("binding must be a CommandBinding")
        object.__setattr__(
            self,
            "covered_seed_obligation_ids",
            _unique_sorted_texts(
                list(self.covered_seed_obligation_ids), "covered_seed_obligation_ids"
            ),
        )
        object.__setattr__(
            self,
            "unresolved_obligation_ids",
            _unique_sorted_texts(
                list(self.unresolved_obligation_ids), "unresolved_obligation_ids"
            ),
        )
        object.__setattr__(
            self,
            "known_test_universe_cid",
            _optional_cid(self.known_test_universe_cid, "known_test_universe_cid"),
        )
        object.__setattr__(
            self,
            "known_test_universe_count",
            _nonneg_int(self.known_test_universe_count, "known_test_universe_count"),
        )
        object.__setattr__(
            self, "policy_cid", _optional_cid(self.policy_cid, "policy_cid")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "selection_cid": self.selection_cid,
            "previous_semantic_state_root_cid": self.previous_semantic_state_root_cid,
            "current_semantic_state_root_cid": self.current_semantic_state_root_cid,
            "producer_fallback": self.producer_fallback,
            "effective_fallback": self.effective_fallback,
            "fallback_reasons": list(self.fallback_reasons),
            "reason_path_cids": list(self.reason_path_cids),
            "binding": self.binding.to_dict(),
            "covered_seed_obligation_ids": list(self.covered_seed_obligation_ids),
            "unresolved_obligation_ids": list(self.unresolved_obligation_ids),
            "known_test_universe_cid": self.known_test_universe_cid,
            "known_test_universe_count": self.known_test_universe_count,
            "policy_cid": self.policy_cid,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CommandProvenance":
        payload = _closed(data, cls._FIELDS, "CommandProvenance")
        binding = payload["binding"]
        if not isinstance(binding, Mapping):
            raise SelectionExecutionError("binding must be an object")
        return cls(
            selection_cid=payload["selection_cid"],
            previous_semantic_state_root_cid=payload[
                "previous_semantic_state_root_cid"
            ],
            current_semantic_state_root_cid=payload[
                "current_semantic_state_root_cid"
            ],
            producer_fallback=payload["producer_fallback"],
            effective_fallback=payload["effective_fallback"],
            fallback_reasons=tuple(payload["fallback_reasons"]),
            reason_path_cids=tuple(payload["reason_path_cids"]),
            binding=CommandBinding.from_dict(binding),
            covered_seed_obligation_ids=tuple(
                payload["covered_seed_obligation_ids"]
            ),
            unresolved_obligation_ids=tuple(payload["unresolved_obligation_ids"]),
            known_test_universe_cid=payload["known_test_universe_cid"],
            known_test_universe_count=payload["known_test_universe_count"],
            policy_cid=payload["policy_cid"],
        )


@dataclass(frozen=True)
class HarnessAssurancePolicy:
    """Harness-side assurance that may only escalate producer fallbacks.

    Ambiguity, opaque reachability, config/dependency, and producer-declared
    full-suite fallbacks cannot be weakened by this policy.
    """

    policy_id: str = "default-assurance"
    require_static_checks: bool = False
    force_full_pytest: bool = False
    force_full_proofs: bool = False
    static_check_commands: tuple[str, ...] = ()
    full_pytest_command: str = "python3.12 -m pytest -q"
    pytest_timeout_seconds: float = _DEFAULT_PYTEST_TIMEOUT_SECONDS
    proof_timeout_seconds: float = _DEFAULT_PROOF_TIMEOUT_SECONDS
    static_timeout_seconds: float = _DEFAULT_STATIC_TIMEOUT_SECONDS
    allow_empty_selection: bool = True

    _FIELDS = frozenset(
        {
            "policy_id",
            "require_static_checks",
            "force_full_pytest",
            "force_full_proofs",
            "static_check_commands",
            "full_pytest_command",
            "pytest_timeout_seconds",
            "proof_timeout_seconds",
            "static_timeout_seconds",
            "allow_empty_selection",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _text(self.policy_id, "policy_id"))
        for name in (
            "require_static_checks",
            "force_full_pytest",
            "force_full_proofs",
            "allow_empty_selection",
        ):
            value = getattr(self, name)
            if type(value) is not bool:
                raise SelectionExecutionError(f"{name} must be a bool")
        commands = tuple(
            _text(item, "static_check_command") for item in self.static_check_commands
        )
        if len(set(commands)) != len(commands):
            raise SelectionExecutionError(
                "static_check_commands must not contain duplicates"
            )
        object.__setattr__(self, "static_check_commands", commands)
        object.__setattr__(
            self,
            "full_pytest_command",
            _text(self.full_pytest_command, "full_pytest_command"),
        )
        object.__setattr__(
            self,
            "pytest_timeout_seconds",
            _positive_float(self.pytest_timeout_seconds, "pytest_timeout_seconds"),
        )
        object.__setattr__(
            self,
            "proof_timeout_seconds",
            _positive_float(self.proof_timeout_seconds, "proof_timeout_seconds"),
        )
        object.__setattr__(
            self,
            "static_timeout_seconds",
            _positive_float(self.static_timeout_seconds, "static_timeout_seconds"),
        )
        if self.require_static_checks and not self.static_check_commands:
            raise SelectionExecutionError(
                "require_static_checks needs at least one static_check_command"
            )

    def assurance_fallback(self) -> str:
        if self.force_full_pytest and self.force_full_proofs:
            return FALLBACK_BOTH
        if self.force_full_pytest:
            return FALLBACK_FULL_PYTEST
        if self.force_full_proofs:
            return FALLBACK_FULL_PROOFS
        return FALLBACK_NONE

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "require_static_checks": self.require_static_checks,
            "force_full_pytest": self.force_full_pytest,
            "force_full_proofs": self.force_full_proofs,
            "static_check_commands": list(self.static_check_commands),
            "full_pytest_command": self.full_pytest_command,
            "pytest_timeout_seconds": self.pytest_timeout_seconds,
            "proof_timeout_seconds": self.proof_timeout_seconds,
            "static_timeout_seconds": self.static_timeout_seconds,
            "allow_empty_selection": self.allow_empty_selection,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HarnessAssurancePolicy":
        payload = _closed(data, cls._FIELDS, "HarnessAssurancePolicy")
        return cls(
            policy_id=payload["policy_id"],
            require_static_checks=payload["require_static_checks"],
            force_full_pytest=payload["force_full_pytest"],
            force_full_proofs=payload["force_full_proofs"],
            static_check_commands=tuple(payload["static_check_commands"]),
            full_pytest_command=payload["full_pytest_command"],
            pytest_timeout_seconds=payload["pytest_timeout_seconds"],
            proof_timeout_seconds=payload["proof_timeout_seconds"],
            static_timeout_seconds=payload["static_timeout_seconds"],
            allow_empty_selection=payload["allow_empty_selection"],
        )


@dataclass(frozen=True)
class MaterializedCommand:
    """One bounded execute unit projected from a sealed selection."""

    kind: str
    command_identity: str
    shell_command: str | None
    validation_command: ValidationCommand | None
    proof_id: str | None
    target_ids: tuple[str, ...]
    provenance: CommandProvenance
    timeout: TypedTimeout
    ordinal: int = 0

    def __post_init__(self) -> None:
        try:
            kind = CommandKind(str(self.kind))
        except ValueError as exc:
            raise SelectionExecutionError(
                f"unsupported command kind {self.kind!r}"
            ) from exc
        object.__setattr__(self, "kind", kind.value)
        object.__setattr__(
            self,
            "command_identity",
            _text(self.command_identity, "command_identity"),
        )
        if self.shell_command is not None:
            object.__setattr__(
                self, "shell_command", _text(self.shell_command, "shell_command")
            )
        if self.proof_id is not None:
            object.__setattr__(self, "proof_id", _text(self.proof_id, "proof_id"))
        object.__setattr__(
            self, "target_ids", _unique_sorted_texts(list(self.target_ids), "target_ids")
        )
        if not isinstance(self.provenance, CommandProvenance):
            raise SelectionExecutionError("provenance must be CommandProvenance")
        if not isinstance(self.timeout, TypedTimeout):
            raise SelectionExecutionError("timeout must be TypedTimeout")
        object.__setattr__(self, "ordinal", _nonneg_int(self.ordinal, "ordinal"))
        if kind in {
            CommandKind.STATIC_CHECK,
            CommandKind.PYTEST_NODE,
            CommandKind.FULL_PYTEST,
        }:
            if self.validation_command is None or not self.shell_command:
                raise SelectionExecutionError(
                    f"{kind.value} commands require shell validation_command"
                )
        if kind in {CommandKind.PROOF, CommandKind.FULL_PROOFS}:
            if not self.proof_id and not self.target_ids:
                raise SelectionExecutionError(
                    f"{kind.value} commands require proof target ids"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "command_identity": self.command_identity,
            "shell_command": self.shell_command,
            "proof_id": self.proof_id,
            "target_ids": list(self.target_ids),
            "provenance": self.provenance.to_dict(),
            "timeout": self.timeout.to_dict(),
            "ordinal": self.ordinal,
            "validation_stage": (
                self.validation_command.stage.label
                if self.validation_command is not None
                else None
            ),
        }


@dataclass(frozen=True)
class MaterializedSelectionPlan:
    """Complete command projection for one sealed selection."""

    selection_ref: TestSelectionRef
    producer_fallback: str
    effective_fallback: str
    fallback_reasons: tuple[str, ...]
    commands: tuple[MaterializedCommand, ...]
    binding: CommandBinding
    reason_path_cids: tuple[str, ...]
    selected_pytest_node_ids: tuple[str, ...]
    selected_proof_ids: tuple[str, ...]
    assurance_policy_id: str

    def validation_commands(self) -> tuple[ValidationCommand, ...]:
        """Return only shell validation commands (static + pytest stages)."""

        return tuple(
            command.validation_command
            for command in self.commands
            if command.validation_command is not None
        )

    def proof_ids(self) -> tuple[str, ...]:
        return tuple(
            command.proof_id
            for command in self.commands
            if command.proof_id is not None
            and command.kind
            in {CommandKind.PROOF.value, CommandKind.FULL_PROOFS.value}
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "selection_ref": self.selection_ref.to_dict(),
            "producer_fallback": self.producer_fallback,
            "effective_fallback": self.effective_fallback,
            "fallback_reasons": list(self.fallback_reasons),
            "commands": [item.to_dict() for item in self.commands],
            "binding": self.binding.to_dict(),
            "reason_path_cids": list(self.reason_path_cids),
            "selected_pytest_node_ids": list(self.selected_pytest_node_ids),
            "selected_proof_ids": list(self.selected_proof_ids),
            "assurance_policy_id": self.assurance_policy_id,
            "schema": SELECTION_EXECUTION_SCHEMA,
            "interface": SELECTION_EXECUTION_INTERFACE,
        }


# ---------------------------------------------------------------------------
# Selection dereference and verification
# ---------------------------------------------------------------------------


def selection_ref_from_selection(selection: Any) -> TestSelectionRef:
    """Build a harness ``TestSelectionRef`` from a producer selection object."""

    selection_cid = getattr(selection, "selection_cid", None)
    if not isinstance(selection_cid, str):
        raise SelectionExecutionError("selection missing selection_cid")
    previous = getattr(selection, "previous_root_cid", None)
    if previous is None and hasattr(selection, "previous_semantic_state_root_cid"):
        previous = getattr(selection, "previous_semantic_state_root_cid")
    current = getattr(selection, "current_root_cid", None)
    if current is None and hasattr(selection, "current_semantic_state_root_cid"):
        current = getattr(selection, "current_semantic_state_root_cid")
    if not isinstance(current, str):
        raise SelectionExecutionError("selection missing current_root_cid")
    return TestSelectionRef.from_dict(
        {
            "selection_cid": selection_cid,
            "previous_semantic_state_root_cid": previous,
            "current_semantic_state_root_cid": current,
        }
    )


def verify_selection_binding(
    selection_ref: TestSelectionRef,
    selection: Any,
) -> None:
    """Verify ``selection_ref`` matches the producer selection root bindings.

    Does not re-select tests.  Only checks identity fields the harness is
    allowed to admit.
    """

    if not isinstance(selection_ref, TestSelectionRef):
        raise SelectionExecutionError("selection_ref must be a TestSelectionRef")
    claimed = selection_ref_from_selection(selection)
    if claimed.selection_cid != selection_ref.selection_cid:
        raise SelectionBindingError(
            "selection_cid mismatch: "
            f"ref={selection_ref.selection_cid!r} selection={claimed.selection_cid!r}"
        )
    if (
        claimed.previous_semantic_state_root_cid
        != selection_ref.previous_semantic_state_root_cid
    ):
        raise SelectionBindingError(
            "previous_semantic_state_root_cid mismatch: "
            f"ref={selection_ref.previous_semantic_state_root_cid!r} "
            f"selection={claimed.previous_semantic_state_root_cid!r}"
        )
    if (
        claimed.current_semantic_state_root_cid
        != selection_ref.current_semantic_state_root_cid
    ):
        raise SelectionBindingError(
            "current_semantic_state_root_cid mismatch: "
            f"ref={selection_ref.current_semantic_state_root_cid!r} "
            f"selection={claimed.current_semantic_state_root_cid!r}"
        )


def _reason_path_cids(selection: Any) -> tuple[str, ...]:
    paths = getattr(selection, "reason_paths", ()) or ()
    cids: list[str] = []
    for item in paths:
        if isinstance(item, Mapping):
            cid = item.get("path_cid")
        else:
            cid = getattr(item, "path_cid", None)
        if isinstance(cid, str) and cid:
            cids.append(validate_opaque_cid(cid, "reason_path_cid"))
    return tuple(sorted(set(cids)))


def _string_sequence(value: Any, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        raise SelectionExecutionError(f"{name} must be a sequence of strings")
    return _unique_sorted_texts(list(value), name)


def producer_fallback_of(selection: Any) -> str:
    return _fallback_value(getattr(selection, "fallback", FALLBACK_NONE))


def producer_fallback_reasons(selection: Any) -> tuple[str, ...]:
    return _string_sequence(
        getattr(selection, "fallback_reasons", ()) or (), "fallback_reasons"
    )


def selected_pytest_node_ids_of(selection: Any) -> tuple[str, ...]:
    return _string_sequence(
        getattr(selection, "selected_pytest_node_ids", ()) or (),
        "selected_pytest_node_ids",
    )


def selected_proof_ids_of(selection: Any) -> tuple[str, ...]:
    return _string_sequence(
        getattr(selection, "selected_proof_ids", ()) or (),
        "selected_proof_ids",
    )


def effective_fallback_for(
    selection: Any,
    assurance: HarnessAssurancePolicy | None = None,
) -> str:
    """Combine producer fallback with harness assurance (escalate only)."""

    producer = producer_fallback_of(selection)
    policy = assurance or HarnessAssurancePolicy()
    effective = combine_fallbacks(producer, policy.assurance_fallback())
    assert_fallback_not_weakened(producer=producer, effective=effective)
    # Explicit guard: opaque/config/dependency producer reasons stay attached.
    return effective


def _command_identity(
    *,
    kind: str,
    shell_command: str | None,
    proof_id: str | None,
    target_ids: Sequence[str],
    provenance: CommandProvenance,
    timeout: TypedTimeout,
    ordinal: int,
) -> str:
    return _digest_identity(
        "sch-cmd",
        {
            "schema": SELECTION_EXECUTION_SCHEMA,
            "kind": kind,
            "shell_command": shell_command,
            "proof_id": proof_id,
            "target_ids": list(target_ids),
            "selection_cid": provenance.selection_cid,
            "binding": provenance.binding.to_dict(),
            "timeout": timeout.to_dict(),
            "ordinal": ordinal,
            "producer_fallback": provenance.producer_fallback,
            "effective_fallback": provenance.effective_fallback,
        },
    )


def _pytest_node_shell(node_id: str) -> str:
    # Authoritative node IDs are already selected by datasets; never invent them.
    return f"python3.12 -m pytest -q -- {shlex.quote(node_id)}"


def _build_validation_command(
    *,
    shell: str,
    stage: ValidationStage,
    timeout_seconds: float,
    ordinal: int,
    validation_id: str,
    requirement_kind: ValidationRequirementKind | None,
    verdict_kind: ValidationVerdictKind,
    fallback: bool = False,
) -> ValidationCommand:
    return ValidationCommand(
        command=shell,
        raw_command=shell,
        stage=stage,
        resource_cost=1,
        impact_paths=(),
        environment_keys=(),
        cacheable=True,
        timeout_seconds=timeout_seconds,
        ordinal=ordinal,
        validation_id=validation_id,
        requirement_kind=requirement_kind,
        verdict_kind=verdict_kind,
        source="sealed_selection",
        fallback=fallback,
    )


def materialize_selection_commands(
    selection: Any,
    *,
    selection_ref: TestSelectionRef | None = None,
    binding: CommandBinding,
    assurance: HarnessAssurancePolicy | None = None,
) -> MaterializedSelectionPlan:
    """Project a sealed producer selection into explicit bounded commands.

    Never walks the semantic graph or reselects affected tests/proofs.
    """

    policy = assurance or HarnessAssurancePolicy()
    ref = selection_ref or selection_ref_from_selection(selection)
    verify_selection_binding(ref, selection)

    producer_fb = producer_fallback_of(selection)
    effective_fb = effective_fallback_for(selection, policy)
    reasons = producer_fallback_reasons(selection)
    path_cids = _reason_path_cids(selection)
    pytest_nodes = selected_pytest_node_ids_of(selection)
    proof_ids = selected_proof_ids_of(selection)

    if (
        not policy.allow_empty_selection
        and not _needs_full_pytest(effective_fb)
        and not pytest_nodes
        and not _needs_full_proofs(effective_fb)
        and not proof_ids
        and not policy.require_static_checks
    ):
        raise SelectionExecutionError(
            "empty selection with no fallback and allow_empty_selection=False",
            reason_code="empty_selection",
        )

    provenance = CommandProvenance(
        selection_cid=ref.selection_cid,
        previous_semantic_state_root_cid=ref.previous_semantic_state_root_cid,
        current_semantic_state_root_cid=ref.current_semantic_state_root_cid,
        producer_fallback=producer_fb,
        effective_fallback=effective_fb,
        fallback_reasons=reasons,
        reason_path_cids=path_cids,
        binding=binding,
        covered_seed_obligation_ids=_string_sequence(
            getattr(selection, "covered_seed_obligation_ids", ()) or (),
            "covered_seed_obligation_ids",
        ),
        unresolved_obligation_ids=_string_sequence(
            getattr(selection, "unresolved_obligation_ids", ()) or (),
            "unresolved_obligation_ids",
        ),
        known_test_universe_cid=_optional_cid(
            getattr(selection, "known_test_universe_cid", None),
            "known_test_universe_cid",
        ),
        known_test_universe_count=int(
            getattr(selection, "known_test_universe_count", 0) or 0
        ),
        policy_cid=_optional_cid(getattr(selection, "policy_cid", None), "policy_cid"),
    )

    commands: list[MaterializedCommand] = []
    ordinal = 0

    if policy.require_static_checks:
        for shell in policy.static_check_commands:
            timeout = TypedTimeout(
                seconds=policy.static_timeout_seconds, stage="static_check"
            )
            validation_id = f"static:{ordinal}"
            validation = _build_validation_command(
                shell=shell,
                stage=ValidationStage.CHEAP,
                timeout_seconds=timeout.seconds,
                ordinal=ordinal,
                validation_id=validation_id,
                requirement_kind=ValidationRequirementKind.STATIC_CHECK,
                verdict_kind=ValidationVerdictKind.DETERMINISTIC,
            )
            identity = _command_identity(
                kind=CommandKind.STATIC_CHECK.value,
                shell_command=shell,
                proof_id=None,
                target_ids=(),
                provenance=provenance,
                timeout=timeout,
                ordinal=ordinal,
            )
            commands.append(
                MaterializedCommand(
                    kind=CommandKind.STATIC_CHECK.value,
                    command_identity=identity,
                    shell_command=shell,
                    validation_command=validation,
                    proof_id=None,
                    target_ids=(),
                    provenance=provenance,
                    timeout=timeout,
                    ordinal=ordinal,
                )
            )
            ordinal += 1

    if _needs_full_pytest(effective_fb):
        shell = policy.full_pytest_command
        timeout = TypedTimeout(
            seconds=policy.pytest_timeout_seconds, stage="full_pytest"
        )
        validation = _build_validation_command(
            shell=shell,
            stage=ValidationStage.BROAD,
            timeout_seconds=timeout.seconds,
            ordinal=ordinal,
            validation_id=f"full_pytest:{ordinal}",
            requirement_kind=ValidationRequirementKind.FOCUSED_TEST,
            verdict_kind=ValidationVerdictKind.TEST,
            fallback=True,
        )
        identity = _command_identity(
            kind=CommandKind.FULL_PYTEST.value,
            shell_command=shell,
            proof_id=None,
            target_ids=pytest_nodes,
            provenance=provenance,
            timeout=timeout,
            ordinal=ordinal,
        )
        commands.append(
            MaterializedCommand(
                kind=CommandKind.FULL_PYTEST.value,
                command_identity=identity,
                shell_command=shell,
                validation_command=validation,
                proof_id=None,
                target_ids=pytest_nodes,
                provenance=provenance,
                timeout=timeout,
                ordinal=ordinal,
            )
        )
        ordinal += 1
    else:
        for node_id in pytest_nodes:
            shell = _pytest_node_shell(node_id)
            timeout = TypedTimeout(
                seconds=policy.pytest_timeout_seconds, stage="pytest_node"
            )
            validation = _build_validation_command(
                shell=shell,
                stage=ValidationStage.TARGETED,
                timeout_seconds=timeout.seconds,
                ordinal=ordinal,
                validation_id=f"pytest:{node_id}",
                requirement_kind=ValidationRequirementKind.FOCUSED_TEST,
                verdict_kind=ValidationVerdictKind.TEST,
            )
            identity = _command_identity(
                kind=CommandKind.PYTEST_NODE.value,
                shell_command=shell,
                proof_id=None,
                target_ids=(node_id,),
                provenance=provenance,
                timeout=timeout,
                ordinal=ordinal,
            )
            commands.append(
                MaterializedCommand(
                    kind=CommandKind.PYTEST_NODE.value,
                    command_identity=identity,
                    shell_command=shell,
                    validation_command=validation,
                    proof_id=None,
                    target_ids=(node_id,),
                    provenance=provenance,
                    timeout=timeout,
                    ordinal=ordinal,
                )
            )
            ordinal += 1

    if _needs_full_proofs(effective_fb):
        # Full-proofs fallback: emit one obligation per selected proof ID when
        # present; empty selected_proof_ids still yields a typed full_proofs
        # marker command so callers cannot treat the absence as "passed".
        targets = proof_ids if proof_ids else ("*",)
        for proof_id in targets:
            timeout = TypedTimeout(
                seconds=policy.proof_timeout_seconds, stage="full_proofs"
            )
            identity = _command_identity(
                kind=CommandKind.FULL_PROOFS.value,
                shell_command=None,
                proof_id=proof_id,
                target_ids=(proof_id,),
                provenance=provenance,
                timeout=timeout,
                ordinal=ordinal,
            )
            commands.append(
                MaterializedCommand(
                    kind=CommandKind.FULL_PROOFS.value,
                    command_identity=identity,
                    shell_command=None,
                    validation_command=None,
                    proof_id=proof_id,
                    target_ids=(proof_id,),
                    provenance=provenance,
                    timeout=timeout,
                    ordinal=ordinal,
                )
            )
            ordinal += 1
    else:
        for proof_id in proof_ids:
            timeout = TypedTimeout(
                seconds=policy.proof_timeout_seconds, stage="proof"
            )
            identity = _command_identity(
                kind=CommandKind.PROOF.value,
                shell_command=None,
                proof_id=proof_id,
                target_ids=(proof_id,),
                provenance=provenance,
                timeout=timeout,
                ordinal=ordinal,
            )
            commands.append(
                MaterializedCommand(
                    kind=CommandKind.PROOF.value,
                    command_identity=identity,
                    shell_command=None,
                    validation_command=None,
                    proof_id=proof_id,
                    target_ids=(proof_id,),
                    provenance=provenance,
                    timeout=timeout,
                    ordinal=ordinal,
                )
            )
            ordinal += 1

    if len(commands) > _MAX_COMMANDS:
        raise SelectionExecutionError(
            f"materialized command count exceeds {_MAX_COMMANDS}",
            reason_code="command_budget_exceeded",
        )

    return MaterializedSelectionPlan(
        selection_ref=ref,
        producer_fallback=producer_fb,
        effective_fallback=effective_fb,
        fallback_reasons=reasons,
        commands=tuple(commands),
        binding=binding,
        reason_path_cids=path_cids,
        selected_pytest_node_ids=pytest_nodes,
        selected_proof_ids=proof_ids,
        assurance_policy_id=policy.policy_id,
    )


# ---------------------------------------------------------------------------
# Execution adapter (projects onto existing schedulers; no reselection)
# ---------------------------------------------------------------------------


def _raise_if_cancelled(
    cancellation: CancellationToken | None,
    *,
    command_identity: str = "",
) -> None:
    if cancellation is None:
        return
    if cancellation.is_cancelled():
        raise SelectionCancelled(
            f"selection execution cancelled: {cancellation.reason or 'cancelled'}",
            cancellation_id=cancellation.cancellation_id,
            reason=cancellation.reason or "cancelled",
            command_identity=command_identity,
        )


@dataclass
class SelectionExecutionAdapter:
    """Adapt a sealed selection into ``ValidationScheduler`` / ``ProofScheduler``.

    The adapter never imports ``run_impact_selected`` and never performs graph
    traversal.  Callers inject already-constructed scheduler instances.
    """

    validation_scheduler: Any | None = None
    proof_scheduler_factory: Callable[..., Any] | None = None
    assurance: HarnessAssurancePolicy = field(default_factory=HarnessAssurancePolicy)
    adapter_id: str = ADAPTER_ID

    def materialize(
        self,
        selection: Any,
        *,
        binding: CommandBinding,
        selection_ref: TestSelectionRef | None = None,
        assurance: HarnessAssurancePolicy | None = None,
    ) -> MaterializedSelectionPlan:
        return materialize_selection_commands(
            selection,
            selection_ref=selection_ref,
            binding=binding,
            assurance=assurance or self.assurance,
        )

    def run_validation_stage(
        self,
        plan: MaterializedSelectionPlan,
        *,
        workspace_path: Path | str,
        cancellation: CancellationToken | None = None,
        environment: Mapping[str, object] | None = None,
        runner: Any | None = None,
        require_full_validation: bool = False,
    ) -> dict[str, Any]:
        """Run static/pytest commands via ``ValidationScheduler.run_staged``.

        Explicit already-selected commands only.  ``changed_files`` is forced
        empty at the scheduler boundary so impact reselection cannot occur.
        """

        _raise_if_cancelled(cancellation)
        if self.validation_scheduler is None:
            return {
                "attempted": False,
                "passed": False,
                "unavailable": True,
                "reason": "validation_scheduler_unavailable",
                "adapter_id": self.adapter_id,
            }
        commands = plan.validation_commands()
        if not commands:
            return {
                "attempted": False,
                "passed": True,
                "returncode": 0,
                "results": [],
                "reason": "no_validation_commands",
                "selection_cid": plan.selection_ref.selection_cid,
                "binding": plan.binding.to_dict(),
            }

        # Defense in depth: never pass graph/impact selectors.
        run_staged = getattr(self.validation_scheduler, "run_staged", None)
        if not callable(run_staged):
            raise SelectionExecutionError(
                "validation_scheduler must provide run_staged",
                reason_code="scheduler_contract",
            )
        if hasattr(self.validation_scheduler, "run_impact_selected"):
            # Presence is fine (shared class); we must not call it.
            pass

        report = run_staged(
            commands,
            workspace_path=workspace_path,
            # Empty changed_files disables second-pass impact selection inside
            # run_staged/run (see ValidationScheduler.run_staged docstring).
            changed_files=(),
            environment=environment,
            require_full_validation=require_full_validation
            or _needs_full_pytest(plan.effective_fallback),
            runner=runner,
        )
        if not isinstance(report, Mapping):
            raise SelectionExecutionError(
                "run_staged must return a mapping report",
                reason_code="scheduler_contract",
            )
        enriched = dict(report)
        enriched.setdefault("selection_cid", plan.selection_ref.selection_cid)
        enriched.setdefault(
            "current_semantic_state_root_cid",
            plan.selection_ref.current_semantic_state_root_cid,
        )
        enriched.setdefault(
            "previous_semantic_state_root_cid",
            plan.selection_ref.previous_semantic_state_root_cid,
        )
        enriched.setdefault("binding", plan.binding.to_dict())
        enriched.setdefault("producer_fallback", plan.producer_fallback)
        enriched.setdefault("effective_fallback", plan.effective_fallback)
        enriched.setdefault("fallback_reasons", list(plan.fallback_reasons))
        enriched.setdefault("reason_path_cids", list(plan.reason_path_cids))
        enriched.setdefault(
            "command_identities",
            [
                item.command_identity
                for item in plan.commands
                if item.validation_command is not None
            ],
        )
        return enriched

    def run_proofs(
        self,
        plan: MaterializedSelectionPlan,
        *,
        cancellation: CancellationToken | None = None,
        prover_available: bool | Callable[[str], bool] | None = None,
        proof_executor: Callable[[str], Mapping[str, Any]] | None = None,
    ) -> tuple[dict[str, Any], ...]:
        """Execute or capability-probe selected proofs.

        Unavailable provers yield typed unavailable results and are never
        reported as passed proofs.
        """

        _raise_if_cancelled(cancellation)
        results: list[dict[str, Any]] = []
        for command in plan.commands:
            if command.kind not in {
                CommandKind.PROOF.value,
                CommandKind.FULL_PROOFS.value,
            }:
                continue
            _raise_if_cancelled(
                cancellation, command_identity=command.command_identity
            )
            proof_id = command.proof_id or (
                command.target_ids[0] if command.target_ids else ""
            )
            available = True
            if prover_available is None:
                available = proof_executor is not None or (
                    self.proof_scheduler_factory is not None
                )
            elif callable(prover_available):
                available = bool(prover_available(proof_id))
            else:
                available = bool(prover_available)

            if not available:
                results.append(
                    UnavailableResult.from_dict(
                        {
                            "operation": "prove",
                            "adapter_id": self.adapter_id,
                            "reason_code": "prover_unavailable",
                            "retryable": True,
                            "diagnostic": _clip(
                                f"prover unavailable for proof_id={proof_id}"
                            ),
                        }
                    ).to_dict()
                    | {
                        "proof_id": proof_id,
                        "status": "unavailable",
                        "passed": False,
                        "command_identity": command.command_identity,
                        "selection_cid": plan.selection_ref.selection_cid,
                        "binding": plan.binding.to_dict(),
                        "timeout": command.timeout.to_dict(),
                    }
                )
                continue

            if proof_executor is None:
                results.append(
                    {
                        "proof_id": proof_id,
                        "status": "unavailable",
                        "passed": False,
                        "reason_code": "proof_executor_missing",
                        "command_identity": command.command_identity,
                        "selection_cid": plan.selection_ref.selection_cid,
                        "binding": plan.binding.to_dict(),
                    }
                )
                continue

            try:
                raw = proof_executor(proof_id)
            except SelectionTimeout:
                raise
            except SelectionCancelled:
                raise
            except Exception as exc:
                results.append(
                    {
                        "proof_id": proof_id,
                        "status": "failed",
                        "passed": False,
                        "error": _clip(f"{type(exc).__name__}:{exc}"),
                        "command_identity": command.command_identity,
                        "selection_cid": plan.selection_ref.selection_cid,
                        "binding": plan.binding.to_dict(),
                        "timeout": command.timeout.to_dict(),
                    }
                )
                continue

            if not isinstance(raw, Mapping):
                raise SelectionExecutionError(
                    "proof_executor must return a mapping",
                    reason_code="proof_executor_contract",
                )
            status = str(raw.get("status") or ("passed" if raw.get("passed") else "failed"))
            if status == "passed" and raw.get("unavailable"):
                status = "unavailable"
            passed = status == "passed" and not raw.get("unavailable")
            # Hard rule: unavailable never becomes passed.
            if status == "unavailable":
                passed = False
            results.append(
                {
                    **dict(raw),
                    "proof_id": proof_id,
                    "status": status,
                    "passed": passed,
                    "command_identity": command.command_identity,
                    "selection_cid": plan.selection_ref.selection_cid,
                    "binding": plan.binding.to_dict(),
                    "timeout": command.timeout.to_dict(),
                    "reason_path_cids": list(plan.reason_path_cids),
                }
            )
        return tuple(results)

    def execute(
        self,
        selection: Any,
        *,
        binding: CommandBinding,
        workspace_path: Path | str,
        selection_ref: TestSelectionRef | None = None,
        assurance: HarnessAssurancePolicy | None = None,
        cancellation: CancellationToken | None = None,
        environment: Mapping[str, object] | None = None,
        runner: Any | None = None,
        prover_available: bool | Callable[[str], bool] | None = None,
        proof_executor: Callable[[str], Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Materialize then run validation and proof projections."""

        plan = self.materialize(
            selection,
            binding=binding,
            selection_ref=selection_ref,
            assurance=assurance,
        )
        validation_report = self.run_validation_stage(
            plan,
            workspace_path=workspace_path,
            cancellation=cancellation,
            environment=environment,
            runner=runner,
        )
        proof_results = self.run_proofs(
            plan,
            cancellation=cancellation,
            prover_available=prover_available,
            proof_executor=proof_executor,
        )
        return {
            "plan": plan.to_dict(),
            "validation": validation_report,
            "proofs": list(proof_results),
            "selection_cid": plan.selection_ref.selection_cid,
            "binding": plan.binding.to_dict(),
            "producer_fallback": plan.producer_fallback,
            "effective_fallback": plan.effective_fallback,
            "fallback_reasons": list(plan.fallback_reasons),
            "reason_path_cids": list(plan.reason_path_cids),
        }


def selection_execution_descriptor() -> dict[str, Any]:
    """Profile-A style descriptor for the selection execution surface."""

    return {
        "interface": SELECTION_EXECUTION_INTERFACE,
        "schema": SELECTION_EXECUTION_SCHEMA,
        "adapter_id": ADAPTER_ID,
        "board_namespace": BOARD_NAMESPACE,
        "operations": (
            "verify_selection_binding",
            "materialize_selection_commands",
            "execute",
        ),
        "forbids": (
            "run_impact_selected",
            "graph_traversal",
            "reselection",
            "guess_node_ids",
            "weaken_producer_fallback",
        ),
    }


__all__ = [
    "ADAPTER_ID",
    "ASSURANCE_POLICY_SCHEMA",
    "COMMAND_BINDING_SCHEMA",
    "COMMAND_PROVENANCE_SCHEMA",
    "CommandBinding",
    "CommandKind",
    "CommandProvenance",
    "FALLBACK_BOTH",
    "FALLBACK_FULL_PROOFS",
    "FALLBACK_FULL_PYTEST",
    "FALLBACK_NONE",
    "FallbackWeakeningError",
    "HarnessAssurancePolicy",
    "MaterializedCommand",
    "MaterializedSelectionPlan",
    "SELECTION_EXECUTION_INTERFACE",
    "SELECTION_EXECUTION_SCHEMA",
    "SelectionBindingError",
    "SelectionCancelled",
    "SelectionExecutionAdapter",
    "SelectionExecutionError",
    "SelectionTimeout",
    "TypedTimeout",
    "assert_fallback_not_weakened",
    "combine_fallbacks",
    "effective_fallback_for",
    "materialize_selection_commands",
    "producer_fallback_of",
    "producer_fallback_reasons",
    "selected_proof_ids_of",
    "selected_pytest_node_ids_of",
    "selection_execution_descriptor",
    "selection_ref_from_selection",
    "verify_selection_binding",
]
