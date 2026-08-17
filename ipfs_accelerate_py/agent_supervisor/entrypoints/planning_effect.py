"""Planning-specific once-only CAS and adoption (ASE3-024).

States advance only through::

    RESERVED -> EFFECT_STARTED -> TERMINAL_OBSERVED -> ADMITTED

or terminal ``UNKNOWN`` which forces ``PROMPT_REPLAY_REQUIRED`` and forbids a
second provider effect. This CAS is intentionally separate from provider
fallback attempt CAS.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final, Mapping

PLANNING_ATTEMPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/planning-attempt@1"
)
PLANNING_ADOPTION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/planning-effect-adoption@1"
)
PROMPT_REPLAY_REQUIRED: Final = "PROMPT_REPLAY_REQUIRED"


class PlanningAttemptState(str, Enum):
    RESERVED = "RESERVED"
    EFFECT_STARTED = "EFFECT_STARTED"
    TERMINAL_OBSERVED = "TERMINAL_OBSERVED"
    ADMITTED = "ADMITTED"
    UNKNOWN = "UNKNOWN"


class PlanningEffectError(ValueError):
    """Raised for planning CAS invariant violations."""


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _now_ms() -> int:
    return int(time.time() * 1000)


def _sha(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


@dataclass(frozen=True)
class DurablePromptIntent:
    """Run/context/policy-bound intent handle (no raw prompt bytes)."""

    schema: str
    run_id: str
    context_cid: str
    policy_cid: str
    prompt_ref: str
    prompt_cid: str
    created_at_ms: int

    def __post_init__(self) -> None:
        for name in (
            "run_id",
            "context_cid",
            "policy_cid",
            "prompt_ref",
            "prompt_cid",
        ):
            if not str(getattr(self, name) or "").strip():
                raise PlanningEffectError(f"{name} is required")
        if self.schema != "ipfs_accelerate_py/agent-supervisor/durable-prompt-intent@1":
            raise PlanningEffectError("unsupported durable prompt intent schema")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "context_cid": self.context_cid,
            "policy_cid": self.policy_cid,
            "prompt_ref": self.prompt_ref,
            "prompt_cid": self.prompt_cid,
            "created_at_ms": self.created_at_ms,
        }

    @property
    def content_id(self) -> str:
        return _sha(self.to_dict())


@dataclass
class PlanningAttemptRecord:
    """Mutable durable planning attempt record."""

    schema: str
    logical_attempt_id: str
    run_id: str
    context_cid: str
    policy_cid: str
    intent_cid: str
    route_plan_cid: str
    state: str
    created_at_ms: int
    effect_started_at_ms: int | None = None
    terminal_observed_at_ms: int | None = None
    admitted_at_ms: int | None = None
    unknown_at_ms: int | None = None
    terminal_output_cid: str = ""
    program_root_cid: str = ""
    replay_required: bool = False
    fence_token: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "logical_attempt_id": self.logical_attempt_id,
            "run_id": self.run_id,
            "context_cid": self.context_cid,
            "policy_cid": self.policy_cid,
            "intent_cid": self.intent_cid,
            "route_plan_cid": self.route_plan_cid,
            "state": self.state,
            "created_at_ms": self.created_at_ms,
            "effect_started_at_ms": self.effect_started_at_ms,
            "terminal_observed_at_ms": self.terminal_observed_at_ms,
            "admitted_at_ms": self.admitted_at_ms,
            "unknown_at_ms": self.unknown_at_ms,
            "terminal_output_cid": self.terminal_output_cid,
            "program_root_cid": self.program_root_cid,
            "replay_required": self.replay_required,
            "fence_token": self.fence_token,
        }

    @property
    def content_id(self) -> str:
        return _sha(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PlanningAttemptRecord":
        return cls(
            schema=str(value.get("schema") or ""),
            logical_attempt_id=str(value.get("logical_attempt_id") or ""),
            run_id=str(value.get("run_id") or ""),
            context_cid=str(value.get("context_cid") or ""),
            policy_cid=str(value.get("policy_cid") or ""),
            intent_cid=str(value.get("intent_cid") or ""),
            route_plan_cid=str(value.get("route_plan_cid") or ""),
            state=str(value.get("state") or ""),
            created_at_ms=int(value.get("created_at_ms") or 0),
            effect_started_at_ms=(
                int(value["effect_started_at_ms"])
                if value.get("effect_started_at_ms") is not None
                else None
            ),
            terminal_observed_at_ms=(
                int(value["terminal_observed_at_ms"])
                if value.get("terminal_observed_at_ms") is not None
                else None
            ),
            admitted_at_ms=(
                int(value["admitted_at_ms"])
                if value.get("admitted_at_ms") is not None
                else None
            ),
            unknown_at_ms=(
                int(value["unknown_at_ms"])
                if value.get("unknown_at_ms") is not None
                else None
            ),
            terminal_output_cid=str(value.get("terminal_output_cid") or ""),
            program_root_cid=str(value.get("program_root_cid") or ""),
            replay_required=bool(value.get("replay_required")),
            fence_token=str(value.get("fence_token") or ""),
        )


@dataclass(frozen=True)
class PlanningEffectAdoptionReceipt:
    """Immutable adoption of a terminal planning output/root."""

    schema: str
    logical_attempt_id: str
    state: str
    terminal_output_cid: str
    program_root_cid: str
    adopted_at_ms: int
    winner: bool
    replay_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "logical_attempt_id": self.logical_attempt_id,
            "state": self.state,
            "terminal_output_cid": self.terminal_output_cid,
            "program_root_cid": self.program_root_cid,
            "adopted_at_ms": self.adopted_at_ms,
            "winner": self.winner,
            "replay_required": self.replay_required,
        }

    @property
    def content_id(self) -> str:
        return _sha(self.to_dict())


@dataclass
class PlanningAttemptCASResult:
    record: PlanningAttemptRecord
    created: bool
    adopted: bool
    provider_effect_authorized: bool
    adoption: PlanningEffectAdoptionReceipt | None = None
    reason_code: str = ""


class PlanningAttemptCAS:
    """File-backed multiproc planning attempt CAS with exclusive locks."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink():
            raise PlanningEffectError("planning CAS root must not be a symlink")
        self._thread_lock = threading.RLock()

    def _path(self, logical_attempt_id: str) -> Path:
        digest = hashlib.sha256(logical_attempt_id.encode("utf-8")).hexdigest()
        return self.root / f"{digest}.json"

    def _lock_path(self, logical_attempt_id: str) -> Path:
        return self._path(logical_attempt_id).with_suffix(".lock")

    def _read(self, logical_attempt_id: str) -> PlanningAttemptRecord | None:
        path = self._path(logical_attempt_id)
        if not path.exists():
            return None
        if path.is_symlink():
            raise PlanningEffectError("planning attempt path is a symlink")
        raw = path.read_bytes()
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PlanningEffectError("planning attempt record is torn") from exc
        if not isinstance(payload, Mapping):
            raise PlanningEffectError("planning attempt record is invalid")
        return PlanningAttemptRecord.from_dict(payload)

    def _write_atomic(self, record: PlanningAttemptRecord) -> None:
        path = self._path(record.logical_attempt_id)
        tmp = path.with_suffix(".tmp")
        payload = _canonical(record.to_dict()) + b"\n"
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(tmp, flags, 0o600)
        try:
            os.write(fd, payload)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, path)
        # Best-effort private mode.
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass

    def reserve(
        self,
        *,
        logical_attempt_id: str,
        run_id: str,
        context_cid: str,
        policy_cid: str,
        intent_cid: str,
        route_plan_cid: str,
        now_ms: int | None = None,
    ) -> PlanningAttemptCASResult:
        """Reserve one logical planning attempt or adopt an existing winner."""

        with self._thread_lock:
            existing = self._read(logical_attempt_id)
            if existing is not None:
                if existing.replay_required or existing.state == PlanningAttemptState.UNKNOWN.value:
                    return PlanningAttemptCASResult(
                        record=existing,
                        created=False,
                        adopted=True,
                        provider_effect_authorized=False,
                        reason_code=PROMPT_REPLAY_REQUIRED,
                    )
                return PlanningAttemptCASResult(
                    record=existing,
                    created=False,
                    adopted=True,
                    provider_effect_authorized=False,
                    reason_code="adopted_existing_reservation",
                    adoption=(
                        PlanningEffectAdoptionReceipt(
                            schema=PLANNING_ADOPTION_SCHEMA,
                            logical_attempt_id=existing.logical_attempt_id,
                            state=existing.state,
                            terminal_output_cid=existing.terminal_output_cid,
                            program_root_cid=existing.program_root_cid,
                            adopted_at_ms=int(now_ms if now_ms is not None else _now_ms()),
                            winner=False,
                            replay_required=existing.replay_required,
                        )
                        if existing.state
                        in {
                            PlanningAttemptState.TERMINAL_OBSERVED.value,
                            PlanningAttemptState.ADMITTED.value,
                        }
                        else None
                    ),
                )

            created_at = int(now_ms if now_ms is not None else _now_ms())
            fence = hashlib.sha256(
                f"{logical_attempt_id}:{created_at}:{os.getpid()}".encode("utf-8")
            ).hexdigest()
            record = PlanningAttemptRecord(
                schema=PLANNING_ATTEMPT_SCHEMA,
                logical_attempt_id=str(logical_attempt_id),
                run_id=str(run_id),
                context_cid=str(context_cid),
                policy_cid=str(policy_cid),
                intent_cid=str(intent_cid),
                route_plan_cid=str(route_plan_cid),
                state=PlanningAttemptState.RESERVED.value,
                created_at_ms=created_at,
                fence_token=fence,
            )
            self._write_atomic(record)
            return PlanningAttemptCASResult(
                record=record,
                created=True,
                adopted=False,
                provider_effect_authorized=True,
                reason_code="reserved",
            )

    def mark_effect_started(
        self,
        logical_attempt_id: str,
        *,
        fence_token: str,
        now_ms: int | None = None,
    ) -> PlanningAttemptRecord:
        with self._thread_lock:
            record = self._require(logical_attempt_id)
            self._require_fence(record, fence_token)
            if record.state == PlanningAttemptState.UNKNOWN.value:
                raise PlanningEffectError(PROMPT_REPLAY_REQUIRED)
            if record.state != PlanningAttemptState.RESERVED.value:
                raise PlanningEffectError(
                    f"cannot start effect from state {record.state}"
                )
            record.state = PlanningAttemptState.EFFECT_STARTED.value
            record.effect_started_at_ms = int(
                now_ms if now_ms is not None else _now_ms()
            )
            self._write_atomic(record)
            return record

    def mark_terminal_observed(
        self,
        logical_attempt_id: str,
        *,
        fence_token: str,
        terminal_output_cid: str,
        program_root_cid: str,
        now_ms: int | None = None,
    ) -> PlanningAttemptRecord:
        with self._thread_lock:
            record = self._require(logical_attempt_id)
            self._require_fence(record, fence_token)
            if record.state == PlanningAttemptState.UNKNOWN.value:
                raise PlanningEffectError(PROMPT_REPLAY_REQUIRED)
            if record.state != PlanningAttemptState.EFFECT_STARTED.value:
                raise PlanningEffectError(
                    f"cannot observe terminal from state {record.state}"
                )
            if not terminal_output_cid or not program_root_cid:
                raise PlanningEffectError("terminal output and program root required")
            record.state = PlanningAttemptState.TERMINAL_OBSERVED.value
            record.terminal_observed_at_ms = int(
                now_ms if now_ms is not None else _now_ms()
            )
            record.terminal_output_cid = str(terminal_output_cid)
            record.program_root_cid = str(program_root_cid)
            self._write_atomic(record)
            return record

    def mark_admitted(
        self,
        logical_attempt_id: str,
        *,
        fence_token: str,
        now_ms: int | None = None,
    ) -> PlanningEffectAdoptionReceipt:
        with self._thread_lock:
            record = self._require(logical_attempt_id)
            self._require_fence(record, fence_token)
            if record.state == PlanningAttemptState.ADMITTED.value:
                return PlanningEffectAdoptionReceipt(
                    schema=PLANNING_ADOPTION_SCHEMA,
                    logical_attempt_id=record.logical_attempt_id,
                    state=record.state,
                    terminal_output_cid=record.terminal_output_cid,
                    program_root_cid=record.program_root_cid,
                    adopted_at_ms=int(record.admitted_at_ms or _now_ms()),
                    winner=False,
                )
            if record.state != PlanningAttemptState.TERMINAL_OBSERVED.value:
                raise PlanningEffectError(
                    f"cannot admit from state {record.state}"
                )
            record.state = PlanningAttemptState.ADMITTED.value
            record.admitted_at_ms = int(now_ms if now_ms is not None else _now_ms())
            self._write_atomic(record)
            return PlanningEffectAdoptionReceipt(
                schema=PLANNING_ADOPTION_SCHEMA,
                logical_attempt_id=record.logical_attempt_id,
                state=record.state,
                terminal_output_cid=record.terminal_output_cid,
                program_root_cid=record.program_root_cid,
                adopted_at_ms=int(record.admitted_at_ms),
                winner=True,
            )

    def mark_unknown(
        self,
        logical_attempt_id: str,
        *,
        fence_token: str | None = None,
        now_ms: int | None = None,
    ) -> PlanningAttemptRecord:
        """Durably record UNKNOWN and require prompt replay (no second effect)."""

        with self._thread_lock:
            record = self._require(logical_attempt_id)
            if fence_token is not None:
                self._require_fence(record, fence_token)
            if record.state == PlanningAttemptState.ADMITTED.value:
                raise PlanningEffectError("admitted attempts cannot become UNKNOWN")
            if record.state == PlanningAttemptState.UNKNOWN.value:
                return record
            record.state = PlanningAttemptState.UNKNOWN.value
            record.unknown_at_ms = int(now_ms if now_ms is not None else _now_ms())
            record.replay_required = True
            self._write_atomic(record)
            return record

    def load(self, logical_attempt_id: str) -> PlanningAttemptRecord | None:
        with self._thread_lock:
            return self._read(logical_attempt_id)

    def authorize_provider_effect(self, logical_attempt_id: str) -> bool:
        """Return True only for RESERVED attempts that are not UNKNOWN."""

        with self._thread_lock:
            record = self._read(logical_attempt_id)
            if record is None:
                return False
            if record.replay_required or record.state == PlanningAttemptState.UNKNOWN.value:
                return False
            return record.state == PlanningAttemptState.RESERVED.value

    def _require(self, logical_attempt_id: str) -> PlanningAttemptRecord:
        record = self._read(logical_attempt_id)
        if record is None:
            raise PlanningEffectError("planning attempt not found")
        return record

    def _require_fence(
        self, record: PlanningAttemptRecord, fence_token: str
    ) -> None:
        if not fence_token or fence_token != record.fence_token:
            raise PlanningEffectError("planning attempt fence token mismatch")


__all__ = [
    "DurablePromptIntent",
    "PLANNING_ADOPTION_SCHEMA",
    "PLANNING_ATTEMPT_SCHEMA",
    "PROMPT_REPLAY_REQUIRED",
    "PlanningAttemptCAS",
    "PlanningAttemptCASResult",
    "PlanningAttemptRecord",
    "PlanningAttemptState",
    "PlanningEffectAdoptionReceipt",
    "PlanningEffectError",
]
