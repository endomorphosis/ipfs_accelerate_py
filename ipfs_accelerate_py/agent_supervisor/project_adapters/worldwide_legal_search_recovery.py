"""Bounded deterministic recovery and lifecycle ownership.

A worker owns a task only while its kernel PID birth, process generation and
lease generation all match the durable lifecycle. Recovery may release a stale
claim, fence a dead worker, or wake an idle runnable queue. It does that at
most ``DEFAULT_LIMITS["recover"]`` times for one incident. Unknown effects stay
unknown, a hold or a passed deadline admits no mutation, and nothing in this
module grants completion authority or starts a process.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .worldwide_legal_search_progress import (
    DEFAULT_LIMITS,
    ProgressLedger,
    canonical_stage,
    digest,
    incident_identity,
)

SCHEMA = "wwl.recovery.decision@1"
STATE_SCHEMA = "wwl.recovery-state@1"
RECOVERY_LIMIT = DEFAULT_LIMITS["recover"]
HOLD_MARKERS = ("HOLD", "STOP", "OPERATOR_STOP")
EFFECT_STATUSES = ("unknown", "known_none", "applied")
DISPOSITIONS = (
    "owner_bound",
    "deadline_exceeded",
    "held",
    "recovery_bound_exhausted",
    "effects_preserved",
    "dead_worker",
    "stale_claim",
    "idle_runnable_queue",
    "idle_within_deadline",
    "owner_live",
    "no_recovery",
)
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_CHECKPOINT = 1_048_576
_OBSERVATION_FIELDS = frozenset({
    "action_id", "observed_at", "deadline_at", "incident", "hold", "hold_markers",
    "owner", "claims", "effects", "runnable_task_ids", "idle_since", "queue_deadline_at",
    "completion_authority",
})
_OWNER_FIELDS = frozenset({"worker_id", "pid", "birth", "process_generation", "lease_generation"})
_CLAIM_FIELDS = _OWNER_FIELDS | {"claim_id", "task_id", "effect_id"}
_EFFECT_FIELDS = frozenset({"effect_id", "task_id", "status", "digest"})
_INCIDENT_FIELDS = frozenset({"task_id", "stage", "defect", "incident_id"})
_RESET_FIELDS = frozenset({
    "reset", "reset_budget", "reset_budgets", "reset_progress", "refund", "budgets_reset", "progress_reset",
})
Probe = Callable[[int], Mapping[str, Any]]


class RecoveryContractError(ValueError):
    """Observation or checkpoint violates the recovery contract."""


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise RecoveryContractError(f"{name} must be a bounded durable identifier")
    return value


def _whole(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise RecoveryContractError(f"{name} must be an integer >= {minimum}")
    return value


def _timestamp(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecoveryContractError(f"{name} must be a finite timestamp")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")) or number < 0:
        raise RecoveryContractError(f"{name} must be a finite timestamp")
    return number


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value.strip().lower()) is None:
        raise RecoveryContractError(f"{name} must be a sha256 digest")
    return value.strip().lower()


def _reject_completion(value: Mapping[str, Any]) -> None:
    if value.get("completion_authority") is True:
        raise RecoveryContractError("recovery cannot grant completion authority")
    present = _RESET_FIELDS.intersection(value)
    if any(value.get(field) not in (False, None) for field in present):
        raise RecoveryContractError("recovery cannot reset a durable budget")


def read_pid_birth(pid: int) -> dict[str, Any]:
    """Read one kernel PID birth. A reused PID has a different start tick."""
    from .worldwide_legal_search_host_lifecycle import process_identity

    checked = _whole(pid, "pid", minimum=1)
    observed = process_identity(checked)
    ticks = int(observed.get("start_ticks") or 0)
    alive = bool(observed.get("alive")) and ticks > 0
    return {"pid": checked, "alive": alive, "start_ticks": ticks if alive else 0, "state": "alive" if alive else "dead"}


def verify_pid_birth(pid: int, birth: int, *, probe: Probe | None = None) -> dict[str, Any]:
    """Compare a claimed birth with the process probe.

    The probe is the authority. An alive process whose start ticks differ is
    PID reuse, not the worker that was given the lease. A dead PID does not
    match any birth, including a caller that still reports it alive.
    """
    expected = _whole(birth, "pid birth", minimum=1)
    checked = _whole(pid, "pid", minimum=1)
    observed = _public_probe(checked, probe)
    matches = observed["alive"] and observed["start_ticks"] == expected
    reused = observed["alive"] and observed["start_ticks"] != expected
    return {
        "pid": checked,
        "expected_birth": expected,
        "observed_birth": observed["start_ticks"],
        "alive": observed["alive"],
        "pid_birth_matches": matches,
        "pid_reused": reused,
        "state": observed["state"],
    }


def verify_generations(
    process_generation: int,
    lease_generation: int,
    *,
    owner_process_generation: int,
    owner_lease_generation: int,
) -> dict[str, Any]:
    """Classify a process/lease pair against the durable lifecycle pair.

    Either generation moving backwards is a stale claim. Either generation
    moving forwards is a future claim and cannot steal ownership. A pair that
    is stale on one counter and future on the other conflicts; it is not current.
    """
    process = _whole(process_generation, "process generation")
    lease = _whole(lease_generation, "lease generation")
    owner_process = _whole(owner_process_generation, "owner process generation")
    owner_lease = _whole(owner_lease_generation, "owner lease generation")
    stale = process < owner_process or lease < owner_lease
    future = process > owner_process or lease > owner_lease
    return {
        "process_generation": process,
        "lease_generation": lease,
        "owner_process_generation": owner_process,
        "owner_lease_generation": owner_lease,
        "process_generation_current": process == owner_process,
        "lease_generation_current": lease == owner_lease,
        "stale": stale and not future,
        "future": future and not stale,
        "conflict": stale and future,
        "match": process == owner_process and lease == owner_lease,
    }


def _public_probe(pid: int, probe: Probe | None) -> dict[str, Any]:
    """Normalize a probe row so a dead PID never counts as a matching birth."""
    raw = probe(pid) if probe is not None else read_pid_birth(pid)
    if not isinstance(raw, Mapping) or int(raw.get("pid", -1)) != pid:
        raise RecoveryContractError("pid birth probe returned a different pid")
    if type(raw.get("alive")) is not bool:
        raise RecoveryContractError("pid birth probe must report alive as a boolean")
    ticks = raw.get("start_ticks", raw.get("birth"))
    if type(ticks) is not int or ticks < 0:
        raise RecoveryContractError("pid birth probe must report start ticks")
    state = raw.get("state", "")
    zombie = isinstance(state, str) and state in {"Z", "X", "x", "zombie", "dead"}
    alive = bool(raw["alive"]) and not zombie and ticks > 0
    return {"pid": pid, "alive": alive, "start_ticks": ticks if alive else 0, "state": "alive" if alive else "dead"}


class RecoveryOwner:
    """Durable lifecycle owner for one campaign run.

    The progress ledger, when bound, is the recover-budget authority. This
    owner charges that same incident and refuses a ledger whose recover limit
    is not the campaign limit. A checkpoint reloads generations, fences,
    unknown effects and spent counts; it does not open a fresh budget.
    """

    def __init__(self, run_id: str, *, ledger: ProgressLedger | None = None, probe: Probe | None = None):
        self.run_id = _identifier(run_id, "run_id")
        if ledger is not None:
            if not isinstance(ledger, ProgressLedger):
                raise RecoveryContractError("recovery ledger must be a progress ledger")
            if ledger.run_id != self.run_id:
                raise RecoveryContractError("recovery run does not match the progress ledger")
            if ledger.limits.get("recover") != RECOVERY_LIMIT:
                raise RecoveryContractError("durable recovery bound differs from the campaign ledger")
        self.ledger = ledger
        self.probe: Probe = probe or read_pid_birth
        self.process_generation = 0
        self.lease_generation = 0
        self.owner: dict[str, Any] | None = None
        self.fences: list[dict[str, Any]] = []
        self.claims: dict[str, dict[str, Any]] = {}
        self.effects: dict[str, dict[str, Any]] = {}
        self.queue: list[str] = []
        self.hold = {"held": False, "marker": ""}
        self.spent: dict[str, int] = {}
        self.woken_epochs: list[str] = []
        self._actions: dict[str, dict[str, Any]] = {}
        self.completion_authority = False

    def bind_owner(
        self,
        *,
        worker_id: str,
        pid: int,
        birth: int,
        process_generation: int,
        lease_generation: int,
    ) -> dict[str, Any]:
        """Admit the current lifecycle owner after PID-birth verification."""
        if self.hold["held"]:
            raise RecoveryContractError("hold blocks lifecycle ownership changes")
        if self._unknown_blocks_advance():
            raise RecoveryContractError("unknown effects still own the lifecycle")
        if self.owner is not None and self.owner["fenced"]:
            raise RecoveryContractError("fenced lifecycle owner cannot be replaced without a new generation")
        worker = _identifier(worker_id, "worker_id")
        checked = verify_pid_birth(pid, birth, probe=self.probe)
        if not checked["pid_birth_matches"]:
            raise RecoveryContractError("pid birth does not match the lifecycle owner")
        process = _whole(process_generation, "process generation", minimum=1)
        lease = _whole(lease_generation, "lease generation", minimum=1)
        identity = {
            "worker_id": worker, "pid": checked["pid"], "birth": checked["expected_birth"],
            "process_generation": process, "lease_generation": lease, "fenced": False,
        }
        if self.owner is not None and not self.owner["fenced"]:
            if _same_owner(self.owner, identity):
                return self._owner_decision(identity, adopted=False)
            relation = verify_generations(
                process, lease,
                owner_process_generation=self.process_generation,
                owner_lease_generation=self.lease_generation,
            )
            if relation["stale"] or relation["conflict"]:
                raise RecoveryContractError("process or lease generation regressed")
            if relation["future"]:
                raise RecoveryContractError("process or lease generation jumped ahead of the durable lifecycle")
            raise RecoveryContractError("lifecycle owner is already bound")
        if any(_same_birth(item, identity) for item in self.fences):
            raise RecoveryContractError("fenced pid birth cannot be rebound")
        if self.process_generation == 0 and self.lease_generation == 0:
            self.process_generation = process
            self.lease_generation = lease
        elif process != self.process_generation or lease != self.lease_generation:
            relation = verify_generations(
                process, lease,
                owner_process_generation=self.process_generation,
                owner_lease_generation=self.lease_generation,
            )
            if relation["stale"] or relation["conflict"]:
                raise RecoveryContractError("process or lease generation regressed")
            raise RecoveryContractError("process or lease generation jumped ahead of the durable lifecycle")
        self.owner = identity
        return self._owner_decision(identity, adopted=True)

    def set_hold(self, held: bool, *, marker: str = "") -> dict[str, Any]:
        """Record or clear a durable hold. Recovery does not override it."""
        if type(held) is not bool:
            raise RecoveryContractError("hold must be a boolean")
        if held:
            if marker not in HOLD_MARKERS:
                raise RecoveryContractError("hold marker is not recognized")
            self.hold = {"held": True, "marker": marker}
        else:
            if marker not in {"", *HOLD_MARKERS}:
                raise RecoveryContractError("hold marker is not recognized")
            self.hold = {"held": False, "marker": ""}
        return {"schema": SCHEMA, "disposition": "held" if held else "no_recovery", "held": self.hold["held"],
                "hold_marker": self.hold["marker"], "applied": False, "completion_authority": False}

    def recover(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        """Take at most one bounded recovery step for one incident."""
        if not isinstance(observation, Mapping):
            raise RecoveryContractError("recovery observation must be an object")
        _reject_completion(observation)
        unknown = set(observation) - _OBSERVATION_FIELDS
        if unknown:
            raise RecoveryContractError("observation field is not recognized")
        normalized = self._normalize(observation)
        action_id = normalized["action_id"]
        if action_id in self._actions:
            stored = self._actions[action_id]
            if stored["digest"] != normalized["digest"]:
                raise RecoveryContractError("action id reused with a different observation")
            return {**stored["decision"], "deduplicated": True}
        preview = self._preview(normalized)
        decision = self._decide(normalized, preview)
        if decision["mutate"]:
            self._apply(decision["mutation"], preview)
            incident_id = normalized["incident_id"]
            if decision["mutation"]["spend"]:
                charge = self._spend(incident_id)
            else:
                charge = self._budget(incident_id, charged=False)
            decision["public"]["budget"] = charge
            decision["public"]["runnable_task_ids"] = list(self.queue)
            decision["public"]["process_generation"] = self.process_generation
            decision["public"]["lease_generation"] = self.lease_generation
            decision["public"]["generation_advanced"] = decision["mutation"]["advance"]
            decision["public"]["effect_records"] = self._effect_records()
            decision["public"]["owner_live"] = self._owner_is_live()
            decision["public"]["owner_fenced"] = bool(self.owner and self.owner["fenced"]) or (
                self.owner is None and bool(self.fences) and decision["mutation"]["advance"]
            )
        self._actions[action_id] = {"digest": normalized["digest"], "decision": decision["public"]}
        return dict(decision["public"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": STATE_SCHEMA,
            "run_id": self.run_id,
            "recover_limit": RECOVERY_LIMIT,
            "process_generation": self.process_generation,
            "lease_generation": self.lease_generation,
            "owner": None if self.owner is None else dict(self.owner),
            "fences": [dict(item) for item in self.fences],
            "claims": {key: dict(self.claims[key]) for key in sorted(self.claims)},
            "effects": {key: dict(self.effects[key]) for key in sorted(self.effects)},
            "queue": list(self.queue),
            "hold": dict(self.hold),
            "spent": {key: self.spent[key] for key in sorted(self.spent)},
            "woken_epochs": list(self.woken_epochs),
            "actions": {key: {"digest": self._actions[key]["digest"], "decision": dict(self._actions[key]["decision"])} for key in sorted(self._actions)},
            "ledger": None if self.ledger is None else self.ledger.to_dict(),
            "completion_authority": False,
        }

    def save(self, path: str | Path) -> None:
        """Atomically replace a checkpoint. A crash cannot widen the recover budget."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
        if len(payload) > _MAX_CHECKPOINT:
            raise RecoveryContractError("recovery checkpoint exceeds the admission bound")
        descriptor, temporary = tempfile.mkstemp(prefix=".wwl-recovery-", dir=target.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            directory = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except Exception:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise

    @classmethod
    def load(cls, path: str | Path, *, probe: Probe | None = None) -> RecoveryOwner:
        raw = Path(path).read_bytes()
        if len(raw) > _MAX_CHECKPOINT:
            raise RecoveryContractError("recovery checkpoint exceeds the admission bound")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise RecoveryContractError("recovery checkpoint is not JSON") from error
        return cls.from_dict(value, probe=probe)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, probe: Probe | None = None) -> RecoveryOwner:
        if not isinstance(value, Mapping) or value.get("schema") != STATE_SCHEMA:
            raise RecoveryContractError("recovery checkpoint schema is not recognized")
        if value.get("completion_authority") is not False:
            raise RecoveryContractError("recovery checkpoint cannot grant completion authority")
        if value.get("recover_limit") != RECOVERY_LIMIT:
            raise RecoveryContractError("durable recovery bound differs from the campaign ledger")
        ledger_value = value.get("ledger")
        ledger = None
        if ledger_value is not None:
            ledger = ProgressLedger.from_dict(ledger_value)
        owner = cls(str(value.get("run_id") or ""), ledger=ledger, probe=probe)
        owner.process_generation = _whole(value.get("process_generation"), "process generation")
        owner.lease_generation = _whole(value.get("lease_generation"), "lease generation")
        owner.owner = _restore_owner(value.get("owner"))
        if owner.owner is not None:
            if owner.owner["process_generation"] > owner.process_generation or owner.owner["lease_generation"] > owner.lease_generation:
                raise RecoveryContractError("stored lifecycle owner is ahead of its generation")
            if not owner.owner["fenced"] and (
                owner.owner["process_generation"] != owner.process_generation or owner.owner["lease_generation"] != owner.lease_generation
            ):
                raise RecoveryContractError("stored lifecycle owner generation does not match")
        owner.fences = [_restore_fence(item) for item in value.get("fences") or []]
        owner.claims = _restore_claims(value.get("claims"))
        owner.effects = _restore_effects(value.get("effects"))
        queue = value.get("queue") or []
        if not isinstance(queue, list):
            raise RecoveryContractError("stored runnable queue is malformed")
        owner.queue = [_identifier(item, "task_id") for item in queue]
        if len(owner.queue) != len(set(owner.queue)):
            raise RecoveryContractError("stored runnable queue repeats a task")
        hold = value.get("hold")
        if not isinstance(hold, Mapping) or type(hold.get("held")) is not bool:
            raise RecoveryContractError("stored hold is malformed")
        marker = hold.get("marker") or ""
        if hold["held"] and marker not in HOLD_MARKERS:
            raise RecoveryContractError("stored hold marker is not recognized")
        if not hold["held"] and marker not in {"", *HOLD_MARKERS}:
            raise RecoveryContractError("stored hold marker is not recognized")
        owner.hold = {"held": hold["held"], "marker": "" if not hold["held"] else marker}
        spent = value.get("spent") or {}
        if not isinstance(spent, Mapping):
            raise RecoveryContractError("stored recovery budget is malformed")
        for incident_id, used in spent.items():
            _digest(incident_id, "incident_id")
            owner.spent[incident_id] = _whole(used, "recover spent")
            if owner.spent[incident_id] > RECOVERY_LIMIT:
                raise RecoveryContractError("stored recovery budget exceeds the durable bound")
        epochs = value.get("woken_epochs") or []
        if not isinstance(epochs, list):
            raise RecoveryContractError("stored idle epochs are malformed")
        owner.woken_epochs = [_digest(item, "idle epoch") for item in epochs]
        actions = value.get("actions") or {}
        if not isinstance(actions, Mapping):
            raise RecoveryContractError("stored recovery actions are malformed")
        for action_id, payload in actions.items():
            _identifier(action_id, "action_id")
            if not isinstance(payload, Mapping) or payload.get("decision", {}).get("completion_authority") is not False:
                raise RecoveryContractError("stored recovery action cannot grant completion authority")
            owner._actions[action_id] = {"digest": _digest(payload.get("digest"), "action digest"), "decision": dict(payload["decision"])}
        owner._require_ledger_alignment()
        return owner

    def _normalize(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        action_id = _identifier(observation.get("action_id"), "action_id")
        observed_at = _timestamp(observation.get("observed_at"), "observed_at")
        deadline_at = _timestamp(observation.get("deadline_at"), "deadline_at")
        incident = observation.get("incident")
        if not isinstance(incident, Mapping) or set(incident) - _INCIDENT_FIELDS:
            raise RecoveryContractError("incident binding is malformed")
        task_id = _identifier(incident.get("task_id"), "task_id")
        stage = canonical_stage(incident.get("stage"))
        incident_id = incident_identity(run_id=self.run_id, task_id=task_id, stage=stage, defect=incident.get("defect", ""))
        if "incident_id" in incident and incident.get("incident_id") not in (None, "") and incident.get("incident_id") != incident_id:
            raise RecoveryContractError("incident id does not match the durable incident identity")
        hold_markers = _markers(observation.get("hold_markers"))
        held = observation.get("hold") is True or bool(hold_markers) or self.hold["held"]
        if observation.get("hold") not in (None, True, False):
            raise RecoveryContractError("hold must be a boolean")
        owner = _optional_owner(observation.get("owner"))
        claims = _claim_list(observation.get("claims"))
        effects = _effect_list(observation.get("effects"))
        runnable = _task_list(observation.get("runnable_task_ids"))
        idle_since = None
        queue_deadline_at = None
        if runnable:
            if "idle_since" not in observation or "queue_deadline_at" not in observation:
                raise RecoveryContractError("an idle runnable queue requires an idle start and a deadline")
            idle_since = _timestamp(observation.get("idle_since"), "idle_since")
            queue_deadline_at = _timestamp(observation.get("queue_deadline_at"), "queue_deadline_at")
            if idle_since > observed_at:
                raise RecoveryContractError("idle runnable queue starts after the observation")
            if queue_deadline_at < idle_since:
                raise RecoveryContractError("idle runnable queue deadline precedes its start")
        body = {
            "action_id": action_id,
            "observed_at": observed_at,
            "deadline_at": deadline_at,
            "incident_id": incident_id,
            "task_id": task_id,
            "stage": stage,
            "held": held,
            "hold_markers": hold_markers,
            "owner": owner,
            "claims": claims,
            "effects": effects,
            "runnable_task_ids": runnable,
            "idle_since": idle_since,
            "queue_deadline_at": queue_deadline_at,
            "completion_authority": False,
        }
        return {**body, "digest": digest(body)}

    def _preview(self, normalized: Mapping[str, Any]) -> dict[str, Any]:
        """Read-only classification. The probe overrules any claimed liveness."""
        births: dict[int, dict[str, Any]] = {}

        def birth_for(pid: int, birth: int) -> dict[str, Any]:
            if pid not in births:
                births[pid] = _public_probe(pid, self.probe)
            checked = verify_pid_birth(pid, birth, probe=lambda requested, row=births[pid]: row)
            return checked

        bound = self.owner
        observed_owner = normalized["owner"]
        if bound is not None and observed_owner is not None and not _same_owner(bound, {**observed_owner, "fenced": bound["fenced"]}):
            if not (bound["fenced"] and _same_birth(bound, observed_owner) and observed_owner["process_generation"] == bound["process_generation"] and observed_owner["lease_generation"] == bound["lease_generation"]):
                raise RecoveryContractError("observation owner differs from the bound lifecycle owner")
        if bound is None and observed_owner is not None and not any(_same_birth(item, observed_owner) and item["process_generation"] == observed_owner["process_generation"] and item["lease_generation"] == observed_owner["lease_generation"] for item in self.fences):
            raise RecoveryContractError("lifecycle owner is not bound")
        subject = None
        if bound is not None:
            subject = bound
        elif observed_owner is not None:
            subject = observed_owner
        owner_check = None
        if subject is not None:
            owner_check = birth_for(subject["pid"], subject["birth"])
        merged_effects = self._merged_effects(normalized["effects"])
        unknown_ids = [item["effect_id"] for item in merged_effects.values() if item["status"] == "unknown"]
        unknown_tasks = {item["task_id"] for item in merged_effects.values() if item["status"] == "unknown"}
        if unknown_tasks.intersection(normalized["runnable_task_ids"]):
            raise RecoveryContractError("unknown effect cannot sit on the runnable queue")
        claim_views = []
        for claim in normalized["claims"]:
            if subject is None and bound is None and self.process_generation == 0:
                raise RecoveryContractError("claim requires a bound lifecycle owner")
            relation = verify_generations(
                claim["process_generation"], claim["lease_generation"],
                owner_process_generation=self.process_generation,
                owner_lease_generation=self.lease_generation,
            )
            if relation["future"] or relation["conflict"]:
                raise RecoveryContractError("process or lease generation is ahead of the durable lifecycle")
            if self.owner is None and (self.process_generation == 0 or relation["match"]):
                raise RecoveryContractError("claim requires a bound lifecycle owner")
            checked = birth_for(claim["pid"], claim["birth"])
            current = bool(
                bound is not None and not bound["fenced"] and owner_check and owner_check["pid_birth_matches"]
                and relation["match"] and checked["pid_birth_matches"]
                and claim["worker_id"] == bound["worker_id"] and claim["pid"] == bound["pid"] and claim["birth"] == bound["birth"]
            )
            stale = not current
            claim_views.append({**claim, "current": current, "stale": stale, "pid_reused": checked["pid_reused"], "alive": checked["alive"]})
        dead = bool(subject is not None and owner_check is not None and not owner_check["pid_birth_matches"])
        already = False
        if subject is not None and dead:
            already = any(
                item["pid"] == subject["pid"] and item["birth"] == subject["birth"]
                and item["process_generation"] == subject["process_generation"] and item["lease_generation"] == subject["lease_generation"]
                for item in self.fences
            ) or bool(bound is not None and bound["fenced"] and bound["pid"] == subject["pid"] and bound["birth"] == subject["birth"])
        live = bool(subject is not None and owner_check is not None and owner_check["pid_birth_matches"] and bound is not None and not bound["fenced"])
        stale_ids = [item["claim_id"] for item in claim_views if item["stale"] and self.claims.get(item["claim_id"], {}).get("state") != "released"]
        return {
            "births": births,
            "owner_check": owner_check,
            "effects": merged_effects,
            "unknown_ids": unknown_ids,
            "unknown_tasks": unknown_tasks,
            "claims": claim_views,
            "stale_ids": stale_ids,
            "dead": dead and not already,
            "already_fenced": already,
            "live": live,
            "pid_reused": bool(owner_check and owner_check["pid_reused"]),
        }

    def _decide(self, normalized: Mapping[str, Any], preview: Mapping[str, Any]) -> dict[str, Any]:
        incident_id = str(normalized["incident_id"])
        findings = ["pid_birth_checked", "process_generation_checked", "lease_generation_checked"]
        if preview["pid_reused"]:
            findings.append("pid_reused")
        if preview["stale_ids"]:
            findings.append("stale_claim")
        if preview["dead"]:
            findings.append("dead_worker")
        idle = bool(normalized["runnable_task_ids"]) and not preview["live"] and not preview["dead"] and not preview["stale_ids"]
        if idle:
            findings.append("idle_runnable_queue")
        if preview["unknown_ids"]:
            findings.append("unknown_effect")
        base = self._shell(normalized, preview, findings)
        if normalized["observed_at"] >= normalized["deadline_at"]:
            base.update(disposition="deadline_exceeded", reason="recovery deadline passed before the action",
                        deadline_exceeded=True, applied=False)
            return {"public": base, "mutate": False}
        if normalized["held"]:
            findings.append("hold")
            base["findings"] = _unique(findings)
            base.update(disposition="held", reason="hold blocks lifecycle mutation", held=True, applied=False)
            return {"public": base, "mutate": False}
        mutation = self._mutation(normalized, preview, idle)
        if mutation["spend"] and self._remaining(incident_id) <= 0:
            findings.append("recovery_bound")
            base["findings"] = _unique(findings)
            base.update(disposition="recovery_bound_exhausted", reason="durable recovery bound exhausted",
                        applied=False, budget=self._budget(incident_id, charged=False, blocked=True))
            return {"public": base, "mutate": False}
        base["findings"] = _unique(findings)
        base.update(disposition=mutation["disposition"], reason=mutation["reason"], applied=mutation["applied"],
                    wake_authorized=mutation["wake"], effects_preserved=mutation["disposition"] == "effects_preserved" or bool(preview["unknown_ids"]))
        return {"public": base, "mutate": mutation["applied"] or mutation["remember"], "mutation": mutation}

    def _mutation(self, normalized: Mapping[str, Any], preview: Mapping[str, Any], idle: bool) -> dict[str, Any]:
        unknown = bool(preview["unknown_ids"])
        release = list(preview["stale_ids"])
        requeue: list[str] = []
        advance = False
        fence = False
        wake = False
        spend = False
        remember = True
        disposition = "no_recovery"
        reason = "no recovery action is due"
        if preview["dead"] and unknown:
            disposition = "effects_preserved"
            reason = "unknown effects remain preserved"
            fence = True
            spend = not (self.owner is not None and self.owner["fenced"])
        elif preview["dead"]:
            disposition = "dead_worker"
            reason = "dead worker fenced and generation advanced"
            fence = True
            advance = True
            requeue = _requeue_tasks(preview, normalized["runnable_task_ids"])
            spend = True
        elif release and unknown:
            disposition = "effects_preserved"
            reason = "unknown effects remain preserved"
            requeue = _requeue_tasks(preview, normalized["runnable_task_ids"], only_stale=True)
            spend = True
        elif release:
            disposition = "stale_claim"
            reason = "stale claim released"
            requeue = _requeue_tasks(preview, normalized["runnable_task_ids"], only_stale=True)
            spend = True
        elif idle:
            epoch = _epoch(self.run_id, normalized)
            past = float(normalized["observed_at"]) >= float(normalized["queue_deadline_at"])
            if past and epoch not in self.woken_epochs:
                disposition = "idle_runnable_queue"
                reason = "idle runnable queue woke inside its recovery bound"
                wake = True
                spend = True
            elif past:
                disposition = "idle_runnable_queue"
                reason = "idle runnable queue already woke for this epoch"
                wake = True
                spend = False
            else:
                disposition = "idle_within_deadline"
                reason = "idle runnable queue is inside its deadline"
                spend = False
        elif preview["live"]:
            disposition = "owner_live"
            reason = "verified owner still holds the current generation"
            spend = False
        else:
            remember = False
            spend = False
        applied = bool(fence or release or advance or (wake and spend))
        if disposition == "no_recovery":
            applied = False
            remember = False
        elif disposition in {"idle_within_deadline", "owner_live", "idle_runnable_queue"}:
            remember = True
        return {
            "disposition": disposition, "reason": reason, "spend": spend, "fence": fence, "advance": advance,
            "release": release, "requeue": requeue, "wake": wake, "applied": applied, "remember": remember,
            "epoch": _epoch(self.run_id, normalized) if normalized["runnable_task_ids"] else "",
            "runnable": list(normalized["runnable_task_ids"]),
        }

    def _apply(self, mutation: Mapping[str, Any], preview: Mapping[str, Any]) -> None:
        for effect in preview["effects"].values():
            current = self.effects.get(effect["effect_id"])
            if current is None:
                self.effects[effect["effect_id"]] = dict(effect)
            elif current["status"] == "unknown":
                continue
            elif current != effect and (current["status"] != effect["status"] or current["digest"] != effect["digest"] or current["task_id"] != effect["task_id"]):
                raise RecoveryContractError("effect identity changed")
        for claim in preview["claims"]:
            previous = self.claims.get(claim["claim_id"])
            if previous is not None and previous.get("binding") != claim["binding"]:
                raise RecoveryContractError("claim id reused with a different binding")
            if claim["current"] and not preview["dead"]:
                state = "current"
            elif claim["claim_id"] in mutation["release"] or (
                claim["stale"] and mutation["disposition"] in {"stale_claim", "effects_preserved", "dead_worker"}
            ):
                state = "released"
            elif previous is not None:
                state = str(previous["state"])
            else:
                state = "open"
            stored = {key: claim[key] for key in ("claim_id", "task_id", "worker_id", "pid", "birth", "process_generation", "lease_generation")}
            stored["effect_id"] = claim.get("effect_id", "")
            stored["binding"] = claim["binding"]
            stored["state"] = state
            self.claims[claim["claim_id"]] = stored
        if mutation["fence"] and self.owner is not None and not any(_same_birth(item, self.owner) and item["process_generation"] == self.owner["process_generation"] and item["lease_generation"] == self.owner["lease_generation"] for item in self.fences):
            self.fences.append({
                "worker_id": self.owner["worker_id"], "pid": self.owner["pid"], "birth": self.owner["birth"],
                "process_generation": self.owner["process_generation"], "lease_generation": self.owner["lease_generation"],
                "reason": mutation["disposition"],
            })
            self.owner["fenced"] = True
            self.owner["fence_reason"] = mutation["disposition"]
        if mutation["advance"] and self.owner is not None and self.owner["fenced"]:
            self.process_generation += 1
            self.lease_generation += 1
            self.owner = None
        for task_id in mutation["requeue"]:
            if task_id not in self.queue and task_id not in preview["unknown_tasks"]:
                self.queue.append(task_id)
        for task_id in mutation["runnable"]:
            if task_id not in self.queue and task_id not in preview["unknown_tasks"]:
                self.queue.append(task_id)
        self.queue.sort()
        if mutation["wake"] and mutation["spend"] and mutation["epoch"] and mutation["epoch"] not in self.woken_epochs:
            self.woken_epochs.append(mutation["epoch"])

    def _merged_effects(self, incoming: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
        merged = {key: dict(value) for key, value in self.effects.items()}
        for item in incoming:
            current = merged.get(item["effect_id"])
            if current is None:
                merged[item["effect_id"]] = dict(item)
                continue
            if current["task_id"] != item["task_id"]:
                raise RecoveryContractError("effect identity changed")
            if current["status"] == "unknown":
                continue
            if current["status"] != item["status"] or current["digest"] != item["digest"]:
                if item["status"] == "unknown":
                    continue
                raise RecoveryContractError("effect identity changed")
        return merged

    def _spend(self, incident_id: str) -> dict[str, Any]:
        if self._remaining(incident_id) <= 0:
            raise RecoveryContractError("durable recovery bound exhausted")
        if self.ledger is not None:
            if incident_id not in self.ledger.incidents:
                raise RecoveryContractError("bound progress ledger has no such incident")
            ledger_spent = self.ledger.incidents[incident_id]["spent"]["recover"]
            if ledger_spent != self.spent.get(incident_id, 0):
                raise RecoveryContractError("durable recovery bound diverged from the progress ledger")
            charged = self.ledger.charge(incident_id, "recover")
            if not charged["charged"]:
                raise RecoveryContractError("durable recovery bound exhausted")
            self.spent[incident_id] = charged["spent"]
            return {
                "spent": charged["spent"], "limit": charged["limit"], "remaining": charged["remaining"],
                "charged": True, "status": "charged", "budgets_reset": False, "completion_authority": False,
            }
        self.spent[incident_id] = self.spent.get(incident_id, 0) + 1
        return self._budget(incident_id, charged=True)

    def _remaining(self, incident_id: str) -> int:
        return RECOVERY_LIMIT - self.spent.get(incident_id, 0)

    def _budget(self, incident_id: str, *, charged: bool, blocked: bool = False) -> dict[str, Any]:
        spent = self.spent.get(incident_id, 0)
        return {
            "spent": spent, "limit": RECOVERY_LIMIT, "remaining": RECOVERY_LIMIT - spent,
            "charged": charged, "status": "blocked" if blocked or spent >= RECOVERY_LIMIT and not charged else ("charged" if charged else "not_charged"),
            "budgets_reset": False, "completion_authority": False,
        }

    def _shell(self, normalized: Mapping[str, Any], preview: Mapping[str, Any], findings: list[str]) -> dict[str, Any]:
        owner_check = preview["owner_check"] or {}
        return {
            "schema": SCHEMA,
            "action_id": normalized["action_id"],
            "disposition": "no_recovery",
            "reason": "",
            "incident_id": normalized["incident_id"],
            "applied": False,
            "deduplicated": False,
            "findings": list(findings),
            "pid_birth_matches": bool(owner_check.get("pid_birth_matches")),
            "pid_reused": bool(preview["pid_reused"]),
            "process_generation": self.process_generation,
            "lease_generation": self.lease_generation,
            "generation_advanced": False,
            "process_generation_current": True,
            "lease_generation_current": True,
            "stale_claim_ids": list(preview["stale_ids"]),
            "dead_worker": bool(preview["dead"]),
            "owner_live": bool(preview["live"]),
            "owner_fenced": bool(self.owner and self.owner["fenced"]),
            "idle_runnable": "idle_runnable_queue" in findings,
            "effects_unknown": bool(preview["unknown_ids"]),
            "effects_preserved": bool(preview["unknown_ids"]),
            "effect_records": self._effect_records(),
            "runnable_task_ids": list(self.queue),
            "wake_authorized": False,
            "held": bool(normalized["held"]),
            "deadline_exceeded": False,
            "budget": self._budget(str(normalized["incident_id"]), charged=False),
            "completion_authority": False,
        }

    def _effect_records(self) -> list[dict[str, Any]]:
        return [dict(self.effects[key]) for key in sorted(self.effects)]

    def _owner_is_live(self) -> bool:
        if self.owner is None or self.owner["fenced"]:
            return False
        checked = verify_pid_birth(self.owner["pid"], self.owner["birth"], probe=self.probe)
        return bool(checked["pid_birth_matches"])

    def _unknown_blocks_advance(self) -> bool:
        return any(item["status"] == "unknown" for item in self.effects.values()) and any(item.get("reason") == "effects_preserved" or (self.owner or {}).get("fence_reason") == "effects_preserved" for item in self.fences + ([self.owner] if self.owner and self.owner.get("fenced") else []))

    def _owner_decision(self, identity: Mapping[str, Any], *, adopted: bool) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "disposition": "owner_bound",
            "reason": "pid birth and process/lease generation verified",
            "applied": adopted,
            "deduplicated": not adopted,
            "worker_id": identity["worker_id"],
            "pid": identity["pid"],
            "birth": identity["birth"],
            "pid_birth_matches": True,
            "pid_reused": False,
            "process_generation": self.process_generation,
            "lease_generation": self.lease_generation,
            "generation_advanced": False,
            "completion_authority": False,
            "held": False,
            "effects_unknown": any(item["status"] == "unknown" for item in self.effects.values()),
        }

    def _require_ledger_alignment(self) -> None:
        if self.ledger is None:
            return
        if self.ledger.limits.get("recover") != RECOVERY_LIMIT:
            raise RecoveryContractError("durable recovery bound differs from the campaign ledger")
        for incident_id, incident in self.ledger.incidents.items():
            used = incident["spent"]["recover"]
            if used != self.spent.get(incident_id, 0):
                raise RecoveryContractError("durable recovery bound diverged from the progress ledger")
        for incident_id, used in self.spent.items():
            if incident_id not in self.ledger.incidents or self.ledger.incidents[incident_id]["spent"]["recover"] != used:
                raise RecoveryContractError("durable recovery bound diverged from the progress ledger")


def _same_owner(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return all(left[key] == right[key] for key in ("worker_id", "pid", "birth", "process_generation", "lease_generation"))


def _same_birth(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return left["pid"] == right["pid"] and left["birth"] == right["birth"]


def _optional_owner(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != _OWNER_FIELDS:
        raise RecoveryContractError("lifecycle owner binding is malformed")
    return {
        "worker_id": _identifier(value.get("worker_id"), "worker_id"),
        "pid": _whole(value.get("pid"), "pid", minimum=1),
        "birth": _whole(value.get("birth"), "pid birth", minimum=1),
        "process_generation": _whole(value.get("process_generation"), "process generation", minimum=1),
        "lease_generation": _whole(value.get("lease_generation"), "lease generation", minimum=1),
    }


def _claim_list(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise RecoveryContractError("claims must be a list")
    claims = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, Mapping) or set(item) - _CLAIM_FIELDS or not _OWNER_FIELDS <= set(item) or "claim_id" not in item or "task_id" not in item:
            raise RecoveryContractError("claim binding is malformed")
        claim_id = _identifier(item.get("claim_id"), "claim_id")
        if claim_id in seen:
            raise RecoveryContractError("claim id repeated")
        seen.add(claim_id)
        body = {
            "claim_id": claim_id,
            "task_id": _identifier(item.get("task_id"), "task_id"),
            "worker_id": _identifier(item.get("worker_id"), "worker_id"),
            "pid": _whole(item.get("pid"), "pid", minimum=1),
            "birth": _whole(item.get("birth"), "pid birth", minimum=1),
            "process_generation": _whole(item.get("process_generation"), "process generation", minimum=1),
            "lease_generation": _whole(item.get("lease_generation"), "lease generation", minimum=1),
            "effect_id": _identifier(item["effect_id"], "effect_id") if item.get("effect_id") else "",
        }
        body["binding"] = digest(body)
        claims.append(body)
    return claims


def _effect_list(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise RecoveryContractError("effects must be a list")
    effects = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, Mapping) or set(item) != _EFFECT_FIELDS:
            raise RecoveryContractError("effect binding is malformed")
        effect_id = _identifier(item.get("effect_id"), "effect_id")
        if effect_id in seen:
            raise RecoveryContractError("effect id repeated")
        seen.add(effect_id)
        status = item.get("status")
        if status not in EFFECT_STATUSES:
            raise RecoveryContractError("effect status is not recognized")
        effects.append({
            "effect_id": effect_id,
            "task_id": _identifier(item.get("task_id"), "task_id"),
            "status": status,
            "digest": _digest(item.get("digest"), "effect digest"),
        })
    return effects


def _task_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise RecoveryContractError("runnable queue must be a list")
    tasks = [_identifier(item, "task_id") for item in value]
    if len(tasks) != len(set(tasks)):
        raise RecoveryContractError("runnable queue repeats a task")
    return sorted(tasks)


def _markers(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise RecoveryContractError("hold markers must be a list")
    markers = []
    for item in value:
        if item not in HOLD_MARKERS:
            raise RecoveryContractError("hold marker is not recognized")
        if item not in markers:
            markers.append(item)
    return markers


def _epoch(run_id: str, normalized: Mapping[str, Any]) -> str:
    return digest({
        "run_id": run_id,
        "runnable_task_ids": list(normalized["runnable_task_ids"]),
        "idle_since": normalized["idle_since"],
        "queue_deadline_at": normalized["queue_deadline_at"],
    })


def _requeue_tasks(preview: Mapping[str, Any], runnable: Sequence[str], *, only_stale: bool = False) -> list[str]:
    blocked = set(preview["unknown_tasks"])
    applied = {item["task_id"] for item in preview["effects"].values() if item["status"] == "applied"}
    tasks: list[str] = []
    for claim in preview["claims"]:
        if only_stale and not claim["stale"]:
            continue
        if claim["current"]:
            continue
        task_id = claim["task_id"]
        if task_id in blocked or task_id in applied:
            continue
        effect_id = claim.get("effect_id") or ""
        effect = preview["effects"].get(effect_id) if effect_id else None
        if effect is None:
            # No linked effect: the claim itself is not proof of an unknown side effect.
            if task_id not in tasks:
                tasks.append(task_id)
            continue
        if effect["status"] == "known_none" and task_id not in tasks:
            tasks.append(task_id)
    for task_id in runnable:
        if task_id not in blocked and task_id not in applied and task_id not in tasks:
            tasks.append(task_id)
    return tasks


def _unique(items: Sequence[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        if item not in result:
            result.append(item)
    return result


def _restore_owner(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise RecoveryContractError("stored lifecycle owner is malformed")
    owner = _optional_owner({key: value.get(key) for key in _OWNER_FIELDS})
    if owner is None:
        raise RecoveryContractError("stored lifecycle owner is malformed")
    if type(value.get("fenced")) is not bool:
        raise RecoveryContractError("stored lifecycle owner is malformed")
    owner["fenced"] = value["fenced"]
    reason = value.get("fence_reason", "")
    if reason not in {"", *DISPOSITIONS}:
        raise RecoveryContractError("stored fence reason is not recognized")
    owner["fence_reason"] = reason
    return owner


def _restore_fence(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RecoveryContractError("stored fence is malformed")
    reason = value.get("reason")
    if reason not in DISPOSITIONS:
        raise RecoveryContractError("stored fence reason is not recognized")
    return {
        "worker_id": _identifier(value.get("worker_id"), "worker_id"),
        "pid": _whole(value.get("pid"), "pid", minimum=1),
        "birth": _whole(value.get("birth"), "pid birth", minimum=1),
        "process_generation": _whole(value.get("process_generation"), "process generation", minimum=1),
        "lease_generation": _whole(value.get("lease_generation"), "lease generation", minimum=1),
        "reason": reason,
    }


def _restore_claims(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise RecoveryContractError("stored claims are malformed")
    restored: dict[str, dict[str, Any]] = {}
    for key, item in value.items():
        _identifier(key, "claim_id")
        if not isinstance(item, Mapping) or item.get("state") not in {"current", "released", "open"}:
            raise RecoveryContractError("stored claim is malformed")
        claim = {
            "claim_id": _identifier(item.get("claim_id"), "claim_id"),
            "task_id": _identifier(item.get("task_id"), "task_id"),
            "worker_id": _identifier(item.get("worker_id"), "worker_id"),
            "pid": _whole(item.get("pid"), "pid", minimum=1),
            "birth": _whole(item.get("birth"), "pid birth", minimum=1),
            "process_generation": _whole(item.get("process_generation"), "process generation", minimum=1),
            "lease_generation": _whole(item.get("lease_generation"), "lease generation", minimum=1),
            "effect_id": _identifier(item["effect_id"], "effect_id") if item.get("effect_id") else "",
        }
        if claim["claim_id"] != key:
            raise RecoveryContractError("stored claim id does not match its key")
        claim["binding"] = digest({key: claim[key] for key in ("claim_id", "task_id", "worker_id", "pid", "birth", "process_generation", "lease_generation", "effect_id")})
        if item.get("binding") != claim["binding"]:
            raise RecoveryContractError("stored claim binding was rewritten")
        claim["state"] = item["state"]
        restored[key] = claim
    return restored


def _restore_effects(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise RecoveryContractError("stored effects are malformed")
    restored: dict[str, dict[str, Any]] = {}
    for key, item in value.items():
        _identifier(key, "effect_id")
        if not isinstance(item, Mapping) or set(item) != _EFFECT_FIELDS:
            raise RecoveryContractError("stored effect is malformed")
        effect = _effect_list([item])[0]
        if effect["effect_id"] != key:
            raise RecoveryContractError("stored effect id does not match its key")
        restored[key] = effect
    return restored
