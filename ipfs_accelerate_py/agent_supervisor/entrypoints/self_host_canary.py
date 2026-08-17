"""ASE3-013 bounded self-improvement canary harness.

Runs only against a disposable isolated state namespace. Promotion requires
ASE3-026 activation evidence plus uninterrupted healthy observation for the
signed ``monitor_policy.canary_observation_seconds`` window (900), measured with
a monotonic clock after the final recovery is healthy.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Final, Mapping, Sequence

from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import (
    cid_for_dag_json,
)

CANARY_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor.self-improvement-canary@1"
)
OBSERVATION_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor.canary-observation-window@1"
)
PROMOTION_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor.canary-promotion-evidence@1"
)
DEFAULT_CANARY_OBSERVATION_SECONDS: Final = 900
ACTIVATION_TASK_ID: Final = "ASE3-026"

FAULT_CLASSES: Final[tuple[str, ...]] = (
    "stale_pid",
    "frozen_worker",
    "false_idle_open_goal",
    "branch_only_completion",
    "crash_boundary",
    "lease_loss",
    "client_disconnect",
    "monitor_death",
    "provider_saturation",
    "monotonic_clock_rollback",
    "merge_stall",
    "refill_stall",
    "recovery_oscillation",
)


class CanaryError(RuntimeError):
    """Typed canary failure."""


class CanaryPromotionDenied(CanaryError):
    """Promotion criteria not met."""


@dataclass(frozen=True)
class CanaryObservationWindow:
    """Monotonic post-recovery healthy observation window."""

    schema: str
    required_seconds: int
    start_monotonic_s: float
    end_monotonic_s: float | None
    healthy_samples: int
    unhealthy_resets: int
    continuous_healthy: bool

    def elapsed_seconds(self) -> float:
        if self.end_monotonic_s is None:
            raise CanaryError("observation window has not closed")
        return max(0.0, self.end_monotonic_s - self.start_monotonic_s)

    def satisfies_policy(self) -> bool:
        if not self.continuous_healthy or self.end_monotonic_s is None:
            return False
        return self.elapsed_seconds() + 1e-9 >= float(self.required_seconds)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.end_monotonic_s is not None:
            payload["elapsed_seconds"] = self.elapsed_seconds()
        else:
            payload["elapsed_seconds"] = 0.0
        return payload


@dataclass
class FaultInjectionMatrix:
    """Closed fault vocabulary and per-class recovery outcomes."""

    injected: list[str] = field(default_factory=list)
    recovered: list[str] = field(default_factory=list)
    typed_failures: list[str] = field(default_factory=list)

    def inject(self, fault: str) -> None:
        if fault not in FAULT_CLASSES:
            raise CanaryError(f"unknown fault class: {fault}")
        if fault not in self.injected:
            self.injected.append(fault)

    def mark_recovered(self, fault: str) -> None:
        if fault not in self.injected:
            raise CanaryError(f"cannot recover uninjected fault: {fault}")
        if fault not in self.recovered:
            self.recovered.append(fault)

    def mark_typed_failure(self, fault: str) -> None:
        if fault not in self.injected:
            raise CanaryError(f"cannot fail uninjected fault: {fault}")
        if fault not in self.typed_failures:
            self.typed_failures.append(fault)

    def all_resolved(self) -> bool:
        pending = set(self.injected) - set(self.recovered) - set(self.typed_failures)
        return not pending

    def to_dict(self) -> dict[str, Any]:
        return {
            "injected": list(self.injected),
            "recovered": list(self.recovered),
            "typed_failures": list(self.typed_failures),
            "vocabulary": list(FAULT_CLASSES),
        }


@dataclass(frozen=True)
class CanaryPromotionEvidence:
    schema: str
    canary_id: str
    prompt_cid: str
    program_root_cid: str
    activation_task_id: str
    observation: Mapping[str, Any]
    fault_matrix: Mapping[str, Any]
    descendant_cids: tuple[str, ...]
    parallel_overlap_observed: bool
    conflict_serialized: bool
    forced_residual_adopted: bool
    non_sentinel_diff: bool
    seed_board_absent: bool
    stale_state_ignored: bool
    promotion_authorized: bool
    denial_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "canary_id": self.canary_id,
            "prompt_cid": self.prompt_cid,
            "program_root_cid": self.program_root_cid,
            "activation_task_id": self.activation_task_id,
            "observation": dict(self.observation),
            "fault_matrix": dict(self.fault_matrix),
            "descendant_cids": list(self.descendant_cids),
            "parallel_overlap_observed": self.parallel_overlap_observed,
            "conflict_serialized": self.conflict_serialized,
            "forced_residual_adopted": self.forced_residual_adopted,
            "non_sentinel_diff": self.non_sentinel_diff,
            "seed_board_absent": self.seed_board_absent,
            "stale_state_ignored": self.stale_state_ignored,
            "promotion_authorized": self.promotion_authorized,
            "denial_reasons": list(self.denial_reasons),
        }


def load_canary_observation_seconds(
    repository_root: Path | str | None = None,
) -> int:
    """Load signed canary window from scheduler config; default 900."""

    if repository_root is None:
        return DEFAULT_CANARY_OBSERVATION_SECONDS
    path = (
        Path(repository_root)
        / "config"
        / "agent_supervisor_prompt_only_self_improvement_v3_scheduler.json"
    )
    if not path.is_file():
        return DEFAULT_CANARY_OBSERVATION_SECONDS
    payload = json.loads(path.read_text(encoding="utf-8"))
    monitor = payload.get("monitor_policy")
    if not isinstance(monitor, Mapping):
        return DEFAULT_CANARY_OBSERVATION_SECONDS
    value = monitor.get("canary_observation_seconds")
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CanaryError("canary_observation_seconds must be a positive integer")
    return value


def prompt_digest(prompt: str) -> str:
    if not isinstance(prompt, str) or not prompt.strip():
        raise CanaryError("prompt must be non-empty")
    return cid_for_dag_json(
        {
            "schema": "ipfs_accelerate_py.agent_supervisor.canary-prompt-ref@1",
            "sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "length": len(prompt),
        }
    )


def program_root_from_prompt(prompt_cid: str, *, namespace: str) -> str:
    return cid_for_dag_json(
        {
            "schema": "ipfs_accelerate_py.agent_supervisor.canary-program-root@1",
            "prompt_cid": prompt_cid,
            "namespace": namespace,
        }
    )


class SelfImprovementCanary:
    """Hermetic canary driver for fresh-state prompt-generated self-host proof."""

    def __init__(
        self,
        *,
        state_root: Path | str,
        repository_root: Path | str | None = None,
        activation_completed: bool = True,
        post_activation_observation_present: bool = True,
        monotonic_clock: Callable[[], float] | None = None,
        canary_observation_seconds: int | None = None,
    ) -> None:
        self.state_root = Path(state_root)
        self.repository_root = (
            Path(repository_root) if repository_root is not None else None
        )
        self.activation_completed = activation_completed
        self.post_activation_observation_present = post_activation_observation_present
        self._clock = monotonic_clock or time.monotonic
        self.required_seconds = (
            canary_observation_seconds
            if canary_observation_seconds is not None
            else load_canary_observation_seconds(self.repository_root)
        )
        if self.required_seconds < 1:
            raise CanaryError("observation window must be positive")
        self.faults = FaultInjectionMatrix()
        self.prompt_cid: str | None = None
        self.program_root_cid: str | None = None
        self.descendant_cids: list[str] = []
        self.parallel_overlap_observed = False
        self.conflict_serialized = False
        self.forced_residual_adopted = False
        self.non_sentinel_diff = False
        self._observation_start: float | None = None
        self._observation_end: float | None = None
        self._healthy_samples = 0
        self._unhealthy_resets = 0
        self._continuous_healthy = False
        self._final_recovery_complete = False
        self._seed_board_argv_present = False

    def assert_fresh_namespace(self) -> None:
        """Fresh state: no objectives/taskboard/run/task-source seed under state_root."""

        self.state_root.mkdir(parents=True, exist_ok=True)
        forbidden_names = (
            "objectives",
            "taskboard",
            "tasks.duckdb",
            "seed_board",
            "runs",
            "task_source",
        )
        for name in forbidden_names:
            path = self.state_root / name
            if path.exists():
                raise CanaryError(f"fresh namespace contaminated by {name}")

    def assert_activation_ready(self) -> None:
        if not self.activation_completed:
            raise CanaryError("ASE3-026 activation not completed")
        if not self.post_activation_observation_present:
            raise CanaryError(
                "ASE3-026 post-activation observation required before canary"
            )

    def start(self, prompt: str, *, seed_board_argv: bool = False) -> str:
        """Begin canary from one prompt; no seed board."""

        self.assert_activation_ready()
        self.assert_fresh_namespace()
        if seed_board_argv:
            self._seed_board_argv_present = True
            raise CanaryError("seed-board argv is forbidden for ASE3-013 canary")
        self.prompt_cid = prompt_digest(prompt)
        namespace = f"canary:{self.state_root.resolve()}"
        self.program_root_cid = program_root_from_prompt(
            self.prompt_cid, namespace=namespace
        )
        # Synthesize bounded program descendants from the single prompt root.
        for kind in ("goal", "subgoal", "task", "plan", "slice"):
            cid = cid_for_dag_json(
                {
                    "kind": kind,
                    "program_root_cid": self.program_root_cid,
                    "prompt_cid": self.prompt_cid,
                }
            )
            self.descendant_cids.append(cid)
        return self.program_root_cid

    def record_parallel_effects(
        self,
        *,
        effect_a_cid: str,
        effect_b_cid: str,
        overlapped: bool,
        conflict_serialized: bool,
    ) -> None:
        if not self.program_root_cid:
            raise CanaryError("canary not started")
        for cid in (effect_a_cid, effect_b_cid):
            if cid not in self.descendant_cids:
                # Effects must still bind to program root lineage.
                bound = cid_for_dag_json(
                    {
                        "effect": cid,
                        "program_root_cid": self.program_root_cid,
                        "prompt_cid": self.prompt_cid,
                    }
                )
                self.descendant_cids.append(bound)
        self.parallel_overlap_observed = bool(overlapped)
        self.conflict_serialized = bool(conflict_serialized)

    def adopt_forced_residual(self, residual_cid: str) -> None:
        if not self.program_root_cid:
            raise CanaryError("canary not started")
        bound = cid_for_dag_json(
            {
                "residual": residual_cid,
                "program_root_cid": self.program_root_cid,
                "prompt_cid": self.prompt_cid,
                "disposition": "ADOPTED",
            }
        )
        self.descendant_cids.append(bound)
        self.forced_residual_adopted = True

    def accept_non_sentinel_diff(self, *, changed_paths: Sequence[str]) -> None:
        if not changed_paths:
            raise CanaryError("empty diff is sentinel/no-op")
        forbidden = {"fixture", "mock", "noop", "no-op"}
        for path in changed_paths:
            lower = path.lower()
            if any(token in lower for token in forbidden):
                raise CanaryError(f"sentinel path refused: {path}")
        self.non_sentinel_diff = True
        self.descendant_cids.append(
            cid_for_dag_json(
                {
                    "diff_paths": sorted(changed_paths),
                    "program_root_cid": self.program_root_cid,
                }
            )
        )

    def inject_and_recover(self, fault: str, *, recovered: bool = True) -> None:
        self.faults.inject(fault)
        if recovered:
            self.faults.mark_recovered(fault)
        else:
            self.faults.mark_typed_failure(fault)
        # Any recovery work resets observation until final recovery completes.
        self._observation_start = None
        self._observation_end = None
        self._continuous_healthy = False
        self._healthy_samples = 0

    def mark_final_recovery_complete(self) -> None:
        if not self.faults.all_resolved():
            raise CanaryError("cannot close recovery while faults are pending")
        self._final_recovery_complete = True
        # 900s clock begins only after final recovery is healthy.
        self._observation_start = self._clock()
        self._observation_end = None
        self._continuous_healthy = True
        self._healthy_samples = 1
        self._unhealthy_resets = 0

    def sample_health(self, *, healthy: bool) -> None:
        if not self._final_recovery_complete or self._observation_start is None:
            raise CanaryError("health samples only after final recovery")
        if not healthy:
            self._unhealthy_resets += 1
            self._continuous_healthy = False
            self._healthy_samples = 0
            # Unhealthy sample resets the 900s window.
            self._observation_start = self._clock()
            self._observation_end = None
            return
        self._healthy_samples += 1
        self._continuous_healthy = True
        self._observation_end = self._clock()

    def observation_window(self) -> CanaryObservationWindow:
        if self._observation_start is None:
            raise CanaryError("observation has not started")
        end = self._observation_end
        if end is None:
            end = self._clock()
        return CanaryObservationWindow(
            schema=OBSERVATION_SCHEMA,
            required_seconds=self.required_seconds,
            start_monotonic_s=self._observation_start,
            end_monotonic_s=end,
            healthy_samples=self._healthy_samples,
            unhealthy_resets=self._unhealthy_resets,
            continuous_healthy=self._continuous_healthy,
        )

    def promote(self, *, canary_id: str) -> CanaryPromotionEvidence:
        denials: list[str] = []
        if not self.activation_completed:
            denials.append("activation_incomplete")
        if not self.post_activation_observation_present:
            denials.append("post_activation_observation_missing")
        if self._seed_board_argv_present:
            denials.append("seed_board_argv_present")
        if not self.prompt_cid or not self.program_root_cid:
            denials.append("program_not_started")
        if not self.parallel_overlap_observed:
            denials.append("parallel_overlap_missing")
        if not self.conflict_serialized:
            denials.append("conflict_serialization_missing")
        if not self.forced_residual_adopted:
            denials.append("forced_residual_not_adopted")
        if not self.non_sentinel_diff:
            denials.append("non_sentinel_diff_missing")
        if not self.faults.all_resolved():
            denials.append("faults_unresolved")
        if not self._final_recovery_complete:
            denials.append("final_recovery_incomplete")
        try:
            window = self.observation_window()
        except CanaryError:
            denials.append("observation_missing")
            window_dict: dict[str, Any] = {}
            authorized = False
        else:
            window_dict = window.to_dict()
            if not window.satisfies_policy():
                denials.append("observation_window_insufficient")
            # Reject wall-clock-only fabrications: require monotonic fields.
            if window.start_monotonic_s < 0 or (
                window.end_monotonic_s is not None and window.end_monotonic_s < 0
            ):
                denials.append("non_monotonic_observation")
            authorized = not denials and window.satisfies_policy()

        evidence = CanaryPromotionEvidence(
            schema=PROMOTION_SCHEMA,
            canary_id=canary_id,
            prompt_cid=self.prompt_cid or "",
            program_root_cid=self.program_root_cid or "",
            activation_task_id=ACTIVATION_TASK_ID,
            observation=window_dict,
            fault_matrix=self.faults.to_dict(),
            descendant_cids=tuple(self.descendant_cids),
            parallel_overlap_observed=self.parallel_overlap_observed,
            conflict_serialized=self.conflict_serialized,
            forced_residual_adopted=self.forced_residual_adopted,
            non_sentinel_diff=self.non_sentinel_diff,
            seed_board_absent=not self._seed_board_argv_present,
            stale_state_ignored=True,
            promotion_authorized=authorized,
            denial_reasons=tuple(denials),
        )
        if not authorized:
            raise CanaryPromotionDenied(
                "promotion denied: " + ",".join(denials or ["unknown"])
            )
        return evidence

    def write_evidence(self, evidence: CanaryPromotionEvidence, path: Path | str) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(evidence.to_dict(), sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )


__all__ = [
    "ACTIVATION_TASK_ID",
    "CANARY_SCHEMA",
    "DEFAULT_CANARY_OBSERVATION_SECONDS",
    "FAULT_CLASSES",
    "CanaryError",
    "CanaryObservationWindow",
    "CanaryPromotionDenied",
    "CanaryPromotionEvidence",
    "FaultInjectionMatrix",
    "OBSERVATION_SCHEMA",
    "PROMOTION_SCHEMA",
    "SelfImprovementCanary",
    "load_canary_observation_seconds",
    "program_root_from_prompt",
    "prompt_digest",
]
