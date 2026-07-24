"""Configured end-to-end boundary for Leanstral-assisted goal development.

The lower-level goal-development, refinement, objective-admission, and
completion modules deliberately do not own one another.  This module provides
the small supervisor boundary which composes them without weakening those
trust boundaries:

* the configured mode defaults to ``shadow``;
* every model result is reparsed through the versioned result contract;
* multiple candidates are retained and one candidate is selected
  deterministically for preview;
* objective materialization remains delegated to ``objective_daemon``; and
* a restartable audit record and a proof-metrics projection are written even
  when every optional model route falls back.

This boundary has no completion authority.  Code-conformance and completion
decisions remain the responsibility of ``code_proof_obligations`` and
``goal_completion`` and may be recorded in the audit metadata by callers.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from .formal_verification_contracts import ContractValidationError
from .goal_development_contracts import (
    GoalAdmissionDecision,
    GoalDevelopmentAdmissionReceipt,
    GoalDevelopmentMode,
    GoalDevelopmentProposalReceipt,
    GoalProposalDecision,
)
from .leanstral_goal_development import (
    GoalDevelopmentFallbackReason,
    GoalDevelopmentProviderResult,
    GoalDevelopmentResultStatus,
    LeanstralGoalDevelopmentInvocation,
    LeanstralGoalDevelopmentProvider,
    create_leanstral_goal_development_provider,
)
from .objective_daemon import (
    ObjectiveGenerationAdmissionResult,
    materialize_admitted_objective_work,
)
from .proof_metrics import (
    ProofMetricsSnapshot,
    build_proof_metrics_snapshot,
    write_proof_metrics_snapshot,
)


LEANSTRAL_GOAL_LIFECYCLE_VERSION: Final = "1.0.0"
LEANSTRAL_GOAL_LIFECYCLE_RUN_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/leanstral-goal-lifecycle-run@1"
)
LEANSTRAL_GOAL_LIFECYCLE_AUDIT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/leanstral-goal-lifecycle-audit@1"
)
DEFAULT_LEANSTRAL_GOAL_LIFECYCLE_AUDIT_FILE: Final = (
    "leanstral-goal-lifecycle.audit.jsonl"
)
DEFAULT_LEANSTRAL_GOAL_LIFECYCLE_STATE_FILE: Final = (
    "leanstral-goal-lifecycle.state.json"
)
DEFAULT_LEANSTRAL_GOAL_LIFECYCLE_METRICS_FILE: Final = (
    "leanstral-goal-lifecycle.metrics.json"
)
DEFAULT_LEANSTRAL_GOAL_LIFECYCLE_GENERATION_FILE: Final = (
    "leanstral-goal-generation.json"
)
DEFAULT_LEANSTRAL_GOAL_LIFECYCLE_MAX_CANDIDATES: Final = 3
MAX_LEANSTRAL_GOAL_LIFECYCLE_CANDIDATES: Final = 8


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_copy(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    )


def _content_id(value: Mapping[str, Any], *, prefix: str) -> str:
    material = dict(value)
    material.pop("run_id", None)
    material.pop("started_at", None)
    material.pop("finished_at", None)
    return prefix + hashlib.sha256(
        json.dumps(
            material,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _file_digest(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        data = b""
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


@dataclass(frozen=True)
class LeanstralGoalLifecycleConfig:
    """Explicit operational controls for one configured supervisor path."""

    state_dir: Path | str
    mode: GoalDevelopmentMode | str = GoalDevelopmentMode.SHADOW
    max_candidates: int = DEFAULT_LEANSTRAL_GOAL_LIFECYCLE_MAX_CANDIDATES
    validator_id: str = "agent-supervisor:goal-contract-validator@1"
    admitter_id: str = "agent-supervisor:objective-admission@1"
    lifecycle_owner: str = "leanstral_goal_lifecycle"
    queryable_metrics: bool = False
    audit_filename: str = DEFAULT_LEANSTRAL_GOAL_LIFECYCLE_AUDIT_FILE
    state_filename: str = DEFAULT_LEANSTRAL_GOAL_LIFECYCLE_STATE_FILE
    metrics_filename: str = DEFAULT_LEANSTRAL_GOAL_LIFECYCLE_METRICS_FILE
    generation_filename: str = DEFAULT_LEANSTRAL_GOAL_LIFECYCLE_GENERATION_FILE

    def __post_init__(self) -> None:
        try:
            raw_state_dir = os.fspath(self.state_dir)
        except TypeError as exc:
            raise ContractValidationError(
                "state_dir must be a filesystem path"
            ) from exc
        if not isinstance(raw_state_dir, str) or not raw_state_dir.strip():
            raise ContractValidationError("state_dir must not be empty")
        state_dir = Path(raw_state_dir)
        try:
            mode = (
                self.mode
                if isinstance(self.mode, GoalDevelopmentMode)
                else GoalDevelopmentMode(str(self.mode))
            )
        except ValueError as exc:
            raise ContractValidationError("unsupported goal lifecycle mode") from exc
        if (
            isinstance(self.max_candidates, bool)
            or not isinstance(self.max_candidates, int)
            or not 1 <= self.max_candidates <= MAX_LEANSTRAL_GOAL_LIFECYCLE_CANDIDATES
        ):
            raise ContractValidationError(
                "max_candidates must be between 1 and "
                f"{MAX_LEANSTRAL_GOAL_LIFECYCLE_CANDIDATES}"
            )
        for name in (
            "validator_id",
            "admitter_id",
            "lifecycle_owner",
            "audit_filename",
            "state_filename",
            "metrics_filename",
            "generation_filename",
        ):
            value = str(getattr(self, name) or "").strip()
            if not value:
                raise ContractValidationError(f"{name} must not be empty")
            if (
                name.endswith("filename")
                and (value in {".", ".."} or Path(value).name != value)
            ):
                raise ContractValidationError(f"{name} must be a plain file name")
            object.__setattr__(self, name, value)
        filenames = (
            self.audit_filename,
            self.state_filename,
            self.metrics_filename,
            self.generation_filename,
        )
        if len(set(filenames)) != len(filenames):
            raise ContractValidationError(
                "goal lifecycle artifact filenames must be distinct"
            )
        if not isinstance(self.queryable_metrics, bool):
            raise ContractValidationError("queryable_metrics must be a boolean")
        object.__setattr__(self, "state_dir", state_dir)
        object.__setattr__(self, "mode", mode)

    @property
    def audit_path(self) -> Path:
        return self.state_dir / self.audit_filename

    @property
    def state_path(self) -> Path:
        return self.state_dir / self.state_filename

    @property
    def metrics_path(self) -> Path:
        return self.state_dir / self.metrics_filename

    @property
    def generation_path(self) -> Path:
        return self.state_dir / self.generation_filename


@dataclass(frozen=True)
class LeanstralGoalLifecycleRun(Mapping[str, Any]):
    """Restartable public projection of one configured lifecycle pass."""

    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        copied = _json_copy(dict(self.payload))
        if copied.get("schema") != LEANSTRAL_GOAL_LIFECYCLE_RUN_SCHEMA:
            raise ContractValidationError("unsupported goal lifecycle run schema")
        expected = _content_id(copied, prefix="leanstral-goal-run-")
        if copied.get("run_id") != expected:
            raise ContractValidationError("goal lifecycle run identity is invalid")
        if copied.get("mode") not in {item.value for item in GoalDevelopmentMode}:
            raise ContractValidationError("goal lifecycle run mode is invalid")
        object.__setattr__(self, "payload", copied)

    @property
    def run_id(self) -> str:
        return str(self.payload["run_id"])

    @property
    def mode(self) -> GoalDevelopmentMode:
        return GoalDevelopmentMode(str(self.payload["mode"]))

    @property
    def selected_draft_id(self) -> str:
        return str(self.payload.get("selected_draft_id") or "")

    @property
    def objective_heap_unchanged(self) -> bool:
        return bool(self.payload.get("objective_heap_unchanged"))

    @property
    def completion_state_unchanged(self) -> bool:
        return bool(self.payload.get("completion_state_unchanged"))

    @property
    def generation_state_unchanged(self) -> bool:
        return bool(self.payload.get("generation_state_unchanged"))

    def to_dict(self) -> dict[str, Any]:
        return _json_copy(self.payload)

    def __getitem__(self, key: str) -> Any:
        return self.payload[key]

    def __iter__(self):
        return iter(self.payload)

    def __len__(self) -> int:
        return len(self.payload)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LeanstralGoalLifecycleRun":
        return cls(payload)


@dataclass(frozen=True)
class _CandidateObservation:
    index: int
    result: GoalDevelopmentProviderResult
    schema_accepted: bool
    reason_codes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_index": self.index,
            "schema_accepted": self.schema_accepted,
            "reason_codes": list(self.reason_codes),
            "result": self.result.to_dict(),
        }


class ConfiguredLeanstralGoalLifecycleSupervisor:
    """Run the reviewed goal-development path with durable shadow auditing."""

    def __init__(
        self,
        config: LeanstralGoalLifecycleConfig,
        *,
        providers: Sequence[LeanstralGoalDevelopmentProvider | Any] = (),
    ) -> None:
        if not isinstance(config, LeanstralGoalLifecycleConfig):
            raise ContractValidationError(
                "config must be LeanstralGoalLifecycleConfig"
            )
        selected = tuple(providers) or (create_leanstral_goal_development_provider(),)
        if len(selected) > config.max_candidates:
            raise ContractValidationError(
                "configured providers exceed max_candidates"
            )
        if any(not callable(getattr(item, "develop", None)) for item in selected):
            raise ContractValidationError(
                "every goal lifecycle provider must expose develop()"
            )
        self.config = config
        self.providers = selected

    def _candidate(
        self,
        index: int,
        provider: Any,
        invocation: LeanstralGoalDevelopmentInvocation,
    ) -> _CandidateObservation:
        try:
            raw = provider.develop(invocation)
            if isinstance(raw, GoalDevelopmentProviderResult):
                result = GoalDevelopmentProviderResult.from_dict(raw.to_dict())
            elif isinstance(raw, Mapping):
                result = GoalDevelopmentProviderResult.from_dict(raw)
            else:
                raise TypeError(
                    "goal-development provider returned an unsupported result type"
                )
            if result.request_id != invocation.request_id:
                raise ContractValidationError(
                    "candidate changed the frozen request identity"
                )
            accepted = result.status is GoalDevelopmentResultStatus.DRAFT
            reasons = (
                ()
                if accepted
                else (
                    "deterministic_fallback",
                    (
                        result.fallback_reason.value
                        if result.fallback_reason is not None
                        else "unknown_fallback"
                    ),
                )
            )
            return _CandidateObservation(index, result, accepted, reasons)
        except (ContractValidationError, TypeError, ValueError):
            result = GoalDevelopmentProviderResult(
                request_id=invocation.request_id,
                status=GoalDevelopmentResultStatus.DETERMINISTIC_FALLBACK,
                fallback_reason=GoalDevelopmentFallbackReason.MALFORMED_OUTPUT,
            )
            return _CandidateObservation(
                index,
                result,
                False,
                ("type_or_schema_rejection",),
            )
        except Exception:
            # Optional route failures cannot stall or authorize the supervisor.
            result = GoalDevelopmentProviderResult(
                request_id=invocation.request_id,
                status=GoalDevelopmentResultStatus.DETERMINISTIC_FALLBACK,
                fallback_reason=GoalDevelopmentFallbackReason.UNAVAILABLE,
            )
            return _CandidateObservation(
                index,
                result,
                False,
                ("provider_unavailable",),
            )

    @staticmethod
    def _select_candidate(
        candidates: Sequence[_CandidateObservation],
    ) -> _CandidateObservation | None:
        drafts = [
            item
            for item in candidates
            if item.schema_accepted and item.result.draft is not None
        ]
        if not drafts:
            return None
        return sorted(
            drafts,
            key=lambda item: (
                -len(item.result.draft.proposals),
                item.result.draft.draft_id,
                item.index,
            ),
        )[0]

    def _non_admission_receipt(
        self,
        receipt: GoalDevelopmentProposalReceipt,
    ) -> GoalDevelopmentAdmissionReceipt | None:
        mode = self.config.mode
        if mode is GoalDevelopmentMode.AUTO_SAFE:
            return None
        if mode is GoalDevelopmentMode.ASSIST:
            decision = GoalAdmissionDecision.REVIEW_REQUIRED
            reasons = ("operator_review_required",)
        else:
            decision = GoalAdmissionDecision.NOT_ADMITTED
            reasons = (
                "shadow_mode"
                if mode is GoalDevelopmentMode.SHADOW
                else f"{mode.value}_mode_not_admissible",
            )
        return GoalDevelopmentAdmissionReceipt.for_proposal(
            receipt,
            mode=mode,
            admitter_id=self.config.admitter_id,
            decision=decision,
            reason_codes=reasons,
        )

    def run(
        self,
        invocation: LeanstralGoalDevelopmentInvocation | Mapping[str, Any],
        *,
        repo_root: Path | str,
        objective_path: Path | str,
        objective_work: Iterable[Any] = (),
        completion_state_path: Path | str | None = None,
        generation_path: Path | str | None = None,
        admission_receipt: (
            GoalDevelopmentAdmissionReceipt | Mapping[str, Any] | None
        ) = None,
        refinement_verification: Any = None,
        authoritative_receipts: Iterable[Any] | Mapping[str, Any] = (),
        required_authoritative_receipt_ids: Sequence[str] = (),
        proposal_bindings: Mapping[str, str] | None = None,
        limits: Any = None,
        hard_policy_gates: Mapping[str, bool] | None = None,
        new_assumption_ids: Sequence[str] = (),
        unsupported_semantics: Sequence[str] = (),
        implementation_conformance: Mapping[str, Any] | None = None,
        completion_decision: Mapping[str, Any] | None = None,
    ) -> LeanstralGoalLifecycleRun:
        """Execute one pass and persist audit, metrics, and restart state.

        ``objective_work`` is the deterministic, reviewed projection from a
        selected decomposition into objective-heap records.  The model cannot
        supply paths, commands, formulas, or canonical heap text through this
        method.
        """

        call = (
            invocation
            if isinstance(invocation, LeanstralGoalDevelopmentInvocation)
            else LeanstralGoalDevelopmentInvocation.from_dict(invocation)
        )
        if call.request.mode is not self.config.mode:
            raise ContractValidationError(
                "invocation mode does not match configured supervisor mode"
            )
        repo = Path(repo_root)
        objective = Path(objective_path)
        completion = (
            None if completion_state_path is None else Path(completion_state_path)
        )
        generation = (
            self.config.generation_path
            if generation_path is None
            else Path(generation_path)
        )
        objective_records = tuple(objective_work)
        started_at = _utc_now()
        objective_before = _file_digest(objective)
        completion_before = _file_digest(completion)
        generation_before = _file_digest(generation)

        candidates = tuple(
            self._candidate(index, provider, call)
            for index, provider in enumerate(
                self.providers[: self.config.max_candidates]
            )
        )
        selected = self._select_candidate(candidates)
        draft = None if selected is None else selected.result.draft
        proposal_receipt = (
            None
            if draft is None
            else GoalDevelopmentProposalReceipt.for_draft(
                draft,
                validator_id=self.config.validator_id,
                decision=GoalProposalDecision.ACCEPTED,
            )
        )
        parsed_admission = admission_receipt
        if parsed_admission is not None and not isinstance(
            parsed_admission, GoalDevelopmentAdmissionReceipt
        ):
            parsed_admission = GoalDevelopmentAdmissionReceipt.from_dict(
                parsed_admission
            )
        if parsed_admission is None and proposal_receipt is not None:
            parsed_admission = self._non_admission_receipt(proposal_receipt)

        admission: ObjectiveGenerationAdmissionResult = (
            materialize_admitted_objective_work(
                objective_records,
                repo_root=repo,
                objective_path=objective,
                generation_path=generation,
                mode=self.config.mode,
                limits=limits,
                root_goal_id=call.request.root_goal_id,
                expected_root_content_id=call.request.root_goal_content_id,
                expected_repository_tree_id=call.request.repository_tree_id,
                lifecycle_owner=self.config.lifecycle_owner,
                draft=draft,
                proposal_receipt=proposal_receipt,
                admission_receipt=parsed_admission,
                refinement_verification=refinement_verification,
                required_authoritative_receipt_ids=required_authoritative_receipt_ids,
                authoritative_receipts=authoritative_receipts,
                proposal_bindings=proposal_bindings,
                new_assumption_ids=new_assumption_ids,
                unsupported_semantics=unsupported_semantics,
                hard_policy_gates=hard_policy_gates,
            )
        )

        objective_after = _file_digest(objective)
        completion_after = _file_digest(completion)
        generation_after = _file_digest(generation)
        attempts = []
        for item in candidates:
            result = item.result
            attempts.append(
                {
                    "attempt_id": (
                        f"{result.result_id}:candidate:{item.index}"
                    ),
                    "stage": "model_draft",
                    "status": (
                        "succeeded" if item.schema_accepted else "failed"
                    ),
                    "provider_id": result.provider_id,
                    "repository_tree_id": call.request.repository_tree_id,
                    "goal_cid": call.request.root_goal_content_id,
                    "resource_class": "accelerator-model-draft",
                    "availability_checked": True,
                    "availability_success": (
                        result.fallback_reason
                        is not GoalDevelopmentFallbackReason.UNAVAILABLE
                    ),
                    "schema_validated": True,
                    "schema_accepted": item.schema_accepted,
                    "deterministic_fallback": result.used_fallback,
                    "input_token_count": 0,
                    "output_token_count": (
                        0 if result.draft is None else result.draft.token_count
                    ),
                }
            )
        metrics: ProofMetricsSnapshot = build_proof_metrics_snapshot(
            attempts=attempts,
            identity={
                "goal_cid": call.request.root_goal_content_id,
                "repository_tree_id": call.request.repository_tree_id,
                "provider_id": "leanstral-goal-development",
                "resource_class": "accelerator-model-draft",
            },
        )
        write_proof_metrics_snapshot(
            self.config.metrics_path,
            metrics,
            queryable=self.config.queryable_metrics,
        )

        material: dict[str, Any] = {
            "schema": LEANSTRAL_GOAL_LIFECYCLE_RUN_SCHEMA,
            "version": LEANSTRAL_GOAL_LIFECYCLE_VERSION,
            "mode": self.config.mode.value,
            "request_id": call.request_id,
            "root_goal_id": call.request.root_goal_id,
            "root_goal_content_id": call.request.root_goal_content_id,
            "repository_tree_id": call.request.repository_tree_id,
            "policy_digest": call.request.policy_digest,
            "started_at": started_at,
            "finished_at": _utc_now(),
            "candidate_count": len(candidates),
            "candidates": [item.to_dict() for item in candidates],
            "selected_draft_id": "" if draft is None else draft.draft_id,
            "proposal_receipt": (
                None if proposal_receipt is None else proposal_receipt.to_dict()
            ),
            "admission_receipt": (
                None
                if parsed_admission is None
                else parsed_admission.to_dict()
            ),
            "objective_admission": admission.to_dict(),
            "objective_heap_before": objective_before,
            "objective_heap_after": objective_after,
            "objective_heap_unchanged": objective_before == objective_after,
            "completion_state_before": completion_before,
            "completion_state_after": completion_after,
            "completion_state_unchanged": completion_before == completion_after,
            "generation_state_before": generation_before,
            "generation_state_after": generation_after,
            "generation_state_unchanged": generation_before == generation_after,
            "implementation_conformance": _json_copy(
                dict(implementation_conformance or {})
            ),
            "completion_decision": _json_copy(dict(completion_decision or {})),
            "metrics_path": str(self.config.metrics_path),
            "metrics_snapshot_id": metrics.snapshot_id,
            "authoritative": False,
            "completion_authority": False,
        }
        material["run_id"] = _content_id(material, prefix="leanstral-goal-run-")
        run = LeanstralGoalLifecycleRun(material)
        audit = {
            "schema": LEANSTRAL_GOAL_LIFECYCLE_AUDIT_SCHEMA,
            "run": run.to_dict(),
        }
        _append_jsonl(self.config.audit_path, audit)
        _atomic_json(self.config.state_path, run.to_dict())
        return run

    def recover(self) -> LeanstralGoalLifecycleRun | None:
        """Recover the latest valid state, falling back to the durable journal."""

        try:
            payload = json.loads(self.config.state_path.read_text(encoding="utf-8"))
            if isinstance(payload, Mapping):
                return LeanstralGoalLifecycleRun.from_dict(payload)
        except (OSError, ValueError, ContractValidationError):
            pass
        try:
            lines = self.config.audit_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return None
        for line in reversed(lines):
            try:
                envelope = json.loads(line)
                if (
                    isinstance(envelope, Mapping)
                    and envelope.get("schema")
                    == LEANSTRAL_GOAL_LIFECYCLE_AUDIT_SCHEMA
                    and isinstance(envelope.get("run"), Mapping)
                ):
                    return LeanstralGoalLifecycleRun.from_dict(envelope["run"])
            except (ValueError, ContractValidationError):
                continue
        return None


def build_configured_leanstral_goal_lifecycle_supervisor(
    *,
    state_dir: Path | str,
    mode: GoalDevelopmentMode | str = GoalDevelopmentMode.SHADOW,
    providers: Sequence[LeanstralGoalDevelopmentProvider | Any] = (),
    **config: Any,
) -> ConfiguredLeanstralGoalLifecycleSupervisor:
    """Build the explicit supervisor path; omitted mode is always shadow."""

    return ConfiguredLeanstralGoalLifecycleSupervisor(
        LeanstralGoalLifecycleConfig(
            state_dir=state_dir,
            mode=mode,
            **config,
        ),
        providers=providers,
    )


__all__ = [
    "ConfiguredLeanstralGoalLifecycleSupervisor",
    "DEFAULT_LEANSTRAL_GOAL_LIFECYCLE_AUDIT_FILE",
    "DEFAULT_LEANSTRAL_GOAL_LIFECYCLE_GENERATION_FILE",
    "DEFAULT_LEANSTRAL_GOAL_LIFECYCLE_MAX_CANDIDATES",
    "DEFAULT_LEANSTRAL_GOAL_LIFECYCLE_METRICS_FILE",
    "DEFAULT_LEANSTRAL_GOAL_LIFECYCLE_STATE_FILE",
    "LEANSTRAL_GOAL_LIFECYCLE_AUDIT_SCHEMA",
    "LEANSTRAL_GOAL_LIFECYCLE_RUN_SCHEMA",
    "LEANSTRAL_GOAL_LIFECYCLE_VERSION",
    "LeanstralGoalLifecycleConfig",
    "LeanstralGoalLifecycleRun",
    "MAX_LEANSTRAL_GOAL_LIFECYCLE_CANDIDATES",
    "build_configured_leanstral_goal_lifecycle_supervisor",
]
