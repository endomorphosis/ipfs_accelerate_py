"""Risk-weighted campaign planning composition (AAE-040).

Interface surface:

* ``plan_mutation_campaign`` — establish unmutated baseline requirements,
  budget risk-weighted targets under a resource envelope, generate bounded
  candidates, predict explained detection sets, and seal a deterministic
  ``MutationCampaignPlan@1`` (optional held-out partition provenance).
* ``generate_mutation_candidates`` — compose the canonical datasets
  ``generate_mutation_candidates@1`` authority.
* ``predict_detection_set`` — compose the canonical datasets
  ``predict_detection_set@1`` authority, projecting ``AssuranceManifest@1``
  into the detection slice when needed.

This module does **not** execute mutants, create worktrees, open a store, or
change production policy. Missing observation inputs and identity disagreements
fail closed.

Cold import is side-effect free.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import blake2b
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ipfs_accelerate_py.agent_supervisor.adversarial_assurance.manifest import (
    ASSURANCE_MANIFEST_INTERFACE,
    AssuranceManifest,
    AssuranceManifestError,
    RepositoryStateBinding,
)
from ipfs_datasets_py.logic.software_contracts.content import (
    cid_for_structured,
    validate_cid,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.common import (
    ArtifactProvenance,
    AssuranceArtifactHeader,
    AssuranceBaseError,
    AssuranceTerminalStatus,
    AuthoritySource,
    ExecutionMode,
    GeneratorIdentity,
    VersionBinding,
    reject_private_and_model_authority,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.detection import (
    DetectionAssuranceManifest,
    DetectionPredictionError,
    PREDICT_DETECTION_SET_INTERFACE as DATASETS_PREDICT_DETECTION_SET_INTERFACE,
    assert_prediction_explained,
    predict_detection_set as datasets_predict_detection_set,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.execution_contracts import (
    ExpectedDetectionSet,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.generator import (
    GENERATE_MUTATION_CANDIDATES_INTERFACE as DATASETS_GENERATE_MUTATION_CANDIDATES_INTERFACE,
    MutationGenerationError,
    MutationGenerationManifest,
    MutationGenerationResult,
    generate_mutation_candidates as datasets_generate_mutation_candidates,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.held_out import (
    HeldOutPolicyError,
    MutantPartitionPlan,
    partition_mutants,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.mutation_contracts import (
    CampaignBudget,
    MUTATION_CAMPAIGN_PLAN_INTERFACE,
    MUTATION_CAMPAIGN_PLAN_SCHEMA,
    MutationCampaignPlan,
    MutationCampaignPolicy,
    MutationCandidate,
    MutationContractError,
    MutationOperatorDefinition,
    MutationRiskClass,
    MutationTarget,
    SeedConfigBinding,
    assert_budget_admits_counts,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.risk import (
    DEFAULT_ALWAYS_SELECT_MIN_RISK_BP,
    DEFAULT_LOW_RISK_SAMPLE_RATE_BP,
    RiskCandidate,
    RiskScore,
    SamplingBudget,
    TargetRiskError,
    rank_mutation_risk,
    selected_risk_scores,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.targets import (
    AssertedProperty,
    ClaimRecord,
    TargetSelectionError,
    select_mutation_targets,
)

# ---------------------------------------------------------------------------
# Schema / interface constants
# ---------------------------------------------------------------------------

PLAN_MUTATION_CAMPAIGN_INTERFACE: Final[str] = "plan_mutation_campaign@1"
GENERATE_MUTATION_CANDIDATES_INTERFACE: Final[str] = (
    DATASETS_GENERATE_MUTATION_CANDIDATES_INTERFACE
)
PREDICT_DETECTION_SET_INTERFACE: Final[str] = (
    DATASETS_PREDICT_DETECTION_SET_INTERFACE
)

BASELINE_REQUIREMENTS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-baseline-requirements@1"
)
CAMPAIGN_RESOURCE_BUDGET_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-campaign-resource-budget@1"
)
CAMPAIGN_PLAN_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-campaign-plan-result@1"
)
CAMPAIGN_PLAN_RESULT_INTERFACE: Final[str] = "MutationCampaignPlanResult@1"

GENERATOR_ID: Final[str] = "campaign_planning"
GENERATOR_VERSION: Final[str] = "1.0.0"

MAX_TEXT_CHARS: Final[int] = 16_384
MAX_DIAGNOSTIC: Final[int] = 1_024
MAX_TARGETS: Final[int] = 1_024
MAX_OPERATORS: Final[int] = 256
MAX_CANDIDATES: Final[int] = 4_096
MAX_SEED: Final[int] = 2**63 - 1
MAX_EXECUTION_SECONDS: Final[int] = 7 * 24 * 3_600
MAX_WORKTREES: Final[int] = 256
MAX_RISK_WEIGHT_BP: Final[int] = 10_000

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_.:/+-]{0,127}$")
_REPOSITORY_ID_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.:/+-]{0,255}$"
)

# Priority order for risk-weighted budgeting (matches datasets risk ranking).
_RISK_CLASS_PRIORITY: Final[tuple[str, ...]] = (
    MutationRiskClass.CRITICAL_SECURITY.value,
    MutationRiskClass.AUTHORIZATION.value,
    MutationRiskClass.FINANCIAL_LEGAL.value,
    MutationRiskClass.DURABILITY.value,
    MutationRiskClass.DISTRIBUTED_TRANSITION.value,
    MutationRiskClass.PROOF_RECEIPT_TRUST.value,
    MutationRiskClass.CRITICAL_INVARIANT.value,
    MutationRiskClass.HIGH.value,
    MutationRiskClass.MEDIUM.value,
    MutationRiskClass.LOCAL_BUG.value,
    MutationRiskClass.LOW.value,
)
_RISK_RANK: Final[Mapping[str, int]] = MappingProxyType(
    {name: index for index, name in enumerate(_RISK_CLASS_PRIORITY)}
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class CampaignPlanningError(AssuranceBaseError):
    """Raised when campaign planning inputs fail closed verification."""

    def __init__(self, message: str, *, reason_code: str = "malformed_input") -> None:
        super().__init__(message)
        self.reason_code = str(reason_code or "malformed_input")


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _text(value: Any, name: str, *, empty: bool = False) -> str:
    if type(value) is not str:
        raise CampaignPlanningError(f"{name} must be a string")
    text = unicodedata.normalize("NFC", value)
    if text != value or (not empty and text != text.strip()):
        raise CampaignPlanningError(f"{name} must be trimmed NFC text")
    if not empty and not text.strip():
        raise CampaignPlanningError(f"{name} must not be empty")
    if len(text) > MAX_TEXT_CHARS or any(not char.isprintable() for char in text):
        raise CampaignPlanningError(f"{name} contains invalid text")
    return text


def _optional_text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    text = _text(value, name, empty=True)
    return text or None


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise CampaignPlanningError(f"{name} must be a boolean")
    return value


def _cid(value: Any, name: str) -> str:
    try:
        return validate_cid(value)
    except Exception as exc:
        raise CampaignPlanningError(f"{name} must be a valid CID") from exc


def _token(value: Any, name: str) -> str:
    text = _text(value, name)
    if _TOKEN_RE.fullmatch(text) is None:
        raise CampaignPlanningError(f"{name} must match token pattern")
    return text


def _repository_id(value: Any, name: str = "repository_id") -> str:
    text = _text(value, name)
    if _REPOSITORY_ID_RE.fullmatch(text) is None:
        raise CampaignPlanningError(f"{name} must be a repository identity")
    return text


def _pos_int(value: Any, name: str, *, maximum: int) -> int:
    if type(value) is not int or isinstance(value, bool) or value < 1:
        raise CampaignPlanningError(f"{name} must be a positive integer")
    if value > maximum:
        raise CampaignPlanningError(f"{name} exceeds maximum")
    return value


def _nonneg_int(value: Any, name: str, *, maximum: int) -> int:
    if type(value) is not int or isinstance(value, bool) or value < 0:
        raise CampaignPlanningError(f"{name} must be a nonnegative integer")
    if value > maximum:
        raise CampaignPlanningError(f"{name} exceeds maximum")
    return value


def _basis_points(value: Any, name: str) -> int:
    return _nonneg_int(value, name, maximum=MAX_RISK_WEIGHT_BP)


def _freeze_structured(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_structured(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_structured(item) for item in value)
    return value


def _thaw_structured(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_structured(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_structured(item) for item in value]
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise CampaignPlanningError(f"{name} must be a mapping")
    raw = dict(value)
    try:
        reject_private_and_model_authority(raw, path=name)
    except AssuranceBaseError as exc:
        raise CampaignPlanningError(str(exc)) from exc
    return MappingProxyType(_freeze_structured(raw))


def _closed(data: Mapping[str, Any], fields: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(data, Mapping):
        raise CampaignPlanningError(f"{name} must be a mapping")
    unknown = set(data) - fields
    if unknown:
        raise CampaignPlanningError(
            f"{name} contains unknown fields: {', '.join(sorted(unknown))}"
        )
    return dict(data)


def _clip(text: str, *, limit: int = MAX_DIAGNOSTIC) -> str:
    raw = str(text or "")
    if len(raw) <= limit:
        return raw
    return raw[: max(0, limit - 3)] + "..."


# ---------------------------------------------------------------------------
# Baseline requirements
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BaselineRequirements:
    """Unmutated green baseline bound into campaign planning.

    Planning refuses to proceed without an explicit baseline receipt identity
    and affirmative unmutated/green observation flags.
    """

    baseline_receipt_cid: str
    repository_id: str
    repository_state_cid: str
    unmutated: bool = True
    verification_green: bool = True
    observation_complete: bool = True
    notes: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "baseline_receipt_cid",
            "repository_id",
            "repository_state_cid",
            "unmutated",
            "verification_green",
            "observation_complete",
            "notes",
            "metadata",
            "baseline_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "baseline_receipt_cid",
            _cid(self.baseline_receipt_cid, "baseline_receipt_cid"),
        )
        object.__setattr__(
            self, "repository_id", _repository_id(self.repository_id)
        )
        object.__setattr__(
            self,
            "repository_state_cid",
            _cid(self.repository_state_cid, "repository_state_cid"),
        )
        object.__setattr__(self, "unmutated", _bool(self.unmutated, "unmutated"))
        object.__setattr__(
            self,
            "verification_green",
            _bool(self.verification_green, "verification_green"),
        )
        object.__setattr__(
            self,
            "observation_complete",
            _bool(self.observation_complete, "observation_complete"),
        )
        if not self.unmutated:
            raise CampaignPlanningError(
                "baseline must be unmutated",
                reason_code="baseline_mutated",
            )
        if not self.verification_green:
            raise CampaignPlanningError(
                "baseline verification must be green",
                reason_code="baseline_not_green",
            )
        if not self.observation_complete:
            raise CampaignPlanningError(
                "baseline observation must be complete",
                reason_code="baseline_incomplete",
            )
        object.__setattr__(self, "notes", _optional_text(self.notes, "notes"))
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": BASELINE_REQUIREMENTS_SCHEMA,
            "baseline_receipt_cid": self.baseline_receipt_cid,
            "repository_id": self.repository_id,
            "repository_state_cid": self.repository_state_cid,
            "unmutated": True,
            "verification_green": True,
            "observation_complete": True,
            "notes": self.notes,
            "metadata": _thaw_structured(self.metadata),
        }

    @property
    def baseline_cid(self) -> str:
        return cid_for_structured(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["baseline_cid"] = self.baseline_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BaselineRequirements":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("baseline_cid", None)
        payload.pop("schema", None)
        result = cls(
            baseline_receipt_cid=payload["baseline_receipt_cid"],
            repository_id=payload["repository_id"],
            repository_state_cid=payload["repository_state_cid"],
            unmutated=payload.get("unmutated", True),
            verification_green=payload.get("verification_green", True),
            observation_complete=payload.get("observation_complete", True),
            notes=payload.get("notes"),
            metadata=payload.get("metadata") or {},
        )
        if claimed is not None and claimed != result.baseline_cid:
            raise CampaignPlanningError(
                "BaselineRequirements baseline_cid identity mismatch"
            )
        return result

    @classmethod
    def normalize(
        cls,
        value: "BaselineRequirements | Mapping[str, Any] | str | None",
        *,
        repository_id: str,
        repository_state_cid: str,
    ) -> "BaselineRequirements":
        if value is None:
            raise CampaignPlanningError(
                "baseline requirements are required for campaign planning",
                reason_code="baseline_missing",
            )
        if isinstance(value, BaselineRequirements):
            sealed = value
        elif isinstance(value, str):
            sealed = cls(
                baseline_receipt_cid=value,
                repository_id=repository_id,
                repository_state_cid=repository_state_cid,
            )
        elif isinstance(value, Mapping):
            if "schema" in value or "baseline_cid" in value:
                sealed = cls.from_dict(value)
            else:
                receipt = value.get("baseline_receipt_cid") or value.get(
                    "receipt_cid"
                )
                if receipt is None:
                    raise CampaignPlanningError(
                        "baseline.baseline_receipt_cid is required",
                        reason_code="baseline_missing",
                    )
                sealed = cls(
                    baseline_receipt_cid=receipt,
                    repository_id=value.get("repository_id", repository_id),
                    repository_state_cid=value.get(
                        "repository_state_cid", repository_state_cid
                    ),
                    unmutated=value.get("unmutated", True),
                    verification_green=value.get("verification_green", True),
                    observation_complete=value.get("observation_complete", True),
                    notes=value.get("notes"),
                    metadata=value.get("metadata") or {},
                )
        else:
            raise CampaignPlanningError(
                "baseline must be BaselineRequirements, mapping, or receipt CID"
            )
        if sealed.repository_id != repository_id:
            raise CampaignPlanningError(
                "baseline repository_id must match repository_state",
                reason_code="identity_mismatch",
            )
        if sealed.repository_state_cid != repository_state_cid:
            raise CampaignPlanningError(
                "baseline repository_state_cid must match repository_state",
                reason_code="identity_mismatch",
            )
        return sealed


# ---------------------------------------------------------------------------
# Resource budget
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CampaignResourceBudget:
    """Resource envelope for risk-weighted campaign planning.

    Hard caps intersect with ``MutationCampaignPolicy.budget``. Sampling fields
    control risk-weighted target admission under the envelope.
    """

    max_total_candidates: int
    max_candidates_per_target: int
    max_candidates_per_operator: int
    max_targets: int
    max_operators: int
    max_execution_seconds: int
    max_worktrees: int
    always_select_min_risk_bp: int = DEFAULT_ALWAYS_SELECT_MIN_RISK_BP
    low_risk_sample_rate_bp: int = DEFAULT_LOW_RISK_SAMPLE_RATE_BP
    sampling_seed: int = 0
    notes: str | None = None

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "max_total_candidates",
            "max_candidates_per_target",
            "max_candidates_per_operator",
            "max_targets",
            "max_operators",
            "max_execution_seconds",
            "max_worktrees",
            "always_select_min_risk_bp",
            "low_risk_sample_rate_bp",
            "sampling_seed",
            "notes",
            "budget_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_total_candidates",
            _pos_int(
                self.max_total_candidates,
                "max_total_candidates",
                maximum=MAX_CANDIDATES,
            ),
        )
        object.__setattr__(
            self,
            "max_candidates_per_target",
            _pos_int(
                self.max_candidates_per_target,
                "max_candidates_per_target",
                maximum=MAX_CANDIDATES,
            ),
        )
        object.__setattr__(
            self,
            "max_candidates_per_operator",
            _pos_int(
                self.max_candidates_per_operator,
                "max_candidates_per_operator",
                maximum=MAX_CANDIDATES,
            ),
        )
        object.__setattr__(
            self,
            "max_targets",
            _pos_int(self.max_targets, "max_targets", maximum=MAX_TARGETS),
        )
        object.__setattr__(
            self,
            "max_operators",
            _pos_int(self.max_operators, "max_operators", maximum=MAX_OPERATORS),
        )
        object.__setattr__(
            self,
            "max_execution_seconds",
            _pos_int(
                self.max_execution_seconds,
                "max_execution_seconds",
                maximum=MAX_EXECUTION_SECONDS,
            ),
        )
        object.__setattr__(
            self,
            "max_worktrees",
            _pos_int(self.max_worktrees, "max_worktrees", maximum=MAX_WORKTREES),
        )
        if self.max_candidates_per_target > self.max_total_candidates:
            raise CampaignPlanningError(
                "max_candidates_per_target cannot exceed max_total_candidates"
            )
        if self.max_candidates_per_operator > self.max_total_candidates:
            raise CampaignPlanningError(
                "max_candidates_per_operator cannot exceed max_total_candidates"
            )
        object.__setattr__(
            self,
            "always_select_min_risk_bp",
            _basis_points(
                self.always_select_min_risk_bp, "always_select_min_risk_bp"
            ),
        )
        object.__setattr__(
            self,
            "low_risk_sample_rate_bp",
            _basis_points(
                self.low_risk_sample_rate_bp, "low_risk_sample_rate_bp"
            ),
        )
        object.__setattr__(
            self,
            "sampling_seed",
            _nonneg_int(self.sampling_seed, "sampling_seed", maximum=MAX_SEED),
        )
        object.__setattr__(self, "notes", _optional_text(self.notes, "notes"))

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": CAMPAIGN_RESOURCE_BUDGET_SCHEMA,
            "max_total_candidates": self.max_total_candidates,
            "max_candidates_per_target": self.max_candidates_per_target,
            "max_candidates_per_operator": self.max_candidates_per_operator,
            "max_targets": self.max_targets,
            "max_operators": self.max_operators,
            "max_execution_seconds": self.max_execution_seconds,
            "max_worktrees": self.max_worktrees,
            "always_select_min_risk_bp": self.always_select_min_risk_bp,
            "low_risk_sample_rate_bp": self.low_risk_sample_rate_bp,
            "sampling_seed": self.sampling_seed,
            "notes": self.notes,
        }

    @property
    def budget_cid(self) -> str:
        return cid_for_structured(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["budget_cid"] = self.budget_cid
        return payload

    def as_campaign_budget(self) -> CampaignBudget:
        return CampaignBudget(
            max_total_candidates=self.max_total_candidates,
            max_candidates_per_target=self.max_candidates_per_target,
            max_candidates_per_operator=self.max_candidates_per_operator,
            max_targets=self.max_targets,
            max_operators=self.max_operators,
            max_execution_seconds=self.max_execution_seconds,
            max_worktrees=self.max_worktrees,
        )

    def as_sampling_budget(self) -> SamplingBudget:
        return SamplingBudget(
            max_targets=self.max_targets,
            always_select_min_risk_bp=self.always_select_min_risk_bp,
            low_risk_sample_rate_bp=self.low_risk_sample_rate_bp,
            seed=self.sampling_seed,
            notes=self.notes,
        )

    def intersect(self, policy_budget: CampaignBudget) -> "CampaignResourceBudget":
        """Return the pointwise minimum envelope with a sealed policy budget."""

        return CampaignResourceBudget(
            max_total_candidates=min(
                self.max_total_candidates, policy_budget.max_total_candidates
            ),
            max_candidates_per_target=min(
                self.max_candidates_per_target,
                policy_budget.max_candidates_per_target,
            ),
            max_candidates_per_operator=min(
                self.max_candidates_per_operator,
                policy_budget.max_candidates_per_operator,
            ),
            max_targets=min(self.max_targets, policy_budget.max_targets),
            max_operators=min(self.max_operators, policy_budget.max_operators),
            max_execution_seconds=min(
                self.max_execution_seconds, policy_budget.max_execution_seconds
            ),
            max_worktrees=min(self.max_worktrees, policy_budget.max_worktrees),
            always_select_min_risk_bp=self.always_select_min_risk_bp,
            low_risk_sample_rate_bp=self.low_risk_sample_rate_bp,
            sampling_seed=self.sampling_seed,
            notes=self.notes,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CampaignResourceBudget":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("budget_cid", None)
        payload.pop("schema", None)
        result = cls(
            max_total_candidates=payload["max_total_candidates"],
            max_candidates_per_target=payload["max_candidates_per_target"],
            max_candidates_per_operator=payload["max_candidates_per_operator"],
            max_targets=payload["max_targets"],
            max_operators=payload["max_operators"],
            max_execution_seconds=payload["max_execution_seconds"],
            max_worktrees=payload["max_worktrees"],
            always_select_min_risk_bp=payload.get(
                "always_select_min_risk_bp", DEFAULT_ALWAYS_SELECT_MIN_RISK_BP
            ),
            low_risk_sample_rate_bp=payload.get(
                "low_risk_sample_rate_bp", DEFAULT_LOW_RISK_SAMPLE_RATE_BP
            ),
            sampling_seed=payload.get("sampling_seed", 0),
            notes=payload.get("notes"),
        )
        if claimed is not None and claimed != result.budget_cid:
            raise CampaignPlanningError(
                "CampaignResourceBudget budget_cid identity mismatch"
            )
        return result

    @classmethod
    def normalize(
        cls, value: "CampaignResourceBudget | CampaignBudget | Mapping[str, Any]"
    ) -> "CampaignResourceBudget":
        if isinstance(value, CampaignResourceBudget):
            return value
        if isinstance(value, CampaignBudget):
            return cls(
                max_total_candidates=value.max_total_candidates,
                max_candidates_per_target=value.max_candidates_per_target,
                max_candidates_per_operator=value.max_candidates_per_operator,
                max_targets=value.max_targets,
                max_operators=value.max_operators,
                max_execution_seconds=value.max_execution_seconds,
                max_worktrees=value.max_worktrees,
            )
        if not isinstance(value, Mapping):
            raise CampaignPlanningError(
                "resource_budget must be CampaignResourceBudget, "
                "CampaignBudget, or mapping"
            )
        if "schema" in value or "budget_cid" in value:
            # CampaignBudget schema is datasets-owned; accept either.
            schema = value.get("schema")
            if schema == CAMPAIGN_RESOURCE_BUDGET_SCHEMA or (
                "always_select_min_risk_bp" in value
                or "sampling_seed" in value
                or "low_risk_sample_rate_bp" in value
            ):
                return cls.from_dict(value)
            try:
                sealed = CampaignBudget.from_dict(value)  # type: ignore[arg-type]
            except MutationContractError as exc:
                # Fall through to open construction.
                try:
                    return cls.from_dict(value)
                except CampaignPlanningError:
                    raise CampaignPlanningError(str(exc)) from exc
            return cls.normalize(sealed)
        required = {
            "max_total_candidates",
            "max_candidates_per_target",
            "max_candidates_per_operator",
            "max_targets",
            "max_operators",
            "max_execution_seconds",
            "max_worktrees",
        }
        missing = required - set(value)
        if missing:
            raise CampaignPlanningError(
                "resource_budget missing required fields: "
                f"{', '.join(sorted(missing))}"
            )
        allowed = required | {
            "always_select_min_risk_bp",
            "low_risk_sample_rate_bp",
            "sampling_seed",
            "notes",
        }
        unknown = set(value) - allowed
        if unknown:
            raise CampaignPlanningError(
                "resource_budget contains unknown fields: "
                f"{', '.join(sorted(unknown))}"
            )
        return cls(
            max_total_candidates=value["max_total_candidates"],
            max_candidates_per_target=value["max_candidates_per_target"],
            max_candidates_per_operator=value["max_candidates_per_operator"],
            max_targets=value["max_targets"],
            max_operators=value["max_operators"],
            max_execution_seconds=value["max_execution_seconds"],
            max_worktrees=value["max_worktrees"],
            always_select_min_risk_bp=value.get(
                "always_select_min_risk_bp", DEFAULT_ALWAYS_SELECT_MIN_RISK_BP
            ),
            low_risk_sample_rate_bp=value.get(
                "low_risk_sample_rate_bp", DEFAULT_LOW_RISK_SAMPLE_RATE_BP
            ),
            sampling_seed=value.get("sampling_seed", 0),
            notes=value.get("notes"),
        )


# ---------------------------------------------------------------------------
# Plan result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MutationCampaignPlanResult:
    """Sealed planning result binding plan, candidates, detections, partition."""

    plan: MutationCampaignPlan
    baseline: BaselineRequirements
    resource_budget: CampaignResourceBudget
    selected_targets: tuple[MutationTarget, ...]
    selected_operators: tuple[MutationOperatorDefinition, ...]
    candidates: tuple[MutationCandidate, ...]
    expected_detections: tuple[ExpectedDetectionSet, ...]
    risk_ranking: tuple[RiskScore, ...]
    generation_result: MutationGenerationResult | None = None
    partition: MutantPartitionPlan | None = None
    assurance_manifest_cid: str | None = None
    notes: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    production_policy_changed: bool = False

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface_id",
            "plan",
            "baseline",
            "resource_budget",
            "selected_targets",
            "selected_operators",
            "candidates",
            "expected_detections",
            "risk_ranking",
            "generation_result",
            "partition",
            "assurance_manifest_cid",
            "notes",
            "metadata",
            "production_policy_changed",
            "result_cid",
        }
    )

    def __post_init__(self) -> None:
        if not isinstance(self.plan, MutationCampaignPlan):
            raise CampaignPlanningError("plan must be MutationCampaignPlan")
        if not isinstance(self.baseline, BaselineRequirements):
            raise CampaignPlanningError("baseline must be BaselineRequirements")
        if not isinstance(self.resource_budget, CampaignResourceBudget):
            raise CampaignPlanningError(
                "resource_budget must be CampaignResourceBudget"
            )
        object.__setattr__(
            self,
            "selected_targets",
            tuple(self.selected_targets),
        )
        object.__setattr__(
            self,
            "selected_operators",
            tuple(self.selected_operators),
        )
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(
            self, "expected_detections", tuple(self.expected_detections)
        )
        object.__setattr__(self, "risk_ranking", tuple(self.risk_ranking))
        if len(self.candidates) != len(self.expected_detections):
            raise CampaignPlanningError(
                "expected_detections must align 1:1 with candidates"
            )
        for candidate, detection in zip(
            self.candidates, self.expected_detections, strict=True
        ):
            if detection.candidate_id != candidate.candidate_id:
                raise CampaignPlanningError(
                    "expected_detections candidate_id must match candidates order"
                )
            if detection.candidate_cid != candidate.candidate_cid:
                raise CampaignPlanningError(
                    "expected_detections candidate_cid must match candidates"
                )
        if self.assurance_manifest_cid is not None:
            object.__setattr__(
                self,
                "assurance_manifest_cid",
                _cid(self.assurance_manifest_cid, "assurance_manifest_cid"),
            )
        object.__setattr__(self, "notes", _optional_text(self.notes, "notes"))
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))
        object.__setattr__(
            self,
            "production_policy_changed",
            _bool(self.production_policy_changed, "production_policy_changed"),
        )
        if self.production_policy_changed:
            raise CampaignPlanningError(
                "plan_mutation_campaign must not change production policy"
            )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": CAMPAIGN_PLAN_RESULT_SCHEMA,
            "interface_id": CAMPAIGN_PLAN_RESULT_INTERFACE,
            "plan": self.plan.identity_payload(),
            "baseline": self.baseline.identity_payload(),
            "resource_budget": self.resource_budget.identity_payload(),
            "selected_targets": [
                item.identity_payload() for item in self.selected_targets
            ],
            "selected_operators": [
                item.identity_payload() for item in self.selected_operators
            ],
            "candidates": [item.identity_payload() for item in self.candidates],
            "expected_detections": [
                item.identity_payload() for item in self.expected_detections
            ],
            "risk_ranking": [item.to_dict() for item in self.risk_ranking],
            "generation_result": (
                self.generation_result.to_dict()
                if self.generation_result is not None
                else None
            ),
            "partition": (
                self.partition.identity_payload()
                if self.partition is not None
                else None
            ),
            "assurance_manifest_cid": self.assurance_manifest_cid,
            "notes": self.notes,
            "metadata": _thaw_structured(self.metadata),
            "production_policy_changed": False,
        }

    @property
    def result_cid(self) -> str:
        return cid_for_structured(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["result_cid"] = self.result_cid
        # Expand sealed plan for callers that need plan_cid.
        payload["plan"] = self.plan.to_dict()
        payload["baseline"] = self.baseline.to_dict()
        payload["resource_budget"] = self.resource_budget.to_dict()
        return payload


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def _normalize_policy(
    value: MutationCampaignPolicy | Mapping[str, Any],
    name: str = "mutation_policy",
) -> MutationCampaignPolicy:
    if isinstance(value, MutationCampaignPolicy):
        return value
    if isinstance(value, Mapping):
        try:
            if "schema" in value or "policy_cid" in value:
                return MutationCampaignPolicy.from_dict(value)
            return MutationCampaignPolicy(
                header=value["header"],
                policy_id=value["policy_id"],
                policy_version=value["policy_version"],
                admitted_operator_classes=value["admitted_operator_classes"],
                admitted_risk_classes=value["admitted_risk_classes"],
                budget=value["budget"],
                seed_config=value["seed_config"],
                require_disposable_worktree=value.get(
                    "require_disposable_worktree", True
                ),
                require_network_disabled=value.get(
                    "require_network_disabled", True
                ),
                require_rollback=value.get("require_rollback", True),
                require_deterministic_seed=value.get(
                    "require_deterministic_seed", True
                ),
                full_suite_fallback_enabled=value.get(
                    "full_suite_fallback_enabled", True
                ),
                held_out_partition_required=value.get(
                    "held_out_partition_required", True
                ),
                operator_cids=value.get("operator_cids", ()),
                notes=value.get("notes"),
                metadata=value.get("metadata") or {},
            )
        except (MutationContractError, KeyError, TypeError) as exc:
            raise CampaignPlanningError(
                f"{name} is not a sealed MutationCampaignPolicy: {exc}"
            ) from exc
    raise CampaignPlanningError(f"{name} must be MutationCampaignPolicy or mapping")


def _normalize_assurance_manifest(
    value: AssuranceManifest | Mapping[str, Any],
    name: str = "assurance_manifest",
) -> AssuranceManifest:
    if isinstance(value, AssuranceManifest):
        return value
    if isinstance(value, Mapping):
        try:
            return AssuranceManifest.from_dict(value)
        except AssuranceManifestError as exc:
            raise CampaignPlanningError(f"{name}: {exc}") from exc
    raise CampaignPlanningError(f"{name} must be AssuranceManifest or mapping")


def _normalize_target(
    value: MutationTarget | Mapping[str, Any], name: str
) -> MutationTarget:
    if isinstance(value, MutationTarget):
        return value
    if isinstance(value, Mapping):
        try:
            if "schema" in value or "target_cid" in value:
                return MutationTarget.from_dict(value)
            return MutationTarget(
                target_id=value["target_id"],
                repository_id=value["repository_id"],
                repository_state_cid=value["repository_state_cid"],
                symbol_ids=value["symbol_ids"],
                artifact_cids=value["artifact_cids"],
                language=value["language"],
                artifact_type=value["artifact_type"],
                prerequisites=value.get("prerequisites", ()),
                risk_class=value["risk_class"],
                risk_weight_bp=value["risk_weight_bp"],
                capsule_cids=value.get("capsule_cids", ()),
                proof_unit_cids=value.get("proof_unit_cids", ()),
                source_path=value.get("source_path"),
                notes=value.get("notes"),
                metadata=value.get("metadata") or {},
            )
        except (MutationContractError, KeyError, TypeError) as exc:
            raise CampaignPlanningError(f"{name}: {exc}") from exc
    raise CampaignPlanningError(f"{name} must be MutationTarget or mapping")


def _normalize_operator(
    value: MutationOperatorDefinition | Mapping[str, Any], name: str
) -> MutationOperatorDefinition:
    if isinstance(value, MutationOperatorDefinition):
        return value
    if isinstance(value, Mapping):
        try:
            if "schema" in value or "operator_cid" in value:
                return MutationOperatorDefinition.from_dict(value)
            return MutationOperatorDefinition(**dict(value))  # type: ignore[arg-type]
        except (MutationContractError, TypeError, KeyError) as exc:
            raise CampaignPlanningError(f"{name}: {exc}") from exc
    raise CampaignPlanningError(
        f"{name} must be MutationOperatorDefinition or mapping"
    )


def _normalize_seed_config(
    value: SeedConfigBinding | Mapping[str, Any],
    name: str = "seed_config",
) -> SeedConfigBinding:
    if isinstance(value, SeedConfigBinding):
        return value
    if isinstance(value, Mapping):
        try:
            if "schema" in value or "binding_cid" in value or "config_cid" in value:
                return SeedConfigBinding.from_dict(value)
            return SeedConfigBinding(
                seed=value["seed"],
                config=value.get("config") or {},
                config_cid=value.get("config_cid"),
            )
        except (MutationContractError, KeyError, TypeError) as exc:
            raise CampaignPlanningError(f"{name}: {exc}") from exc
    raise CampaignPlanningError(f"{name} must be SeedConfigBinding or mapping")


def _risk_sort_key(target: MutationTarget) -> tuple[int, int, str]:
    return (
        _RISK_RANK.get(target.risk_class, len(_RISK_CLASS_PRIORITY)),
        -int(target.risk_weight_bp),
        target.target_id,
    )


def _budget_risk_weighted_targets(
    targets: Sequence[MutationTarget],
    *,
    policy: MutationCampaignPolicy,
    resource_budget: CampaignResourceBudget,
) -> tuple[tuple[MutationTarget, ...], tuple[RiskScore, ...]]:
    """Admit targets by policy risk class and risk-weighted sampling budget."""

    admitted = [target for target in targets if policy.admits_target(target)]
    if not admitted:
        raise CampaignPlanningError(
            "no targets admitted by mutation_policy.admitted_risk_classes",
            reason_code="no_admitted_targets",
        )

    risk_candidates = [
        RiskCandidate(
            subject_id=target.target_id,
            risk_class=target.risk_class,
            signals={
                "fan_out": 0,
                "recent_change_bp": 0,
                "uncertainty_bp": 0,
                "defect_count": 0,
                "frequency_bp": 0,
                "failure_cost_bp": min(
                    int(target.risk_weight_bp), MAX_RISK_WEIGHT_BP
                ),
                "missing_tests": False,
                "is_formatting": False,
                "is_generated_proven": False,
                "is_immutable_dependency": False,
                "is_boilerplate": False,
            },
            property_classes=(),
            metadata={
                "target_cid": target.target_cid,
                "risk_weight_bp": int(target.risk_weight_bp),
            },
        )
        for target in admitted
    ]
    try:
        ranking = rank_mutation_risk(
            risk_candidates,
            budget=resource_budget.as_sampling_budget(),
            apply_sampling=True,
        )
    except TargetRiskError as exc:
        raise CampaignPlanningError(str(exc)) from exc

    by_id = {target.target_id: target for target in admitted}
    selected = [by_id[score.subject_id] for score in selected_risk_scores(ranking)]

    # When every admitted target sits below the always-select threshold and
    # bounded sampling rejects the residual pool, keep a deterministic
    # risk-ordered floor so planning can still establish a non-empty budgeted
    # campaign from valid observation targets.
    if not selected:
        floor = sorted(admitted, key=_risk_sort_key)[
            : max(1, min(resource_budget.max_targets, len(admitted)))
        ]
        selected = list(floor)

    # Preserve risk priority then stable id among the sampled set.
    selected_sorted = tuple(sorted(selected, key=_risk_sort_key))
    if len(selected_sorted) > resource_budget.max_targets:
        selected_sorted = selected_sorted[: resource_budget.max_targets]
    return selected_sorted, ranking


def _select_operators(
    operators: Sequence[MutationOperatorDefinition],
    *,
    policy: MutationCampaignPolicy,
    resource_budget: CampaignResourceBudget,
) -> tuple[MutationOperatorDefinition, ...]:
    admitted = [
        operator
        for operator in operators
        if policy.admits_operator(operator)
        and (
            not policy.operator_cids
            or operator.operator_cid in set(policy.operator_cids)
        )
        and operator.required_sandbox.network_disabled
        and operator.required_sandbox.disposable_worktree_required
        and operator.rollback.preserves_production
    ]
    if not admitted:
        raise CampaignPlanningError(
            "no operators admitted by mutation_policy under resource budget",
            reason_code="no_admitted_operators",
        )
    # Stable order by (operator_id, operator_version).
    ordered = tuple(
        sorted(admitted, key=lambda item: (item.operator_id, item.operator_version))
    )
    if len(ordered) > resource_budget.max_operators:
        ordered = ordered[: resource_budget.max_operators]
    return ordered


def _stable_plan_id(
    *,
    policy_id: str,
    repository_state_cid: str,
    baseline_receipt_cid: str,
    seed: int,
    config_cid: str,
    target_cids: Sequence[str],
    operator_cids: Sequence[str],
) -> str:
    digest = blake2b(
        digest_size=16,
        person=b"aae-plan-id",
    )
    for part in (
        PLAN_MUTATION_CAMPAIGN_INTERFACE,
        policy_id,
        repository_state_cid,
        baseline_receipt_cid,
        str(seed),
        config_cid,
        *target_cids,
        *operator_cids,
    ):
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return f"plan_{digest.hexdigest()[:24]}"


def _plan_header(
    *,
    repository_id: str,
    repository_state_cid: str,
    policy: MutationCampaignPolicy,
    targets: Sequence[MutationTarget],
    environment_cid: str,
    dependency_lock_cid: str,
    baseline_receipt_cid: str,
    seed_config: SeedConfigBinding,
) -> AssuranceArtifactHeader:
    symbol_ids: list[str] = []
    artifact_cids: list[str] = []
    capsule_cids: list[str] = []
    proof_unit_cids: list[str] = []
    for target in targets:
        symbol_ids.extend(target.symbol_ids)
        artifact_cids.extend(target.artifact_cids)
        capsule_cids.extend(target.capsule_cids)
        proof_unit_cids.extend(target.proof_unit_cids)
    # Deduplicate while preserving stable sort (header requires sorted unique).
    symbol_ids = sorted(set(symbol_ids))
    artifact_cids = sorted(set(artifact_cids))
    capsule_cids = sorted(set(capsule_cids))
    proof_unit_cids = sorted(set(proof_unit_cids))
    if not symbol_ids:
        symbol_ids = ["campaign.plan"]
    if not artifact_cids:
        artifact_cids = [repository_state_cid]

    generator = GeneratorIdentity(
        generator_id=GENERATOR_ID,
        generator_version=GENERATOR_VERSION,
        interface_id=PLAN_MUTATION_CAMPAIGN_INTERFACE,
    )
    versions = VersionBinding(
        operator_id="campaign_planning",
        operator_version=GENERATOR_VERSION,
        campaign_policy_id=policy.policy_id,
        campaign_policy_version=policy.policy_version,
        generator=generator,
    )
    provenance = ArtifactProvenance(
        producer_id="adversarial_assurance",
        producer_version=GENERATOR_VERSION,
        execution_mode=ExecutionMode.LIVE,
        authority_source=AuthoritySource.DETERMINISTIC,
        input_cids=(
            repository_state_cid,
            policy.policy_cid,
            baseline_receipt_cid,
            seed_config.config_cid or seed_config.binding_cid,
        ),
        tool_ids=("campaign_planning.v1",),
        policy_cid=policy.policy_cid,
        notes=None,
    )
    return AssuranceArtifactHeader(
        artifact_kind="mutation_campaign_plan",
        repository_id=repository_id,
        repository_state_cid=repository_state_cid,
        target_symbol_ids=tuple(symbol_ids),
        target_artifact_cids=tuple(artifact_cids),
        capsule_cids=tuple(capsule_cids) if capsule_cids else (repository_state_cid,),
        proof_unit_cids=(
            tuple(proof_unit_cids) if proof_unit_cids else (repository_state_cid,)
        ),
        environment_cid=environment_cid,
        dependency_lock_cid=dependency_lock_cid,
        versions=versions,
        provenance=provenance,
        terminal_status=AssuranceTerminalStatus.COMPLETE,
        receipt_cids=(baseline_receipt_cid,),
        proof_cids=(),
        metadata={
            "interface_id": PLAN_MUTATION_CAMPAIGN_INTERFACE,
            "generator_id": GENERATOR_ID,
            "generator_version": GENERATOR_VERSION,
        },
    )


def _build_partition(
    *,
    candidates: Sequence[MutationCandidate],
    plan: MutationCampaignPlan,
    policy: MutationCampaignPolicy,
    header: AssuranceArtifactHeader,
) -> MutantPartitionPlan | None:
    """Build a deterministic diagnosis/development/held-out partition plan.

    With fewer than two explained candidates a leakage-resistant held-out set
    cannot be formed (diagnosis is mandatory and must be disjoint from
    held-out); partitioning is deferred rather than inventing members.
    """

    if not policy.held_out_partition_required:
        return None
    if len(candidates) < 2:
        return None

    mutants = [
        {
            "mutant_id": candidate.candidate_id,
            "relatedness_key": candidate.target_id,
            # Mutation candidates are not remediation-generation inputs yet.
            "used_for_candidate_generation": False,
            "candidate_cid": candidate.candidate_cid,
            "operator_id": candidate.operator_id,
            "target_id": candidate.target_id,
            "notes": None,
        }
        for candidate in candidates
    ]
    # Highest-risk first for diagnosis reservation (stable on candidate_id).
    risk_order = sorted(
        candidates,
        key=lambda item: (
            _RISK_RANK.get(item.risk_class, len(_RISK_CLASS_PRIORITY)),
            item.candidate_id,
        ),
    )
    diagnosis_ids = (risk_order[0].candidate_id,)
    try:
        partition = partition_mutants(
            mutants,
            diagnosis_ids,
            header=header,
            campaign_id=policy.policy_id,
            partition_seed=plan.seed_config.seed,
            plan_id=f"partition_{plan.plan_id}",
            require_held_out=True,
            notes="aae-040 campaign planning partition",
            metadata={
                "plan_cid": plan.plan_cid,
                "policy_cid": plan.policy_cid,
            },
        )
    except HeldOutPolicyError:
        # Hash sampling on a small remainder can leave held-out empty even when
        # require_held_out=true. Retry with seed offsets, then force a minimal
        # diagnosis+held-out corpus with development_ratio_bp=0.
        remainder = [
            item.candidate_id
            for item in risk_order
            if item.candidate_id not in set(diagnosis_ids)
        ]
        if not remainder:
            return None
        held_out_ids = (remainder[-1],)
        try:
            for seed_offset in range(0, 64):
                try:
                    return partition_mutants(
                        mutants,
                        diagnosis_ids,
                        header=header,
                        campaign_id=policy.policy_id,
                        partition_seed=plan.seed_config.seed + seed_offset,
                        plan_id=f"partition_{plan.plan_id}",
                        require_held_out=True,
                        notes="aae-040 campaign planning partition",
                        metadata={
                            "plan_cid": plan.plan_cid,
                            "policy_cid": plan.policy_cid,
                            "partition_seed_offset": seed_offset,
                        },
                    )
                except HeldOutPolicyError:
                    continue
            minimal = [
                item
                for item in mutants
                if item["mutant_id"] in set(diagnosis_ids) | set(held_out_ids)
            ]
            return partition_mutants(
                minimal,
                diagnosis_ids,
                header=header,
                campaign_id=policy.policy_id,
                partition_seed=plan.seed_config.seed,
                plan_id=f"partition_{plan.plan_id}",
                development_ratio_bp=0,
                held_out_ratio_bp=10_000,
                require_held_out=True,
                notes="aae-040 campaign planning partition (forced held-out)",
                metadata={
                    "plan_cid": plan.plan_cid,
                    "policy_cid": plan.policy_cid,
                    "forced_held_out": True,
                },
            )
        except HeldOutPolicyError as exc:
            raise CampaignPlanningError(
                f"held-out partition failed: {exc}",
                reason_code="partition_failed",
            ) from exc
    return partition


# ---------------------------------------------------------------------------
# Public composition surfaces
# ---------------------------------------------------------------------------


def generate_mutation_candidates(
    manifest: MutationGenerationManifest | Mapping[str, Any],
    mutation_policy: MutationCampaignPolicy | Mapping[str, Any],
    *,
    seed_config: SeedConfigBinding | Mapping[str, Any] | None = None,
    notes: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    return_result: bool = False,
) -> tuple[MutationCandidate, ...] | MutationGenerationResult:
    """Compose canonical ``generate_mutation_candidates@1``.

    Interface: ``generate_mutation_candidates@1``

    Delegates to the datasets semantic mutation generator without altering
    production policy. Same source root, targets, operators, seed, and policy
    yield byte-identical ordered candidates.
    """

    try:
        return datasets_generate_mutation_candidates(
            manifest,
            mutation_policy,
            seed_config=seed_config,
            notes=notes,
            metadata=metadata,
            return_result=return_result,
        )
    except MutationGenerationError as exc:
        raise CampaignPlanningError(
            f"generate_mutation_candidates failed: {exc}",
            reason_code="generation_failed",
        ) from exc


def predict_detection_set(
    mutation: MutationCandidate | Mapping[str, Any],
    assurance_manifest: (
        AssuranceManifest
        | DetectionAssuranceManifest
        | Mapping[str, Any]
    ),
    *,
    notes: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ExpectedDetectionSet:
    """Compose canonical ``predict_detection_set@1``.

    Interface: ``predict_detection_set@1``

    Accepts ``AssuranceManifest@1`` (projects via
    :meth:`AssuranceManifest.as_detection_manifest`) or a sealed
    ``DetectionAssuranceManifest@1``. Every returned prediction is fully
    explained (claim, rationale, dependency path, strength, terminal status,
    detector identity/revision).
    """

    if isinstance(assurance_manifest, AssuranceManifest):
        detection_manifest: DetectionAssuranceManifest | Mapping[str, Any] = (
            assurance_manifest.as_detection_manifest()
        )
    else:
        detection_manifest = assurance_manifest

    try:
        detection_set = datasets_predict_detection_set(
            mutation,
            detection_manifest,
            notes=notes,
            metadata=metadata,
        )
    except DetectionPredictionError as exc:
        raise CampaignPlanningError(
            f"predict_detection_set failed: {exc}",
            reason_code="detection_failed",
        ) from exc

    for prediction in detection_set.predicted_detectors:
        try:
            assert_prediction_explained(prediction)
        except DetectionPredictionError as exc:
            raise CampaignPlanningError(
                f"detector prediction not fully explained: {exc}",
                reason_code="detection_unexplained",
            ) from exc
    return detection_set


def plan_mutation_campaign(
    repository_state: Any,
    assurance_manifest: AssuranceManifest | Mapping[str, Any],
    mutation_policy: MutationCampaignPolicy | Mapping[str, Any],
    resource_budget: CampaignResourceBudget | CampaignBudget | Mapping[str, Any],
    *,
    baseline: BaselineRequirements | Mapping[str, Any] | str | None = None,
    baseline_receipt_cid: str | None = None,
    targets: Sequence[MutationTarget | Mapping[str, Any]] | None = None,
    operators: Sequence[MutationOperatorDefinition | Mapping[str, Any]] | None = None,
    properties: Sequence[
        AssertedProperty | ClaimRecord | Mapping[str, Any]
    ]
    | None = None,
    generation_manifest: MutationGenerationManifest | Mapping[str, Any] | None = None,
    seed_config: SeedConfigBinding | Mapping[str, Any] | None = None,
    source_root_cid: str | None = None,
    environment_cid: str | None = None,
    dependency_lock_cid: str | None = None,
    plan_id: str | None = None,
    partition: bool = True,
    notes: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    return_result: bool = True,
) -> MutationCampaignPlan | MutationCampaignPlanResult:
    """Compose a risk-weighted ``MutationCampaignPlan@1``.

    Interface: ``plan_mutation_campaign@1``

    Pipeline (fail-closed, pure composition):

    1. Establish unmutated green baseline requirements.
    2. Intersect caller resource budget with policy campaign budget.
    3. Risk-weight and sample targets under the envelope.
    4. Admit operators under policy/sandbox/rollback constraints.
    5. Generate bounded candidates via canonical semantic generation.
    6. Predict explained detection sets for every candidate.
    7. Seal deterministic plan identity (and optional held-out partition).

    Parameters
    ----------
    repository_state:
        Mapping/view with ``repository_id`` and ``repository_state_cid``.
    assurance_manifest:
        Sealed ``AssuranceManifest@1`` (detection catalog/claims/edges).
    mutation_policy:
        Sealed ``MutationCampaignPolicy@1``.
    resource_budget:
        Hard resource envelope (intersected with policy budget).
    baseline / baseline_receipt_cid:
        Required unmutated green baseline receipt identity.
    targets / operators / properties / generation_manifest:
        Observation inputs for selection and generation. Fail closed when
        insufficient to produce a non-empty admitted plan.

    Does not execute mutants, create worktrees, or change production policy.
    """

    try:
        repo = RepositoryStateBinding.normalize(repository_state)
    except AssuranceManifestError as exc:
        raise CampaignPlanningError(f"repository_state: {exc}") from exc

    sealed_manifest = _normalize_assurance_manifest(assurance_manifest)
    sealed_policy = _normalize_policy(mutation_policy)
    sealed_resource = CampaignResourceBudget.normalize(resource_budget).intersect(
        sealed_policy.budget
    )

    if sealed_manifest.repository_id != repo.repository_id:
        raise CampaignPlanningError(
            "assurance_manifest repository_id must match repository_state",
            reason_code="identity_mismatch",
        )
    if sealed_manifest.repository_state_cid != repo.repository_state_cid:
        raise CampaignPlanningError(
            "assurance_manifest repository_state_cid must match repository_state",
            reason_code="identity_mismatch",
        )
    if sealed_policy.header.repository_id != repo.repository_id:
        raise CampaignPlanningError(
            "mutation_policy repository_id must match repository_state",
            reason_code="identity_mismatch",
        )
    if sealed_policy.header.repository_state_cid != repo.repository_state_cid:
        raise CampaignPlanningError(
            "mutation_policy repository_state_cid must match repository_state",
            reason_code="identity_mismatch",
        )
    if not sealed_manifest.observation_complete:
        raise CampaignPlanningError(
            "assurance_manifest observation must be complete",
            reason_code="observation_incomplete",
        )

    # Baseline requirements (required).
    baseline_input: BaselineRequirements | Mapping[str, Any] | str | None = baseline
    if baseline_input is None and baseline_receipt_cid is not None:
        baseline_input = baseline_receipt_cid
    if baseline_input is None:
        # Allow embedding under repository_state.metadata.baseline_receipt_cid.
        meta_receipt = None
        if isinstance(repo.metadata, Mapping):
            meta_receipt = repo.metadata.get("baseline_receipt_cid")
        if meta_receipt is not None:
            baseline_input = str(meta_receipt)
    sealed_baseline = BaselineRequirements.normalize(
        baseline_input,
        repository_id=repo.repository_id,
        repository_state_cid=repo.repository_state_cid,
    )

    active_seed = (
        sealed_policy.seed_config
        if seed_config is None
        else _normalize_seed_config(seed_config)
    )
    if sealed_policy.require_deterministic_seed and active_seed.seed < 0:
        raise CampaignPlanningError("seed must be nonnegative")

    # Resolve targets.
    sealed_targets: tuple[MutationTarget, ...]
    if generation_manifest is not None:
        try:
            gen_manifest = MutationGenerationManifest.normalize(generation_manifest)
        except MutationGenerationError as exc:
            raise CampaignPlanningError(f"generation_manifest: {exc}") from exc
        if gen_manifest.repository_id != repo.repository_id:
            raise CampaignPlanningError(
                "generation_manifest repository_id must match repository_state",
                reason_code="identity_mismatch",
            )
        if gen_manifest.repository_state_cid != repo.repository_state_cid:
            raise CampaignPlanningError(
                "generation_manifest repository_state_cid must match "
                "repository_state",
                reason_code="identity_mismatch",
            )
        sealed_targets = tuple(gen_manifest.targets)
        sealed_ops_from_manifest = tuple(gen_manifest.operators)
        resolved_source_root = gen_manifest.source_root_cid
        resolved_env = gen_manifest.environment_cid
        resolved_dep = gen_manifest.dependency_lock_cid
    else:
        sealed_ops_from_manifest = ()
        resolved_source_root = source_root_cid or repo.source_root_cid
        resolved_env = environment_cid or repo.environment_cid
        resolved_dep = dependency_lock_cid or repo.dependency_lock_cid
        if targets is not None:
            if not isinstance(targets, (list, tuple)):
                raise CampaignPlanningError("targets must be a sequence")
            if not targets:
                raise CampaignPlanningError("targets must not be empty")
            sealed_targets = tuple(
                _normalize_target(item, f"targets[{index}]")
                for index, item in enumerate(targets)
            )
        elif properties is not None:
            try:
                selected = select_mutation_targets(
                    properties,
                    repository_id=repo.repository_id,
                    repository_state_cid=repo.repository_state_cid,
                    budget=sealed_resource.as_sampling_budget(),
                )
            except TargetSelectionError as exc:
                raise CampaignPlanningError(
                    f"select_mutation_targets failed: {exc}",
                    reason_code="target_selection_failed",
                ) from exc
            sealed_targets = tuple(selected)
            if not sealed_targets:
                raise CampaignPlanningError(
                    "select_mutation_targets produced no targets",
                    reason_code="no_selected_targets",
                )
        else:
            raise CampaignPlanningError(
                "targets, properties, or generation_manifest is required",
                reason_code="missing_targets",
            )

    for target in sealed_targets:
        if target.repository_id != repo.repository_id:
            raise CampaignPlanningError(
                "target repository_id must match repository_state",
                reason_code="identity_mismatch",
            )
        if target.repository_state_cid != repo.repository_state_cid:
            raise CampaignPlanningError(
                "target repository_state_cid must match repository_state",
                reason_code="identity_mismatch",
            )

    selected_targets, risk_ranking = _budget_risk_weighted_targets(
        sealed_targets,
        policy=sealed_policy,
        resource_budget=sealed_resource,
    )

    # Resolve operators.
    if operators is not None:
        if not isinstance(operators, (list, tuple)):
            raise CampaignPlanningError("operators must be a sequence")
        if not operators:
            raise CampaignPlanningError("operators must not be empty")
        raw_ops = tuple(
            _normalize_operator(item, f"operators[{index}]")
            for index, item in enumerate(operators)
        )
    elif sealed_ops_from_manifest:
        raw_ops = sealed_ops_from_manifest
    else:
        raise CampaignPlanningError(
            "operators or generation_manifest.operators is required",
            reason_code="missing_operators",
        )
    selected_operators = _select_operators(
        raw_ops,
        policy=sealed_policy,
        resource_budget=sealed_resource,
    )

    if resolved_source_root is None:
        raise CampaignPlanningError(
            "source_root_cid is required (via generation_manifest, "
            "repository_state, or source_root_cid=)",
            reason_code="missing_source_root",
        )
    if resolved_env is None:
        raise CampaignPlanningError(
            "environment_cid is required (via generation_manifest, "
            "repository_state, or environment_cid=)",
            reason_code="missing_environment",
        )
    if resolved_dep is None:
        raise CampaignPlanningError(
            "dependency_lock_cid is required (via generation_manifest, "
            "repository_state, or dependency_lock_cid=)",
            reason_code="missing_dependency_lock",
        )
    resolved_source_root = _cid(resolved_source_root, "source_root_cid")
    resolved_env = _cid(resolved_env, "environment_cid")
    resolved_dep = _cid(resolved_dep, "dependency_lock_cid")

    # Effective policy budget is the intersected resource envelope so generation
    # and plan sealing share one hard bound.
    effective_budget = sealed_resource.as_campaign_budget()
    try:
        effective_policy = MutationCampaignPolicy(
            header=sealed_policy.header,
            policy_id=sealed_policy.policy_id,
            policy_version=sealed_policy.policy_version,
            admitted_operator_classes=sealed_policy.admitted_operator_classes,
            admitted_risk_classes=sealed_policy.admitted_risk_classes,
            budget=effective_budget,
            seed_config=active_seed,
            require_disposable_worktree=sealed_policy.require_disposable_worktree,
            require_network_disabled=sealed_policy.require_network_disabled,
            require_rollback=sealed_policy.require_rollback,
            require_deterministic_seed=sealed_policy.require_deterministic_seed,
            full_suite_fallback_enabled=sealed_policy.full_suite_fallback_enabled,
            held_out_partition_required=sealed_policy.held_out_partition_required,
            operator_cids=tuple(op.operator_cid for op in selected_operators),
            notes=sealed_policy.notes,
            metadata={
                **_thaw_structured(sealed_policy.metadata),
                "resource_budget_cid": sealed_resource.budget_cid,
                "planning_interface": PLAN_MUTATION_CAMPAIGN_INTERFACE,
            },
        )
    except MutationContractError as exc:
        raise CampaignPlanningError(f"effective policy seal failed: {exc}") from exc

    gen_input = MutationGenerationManifest(
        repository_id=repo.repository_id,
        repository_state_cid=repo.repository_state_cid,
        source_root_cid=resolved_source_root,
        targets=selected_targets,
        operators=selected_operators,
        environment_cid=resolved_env,
        dependency_lock_cid=resolved_dep,
        notes=_optional_text(notes, "notes"),
        metadata={
            "assurance_manifest_cid": sealed_manifest.manifest_cid,
            "baseline_receipt_cid": sealed_baseline.baseline_receipt_cid,
            "planning_interface": PLAN_MUTATION_CAMPAIGN_INTERFACE,
        },
    )

    generation = generate_mutation_candidates(
        gen_input,
        effective_policy,
        seed_config=active_seed,
        notes=notes,
        metadata={
            "assurance_manifest_cid": sealed_manifest.manifest_cid,
            "baseline_cid": sealed_baseline.baseline_cid,
        },
        return_result=True,
    )
    assert isinstance(generation, MutationGenerationResult)
    candidates = tuple(generation.candidates)
    if not candidates:
        raise CampaignPlanningError(
            "generation produced no candidates",
            reason_code="no_candidates",
        )

    # Predict explained detections for every candidate (stable candidate order).
    # Candidates whose violated property classes have no reachable detector in
    # the assurance catalog (and no applicable synthetic fallback) are dropped
    # rather than sealed into an unexplained plan. Fail closed if none remain.
    kept_candidates: list[MutationCandidate] = []
    detections: list[ExpectedDetectionSet] = []
    skipped_unexplained = 0
    for candidate in candidates:
        try:
            detection = predict_detection_set(
                candidate,
                sealed_manifest,
                notes=notes,
                metadata={
                    "plan_interface": PLAN_MUTATION_CAMPAIGN_INTERFACE,
                    "assurance_manifest_cid": sealed_manifest.manifest_cid,
                },
            )
        except CampaignPlanningError as exc:
            if getattr(exc, "reason_code", "") == "detection_failed":
                skipped_unexplained += 1
                continue
            raise
        kept_candidates.append(candidate)
        detections.append(detection)
    if not kept_candidates:
        raise CampaignPlanningError(
            "no candidates produced explained detection sets under the "
            "assurance_manifest catalog and synthetic fallbacks"
            + (
                f" ({skipped_unexplained} candidates lacked reachable detectors)"
                if skipped_unexplained
                else ""
            ),
            reason_code="no_explained_detections",
        )
    candidates = tuple(kept_candidates)
    expected_detections = tuple(detections)

    target_cids = tuple(
        sorted({target.target_cid for target in selected_targets})
    )
    operator_cids = tuple(
        sorted({operator.operator_cid for operator in selected_operators})
    )
    # Preserve generation order for candidate_cids identity (unique sorted by
    # MutationCampaignPlan contract).
    candidate_cids = tuple(candidate.candidate_cid for candidate in candidates)

    try:
        assert_budget_admits_counts(
            effective_budget,
            target_count=len(selected_targets),
            operator_count=len(selected_operators),
            candidate_count=len(candidates),
        )
    except MutationContractError as exc:
        raise CampaignPlanningError(str(exc)) from exc

    resolved_plan_id = (
        _token(plan_id, "plan_id")
        if plan_id is not None
        else _stable_plan_id(
            policy_id=effective_policy.policy_id,
            repository_state_cid=repo.repository_state_cid,
            baseline_receipt_cid=sealed_baseline.baseline_receipt_cid,
            seed=active_seed.seed,
            config_cid=active_seed.config_cid or active_seed.binding_cid,
            target_cids=target_cids,
            operator_cids=operator_cids,
        )
    )

    header = _plan_header(
        repository_id=repo.repository_id,
        repository_state_cid=repo.repository_state_cid,
        policy=effective_policy,
        targets=selected_targets,
        environment_cid=resolved_env,
        dependency_lock_cid=resolved_dep,
        baseline_receipt_cid=sealed_baseline.baseline_receipt_cid,
        seed_config=active_seed,
    )

    plan_metadata: dict[str, Any] = {
        "interface_id": PLAN_MUTATION_CAMPAIGN_INTERFACE,
        "generator_id": GENERATOR_ID,
        "generator_version": GENERATOR_VERSION,
        "assurance_manifest_cid": sealed_manifest.manifest_cid,
        "baseline_cid": sealed_baseline.baseline_cid,
        "resource_budget_cid": sealed_resource.budget_cid,
        "generation_manifest_cid": gen_input.manifest_cid,
        "candidate_count": len(candidates),
        "detection_count": len(expected_detections),
        "production_policy_changed": False,
    }
    if metadata:
        plan_metadata.update(_thaw_structured(_mapping(metadata, "metadata")))

    try:
        plan = MutationCampaignPlan(
            header=header,
            plan_id=resolved_plan_id,
            policy_id=effective_policy.policy_id,
            policy_version=effective_policy.policy_version,
            policy_cid=effective_policy.policy_cid,
            repository_id=repo.repository_id,
            repository_state_cid=repo.repository_state_cid,
            baseline_receipt_cid=sealed_baseline.baseline_receipt_cid,
            seed_config=active_seed,
            budget=effective_budget,
            target_cids=target_cids,
            operator_cids=operator_cids,
            candidate_cids=candidate_cids,
            admitted_risk_classes=effective_policy.admitted_risk_classes,
            require_sandbox=True,
            require_rollback=True,
            notes=_optional_text(notes, "notes"),
            metadata=plan_metadata,
        )
    except MutationContractError as exc:
        raise CampaignPlanningError(f"failed to seal MutationCampaignPlan: {exc}") from exc

    partition_plan: MutantPartitionPlan | None = None
    if partition and sealed_policy.held_out_partition_required:
        partition_plan = _build_partition(
            candidates=candidates,
            plan=plan,
            policy=effective_policy,
            header=header,
        )

    result = MutationCampaignPlanResult(
        plan=plan,
        baseline=sealed_baseline,
        resource_budget=sealed_resource,
        selected_targets=selected_targets,
        selected_operators=selected_operators,
        candidates=candidates,
        expected_detections=expected_detections,
        risk_ranking=risk_ranking,
        generation_result=generation,
        partition=partition_plan,
        assurance_manifest_cid=sealed_manifest.manifest_cid,
        notes=_optional_text(notes, "notes"),
        metadata={
            "interface_id": PLAN_MUTATION_CAMPAIGN_INTERFACE,
            "generator_id": GENERATOR_ID,
            "generator_version": GENERATOR_VERSION,
            "assurance_manifest_interface": ASSURANCE_MANIFEST_INTERFACE,
            "mutation_campaign_plan_schema": MUTATION_CAMPAIGN_PLAN_SCHEMA,
            "mutation_campaign_plan_interface": MUTATION_CAMPAIGN_PLAN_INTERFACE,
        },
        production_policy_changed=False,
    )
    return result if return_result else result.plan


__all__ = [
    "BASELINE_REQUIREMENTS_SCHEMA",
    "CAMPAIGN_PLAN_RESULT_INTERFACE",
    "CAMPAIGN_PLAN_RESULT_SCHEMA",
    "CAMPAIGN_RESOURCE_BUDGET_SCHEMA",
    "GENERATE_MUTATION_CANDIDATES_INTERFACE",
    "GENERATOR_ID",
    "GENERATOR_VERSION",
    "PLAN_MUTATION_CAMPAIGN_INTERFACE",
    "PREDICT_DETECTION_SET_INTERFACE",
    "BaselineRequirements",
    "CampaignPlanningError",
    "CampaignResourceBudget",
    "MutationCampaignPlanResult",
    "generate_mutation_candidates",
    "plan_mutation_campaign",
    "predict_detection_set",
]
