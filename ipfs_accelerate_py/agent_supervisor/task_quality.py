"""Deterministic quality, sizing, and semantic admission for generated tasks.

The task generator has two distinct responsibilities:

* describe a complete, executable unit of work; and
* keep the open board useful by rejecting duplicates and resizing work before
  it consumes scheduler capacity.

This module implements that boundary without depending on a model provider or
on Markdown display IDs.  Every decision is reproducible from canonical task
content, policy, history, and the current open-work count.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any, Final

from .task_identity import (
    canonical_content_cid,
    canonical_json_bytes,
    canonical_task_identity,
)


TASK_QUALITY_SCHEMA = "ipfs_accelerate_py/agent-supervisor/task-quality@1"
TASK_SEMANTIC_IDENTITY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/task-semantic-identity@1"
)
TASK_WORK_CONTRACT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/task-work-contract@1"
)
TASK_GRANULARITY_MEASUREMENT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/task-granularity-measurement@1"
)
TASK_GRANULARITY_CALIBRATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/task-granularity-calibration@1"
)
TASK_COMPLETION_PROPAGATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/task-completion-propagation@1"
)
TASK_GRANULARITY_RUN_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/task-granularity-run@1"
)
TASK_QUALITY_EVALUATOR_VERSION = "task-quality/v1"
TASK_GENERATION_OBJECTIVE_ID: Final = "ASI-G050"
TASK_GENERATION_REQUIRED_EXHAUSTIVE_RECEIPTS: Final = 2
TASK_GENERATION_PRODUCING_TASK_IDS: Final[tuple[str, ...]] = (
    "ASI-013",
    "ASI-014",
)
TASK_GENERATION_CHILD_GOAL_IDS: Final[tuple[str, ...]] = (
    "ASI-G106",
    "ASI-G107",
    "ASI-G108",
)
TASK_GENERATION_ACCEPTANCE_CRITERIA: Final[tuple[str, ...]] = (
    "Tasks bind one coherent acceptance/effect subset with predicted scope and costs",
    "broad tasks split and compatible tiny tasks coalesce",
    "semantic duplicates are rejected across refills",
    "bundles preserve critical-path width and serialize conflicts",
    "model calls per accepted work item improve without increasing merge conflicts",
)
TASK_SPLIT_REFILL_REQUIREMENT_ID: Final = (
    "127990245919649912156052660092678945998"
)
TASK_SPLIT_REFILL_EVIDENCE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/task-split-refill-evidence@1"
)
_TASK_SPLIT_REFILL_EVIDENCE_SEAL: Final = object()

RESOURCE_CLASSES = frozenset(
    {
        "cpu-small",
        "cpu-medium",
        "cpu-large",
        "cpu-proof-solver",
        "gpu-small",
        "gpu-medium",
        "gpu-large",
        "io-small",
        "io-medium",
        "network",
    }
)
TOKEN_CLASSES = frozenset({"tiny", "small", "medium", "large", "xlarge"})
TOKEN_CLASS_LIMITS = {
    "tiny": 1_024,
    "small": 4_096,
    "medium": 16_384,
    "large": 32_768,
    "xlarge": 65_536,
}


def _normalized_text(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def _normalized_display_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _normalized_path(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return re.sub(r"/+", "/", text).rstrip("/")


def _strings(value: Any, *, paths: bool = False) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        raw: Iterable[Any] = re.split(r"[,;\n]+", value)
    elif isinstance(value, Mapping):
        raw = value.keys()
    elif isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        raw = value
    else:
        raw = (value,)
    normalize = _normalized_path if paths else _normalized_display_text
    values = {normalize(item) for item in raw if normalize(item)}
    return tuple(sorted(values, key=lambda item: (item.casefold(), item)))


def _mapping_value(payload: Mapping[str, Any], *names: str, default: Any = "") -> Any:
    normalized = {
        str(key).strip().casefold().replace("_", " "): value
        for key, value in payload.items()
    }
    for name in names:
        value = normalized.get(name.casefold().replace("_", " "))
        if value not in (None, ""):
            return value
    return default


def _non_negative_int(value: Any, name: str) -> int:
    if value in (None, ""):
        return 0
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return parsed


def _finite_ratio(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number between zero and one") from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{name} must be a number between zero and one")
    return parsed


def _positive_int(value: Any, name: str) -> int:
    parsed = _non_negative_int(value, name)
    if parsed == 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _evidence_hash_material(value: Any) -> Any:
    """Project finite floats to stable strings for canonical receipt hashing."""

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("task-quality evidence cannot contain non-finite floats")
        return {"$finite_float": format(value, ".17g")}
    if isinstance(value, Mapping):
        return {
            str(key): _evidence_hash_material(child)
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_evidence_hash_material(child) for child in value]
    return value


def _task_quality_evidence_bytes(value: Any) -> bytes:
    return canonical_json_bytes(_evidence_hash_material(value))


def _task_quality_evidence_cid(value: Any) -> str:
    return canonical_content_cid(_evidence_hash_material(value))


def _authoritative_repository_tree(value: Any) -> bool:
    """Return whether a receipt names a concrete repository snapshot."""

    normalized = str(value or "").strip().casefold()
    return normalized not in {
        "",
        "in-memory",
        "in_memory",
        "memory",
        "unknown",
        "unbound",
        "working-tree",
        "working_tree",
    }


def _task_work_contract_material(candidate: "TaskCandidate") -> dict[str, Any]:
    """Project one candidate's complete acceptance, scope, and cost contract.

    Acceptance criteria, effects, and their evidence delta deliberately live
    in one subset instead of independent arrays which a downstream projection
    could accidentally mix with another task.  Scope and cost estimates are
    part of the same canonical object, so changing an execution estimate
    changes the semantic task identity rather than silently reusing stale
    admission or bundle evidence.
    """

    result = {
        "schema": TASK_WORK_CONTRACT_SCHEMA,
        "goal_id": _normalized_text(candidate.goal_id),
        "acceptance_effect_subset": {
            "acceptance": sorted(
                _normalized_text(item) for item in candidate.acceptance
            ),
            "effects": sorted(
                _normalized_text(item) for item in candidate.effects
            ),
            "evidence_subset": sorted(
                _normalized_text(item) for item in candidate.evidence_subset
            ),
        },
        "predicted_scope": {
            "paths": sorted(
                path.casefold()
                for path in set(candidate.outputs) | set(candidate.predicted_paths)
            ),
            "symbols": sorted(
                _normalized_text(item) for item in candidate.predicted_symbols
            ),
            "context_paths": sorted(
                path.casefold() for path in candidate.context_paths
            ),
        },
        "predicted_costs": {
            "context_tokens": candidate.estimated_context_tokens,
            "validation_seconds": candidate.estimated_validation_seconds,
            "task_tokens": candidate.estimated_tokens,
            "resource_class": candidate.resource_class.casefold(),
            "token_class": candidate.token_class.casefold(),
            "dependency_count": len(candidate.dependencies),
            "conflict_count": len(candidate.conflicts),
        },
        "execution_boundary": {
            "preconditions": sorted(
                _normalized_text(item) for item in candidate.preconditions
            ),
            "dependencies": sorted(
                _normalized_display_text(item) for item in candidate.dependencies
            ),
            "conflicts": sorted(
                _normalized_display_text(item) for item in candidate.conflicts
            ),
            "validation_commands": sorted(
                _normalized_display_text(item)
                for item in candidate.validation_commands
            ),
            "merge_fate": _normalized_text(candidate.merge_fate),
        },
    }
    # Keep the v1 projection byte-for-byte compatible for legacy candidates,
    # while binding the richer ASI-110 interface/proof/risk predictions whenever
    # they are declared.
    if candidate.predicted_interfaces:
        result["predicted_scope"]["interfaces"] = sorted(
            _normalized_text(item) for item in candidate.predicted_interfaces
        )
    if candidate.proof_obligations or candidate.proof_commands:
        result["predicted_proof"] = {
            "obligations": sorted(
                _normalized_text(item) for item in candidate.proof_obligations
            ),
            "commands": sorted(
                _normalized_display_text(item) for item in candidate.proof_commands
            ),
            "estimated_seconds": candidate.estimated_proof_seconds,
        }
    if candidate.estimated_merge_risk_millionths:
        result["predicted_costs"]["merge_risk_millionths"] = (
            candidate.estimated_merge_risk_millionths
        )
    shard = _normalized_display_text(
        candidate.metadata.get("granularity_shard")
    )
    if shard:
        result["execution_boundary"]["granularity_shard"] = shard
    return result


def _semantic_material(value: "TaskCandidate | Mapping[str, Any]") -> dict[str, Any]:
    candidate = (
        value
        if isinstance(value, TaskCandidate)
        else TaskCandidate.from_mapping(value, validate_identity=False)
    )
    material: dict[str, Any] = {
        "schema": TASK_SEMANTIC_IDENTITY_SCHEMA,
        "work_contract": _task_work_contract_material(candidate),
        "goal_id": _normalized_text(candidate.goal_id),
        "acceptance": sorted(_normalized_text(item) for item in candidate.acceptance),
        "preconditions": sorted(_normalized_text(item) for item in candidate.preconditions),
        "effects": sorted(_normalized_text(item) for item in candidate.effects),
        "evidence_subset": sorted(
            _normalized_text(item) for item in candidate.evidence_subset
        ),
        "outputs": sorted(path.casefold() for path in candidate.outputs),
        "predicted_symbols": sorted(
            _normalized_text(item) for item in candidate.predicted_symbols
        ),
        "validation_commands": sorted(
            _normalized_display_text(item) for item in candidate.validation_commands
        ),
        "merge_fate": _normalized_text(candidate.merge_fate),
    }
    if candidate.predicted_interfaces:
        material["predicted_interfaces"] = sorted(
            _normalized_text(item) for item in candidate.predicted_interfaces
        )
    if candidate.proof_obligations or candidate.proof_commands:
        material["proof_obligations"] = sorted(
            _normalized_text(item) for item in candidate.proof_obligations
        )
        material["proof_commands"] = sorted(
            _normalized_display_text(item) for item in candidate.proof_commands
        )
    shard = _normalized_display_text(
        candidate.metadata.get("granularity_shard")
    )
    if shard:
        material["granularity_shard"] = shard
    # Context changes execution cost, but not the work's purpose.  It is used
    # as identity material only when there is no concrete output/symbol surface.
    if not material["outputs"] and not material["predicted_symbols"]:
        material["context_paths"] = sorted(
            path.casefold() for path in candidate.context_paths
        )
    if not any(
        material[name]
        for name in ("acceptance", "effects", "evidence_subset", "outputs", "predicted_symbols")
    ):
        material["title"] = _normalized_text(candidate.title)
    return material


def canonical_semantic_identity(value: "TaskCandidate | Mapping[str, Any]") -> str:
    """Return a display-ID-independent identity for one semantic unit of work."""

    digest = hashlib.sha256(canonical_json_bytes(_semantic_material(value))).hexdigest()
    return f"task-quality/v1/{digest}"


@dataclass(frozen=True)
class TaskCandidate:
    """A candidate task at the generated-task admission boundary.

    Fields which make a task executable intentionally default to empty.  This
    lets :func:`score_task_candidate` return explicit rejection reasons instead
    of failing while decoding an incomplete model response.
    """

    title: str = ""
    goal_id: str = ""
    acceptance: tuple[str, ...] = ()
    acceptance_criteria: tuple[str, ...] = ()
    preconditions: tuple[str, ...] = ()
    effects: tuple[str, ...] = ()
    evidence_subset: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    validation_commands: tuple[str, ...] = ()
    context_paths: tuple[str, ...] = ()
    context_keys: tuple[str, ...] = ()
    predicted_paths: tuple[str, ...] = ()
    predicted_symbols: tuple[str, ...] = ()
    predicted_interfaces: tuple[str, ...] = ()
    proof_obligations: tuple[str, ...] = ()
    proof_commands: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    resources: tuple[str, ...] = ()
    resource_class: str = ""
    token_class: str = ""
    merge_fate: str = ""
    priority: str = ""
    track: str = ""
    estimated_context_tokens: int = 0
    estimated_validation_seconds: int = 0
    estimated_proof_seconds: int = 0
    estimated_tokens: int = 0
    estimated_merge_risk_millionths: int = 0
    historical_duplicate_similarity: float = 0.0
    historical_failure_similarity: float = 0.0
    source_id: str = ""
    semantic_identity: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        normalized_acceptance = _strings(
            self.acceptance or self.acceptance_criteria
        )
        normalized_context = _strings(
            self.context_paths or self.context_keys,
            paths=True,
        )
        object.__setattr__(self, "acceptance", normalized_acceptance)
        object.__setattr__(self, "acceptance_criteria", normalized_acceptance)
        object.__setattr__(self, "context_paths", normalized_context)
        object.__setattr__(self, "context_keys", normalized_context)
        for name in (
            "title",
            "goal_id",
            "resource_class",
            "token_class",
            "merge_fate",
            "priority",
            "track",
            "source_id",
        ):
            object.__setattr__(self, name, _normalized_display_text(getattr(self, name)))
        for name in (
            "preconditions",
            "effects",
            "evidence_subset",
            "validation_commands",
            "predicted_symbols",
            "predicted_interfaces",
            "proof_obligations",
            "proof_commands",
            "dependencies",
            "conflicts",
            "resources",
        ):
            object.__setattr__(self, name, _strings(getattr(self, name)))
        for name in ("outputs", "predicted_paths"):
            object.__setattr__(self, name, _strings(getattr(self, name), paths=True))
        for name in (
            "estimated_context_tokens",
            "estimated_validation_seconds",
            "estimated_proof_seconds",
            "estimated_tokens",
            "estimated_merge_risk_millionths",
        ):
            object.__setattr__(
                self,
                name,
                _non_negative_int(getattr(self, name), name),
            )
        for name in (
            "historical_duplicate_similarity",
            "historical_failure_similarity",
        ):
            object.__setattr__(self, name, _finite_ratio(getattr(self, name), name))
        metadata = dict(self.metadata) if isinstance(self.metadata, Mapping) else {}
        object.__setattr__(self, "metadata", metadata)
        expected = canonical_semantic_identity(self)
        supplied = str(self.semantic_identity or "").strip()
        if supplied and supplied != expected:
            raise ValueError(
                "semantic_identity does not match canonical semantic task content"
            )
        object.__setattr__(self, "semantic_identity", expected)

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        validate_identity: bool = True,
    ) -> "TaskCandidate":
        if not isinstance(payload, Mapping):
            raise TypeError("task candidates must be mappings")
        candidate = cls(
            title=_mapping_value(payload, "title", "summary"),
            goal_id=_mapping_value(
                payload,
                "goal id",
                "parent goal id",
                "parent objective id",
                "goal",
            ),
            acceptance=_strings(
                _mapping_value(
                    payload,
                    "acceptance",
                    "acceptance subset",
                    "acceptance criteria",
                )
            ),
            preconditions=_strings(
                _mapping_value(payload, "preconditions", "required preconditions")
            ),
            effects=_strings(
                _mapping_value(payload, "effects", "expected effects")
            ),
            evidence_subset=_strings(
                _mapping_value(
                    payload,
                    "evidence subset",
                    "expected evidence delta",
                    "missing evidence",
                    "evidence",
                )
            ),
            outputs=_strings(
                _mapping_value(payload, "outputs", "files"),
                paths=True,
            ),
            validation_commands=_strings(
                _mapping_value(payload, "validation commands", "validation")
            ),
            context_paths=_strings(
                _mapping_value(
                    payload,
                    "context paths",
                    "context keys",
                    "context files",
                    "context",
                ),
                paths=True,
            ),
            predicted_paths=_strings(
                _mapping_value(payload, "predicted paths", "predicted files"),
                paths=True,
            ),
            predicted_symbols=_strings(
                _mapping_value(payload, "predicted symbols", "ast symbols", "symbols")
            ),
            predicted_interfaces=_strings(
                _mapping_value(
                    payload,
                    "predicted interfaces",
                    "interfaces",
                    "interface contracts",
                )
            ),
            proof_obligations=_strings(
                _mapping_value(
                    payload,
                    "proof obligations",
                    "predicted proof",
                    "proof subset",
                )
            ),
            proof_commands=_strings(
                _mapping_value(payload, "proof commands", "proof validation")
            ),
            dependencies=_strings(
                _mapping_value(payload, "dependencies", "depends on")
            ),
            conflicts=_strings(
                _mapping_value(payload, "conflicts", "conflict keys")
            ),
            resources=_strings(
                _mapping_value(payload, "resources", "required resources")
            ),
            resource_class=_mapping_value(payload, "resource class"),
            token_class=_mapping_value(payload, "token class", "token budget class"),
            merge_fate=_mapping_value(payload, "merge fate", "merge family", "merge key"),
            priority=_mapping_value(payload, "priority"),
            track=_mapping_value(payload, "track"),
            estimated_context_tokens=_mapping_value(
                payload, "estimated context tokens", "context tokens", default=0
            ),
            estimated_validation_seconds=_mapping_value(
                payload,
                "estimated validation seconds",
                "validation seconds",
                "validation cost",
                default=0,
            ),
            estimated_proof_seconds=_mapping_value(
                payload,
                "estimated proof seconds",
                "proof seconds",
                "proof cost",
                default=0,
            ),
            estimated_tokens=_mapping_value(
                payload, "estimated tokens", "token cost", default=0
            ),
            estimated_merge_risk_millionths=_mapping_value(
                payload,
                "estimated merge risk millionths",
                "merge risk millionths",
                default=0,
            ),
            historical_duplicate_similarity=_mapping_value(
                payload,
                "historical duplicate similarity",
                "duplicate similarity",
                default=0.0,
            ),
            historical_failure_similarity=_mapping_value(
                payload,
                "historical failure similarity",
                "failure similarity",
                default=0.0,
            ),
            source_id=_mapping_value(
                payload, "source id", "task id", "finding id", "proposal id"
            ),
            semantic_identity=(
                _mapping_value(
                    payload,
                    "canonical semantic identity",
                )
                if validate_identity
                else ""
            ),
            metadata=payload.get("metadata", {}),
        )
        if validate_identity:
            supplied_contract = _mapping_value(payload, "work contract")
            if supplied_contract not in (None, ""):
                if (
                    not isinstance(supplied_contract, Mapping)
                    or dict(supplied_contract) != candidate.work_contract
                ):
                    raise ValueError(
                        "work_contract does not match canonical task work content"
                    )
            supplied_contract_id = str(
                _mapping_value(payload, "work contract id") or ""
            ).strip()
            if (
                supplied_contract_id
                and supplied_contract_id != candidate.work_contract_id
            ):
                raise ValueError(
                    "work_contract_id does not match canonical task work content"
                )
            supplied_key = str(
                _mapping_value(payload, "canonical task key") or ""
            ).strip()
            supplied_cid = str(
                _mapping_value(payload, "canonical task cid", "task cid") or ""
            ).strip()
            if supplied_key and supplied_key != candidate.canonical_task_key:
                raise ValueError(
                    "canonical_task_key does not match canonical semantic task content"
                )
            if supplied_cid and supplied_cid != candidate.canonical_task_cid:
                raise ValueError(
                    "canonical_task_cid does not match canonical semantic task content"
                )
        return candidate

    from_dict = from_mapping

    @property
    def predicted_path_breadth(self) -> int:
        return len(set(self.outputs) | set(self.predicted_paths))

    @property
    def predicted_symbol_breadth(self) -> int:
        return len(self.predicted_symbols)

    @property
    def predicted_interface_breadth(self) -> int:
        return len(self.predicted_interfaces)

    @property
    def predicted_proof_breadth(self) -> int:
        return len(self.proof_obligations)

    @property
    def work_contract(self) -> dict[str, Any]:
        """Return the canonical single-subset work contract."""

        return _task_work_contract_material(self)

    @property
    def work_contract_id(self) -> str:
        """Return the content identity of acceptance, scope, and cost binding."""

        return _task_quality_evidence_cid(self.work_contract)

    @property
    def predicted_costs_complete(self) -> bool:
        """Return whether every mandatory execution-cost estimate is bound."""

        return (
            self.estimated_context_tokens > 0
            and self.estimated_validation_seconds > 0
            and self.estimated_tokens > 0
        )

    @property
    def canonical_task_key(self) -> str:
        """Return the repository-wide canonical task key for this candidate."""

        return canonical_task_identity(
            {"dedupe_key": self.semantic_identity}
        ).canonical_task_key

    @property
    def canonical_task_cid(self) -> str:
        """Return the repository-wide canonical task CID for this candidate."""

        return canonical_task_identity(
            {"dedupe_key": self.semantic_identity}
        ).canonical_task_cid

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for name in (
            "acceptance",
            "acceptance_criteria",
            "preconditions",
            "effects",
            "evidence_subset",
            "outputs",
            "validation_commands",
            "context_paths",
            "context_keys",
            "predicted_paths",
            "predicted_symbols",
            "predicted_interfaces",
            "proof_obligations",
            "proof_commands",
            "dependencies",
            "conflicts",
            "resources",
        ):
            result[name] = list(result[name])
        result["canonical_semantic_identity"] = self.semantic_identity
        result["canonical_task_key"] = self.canonical_task_key
        result["canonical_task_cid"] = self.canonical_task_cid
        result["task_cid"] = self.canonical_task_cid
        result["work_contract"] = self.work_contract
        result["work_contract_id"] = self.work_contract_id
        result["predicted_costs_complete"] = self.predicted_costs_complete
        result["predicted_path_breadth"] = self.predicted_path_breadth
        result["predicted_symbol_breadth"] = self.predicted_symbol_breadth
        result["predicted_interface_breadth"] = self.predicted_interface_breadth
        result["predicted_proof_breadth"] = self.predicted_proof_breadth
        return result


@dataclass(frozen=True)
class HistoricalTask:
    """A normalized historical work item used for similarity penalties."""

    candidate: TaskCandidate
    outcome: str = "accepted"
    failure_reason: str = ""

    @classmethod
    def from_value(cls, value: Any, *, default_outcome: str) -> "HistoricalTask":
        if isinstance(value, cls):
            return value
        if isinstance(value, TaskCandidate):
            return cls(value, default_outcome)
        if not isinstance(value, Mapping):
            raise TypeError("historical task records must be candidates or mappings")
        task_payload = value.get("candidate") or value.get("task") or value
        return cls(
            candidate=(
                task_payload
                if isinstance(task_payload, TaskCandidate)
                else TaskCandidate.from_mapping(task_payload)
            ),
            outcome=_normalized_display_text(value.get("outcome") or default_outcome).casefold(),
            failure_reason=_normalized_display_text(
                value.get("failure_reason") or value.get("reason") or ""
            ),
        )


@dataclass(frozen=True)
class TaskQualityPolicy:
    """Admission thresholds and deterministic sizing limits."""

    min_quality_score: float = 0.55
    duplicate_similarity_threshold: float = 0.82
    failure_similarity_threshold: float = 0.90
    max_predicted_paths: int = 8
    max_predicted_symbols: int = 24
    max_predicted_interfaces: int = 16
    max_acceptance_criteria: int = 12
    max_effects: int = 12
    max_evidence_items: int = 16
    max_context_paths: int = 16
    max_context_tokens: int = 24_000
    max_validation_seconds: int = 1_800
    max_proof_items: int = 16
    max_proof_seconds: int = 1_800
    max_estimated_tokens: int = 32_768
    max_merge_risk_millionths: int = 1_000_000
    max_dependencies: int = 16
    max_conflicts: int = 8
    tiny_max_paths: int = 2
    tiny_max_symbols: int = 2
    tiny_max_tokens: int = 1_024
    max_open_work: int = 48
    max_new_work: int = 12
    max_split_parts: int = 32
    split_over_broad: bool = True
    coalesce_tiny: bool = True

    def __post_init__(self) -> None:
        for name in (
            "min_quality_score",
            "duplicate_similarity_threshold",
            "failure_similarity_threshold",
        ):
            object.__setattr__(self, name, _finite_ratio(getattr(self, name), name))
        for name in (
            "max_predicted_paths",
            "max_predicted_symbols",
            "max_predicted_interfaces",
            "max_acceptance_criteria",
            "max_effects",
            "max_evidence_items",
            "max_context_paths",
            "max_context_tokens",
            "max_validation_seconds",
            "max_proof_items",
            "max_proof_seconds",
            "max_estimated_tokens",
            "max_merge_risk_millionths",
            "max_dependencies",
            "max_conflicts",
            "tiny_max_paths",
            "tiny_max_symbols",
            "tiny_max_tokens",
            "max_open_work",
            "max_new_work",
            "max_split_parts",
        ):
            value = _non_negative_int(getattr(self, name), name)
            object.__setattr__(self, name, value)
        if (
            self.max_predicted_paths == 0
            or self.max_predicted_symbols == 0
            or self.max_predicted_interfaces == 0
            or self.max_acceptance_criteria == 0
            or self.max_effects == 0
            or self.max_evidence_items == 0
            or self.max_proof_items == 0
            or self.max_split_parts == 0
        ):
            raise ValueError("task breadth limits must be positive")
        if self.max_new_work > self.max_open_work:
            object.__setattr__(self, "max_new_work", self.max_open_work)

    @property
    def policy_id(self) -> str:
        """Return the content identity of every sizing and admission threshold."""

        digest = hashlib.sha256(
            _task_quality_evidence_bytes(asdict(self))
        ).hexdigest()
        return f"task-quality-policy/v1/{digest}"


@dataclass(frozen=True)
class TaskCostMeasurement:
    """One successful task execution used to calibrate granularity.

    Measurements are deliberately feature-scoped.  A value from another tree,
    policy, or toolchain is retained as an excluded diagnostic by calibration
    and can never affect an effective bound.
    """

    fixture_id: str
    repository_tree: str
    policy_id: str
    toolchain_features: tuple[str, ...]
    acceptance_count: int
    context_path_count: int
    context_tokens: int
    predicted_file_count: int
    predicted_symbol_count: int
    predicted_interface_count: int
    validation_seconds: int
    proof_item_count: int
    proof_seconds: int
    task_tokens: int
    merge_risk_millionths: int
    model_calls: int
    accepted_criteria: int
    measurement_id: str = ""

    def __post_init__(self) -> None:
        for name in ("fixture_id", "repository_tree", "policy_id"):
            object.__setattr__(
                self, name, _normalized_display_text(getattr(self, name))
            )
            if not getattr(self, name):
                raise ValueError(f"{name} is required")
        features = _strings(self.toolchain_features)
        if not features:
            raise ValueError("toolchain_features must identify the measured toolchain")
        object.__setattr__(self, "toolchain_features", features)
        for name in (
            "acceptance_count",
            "context_path_count",
            "context_tokens",
            "predicted_file_count",
            "predicted_symbol_count",
            "predicted_interface_count",
            "validation_seconds",
            "proof_item_count",
            "proof_seconds",
            "task_tokens",
            "accepted_criteria",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        object.__setattr__(
            self, "model_calls", _non_negative_int(self.model_calls, "model_calls")
        )
        if self.accepted_criteria > self.acceptance_count:
            raise ValueError("accepted_criteria cannot exceed acceptance_count")
        risk = _non_negative_int(
            self.merge_risk_millionths, "merge_risk_millionths"
        )
        if risk > 1_000_000:
            raise ValueError("merge_risk_millionths cannot exceed 1000000")
        object.__setattr__(self, "merge_risk_millionths", risk)
        expected = _task_quality_evidence_cid(self._material())
        supplied = str(self.measurement_id or "").strip()
        if supplied and supplied != expected:
            raise ValueError(
                "measurement_id does not match canonical task cost measurement"
            )
        object.__setattr__(self, "measurement_id", expected)

    def _material(self) -> dict[str, Any]:
        return {
            "schema": TASK_GRANULARITY_MEASUREMENT_SCHEMA,
            "fixture_id": self.fixture_id,
            "repository_tree": self.repository_tree,
            "policy_id": self.policy_id,
            "toolchain_features": list(self.toolchain_features),
            "acceptance_count": self.acceptance_count,
            "context_path_count": self.context_path_count,
            "context_tokens": self.context_tokens,
            "predicted_file_count": self.predicted_file_count,
            "predicted_symbol_count": self.predicted_symbol_count,
            "predicted_interface_count": self.predicted_interface_count,
            "validation_seconds": self.validation_seconds,
            "proof_item_count": self.proof_item_count,
            "proof_seconds": self.proof_seconds,
            "task_tokens": self.task_tokens,
            "merge_risk_millionths": self.merge_risk_millionths,
            "model_calls": self.model_calls,
            "accepted_criteria": self.accepted_criteria,
        }

    def matches(
        self,
        *,
        repository_tree: str,
        policy_id: str,
        toolchain_features: Iterable[str],
    ) -> bool:
        return (
            self.repository_tree == _normalized_display_text(repository_tree)
            and self.policy_id == _normalized_display_text(policy_id)
            and self.toolchain_features == _strings(toolchain_features)
        )

    @property
    def model_calls_per_accepted_criterion(self) -> float:
        return self.model_calls / self.accepted_criteria

    def to_dict(self) -> dict[str, Any]:
        result = self._material()
        result["measurement_id"] = self.measurement_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TaskCostMeasurement":
        allowed = {
            "schema",
            "fixture_id",
            "repository_tree",
            "policy_id",
            "toolchain_features",
            "acceptance_count",
            "context_path_count",
            "context_tokens",
            "predicted_file_count",
            "predicted_symbol_count",
            "predicted_interface_count",
            "validation_seconds",
            "proof_item_count",
            "proof_seconds",
            "task_tokens",
            "merge_risk_millionths",
            "model_calls",
            "accepted_criteria",
            "measurement_id",
        }
        extras = set(payload) - allowed
        if extras:
            raise ValueError(
                "task cost measurement has unknown fields: "
                + ", ".join(sorted(str(item) for item in extras))
            )
        if str(payload.get("schema") or "") != TASK_GRANULARITY_MEASUREMENT_SCHEMA:
            raise ValueError("task cost measurement schema mismatch")
        return cls(
            **{
                name: payload.get(name)
                for name in cls.__dataclass_fields__
            }
        )


@dataclass(frozen=True)
class TaskGranularityCalibration:
    """Deterministic effective policy derived from matching measurements."""

    repository_tree: str
    source_policy_id: str
    toolchain_features: tuple[str, ...]
    matching_measurement_ids: tuple[str, ...]
    excluded_measurement_ids: tuple[str, ...]
    effective_policy: TaskQualityPolicy
    calibration_id: str = ""

    def __post_init__(self) -> None:
        tree = _normalized_display_text(self.repository_tree)
        policy_id = _normalized_display_text(self.source_policy_id)
        features = _strings(self.toolchain_features)
        matching = _strings(self.matching_measurement_ids)
        excluded = _strings(self.excluded_measurement_ids)
        if not tree or not policy_id or not features or not matching:
            raise ValueError(
                "calibration requires tree, source policy, toolchain, and "
                "at least one matching measurement"
            )
        if set(matching) & set(excluded):
            raise ValueError("matching and excluded measurements must be disjoint")
        object.__setattr__(self, "repository_tree", tree)
        object.__setattr__(self, "source_policy_id", policy_id)
        object.__setattr__(self, "toolchain_features", features)
        object.__setattr__(self, "matching_measurement_ids", matching)
        object.__setattr__(self, "excluded_measurement_ids", excluded)
        expected = _task_quality_evidence_cid(self._identity_material())
        supplied = str(self.calibration_id or "").strip()
        if supplied and supplied != expected:
            raise ValueError(
                "calibration_id does not match canonical task granularity calibration"
            )
        object.__setattr__(self, "calibration_id", expected)

    def _identity_material(self) -> dict[str, Any]:
        return {
            "schema": TASK_GRANULARITY_CALIBRATION_SCHEMA,
            "repository_tree": self.repository_tree,
            "source_policy_id": self.source_policy_id,
            "toolchain_features": list(self.toolchain_features),
            "matching_measurement_ids": list(self.matching_measurement_ids),
            "effective_policy": asdict(self.effective_policy),
            "effective_policy_id": self.effective_policy.policy_id,
        }

    def applies_to(
        self,
        policy: TaskQualityPolicy,
        *,
        repository_tree: str | None = None,
        toolchain_features: Iterable[str] | None = None,
    ) -> bool:
        if self.source_policy_id != policy.policy_id:
            return False
        if repository_tree is not None and self.repository_tree != _normalized_display_text(
            repository_tree
        ):
            return False
        if toolchain_features is not None and self.toolchain_features != _strings(
            toolchain_features
        ):
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        result = self._identity_material()
        result["excluded_measurement_ids"] = list(self.excluded_measurement_ids)
        result["calibration_id"] = self.calibration_id
        return result


def calibrate_task_granularity(
    measurements: Iterable[TaskCostMeasurement | Mapping[str, Any]],
    *,
    repository_tree: str,
    policy: TaskQualityPolicy | None = None,
    toolchain_features: Iterable[str],
) -> TaskGranularityCalibration:
    """Derive conservative bounds from exactly matching successful history."""

    selected = policy or TaskQualityPolicy()
    requested_features = _strings(toolchain_features)
    history = tuple(
        item
        if isinstance(item, TaskCostMeasurement)
        else TaskCostMeasurement.from_dict(item)
        for item in measurements
    )
    matching = tuple(
        sorted(
            (
                item
                for item in history
                if item.matches(
                    repository_tree=repository_tree,
                    policy_id=selected.policy_id,
                    toolchain_features=requested_features,
                )
            ),
            key=lambda item: item.measurement_id,
        )
    )
    if not matching:
        raise ValueError(
            "no task cost measurements match repository tree, policy, and "
            "toolchain features"
        )

    def observed_max(name: str, policy_limit: int) -> int:
        return max(1, min(policy_limit, max(getattr(item, name) for item in matching)))

    effective = replace(
        selected,
        max_acceptance_criteria=observed_max(
            "acceptance_count", selected.max_acceptance_criteria
        ),
        max_context_paths=observed_max(
            "context_path_count", selected.max_context_paths
        ),
        max_context_tokens=observed_max(
            "context_tokens", selected.max_context_tokens
        ),
        max_predicted_paths=observed_max(
            "predicted_file_count", selected.max_predicted_paths
        ),
        max_predicted_symbols=observed_max(
            "predicted_symbol_count", selected.max_predicted_symbols
        ),
        max_predicted_interfaces=observed_max(
            "predicted_interface_count", selected.max_predicted_interfaces
        ),
        max_validation_seconds=observed_max(
            "validation_seconds", selected.max_validation_seconds
        ),
        max_proof_items=observed_max(
            "proof_item_count", selected.max_proof_items
        ),
        max_proof_seconds=observed_max(
            "proof_seconds", selected.max_proof_seconds
        ),
        max_estimated_tokens=observed_max(
            "task_tokens", selected.max_estimated_tokens
        ),
        max_merge_risk_millionths=min(
            selected.max_merge_risk_millionths,
            max(item.merge_risk_millionths for item in matching),
        ),
    )
    matching_ids = tuple(item.measurement_id for item in matching)
    matching_set = set(matching_ids)
    excluded_ids = tuple(
        sorted(
            item.measurement_id
            for item in history
            if item.measurement_id not in matching_set
        )
    )
    return TaskGranularityCalibration(
        repository_tree=repository_tree,
        source_policy_id=selected.policy_id,
        toolchain_features=requested_features,
        matching_measurement_ids=matching_ids,
        excluded_measurement_ids=excluded_ids,
        effective_policy=effective,
    )


def _effective_granularity_policy(
    policy: TaskQualityPolicy,
    calibration: TaskGranularityCalibration | None,
) -> TaskQualityPolicy:
    if calibration is None:
        return policy
    if not calibration.applies_to(policy):
        raise ValueError("task granularity calibration does not match source policy")
    return calibration.effective_policy


@dataclass(frozen=True)
class TaskQualityScore:
    """The complete auditable score for one task candidate."""

    acceptance_coverage: float
    coherent_effects: float
    breadth_fit: float
    context_cost: float
    validation_cost: float
    dependency_cost: float
    conflict_cost: float
    resource_fit: float
    historical_novelty: float
    historical_failure_safety: float
    duplicate_similarity: float
    failure_similarity: float
    total: float
    rationale: tuple[str, ...] = ()
    rejection_reasons: tuple["TaskRejection", ...] = ()

    @property
    def score(self) -> float:
        return self.total

    @property
    def total_score(self) -> float:
        return self.total

    @property
    def accepted(self) -> bool:
        return not self.rejection_reasons

    @property
    def dimensions(self) -> dict[str, float]:
        return {
            "acceptance_coverage": self.acceptance_coverage,
            "effect_coherence": self.coherent_effects,
            "scope": self.breadth_fit,
            "context_cost": self.context_cost,
            "validation_cost": self.validation_cost,
            "dependency_quality": self.dependency_cost,
            "conflict_cost": self.conflict_cost,
            "resource_fit": self.resource_fit,
            "duplicate_novelty": self.historical_novelty,
            "failure_risk": self.historical_failure_safety,
        }

    @property
    def total_millionths(self) -> int:
        return int(round(self.total * 1_000_000))

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["rationale"] = list(self.rationale)
        result["rejection_reasons"] = [
            item.to_dict() for item in self.rejection_reasons
        ]
        result["accepted"] = self.accepted
        result["dimensions"] = self.dimensions
        result["total_millionths"] = self.total_millionths
        return result


@dataclass(frozen=True)
class TaskRejection:
    """A machine-readable reason a candidate did not consume board capacity."""

    reason: str
    semantic_identity: str
    detail: str = ""

    @property
    def code(self) -> str:
        return self.reason

    @property
    def message(self) -> str:
        return self.detail

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


class TaskAdmissionStatus(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    SPLIT = "split"
    COALESCED = "coalesced"


@dataclass(frozen=True)
class TaskAdmissionDecision:
    candidate: TaskCandidate
    score: TaskQualityScore
    status: TaskAdmissionStatus
    rejections: tuple[TaskRejection, ...] = ()
    source_identities: tuple[str, ...] = ()

    @property
    def accepted(self) -> bool:
        return self.status in {
            TaskAdmissionStatus.ACCEPTED,
            TaskAdmissionStatus.SPLIT,
            TaskAdmissionStatus.COALESCED,
        }

    @property
    def rejection_reasons(self) -> tuple[str, ...]:
        return tuple(item.reason for item in self.rejections)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "score": self.score.to_dict(),
            "status": self.status.value,
            "accepted": self.accepted,
            "rejections": [item.to_dict() for item in self.rejections],
            "rejection_reasons": list(self.rejection_reasons),
            "source_identities": list(self.source_identities),
        }


@dataclass(frozen=True)
class TaskAdmissionResult:
    """Stable output of one split/coalesce/dedupe/admission cycle."""

    decisions: tuple[TaskAdmissionDecision, ...]
    initial_open_work: int
    max_open_work: int
    candidate_count: int
    granularity_calibration_id: str = ""

    @property
    def accepted(self) -> tuple[TaskCandidate, ...]:
        return tuple(item.candidate for item in self.decisions if item.accepted)

    @property
    def rejected(self) -> tuple[TaskAdmissionDecision, ...]:
        return tuple(item for item in self.decisions if not item.accepted)

    @property
    def final_open_work(self) -> int:
        return self.initial_open_work + len(self.accepted)

    @property
    def final_open_work_count(self) -> int:
        return self.final_open_work

    @property
    def bounded(self) -> bool:
        return self.final_open_work <= self.max_open_work

    def to_dict(self) -> dict[str, Any]:
        rejection_counts: dict[str, int] = {}
        for decision in self.rejected:
            for reason in decision.rejection_reasons:
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
        result = {
            "schema": TASK_QUALITY_SCHEMA,
            "evaluator_version": TASK_QUALITY_EVALUATOR_VERSION,
            "decisions": [item.to_dict() for item in self.decisions],
            "accepted": [item.to_dict() for item in self.accepted],
            "rejected": [item.to_dict() for item in self.rejected],
            "rejection_counts": dict(sorted(rejection_counts.items())),
            "candidate_count": self.candidate_count,
            "initial_open_work": self.initial_open_work,
            "final_open_work": self.final_open_work,
            "max_open_work": self.max_open_work,
            "bounded": self.bounded,
        }
        if self.granularity_calibration_id:
            result["granularity_calibration_id"] = self.granularity_calibration_id
        return result


def _semantic_tokens(candidate: TaskCandidate) -> set[str]:
    values = [
        candidate.goal_id,
        *candidate.acceptance,
        *candidate.effects,
        *candidate.evidence_subset,
        *candidate.outputs,
        *candidate.predicted_symbols,
    ]
    return set(re.findall(r"[a-z0-9]+", " ".join(values).casefold()))


def task_semantic_similarity(left: TaskCandidate, right: TaskCandidate) -> float:
    """Return a conservative semantic Jaccard similarity in ``[0, 1]``."""

    if left.semantic_identity == right.semantic_identity:
        return 1.0
    if (
        left.goal_id
        and right.goal_id
        and _normalized_text(left.goal_id) != _normalized_text(right.goal_id)
    ):
        return 0.0
    left_tokens = _semantic_tokens(left)
    right_tokens = _semantic_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    token_similarity = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
    left_outputs, right_outputs = set(left.outputs), set(right.outputs)
    output_similarity = (
        len(left_outputs & right_outputs) / len(left_outputs | right_outputs)
        if left_outputs and right_outputs
        else 0.0
    )
    # Requiring both semantic and output overlap avoids treating tasks that
    # merely share a broad goal as duplicates.
    return 0.75 * token_similarity + 0.25 * output_similarity


def _history(
    values: Iterable[Any],
    *,
    default_outcome: str,
) -> tuple[HistoricalTask, ...]:
    return tuple(
        HistoricalTask.from_value(value, default_outcome=default_outcome)
        for value in values
    )


def _bounded_cost(value: int, limit: int) -> float:
    if value <= 0:
        return 1.0
    if limit <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - value / (limit * 1.25)))


def _acceptance_coverage(candidate: TaskCandidate) -> float:
    if not candidate.acceptance:
        return 0.0
    dimensions = [
        bool(candidate.evidence_subset),
        bool(candidate.validation_commands),
        bool(candidate.effects),
        bool(candidate.outputs or candidate.predicted_paths),
    ]
    return 0.5 + 0.5 * sum(dimensions) / len(dimensions)


def _effect_coherence(candidate: TaskCandidate) -> float:
    if not candidate.effects:
        return 0.0
    effect_tokens = set(
        re.findall(r"[a-z0-9]+", " ".join(candidate.effects).casefold())
    )
    surface_tokens = set(
        re.findall(
            r"[a-z0-9]+",
            " ".join(
                [
                    *candidate.acceptance,
                    *candidate.outputs,
                    *candidate.predicted_paths,
                    *candidate.predicted_symbols,
                ]
            ).casefold(),
        )
    )
    if not surface_tokens:
        return 0.5
    return 0.5 + 0.5 * min(1.0, len(effect_tokens & surface_tokens) / 3)


def _breadth_fit(candidate: TaskCandidate, policy: TaskQualityPolicy) -> float:
    paths = candidate.predicted_path_breadth
    symbols = candidate.predicted_symbol_breadth
    path_fit = 1.0 if paths else 0.25
    symbol_fit = 1.0
    if paths > policy.max_predicted_paths:
        path_fit = max(0.0, policy.max_predicted_paths / paths)
    if symbols > policy.max_predicted_symbols:
        symbol_fit = max(0.0, policy.max_predicted_symbols / symbols)
    return (path_fit + symbol_fit) / 2


def _resource_fit(candidate: TaskCandidate) -> float:
    if not candidate.resource_class or not candidate.token_class:
        return 0.0
    if (
        candidate.resource_class.casefold() not in RESOURCE_CLASSES
        or candidate.token_class.casefold() not in TOKEN_CLASSES
    ):
        return 0.0
    token_limit = TOKEN_CLASS_LIMITS[candidate.token_class.casefold()]
    if candidate.estimated_tokens and candidate.estimated_tokens > token_limit:
        return max(0.0, token_limit / candidate.estimated_tokens)
    return 1.0


def score_task_candidate(
    candidate: TaskCandidate | Mapping[str, Any],
    *,
    policy: TaskQualityPolicy | None = None,
    historical_tasks: Iterable[Any] = (),
    historical_failures: Iterable[Any] = (),
    calibration: TaskGranularityCalibration | None = None,
) -> TaskQualityScore:
    """Score all acceptance dimensions without mutating scheduler state."""

    item = (
        candidate if isinstance(candidate, TaskCandidate) else TaskCandidate.from_mapping(candidate)
    )
    source_policy = policy or TaskQualityPolicy()
    selected = _effective_granularity_policy(source_policy, calibration)
    tasks = _history(historical_tasks, default_outcome="accepted")
    failures = _history(historical_failures, default_outcome="failed")
    duplicate_similarity = max(
        item.historical_duplicate_similarity,
        max(
            (task_semantic_similarity(item, prior.candidate) for prior in tasks),
            default=0.0,
        ),
    )
    failure_similarity = max(
        item.historical_failure_similarity,
        max(
            (task_semantic_similarity(item, prior.candidate) for prior in failures),
            default=0.0,
        ),
    )
    dimensions = {
        "acceptance_coverage": _acceptance_coverage(item),
        "coherent_effects": _effect_coherence(item),
        "breadth_fit": _breadth_fit(item, selected),
        "context_cost": (
            _bounded_cost(len(item.context_paths), selected.max_context_paths)
            + _bounded_cost(item.estimated_context_tokens, selected.max_context_tokens)
        )
        / 2,
        "validation_cost": _bounded_cost(
            item.estimated_validation_seconds,
            selected.max_validation_seconds,
        ),
        "dependency_cost": _bounded_cost(
            len(item.dependencies),
            selected.max_dependencies,
        ),
        "conflict_cost": _bounded_cost(len(item.conflicts), selected.max_conflicts),
        "resource_fit": _resource_fit(item),
        "historical_novelty": 1.0 - duplicate_similarity,
        "historical_failure_safety": 1.0 - failure_similarity,
    }
    weights = {
        "acceptance_coverage": 0.16,
        "coherent_effects": 0.14,
        "breadth_fit": 0.12,
        "context_cost": 0.09,
        "validation_cost": 0.09,
        "dependency_cost": 0.08,
        "conflict_cost": 0.08,
        "resource_fit": 0.08,
        "historical_novelty": 0.10,
        "historical_failure_safety": 0.06,
    }
    total = sum(dimensions[name] * weights[name] for name in weights)
    rationale = tuple(
        f"{name}={dimensions[name]:.6f} weight={weights[name]:.6f}"
        for name in weights
    )
    raw_score = TaskQualityScore(
        **dimensions,
        duplicate_similarity=duplicate_similarity,
        failure_similarity=failure_similarity,
        total=max(0.0, min(1.0, total)),
        rationale=rationale,
    )
    return replace(
        raw_score,
        rejection_reasons=_task_rejections(item, raw_score, selected),
    )


def _task_rejections(
    candidate: TaskCandidate,
    score: TaskQualityScore,
    policy: TaskQualityPolicy,
) -> tuple[TaskRejection, ...]:
    identity = candidate.semantic_identity
    reasons: list[TaskRejection] = []

    def reject(reason: str, detail: str) -> None:
        reasons.append(TaskRejection(reason, identity, detail))

    required = (
        ("missing_goal_id", candidate.goal_id, "goal_id is required"),
        ("missing_acceptance", candidate.acceptance, "acceptance coverage is required"),
        ("missing_preconditions", candidate.preconditions, "preconditions are required"),
        ("missing_effects", candidate.effects, "effects are required"),
        (
            "missing_evidence_subset",
            candidate.evidence_subset,
            "an evidence subset is required",
        ),
        (
            "missing_outputs",
            candidate.outputs or candidate.predicted_paths,
            "at least one output or predicted path is required",
        ),
        (
            "missing_validation",
            candidate.validation_commands,
            "validation commands are required",
        ),
        (
            "missing_estimated_context_tokens",
            candidate.estimated_context_tokens,
            "a positive estimated_context_tokens cost binding is required",
        ),
        (
            "missing_estimated_validation_seconds",
            candidate.estimated_validation_seconds,
            "a positive estimated_validation_seconds cost binding is required",
        ),
        (
            "missing_estimated_tokens",
            candidate.estimated_tokens,
            "a positive estimated_tokens cost binding is required",
        ),
        (
            "missing_resource_class",
            candidate.resource_class,
            "resource_class is required",
        ),
        ("missing_token_class", candidate.token_class, "token_class is required"),
        ("missing_merge_fate", candidate.merge_fate, "merge_fate is required"),
    )
    for reason, value, detail in required:
        if not value:
            reject(reason, detail)
    if (
        candidate.resource_class
        and candidate.resource_class.casefold() not in RESOURCE_CLASSES
    ):
        reject(
            "invalid_resource_class",
            f"unsupported resource class {candidate.resource_class!r}",
        )
    if candidate.token_class and candidate.token_class.casefold() not in TOKEN_CLASSES:
        reject("invalid_token_class", f"unsupported token class {candidate.token_class!r}")
    if candidate.predicted_path_breadth > policy.max_predicted_paths:
        reject(
            "predicted_path_breadth",
            f"{candidate.predicted_path_breadth} paths exceed {policy.max_predicted_paths}",
        )
    if candidate.predicted_symbol_breadth > policy.max_predicted_symbols:
        reject(
            "predicted_symbol_breadth",
            f"{candidate.predicted_symbol_breadth} symbols exceed {policy.max_predicted_symbols}",
        )
    if candidate.predicted_interface_breadth > policy.max_predicted_interfaces:
        reject(
            "predicted_interface_breadth",
            f"{candidate.predicted_interface_breadth} interfaces exceed "
            f"{policy.max_predicted_interfaces}",
        )
    if len(candidate.acceptance) > policy.max_acceptance_criteria:
        reject(
            "acceptance_breadth",
            f"{len(candidate.acceptance)} criteria exceed {policy.max_acceptance_criteria}",
        )
    if len(candidate.effects) > policy.max_effects:
        reject(
            "effect_breadth",
            f"{len(candidate.effects)} effects exceed {policy.max_effects}",
        )
    if len(candidate.evidence_subset) > policy.max_evidence_items:
        reject(
            "evidence_breadth",
            f"{len(candidate.evidence_subset)} evidence items exceed {policy.max_evidence_items}",
        )
    if len(candidate.context_paths) > policy.max_context_paths:
        reject(
            "context_path_cost",
            f"{len(candidate.context_paths)} context paths exceed {policy.max_context_paths}",
        )
    if candidate.estimated_context_tokens > policy.max_context_tokens:
        reject(
            "context_token_cost",
            f"{candidate.estimated_context_tokens} exceeds {policy.max_context_tokens}",
        )
    if candidate.estimated_validation_seconds > policy.max_validation_seconds:
        reject(
            "validation_cost",
            f"{candidate.estimated_validation_seconds}s exceeds {policy.max_validation_seconds}s",
        )
    if candidate.proof_obligations and not candidate.proof_commands:
        reject(
            "missing_proof_validation",
            "declared proof obligations require proof commands",
        )
    if candidate.proof_commands and not candidate.proof_obligations:
        reject(
            "missing_proof_obligations",
            "declared proof commands require proof obligations",
        )
    if (
        candidate.proof_obligations or candidate.proof_commands
    ) and candidate.estimated_proof_seconds <= 0:
        reject(
            "missing_estimated_proof_seconds",
            "declared proof obligations require a positive proof cost",
        )
    if len(candidate.proof_obligations) > policy.max_proof_items:
        reject(
            "proof_breadth",
            f"{len(candidate.proof_obligations)} proof obligations exceed "
            f"{policy.max_proof_items}",
        )
    if candidate.estimated_proof_seconds > policy.max_proof_seconds:
        reject(
            "proof_cost",
            f"{candidate.estimated_proof_seconds}s exceeds "
            f"{policy.max_proof_seconds}s",
        )
    if candidate.estimated_tokens > policy.max_estimated_tokens:
        reject(
            "task_token_cost",
            f"{candidate.estimated_tokens} exceeds {policy.max_estimated_tokens}",
        )
    if len(candidate.dependencies) > policy.max_dependencies:
        reject(
            "dependency_cost",
            f"{len(candidate.dependencies)} dependencies exceed {policy.max_dependencies}",
        )
    if len(candidate.conflicts) > policy.max_conflicts:
        reject(
            "conflict_cost",
            f"{len(candidate.conflicts)} conflicts exceed {policy.max_conflicts}",
        )
    if (
        candidate.estimated_merge_risk_millionths
        > policy.max_merge_risk_millionths
    ):
        reject(
            "merge_risk",
            f"{candidate.estimated_merge_risk_millionths} exceeds "
            f"{policy.max_merge_risk_millionths} millionths",
        )
    if score.duplicate_similarity >= policy.duplicate_similarity_threshold:
        reject(
            "historical_duplicate",
            f"similarity {score.duplicate_similarity:.6f} reaches "
            f"{policy.duplicate_similarity_threshold:.6f}",
        )
    if score.failure_similarity >= policy.failure_similarity_threshold:
        reject(
            "historical_failure",
            f"similarity {score.failure_similarity:.6f} reaches "
            f"{policy.failure_similarity_threshold:.6f}",
        )
    if score.total < policy.min_quality_score:
        reject(
            "quality_below_threshold",
            f"score {score.total:.6f} is below {policy.min_quality_score:.6f}",
        )
    return tuple(reasons)


def admit_task_candidate(
    candidate: TaskCandidate | Mapping[str, Any],
    *,
    policy: TaskQualityPolicy | None = None,
    historical_tasks: Iterable[Any] = (),
    historical_failures: Iterable[Any] = (),
    calibration: TaskGranularityCalibration | None = None,
) -> TaskAdmissionDecision:
    item = (
        candidate if isinstance(candidate, TaskCandidate) else TaskCandidate.from_mapping(candidate)
    )
    source_policy = policy or TaskQualityPolicy()
    selected = _effective_granularity_policy(source_policy, calibration)
    score = score_task_candidate(
        item,
        policy=selected,
        historical_tasks=historical_tasks,
        historical_failures=historical_failures,
    )
    rejections = _task_rejections(item, score, selected)
    return TaskAdmissionDecision(
        candidate=item,
        score=score,
        status=(
            TaskAdmissionStatus.REJECTED
            if rejections
            else TaskAdmissionStatus.ACCEPTED
        ),
        rejections=rejections,
        source_identities=(item.semantic_identity,),
    )


def is_over_broad(
    candidate: TaskCandidate,
    policy: TaskQualityPolicy | None = None,
    *,
    calibration: TaskGranularityCalibration | None = None,
) -> bool:
    source_policy = policy or TaskQualityPolicy()
    selected = _effective_granularity_policy(source_policy, calibration)
    return (
        candidate.predicted_path_breadth > selected.max_predicted_paths
        or candidate.predicted_symbol_breadth > selected.max_predicted_symbols
        or candidate.predicted_interface_breadth
        > selected.max_predicted_interfaces
        or len(candidate.acceptance) > selected.max_acceptance_criteria
        or len(candidate.effects) > selected.max_effects
        or len(candidate.evidence_subset) > selected.max_evidence_items
        or len(candidate.context_paths) > selected.max_context_paths
        or candidate.estimated_context_tokens > selected.max_context_tokens
        or candidate.estimated_validation_seconds
        > selected.max_validation_seconds
        or len(candidate.proof_obligations) > selected.max_proof_items
        or candidate.estimated_proof_seconds > selected.max_proof_seconds
        or candidate.estimated_tokens > selected.max_estimated_tokens
        or candidate.estimated_merge_risk_millionths
        > selected.max_merge_risk_millionths
    )


def is_tiny(
    candidate: TaskCandidate,
    policy: TaskQualityPolicy | None = None,
    *,
    calibration: TaskGranularityCalibration | None = None,
) -> bool:
    source_policy = policy or TaskQualityPolicy()
    selected = _effective_granularity_policy(source_policy, calibration)
    return (
        candidate.predicted_path_breadth <= selected.tiny_max_paths
        and candidate.predicted_symbol_breadth <= selected.tiny_max_symbols
        and candidate.estimated_tokens <= selected.tiny_max_tokens
    )


def split_task_candidate(
    candidate: TaskCandidate | Mapping[str, Any],
    policy: TaskQualityPolicy | None = None,
    *,
    max_paths: int | None = None,
    max_symbols: int | None = None,
    calibration: TaskGranularityCalibration | None = None,
) -> tuple[TaskCandidate, ...]:
    """Split over-broad work deterministically while preserving dependencies.

    Every child retains the acceptance/proof contract, validation, goal, merge
    fate, and external prerequisites.  Disjoint children remain independent;
    their source identity is carried by the admission decision instead of
    inventing dependency edges which would collapse critical-path width.
    """

    item = (
        candidate if isinstance(candidate, TaskCandidate) else TaskCandidate.from_mapping(candidate)
    )
    source_policy = policy or TaskQualityPolicy()
    selected = _effective_granularity_policy(source_policy, calibration)
    path_limit = max(1, int(max_paths or selected.max_predicted_paths))
    symbol_limit = max(1, int(max_symbols or selected.max_predicted_symbols))
    paths = tuple(sorted(set(item.outputs) | set(item.predicted_paths)))
    symbols = item.predicted_symbols
    required_parts = max(
        1,
        math.ceil(len(paths) / path_limit),
        math.ceil(len(symbols) / symbol_limit),
        math.ceil(
            len(item.predicted_interfaces) / selected.max_predicted_interfaces
        ),
        math.ceil(len(item.acceptance) / selected.max_acceptance_criteria),
        math.ceil(len(item.effects) / selected.max_effects),
        math.ceil(len(item.evidence_subset) / selected.max_evidence_items),
        (
            math.ceil(len(item.context_paths) / selected.max_context_paths)
            if selected.max_context_paths
            else 1
        ),
        (
            math.ceil(item.estimated_context_tokens / selected.max_context_tokens)
            if selected.max_context_tokens
            else 1
        ),
        (
            math.ceil(
                item.estimated_validation_seconds
                / selected.max_validation_seconds
            )
            if selected.max_validation_seconds
            else (
                selected.max_split_parts + 1
                if item.estimated_validation_seconds
                else 1
            )
        ),
        math.ceil(len(item.proof_obligations) / selected.max_proof_items),
        (
            math.ceil(item.estimated_proof_seconds / selected.max_proof_seconds)
            if selected.max_proof_seconds
            else (
                selected.max_split_parts + 1
                if item.estimated_proof_seconds
                else 1
            )
        ),
        (
            math.ceil(item.estimated_tokens / selected.max_estimated_tokens)
            if selected.max_estimated_tokens
            else 1
        ),
        (
            math.ceil(
                item.estimated_merge_risk_millionths
                / selected.max_merge_risk_millionths
            )
            if selected.max_merge_risk_millionths
            else (
                selected.max_split_parts + 1
                if item.estimated_merge_risk_millionths
                else 1
            )
        ),
    )
    part_count = min(required_parts, selected.max_split_parts)
    if part_count == 1:
        return (item,)

    path_chunks = [paths[index::part_count] for index in range(part_count)]
    symbol_chunks = [symbols[index::part_count] for index in range(part_count)]
    interface_chunks = [
        item.predicted_interfaces[index::part_count]
        for index in range(part_count)
    ]
    acceptance_chunks = [
        item.acceptance[index::part_count] for index in range(part_count)
    ]
    effect_chunks = [item.effects[index::part_count] for index in range(part_count)]
    evidence_chunks = [
        item.evidence_subset[index::part_count] for index in range(part_count)
    ]
    proof_chunks = [
        item.proof_obligations[index::part_count]
        for index in range(part_count)
    ]
    context_chunks: list[list[str]] = [[] for _ in range(part_count)]
    path_part = {
        path: index
        for index, chunk in enumerate(path_chunks)
        for path in chunk
    }
    residual_context_index = 0
    for path in item.context_paths:
        target = path_part.get(path)
        if target is None:
            target = residual_context_index % part_count
            residual_context_index += 1
        context_chunks[target].append(path)
    children: list[TaskCandidate] = []

    def partitioned(total: int, index: int) -> int:
        quotient, remainder = divmod(total, part_count)
        return quotient + (1 if index < remainder else 0)

    split_validation_cost = (
        item.estimated_validation_seconds > selected.max_validation_seconds
    )
    split_proof_cost = item.estimated_proof_seconds > selected.max_proof_seconds
    split_merge_risk = (
        item.estimated_merge_risk_millionths
        > selected.max_merge_risk_millionths
    )
    for index in range(part_count):
        def exact_subset(
            chunk: Sequence[str],
            source: Sequence[str],
        ) -> tuple[str, ...]:
            if chunk or not source:
                return tuple(chunk)
            return (source[index % len(source)],)

        child_paths = exact_subset(path_chunks[index], paths)
        outputs = tuple(path for path in item.outputs if path in child_paths)
        if not outputs and child_paths:
            outputs = child_paths
        # A task may legitimately need context which is not itself an output.
        # Partition residual context while keeping exact output context with
        # its owner; copying whole directories into every child would leave
        # each split over-broad and collapse independent execution width.
        contexts = _strings(
            (
                *context_chunks[index],
                *(path for path in child_paths if path in item.context_paths),
            ),
            paths=True,
        )
        child_metadata = dict(item.metadata)
        child_metadata["granularity_shard"] = (
            f"{item.semantic_identity}:{index + 1}/{part_count}"
        )
        child = replace(
            item,
            title=f"{item.title} [{index + 1}/{part_count}]",
            outputs=outputs,
            predicted_paths=child_paths,
            predicted_symbols=exact_subset(
                symbol_chunks[index], item.predicted_symbols
            ),
            predicted_interfaces=exact_subset(
                interface_chunks[index], item.predicted_interfaces
            ),
            acceptance=exact_subset(
                acceptance_chunks[index], item.acceptance
            ),
            effects=exact_subset(effect_chunks[index], item.effects),
            evidence_subset=exact_subset(
                evidence_chunks[index], item.evidence_subset
            ),
            proof_obligations=exact_subset(
                proof_chunks[index], item.proof_obligations
            ),
            dependencies=item.dependencies,
            context_paths=contexts,
            estimated_context_tokens=partitioned(
                item.estimated_context_tokens, index
            ),
            estimated_validation_seconds=(
                partitioned(item.estimated_validation_seconds, index)
                if split_validation_cost
                else item.estimated_validation_seconds
            ),
            estimated_proof_seconds=(
                partitioned(item.estimated_proof_seconds, index)
                if split_proof_cost
                else item.estimated_proof_seconds
            ),
            estimated_tokens=partitioned(item.estimated_tokens, index),
            estimated_merge_risk_millionths=(
                partitioned(item.estimated_merge_risk_millionths, index)
                if split_merge_risk
                else item.estimated_merge_risk_millionths
            ),
            source_id=f"{item.source_id}:split:{index + 1}" if item.source_id else "",
            semantic_identity="",
            metadata=child_metadata,
        )
        children.append(child)
    return tuple(children)


def can_coalesce_tasks(
    left: TaskCandidate | Mapping[str, Any],
    right: TaskCandidate | Mapping[str, Any],
    *,
    policy: TaskQualityPolicy | None = None,
    calibration: TaskGranularityCalibration | None = None,
) -> bool:
    """Return whether tiny tasks share every required merge-fate boundary."""

    first = left if isinstance(left, TaskCandidate) else TaskCandidate.from_mapping(left)
    second = right if isinstance(right, TaskCandidate) else TaskCandidate.from_mapping(right)
    source_policy = policy or TaskQualityPolicy()
    selected = _effective_granularity_policy(source_policy, calibration)
    compatible = (
        first.semantic_identity != second.semantic_identity
        and not first.metadata.get("granularity_shard")
        and not second.metadata.get("granularity_shard")
        and is_tiny(first, selected)
        and is_tiny(second, selected)
        and _normalized_text(first.goal_id) == _normalized_text(second.goal_id)
        and first.context_paths == second.context_paths
        and first.outputs == second.outputs
        and first.validation_commands == second.validation_commands
        and first.proof_commands == second.proof_commands
        and _normalized_text(first.merge_fate) == _normalized_text(second.merge_fate)
        and first.resource_class.casefold() == second.resource_class.casefold()
        and first.token_class.casefold() == second.token_class.casefold()
    )
    if not compatible:
        return False
    return (
        len(set(first.acceptance) | set(second.acceptance))
        <= selected.max_acceptance_criteria
        and len(set(first.effects) | set(second.effects)) <= selected.max_effects
        and len(set(first.evidence_subset) | set(second.evidence_subset))
        <= selected.max_evidence_items
        and len(set(first.predicted_paths) | set(second.predicted_paths))
        <= selected.max_predicted_paths
        and len(set(first.predicted_symbols) | set(second.predicted_symbols))
        <= selected.max_predicted_symbols
        and len(
            set(first.predicted_interfaces) | set(second.predicted_interfaces)
        )
        <= selected.max_predicted_interfaces
        and len(set(first.proof_obligations) | set(second.proof_obligations))
        <= selected.max_proof_items
        and first.estimated_tokens + second.estimated_tokens
        <= selected.max_estimated_tokens
        and max(
            first.estimated_context_tokens,
            second.estimated_context_tokens,
        )
        <= selected.max_context_tokens
        and max(
            first.estimated_validation_seconds,
            second.estimated_validation_seconds,
        )
        <= selected.max_validation_seconds
        and max(
            first.estimated_proof_seconds,
            second.estimated_proof_seconds,
        )
        <= selected.max_proof_seconds
        and first.estimated_merge_risk_millionths
        + second.estimated_merge_risk_millionths
        <= selected.max_merge_risk_millionths
    )


def coalesce_task_candidates(
    candidates: Iterable[TaskCandidate | Mapping[str, Any]],
    *,
    policy: TaskQualityPolicy | None = None,
    calibration: TaskGranularityCalibration | None = None,
) -> TaskCandidate:
    """Coalesce compatible tiny candidates into one task or fail closed."""

    items = tuple(sorted((
        item if isinstance(item, TaskCandidate) else TaskCandidate.from_mapping(item)
        for item in candidates
    ), key=lambda item: item.semantic_identity))
    if not items:
        raise ValueError("at least one task candidate is required")
    source_policy = policy or TaskQualityPolicy()
    selected = _effective_granularity_policy(source_policy, calibration)
    anchor = items[0]
    if any(
        not can_coalesce_tasks(anchor, item, policy=selected)
        for item in items[1:]
    ):
        raise ValueError(
            "tiny candidates may coalesce only with shared goal, context, outputs, "
            "validation, resource/token class, and merge fate"
        )

    def union(name: str) -> tuple[str, ...]:
        return _strings(item for value in items for item in getattr(value, name))

    merged = replace(
        anchor,
        title="; ".join(item.title for item in items if item.title),
        acceptance=union("acceptance"),
        preconditions=union("preconditions"),
        effects=union("effects"),
        evidence_subset=union("evidence_subset"),
        predicted_paths=_strings(
            (
                path
                for item in items
                for path in item.predicted_paths
            ),
            paths=True,
        ),
        predicted_symbols=union("predicted_symbols"),
        predicted_interfaces=union("predicted_interfaces"),
        proof_obligations=union("proof_obligations"),
        dependencies=union("dependencies"),
        conflicts=union("conflicts"),
        resources=union("resources"),
        estimated_context_tokens=max(item.estimated_context_tokens for item in items),
        estimated_validation_seconds=max(
            item.estimated_validation_seconds for item in items
        ),
        estimated_proof_seconds=max(
            item.estimated_proof_seconds for item in items
        ),
        estimated_tokens=sum(item.estimated_tokens for item in items),
        estimated_merge_risk_millionths=sum(
            item.estimated_merge_risk_millionths for item in items
        ),
        source_id=",".join(item.source_id for item in items if item.source_id),
        semantic_identity="",
    )
    if is_over_broad(merged, selected):
        raise ValueError("coalesced candidate exceeds a task granularity bound")
    return merged


def _coalesce_tiny_groups(
    candidates: Sequence[TaskCandidate],
    policy: TaskQualityPolicy,
    calibration: TaskGranularityCalibration | None = None,
) -> tuple[tuple[TaskCandidate, tuple[str, ...]], ...]:
    remaining = list(sorted(candidates, key=lambda item: item.semantic_identity))
    result: list[tuple[TaskCandidate, tuple[str, ...]]] = []
    while remaining:
        anchor = remaining.pop(0)
        group = [anchor]
        for item in remaining:
            try:
                coalesce_task_candidates(
                    (*group, item),
                    policy=policy,
                    calibration=calibration,
                )
            except ValueError:
                continue
            group.append(item)
        compatible = group[1:]
        if not compatible:
            result.append((anchor, (anchor.semantic_identity,)))
            continue
        compatible_ids = {item.semantic_identity for item in compatible}
        remaining = [
            item for item in remaining if item.semantic_identity not in compatible_ids
        ]
        merged = coalesce_task_candidates(
            group,
            policy=policy,
            calibration=calibration,
        )
        result.append((merged, tuple(item.semantic_identity for item in group)))
    return tuple(result)


def _granularity_source_identity(candidate: TaskCandidate) -> str:
    shard = _normalized_display_text(
        candidate.metadata.get("granularity_shard")
    )
    return shard.rpartition(":")[0] if shard else ""


def refine_task_candidates(
    candidates: Iterable[TaskCandidate | Mapping[str, Any]],
    *,
    policy: TaskQualityPolicy | None = None,
    existing_tasks: Iterable[Any] = (),
    existing: Iterable[Any] | None = None,
    historical_tasks: Iterable[Any] = (),
    historical_failures: Iterable[Any] = (),
    current_open_work: int = 0,
    open_work_count: int | None = None,
    max_open_work: int | None = None,
    calibration: TaskGranularityCalibration | None = None,
    cost_measurements: Iterable[TaskCostMeasurement | Mapping[str, Any]] = (),
    repository_tree: str = "",
    toolchain_features: Iterable[str] = (),
) -> TaskAdmissionResult:
    """Resize, semantically deduplicate, score, and pressure-bound candidates."""

    source_policy = policy or TaskQualityPolicy()
    measurement_history = tuple(cost_measurements)
    requested_features = _strings(toolchain_features)
    if calibration is not None and measurement_history:
        raise ValueError("pass either calibration or cost_measurements, not both")
    if measurement_history:
        calibration = calibrate_task_granularity(
            measurement_history,
            repository_tree=repository_tree,
            policy=source_policy,
            toolchain_features=requested_features,
        )
    if calibration is not None and (
        not calibration.applies_to(
            source_policy,
            repository_tree=(repository_tree or None),
            toolchain_features=(requested_features or None),
        )
    ):
        raise ValueError(
            "task granularity calibration does not match requested execution features"
        )
    selected = _effective_granularity_policy(source_policy, calibration)
    provided_existing = tuple(existing_tasks)
    if existing is not None:
        if provided_existing:
            raise ValueError("pass either existing_tasks or existing, not both")
        provided_existing = tuple(existing)
    if open_work_count is not None:
        if current_open_work:
            raise ValueError(
                "pass either current_open_work or open_work_count, not both"
            )
        current_open_work = open_work_count
    open_count = _non_negative_int(current_open_work, "current_open_work")
    open_limit = (
        selected.max_open_work
        if max_open_work is None
        else _non_negative_int(max_open_work, "max_open_work")
    )
    raw = tuple(
        item if isinstance(item, TaskCandidate) else TaskCandidate.from_mapping(item)
        for item in candidates
    )
    historical = _history(historical_tasks, default_outcome="accepted")
    existing_history = _history(provided_existing, default_outcome="open")
    failures = _history(historical_failures, default_outcome="failed")
    known = tuple(item.candidate for item in (*existing_history, *historical))

    resized: list[TaskCandidate] = []
    source_map: dict[str, tuple[str, ...]] = {}
    for candidate in sorted(raw, key=lambda item: item.semantic_identity):
        pieces = (
            split_task_candidate(candidate, policy=selected)
            if selected.split_over_broad and is_over_broad(candidate, selected)
            else (candidate,)
        )
        for piece in pieces:
            resized.append(piece)
            source_map[piece.semantic_identity] = (candidate.semantic_identity,)

    grouped = (
        _coalesce_tiny_groups(resized, selected)
        if selected.coalesce_tiny
        else tuple((item, (item.semantic_identity,)) for item in resized)
    )
    candidates_with_sources = sorted(grouped, key=lambda pair: pair[0].semantic_identity)
    decisions: list[TaskAdmissionDecision] = []
    observed_identities = {item.semantic_identity for item in known}
    observed_candidates = list(known)
    capacity = min(selected.max_new_work, max(0, open_limit - open_count))
    accepted_count = 0

    for candidate, coalesced_sources in candidates_with_sources:
        source_identities = tuple(
            source
            for identity in coalesced_sources
            for source in source_map.get(identity, (identity,))
        )
        granularity_source = _granularity_source_identity(candidate)
        comparable_history = (
            prior
            for prior in observed_candidates
            if not (
                granularity_source
                and _granularity_source_identity(prior) == granularity_source
            )
        )
        score = score_task_candidate(
            candidate,
            policy=selected,
            historical_tasks=comparable_history,
            historical_failures=(item.candidate for item in failures),
        )
        rejections = list(_task_rejections(candidate, score, selected))
        if candidate.semantic_identity in observed_identities:
            rejections.append(
                TaskRejection(
                    "duplicate_semantic_identity",
                    candidate.semantic_identity,
                    "canonical semantic identity already exists",
                )
            )
        if not rejections and accepted_count >= capacity:
            rejections.append(
                TaskRejection(
                    "open_work_limit",
                    candidate.semantic_identity,
                    f"cycle has {capacity} available open-work slots",
                )
            )
        if rejections:
            decisions.append(
                TaskAdmissionDecision(
                    candidate=candidate,
                    score=score,
                    status=TaskAdmissionStatus.REJECTED,
                    rejections=tuple(rejections),
                    source_identities=source_identities,
                )
            )
            observed_identities.add(candidate.semantic_identity)
            observed_candidates.append(candidate)
            continue
        status = TaskAdmissionStatus.ACCEPTED
        if len(source_identities) > 1:
            status = TaskAdmissionStatus.COALESCED
        elif source_identities and source_identities[0] != candidate.semantic_identity:
            status = TaskAdmissionStatus.SPLIT
        decisions.append(
            TaskAdmissionDecision(
                candidate=candidate,
                score=score,
                status=status,
                source_identities=source_identities,
            )
        )
        accepted_count += 1
        observed_identities.add(candidate.semantic_identity)
        observed_candidates.append(candidate)

    return TaskAdmissionResult(
        decisions=tuple(decisions),
        initial_open_work=open_count,
        max_open_work=open_limit,
        candidate_count=len(raw),
        granularity_calibration_id=(
            calibration.calibration_id if calibration is not None else ""
        ),
    )


def _resolve_completed_task_identities(
    tasks: Sequence[TaskCandidate],
    completed: Iterable[str],
) -> tuple[str, ...]:
    aliases: dict[str, str] = {}
    for task in tasks:
        for alias in (
            task.semantic_identity,
            task.canonical_task_key,
            task.canonical_task_cid,
        ):
            existing = aliases.get(alias)
            if existing is not None and existing != task.semantic_identity:
                raise ValueError("task completion alias is ambiguous")
            aliases[alias] = task.semantic_identity
    resolved: set[str] = set()
    unknown: list[str] = []
    for value in completed:
        identity = aliases.get(str(value).strip())
        if identity is None:
            unknown.append(str(value))
        else:
            resolved.add(identity)
    if unknown:
        raise ValueError(
            "completion names unknown or unaccepted tasks: "
            + ", ".join(sorted(unknown))
        )
    return tuple(sorted(resolved))


@dataclass(frozen=True)
class TaskCompletionPropagation:
    """Exact descendant-to-source completion projection."""

    completed_task_identities: tuple[str, ...]
    completed_task_cids: tuple[str, ...]
    completed_source_identities: tuple[str, ...]
    incomplete_source_identities: tuple[str, ...]
    completed_acceptance: tuple[str, ...]
    propagation_id: str = ""

    def __post_init__(self) -> None:
        for name in (
            "completed_task_identities",
            "completed_task_cids",
            "completed_source_identities",
            "incomplete_source_identities",
            "completed_acceptance",
        ):
            object.__setattr__(self, name, _strings(getattr(self, name)))
        if set(self.completed_source_identities) & set(
            self.incomplete_source_identities
        ):
            raise ValueError("source completion populations must be disjoint")
        expected = _task_quality_evidence_cid(self._material())
        supplied = str(self.propagation_id or "").strip()
        if supplied and supplied != expected:
            raise ValueError("completion propagation identity mismatch")
        object.__setattr__(self, "propagation_id", expected)

    def _material(self) -> dict[str, Any]:
        return {
            "schema": TASK_COMPLETION_PROPAGATION_SCHEMA,
            "completed_task_identities": list(self.completed_task_identities),
            "completed_task_cids": list(self.completed_task_cids),
            "completed_source_identities": list(self.completed_source_identities),
            "incomplete_source_identities": list(
                self.incomplete_source_identities
            ),
            "completed_acceptance": list(self.completed_acceptance),
        }

    def to_dict(self) -> dict[str, Any]:
        result = self._material()
        result["propagation_id"] = self.propagation_id
        return result


def propagate_task_completion(
    result: TaskAdmissionResult,
    completed_tasks: Iterable[str],
) -> TaskCompletionPropagation:
    """Propagate completion only through explicit admission source bindings."""

    accepted_decisions = tuple(
        decision for decision in result.decisions if decision.accepted
    )
    accepted_tasks = tuple(decision.candidate for decision in accepted_decisions)
    completed = set(
        _resolve_completed_task_identities(accepted_tasks, completed_tasks)
    )
    completed_cids = tuple(
        sorted(
            task.canonical_task_cid
            for task in accepted_tasks
            if task.semantic_identity in completed
        )
    )

    criterion_owners: dict[str, set[str]] = {}
    for task in accepted_tasks:
        for criterion in task.acceptance:
            criterion_owners.setdefault(criterion, set()).add(
                task.semantic_identity
            )
    completed_acceptance = tuple(
        sorted(
            criterion
            for criterion, owners in criterion_owners.items()
            if owners and owners.issubset(completed)
        )
    )

    source_decisions: dict[str, list[TaskAdmissionDecision]] = {}
    for decision in result.decisions:
        for source in decision.source_identities:
            source_decisions.setdefault(source, []).append(decision)
    completed_sources: list[str] = []
    incomplete_sources: list[str] = []
    for source, decisions in source_decisions.items():
        if decisions and all(
            decision.accepted
            and decision.candidate.semantic_identity in completed
            for decision in decisions
        ):
            completed_sources.append(source)
        else:
            incomplete_sources.append(source)

    return TaskCompletionPropagation(
        completed_task_identities=tuple(completed),
        completed_task_cids=completed_cids,
        completed_source_identities=tuple(completed_sources),
        incomplete_source_identities=tuple(incomplete_sources),
        completed_acceptance=completed_acceptance,
    )


@dataclass(frozen=True)
class TaskGranularityRun:
    """One arm of a paired granularity fixture."""

    fixture_id: str
    tasks: tuple[TaskCandidate, ...]
    completed_tasks: tuple[str, ...]
    model_calls: int
    run_id: str = ""

    def __post_init__(self) -> None:
        fixture = _normalized_display_text(self.fixture_id)
        if not fixture:
            raise ValueError("fixture_id is required")
        object.__setattr__(self, "fixture_id", fixture)
        normalized_tasks = tuple(
            sorted(
                (
                    item
                    if isinstance(item, TaskCandidate)
                    else TaskCandidate.from_mapping(item)
                    for item in self.tasks
                ),
                key=lambda item: item.canonical_task_cid,
            )
        )
        object.__setattr__(self, "tasks", normalized_tasks)
        object.__setattr__(
            self,
            "completed_tasks",
            _resolve_completed_task_identities(
                normalized_tasks, self.completed_tasks
            ),
        )
        object.__setattr__(
            self,
            "model_calls",
            _non_negative_int(self.model_calls, "model_calls"),
        )
        expected = _task_quality_evidence_cid(self._material())
        supplied = str(self.run_id or "").strip()
        if supplied and supplied != expected:
            raise ValueError("task granularity run identity mismatch")
        object.__setattr__(self, "run_id", expected)

    @property
    def acceptance_surface(self) -> tuple[str, ...]:
        return _strings(
            criterion for task in self.tasks for criterion in task.acceptance
        )

    @property
    def completed_acceptance(self) -> tuple[str, ...]:
        completed = set(self.completed_tasks)
        owners: dict[str, set[str]] = {}
        for task in self.tasks:
            for criterion in task.acceptance:
                owners.setdefault(criterion, set()).add(task.semantic_identity)
        return tuple(
            sorted(
                criterion
                for criterion, identities in owners.items()
                if identities and identities.issubset(completed)
            )
        )

    @property
    def duplicate_semantic_task_count(self) -> int:
        identities = [task.semantic_identity for task in self.tasks]
        return len(identities) - len(set(identities))

    @property
    def calls_per_accepted_criterion(self) -> float | None:
        if not self.completed_acceptance:
            return None
        return self.model_calls / len(self.completed_acceptance)

    def _material(self) -> dict[str, Any]:
        return {
            "schema": TASK_GRANULARITY_RUN_SCHEMA,
            "fixture_id": self.fixture_id,
            "task_cids": [task.canonical_task_cid for task in self.tasks],
            "completed_tasks": list(self.completed_tasks),
            "model_calls": self.model_calls,
        }

    def to_dict(self) -> dict[str, Any]:
        result = self._material()
        result.update(
            {
                "run_id": self.run_id,
                "acceptance_surface": list(self.acceptance_surface),
                "completed_acceptance": list(self.completed_acceptance),
                "duplicate_semantic_task_count": (
                    self.duplicate_semantic_task_count
                ),
                "calls_per_accepted_criterion": (
                    self.calls_per_accepted_criterion
                ),
            }
        )
        return result


@dataclass(frozen=True)
class TaskGranularityComparison:
    fixture_id: str
    baseline_run_id: str
    candidate_run_id: str
    source_coverage_preserved: bool
    completion_exact: bool
    zero_duplicate_semantic_tasks: bool
    fewer_model_calls_per_accepted_criterion: bool

    @property
    def qualifies(self) -> bool:
        return (
            self.source_coverage_preserved
            and self.completion_exact
            and self.zero_duplicate_semantic_tasks
            and self.fewer_model_calls_per_accepted_criterion
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["qualifies"] = self.qualifies
        return result


def compare_task_granularity_runs(
    baseline: TaskGranularityRun,
    candidate: TaskGranularityRun,
) -> TaskGranularityComparison:
    """Compare paired fixtures without allowing criterion-denominator drift."""

    if baseline.fixture_id != candidate.fixture_id:
        raise ValueError("task granularity runs must name the same fixture")
    same_surface = baseline.acceptance_surface == candidate.acceptance_surface
    same_completion = (
        baseline.completed_acceptance == baseline.acceptance_surface
        and candidate.completed_acceptance == candidate.acceptance_surface
        and baseline.completed_acceptance == candidate.completed_acceptance
    )
    baseline_count = len(baseline.completed_acceptance)
    candidate_count = len(candidate.completed_acceptance)
    fewer_calls = (
        baseline_count > 0
        and candidate_count > 0
        and candidate.model_calls * baseline_count
        < baseline.model_calls * candidate_count
    )
    return TaskGranularityComparison(
        fixture_id=baseline.fixture_id,
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        source_coverage_preserved=same_surface,
        completion_exact=same_completion,
        zero_duplicate_semantic_tasks=(
            candidate.duplicate_semantic_task_count == 0
        ),
        fewer_model_calls_per_accepted_criterion=fewer_calls,
    )


def _task_split_refill_material(
    *,
    source_candidate: TaskCandidate,
    policy: TaskQualityPolicy,
    initial_open_work: int,
    first_admission: TaskAdmissionResult,
    refill_admission: TaskAdmissionResult,
    repository_tree: str,
) -> dict[str, Any]:
    return {
        "schema": TASK_SPLIT_REFILL_EVIDENCE_SCHEMA,
        "requirement_id": TASK_SPLIT_REFILL_REQUIREMENT_ID,
        "repository_tree": str(repository_tree),
        "policy_id": policy.policy_id,
        "policy": asdict(policy),
        "source_candidate": source_candidate.to_dict(),
        "initial_open_work": initial_open_work,
        "first_admission": first_admission.to_dict(),
        "refill_admission": refill_admission.to_dict(),
    }


def _task_split_refill_qualifies(material: Mapping[str, Any]) -> bool:
    """Independently reproduce the complete split-then-refill obligation."""

    try:
        source_payload = material.get("source_candidate")
        policy_payload = material.get("policy")
        first_payload = material.get("first_admission")
        refill_payload = material.get("refill_admission")
        if not all(
            isinstance(value, Mapping)
            for value in (
                source_payload,
                policy_payload,
                first_payload,
                refill_payload,
            )
        ):
            return False
        repository_tree = str(material.get("repository_tree") or "").strip()
        if not _authoritative_repository_tree(repository_tree):
            return False
        source = TaskCandidate.from_dict(source_payload)
        policy = TaskQualityPolicy(**dict(policy_payload))
        if str(material.get("policy_id") or "") != policy.policy_id:
            return False
        initial_open_work = _non_negative_int(
            material.get("initial_open_work"), "initial_open_work"
        )
        if not is_over_broad(source, policy):
            return False
        expected_children = split_task_candidate(source, policy=policy)
        if len(expected_children) < 2 or any(
            is_over_broad(child, policy) for child in expected_children
        ):
            return False
        expected_semantic_ids = tuple(
            sorted(child.semantic_identity for child in expected_children)
        )
        expected_task_cids = tuple(
            sorted(child.canonical_task_cid for child in expected_children)
        )
        if (
            len(set(expected_semantic_ids)) != len(expected_semantic_ids)
            or len(set(expected_task_cids)) != len(expected_task_cids)
        ):
            return False

        first = refine_task_candidates(
            (source,),
            policy=policy,
            current_open_work=initial_open_work,
        )
        refill = refine_task_candidates(
            (source,),
            policy=policy,
            existing_tasks=first.accepted,
            current_open_work=first.final_open_work,
        )
        if first.to_dict() != dict(first_payload):
            return False
        if refill.to_dict() != dict(refill_payload):
            return False

        admitted_semantic_ids = tuple(
            sorted(candidate.semantic_identity for candidate in first.accepted)
        )
        admitted_task_cids = tuple(
            sorted(candidate.canonical_task_cid for candidate in first.accepted)
        )
        if (
            admitted_semantic_ids != expected_semantic_ids
            or admitted_task_cids != expected_task_cids
            or any(
                decision.status is not TaskAdmissionStatus.SPLIT
                or decision.source_identities != (source.semantic_identity,)
                for decision in first.decisions
            )
        ):
            return False
        if first.final_open_work != initial_open_work + len(expected_children):
            return False

        # Splitting must cover the complete work surface and retain external
        # prerequisites without creating sibling dependencies.
        if {
            value for child in expected_children for value in child.acceptance
        } != set(source.acceptance):
            return False
        if {
            value for child in expected_children for value in child.effects
        } != set(source.effects):
            return False
        if {
            value for child in expected_children for value in child.evidence_subset
        } != set(source.evidence_subset):
            return False
        if {
            value for child in expected_children for value in child.predicted_paths
        } != set(source.outputs) | set(source.predicted_paths):
            return False
        if {
            value for child in expected_children for value in child.predicted_symbols
        } != set(source.predicted_symbols):
            return False
        if {
            value for child in expected_children for value in child.context_paths
        } != set(source.context_paths):
            return False
        if any(
            child.goal_id != source.goal_id
            or child.preconditions != source.preconditions
            or child.dependencies != source.dependencies
            or child.conflicts != source.conflicts
            or child.resources != source.resources
            or child.resource_class != source.resource_class
            or child.token_class != source.token_class
            or child.validation_commands != source.validation_commands
            or child.merge_fate != source.merge_fate
            for child in expected_children
        ):
            return False

        replay_semantic_ids = tuple(
            sorted(decision.candidate.semantic_identity for decision in refill.decisions)
        )
        duplicate_codes = {
            "duplicate_semantic_identity",
            "historical_duplicate",
        }
        return (
            not refill.accepted
            and replay_semantic_ids == expected_semantic_ids
            and refill.initial_open_work == first.final_open_work
            and refill.final_open_work == first.final_open_work
            and refill.bounded
            and all(
                duplicate_codes.intersection(decision.rejection_reasons)
                for decision in refill.decisions
            )
        )
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class TaskSplitRefillEvidence:
    """Content-addressed proof that broad work cannot duplicate on refill."""

    repository_tree: str
    policy_id: str
    policy: Mapping[str, Any]
    source_candidate: Mapping[str, Any]
    initial_open_work: int
    first_admission: Mapping[str, Any]
    refill_admission: Mapping[str, Any]
    evidence_id: str
    integrity_digest: str
    _producer_seal: Any = field(default=None, compare=False, repr=False)

    @classmethod
    def create(
        cls,
        source_candidate: TaskCandidate | Mapping[str, Any],
        *,
        policy: TaskQualityPolicy | None = None,
        initial_open_work: int = 0,
        repository_tree: str = "",
    ) -> "TaskSplitRefillEvidence":
        """Execute both admission cycles and bind their exact canonical result."""

        source = (
            source_candidate
            if isinstance(source_candidate, TaskCandidate)
            else TaskCandidate.from_mapping(source_candidate)
        )
        selected = policy or TaskQualityPolicy()
        open_work = _non_negative_int(initial_open_work, "initial_open_work")
        first = refine_task_candidates(
            (source,),
            policy=selected,
            current_open_work=open_work,
        )
        refill = refine_task_candidates(
            (source,),
            policy=selected,
            existing_tasks=first.accepted,
            current_open_work=first.final_open_work,
        )
        material = _task_split_refill_material(
            source_candidate=source,
            policy=selected,
            initial_open_work=open_work,
            first_admission=first,
            refill_admission=refill,
            repository_tree=repository_tree,
        )
        digest = hashlib.sha256(_task_quality_evidence_bytes(material)).hexdigest()
        return cls(
            repository_tree=str(repository_tree),
            policy_id=selected.policy_id,
            policy=dict(material["policy"]),
            source_candidate=dict(material["source_candidate"]),
            initial_open_work=open_work,
            first_admission=dict(material["first_admission"]),
            refill_admission=dict(material["refill_admission"]),
            evidence_id=_task_quality_evidence_cid(material),
            integrity_digest=digest,
            _producer_seal=_TASK_SPLIT_REFILL_EVIDENCE_SEAL,
        )

    def _material(self) -> dict[str, Any]:
        return {
            "schema": TASK_SPLIT_REFILL_EVIDENCE_SCHEMA,
            "requirement_id": TASK_SPLIT_REFILL_REQUIREMENT_ID,
            "repository_tree": self.repository_tree,
            "policy_id": self.policy_id,
            "policy": dict(self.policy),
            "source_candidate": dict(self.source_candidate),
            "initial_open_work": self.initial_open_work,
            "first_admission": dict(self.first_admission),
            "refill_admission": dict(self.refill_admission),
        }

    def verify_integrity(self) -> bool:
        material = self._material()
        return (
            self.integrity_digest
            == hashlib.sha256(_task_quality_evidence_bytes(material)).hexdigest()
            and self.evidence_id == _task_quality_evidence_cid(material)
        )

    @property
    def proved_requirement_ids(self) -> tuple[str, ...]:
        if (
            self._producer_seal is _TASK_SPLIT_REFILL_EVIDENCE_SEAL
            and self.verify_integrity()
            and _task_split_refill_qualifies(self._material())
        ):
            return (TASK_SPLIT_REFILL_REQUIREMENT_ID,)
        return ()

    def to_dict(self) -> dict[str, Any]:
        material = self._material()
        qualifies = bool(self.proved_requirement_ids)
        material.update(
            {
                "evidence_id": self.evidence_id,
                "integrity_digest": self.integrity_digest,
                "proved_requirement_ids": list(self.proved_requirement_ids),
                "status": "passed" if qualifies else "diagnostic",
                "complete": qualifies,
                "coverage_complete": qualifies,
                "source_tier": "validation",
            }
        )
        return material

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TaskSplitRefillEvidence":
        """Restore integrity-checked diagnostics without producer authority."""

        evidence = cls(
            repository_tree=str(payload.get("repository_tree") or ""),
            policy_id=str(payload.get("policy_id") or ""),
            policy=dict(payload.get("policy") or {}),
            source_candidate=dict(payload.get("source_candidate") or {}),
            initial_open_work=_non_negative_int(
                payload.get("initial_open_work"), "initial_open_work"
            ),
            first_admission=dict(payload.get("first_admission") or {}),
            refill_admission=dict(payload.get("refill_admission") or {}),
            evidence_id=str(payload.get("evidence_id") or ""),
            integrity_digest=str(payload.get("integrity_digest") or ""),
            _producer_seal=None,
        )
        if not evidence.verify_integrity():
            raise ValueError("task split/refill evidence digest mismatch")
        return evidence


def prove_task_split_refill(
    source_candidate: TaskCandidate | Mapping[str, Any],
    *,
    policy: TaskQualityPolicy | None = None,
    initial_open_work: int = 0,
    repository_tree: str = "",
) -> TaskSplitRefillEvidence:
    """Produce the authoritative two-cycle split/refill validation receipt."""

    return TaskSplitRefillEvidence.create(
        source_candidate,
        policy=policy,
        initial_open_work=initial_open_work,
        repository_tree=repository_tree,
    )


SUCCESSOR_GOAL_QUALITY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/successor-goal-quality-lint@1"
)
SUCCESSOR_GOAL_QUALITY_POLICY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/successor-goal-quality-policy@1"
)


class GoalQualityIssueCode(str, Enum):
    """Closed reasons a generated successor goal fails quality admission."""

    MISSING_TITLE = "missing_title"
    MISSING_PARENT_GOAL = "missing_parent_goal"
    MISSING_OUTCOME = "missing_outcome"
    MISSING_SCOPE = "missing_scope"
    MISSING_ASSUMPTIONS = "missing_assumptions"
    MISSING_NON_GOALS = "missing_non_goals"
    MISSING_ACCEPTANCE = "missing_acceptance"
    MISSING_EVIDENCE = "missing_evidence"
    MISSING_VALIDATION = "missing_validation"
    MISSING_OUTPUTS = "missing_outputs"
    INVALID_KIND = "invalid_kind"
    NON_FINITE_CONFIDENCE = "non_finite_confidence"
    CONFIDENCE_BELOW_THRESHOLD = "confidence_below_threshold"
    NON_FINITE_NOVELTY = "non_finite_novelty"
    NOVELTY_BELOW_THRESHOLD = "novelty_below_threshold"
    INVALID_DEPTH = "invalid_depth"
    DEPTH_EXCEEDED = "depth_exceeded"
    UNBOUNDED_SCOPE = "unbounded_scope"
    BREADTH_EXCEEDED = "breadth_exceeded"
    INVALID_TOKEN_ESTIMATE = "invalid_token_estimate"
    TOKEN_BUDGET_EXCEEDED = "token_budget_exceeded"
    INVALID_GOAL_COUNT = "invalid_goal_count"
    GOAL_BUDGET_EXCEEDED = "goal_budget_exceeded"
    INVALID_TASK_COUNT = "invalid_task_count"
    TASK_BUDGET_EXCEEDED = "task_budget_exceeded"
    INVALID_OPEN_WORK_COUNT = "invalid_open_work_count"
    OPEN_WORK_BUDGET_EXCEEDED = "open_work_budget_exceeded"
    UNSUPPORTED_DEPENDENCY = "unsupported_dependency"


@dataclass(frozen=True)
class GoalQualityLintPolicy:
    """Finite local bounds for one mapping-based successor goal contract."""

    minimum_confidence: float = 0.5
    minimum_novelty: float = 0.5
    max_depth: int = 3
    max_scope_items: int = 16
    max_acceptance_items: int = 12
    max_evidence_items: int = 16
    max_validation_commands: int = 12
    max_dependencies: int = 16
    max_outputs: int = 16
    max_total_breadth: int = 64
    max_estimated_tokens: int = 8_192
    max_goals_per_batch: int = 8
    max_tasks_per_goal: int = 3
    max_open_work: int = 48
    max_issues: int = 32

    def __post_init__(self) -> None:
        for name in ("minimum_confidence", "minimum_novelty"):
            object.__setattr__(self, name, _finite_ratio(getattr(self, name), name))
        for name in (
            "max_scope_items",
            "max_acceptance_items",
            "max_evidence_items",
            "max_validation_commands",
            "max_dependencies",
            "max_outputs",
            "max_total_breadth",
            "max_issues",
        ):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name),
            )
        for name in (
            "max_depth",
            "max_estimated_tokens",
            "max_goals_per_batch",
            "max_tasks_per_goal",
            "max_open_work",
        ):
            object.__setattr__(
                self,
                name,
                _non_negative_int(getattr(self, name), name),
            )

    @property
    def policy_id(self) -> str:
        return _task_quality_evidence_cid(
            {
                "schema": SUCCESSOR_GOAL_QUALITY_POLICY_SCHEMA,
                **asdict(self),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SUCCESSOR_GOAL_QUALITY_POLICY_SCHEMA,
            **asdict(self),
            "policy_id": self.policy_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GoalQualityLintPolicy":
        if not isinstance(payload, Mapping):
            raise TypeError("successor goal quality policy must be a mapping")
        allowed = {"schema", "policy_id", *cls.__dataclass_fields__}
        unknown = sorted(str(key) for key in payload if key not in allowed)
        if unknown:
            raise ValueError(
                "successor goal quality policy contains unknown fields: "
                + ", ".join(unknown)
            )
        if payload.get("schema") != SUCCESSOR_GOAL_QUALITY_POLICY_SCHEMA:
            raise ValueError("unsupported successor goal quality policy schema")
        result = cls(
            **{
                name: payload.get(name, field_.default)
                for name, field_ in cls.__dataclass_fields__.items()
            }
        )
        if payload.get("policy_id") != result.policy_id:
            raise ValueError("successor goal quality policy identity does not match")
        return result


@dataclass(frozen=True)
class GoalQualityIssue:
    """One typed, content-bound successor-goal lint finding."""

    code: GoalQualityIssueCode | str
    field: str
    detail: str
    related_values: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        try:
            code = (
                self.code
                if isinstance(self.code, GoalQualityIssueCode)
                else GoalQualityIssueCode(str(self.code))
            )
        except ValueError as exc:
            raise ValueError(f"unknown successor goal quality issue: {self.code!r}") from exc
        field_name = _normalized_display_text(self.field)
        detail = _normalized_display_text(self.detail)
        if not field_name or not detail:
            raise ValueError("goal quality issues require field and detail")
        if len(detail.encode("utf-8")) > 1_024:
            raise ValueError("goal quality issue detail exceeds 1024 UTF-8 bytes")
        related = _strings(self.related_values)
        if len(related) > 8:
            raise ValueError("goal quality issues may bind at most 8 related values")
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "field", field_name)
        object.__setattr__(self, "detail", detail)
        object.__setattr__(self, "related_values", related)

    @property
    def reason(self) -> str:
        return self.code.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "reason": self.reason,
            "field": self.field,
            "detail": self.detail,
            "related_values": list(self.related_values),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GoalQualityIssue":
        if not isinstance(payload, Mapping):
            raise TypeError("successor goal quality issue must be a mapping")
        allowed = {"code", "reason", "field", "detail", "related_values"}
        unknown = sorted(str(key) for key in payload if key not in allowed)
        if unknown:
            raise ValueError(
                "successor goal quality issue contains unknown fields: "
                + ", ".join(unknown)
            )
        result = cls(
            code=payload.get("code", ""),
            field=str(payload.get("field") or ""),
            detail=str(payload.get("detail") or ""),
            related_values=tuple(payload.get("related_values") or ()),
        )
        if payload.get("reason") != result.reason:
            raise ValueError("successor goal quality issue reason does not match code")
        return result


@dataclass(frozen=True)
class GoalQualityLintResult:
    """Deterministic bounded accounting for one successor goal candidate."""

    candidate_id: str
    policy_id: str
    issues: tuple[GoalQualityIssue, ...]
    total_issue_count: int

    def __post_init__(self) -> None:
        for name in ("candidate_id", "policy_id"):
            normalized = _normalized_display_text(getattr(self, name))
            if not normalized:
                raise ValueError(f"{name} is required")
            object.__setattr__(self, name, normalized)
        issues = tuple(self.issues)
        if any(not isinstance(item, GoalQualityIssue) for item in issues):
            raise TypeError("issues must contain GoalQualityIssue values")
        total = _non_negative_int(self.total_issue_count, "total_issue_count")
        if total < len(issues):
            raise ValueError("total_issue_count cannot be smaller than retained issues")
        object.__setattr__(self, "issues", issues)
        object.__setattr__(self, "total_issue_count", total)

    @property
    def accepted(self) -> bool:
        return self.total_issue_count == 0

    @property
    def rejection_reasons(self) -> tuple[str, ...]:
        return tuple(item.code.value for item in self.issues)

    @property
    def issues_truncated(self) -> int:
        return self.total_issue_count - len(self.issues)

    @property
    def result_id(self) -> str:
        return _task_quality_evidence_cid(
            {
                "schema": SUCCESSOR_GOAL_QUALITY_SCHEMA,
                "candidate_id": self.candidate_id,
                "policy_id": self.policy_id,
                "issues": [item.to_dict() for item in self.issues],
                "total_issue_count": self.total_issue_count,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SUCCESSOR_GOAL_QUALITY_SCHEMA,
            "candidate_id": self.candidate_id,
            "policy_id": self.policy_id,
            "accepted": self.accepted,
            "issues": [item.to_dict() for item in self.issues],
            "rejection_reasons": list(self.rejection_reasons),
            "total_issue_count": self.total_issue_count,
            "issues_truncated": self.issues_truncated,
            "result_id": self.result_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GoalQualityLintResult":
        if not isinstance(payload, Mapping):
            raise TypeError("successor goal quality result must be a mapping")
        allowed = {
            "schema",
            "candidate_id",
            "policy_id",
            "accepted",
            "issues",
            "rejection_reasons",
            "total_issue_count",
            "issues_truncated",
            "result_id",
        }
        unknown = sorted(str(key) for key in payload if key not in allowed)
        if unknown:
            raise ValueError(
                "successor goal quality result contains unknown fields: "
                + ", ".join(unknown)
            )
        if payload.get("schema") != SUCCESSOR_GOAL_QUALITY_SCHEMA:
            raise ValueError("unsupported successor goal quality result schema")
        raw_issues = payload.get("issues") or ()
        if isinstance(raw_issues, (str, bytes, bytearray, Mapping)):
            raise TypeError("successor goal quality result issues must be a sequence")
        result = cls(
            candidate_id=str(payload.get("candidate_id") or ""),
            policy_id=str(payload.get("policy_id") or ""),
            issues=tuple(GoalQualityIssue.from_dict(item) for item in raw_issues),
            total_issue_count=payload.get("total_issue_count", -1),
        )
        if payload.get("accepted") is not result.accepted:
            raise ValueError("successor goal quality accepted projection was forged")
        if tuple(payload.get("rejection_reasons") or ()) != result.rejection_reasons:
            raise ValueError("successor goal quality rejection projection was forged")
        if payload.get("issues_truncated") != result.issues_truncated:
            raise ValueError("successor goal quality truncation projection was forged")
        if payload.get("result_id") != result.result_id:
            raise ValueError("successor goal quality result identity does not match")
        return result


def _successor_value(payload: Mapping[str, Any], *names: str) -> Any:
    """Return the first present successor field, preserving false and zero."""

    normalized = {
        str(key).strip().casefold().replace("-", "_").replace(" ", "_"): value
        for key, value in payload.items()
    }
    for name in names:
        key = name.casefold().replace("-", "_").replace(" ", "_")
        if key in normalized:
            return normalized[key]
    return None


def _successor_strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        value = _successor_value(value, "include", "items", "paths")
    return _strings(value)


def _successor_finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _successor_integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, float) and (
        not math.isfinite(value) or not value.is_integer()
    ):
        return None
    if isinstance(value, str) and not re.fullmatch(r"\+?[0-9]+", value.strip()):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def lint_successor_goal_candidate(
    candidate: Mapping[str, Any],
    *,
    policy: GoalQualityLintPolicy | None = None,
    supported_dependencies: Iterable[str] | None = None,
    open_work_count: int | None = None,
) -> GoalQualityLintResult:
    """Lint one complete successor goal mapping without importing v2 runtime types.

    The accepted spellings intentionally overlap :class:`ObjectiveWorkProposal`
    so provider packets can be checked before constructing a mutation-capable
    objective object.  ``supported_dependencies=None`` means the caller has not
    supplied an authoritative dependency population; an empty iterable means
    no external dependencies are available.
    """

    if not isinstance(candidate, Mapping):
        raise TypeError("successor goal candidate must be a mapping")
    selected = policy or GoalQualityLintPolicy()
    if not isinstance(selected, GoalQualityLintPolicy):
        raise TypeError("policy must be a GoalQualityLintPolicy")

    title = _normalized_display_text(_successor_value(candidate, "title", "summary"))
    parent_goal = _normalized_display_text(
        _successor_value(candidate, "parent_goal_id", "parent_objective_id", "goal_id")
    )
    outcome = _normalized_display_text(
        _successor_value(candidate, "outcome", "expected_outcome")
    )
    raw_scope = _successor_value(candidate, "scope", "scope_include")
    scope = _successor_strings(raw_scope)
    assumptions = _successor_strings(_successor_value(candidate, "assumptions"))
    non_goals = _successor_strings(
        _successor_value(candidate, "non_goals", "excluded_outcomes")
    )
    acceptance = _successor_strings(
        _successor_value(
            candidate,
            "acceptance_subset",
            "acceptance",
            "acceptance_criteria",
            "parent_objective_terms",
        )
    )
    evidence = _successor_strings(
        _successor_value(
            candidate,
            "expected_evidence_delta",
            "evidence_subset",
            "evidence",
            "evidence_requirements",
        )
    )
    validation = _successor_strings(
        _successor_value(candidate, "validation_commands", "validation")
    )
    outputs = _successor_strings(
        _successor_value(candidate, "predicted_files", "outputs", "predicted_paths")
    )
    dependencies = _successor_strings(
        _successor_value(candidate, "dependencies", "depends_on")
    )
    explicit_unsupported = _successor_strings(
        _successor_value(candidate, "unsupported_dependencies")
    )
    kind = _normalized_display_text(
        _successor_value(candidate, "kind", "work_kind", "proposal_kind") or "goal"
    ).casefold()
    confidence = _successor_finite(
        _successor_value(candidate, "confidence", "goal_confidence")
    )
    novelty = _successor_finite(
        _successor_value(candidate, "novelty", "semantic_novelty")
    )
    depth = _successor_integer(
        _successor_value(candidate, "depth", "graph_depth")
    )
    estimated_tokens = _successor_integer(
        _successor_value(candidate, "estimated_tokens", "token_cost")
    )
    raw_goal_count = _successor_value(
        candidate, "goal_count", "requested_goal_count"
    )
    goal_count = _successor_integer(
        1 if raw_goal_count is None else raw_goal_count
    )
    tasks_value = _successor_value(candidate, "tasks", "task_candidates")
    raw_task_count = _successor_value(candidate, "task_count", "estimated_task_count")
    if raw_task_count is None and tasks_value is not None:
        if isinstance(tasks_value, Sequence) and not isinstance(
            tasks_value, (str, bytes, bytearray)
        ):
            raw_task_count = len(tasks_value)
        else:
            raw_task_count = -1
    task_count = _successor_integer(raw_task_count)
    if raw_task_count is None:
        task_count = 1
    raw_open_work = (
        open_work_count
        if open_work_count is not None
        else _successor_value(candidate, "open_work_count", "current_open_work")
    )
    if raw_open_work is None:
        raw_open_work = 0
    current_open_work = _successor_integer(raw_open_work)

    identity_material = {
        "schema": SUCCESSOR_GOAL_QUALITY_SCHEMA,
        "title": title,
        "parent_goal_id": parent_goal,
        "outcome": outcome,
        "scope": scope,
        "assumptions": assumptions,
        "non_goals": non_goals,
        "acceptance": acceptance,
        "evidence": evidence,
        "validation": validation,
        "outputs": outputs,
        "dependencies": dependencies,
        "kind": kind,
        "confidence": confidence if confidence is not None else str(
            _successor_value(candidate, "confidence", "goal_confidence")
        ),
        "novelty": novelty if novelty is not None else str(
            _successor_value(candidate, "novelty", "semantic_novelty")
        ),
        "depth": depth,
        "estimated_tokens": estimated_tokens,
        "goal_count": goal_count,
        "task_count": task_count,
    }
    candidate_id = _task_quality_evidence_cid(identity_material)
    findings: list[GoalQualityIssue] = []

    def reject(
        code: GoalQualityIssueCode,
        field_name: str,
        detail: str,
        related: Iterable[str] = (),
    ) -> None:
        bounded_detail = detail.encode("utf-8")[:1_024].decode(
            "utf-8", errors="ignore"
        )
        related_values = tuple(
            sorted(
                {
                    str(item).encode("utf-8")[:256].decode(
                        "utf-8", errors="ignore"
                    )
                    for item in related
                }
            )
        )[:8]
        findings.append(
            GoalQualityIssue(code, field_name, bounded_detail, related_values)
        )

    for missing, code, field_name, detail in (
        (not title, GoalQualityIssueCode.MISSING_TITLE, "title", "title is required"),
        (
            not parent_goal,
            GoalQualityIssueCode.MISSING_PARENT_GOAL,
            "parent_goal_id",
            "parent_goal_id is required",
        ),
        (not outcome, GoalQualityIssueCode.MISSING_OUTCOME, "outcome", "outcome is required"),
        (not scope, GoalQualityIssueCode.MISSING_SCOPE, "scope", "finite scope is required"),
        (
            not assumptions,
            GoalQualityIssueCode.MISSING_ASSUMPTIONS,
            "assumptions",
            "assumptions require an explicit value or reviewed-none marker",
        ),
        (
            not non_goals,
            GoalQualityIssueCode.MISSING_NON_GOALS,
            "non_goals",
            "non_goals require an explicit value or reviewed-none marker",
        ),
        (
            not acceptance,
            GoalQualityIssueCode.MISSING_ACCEPTANCE,
            "acceptance",
            "acceptance criteria are required",
        ),
        (
            not evidence,
            GoalQualityIssueCode.MISSING_EVIDENCE,
            "evidence",
            "an evidence delta is required",
        ),
        (
            not validation,
            GoalQualityIssueCode.MISSING_VALIDATION,
            "validation_commands",
            "validation commands are required",
        ),
        (
            not outputs,
            GoalQualityIssueCode.MISSING_OUTPUTS,
            "outputs",
            "predicted output paths are required",
        ),
    ):
        if missing:
            reject(code, field_name, detail)
    if kind not in {"goal", "subgoal"}:
        reject(
            GoalQualityIssueCode.INVALID_KIND,
            "kind",
            "successor work kind must be goal or subgoal",
            (kind,),
        )
    if confidence is None or not 0.0 <= confidence <= 1.0:
        reject(
            GoalQualityIssueCode.NON_FINITE_CONFIDENCE,
            "confidence",
            "confidence must be a finite ratio between zero and one",
        )
    elif confidence < selected.minimum_confidence:
        reject(
            GoalQualityIssueCode.CONFIDENCE_BELOW_THRESHOLD,
            "confidence",
            f"{confidence:.6f} is below {selected.minimum_confidence:.6f}",
        )
    if novelty is None or not 0.0 <= novelty <= 1.0:
        reject(
            GoalQualityIssueCode.NON_FINITE_NOVELTY,
            "novelty",
            "novelty must be a finite ratio between zero and one",
        )
    elif novelty < selected.minimum_novelty:
        reject(
            GoalQualityIssueCode.NOVELTY_BELOW_THRESHOLD,
            "novelty",
            f"{novelty:.6f} is below {selected.minimum_novelty:.6f}",
        )
    if depth is None:
        reject(GoalQualityIssueCode.INVALID_DEPTH, "depth", "depth must be non-negative")
    elif depth > selected.max_depth:
        reject(
            GoalQualityIssueCode.DEPTH_EXCEEDED,
            "depth",
            f"{depth} exceeds {selected.max_depth}",
        )
    unbounded = tuple(
        item
        for item in scope
        if item.casefold().strip() in {"*", "**", ".", "/", "all", "any", "repository"}
        or "*" in item
    )
    if unbounded:
        reject(
            GoalQualityIssueCode.UNBOUNDED_SCOPE,
            "scope",
            "scope contains wildcard or repository-wide subjects",
            unbounded,
        )
    breadth = {
        "scope": (len(scope), selected.max_scope_items),
        "acceptance": (len(acceptance), selected.max_acceptance_items),
        "evidence": (len(evidence), selected.max_evidence_items),
        "validation_commands": (len(validation), selected.max_validation_commands),
        "dependencies": (len(dependencies), selected.max_dependencies),
        "outputs": (len(outputs), selected.max_outputs),
    }
    for field_name, (count, limit) in breadth.items():
        if count > limit:
            reject(
                GoalQualityIssueCode.BREADTH_EXCEEDED,
                field_name,
                f"{count} items exceed {limit}",
            )
    total_breadth = sum(count for count, _limit in breadth.values())
    if total_breadth > selected.max_total_breadth:
        reject(
            GoalQualityIssueCode.BREADTH_EXCEEDED,
            "total_breadth",
            f"{total_breadth} items exceed {selected.max_total_breadth}",
        )
    if estimated_tokens is None:
        reject(
            GoalQualityIssueCode.INVALID_TOKEN_ESTIMATE,
            "estimated_tokens",
            "estimated_tokens must be a non-negative integer",
        )
    elif estimated_tokens > selected.max_estimated_tokens:
        reject(
            GoalQualityIssueCode.TOKEN_BUDGET_EXCEEDED,
            "estimated_tokens",
            f"{estimated_tokens} exceeds {selected.max_estimated_tokens}",
        )
    if goal_count is None or goal_count < 1:
        reject(
            GoalQualityIssueCode.INVALID_GOAL_COUNT,
            "goal_count",
            "goal_count must be a positive integer",
        )
    elif goal_count > selected.max_goals_per_batch:
        reject(
            GoalQualityIssueCode.GOAL_BUDGET_EXCEEDED,
            "goal_count",
            f"{goal_count} exceeds {selected.max_goals_per_batch}",
        )
    if task_count is None or task_count < 1:
        reject(
            GoalQualityIssueCode.INVALID_TASK_COUNT,
            "task_count",
            "task_count must be a positive integer",
        )
    elif task_count > selected.max_tasks_per_goal:
        reject(
            GoalQualityIssueCode.TASK_BUDGET_EXCEEDED,
            "task_count",
            f"{task_count} exceeds {selected.max_tasks_per_goal}",
        )
    if current_open_work is None:
        reject(
            GoalQualityIssueCode.INVALID_OPEN_WORK_COUNT,
            "open_work_count",
            "open_work_count must be a non-negative integer",
        )
    elif current_open_work + (task_count or 0) > selected.max_open_work:
        reject(
            GoalQualityIssueCode.OPEN_WORK_BUDGET_EXCEEDED,
            "open_work_count",
            f"{current_open_work} open plus {task_count or 0} new exceeds "
            f"{selected.max_open_work}",
        )

    unsupported = set(explicit_unsupported)
    if supported_dependencies is not None:
        supported = set(_strings(supported_dependencies))
        unsupported.update(item for item in dependencies if item not in supported)
    if unsupported:
        unsupported_values = tuple(
            sorted(unsupported, key=lambda item: (item.casefold(), item))
        )
        reject(
            GoalQualityIssueCode.UNSUPPORTED_DEPENDENCY,
            "dependencies",
            f"{len(unsupported_values)} declared dependencies are unavailable",
            unsupported_values,
        )

    ordered = tuple(
        sorted(
            findings,
            key=lambda item: (
                item.code.value,
                item.field.casefold(),
                item.detail.casefold(),
                item.related_values,
            ),
        )
    )
    return GoalQualityLintResult(
        candidate_id=candidate_id,
        policy_id=selected.policy_id,
        issues=ordered[: selected.max_issues],
        total_issue_count=len(ordered),
    )


# The shorter spelling is useful to callers which already operate exclusively
# on successor candidates.
lint_goal_candidate = lint_successor_goal_candidate


# Compatibility spellings used by existing proposal evaluators and early ASI
# design notes.
TaskQualityResult = TaskAdmissionResult
TaskRefinementResult = TaskAdmissionResult
TaskQualityRejection = TaskRejection
TaskGranularityMeasurement = TaskCostMeasurement
canonical_task_semantic_identity = canonical_semantic_identity
coalesce_tasks = coalesce_task_candidates
split_over_broad_candidate = split_task_candidate
evaluate_task_candidates = refine_task_candidates


__all__ = [
    "RESOURCE_CLASSES",
    "TASK_GENERATION_ACCEPTANCE_CRITERIA",
    "TASK_GENERATION_CHILD_GOAL_IDS",
    "TASK_GENERATION_OBJECTIVE_ID",
    "TASK_GENERATION_PRODUCING_TASK_IDS",
    "TASK_GENERATION_REQUIRED_EXHAUSTIVE_RECEIPTS",
    "TASK_SPLIT_REFILL_EVIDENCE_SCHEMA",
    "TASK_SPLIT_REFILL_REQUIREMENT_ID",
    "TASK_QUALITY_EVALUATOR_VERSION",
    "TASK_QUALITY_SCHEMA",
    "TASK_SEMANTIC_IDENTITY_SCHEMA",
    "SUCCESSOR_GOAL_QUALITY_POLICY_SCHEMA",
    "SUCCESSOR_GOAL_QUALITY_SCHEMA",
    "TASK_WORK_CONTRACT_SCHEMA",
    "TASK_COMPLETION_PROPAGATION_SCHEMA",
    "TASK_GRANULARITY_CALIBRATION_SCHEMA",
    "TASK_GRANULARITY_MEASUREMENT_SCHEMA",
    "TASK_GRANULARITY_RUN_SCHEMA",
    "TOKEN_CLASSES",
    "TOKEN_CLASS_LIMITS",
    "HistoricalTask",
    "GoalQualityIssue",
    "GoalQualityIssueCode",
    "GoalQualityLintPolicy",
    "GoalQualityLintResult",
    "TaskCompletionPropagation",
    "TaskCostMeasurement",
    "TaskGranularityCalibration",
    "TaskGranularityComparison",
    "TaskGranularityMeasurement",
    "TaskGranularityRun",
    "TaskAdmissionDecision",
    "TaskAdmissionResult",
    "TaskAdmissionStatus",
    "TaskCandidate",
    "TaskQualityPolicy",
    "TaskQualityRejection",
    "TaskQualityResult",
    "TaskQualityScore",
    "TaskRefinementResult",
    "TaskRejection",
    "TaskSplitRefillEvidence",
    "admit_task_candidate",
    "can_coalesce_tasks",
    "calibrate_task_granularity",
    "canonical_semantic_identity",
    "canonical_task_semantic_identity",
    "coalesce_task_candidates",
    "coalesce_tasks",
    "compare_task_granularity_runs",
    "evaluate_task_candidates",
    "is_over_broad",
    "is_tiny",
    "lint_goal_candidate",
    "lint_successor_goal_candidate",
    "prove_task_split_refill",
    "propagate_task_completion",
    "refine_task_candidates",
    "score_task_candidate",
    "split_over_broad_candidate",
    "split_task_candidate",
    "task_semantic_similarity",
]
