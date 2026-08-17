"""Learning checkpoint binding and compatible-resume contract.

This module is the campaign durability adapter for training-run identity.  It
does not own a checkpoint store, scheduler, or promotion pointer.  Persistence
is performed by the existing recovery checkpoint store; this contract only
decides whether a bound snapshot may be written or resumed.

A checkpoint is admitted only when every lineage and progress identity listed
by the durable-runtime contract is present.  Compatible resume requires an
exact lineage match and a monotonic progress cursor.  Promotion authority is
never implied by a successful resume.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

from ..proof.formal_verification_contracts import content_identity


CAMPAIGN_DURABILITY_REQUIREMENT_ID: Final = (
    "campaign:durable-runtime-checkpoint-resume-leases-refill@1"
)
LEARNING_CHECKPOINT_BINDING_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/learning-checkpoint-binding@1"
)
LEARNING_RESUME_DECISION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/learning-resume-decision@1"
)

# LEASE-DEFAULT: one renewable 30-minute lease, heartbeat at most 60 seconds,
# monotonically increasing fence, maximum three attempts.
LEASE_DEFAULT_DURATION_SECONDS: Final = 30 * 60
LEASE_DEFAULT_DURATION_MS: Final = LEASE_DEFAULT_DURATION_SECONDS * 1000
LEASE_DEFAULT_HEARTBEAT_SECONDS: Final = 60
LEASE_DEFAULT_HEARTBEAT_MS: Final = LEASE_DEFAULT_HEARTBEAT_SECONDS * 1000
LEASE_DEFAULT_MAX_ATTEMPTS: Final = 3

# Closed L3 ownership vocabulary.  Each kind has a distinct exclusive key.
# LEASE-DEFAULT names the mutation surfaces; the campaign objective also owns
# the training-run lease so a crash cannot start a second overlapping run.
class L3ResourceKind(str, Enum):
    CHECKPOINT = "checkpoint"
    TOKENIZER = "tokenizer"
    CORPUS = "corpus"
    SPLIT = "split"
    COMPILER_CONTRACT = "compiler-contract"
    LOSS_CONFIG = "loss-config"
    PROOF_SHARD = "proof-shard"
    EVALUATION_SHARD = "evaluation-shard"
    PROMOTION_POINTER = "promotion-pointer"
    PUBLICATION = "publication"
    RUN = "run"


NAMED_L3_RESOURCES: Final[tuple[L3ResourceKind, ...]] = tuple(L3ResourceKind)

LEARNING_CHECKPOINT_BINDING_FIELDS: Final[tuple[str, ...]] = (
    "architecture_id",
    "weights_id",
    "optimizer_id",
    "scheduler_id",
    "tokenizer_id",
    "vocab_id",
    "cursor_id",
    "corpus_id",
    "split_id",
    "curriculum_id",
    "loss_id",
    "random_id",
    "env_id",
    "code_id",
    "compiler_id",
)

# Lineage must be identical across a compatible resume.  Progress identities
# may advance but must not rewind or fork.
LINEAGE_BINDING_FIELDS: Final[tuple[str, ...]] = (
    "architecture_id",
    "optimizer_id",
    "scheduler_id",
    "tokenizer_id",
    "vocab_id",
    "corpus_id",
    "split_id",
    "curriculum_id",
    "loss_id",
    "env_id",
    "code_id",
    "compiler_id",
)

PROGRESS_BINDING_FIELDS: Final[tuple[str, ...]] = (
    "weights_id",
    "cursor_id",
    "random_id",
)

PROMOTION_AUTHORITY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "promotion",
        "promotion_pointer",
        "promotion_pointer_id",
        "current_checkpoint_pointer",
        "mutable_promotion_authority",
    }
)


class LearningCheckpointError(ValueError):
    """Malformed or unsafe learning-checkpoint operation."""


class IncompatibleResumeError(LearningCheckpointError):
    """A stored checkpoint cannot be resumed with the requested binding."""


class StaleFenceError(LearningCheckpointError):
    """A write or resume used a superseded fencing token."""


class PromotionMutationError(LearningCheckpointError):
    """The adapter refused to treat a checkpoint as promotion authority."""


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not str(value).strip():
        raise LearningCheckpointError(f"{name} must be a non-empty string")
    text = str(value).strip()
    if "\x00" in text:
        raise LearningCheckpointError(f"{name} must not contain NUL")
    return text


def _required_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LearningCheckpointError(f"{name} must be an integer")
    if value < minimum:
        raise LearningCheckpointError(f"{name} must be at least {minimum}")
    return value


def exclusive_lease_key(
    kind: L3ResourceKind | str,
    *,
    resource_id: str = "",
) -> str:
    """Return the distinct exclusive key for one named L3 resource."""

    selected = kind if isinstance(kind, L3ResourceKind) else L3ResourceKind(str(kind))
    identity = str(resource_id or "").strip()
    if identity:
        return f"l3:{selected.value}:{identity}"
    return f"l3:{selected.value}"


def default_l3_lease_keys() -> dict[L3ResourceKind, str]:
    """Return the closed unscoped exclusive-key catalog."""

    return {kind: exclusive_lease_key(kind) for kind in NAMED_L3_RESOURCES}


def assert_distinct_l3_lease_keys(
    keys: Mapping[L3ResourceKind | str, str] | None = None,
) -> tuple[str, ...]:
    """Fail closed when two L3 resources would share a mutation key."""

    catalog = default_l3_lease_keys() if keys is None else keys
    normalized = {
        (item if isinstance(item, L3ResourceKind) else L3ResourceKind(str(item))): _required_text(
            key, "lease key"
        )
        for item, key in catalog.items()
    }
    missing = [kind for kind in NAMED_L3_RESOURCES if kind not in normalized]
    if missing:
        raise LearningCheckpointError(
            "L3 lease catalog is missing: " + ", ".join(kind.value for kind in missing)
        )
    values = tuple(normalized[kind] for kind in NAMED_L3_RESOURCES)
    if len(set(values)) != len(values):
        raise LearningCheckpointError("named L3 resources must use distinct exclusive keys")
    return values


def _reject_promotion_authority(payload: Mapping[str, Any], *, noun: str) -> None:
    for key in payload:
        marker = str(key or "").strip().casefold()
        if marker in PROMOTION_AUTHORITY_FIELDS:
            raise PromotionMutationError(f"{noun} must not carry mutable promotion authority")


@dataclass(frozen=True)
class LearningCheckpointBinding:
    """Exact architecture-through-compiler identity for one training snapshot."""

    architecture_id: str
    weights_id: str
    optimizer_id: str
    scheduler_id: str
    tokenizer_id: str
    vocab_id: str
    cursor_id: str
    corpus_id: str
    split_id: str
    curriculum_id: str
    loss_id: str
    random_id: str
    env_id: str
    code_id: str
    compiler_id: str
    cursor_step: int = 0
    schema: str = LEARNING_CHECKPOINT_BINDING_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != LEARNING_CHECKPOINT_BINDING_SCHEMA:
            raise LearningCheckpointError("unsupported learning checkpoint binding schema")
        for name in LEARNING_CHECKPOINT_BINDING_FIELDS:
            object.__setattr__(self, name, _required_text(getattr(self, name), name))
        object.__setattr__(
            self, "cursor_step", _required_int(self.cursor_step, "cursor_step", minimum=0)
        )

    @property
    def lineage_id(self) -> str:
        return content_identity(
            {
                "kind": "learning-checkpoint-lineage",
                "fields": {name: getattr(self, name) for name in LINEAGE_BINDING_FIELDS},
            }
        )

    @property
    def progress_id(self) -> str:
        return content_identity(
            {
                "kind": "learning-checkpoint-progress",
                "cursor_step": self.cursor_step,
                "fields": {name: getattr(self, name) for name in PROGRESS_BINDING_FIELDS},
            }
        )

    @property
    def binding_id(self) -> str:
        return content_identity(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "architecture_id": self.architecture_id,
            "weights_id": self.weights_id,
            "optimizer_id": self.optimizer_id,
            "scheduler_id": self.scheduler_id,
            "tokenizer_id": self.tokenizer_id,
            "vocab_id": self.vocab_id,
            "cursor_id": self.cursor_id,
            "corpus_id": self.corpus_id,
            "split_id": self.split_id,
            "curriculum_id": self.curriculum_id,
            "loss_id": self.loss_id,
            "random_id": self.random_id,
            "env_id": self.env_id,
            "code_id": self.code_id,
            "compiler_id": self.compiler_id,
            "cursor_step": self.cursor_step,
            "lineage_id": self.lineage_id,
            "progress_id": self.progress_id,
        }
        if include_id:
            payload["binding_id"] = self.binding_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LearningCheckpointBinding":
        if not isinstance(payload, Mapping):
            raise LearningCheckpointError("learning checkpoint binding must be an object")
        _reject_promotion_authority(payload, noun="learning checkpoint binding")
        return cls(
            architecture_id=str(payload.get("architecture_id") or ""),
            weights_id=str(payload.get("weights_id") or ""),
            optimizer_id=str(payload.get("optimizer_id") or ""),
            scheduler_id=str(payload.get("scheduler_id") or ""),
            tokenizer_id=str(payload.get("tokenizer_id") or ""),
            vocab_id=str(payload.get("vocab_id") or ""),
            cursor_id=str(payload.get("cursor_id") or ""),
            corpus_id=str(payload.get("corpus_id") or ""),
            split_id=str(payload.get("split_id") or ""),
            curriculum_id=str(payload.get("curriculum_id") or ""),
            loss_id=str(payload.get("loss_id") or ""),
            random_id=str(payload.get("random_id") or ""),
            env_id=str(payload.get("env_id") or ""),
            code_id=str(payload.get("code_id") or ""),
            compiler_id=str(payload.get("compiler_id") or ""),
            cursor_step=payload.get("cursor_step", 0),  # type: ignore[arg-type]
            schema=str(payload.get("schema") or LEARNING_CHECKPOINT_BINDING_SCHEMA),
        )

    def replace_progress(
        self,
        *,
        weights_id: str | None = None,
        cursor_id: str | None = None,
        random_id: str | None = None,
        cursor_step: int | None = None,
    ) -> "LearningCheckpointBinding":
        return LearningCheckpointBinding(
            architecture_id=self.architecture_id,
            weights_id=self.weights_id if weights_id is None else weights_id,
            optimizer_id=self.optimizer_id,
            scheduler_id=self.scheduler_id,
            tokenizer_id=self.tokenizer_id,
            vocab_id=self.vocab_id,
            cursor_id=self.cursor_id if cursor_id is None else cursor_id,
            corpus_id=self.corpus_id,
            split_id=self.split_id,
            curriculum_id=self.curriculum_id,
            loss_id=self.loss_id,
            random_id=self.random_id if random_id is None else random_id,
            env_id=self.env_id,
            code_id=self.code_id,
            compiler_id=self.compiler_id,
            cursor_step=self.cursor_step if cursor_step is None else cursor_step,
        )


def assert_compatible_resume(
    stored: LearningCheckpointBinding,
    requested: LearningCheckpointBinding,
) -> None:
    """Reject an incompatible or rewound resume before any mutation."""

    if stored.lineage_id != requested.lineage_id:
        mismatched = [
            name
            for name in LINEAGE_BINDING_FIELDS
            if getattr(stored, name) != getattr(requested, name)
        ]
        raise IncompatibleResumeError(
            "incompatible resume lineage: " + ", ".join(mismatched or ("lineage_id",))
        )
    if requested.cursor_step < stored.cursor_step:
        raise IncompatibleResumeError("incompatible resume: cursor_step moved backwards")
    if requested.cursor_step == stored.cursor_step and requested.progress_id != stored.progress_id:
        raise IncompatibleResumeError("incompatible resume: progress forked at the same cursor")


def resume_decision(
    stored: LearningCheckpointBinding,
    requested: LearningCheckpointBinding,
) -> dict[str, Any]:
    """Return a content-addressed compatible-resume decision."""

    assert_compatible_resume(stored, requested)
    payload = {
        "schema": LEARNING_RESUME_DECISION_SCHEMA,
        "requirement_id": CAMPAIGN_DURABILITY_REQUIREMENT_ID,
        "compatible": True,
        "stored_binding_id": stored.binding_id,
        "requested_binding_id": requested.binding_id,
        "lineage_id": stored.lineage_id,
        "stored_progress_id": stored.progress_id,
        "requested_progress_id": requested.progress_id,
        "cursor_step": requested.cursor_step,
        "promotion_authority": False,
    }
    payload["decision_id"] = content_identity(payload)
    return payload


def checkpoint_state_payload(
    binding: LearningCheckpointBinding,
    *,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the state object persisted through the existing recovery store."""

    extras = dict(extra or {})
    _reject_promotion_authority(extras, noun="checkpoint state")
    reserved = set(LEARNING_CHECKPOINT_BINDING_FIELDS) | {
        "schema",
        "binding",
        "binding_id",
        "lineage_id",
        "progress_id",
        "cursor_step",
        "promotion_authority",
    }
    collision = reserved.intersection(extras)
    if collision:
        raise LearningCheckpointError(
            "checkpoint extra state collides with binding fields: " + ", ".join(sorted(collision))
        )
    return {
        "schema": LEARNING_CHECKPOINT_BINDING_SCHEMA,
        "binding": binding.to_dict(),
        "binding_id": binding.binding_id,
        "lineage_id": binding.lineage_id,
        "progress_id": binding.progress_id,
        "cursor_step": binding.cursor_step,
        "promotion_authority": False,
        "extra": extras,
    }


def binding_from_checkpoint_state(state: Mapping[str, Any]) -> LearningCheckpointBinding:
    """Decode a recovery-checkpoint state written by this adapter."""

    if not isinstance(state, Mapping):
        raise LearningCheckpointError("recovery checkpoint state must be an object")
    _reject_promotion_authority(state, noun="recovery checkpoint state")
    raw = state.get("binding")
    if isinstance(raw, Mapping):
        return LearningCheckpointBinding.from_dict(raw)
    payload = {name: state.get(name) for name in LEARNING_CHECKPOINT_BINDING_FIELDS}
    payload["cursor_step"] = state.get("cursor_step", 0)
    payload["schema"] = state.get("schema") or LEARNING_CHECKPOINT_BINDING_SCHEMA
    return LearningCheckpointBinding.from_dict(payload)


def semantic_roots_for(binding: LearningCheckpointBinding) -> dict[str, str]:
    """Project the binding into recovery semantic-root identities."""

    return {
        "architecture": binding.architecture_id,
        "tokenizer": binding.tokenizer_id,
        "vocab": binding.vocab_id,
        "corpus": binding.corpus_id,
        "split": binding.split_id,
        "curriculum": binding.curriculum_id,
        "loss": binding.loss_id,
        "env": binding.env_id,
        "code": binding.code_id,
        "compiler": binding.compiler_id,
        "lineage": binding.lineage_id,
    }


__all__ = (
    "CAMPAIGN_DURABILITY_REQUIREMENT_ID",
    "LEARNING_CHECKPOINT_BINDING_FIELDS",
    "LEARNING_CHECKPOINT_BINDING_SCHEMA",
    "LEASE_DEFAULT_DURATION_MS",
    "LEASE_DEFAULT_DURATION_SECONDS",
    "LEASE_DEFAULT_HEARTBEAT_MS",
    "LEASE_DEFAULT_HEARTBEAT_SECONDS",
    "LEASE_DEFAULT_MAX_ATTEMPTS",
    "LINEAGE_BINDING_FIELDS",
    "L3ResourceKind",
    "NAMED_L3_RESOURCES",
    "PROGRESS_BINDING_FIELDS",
    "IncompatibleResumeError",
    "LearningCheckpointBinding",
    "LearningCheckpointError",
    "PromotionMutationError",
    "StaleFenceError",
    "assert_compatible_resume",
    "assert_distinct_l3_lease_keys",
    "binding_from_checkpoint_state",
    "checkpoint_state_payload",
    "default_l3_lease_keys",
    "exclusive_lease_key",
    "resume_decision",
    "semantic_roots_for",
)
