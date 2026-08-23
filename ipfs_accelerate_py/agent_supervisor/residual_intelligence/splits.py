"""Lineage-safe grouped splitting for residual distillation examples."""

# Python 3.8 support requires ``str, Enum`` rather than ``enum.StrEnum``.
# ruff: noqa: UP042

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

from .contracts import (
    ResidualIntelligenceError,
    bounded_int,
    canonical_id,
    required_text,
    text_tuple,
)
from .corpus import ResidualDistillationExample
from .rights import LeakageAudit

SEMANTIC_SPLIT_POLICY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-semantic-split-policy@1"
)
SEMANTIC_SPLIT_MANIFEST_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-semantic-split-manifest@1"
)


class SplitPartition(str, Enum):
    TRAIN = "training"
    DEVELOPMENT = "development"
    HELD_OUT = "held_out"
    ADVERSARIAL = "adversarial"


@dataclass(frozen=True)
class SemanticSplitPolicy:
    """Frozen component-level split policy; never a random row split."""

    policy_id: str
    seed_identity: str
    train_basis_points: int = 7000
    development_basis_points: int = 1000
    held_out_basis_points: int = 1000
    adversarial_basis_points: int = 1000
    forced_development_groups: tuple[str, ...] = ()
    forced_held_out_groups: tuple[str, ...] = ()
    forced_adversarial_groups: tuple[str, ...] = ()
    schema: str = SEMANTIC_SPLIT_POLICY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SEMANTIC_SPLIT_POLICY_SCHEMA:
            raise ResidualIntelligenceError("unsupported semantic split policy schema")
        object.__setattr__(self, "policy_id", required_text(self.policy_id, "policy_id"))
        object.__setattr__(
            self, "seed_identity", required_text(self.seed_identity, "seed_identity")
        )
        for field in (
            "train_basis_points",
            "development_basis_points",
            "held_out_basis_points",
            "adversarial_basis_points",
        ):
            object.__setattr__(
                self,
                field,
                bounded_int(getattr(self, field), field, minimum=0, maximum=10_000),
            )
        if (
            sum(
                (
                    self.train_basis_points,
                    self.development_basis_points,
                    self.held_out_basis_points,
                    self.adversarial_basis_points,
                )
            )
            != 10_000
        ):
            raise ResidualIntelligenceError("split basis points must sum to 10000")
        forced_sets: list[set[str]] = []
        for field in (
            "forced_development_groups",
            "forced_held_out_groups",
            "forced_adversarial_groups",
        ):
            values = text_tuple(getattr(self, field), field)
            object.__setattr__(self, field, values)
            forced_sets.append(set(values))
        if any(forced_sets[i] & forced_sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise ResidualIntelligenceError("a forced split group names multiple partitions")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "policy_id": self.policy_id,
            "seed_identity": self.seed_identity,
            "train_basis_points": self.train_basis_points,
            "development_basis_points": self.development_basis_points,
            "held_out_basis_points": self.held_out_basis_points,
            "adversarial_basis_points": self.adversarial_basis_points,
            "forced_development_groups": list(self.forced_development_groups),
            "forced_held_out_groups": list(self.forced_held_out_groups),
            "forced_adversarial_groups": list(self.forced_adversarial_groups),
        }


@dataclass(frozen=True)
class SemanticSplitAssignment:
    example_identity: str
    component_id: str
    split_group: str
    partition: SplitPartition
    hidden_from_training: bool

    def __post_init__(self) -> None:
        for field in ("example_identity", "component_id", "split_group"):
            object.__setattr__(self, field, required_text(getattr(self, field), field))
        object.__setattr__(self, "partition", SplitPartition(self.partition))
        expected_hidden = self.partition in {
            SplitPartition.HELD_OUT,
            SplitPartition.ADVERSARIAL,
        }
        if (
            type(self.hidden_from_training) is not bool
            or self.hidden_from_training != expected_hidden
        ):
            raise ResidualIntelligenceError(
                "hidden_from_training must match held-out/adversarial partition"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_identity": self.example_identity,
            "component_id": self.component_id,
            "split_group": self.split_group,
            "partition": self.partition.value,
            "hidden_from_training": self.hidden_from_training,
        }


@dataclass(frozen=True)
class SemanticSplitManifest:
    policy: SemanticSplitPolicy
    assignments: tuple[SemanticSplitAssignment, ...]
    hidden_test_commitment: str
    schema: str = SEMANTIC_SPLIT_MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SEMANTIC_SPLIT_MANIFEST_SCHEMA:
            raise ResidualIntelligenceError("unsupported semantic split manifest schema")
        if not isinstance(self.policy, SemanticSplitPolicy):
            raise ResidualIntelligenceError("split manifest requires a typed policy")
        assignments = tuple(self.assignments)
        if any(not isinstance(item, SemanticSplitAssignment) for item in assignments):
            raise ResidualIntelligenceError("split assignments must be typed records")
        ids = [item.example_identity for item in assignments]
        if len(set(ids)) != len(ids):
            raise ResidualIntelligenceError("an example appears more than once in a split")
        component_partitions: dict[str, set[SplitPartition]] = defaultdict(set)
        for item in assignments:
            component_partitions[item.component_id].add(item.partition)
        if any(len(value) != 1 for value in component_partitions.values()):
            raise ResidualIntelligenceError("semantic lineage crosses split partitions")
        object.__setattr__(self, "assignments", assignments)
        object.__setattr__(
            self,
            "hidden_test_commitment",
            required_text(self.hidden_test_commitment, "hidden_test_commitment"),
        )

    @property
    def split_root(self) -> str:
        return canonical_id(self.to_dict(include_root=False))

    def to_dict(self, *, include_root: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "policy": self.policy.to_dict(),
            "assignments": [item.to_dict() for item in self.assignments],
            "hidden_test_commitment": self.hidden_test_commitment,
        }
        if include_root:
            result["split_root"] = self.split_root
        return result

    def leakage_audit(self) -> LeakageAudit:
        component_partition: dict[str, SplitPartition] = {}
        for item in self.assignments:
            component_partition[item.component_id] = item.partition
        counts = Counter(component_partition.values())
        duplicate_count = len(self.assignments) - len(
            {item.example_identity for item in self.assignments}
        )
        cross_count = sum(
            1
            for component in {item.component_id for item in self.assignments}
            if len({item.partition for item in self.assignments if item.component_id == component})
            > 1
        )
        passed = (
            cross_count == 0
            and duplicate_count == 0
            and counts[SplitPartition.HELD_OUT] > 0
            and counts[SplitPartition.ADVERSARIAL] > 0
        )
        return LeakageAudit(
            split_root=self.split_root,
            grouping_policy_id=self.policy.policy_id,
            train_group_count=counts[SplitPartition.TRAIN],
            development_group_count=counts[SplitPartition.DEVELOPMENT],
            holdout_group_count=counts[SplitPartition.HELD_OUT],
            adversarial_group_count=counts[SplitPartition.ADVERSARIAL],
            cross_partition_group_count=cross_count,
            duplicate_example_count=duplicate_count,
            hidden_test_commitment=self.hidden_test_commitment,
            hidden_test_bodies_accessed=False,
            passed=passed,
        )


class _UnionFind:
    def __init__(self, values: Sequence[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        keep, merge = sorted((left_root, right_root))
        self.parent[merge] = keep


def _component_groups(
    examples: Sequence[ResidualDistillationExample],
) -> dict[str, tuple[str, ...]]:
    ids = [item.example_identity for item in examples]
    union = _UnionFind(ids)
    by_lineage: dict[str, list[str]] = defaultdict(list)
    for item in examples:
        for lineage in {item.split_group, *item.semantic_lineage}:
            by_lineage[lineage].append(item.example_identity)
    for members in by_lineage.values():
        for member in members[1:]:
            union.union(members[0], member)
    components: dict[str, list[str]] = defaultdict(list)
    for example_id in ids:
        components[union.find(example_id)].append(example_id)
    return {root: tuple(sorted(values)) for root, values in components.items()}


def _partition_for(
    *,
    policy: SemanticSplitPolicy,
    component_id: str,
    group_names: set[str],
    has_adversarial_example: bool,
) -> SplitPartition:
    forced: list[SplitPartition] = []
    if group_names & set(policy.forced_development_groups):
        forced.append(SplitPartition.DEVELOPMENT)
    if group_names & set(policy.forced_held_out_groups):
        forced.append(SplitPartition.HELD_OUT)
    if group_names & set(policy.forced_adversarial_groups):
        forced.append(SplitPartition.ADVERSARIAL)
    if len(set(forced)) > 1:
        raise ResidualIntelligenceError("one semantic component has conflicting forced splits")
    if forced:
        return forced[0]
    if has_adversarial_example:
        return SplitPartition.ADVERSARIAL
    digest = hashlib.sha256(f"{policy.seed_identity}\x00{component_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 10_000
    train_end = policy.train_basis_points
    development_end = train_end + policy.development_basis_points
    heldout_end = development_end + policy.held_out_basis_points
    if bucket < train_end:
        return SplitPartition.TRAIN
    if bucket < development_end:
        return SplitPartition.DEVELOPMENT
    if bucket < heldout_end:
        return SplitPartition.HELD_OUT
    return SplitPartition.ADVERSARIAL


def semantic_lineage_split(
    examples: Sequence[ResidualDistillationExample],
    *,
    policy: SemanticSplitPolicy,
    hidden_test_commitment: str,
) -> SemanticSplitManifest:
    """Split connected semantic-lineage components as indivisible units."""

    typed = tuple(examples)
    if not typed:
        raise ResidualIntelligenceError("cannot split an empty corpus")
    if any(not isinstance(item, ResidualDistillationExample) for item in typed):
        raise ResidualIntelligenceError("semantic split requires typed examples")
    components = _component_groups(typed)
    example_by_id = {item.example_identity: item for item in typed}
    assignments: list[SemanticSplitAssignment] = []
    for _root, member_ids in sorted(components.items()):
        members = [example_by_id[item] for item in member_ids]
        group_names = {
            lineage for item in members for lineage in {item.split_group, *item.semantic_lineage}
        }
        component_id = canonical_id(
            {"semantic_lineage": sorted(group_names), "member_ids": list(member_ids)}
        )
        partition = _partition_for(
            policy=policy,
            component_id=component_id,
            group_names=group_names,
            has_adversarial_example=any(item.adversarial for item in members),
        )
        for item in sorted(members, key=lambda candidate: candidate.example_identity):
            assignments.append(
                SemanticSplitAssignment(
                    example_identity=item.example_identity,
                    component_id=component_id,
                    split_group=item.split_group,
                    partition=partition,
                    hidden_from_training=partition
                    in {SplitPartition.HELD_OUT, SplitPartition.ADVERSARIAL},
                )
            )
    return SemanticSplitManifest(
        policy=policy,
        assignments=tuple(assignments),
        hidden_test_commitment=hidden_test_commitment,
    )


def assert_training_view_excludes_hidden(
    manifest: SemanticSplitManifest,
    example_ids: Sequence[str],
) -> None:
    hidden = {item.example_identity for item in manifest.assignments if item.hidden_from_training}
    exposed = hidden & set(text_tuple(tuple(example_ids), "example_ids"))
    if exposed:
        raise ResidualIntelligenceError("training view includes hidden split examples")


__all__ = (
    "SEMANTIC_SPLIT_MANIFEST_SCHEMA",
    "SEMANTIC_SPLIT_POLICY_SCHEMA",
    "SemanticSplitAssignment",
    "SemanticSplitManifest",
    "SemanticSplitPolicy",
    "SplitPartition",
    "assert_training_view_excludes_hidden",
    "semantic_lineage_split",
)
