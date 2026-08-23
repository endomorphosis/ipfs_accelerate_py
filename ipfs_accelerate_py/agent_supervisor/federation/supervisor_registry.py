"""Supervisor-registry compatibility surface.

The implementation remains one :class:`FederationStateRepository`; this
module intentionally does not create a second state repository or database
authority. Shard compilation is a bounded specialization helper over that
same owner, not a second assignment authority.
"""

from collections.abc import Mapping, Sequence

from .contracts import FederationBinding
from .registry import (
    FederationRepositoryConflict,
    FederationRepositoryError,
    FederationRepositoryNotFound,
    FederationStateRepository,
)
from .sharding import (
    CompiledShardPlan,
    ShardWork,
    SupervisorSpecializationBound,
    compile_supervisor_shards,
)

SupervisorRegistry = FederationStateRepository


def compile_registered_shards(
    work: Sequence[ShardWork],
    specializations: Sequence[SupervisorSpecializationBound],
    *,
    binding: FederationBinding,
    fencing_epoch: int = 1,
    assignment_revision: int = 1,
    ducklake_receipt: Mapping[str, object] | None = None,
) -> CompiledShardPlan:
    """Compile conflict-free shards without opening a second registry."""

    return compile_supervisor_shards(
        work,
        specializations,
        binding=binding,
        fencing_epoch=fencing_epoch,
        assignment_revision=assignment_revision,
        ducklake_receipt=ducklake_receipt,
    )


__all__ = [
    "FederationRepositoryConflict",
    "FederationRepositoryError",
    "FederationRepositoryNotFound",
    "FederationStateRepository",
    "SupervisorRegistry",
    "compile_registered_shards",
]
