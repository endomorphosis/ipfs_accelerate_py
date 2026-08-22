"""Supervisor-registry compatibility surface.

The implementation remains one :class:`FederationStateRepository`; this
module intentionally does not create a second state repository or database
authority.
"""

from .registry import (
    FederationRepositoryConflict,
    FederationRepositoryError,
    FederationRepositoryNotFound,
    FederationStateRepository,
)

SupervisorRegistry = FederationStateRepository

__all__ = [
    "FederationRepositoryConflict",
    "FederationRepositoryError",
    "FederationRepositoryNotFound",
    "FederationStateRepository",
    "SupervisorRegistry",
]
