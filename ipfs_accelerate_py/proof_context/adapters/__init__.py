"""Public, closed adapter surface for Proof-Carrying Context Engine v0.1."""

from ipfs_accelerate_py.proof_context.adapters.base import (
    AdapterResult,
    CancellationToken,
    CodingAgentAdapter,
    execute_propose,
)
from ipfs_accelerate_py.proof_context.adapters.registry import (
    ADAPTER_NAMES,
    AdapterConfiguration,
    AdapterRegistry,
    DEFAULT_ADAPTER_REGISTRY,
    REGISTRY_DESCRIPTOR,
    REGISTRY_DESCRIPTOR_CID,
    adapter_registry_descriptor,
    create_adapter,
)

__all__ = [
    "ADAPTER_NAMES",
    "AdapterConfiguration",
    "AdapterRegistry",
    "AdapterResult",
    "CancellationToken",
    "CodingAgentAdapter",
    "DEFAULT_ADAPTER_REGISTRY",
    "REGISTRY_DESCRIPTOR",
    "REGISTRY_DESCRIPTOR_CID",
    "adapter_registry_descriptor",
    "create_adapter",
    "execute_propose",
]
