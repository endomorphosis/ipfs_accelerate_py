"""
Mock implementations of IPFS functionality for the MCP server.

PCPR-032: this compatibility shim no longer emits Qm-prefixed random strings
or hexadecimal slices as CIDs. Ordinary runtime cannot instantiate the mock
client. Explicit simulation uses canonical CIDv1 from retained bytes and is
never live.


.. deprecated::
    This module has been migrated to the canonical runtime at
    ``ipfs_accelerate_py.mcp_server.tools.ipfs`` and the simulation namespace
    ``ipfs_accelerate_py.compatibility.simulation.pseudo_cid``.
    Import from those modules instead. This file is a compatibility shim only.
"""

from __future__ import annotations

from ipfs_accelerate_py.compatibility.simulation.pseudo_cid import (
    MOCK_IPFS_NAMESPACE,
    MockIPFSClient,
    PseudoCidIdentityError,
    UnavailableIpfsClient,
    instantiate_mock_ipfs_client,
    load_ordinary_ipfs_client,
    mint_canonical_cid,
    random_cid,
)

__all__ = (
    "MOCK_IPFS_NAMESPACE",
    "MockIPFSClient",
    "PseudoCidIdentityError",
    "UnavailableIpfsClient",
    "instantiate_mock_ipfs_client",
    "load_ordinary_ipfs_client",
    "mint_canonical_cid",
    "random_cid",
)
