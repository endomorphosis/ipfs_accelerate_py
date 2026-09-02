"""Native IPFS tools for unified mcp_server Wave A migration."""

from .native_ipfs_tools import (
    ipfs_files_add_file,
    ipfs_files_cat,
    ipfs_files_get_file,
    ipfs_files_list_files,
    ipfs_files_pin_file,
    ipfs_files_read,
    ipfs_files_unpin_file,
    ipfs_files_validate_cid,
    ipfs_files_write,
    ipfs_mkdir,
    ipfs_pin_add,
    ipfs_pin_rm,
    register_native_ipfs_tools,
)


def _load_mock_ipfs_client():
    """Resolve MockIPFSClient from the quarantined simulation namespace.

    PCPR-032: the former fallback stub minted Qm-prefixed random strings.
    Ordinary runtime cannot instantiate that generator. The simulation
    client mints canonical CIDv1 from retained bytes and is never live.
    """
    from ipfs_accelerate_py.compatibility.simulation.pseudo_cid import (
        MockIPFSClient as _MockIPFSClient,
    )

    return _MockIPFSClient


# Lazily resolved to avoid hard dependency on the simulation package at import.
_MockIPFSClientClass = None


class MockIPFSClient:
    """Canonical MockIPFSClient shim.

    Delegates to ``ipfs_accelerate_py.compatibility.simulation.pseudo_cid.MockIPFSClient``.
    Instantiation requires explicit simulation. Random Qm strings are not emitted.

    Migration note:
        Replace ``from ipfs_accelerate_py.mcp.tools.mock_ipfs import MockIPFSClient``
        with ``from ipfs_accelerate_py.mcp_server.tools.ipfs import MockIPFSClient``.
    """

    def __new__(cls, *args, **kwargs):
        global _MockIPFSClientClass
        if _MockIPFSClientClass is None:
            _MockIPFSClientClass = _load_mock_ipfs_client()
        return _MockIPFSClientClass(*args, **kwargs)


__all__ = [
    "ipfs_files_list_files",
    "ipfs_files_add_file",
    "ipfs_files_pin_file",
    "ipfs_files_unpin_file",
    "ipfs_files_get_file",
    "ipfs_files_cat",
    "ipfs_files_validate_cid",
    "ipfs_mkdir",
    "ipfs_pin_add",
    "ipfs_pin_rm",
    "ipfs_files_write",
    "ipfs_files_read",
    "register_native_ipfs_tools",
    "MockIPFSClient",
]
