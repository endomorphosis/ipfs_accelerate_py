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
    """Resolve MockIPFSClient from the canonical or legacy location."""
    try:
        from ipfs_accelerate_py.mcp.tools.mock_ipfs import MockIPFSClient as _MockIPFSClient  # type: ignore
        return _MockIPFSClient
    except Exception:
        pass

    # Canonical fallback stub when legacy module is removed.
    import random
    import string

    def _random_cid():
        letters = string.ascii_lowercase + string.digits
        return "Qm" + "".join(random.choice(letters) for _ in range(44))

    class _CanonicalMockIPFSClient:
        """Minimal in-memory MockIPFSClient for testing without ipfs_kit_py."""

        def __init__(self):
            self._files = {}
            self._pins = set()

        async def add(self, content, **_kwargs):
            cid = _random_cid()
            self._files[cid] = content if isinstance(content, bytes) else str(content).encode()
            return {"Hash": cid}

        async def cat(self, cid, **_kwargs):
            return self._files.get(cid, b"")

        async def pin_add(self, cid, **_kwargs):
            self._pins.add(cid)
            return {"Pins": [cid]}

        async def pin_rm(self, cid, **_kwargs):
            self._pins.discard(cid)
            return {"Pins": [cid]}

        async def id(self, **_kwargs):
            return {"ID": "QmMockPeerID", "Addresses": []}

    return _CanonicalMockIPFSClient


# Lazily resolved to avoid hard dependency on legacy mcp package.
_MockIPFSClientClass = None


class MockIPFSClient:
    """Canonical MockIPFSClient shim.

    Delegates to ``ipfs_accelerate_py.mcp.tools.mock_ipfs.MockIPFSClient``
    while that module exists; falls back to a built-in stub afterwards.

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
