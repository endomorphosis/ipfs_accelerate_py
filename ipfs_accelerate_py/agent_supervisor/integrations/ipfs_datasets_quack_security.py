"""Thin high-level adapter to datasets ``quack_security``.

Datasets remains the server-side authorization authority. This module does not
reimplement policy; it only exposes the public datasets surface to the
accelerate daemon integration path.
"""

from __future__ import annotations

from ipfs_datasets_py.duckdb_control import quack_security as datasets_quack_security

AUTHORITY = "ipfs_datasets_py.duckdb_control.quack_security"
ADAPTER = "lgswf/ipfs-datasets-quack-security-adapter@1"


def scoped_authority():
    """Return the datasets module that owns Quack scoped authorization."""

    return datasets_quack_security
