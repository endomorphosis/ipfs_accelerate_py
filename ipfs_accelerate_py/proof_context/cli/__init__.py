"""Public, dependency-free surface for the proof-context module CLI.

Importing this package never opens a repository or starts runtime work.  The
``python -m ipfs_accelerate_py.proof_context.cli`` entry point is intentionally
kept separate so packaging can add a console script later without changing the
command contract.
"""

from .app import CONTRACT_VERSION, INTERFACE, PROG

__all__ = ["CONTRACT_VERSION", "INTERFACE", "PROG"]
