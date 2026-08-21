"""Kernel boundary for processes that retain state-authority credentials.

Implementation providers run under the same host account as the supervisor in
the current deployment profile.  Environment scrubbing alone is therefore not
enough: a same-UID child can ordinarily read a dumpable parent's
``/proc/<pid>/environ``.  Trusted control processes call this module before
they spawn provider code.  Linux then denies same-UID process introspection,
while ordinary provider children receive no state credential in their own
environment.

This is an isolation boundary, not an authorization decision.  Typed owner
commands and canonical repository validation remain mandatory.
"""

from __future__ import annotations

import ctypes
import os
import sys
from collections.abc import Mapping
from typing import Final

PR_GET_DUMPABLE: Final = 3
PR_SET_DUMPABLE: Final = 4
STATE_AUTHORITY_CREDENTIAL_NAMES: Final = frozenset(
    {
        "IPFS_ACCELERATE_AGENT_QUACK_TOKEN",
        "IPFS_ACCELERATE_AGENT_OWNER_STATE_TOKEN",
    }
)


class StateAuthorityProcessIsolationError(RuntimeError):
    """A credential-bearing process could not establish its kernel boundary."""


def state_authority_credentials_present(
    environment: Mapping[str, str] | None = None,
) -> bool:
    """Return whether an admitted raw state credential is present."""

    source = os.environ if environment is None else environment
    return any(bool(str(source.get(name, "") or "").strip()) for name in STATE_AUTHORITY_CREDENTIAL_NAMES)


def harden_state_authority_process(
    environment: Mapping[str, str] | None = None,
) -> bool:
    """Make a credential-bearing Linux process non-dumpable, or fail closed.

    Returns ``False`` when no credential is present, so ordinary provider-free
    imports and hermetic tests retain their normal process behavior.
    """

    if not state_authority_credentials_present(environment):
        return False
    if not sys.platform.startswith("linux"):
        raise StateAuthorityProcessIsolationError(
            "state credentials require a qualified Linux non-dumpable process"
        )
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise StateAuthorityProcessIsolationError(
            f"PR_SET_DUMPABLE failed with errno {error_number}"
        )
    if libc.prctl(PR_GET_DUMPABLE, 0, 0, 0, 0) != 0:
        raise StateAuthorityProcessIsolationError(
            "state-authority process remained dumpable"
        )
    return True


__all__ = (
    "PR_GET_DUMPABLE",
    "PR_SET_DUMPABLE",
    "STATE_AUTHORITY_CREDENTIAL_NAMES",
    "StateAuthorityProcessIsolationError",
    "harden_state_authority_process",
    "state_authority_credentials_present",
)
