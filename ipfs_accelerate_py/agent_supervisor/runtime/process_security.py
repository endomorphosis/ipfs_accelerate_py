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
import threading
from collections.abc import Mapping, MutableMapping
from typing import Final

PR_GET_DUMPABLE: Final = 3
PR_SET_DUMPABLE: Final = 4
STATE_AUTHORITY_CREDENTIAL_NAMES: Final = frozenset(
    {
        "IPFS_ACCELERATE_AGENT_QUACK_TOKEN",
        "IPFS_ACCELERATE_AGENT_OWNER_STATE_TOKEN",
    }
)
_CAPTURED_STATE_AUTHORITY_CREDENTIALS: dict[str, str] = {}
_CAPTURED_STATE_AUTHORITY_LOCK = threading.RLock()


class StateAuthorityProcessIsolationError(RuntimeError):
    """A credential-bearing process could not establish its kernel boundary."""


def env_secret_handle_target(secret_handle: str) -> str:
    """Return the environment variable named by an ``env://`` secret handle."""

    handle = str(secret_handle or "").strip()
    if not handle.startswith("env://"):
        return ""
    target = handle[len("env://") :].strip()
    if not target or not target.isidentifier():
        return ""
    return target


def forward_env_secret_handle_credentials(
    child_environment: MutableMapping[str, str],
    *,
    secret_handle: str,
    source_environment: Mapping[str, str] | None = None,
) -> MutableMapping[str, str]:
    """Copy an already-admitted ``env://`` credential into a trusted child.

    This never mints a token.  Provider children must still go through
    ``provider_subprocess_environment``, which scrubs these names.
    """

    target = env_secret_handle_target(secret_handle)
    if not target:
        return child_environment
    source = os.environ if source_environment is None else source_environment
    value = state_authority_credential(target, environment=source)
    if value:
        child_environment[target] = value
    return child_environment


def state_authority_credential(
    name: str,
    *,
    environment: Mapping[str, str] | None = None,
) -> str:
    """Resolve a trusted process credential without exposing it to children."""

    source = os.environ if environment is None else environment
    value = str(source.get(name, "") or "").strip()
    if value:
        return value
    if name not in STATE_AUTHORITY_CREDENTIAL_NAMES:
        return ""
    with _CAPTURED_STATE_AUTHORITY_LOCK:
        return str(_CAPTURED_STATE_AUTHORITY_CREDENTIALS.get(name, "") or "")


def state_authority_credentials_present(
    environment: Mapping[str, str] | None = None,
) -> bool:
    """Return whether an admitted raw state credential is present."""

    source = os.environ if environment is None else environment
    if any(
        bool(str(source.get(name, "") or "").strip())
        for name in STATE_AUTHORITY_CREDENTIAL_NAMES
    ):
        return True
    if environment is not None:
        return False
    with _CAPTURED_STATE_AUTHORITY_LOCK:
        return any(_CAPTURED_STATE_AUTHORITY_CREDENTIALS.values())


def establish_state_authority_process_boundary() -> bool:
    """Make the current process non-dumpable before it mints a credential."""

    if not sys.platform.startswith("linux"):
        raise StateAuthorityProcessIsolationError(
            "state authority requires a qualified Linux non-dumpable process"
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


def capture_state_authority_credentials() -> bool:
    """Harden, retain credentials in memory, and remove them from ``environ``.

    Ordinary subprocess APIs inherit ``os.environ`` when no explicit mapping is
    supplied.  Capturing at each trusted module entry therefore makes every
    unclassified child token-free by default.  The few sealed authority hops
    re-add the captured value through ``forward_env_secret_handle_credentials``.
    """

    present = {
        name: str(os.environ.get(name, "") or "").strip()
        for name in STATE_AUTHORITY_CREDENTIAL_NAMES
        if str(os.environ.get(name, "") or "").strip()
    }
    if not present:
        return False
    establish_state_authority_process_boundary()
    with _CAPTURED_STATE_AUTHORITY_LOCK:
        for name, value in present.items():
            prior = _CAPTURED_STATE_AUTHORITY_CREDENTIALS.get(name, "")
            if prior and prior != value:
                raise StateAuthorityProcessIsolationError(
                    "state-authority credential changed within one process"
                )
        _CAPTURED_STATE_AUTHORITY_CREDENTIALS.update(present)
        for name in present:
            os.environ.pop(name, None)
    return True


def harden_state_authority_process(
    environment: Mapping[str, str] | None = None,
) -> bool:
    """Make a credential-bearing Linux process non-dumpable, or fail closed.

    Returns ``False`` when no credential is present, so ordinary provider-free
    imports and hermetic tests retain their normal process behavior.
    """

    if not state_authority_credentials_present(environment):
        return False
    return establish_state_authority_process_boundary()


__all__ = (
    "PR_GET_DUMPABLE",
    "PR_SET_DUMPABLE",
    "STATE_AUTHORITY_CREDENTIAL_NAMES",
    "StateAuthorityProcessIsolationError",
    "capture_state_authority_credentials",
    "establish_state_authority_process_boundary",
    "env_secret_handle_target",
    "forward_env_secret_handle_credentials",
    "harden_state_authority_process",
    "state_authority_credential",
    "state_authority_credentials_present",
)
