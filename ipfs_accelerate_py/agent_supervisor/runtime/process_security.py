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
import stat
import sys
from collections.abc import Mapping, MutableMapping
from typing import Final

PR_GET_DUMPABLE: Final = 3
PR_SET_DUMPABLE: Final = 4
STATE_AUTHORITY_CREDENTIAL_NAMES: Final = frozenset(
    {
        "IPFS_ACCELERATE_AGENT_QUACK_TOKEN",
        "IPFS_ACCELERATE_AGENT_OWNER_STATE_TOKEN",
        "IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SECRET_FD",
    }
)

STATE_AUTHORITY_DESCRIPTOR_ENV_NAMES: Final = frozenset(
    {"IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SECRET_FD"}
)
STATE_AUTHORITY_DESCRIPTOR_SOCKET_ENV: Final = (
    "IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SOCKET"
)


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
    value = str(source.get(target, "") or "").strip()
    if value:
        child_environment[target] = value
    return child_environment


def state_authority_credentials_present(
    environment: Mapping[str, str] | None = None,
) -> bool:
    """Return whether an admitted raw state credential is present."""

    source = os.environ if environment is None else environment
    return any(bool(str(source.get(name, "") or "").strip()) for name in STATE_AUTHORITY_CREDENTIAL_NAMES)


def state_authority_pass_fds(
    environment: Mapping[str, str] | None = None,
) -> tuple[int, ...]:
    """Return validated inherited descriptors for trusted control children."""

    source = os.environ if environment is None else environment
    broker_socket_present = bool(
        str(source.get(STATE_AUTHORITY_DESCRIPTOR_SOCKET_ENV, "") or "").strip()
    )
    broker_descriptor_present = bool(
        str(
            source.get(
                "IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SECRET_FD",
                "",
            )
            or ""
        ).strip()
    )
    if broker_socket_present != broker_descriptor_present:
        raise StateAuthorityProcessIsolationError(
            "state-authority broker binding is incomplete"
        )
    descriptors: list[int] = []
    for name in STATE_AUTHORITY_DESCRIPTOR_ENV_NAMES:
        raw = str(source.get(name, "") or "").strip()
        if not raw:
            continue
        if not raw.isascii() or not raw.isdecimal():
            raise StateAuthorityProcessIsolationError(
                "state-authority descriptor binding is invalid"
            )
        descriptor = int(raw)
        if descriptor < 3 or descriptor > 1_048_576:
            raise StateAuthorityProcessIsolationError(
                "state-authority descriptor binding is invalid"
            )
        try:
            observed = os.fstat(descriptor)
        except OSError as exc:
            raise StateAuthorityProcessIsolationError(
                "state-authority descriptor is unavailable"
            ) from exc
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_uid != os.geteuid()
            or not 32 <= observed.st_size <= 256
        ):
            raise StateAuthorityProcessIsolationError(
                "state-authority descriptor is not a bounded owner memfd"
            )
        if not sys.platform.startswith("linux"):
            raise StateAuthorityProcessIsolationError(
                "state-authority descriptor requires a qualified Linux memfd"
            )
        import fcntl

        required_seals = (
            int(getattr(fcntl, "F_SEAL_SEAL", 0x0001))
            | int(getattr(fcntl, "F_SEAL_SHRINK", 0x0002))
            | int(getattr(fcntl, "F_SEAL_GROW", 0x0004))
            | int(getattr(fcntl, "F_SEAL_WRITE", 0x0008))
        )
        try:
            observed_seals = int(
                fcntl.fcntl(
                    descriptor,
                    int(getattr(fcntl, "F_GET_SEALS", 1034)),
                )
            )
        except OSError as exc:
            raise StateAuthorityProcessIsolationError(
                "state-authority descriptor is not sealed"
            ) from exc
        if observed_seals & required_seals != required_seals:
            raise StateAuthorityProcessIsolationError(
                "state-authority descriptor is not sealed"
            )
        descriptors.append(descriptor)
    return tuple(sorted(set(descriptors)))


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
    "STATE_AUTHORITY_DESCRIPTOR_ENV_NAMES",
    "STATE_AUTHORITY_DESCRIPTOR_SOCKET_ENV",
    "StateAuthorityProcessIsolationError",
    "env_secret_handle_target",
    "forward_env_secret_handle_credentials",
    "harden_state_authority_process",
    "state_authority_credentials_present",
    "state_authority_pass_fds",
)
