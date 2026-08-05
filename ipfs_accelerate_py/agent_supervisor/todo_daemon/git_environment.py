"""Fail-closed environment construction for repository-trust Git commands."""

from __future__ import annotations

import os
from collections.abc import Mapping


def sanitized_git_environment(
    source: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Remove every ambient ``GIT_*`` input and add only safe controls.

    Repository routing, alternate object stores, replacement refs, legacy
    grafts, indexes, diff drivers, and indexed config injection are all
    attacker-controlled inputs when inherited from a long-running supervisor.
    Trust-boundary Git calls therefore inherit ordinary process variables but
    no Git variable.
    """

    inherited = os.environ if source is None else source
    environment = {
        str(key): str(value)
        for key, value in inherited.items()
        if not str(key).upper().startswith("GIT_")
    }
    environment.update(
        {
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_CONFIG_COUNT": "0",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_GRAFT_FILE": os.devnull,
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
            "LANG": "C",
            "LC_ALL": "C",
        }
    )
    return environment


__all__ = ["sanitized_git_environment"]
