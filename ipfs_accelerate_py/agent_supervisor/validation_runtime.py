"""Fail-closed subprocess policy for supervisor validation commands.

Validation commands are reviewed shell text, but the process environment that
launches the supervisor is not part of that text.  Keep profile hooks, secrets,
and transient executable search paths outside the validation boundary.
"""

from __future__ import annotations

import os
import stat
import sys
from collections.abc import Mapping
from pathlib import Path

VALIDATION_PATH_ENV = "IPFS_ACCELERATE_AGENT_VALIDATION_PATH"
VALIDATION_PYTHON_ENV = "IPFS_ACCELERATE_AGENT_VALIDATION_PYTHON"
VALIDATION_NPM_CACHE_ENV = "IPFS_ACCELERATE_AGENT_VALIDATION_NPM_CACHE"
_CHILD_PYTHON_ENV = "IPFS_ACCELERATE_VALIDATION_PYTHON_EXECUTABLE"
_NEUTRAL_HOME = "/nonexistent/ipfs-accelerate-validation"

# These values affect deterministic/offline validation without carrying the
# provider, wallet, registry, signing, or cloud credentials commonly present
# in a long-running supervisor environment.
VALIDATION_ENVIRONMENT_ALLOWLIST = frozenset(
    {
        "CARGO_NET_OFFLINE",
        "CI",
        "HF_DATASETS_OFFLINE",
        "HF_HUB_OFFLINE",
        "IPFS_ACCEL_SKIP_CORE",
        "IPFS_KIT_DISABLE",
        "LANG",
        "LANGUAGE",
        "LC_ALL",
        "NODE_ENV",
        "NPM_CONFIG_OFFLINE",
        "PIP_NO_INDEX",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONHASHSEED",
        "TRANSFORMERS_OFFLINE",
        "TZ",
        "WORLD_AID_EXTERNAL_CALLS_ENABLED",
        "WORLD_AID_WLD_TRANSFERS_ENABLED",
        "WORLD_ID_ENABLED",
    }
)

_SYSTEM_VALIDATION_PATHS = (
    Path("/usr/local/sbin"),
    Path("/usr/local/bin"),
    Path("/usr/sbin"),
    Path("/usr/bin"),
    Path("/sbin"),
    Path("/bin"),
)


class ValidationRuntimeError(ValueError):
    """Raised before execution when the validation runtime policy is invalid."""


def _reject_writable_path(path: Path, *, source: str) -> None:
    inspected = path
    while True:
        try:
            mode = inspected.stat().st_mode
        except OSError as exc:
            raise ValidationRuntimeError(
                f"{source} path cannot be inspected: {inspected}"
            ) from exc
        if mode & (stat.S_IWGRP | stat.S_IWOTH) or os.access(inspected, os.W_OK):
            relationship = "path" if inspected == path else "ancestor"
            raise ValidationRuntimeError(
                f"{source} {relationship} must not be writable by the "
                f"validation user or group: {inspected}"
            )
        parent = inspected.parent
        if parent == inspected:
            break
        inspected = parent


def _validated_path_entries(
    value: str,
    *,
    source: str,
) -> tuple[str, ...]:
    entries: list[str] = []
    for raw_entry in value.split(os.pathsep):
        if not raw_entry:
            raise ValidationRuntimeError(f"{source} contains an empty PATH entry")
        path = Path(raw_entry)
        if not path.is_absolute():
            raise ValidationRuntimeError(
                f"{source} entries must be absolute: {raw_entry!r}"
            )
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise ValidationRuntimeError(
                f"{source} entry is unavailable: {raw_entry!r}"
            ) from exc
        if not resolved.is_dir():
            raise ValidationRuntimeError(
                f"{source} entry is not a directory: {raw_entry!r}"
            )
        _reject_writable_path(resolved, source=source)
        rendered = str(resolved)
        if rendered not in entries:
            entries.append(rendered)
    if not entries:
        raise ValidationRuntimeError(f"{source} must contain at least one directory")
    return tuple(entries)


def validation_executable_path(
    environment: Mapping[str, object] | None = None,
) -> str:
    """Return the exact executable search path for validation children.

    The inherited ``PATH`` is never reused.  Operators that need an approved
    non-system toolchain must supply its complete absolute path list through
    :data:`VALIDATION_PATH_ENV`; otherwise only standard system directories are
    exposed.
    """

    source = os.environ if environment is None else environment
    override = str(source.get(VALIDATION_PATH_ENV) or "").strip()
    if override:
        return os.pathsep.join(
            _validated_path_entries(override, source=VALIDATION_PATH_ENV)
        )

    candidates = list(_SYSTEM_VALIDATION_PATHS)

    entries: list[str] = []
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        resolved = candidate.resolve()
        try:
            _reject_writable_path(resolved, source="default validation PATH")
        except ValidationRuntimeError:
            continue
        rendered = str(resolved)
        if rendered not in entries:
            entries.append(rendered)
    if not entries:
        raise ValidationRuntimeError(
            "no trusted executable directories are available for validation"
        )
    return os.pathsep.join(entries)


def validation_python_executable(
    environment: Mapping[str, object] | None = None,
) -> str:
    """Return one non-writable Python executable for the ``python`` command."""

    source = os.environ if environment is None else environment
    configured = str(source.get(VALIDATION_PYTHON_ENV) or "").strip()
    candidate = Path(configured) if configured else Path(sys.executable).resolve()
    if not candidate.is_absolute():
        raise ValidationRuntimeError(
            f"{VALIDATION_PYTHON_ENV} must be an absolute executable path"
        )
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ValidationRuntimeError(
            f"validation Python is unavailable: {candidate}"
        ) from exc
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ValidationRuntimeError(
            f"validation Python is not executable: {candidate}"
        )
    _reject_writable_path(candidate.parent.resolve(), source="validation Python")
    _reject_writable_path(resolved, source="validation Python")
    # Preserve an explicitly approved read-only virtual-environment launcher;
    # otherwise use the canonical system executable rather than a user-owned
    # shim from the supervisor's inherited PATH.
    return str(candidate if configured else resolved)


def _approved_directory(
    source: Mapping[str, object],
    variable: str,
) -> str | None:
    raw = str(source.get(variable) or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        raise ValidationRuntimeError(f"{variable} must be an absolute directory")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValidationRuntimeError(f"{variable} is unavailable: {path}") from exc
    if not resolved.is_dir():
        raise ValidationRuntimeError(f"{variable} is not a directory: {path}")
    return str(resolved)


def build_validation_environment(
    environment: Mapping[str, object] | None = None,
) -> dict[str, str]:
    """Build the complete allowlisted environment for a validation child."""

    source = os.environ if environment is None else environment
    result = {
        key: str(source[key])
        for key in sorted(VALIDATION_ENVIRONMENT_ALLOWLIST)
        if key in source and source[key] is not None
    }
    result.update(
        {
            _CHILD_PYTHON_ENV: validation_python_executable(source),
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_PAGER": "cat",
            "GIT_TERMINAL_PROMPT": "0",
            "HOME": _NEUTRAL_HOME,
            "NO_COLOR": "1",
            "NPM_CONFIG_GLOBALCONFIG": "/dev/null",
            "NPM_CONFIG_USERCONFIG": "/dev/null",
            "PAGER": "cat",
            "PATH": validation_executable_path(source),
            "PIP_CONFIG_FILE": "/dev/null",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INPUT": "1",
            "PYTHONNOUSERSITE": "1",
            "TERM": "dumb",
            "XDG_CACHE_HOME": _NEUTRAL_HOME,
            "XDG_CONFIG_HOME": _NEUTRAL_HOME,
            "XDG_DATA_HOME": _NEUTRAL_HOME,
            "XDG_STATE_HOME": _NEUTRAL_HOME,
        }
    )
    npm_cache = _approved_directory(source, VALIDATION_NPM_CACHE_ENV)
    if npm_cache is not None:
        result["NPM_CONFIG_CACHE"] = npm_cache
    result.setdefault("LANG", "C")
    result.setdefault("LC_ALL", "C")
    result.setdefault("PYTHONHASHSEED", "0")
    result.setdefault("TZ", "UTC")
    return result


def validation_shell_command(command: str) -> list[str]:
    """Return a non-login, non-interactive Bash invocation for reviewed text."""

    text = str(command)
    if not text.strip():
        raise ValidationRuntimeError("validation command must not be empty")
    guarded = (
        f'python() {{ "${{{_CHILD_PYTHON_ENV}}}" "$@"; }}; '
        "readonly -f python; "
        f"{text}"
    )
    return ["/bin/bash", "--noprofile", "--norc", "-c", guarded]
