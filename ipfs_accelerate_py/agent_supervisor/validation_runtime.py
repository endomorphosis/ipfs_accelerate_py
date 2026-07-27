"""Fail-closed subprocess policy for supervisor validation commands.

Validation commands are reviewed shell text, but the process environment that
launches the supervisor is not part of that text.  Keep profile hooks, secrets,
and transient executable search paths outside the validation boundary.
"""

from __future__ import annotations

import os
import shlex
import site
import stat
import sys
import sysconfig
from collections.abc import Mapping, Sequence
from pathlib import Path

VALIDATION_PATH_ENV = "IPFS_ACCELERATE_AGENT_VALIDATION_PATH"
VALIDATION_PYTHON_ENV = "IPFS_ACCELERATE_AGENT_VALIDATION_PYTHON"
VALIDATION_PYTHONPATH_ENV = "IPFS_ACCELERATE_AGENT_VALIDATION_PYTHONPATH"
VALIDATION_NPM_CACHE_ENV = "IPFS_ACCELERATE_AGENT_VALIDATION_NPM_CACHE"
_CHILD_PYTHON_ENV = "IPFS_ACCELERATE_VALIDATION_PYTHON_EXECUTABLE"
_NEUTRAL_HOME = "/nonexistent/ipfs-accelerate-validation"
_NPM_DISABLED_USER_CONFIG = "/dev/null/npmrc"

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
    """Return the canonical approved Python executable for validation commands.

    Never execute the original launcher spelling after validating only its
    target: a launcher in a writable directory could be replaced between the
    check and ``exec``.  The child instead executes the already-resolved,
    non-writable system binary.  Approved package roots are supplied
    separately by :func:`_runtime_python_path_entries`.
    """

    source = os.environ if environment is None else environment
    configured = str(source.get(VALIDATION_PYTHON_ENV) or "").strip()
    candidate = Path(configured) if configured else Path(sys.executable)
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
    _reject_writable_path(resolved, source="validation Python")
    return str(resolved)


def _known_runtime_package_roots() -> set[Path]:
    """Return package roots derived from Python installation metadata.

    This deliberately does not trust arbitrary entries already present in
    ``sys.path``.  In particular, a process-start ``PYTHONPATH`` ending in a
    directory named ``site-packages`` must not be laundered into the scrubbed
    validation environment.
    """

    candidates: set[Path] = set()
    try:
        candidates.update(Path(item) for item in site.getsitepackages())
    except (AttributeError, OSError):
        pass
    for key in ("purelib", "platlib"):
        configured = sysconfig.get_path(key)
        if configured:
            candidates.add(Path(configured))
    if os.name == "posix":
        try:
            import pwd

            account_uid = os.getuid()
            # ``unshare -Ur`` maps the invoking host account to namespace
            # uid 0. Resolve that mapping through the kernel-owned uid map so
            # the approved canonical user site remains stable without
            # trusting inherited HOME or PYTHONUSERBASE values.
            try:
                uid_map = Path("/proc/self/uid_map").read_text(
                    encoding="utf-8"
                )
            except OSError:
                uid_map = ""
            for line in uid_map.splitlines():
                fields = line.split()
                if len(fields) != 3:
                    continue
                try:
                    inside_start, outside_start, length = (
                        int(field) for field in fields
                    )
                except ValueError:
                    continue
                if inside_start <= account_uid < inside_start + length:
                    account_uid = outside_start + (account_uid - inside_start)
                    break
            account_home = Path(pwd.getpwuid(account_uid).pw_dir)
            user_base = account_home / ".local"
            for key in ("purelib", "platlib"):
                configured = sysconfig.get_path(
                    key,
                    scheme="posix_user",
                    vars={"userbase": str(user_base)},
                )
                if configured:
                    candidates.add(Path(configured))
        except (ImportError, KeyError, OSError):
            pass

    resolved: set[Path] = set()
    for candidate in candidates:
        try:
            root = candidate.resolve(strict=True)
        except OSError:
            continue
        if root.is_dir():
            resolved.add(root)
    return resolved


def _runtime_python_path_entries(
    source: Mapping[str, object],
) -> tuple[str, ...]:
    """Return explicitly approved or already-active Python package roots.

    Automatic user-site discovery remains disabled in the child.  For the
    default supervisor interpreter, retain only active ``site-packages`` and
    ``dist-packages`` directories from this process.  This supports a
    supervisor installed with ``pip --user`` without re-reading ``HOME`` or an
    inherited ``PYTHONPATH``.  A separately configured validation interpreter
    is assumed to carry its own isolated package environment unless an
    explicit validation PYTHONPATH is also supplied.
    """

    configured = str(source.get(VALIDATION_PYTHONPATH_ENV) or "").strip()
    if configured:
        entries: list[str] = []
        for raw_entry in configured.split(os.pathsep):
            if not raw_entry:
                raise ValidationRuntimeError(
                    f"{VALIDATION_PYTHONPATH_ENV} contains an empty entry"
                )
            path = Path(raw_entry)
            if not path.is_absolute():
                raise ValidationRuntimeError(
                    f"{VALIDATION_PYTHONPATH_ENV} entries must be absolute: "
                    f"{raw_entry!r}"
                )
            try:
                resolved = path.resolve(strict=True)
            except OSError as exc:
                raise ValidationRuntimeError(
                    f"{VALIDATION_PYTHONPATH_ENV} entry is unavailable: "
                    f"{raw_entry!r}"
                ) from exc
            if not resolved.is_dir():
                raise ValidationRuntimeError(
                    f"{VALIDATION_PYTHONPATH_ENV} entry is not a directory: "
                    f"{raw_entry!r}"
                )
            _reject_writable_path(
                resolved,
                source=VALIDATION_PYTHONPATH_ENV,
            )
            rendered = str(resolved)
            if rendered not in entries:
                entries.append(rendered)
        return tuple(entries)

    if str(source.get(VALIDATION_PYTHON_ENV) or "").strip():
        return ()

    approved_roots = _known_runtime_package_roots()
    entries = []
    for raw_entry in sys.path:
        if not raw_entry:
            continue
        path = Path(raw_entry)
        if not path.is_absolute():
            continue
        try:
            resolved = path.resolve(strict=True)
        except OSError:
            continue
        if not resolved.is_dir() or resolved not in approved_roots:
            continue
        rendered = str(resolved)
        if rendered not in entries:
            entries.append(rendered)
    return tuple(entries)


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
            # npm rejects loading one path in two config scopes.  A child of
            # /dev/null is both distinct from the global config and guaranteed
            # to remain unavailable, so neither scope can import host settings.
            "NPM_CONFIG_USERCONFIG": _NPM_DISABLED_USER_CONFIG,
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
    python_path = _runtime_python_path_entries(source)
    if python_path:
        result["PYTHONPATH"] = os.pathsep.join(python_path)
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
    if "`" in text or "$(" in text:
        raise ValidationRuntimeError(
            "dynamic command substitution is not permitted for validation"
        )
    try:
        lexer = shlex.shlex(
            text,
            posix=True,
            punctuation_chars=";&|()<>",
        )
        lexer.whitespace_split = True
        lexer.commenters = ""
        leading = list(lexer)
    except ValueError as exc:
        raise ValidationRuntimeError(
            "validation command has invalid shell quoting"
        ) from exc
    if any(Path(token).name in {"bash", "sh"} for token in leading):
        raise ValidationRuntimeError(
            "nested validation shells are not permitted; provide the inner "
            "command directly"
        )
    if any(Path(token).name == "eval" for token in leading):
        raise ValidationRuntimeError(
            "dynamic shell evaluation is not permitted for validation"
        )
    guarded = (
        f"readonly {_CHILD_PYTHON_ENV}; "
        f'python() {{ "${{{_CHILD_PYTHON_ENV}}}" "$@"; }}; '
        f'python3() {{ "${{{_CHILD_PYTHON_ENV}}}" "$@"; }}; '
        f'pytest() {{ "${{{_CHILD_PYTHON_ENV}}}" -m pytest "$@"; }}; '
        "readonly -f python python3 pytest; "
        f"{text}"
    )
    return ["/bin/bash", "--noprofile", "--norc", "-c", guarded]


def validation_argv_command(command: Sequence[str]) -> list[str]:
    """Normalize an argv validation without permitting a login shell.

    Most argv validations execute directly.  Historical callers may have
    stored ``bash -lc`` or ``sh -c`` arrays; retain their command text and
    positional arguments while routing them through the same guarded Bash
    contract as string validations.  Bare Python and pytest launchers are
    pinned to the supervisor's approved interpreter because its parent
    directory is intentionally absent from the restricted executable PATH.
    """

    parts = [str(part) for part in command]
    if not parts or not parts[0]:
        raise ValidationRuntimeError("validation argv must not be empty")
    if parts[0] in {"python", "python3"}:
        return [validation_python_executable(), *parts[1:]]
    if parts[0] == "pytest":
        return [validation_python_executable(), "-m", "pytest", *parts[1:]]
    executable = Path(parts[0]).name
    if executable not in {"bash", "sh"}:
        if any(Path(part).name in {"bash", "sh"} for part in parts[1:]):
            raise ValidationRuntimeError(
                "wrapped validation shells are not permitted"
            )
        return parts

    command_index: int | None = None
    for index, argument in enumerate(parts[1:], start=1):
        if argument == "--login":
            continue
        if argument == "-c" or (
            argument.startswith("-")
            and not argument.startswith("--")
            and "c" in argument[1:]
        ):
            command_index = index + 1
            break

    if command_index is None:
        raise ValidationRuntimeError(
            "validation shell argv must provide command text with -c"
        )
    if command_index >= len(parts):
        raise ValidationRuntimeError("validation shell command text is missing")
    normalized = validation_shell_command(parts[command_index])
    normalized.extend(parts[command_index + 1 :])
    return normalized
