"""Sealed environment contract for model-provider shell commands.

Implementation providers are allowed to edit an isolated worktree, but their
shell tools may intentionally replace ``PATH``.  That is a useful provider
sandbox default, yet it can make a command observe a different toolchain from
the one declared by the supervisor process.  This module projects the small,
non-secret part of that declaration and exposes it through a sealed launcher.

The launcher is *not* an authoritative validation environment.  It exists only
so provider-side discovery, preflight, and evidence-generation commands can
observe the same declared executable path and managed prover roots.  Final
validation still uses :mod:`agent_supervisor.validation.validation_runtime`.
Managed roots therefore have to be deployed beneath root-owned/read-only
ancestors before supervisor dispatch; profile-home installs are useful for
development but cannot enter this certification boundary.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from ..validation.validation_runtime import (
    FORMAL_TOOLCHAIN_CONTRACT_SHA256_ENV,
    FORMAL_TOOLCHAIN_REQUIRED_COMMANDS_ENV,
    formal_toolchain_deployment_manifest,
)


PROVIDER_COMMAND_ENVIRONMENT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/provider-command-environment@1"
)
PROVIDER_COMMAND_ENV_WRAPPER_ENV = (
    "IPFS_ACCELERATE_AGENT_COMMAND_ENV_WRAPPER"
)
PROVIDER_COMMAND_ENV_DIGEST_ENV = (
    "IPFS_ACCELERATE_AGENT_COMMAND_ENV_SHA256"
)
PROVIDER_COMMAND_REQUIRED_COMMANDS_ENV = (
    FORMAL_TOOLCHAIN_REQUIRED_COMMANDS_ENV
)

# This is deliberately not a general environment allowlist.  In particular,
# provider credentials, signing material, registry configuration, and cloud
# variables must never be serialized into the launcher or its receipt.
APPROVED_PROVIDER_COMMAND_ENVIRONMENT_NAMES = (
    FORMAL_TOOLCHAIN_CONTRACT_SHA256_ENV,
    "IPFS_ACCELERATE_AGENT_VALIDATION_PATH",
    "IPFS_ACCELERATE_AGENT_VALIDATION_PYTHON",
    "IPFS_DATASETS_PY_EXTERNAL_PROVER_ROOT",
    "IPFS_DATASETS_PY_THEOREM_PROVERS_ROOT",
    "LANG",
    "LANGUAGE",
    "LC_ALL",
    "PATH",
    "PYTHONHASHSEED",
    "PYTHONNOUSERSITE",
    "TMPDIR",
    "TZ",
)

_PATH_LIKE_NAMES = frozenset(
    {
        "IPFS_DATASETS_PY_EXTERNAL_PROVER_ROOT",
        "IPFS_DATASETS_PY_THEOREM_PROVERS_ROOT",
        "TMPDIR",
    }
)
_PATH_LIST_NAMES = frozenset(
    {
        "IPFS_ACCELERATE_AGENT_VALIDATION_PATH",
        "PATH",
    }
)
_BARE_COMMAND_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,127}")
_MAX_ENV_VALUE_BYTES = 65_536
_MAX_PATH_ENTRIES = 256
_MAX_REQUIRED_COMMANDS = 64


class ProviderCommandEnvironmentError(RuntimeError):
    """Raised before provider dispatch when the declared environment is unsafe."""


@dataclass(frozen=True)
class ProviderCommandEnvironment:
    """Public, value-free receipt for one sealed provider command launcher."""

    wrapper_path: str
    contract_sha256: str
    formal_toolchain_contract_sha256: str
    environment_names: tuple[str, ...]
    required_commands: tuple[str, ...]
    required_command_identity_sha256: tuple[tuple[str, str], ...]
    sealed: bool


def _safe_text(name: str, value: object) -> str:
    text = str(value)
    encoded = text.encode("utf-8")
    if len(encoded) > _MAX_ENV_VALUE_BYTES:
        raise ProviderCommandEnvironmentError(
            f"approved provider command environment value is too large: {name}"
        )
    if "\x00" in text or "\r" in text or "\n" in text:
        raise ProviderCommandEnvironmentError(
            f"approved provider command environment value is malformed: {name}"
        )
    return text


def _absolute_path(name: str, value: str, *, require_directory: bool) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ProviderCommandEnvironmentError(
            f"approved provider command environment path must be absolute: {name}"
        )
    if require_directory:
        try:
            if not path.is_dir():
                raise ProviderCommandEnvironmentError(
                    "approved managed provider command root is unavailable: "
                    f"{name}"
                )
        except OSError as exc:
            raise ProviderCommandEnvironmentError(
                f"unable to inspect approved provider command root: {name}"
            ) from exc
    return str(path)


def _absolute_path_list(name: str, value: str) -> str:
    entries = value.split(os.pathsep)
    if not entries or len(entries) > _MAX_PATH_ENTRIES:
        raise ProviderCommandEnvironmentError(
            f"approved provider command path list is malformed: {name}"
        )
    normalized: list[str] = []
    for entry in entries:
        if not entry:
            raise ProviderCommandEnvironmentError(
                f"approved provider command path list has an empty entry: {name}"
            )
        normalized.append(
            _absolute_path(name, entry, require_directory=False)
        )
    return os.pathsep.join(normalized)


def project_provider_command_environment(
    environment: Mapping[str, object] | None = None,
) -> dict[str, str]:
    """Project and validate the non-secret provider command environment.

    ``PATH`` is mandatory.  Managed prover roots are optional for ordinary
    implementation tasks, but a declared root must already exist as a
    directory.  The projection never scans ambient variable names.
    """

    source = os.environ if environment is None else environment
    formal_toolchain = formal_toolchain_deployment_manifest(source)
    formal_path = os.pathsep.join(
        str(item) for item in formal_toolchain["path_entries"]
    )
    projected: dict[str, str] = {}
    for name in APPROVED_PROVIDER_COMMAND_ENVIRONMENT_NAMES:
        if name == "PATH":
            value = formal_path
        elif name not in source or source[name] is None:
            continue
        else:
            value = _safe_text(name, source[name])
        if not value:
            continue
        if name in _PATH_LIST_NAMES:
            value = _absolute_path_list(name, value)
        elif name in _PATH_LIKE_NAMES:
            value = _absolute_path(
                name,
                value,
                require_directory=name.startswith(
                    "IPFS_DATASETS_PY_"
                ),
            )
        elif name == "IPFS_ACCELERATE_AGENT_VALIDATION_PYTHON":
            value = _absolute_path(
                name,
                value,
                require_directory=False,
            )
        projected[name] = value
    if not projected.get("PATH"):
        raise ProviderCommandEnvironmentError(
            "approved provider command environment is missing PATH"
        )
    return projected


def provider_command_environment_sha256(
    environment: Mapping[str, str],
) -> str:
    """Return the deterministic digest of the private environment projection."""

    payload = {
        "schema": PROVIDER_COMMAND_ENVIRONMENT_SCHEMA,
        "environment": {
            str(name): str(environment[name])
            for name in sorted(environment)
        },
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def normalize_required_commands(
    commands: Sequence[str],
) -> tuple[str, ...]:
    """Normalize a bounded list of bare executable names."""

    normalized: list[str] = []
    for raw in commands:
        for item in str(raw).split(","):
            command = item.strip()
            if not command:
                continue
            if not _BARE_COMMAND_RE.fullmatch(command):
                raise ProviderCommandEnvironmentError(
                    "required provider command must be a bare executable name"
                )
            if command not in normalized:
                normalized.append(command)
            if len(normalized) > _MAX_REQUIRED_COMMANDS:
                raise ProviderCommandEnvironmentError(
                    "too many required provider commands"
                )
    return tuple(normalized)


def _executable_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as exc:
        raise ProviderCommandEnvironmentError(
            "unable to bind required provider command identity"
        ) from exc
    return digest.hexdigest()


def preflight_required_commands(
    environment: Mapping[str, str],
    required_commands: Sequence[str],
) -> tuple[tuple[str, str], ...]:
    """Bind declared command names to executable content before provider launch."""

    normalized = normalize_required_commands(required_commands)
    identities: list[tuple[str, str]] = []
    search_path = str(environment.get("PATH") or "")
    for command in normalized:
        found = shutil.which(command, path=search_path)
        if not found:
            raise ProviderCommandEnvironmentError(
                f"required provider command is unavailable: {command}"
            )
        executable = Path(found)
        try:
            resolved = executable.resolve(strict=True)
        except OSError as exc:
            raise ProviderCommandEnvironmentError(
                f"required provider command is unavailable: {command}"
            ) from exc
        if not resolved.is_file() or not os.access(resolved, os.X_OK):
            raise ProviderCommandEnvironmentError(
                f"required provider command is not executable: {command}"
            )
        identities.append((command, _executable_sha256(resolved)))
    return tuple(identities)


def _sealed_launcher_source(environment: Mapping[str, str]) -> bytes:
    interpreter = str(Path(sys.executable).resolve())
    if "\n" in interpreter or len(interpreter.encode("utf-8")) > 120:
        raise ProviderCommandEnvironmentError(
            "provider command launcher Python interpreter is unsupported"
        )
    serialized = json.dumps(
        dict(sorted(environment.items())),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    source = f"""#!{interpreter}
import os
import re
import shutil
import sys

environment = {serialized!r}
approved = __import__("json").loads(environment)
bare = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{{0,127}}")
arguments = sys.argv[1:]
preflight = bool(arguments and arguments[0] == "--preflight")
if preflight:
    arguments = arguments[1:]
elif arguments and arguments[0] == "--":
    arguments = arguments[1:]
else:
    print("usage: provider-command-env (-- COMMAND... | --preflight NAME...)", file=sys.stderr)
    raise SystemExit(64)
if not arguments:
    print("provider command environment requires a command", file=sys.stderr)
    raise SystemExit(64)
if preflight:
    if len(arguments) > {_MAX_REQUIRED_COMMANDS}:
        print("too many provider command preflight names", file=sys.stderr)
        raise SystemExit(64)
    for name in arguments:
        if not bare.fullmatch(name):
            print("provider command preflight requires bare names", file=sys.stderr)
            raise SystemExit(64)
        if not shutil.which(name, path=approved["PATH"]):
            print("provider command unavailable: " + name, file=sys.stderr)
            raise SystemExit(127)
        print("provider command available: " + name)
    raise SystemExit(0)
command = arguments[0]
if "/" in command:
    resolved = os.path.abspath(command)
    if not os.path.isfile(resolved) or not os.access(resolved, os.X_OK):
        print("provider command is not executable", file=sys.stderr)
        raise SystemExit(126)
else:
    if not bare.fullmatch(command):
        print("provider command name is malformed", file=sys.stderr)
        raise SystemExit(64)
    resolved = shutil.which(command, path=approved["PATH"])
    if not resolved:
        print("provider command unavailable: " + command, file=sys.stderr)
        raise SystemExit(127)
os.execve(resolved, arguments, approved)
"""
    return source.encode("utf-8")


def _write_all(fd: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        try:
            written = os.write(fd, payload[offset:])
        except OSError as exc:
            raise ProviderCommandEnvironmentError(
                "unable to write sealed provider command launcher"
            ) from exc
        if written <= 0:
            raise ProviderCommandEnvironmentError(
                "sealed provider command launcher write was incomplete"
            )
        offset += written


@contextmanager
def sealed_provider_command_environment(
    environment: Mapping[str, object] | None = None,
    *,
    required_commands: Sequence[str] = (),
) -> Iterator[ProviderCommandEnvironment]:
    """Yield a sealed launcher for the approved command environment.

    Linux ``memfd`` sealing makes the launcher immutable for the complete Grok
    subprocess lifetime.  Unsupported platforms fail before model dispatch
    instead of silently falling back to a mutable wrapper.
    """

    if not sys.platform.startswith("linux") or not hasattr(os, "memfd_create"):
        raise ProviderCommandEnvironmentError(
            "sealed provider command launcher is unavailable on this platform"
        )
    normalized_required = normalize_required_commands(required_commands)
    source = dict(os.environ if environment is None else environment)
    if normalized_required:
        source[FORMAL_TOOLCHAIN_REQUIRED_COMMANDS_ENV] = ",".join(
            normalized_required
        )
    formal_toolchain = formal_toolchain_deployment_manifest(source)
    source[FORMAL_TOOLCHAIN_CONTRACT_SHA256_ENV] = str(
        formal_toolchain["manifest_sha256"]
    )
    projected = project_provider_command_environment(source)
    identities = preflight_required_commands(
        projected,
        normalized_required,
    )
    content = _sealed_launcher_source(projected)
    flags = int(getattr(os, "MFD_CLOEXEC", 0)) | int(
        getattr(os, "MFD_ALLOW_SEALING", 0)
    )
    if not getattr(os, "MFD_ALLOW_SEALING", 0):
        raise ProviderCommandEnvironmentError(
            "provider command launcher sealing is unavailable"
        )
    try:
        fd = os.memfd_create("ipfs-provider-command-env", flags=flags)
    except OSError as exc:
        raise ProviderCommandEnvironmentError(
            "unable to create sealed provider command launcher"
        ) from exc
    try:
        _write_all(fd, content)
        os.fchmod(fd, 0o500)
        seals = (
            fcntl.F_SEAL_WRITE
            | fcntl.F_SEAL_GROW
            | fcntl.F_SEAL_SHRINK
            | fcntl.F_SEAL_SEAL
        )
        try:
            fcntl.fcntl(fd, fcntl.F_ADD_SEALS, seals)
            observed = fcntl.fcntl(fd, fcntl.F_GET_SEALS)
        except OSError as exc:
            raise ProviderCommandEnvironmentError(
                "unable to seal provider command launcher"
            ) from exc
        if observed & seals != seals:
            raise ProviderCommandEnvironmentError(
                "provider command launcher seal verification failed"
            )
        launcher_path = f"/proc/{os.getpid()}/fd/{fd}"
        if not os.access(launcher_path, os.X_OK):
            raise ProviderCommandEnvironmentError(
                "sealed provider command launcher is not executable"
            )
        yield ProviderCommandEnvironment(
            wrapper_path=launcher_path,
            contract_sha256=provider_command_environment_sha256(projected),
            formal_toolchain_contract_sha256=str(
                formal_toolchain["manifest_sha256"]
            ),
            environment_names=tuple(sorted(projected)),
            required_commands=normalized_required,
            required_command_identity_sha256=identities,
            sealed=True,
        )
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


__all__ = [
    "APPROVED_PROVIDER_COMMAND_ENVIRONMENT_NAMES",
    "FORMAL_TOOLCHAIN_CONTRACT_SHA256_ENV",
    "FORMAL_TOOLCHAIN_REQUIRED_COMMANDS_ENV",
    "PROVIDER_COMMAND_ENVIRONMENT_SCHEMA",
    "PROVIDER_COMMAND_ENV_DIGEST_ENV",
    "PROVIDER_COMMAND_ENV_WRAPPER_ENV",
    "PROVIDER_COMMAND_REQUIRED_COMMANDS_ENV",
    "ProviderCommandEnvironment",
    "ProviderCommandEnvironmentError",
    "normalize_required_commands",
    "preflight_required_commands",
    "project_provider_command_environment",
    "provider_command_environment_sha256",
    "sealed_provider_command_environment",
]
