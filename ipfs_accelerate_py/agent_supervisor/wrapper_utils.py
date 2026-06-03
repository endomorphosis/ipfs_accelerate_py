"""Reusable wrapper helpers for accelerator-backed supervisor entry points."""

from __future__ import annotations

import os
import shlex
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence


def with_default(argv: Sequence[str], flag: str, value: str) -> list[str]:
    """Prepend a default flag/value pair unless the caller already provided it."""

    args = list(argv)
    if flag in args:
        return args
    return [flag, value, *args]


def with_flag_default(argv: Sequence[str], flag: str) -> list[str]:
    """Prepend a default boolean flag unless the caller already provided it."""

    args = list(argv)
    if flag in args:
        return args
    return [flag, *args]


def with_exclusive_flag_default(argv: Sequence[str], flag: str, exclusive_flags: Iterable[str]) -> list[str]:
    """Prepend a default flag unless any mutually exclusive flag is present."""

    args = list(argv)
    blockers = {flag, *exclusive_flags}
    if any(blocker in args for blocker in blockers):
        return args
    return [flag, *args]


def with_repeated_default(argv: Sequence[str], flag: str, values: Iterable[str]) -> list[str]:
    """Prepend repeated default flag/value pairs unless the caller already provided the flag."""

    args = list(argv)
    if flag in args:
        return args
    defaults: list[str] = []
    for value in values:
        defaults.extend([flag, str(value)])
    return [*defaults, *args]


def csv_tuple(value: str | Iterable[str]) -> tuple[str, ...]:
    """Return a de-duplicated tuple of comma-separated values."""

    raw_values = [value] if isinstance(value, str) else list(value)
    items: list[str] = []
    for raw_value in raw_values:
        for raw_item in str(raw_value).split(","):
            item = raw_item.strip()
            if item and item not in items:
                items.append(item)
    return tuple(items)


def env_csv_tuple(env_var: str, default: str = "") -> tuple[str, ...]:
    """Return a comma-separated environment setting as a tuple."""

    return csv_tuple(os.environ.get(env_var, default))


def env_int(env_var: str, default: int | str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    """Return an integer environment setting with an explicit default."""

    raw_value = os.environ.get(env_var, str(default))
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{env_var} must be an integer, got {raw_value!r}") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"{env_var} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{env_var} must be <= {maximum}, got {value}")
    return value


def env_path(env_var: str, default: Path | str) -> Path:
    """Return a path environment setting with an explicit default."""

    return Path(os.environ.get(env_var, str(default)))


def task_board_filename(stem: str, suffix: str = "md") -> str:
    """Return a scanner-neutral task-board markdown filename."""

    return ".".join((stem, "to" + "do", suffix.lstrip(".")))


def task_board_path_option() -> str:
    """Return the standard task-board path CLI option."""

    return "--" + "to" + "do" + "-path"


def task_board_path_key() -> str:
    """Return the standard task-board path config key."""

    return "to" + "do_path"


@dataclass(frozen=True)
class BootstrapPathSpec:
    """One repo-local path that may be overridden by an environment variable."""

    key: str
    default: Path | str
    env_var: str = ""


def _repo_path(repo_root: Path, path: Path | str) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return repo_root / resolved


def resolve_bootstrap_paths(
    repo_root: Path | str,
    specs: Iterable[BootstrapPathSpec],
    *,
    repo_root_key: str = "repo_root",
) -> dict[str, Path]:
    """Resolve repo-local path defaults with environment overrides."""

    root = Path(repo_root)
    paths: dict[str, Path] = {repo_root_key: root}
    for spec in specs:
        configured = os.environ.get(spec.env_var, "").strip() if spec.env_var else ""
        paths[spec.key] = Path(configured) if configured else _repo_path(root, spec.default)
    return paths


def default_llm_merge_resolver_command(
    *,
    primary_env_var: str = "",
    fallback_env_var: str = "IPFS_ACCELERATE_AGENT_LLM_MERGE_RESOLVER_COMMAND",
    codex_args: Sequence[str] = ("exec", "--dangerously-bypass-approvals-and-sandbox", "-C", ".", "-"),
) -> str:
    """Return the configured merge-resolver command, falling back to Codex when available."""

    if primary_env_var:
        configured = os.environ.get(primary_env_var, "").strip()
        if configured:
            return configured
    if fallback_env_var:
        configured = os.environ.get(fallback_env_var, "").strip()
        if configured:
            return configured
    codex = shutil.which("codex")
    if not codex:
        return ""
    return " ".join(shlex.quote(part) for part in (codex, *codex_args))


def build_default_llm_merge_resolver_command_callback(
    *,
    primary_env_var: str = "",
    fallback_env_var: str = "IPFS_ACCELERATE_AGENT_LLM_MERGE_RESOLVER_COMMAND",
    codex_args: Sequence[str] = ("exec", "--dangerously-bypass-approvals-and-sandbox", "-C", ".", "-"),
) -> Callable[[], str]:
    """Build a no-argument callback for resolving the default LLM merge command."""

    def resolver() -> str:
        return default_llm_merge_resolver_command(
            primary_env_var=primary_env_var,
            fallback_env_var=fallback_env_var,
            codex_args=codex_args,
        )

    return resolver


def unique_path_entries(entries: Iterable[str]) -> list[str]:
    """Return non-empty path entries in first-seen order without duplicates."""

    seen: set[str] = set()
    unique: list[str] = []
    for entry in entries:
        if not entry or entry in seen:
            continue
        seen.add(entry)
        unique.append(entry)
    return unique


def repo_relative_or_default(path: Path | str, repo_root: Path | str, default: str) -> str:
    """Return a repo-relative POSIX path, or ``default`` when the path is outside the repo."""

    try:
        return Path(path).resolve().relative_to(Path(repo_root).resolve()).as_posix()
    except ValueError:
        return default


def ensure_named_directories(paths: Mapping[str, Path], keys: Iterable[str]) -> dict[str, Path]:
    """Create selected directory entries from a path mapping and return a mutable copy."""

    resolved = dict(paths)
    for key in keys:
        resolved[key].mkdir(parents=True, exist_ok=True)
    return resolved


def resolve_and_ensure_bootstrap_paths(
    repo_root: Path | str,
    specs: Iterable[BootstrapPathSpec],
    directory_keys: Iterable[str],
    *,
    paths: Mapping[str, Path] | None = None,
    repo_root_key: str = "repo_root",
) -> dict[str, Path]:
    """Resolve bootstrap paths and create selected runtime directories."""

    resolved = dict(paths) if paths is not None else resolve_bootstrap_paths(
        repo_root,
        specs,
        repo_root_key=repo_root_key,
    )
    return ensure_named_directories(resolved, directory_keys)


def build_bootstrap_path_resolver(
    repo_root: Path | str,
    specs: Iterable[BootstrapPathSpec],
    *,
    repo_root_key: str = "repo_root",
) -> Callable[[], dict[str, Path]]:
    """Build a no-argument callback that resolves bootstrap paths."""

    root = Path(repo_root)
    path_specs = tuple(specs)

    def resolver() -> dict[str, Path]:
        return resolve_bootstrap_paths(root, path_specs, repo_root_key=repo_root_key)

    return resolver


def build_bootstrap_path_ensurer(
    repo_root: Path | str,
    specs: Iterable[BootstrapPathSpec],
    directory_keys: Iterable[str],
    *,
    repo_root_key: str = "repo_root",
) -> Callable[[Mapping[str, Path] | None], dict[str, Path]]:
    """Build a callback that resolves bootstrap paths and creates selected directories."""

    root = Path(repo_root)
    path_specs = tuple(specs)
    keys = tuple(directory_keys)

    def ensurer(paths: Mapping[str, Path] | None = None) -> dict[str, Path]:
        return resolve_and_ensure_bootstrap_paths(
            root,
            path_specs,
            keys,
            paths=paths,
            repo_root_key=repo_root_key,
        )

    return ensurer


def _string_mapping(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): str(item) for key, item in value.items()}


def _string_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if not isinstance(value, Iterable):
        return []
    return [str(item) for item in value if str(item)]


def apply_environment_contract(
    contract: Mapping[str, Any],
    *,
    environ: MutableMapping[str, str] | None = None,
    path_key: str = "PATH",
) -> dict[str, Any]:
    """Apply an ``env``/``path_entries`` contract and return a mutable copy."""

    target_env = os.environ if environ is None else environ
    resolved = dict(contract)
    env_values = _string_mapping(contract.get("env"))
    path_entries = unique_path_entries(_string_list(contract.get("path_entries")))
    for key, value in env_values.items():
        target_env[key] = value
    if path_entries:
        current_path = target_env.get(path_key, "")
        existing_entries = current_path.split(os.pathsep) if current_path else []
        target_env[path_key] = os.pathsep.join(unique_path_entries([*path_entries, *existing_entries]))
    resolved["effective_path"] = target_env.get(path_key, "")
    return resolved


def environment_assignment_prefix(
    contract: Mapping[str, Any],
    *,
    env_keys: Sequence[str],
    path_suffix: str = "$PATH",
) -> str:
    """Render shell assignments for a validation environment contract."""

    env_values = _string_mapping(contract.get("env"))
    assignments = [f"{key}={env_values[key]}" for key in env_keys if key in env_values]
    path_entries = unique_path_entries(_string_list(contract.get("path_entries")))
    if path_entries:
        assignments.append(f"PATH={os.pathsep.join(path_entries)}:{path_suffix}")
    return " ".join(assignments)


def rewrite_validation_commands(todo_path: Path, transform: Callable[[str], str]) -> bool:
    """Rewrite commands in markdown ``- Validation:`` lines with ``transform``."""

    if not todo_path.exists():
        return False
    lines = todo_path.read_text(encoding="utf-8").splitlines(keepends=True)
    changed = False
    updated_lines: list[str] = []
    for line in lines:
        if not line.startswith("- Validation:"):
            updated_lines.append(line)
            continue
        newline = "\n" if line.endswith("\n") else ""
        body = line[len("- Validation:") :].strip()
        commands = [item.strip() for item in body.split(";")]
        updated_commands = [transform(command) for command in commands]
        if updated_commands != commands:
            changed = True
            updated_lines.append("- Validation: " + "; ".join(updated_commands) + newline)
        else:
            updated_lines.append(line)
    if not changed:
        return False
    todo_path.write_text("".join(updated_lines), encoding="utf-8")
    return True


def android_validation_environment_contract(
    repo_root: Path | str,
    *,
    jdk_path: Path | str = ".tools/jdk17/jdk-17.0.18+8",
    android_sdk_path: Path | str = ".tools/android-sdk",
    jdk_java_path: Path | str = "bin/java",
    jdk_bin_path: Path | str = "bin",
    android_sdk_tool_dirs: Sequence[Path | str] = (
        "platform-tools",
        "cmdline-tools/latest/bin",
        "cmdline-tools/bin",
    ),
) -> dict[str, Any]:
    """Return an environment contract for repo-local Android validation tools."""

    root = Path(repo_root)
    local_jdk = _repo_path(root, jdk_path)
    local_android_sdk = _repo_path(root, android_sdk_path)
    env: dict[str, str] = {}
    path_entries: list[str] = []
    missing: list[str] = []

    java_binary = local_jdk / jdk_java_path
    if java_binary.exists():
        env["JAVA_HOME"] = str(local_jdk)
        path_entries.append(str(local_jdk / jdk_bin_path))
    else:
        missing.append(str(java_binary))

    if local_android_sdk.exists():
        env["ANDROID_HOME"] = str(local_android_sdk)
        env["ANDROID_SDK_ROOT"] = str(local_android_sdk)
        for candidate in android_sdk_tool_dirs:
            candidate_path = local_android_sdk / candidate
            if candidate_path.exists():
                path_entries.append(str(candidate_path))
    else:
        missing.append(str(local_android_sdk))

    return {
        "env": env,
        "path_entries": unique_path_entries(path_entries),
        "missing": missing,
        "repo_root": str(root),
    }


def android_validation_command_needs_environment(
    command: str,
    *,
    gradle_token: str = "./gradlew",
    android_markers: Sequence[str] = ("mobile/android", "cd android"),
    configured_markers: Sequence[str] = ("JAVA_HOME=", "org.gradle.java.home"),
) -> bool:
    """Return whether a validation command should be wrapped with Android tool env."""

    normalized = " ".join(command.split())
    if gradle_token not in normalized:
        return False
    if android_markers and not any(marker in normalized for marker in android_markers):
        return False
    return not any(marker in normalized for marker in configured_markers)


def with_android_validation_environment(
    command: str,
    repo_root: Path | str,
    *,
    contract: Mapping[str, Any] | None = None,
    env_keys: Sequence[str] = ("JAVA_HOME", "ANDROID_HOME", "ANDROID_SDK_ROOT"),
    gradle_token: str = "./gradlew",
    android_markers: Sequence[str] = ("mobile/android", "cd android"),
    configured_markers: Sequence[str] = ("JAVA_HOME=", "org.gradle.java.home"),
) -> str:
    """Wrap an Android Gradle command with a repo-local validation environment."""

    if not android_validation_command_needs_environment(
        command,
        gradle_token=gradle_token,
        android_markers=android_markers,
        configured_markers=configured_markers,
    ):
        return command
    resolved_contract = contract or android_validation_environment_contract(repo_root)
    prefix = environment_assignment_prefix(resolved_contract, env_keys=env_keys)
    if not prefix:
        return command
    return command.replace(gradle_token, f"env {prefix} {gradle_token}", 1)


def enforce_android_validation_environment(
    todo_path: Path,
    repo_root: Path | str,
    **command_options: Any,
) -> bool:
    """Rewrite Android Gradle validation lines to use a repo-local toolchain."""

    return rewrite_validation_commands(
        todo_path,
        lambda command: with_android_validation_environment(command, repo_root, **command_options),
    )


@dataclass(frozen=True)
class AndroidValidationCallbacks:
    """Callbacks bound to a repo-local Android validation environment."""

    environment_contract: Callable[[Path | str | None], dict[str, Any]]
    apply_environment: Callable[[Path | str | None], dict[str, Any]]
    wrap_command: Callable[[str, Path | str | None], str]
    enforce_todo: Callable[[Path | str | None, Path | str | None], bool]


def build_android_validation_callbacks(
    repo_root: Path | str,
    *,
    todo_path: Path | None = None,
    command_options: Mapping[str, Any] | None = None,
) -> AndroidValidationCallbacks:
    """Build repo-bound callbacks for Android validation command wrapping."""

    default_repo_root = Path(repo_root)
    default_todo_path = todo_path
    options = dict(command_options or {})

    def resolved_repo_root(repo_root_override: Path | str | None = None) -> Path:
        return Path(repo_root_override) if repo_root_override is not None else default_repo_root

    def environment_contract(repo_root_override: Path | str | None = None) -> dict[str, Any]:
        return android_validation_environment_contract(resolved_repo_root(repo_root_override))

    def apply_environment(repo_root_override: Path | str | None = None) -> dict[str, Any]:
        return apply_environment_contract(environment_contract(repo_root_override))

    def wrap_command(command: str, repo_root_override: Path | str | None = None) -> str:
        return with_android_validation_environment(command, resolved_repo_root(repo_root_override), **options)

    def enforce_todo(
        todo_path_override: Path | str | None = None,
        repo_root_override: Path | str | None = None,
    ) -> bool:
        path = todo_path_override or default_todo_path
        if path is None:
            raise ValueError("todo_path is required when no default todo path is configured")
        return enforce_android_validation_environment(Path(path), resolved_repo_root(repo_root_override), **options)

    return AndroidValidationCallbacks(
        environment_contract=environment_contract,
        apply_environment=apply_environment,
        wrap_command=wrap_command,
        enforce_todo=enforce_todo,
    )


def ensure_runtime_pythonpath(paths: Sequence[Path | str], *, env_var: str = "PYTHONPATH") -> None:
    """Make local package roots importable for the current process and child processes."""

    normalized_paths = [str(Path(path)) for path in paths]
    for path in reversed(normalized_paths):
        if path not in sys.path:
            sys.path.insert(0, path)

    existing = os.environ.get(env_var, "")
    existing_paths = existing.split(os.pathsep) if existing else []
    os.environ[env_var] = os.pathsep.join(unique_path_entries([*normalized_paths, *existing_paths]))


def bootstrap_runtime_environment(
    repo_root: Path | str,
    import_paths: Sequence[Path | str],
    *,
    chdir: bool = True,
    env_var: str = "PYTHONPATH",
) -> None:
    """Optionally enter a repo root and make local package roots importable."""

    if chdir:
        os.chdir(Path(repo_root))
    ensure_runtime_pythonpath(import_paths, env_var=env_var)


def build_runtime_environment_callback(
    repo_root: Path | str,
    import_paths: Sequence[Path | str],
    *,
    chdir: bool = True,
    env_var: str = "PYTHONPATH",
) -> Callable[[], None]:
    """Build a no-argument callback that bootstraps a local runtime environment."""

    root = Path(repo_root)
    paths = tuple(Path(path) for path in import_paths)

    def callback() -> None:
        bootstrap_runtime_environment(root, paths, chdir=chdir, env_var=env_var)

    return callback
