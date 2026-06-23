"""Reusable wrapper helpers for accelerator-backed supervisor entry points."""

from __future__ import annotations

import os
import shlex
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

from .validation_commands import split_validation_commands


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


DEFAULT_CODEBASE_SCAN_DATA_SUBDIRS = (
    "discovery",
    "objective_bundles",
    "objective_datasets",
    "state",
    "worktrees",
)


def _prefix_path(*parts: object) -> str:
    normalized = [str(part).strip().strip("/") for part in parts if str(part).strip().strip("/")]
    if not normalized:
        return ""
    return "/".join(normalized) + "/"


def _string_tuple(value: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,) if value else ()
    return tuple(str(item) for item in value if str(item))


def data_namespace_scan_skip_prefixes(
    namespaces: Iterable[str] | Mapping[str, Iterable[str]],
    *,
    root: str = "data",
    subdirs: Iterable[str] = DEFAULT_CODEBASE_SCAN_DATA_SUBDIRS,
    include_scripts: bool = False,
    script_prefix: str = "scripts/",
    extra_prefixes: Iterable[str] = (),
) -> tuple[str, ...]:
    """Build repo-relative generated-data prefixes to skip during codebase scans."""

    prefixes: list[str] = []
    if include_scripts:
        prefixes.append(_prefix_path(script_prefix))

    default_subdirs = _string_tuple(subdirs)
    if isinstance(namespaces, Mapping):
        namespace_items = namespaces.items()
    else:
        namespace_items = ((namespace, default_subdirs) for namespace in namespaces)

    for namespace, namespace_subdirs in namespace_items:
        for subdir in _string_tuple(namespace_subdirs):
            prefixes.append(_prefix_path(root, namespace, subdir))

    prefixes.extend(_prefix_path(prefix) for prefix in extra_prefixes)
    return tuple(unique_path_entries(prefix for prefix in prefixes if prefix))


@dataclass(frozen=True)
class AgentSupervisorNamespacePaths:
    """Standard repo-local data paths for one supervisor namespace."""

    repo_root: Path
    namespace: str
    data_root: Path
    namespace_root: Path
    discovery_dir: Path
    state_dir: Path
    worktree_root: Path
    objective_graph_path: Path
    objective_bundle_dir: Path
    objective_dataset_dir: Path
    objective_todo_vector_index_path: Path

    def repo_relative_path(self, key: str, default: str) -> str:
        """Return one stored path as repo-relative text, falling back to ``default``."""

        value = getattr(self, key)
        return repo_relative_or_default(value, self.repo_root, default)

    def discovery_output_path(self, default: str | None = None) -> str:
        """Return the namespace discovery directory as repo-relative output text."""

        fallback = default or str(Path("data") / self.namespace / "discovery")
        return self.repo_relative_path("discovery_dir", fallback)


def agent_supervisor_namespace_paths(
    repo_root: Path | str,
    namespace: str,
    *,
    data_root: Path | str = "data",
    discovery_subdir: str = "discovery",
    state_subdir: str = "state",
    worktree_subdir: str = "worktrees",
    objective_graph_filename: str = "objective_graph.json",
    objective_bundle_subdir: str = "objective_bundles",
    objective_dataset_subdir: str = "objective_datasets",
    todo_vector_index_filename: str = "todo_vector_index.json",
) -> AgentSupervisorNamespacePaths:
    """Return the conventional generated-data paths for a supervisor namespace."""

    root = Path(repo_root)
    data_root_path = _repo_path(root, data_root)
    namespace_root = data_root_path / namespace
    objective_bundle_dir = namespace_root / objective_bundle_subdir
    return AgentSupervisorNamespacePaths(
        repo_root=root,
        namespace=namespace,
        data_root=data_root_path,
        namespace_root=namespace_root,
        discovery_dir=namespace_root / discovery_subdir,
        state_dir=namespace_root / state_subdir,
        worktree_root=namespace_root / worktree_subdir,
        objective_graph_path=namespace_root / objective_graph_filename,
        objective_bundle_dir=objective_bundle_dir,
        objective_dataset_dir=namespace_root / objective_dataset_subdir,
        objective_todo_vector_index_path=objective_bundle_dir / todo_vector_index_filename,
    )


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


def env_str(env_var: str, default: str = "") -> str:
    """Return a stripped string environment setting with an explicit default."""

    value = os.environ.get(env_var, "").strip()
    return value or default


def apply_env_defaults(
    defaults: Mapping[str, str],
    *,
    environ: MutableMapping[str, str] | None = None,
    replace_empty: bool = False,
) -> dict[str, str]:
    """Apply environment defaults and return the effective values for those keys."""

    target_env = os.environ if environ is None else environ
    effective: dict[str, str] = {}
    for key, value in defaults.items():
        name = str(key)
        default_value = str(value)
        if name not in target_env or (replace_empty and not target_env.get(name, "")):
            target_env[name] = default_value
        effective[name] = target_env[name]
    return effective


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


def ensure_sys_path(paths: Sequence[Path | str]) -> None:
    """Make local import paths importable for the current process only."""

    normalized_paths = [str(Path(path)) for path in paths]
    for path in reversed(normalized_paths):
        if path not in sys.path:
            sys.path.insert(0, path)


def repo_root_from_env(
    env_var: str = "REPO_ROOT",
    *,
    fallback: Path | str | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve a wrapper repo root from an environment override or fallback path."""

    source = os.environ if environ is None else environ
    configured = source.get(env_var, "").strip() if env_var else ""
    return Path(configured or fallback or Path.cwd()).resolve()


@dataclass(frozen=True)
class RepoScriptBootstrap:
    """Resolved root and package path information for a repo-local wrapper script."""

    script_path: Path
    script_repo_root: Path
    repo_root: Path
    package_root: Path
    script_dir: Path


def build_repo_script_bootstrap(
    script_file: Path | str,
    *,
    package_name: Path | str = "ipfs_accelerate",
    external_dir: Path | str = "external",
    repo_root_env_var: str = "REPO_ROOT",
    script_repo_root_parent: int = 1,
    ensure_package_path: bool = True,
    include_script_dir: bool = False,
    environ: Mapping[str, str] | None = None,
) -> RepoScriptBootstrap:
    """Build env-aware bootstrap paths for a repo-local wrapper script."""

    script_path = Path(script_file).resolve()
    script_dir = script_path.parent
    script_repo_root = script_path.parents[script_repo_root_parent]
    package_root = repo_external_package_root(
        script_repo_root,
        package_name,
        external_dir=external_dir,
    )
    if ensure_package_path:
        import_paths = (package_root, script_dir) if include_script_dir else (package_root,)
        ensure_sys_path(import_paths)
    return RepoScriptBootstrap(
        script_path=script_path,
        script_repo_root=script_repo_root,
        repo_root=repo_root_from_env(
            repo_root_env_var,
            fallback=script_repo_root,
            environ=environ,
        ),
        package_root=package_root,
        script_dir=script_dir,
    )


def repo_external_package_root(
    repo_root: Path | str,
    package_name: Path | str,
    *,
    external_dir: Path | str = "external",
) -> Path:
    """Return one package root under a repo-local external dependency directory."""

    package_path = Path(package_name)
    if package_path.is_absolute():
        return package_path
    base_path = Path(external_dir)
    base = base_path if base_path.is_absolute() else Path(repo_root) / base_path
    return base / package_path


def repo_external_package_roots(
    repo_root: Path | str,
    package_names: Sequence[Path | str],
    *,
    external_dir: Path | str = "external",
) -> tuple[Path, ...]:
    """Return package roots under a repo-local external dependency directory."""

    return tuple(
        repo_external_package_root(repo_root, package_name, external_dir=external_dir)
        for package_name in package_names
    )


def repo_script_path(
    repo_root: Path | str,
    script_path: Path | str,
    *,
    scripts_dir: Path | str = "scripts",
) -> Path:
    """Return a repo-local script path from a filename or relative path."""

    root = Path(repo_root)
    path = Path(script_path)
    if path.is_absolute():
        return path
    if path.parent != Path("."):
        return _repo_path(root, path)
    return _repo_path(root, Path(scripts_dir) / path)


def repo_script_command(
    repo_root: Path | str,
    script_path: Path | str,
    *,
    command: Sequence[str] = ("bash",),
) -> str:
    """Return a shell-safe command for a repo-local script."""

    script = repo_script_path(repo_root, script_path)
    parts = [str(part) for part in command]
    parts.append(str(script))
    return shlex.join(parts)


def task_board_filename(stem: str, suffix: str = "md") -> str:
    """Return a scanner-neutral task-board markdown filename."""

    return ".".join((stem, "to" + "do", suffix.lstrip(".")))


DEFAULT_REPO_DOCS_DIR = "implementation_plan/docs"


def repo_doc_path(
    repo_root: Path | str,
    filename: Path | str,
    *,
    docs_dir: Path | str = DEFAULT_REPO_DOCS_DIR,
) -> Path:
    """Return a repo-local documentation path from a filename or relative path."""

    root = Path(repo_root)
    path = Path(filename)
    if path.is_absolute():
        return path
    if path.parent != Path("."):
        return _repo_path(root, path)
    return _repo_path(root, Path(docs_dir) / path)


def repo_task_board_path(
    repo_root: Path | str,
    stem: str,
    *,
    docs_dir: Path | str = DEFAULT_REPO_DOCS_DIR,
    suffix: str = "md",
) -> Path:
    """Return a repo-local task-board path using the standard task-board filename."""

    return repo_doc_path(repo_root, task_board_filename(stem, suffix), docs_dir=docs_dir)


def task_board_path_option() -> str:
    """Return the standard task-board path CLI option."""

    return "--" + "to" + "do" + "-path"


def task_board_path_key() -> str:
    """Return the standard task-board path config key."""

    return "to" + "do_path"


def prefixed_env_var(prefix: str, *parts: str) -> str:
    """Return a conventional environment variable name from a prefix and parts."""

    values = [prefix, *parts]
    return "_".join(value.strip("_").upper() for value in values if value.strip("_"))


def prefixed_env_csv_tuple(prefix: str, setting: str, default: str = "") -> tuple[str, ...]:
    """Return a prefixed comma-separated environment setting as a tuple."""

    return env_csv_tuple(prefixed_env_var(prefix, setting), default)


def prefixed_env_str(prefix: str, setting: str, default: str = "") -> str:
    """Return a prefixed string environment setting with an explicit default."""

    return env_str(prefixed_env_var(prefix, setting), default)


def prefixed_interoperability_focus(prefix: str, default: str = "") -> tuple[str, ...]:
    """Return the conventional prefixed interoperability-focus setting."""

    return prefixed_env_csv_tuple(prefix, "INTEROPERABILITY_FOCUS", default)


def prefixed_env_int(
    prefix: str,
    setting: str,
    default: int | str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Return a prefixed integer environment setting with an explicit default."""

    return env_int(prefixed_env_var(prefix, setting), default, minimum=minimum, maximum=maximum)


def prefixed_env_path(prefix: str, setting: str, default: Path | str) -> Path:
    """Return a prefixed path environment setting with an explicit default."""

    return env_path(prefixed_env_var(prefix, setting), default)


@dataclass(frozen=True)
class CodebaseScanEnvSettings:
    """Prefixed environment settings for codebase-scan behavior."""

    min_open_tasks: int
    max_findings: int
    cooldown_seconds: int
    timeout_seconds: int

    def recorder_kwargs(self) -> dict[str, int]:
        """Return keyword arguments for scan/backlog recorder defaults."""

        return {
            "min_open_tasks": self.min_open_tasks,
            "max_findings": self.max_findings,
            "cooldown_seconds": self.cooldown_seconds,
        }

    def codebase_refill_kwargs(self) -> dict[str, int]:
        """Return keyword arguments for codebase-refill supervisor defaults."""

        return {
            "codebase_scan_min_open_tasks": self.min_open_tasks,
            "codebase_scan_max_findings": self.max_findings,
            "codebase_scan_cooldown_seconds": self.cooldown_seconds,
            "codebase_refill_timeout_seconds": self.timeout_seconds,
        }


def prefixed_codebase_scan_env_settings(
    prefix: str,
    *,
    min_open_tasks: int = 5,
    max_findings: int = 5,
    cooldown_seconds: int = 21600,
    timeout_seconds: int = 600,
) -> CodebaseScanEnvSettings:
    """Return codebase-scan settings from conventional prefixed environment variables."""

    return CodebaseScanEnvSettings(
        min_open_tasks=prefixed_env_int(prefix, "CODEBASE_SCAN_MIN_OPEN_TASKS", min_open_tasks),
        max_findings=prefixed_env_int(prefix, "CODEBASE_SCAN_MAX_FINDINGS", max_findings),
        cooldown_seconds=prefixed_env_int(prefix, "CODEBASE_SCAN_COOLDOWN_SECONDS", cooldown_seconds),
        timeout_seconds=prefixed_env_int(prefix, "CODEBASE_REFILL_TIMEOUT_SECONDS", timeout_seconds),
    )


@dataclass(frozen=True)
class ObjectiveRefillEnvSettings:
    """Prefixed environment settings for objective-refill scan behavior."""

    min_open_tasks: int
    max_findings: int
    cooldown_seconds: int
    timeout_seconds: int
    surplus_findings_per_goal: int
    surplus_min_terms_per_todo: int

    def recorder_kwargs(self) -> dict[str, int]:
        """Return keyword arguments for objective-refill recorder defaults."""

        return {
            "min_open_tasks": self.min_open_tasks,
            "max_findings": self.max_findings,
            "cooldown_seconds": self.cooldown_seconds,
            "surplus_findings_per_goal": self.surplus_findings_per_goal,
            "surplus_min_terms_per_todo": self.surplus_min_terms_per_todo,
        }

    def objective_refill_kwargs(self) -> dict[str, int]:
        """Return keyword arguments for objective-refill supervisor defaults."""

        return {
            "objective_scan_min_open_tasks": self.min_open_tasks,
            "objective_scan_max_findings": self.max_findings,
            "objective_scan_cooldown_seconds": self.cooldown_seconds,
            "objective_refill_timeout_seconds": self.timeout_seconds,
            "objective_surplus_findings_per_goal": self.surplus_findings_per_goal,
            "objective_surplus_min_terms_per_todo": self.surplus_min_terms_per_todo,
        }


def prefixed_objective_refill_env_settings(
    prefix: str,
    *,
    min_open_tasks: int = 20,
    max_findings: int = 12,
    cooldown_seconds: int = 900,
    timeout_seconds: int = 600,
    surplus_findings_per_goal: int = 6,
    surplus_min_terms_per_todo: int = 4,
) -> ObjectiveRefillEnvSettings:
    """Return objective-refill settings from conventional prefixed environment variables."""

    return ObjectiveRefillEnvSettings(
        min_open_tasks=prefixed_env_int(prefix, "OBJECTIVE_SCAN_MIN_OPEN_TASKS", min_open_tasks),
        max_findings=prefixed_env_int(prefix, "OBJECTIVE_SCAN_MAX_FINDINGS", max_findings),
        cooldown_seconds=prefixed_env_int(prefix, "OBJECTIVE_SCAN_COOLDOWN_SECONDS", cooldown_seconds),
        timeout_seconds=prefixed_env_int(prefix, "OBJECTIVE_REFILL_TIMEOUT_SECONDS", timeout_seconds),
        surplus_findings_per_goal=prefixed_env_int(
            prefix,
            "OBJECTIVE_SURPLUS_FINDINGS_PER_GOAL",
            surplus_findings_per_goal,
        ),
        surplus_min_terms_per_todo=prefixed_env_int(
            prefix,
            "OBJECTIVE_SURPLUS_MIN_TERMS_PER_TODO",
            surplus_min_terms_per_todo,
        ),
    )


def task_board_env_var(prefix: str) -> str:
    """Return the conventional environment variable for a task-board path."""

    return prefixed_env_var(prefix, "TO" + "DO", "PATH")


@dataclass(frozen=True)
class BootstrapPathSpec:
    """One repo-local path that may be overridden by an environment variable."""

    key: str
    default: Path | str
    env_var: str = ""


@dataclass(frozen=True)
class BootstrapPathCallbacks:
    """Resolved bootstrap specs plus callbacks for wrapper entry points."""

    specs: tuple[BootstrapPathSpec, ...]
    repo_root: Path
    resolve: Callable[[], dict[str, Path]]
    ensure: Callable[[Mapping[str, Path] | None], dict[str, Path]]

    def output_path(self, key: str, default: str, paths: Mapping[str, Path | str] | None = None) -> str:
        """Return a repo-relative output path for a resolved bootstrap path."""

        resolved = self.resolve() if paths is None else paths
        return repo_relative_or_default(resolved[key], self.repo_root, default)

    def output_path_factory(self, key: str, default: str) -> Callable[[Mapping[str, Path | str] | None], str]:
        """Return a callback that resolves one repo-relative bootstrap output path."""

        def factory(paths: Mapping[str, Path | str] | None = None) -> str:
            return self.output_path(key, default, paths)

        return factory

    def output_path_kwargs_factory(
        self,
        output_key: str,
        path_key: str,
        default: str,
    ) -> Callable[[Mapping[str, Path | str] | None], dict[str, str]]:
        """Return a callback that exposes one bootstrap output path as kwargs."""

        def factory(paths: Mapping[str, Path | str] | None = None) -> dict[str, str]:
            return {output_key: self.output_path(path_key, default, paths)}

        return factory


def prefixed_bootstrap_path_spec(
    key: str,
    default: Path | str,
    prefix: str,
    setting: str | None = None,
) -> BootstrapPathSpec:
    """Build a bootstrap path spec with a conventional prefixed environment name."""

    return BootstrapPathSpec(key, default, prefixed_env_var(prefix, setting or key))


def prefixed_bootstrap_path_specs(
    prefix: str,
    entries: Iterable[tuple[str, Path | str] | tuple[str, Path | str, str | None]],
) -> tuple[BootstrapPathSpec, ...]:
    """Build prefixed bootstrap path specs from key/default entries."""

    specs: list[BootstrapPathSpec] = []
    for entry in entries:
        if len(entry) == 2:
            key, default = entry
            setting = None
        elif len(entry) == 3:
            key, default, setting = entry
        else:
            raise ValueError(f"bootstrap path entries must have 2 or 3 fields, got {len(entry)}")
        specs.append(prefixed_bootstrap_path_spec(key, default, prefix, setting))
    return tuple(specs)


AGENT_SUPERVISOR_DIRECTORY_BOOTSTRAP_KEYS = (
    "state_dir",
    "worktree_root",
    "discovery_dir",
    "objective_bundle_dir",
    "objective_dataset_dir",
)


def agent_supervisor_bootstrap_path_entries(
    todo_path: Path | str,
    namespace_paths: AgentSupervisorNamespacePaths,
    *,
    todo_key: str = "todo_path",
    todo_setting: str | None = None,
    objective_path: Path | str | None = None,
    objective_path_key: str = "objective_heap_path",
    objective_path_setting: str | None = None,
    namespace_keys: Iterable[str] = ("state_dir", "worktree_root"),
    extra_entries: Iterable[tuple[str, Path | str] | tuple[str, Path | str, str | None]] = (),
) -> tuple[tuple[str, Path | str] | tuple[str, Path | str, str | None], ...]:
    """Build standard bootstrap entries from a task board and namespace paths."""

    entries: list[tuple[str, Path | str] | tuple[str, Path | str, str | None]] = []
    entries.append((todo_key, todo_path) if todo_setting is None else (todo_key, todo_path, todo_setting))
    if objective_path is not None:
        entries.append(
            (objective_path_key, objective_path)
            if objective_path_setting is None
            else (objective_path_key, objective_path, objective_path_setting)
        )
    for key in namespace_keys:
        entries.append((key, getattr(namespace_paths, key)))
    entries.extend(extra_entries)
    return tuple(entries)


def build_agent_supervisor_bootstrap_path_callbacks(
    repo_root: Path | str,
    prefix: str,
    todo_path: Path | str,
    namespace_paths: AgentSupervisorNamespacePaths,
    *,
    todo_key: str = "todo_path",
    todo_setting: str | None = None,
    objective_path: Path | str | None = None,
    objective_path_key: str = "objective_heap_path",
    objective_path_setting: str | None = None,
    namespace_keys: Iterable[str] = ("state_dir", "worktree_root"),
    directory_keys: Iterable[str] | None = None,
    extra_entries: Iterable[tuple[str, Path | str] | tuple[str, Path | str, str | None]] = (),
    repo_root_key: str = "repo_root",
) -> BootstrapPathCallbacks:
    """Build prefixed bootstrap callbacks for the standard supervisor namespace layout."""

    keys = tuple(namespace_keys)
    entries = agent_supervisor_bootstrap_path_entries(
        todo_path,
        namespace_paths,
        todo_key=todo_key,
        todo_setting=todo_setting,
        objective_path=objective_path,
        objective_path_key=objective_path_key,
        objective_path_setting=objective_path_setting,
        namespace_keys=keys,
        extra_entries=extra_entries,
    )
    resolved_directory_keys = tuple(
        key for key in keys if key in AGENT_SUPERVISOR_DIRECTORY_BOOTSTRAP_KEYS
    ) if directory_keys is None else tuple(directory_keys)
    return build_prefixed_bootstrap_path_callbacks(
        repo_root,
        prefix,
        entries,
        resolved_directory_keys,
        repo_root_key=repo_root_key,
    )


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
    codex_args: Sequence[str] = (
        "exec",
        "--ignore-user-config",
        "--dangerously-bypass-approvals-and-sandbox",
        "-C",
        ".",
        "-",
    ),
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
    codex_args: Sequence[str] = (
        "exec",
        "--ignore-user-config",
        "--dangerously-bypass-approvals-and-sandbox",
        "-C",
        ".",
        "-",
    ),
) -> Callable[[], str]:
    """Build a no-argument callback for resolving the default LLM merge command."""

    def resolver() -> str:
        return default_llm_merge_resolver_command(
            primary_env_var=primary_env_var,
            fallback_env_var=fallback_env_var,
            codex_args=codex_args,
        )

    return resolver


def build_prefixed_default_llm_merge_resolver_command_callback(
    prefix: str,
    setting: str = "LLM_MERGE_RESOLVER_COMMAND",
    *,
    fallback_env_var: str = "IPFS_ACCELERATE_AGENT_LLM_MERGE_RESOLVER_COMMAND",
    codex_args: Sequence[str] = (
        "exec",
        "--ignore-user-config",
        "--dangerously-bypass-approvals-and-sandbox",
        "-C",
        ".",
        "-",
    ),
) -> Callable[[], str]:
    """Build a merge-resolver command callback from a prefixed environment setting."""

    return build_default_llm_merge_resolver_command_callback(
        primary_env_var=prefixed_env_var(prefix, setting),
        fallback_env_var=fallback_env_var,
        codex_args=codex_args,
    )


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


def build_prefixed_bootstrap_path_callbacks(
    repo_root: Path | str,
    prefix: str,
    entries: Iterable[tuple[str, Path | str] | tuple[str, Path | str, str | None]],
    directory_keys: Iterable[str],
    *,
    repo_root_key: str = "repo_root",
) -> BootstrapPathCallbacks:
    """Build prefixed path specs plus resolver and directory-ensurer callbacks."""

    specs = prefixed_bootstrap_path_specs(prefix, entries)
    root = Path(repo_root)
    return BootstrapPathCallbacks(
        specs=specs,
        repo_root=root,
        resolve=build_bootstrap_path_resolver(root, specs, repo_root_key=repo_root_key),
        ensure=build_bootstrap_path_ensurer(root, specs, directory_keys, repo_root_key=repo_root_key),
    )


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
        commands = split_validation_commands(body)
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
    env["JAVA_HOME"] = str(local_jdk)
    path_entries.append(str(local_jdk / jdk_bin_path))
    if not java_binary.exists():
        missing.append(str(java_binary))

    env["ANDROID_HOME"] = str(local_android_sdk)
    env["ANDROID_SDK_ROOT"] = str(local_android_sdk)
    if local_android_sdk.exists():
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
    ensure_sys_path(normalized_paths)

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


@dataclass(frozen=True)
class RuntimeEnvironmentCallbacks:
    """Reusable runtime environment callbacks for wrapper entry points."""

    enter: Callable[[], None]
    ensure_pythonpath: Callable[[], None]
    ensure_primary_pythonpath: Callable[[], None]


@dataclass(frozen=True)
class AgentSupervisorRuntimeBootstrapCallbacks:
    """Standard bootstrap paths and runtime callbacks for one supervisor namespace."""

    bootstrap_paths: BootstrapPathCallbacks
    runtime_environment: RuntimeEnvironmentCallbacks

    @property
    def specs(self) -> tuple[BootstrapPathSpec, ...]:
        """Return the underlying bootstrap path specs."""

        return self.bootstrap_paths.specs

    @property
    def resolve(self) -> Callable[[], dict[str, Path]]:
        """Return the bootstrap path resolver callback."""

        return self.bootstrap_paths.resolve

    @property
    def ensure(self) -> Callable[[Mapping[str, Path] | None], dict[str, Path]]:
        """Return the bootstrap path directory-ensurer callback."""

        return self.bootstrap_paths.ensure

    @property
    def enter(self) -> Callable[[], None]:
        """Return the runtime environment enter callback."""

        return self.runtime_environment.enter

    @property
    def ensure_pythonpath(self) -> Callable[[], None]:
        """Return the callback that only ensures all runtime import paths."""

        return self.runtime_environment.ensure_pythonpath

    @property
    def ensure_primary_pythonpath(self) -> Callable[[], None]:
        """Return the callback that only ensures primary runtime import paths."""

        return self.runtime_environment.ensure_primary_pythonpath


@dataclass(frozen=True)
class AgentSupervisorNamespaceContext:
    """Reusable namespace wrapper context for repo-local daemon/supervisor scripts."""

    repo_root: Path
    env_prefix: str
    task_board_path: Path
    task_board_path_key: str
    task_board_path_option: str
    namespace_paths: AgentSupervisorNamespacePaths
    runtime_bootstrap: AgentSupervisorRuntimeBootstrapCallbacks

    @property
    def bootstrap_paths(self) -> BootstrapPathCallbacks:
        """Return bootstrap path callbacks for this namespace context."""

        return self.runtime_bootstrap.bootstrap_paths

    @property
    def runtime_environment(self) -> RuntimeEnvironmentCallbacks:
        """Return runtime environment callbacks for this namespace context."""

        return self.runtime_bootstrap.runtime_environment

    @property
    def specs(self) -> tuple[BootstrapPathSpec, ...]:
        """Return bootstrap path specs for this namespace context."""

        return self.runtime_bootstrap.specs


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


def build_runtime_environment_callbacks(
    repo_root: Path | str,
    import_paths: Sequence[Path | str],
    *,
    primary_import_paths: Sequence[Path | str] | None = None,
    env_var: str = "PYTHONPATH",
) -> RuntimeEnvironmentCallbacks:
    """Build standard enter/ensure callbacks for a local wrapper runtime."""

    primary_paths = tuple(import_paths if primary_import_paths is None else primary_import_paths)
    return RuntimeEnvironmentCallbacks(
        enter=build_runtime_environment_callback(repo_root, import_paths, env_var=env_var),
        ensure_pythonpath=build_runtime_environment_callback(
            repo_root,
            import_paths,
            chdir=False,
            env_var=env_var,
        ),
        ensure_primary_pythonpath=build_runtime_environment_callback(
            repo_root,
            primary_paths,
            chdir=False,
            env_var=env_var,
        ),
    )


def build_repo_runtime_environment_callbacks(
    repo_root: Path | str,
    package_names: Sequence[Path | str] = ("ipfs_accelerate", "ipfs_datasets"),
    *,
    external_dir: Path | str = "external",
    primary_package_names: Sequence[Path | str] | None = None,
    env_var: str = "PYTHONPATH",
) -> RuntimeEnvironmentCallbacks:
    """Build runtime callbacks for repo-local external package roots."""

    package_roots = repo_external_package_roots(
        repo_root,
        package_names,
        external_dir=external_dir,
    )
    primary_package_roots = (
        None
        if primary_package_names is None
        else repo_external_package_roots(
            repo_root,
            primary_package_names,
            external_dir=external_dir,
        )
    )
    return build_runtime_environment_callbacks(
        repo_root,
        package_roots,
        primary_import_paths=primary_package_roots,
        env_var=env_var,
    )


def build_agent_supervisor_runtime_bootstrap_callbacks(
    repo_root: Path | str,
    prefix: str,
    todo_path: Path | str,
    namespace_paths: AgentSupervisorNamespacePaths,
    *,
    todo_key: str = "todo_path",
    todo_setting: str | None = None,
    objective_path: Path | str | None = None,
    objective_path_key: str = "objective_heap_path",
    objective_path_setting: str | None = None,
    namespace_keys: Iterable[str] = ("state_dir", "worktree_root"),
    directory_keys: Iterable[str] | None = None,
    extra_entries: Iterable[tuple[str, Path | str] | tuple[str, Path | str, str | None]] = (),
    repo_root_key: str = "repo_root",
    runtime_package_names: Sequence[Path | str] = ("ipfs_accelerate", "ipfs_datasets"),
    runtime_external_dir: Path | str = "external",
    runtime_primary_package_names: Sequence[Path | str] | None = None,
    runtime_env_var: str = "PYTHONPATH",
) -> AgentSupervisorRuntimeBootstrapCallbacks:
    """Build standard bootstrap path and runtime callbacks for a namespace wrapper."""

    return AgentSupervisorRuntimeBootstrapCallbacks(
        bootstrap_paths=build_agent_supervisor_bootstrap_path_callbacks(
            repo_root,
            prefix,
            todo_path,
            namespace_paths,
            todo_key=todo_key,
            todo_setting=todo_setting,
            objective_path=objective_path,
            objective_path_key=objective_path_key,
            objective_path_setting=objective_path_setting,
            namespace_keys=namespace_keys,
            directory_keys=directory_keys,
            extra_entries=extra_entries,
            repo_root_key=repo_root_key,
        ),
        runtime_environment=build_repo_runtime_environment_callbacks(
            repo_root,
            runtime_package_names,
            external_dir=runtime_external_dir,
            primary_package_names=runtime_primary_package_names,
            env_var=runtime_env_var,
        ),
    )


def build_agent_supervisor_namespace_context(
    repo_root: Path | str,
    prefix: str,
    *,
    namespace: str | None = None,
    namespace_paths: AgentSupervisorNamespacePaths | None = None,
    task_board_stem: str | None = None,
    task_board_path: Path | str | None = None,
    task_board_docs_dir: Path | str = DEFAULT_REPO_DOCS_DIR,
    task_board_suffix: str = "md",
    task_board_key: str = "todo_path",
    task_board_setting: str | None = None,
    task_board_option: str | None = None,
    objective_path: Path | str | None = None,
    objective_path_key: str = "objective_heap_path",
    objective_path_setting: str | None = None,
    namespace_keys: Iterable[str] = ("state_dir", "worktree_root"),
    directory_keys: Iterable[str] | None = None,
    extra_entries: Iterable[tuple[str, Path | str] | tuple[str, Path | str, str | None]] = (),
    repo_root_key: str = "repo_root",
    runtime_package_names: Sequence[Path | str] = ("ipfs_accelerate", "ipfs_datasets"),
    runtime_external_dir: Path | str = "external",
    runtime_primary_package_names: Sequence[Path | str] | None = None,
    runtime_env_var: str = "PYTHONPATH",
) -> AgentSupervisorNamespaceContext:
    """Build standard repo, namespace, bootstrap, and runtime bindings for a wrapper."""

    root = Path(repo_root)
    if namespace_paths is None:
        if namespace is None:
            raise ValueError("namespace is required when namespace_paths is not provided")
        namespace_paths = agent_supervisor_namespace_paths(root, namespace)
    if task_board_path is None:
        if task_board_stem is None:
            raise ValueError("task_board_stem or task_board_path is required")
        resolved_task_board_path = repo_task_board_path(
            root,
            task_board_stem,
            docs_dir=task_board_docs_dir,
            suffix=task_board_suffix,
        )
    else:
        resolved_task_board_path = _repo_path(root, task_board_path)
    resolved_task_board_option = task_board_path_option() if task_board_option is None else task_board_option
    runtime_bootstrap = build_agent_supervisor_runtime_bootstrap_callbacks(
        root,
        prefix,
        resolved_task_board_path,
        namespace_paths,
        todo_key=task_board_key,
        todo_setting=task_board_setting,
        objective_path=objective_path,
        objective_path_key=objective_path_key,
        objective_path_setting=objective_path_setting,
        namespace_keys=namespace_keys,
        directory_keys=directory_keys,
        extra_entries=extra_entries,
        repo_root_key=repo_root_key,
        runtime_package_names=runtime_package_names,
        runtime_external_dir=runtime_external_dir,
        runtime_primary_package_names=runtime_primary_package_names,
        runtime_env_var=runtime_env_var,
    )
    return AgentSupervisorNamespaceContext(
        repo_root=root,
        env_prefix=prefix,
        task_board_path=resolved_task_board_path,
        task_board_path_key=task_board_key,
        task_board_path_option=resolved_task_board_option,
        namespace_paths=namespace_paths,
        runtime_bootstrap=runtime_bootstrap,
    )
