"""Reusable interface-descriptor action contract rendering helpers."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


ActionDefinition = dict[str, str]


@dataclass(frozen=True)
class PythonActionContractConfig:
    """Export names and docstring for a generated Python action contract module."""

    contract_name: str
    definitions_name: str
    ids_name: str
    operations_name: str
    docstring: str
    definition_fields: Sequence[str] = ("action", "operation", "id", "label", "phrase")


@dataclass(frozen=True)
class JavaScriptActionContractConfig:
    """Export names for a generated JavaScript action contract module."""

    contract_name: str
    ids_name: str
    ids_set_name: str
    action_by_id_name: str
    operation_by_id_name: str
    validator_function_name: str
    extra_id_maps: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionContractSyncTarget:
    """One generated contract artifact to verify or update."""

    path: Path
    content: str


@dataclass(frozen=True)
class ActionContractSyncSpec:
    """Portable action-contract sync inputs using repo-relative artifact paths."""

    descriptor_path: Path | str
    contract: str
    operation_to_action: Mapping[str, str]
    action_metadata: Mapping[str, Mapping[str, str]]
    python_target_path: Path | str
    python_config: PythonActionContractConfig
    js_target_path: Path | str
    js_config: JavaScriptActionContractConfig
    operation_label: str = "operation"
    description: str = "Sync or verify generated action contract modules."


@dataclass(frozen=True)
class ActionContractCodegenConfig:
    """Inputs for syncing generated action contract modules from a descriptor."""

    descriptor_path: Path
    contract: str
    operation_to_action: Callable[[str], str]
    action_metadata: Mapping[str, Mapping[str, str]]
    python_target_path: Path
    python_config: PythonActionContractConfig
    js_target_path: Path
    js_config: JavaScriptActionContractConfig
    repo_root: Path | None = None
    description: str = "Sync or verify generated action contract modules."


@dataclass(frozen=True)
class ConfiguredActionContractSyncRunner:
    """Project-bound runner wiring for action-contract sync CLIs."""

    config: ActionContractCodegenConfig

    def parse_args(self, argv: Sequence[str] | None = None) -> argparse.Namespace:
        """Parse action-contract sync CLI args using the bound config."""

        return build_action_contract_sync_arg_parser(self.config).parse_args(argv)

    def build_targets(self) -> tuple[ActionContractSyncTarget, ...]:
        """Render sync targets from the bound config."""

        return build_action_contract_sync_targets(self.config)

    def run(self, argv: Sequence[str] | None = None) -> int:
        """Run the configured action-contract sync CLI."""

        return run_action_contract_sync(self.config, argv)


def build_configured_action_contract_sync_runner(
    *,
    descriptor_path: Path | str,
    contract: str,
    operation_to_action: Callable[[str], str],
    action_metadata: Mapping[str, Mapping[str, str]],
    python_target_path: Path | str,
    python_config: PythonActionContractConfig,
    js_target_path: Path | str,
    js_config: JavaScriptActionContractConfig,
    repo_root: Path | str | None = None,
    description: str = "Sync or verify generated action contract modules.",
) -> ConfiguredActionContractSyncRunner:
    """Build reusable action-contract sync runner wiring bound to project inputs."""

    return ConfiguredActionContractSyncRunner(
        ActionContractCodegenConfig(
            descriptor_path=Path(descriptor_path),
            contract=contract,
            operation_to_action=operation_to_action,
            action_metadata=action_metadata,
            python_target_path=Path(python_target_path),
            python_config=python_config,
            js_target_path=Path(js_target_path),
            js_config=js_config,
            repo_root=Path(repo_root) if repo_root is not None else None,
            description=description,
        )
    )


def _repo_bound_path(repo_root: Path, path: Path | str) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return repo_root / value


def build_action_contract_sync_runner_from_spec(
    *,
    repo_root: Path | str,
    sync_spec: ActionContractSyncSpec,
) -> ConfiguredActionContractSyncRunner:
    """Build an action-contract sync runner from a reusable repo-relative spec."""

    root = Path(repo_root)
    return build_configured_action_contract_sync_runner(
        descriptor_path=_repo_bound_path(root, sync_spec.descriptor_path),
        contract=sync_spec.contract,
        operation_to_action=operation_action_mapper(
            sync_spec.operation_to_action,
            label=sync_spec.operation_label,
        ),
        action_metadata=sync_spec.action_metadata,
        python_target_path=_repo_bound_path(root, sync_spec.python_target_path),
        python_config=sync_spec.python_config,
        js_target_path=_repo_bound_path(root, sync_spec.js_target_path),
        js_config=sync_spec.js_config,
        repo_root=root,
        description=sync_spec.description,
    )


def operation_action_mapper(mapping: Mapping[str, str], *, label: str = "operation"):
    """Return a mapper that converts descriptor operation names into action names."""

    values = dict(mapping)

    def mapper(operation: str) -> str:
        try:
            return values[operation]
        except KeyError as exc:
            raise ValueError(f"Unsupported {label}: {operation}") from exc

    return mapper


def _python_string(value: str) -> str:
    return json.dumps(str(value))


def _js_string(value: str) -> str:
    escaped = str(value).replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")
    return f"'{escaped}'"


def _method_action_id(method: Mapping[str, Any]) -> str:
    try:
        return str(method["outputSchema"]["properties"]["type"]["const"])
    except KeyError as exc:
        operation = str(method.get("name") or "<unknown>")
        raise ValueError(f"Descriptor method {operation!r} is missing outputSchema.properties.type.const") from exc


def load_action_definitions_from_descriptor(
    descriptor_path: Path,
    *,
    operation_to_action: Callable[[str], str],
    action_metadata: Mapping[str, Mapping[str, str]],
) -> list[ActionDefinition]:
    """Load ordered action definitions from an interface descriptor JSON file."""

    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    definitions: list[ActionDefinition] = []
    for method in descriptor.get("methods", ()):
        operation = str(method["name"])
        action = str(operation_to_action(operation))
        metadata = dict(action_metadata.get(action, {}))
        definitions.append(
            {
                "action": action,
                "operation": operation,
                "id": _method_action_id(method),
                **metadata,
            }
        )
    return definitions


def render_python_action_contract(
    definitions: Sequence[Mapping[str, str]],
    *,
    contract: str,
    config: PythonActionContractConfig,
) -> str:
    """Render a Python constants module for action ids and ORB operations."""

    tuple_entries = "\n".join(
        [
            "    {\n"
            + "\n".join(
                f'        "{field_name}": {_python_string(definition[field_name])},'
                for field_name in config.definition_fields
            )
            + "\n    },"
            for definition in definitions
        ]
    )
    docstring = config.docstring.strip()
    return (
        f'"""{docstring}\n"""\n\n'
        "from __future__ import annotations\n\n"
        "from typing import Final\n\n\n"
        f"{config.contract_name}: Final = {_python_string(contract)}\n\n"
        f"{config.definitions_name}: Final[tuple[dict[str, str], ...]] = (\n"
        f"{tuple_entries}\n"
        ")\n\n"
        f"{config.ids_name}: Final[tuple[str, ...]] = tuple(\n"
        f'    definition["id"] for definition in {config.definitions_name}\n'
        ")\n\n"
        f"{config.operations_name}: Final[tuple[str, ...]] = tuple(\n"
        f'    definition["operation"] for definition in {config.definitions_name}\n'
        ")\n"
    )


def _render_js_id_map(definitions: Sequence[Mapping[str, str]], definition_key: str) -> str:
    return "\n".join(
        f"  {definition['id']}: {_js_string(definition[definition_key])},"
        for definition in definitions
    )


def render_js_action_contract(
    definitions: Sequence[Mapping[str, str]],
    *,
    contract: str,
    config: JavaScriptActionContractConfig,
) -> str:
    """Render a JavaScript constants module for action ids and bridge maps."""

    action_ids = "\n".join(f"  {_js_string(definition['id'])}," for definition in definitions)
    chunks = [
        f"export const {config.contract_name} =\n  {_js_string(contract)};\n\n",
        f"export const {config.ids_name} = [\n{action_ids}\n];\n\n",
        f"export const {config.action_by_id_name} = {{\n"
        f"{_render_js_id_map(definitions, 'action')}\n"
        "};\n\n",
        f"export const {config.operation_by_id_name} = {{\n"
        f"{_render_js_id_map(definitions, 'operation')}\n"
        "};\n\n",
    ]
    for definition_key, export_name in config.extra_id_maps.items():
        chunks.append(
            f"export const {export_name} = {{\n"
            f"{_render_js_id_map(definitions, definition_key)}\n"
            "};\n\n"
        )
    chunks.extend(
        [
            f"const {config.ids_set_name} = new Set({config.ids_name});\n\n",
            f"export function {config.validator_function_name}(actionId) {{\n"
            f"  return {config.ids_set_name}.has(actionId);\n"
            "}\n",
        ]
    )
    return "".join(chunks)


def write_contract_target(
    target: ActionContractSyncTarget,
    *,
    check: bool,
    write: bool,
    repo_root: Path | None = None,
) -> bool:
    """Verify or update one generated artifact and report whether it changed."""

    existing = target.path.read_text(encoding="utf-8") if target.path.exists() else ""
    if existing == target.content:
        return False
    label = target.path
    if repo_root is not None:
        try:
            label = target.path.relative_to(repo_root)
        except ValueError:
            label = target.path
    if check:
        print(f"drift:{label}")
        return True
    if write:
        target.path.parent.mkdir(parents=True, exist_ok=True)
        target.path.write_text(target.content, encoding="utf-8")
        print(f"updated:{label}")
        return True
    print(f"would-update:{label}")
    return True


def sync_contract_targets(
    targets: Sequence[ActionContractSyncTarget],
    *,
    check: bool,
    write: bool,
    repo_root: Path | None = None,
) -> bool:
    """Verify or update generated contract artifacts and return whether any changed."""

    changed = False
    for target in targets:
        changed |= write_contract_target(target, check=check, write=write, repo_root=repo_root)
    return changed


def build_action_contract_sync_targets(config: ActionContractCodegenConfig) -> tuple[ActionContractSyncTarget, ...]:
    """Render the configured generated contract artifacts."""

    definitions = load_action_definitions_from_descriptor(
        config.descriptor_path,
        operation_to_action=config.operation_to_action,
        action_metadata=config.action_metadata,
    )
    return (
        ActionContractSyncTarget(
            config.python_target_path,
            render_python_action_contract(
                definitions,
                contract=config.contract,
                config=config.python_config,
            ),
        ),
        ActionContractSyncTarget(
            config.js_target_path,
            render_js_action_contract(
                definitions,
                contract=config.contract,
                config=config.js_config,
            ),
        ),
    )


def build_action_contract_sync_arg_parser(config: ActionContractCodegenConfig) -> argparse.ArgumentParser:
    """Build the standard check/write parser for action contract sync wrappers."""

    parser = argparse.ArgumentParser(description=config.description)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Overwrite generated contract modules with rendered content.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if rendered content differs from checked-in modules.",
    )
    return parser


def run_action_contract_sync(config: ActionContractCodegenConfig, argv: Sequence[str] | None = None) -> int:
    """Run the standard action-contract sync CLI."""

    args = build_action_contract_sync_arg_parser(config).parse_args(argv)
    if not args.check and not args.write:
        args.check = True
    changed = sync_contract_targets(
        build_action_contract_sync_targets(config),
        check=bool(args.check),
        write=bool(args.write),
        repo_root=config.repo_root,
    )
    return 1 if args.check and changed else 0
