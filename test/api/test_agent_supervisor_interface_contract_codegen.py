from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.interface_contract_codegen import (
    ActionContractCodegenConfig,
    ActionContractSyncSpec,
    ActionContractSyncTarget,
    ConfiguredActionContractSyncRunner,
    JavaScriptActionContractConfig,
    PythonActionContractConfig,
    build_action_contract_sync_arg_parser,
    build_action_contract_sync_runner_from_spec,
    build_action_contract_sync_targets,
    build_configured_action_contract_sync_runner,
    load_action_definitions_from_descriptor,
    operation_action_mapper,
    render_js_action_contract,
    render_python_action_contract,
    run_action_contract_sync,
    sync_contract_targets,
)


def _write_descriptor(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "methods": [
                    {
                        "name": "render_widget",
                        "outputSchema": {
                            "properties": {
                                "type": {"const": "mobile_render_display_widget"},
                            },
                        },
                    },
                    {
                        "name": "focus_next",
                        "outputSchema": {
                            "properties": {
                                "type": {"const": "mobile_focus_display_widget"},
                            },
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def test_interface_contract_codegen_projects_descriptor_and_renders_modules(tmp_path: Path):
    descriptor_path = tmp_path / "interface.json"
    _write_descriptor(descriptor_path)

    definitions = load_action_definitions_from_descriptor(
        descriptor_path,
        operation_to_action=operation_action_mapper(
            {
                "render_widget": "render",
                "focus_next": "focus",
            },
            label="test operation",
        ),
        action_metadata={
            "render": {
                "label": "Render Widget",
                "phrase": "render the widget",
                "dat_method": "renderWidget",
            },
            "focus": {
                "label": "Focus Widget",
                "phrase": "focus the widget",
                "dat_method": "focusWidget",
            },
        },
    )

    assert definitions == [
        {
            "action": "render",
            "operation": "render_widget",
            "id": "mobile_render_display_widget",
            "label": "Render Widget",
            "phrase": "render the widget",
            "dat_method": "renderWidget",
        },
        {
            "action": "focus",
            "operation": "focus_next",
            "id": "mobile_focus_display_widget",
            "label": "Focus Widget",
            "phrase": "focus the widget",
            "dat_method": "focusWidget",
        },
    ]

    python_module = render_python_action_contract(
        definitions,
        contract="example.contract@1",
        config=PythonActionContractConfig(
            contract_name="WIDGET_CONTRACT",
            definitions_name="WIDGET_DEFINITIONS",
            ids_name="WIDGET_IDS",
            operations_name="WIDGET_OPERATIONS",
            docstring="Generated widget contract.",
        ),
    )
    js_module = render_js_action_contract(
        definitions,
        contract="example.contract@1",
        config=JavaScriptActionContractConfig(
            contract_name="WIDGET_CONTRACT",
            ids_name="WIDGET_IDS",
            ids_set_name="WIDGET_ID_SET",
            action_by_id_name="WIDGET_ACTION_BY_ID",
            operation_by_id_name="WIDGET_OPERATION_BY_ID",
            validator_function_name="isWidgetActionId",
            extra_id_maps={"dat_method": "WIDGET_DAT_METHOD_BY_ID"},
        ),
    )

    assert 'WIDGET_CONTRACT: Final = "example.contract@1"' in python_module
    assert '"operation": "focus_next"' in python_module
    assert "export const WIDGET_DAT_METHOD_BY_ID = {" in js_module
    assert "mobile_focus_display_widget: 'focusWidget'," in js_module
    assert "return WIDGET_ID_SET.has(actionId);" in js_module


def test_interface_contract_codegen_sync_targets_check_and_write(tmp_path: Path, capsys):
    repo_root = tmp_path / "repo"
    target_path = repo_root / "generated" / "contract.py"
    target = ActionContractSyncTarget(target_path, "generated = True\n")

    assert sync_contract_targets((target,), check=True, write=False, repo_root=repo_root) is True
    assert capsys.readouterr().out == "drift:generated/contract.py\n"
    assert not target_path.exists()

    assert sync_contract_targets((target,), check=False, write=True, repo_root=repo_root) is True
    assert capsys.readouterr().out == "updated:generated/contract.py\n"
    assert target_path.read_text(encoding="utf-8") == "generated = True\n"

    assert sync_contract_targets((target,), check=True, write=False, repo_root=repo_root) is False
    assert capsys.readouterr().out == ""


def test_interface_contract_codegen_rejects_unknown_operation(tmp_path: Path):
    descriptor_path = tmp_path / "interface.json"
    _write_descriptor(descriptor_path)

    with pytest.raises(ValueError, match="Unsupported test operation: focus_next"):
        load_action_definitions_from_descriptor(
            descriptor_path,
            operation_to_action=operation_action_mapper(
                {"render_widget": "render"},
                label="test operation",
            ),
            action_metadata={"render": {}},
        )


def test_interface_contract_codegen_runner_defaults_to_check_and_can_write(tmp_path: Path, capsys):
    descriptor_path = tmp_path / "interface.json"
    _write_descriptor(descriptor_path)
    config = ActionContractCodegenConfig(
        descriptor_path=descriptor_path,
        contract="example.contract@1",
        operation_to_action=operation_action_mapper(
            {
                "render_widget": "render",
                "focus_next": "focus",
            }
        ),
        action_metadata={
            "render": {
                "label": "Render Widget",
                "phrase": "render the widget",
                "dat_method": "renderWidget",
            },
            "focus": {
                "label": "Focus Widget",
                "phrase": "focus the widget",
                "dat_method": "focusWidget",
            },
        },
        python_target_path=tmp_path / "pkg" / "widget_contract.py",
        python_config=PythonActionContractConfig(
            contract_name="WIDGET_CONTRACT",
            definitions_name="WIDGET_DEFINITIONS",
            ids_name="WIDGET_IDS",
            operations_name="WIDGET_OPERATIONS",
            docstring="Generated widget contract.",
        ),
        js_target_path=tmp_path / "web" / "widgetContract.js",
        js_config=JavaScriptActionContractConfig(
            contract_name="WIDGET_CONTRACT",
            ids_name="WIDGET_IDS",
            ids_set_name="WIDGET_ID_SET",
            action_by_id_name="WIDGET_ACTION_BY_ID",
            operation_by_id_name="WIDGET_OPERATION_BY_ID",
            validator_function_name="isWidgetActionId",
            extra_id_maps={"dat_method": "WIDGET_DAT_METHOD_BY_ID"},
        ),
        repo_root=tmp_path,
        description="Sync test widget contracts.",
    )

    parser = build_action_contract_sync_arg_parser(config)
    assert parser.description == "Sync test widget contracts."
    targets = build_action_contract_sync_targets(config)
    assert [target.path for target in targets] == [config.python_target_path, config.js_target_path]

    assert run_action_contract_sync(config, []) == 1
    assert capsys.readouterr().out == "drift:pkg/widget_contract.py\ndrift:web/widgetContract.js\n"
    assert not config.python_target_path.exists()
    assert not config.js_target_path.exists()

    assert run_action_contract_sync(config, ["--write"]) == 0
    assert capsys.readouterr().out == "updated:pkg/widget_contract.py\nupdated:web/widgetContract.js\n"
    assert config.python_target_path.exists()
    assert config.js_target_path.exists()

    assert run_action_contract_sync(config, ["--check"]) == 0
    assert capsys.readouterr().out == ""


def test_configured_action_contract_sync_runner_reuses_binding(tmp_path: Path, capsys):
    descriptor_path = tmp_path / "interface.json"
    _write_descriptor(descriptor_path)

    runner = build_configured_action_contract_sync_runner(
        descriptor_path=str(descriptor_path),
        contract="example.contract@1",
        operation_to_action=operation_action_mapper(
            {
                "render_widget": "render",
                "focus_next": "focus",
            }
        ),
        action_metadata={
            "render": {
                "label": "Render Widget",
                "phrase": "render the widget",
                "dat_method": "renderWidget",
            },
            "focus": {
                "label": "Focus Widget",
                "phrase": "focus the widget",
                "dat_method": "focusWidget",
            },
        },
        python_target_path=str(tmp_path / "pkg" / "widget_contract.py"),
        python_config=PythonActionContractConfig(
            contract_name="WIDGET_CONTRACT",
            definitions_name="WIDGET_DEFINITIONS",
            ids_name="WIDGET_IDS",
            operations_name="WIDGET_OPERATIONS",
            docstring="Generated widget contract.",
        ),
        js_target_path=str(tmp_path / "web" / "widgetContract.js"),
        js_config=JavaScriptActionContractConfig(
            contract_name="WIDGET_CONTRACT",
            ids_name="WIDGET_IDS",
            ids_set_name="WIDGET_ID_SET",
            action_by_id_name="WIDGET_ACTION_BY_ID",
            operation_by_id_name="WIDGET_OPERATION_BY_ID",
            validator_function_name="isWidgetActionId",
            extra_id_maps={"dat_method": "WIDGET_DAT_METHOD_BY_ID"},
        ),
        repo_root=str(tmp_path),
        description="Sync test widget contracts.",
    )

    parsed = runner.parse_args(["--write"])
    targets = runner.build_targets()

    assert isinstance(runner, ConfiguredActionContractSyncRunner)
    assert parsed.write is True
    assert parsed.check is False
    assert runner.config.descriptor_path == descriptor_path
    assert runner.config.repo_root == tmp_path
    assert [target.path for target in targets] == [
        tmp_path / "pkg" / "widget_contract.py",
        tmp_path / "web" / "widgetContract.js",
    ]

    assert runner.run(["--write"]) == 0
    assert capsys.readouterr().out == "updated:pkg/widget_contract.py\nupdated:web/widgetContract.js\n"
    assert runner.run(["--check"]) == 0
    assert capsys.readouterr().out == ""


def test_action_contract_sync_spec_binds_repo_relative_paths(tmp_path: Path, capsys):
    repo_root = tmp_path / "repo"
    descriptor_path = repo_root / "spec" / "interface.json"
    descriptor_path.parent.mkdir(parents=True)
    _write_descriptor(descriptor_path)

    sync_spec = ActionContractSyncSpec(
        descriptor_path="spec/interface.json",
        contract="example.contract@1",
        operation_to_action={
            "render_widget": "render",
            "focus_next": "focus",
        },
        operation_label="portable operation",
        action_metadata={
            "render": {
                "label": "Render Widget",
                "phrase": "render the widget",
                "dat_method": "renderWidget",
            },
            "focus": {
                "label": "Focus Widget",
                "phrase": "focus the widget",
                "dat_method": "focusWidget",
            },
        },
        python_target_path="pkg/widget_contract.py",
        python_config=PythonActionContractConfig(
            contract_name="WIDGET_CONTRACT",
            definitions_name="WIDGET_DEFINITIONS",
            ids_name="WIDGET_IDS",
            operations_name="WIDGET_OPERATIONS",
            docstring="Generated widget contract.",
        ),
        js_target_path="web/widgetContract.js",
        js_config=JavaScriptActionContractConfig(
            contract_name="WIDGET_CONTRACT",
            ids_name="WIDGET_IDS",
            ids_set_name="WIDGET_ID_SET",
            action_by_id_name="WIDGET_ACTION_BY_ID",
            operation_by_id_name="WIDGET_OPERATION_BY_ID",
            validator_function_name="isWidgetActionId",
            extra_id_maps={"dat_method": "WIDGET_DAT_METHOD_BY_ID"},
        ),
        description="Sync portable widget contracts.",
    )

    runner = build_action_contract_sync_runner_from_spec(
        repo_root=repo_root,
        sync_spec=sync_spec,
    )

    assert isinstance(runner, ConfiguredActionContractSyncRunner)
    assert runner.config.descriptor_path == repo_root / "spec" / "interface.json"
    assert runner.config.python_target_path == repo_root / "pkg" / "widget_contract.py"
    assert runner.config.js_target_path == repo_root / "web" / "widgetContract.js"
    assert runner.config.description == "Sync portable widget contracts."

    assert runner.run(["--write"]) == 0
    assert capsys.readouterr().out == "updated:pkg/widget_contract.py\nupdated:web/widgetContract.js\n"

    with pytest.raises(ValueError, match="Unsupported portable operation: missing"):
        runner.config.operation_to_action("missing")
