from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.interface_contract_codegen import (
    ActionContractSyncTarget,
    JavaScriptActionContractConfig,
    PythonActionContractConfig,
    load_action_definitions_from_descriptor,
    operation_action_mapper,
    render_js_action_contract,
    render_python_action_contract,
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
