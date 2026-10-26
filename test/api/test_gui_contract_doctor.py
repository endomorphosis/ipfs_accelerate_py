"""Click, DOM, and backend contracts stay deterministic and fail closed."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

_ACCEL = Path(__file__).resolve().parents[2]
_DATASETS = Path(
    os.environ.get("IPFS_DATASETS_REPO_ROOT", _ACCEL / "ipfs_datasets_py")
)
_SWISSKNIFE = Path(
    os.environ.get("SWISSKNIFE_REPO_ROOT", _ACCEL.parent.parent / "swissknife")
)
for _path in (_ACCEL, _DATASETS):
    text = str(_path)
    if text not in sys.path:
        sys.path.insert(0, text)

from ipfs_accelerate_py.agent_supervisor.gui_contract_doctor import (
    _LOCAL_ZERO_CALL_APPS,
    build_click_dom_backend_contract,
    build_click_worklist,
    doctor_status_receipt,
    posted_calls_from_sources,
    sketch_finite_read_alias,
    sketch_gateway_binding_attribute,
)


def _swissknife_file(relative_path: str) -> Path:
    path = _SWISSKNIFE / relative_path
    if not path.is_file():
        pytest.skip(
            "SwissKnife integration fixture unavailable; set SWISSKNIFE_REPO_ROOT"
        )
    return path


def _census() -> dict:
    return {
        "rows": [
            {"app_id": "file-manager", "selector": "#back-btn", "role": "button", "accessible_name": "Back"},
            {"app_id": "file-manager", "selector": "#copy-a", "role": "button", "accessible_name": "Copy"},
            {"app_id": "file-manager", "selector": "#copy-b", "role": "button", "accessible_name": "Copy"},
            {
                "app_id": "file-manager",
                "selector": ".window-control.minimize",
                "role": "button",
                "accessible_name": "Minimize",
            },
            {
                "app_id": "file-manager",
                "selector": '[data-live-gateway-binding="file-manager.ipfs_kit_py.retrieve_content"]',
                "role": "button",
                "accessible_name": "Invoke ipfs_cat through the desktop gateway",
            },
        ]
    }


def _matrix() -> dict:
    return {
        "rows": [
            {"app_id": "file-manager", "selector": 'button[aria-label="Back"]', "disposition": "unsupported"},
            {"app_id": "file-manager", "selector": 'button[aria-label="Copy"]', "disposition": "unsupported"},
            {"app_id": "file-manager", "selector": 'button[aria-label="${name}"]', "disposition": "unsupported"},
            {"app_id": "file-manager", "selector": '[data-action="minimize"]', "disposition": "local-only"},
            {"app_id": "file-manager", "selector": 'button[aria-label="Missing"]', "disposition": "unsupported"},
            {"app_id": "file-manager", "selector": '[data-action="upload-ipfs"]', "disposition": "unsupported"},
        ]
    }


def test_unique_dom_alias_and_backend_abstain() -> None:
    posted = posted_calls_from_sources(
        [
            (
                "file-manager",
                "fetch('/mcp/tools/call', { body: JSON.stringify({ method: 'request_action', tool: 'mcp++:ipfs_kit_py/artifacts/get', arguments: {} }) });",
            )
        ]
    )
    contract = build_click_dom_backend_contract(
        census=_census(),
        matrix=_matrix(),
        live_bindings={
            "bindings": [
                {
                    "binding_id": "file-manager.ipfs_kit_py.retrieve_content",
                    "owner": "ipfs_kit_py",
                }
            ]
        },
        executable_bindings=[
            {
                "app_id": "file-manager",
                "binding_id": "file-manager.ipfs_kit_py.retrieve_content",
                "preferred_tool_ids": ["ipfs_cat"],
            }
        ],
        posted_calls=posted,
    )
    aliases = {(row["from_selector"], row["to_selector"]) for row in contract["dom_aliases"]}
    assert ('button[aria-label="Back"]', "#back-btn") in aliases
    assert ('[data-action="minimize"]', ".window-control.minimize") in aliases
    assert all(row["operator_disposition"] == "unique" for row in contract["dom_aliases"])
    reasons = {row["selector"]: row["reason"] for row in contract["dom_abstentions"]}
    assert reasons['button[aria-label="Copy"]'] == "ambiguous_accessible_name"
    assert reasons['button[aria-label="${name}"]'] == "template_selector"
    assert reasons['button[aria-label="Missing"]'] == "accessible_name_absent"
    assert reasons['[data-action="upload-ipfs"]'] == "no_dom_name"
    assert contract["ui_ux_ir"]["execution_performed"] is False
    assert contract["ui_ux_ir"]["app_count"] == 1
    assert contract["app_source_rewritten"] is False
    assert contract["mutation_authorized"] is False
    assert contract["world_projection_role"] == "bounded_planning_projection_only"
    gateway = contract["gateway_controls"][0]
    assert gateway["binding_id"] == "file-manager.ipfs_kit_py.retrieve_content"
    assert gateway["catalog_match"] is True
    assert gateway["owner"] == "ipfs_kit_py"
    abstained = contract["backend_abstentions"]
    assert len(abstained) == 1
    assert abstained[0]["tool_id"] == "mcp++:ipfs_kit_py/artifacts/get"
    assert abstained[0]["repair"] == "abstain"
    assert abstained[0]["reason"] == "tool_not_preferred"
    assert contract["preferred_tool_matches"] == []


def test_exact_preferred_tool_is_recorded_without_a_rewrite() -> None:
    contract = build_click_dom_backend_contract(
        census={"rows": []},
        matrix={"rows": []},
        executable_bindings=[
            {
                "app_id": "image-viewer",
                "binding_id": "image-viewer.ipfs_kit_py.retrieve_content",
                "preferred_tool_ids": ["ipfs_cat"],
            }
        ],
        posted_calls=[
            {
                "app_id": "image-viewer",
                "tool_id": "ipfs_cat",
                "source": "image-viewer",
                "envelope": "gateway",
            }
        ],
    )
    assert contract["backend_abstentions"] == []
    match = contract["preferred_tool_matches"][0]
    assert match["binding_id"] == "image-viewer.ipfs_kit_py.retrieve_content"
    assert match["repair"] == "already_preferred_tool"
    assert contract["app_source_rewritten"] is False
    assert contract["ui_ux_ir"]["execution_performed"] is False


def _observation(outcome: str, *, dry_run: bool = True) -> dict:
    return {
        "route": "/mcp/tools/call",
        "tool_id": "ipfs_cat",
        "transport": "http",
        "correlation_id": "image-viewer:retrieve:1",
        "policy_outcome": "require_confirmation",
        "receipt_id": "bafkreiabc",
        "event_dag_cid": "bafkreidef",
        "dry_run": dry_run,
        "outcome": outcome,
    }


def test_worklist_keeps_census_controls_and_does_not_guess_among_tools() -> None:
    census = {
        "rows": [
            {
                "app_id": "file-manager",
                "launch_id": "file-manager",
                "selector": "#back-btn",
                "role": "button",
                "accessible_name": "Back",
            },
            {
                "app_id": "file-manager",
                "launch_id": "file-manager",
                "selector": ".window-control.minimize",
                "role": "button",
                "accessible_name": "Minimize",
            },
            {
                "app_id": "file-manager",
                "launch_id": "file-manager",
                "selector": '[data-live-gateway-binding="file-manager.ipfs_kit_py.retrieve_content"]',
                "role": "button",
                "accessible_name": "Retrieve",
            },
            {
                "app_id": "notes",
                "launch_id": "notes",
                "selector": '[data-live-gateway-binding="notes.ipfs_datasets_py.record_provenance"]',
                "role": "button",
                "accessible_name": "Record",
            },
            {
                "app_id": "file-manager",
                "launch_id": "file-manager",
                "selector": 'button[aria-label="${name}"]',
                "role": "button",
                "accessible_name": "templated",
            },
        ]
    }
    matrix = _matrix()
    live = {
        "bindings": [
            {"binding_id": "file-manager.ipfs_kit_py.retrieve_content", "owner": "ipfs_kit_py"},
            {"binding_id": "notes.ipfs_datasets_py.record_provenance", "owner": "ipfs_datasets_py"},
        ]
    }
    executable = [
        {
            "app_id": "file-manager",
            "binding_id": "file-manager.ipfs_kit_py.retrieve_content",
            "preferred_tool_ids": ["ipfs_cat", "ipfs_ls", "ipfs_add"],
        },
        {
            "app_id": "notes",
            "binding_id": "notes.ipfs_datasets_py.record_provenance",
            "preferred_tool_ids": ["record_provenance"],
        },
    ]
    contract = build_click_dom_backend_contract(
        census=census,
        matrix=matrix,
        live_bindings=live,
        executable_bindings=executable,
    )
    work = build_click_worklist(
        census=census,
        matrix=matrix,
        live_bindings=live,
        executable_bindings=executable,
        dom_aliases=contract["dom_aliases"],
    )
    selectors = [row["selector"] for row in work["rows"]]
    assert "#back-btn" in selectors
    assert 'button[aria-label="${name}"]' not in selectors
    assert all("${" not in row["selector"] for row in work["rows"])
    by_selector = {row["selector"]: row for row in work["rows"]}
    assert by_selector["#back-btn"]["disposition"] == "unsupported"
    assert by_selector[".window-control.minimize"]["disposition"] == "local-only"
    storage = by_selector['[data-live-gateway-binding="file-manager.ipfs_kit_py.retrieve_content"]']
    assert storage["disposition"] == "unlisted"
    assert storage["tool_id"] == ""
    assert storage["reason"] == "multiple_preferred_tools"
    notes = by_selector['[data-live-gateway-binding="notes.ipfs_datasets_py.record_provenance"]']
    assert notes["disposition"] == "invoke"
    assert notes["tool_id"] == "record_provenance"
    assert work["doctor_operation"] == "plan"
    assert work["mutation_authorized"] is False
    assert work["app_source_rewritten"] is False
    assert work["operator"]["matrix_disposition"] == "out_of_scope"
    assert work["operator"]["matrix_rewritten"] is False
    assert contract["mutation_authorized"] is False


def test_unrendered_census_selectors_stay_loss_receipts() -> None:
    census = {
        "rows": [
            {
                "app_id": "image-viewer",
                "launch_id": "image-viewer",
                "selector": "#back-btn",
                "role": "button",
                "accessible_name": "Back",
            },
            {
                "app_id": "image-viewer",
                "launch_id": "image-viewer",
                "selector": "#blur-slider",
                "role": "slider",
                "accessible_name": "Blur",
            },
            {
                "app_id": "image-viewer",
                "launch_id": "image-viewer",
                "selector": '[data-live-gateway-binding="mcp++:ipfs_kit_py/artifacts/get"]',
                "role": "button",
                "accessible_name": "Retrieve",
            },
            {
                "app_id": "image-viewer",
                "launch_id": "image-viewer",
                "selector": '[data-live-gateway-binding="image-viewer.ipfs_kit_py.retrieve_content"]',
                "role": "button",
                "accessible_name": "Retrieve catalog",
            },
            {
                "app_id": "image-viewer",
                "launch_id": "image-viewer",
                "selector": '[data-action="retrieve-cid-image"]',
                "role": "button",
                "accessible_name": "Retrieve CID",
            },
        ]
    }
    live = {
        "bindings": [
            {"binding_id": "image-viewer.ipfs_kit_py.retrieve_content", "owner": "ipfs_kit_py"},
        ]
    }
    executable = [
        {
            "app_id": "image-viewer",
            "binding_id": "image-viewer.ipfs_kit_py.retrieve_content",
            "preferred_tool_ids": ["ipfs_cat", "ipfs_ls", "ipfs_add"],
        }
    ]
    source = (
        '<button id="back-btn" data-action="retrieve-cid-image" '
        'data-live-gateway-binding="image-viewer.ipfs_kit_py.retrieve_content"></button>'
    )
    work = build_click_worklist(
        census=census,
        matrix={"rows": []},
        live_bindings=live,
        executable_bindings=executable,
        app_sources={"image-viewer": source},
    )
    selectors = [row["selector"] for row in work["rows"]]
    assert "#back-btn" in selectors
    assert '[data-action="retrieve-cid-image"]' in selectors
    assert '[data-live-gateway-binding="image-viewer.ipfs_kit_py.retrieve_content"]' in selectors
    assert "#blur-slider" not in selectors
    assert '[data-live-gateway-binding="mcp++:ipfs_kit_py/artifacts/get"]' not in selectors
    reasons = {item["selector"]: item["reason"] for item in work["unrendered_losses"]}
    assert reasons["#blur-slider"] == "dom_id_absent"
    assert reasons['[data-live-gateway-binding="mcp++:ipfs_kit_py/artifacts/get"]'] == (
        "binding_attribute_absent"
    )
    assert work["mutation_authorized"] is False


def test_finite_adapter_sketches_one_get_span_and_leaves_put_bytes() -> None:
    source = (
        "const body = JSON.stringify({ method: 'request_action', "
        "tool: 'mcp++:ipfs_kit_py/artifacts/get', arguments: args });"
    )
    bindings = [
        {
            "app_id": "notes",
            "binding_id": "notes.ipfs_kit_py.retrieve_content",
            "preferred_tool_ids": ["ipfs_cat", "ipfs_ls", "ipfs_add"],
        }
    ]
    path = "swissknife/web/js/apps/notes.js"
    sketch = sketch_finite_read_alias(
        source,
        app_id="notes",
        path=path,
        executable_bindings=bindings,
    )
    assert sketch["disposition"] == "unique"
    assert sketch["applied"] is False
    assert sketch["source_text"] == source
    assert "ipfs_cat" in sketch["rewritten_text"]
    assert "swissknife.all-app-tool-gateway.v1" in sketch["rewritten_text"]
    assert "artifacts/get" not in sketch["rewritten_text"]
    twice = sketch_finite_read_alias(
        source + "\n" + source,
        app_id="notes",
        path=path,
        executable_bindings=bindings,
    )
    assert twice["disposition"] == "ambiguous"
    assert "adapter_span_not_unique" in twice["reason_codes"]
    assert twice["rewritten_text"] == source + "\n" + source
    assert twice["applied"] is False
    put = source.replace("artifacts/get", "artifacts/put")
    refused = sketch_finite_read_alias(
        put,
        app_id="notes",
        path=path,
        executable_bindings=bindings,
    )
    assert "adapter_mapping_incomplete" in refused["reason_codes"]
    assert refused["rewritten_text"] == put
    assert refused["applied"] is False


def test_call_mcp_function_branches_get_and_keeps_put() -> None:
    function = (
        "async function callMcpTool(tool, toolArguments) {\n"
        "    const response = await fetch('/mcp/tools/call', {\n"
        "        method: 'POST',\n"
        "        headers: { 'Content-Type': 'application/json' },\n"
        "        body: JSON.stringify({\n"
        "            method: 'request_action',\n"
        "            tool,\n"
        "            arguments: toolArguments\n"
        "        })\n"
        "    });\n"
        "    return parseToolResponse(response);\n"
        "}"
    )
    source = (
        function
        + "\nconst got = await callMcpTool('mcp++:ipfs_kit_py/artifacts/get', { cid });\n"
        + "const put = await callMcpTool('mcp++:ipfs_kit_py/artifacts/put', { cid });\n"
        + "const dag = await callMcpTool('mcp++:ipfs_kit_py/dag/append', { event });\n"
    )
    bindings = [
        {
            "app_id": "media-player",
            "binding_id": "media-player.ipfs_kit_py.retrieve_content",
            "preferred_tool_ids": ["ipfs_cat", "ipfs_ls", "ipfs_add"],
        }
    ]
    path = "swissknife/web/js/apps/media-player.js"
    sketch = sketch_finite_read_alias(
        source,
        app_id="media-player",
        path=path,
        executable_bindings=bindings,
    )
    assert sketch["disposition"] == "unique"
    assert sketch["applied"] is False
    rewritten = sketch["rewritten_text"]
    assert "tool_id: 'ipfs_cat'" in rewritten
    assert "tool === 'mcp++:ipfs_kit_py/artifacts/get'" in rewritten
    assert rewritten.count("artifacts/put") == 1
    assert "dag/append" in rewritten
    assert "method: 'request_action'" in rewritten
    put_only = source.replace("artifacts/get", "artifacts/put")
    refused = sketch_finite_read_alias(
        put_only,
        app_id="media-player",
        path=path,
        executable_bindings=bindings,
    )
    assert "adapter_mapping_incomplete" in refused["reason_codes"]
    assert refused["rewritten_text"] == put_only
    twice = sketch_finite_read_alias(
        source + "\n" + function,
        app_id="media-player",
        path=path,
        executable_bindings=bindings,
    )
    assert twice["disposition"] == "ambiguous"
    assert twice["rewritten_text"] == source + "\n" + function
    assert twice["applied"] is False


def test_gateway_attribute_span_is_unique_only_once() -> None:
    source = '<button data-live-gateway-binding="mcp++:ipfs_kit_py/artifacts/get"></button>'
    path = "swissknife/web/js/apps/image-viewer.js"
    sketch = sketch_gateway_binding_attribute(source, path=path)
    assert sketch["disposition"] == "unique"
    assert sketch["applied"] is False
    assert "image-viewer.ipfs_kit_py.retrieve_content" in sketch["rewritten_text"]
    assert sketch["source_text"] == source
    twice = sketch_gateway_binding_attribute(source + source, path=path)
    assert twice["disposition"] == "ambiguous"
    assert twice["rewritten_text"] == source + source
    assert twice["applied"] is False


def test_receipt_is_not_owner_execution() -> None:
    validator_path = _swissknife_file(
        "scripts/vog_live_click_requires_observation.py"
    )
    spec = importlib.util.spec_from_file_location(
        "vog_live_click_requires_observation",
        validator_path,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    unreachable = {
        "app_id": "image-viewer",
        "disposition": "invoke",
        "observation": _observation("unreachable"),
    }
    assert module.owner_execution_ok(unreachable) is False
    executed = {
        "app_id": "image-viewer",
        "disposition": "invoke",
        "observation": _observation("executed", dry_run=True),
    }
    assert module.owner_execution_ok(executed) is True
    mutating = {
        "app_id": "image-viewer",
        "disposition": "invoke",
        "observation": _observation("executed", dry_run=False),
    }
    assert module.owner_execution_ok(mutating) is False


def test_doctor_status_stays_report_only() -> None:
    receipt = doctor_status_receipt()
    assert receipt["mode"] == "report_only"
    assert receipt["mutation_authorized"] is False
    assert receipt["completion_authoritative"] is False


def test_local_family_gateway_rows_make_no_tool_choice() -> None:
    helper = _swissknife_file("test/e2e/helpers/live-click-session.ts")
    text = helper.read_text(encoding="utf-8")
    local = text.split("local: {", 1)[1].split("apps: [", 1)[1].split("]", 1)[0]
    listed = {item.strip().strip("'\"") for item in local.split(",") if item.strip()}
    assert listed == set(_LOCAL_ZERO_CALL_APPS)

    census = {
        "rows": [
            {
                "app_id": "calendar",
                "launch_id": "calendar",
                "selector": '[data-live-gateway-binding="calendar.ipfs_kit_py.retrieve_content"]',
                "role": "button",
                "accessible_name": "Retrieve",
            },
            {
                "app_id": "todo",
                "launch_id": "todo",
                "selector": '[data-live-gateway-binding="todo.ipfs_datasets_py.record_provenance"]',
                "role": "button",
                "accessible_name": "Record",
            },
            {
                "app_id": "openrouter",
                "launch_id": "openrouter",
                "selector": 'button[aria-label="Model Info"]',
                "role": "button",
                "accessible_name": "Model Info",
            },
            {
                "app_id": "mcp-control",
                "launch_id": "mcp-control",
                "selector": '[data-live-gateway-binding="mcp-control.ipfs_kit_py.retrieve_content"]',
                "role": "button",
                "accessible_name": "Retrieve",
            },
            {
                "app_id": "github",
                "launch_id": "github",
                "selector": '[data-live-gateway-binding="github.ipfs_kit_py.retrieve_content"]',
                "role": "button",
                "accessible_name": "Retrieve",
            },
            {
                "app_id": "calculator",
                "launch_id": "calculator",
                "selector": ".window-control.close",
                "role": "button",
                "accessible_name": "Close",
            },
        ]
    }
    matrix = {
        "rows": [
            {
                "app_id": "openrouter",
                "selector": 'button[aria-label="Model Info"]',
                "disposition": "external-provider",
                "tool_id": "",
            }
        ]
    }
    live = {
        "bindings": [
            {"binding_id": "calendar.ipfs_kit_py.retrieve_content"},
            {"binding_id": "todo.ipfs_datasets_py.record_provenance"},
            {"binding_id": "mcp-control.ipfs_kit_py.retrieve_content"},
            {"binding_id": "github.ipfs_kit_py.retrieve_content"},
        ]
    }
    executable = [
        {
            "app_id": "calendar",
            "binding_id": "calendar.ipfs_kit_py.retrieve_content",
            "preferred_tool_ids": ["ipfs_cat", "ipfs_ls", "ipfs_add"],
        },
        {
            "app_id": "todo",
            "binding_id": "todo.ipfs_datasets_py.record_provenance",
            "preferred_tool_ids": ["record_provenance"],
        },
        {
            "app_id": "mcp-control",
            "binding_id": "mcp-control.ipfs_kit_py.retrieve_content",
            "preferred_tool_ids": ["ipfs_cat"],
        },
        {
            "app_id": "github",
            "binding_id": "github.ipfs_kit_py.retrieve_content",
            "preferred_tool_ids": ["ipfs_cat", "ipfs_ls"],
        },
    ]
    work = build_click_worklist(
        census=census,
        matrix=matrix,
        live_bindings=live,
        executable_bindings=executable,
    )
    by_app = {row["app_id"]: row for row in work["rows"]}
    calendar = by_app["calendar"]
    assert calendar["disposition"] == "unsupported"
    assert calendar["tool_id"] == ""
    assert calendar["reason"] == "local_family_zero_call"
    todo = by_app["todo"]
    assert todo["disposition"] == "unsupported"
    assert todo["tool_id"] == ""
    assert todo["reason"] == "local_family_zero_call"
    assert by_app["openrouter"]["disposition"] == "external-provider"
    assert by_app["mcp-control"]["disposition"] == "invoke"
    assert by_app["mcp-control"]["tool_id"] == "ipfs_cat"
    assert by_app["github"]["disposition"] == "unlisted"
    assert by_app["github"]["tool_id"] == ""
    assert by_app["calculator"]["disposition"] == "local-only"
    assert work["mutation_authorized"] is False
    assert work["doctor_operation"] == "plan"
