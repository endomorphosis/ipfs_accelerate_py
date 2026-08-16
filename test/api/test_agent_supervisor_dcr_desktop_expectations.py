"""DCR-014 static SwissKnife desktop expectation inventory tests."""

from __future__ import annotations

from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.analysis.deterministic_desktop_expectations import (
    DesktopAuthorityClass,
    capture_desktop_expectations,
)


def test_static_inventory_binds_spans_consumers_precedence_and_blockers(
    tmp_path: Path,
) -> None:
    swissknife = tmp_path / "swissknife"
    mcp = tmp_path / "Mcp-Plus-Plus"
    (swissknife / "src/desktop").mkdir(parents=True)
    (swissknife / "src/desktop/client.ts").write_text(
        "const desktopMcp = client.callTool('desktop.open', { request: 'OpenRequest' });\n"
        "const ui = { onClick: 'open-window', operation: 'desktop.open', version: '1', request: 'OpenRequest', result: 'OpenResult', error: 'Denied', transport: 'http' };\n",
        encoding="utf-8",
    )
    (swissknife / "src/desktop/unresolved_client.ts").write_text(
        "const desktopMcp = createMcpClient();\n", encoding="utf-8"
    )
    (mcp / "idl").mkdir(parents=True)
    (mcp / "idl/desktop.idl").write_text(
        "operation: 'desktop.open', version: '1', request: 'OpenRequest', result: 'OpenResult', error: 'Denied', transport: 'http'\n",
        encoding="utf-8",
    )
    (mcp / "tests").mkdir()
    (mcp / "tests/desktop.test.ts").write_text(
        "operation: 'desktop.open', version: '1', request: 'OtherRequest', result: 'OpenResult', error: 'Denied', transport: 'http'\n",
        encoding="utf-8",
    )
    (mcp / "archive").mkdir()
    (mcp / "archive/desktop.ts").write_text(
        "operation: 'desktop.open', version: '0', request: 'LegacyRequest'\n",
        encoding="utf-8",
    )

    inventory = capture_desktop_expectations(swissknife_root=swissknife, mcp_plus_plus_root=mcp)

    assert inventory["scan_mode"] == "static_source_only"
    assert inventory["identity"].startswith("sha256:")
    assert len(inventory["consumers"]) == 2
    assert all(
        item["source_span"]["sha256"].startswith("sha256:") for item in inventory["evidence"]
    )
    effective = inventory["effective_expectations"]
    assert len(effective) == 1
    assert effective[0]["authority_class"] == DesktopAuthorityClass.REVIEWED_DECLARATION.value
    assert effective[0]["ui_action"] == ""
    assert {item["kind"] for item in inventory["blockers"]} == {
        "contradictory_desktop_expectation",
        "unresolved_desktop_mcp_consumer",
    }


def test_scan_is_deterministic_and_does_not_require_application_imports(tmp_path: Path) -> None:
    swissknife = tmp_path / "swissknife"
    mcp = tmp_path / "Mcp-Plus-Plus"
    (swissknife / "ui").mkdir(parents=True)
    (mcp / "registries").mkdir(parents=True)
    (swissknife / "ui/desktop.ts").write_text(
        "mcp operation: 'desktop.status', request: 'StatusRequest'\n", encoding="utf-8"
    )
    (mcp / "registries/tools.json").write_text(
        '{"operation": "desktop.status", "request": "StatusRequest"}\n',
        encoding="utf-8",
    )

    first = capture_desktop_expectations(swissknife_root=swissknife, mcp_plus_plus_root=mcp)
    second = capture_desktop_expectations(swissknife_root=swissknife, mcp_plus_plus_root=mcp)

    assert first == second
    assert not first["blockers"]
