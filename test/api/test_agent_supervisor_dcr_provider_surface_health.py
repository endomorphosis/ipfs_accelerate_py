"""DCR-013 static multi-root provider surface tests."""

from __future__ import annotations

from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.analysis.provider_surface_health import (
    ProviderSurfaceExpectation,
    SurfaceStatus,
    scan_provider_surfaces,
)


def put(root: Path, name: str, body: str) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


def test_static_registration_alias_and_indirect_dispatcher_are_bound_stably(tmp_path: Path) -> None:
    accelerate, datasets, kit, mcp = (
        tmp_path / name for name in ("accelerate", "datasets", "kit", "mcp")
    )
    put(
        accelerate,
        "a.py",
        '@registry.register_tool("op", handler=handle, dispatcher="tools.call", effect="read", request_schema="Req@1", result_schema="Res@1", error_schema="Err@1", aliases="old-op")\ndef handle(): pass\n',
    )
    put(
        datasets,
        "b.py",
        'server.register(name="other", callback=worker, dispatcher="router", effect="write", request="Req@1", result="Res@1", error="Err@1")\ndef worker(): pass\n',
    )
    put(kit, "c.py", "x = 1\n")
    put(mcp, "d.py", "x = 2\n")
    roots = {"accelerate": accelerate, "datasets": datasets, "kit": kit, "mcp++": mcp}
    one = scan_provider_surfaces(
        roots,
        forest_identity="forest:1",
        index_identity="index:1",
        expectations=(ProviderSurfaceExpectation("op"), ProviderSurfaceExpectation("other")),
    )
    two = scan_provider_surfaces(
        roots,
        forest_identity="forest:1",
        index_identity="index:1",
        expectations=(ProviderSurfaceExpectation("other"), ProviderSurfaceExpectation("op")),
    )
    assert one.to_dict() == two.to_dict() and one.parity_ready
    assert one.rows[0].aliases == ("old-op",) and one.rows[0].dispatcher


def test_duplicates_syntax_failures_and_unresolved_mandatory_rows_block_parity(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    body = 'r.register("dup", handler=h, dispatcher="d", effect="e", request_schema="q", result_schema="r", error_schema="x")\n'
    put(root, "a.py", body)
    put(root, "b.py", body)
    put(root, "bad.py", "def broken(:\n")
    report = scan_provider_surfaces(
        {"accelerate": root},
        forest_identity="forest:1",
        index_identity="index:1",
        expectations=(ProviderSurfaceExpectation("missing"),),
    )
    assert not report.parity_ready
    assert any(row.status is SurfaceStatus.AMBIGUOUS for row in report.rows)
    assert any(row.status is SurfaceStatus.PARSER_FAILURE for row in report.rows)
    assert "missing" in report.mandatory_unresolved
