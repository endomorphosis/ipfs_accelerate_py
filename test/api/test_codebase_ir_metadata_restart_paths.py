"""Native restart uses the loaded datasets package despite a shadow path."""

import json
from pathlib import Path
import sys

import ipfs_datasets_py

from benchmarks.agent_supervisor.container_coding import codebase_ir_metadata as metadata


def test_native_restart_prefers_loaded_package_to_unrelated_shadow(tmp_path, monkeypatch):
    # Preload the genuine native history before introducing the unrelated clone.
    # The test exercises real DuckDB/DuckLake; no subprocess or native API is mocked.
    history = metadata._history()
    package_file = Path(ipfs_datasets_py.__file__)
    assert package_file.is_absolute() and package_file.resolve(strict=True) == package_file
    assert Path(history.__file__).resolve(strict=True).parent == package_file.parent / "ducklake"
    assert hasattr(history, "IsolatedNativeDuckLakeHistory")
    assert sys.modules["ipfs_datasets_py"] is ipfs_datasets_py

    shadow = tmp_path / "unrelated-clone"
    shadow_package = shadow / "ipfs_datasets_py"
    shadow_package.mkdir(parents=True)
    marker = tmp_path / "shadow-package-imported"
    (shadow_package / "__init__.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('shadow package executed', encoding='utf-8')\n"
        "SHADOW_PACKAGE = True\n", encoding="utf-8")
    assert not (shadow_package / "ducklake").exists()
    monkeypatch.syspath_prepend(str(shadow))
    assert sys.path[0] == str(shadow)

    payload = {"row_id": "source-unit", "symbol": "increment", "line": 1,
               "unicode": "é", "candidate": True, "proof_authority": False}
    output = tmp_path / "metadata"
    report = metadata.hydrate_codebase_ir_metadata(
        records={"ast": [payload]}, output=output,
        source_snapshot={"schema": "restart-path-native-fixture@1", "snapshot": "one-source-unit"})

    # _fresh compares its child's complete native report to the parent report.
    # A second local validation independently reopens both native stores.
    local = metadata.validate_codebase_ir_metadata(output=output, expected=report)
    assert local == {key: value for key, value in report.items() if key != "fresh_process_readback"}
    assert report["family_counts"] == {"ast": 1, "contracts": 0, "kg": 0, "vectors": 0}
    assert report["lake_packet_count"] == report["row_count"] == 1
    assert len(report["lake_snapshot_ids"]) == 1
    assert report["fresh_process_readback"] == {
        "verified": True,
        "method": "new_python_process_native_duckdb_and_ducklake_readback",
        "manifest_sha256": local["manifest_sha256"],
        "row_root_sha256": local["row_root_sha256"],
        "row_count": 1,
        "lake_snapshot_digest": local["lake_snapshot_digest"],
    }
    exported = (output / report["exports"]["ast"]["relative_path"]).read_text().splitlines()
    assert len(exported) == 1 and json.loads(exported[0])["payload"] == payload
    assert not marker.exists(), "restart executed the unrelated shadow package"
