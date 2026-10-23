"""Validation-gap rules must travel with a relocated supervisor checkout."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.rescue import lane_resume


def test_packaged_refill_runs_without_workspace_siblings(tmp_path, monkeypatch) -> None:
    """Load a copied lane and its packaged rules without either datasets path."""

    source = Path(lane_resume.__file__).resolve().parent
    package_root = tmp_path / "relocated_rescue"
    package_root.mkdir()
    for name in ("lane_resume.py", "board_gap_refill.py"):
        (package_root / name).write_bytes((source / name).read_bytes())

    package_name = "relocated_vog_rescue"
    package = types.ModuleType(package_name)
    package.__path__ = [str(package_root)]
    monkeypatch.setitem(sys.modules, package_name, package)
    # Loading the lane must share only its unrelated process/gate imports;
    # the admission module itself can resolve solely from the copied package.
    for name in ("command_processes", "gate_recovery"):
        module = sys.modules[f"{lane_resume.__package__}.{name}"]
        monkeypatch.setitem(sys.modules, f"{package_name}.{name}", module)
    module_name = f"{package_name}.lane_resume"
    spec = importlib.util.spec_from_file_location(module_name, package_root / "lane_resume.py")
    assert spec and spec.loader
    relocated_lane = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, relocated_lane)
    try:
        spec.loader.exec_module(relocated_lane)
        rules = relocated_lane._refill_module(tmp_path / "independent_workspace")
        assert Path(rules.__file__).resolve() == package_root / "board_gap_refill.py"
        seed = (
            "## VOG-012 Census\n- Status: pending\n- Depends on:\n"
            "- Validation: python3 swissknife/scripts/vog_rendered_census_requires_live_dom.py\n"
        )
        result = rules.admit_validation_gap(
            seed,
            task_header_prefix="## VOG-",
            task_id="VOG-012",
            fields={"validation": "python3 swissknife/scripts/vog_rendered_census_requires_live_dom.py"},
            artifact_text=json.dumps({"rows": []}),
        )
        assert result["admitted"] is True
        assert result["derived_task_id"] == "VOG-R012"
        assert result["todo_text"].startswith(seed)
        assert result["work_complete"] is result["completion_authority"] is False
    finally:
        sys.modules.pop(f"{package_name}.board_gap_refill", None)
