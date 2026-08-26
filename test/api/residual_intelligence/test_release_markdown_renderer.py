from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.residual_intelligence.release import (
    render_vrif_release_report_markdown,
)


ROOT = Path(__file__).resolve().parents[3]
OPERATOR_PATH = ROOT / "scripts/run_agent_supervisor_residual_intelligence.py"
LEGACY_FIXTURE_SIZE = 803
LEGACY_FIXTURE_SHA256 = "1a3132c871de9b385fd40f0da474eba8d2acd84e8c34de881a0bbf57d04e197e"


def _operator():
    specification = importlib.util.spec_from_file_location(
        "vrif_release_renderer_operator",
        OPERATOR_PATH,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_shared_renderer_preserves_legacy_bytes_and_operator_wrapper() -> None:
    report = {"start_tree": "start", "end_tree": "end", "gaps": {"β": "✓"}}

    rendered = render_vrif_release_report_markdown(report)
    rendered_bytes = rendered.encode("utf-8")

    assert len(rendered_bytes) == LEGACY_FIXTURE_SIZE
    assert hashlib.sha256(rendered_bytes).hexdigest() == LEGACY_FIXTURE_SHA256
    assert _operator()._vrif_release_report_markdown(report) == rendered
