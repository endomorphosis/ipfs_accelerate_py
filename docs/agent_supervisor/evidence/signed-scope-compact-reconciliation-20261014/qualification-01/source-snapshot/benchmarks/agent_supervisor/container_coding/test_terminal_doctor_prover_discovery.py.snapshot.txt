"""Explicit local prover selection must not cause an avoidable model fallback."""
from pathlib import Path

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_doctor_dispatch as doctor


@pytest.mark.parametrize("nominated,available,expected", [
    ("/opt/isolated tools/z3", "/usr/bin/z3", "/opt/isolated tools/z3"),
    ("/opt/isolated/z3", None, "/opt/isolated/z3"),
    ("", "/usr/bin/z3", "/usr/bin/z3"),
    (None, None, "/unavailable/z3"),
])
def test_explicit_z3_precedes_path_and_no_tool_is_installed(monkeypatch, nominated, available, expected):
    if nominated is None:
        monkeypatch.delenv("DOCTOR_COMPOSITION_Z3", raising=False)
    else:
        monkeypatch.setenv("DOCTOR_COMPOSITION_Z3", nominated)
    monkeypatch.setenv("DOCTOR_COMPOSITION_LEAN", "/opt/isolated/lean")
    monkeypatch.setattr(doctor.shutil, "which", lambda name: available if name == "z3" else None)
    monkeypatch.setattr(doctor.subprocess, "run", lambda *a, **kw: pytest.fail("discovery must not launch or install nominated tools"))
    assert doctor._installed_provers() == (Path(expected), Path("/opt/isolated/lean"))
