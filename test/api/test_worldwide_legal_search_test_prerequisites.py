"""An optional sibling dependency cannot prevent unrelated native gate collection."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


def _run_fixture(tmp_path: Path, package: str | None, *, nested: bool = True):
    root = tmp_path / "checkout"
    conftest = root / "test/api/conftest.py"
    conftest.parent.mkdir(parents=True)
    conftest.write_bytes(Path(__file__).with_name("conftest.py").read_bytes())
    (root / "ipfs_datasets_py").mkdir()
    poison = tmp_path / "installed/ipfs_datasets_py"
    poison.mkdir(parents=True)
    (poison / "__init__.py").write_text("raise RuntimeError('optional-global-imported')\n")
    if package is not None:
        local = root / "ipfs_datasets_py"
        if nested:
            local /= "ipfs_datasets_py"
        local.mkdir(parents=True, exist_ok=True)
        (local / "__init__.py").write_text(package)
        (local / "local_impl.py").write_text("answer = 42\n")
    script = """
import json, runpy, sys
runpy.run_path(sys.argv[1])
present = 'ipfs_datasets_py' in sys.modules
if present:
    module = sys.modules['ipfs_datasets_py']
    print(json.dumps({'present': True, 'file': module.__file__, 'answer': module.answer}))
else:
    try:
        import ipfs_datasets_py
    except RuntimeError as exc:
        print(json.dumps({'present': False, 'required_import_error': str(exc)}))
    else:
        raise AssertionError('an explicit required dependency import must still execute')
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(poison.parent)
    return subprocess.run([sys.executable, "-B", "-c", script, str(conftest)],
        env=env, capture_output=True, text=True, timeout=30)


def test_empty_gitlink_does_not_import_unrelated_installed_package(tmp_path):
    result = _run_fixture(tmp_path, None)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "present": False, "required_import_error": "optional-global-imported"}


def test_initialized_submodule_pins_its_actual_package(tmp_path):
    result = _run_fixture(tmp_path, "from .local_impl import answer\n")
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    assert observed["answer"] == 42 and observed["present"] is True
    assert observed["file"] == str(tmp_path / "checkout/ipfs_datasets_py/ipfs_datasets_py/__init__.py")


def test_direct_local_package_remains_supported(tmp_path):
    result = _run_fixture(tmp_path, "from .local_impl import answer\n", nested=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["file"] == str(tmp_path / "checkout/ipfs_datasets_py/__init__.py")


def test_local_package_dependency_failures_are_not_silenced(tmp_path):
    result = _run_fixture(tmp_path, "raise ImportError('required-local-capability-missing')\n")
    assert result.returncode != 0
    assert "required-local-capability-missing" in result.stderr
