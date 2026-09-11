"""Exercise the exact isolated launcher without enabling user-site imports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from scripts.ops.agent_supervisor import spar_capture_runtime as runtime

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def manifest(tmp_path):
    path = tmp_path / "reviewed-runtime.json"
    path.write_text(json.dumps(runtime.observed_runtime(), sort_keys=True) + "\n")
    return path


def _invoke(tmp_path, manifest=None, digest=None):
    native = tmp_path / "native"
    output = tmp_path / "output"
    arguments = [
        "/usr/bin/python3", "-I", str(ROOT / "scripts/run_spar_retained_capture.py"),
        "--repository-root", str(native), "--config-path", str(native / "config.json"),
        "--fleet-config", str(tmp_path / "fleet.json"), "--operation-root", str(output),
        "--expected-source-commit", "1" * 40, "--expected-source-tree", "2" * 40,
    ]
    if manifest is not None:
        arguments += ["--runtime-manifest", str(manifest), "--runtime-manifest-sha256", digest]
    result = subprocess.run(arguments, capture_output=True, text=True, timeout=20)
    assert result.returncode == 1
    assert not output.exists() and not native.exists()
    return json.loads(result.stdout)


def test_missing_duckdb_fails_before_native_inspection_or_output(tmp_path):
    report = _invoke(tmp_path)
    assert report["stage"] == "new"
    assert report["diagnostic"]["missing_module"] == "duckdb"
    assert report["diagnostic"]["traceback"]


def test_reviewed_runtime_allows_isolated_probe_before_missing_native_source(tmp_path, manifest):
    report = _invoke(tmp_path, manifest, hashlib.sha256(manifest.read_bytes()).hexdigest())
    assert report["stage"] == "new"
    assert report["diagnostic"]["missing_module"] is None
    assert report["diagnostic"]["exception"] == "SparMergeOwnerError"
    assert any(frame["function"] == "_native_operator" for frame in report["diagnostic"]["traceback"])


@pytest.mark.parametrize("change", ["manifest-digest", "dependency-digest", "interpreter", "escape"])
def test_runtime_bindings_fail_before_native_inspection(tmp_path, manifest, change):
    value = json.loads(manifest.read_text())
    if change == "dependency-digest":
        value["files"]["duckdb/__init__.py"] = "0" * 64
    elif change == "interpreter":
        value["interpreter"] = "/other/python3"
    elif change == "escape":
        value["files"]["duckdb/../../unreviewed.py"] = "0" * 64
    manifest.write_text(json.dumps(value))
    digest = "0" * 64 if change == "manifest-digest" else hashlib.sha256(manifest.read_bytes()).hexdigest()
    report = _invoke(tmp_path, manifest, digest)
    assert report["stage"] == "new"
    assert report["diagnostic"]["exception"] == "CaptureRuntimeDenied"
    assert not any(frame["function"] == "_native_operator" for frame in report["diagnostic"]["traceback"])


def test_explicit_runtime_keeps_user_site_and_hooks_disabled(tmp_path, manifest):
    script = tmp_path / "probe.py"
    script.write_text("""import sys,hashlib,json,fcntl
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from scripts.ops.agent_supervisor.spar_capture_runtime import admit_runtime
p=Path(sys.argv[2]);r=admit_runtime(str(p),hashlib.sha256(p.read_bytes()).hexdigest())
assert sys.flags.isolated == 1
assert not any('/.local/' in entry and 'site-packages' in entry for entry in sys.path)
assert r.extension_fd is not None
assert fcntl.fcntl(r.extension_fd,fcntl.F_GET_SEALS) & fcntl.F_SEAL_WRITE
assert admit_runtime() is r
print(json.dumps({'duckdb_version':r.manifest['duckdb_version'],'isolated':True,
 'user_site_activated':False,'extension_loaded_from_sealed_memfd':True}))
""")
    result = subprocess.run(["/usr/bin/python3", "-I", str(script), str(ROOT), str(manifest)],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["extension_loaded_from_sealed_memfd"] is True
