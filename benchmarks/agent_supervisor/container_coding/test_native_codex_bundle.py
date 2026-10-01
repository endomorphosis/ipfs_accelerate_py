"""The root installer refuses incomplete CLI tool runtimes before model calls."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from benchmarks.agent_supervisor.container_coding.terminal_deployment import native_codex_exposure_script


def test_native_bundle_installer_requires_actual_root(tmp_path):
    if os.geteuid() == 0:
        pytest.skip("the other cases exercise the actual root installer")
    result = subprocess.run([sys.executable, "-I", "-c", native_codex_exposure_script(str(tmp_path))],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "requires container root" in result.stderr
    assert not (tmp_path / "provider-bin").exists()


@pytest.fixture
def bundle(tmp_path):
    if os.geteuid() != 0:
        pytest.skip("native bundle deployment must be tested as container root")
    vendor = tmp_path / "home/.nvm/versions/node/pinned/lib/node_modules/@openai/codex/node_modules/@openai/codex-test/vendor/test/bin"
    vendor.mkdir(parents=True)
    for name, output in (("codex", "codex-cli 0.158.0"), ("codex-code-mode-host", "Usage: codex-code-mode-host"),
                         ("another-native-helper", "helper")):
        p = vendor / name
        p.write_text(f"#!{sys.executable}\nprint({output!r})\n")
        p.chmod(0o755)
    return tmp_path, vendor


def test_complete_vendor_runtime_is_copied_protected_and_executed(bundle):
    root, vendor = bundle
    result = subprocess.run([sys.executable, "-I", "-c", native_codex_exposure_script(str(root))],
                            capture_output=True, text=True, check=True)
    receipt = json.loads(result.stdout)
    assert receipt["provider_calls"] == 0
    assert {f["name"] for f in receipt["files"]} == {p.name for p in vendor.iterdir()}
    assert all(f["uid"] == 0 and f["mode"] == 0o755 for f in receipt["files"])
    assert all(f["sha256"] == f["source_sha256"] for f in receipt["files"])
    assert set(receipt["executable_checks"]) == {"codex", "codex-code-mode-host"}
    assert all(x["returncode"] == 0 for x in receipt["executable_checks"].values())


@pytest.mark.parametrize("mutation", ["missing_host", "symlink_host", "nonexecuting_host", "wrong_version", "host_cannot_start"])
def test_incomplete_or_nonrunning_native_runtime_refuses_deployment(bundle, mutation):
    root, vendor = bundle
    host = vendor / "codex-code-mode-host"
    if mutation == "missing_host":
        host.unlink()
    elif mutation == "symlink_host":
        host.unlink()
        host.symlink_to(vendor / "codex")
    elif mutation == "nonexecuting_host":
        host.chmod(0o644)
    elif mutation == "wrong_version":
        (vendor / "codex").write_text(f"#!{sys.executable}\nprint('codex-cli unpinned')\n")
    else:
        host.write_text(f"#!{sys.executable}\nraise SystemExit(3)\n")
    result = subprocess.run([sys.executable, "-I", "-c", native_codex_exposure_script(str(root))],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "ValueError" in result.stderr
    assert not result.stdout.strip()
