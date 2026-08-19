from __future__ import annotations

import importlib.util
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ARTIFACTS = (
    "dashboard.pid",
    "data/model_manager.duckdb.wal",
    "state/p2p_gpt2_2peer/peer1_queue.duckdb.wal",
    "state/p2p_gpt2_2peer/peer2_queue.duckdb.wal",
    "state/smoketest_logs/driver.out",
    "state/tls/mcpplusplus.crt",
    "state/tls/mcpplusplus.key",
    "test/kitchen_sink_models.db.wal",
)


def test_runtime_and_private_key_shaped_artifacts_are_forward_removed() -> None:
    assert all(not (ROOT / relative).exists() for relative in RUNTIME_ARTIFACTS)

    tracked = subprocess.run(
        ["git", "ls-files", "--", *RUNTIME_ARTIFACTS],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert tracked.returncode == 0, tracked.stderr
    assert tracked.stdout == ""


def test_forward_removed_runtime_paths_remain_ignored() -> None:
    ignored = subprocess.run(
        ["git", "check-ignore", "--no-index", "--", *RUNTIME_ARTIFACTS],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert ignored.returncode == 0, ignored.stderr
    assert set(ignored.stdout.splitlines()) == set(RUNTIME_ARTIFACTS)


def test_generated_replacement_private_key_is_published_mode_0600(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    script_path = ROOT / "scripts/systemd/generate_self_signed_cert.py"
    spec = importlib.util.spec_from_file_location("eaaef_tls_generator", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr(module.shutil, "which", lambda _name: "/usr/bin/openssl")
    monkeypatch.setattr(module, "_detect_lan_ip", lambda: "")
    monkeypatch.setattr(module, "_hostname_sans", lambda: ["DNS:localhost"])

    def fake_openssl(argv: list[str]) -> None:
        key_path = Path(argv[argv.index("-keyout") + 1])
        cert_path = Path(argv[argv.index("-out") + 1])
        key_path.write_text("test-only-key", encoding="utf-8")
        cert_path.write_text("test-only-certificate", encoding="utf-8")

    monkeypatch.setattr(module, "_run", fake_openssl)
    key_path = tmp_path / "state/tls/generated.key"
    cert_path = tmp_path / "state/tls/generated.crt"
    assert (
        module.main(
            [
                "--keyfile",
                str(key_path),
                "--certfile",
                str(cert_path),
            ]
        )
        == 0
    )
    assert stat.S_IMODE(key_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(cert_path.stat().st_mode) == 0o644
