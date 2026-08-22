"""EAAEF-162: supported profiles exist; remote multi-host stays disabled."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "deploy" / "external-agent"
FILES = (
    "local-supervised.yaml",
    "detached-single-host.yaml",
    "multi-container-single-host.yaml",
)


def test_profiles_disable_remote_multi_host_and_docker_socket() -> None:
    for name in FILES:
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "remote_multi_host: disabled" in text
        assert "docker_socket: false" in text
