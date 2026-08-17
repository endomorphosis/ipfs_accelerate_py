"""IPS-043: zk-seal CLI surface."""

from __future__ import annotations

import json
import socket
from io import StringIO
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.cli import (
    CLI_EVIDENCE,
    COMMANDS,
    build_parser,
    main,
)


def test_cli_evidence_and_nine_commands() -> None:
    assert CLI_EVIDENCE == "ips/cli@1"
    assert COMMANDS == (
        "full",
        "incremental",
        "verify",
        "plan",
        "explain-reuse",
        "explain-invalidation",
        "benchmark",
        "cache-status",
        "force-full",
    )
    parser = build_parser()
    help_text = parser.format_help()
    for name in COMMANDS:
        assert name in help_text


def test_help_and_import_have_no_network_or_state_side_effects(tmp_path: Path) -> None:
    opened: list[str] = []
    original = socket.socket

    class GuardedSocket(original):  # type: ignore[misc,valid-type]
        def __init__(self, *args, **kwargs):
            opened.append("socket")
            raise AssertionError("CLI help must not open sockets")

    socket.socket = GuardedSocket  # type: ignore[misc]
    try:
        with pytest.raises(SystemExit) as caught:
            main(["--help"])
        assert caught.value.code == 0
    finally:
        socket.socket = original
    assert opened == []
    assert list(tmp_path.iterdir()) == []


def test_cache_status_is_typed_unavailable() -> None:
    buf = StringIO()
    from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing import cli as cli_mod

    original = cli_mod._emit

    captured: dict[str, object] = {}

    def fake_emit(payload, *, stream=None):
        captured.update(payload)
        return original(payload, stream=buf)

    cli_mod._emit = fake_emit  # type: ignore[method-assign]
    try:
        code = main(["cache-status"])
    finally:
        cli_mod._emit = original  # type: ignore[method-assign]
    assert code == 2
    assert captured["status"] == "unavailable"
    assert captured["ok"] is False
    assert captured["details"]["typed"] is True
    assert captured["side_effects"]["network"] is False


def test_force_full_is_typed_reproof(tmp_path: Path) -> None:
    state = {
        "identity_cid": "sha256:" + ("bb" * 32),
        "repository_state_cid": "sha256:" + ("bb" * 32),
    }
    path = tmp_path / "state.json"
    path.write_text(json.dumps(state), encoding="utf-8")
    buf = StringIO()
    from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing import cli as cli_mod

    captured: dict[str, object] = {}
    original = cli_mod._emit

    def fake_emit(payload, *, stream=None):
        captured.update(payload)
        return original(payload, stream=buf)

    cli_mod._emit = fake_emit  # type: ignore[method-assign]
    try:
        code = main(["force-full", "--old", str(path), "--new", str(path)])
    finally:
        cli_mod._emit = original  # type: ignore[method-assign]
    assert captured["command"] == "force-full"
    assert captured["status"] == "full_reproof_required"
    assert code == 2


def test_production_full_rejects_simulated_payload(tmp_path: Path) -> None:
    path = tmp_path / "sim.json"
    path.write_text(json.dumps({"simulated": True, "repository_state_cid": "x"}), encoding="utf-8")
    buf = StringIO()
    from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing import cli as cli_mod

    captured: dict[str, object] = {}
    original = cli_mod._emit

    def fake_emit(payload, *, stream=None):
        captured.update(payload)
        return original(payload, stream=buf)

    cli_mod._emit = fake_emit  # type: ignore[method-assign]
    try:
        code = main(["full", "--state", str(path)])
    finally:
        cli_mod._emit = original  # type: ignore[method-assign]
    assert captured["status"] == "simulated_only"
    assert code == 2


def test_benchmark_missing_records_are_typed() -> None:
    buf = StringIO()
    from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing import cli as cli_mod

    captured: dict[str, object] = {}
    original = cli_mod._emit

    def fake_emit(payload, *, stream=None):
        captured.update(payload)
        return original(payload, stream=buf)

    cli_mod._emit = fake_emit  # type: ignore[method-assign]
    try:
        code = main(["benchmark"])
    finally:
        cli_mod._emit = original  # type: ignore[method-assign]
    assert captured["status"] == "unavailable"
    assert code == 2
