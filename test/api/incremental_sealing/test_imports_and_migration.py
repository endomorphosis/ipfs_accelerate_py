"""IPS-044: hermetic bootstrap and truthful legacy migration."""

from __future__ import annotations

import socket
from pathlib import Path

import tomllib

from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.bootstrap import (
    BOOTSTRAP_EVIDENCE,
    IncrementalSealingBootstrap,
    bind_bootstrap,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.migration import (
    MIGRATION_EVIDENCE,
    CacheAdmission,
    migrate_legacy_evidence,
)


def test_evidence_subsets() -> None:
    assert BOOTSTRAP_EVIDENCE == "ips/import-hermeticity@1"
    assert MIGRATION_EVIDENCE == "ips/cross-repository-migration@1"


def test_import_and_bind_have_no_process_or_network_side_effects() -> None:
    opened: list[str] = []
    original = socket.socket

    class Guarded(original):  # type: ignore[misc,valid-type]
        def __init__(self, *args, **kwargs):
            opened.append("socket")
            raise AssertionError("bootstrap must not open sockets")

    socket.socket = Guarded  # type: ignore[misc]
    try:
        bootstrap = bind_bootstrap()
        assert isinstance(bootstrap, IncrementalSealingBootstrap)
        assert bootstrap.has_datasets_adapter() is False
        assert bootstrap.to_canonical()["import_side_effects"] is False
    finally:
        socket.socket = original
    assert opened == []


def test_simulated_legacy_evidence_never_enters_cache() -> None:
    result = migrate_legacy_evidence({"kind": "legacy", "simulated": True})
    assert result.reusable is False
    assert result.simulated is True
    assert result.cache_admission is CacheAdmission.REJECTED
    assert result.disposition == "reject"


def test_legacy_integrity_requires_reverify_before_cache() -> None:
    result = migrate_legacy_evidence({"digest": "sha256:" + ("ab" * 32)})
    assert result.reusable is False
    assert result.cache_admission is CacheAdmission.REQUIRES_REVERIFY


def test_current_policy_admission_required_for_reuse() -> None:
    rejected = migrate_legacy_evidence(
        {"digest": "sha256:" + ("ab" * 32)},
        admit=lambda payload: False,
    )
    assert rejected.reusable is False
    assert rejected.cache_admission is CacheAdmission.REJECTED

    admitted = migrate_legacy_evidence(
        {"digest": "sha256:" + ("ab" * 32)},
        admit=lambda payload: True,
    )
    assert admitted.reusable is True
    assert admitted.cache_admission is CacheAdmission.ADMITTED
    assert admitted.simulated is False


def test_accelerate_pyproject_declares_pytest11_proof_reuse() -> None:
    root = Path(__file__).resolve().parents[3]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    entry = project["project"]["entry-points"]["pytest11"]
    assert entry["ipfs-proof-reuse"] == "ipfs_accelerate_py.testing.proof_reuse.plugin"
