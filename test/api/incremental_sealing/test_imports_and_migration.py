"""IPS-044: hermetic bootstrap and truthful legacy proof-receipt migration."""

from __future__ import annotations

import importlib
import json
import os
import socket
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.bootstrap import (
    BOOTSTRAP_INTERFACE,
    BOOTSTRAP_SCHEMA,
    IMPORT_HERMETICITY_EVIDENCE,
    BootstrapDependencies,
    BootstrapError,
    IncrementalSealingBootstrap,
    assert_import_is_hermetic,
    bootstrap_dependencies,
    hermetic_bootstrap,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.migration import (
    MIGRATION_EVIDENCE,
    MIGRATION_SCHEMA,
    LegacyEvidenceMigrationResult,
    MigrationDisposition,
    MigrationError,
    closed_migration_dispositions,
    migrate_legacy_evidence,
    migrate_legacy_evidence_batch,
    migration_contract,
)
from ipfs_datasets_py.logic.zkp.incremental_sealing.evidence import (
    IntegrityCommitment,
    ProofMode,
    ProofTerminalStatus,
    SignedExecutionReceipt,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
_DIGEST = "sha256:" + ("ab" * 32)
_DIGEST_B = "sha256:" + ("cd" * 32)
_DIGEST_C = "sha256:" + ("ef" * 32)


# ---------------------------------------------------------------------------
# Evidence / contract surface
# ---------------------------------------------------------------------------


def test_evidence_subsets_and_closed_dispositions() -> None:
    assert IMPORT_HERMETICITY_EVIDENCE == "ips/import-hermeticity@1"
    assert MIGRATION_EVIDENCE == "ips/cross-repository-migration@1"
    assert BOOTSTRAP_SCHEMA.endswith("hermetic-bootstrap@1")
    assert MIGRATION_SCHEMA.endswith("legacy-evidence-migration@1")
    assert BOOTSTRAP_INTERFACE == "IncrementalSealingBootstrap@1"
    assert closed_migration_dispositions() == {
        "accept",
        "adapt",
        "reverify",
        "reject",
    }
    contract = migration_contract()
    assert contract["cache_admission_gate"] == "current_policy_verification_required"
    assert contract["assurance_upgrade_forbidden"] is True
    assert contract["staging_is_not_admission"] is True
    assert contract["simulated_never_cache_admitted"] is True


def test_assert_import_is_hermetic_contract() -> None:
    report = assert_import_is_hermetic()
    assert report["evidence"] == IMPORT_HERMETICITY_EVIDENCE
    assert report["ordinary_import_side_effects"] == []
    forbidden = set(report["forbidden_on_import"])
    for item in (
        "file_create",
        "key_generate",
        "subprocess",
        "package_install",
        "network_access",
        "daemon_dependency",
    ):
        assert item in forbidden


# ---------------------------------------------------------------------------
# Hermetic cold import
# ---------------------------------------------------------------------------


def test_ordinary_package_import_is_hermetic(tmp_path: Path) -> None:
    """Cold import creates no files, keys, subprocesses, installs, or network."""

    home = tmp_path / "home"
    home.mkdir()
    ipfs_path = tmp_path / "ipfs-path"
    env = dict(os.environ)
    env["HOME"] = str(home)
    env["IPFS_PATH"] = str(ipfs_path)
    env["XDG_CACHE_HOME"] = str(home / ".cache")
    env["XDG_CONFIG_HOME"] = str(home / ".config")
    env["XDG_DATA_HOME"] = str(home / ".local" / "share")
    env["XDG_STATE_HOME"] = str(home / ".local" / "state")
    env["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (
            str(REPO_ROOT),
            str(REPO_ROOT / "ipfs_datasets_py"),
            str(REPO_ROOT / "ipfs_kit_py"),
            env.get("PYTHONPATH", ""),
        )
        if part
    )
    probe = tmp_path / "cold_import_probe.py"
    probe.write_text(
        f"""
import os
import socket
import subprocess
import sys
from pathlib import Path

home = Path(os.environ["HOME"])
ipfs_path = Path(os.environ["IPFS_PATH"])
before = {{p.name for p in home.iterdir()}} if home.exists() else set()

original_socket = socket.socket

class NoNetwork(original_socket):
    def connect(self, *a, **k):
        raise AssertionError("network connect during import")
    def connect_ex(self, *a, **k):
        raise AssertionError("network connect_ex during import")

socket.socket = NoNetwork

orig_popen = subprocess.Popen
def no_popen(*a, **k):
    raise AssertionError("subprocess during import")
subprocess.Popen = no_popen

import importlib
boot = importlib.import_module(
    "ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.bootstrap"
)
mig = importlib.import_module(
    "ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.migration"
)
assert boot.IMPORT_HERMETICITY_EVIDENCE == "ips/import-hermeticity@1"
assert mig.MIGRATION_EVIDENCE == "ips/cross-repository-migration@1"
# Construction without store binding stays pure.
bs = boot.hermetic_bootstrap()
assert bs.bound is False
report = bs.report()
assert report.import_side_effects == ()
assert report.bound is False
# Classification without admission is pure.
result = mig.migrate_legacy_evidence(
    {{"digest": {_DIGEST!r}, "cid": {_DIGEST_B!r}}}
)
assert result.cache_admitted is False
assert result.requires_current_policy_verification is True

after = {{p.name for p in home.iterdir()}} if home.exists() else set()
assert after == before, (before, after)
assert not ipfs_path.exists(), "IPFS_PATH must not be created"
assert "provekit" not in sys.modules
print("hermetic-import-ok")
""",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, str(probe)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert "hermetic-import-ok" in completed.stdout


def test_bootstrap_rejects_home_and_xdg_store_defaults() -> None:
    with pytest.raises(BootstrapError, match="explicit absolute path"):
        BootstrapDependencies(store_root="~/.cache/ips")
    with pytest.raises(BootstrapError, match="explicit absolute path"):
        bootstrap_dependencies(store_root="$HOME/proof-store")
    with pytest.raises(BootstrapError, match="explicit absolute path"):
        hermetic_bootstrap(store_root="$XDG_STATE_HOME/seal")


def test_bootstrap_unbound_cannot_build_sealer() -> None:
    boot = hermetic_bootstrap()
    assert boot.bound is False
    with pytest.raises(BootstrapError, match="explicit store_root"):
        boot.build_sealer()
    report = boot.report()
    assert report.bound is False
    assert report.evidence == IMPORT_HERMETICITY_EVIDENCE
    context = boot.build_migration_context()
    assert context["cache_admission_requires_current_policy_verification"] is True


def test_bootstrap_explicit_injection_builds_sealer(tmp_path: Path) -> None:
    root = tmp_path / "seal-root"
    root.mkdir()
    boot = hermetic_bootstrap(
        store_root=root,
        create_store=True,
        branch_id="main",
    )
    assert boot.bound is True
    sealer = boot.build_sealer()
    assert sealer is not None
    assert sealer.root_path == root
    verifier = boot.build_evidence_verifier()
    assert verifier is not None
    # Second build returns the cached instance.
    assert boot.build_sealer() is sealer


def test_bootstrap_bind_store_root_is_pure_until_build(tmp_path: Path) -> None:
    target = tmp_path / "not-created-yet"
    boot = hermetic_bootstrap().bind_store_root(target, create_store=False)
    assert boot.bound is True
    assert not target.exists()
    report = boot.report()
    assert report.store_root == str(target)
    assert report.create_store is False


# ---------------------------------------------------------------------------
# Truthful migration
# ---------------------------------------------------------------------------


def _integrity_payload() -> dict:
    return {
        "evidence_class": "IntegrityCommitment",
        "digest": _DIGEST,
        "cid": _DIGEST_B,
        "merkle_inclusion": "leaf:0",
        "byte_length": 32,
    }


def _signed_payload() -> dict:
    return {
        "evidence_class": "SignedExecutionReceipt",
        "signer_id": "allowlist/operator-1",
        "receipt_digest": _DIGEST,
        "signature": "ed25519:sig-valid",
        "statement": "pytest node completed",
    }


def test_null_payload_is_rejected() -> None:
    result = migrate_legacy_evidence(None)
    assert isinstance(result, LegacyEvidenceMigrationResult)
    assert result.disposition is MigrationDisposition.REJECT
    assert result.cache_admitted is False
    assert result.cache_eligible is False
    assert result.requires_current_policy_verification is True


def test_simulated_evidence_never_enters_cache(tmp_path: Path) -> None:
    result = migrate_legacy_evidence(
        {
            "backend": "simulated",
            "proof_system": "mock-demo",
            "passed": True,
        },
        admit_to_cache=True,
        stage_root=tmp_path / "stage",
    )
    assert result.disposition is MigrationDisposition.REJECT
    assert result.assurance == "simulated"
    assert result.cache_admitted is False
    assert result.cache_eligible is False
    assert result.production_seal_allowed is False


def test_canonical_integrity_accepts_without_auto_cache() -> None:
    result = migrate_legacy_evidence(_integrity_payload())
    assert result.disposition is MigrationDisposition.ACCEPT
    assert result.assurance == "integrity_only"
    assert result.cache_admitted is False
    assert result.cache_eligible is False
    assert result.requires_current_policy_verification is True
    assert result.datasets_disposition == "accept"


def test_legacy_digest_adapts_without_cache_admission() -> None:
    result = migrate_legacy_evidence({"digest": _DIGEST, "cid": _DIGEST_B})
    assert result.disposition in {
        MigrationDisposition.ADAPT,
        MigrationDisposition.ACCEPT,
    }
    assert result.cache_admitted is False
    assert "never enters reusable cache" in " ".join(result.reasons)


def test_legacy_cache_candidate_requires_reverify() -> None:
    result = migrate_legacy_evidence(
        {
            "digest": _DIGEST,
            "cid": _DIGEST_B,
            "cache_key": "proof-cache-key:legacy-1",
            "role": "candidate",
            "cache_status": "unverified",
        }
    )
    assert result.disposition is MigrationDisposition.REVERIFY
    assert result.cache_admitted is False
    assert result.admission_reason_code == "reverify_required"
    assert result.requires_current_policy_verification is True


def test_admit_to_cache_requires_current_policy_verification() -> None:
    result = migrate_legacy_evidence(
        _integrity_payload(),
        admit_to_cache=True,
        proof_unit_id="unit/integrity-1",
        public_input_cid=_DIGEST,
        proof_object_cid=_DIGEST_C,
        required_for_seal=True,
    )
    # Integrity under production required-for-seal may admit when verification
    # succeeds; either way classification alone is never the authority.
    if result.cache_admitted:
        assert result.cache_eligible is True
        assert result.cache_admission_record is not None
        assert result.cache_admission_record["verified"] is True
        assert result.verification_digest
        assert result.disposition in {
            MigrationDisposition.ACCEPT,
            MigrationDisposition.ADAPT,
        }
    else:
        assert result.cache_eligible is False
        assert result.disposition is MigrationDisposition.REVERIFY
        assert result.admission_reason_code is not None


def test_staging_is_not_admission(tmp_path: Path) -> None:
    stage = tmp_path / "legacy-stage"
    stage.mkdir()
    result = migrate_legacy_evidence(
        {"digest": _DIGEST, "cid": _DIGEST_B, "note": "legacy"},
        stage_root=stage,
        admit_to_cache=False,
    )
    assert result.cache_admitted is False
    if result.staged_cid is not None:
        assert result.staged_only is True
        assert "staging is not cache admission" in " ".join(result.reasons)


def test_signed_receipt_preserves_assurance() -> None:
    result = migrate_legacy_evidence(_signed_payload())
    assert result.assurance == "signed_receipt"
    assert result.disposition is MigrationDisposition.ACCEPT
    assert result.cache_admitted is False
    assert result.production_seal_allowed is True


def test_batch_migration_preserves_per_item_truth() -> None:
    results = migrate_legacy_evidence_batch(
        [
            None,
            {"backend": "simulated", "mode": "demo"},
            _integrity_payload(),
        ]
    )
    assert len(results) == 3
    assert results[0].disposition is MigrationDisposition.REJECT
    assert results[1].disposition is MigrationDisposition.REJECT
    assert results[1].assurance == "simulated"
    assert results[2].disposition is MigrationDisposition.ACCEPT
    assert all(item.cache_admitted is False for item in results)


def test_migration_result_canonical_round_trip() -> None:
    result = migrate_legacy_evidence(_integrity_payload())
    payload = result.to_canonical()
    assert payload["schema"] == MIGRATION_SCHEMA
    assert payload["evidence"] == MIGRATION_EVIDENCE
    assert payload["cache_admitted"] is False
    text = result.to_canonical_json()
    assert json.loads(text) == payload


def test_migration_result_rejects_simulated_cache_admission() -> None:
    with pytest.raises(MigrationError, match="cannot be cache-admitted"):
        LegacyEvidenceMigrationResult(
            disposition=MigrationDisposition.REJECT,
            path_family="simulated_zkp_proof",
            assurance="simulated",
            proof_mode=ProofMode.SIMULATED.value,
            target_evidence_class="n/a",
            establishes="nothing",
            does_not_establish="any production claim",
            production_seal_allowed=False,
            reasons=("simulated",),
            cache_eligible=True,
            cache_admitted=True,
        )


# ---------------------------------------------------------------------------
# Cross-repository pytest entry-point reconciliation
# ---------------------------------------------------------------------------


def _load_pyproject(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def test_accelerate_declares_pytest11_proof_reuse_entry_point() -> None:
    project = _load_pyproject(REPO_ROOT / "pyproject.toml")
    entry = project["project"]["entry-points"]["pytest11"]
    assert entry["ipfs-proof-reuse"] == (
        "ipfs_accelerate_py.testing.proof_reuse.plugin"
    )


def test_kit_declares_distinct_pytest11_proof_reuse_entry_point() -> None:
    project = _load_pyproject(REPO_ROOT / "ipfs_kit_py" / "pyproject.toml")
    entry = project["project"]["entry-points"]["pytest11"]
    assert entry["ipfs-kit-proof-reuse"] == "ipfs_kit_py.pytest_proof_reuse"
    assert "ipfs-proof-reuse" not in entry or entry.get("ipfs-proof-reuse") != (
        "ipfs_accelerate_py.testing.proof_reuse.plugin"
    )


def test_datasets_declares_distinct_pytest11_proof_reuse_entry_point() -> None:
    project = _load_pyproject(REPO_ROOT / "ipfs_datasets_py" / "pyproject.toml")
    entry = project["project"]["entry-points"]["pytest11"]
    assert entry["ipfs-datasets-proof-reuse"] == (
        "ipfs_datasets_py.pytest_proof_reuse"
    )


def test_entry_point_names_are_cross_repository_distinct() -> None:
    accelerate = _load_pyproject(REPO_ROOT / "pyproject.toml")
    kit = _load_pyproject(REPO_ROOT / "ipfs_kit_py" / "pyproject.toml")
    datasets = _load_pyproject(REPO_ROOT / "ipfs_datasets_py" / "pyproject.toml")
    names = {
        next(iter(accelerate["project"]["entry-points"]["pytest11"])),
        next(iter(kit["project"]["entry-points"]["pytest11"])),
        next(iter(datasets["project"]["entry-points"]["pytest11"])),
    }
    # Each repository owns a distinct pytest11 plugin name after drift fix.
    assert "ipfs-proof-reuse" in names
    assert "ipfs-kit-proof-reuse" in names
    assert "ipfs-datasets-proof-reuse" in names
    assert len(names) == 3


def test_public_interfaces_are_importable() -> None:
    boot = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.bootstrap"
    )
    mig = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.migration"
    )
    assert boot.IncrementalSealingBootstrap is IncrementalSealingBootstrap
    assert mig.LegacyEvidenceMigrationResult is LegacyEvidenceMigrationResult
    assert mig.migrate_legacy_evidence is migrate_legacy_evidence
    assert callable(boot.hermetic_bootstrap)
    assert callable(mig.migrate_legacy_evidence)


def test_integrity_and_signed_constructors_still_honest() -> None:
    # Guard against accidental assurance collapse in migration helpers.
    integrity = IntegrityCommitment(
        digest=_DIGEST,
        cid=_DIGEST_B,
        merkle_inclusion="leaf:0",
        byte_length=32,
    )
    signed = SignedExecutionReceipt(
        signer_id="allowlist/operator-1",
        receipt_digest=_DIGEST,
        signature="ed25519:sig-valid",
        statement="pytest node completed",
    )
    assert integrity.to_canonical()["evidence_class"] == "IntegrityCommitment"
    assert signed.to_canonical()["evidence_class"] == "SignedExecutionReceipt"
    assert ProofTerminalStatus.INTEGRITY_VERIFIED.value == "integrity_verified"
