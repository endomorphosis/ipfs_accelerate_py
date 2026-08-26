from __future__ import annotations

import importlib.util
import json
import stat
import subprocess
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
RECONCILIATION_RELATIVE = (
    "docs/architecture/external_agent_autonomous_execution_fabric/"
    "reconciliation/accelerator.json"
)
RECONCILIATION_PATH = ROOT / RECONCILIATION_RELATIVE
SOURCE_MANIFEST_RELATIVE = (
    "docs/architecture/external_agent_autonomous_execution_fabric/"
    "source_reconciliation_manifest.json"
)
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


def _git(*argv: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *argv],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=check,
    )


def _load_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _tree_blob(revision: str, path: str) -> tuple[str, str, int]:
    result = _git("ls-tree", "-l", revision, "--", path)
    lines = result.stdout.rstrip("\n").splitlines()
    assert len(lines) == 1, (revision, path, result.stdout)
    metadata, actual_path = lines[0].split("\t", 1)
    mode, kind, blob, size = metadata.split()
    assert actual_path == path
    assert kind == "blob"
    return mode, blob, int(size)


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


def test_reconciliation_tombstone_is_closed_evidence_only_output() -> None:
    raw = RECONCILIATION_PATH.read_text(encoding="utf-8")
    reconciliation = _load_mapping(RECONCILIATION_PATH)

    assert set(reconciliation) == {
        "schema",
        "task_id",
        "artifact_kind",
        "authority",
        "declared_output_resolution",
        "reviewed_baseline",
        "residual_lineages",
        "forward_removal",
        "replacement_tls",
    }
    assert reconciliation["schema"] == (
        "ExternalAgentAcceleratorReconciliationTombstone@1"
    )
    assert reconciliation["task_id"] == "EAAEF-002"
    assert reconciliation["artifact_kind"] == (
        "declared_output_forward_removal_tombstone"
    )
    assert reconciliation["authority"] == {
        "evidence_only": True,
        "authorizes_history_rewrite": False,
        "authorizes_runtime_materialization": False,
        "authorizes_secret_reuse": False,
        "authorizes_wholesale_lineage_merge": False,
    }

    resolution = reconciliation["declared_output_resolution"]
    assert set(resolution) == {
        "materialized_evidence_output",
        "tombstoned_outputs",
        "resolution",
    }
    assert resolution["materialized_evidence_output"] == RECONCILIATION_RELATIVE
    assert tuple(resolution["tombstoned_outputs"]) == RUNTIME_ARTIFACTS
    assert resolution["resolution"] == (
        "The tombstoned outputs are intentionally absent, untracked and ignored. "
        "This evidence artifact must never be copied or linked to any tombstoned "
        "path."
    )

    unignored = _git(
        "check-ignore",
        "--no-index",
        "--",
        RECONCILIATION_RELATIVE,
        check=False,
    )
    assert unignored.returncode == 1, unignored.stdout
    assert "BEGIN PRIVATE KEY" not in raw
    assert "BEGIN RSA PRIVATE KEY" not in raw


def test_reconciliation_tombstone_binds_forward_removal_history() -> None:
    reconciliation = _load_mapping(RECONCILIATION_PATH)
    removal = reconciliation["forward_removal"]
    assert set(removal) == {
        "commit",
        "tree",
        "parent_commit",
        "parent_tree",
        "required_checkout_state",
        "history_policy",
        "supporting_source_blobs_at_removal",
        "artifacts",
    }
    assert removal["required_checkout_state"] == [
        "absent",
        "untracked",
        "ignored",
    ]
    assert removal["history_policy"] == (
        "metadata_provenance_only_never_checkout_materialization"
    )

    commit = removal["commit"]
    parent_commit = removal["parent_commit"]
    assert _git("show", "-s", "--format=%T", commit).stdout.strip() == removal["tree"]
    assert _git("rev-parse", f"{commit}^").stdout.strip() == parent_commit
    assert (
        _git("show", "-s", "--format=%T", parent_commit).stdout.strip()
        == removal["parent_tree"]
    )
    assert _git("merge-base", "--is-ancestor", commit, "HEAD").returncode == 0

    recorded_artifacts = removal["artifacts"]
    assert tuple(item["path"] for item in recorded_artifacts) == RUNTIME_ARTIFACTS
    expected_classifications = {
        "dashboard.pid": "runtime_pid",
        "data/model_manager.duckdb.wal": "database_write_ahead_log",
        "state/p2p_gpt2_2peer/peer1_queue.duckdb.wal": ("database_write_ahead_log"),
        "state/p2p_gpt2_2peer/peer2_queue.duckdb.wal": ("database_write_ahead_log"),
        "state/smoketest_logs/driver.out": "runtime_log",
        "state/tls/mcpplusplus.crt": "certificate_shaped_runtime_artifact",
        "state/tls/mcpplusplus.key": "private_key_shaped_secret",
        "test/kitchen_sink_models.db.wal": "database_write_ahead_log",
    }
    for item in recorded_artifacts:
        assert set(item) == {
            "path",
            "classification",
            "historical_mode",
            "historical_blob",
            "historical_size_bytes",
        }
        assert item["classification"] == expected_classifications[item["path"]]
        mode, blob, size = _tree_blob(parent_commit, item["path"])
        assert (mode, blob, size) == (
            item["historical_mode"],
            item["historical_blob"],
            item["historical_size_bytes"],
        )

    removed_paths = _git(
        "ls-tree", "-r", "--name-only", commit, "--", *RUNTIME_ARTIFACTS
    )
    assert removed_paths.stdout == ""

    supporting_blobs = removal["supporting_source_blobs_at_removal"]
    assert set(supporting_blobs) == {
        ".gitignore",
        "scripts/systemd/generate_self_signed_cert.py",
        "test/api/test_external_agent_source_hygiene.py",
    }
    for path, expected_blob in supporting_blobs.items():
        _mode, blob, _size = _tree_blob(commit, path)
        assert blob == expected_blob


def test_reconciliation_tombstone_binds_reviewed_residual_dispositions() -> None:
    reconciliation = _load_mapping(RECONCILIATION_PATH)
    reviewed = reconciliation["reviewed_baseline"]
    assert reviewed["repository_manifest_pointer"] == (
        "/repositories/ipfs_accelerate_py"
    )
    for evidence in (reviewed["machine_manifest"], reviewed["human_report"]):
        assert set(evidence) == {"path", "git_blob"}
        assert (ROOT / evidence["path"]).is_file()
        assert (
            _git("hash-object", "--", evidence["path"]).stdout.strip()
            == evidence["git_blob"]
        )

    source_manifest = _load_mapping(ROOT / SOURCE_MANIFEST_RELATIVE)
    source_lineages = source_manifest["repositories"]["ipfs_accelerate_py"][
        "relevant_unmerged"
    ]
    by_head = {item["head"]: item for item in source_lineages}
    expected_scopes = (
        "dcr_git_authority_and_replay",
        "self_hosting_recovery",
        "semantic_runtime_residual",
        "task_contract_residual",
    )
    recorded_lineages = reconciliation["residual_lineages"]
    assert tuple(item["scope"] for item in recorded_lineages) == expected_scopes
    for recorded in recorded_lineages:
        assert set(recorded) == {
            "scope",
            "branch",
            "head",
            "merge_base",
            "classification",
            "safe_to_cherry_pick",
            "disposition",
        }
        actual = by_head[recorded["head"]]
        for field in (
            "branch",
            "head",
            "merge_base",
            "classification",
            "safe_to_cherry_pick",
            "disposition",
        ):
            assert recorded[field] == actual[field]
        assert recorded["safe_to_cherry_pick"] is False

    replacement = reconciliation["replacement_tls"]
    assert replacement == {
        "generator_path": "scripts/systemd/generate_self_signed_cert.py",
        "private_key_mode": "0600",
        "certificate_mode": "0644",
        "permission_failure": "raise_before_final_path_publication",
        "historical_private_key_trusted": False,
        "rotation_required_if_ever_trusted": True,
    }


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
