"""DCR-003 tests for real-path repair ownership and Gitlink admission."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.root_ownership import (
    REPAIR_ROOT_OWNERSHIP_INTERFACE,
    RepairRootOwnership,
    RootOwnershipDenied,
    SubmodulePinAdmission,
)

ROOT_IDS = (
    ("swissknife", "swissknife", "consumer"),
    ("mcp-plus-plus", "Mcp-Plus-Plus", "consumer"),
    ("ipfs-accelerate", "external/ipfs_accelerate", "provider"),
    ("ipfs-datasets", "external/ipfs_datasets", "provider"),
    ("ipfs-kit", "external/ipfs_kit", "provider"),
)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _configure(repo: Path) -> None:
    _git(repo, "config", "user.name", "DCR Root Test")
    _git(repo, "config", "user.email", "dcr-root@example.invalid")


def _canonical_id(record: dict[str, object]) -> str:
    encoded = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


@pytest.fixture
def ownership(tmp_path: Path) -> RepairRootOwnership:
    sources = tmp_path / "sources"
    sources.mkdir()
    source_by_id: dict[str, Path] = {}
    for root_id, _relative_path, _role in ROOT_IDS:
        source = sources / root_id
        source.mkdir()
        _git(source, "init", "-b", "main")
        _configure(source)
        _write(source / "src/module.py", "VALUE = 1\n")
        _git(source, "add", ".")
        _git(source, "commit", "-m", "seed child")
        source_by_id[root_id] = source

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _git(workspace, "init", "-b", "main")
    _configure(workspace)
    _write(workspace / "README.md", "orchestration only\n")
    _git(workspace, "add", "README.md")
    _git(workspace, "commit", "-m", "seed parent")
    for root_id, relative_path, _role in ROOT_IDS:
        _git(
            workspace,
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            str(source_by_id[root_id]),
            relative_path,
        )
    _git(workspace, "add", ".gitmodules", *[item[1] for item in ROOT_IDS])
    _git(workspace, "commit", "-m", "add DCR roots")

    roots = [
        {
            "id": "orchestration",
            "relative_path": ".",
            "role": "orchestration_only",
            "allowed_write_prefixes": [],
            "pin_path": "",
        }
    ]
    roots.extend(
        {
            "id": root_id,
            "relative_path": relative_path,
            "role": role,
            "allowed_write_prefixes": ["."],
            "pin_path": relative_path,
        }
        for root_id, relative_path, role in ROOT_IDS
    )
    return RepairRootOwnership.from_mapping(
        {
            "schema": ("ipfs_accelerate_py/agent-supervisor/deterministic-repair-roots@1"),
            "interface": REPAIR_ROOT_OWNERSHIP_INTERFACE,
            "roots": roots,
        },
        workspace_root=workspace,
    )


def _path(ownership: RepairRootOwnership, root_id: str) -> Path:
    return ownership.root_path(root_id) / "src/module.py"


def test_admission_binds_all_roots_and_mints_canonical_receipt(
    ownership: RepairRootOwnership,
) -> None:
    bindings = ownership.capture_bindings()
    receipt = ownership.admit_write(
        [_path(ownership, "ipfs-accelerate")],
        claimed_root_id="ipfs-accelerate",
        defect_root_id="ipfs-accelerate",
        bindings=bindings,
    )

    assert receipt.to_dict()["receipt_id"] == receipt.receipt_id
    assert receipt.write_paths == ("external/ipfs_accelerate/src/module.py",)
    assert [binding.root_id for binding in receipt.bindings] == sorted(bindings)


def test_realpath_escape_orchestration_and_cross_root_writes_deny(
    ownership: RepairRootOwnership,
    tmp_path: Path,
) -> None:
    bindings = ownership.capture_bindings()
    outside = tmp_path / "outside.py"
    _write(outside, "outside\n")
    escaped = ownership.root_path("ipfs-accelerate") / "src/escaped.py"
    escaped.symlink_to(outside)

    with pytest.raises(RootOwnershipDenied, match="escapes workspace"):
        ownership.owner_for(escaped)
    escaped.unlink()
    with pytest.raises(RootOwnershipDenied, match="orchestration-only"):
        ownership.admit_write(
            [ownership.workspace_root / "README.md"],
            claimed_root_id="orchestration",
            bindings=bindings,
        )
    with pytest.raises(RootOwnershipDenied, match="cross-root"):
        ownership.admit_write(
            [_path(ownership, "swissknife"), _path(ownership, "ipfs-accelerate")],
            claimed_root_id="swissknife",
            bindings=bindings,
        )


def test_unbound_dirty_changed_roots_and_consumer_weakening_deny(
    ownership: RepairRootOwnership,
) -> None:
    bindings = ownership.capture_bindings()
    bindings.pop("ipfs-accelerate")
    with pytest.raises(RootOwnershipDenied, match="unbound dirty"):
        ownership.admit_write(
            [_path(ownership, "swissknife")],
            claimed_root_id="swissknife",
            bindings=bindings,
        )

    bindings = ownership.capture_bindings()
    _write(_path(ownership, "ipfs-accelerate"), "VALUE = 2\n")
    with pytest.raises(RootOwnershipDenied, match="changed roots"):
        ownership.admit_write(
            [_path(ownership, "ipfs-accelerate")],
            claimed_root_id="ipfs-accelerate",
            bindings=bindings,
        )

    current = ownership.capture_bindings()
    with pytest.raises(RootOwnershipDenied, match="consumer weakening"):
        ownership.admit_write(
            [_path(ownership, "swissknife")],
            claimed_root_id="swissknife",
            defect_root_id="ipfs-accelerate",
            bindings=current,
        )


def test_ignored_and_symlink_overlay_changes_invalidate_bindings(
    ownership: RepairRootOwnership,
) -> None:
    target = ownership.root_path("ipfs-accelerate")
    git_dir = Path(_git(target, "rev-parse", "--git-dir"))
    if not git_dir.is_absolute():
        git_dir = (target / git_dir).resolve()
    _write(git_dir / "info/exclude", "ignored-state.txt\nignored-link\n")
    _write(target / "ignored-state.txt", "first\n")
    (target / "ignored-link").symlink_to("src/module.py")
    bindings = ownership.capture_bindings()

    _write(target / "ignored-state.txt", "second\n")
    with pytest.raises(RootOwnershipDenied, match="changed roots"):
        ownership.admit_write(
            [_path(ownership, "ipfs-accelerate")],
            claimed_root_id="ipfs-accelerate",
            bindings=bindings,
        )

    bindings = ownership.capture_bindings()
    (target / "ignored-link").unlink()
    (target / "ignored-link").symlink_to("ignored-state.txt")
    with pytest.raises(RootOwnershipDenied, match="changed roots"):
        ownership.admit_write(
            [_path(ownership, "ipfs-accelerate")],
            claimed_root_id="ipfs-accelerate",
            bindings=bindings,
        )


def test_submodule_pin_requires_bound_clean_target_receipt_and_validation(
    ownership: RepairRootOwnership,
) -> None:
    target = ownership.root_path("ipfs-accelerate")
    predecessor = _git(
        ownership.workspace_root,
        "rev-parse",
        "HEAD:external/ipfs_accelerate",
    )
    _write(target / "src/module.py", "VALUE = 2\n")
    _git(target, "add", "src/module.py")
    _git(target, "commit", "-m", "repair provider")
    successor = _git(target, "rev-parse", "HEAD")
    bindings = ownership.capture_bindings()
    root_receipt = ownership.admit_write(
        [target / "src/module.py"],
        claimed_root_id="ipfs-accelerate",
        defect_root_id="ipfs-accelerate",
        bindings=bindings,
    )
    validation = {
        "head": successor,
        "passed": True,
        "root_id": "ipfs-accelerate",
        "schema": "test/dcr-validation@1",
    }
    validation["receipt_id"] = _canonical_id(validation)
    admission = SubmodulePinAdmission(ownership)

    with pytest.raises(RootOwnershipDenied, match="premature pin"):
        admission.admit_pin_update(
            target_root_id="ipfs-accelerate",
            predecessor=predecessor,
            successor=successor,
            bindings=bindings,
            root_receipt=root_receipt,
            validation_receipt={"passed": False},
            changed_root_ids=("ipfs-accelerate",),
        )

    receipt = admission.admit_pin_update(
        target_root_id="ipfs-accelerate",
        predecessor=predecessor,
        successor=successor,
        bindings=bindings,
        root_receipt=root_receipt,
        validation_receipt=validation,
        changed_root_ids=("ipfs-accelerate",),
    )
    assert receipt.successor == successor
    assert receipt.to_dict()["receipt_id"] == receipt.receipt_id

    _git(ownership.workspace_root, "add", "external/ipfs_accelerate")
    staged_bindings = ownership.capture_bindings()
    staged_root_receipt = ownership.admit_write(
        [target / "src/module.py"],
        claimed_root_id="ipfs-accelerate",
        defect_root_id="ipfs-accelerate",
        bindings=staged_bindings,
    )
    with pytest.raises(RootOwnershipDenied, match="predecessor"):
        admission.admit_pin_update(
            target_root_id="ipfs-accelerate",
            predecessor=predecessor,
            successor=successor,
            bindings=staged_bindings,
            root_receipt=staged_root_receipt,
            validation_receipt=validation,
            changed_root_ids=("ipfs-accelerate",),
        )


def test_policy_rejects_root_set_weakening(
    ownership: RepairRootOwnership,
) -> None:
    raw = {
        "schema": "ipfs_accelerate_py/agent-supervisor/deterministic-repair-roots@1",
        "interface": REPAIR_ROOT_OWNERSHIP_INTERFACE,
        "roots": [
            {
                "id": root.root_id,
                "relative_path": root.relative_path,
                "role": root.role,
                "allowed_write_prefixes": list(root.allowed_write_prefixes),
                "pin_path": root.pin_path,
            }
            for root in ownership.roots
        ],
    }
    raw["roots"] = raw["roots"][:-1]
    with pytest.raises(ValueError, match="exact DCR-003 root set"):
        RepairRootOwnership.from_mapping(raw, workspace_root=ownership.workspace_root)


def test_reviewed_policy_loads_and_unknown_or_duplicate_fields_fail_closed(
    ownership: RepairRootOwnership,
    tmp_path: Path,
) -> None:
    workspace_root = Path(__file__).resolve().parents[4]
    policy_path = workspace_root / "config/deterministic_contract_repair_roots.json"
    loaded = RepairRootOwnership.from_file(
        policy_path,
        workspace_root=workspace_root,
    )
    assert {root.root_id for root in loaded.roots} == {
        "orchestration",
        "swissknife",
        "mcp-plus-plus",
        "ipfs-accelerate",
        "ipfs-datasets",
        "ipfs-kit",
    }

    raw = json.loads(policy_path.read_text(encoding="utf-8"))
    raw["unreviewed"] = True
    unknown = tmp_path / "unknown.json"
    unknown.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="unreadable|reviewed schema"):
        RepairRootOwnership.from_file(unknown, workspace_root=ownership.workspace_root)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema":"one","schema":"two","interface":"x","roots":[]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unreadable"):
        RepairRootOwnership.from_file(duplicate, workspace_root=ownership.workspace_root)
