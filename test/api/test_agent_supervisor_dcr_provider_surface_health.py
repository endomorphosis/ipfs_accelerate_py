"""Adversarial DCR-013 provider surface health lifecycle tests."""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.deterministic_repair_forest import (
    DCR_SCHEDULER_POLICY_PATH,
    DCR_TODO_PATH,
    write_repair_forest,
)
from ipfs_accelerate_py.agent_supervisor.analysis.provider_surface_health import (
    DCR_ARTIFACT_PATH,
    DCR_CARRIER_SUBJECT,
    DCR_FOREST_PATH,
    DCR_TODO_SUBJECT,
    DEFAULT_MAX_BYTES,
    MANDATORY_PACKAGE_ROOTS,
    PROVIDER_SURFACE_HEALTH_INTERFACE,
    PROVIDER_SURFACE_HEALTH_SCHEMA,
    SURFACE_CODEC,
    ProviderSurfaceHealth,
    ProviderSurfaceValidation,
    decode_surface_ledger,
    validate_provider_surface_health,
    write_provider_surface_health,
)
from ipfs_accelerate_py.agent_supervisor.analysis.provider_surface_health import (
    main as surface_main,
)
from ipfs_accelerate_py.agent_supervisor.analysis.python_mcp_surface_extractor import (
    PYTHON_MCP_SURFACE_EXTRACTOR_INTERFACE,
)

_ROOTS_SCHEMA = "ipfs_accelerate_py/agent-supervisor/deterministic-repair-roots@1"
_AUTHORITY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-repair-authority-policy@1"
)
_SCHEDULER_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor."
    "deterministic_swissknife_mcplusplus_repair.scheduler_config@1"
)
_RUNTIME_ROOT = "data/agent_supervisor/deterministic_contract_repair"
_FIXTURE_GIT_USER_NAME = "DCR provider surface test"
_FIXTURE_GIT_USER_EMAIL = "dcr-provider-surface@example.invalid"
_FIXTURE_GIT_CONFIG = (
    "-c",
    "protocol.file.allow=always",
    "-c",
    f"user.name={_FIXTURE_GIT_USER_NAME}",
    "-c",
    f"user.email={_FIXTURE_GIT_USER_EMAIL}",
)


def _git(path: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", *_FIXTURE_GIT_CONFIG, *arguments),
        cwd=path,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode:
        raise AssertionError(
            f"git {' '.join(arguments)} failed in {path}: {result.stderr}"
        )
    return result.stdout.strip()


def _initialize_repository(path: Path) -> None:
    path.mkdir(parents=True)
    _git(path, "init", "-b", "main")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _seed_provider(path: Path, package: str, *, with_duplicate: bool = False) -> None:
    _initialize_repository(path)
    (path / ".gitignore").write_text(
        "ignored-state.txt\n__pycache__/\n.pytest_cache/\n",
        encoding="utf-8",
    )
    package_dir = path / package
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text(
        f'"""{package} fixture package."""\n',
        encoding="utf-8",
    )
    tools = package_dir / "mcp_server"
    tools.mkdir()
    (tools / "__init__.py").write_text("", encoding="utf-8")
    (tools / "register.py").write_text(
        f'''
def require_capability(ctx):
    return True

def store_blob(content: bytes, pin: bool = True) -> dict:
    require_capability("write")
    return {{"ok": True, "pin": pin}}

server.add_tool(store_blob, name="{package}.store")

@server.tool(name="tools.list_tools")
async def list_tools(category: str) -> list:
    return []
''',
        encoding="utf-8",
    )
    if with_duplicate:
        (tools / "register_dup.py").write_text(
            f'''
def other_store(content: bytes) -> dict:
    return {{"ok": False}}

server.add_tool(other_store, name="{package}.store")
''',
            encoding="utf-8",
        )
    (package_dir / "archive").mkdir()
    (package_dir / "archive" / "legacy.py").write_text(
        '''
def legacy_echo(value: str) -> str:
    return value

server.add_tool(legacy_echo, name="legacy.echo")
''',
        encoding="utf-8",
    )
    (package_dir / "tests").mkdir()
    (package_dir / "tests" / "test_helpers.py").write_text(
        "def test_noop():\n    assert True\n",
        encoding="utf-8",
    )
    _git(path, "add", ".")
    _git(path, "commit", "-m", f"seed {package}")


def _seed_mcp_plus_plus(path: Path) -> None:
    _initialize_repository(path)
    (path / ".gitignore").write_text(
        "ignored-state.txt\n__pycache__/\n.pytest_cache/\n",
        encoding="utf-8",
    )
    (path / "README.md").write_text("# Mcp-Plus-Plus fixture\n", encoding="utf-8")
    validators = path / "tests-py" / "validators"
    validators.mkdir(parents=True)
    (validators / "__init__.py").write_text("", encoding="utf-8")
    (validators / "base_mcp.py").write_text(
        '''
def validate(payload: dict) -> bool:
    return True
''',
        encoding="utf-8",
    )
    tools = path / "tests-py" / "tools"
    tools.mkdir(parents=True)
    (tools / "register.py").write_text(
        '''
def echo(value: str) -> str:
    return value

server.add_tool(echo, name="mcp_plus_plus.echo")
''',
        encoding="utf-8",
    )
    _git(path, "add", ".")
    _git(path, "commit", "-m", "seed mcp-plus-plus")


def _seed_swissknife(path: Path) -> None:
    _initialize_repository(path)
    (path / ".gitignore").write_text(
        "ignored-state.txt\n__pycache__/\n.pytest_cache/\n",
        encoding="utf-8",
    )
    (path / "README.md").write_text("# swissknife fixture\n", encoding="utf-8")
    _git(path, "add", ".")
    _git(path, "commit", "-m", "seed swissknife")


def _root_policy() -> dict[str, Any]:
    declarations = (
        ("orchestration", ".", "orchestration_only"),
        ("swissknife", "swissknife", "consumer"),
        ("mcp-plus-plus", "Mcp-Plus-Plus", "consumer"),
        ("ipfs-accelerate", "external/ipfs_accelerate", "provider"),
        ("ipfs-datasets", "external/ipfs_datasets", "provider"),
        ("ipfs-kit", "external/ipfs_kit", "provider"),
    )
    return {
        "schema": _ROOTS_SCHEMA,
        "interface": "RepairRootOwnership@1",
        "roots": [
            {
                "id": root_id,
                "relative_path": relative,
                "role": role,
                "allowed_write_prefixes": [] if relative == "." else ["."],
                "pin_path": "" if relative == "." else relative,
            }
            for root_id, relative, role in declarations
        ],
    }


def _authority_policy() -> dict[str, Any]:
    return {
        "schema": _AUTHORITY_SCHEMA,
        "interface": "DeterministicRepairAuthorityPolicy@1",
        "runtime": "target_repair_runtime",
        "pin_strategy": "verified_capability_receipt_ids",
        "local_logic_pins": [],
        "prover_subprocess_pins": [],
        "loopback_mcp_pins": [],
        "model_call_budget": 0,
        "llm_call_budget": 0,
        "remote_provider_call_budget": 0,
        "network_policy": "deny_except_explicit_loopback",
        "model_or_remote_fallback_authorized": False,
    }


def _scheduler_policy() -> dict[str, Any]:
    return {
        "schema": _SCHEDULER_SCHEMA,
        "runtime_paths": {
            "root": _RUNTIME_ROOT,
            "state": f"{_RUNTIME_ROOT}/state",
            "worktrees": f"{_RUNTIME_ROOT}/worktrees",
            "merge_queue": f"{_RUNTIME_ROOT}/merge-queue",
            "logs": f"{_RUNTIME_ROOT}/logs",
            "evidence": f"{_RUNTIME_ROOT}/evidence",
            "generated_runtime_artifacts_are_completion_authority": False,
        },
    }


def _make_workspace(tmp_path: Path, *, with_duplicate: bool = False) -> Path:
    sources = tmp_path / "FixtureSources"
    repositories = {
        "swissknife": sources / "SwissKnifeSource",
        "Mcp-Plus-Plus": sources / "McpPlusPlusSource",
        "external/ipfs_accelerate": sources / "AccelerateSource",
        "external/ipfs_datasets": sources / "DatasetsSource",
        "external/ipfs_kit": sources / "KitSource",
    }
    _seed_swissknife(repositories["swissknife"])
    _seed_mcp_plus_plus(repositories["Mcp-Plus-Plus"])
    _seed_provider(
        repositories["external/ipfs_accelerate"],
        "ipfs_accelerate_py",
        with_duplicate=with_duplicate,
    )
    _seed_provider(repositories["external/ipfs_datasets"], "ipfs_datasets_py")
    _seed_provider(repositories["external/ipfs_kit"], "ipfs_kit_py")

    workspace = tmp_path / "CaseSensitiveWorkspace"
    _initialize_repository(workspace)
    (workspace / ".gitignore").write_text(
        "ignored-state.txt\n__pycache__/\n.pytest_cache/\ndata/*\n",
        encoding="utf-8",
    )
    _write_json(
        workspace / "config/deterministic_contract_repair_roots.json",
        _root_policy(),
    )
    _write_json(
        workspace / "config/deterministic_contract_repair_authority.json",
        _authority_policy(),
    )
    _write_json(
        workspace.joinpath(*Path(DCR_SCHEDULER_POLICY_PATH).parts),
        _scheduler_policy(),
    )
    todo = workspace.joinpath(*Path(DCR_TODO_PATH).parts)
    todo.parent.mkdir(parents=True)
    todo.write_text(
        "# Deterministic repair\n\n"
        "## DCR-011 Multi-root forest\n\n"
        "- Status: completed\n"
        "- Acceptance: bind the real repository forest.\n\n"
        "## DCR-013 Index complete actual provider registration and handler surfaces\n\n"
        "- Status: todo\n"
        "- Acceptance: compact actual surface inventory.\n",
        encoding="utf-8",
    )
    (workspace / "root_note.md").write_text("orchestration note\n", encoding="utf-8")
    _git(workspace, "add", ".gitignore", "config", "implementation_plan", "root_note.md")
    _git(workspace, "commit", "-m", "seed orchestration policy")
    for relative, source in repositories.items():
        _git(workspace, "submodule", "add", str(source), relative)
    _git(workspace, "commit", "-am", "pin the six-root test forest")
    _git(workspace, "submodule", "update", "--init", "--recursive")
    assert not _git(workspace, "status", "--porcelain=v1")
    return workspace


@dataclass
class LifecycleFixture:
    workspace: Path
    forest_manifest: Any
    surface_payload: dict[str, Any]
    subject: str
    branch: str = "implementation/dcr-013-provider"

    @property
    def artifact(self) -> Path:
        return self.workspace.joinpath(*Path(DCR_ARTIFACT_PATH).parts)

    @property
    def forest_artifact(self) -> Path:
        return self.workspace.joinpath(*Path(DCR_FOREST_PATH).parts)

    def validate(self) -> ProviderSurfaceValidation:
        return validate_provider_surface_health(
            self.artifact,
            self.workspace,
            forest_path=self.forest_artifact,
        )

    def carry(self) -> str:
        _git(self.workspace, "add", "--force", DCR_ARTIFACT_PATH)
        _git(self.workspace, "commit", "-m", DCR_CARRIER_SUBJECT)
        return _git(self.workspace, "rev-parse", "HEAD")

    def merge(self) -> str:
        _git(self.workspace, "switch", "main")
        _git(self.workspace, "submodule", "update", "--init", "--recursive")
        _git(
            self.workspace,
            "merge",
            "--no-ff",
            self.branch,
            "-m",
            "Merge branch 'implementation/dcr-013-provider' into main",
        )
        _git(self.workspace, "submodule", "update", "--init", "--recursive")
        return _git(self.workspace, "rev-parse", "HEAD")

    def complete_todo(self) -> str:
        todo = self.workspace.joinpath(*Path(DCR_TODO_PATH).parts)
        contents = todo.read_text(encoding="utf-8")
        contents = contents.replace(
            "## DCR-013 Index complete actual provider registration and handler surfaces\n\n"
            "- Status: todo\n",
            "## DCR-013 Index complete actual provider registration and handler surfaces\n\n"
            "- Status: completed\n",
            1,
        )
        todo.write_text(contents, encoding="utf-8")
        _git(self.workspace, "add", DCR_TODO_PATH)
        _git(self.workspace, "commit", "-m", DCR_TODO_SUBJECT)
        return _git(self.workspace, "rev-parse", "HEAD")


def _prepare_subject(
    tmp_path: Path, *, with_duplicate: bool = False
) -> LifecycleFixture:
    workspace = _make_workspace(tmp_path, with_duplicate=with_duplicate)
    forest_branch = "implementation/dcr-011-provider"
    _git(workspace, "switch", "-c", forest_branch)
    forest_path = workspace.joinpath(*Path(DCR_FOREST_PATH).parts)
    forest_path.parent.mkdir(parents=True, exist_ok=True)
    forest_manifest = write_repair_forest(forest_path, workspace)
    _git(workspace, "add", "--force", DCR_FOREST_PATH)
    _git(
        workspace,
        "commit",
        "-m",
        "DCR-011: Materialize one current multi-root forest and overlay identity",
    )
    _git(workspace, "switch", "main")
    _git(
        workspace,
        "merge",
        "--no-ff",
        forest_branch,
        "-m",
        "Merge branch 'implementation/dcr-011-provider' into main",
    )
    assert forest_path.is_file()

    branch = "implementation/dcr-013-provider"
    subject = _git(workspace, "rev-parse", "HEAD")
    _git(workspace, "switch", "-c", branch)
    artifact = workspace.joinpath(*Path(DCR_ARTIFACT_PATH).parts)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    payload = write_provider_surface_health(
        artifact,
        workspace,
        forest_path=forest_path,
    )
    return LifecycleFixture(
        workspace=workspace,
        forest_manifest=forest_manifest,
        surface_payload=payload,
        subject=subject,
        branch=branch,
    )


def test_mandatory_package_roots_cover_accelerate_datasets_kit_and_mcp() -> None:
    root_ids = {item["root_id"] for item in MANDATORY_PACKAGE_ROOTS}
    assert root_ids == {
        "ipfs-accelerate",
        "ipfs-datasets",
        "ipfs-kit",
        "mcp-plus-plus",
    }
    assert PROVIDER_SURFACE_HEALTH_INTERFACE == "ProviderSurfaceHealth@1"
    assert ProviderSurfaceHealth.interface == PROVIDER_SURFACE_HEALTH_INTERFACE
    assert ProviderSurfaceHealth.schema == PROVIDER_SURFACE_HEALTH_SCHEMA
    assert PYTHON_MCP_SURFACE_EXTRACTOR_INTERFACE.startswith("PythonMcpSurfaceExtractor")


def test_surface_ledger_round_trip_and_classifications(tmp_path: Path) -> None:
    fixture = _prepare_subject(tmp_path)
    ledger = fixture.surface_payload["surface_ledger"]
    assert ledger["codec"] == SURFACE_CODEC
    rows, digest = decode_surface_ledger(ledger)
    assert digest == fixture.surface_payload["surface_uncompressed_digest"]
    assert len(rows) == ledger["row_count"]

    packages = {item["root_id"]: item for item in fixture.surface_payload["packages"]}
    assert set(packages) == {
        "ipfs-accelerate",
        "ipfs-datasets",
        "ipfs-kit",
        "mcp-plus-plus",
    }
    for package in packages.values():
        assert package["scanned_file_count"] > 0
        assert package["inventory_merkle"].startswith("sha256:")
        assert package["surface_merkle"].startswith("sha256:")

    registrations = [item for item in rows if item.row_kind.value == "registration"]
    assert registrations
    assert any(item.symbol.endswith(".store") for item in registrations)
    assert any(item.symbol == "tools.list_tools" for item in registrations)
    assert any(item.classification.value == "archive" for item in registrations)
    # Expected descriptors are not required: only actual registrations appear.
    assert all(item.registration_api for item in registrations)


def test_artifact_stays_under_admission_limit_and_binds_forest(
    tmp_path: Path,
) -> None:
    fixture = _prepare_subject(tmp_path)
    encoded = fixture.artifact.read_bytes()
    assert len(encoded) <= DEFAULT_MAX_BYTES
    payload = json.loads(encoded.decode("utf-8"))
    assert payload["schema"] == PROVIDER_SURFACE_HEALTH_SCHEMA
    assert payload["interface"] == PROVIDER_SURFACE_HEALTH_INTERFACE
    assert payload["extractor_interface"] == PYTHON_MCP_SURFACE_EXTRACTOR_INTERFACE
    assert payload["forest_id"] == fixture.forest_manifest.forest_id
    assert payload["forest_historical_proof"]["integrity_valid"] is True
    assert payload["lifecycle"]["task_id"] == "DCR-013"
    assert payload["authoritative"] is False
    assert payload["completion_authorized"] is False


def test_validate_accepts_capture_and_strict_lifecycle(tmp_path: Path) -> None:
    fixture = _prepare_subject(tmp_path)
    captured = fixture.validate()
    assert captured.integrity_valid, captured.reason_codes
    assert captured.current, captured.reason_codes
    assert captured.lifecycle_state == "captured"
    assert captured.downstream_authorized

    fixture.carry()
    carried = fixture.validate()
    assert carried.current, carried.reason_codes
    assert carried.lifecycle_state == "artifact_carried"

    fixture.merge()
    integrated = fixture.validate()
    assert integrated.current, integrated.reason_codes
    assert integrated.lifecycle_state == "integrated"

    fixture.complete_todo()
    completed = fixture.validate()
    assert completed.integrity_valid, completed.reason_codes
    assert completed.current, completed.reason_codes
    assert completed.lifecycle_state == "todo_completed"


def test_duplicate_active_registrations_block_parity(tmp_path: Path) -> None:
    fixture = _prepare_subject(tmp_path, with_duplicate=True)
    health = fixture.surface_payload["health"]
    assert health["parity_authorized"] is False
    assert "duplicate_equivalence_rows" in health["reason_codes"]
    assert health["blockers"]["duplicate_equivalence_rows"] >= 1
    rows, _ = decode_surface_ledger(fixture.surface_payload["surface_ledger"])
    assert any(item.row_kind.value == "duplicate_equivalence" for item in rows)
    # Capture remains integrity-valid even when parity is blocked.
    result = fixture.validate()
    assert result.integrity_valid
    assert result.current


def test_live_cli_validate_rejects_wrong_artifact_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = _prepare_subject(tmp_path)
    wrong = fixture.workspace / "wrong-provider-surfaces.json"
    wrong.write_bytes(fixture.artifact.read_bytes())
    code = surface_main(
        [
            "validate",
            "--workspace",
            str(fixture.workspace),
            "--forest",
            str(fixture.forest_artifact),
            "--artifact",
            str(wrong),
            "--max-bytes",
            str(DEFAULT_MAX_BYTES),
        ]
    )
    assert code == 1
    output = json.loads(capsys.readouterr().out)
    assert "provider_surface_output_path_invalid" in output["reason_codes"]


def test_tampered_ledger_payload_fails_closed(tmp_path: Path) -> None:
    fixture = _prepare_subject(tmp_path)
    payload = json.loads(fixture.artifact.read_text(encoding="utf-8"))
    ledger = payload["surface_ledger"]
    raw = base64.b64decode(ledger["payload"].encode("ascii"))
    tampered = bytearray(zlib.decompress(raw))
    tampered[0] ^= 0xFF
    recompressed = zlib.compress(bytes(tampered), level=9)
    ledger["payload"] = base64.b64encode(recompressed).decode("ascii")
    ledger["payload_sha256"] = "sha256:" + hashlib.sha256(recompressed).hexdigest()
    body = {
        key: value for key, value in payload.items() if key != "provider_surface_id"
    }
    payload["provider_surface_id"] = "sha256:" + hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    fixture.artifact.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result = fixture.validate()
    assert not result.integrity_valid
    assert result.reason_codes


def test_dirty_worktree_does_not_change_forest_bound_scan(tmp_path: Path) -> None:
    fixture = _prepare_subject(tmp_path)
    dirty = (
        fixture.workspace
        / "external"
        / "ipfs_accelerate"
        / "ipfs_accelerate_py"
        / "mcp_server"
        / "dirty_only.py"
    )
    dirty.write_text(
        '''
def dirty_only(value: str) -> str:
    return value

server.add_tool(dirty_only, name="dirty.only")
''',
        encoding="utf-8",
    )
    # Re-materialize against the same forest-bound trees.
    rewritten = write_provider_surface_health(
        fixture.artifact,
        fixture.workspace,
        forest_path=fixture.forest_artifact,
    )
    rows, _ = decode_surface_ledger(rewritten["surface_ledger"])
    assert not any(item.symbol == "dirty.only" for item in rows)
    assert rewritten["provider_surface_id"] == fixture.surface_payload["provider_surface_id"]
