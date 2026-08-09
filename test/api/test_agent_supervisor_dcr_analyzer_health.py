"""Adversarial DCR-012 analyzer-health lifecycle and disposition tests."""

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

from ipfs_accelerate_py.agent_supervisor.analysis.deterministic_repair_analyzer_health import (
    ANALYZER_HEALTH_SCHEMA,
    DCR_ARTIFACT_PATH,
    DCR_CARRIER_SUBJECT,
    DCR_FOREST_PATH,
    DCR_TODO_SUBJECT,
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_SOURCE_BYTES,
    DISPOSITION_CODEC,
    AnalyzerHealthValidation,
    decode_disposition_ledger,
    validate_analyzer_health,
    write_analyzer_health,
)
from ipfs_accelerate_py.agent_supervisor.analysis.deterministic_repair_analyzer_health import (
    main as analyzer_main,
)
from ipfs_accelerate_py.agent_supervisor.analysis.deterministic_repair_forest import (
    DCR_SCHEDULER_POLICY_PATH,
    DCR_TODO_PATH,
    write_repair_forest,
)
from ipfs_accelerate_py.agent_supervisor.analysis.polyglot_ast_provider import (
    HARD_MAX_FILE_BYTES,
)
from ipfs_accelerate_py.agent_supervisor.analysis.repository_indexer import (
    DEFAULT_MAX_PARSER_SOURCE_BYTES,
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
_FIXTURE_GIT_USER_NAME = "DCR analyzer health test"
_FIXTURE_GIT_USER_EMAIL = "dcr-analyzer-health@example.invalid"
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


def _seed_repository(path: Path, label: str) -> None:
    _initialize_repository(path)
    (path / ".gitignore").write_text(
        "ignored-state.txt\n__pycache__/\n.pytest_cache/\n",
        encoding="utf-8",
    )
    (path / "README.md").write_text(f"# {label}\n", encoding="utf-8")
    (path / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (path / "data.json").write_text('{"ok": true}\n', encoding="utf-8")
    (path / "notes.jsonc").write_text(
        '// comment\n{"ok": true}\n',
        encoding="utf-8",
    )
    (path / "broken.py").write_text("def missing:\n", encoding="utf-8")
    (path / "script.ts").write_text("export const value = 1;\n", encoding="utf-8")
    _git(
        path,
        "add",
        ".gitignore",
        "README.md",
        "module.py",
        "data.json",
        "notes.jsonc",
        "broken.py",
        "script.ts",
    )
    _git(path, "commit", "-m", f"seed {label}")


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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _make_workspace(tmp_path: Path) -> Path:
    sources = tmp_path / "FixtureSources"
    repositories = {
        "swissknife": sources / "SwissKnifeSource",
        "Mcp-Plus-Plus": sources / "McpPlusPlusSource",
        "external/ipfs_accelerate": sources / "AccelerateSource",
        "external/ipfs_datasets": sources / "DatasetsSource",
        "external/ipfs_kit": sources / "KitSource",
    }
    for relative, source in repositories.items():
        _seed_repository(source, relative)

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
        "## DCR-012 Restore analyzer health and exact parser accounting\n\n"
        "- Status: todo\n"
        "- Acceptance: one disposition for every forest path.\n",
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
    analyzer_payload: dict[str, Any]
    subject: str
    branch: str = "implementation/dcr-012-provider"

    @property
    def artifact(self) -> Path:
        return self.workspace.joinpath(*Path(DCR_ARTIFACT_PATH).parts)

    @property
    def forest_artifact(self) -> Path:
        return self.workspace.joinpath(*Path(DCR_FOREST_PATH).parts)

    def validate(self) -> AnalyzerHealthValidation:
        return validate_analyzer_health(
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
            "Merge branch 'implementation/dcr-012-provider' into main",
        )
        _git(self.workspace, "submodule", "update", "--init", "--recursive")
        return _git(self.workspace, "rev-parse", "HEAD")

    def complete_todo(self) -> str:
        todo = self.workspace.joinpath(*Path(DCR_TODO_PATH).parts)
        contents = todo.read_text(encoding="utf-8")
        contents = contents.replace(
            "## DCR-012 Restore analyzer health and exact parser accounting\n\n"
            "- Status: todo\n",
            "## DCR-012 Restore analyzer health and exact parser accounting\n\n"
            "- Status: completed\n",
            1,
        )
        todo.write_text(contents, encoding="utf-8")
        _git(self.workspace, "add", DCR_TODO_PATH)
        _git(self.workspace, "commit", "-m", DCR_TODO_SUBJECT)
        return _git(self.workspace, "rev-parse", "HEAD")


def _prepare_subject(tmp_path: Path) -> LifecycleFixture:
    workspace = _make_workspace(tmp_path)
    # Materialize and integrate a DCR-011 forest first.
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

    branch = "implementation/dcr-012-provider"
    subject = _git(workspace, "rev-parse", "HEAD")
    _git(workspace, "switch", "-c", branch)
    artifact = workspace.joinpath(*Path(DCR_ARTIFACT_PATH).parts)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    payload = write_analyzer_health(
        artifact,
        workspace,
        forest_path=forest_path,
        parse_sources=True,
    )
    return LifecycleFixture(
        workspace=workspace,
        forest_manifest=forest_manifest,
        analyzer_payload=payload,
        subject=subject,
        branch=branch,
    )


def test_parser_source_bound_is_thirty_two_mebibytes() -> None:
    assert DEFAULT_MAX_SOURCE_BYTES == 32 * 1024 * 1024
    assert HARD_MAX_FILE_BYTES == 32 * 1024 * 1024
    assert DEFAULT_MAX_PARSER_SOURCE_BYTES == 32 * 1024 * 1024


def test_disposition_ledger_round_trip_is_lossless(tmp_path: Path) -> None:
    fixture = _prepare_subject(tmp_path)
    ledger = fixture.analyzer_payload["disposition_ledger"]
    assert ledger["codec"] == DISPOSITION_CODEC
    rows, digest = decode_disposition_ledger(ledger)
    assert digest == fixture.analyzer_payload["disposition_uncompressed_digest"]
    assert len(rows) == ledger["row_count"]
    assert len(rows) == len({(item.root_id, item.path) for item in rows})
    assert {item.root_id for item in rows} == {
        "orchestration",
        "swissknife",
        "mcp-plus-plus",
        "ipfs-accelerate",
        "ipfs-datasets",
        "ipfs-kit",
    }
    # JSONC must not be reported as a false JSON syntax error.
    jsonc_rows = [item for item in rows if item.path.endswith(".jsonc")]
    assert jsonc_rows
    assert all(item.reason_code == "jsonc_structured" for item in jsonc_rows)
    # Python syntax failures remain typed parse failures.
    broken = [item for item in rows if item.path.endswith("broken.py")]
    assert broken
    assert all(item.parser_status.value == "parse_failure" for item in broken)


def test_artifact_stays_under_admission_limit_and_binds_forest(
    tmp_path: Path,
) -> None:
    fixture = _prepare_subject(tmp_path)
    encoded = fixture.artifact.read_bytes()
    assert len(encoded) <= DEFAULT_MAX_BYTES
    payload = json.loads(encoded.decode("utf-8"))
    assert payload["schema"] == ANALYZER_HEALTH_SCHEMA
    assert payload["forest_id"] == fixture.forest_manifest.forest_id
    assert payload["forest_historical_proof"]["integrity_valid"] is True
    assert payload["lifecycle"]["task_id"] == "DCR-012"
    assert payload["parser_versions"]["persistent_typescript_worker"] is True


def test_validate_accepts_capture_and_strict_lifecycle(tmp_path: Path) -> None:
    fixture = _prepare_subject(tmp_path)
    captured = fixture.validate()
    assert captured.integrity_valid
    assert captured.current
    assert captured.lifecycle_state == "captured"

    fixture.carry()
    carried = fixture.validate()
    assert carried.current
    assert carried.lifecycle_state == "artifact_carried"

    fixture.merge()
    integrated = fixture.validate()
    assert integrated.current
    assert integrated.lifecycle_state == "integrated"

    fixture.complete_todo()
    completed = fixture.validate()
    assert completed.integrity_valid
    assert completed.current
    assert completed.lifecycle_state == "todo_completed"


def test_live_cli_validate_rejects_wrong_artifact_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = _prepare_subject(tmp_path)
    wrong = fixture.workspace / "wrong-analyzer-health.json"
    wrong.write_bytes(fixture.artifact.read_bytes())
    code = analyzer_main(
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
    assert "analyzer_output_path_invalid" in output["reason_codes"]


def test_tampered_ledger_payload_fails_closed(tmp_path: Path) -> None:
    fixture = _prepare_subject(tmp_path)
    payload = json.loads(fixture.artifact.read_text(encoding="utf-8"))
    ledger = payload["disposition_ledger"]
    raw = base64.b64decode(ledger["payload"].encode("ascii"))
    tampered = bytearray(zlib.decompress(raw))
    tampered[0] ^= 0xFF
    recompressed = zlib.compress(bytes(tampered), level=9)
    ledger["payload"] = base64.b64encode(recompressed).decode("ascii")
    ledger["payload_sha256"] = "sha256:" + hashlib.sha256(recompressed).hexdigest()
    body = {key: value for key, value in payload.items() if key != "analyzer_health_id"}
    payload["analyzer_health_id"] = "sha256:" + hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    fixture.artifact.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result = fixture.validate()
    assert not result.integrity_valid
    assert result.reason_codes


def test_materialize_does_not_use_stale_baseline_failure_count(
    tmp_path: Path,
) -> None:
    fixture = _prepare_subject(tmp_path)
    # The stored SCA 22-failure baseline is not an authority for DCR-012.
    assert fixture.analyzer_payload["funnel"]["path_count"] != 22
    assert "stale_baseline" not in fixture.analyzer_payload["health"]["reason_codes"]
