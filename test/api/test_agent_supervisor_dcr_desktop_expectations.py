"""Adversarial DCR-014 SwissKnife desktop expected-contract index tests."""

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

from ipfs_accelerate_py.agent_supervisor.analysis.deterministic_desktop_expectations import (
    CONSUMER_CODEC,
    DCR_ARTIFACT_PATH,
    DCR_CARRIER_SUBJECT,
    DCR_FOREST_PATH,
    DCR_TODO_SUBJECT,
    DEFAULT_MAX_BYTES,
    DESKTOP_CONSUMER_LEDGER_SCHEMA,
    DESKTOP_EXPECTATIONS_SCHEMA,
    DESKTOP_INVENTORY_LEDGER_SCHEMA,
    INVENTORY_CODEC,
    DesktopExpectationsValidation,
    decode_prefix_ledger,
    validate_desktop_expectations,
    write_desktop_expectations,
)
from ipfs_accelerate_py.agent_supervisor.analysis.deterministic_desktop_expectations import (
    main as desktop_main,
)
from ipfs_accelerate_py.agent_supervisor.analysis.deterministic_repair_forest import (
    DCR_SCHEDULER_POLICY_PATH,
    DCR_TODO_PATH,
    write_repair_forest,
)
from ipfs_accelerate_py.agent_supervisor.analysis.mcp_contract_catalog import (
    MCP_CONTRACT_CATALOG_INTERFACE,
    SourceAuthorityClass,
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
_FIXTURE_GIT_USER_NAME = "DCR desktop expectations test"
_FIXTURE_GIT_USER_EMAIL = "dcr-desktop-expectations@example.invalid"
_FIXTURE_GIT_CONFIG = (
    "-c",
    "protocol.file.allow=always",
    "-c",
    f"user.name={_FIXTURE_GIT_USER_NAME}",
    "-c",
    f"user.email={_FIXTURE_GIT_USER_EMAIL}",
)

_CANONICAL_DESCRIPTOR = """
export const IPFS_KIT_INTERFACE: MCPPPInterfaceDescriptor = {
  name: 'ipfs-kit',
  namespace: 'com.ipfs.kit',
  version: '1.0.0',
  interface_cid: 'bafy-kit',
  methods: [{
    name: 'ipfs.add',
    input_schema_cid: 'bafy-in',
    output_schema_cid: 'bafy-out',
    error_schema_cids: ['bafy-error'],
  }],
  errors: [{ name: 'Unavailable', code: 503 }],
  requires: ['mcp++/cid-envelope', 'mcp++/deontic-policy', 'mcp++/p2p-transport'],
  compatibility: { compatible_with: [], supersedes: [] },
};
export const IPFS_ACCELERATE_INTERFACE: MCPPPInterfaceDescriptor = {
  name: 'ipfs-accelerate',
  namespace: 'com.ipfs.accelerate',
  version: '1.0.0',
  interface_cid: 'bafy-accelerate',
  methods: [{
    name: 'accelerate.inference',
    input_schema_cid: 'bafy-in',
    output_schema_cid: 'bafy-out',
    error_schema_cids: ['bafy-error'],
    interaction_pattern: 'stream',
  }],
  errors: [],
  requires: ['mcp++/cid-envelope'],
  compatibility: { compatible_with: [], supersedes: [] },
};
export const IPFS_DATASETS_INTERFACE: MCPPPInterfaceDescriptor = {
  name: 'ipfs-datasets',
  namespace: 'com.ipfs.datasets',
  version: '1.0.0',
  interface_cid: 'bafy-datasets',
  methods: [{
    name: 'datasets.search',
    input_schema_cid: 'bafy-in',
    output_schema_cid: 'bafy-out',
    error_schema_cids: ['bafy-error'],
  }],
  errors: [],
  requires: ['mcp++/cid-envelope'],
  compatibility: { compatible_with: [], supersedes: [] },
};
"""

_CAPABILITY_REGISTRY = """
export const swissknifeMCPCapabilityRegistry = [{
  server_package: 'ipfs_kit_py',
  transport: 'mcp-server',
  capability_descriptor: {
    command_intents: [
      { intent: 'storage.add', tool_name: 'ipfs_add',
        upstream_function: '/api/v0/ipfs/add',
        payload_contracts: ['content_ref'] },
    ],
  },
}];
"""

_CONNECTOR = """
export class Connector {
  async list() { return this.jsonRpc('tools/list', {}); }
  async call(name: string, args: object) {
    return this.jsonRpc('tools/call', { name, arguments: args });
  }
}
"""

_UIIR = """
export const UI_UX_IR_SCHEMA_VERSION = 'ui-ux-ir/v1' as const;
export const UIIR_DOCUMENT_FIELDS = Object.freeze([
  'schema_version',
  'document_id',
  'title',
  'components',
]);
export function decodeUIIRDocument(value: unknown) {
  return value;
}
"""

_ORB = """
export const ORB_TRANSPORT_KINDS = ['local', 'websocket', 'http', 'mcp-server'] as const;
export interface ORBDescriptorSource {
  cid: string;
  descriptor: object;
}
export function bindORB(source: ORBDescriptorSource) {
  return source.cid;
}
"""

_SCHEMA = """
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://example.invalid/contracts/desktop.json",
  "title": "Desktop contract",
  "type": "object",
  "properties": {
    "tool": { "type": "string" }
  }
}
"""


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


def _seed_repository(path: Path, label: str, *, swissknife: bool = False) -> None:
    _initialize_repository(path)
    (path / ".gitignore").write_text(
        "ignored-state.txt\n__pycache__/\n.pytest_cache/\n",
        encoding="utf-8",
    )
    (path / "README.md").write_text(f"# {label}\n", encoding="utf-8")
    if swissknife:
        mcp = path / "src/services/mcp"
        apps = path / "src/services/apps"
        contracts = path / "contracts"
        archive = path / "src/services/mcp/archive"
        generated = path / "src/services/mcp/generated"
        tests = path / "test/mcp-plus-plus"
        for directory in (mcp, apps, contracts, archive, generated, tests):
            directory.mkdir(parents=True, exist_ok=True)
        (mcp / "mcp-plus-plus.ts").write_text(_CANONICAL_DESCRIPTOR, encoding="utf-8")
        (mcp / "mcp-plus-plus-connector.ts").write_text(_CONNECTOR, encoding="utf-8")
        (mcp / "ui-ux-ir-codec.ts").write_text(_UIIR, encoding="utf-8")
        (mcp / "mcp-orb-capability-router.ts").write_text(_ORB, encoding="utf-8")
        (apps / "swissknife-mcp-capability-registry.ts").write_text(
            _CAPABILITY_REGISTRY, encoding="utf-8"
        )
        (contracts / "desktop-tool.schema.json").write_text(_SCHEMA, encoding="utf-8")
        (archive / "old-descriptor.ts").write_text(
            "export const OLD = { name: 'stale' };\n",
            encoding="utf-8",
        )
        (generated / "generated-binding.ts").write_text(
            "// auto-generated\nexport const GEN = 1;\n",
            encoding="utf-8",
        )
        (tests / "descriptor.test.ts").write_text(
            "export const TEST_DESC: MCPPPInterfaceDescriptor = {\n"
            "  name: 'fixture-test',\n"
            "  namespace: 'com.fixture.test',\n"
            "  version: '0.0.1',\n"
            "  interface_cid: 'bafy-test',\n"
            "  methods: [],\n"
            "  errors: [],\n"
            "  requires: [],\n"
            "  compatibility: { compatible_with: [], supersedes: [] },\n"
            "};\n",
            encoding="utf-8",
        )
        _git(
            path,
            "add",
            ".gitignore",
            "README.md",
            "src",
            "contracts",
            "test",
        )
    else:
        _git(path, "add", ".gitignore", "README.md")
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
        _seed_repository(source, relative, swissknife=(relative == "swissknife"))

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
        "## DCR-014 Index SwissKnife desktop expected contracts and UI bindings\n\n"
        "- Status: todo\n"
        "- Acceptance: every active desktop MCP consumer is accounted for.\n",
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
    desktop_payload: dict[str, Any]
    subject: str
    branch: str = "implementation/dcr-014-provider"

    @property
    def artifact(self) -> Path:
        return self.workspace.joinpath(*Path(DCR_ARTIFACT_PATH).parts)

    @property
    def forest_artifact(self) -> Path:
        return self.workspace.joinpath(*Path(DCR_FOREST_PATH).parts)

    def validate(self) -> DesktopExpectationsValidation:
        return validate_desktop_expectations(
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
            "Merge branch 'implementation/dcr-014-provider' into main",
        )
        _git(self.workspace, "submodule", "update", "--init", "--recursive")
        return _git(self.workspace, "rev-parse", "HEAD")

    def complete_todo(self) -> str:
        todo = self.workspace.joinpath(*Path(DCR_TODO_PATH).parts)
        contents = todo.read_text(encoding="utf-8")
        contents = contents.replace(
            "## DCR-014 Index SwissKnife desktop expected contracts and UI bindings\n\n"
            "- Status: todo\n",
            "## DCR-014 Index SwissKnife desktop expected contracts and UI bindings\n\n"
            "- Status: completed\n",
            1,
        )
        todo.write_text(contents, encoding="utf-8")
        _git(self.workspace, "add", DCR_TODO_PATH)
        _git(self.workspace, "commit", "-m", DCR_TODO_SUBJECT)
        return _git(self.workspace, "rev-parse", "HEAD")


def _prepare_subject(tmp_path: Path) -> LifecycleFixture:
    workspace = _make_workspace(tmp_path)
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

    branch = "implementation/dcr-014-provider"
    subject = _git(workspace, "rev-parse", "HEAD")
    _git(workspace, "switch", "-c", branch)
    artifact = workspace.joinpath(*Path(DCR_ARTIFACT_PATH).parts)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    payload = write_desktop_expectations(
        artifact,
        workspace,
        forest_path=forest_path,
    )
    return LifecycleFixture(
        workspace=workspace,
        forest_manifest=forest_manifest,
        desktop_payload=payload,
        subject=subject,
        branch=branch,
    )


def test_catalog_interface_and_desktop_interfaces_are_bound(tmp_path: Path) -> None:
    fixture = _prepare_subject(tmp_path)
    interfaces = fixture.desktop_payload["interfaces"]
    assert interfaces["mcp_contract_catalog"] == MCP_CONTRACT_CATALOG_INTERFACE
    assert interfaces["uiir_document"] == "UIIRDocument"
    assert interfaces["mcp_idl"] == "MCP-IDL"
    assert interfaces["orb"] == "ORB"
    assert interfaces["desktop_expectation_index"] == "DesktopExpectationIndex@1"


def test_inventory_and_consumer_ledgers_round_trip(tmp_path: Path) -> None:
    fixture = _prepare_subject(tmp_path)
    inventory = fixture.desktop_payload["inventory"]
    consumers = fixture.desktop_payload["consumers"]
    assert inventory["scanned_file_count"] >= 6
    assert consumers["active_count"] >= 1
    assert inventory["ledger"]["codec"] == INVENTORY_CODEC
    assert consumers["ledger"]["codec"] == CONSUMER_CODEC
    inv_rows, inv_digest = decode_prefix_ledger(
        inventory["ledger"],
        expected_schema=DESKTOP_INVENTORY_LEDGER_SCHEMA,
        expected_codec=INVENTORY_CODEC,
    )
    con_rows, con_digest = decode_prefix_ledger(
        consumers["ledger"],
        expected_schema=DESKTOP_CONSUMER_LEDGER_SCHEMA,
        expected_codec=CONSUMER_CODEC,
    )
    assert inv_digest == inventory["uncompressed_digest"]
    assert con_digest == consumers["uncompressed_digest"]
    assert len(inv_rows) == inventory["scanned_file_count"]
    assert len(con_rows) == consumers["total_count"]
    classifications = {row[2] for row in inv_rows}
    assert "active" in classifications
    assert "archive" in classifications
    assert "generated" in classifications
    assert "test" in classifications


def test_archive_and_generated_cannot_authorize_reviewed_contracts(
    tmp_path: Path,
) -> None:
    fixture = _prepare_subject(tmp_path)
    expectations = fixture.desktop_payload["expectations"]
    rows, _ = decode_prefix_ledger(
        expectations["ledger"],
        expected_schema=expectations["ledger"]["schema"],
        expected_codec=expectations["ledger"]["codec"],
    )
    inventory_rows, _ = decode_prefix_ledger(
        fixture.desktop_payload["inventory"]["ledger"],
        expected_schema=DESKTOP_INVENTORY_LEDGER_SCHEMA,
        expected_codec=INVENTORY_CODEC,
    )
    path_class = {row[0]: row[2] for row in inventory_rows}
    authorizing = {
        SourceAuthorityClass.AUTHORITATIVE.value,
        SourceAuthorityClass.CONFORMANCE.value,
        SourceAuthorityClass.REGISTRATION.value,
        SourceAuthorityClass.MANIFEST.value,
    }
    for row in rows:
        path = row[6]
        authority = row[3]
        if path_class.get(path) in {"archive", "generated"}:
            assert authority not in authorizing


def test_artifact_stays_under_admission_limit_and_binds_forest(
    tmp_path: Path,
) -> None:
    fixture = _prepare_subject(tmp_path)
    encoded = fixture.artifact.read_bytes()
    assert len(encoded) <= DEFAULT_MAX_BYTES
    payload = json.loads(encoded.decode("utf-8"))
    assert payload["schema"] == DESKTOP_EXPECTATIONS_SCHEMA
    assert payload["forest_id"] == fixture.forest_manifest.forest_id
    assert payload["forest_historical_proof"]["integrity_valid"] is True
    assert payload["lifecycle"]["task_id"] == "DCR-014"
    assert payload["parity"]["active_consumers_accounted"] is True
    assert payload["parity"]["safe_for_completion_reasoning"] is True
    assert payload["authority"]["nominating_cannot_override_reviewed"] is True


def test_validate_accepts_capture_and_strict_lifecycle(tmp_path: Path) -> None:
    fixture = _prepare_subject(tmp_path)
    captured = fixture.validate()
    assert captured.integrity_valid
    assert captured.current
    assert captured.lifecycle_state == "captured"
    assert captured.downstream_authorized

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
    assert completed.downstream_authorized


def test_live_cli_validate_rejects_wrong_artifact_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = _prepare_subject(tmp_path)
    wrong = fixture.workspace / "wrong-desktop-expectations.json"
    wrong.write_bytes(fixture.artifact.read_bytes())
    code = desktop_main(
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
    assert "desktop_output_path_invalid" in output["reason_codes"]


def test_tampered_consumer_ledger_fails_closed(tmp_path: Path) -> None:
    fixture = _prepare_subject(tmp_path)
    payload = json.loads(fixture.artifact.read_text(encoding="utf-8"))
    ledger = payload["consumers"]["ledger"]
    raw = base64.b64decode(ledger["payload"].encode("ascii"))
    tampered = bytearray(zlib.decompress(raw))
    tampered[0] ^= 0xFF
    recompressed = zlib.compress(bytes(tampered), level=9)
    ledger["payload"] = base64.b64encode(recompressed).decode("ascii")
    ledger["payload_sha256"] = "sha256:" + hashlib.sha256(recompressed).hexdigest()
    body = {
        key: value
        for key, value in payload.items()
        if key != "desktop_expectations_id"
    }
    payload["desktop_expectations_id"] = "sha256:" + hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    fixture.artifact.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result = fixture.validate()
    assert not result.integrity_valid
    assert result.reason_codes


def test_dirty_worktree_cannot_silently_authorize_without_carrier(
    tmp_path: Path,
) -> None:
    fixture = _prepare_subject(tmp_path)
    # Mutate the artifact in place without the reviewed carrier transition.
    payload = json.loads(fixture.artifact.read_text(encoding="utf-8"))
    payload["parity"]["blocking_reason_codes"] = ["injected"]
    payload["parity"]["safe_for_completion_reasoning"] = True
    body = {
        key: value
        for key, value in payload.items()
        if key != "desktop_expectations_id"
    }
    payload["desktop_expectations_id"] = "sha256:" + hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    fixture.artifact.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    result = fixture.validate()
    assert result.integrity_valid or not result.downstream_authorized
    assert not result.downstream_authorized or "completion_safe_claim_forged" in (
        result.reason_codes
    )


def test_repository_workspace_desktop_artifact_materializes_under_limit() -> None:
    """Materialize the checked-in projection when the live forest is present."""

    repository_root = Path(__file__).resolve().parents[4]
    forest = repository_root.joinpath(*Path(DCR_FOREST_PATH).parts)
    if not forest.is_file():
        pytest.skip("repository forest artifact is not present")
    artifact = repository_root.joinpath(*Path(DCR_ARTIFACT_PATH).parts)
    try:
        payload = write_desktop_expectations(
            artifact,
            repository_root,
            forest_path=forest,
            max_bytes=DEFAULT_MAX_BYTES,
        )
    except Exception as exc:  # pragma: no cover - surface exact failure
        pytest.fail(f"desktop expectation materialize failed: {exc}")
    encoded = artifact.read_bytes()
    assert len(encoded) <= DEFAULT_MAX_BYTES
    assert payload["schema"] == DESKTOP_EXPECTATIONS_SCHEMA
    assert payload["inventory"]["scanned_file_count"] >= 1
    assert payload["consumers"]["active_count"] >= 1
    assert payload["parity"]["active_consumers_accounted"] is True
    validation = validate_desktop_expectations(
        artifact,
        repository_root,
        forest_path=forest,
        max_bytes=DEFAULT_MAX_BYTES,
    )
    assert validation.integrity_valid, validation.reason_codes
    assert validation.current, validation.reason_codes
    assert validation.downstream_authorized, validation.reason_codes
