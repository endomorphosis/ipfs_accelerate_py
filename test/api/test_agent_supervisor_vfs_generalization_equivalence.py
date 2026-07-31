"""Prove locked-source-to-generic VFS assurance equivalence (LPR-028).

Computes a structured ProgramContractDelta-style impact closure over the
source-lock modules, records Tactician/Hammer-style dispositions for supported
clauses (and explicit unsupported/approval-required dispositions otherwise),
and verifies caller migration, schema/identity parity, and thin-ops delegation.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pytest

from ipfs_accelerate_py.agent_supervisor.integrations.ipfs_kit_vfs_assurance import (
    CLOSED_ADAPTERS,
    build_ipfs_kit_vfs_assurance_profile,
    lazy_import_adapter,
    load_assurance_config,
    optional_providers_loaded,
    run_contracts,
    run_rollout,
    run_verify,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = REPO_ROOT / "config" / "agent_supervisor_vfs_generalization_sources.lock.json"
CONFIG_PATH = REPO_ROOT / "config" / "ipfs_kit_vfs_symbolic_assurance.json"
OPS_CLI = (
    REPO_ROOT
    / "scripts"
    / "ops"
    / "agent_supervisor"
    / "ipfs_kit_vfs_symbolic_assurance.py"
)
AGENT_SUPERVISOR_ROOT = REPO_ROOT / "ipfs_accelerate_py" / "agent_supervisor"
MAP_PATH = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "agent_supervisor"
    / "VFS_ASSURANCE_GENERALIZATION_MAP.md"
)

ORIGINAL_SCHEMAS = {
    "adversarial_e2e_gate": "vfs/adversarial-e2e-gate@1",
    "shadow_rollout_report": "vfs/shadow-rollout-report@1",
    "rollout_decision": "vfs/symbolic-rollout-decision@1",
    "control_request": "vfs/symbolic-control-request@1",
    "control_result": "vfs/symbolic-control-result@1",
    "bounded_status": "vfs/symbolic-bounded-status@1",
    "bounded_findings": "vfs/symbolic-bounded-findings@1",
    "bounded_receipts": "vfs/symbolic-bounded-receipts@1",
    "public_api": "vfs/symbolic-public-api@1",
}

ORIGINAL_IDS = {
    "behavior_id": "behavior:vfs-symbolic-assurance-rollout@1",
    "objective_id": "VFS-G130",
    "objective_revision": "VFS-G130@vfs-036",
    "requirement_id": "vfs-036:adversarial-e2e-control-parity-recovery-rollback",
}

SOURCE_TO_GENERIC: dict[str, str] = {
    "ipfs_accelerate_py/agent_supervisor/vfs_surface_inventory.py": (
        "ipfs_accelerate_py/agent_supervisor/analysis/repository_surface_inventory.py"
    ),
    "ipfs_accelerate_py/agent_supervisor/vfs_contract_pack.py": (
        "ipfs_accelerate_py/agent_supervisor/analysis/program_contract_profile.py"
    ),
    "ipfs_accelerate_py/agent_supervisor/vfs_differential_harness.py": (
        "ipfs_accelerate_py/agent_supervisor/validation/differential_contract_harness.py"
    ),
    "ipfs_accelerate_py/agent_supervisor/vfs_mcp_contract_checker.py": (
        "ipfs_accelerate_py/agent_supervisor/analysis/interface_contract_parity.py"
    ),
    "ipfs_accelerate_py/agent_supervisor/vfs_symbolic_benchmark.py": (
        "ipfs_accelerate_py/agent_supervisor/validation/symbolic_efficiency_benchmark.py"
    ),
    "ipfs_accelerate_py/agent_supervisor/vfs_symbolic_pilot.py": (
        "ipfs_accelerate_py/agent_supervisor/runtime/symbolic_assurance_pilot.py"
    ),
    "ipfs_accelerate_py/agent_supervisor/vfs_symbolic_rollout.py": (
        "ipfs_accelerate_py/agent_supervisor/control/symbolic_assurance_rollout.py"
    ),
}

SOURCE_TEST_TO_GENERIC: dict[str, str] = {
    "test/api/test_agent_supervisor_vfs_surface_inventory.py": (
        "test/api/test_agent_supervisor_repository_surface_inventory.py"
    ),
    "test/api/test_agent_supervisor_vfs_contract_pack.py": (
        "test/api/test_agent_supervisor_program_contract_profile.py"
    ),
    "test/api/test_agent_supervisor_vfs_differential_harness.py": (
        "test/api/test_agent_supervisor_differential_contract_harness.py"
    ),
    "test/api/test_agent_supervisor_vfs_mcp_contract_checker.py": (
        "test/api/test_agent_supervisor_interface_contract_parity.py"
    ),
    "test/api/test_agent_supervisor_vfs_symbolic_benchmark.py": (
        "test/api/test_agent_supervisor_symbolic_efficiency_benchmark.py"
    ),
    "test/api/test_agent_supervisor_vfs_symbolic_pilot.py": (
        "test/api/test_agent_supervisor_symbolic_assurance_pilot.py"
    ),
    "test/api/test_vfs_symbolic_assurance_e2e.py": (
        "test/api/test_agent_supervisor_symbolic_assurance_rollout.py"
    ),
}

_DISPOSITION_PROVED = "proved"
_DISPOSITION_UNSUPPORTED = "unsupported"
_DISPOSITION_APPROVAL_REQUIRED = "approval_required"
_DISPOSITION_MIGRATED = "migrated"

_IMPORT_VFS = re.compile(
    r"ipfs_accelerate_py\.agent_supervisor\.vfs_[A-Za-z0-9_]+"
)


@dataclass(frozen=True)
class ClauseDisposition:
    clause_id: str
    subject: str
    disposition: str
    evidence: str
    notes: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "clause_id": self.clause_id,
            "subject": self.subject,
            "disposition": self.disposition,
            "evidence": self.evidence,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class VfsCallerMigrationReceipt:
    """Impact closure over imports, string imports, entry points, and board outputs."""

    schema: str = (
        "ipfs_accelerate_py/agent-supervisor/vfs-caller-migration-receipt@1"
    )
    module_migrations: tuple[dict[str, str], ...] = ()
    test_migrations: tuple[dict[str, str], ...] = ()
    remaining_import_hits: tuple[str, ...] = ()
    open_board_root_outputs: tuple[str, ...] = ()
    dispositions: tuple[ClauseDisposition, ...] = ()

    @property
    def passed(self) -> bool:
        return not self.remaining_import_hits and not self.open_board_root_outputs

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "passed": self.passed,
            "module_migrations": list(self.module_migrations),
            "test_migrations": list(self.test_migrations),
            "remaining_import_hits": list(self.remaining_import_hits),
            "open_board_root_outputs": list(self.open_board_root_outputs),
            "dispositions": [item.to_dict() for item in self.dispositions],
        }


@dataclass(frozen=True)
class VfsGeneralizationEquivalenceReceipt:
    """Locked-source-to-generic equivalence receipt with Tactician/Hammer dispositions."""

    schema: str = (
        "ipfs_accelerate_py/agent-supervisor/vfs-generalization-equivalence-receipt@1"
    )
    source_lock_content_id: str = ""
    source_revision: str = ""
    clauses: tuple[ClauseDisposition, ...] = ()
    caller_migration: VfsCallerMigrationReceipt | None = None
    profile_identity: Mapping[str, str] = field(default_factory=dict)
    schema_parity: Mapping[str, str] = field(default_factory=dict)
    authority_flags: Mapping[str, bool] = field(default_factory=dict)
    content_id: str = ""

    @property
    def passed(self) -> bool:
        if self.caller_migration is not None and not self.caller_migration.passed:
            return False
        blocking = {
            _DISPOSITION_UNSUPPORTED,
            _DISPOSITION_APPROVAL_REQUIRED,
        }
        # Unsupported clauses are allowed only when explicitly recorded and
        # non-blocking (byte-identical source reconstruction is not claimed).
        required_proved = {
            "layout-cutover",
            "generic-engine-presence",
            "profile-identity-parity",
            "schema-parity",
            "authority-flags",
            "ops-delegation",
            "cold-import",
            "caller-import-closure",
        }
        proved_ids = {
            clause.clause_id
            for clause in self.clauses
            if clause.disposition in {_DISPOSITION_PROVED, _DISPOSITION_MIGRATED}
        }
        return required_proved.issubset(proved_ids)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "passed": self.passed,
            "source_lock_content_id": self.source_lock_content_id,
            "source_revision": self.source_revision,
            "clauses": [item.to_dict() for item in self.clauses],
            "caller_migration": (
                self.caller_migration.to_dict() if self.caller_migration else None
            ),
            "profile_identity": dict(self.profile_identity),
            "schema_parity": dict(self.schema_parity),
            "authority_flags": dict(self.authority_flags),
        }
        unsigned = dict(payload)
        unsigned.pop("content_id", None)
        digest = "sha256:" + hashlib.sha256(
            json.dumps(
                unsigned,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        payload["content_id"] = digest
        return payload


def _load_lock() -> dict[str, Any]:
    return json.loads(LOCK_PATH.read_text(encoding="utf-8"))


def _scan_import_hits() -> tuple[str, ...]:
    hits: list[str] = []
    patterns = (
        "ipfs_accelerate_py/agent_supervisor/**/*.py",
        "scripts/ops/agent_supervisor/**/*.py",
        "test/api/test_agent_supervisor_*.py",
        "test/api/test_vfs_*.py",
        "test/api/test_ipfs_kit_vfs_*.py",
    )
    skip = {
        "test/api/test_agent_supervisor_vfs_generalization_equivalence.py",
        "test/api/test_agent_supervisor_vfs_root_layout_guard.py",
    }
    for pattern in patterns:
        for path in REPO_ROOT.glob(pattern):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            rel = path.relative_to(REPO_ROOT).as_posix()
            if rel in skip:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for match in _IMPORT_VFS.finditer(text):
                # Allow documentation of source_path coordinates in locks/maps
                # only when not forming an import statement.
                line_start = text.rfind("\n", 0, match.start()) + 1
                line = text[line_start : text.find("\n", match.start())]
                if re.search(r"^\s*(from|import)\s+", line) or (
                    "importlib" in line or "import_module" in line
                ):
                    line_no = text.count("\n", 0, match.start()) + 1
                    hits.append(f"{rel}:{line_no}:{match.group(0)}")
                elif re.search(
                    r"""['"]ipfs_accelerate_py\.agent_supervisor\.vfs_""",
                    line,
                ) and (
                    "import" in line
                    or "__import__" in line
                    or "import_module" in line
                ):
                    line_no = text.count("\n", 0, match.start()) + 1
                    hits.append(f"{rel}:{line_no}:string:{match.group(0)}")
    return tuple(sorted(set(hits)))


def _open_board_root_outputs() -> tuple[str, ...]:
    """Return open VFS-board outputs that still target root vfs_*.py modules."""

    todo = REPO_ROOT / "docs/architecture/ipfs_kit_vfs_symbolic_assurance.todo.md"
    if not todo.is_file():
        return ("missing:vfs-todo",)
    text = todo.read_text(encoding="utf-8")
    # Split on task headers.
    parts = re.split(r"\n## (VFS-\d+)\s+", text)
    hits: list[str] = []
    for index in range(1, len(parts), 2):
        task_id = parts[index]
        body = parts[index + 1]
        status_match = re.search(r"- Status:\s*(\S+)", body)
        status = status_match.group(1) if status_match else "unknown"
        if status == "completed":
            continue
        for field_name in ("Outputs", "Predicted files", "Validation"):
            field_match = re.search(rf"- {field_name}:\s*(.+)", body)
            if not field_match:
                continue
            value = field_match.group(1)
            if re.search(
                r"agent_supervisor/vfs_[A-Za-z0-9_]+\.py|"
                r"agent_supervisor\.vfs_|"
                r"python -m ipfs_accelerate_py\.agent_supervisor\.vfs_",
                value,
            ):
                hits.append(f"{task_id}:{field_name}:{value[:160]}")
    return tuple(hits)


def _module_export_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    try:
                        value = ast.literal_eval(node.value)
                        if isinstance(value, (list, tuple)):
                            names.update(str(item) for item in value)
                    except Exception:
                        pass
    return names


def compute_caller_migration_receipt(lock: Mapping[str, Any]) -> VfsCallerMigrationReceipt:
    module_migrations: list[dict[str, str]] = []
    for module in lock.get("modules") or []:
        source = str(module.get("source_path") or "")
        planned = str(module.get("planned_path") or SOURCE_TO_GENERIC.get(source, ""))
        module_migrations.append(
            {
                "source_path": source,
                "planned_path": planned,
                "source_blob": str(module.get("source_blob") or ""),
                "present": str((REPO_ROOT / planned).is_file()),
            }
        )
    test_migrations = [
        {"source_test": src, "planned_test": dst, "present": str((REPO_ROOT / dst).is_file())}
        for src, dst in SOURCE_TEST_TO_GENERIC.items()
    ]
    import_hits = _scan_import_hits()
    board_hits = _open_board_root_outputs()
    dispositions = (
        ClauseDisposition(
            clause_id="caller-import-closure",
            subject="agent_supervisor.vfs_* imports",
            disposition=(
                _DISPOSITION_PROVED if not import_hits else _DISPOSITION_APPROVAL_REQUIRED
            ),
            evidence="static-scan",
            notes=f"hits={len(import_hits)}",
        ),
        ClauseDisposition(
            clause_id="open-board-output-migration",
            subject="ipfs_kit_vfs_symbolic_assurance.todo.md",
            disposition=(
                _DISPOSITION_MIGRATED if not board_hits else _DISPOSITION_APPROVAL_REQUIRED
            ),
            evidence="board-scan",
            notes=f"hits={len(board_hits)}",
        ),
    )
    return VfsCallerMigrationReceipt(
        module_migrations=tuple(module_migrations),
        test_migrations=tuple(test_migrations),
        remaining_import_hits=import_hits,
        open_board_root_outputs=board_hits,
        dispositions=dispositions,
    )


def compute_equivalence_receipt() -> VfsGeneralizationEquivalenceReceipt:
    lock = _load_lock()
    clauses: list[ClauseDisposition] = []

    # Layout cutover: no root vfs_*.py.
    root_vfs = sorted(p.name for p in AGENT_SUPERVISOR_ROOT.glob("vfs_*.py"))
    clauses.append(
        ClauseDisposition(
            clause_id="layout-cutover",
            subject="agent_supervisor/vfs_*.py",
            disposition=_DISPOSITION_PROVED if not root_vfs else _DISPOSITION_APPROVAL_REQUIRED,
            evidence="filesystem-scan",
            notes=f"root_vfs={root_vfs}",
        )
    )

    # Generic engine presence + generic symbols seed coverage.
    missing = [
        path
        for path in SOURCE_TO_GENERIC.values()
        if not (REPO_ROOT / path).is_file()
    ]
    clauses.append(
        ClauseDisposition(
            clause_id="generic-engine-presence",
            subject="planned generic engines",
            disposition=_DISPOSITION_PROVED if not missing else _DISPOSITION_APPROVAL_REQUIRED,
            evidence="filesystem-scan",
            notes=f"missing={missing}",
        )
    )

    # Profile identity and schema parity (supported, proved via runtime).
    config = load_assurance_config(CONFIG_PATH)
    profile = config.profile
    identity = {
        "behavior_id": profile.behavior_id,
        "objective_id": profile.objective_id,
        "objective_revision": profile.objective_revision,
        "requirement_id": profile.requirement_id,
    }
    identity_ok = identity == ORIGINAL_IDS
    clauses.append(
        ClauseDisposition(
            clause_id="profile-identity-parity",
            subject="AssuranceRolloutProfile identity",
            disposition=_DISPOSITION_PROVED if identity_ok else _DISPOSITION_APPROVAL_REQUIRED,
            evidence="build_ipfs_kit_vfs_assurance_profile",
            notes=json.dumps(identity, sort_keys=True),
        )
    )

    schema_parity = {
        key: getattr(profile.schemas, key)
        for key in ORIGINAL_SCHEMAS
    }
    schema_ok = schema_parity == ORIGINAL_SCHEMAS
    clauses.append(
        ClauseDisposition(
            clause_id="schema-parity",
            subject="rollout schemas",
            disposition=_DISPOSITION_PROVED if schema_ok else _DISPOSITION_APPROVAL_REQUIRED,
            evidence="profile.schemas",
            notes=json.dumps(schema_parity, sort_keys=True),
        )
    )

    authority = dict(config.authority_flags)
    authority_ok = authority and not any(authority.values())
    clauses.append(
        ClauseDisposition(
            clause_id="authority-flags",
            subject="authority_flags",
            disposition=_DISPOSITION_PROVED if authority_ok else _DISPOSITION_APPROVAL_REQUIRED,
            evidence="config.authority_flags",
            notes=json.dumps(authority, sort_keys=True),
        )
    )

    # Canonical vectors / operations preserved through contracts adapter.
    contracts = run_contracts(config=config)
    vectors_ok = bool(contracts.get("canonical_vectors")) and "read" in contracts.get(
        "operations", []
    )
    clauses.append(
        ClauseDisposition(
            clause_id="canonical-vectors",
            subject="operation/invariant/error/vector mappings",
            disposition=_DISPOSITION_PROVED if vectors_ok else _DISPOSITION_APPROVAL_REQUIRED,
            evidence="run_contracts",
            notes=f"ops={len(contracts.get('operations') or [])}",
        )
    )

    # Ops delegation / thin wrapper.
    ops_source = OPS_CLI.read_text(encoding="utf-8")
    ops_tree = ast.parse(ops_source)
    thin = (
        not any(isinstance(n, ast.ClassDef) for n in ops_tree.body)
        and "argparse" in ops_source
        and ("dispatch" in ops_source or "load_assurance_config" in ops_source)
    )
    clauses.append(
        ClauseDisposition(
            clause_id="ops-delegation",
            subject=str(OPS_CLI.relative_to(REPO_ROOT)),
            disposition=_DISPOSITION_PROVED if thin else _DISPOSITION_APPROVAL_REQUIRED,
            evidence="ast+source",
            notes="thin facade only",
        )
    )

    # Cold import side-effect freedom for integration.
    before = optional_providers_loaded()
    for name in sorted(CLOSED_ADAPTERS):
        lazy_import_adapter(name, config=config)
    after = optional_providers_loaded()
    cold_ok = before == after == ()
    clauses.append(
        ClauseDisposition(
            clause_id="cold-import",
            subject="integrations.ipfs_kit_vfs_assurance",
            disposition=_DISPOSITION_PROVED if cold_ok else _DISPOSITION_APPROVAL_REQUIRED,
            evidence="optional_providers_loaded",
            notes=f"before={before} after={after}",
        )
    )

    # Rollout + verify through identical generic control engine.
    rollout = run_rollout(config=config, desired_mode="assist")
    verified = run_verify(config=config)
    rollout_ok = (
        rollout.get("automatic_mutation_enabled") is False
        and rollout["adversarial_e2e_gate"]["schema"] == ORIGINAL_SCHEMAS["adversarial_e2e_gate"]
        and verified.get("verified") is True
    )
    clauses.append(
        ClauseDisposition(
            clause_id="rollout-verify-parity",
            subject="control.symbolic_assurance_rollout via profile",
            disposition=_DISPOSITION_PROVED if rollout_ok else _DISPOSITION_APPROVAL_REQUIRED,
            evidence="run_rollout+run_verify",
            notes=f"mode={rollout.get('decision', {}).get('effective_mode')}",
        )
    )

    # Tactician/Hammer: full source-blob behavioral reconstruction is
    # unsupported without importing locked blobs as code; record explicitly.
    clauses.append(
        ClauseDisposition(
            clause_id="source-blob-byte-equivalence",
            subject="locked Git blob vs generic module body",
            disposition=_DISPOSITION_UNSUPPORTED,
            evidence="tactician-hammer-abstention",
            notes=(
                "Workers read exact source-lock blobs but must not execute or "
                "merge them; profile-driven public contract parity is the "
                "proved equivalence surface."
            ),
        )
    )
    clauses.append(
        ClauseDisposition(
            clause_id="dynamic-native-public-api-diff",
            subject="unresolved dynamic/native differences",
            disposition=_DISPOSITION_UNSUPPORTED,
            evidence="tactician-hammer-abstention",
            notes="Abstain rather than guess unresolved semantic drift.",
        )
    )

    migration = compute_caller_migration_receipt(lock)
    clauses.extend(migration.dispositions)

    # Export-name migration heuristic: generic modules expose non-empty public API.
    for planned in SOURCE_TO_GENERIC.values():
        path = REPO_ROOT / planned
        if not path.is_file():
            continue
        exports = _module_export_names(path)
        clauses.append(
            ClauseDisposition(
                clause_id=f"export-surface:{Path(planned).stem}",
                subject=planned,
                disposition=_DISPOSITION_PROVED if exports else _DISPOSITION_APPROVAL_REQUIRED,
                evidence="ast-export-scan",
                notes=f"export_count={len(exports)}",
            )
        )

    receipt = VfsGeneralizationEquivalenceReceipt(
        source_lock_content_id=str(lock.get("content_id") or ""),
        source_revision=str((lock.get("source") or {}).get("revision") or ""),
        clauses=tuple(clauses),
        caller_migration=migration,
        profile_identity=identity,
        schema_parity=schema_parity,
        authority_flags={k: bool(v) for k, v in authority.items()},
    )
    # Attach content_id via to_dict for fixed-point checks.
    materialized = receipt.to_dict()
    object.__setattr__(receipt, "content_id", materialized["content_id"])  # type: ignore[misc]
    return receipt


def test_source_lock_pins_seven_modules_and_forbids_broad_merge() -> None:
    lock = _load_lock()
    assert lock["schema"].endswith("vfs-generalization-sources-lock@1")
    assert lock["source"]["revision"] == "0cc04ebb640c4c981cf4650016e096a73ab0e8c0"
    assert lock["source"]["merge_or_cherry_pick_source_revision"] is False
    assert lock["source"]["read_blobs_only"] is True
    modules = lock["modules"]
    assert len(modules) == 7
    for module in modules:
        source = module["source_path"]
        assert source in SOURCE_TO_GENERIC
        planned = SOURCE_TO_GENERIC[source]
        assert (REPO_ROOT / planned).is_file(), planned
        assert module["source_path_state"] == "source_only"
        assert module["merge_or_cherry_pick_source"] is False


def test_generalization_map_documents_cutover() -> None:
    text = MAP_PATH.read_text(encoding="utf-8")
    assert "LPR-028" in text or "cutover" in text.lower() or "Forbidden after cutover" in text
    for planned in SOURCE_TO_GENERIC.values():
        assert Path(planned).name in text or planned in text


def test_locked_profile_preserves_schemas_vectors_and_cli_surface() -> None:
    config = load_assurance_config(CONFIG_PATH)
    profile = build_ipfs_kit_vfs_assurance_profile(CONFIG_PATH)
    assert profile.behavior_id == ORIGINAL_IDS["behavior_id"]
    assert profile.objective_id == ORIGINAL_IDS["objective_id"]
    for key, value in ORIGINAL_SCHEMAS.items():
        assert getattr(profile.schemas, key) == value
    projected = run_contracts(config=config)
    assert "read" in projected["operations"]
    assert "path-normalized" in projected["invariants"]
    assert "not-found" in projected["error_codes"]
    assert any(v["vector_id"] == "read-empty" for v in projected["canonical_vectors"])
    assert set(config.cli_subcommands) == set(CLOSED_ADAPTERS)
    assert not any(config.authority_flags.values())


def test_ops_cli_delegates_rollout_and_verify() -> None:
    env = {
        **dict(__import__("os").environ),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "IPFS_ACCEL_SKIP_CORE": "1",
        "PYTHONPATH": str(REPO_ROOT)
        + (
            ":" + __import__("os").environ["PYTHONPATH"]
            if __import__("os").environ.get("PYTHONPATH")
            else ""
        ),
    }
    verify = subprocess.run(
        [sys.executable, str(OPS_CLI), "verify"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert verify.returncode == 0, verify.stderr
    payload = json.loads(verify.stdout)
    assert payload["verified"] is True
    assert payload["automatic_mutation_enabled"] is False

    contracts = subprocess.run(
        [sys.executable, str(OPS_CLI), "contracts"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert contracts.returncode == 0, contracts.stderr
    body = json.loads(contracts.stdout)
    assert "read" in body["operations"]


def test_caller_migration_receipt_is_closed() -> None:
    lock = _load_lock()
    receipt = compute_caller_migration_receipt(lock)
    assert isinstance(receipt, VfsCallerMigrationReceipt)
    assert len(receipt.module_migrations) == 7
    assert all(item["present"] == "True" for item in receipt.module_migrations)
    assert receipt.remaining_import_hits == (), receipt.remaining_import_hits
    assert receipt.open_board_root_outputs == (), receipt.open_board_root_outputs
    assert receipt.passed


def test_equivalence_receipt_fixed_point() -> None:
    first = compute_equivalence_receipt()
    second = compute_equivalence_receipt()
    assert isinstance(first, VfsGeneralizationEquivalenceReceipt)
    assert first.passed, first.to_dict()
    assert first.content_id == second.content_id
    assert first.content_id.startswith("sha256:")
    # Supported clauses proved; unsupported retained explicitly.
    by_id = {clause.clause_id: clause for clause in first.clauses}
    assert by_id["layout-cutover"].disposition == _DISPOSITION_PROVED
    assert by_id["source-blob-byte-equivalence"].disposition == _DISPOSITION_UNSUPPORTED
    assert by_id["dynamic-native-public-api-diff"].disposition == _DISPOSITION_UNSUPPORTED
    assert by_id["caller-import-closure"].disposition == _DISPOSITION_PROVED
    assert by_id["open-board-output-migration"].disposition == _DISPOSITION_MIGRATED


def test_generic_engines_importable_without_optional_providers() -> None:
    before = optional_providers_loaded()
    modules = [
        "ipfs_accelerate_py.agent_supervisor.analysis.repository_surface_inventory",
        "ipfs_accelerate_py.agent_supervisor.analysis.program_contract_profile",
        "ipfs_accelerate_py.agent_supervisor.analysis.interface_contract_parity",
        "ipfs_accelerate_py.agent_supervisor.validation.differential_contract_harness",
        "ipfs_accelerate_py.agent_supervisor.validation.symbolic_efficiency_benchmark",
        "ipfs_accelerate_py.agent_supervisor.runtime.symbolic_assurance_pilot",
        "ipfs_accelerate_py.agent_supervisor.control.symbolic_assurance_rollout",
        "ipfs_accelerate_py.agent_supervisor.integrations.ipfs_kit_vfs_assurance",
    ]
    for name in modules:
        __import__(name)
    assert optional_providers_loaded() == before
