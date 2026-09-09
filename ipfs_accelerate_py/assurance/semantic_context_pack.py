"""Fail-closed PCPR-061 Accelerate consumer of the Datasets ContextPack.

Accelerate binds the Datasets-owned DatasetsContextPack@1 identity as
SupervisorContextPack and refuses reminting. It does not construct the
pack, does not store bytes (PCPR-062), does not execute the
deterministic route (PCPR-063), and does not write DuckDB or Quack
state.

Sibling Datasets source is observed when present and never required.
Live supervisor admission stays typed unavailable. Simulated results
are not live.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from ipfs_accelerate_py.assurance.dependency_locks import (
    CLOSED_RELEASE_OUTCOMES,
    PACKAGE_NAME,
    PACKAGE_VERSION,
    SEALED_PATH,
    SEALED_PYTHON,
    content_identity,
    discover_accelerate_root,
    observe_sealed_validation_environment,
    pretty_json,
    sha256_bytes,
    typed_unavailable,
)
from ipfs_accelerate_py.assurance.portfolio_compatibility_lock import (
    PINNED_LOCK_CID as PCPR_056_LOCK_CID,
    PORTFOLIO_ID,
    PORTFOLIO_VERSION,
)
from ipfs_accelerate_py.assurance.shared_contracts import (
    OWNER_INTERFACES,
    OWNER_SCHEMAS,
    shared_interface_id,
    shared_schema_id,
)

INTERFACE: Final = "AccelerateSemanticContextPackBinding@1"
SCHEMA: Final = "ipfs_accelerate_py/assurance/semantic-context-pack-binding@1"
BINDING_SCHEMA: Final = (
    "ipfs_accelerate_py/assurance/declared-semantic-context-pack-binding@1"
)
VERDICT_SCHEMA: Final = (
    "ipfs_accelerate_py/assurance/semantic-context-pack-verdict@1"
)
CATALOG_SCHEMA: Final = (
    "ipfs_accelerate_py/assurance/platform-semantic-context-pack-catalog@1"
)
CATALOG_INTERFACE: Final = "PlatformSemanticContextPack@1"
OWNER_INTERFACE: Final = "DatasetsContextPack@1"
OWNER_SCHEMA: Final = "ipfs_datasets_py/datasets-context-pack@1"
OWNER_REPOSITORY: Final = "ipfs_datasets_py"
PCPR_061_TASK_ID: Final = "PCPR-061"
PCPR_061_GOAL_ID: Final = "PCPR-G700"
PCPR_062_TASK_ID: Final = "PCPR-062"
PCPR_063_TASK_ID: Final = "PCPR-063"
PCPR_060_TASK_ID: Final = "PCPR-060"
PCPR_040_TASK_ID: Final = "PCPR-040"
PCPR_001_TASK_ID: Final = "PCPR-001"
PCPR_PROGRAM_ID: Final = "proof-carrying-platform-qualification-and-release-v1"
PCPR_BOARD_NAMESPACE: Final = (
    "proof-carrying-platform-qualification-and-release-v1"
)
OBJECTIVE_KIND: Final = "declared_semantic_context_pack_binding"
OBJECTIVE_ID: Final = "PCPR-G700"
LANGUAGE: Final = "Python"
OPERATOR_BLOCKING_TASK_ID: Final = "pcpr-061-operator-live-context-pack-admission"
BINDING_DIR_RELPATH: Final = "packaging/pcpr/reference-workflow/cpython312"
BINDING_JSON_NAME: Final = "reference.context-pack.json"
BINDING_README_RELPATH: Final = "packaging/pcpr/reference-workflow/CONTEXT_PACK.md"
CATALOG_RELPATH: Final = (
    "packaging/pcpr/reference-workflow/platform-context-pack-catalog.json"
)
SOURCE_DATE_EPOCH: Final = "0"
SOURCE_REPOSITORY: Final = "endomorphosis/ipfs_accelerate_py"

PINNED_IDEA_DIGEST: Final = (
    "baguqeeracbayojdov4jmqiirx22pavrg6nabazcocru6y3scrdx5e54mw2zq"
)
PINNED_OBJECTIVE_CID: Final = (
    "baguqeeraynsn7tjr3iaggnreylzxo3akaooqwp5bf5oheaa6eubth2zwqeba"
)
PINNED_LOCK_CID: Final = PCPR_056_LOCK_CID

if PINNED_LOCK_CID != (
    "baguqeerawsekbbbt5ccctjt4tydzeahfkahsatb6q5x7k6cy4inrii5lhqmq"
):
    raise RuntimeError("PCPR-061 lock CID remints PCPR-056")

if OWNER_INTERFACES["SupervisorContextPack"] != OWNER_INTERFACE:
    raise RuntimeError("PCPR-061 remints SupervisorContextPack owner interface")
if OWNER_SCHEMAS["SupervisorContextPack"] != OWNER_SCHEMA:
    raise RuntimeError("PCPR-061 remints SupervisorContextPack owner schema")

REFERENCE_OBJECTIVE_IDEA: Final = (
    "Modify a typed formal-logic API while reusing unaffected proofs, "
    "selecting only impacted tests, rejecting stale-tree evidence, and "
    "producing a complete proof-carrying execution receipt."
)

HERMETIC_CANDIDATE_SUITES: Final[tuple[str, ...]] = (
    "test/api/test_agent_supervisor_semantic_context_pack.py",
)

EVIDENCE_KINDS: Final[frozenset[str]] = frozenset(
    {
        "measured",
        "measured_live",
        "measured_hermetic",
        "estimated",
        "simulated",
        "unavailable",
    }
)

REQUIRED_GOOD_PROBE_IDS: Final[frozenset[str]] = frozenset(
    {
        "binding_files_match_generator",
        "pyproject_semantic_context_pack_table",
        "owner_pack_cid_matches_pin",
        "owner_identity_not_reminted",
        "supervisor_context_pack_bound_not_minted",
        "construction_owned_by_datasets",
        "storage_deferred_to_pcpr_062",
        "execution_deferred_to_pcpr_063",
        "duckdb_or_quack_not_written",
        "operator_blocking_task_emitted",
        "no_closed_release_outcome",
    }
)
FORBIDDEN_PRESENT_PROBE_IDS: Final[frozenset[str]] = frozenset(
    {
        "simulated_results_represented_as_live",
        "live_context_pack_admission_represented_as_live",
        "closed_release_represented_as_live",
        "compatibility_identities_reminted",
        "direct_database_bypass_used",
        "current_root_published",
        "datasets_identity_reminted",
    }
)

BINDING_README: Final = """# PCPR-061 Accelerate semantic ContextPack binding

These files bind Accelerate to the Datasets-owned DatasetsContextPack@1
identity for the PCPR-060 reference idea. Accelerate is a consumer. It
does not remint the owner pack CID, does not store bytes, and does not
execute the deterministic-first route.

- `cpython312/reference.context-pack.json` binds SupervisorContextPack
  to the Datasets owner identity. Exact commit and tree are bound by
  the PCPR-061 receipt `current_tree_binding`.
- `platform-context-pack-catalog.json` observes sibling Datasets and
  Kit pack documents when present. Sibling source is never required.
- Durable storage remains PCPR-062. Deterministic execution remains
  PCPR-063.
- Missing a live Quack-fenced session emits operator-blocking task
  `pcpr-061-operator-live-context-pack-admission`.

Sealed validation PATH is exactly `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin`.
"""


class AccelerateSemanticContextPackError(Exception):
    """Fail-closed PCPR-061 Accelerate ContextPack binding error."""


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AccelerateSemanticContextPackError(f"{name} must be a non-empty string")
    return value.strip()


def _kind(value: Any, name: str) -> str:
    kind = _text(value, name)
    if kind not in EVIDENCE_KINDS:
        raise AccelerateSemanticContextPackError(
            f"{name} is not an admitted evidence kind"
        )
    return kind


def _reject_closed_release_value(value: Any, name: str) -> None:
    if isinstance(value, str) and value in CLOSED_RELEASE_OUTCOMES:
        raise AccelerateSemanticContextPackError(
            f"{name} must not be a closed PCPR release outcome"
        )


def _atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    tmp.replace(path)


def parse_pyproject_context_pack_table(text: str) -> dict[str, Any]:
    import tomllib

    table = tomllib.loads(_text(text, "pyproject.toml"))
    if not isinstance(table, dict):
        raise AccelerateSemanticContextPackError("pyproject.toml must be a table")
    tool = table.get("tool")
    payload: dict[str, Any] = {}
    if isinstance(tool, dict):
        accelerate = tool.get("ipfs_accelerate_py")
        if isinstance(accelerate, dict):
            raw = accelerate.get("semantic-context-pack")
            if isinstance(raw, dict):
                payload = dict(raw)
    return payload


@dataclass(frozen=True)
class OutcomeProbe:
    probe_id: str
    present: bool | None
    evidence_kind: str
    live: bool
    simulated_represented_as_live: bool
    reason: str
    details: Mapping[str, Any] = MappingProxyType({})

    def to_mapping(self) -> dict[str, Any]:
        return {
            "probe_id": self.probe_id,
            "present": self.present,
            "evidence_kind": self.evidence_kind,
            "live": self.live,
            "simulated_represented_as_live": self.simulated_represented_as_live,
            "reason": self.reason,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class AccelerateSemanticContextPackVerdict:
    schema: str
    interface: str
    promotion_status: str
    supervisor_disposition: str
    closed_release_outcome: str | None
    release_claim: bool
    completion_authoritative: bool
    contracts_frozen: bool
    duckdb_or_quack_state_written: bool
    sibling_source_required: bool
    live_context_pack_admission: bool
    live_context_pack_admission_evidence_kind: str
    live_storage: bool
    live_execution: bool
    operator_blocking_task: str
    simulated_results_represented_as_live: bool
    this_task_created_competing_authority: bool
    probes: tuple[OutcomeProbe, ...]
    blockers: tuple[str, ...]
    verdict_cid: str
    pack_cid: str
    binding_cid: str
    catalog_cid: str
    objective_cid: str
    idea_digest: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.interface,
            "verdict_cid": self.verdict_cid,
            "pack_cid": self.pack_cid,
            "binding_cid": self.binding_cid,
            "catalog_cid": self.catalog_cid,
            "objective_cid": self.objective_cid,
            "idea_digest": self.idea_digest,
            "promotion_status": self.promotion_status,
            "supervisor_disposition": self.supervisor_disposition,
            "closed_release_outcome": self.closed_release_outcome,
            "release_claim": self.release_claim,
            "completion_authoritative": self.completion_authoritative,
            "contracts_frozen": self.contracts_frozen,
            "duckdb_or_quack_state_written": self.duckdb_or_quack_state_written,
            "sibling_source_required": self.sibling_source_required,
            "live_context_pack_admission": self.live_context_pack_admission,
            "live_context_pack_admission_evidence_kind": (
                self.live_context_pack_admission_evidence_kind
            ),
            "live_storage": self.live_storage,
            "live_execution": self.live_execution,
            "operator_blocking_task": self.operator_blocking_task,
            "simulated_results_represented_as_live": (
                self.simulated_results_represented_as_live
            ),
            "this_task_created_competing_authority": (
                self.this_task_created_competing_authority
            ),
            "blocker_count": len(self.blockers),
            "blockers": list(self.blockers),
            "evidence_kind": "measured",
        }


def _probe(
    probe_id: str,
    present: bool | None,
    *,
    reason: str,
    evidence_kind: str = "measured",
    live: bool = False,
    details: Mapping[str, Any] | None = None,
) -> OutcomeProbe:
    return OutcomeProbe(
        probe_id=probe_id,
        present=present,
        evidence_kind=evidence_kind,
        live=live,
        simulated_represented_as_live=False,
        reason=reason,
        details=MappingProxyType(dict(details or {})),
    )


def _operator_blocking_task() -> dict[str, Any]:
    return {
        "task_id": OPERATOR_BLOCKING_TASK_ID,
        "status": "typed_blocked",
        "evidence_kind": "unavailable",
        "live": False,
        "applied": False,
        "requires": (
            "An admitted Quack-fenced state-owner session before Supervisor "
            "admits the Datasets-owned ContextPack into live task state"
        ),
        "action": (
            "Bind the Datasets pack CID. Do not remint it. Do not write "
            "DuckDB or Quack state. Keep storage as PCPR-062 and execution "
            "as PCPR-063."
        ),
        "reason": (
            "Accelerate consumption of DatasetsContextPack@1 is not live "
            "supervisor admission. Direct DuckDB writes are prohibited."
        ),
    }


def observe_state_owner() -> dict[str, Any]:
    env = observe_sealed_validation_environment()
    duckdb_module = "unavailable"
    try:
        import duckdb as duckdb_mod

        origin = str(getattr(duckdb_mod, "__file__", "") or "")
        if origin and "/home/" not in origin and ".local" not in origin:
            duckdb_module = origin
    except Exception:
        duckdb_module = "unavailable"
    return {
        **env,
        "duckdb_module": duckdb_module,
        "duckdb_module_is_not_live_materialization": True,
        "quack": "unavailable",
        "quack_fenced_session": typed_unavailable(
            reason=(
                "No admitted Quack-fenced state-owner session was observed. "
                "Direct DuckDB writes are prohibited."
            )
        ),
        "duckdb_or_quack_state_written": False,
        "direct_duckdb_write_prohibited": True,
        "direct_quack_write_prohibited": True,
        "evidence_kind": "measured",
    }


def _sibling_pack(path: Path, relative_path: str) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": relative_path,
            "status": "unavailable",
            "evidence_kind": "unavailable",
            "reason": "Sibling ContextPack file is not present beside this checkout.",
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    pack = payload.get("context_pack")
    pack_cid = None
    if isinstance(pack, Mapping):
        pack_cid = pack.get("pack_cid")
    elif isinstance(payload.get("pack_cid"), str):
        pack_cid = payload.get("pack_cid")
    return {
        "path": relative_path,
        "status": "observed",
        "evidence_kind": "measured",
        "package_name": payload.get("package_name"),
        "package_version": payload.get("package_version"),
        "objective_kind": payload.get("objective_kind"),
        "objective_cid": payload.get("objective_cid"),
        "idea_digest": payload.get("idea_digest"),
        "pack_cid": pack_cid,
        "pack_document_cid": payload.get("pack_document_cid")
        or payload.get("binding_cid"),
        "live": False,
        "applied": False,
        "sha256": sha256_bytes(path.read_bytes()),
    }


def render_declared_binding(root: Path | None = None) -> dict[str, Any]:
    package_root = root or discover_accelerate_root()
    if package_root is None:
        raise AccelerateSemanticContextPackError(
            "Accelerate package root was not found"
        )
    state_owner = observe_state_owner()
    document = {
        "schema": BINDING_SCHEMA,
        "interface": INTERFACE,
        "task_id": PCPR_061_TASK_ID,
        "goal_id": PCPR_061_GOAL_ID,
        "program_id": PCPR_PROGRAM_ID,
        "board_namespace": PCPR_BOARD_NAMESPACE,
        "portfolio_id": PORTFOLIO_ID,
        "portfolio_version": PORTFOLIO_VERSION,
        "objective_kind": OBJECTIVE_KIND,
        "package_name": PACKAGE_NAME,
        "package_version": PACKAGE_VERSION,
        "language": LANGUAGE,
        "objective_id": OBJECTIVE_ID,
        "idea": REFERENCE_OBJECTIVE_IDEA,
        "idea_digest": PINNED_IDEA_DIGEST,
        "objective_cid": PINNED_OBJECTIVE_CID,
        "lock_cid": PINNED_LOCK_CID,
        "owner_repository": OWNER_REPOSITORY,
        "owner_interface": OWNER_INTERFACE,
        "owner_schema": OWNER_SCHEMA,
        "shared_contract": {
            "name": "SupervisorContextPack",
            "interface": shared_interface_id("SupervisorContextPack"),
            "schema": shared_schema_id("SupervisorContextPack"),
            "owner_interface": OWNER_INTERFACE,
            "owner_schema": OWNER_SCHEMA,
            "owner_repository": OWNER_REPOSITORY,
            "reminted": False,
        },
        "context_pack": {
            "task_id": PCPR_061_TASK_ID,
            "constructed_by": OWNER_REPOSITORY,
            "constructed": True,
            "consumer": PACKAGE_NAME,
            "pack_cid": PINNED_PACK_CID,
            "owner_interface": OWNER_INTERFACE,
            "owner_schema": OWNER_SCHEMA,
            "reminted": False,
            "live": False,
            "admitted_live": False,
            "stored": False,
            "current_root_published": False,
            "evidence_kind": "measured",
        },
        "storage": {
            "task_id": PCPR_062_TASK_ID,
            "stored": False,
            "current_root_published": False,
            "live": False,
            "deferred": True,
        },
        "execution": {
            "task_id": PCPR_063_TASK_ID,
            "performed": False,
            "live": False,
            "deferred": True,
        },
        "materialization": {
            "kind": "declared_binding_not_live",
            "admitted": False,
            "live": False,
            "applied": False,
            "duckdb_or_quack_state_written": False,
            "evidence_kind": "unavailable",
        },
        "live_context_pack_admission": typed_unavailable(
            reason=(
                "No admitted Quack-fenced supervisor session was observed. "
                "Binding the owner pack CID is not live admission."
            )
        ),
        "state_owner": {
            "duckdb_cli": state_owner.get("duckdb"),
            "duckdb_module": state_owner.get("duckdb_module"),
            "duckdb_module_is_not_live_materialization": True,
            "quack": "unavailable",
            "duckdb_or_quack_state_written": False,
            "direct_duckdb_write_prohibited": True,
            "direct_quack_write_prohibited": True,
        },
        "source": {
            "kind": "git",
            "repository": SOURCE_REPOSITORY,
            "binding": "current_head_not_mutable_main",
            "mutable_main_reference": False,
            "commit": {
                "status": "observed_at_evaluation",
                "evidence_kind": "measured",
                "live": False,
                "field": "PCPR-061 receipt current_tree_binding",
            },
        },
        "operator_blocking_task": _operator_blocking_task(),
        "live": False,
        "applied": False,
        "submitted_live": False,
        "release_claim": False,
        "closed_release_outcome": None,
        "contracts_frozen": False,
        "hashes_invented": False,
        "signatures_invented": False,
        "sibling_source_required": False,
        "this_task_created_competing_authority": False,
        "duckdb_or_quack_state_written": False,
        "source_date_epoch": SOURCE_DATE_EPOCH,
        "evidence_kind": "measured",
    }
    document["binding_cid"] = content_identity(
        {key: value for key, value in document.items() if key != "binding_cid"}
    )
    return document


def artifact_paths(root: Path) -> dict[str, Path]:
    return {
        "binding": root / BINDING_DIR_RELPATH / BINDING_JSON_NAME,
        "readme": root / BINDING_README_RELPATH,
        "catalog": root / CATALOG_RELPATH,
    }


def platform_context_pack_catalog(start: Path | None = None) -> dict[str, Any]:
    root = discover_accelerate_root(start)
    if root is None:
        raise AccelerateSemanticContextPackError(
            "Accelerate package root was not found"
        )
    parent = root.parent
    binding = render_declared_binding(root)
    datasets_pack = _sibling_pack(
        parent / "ipfs_datasets" / BINDING_DIR_RELPATH / "reference.context-pack.json",
        relative_path=(
            f"../ipfs_datasets/{BINDING_DIR_RELPATH}/reference.context-pack.json"
        ),
    )
    kit_pack = _sibling_pack(
        parent
        / "ipfs_kit"
        / BINDING_DIR_RELPATH
        / "reference.context-pack.binding.json",
        relative_path=(
            f"../ipfs_kit/{BINDING_DIR_RELPATH}/reference.context-pack.binding.json"
        ),
    )
    catalog = {
        "schema": CATALOG_SCHEMA,
        "interface": CATALOG_INTERFACE,
        "task_id": PCPR_061_TASK_ID,
        "goal_id": PCPR_061_GOAL_ID,
        "objective_kind": OBJECTIVE_KIND,
        "portfolio_id": PORTFOLIO_ID,
        "portfolio_version": PORTFOLIO_VERSION,
        "objective_cid": PINNED_OBJECTIVE_CID,
        "idea_digest": PINNED_IDEA_DIGEST,
        "pack_cid": PINNED_PACK_CID,
        "lock_cid": PINNED_LOCK_CID,
        "components": {
            "ipfs_accelerate_py": {
                "binding_path": f"{BINDING_DIR_RELPATH}/{BINDING_JSON_NAME}",
                "status": "observed",
                "evidence_kind": "measured",
                "package_name": PACKAGE_NAME,
                "package_version": PACKAGE_VERSION,
                "objective_kind": OBJECTIVE_KIND,
                "objective_cid": PINNED_OBJECTIVE_CID,
                "idea_digest": PINNED_IDEA_DIGEST,
                "pack_cid": PINNED_PACK_CID,
                "binding_cid": binding["binding_cid"],
                "live": False,
                "applied": False,
                "reminted": False,
            },
            "ipfs_datasets_py": {"pack": datasets_pack},
            "ipfs_kit_py": {"binding": kit_pack},
        },
        "live_context_pack_admission": False,
        "live_storage": False,
        "live_execution": False,
        "closed_release_outcome": None,
        "release_claim": False,
        "sibling_source_required": False,
        "evidence_kind": "measured",
    }
    catalog["catalog_cid"] = content_identity(
        {key: value for key, value in catalog.items() if key != "catalog_cid"}
    )
    return catalog


def write_semantic_context_pack_files(start: Path | None = None) -> dict[str, Any]:
    root = discover_accelerate_root(start)
    if root is None:
        raise AccelerateSemanticContextPackError(
            "Accelerate package root was not found"
        )
    binding = render_declared_binding(root)
    paths = artifact_paths(root)
    _atomic_write(paths["binding"], pretty_json(binding))
    readme = BINDING_README if BINDING_README.endswith("\n") else BINDING_README + "\n"
    _atomic_write(paths["readme"], readme)
    catalog = platform_context_pack_catalog(start)
    _atomic_write(paths["catalog"], pretty_json(catalog))
    return {
        "binding": binding,
        "catalog": catalog,
        "paths": {name: str(path) for name, path in paths.items()},
    }


def verify_semantic_context_pack_files(start: Path | None = None) -> dict[str, Any]:
    root = discover_accelerate_root(start)
    if root is None:
        raise AccelerateSemanticContextPackError(
            "Accelerate package root was not found"
        )
    binding = render_declared_binding(root)
    paths = artifact_paths(root)
    missing: list[str] = []
    binding_ok = False
    readme_ok = False
    expected_readme = (
        BINDING_README if BINDING_README.endswith("\n") else BINDING_README + "\n"
    )
    for name, path in paths.items():
        if name == "catalog":
            continue
        if not path.is_file():
            missing.append(name)
            continue
        if name == "binding":
            binding_ok = json.loads(path.read_text(encoding="utf-8")) == binding
        elif name == "readme":
            readme_ok = path.read_text(encoding="utf-8") == expected_readme
    return {
        "ok": not missing and binding_ok and readme_ok,
        "missing": missing,
        "binding_ok": binding_ok,
        "readme_ok": readme_ok,
        "pack_cid": binding["context_pack"]["pack_cid"],
        "binding_cid": binding["binding_cid"],
        "idea_digest": binding["idea_digest"],
        "binding_sha256": (
            sha256_bytes(paths["binding"].read_bytes())
            if paths["binding"].is_file()
            else "unavailable"
        ),
    }


def current_head_static_probes(
    start: Path | None = None,
) -> tuple[OutcomeProbe, ...]:
    root = discover_accelerate_root(start)
    if root is None:
        raise AccelerateSemanticContextPackError(
            "Accelerate package root was not found"
        )
    binding = render_declared_binding(root)
    verified = verify_semantic_context_pack_files(root)
    table = parse_pyproject_context_pack_table(
        (root / "pyproject.toml").read_text(encoding="utf-8")
    )
    remint = (
        binding["context_pack"]["pack_cid"] != PINNED_PACK_CID
        or binding["objective_cid"] != PINNED_OBJECTIVE_CID
        or binding["idea_digest"] != PINNED_IDEA_DIGEST
        or binding["lock_cid"] != PINNED_LOCK_CID
        or binding["context_pack"]["reminted"] is True
    )
    probes = [
        _probe(
            "binding_files_match_generator",
            verified.get("ok") is True and verified.get("binding_ok") is True,
            reason=(
                "Committed Accelerate ContextPack binding matches the generator."
                if verified.get("binding_ok") is True
                else "Committed Accelerate ContextPack binding is missing or drifts."
            ),
        ),
        _probe(
            "pyproject_semantic_context_pack_table",
            table.get("interface") == INTERFACE
            and table.get("schema") == SCHEMA
            and table.get("task-id") == PCPR_061_TASK_ID
            and table.get("objective-kind") == OBJECTIVE_KIND,
            reason=(
                "pyproject.toml declares AccelerateSemanticContextPackBinding@1."
                if table.get("interface") == INTERFACE
                else "pyproject.toml does not declare the PCPR-061 binding."
            ),
        ),
        _probe(
            "owner_pack_cid_matches_pin",
            binding["context_pack"]["pack_cid"] == PINNED_PACK_CID,
            reason="Accelerate binds the Datasets-owned pack CID without remint.",
        ),
        _probe(
            "owner_identity_not_reminted",
            binding["owner_interface"] == OWNER_INTERFACE
            and binding["owner_schema"] == OWNER_SCHEMA
            and binding["shared_contract"]["reminted"] is False,
            reason="SupervisorContextPack remains a binding of DatasetsContextPack@1.",
        ),
        _probe(
            "supervisor_context_pack_bound_not_minted",
            binding["shared_contract"]["name"] == "SupervisorContextPack"
            and binding["context_pack"]["constructed_by"] == OWNER_REPOSITORY
            and binding["context_pack"]["consumer"] == PACKAGE_NAME,
            reason="Accelerate consumes the owner pack and does not mint a sibling identity.",
        ),
        _probe(
            "construction_owned_by_datasets",
            binding["context_pack"]["constructed_by"] == OWNER_REPOSITORY
            and binding["context_pack"]["constructed"] is True
            and binding["context_pack"]["live"] is False,
            reason="Datasets owns construction. Accelerate does not reconstruct the pack.",
        ),
        _probe(
            "storage_deferred_to_pcpr_062",
            binding["storage"]["task_id"] == PCPR_062_TASK_ID
            and binding["storage"]["stored"] is False,
            reason="Durable storage remains PCPR-062.",
        ),
        _probe(
            "execution_deferred_to_pcpr_063",
            binding["execution"]["task_id"] == PCPR_063_TASK_ID
            and binding["execution"]["performed"] is False,
            reason="Deterministic-first execution remains PCPR-063.",
        ),
        _probe(
            "current_root_published",
            False,
            reason="This task must not publish a current ContextPack root.",
        ),
        _probe(
            "duckdb_or_quack_not_written",
            binding["duckdb_or_quack_state_written"] is False
            and binding["state_owner"]["direct_duckdb_write_prohibited"] is True,
            reason="This task does not write DuckDB or Quack state.",
        ),
        _probe(
            "operator_blocking_task_emitted",
            binding["operator_blocking_task"]["task_id"] == OPERATOR_BLOCKING_TASK_ID
            and binding["operator_blocking_task"]["status"] == "typed_blocked",
            reason="Missing live admission emits the operator-blocking task.",
        ),
        _probe(
            "no_closed_release_outcome",
            binding["closed_release_outcome"] is None
            and binding["release_claim"] is False,
            reason="This task does not emit a closed PCPR release outcome.",
        ),
        _probe(
            "compatibility_identities_reminted",
            remint,
            reason="A reminted pack, objective, idea, or lock CID is forbidden.",
        ),
        _probe(
            "datasets_identity_reminted",
            binding["context_pack"]["reminted"] is True,
            reason="Accelerate must not remint DatasetsContextPack@1.",
        ),
        _probe(
            "direct_database_bypass_used",
            False,
            reason="Direct DuckDB or Quack writes were not used.",
        ),
        _probe(
            "simulated_results_represented_as_live",
            False,
            reason="Simulated results are not represented as live.",
        ),
        _probe(
            "live_context_pack_admission_represented_as_live",
            False,
            reason="Owner-CID binding is not live supervisor admission.",
        ),
        _probe(
            "closed_release_represented_as_live",
            False,
            reason="This task does not publish a PCPR release.",
        ),
        _probe(
            "live_context_pack_admission",
            None,
            evidence_kind="unavailable",
            reason="Live supervisor ContextPack admission was not performed.",
        ),
    ]
    return tuple(probes)


def qualify_semantic_context_pack(
    probes: Sequence[OutcomeProbe],
    *,
    pack_cid: str,
    binding_cid: str,
    catalog_cid: str,
    objective_cid: str,
    idea_digest_cid: str,
) -> AccelerateSemanticContextPackVerdict:
    if not probes:
        raise AccelerateSemanticContextPackError("at least one probe is required")
    normalized: list[OutcomeProbe] = []
    blockers: list[str] = []
    for probe in probes:
        kind = _kind(probe.evidence_kind, "evidence_kind")
        if probe.live and kind != "measured_live":
            raise AccelerateSemanticContextPackError(
                "live claims require measured_live evidence"
            )
        if probe.simulated_represented_as_live:
            raise AccelerateSemanticContextPackError(
                "simulated results must not be represented as live"
            )
        normalized.append(probe)
        if probe.probe_id in FORBIDDEN_PRESENT_PROBE_IDS and probe.present is True:
            blockers.append(probe.probe_id)
        if probe.probe_id in REQUIRED_GOOD_PROBE_IDS and probe.present is not True:
            blockers.append(probe.probe_id)

    promotion_status = "rnd_non_promoted"
    _reject_closed_release_value(promotion_status, "promotion_status")
    payload = {
        "schema": VERDICT_SCHEMA,
        "interface": INTERFACE,
        "task_id": PCPR_061_TASK_ID,
        "goal_id": PCPR_061_GOAL_ID,
        "promotion_status": promotion_status,
        "supervisor_disposition": "supervisor_non_promoted",
        "closed_release_outcome": None,
        "release_claim": False,
        "completion_authoritative": False,
        "contracts_frozen": False,
        "duckdb_or_quack_state_written": False,
        "sibling_source_required": False,
        "live_context_pack_admission": False,
        "live_context_pack_admission_evidence_kind": "unavailable",
        "live_storage": False,
        "live_execution": False,
        "operator_blocking_task": OPERATOR_BLOCKING_TASK_ID,
        "simulated_results_represented_as_live": False,
        "this_task_created_competing_authority": False,
        "probes": [item.to_mapping() for item in normalized],
        "blockers": list(dict.fromkeys(blockers)),
        "pack_cid": pack_cid,
        "binding_cid": binding_cid,
        "catalog_cid": catalog_cid,
        "objective_cid": objective_cid,
        "idea_digest": idea_digest_cid,
    }
    return AccelerateSemanticContextPackVerdict(
        schema=VERDICT_SCHEMA,
        interface=INTERFACE,
        promotion_status=promotion_status,
        supervisor_disposition="supervisor_non_promoted",
        closed_release_outcome=None,
        release_claim=False,
        completion_authoritative=False,
        contracts_frozen=False,
        duckdb_or_quack_state_written=False,
        sibling_source_required=False,
        live_context_pack_admission=False,
        live_context_pack_admission_evidence_kind="unavailable",
        live_storage=False,
        live_execution=False,
        operator_blocking_task=OPERATOR_BLOCKING_TASK_ID,
        simulated_results_represented_as_live=False,
        this_task_created_competing_authority=False,
        probes=tuple(normalized),
        blockers=tuple(dict.fromkeys(blockers)),
        verdict_cid=content_identity(payload),
        pack_cid=pack_cid,
        binding_cid=binding_cid,
        catalog_cid=catalog_cid,
        objective_cid=objective_cid,
        idea_digest=idea_digest_cid,
    )


def qualify_current_head_semantic_context_pack(
    start: Path | None = None,
) -> AccelerateSemanticContextPackVerdict:
    binding = render_declared_binding(start)
    catalog = platform_context_pack_catalog(start)
    return qualify_semantic_context_pack(
        current_head_static_probes(start),
        pack_cid=str(binding["context_pack"]["pack_cid"]),
        binding_cid=str(binding["binding_cid"]),
        catalog_cid=str(catalog["catalog_cid"]),
        objective_cid=str(binding["objective_cid"]),
        idea_digest_cid=str(binding["idea_digest"]),
    )


def pcpr_061_receipt_promotion(
    verdict: AccelerateSemanticContextPackVerdict,
) -> dict[str, Any]:
    if verdict.closed_release_outcome is not None:
        raise AccelerateSemanticContextPackError(
            "semantic ContextPack binding must not mint a closed release outcome"
        )
    if verdict.release_claim:
        raise AccelerateSemanticContextPackError(
            "semantic ContextPack binding must not claim a PCPR release"
        )
    if verdict.completion_authoritative:
        raise AccelerateSemanticContextPackError(
            "semantic ContextPack completion is not authoritative"
        )
    if verdict.duckdb_or_quack_state_written:
        raise AccelerateSemanticContextPackError(
            "semantic ContextPack binding must not write DuckDB or Quack state"
        )
    if verdict.promotion_status in CLOSED_RELEASE_OUTCOMES:
        raise AccelerateSemanticContextPackError(
            "promotion_status must not be a closed release outcome"
        )
    if verdict.live_context_pack_admission:
        raise AccelerateSemanticContextPackError(
            "live ContextPack admission requires measured_live evidence"
        )
    if verdict.live_storage or verdict.live_execution:
        raise AccelerateSemanticContextPackError(
            "storage is PCPR-062 and execution is PCPR-063"
        )
    return verdict.to_mapping()


def refuse_pack_cid_remint(cid: str) -> str:
    if cid != PINNED_PACK_CID:
        raise AccelerateSemanticContextPackError(
            f"ContextPack CID {cid} remints {PINNED_PACK_CID}"
        )
    return cid


# Pinned after Datasets encoder measurement. Drift is a remint.
PINNED_PACK_CID: Final = (
    "bafkreih72d3nncekez43wmtlczq5mdtymzniluujypwybpgu3mtt7i4v2e"
)
PINNED_BINDING_CID: Final = (
    "baguqeeraqus4od6vgxrlrgqjxp5k37jc6u22gwrnpyg2qmswiaxr2lva4wga"
)
PINNED_CATALOG_CID: Final = (
    "baguqeerazpi4ztggkhyigjr6jpktppzwmwoxvytllqgmyqutfy6gvbrdj6za"
)
CURRENT_HEAD_NON_PROMOTION_VERDICT_CID: Final = (
    "baguqeeraubcaklbfjb27kaz5amu4rihpuurauhfdi4m7hv5deqpircqxm6ra"
)


__all__ = [
    "AccelerateSemanticContextPackError",
    "AccelerateSemanticContextPackVerdict",
    "CLOSED_RELEASE_OUTCOMES",
    "CURRENT_HEAD_NON_PROMOTION_VERDICT_CID",
    "HERMETIC_CANDIDATE_SUITES",
    "INTERFACE",
    "OBJECTIVE_KIND",
    "OPERATOR_BLOCKING_TASK_ID",
    "OutcomeProbe",
    "PCPR_061_GOAL_ID",
    "PCPR_061_TASK_ID",
    "PINNED_BINDING_CID",
    "PINNED_CATALOG_CID",
    "PINNED_IDEA_DIGEST",
    "PINNED_OBJECTIVE_CID",
    "PINNED_PACK_CID",
    "SCHEMA",
    "SEALED_PATH",
    "SEALED_PYTHON",
    "current_head_static_probes",
    "pcpr_061_receipt_promotion",
    "platform_context_pack_catalog",
    "qualify_current_head_semantic_context_pack",
    "qualify_semantic_context_pack",
    "refuse_pack_cid_remint",
    "render_declared_binding",
    "verify_semantic_context_pack_files",
    "write_semantic_context_pack_files",
]
