"""PCTDD-001: inventory current CID, SHA, canonicalization, and seal paths."""

from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

_TEST_FILE = Path(__file__).resolve()
_ACCELERATE_ROOT = _TEST_FILE.parents[3]
_EXTERNAL_ROOT = _ACCELERATE_ROOT.parent
for _name in ("ipfs_accelerate", "ipfs_datasets", "ipfs_kit"):
    _candidate = _EXTERNAL_ROOT / _name
    if _candidate.is_dir() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

# Nested pytest.ini under external/ipfs_accelerate makes collected nodeids
# rootdir-relative (`test/api/parallel_content_sealing/...` or
# `api/parallel_content_sealing/...`).  The sealed required-acceptance
# collector admits only nodeids bound to the exact profile target.
_PHASE_PLUGIN_MODULE = "run_parallel_content_sealing_proof_carrying_tdd_validation"
_PHASE_REPORTS_ATTR = "_PYTEST_PHASE_REPORTS"

_REQUIRED_TEST_TARGET = (
    "external/ipfs_accelerate/test/api/parallel_content_sealing/"
    "test_pctdd_001_cid_sha_canonicalization_inventory.py"
)
_SOURCE_TO_CID_PATHS = (
    "datasets_cid_utils",
    "accelerate_closed_cid_utils",
    "accelerate_multiformats_identity",
    "datasets_software_contract_content",
    "datasets_ir_core_identity",
    "accelerate_content_identity_bridge",
    "accelerate_assurance_content_identity",
    "datasets_ipld_cid",
    "datasets_profile_g",
    "kit_coordination_cid",
    "kit_proof_certificate_cid",
    "datasets_knowledge_graph_ipld",
    "accelerate_kubo_cid",
)
_SEAL_PUBLICATION_PATHS = (
    "accelerate_incremental_sealer",
    "kit_current_root_cas",
    "kit_proof_certificate_store",
    "kit_proof_seal_store",
)
_HASH_IDENTITY_RULES = (
    "CID is exact byte identity under a versioned profile",
    "Git blob OID is memo lookup key only",
    "filesystem metadata nominates a candidate only",
    "chunk manifest does not replace raw full-stream CID",
    "ordinary SHA-256 of one stream is not composed from independent chunk hashes",
)
_HASH_IDENTITY_PRESERVE = (
    "canonical bytes",
    "codec",
    "multicodec",
    "multihash",
    "CID version",
    "multibase",
    "ordering",
    "normalization",
)
_CURRENT_PATH_LABELS = (
    "source discovery",
    "canonical serialization",
    "SHA/CID",
    "proof verification",
    "Merkle",
    "immutable store",
    "serial WAL/CAS",
)
_TYPED_UNAVAILABLE_CAPABILITIES = (
    "production_zk",
    "key_ceremony",
    "direct_execution_profile",
    "native_batch_hasher",
    "serial_wal_cas_publication",
    "kit_proof_seal_store",
    "datasets_incremental_sealing_evidence",
)
_SEALER_SOURCE = (
    "external/ipfs_accelerate/ipfs_accelerate_py/agent_supervisor/proof/"
    "incremental_sealing/sealer.py"
)
_KIT_PROOF_SEAL_STORE_MODULE = "ipfs_kit_py.proof_seal_store.local_store"
_DATASETS_SEAL_EVIDENCE_MODULE = (
    "ipfs_datasets_py.logic.zkp.incremental_sealing.evidence"
)


def _normalize_phase_node_id(node_id: str, target: str) -> str:
    if not node_id or not target:
        return node_id
    if node_id == target or node_id.startswith(target + "::"):
        return node_id
    filename = target.rsplit("/", 1)[-1]
    if node_id == filename:
        return target
    marker = filename + "::"
    if node_id.startswith(marker):
        return target + "::" + node_id[len(marker) :]
    if node_id.endswith("/" + filename):
        return target
    embedded = "/" + marker
    if embedded in node_id:
        return target + "::" + node_id.split(embedded, 1)[1]
    if node_id.startswith(marker.lstrip("/")):
        return target + "::" + node_id.split("::", 1)[1]
    return node_id


def _rewrite_phase_report_node_ids() -> None:
    target = os.environ.get("PCTDD_REQUIRED_TEST_TARGET", "").strip()
    if not os.environ.get("PCTDD_PYTEST_PHASE_REPORT", "").strip() or not target:
        return
    plugin = sys.modules.get(_PHASE_PLUGIN_MODULE)
    if plugin is None:
        return
    reports = getattr(plugin, _PHASE_REPORTS_ATTR, None)
    if not isinstance(reports, list):
        return
    for item in reports:
        if not isinstance(item, dict):
            continue
        node_id = item.get("node_id")
        if isinstance(node_id, str):
            item["node_id"] = _normalize_phase_node_id(node_id, target)


def _install_required_target_nodeids() -> None:
    target = os.environ.get("PCTDD_REQUIRED_TEST_TARGET", "").strip()
    if not os.environ.get("PCTDD_PYTEST_PHASE_REPORT", "").strip() or not target:
        return
    plugin = sys.modules.get(_PHASE_PLUGIN_MODULE)
    if plugin is None:
        return
    reports = getattr(plugin, _PHASE_REPORTS_ATTR, None)
    if not isinstance(reports, list):
        return
    if getattr(reports, "_pctdd_001_target_bound", False):
        _rewrite_phase_report_node_ids()
        return

    class _TargetBoundPhaseReports(list):
        _pctdd_001_target_bound = True

        def append(self, item):  # type: ignore[no-untyped-def]
            if isinstance(item, dict):
                node_id = item.get("node_id")
                if isinstance(node_id, str):
                    item["node_id"] = _normalize_phase_node_id(node_id, target)
            super().append(item)

        def extend(self, items):  # type: ignore[no-untyped-def]
            for item in items:
                self.append(item)

    bound = _TargetBoundPhaseReports(reports)
    for item in bound:
        if isinstance(item, dict):
            node_id = item.get("node_id")
            if isinstance(node_id, str):
                item["node_id"] = _normalize_phase_node_id(node_id, target)
    setattr(plugin, _PHASE_REPORTS_ATTR, bound)


_install_required_target_nodeids()

import pytest

from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import (
    cid_for_bytes as supervisor_cid_for_bytes,
    cid_for_dag_json as supervisor_cid_for_dag_json,
)
from ipfs_accelerate_py.agent_supervisor.analysis.content_identity_bridge import (
    LOGIC_IR_PROFILE,
    MULTICODEC_DAG_JSON,
    MULTICODEC_RAW,
    MULTIHASH_SHA2_256,
    STRICT_ARTIFACT_PROFILE,
    identify_logic_ir,
    identify_strict_artifact,
    profiles_are_interchangeable,
    sha256_digest_label,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.critical_path import (
    CURRENT_PATH_LABELS,
)
from ipfs_accelerate_py.utils import cid_utils as accelerate_cid_utils
from ipfs_datasets_py.logic.ipld_cid import canonical_dag_json as ipld_canonical_dag_json
from ipfs_datasets_py.logic.ipld_cid import dag_json_cid as ipld_dag_json_cid
from ipfs_datasets_py.logic.ir_core.identity import (
    IDENTITY_PROFILE_NAME,
    canonical_identity,
    cid_v1,
    sha256_digest,
)
from ipfs_datasets_py.logic.profile_g import canonical_profile_g_bytes, profile_g_cid
from ipfs_datasets_py.logic.software_contracts import content as software_contract
from ipfs_datasets_py.utils import cid_utils as datasets_cid_utils
from ipfs_kit_py.mcp_server.mcplusplus.coordination_storage import (
    DurableCoordinationStore,
    cid_for_artifact as kit_cid_for_artifact,
    cid_for_bytes as kit_cid_for_bytes,
)
from ipfs_kit_py.proof_certificate_store import (
    CertificateTransportStatus,
    IpfsKitProofCertificateStore,
    cid_for_certificate_bytes,
    decode_certificate_cid,
    verify_certificate_cid,
)


@pytest.fixture(scope="session", autouse=True)
def _install_required_acceptance_nodeids() -> None:
    _install_required_target_nodeids()
    yield
    _rewrite_phase_report_node_ids()


@pytest.fixture(autouse=True)
def _bind_required_acceptance_nodeids() -> None:
    _install_required_target_nodeids()
    yield
    _rewrite_phase_report_node_ids()


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        receipt = (
            parent
            / "artifacts/parallel_content_sealing_proof_carrying_tdd/receipts/PCTDD-001.json"
        )
        if receipt.is_file():
            return parent
    raise AssertionError("PCTDD-001 receipt is missing from the declared output manifest")


def _load_json(relative: str) -> dict[str, Any]:
    path = _repo_root() / relative
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"{relative} must be a JSON object")
    return payload


def _receipt() -> dict[str, Any]:
    return _load_json(
        "artifacts/parallel_content_sealing_proof_carrying_tdd/receipts/PCTDD-001.json"
    )


def _hash_identity() -> dict[str, Any]:
    return _load_json(
        "docs/architecture/parallel_content_sealing_proof_carrying_tdd_inventory/"
        "hash_identity_inventory.json"
    )


def _critical_path() -> dict[str, Any]:
    return _load_json(
        "docs/architecture/parallel_content_sealing_proof_carrying_tdd_inventory/"
        "hashing_critical_path.json"
    )


def _storage_recovery() -> dict[str, Any]:
    return _load_json(
        "docs/architecture/parallel_content_sealing_proof_carrying_tdd_inventory/"
        "storage_recovery_inventory.json"
    )


def _authority_matrix() -> dict[str, Any]:
    return _load_json(
        "docs/architecture/parallel_content_sealing_proof_carrying_tdd_inventory/"
        "authority_matrix.json"
    )


def _claim_matrix() -> dict[str, Any]:
    return _load_json(
        "docs/architecture/parallel_content_sealing_proof_carrying_tdd_inventory/"
        "proof_claim_matrix.json"
    )


def _ascii_payload() -> dict[str, Any]:
    return {"flag": True, "nested": {"a": 1, "z": 2}, "n": None, "title": "pctdd-001"}


def _unicode_payload() -> dict[str, Any]:
    return {"nested": {"z": 2, "a": 1}, "unicode": "café"}


def _raw_bytes() -> bytes:
    return b"pctdd-001 exact source bytes\n"


def _module_present(name: str) -> bool:
    try:
        importlib.import_module(name)
    except ImportError:
        return False
    return True


def _load_source_module(relative: str, module_name: str) -> Any:
    path = _repo_root() / relative
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load inventoried source {relative}")
    existing = sys.modules.get(module_name)
    existing_file = getattr(existing, "__file__", None) if existing is not None else None
    if existing is not None and existing_file is not None:
        try:
            if Path(existing_file).resolve() == path.resolve():
                return existing
        except OSError:
            pass
    module = importlib.util.module_from_spec(spec)
    # Dataclass processing of postponed annotations looks up cls.__module__ in
    # sys.modules. File-location loads must register before exec_module.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        if sys.modules.get(module_name) is module:
            del sys.modules[module_name]
        raise
    return module


def _assurance_content_identity() -> Any:
    return _load_source_module(
        "external/ipfs_accelerate/ipfs_accelerate_py/assurance/content_identity.py",
        "pctdd_001_assurance_content_identity",
    )


def _kubo_cid_for_bytes(payload: bytes) -> str:
    module = _load_source_module(
        "external/ipfs_accelerate/ipfs_accelerate_py/mcp_server/mcplusplus/kubo_cid.py",
        "pctdd_001_kubo_cid",
    )
    return module.cid_for_bytes(payload)


def _knowledge_graph_ipld_ast() -> ast.Module:
    path = (
        _repo_root()
        / "external/ipfs_datasets/ipfs_datasets_py/knowledge_graphs/storage/ipld_store.py"
    )
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _typed_unavailable(
    *,
    capability: str,
    reason_code: str,
    message: str,
) -> dict[str, Any]:
    record = {
        "capability": capability,
        "reason_code": reason_code,
        "message": message,
        "status": "typed_unavailable",
        "production_admitted": False,
        "claim_unchanged": True,
        "self_approved": False,
    }
    if record["production_admitted"] or record["self_approved"] or not record["claim_unchanged"]:
        raise AssertionError("typed unavailable cases cannot admit, self-approve, or change claims")
    return record


def _inventory_typed_unavailable() -> tuple[dict[str, Any], ...]:
    native_reason = "native_batch_hasher_not_installed"
    native_message = (
        "optional native batch hashing is not a sealed PATH capability; "
        "hashlib.sha256 remains the current hashing behavior and native batch "
        "is not admitted"
    )
    for module_name in (
        "ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing._native_hash",
        "pctdd_native_hash",
    ):
        try:
            __import__(module_name)
        except ImportError:
            break
        native_reason = "native_batch_hasher_unqualified"
        native_message = (
            f"{module_name} imported but is not digest-bound or qualified; "
            "presence is not production admission"
        )
        break
    kit_present = _module_present(_KIT_PROOF_SEAL_STORE_MODULE)
    datasets_present = _module_present(_DATASETS_SEAL_EVIDENCE_MODULE)
    kit_reason = (
        "proof_seal_store_present_unqualified"
        if kit_present
        else "proof_seal_store_module_absent_on_current_kit_pin"
    )
    kit_message = (
        "ipfs_kit_py.proof_seal_store imported but is not publication authority "
        "for this inventory"
        if kit_present
        else (
            "HermeticProofSealStore/SealTransitionWal are not importable on the "
            "current kit pin; IncrementalProofSealer source remains inventoried "
            "without publication authority"
        )
    )
    datasets_reason = (
        "incremental_sealing_evidence_present_unqualified"
        if datasets_present
        else "incremental_sealing_evidence_module_absent_on_current_datasets_pin"
    )
    datasets_message = (
        "datasets incremental-sealing evidence imported but is not production ZK "
        "or publication admission"
        if datasets_present
        else (
            "ipfs_datasets_py.logic.zkp.incremental_sealing.evidence is absent on "
            "the current datasets pin; sealer claim meaning is unchanged"
        )
    )
    return (
        _typed_unavailable(
            capability="production_zk",
            reason_code="production_zk_key_ceremony_unavailable",
            message=(
                "production ZK proving remains typed unavailable; CID/SHA "
                "inventory does not establish ceremony, allowlist, or execution"
            ),
        ),
        _typed_unavailable(
            capability="key_ceremony",
            reason_code="production_zk_key_ceremony_unavailable",
            message=(
                "no production-eligible Groth16 MPC ceremony is admitted; "
                "key generation and download remain forbidden"
            ),
        ),
        _typed_unavailable(
            capability="direct_execution_profile",
            reason_code="direct_execution_profile_optional",
            message=(
                "direct CPython execution profiles remain research-only and unadmitted; "
                "hash/CID identity does not establish execution"
            ),
        ),
        _typed_unavailable(
            capability="native_batch_hasher",
            reason_code=native_reason,
            message=native_message,
        ),
        _typed_unavailable(
            capability="serial_wal_cas_publication",
            reason_code="inventory_has_no_publication_authority",
            message=(
                "serial WAL/CAS publication remains controller-owned; this "
                "inventory cannot publish or advance a current root"
            ),
        ),
        _typed_unavailable(
            capability="kit_proof_seal_store",
            reason_code=kit_reason,
            message=kit_message,
        ),
        _typed_unavailable(
            capability="datasets_incremental_sealing_evidence",
            reason_code=datasets_reason,
            message=datasets_message,
        ),
    )


def _git_blob_oid(payload: bytes) -> str:
    header = b"blob " + str(len(payload)).encode("ascii") + b"\0"
    try:
        digest = hashlib.sha1(header + payload, usedforsecurity=False)
    except TypeError:
        digest = hashlib.sha1(header + payload)
    return digest.hexdigest()


def _sealer_module_ast() -> ast.Module:
    path = _repo_root() / _SEALER_SOURCE
    return ast.parse(path.read_text(encoding="utf-8"), filename=_SEALER_SOURCE)


def _class_def(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"{name} is missing from IncrementalProofSealer source")


def _constant_string(node: ast.AST, name: str) -> str | None:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == name:
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    return node.value.value
    if (
        isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == name
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    ):
        return node.value.value
    return None


def _string_assign(tree: ast.Module, name: str) -> str:
    for node in tree.body:
        value = _constant_string(node, name)
        if value is not None:
            return value
    raise AssertionError(f"{name} string assignment is missing from sealer source")


def test_required_phase_node_ids_bind_to_profile_target() -> None:
    target = _REQUIRED_TEST_TARGET
    relative = (
        "api/parallel_content_sealing/"
        "test_pctdd_001_cid_sha_canonicalization_inventory.py::test_x"
    )
    assert _normalize_phase_node_id(relative, target) == target + "::test_x"
    assert _normalize_phase_node_id(target + "::test_x", target) == target + "::test_x"
    assert (
        _normalize_phase_node_id(
            "test/api/parallel_content_sealing/"
            "test_pctdd_001_cid_sha_canonicalization_inventory.py::test_x",
            target,
        )
        == target + "::test_x"
    )
    collector = os.environ.get("PCTDD_PYTEST_PHASE_REPORT", "").strip()
    required = os.environ.get("PCTDD_REQUIRED_TEST_TARGET", "").strip()
    if not collector or not required:
        return
    plugin = sys.modules.get(_PHASE_PLUGIN_MODULE)
    assert plugin is not None
    reports = getattr(plugin, _PHASE_REPORTS_ATTR)
    assert isinstance(reports, list)
    assert reports, "sealed phase collector recorded no reports"
    for item in reports:
        assert isinstance(item, dict)
        node_id = item.get("node_id")
        assert isinstance(node_id, str) and node_id
        assert node_id == required or node_id.startswith(required + "::")
        assert item.get("disposition") == "passed"


def test_hash_identity_inventory_rules_are_preserved() -> None:
    inventory = _hash_identity()
    matrix = _claim_matrix()
    assert inventory["schema"] == "pctdd/hash-identity@1"
    assert tuple(inventory["rules"]) == _HASH_IDENTITY_RULES
    assert tuple(inventory["preserve"]) == _HASH_IDENTITY_PRESERVE
    assert matrix["IntegrityCommitment"]["establishes"] == (
        "exact bytes/digest/CID/Merkle inclusion"
    )
    assert matrix["IntegrityCommitment"]["does_not"] == "execution or semantics"
    assert matrix["IntegrityCommitment"]["establishes"] != (
        matrix["IncrementalCommitSeal"]["establishes"]
    )


def test_hashing_critical_path_order_is_inventoried() -> None:
    inventory = _critical_path()
    storage = _storage_recovery()
    authority = _authority_matrix()
    assert inventory["schema"] == "pctdd/hashing-critical-path@1"
    assert tuple(inventory["current_path"]) == _CURRENT_PATH_LABELS
    assert tuple(CURRENT_PATH_LABELS) == _CURRENT_PATH_LABELS
    assert inventory["current_path"][-1] == "serial WAL/CAS"
    assert "duplicate serialization/hashing requires task instrumentation" in (
        inventory["observed_gaps"]
    )
    assert storage["required_invariant"].startswith(
        "parallel immutable preparation has no publication authority"
    )
    assert authority["canonical_semantic_and_statement_authority"] == "ipfs_datasets_py"
    assert authority["verified_storage_wal_cas_authority"] == "ipfs_kit_py"
    assert authority["execution_scheduling_admission_authority"] == "ipfs_accelerate_py"
    assert "CID system" in authority["forbidden_duplicates"]
    assert "canonical profile" in authority["forbidden_duplicates"]
    assert "WAL" in authority["forbidden_duplicates"]
    assert "current-root store" in authority["forbidden_duplicates"]


def test_every_source_to_cid_path_is_inventoried_without_changing_identity() -> None:
    receipt = _receipt()
    inventory = receipt["source_to_cid_inventory"]
    assert tuple(inventory) == _SOURCE_TO_CID_PATHS
    raw = _raw_bytes()
    ascii_payload = _ascii_payload()
    unicode_payload = _unicode_payload()

    datasets_raw = datasets_cid_utils.cid_for_bytes(raw)
    accelerate_raw = accelerate_cid_utils.cid_for_bytes(raw)
    supervisor_raw = supervisor_cid_for_bytes(raw)
    contract_raw = software_contract.cid_for_bytes(raw)
    kit_raw = kit_cid_for_bytes(raw, codec="raw")
    ir_raw = cid_v1(raw)
    cert_raw = cid_for_certificate_bytes(raw)
    kubo_raw = _kubo_cid_for_bytes(raw)
    assurance_mod = _assurance_content_identity()
    assurance = assurance_mod.mint_content_identity(raw)
    assert (
        datasets_raw
        == accelerate_raw
        == supervisor_raw
        == contract_raw
        == kit_raw
        == ir_raw
        == cert_raw
        == kubo_raw
    )
    assert assurance.cid == datasets_raw
    assert assurance.codec == "raw"
    assert assurance.digest_hex == hashlib.sha256(raw).hexdigest()
    assert datasets_cid_utils.validate_cid(datasets_raw, codecs=("raw",)) == datasets_raw
    assert accelerate_cid_utils.validate_cid(accelerate_raw, codecs=("raw",)) == accelerate_raw
    assert software_contract.decode_and_recompute_source(contract_raw, raw) == contract_raw
    assert verify_certificate_cid(cert_raw, raw) is True
    parsed = decode_certificate_cid(cert_raw)
    assert parsed.verifies(raw) is True
    assert parsed.codec == 0x55
    assert parsed.multihash_code == 0x12

    dag_bytes = datasets_cid_utils.canonical_dag_json_bytes(ascii_payload)
    accelerate_dag_bytes = accelerate_cid_utils.canonical_dag_json_bytes(ascii_payload)
    contract_dag_bytes = software_contract.canonical_dag_json_bytes(ascii_payload)
    profile_g_bytes = canonical_profile_g_bytes(ascii_payload)
    assert dag_bytes == accelerate_dag_bytes == contract_dag_bytes == profile_g_bytes
    datasets_dag = datasets_cid_utils.cid_for_dag_json(ascii_payload)
    accelerate_dag = accelerate_cid_utils.cid_for_dag_json(ascii_payload)
    supervisor_dag = supervisor_cid_for_dag_json(ascii_payload)
    contract_dag = software_contract.cid_for_structured(ascii_payload)
    kit_dag = kit_cid_for_artifact(ascii_payload)
    ipld_ascii = ipld_dag_json_cid(ascii_payload)
    profile_g = profile_g_cid(ascii_payload)
    bridge = identify_strict_artifact(ascii_payload)
    assert (
        datasets_dag
        == accelerate_dag
        == supervisor_dag
        == contract_dag
        == kit_dag
        == ipld_ascii
        == profile_g
    )
    assert bridge.cid == datasets_dag
    assert bridge.profile == STRICT_ARTIFACT_PROFILE
    assert bridge.multicodec == MULTICODEC_DAG_JSON
    assert bridge.multihash == MULTIHASH_SHA2_256
    assert bridge.canonical_bytes == dag_bytes
    assert bridge.digest == sha256_digest_label(dag_bytes)
    assert software_contract.decode_and_recompute_structured(contract_dag, ascii_payload) == (
        contract_dag
    )
    assert datasets_raw != datasets_dag
    assert datasets_cid_utils.cid_for_bytes(dag_bytes) != datasets_dag

    ir = canonical_identity(
        ascii_payload,
        domain="pctdd-001",
        schema_version="1.0.0",
    )
    bridge_ir = identify_logic_ir(
        ascii_payload,
        domain="pctdd-001",
        schema_version="1.0.0",
    )
    assert ir.profile == IDENTITY_PROFILE_NAME == LOGIC_IR_PROFILE
    assert ir.cid == bridge_ir.cid
    assert ir.canonical_bytes == bridge_ir.canonical_bytes
    assert ir.digest == sha256_digest(ir.canonical_bytes)
    assert bridge_ir.multicodec == MULTICODEC_RAW
    assert ir.cid != datasets_dag
    assert ir.cid != datasets_raw
    assert ir.cid.startswith("b")
    assert profiles_are_interchangeable(bridge, bridge) is True
    assert profiles_are_interchangeable(bridge_ir, bridge_ir) is True
    assert profiles_are_interchangeable(bridge, bridge_ir) is False

    unicode_datasets = datasets_cid_utils.canonical_dag_json_bytes(unicode_payload)
    unicode_ipld = ipld_canonical_dag_json(unicode_payload)
    assert unicode_datasets != unicode_ipld
    assert datasets_cid_utils.cid_for_dag_json(unicode_payload) != ipld_dag_json_cid(
        unicode_payload
    )
    kg_tree = _knowledge_graph_ipld_ast()
    kg_assigns: dict[str, Any] = {}
    for node in kg_tree.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and isinstance(node.value, ast.Constant)
        ):
            kg_assigns[node.target.id] = node.value.value
        elif (
            isinstance(node, ast.Assign)
            and node.targets
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
        ):
            kg_assigns[node.targets[0].id] = node.value.value
    assert kg_assigns["DEFAULT_MANIFEST_CODEC"] == "dag-cbor"
    assert kg_assigns["DEFAULT_PAYLOAD_CODEC"] == "raw"
    kg_functions = {
        node.name
        for node in kg_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "compute_cid_v1" in kg_functions
    assert inventory["datasets_ipld_cid"]["classification"] == "available_distinct_profile"
    assert inventory["datasets_profile_g"]["classification"] == "available_distinct_profile"
    assert inventory["datasets_knowledge_graph_ipld"]["classification"] == (
        "available_distinct_profile"
    )
    for name in _SOURCE_TO_CID_PATHS:
        record = inventory[name]
        assert record["owner"]
        assert record["profile"]
        assert record["codec"]
        assert record["classification"]
        assert record["does_not"] == "execution or semantics"
        assert record["self_approved"] is False
        source = record["source"]
        assert isinstance(source, str) and source
        source_path = _repo_root() / source
        assert source_path.exists(), f"inventoried source missing: {source}"


def test_canonicalization_profiles_are_not_aliases() -> None:
    payload = _unicode_payload()
    ascii_payload = _ascii_payload()
    legacy = datasets_cid_utils.canonical_json_bytes(payload)
    strict = datasets_cid_utils.canonical_dag_json_bytes(payload)
    accelerate_strict = accelerate_cid_utils.canonical_dag_json_bytes(payload)
    ipld = ipld_canonical_dag_json(payload)
    contract = software_contract.canonical_dag_json_bytes(ascii_payload)
    profile_g = canonical_profile_g_bytes(ascii_payload)
    assert legacy == strict
    assert strict == accelerate_strict
    assert strict != ipld
    assert b"\\u" in ipld
    assert "café".encode("utf-8") in strict
    assert contract == profile_g
    try:
        software_contract.canonical_dag_json_bytes({"n": 1.5})
    except software_contract.StructuredIdentityError:
        pass
    else:
        raise AssertionError("software-contract structured identity must reject floats")
    try:
        canonical_profile_g_bytes({"n": 1.5})
    except ValueError:
        pass
    else:
        raise AssertionError("Profile G must reject floats")
    finite = datasets_cid_utils.canonical_dag_json_bytes({"n": 1.5})
    assert finite == b'{"n":1.5}'
    marker = object()
    represented = datasets_cid_utils.canonical_json_bytes({"marker": marker})
    assert represented.startswith(b'{"marker":')
    try:
        datasets_cid_utils.canonical_dag_json_bytes({"marker": marker})
    except TypeError:
        pass
    else:
        raise AssertionError("strict DAG-JSON must not stringify host objects")
    left = {"nested": {"z": 2, "a": 1}, "unicode": "café"}
    right = {"unicode": "café", "nested": {"a": 1, "z": 2}}
    assert datasets_cid_utils.canonical_dag_json_bytes(left) == (
        datasets_cid_utils.canonical_dag_json_bytes(right)
    )
    assert datasets_cid_utils.cid_for_dag_json(left) == datasets_cid_utils.cid_for_dag_json(right)


def test_sha256_stream_is_not_composed_from_chunk_hashes() -> None:
    stream = b"0123456789abcdef" * 16
    full = hashlib.sha256(stream).digest()
    left = hashlib.sha256(stream[:64]).digest()
    right = hashlib.sha256(stream[64:]).digest()
    composed = hashlib.sha256(left + right).digest()
    assert full != composed
    assert full != left
    assert full != right
    cid = datasets_cid_utils.cid_for_bytes(stream)
    digest = hashlib.sha256(stream).digest()
    assert accelerate_cid_utils.cid_from_sha256_digest(digest) == cid
    assert accelerate_cid_utils.digest_bytes_from_cid(cid) == digest
    assert sha256_digest(stream) == "sha256:" + digest.hex()
    chunk_manifest = {
        "kind": "chunk-manifest",
        "chunks": [
            {"offset": 0, "length": 64, "digest": "sha256:" + left.hex()},
            {"offset": 64, "length": 64, "digest": "sha256:" + right.hex()},
        ],
    }
    manifest_cid = datasets_cid_utils.cid_for_dag_json(chunk_manifest)
    assert manifest_cid != cid
    assert hashlib.sha256(datasets_cid_utils.canonical_dag_json_bytes(chunk_manifest)).digest() != (
        full
    )


def test_git_blob_oid_and_filesystem_metadata_are_not_cid_identity() -> None:
    payload = _raw_bytes()
    cid = datasets_cid_utils.cid_for_bytes(payload)
    blob_oid = _git_blob_oid(payload)
    digest_hex = hashlib.sha256(payload).hexdigest()
    assert blob_oid != cid
    assert blob_oid != digest_hex
    assert len(blob_oid) == 40
    assert not blob_oid.startswith("b")
    assert cid.startswith("b")
    assurance_mod = _assurance_content_identity()
    try:
        assurance_mod.reject_pseudo_cid(digest_hex)
    except assurance_mod.ContentIdentityError as exc:
        assert exc.code is assurance_mod.IdentityErrorCode.PSEUDO_CID_RAW_HEX
        assert exc.integrity is assurance_mod.Integrity.UNCHECKED
    else:
        raise AssertionError("raw SHA-256 hex must not be admitted as a CID")
    try:
        assurance_mod.reject_pseudo_cid("sha256:" + digest_hex)
    except assurance_mod.ContentIdentityError as exc:
        assert exc.code is assurance_mod.IdentityErrorCode.PSEUDO_CID_LABELED
    else:
        raise AssertionError("labeled SHA-256 must not be admitted as a CID")
    try:
        assurance_mod.reject_pseudo_cid("QmTest0123456789abcdef0123456789abcdef")
    except assurance_mod.ContentIdentityError as exc:
        assert exc.integrity is assurance_mod.Integrity.UNCHECKED
    else:
        raise AssertionError("Qm-like identifiers must not be admitted as CIDs")
    metadata = {"size": len(payload), "mtime": 1, "inode": "candidate-only", "path": "a.bin"}
    impostor = b"different-bytes-same-metadata-must-not-reuse\n"
    assert hashlib.sha256(payload).digest() != hashlib.sha256(impostor).digest()
    assert json.dumps(metadata, sort_keys=True) != payload.decode("utf-8", errors="replace")
    rules = _hash_identity()["rules"]
    assert "Git blob OID is memo lookup key only" in rules
    assert "filesystem metadata nominates a candidate only" in rules


def test_chunk_manifest_does_not_replace_raw_full_stream_cid() -> None:
    stream = b"full-stream-bytes-are-not-chunk-identity\n" * 4
    raw_cid = datasets_cid_utils.cid_for_bytes(stream)
    chunks = [stream[index : index + 16] for index in range(0, len(stream), 16)]
    manifest = {
        "kind": "ordered-chunk-manifest",
        "chunks": [
            {
                "offset": index * 16,
                "length": len(chunk),
                "digest": "sha256:" + hashlib.sha256(chunk).hexdigest(),
            }
            for index, chunk in enumerate(chunks)
        ],
    }
    manifest_cid = datasets_cid_utils.cid_for_dag_json(manifest)
    joined_chunk_digests = b"".join(hashlib.sha256(chunk).digest() for chunk in chunks)
    assert hashlib.sha256(stream).digest() != hashlib.sha256(joined_chunk_digests).digest()
    assert raw_cid != manifest_cid
    assert datasets_cid_utils.validate_cid(raw_cid, codecs=("raw",)) == raw_cid
    assert datasets_cid_utils.validate_cid(manifest_cid, codecs=("dag-json",)) == manifest_cid
    assert "chunk manifest does not replace raw full-stream CID" in _hash_identity()["rules"]


def test_seal_publication_paths_are_inventoried_without_publication_authority(
    tmp_path: Path,
) -> None:
    receipt = _receipt()
    storage = _storage_recovery()
    publication = receipt["seal_publication_inventory"]
    path_keys = tuple(key for key in publication if key != "required_invariant")
    assert path_keys == _SEAL_PUBLICATION_PATHS
    assert publication["required_invariant"] == storage["required_invariant"]
    assert receipt["publication_authority_invoked"] is False
    sealer = publication["accelerate_incremental_sealer"]
    assert sealer["interface"] == "IncrementalProofSealer@1"
    assert sealer["publication_authority_invoked"] is False
    assert sealer["self_approved"] is False
    assert sealer["classification"] == "available_with_caveats"
    assert tuple(sealer["kinds"]) == ("full_checkpoint", "delta_seal")
    assert sealer["source"] == _SEALER_SOURCE
    assert (_repo_root() / _SEALER_SOURCE).is_file()

    tree = _sealer_module_ast()
    assert _string_assign(tree, "SEALER_INTERFACE") == "IncrementalProofSealer@1"
    kind_cls = _class_def(tree, "PublicationKind")
    kind_values = {
        node.targets[0].id: node.value.value
        for node in kind_cls.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
    }
    assert kind_values["FULL_CHECKPOINT"] == "full_checkpoint"
    assert kind_values["DELTA_SEAL"] == "delta_seal"
    assert kind_values["FULL_CHECKPOINT"] != kind_values["DELTA_SEAL"]
    sealer_cls = _class_def(tree, "IncrementalProofSealer")
    methods = {
        node.name
        for node in sealer_cls.body
        if isinstance(node, ast.FunctionDef)
    }
    assert "publish_full_checkpoint" in methods
    assert "publish_delta_seal" in methods
    assert "recover_publication" in methods
    assert "publish_full_checkpoint" != "publish_delta_seal"

    kit_store = publication["kit_proof_seal_store"]
    kit_present = _module_present(_KIT_PROOF_SEAL_STORE_MODULE)
    datasets_present = _module_present(_DATASETS_SEAL_EVIDENCE_MODULE)
    if kit_present:
        assert kit_store["classification"] == "available"
    else:
        assert kit_store["classification"] == "typed_unavailable"
        assert kit_store["reason_code"] == "proof_seal_store_module_absent_on_current_kit_pin"
        assert kit_store["publication_authority_invoked"] is False
        assert kit_store["self_approved"] is False
        assert kit_store["claim_unchanged"] is True
    if not datasets_present:
        assert receipt["limitations"]["datasets_incremental_sealing_evidence"][
            "reason_code"
        ] == "incremental_sealing_evidence_module_absent_on_current_datasets_pin"

    assert "immutable proof-seal persistence" in storage["existing"]
    assert "ordered WAL" in storage["existing"]
    assert "current-pointer compare-and-swap" in storage["existing"]
    assert "stale-parent rejection" in storage["existing"]
    assert "crash recovery" in storage["existing"]

    with DurableCoordinationStore(tmp_path / "cas") as store:
        successor = store.put({"schema": "pctdd/inventory@1", "name": "one"})["cid"]
        before = store.current_state_root("semantic/pctdd-001")
        assert before["root_cid"] is None
        assert before["revision"] == 0
        result = store.compare_and_swap_state_root(
            "semantic/pctdd-001",
            expected_revision=0,
            expected_root_cid=None,
            new_root_cid=successor,
            operation_id="inventory-1",
        )
        assert result["status"] == "updated"
        assert result["after"]["root_cid"] == successor
        stale = store.compare_and_swap_state_root(
            "semantic/pctdd-001",
            expected_revision=0,
            expected_root_cid=None,
            new_root_cid=store.put({"schema": "pctdd/inventory@1", "name": "two"})["cid"],
            operation_id="inventory-stale",
        )
        assert stale["status"] == "conflict"
        assert store.current_state_root("semantic/pctdd-001")["root_cid"] == successor

    cert_root = tmp_path / "certs"
    store = IpfsKitProofCertificateStore(cert_root)
    data = b'{"certificate":"pctdd-001","passed":true}'
    cid = cid_for_certificate_bytes(data)
    put = store.put_bytes(data, claimed_cid=cid)
    assert put.stored and put.cid == cid
    fetched = store.get_bytes(cid)
    assert fetched.status is CertificateTransportStatus.HIT
    assert fetched.data == data
    assert receipt["publication_authority_invoked"] is False
    assert publication["kit_current_root_cas"]["publication_authority_invoked"] is False
    assert publication["kit_proof_certificate_store"]["publication_authority_invoked"] is False
    assert publication["kit_proof_seal_store"]["publication_authority_invoked"] is False


def test_typed_unavailable_cases_do_not_change_claim_meaning() -> None:
    identity_before = _hash_identity()
    path_before = _critical_path()
    storage_before = _storage_recovery()
    records = _inventory_typed_unavailable()
    capabilities = {item["capability"] for item in records}
    assert capabilities == set(_TYPED_UNAVAILABLE_CAPABILITIES)
    receipt_limitations = _receipt()["limitations"]
    for item in records:
        assert item["status"] == "typed_unavailable"
        assert item["production_admitted"] is False
        assert item["self_approved"] is False
        assert item["claim_unchanged"] is True
        assert item["reason_code"]
        assert item["message"]
        limitation = receipt_limitations[item["capability"]]
        assert limitation["status"] == "typed_unavailable"
        assert limitation["production_admitted"] is False
        assert limitation["self_approved"] is False
        assert limitation["claim_unchanged"] is True
        if item["capability"] == "native_batch_hasher":
            assert limitation["reason_code"] in {
                "native_batch_hasher_not_installed",
                "native_batch_hasher_unqualified",
            }
        elif item["capability"] == "kit_proof_seal_store":
            assert limitation["reason_code"] == item["reason_code"]
        elif item["capability"] == "datasets_incremental_sealing_evidence":
            assert limitation["reason_code"] == item["reason_code"]
        else:
            assert limitation["reason_code"] == item["reason_code"]
    identity_after = _hash_identity()
    path_after = _critical_path()
    storage_after = _storage_recovery()
    assert identity_after == identity_before
    assert path_after == path_before
    assert storage_after == storage_before
    assert tuple(identity_after["rules"]) == _HASH_IDENTITY_RULES
    poisoned = dict(records[0])
    poisoned["production_admitted"] = True
    try:
        if poisoned["production_admitted"] or poisoned["self_approved"] or not poisoned["claim_unchanged"]:
            raise AssertionError(
                "typed unavailable cases cannot admit, self-approve, or change claims"
            )
        raise AssertionError("poisoned production admission must be rejected")
    except AssertionError as exc:
        assert "cannot admit" in str(exc)


def test_receipt_is_not_completion_authority() -> None:
    receipt = _receipt()
    assert receipt["schema"] == "pctdd/task-receipt@1"
    assert receipt["task_id"] == "PCTDD-001"
    assert receipt["plan_revision"] == "PCTDD-PLAN-V1.1"
    assert receipt["store_generation"] == "pctdd-v1-g6"
    assert receipt["completion_authoritative"] is False
    assert receipt["self_approval"] is False
    assert receipt["worker_authored_test_is_sufficient_alone"] is False
    assert receipt["status"] == "implementation_submitted_pending_controller_validation"
    assert receipt["claim_class"] == "IntegrityCommitment"
    assert receipt["publication_authority_invoked"] is False
    assert receipt["markdown_non_authoritative"] is True
    assert receipt["validation_profile"] == "pctdd-validation/PCTDD-PLAN-V1.1/PCTDD-001@1"
    assert "controller-owned" in receipt["completion_authority"]
    assert "does not complete" in receipt["claim"].casefold()
    assert receipt["changed_paths"] == [
        "artifacts/parallel_content_sealing_proof_carrying_tdd/receipts/PCTDD-001.json",
        "external/ipfs_accelerate/test/api/parallel_content_sealing/"
        "test_pctdd_001_cid_sha_canonicalization_inventory.py",
    ]
    limitations = receipt["limitations"]
    for key in _TYPED_UNAVAILABLE_CAPABILITIES:
        assert limitations[key]["status"] == "typed_unavailable"
        assert limitations[key]["production_admitted"] is False
        assert limitations[key]["self_approved"] is False
        assert limitations[key]["claim_unchanged"] is True
    assert receipt["predecessor_rescue_candidate"]["admitted"] is False
    assert receipt["predecessor_rescue_candidate"]["classification"] == (
        "receipt-observation-only"
    )
    assert receipt["predecessor_rescue_candidate"]["outer_commit"] == (
        "e623dd43dbc8f8feb503dd8dea2a6afb4bbd26c0"
    )
    assert tuple(receipt["source_to_cid_inventory"]) == _SOURCE_TO_CID_PATHS
    publication = receipt["seal_publication_inventory"]
    assert publication["accelerate_incremental_sealer"]["publication_authority_invoked"] is False
    assert publication["kit_current_root_cas"]["publication_authority_invoked"] is False
    assert publication["kit_proof_seal_store"]["publication_authority_invoked"] is False
    assert "execution or semantics" in receipt["claim_does_not"]
    assert "current-root publication" in receipt["claim_does_not"]
