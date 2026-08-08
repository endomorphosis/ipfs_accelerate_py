"""Fail-closed materialization of reviewed DCR-001..004 JSON artifacts.

These files are projections of reviewed inputs, not execution authority.  In
particular, a capability inventory is not made selectable merely because it
contains boolean self-test fields: every available entry must be accompanied
by matching, content-bound :class:`CapabilityEvidenceReceipt` records.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from .capabilities import (
    CAPABILITY_EVIDENCE_RECEIPT_SCHEMA,
    DETERMINISTIC_REPAIR_CAPABILITIES_INTERFACE,
    DETERMINISTIC_REPAIR_CAPABILITIES_SCHEMA,
    CapabilityEvidenceReceipt,
    DeterministicRepairCapabilities,
)
from .contracts import (
    DETERMINISTIC_REPAIR_CONTRACT_VERSION,
    DETERMINISTIC_REPAIR_INTERFACE,
    POST_EDIT_VALIDATION_RECEIPT_SCHEMA,
    PUBLICATION_RECEIPT_SCHEMA,
    REPAIR_ADMISSION_RECEIPT_SCHEMA,
    REPROOF_RECEIPT_SCHEMA,
    AuthorityStage,
    DeterministicRepairDisposition,
)
from .no_llm_policy import (
    DETERMINISTIC_REPAIR_AUTHORITY_POLICY_INTERFACE,
    DETERMINISTIC_REPAIR_AUTHORITY_POLICY_SCHEMA,
    DeterministicRepairAuthorityPolicy,
)
from .root_ownership import (
    REPAIR_ROOT_OWNERSHIP_INTERFACE,
    REPAIR_ROOTS_SCHEMA,
    RepairRootOwnership,
)

DETERMINISTIC_ARTIFACT_MATERIALIZER_INTERFACE: Final[str] = (
    "DeterministicRepairArtifactMaterializer@1"
)
NO_LLM_POLICY_ARTIFACT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-repair/no-llm-policy-artifact@1"
)
DISPOSITION_SCHEMA_ARTIFACT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-repair/disposition-schema-artifact@1"
)
ROOT_POLICY_ARTIFACT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-repair/root-policy-artifact@1"
)
CAPABILITIES_ARTIFACT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-repair/capabilities-artifact@1"
)
_ARTIFACT_FILENAMES: Final[tuple[str, ...]] = (
    "no-llm-policy.json",
    "disposition-schema.json",
    "root-policy.json",
    "capabilities.json",
)


class DeterministicArtifactError(ValueError):
    """An input, output location, or generated artifact is not admissible."""


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DeterministicArtifactError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise DeterministicArtifactError(f"non-canonical JSON constant: {value}")


def _read_json(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    try:
        source = path.read_bytes()
        payload = json.loads(
            source.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, DeterministicArtifactError) as exc:
        raise DeterministicArtifactError(f"{label} is unreadable or non-canonical") from exc
    if not isinstance(payload, dict):
        raise DeterministicArtifactError(f"{label} must be a JSON object")
    return payload, hashlib.sha256(source).hexdigest()


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _source_binding(*, schema: str, interface: str, sha256: str) -> dict[str, str]:
    return {
        "interface": interface,
        "schema": schema,
        "sha256": sha256,
    }


def _require_output_directory(output_dir: Path | str) -> Path:
    destination = Path(output_dir)
    if destination.is_symlink() or not destination.is_dir():
        raise DeterministicArtifactError(
            "output_dir must be an existing non-symlink directory explicitly supplied by the caller"
        )
    return destination


def _verified_capabilities(
    inventory: DeterministicRepairCapabilities | None,
    evidence: Sequence[CapabilityEvidenceReceipt] | None,
) -> dict[str, Any]:
    unavailable: dict[str, Any] = {
        "authoritative": False,
        "availability": "unavailable",
        "evidence_schema": CAPABILITY_EVIDENCE_RECEIPT_SCHEMA,
    }
    if inventory is None:
        return {**unavailable, "reason": "capability_inventory_not_supplied"}
    if not isinstance(inventory, DeterministicRepairCapabilities):
        raise DeterministicArtifactError("capabilities must be DeterministicRepairCapabilities")
    if evidence is None:
        return {**unavailable, "reason": "capability_evidence_not_supplied"}
    if isinstance(evidence, (str, bytes)) or not all(
        isinstance(item, CapabilityEvidenceReceipt) for item in evidence
    ):
        raise DeterministicArtifactError("capability evidence must contain only typed receipts")
    if not inventory.modules and not inventory.toolchains:
        return {**unavailable, "reason": "capability_inventory_empty"}
    if not inventory.available:
        return {**unavailable, "reason": "capability_inventory_unavailable"}

    receipts = tuple(evidence)

    def has_receipt(
        *,
        evidence_id: str,
        evidence_kind: str,
        subject_id: str,
        subject_digest: str,
        subject_version: str,
    ) -> bool:
        return any(
            receipt.verifies(
                evidence_id=evidence_id,
                evidence_kind=evidence_kind,
                subject_id=subject_id,
                subject_digest=subject_digest,
                subject_version=subject_version,
            )
            for receipt in receipts
        )

    for module in inventory.modules:
        required = ("initialization", "reconstruction", "self_test")
        if not all(
            has_receipt(
                evidence_id=module.capability_id,
                evidence_kind=kind,
                subject_id=module.capability_id,
                subject_digest=module.content_digest,
                subject_version=module.distribution_version,
            )
            for kind in required
        ):
            return {**unavailable, "reason": "module_evidence_missing_or_mismatched"}
    for toolchain in inventory.toolchains:
        required = (
            (toolchain.self_test_id, "self_test"),
            (toolchain.reconstruction_id, "reconstruction"),
        )
        if not all(
            has_receipt(
                evidence_id=evidence_id,
                evidence_kind=kind,
                subject_id=toolchain.tool_id,
                subject_digest=toolchain.executable_digest,
                subject_version=toolchain.version,
            )
            for evidence_id, kind in required
        ):
            return {**unavailable, "reason": "toolchain_evidence_missing_or_mismatched"}
    return {
        "authoritative": True,
        "availability": "available",
        "evidence_receipt_ids": sorted(receipt.receipt_id for receipt in receipts),
        "evidence_schema": CAPABILITY_EVIDENCE_RECEIPT_SCHEMA,
        "inventory": inventory.to_dict(),
    }


def _build_artifacts(
    *,
    authority_policy_path: Path | str,
    root_policy_path: Path | str,
    workspace_root: Path | str,
    capabilities: DeterministicRepairCapabilities | None,
    capability_evidence: Sequence[CapabilityEvidenceReceipt] | None,
) -> dict[str, dict[str, Any]]:
    authority_path = Path(authority_policy_path)
    root_path = Path(root_policy_path)
    authority_raw, authority_sha256 = _read_json(authority_path, label="authority policy")
    root_raw, root_sha256 = _read_json(root_path, label="root policy")
    try:
        authority_policy = DeterministicRepairAuthorityPolicy.from_mapping(authority_raw)
        RepairRootOwnership.from_mapping(root_raw, workspace_root=workspace_root)
    except ValueError as exc:
        raise DeterministicArtifactError("reviewed policy input was rejected") from exc

    authority_binding = _source_binding(
        schema=DETERMINISTIC_REPAIR_AUTHORITY_POLICY_SCHEMA,
        interface=DETERMINISTIC_REPAIR_AUTHORITY_POLICY_INTERFACE,
        sha256=authority_sha256,
    )
    root_binding = _source_binding(
        schema=REPAIR_ROOTS_SCHEMA,
        interface=REPAIR_ROOT_OWNERSHIP_INTERFACE,
        sha256=root_sha256,
    )
    common = {
        "materializer_interface": DETERMINISTIC_ARTIFACT_MATERIALIZER_INTERFACE,
        "non_executable": True,
        "source_authority_policy": authority_binding,
        "source_root_policy": root_binding,
    }
    return {
        "no-llm-policy.json": {
            **common,
            "artifact_schema": NO_LLM_POLICY_ARTIFACT_SCHEMA,
            "policy": authority_policy.to_dict(),
        },
        "disposition-schema.json": {
            **common,
            "artifact_schema": DISPOSITION_SCHEMA_ARTIFACT_SCHEMA,
            "contract_interface": DETERMINISTIC_REPAIR_INTERFACE,
            "contract_version": DETERMINISTIC_REPAIR_CONTRACT_VERSION,
            "public_dispositions": [item.value for item in DeterministicRepairDisposition],
            "authority_stages": [item.value for item in AuthorityStage],
            "receipt_schemas": {
                "admission": REPAIR_ADMISSION_RECEIPT_SCHEMA,
                "post_edit_validation": POST_EDIT_VALIDATION_RECEIPT_SCHEMA,
                "publication": PUBLICATION_RECEIPT_SCHEMA,
                "reproof": REPROOF_RECEIPT_SCHEMA,
            },
        },
        "root-policy.json": {
            **common,
            "artifact_schema": ROOT_POLICY_ARTIFACT_SCHEMA,
            "root_policy": root_raw,
        },
        "capabilities.json": {
            **common,
            "artifact_schema": CAPABILITIES_ARTIFACT_SCHEMA,
            "capabilities_interface": DETERMINISTIC_REPAIR_CAPABILITIES_INTERFACE,
            "capabilities_schema": DETERMINISTIC_REPAIR_CAPABILITIES_SCHEMA,
            "capabilities": _verified_capabilities(capabilities, capability_evidence),
        },
    }


def materialize_deterministic_repair_artifacts(
    *,
    authority_policy_path: Path | str,
    root_policy_path: Path | str,
    workspace_root: Path | str,
    output_dir: Path | str,
    capabilities: DeterministicRepairCapabilities | None = None,
    capability_evidence: Sequence[CapabilityEvidenceReceipt] | None = None,
) -> dict[str, Path]:
    """Write canonical DCR artifact projections to one explicit output directory."""

    destination = _require_output_directory(output_dir)
    artifacts = _build_artifacts(
        authority_policy_path=authority_policy_path,
        root_policy_path=root_policy_path,
        workspace_root=workspace_root,
        capabilities=capabilities,
        capability_evidence=capability_evidence,
    )
    outputs: dict[str, Path] = {}
    for filename in _ARTIFACT_FILENAMES:
        target = destination / filename
        if target.is_symlink() or (target.exists() and not target.is_file()):
            raise DeterministicArtifactError(f"artifact target is unsafe: {filename}")
        target.write_bytes(_canonical_json_bytes(artifacts[filename]))
        outputs[filename] = target
    return outputs


def verify_deterministic_repair_artifacts(
    *,
    authority_policy_path: Path | str,
    root_policy_path: Path | str,
    workspace_root: Path | str,
    output_dir: Path | str,
    capabilities: DeterministicRepairCapabilities | None = None,
    capability_evidence: Sequence[CapabilityEvidenceReceipt] | None = None,
) -> None:
    """Reject missing, changed, or non-canonical artifacts without rewriting them."""

    destination = _require_output_directory(output_dir)
    artifacts = _build_artifacts(
        authority_policy_path=authority_policy_path,
        root_policy_path=root_policy_path,
        workspace_root=workspace_root,
        capabilities=capabilities,
        capability_evidence=capability_evidence,
    )
    for filename in _ARTIFACT_FILENAMES:
        target = destination / filename
        if target.is_symlink() or not target.is_file():
            raise DeterministicArtifactError(f"artifact is missing or unsafe: {filename}")
        try:
            actual = target.read_bytes()
        except OSError as exc:
            raise DeterministicArtifactError(f"artifact cannot be read: {filename}") from exc
        if actual != _canonical_json_bytes(artifacts[filename]):
            raise DeterministicArtifactError(
                f"artifact bytes do not match reviewed inputs: {filename}"
            )


__all__ = [
    "CAPABILITIES_ARTIFACT_SCHEMA",
    "DETERMINISTIC_ARTIFACT_MATERIALIZER_INTERFACE",
    "DISPOSITION_SCHEMA_ARTIFACT_SCHEMA",
    "DeterministicArtifactError",
    "NO_LLM_POLICY_ARTIFACT_SCHEMA",
    "ROOT_POLICY_ARTIFACT_SCHEMA",
    "materialize_deterministic_repair_artifacts",
    "verify_deterministic_repair_artifacts",
]
