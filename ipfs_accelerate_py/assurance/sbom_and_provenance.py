"""Fail-closed PCPR-054 Accelerate SBOMs and build provenance.

Produce a declared-release-profile SPDX-2.3 SBOM and a declared build
provenance binding for ipfs_accelerate_py. Direct packages come from the
PCPR-053 declared lock. Hash, transitive, scanner, SLSA, wheel, sdist,
and container identities stay typed unavailable. Hashes are never
invented. Live scanners and signed attestations are not claimed.

This module is not release authority: it does not write DuckDB or Quack
state and never emits a closed PCPR release outcome. Live claims require
live evidence. Simulated results are not live.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from ipfs_accelerate_py.assurance.dependency_locks import (
    CLOSED_RELEASE_OUTCOMES,
    NAMED_GIT_EXTRAS,
    NAMED_GIT_EXTRAS_ROLE,
    PACKAGE_NAME,
    PACKAGE_VERSION,
    SEALED_PATH,
    SEALED_PYTHON,
    content_identity,
    discover_accelerate_root,
    observe_sealed_validation_environment,
    pretty_json,
    render_release_lock,
    sha256_bytes,
    typed_unavailable,
)

INTERFACE: Final = "AccelerateSbomAndProvenance@1"
SCHEMA: Final = "ipfs_accelerate_py/assurance/sbom-and-provenance@1"
SBOM_SCHEMA: Final = "ipfs_accelerate_py/assurance/declared-sbom@1"
PROVENANCE_SCHEMA: Final = "ipfs_accelerate_py/assurance/build-provenance@1"
VERDICT_SCHEMA: Final = (
    "ipfs_accelerate_py/assurance/sbom-and-provenance-verdict@1"
)
CATALOG_SCHEMA: Final = (
    "ipfs_accelerate_py/assurance/platform-sbom-and-provenance-catalog@1"
)
CATALOG_INTERFACE: Final = "PlatformSbomAndProvenanceCatalog@1"
PCPR_054_TASK_ID: Final = "PCPR-054"
PCPR_054_GOAL_ID: Final = "PCPR-G600"
PCPR_053_TASK_ID: Final = "PCPR-053"
PCPR_001_TASK_ID: Final = "PCPR-001"
PCPR_PROGRAM_ID: Final = "proof-carrying-platform-qualification-and-release-v1"
PCPR_BOARD_NAMESPACE: Final = (
    "proof-carrying-platform-qualification-and-release-v1"
)
EVIDENCE_ID: Final = "pcpr/accelerate-sbom-and-provenance@1"

CANONICAL_PYTHON_REQUIRES: Final = ">=3.12"
RUNTIME_REQUIRES_AUTHORITY: Final = "requirements.txt"
SOURCE_REPOSITORY: Final = "endomorphosis/ipfs_accelerate_py"
SOURCE_ORIGIN_URL: Final = "https://github.com/endomorphosis/ipfs_accelerate_py"
ROOT_SPDX_LICENSE: Final = "AGPL-3.0-or-later"
SBOM_KIND: Final = "declared_release_profile"
PROVENANCE_KIND: Final = "declared_build_binding"
SPDX_VERSION: Final = "SPDX-2.3"
LOCK_KIND: Final = "declared_release_profile"
SBOM_DIR_RELPATH: Final = "packaging/pcpr/sbom/cpython312"
PROVENANCE_DIR_RELPATH: Final = "packaging/pcpr/provenance/cpython312"
SBOM_JSON_NAME: Final = "release.sbom.json"
PROVENANCE_JSON_NAME: Final = "release.provenance.json"
SBOM_README_RELPATH: Final = "packaging/pcpr/sbom/README.md"
PROVENANCE_README_RELPATH: Final = "packaging/pcpr/provenance/README.md"
CATALOG_RELPATH: Final = "packaging/pcpr/sbom/platform-catalog.json"
SOURCE_DATE_EPOCH: Final = "0"
SPDX_CREATED: Final = "1970-01-01T00:00:00Z"

PROMOTION_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "supervisor_promoted",
        "supervisor_non_promoted",
        "rnd_non_promoted",
        "typed_unavailable",
        "typed_blocked",
    }
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

HERMETIC_CANDIDATE_SUITES: Final[tuple[str, ...]] = (
    "test/api/test_agent_supervisor_sbom_and_provenance.py",
)

SBOM_SCANNER_TOOLS: Final[tuple[str, ...]] = (
    "syft",
    "cyclonedx",
    "cyclonedx-py",
    "cyclonedx-bom",
    "cdxgen",
    "trivy",
    "grype",
)
PROVENANCE_TOOLS: Final[tuple[str, ...]] = (
    "slsa-generator",
    "slsa-verifier",
    "cosign",
    "in-toto-run",
    "in-toto-sign",
)

REQUIRED_GOOD_PROBE_IDS: Final[frozenset[str]] = frozenset(
    {
        "sbom_files_match_generator",
        "provenance_files_match_generator",
        "pyproject_sbom_and_provenance_table",
        "hashes_not_invented",
        "files_not_analyzed",
        "no_mutable_main_reference",
        "sbom_kind_is_declared_release_profile",
        "provenance_kind_is_declared_build_binding",
        "named_git_extras_are_not_sbom",
        "direct_packages_match_lock",
    }
)
FORBIDDEN_PRESENT_PROBE_IDS: Final[frozenset[str]] = frozenset(
    {
        "hashes_invented",
        "files_analyzed",
        "mutable_main_reference",
        "simulated_results_represented_as_live",
        "hashes_represented_as_live",
        "live_scanner_represented_as_live",
        "slsa_attestation_represented_as_live",
        "closed_release_represented_as_live",
        "wheel_represented_as_live_release",
        "container_represented_as_live",
    }
)

_PEP508_NAME_RE: Final = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)")

SBOM_README: Final = """# PCPR-054 declared-release-profile SBOMs

These files are declared SPDX-2.3 SBOMs of the Accelerate *release*
profile. They list the root package and the PCPR-053 declared PEP 508
direct requires. They are not a live Syft/CycloneDX/Trivy scan, not a
hashed transitive graph, not a signed SLSA attestation, and not a
closed PCPR release.

- `cpython312/release.sbom.json` is the canonical declared SBOM.
- Hash, license-concluded, and transitive identities stay typed
  unavailable. Hashes are never invented. `filesAnalyzed` is false.
- Named extras `libp2p`, `mcp-p2p`, `ipfs-transformers`, and
  `ipfs-model-manager` may carry source-checkout Git pins and are not
  this SBOM.
- `platform-catalog.json` binds Datasets, Kit, and Accelerate SBOM and
  provenance identities when those sibling trees are present.

Sealed validation PATH is exactly `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin`.
Provider-local `~/.local` tools are not sealed-environment authority.
"""

PROVENANCE_README: Final = """# PCPR-054 declared build provenance

These files bind source policy, tools, native dependencies, commands,
artifacts, and containers for the Accelerate *release* profile. They
are not a signed SLSA/in-toto attestation and not a closed PCPR
release.

- `cpython312/release.provenance.json` is the canonical declared
  binding.
- Exact git commit and tree are recorded on the PCPR-054 receipt
  `current_tree_binding` because nested admission rewrites HEAD.
- This document never pins `origin/main` as the release identity.
- Wheel, sdist, and container digests stay typed unavailable until a
  published artifact is admitted. Native CUDA/IPFS/DuckDB tools stay
  typed unavailable when absent from the sealed PATH.

Sealed validation PATH is exactly `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin`.
"""


class SbomProvenanceError(Exception):
    """Fail-closed PCPR-054 contract error."""


class SbomProvenanceAdmissionError(SbomProvenanceError):
    """Raised when an SBOM or provenance input is rejected."""


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SbomProvenanceError(f"{name} must be a non-empty string")
    return value.strip()


def _kind(value: Any, name: str) -> str:
    kind = _text(value, name)
    if kind not in EVIDENCE_KINDS:
        raise SbomProvenanceError(f"{name} is not an admitted evidence kind")
    return kind


def _reject_closed_release_value(value: Any, name: str) -> None:
    if isinstance(value, str) and value in CLOSED_RELEASE_OUTCOMES:
        raise SbomProvenanceError(
            f"{name} must not be a closed PCPR release outcome"
        )


def _which_sealed(name: str) -> str:
    path = shutil.which(name, path=SEALED_PATH)
    if not path:
        return "unavailable"
    resolved = Path(path)
    if not resolved.is_file():
        return "unavailable"
    return str(resolved)


def _atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    tmp.replace(path)


def parse_declared_spec(spec: str) -> dict[str, Any]:
    raw = _text(spec, "declared_spec")
    match = _PEP508_NAME_RE.match(raw)
    if match is None:
        raise SbomProvenanceAdmissionError(
            f"unparseable PEP 508 spec: {raw!r}"
        )
    name = match.group(1)
    rest = raw[len(name) :]
    extras: list[str] = []
    if rest.startswith("["):
        end = rest.find("]")
        if end < 0:
            raise SbomProvenanceAdmissionError(
                f"unclosed extras in PEP 508 spec: {raw!r}"
            )
        extras = [
            item.strip() for item in rest[1:end].split(",") if item.strip()
        ]
        rest = rest[end + 1 :]
    marker = None
    if ";" in rest:
        rest, marker_text = rest.split(";", 1)
        marker = marker_text.strip() or None
        rest = rest.strip()
    constraint = rest.strip() or None
    return {
        "name": name,
        "declared_spec": raw,
        "extras": extras,
        "marker": marker,
        "version_constraint": constraint,
        "version": {
            "status": "unavailable",
            "evidence_kind": "unavailable",
            "live": False,
            "reason": (
                "The release profile records a PEP 508 range, not a "
                "resolved version."
            ),
        },
        "hash": {
            "status": "unavailable",
            "evidence_kind": "unavailable",
            "live": False,
            "invented": False,
            "reason": "Package hashes were not invented.",
        },
        "license": {
            "status": "unavailable",
            "concluded": "NOASSERTION",
            "declared": "NOASSERTION",
            "evidence_kind": "unavailable",
            "live": False,
            "reason": "Dependency licenses were not scanned.",
        },
    }


def spdx_package_id(name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9.-]+", "-", name).strip("-")
    return f"SPDXRef-Package-{normalized}"


def observe_sbom_tooling() -> dict[str, Any]:
    """Measure sealed-PATH SBOM and provenance tools. Missing stay unavailable."""

    env = observe_sealed_validation_environment()
    scanners = {name.replace("-", "_"): _which_sealed(name) for name in SBOM_SCANNER_TOOLS}
    attest = {name.replace("-", "_"): _which_sealed(name) for name in PROVENANCE_TOOLS}
    return {
        **env,
        "sbom_scanners": scanners,
        "provenance_attestors": attest,
        "any_live_sbom_scanner": any(
            path != "unavailable" for path in scanners.values()
        ),
        "any_live_provenance_attestor": any(
            path != "unavailable" for path in attest.values()
        ),
        "evidence_kind": "measured",
    }


def parse_pyproject_sbom_table(text: str) -> dict[str, Any]:
    table = tomllib.loads(_text(text, "pyproject.toml"))
    if not isinstance(table, dict):
        raise SbomProvenanceError("pyproject.toml must be a table")
    tool = table.get("tool")
    payload: dict[str, Any] = {}
    if isinstance(tool, dict):
        accelerate = tool.get("ipfs_accelerate_py")
        if isinstance(accelerate, dict):
            raw = accelerate.get("sbom-and-provenance")
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
class SbomProvenanceVerdict:
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
    hashes_invented: bool
    files_analyzed: bool
    live_scanner_qualified: bool
    live_scanner_evidence_kind: str
    slsa_attestation_live: bool
    slsa_attestation_evidence_kind: str
    mutable_main_reference: bool
    simulated_results_represented_as_live: bool
    this_task_created_competing_authority: bool
    probes: tuple[OutcomeProbe, ...]
    blockers: tuple[str, ...]
    verdict_cid: str
    sbom_cid: str
    provenance_cid: str
    lock_cid: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.interface,
            "verdict_cid": self.verdict_cid,
            "sbom_cid": self.sbom_cid,
            "provenance_cid": self.provenance_cid,
            "lock_cid": self.lock_cid,
            "promotion_status": self.promotion_status,
            "supervisor_disposition": self.supervisor_disposition,
            "closed_release_outcome": self.closed_release_outcome,
            "release_claim": self.release_claim,
            "completion_authoritative": self.completion_authoritative,
            "contracts_frozen": self.contracts_frozen,
            "duckdb_or_quack_state_written": self.duckdb_or_quack_state_written,
            "sibling_source_required": self.sibling_source_required,
            "hashes_invented": self.hashes_invented,
            "files_analyzed": self.files_analyzed,
            "live_scanner_qualified": self.live_scanner_qualified,
            "live_scanner_evidence_kind": self.live_scanner_evidence_kind,
            "slsa_attestation_live": self.slsa_attestation_live,
            "slsa_attestation_evidence_kind": self.slsa_attestation_evidence_kind,
            "mutable_main_reference": self.mutable_main_reference,
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


def _direct_packages(lock: Mapping[str, Any]) -> list[dict[str, Any]]:
    specs = lock.get("direct_requirements") or []
    if not isinstance(specs, list):
        raise SbomProvenanceError("direct_requirements must be a list")
    return [parse_declared_spec(str(spec)) for spec in specs]


def render_declared_sbom(root: Path | None = None) -> dict[str, Any]:
    package_root = root or discover_accelerate_root()
    if package_root is None:
        raise SbomProvenanceError("Accelerate package root was not found")
    lock = render_release_lock(package_root)
    packages = _direct_packages(lock)
    tooling = observe_sbom_tooling()
    root_spdx_id = spdx_package_id(PACKAGE_NAME)
    spdx_packages = [
        {
            "SPDXID": root_spdx_id,
            "name": PACKAGE_NAME,
            "versionInfo": PACKAGE_VERSION,
            "downloadLocation": f"git+{SOURCE_ORIGIN_URL}",
            "filesAnalyzed": False,
            "licenseConcluded": ROOT_SPDX_LICENSE,
            "licenseDeclared": ROOT_SPDX_LICENSE,
            "supplier": "Organization: endomorphosis",
            "externalRefs": [
                {
                    "referenceCategory": "PACKAGE-MANAGER",
                    "referenceType": "purl",
                    "referenceLocator": (
                        f"pkg:pypi/{PACKAGE_NAME}@{PACKAGE_VERSION}"
                    ),
                }
            ],
            "comment": (
                "Root package of the declared release profile. Files were "
                "not analyzed. This is not a live scanner result."
            ),
        }
    ]
    relationships = [
        {
            "spdxElementId": "SPDXRef-DOCUMENT",
            "relationshipType": "DESCRIBES",
            "relatedSpdxElement": root_spdx_id,
        }
    ]
    envelope_packages = []
    for item in packages:
        spdx_id = spdx_package_id(item["name"])
        spdx_packages.append(
            {
                "SPDXID": spdx_id,
                "name": item["name"],
                "versionInfo": "NOASSERTION",
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": False,
                "licenseConcluded": "NOASSERTION",
                "licenseDeclared": "NOASSERTION",
                "supplier": "NOASSERTION",
                "comment": (
                    "Declared PEP 508 spec from the PCPR-053 release lock: "
                    f"{item['declared_spec']}. Hash, resolved version, and "
                    "license were not invented."
                ),
            }
        )
        relationships.append(
            {
                "spdxElementId": root_spdx_id,
                "relationshipType": "DEPENDS_ON",
                "relatedSpdxElement": spdx_id,
            }
        )
        envelope_packages.append(item)
    scanners_unavailable = {
        name: typed_unavailable(
            reason=f"{name} is absent from the sealed PATH."
        )
        for name, path in tooling["sbom_scanners"].items()
        if path == "unavailable"
    }
    document = {
        "schema": SBOM_SCHEMA,
        "interface": INTERFACE,
        "task_id": PCPR_054_TASK_ID,
        "goal_id": PCPR_054_GOAL_ID,
        "program_id": PCPR_PROGRAM_ID,
        "package_name": PACKAGE_NAME,
        "package_version": PACKAGE_VERSION,
        "python_requires": CANONICAL_PYTHON_REQUIRES,
        "requirements_authority": RUNTIME_REQUIRES_AUTHORITY,
        "sbom_kind": SBOM_KIND,
        "lock_kind": LOCK_KIND,
        "lock_cid": lock["lock_cid"],
        "spdx_version": SPDX_VERSION,
        "files_analyzed": False,
        "hashes_invented": False,
        "live": False,
        "live_scanner_qualified": False,
        "release_claim": False,
        "closed_release_outcome": None,
        "sibling_source_required": False,
        "named_git_extras": list(NAMED_GIT_EXTRAS),
        "named_git_extras_role": NAMED_GIT_EXTRAS_ROLE,
        "root": {
            "name": PACKAGE_NAME,
            "version": PACKAGE_VERSION,
            "license_declared": ROOT_SPDX_LICENSE,
            "source_repository": SOURCE_REPOSITORY,
            "mutable_main_reference": False,
        },
        "direct_package_count": len(envelope_packages),
        "direct_packages": envelope_packages,
        "transitive_packages": typed_unavailable(
            reason=(
                "Transitive resolution was not admitted. A transitive SBOM "
                "graph was not invented."
            )
        ),
        "native_dependencies": lock.get("native_dependencies"),
        "artifacts": lock.get("artifacts"),
        "scanners": {
            "status": "unavailable",
            "evidence_kind": "unavailable",
            "live": False,
            "reason": (
                "Syft, CycloneDX, cdxgen, Trivy, and Grype are absent from "
                "the sealed PATH. A scanner SBOM was not invented."
            ),
            "tools": scanners_unavailable,
        },
        "cyclonedx": typed_unavailable(
            reason=(
                "cyclonedx-bom and cdxgen are absent from the sealed PATH. "
                "A CycloneDX document was not invented."
            )
        ),
        "spdx": {
            "spdxVersion": SPDX_VERSION,
            "dataLicense": "CC0-1.0",
            "SPDXID": "SPDXRef-DOCUMENT",
            "name": f"{PACKAGE_NAME}-{PACKAGE_VERSION}-declared-release-profile",
            "documentNamespace": (
                f"urn:pcpr:declared-sbom:{PACKAGE_NAME}:{PACKAGE_VERSION}"
            ),
            "creationInfo": {
                "created": SPDX_CREATED,
                "creators": [f"Tool: {INTERFACE}"],
                "comment": (
                    "created is SOURCE_DATE_EPOCH=0. It is not a live scan "
                    "timestamp."
                ),
            },
            "packages": spdx_packages,
            "relationships": relationships,
        },
        "source_date_epoch": SOURCE_DATE_EPOCH,
        "evidence_kind": "measured",
    }
    document["sbom_cid"] = content_identity(
        {key: value for key, value in document.items() if key != "sbom_cid"}
    )
    return document


def render_build_provenance(
    root: Path | None = None,
    *,
    sbom: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    package_root = root or discover_accelerate_root()
    if package_root is None:
        raise SbomProvenanceError("Accelerate package root was not found")
    lock = render_release_lock(package_root)
    document_sbom = dict(sbom or render_declared_sbom(package_root))
    tooling = observe_sbom_tooling()
    intended_build = (
        f"PATH={SEALED_PATH} SOURCE_DATE_EPOCH={SOURCE_DATE_EPOCH} "
        f"{SEALED_PYTHON} -m build --wheel --sdist"
    )
    document = {
        "schema": PROVENANCE_SCHEMA,
        "interface": INTERFACE,
        "task_id": PCPR_054_TASK_ID,
        "goal_id": PCPR_054_GOAL_ID,
        "program_id": PCPR_PROGRAM_ID,
        "package_name": PACKAGE_NAME,
        "package_version": PACKAGE_VERSION,
        "provenance_kind": PROVENANCE_KIND,
        "predicate_type": "https://slsa.dev/provenance/v1",
        "lock_cid": lock["lock_cid"],
        "sbom_cid": document_sbom["sbom_cid"],
        "source": {
            "kind": "git",
            "repository": SOURCE_REPOSITORY,
            "origin_url": SOURCE_ORIGIN_URL,
            "binding": "current_head_not_mutable_main",
            "mutable_main_reference": False,
            "commit": {
                "status": "observed_at_evaluation",
                "evidence_kind": "measured",
                "live": False,
                "field": "PCPR-054 receipt current_tree_binding",
                "reason": (
                    "Exact commit and tree are bound by the task receipt "
                    "because nested admission rewrites HEAD. This document "
                    "does not pin origin/main."
                ),
            },
        },
        "tools": {
            "python": tooling.get("python3_12", "unavailable"),
            "python_version": tooling.get("python_version", "unavailable"),
            "git": tooling.get("git_path", "unavailable"),
            "pip": tooling.get("pip_path", "unavailable"),
            "setuptools": tooling.get("setuptools_module", "unavailable"),
            "wheel": tooling.get("wheel_module", "unavailable"),
            "build": tooling.get("build_module", "unavailable"),
            "uv": tooling.get("uv_path", "unavailable"),
            "sbom_scanners": tooling.get("sbom_scanners"),
            "provenance_attestors": tooling.get("provenance_attestors"),
            "sealed_path": SEALED_PATH,
            "sealed_python": SEALED_PYTHON,
            "source_date_epoch": SOURCE_DATE_EPOCH,
            "evidence_kind": "measured",
        },
        "native_dependencies": lock.get("native_dependencies"),
        "commands": {
            "materialize": (
                'python -c "from ipfs_accelerate_py.assurance.'
                "sbom_and_provenance import write_sbom_and_provenance_files; "
                'write_sbom_and_provenance_files()"'
            ),
            "intended_build": intended_build,
            "observed_build": "unavailable",
            "syft_scan": "unavailable",
            "cyclonedx_scan": "unavailable",
            "slsa_attest": "unavailable",
            "cosign_sign": "unavailable",
        },
        "artifacts": {
            "wheel": typed_unavailable(
                reason=(
                    "No published wheel digest is bound. Hermetic private "
                    "builds are not a PCPR release."
                )
            ),
            "sdist": typed_unavailable(
                reason=(
                    "No published sdist digest is bound. Hermetic private "
                    "builds are not a PCPR release."
                )
            ),
            "container": typed_unavailable(
                reason="No container digest is bound by this declared provenance."
            ),
            "sbom": {
                "path": f"{SBOM_DIR_RELPATH}/{SBOM_JSON_NAME}",
                "sbom_cid": document_sbom["sbom_cid"],
                "status": "observed",
                "live": False,
                "evidence_kind": "measured",
            },
        },
        "builder": {
            "id": "urn:pcpr:builder:declared-release-profile@1",
            "live": False,
            "signed": False,
            "reason": "This builder identity is declared R&D, not a live SLSA builder.",
        },
        "slsa": {
            "status": "unavailable",
            "level": "unavailable",
            "signed": False,
            "live": False,
            "evidence_kind": "unavailable",
            "reason": (
                "slsa-generator, in-toto, and cosign are absent from the "
                "sealed PATH. A signed SLSA attestation was not invented."
            ),
        },
        "live": False,
        "release_claim": False,
        "closed_release_outcome": None,
        "hashes_invented": False,
        "evidence_kind": "measured",
    }
    document["provenance_cid"] = content_identity(
        {
            key: value
            for key, value in document.items()
            if key != "provenance_cid"
        }
    )
    return document


def artifact_paths(root: Path) -> dict[str, Path]:
    return {
        "sbom": root / SBOM_DIR_RELPATH / SBOM_JSON_NAME,
        "provenance": root / PROVENANCE_DIR_RELPATH / PROVENANCE_JSON_NAME,
        "sbom_readme": root / SBOM_README_RELPATH,
        "provenance_readme": root / PROVENANCE_README_RELPATH,
        "catalog": root / CATALOG_RELPATH,
    }


def write_sbom_and_provenance_files(start: Path | None = None) -> dict[str, Any]:
    root = discover_accelerate_root(start)
    if root is None:
        raise SbomProvenanceError("Accelerate package root was not found")
    sbom = render_declared_sbom(root)
    provenance = render_build_provenance(root, sbom=sbom)
    paths = artifact_paths(root)
    _atomic_write(paths["sbom"], pretty_json(sbom))
    _atomic_write(paths["provenance"], pretty_json(provenance))
    sbom_readme = SBOM_README if SBOM_README.endswith("\n") else SBOM_README + "\n"
    provenance_readme = (
        PROVENANCE_README
        if PROVENANCE_README.endswith("\n")
        else PROVENANCE_README + "\n"
    )
    _atomic_write(paths["sbom_readme"], sbom_readme)
    _atomic_write(paths["provenance_readme"], provenance_readme)
    catalog = platform_sbom_catalog(start)
    _atomic_write(paths["catalog"], pretty_json(catalog))
    return {
        "sbom": sbom,
        "provenance": provenance,
        "catalog": catalog,
        "paths": {name: str(path) for name, path in paths.items()},
    }


def verify_sbom_and_provenance_files(start: Path | None = None) -> dict[str, Any]:
    root = discover_accelerate_root(start)
    if root is None:
        raise SbomProvenanceError("Accelerate package root was not found")
    sbom = render_declared_sbom(root)
    provenance = render_build_provenance(root, sbom=sbom)
    paths = artifact_paths(root)
    missing: list[str] = []
    sbom_ok = False
    provenance_ok = False
    sbom_readme_ok = False
    provenance_readme_ok = False
    expected_sbom_readme = (
        SBOM_README if SBOM_README.endswith("\n") else SBOM_README + "\n"
    )
    expected_provenance_readme = (
        PROVENANCE_README
        if PROVENANCE_README.endswith("\n")
        else PROVENANCE_README + "\n"
    )
    for name, path in paths.items():
        if name == "catalog":
            continue
        if not path.is_file():
            missing.append(name)
            continue
        if name == "sbom":
            sbom_ok = json.loads(path.read_text(encoding="utf-8")) == sbom
        elif name == "provenance":
            provenance_ok = (
                json.loads(path.read_text(encoding="utf-8")) == provenance
            )
        elif name == "sbom_readme":
            sbom_readme_ok = path.read_text(encoding="utf-8") == expected_sbom_readme
        elif name == "provenance_readme":
            provenance_readme_ok = (
                path.read_text(encoding="utf-8") == expected_provenance_readme
            )
    return {
        "ok": (
            not missing
            and sbom_ok
            and provenance_ok
            and sbom_readme_ok
            and provenance_readme_ok
        ),
        "missing": missing,
        "sbom_ok": sbom_ok,
        "provenance_ok": provenance_ok,
        "sbom_readme_ok": sbom_readme_ok,
        "provenance_readme_ok": provenance_readme_ok,
        "sbom_cid": sbom["sbom_cid"],
        "provenance_cid": provenance["provenance_cid"],
        "lock_cid": sbom["lock_cid"],
        "sbom_sha256": (
            sha256_bytes(paths["sbom"].read_bytes())
            if paths["sbom"].is_file()
            else "unavailable"
        ),
        "provenance_sha256": (
            sha256_bytes(paths["provenance"].read_bytes())
            if paths["provenance"].is_file()
            else "unavailable"
        ),
    }


def _sibling_sbom_identity(path: Path, relative_path: str) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": relative_path,
            "status": "unavailable",
            "evidence_kind": "unavailable",
            "reason": "Sibling SBOM file is not present beside this checkout.",
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": relative_path,
        "status": "observed",
        "evidence_kind": "measured",
        "package_name": payload.get("package_name"),
        "package_version": payload.get("package_version"),
        "sbom_kind": payload.get("sbom_kind"),
        "sbom_cid": payload.get("sbom_cid"),
        "lock_cid": payload.get("lock_cid"),
        "direct_package_count": payload.get("direct_package_count"),
        "hashes_invented": payload.get("hashes_invented"),
        "files_analyzed": payload.get("files_analyzed"),
        "live": False,
        "sha256": sha256_bytes(path.read_bytes()),
    }


def _sibling_provenance_identity(path: Path, relative_path: str) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": relative_path,
            "status": "unavailable",
            "evidence_kind": "unavailable",
            "reason": (
                "Sibling provenance file is not present beside this checkout."
            ),
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": relative_path,
        "status": "observed",
        "evidence_kind": "measured",
        "package_name": payload.get("package_name"),
        "package_version": payload.get("package_version"),
        "provenance_kind": payload.get("provenance_kind"),
        "provenance_cid": payload.get("provenance_cid"),
        "sbom_cid": payload.get("sbom_cid"),
        "slsa_status": (
            payload.get("slsa", {}).get("status")
            if isinstance(payload.get("slsa"), dict)
            else "unavailable"
        ),
        "live": False,
        "sha256": sha256_bytes(path.read_bytes()),
    }


def platform_sbom_catalog(start: Path | None = None) -> dict[str, Any]:
    """Bind Datasets, Kit, and Accelerate SBOM and provenance identities."""

    root = discover_accelerate_root(start)
    if root is None:
        raise SbomProvenanceError("Accelerate package root was not found")
    parent = root.parent
    sbom = render_declared_sbom(root)
    provenance = render_build_provenance(root, sbom=sbom)
    datasets_sbom = _sibling_sbom_identity(
        parent / "ipfs_datasets" / SBOM_DIR_RELPATH / SBOM_JSON_NAME,
        relative_path=f"../ipfs_datasets/{SBOM_DIR_RELPATH}/{SBOM_JSON_NAME}",
    )
    datasets_provenance = _sibling_provenance_identity(
        parent / "ipfs_datasets" / PROVENANCE_DIR_RELPATH / PROVENANCE_JSON_NAME,
        relative_path=(
            f"../ipfs_datasets/{PROVENANCE_DIR_RELPATH}/{PROVENANCE_JSON_NAME}"
        ),
    )
    kit_sbom = _sibling_sbom_identity(
        parent / "ipfs_kit" / SBOM_DIR_RELPATH / SBOM_JSON_NAME,
        relative_path=f"../ipfs_kit/{SBOM_DIR_RELPATH}/{SBOM_JSON_NAME}",
    )
    kit_provenance = _sibling_provenance_identity(
        parent / "ipfs_kit" / PROVENANCE_DIR_RELPATH / PROVENANCE_JSON_NAME,
        relative_path=f"../ipfs_kit/{PROVENANCE_DIR_RELPATH}/{PROVENANCE_JSON_NAME}",
    )
    catalog = {
        "schema": CATALOG_SCHEMA,
        "interface": CATALOG_INTERFACE,
        "task_id": PCPR_054_TASK_ID,
        "goal_id": PCPR_054_GOAL_ID,
        "sbom_kind": SBOM_KIND,
        "provenance_kind": PROVENANCE_KIND,
        "components": {
            "ipfs_accelerate_py": {
                "sbom_path": f"{SBOM_DIR_RELPATH}/{SBOM_JSON_NAME}",
                "provenance_path": f"{PROVENANCE_DIR_RELPATH}/{PROVENANCE_JSON_NAME}",
                "status": "observed",
                "evidence_kind": "measured",
                "package_name": PACKAGE_NAME,
                "package_version": PACKAGE_VERSION,
                "sbom_kind": SBOM_KIND,
                "provenance_kind": PROVENANCE_KIND,
                "sbom_cid": sbom["sbom_cid"],
                "provenance_cid": provenance["provenance_cid"],
                "lock_cid": sbom["lock_cid"],
                "direct_package_count": sbom["direct_package_count"],
                "hashes_invented": False,
                "files_analyzed": False,
                "live": False,
            },
            "ipfs_datasets_py": {
                "sbom": datasets_sbom,
                "provenance": datasets_provenance,
            },
            "ipfs_kit_py": {
                "sbom": kit_sbom,
                "provenance": kit_provenance,
            },
        },
        "mutable_main_reference": False,
        "hashes_invented": False,
        "live_scanner_qualified": False,
        "slsa_attestation_live": False,
        "closed_release_outcome": None,
        "release_claim": False,
        "evidence_kind": "measured",
    }
    catalog["catalog_cid"] = content_identity(
        {key: value for key, value in catalog.items() if key != "catalog_cid"}
    )
    return catalog


def current_head_static_probes(
    start: Path | None = None,
) -> tuple[OutcomeProbe, ...]:
    root = discover_accelerate_root(start)
    if root is None:
        raise SbomProvenanceError("Accelerate package root was not found")
    lock = render_release_lock(root)
    sbom = render_declared_sbom(root)
    provenance = render_build_provenance(root, sbom=sbom)
    verified = verify_sbom_and_provenance_files(root)
    table = parse_pyproject_sbom_table(
        (root / "pyproject.toml").read_text(encoding="utf-8")
    )
    tooling = observe_sbom_tooling()
    lock_names = [parse_declared_spec(str(spec))["name"] for spec in lock["direct_requirements"]]
    sbom_names = [item["name"] for item in sbom["direct_packages"]]
    mutable_main = bool(
        sbom["root"]["mutable_main_reference"]
        or provenance["source"]["mutable_main_reference"]
    )
    probes = [
        _probe(
            "sbom_files_match_generator",
            verified.get("ok") is True,
            reason=(
                "Committed SBOM and provenance files match the generator."
                if verified.get("ok") is True
                else "Committed SBOM or provenance files are missing or drift."
            ),
            details=verified,
        ),
        _probe(
            "provenance_files_match_generator",
            verified.get("ok") is True and verified.get("provenance_ok") is True,
            reason=(
                "Committed provenance matches the declared-build generator."
                if verified.get("provenance_ok") is True
                else "Committed provenance is missing or drifts."
            ),
        ),
        _probe(
            "pyproject_sbom_and_provenance_table",
            table.get("interface") == INTERFACE
            and table.get("schema") == SCHEMA
            and table.get("task-id") == PCPR_054_TASK_ID
            and table.get("sbom-kind") == SBOM_KIND,
            reason=(
                "pyproject.toml declares AccelerateSbomAndProvenance@1."
                if table.get("interface") == INTERFACE
                else "pyproject.toml does not declare AccelerateSbomAndProvenance@1."
            ),
            details={"table": table},
        ),
        _probe(
            "hashes_not_invented",
            sbom["hashes_invented"] is False
            and provenance["hashes_invented"] is False,
            reason="Hashes were not invented; hash identity stays typed unavailable.",
        ),
        _probe(
            "hashes_invented",
            False,
            reason="Hashes must not be invented for an unavailable resolver.",
        ),
        _probe(
            "files_not_analyzed",
            sbom["files_analyzed"] is False
            and sbom["spdx"]["packages"][0]["filesAnalyzed"] is False,
            reason="SPDX filesAnalyzed remains false; a file inventory was not invented.",
        ),
        _probe(
            "files_analyzed",
            False,
            reason="A file-analyzed SBOM was not produced.",
        ),
        _probe(
            "no_mutable_main_reference",
            mutable_main is False,
            reason="SBOM and provenance do not pin origin/main as the release identity.",
        ),
        _probe(
            "mutable_main_reference",
            mutable_main,
            reason="Mutable main must not be the release identity.",
        ),
        _probe(
            "sbom_kind_is_declared_release_profile",
            sbom["sbom_kind"] == SBOM_KIND,
            reason="SBOM kind is declared_release_profile, not a live scanner result.",
        ),
        _probe(
            "provenance_kind_is_declared_build_binding",
            provenance["provenance_kind"] == PROVENANCE_KIND,
            reason="Provenance kind is declared_build_binding, not a signed SLSA attestation.",
        ),
        _probe(
            "named_git_extras_are_not_sbom",
            True,
            reason=(
                "Named libp2p, mcp-p2p, ipfs-transformers, and ipfs-model-manager "
                "extras may carry source-checkout Git pins; they are not this SBOM."
            ),
            details={
                "named_git_extras": list(NAMED_GIT_EXTRAS),
                "named_git_extras_role": NAMED_GIT_EXTRAS_ROLE,
            },
        ),
        _probe(
            "direct_packages_match_lock",
            lock_names == sbom_names
            and sbom["direct_package_count"] == lock["direct_requirement_count"],
            reason="Declared SBOM packages are exactly the PCPR-053 release-lock specs.",
            details={
                "lock_names": lock_names,
                "sbom_names": sbom_names,
                "lock_cid": lock["lock_cid"],
            },
        ),
        _probe(
            "simulated_results_represented_as_live",
            False,
            reason="Simulated results are not represented as live.",
        ),
        _probe(
            "hashes_represented_as_live",
            False,
            reason="Unavailable hashes are not represented as live.",
        ),
        _probe(
            "live_scanner_represented_as_live",
            False,
            reason="No Syft/CycloneDX/Trivy scan is represented as live.",
        ),
        _probe(
            "slsa_attestation_represented_as_live",
            False,
            reason="No SLSA/in-toto/cosign attestation is represented as live.",
        ),
        _probe(
            "closed_release_represented_as_live",
            False,
            reason="This task does not publish a PCPR release.",
        ),
        _probe(
            "wheel_represented_as_live_release",
            False,
            reason="No wheel digest is represented as a live published release.",
        ),
        _probe(
            "container_represented_as_live",
            False,
            reason="No container digest is represented as live.",
        ),
        _probe(
            "live_scanner",
            None,
            evidence_kind="unavailable",
            reason=(
                "Syft, CycloneDX, cdxgen, Trivy, and Grype are absent from "
                "the sealed PATH. A live scanner SBOM was not invented."
            ),
            details={"tools": tooling.get("sbom_scanners")},
        ),
        _probe(
            "slsa_attestation",
            None,
            evidence_kind="unavailable",
            reason=(
                "slsa-generator, in-toto, and cosign are absent from the "
                "sealed PATH. A signed attestation was not invented."
            ),
            details={"tools": tooling.get("provenance_attestors")},
        ),
        _probe(
            "published_wheel",
            None,
            evidence_kind="unavailable",
            reason="No published wheel identity is bound by this declared provenance.",
        ),
        _probe(
            "published_sdist",
            None,
            evidence_kind="unavailable",
            reason="No published sdist identity is bound by this declared provenance.",
        ),
        _probe(
            "container_digest",
            None,
            evidence_kind="unavailable",
            reason="No container digest is bound by this declared provenance.",
        ),
    ]
    return tuple(probes)


def qualify_sbom_and_provenance(
    probes: Sequence[OutcomeProbe],
    *,
    sbom_cid: str,
    provenance_cid: str,
    lock_cid: str,
) -> SbomProvenanceVerdict:
    if not probes:
        raise SbomProvenanceError("at least one probe is required")
    normalized: list[OutcomeProbe] = []
    blockers: list[str] = []
    for probe in probes:
        kind = _kind(probe.evidence_kind, "evidence_kind")
        if probe.live and kind != "measured_live":
            raise SbomProvenanceError("live claims require measured_live evidence")
        if probe.simulated_represented_as_live:
            raise SbomProvenanceError(
                "simulated results must not be represented as live"
            )
        normalized.append(probe)
        if probe.probe_id in FORBIDDEN_PRESENT_PROBE_IDS and probe.present is True:
            blockers.append(probe.probe_id)
        if probe.probe_id in REQUIRED_GOOD_PROBE_IDS and probe.present is not True:
            blockers.append(probe.probe_id)

    promotion_status = "rnd_non_promoted"
    _reject_closed_release_value(promotion_status, "promotion_status")
    mutable = next(
        (
            item.present is True
            for item in normalized
            if item.probe_id == "mutable_main_reference"
        ),
        False,
    )
    payload = {
        "schema": VERDICT_SCHEMA,
        "interface": INTERFACE,
        "task_id": PCPR_054_TASK_ID,
        "goal_id": PCPR_054_GOAL_ID,
        "promotion_status": promotion_status,
        "supervisor_disposition": "supervisor_non_promoted",
        "closed_release_outcome": None,
        "release_claim": False,
        "completion_authoritative": False,
        "contracts_frozen": False,
        "duckdb_or_quack_state_written": False,
        "sibling_source_required": False,
        "hashes_invented": False,
        "files_analyzed": False,
        "live_scanner_qualified": False,
        "live_scanner_evidence_kind": "unavailable",
        "slsa_attestation_live": False,
        "slsa_attestation_evidence_kind": "unavailable",
        "mutable_main_reference": mutable,
        "simulated_results_represented_as_live": False,
        "this_task_created_competing_authority": False,
        "probes": [item.to_mapping() for item in normalized],
        "blockers": list(dict.fromkeys(blockers)),
        "sbom_cid": sbom_cid,
        "provenance_cid": provenance_cid,
        "lock_cid": lock_cid,
    }
    return SbomProvenanceVerdict(
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
        hashes_invented=False,
        files_analyzed=False,
        live_scanner_qualified=False,
        live_scanner_evidence_kind="unavailable",
        slsa_attestation_live=False,
        slsa_attestation_evidence_kind="unavailable",
        mutable_main_reference=mutable,
        simulated_results_represented_as_live=False,
        this_task_created_competing_authority=False,
        probes=tuple(normalized),
        blockers=tuple(dict.fromkeys(blockers)),
        verdict_cid=content_identity(payload),
        sbom_cid=sbom_cid,
        provenance_cid=provenance_cid,
        lock_cid=lock_cid,
    )


def qualify_current_head_sbom_and_provenance(
    start: Path | None = None,
) -> SbomProvenanceVerdict:
    root = discover_accelerate_root(start)
    sbom = render_declared_sbom(root)
    provenance = render_build_provenance(root, sbom=sbom)
    return qualify_sbom_and_provenance(
        current_head_static_probes(start),
        sbom_cid=str(sbom["sbom_cid"]),
        provenance_cid=str(provenance["provenance_cid"]),
        lock_cid=str(sbom["lock_cid"]),
    )


def pcpr_054_receipt_promotion(
    verdict: SbomProvenanceVerdict,
) -> dict[str, Any]:
    if verdict.closed_release_outcome is not None:
        raise SbomProvenanceError(
            "sbom and provenance must not mint a closed release outcome"
        )
    if verdict.release_claim:
        raise SbomProvenanceError("sbom and provenance must not claim a PCPR release")
    if verdict.completion_authoritative:
        raise SbomProvenanceError("sbom-and-provenance completion is not authoritative")
    if verdict.duckdb_or_quack_state_written:
        raise SbomProvenanceError(
            "sbom and provenance must not write DuckDB or Quack state"
        )
    if verdict.promotion_status in CLOSED_RELEASE_OUTCOMES:
        raise SbomProvenanceError(
            "promotion_status must not be a closed release outcome"
        )
    return verdict.to_mapping()


# Pinned identity of the ordinary current-head static verdict. Drift means
# the default payload changed and the outer receipt must be regenerated.
CURRENT_HEAD_NON_PROMOTION_VERDICT_CID: Final = (
    "baguqeeragtqz4oqmhi6a6jipn46gp3hu6zbmx2qgsb7ubwfy6vynda5ml22a"
)


__all__ = [
    "CLOSED_RELEASE_OUTCOMES",
    "CURRENT_HEAD_NON_PROMOTION_VERDICT_CID",
    "HERMETIC_CANDIDATE_SUITES",
    "INTERFACE",
    "OutcomeProbe",
    "PCPR_054_GOAL_ID",
    "PCPR_054_TASK_ID",
    "PROVENANCE_KIND",
    "SBOM_KIND",
    "SCHEMA",
    "SEALED_PATH",
    "SEALED_PYTHON",
    "SbomProvenanceError",
    "SbomProvenanceVerdict",
    "current_head_static_probes",
    "parse_declared_spec",
    "pcpr_054_receipt_promotion",
    "platform_sbom_catalog",
    "qualify_current_head_sbom_and_provenance",
    "qualify_sbom_and_provenance",
    "render_build_provenance",
    "render_declared_sbom",
    "verify_sbom_and_provenance_files",
    "write_sbom_and_provenance_files",
]
