"""Fail-closed PCPR-056 Accelerate portfolio compatibility lock.

Produce one declared proof-carrying-platform-0.1.0 compatibility lock
that binds Accelerate, Datasets, and Kit package versions, intended
tags, PCPR-053/054/055 artifact CIDs, PCPR-040/041/042/043 contract and
vector identities, the one supported combination, hermetic candidate
test suites, and named qualification-receipt identities.

This is the lock (lock=True). It is not a freeze (PCPR-002), not a live
signed Git tag or cosign signature, not branch protection (PCPR-057),
and not a closed PCPR release. Live signatures, published
wheel/sdist/container identities, and live supervisor/CUDA/solver
qualification stay typed unavailable. Signatures and unpublished
artifact hashes are never invented. Exact commit and tree are bound by
the PCPR-056 receipt current_tree_binding because nested admission
rewrites HEAD. origin/main is not the release identity.

This module does not import sibling Datasets or Kit packages. Sibling
packaging files, when present beside this checkout, may be observed to
confirm the pinned identities; they are never required.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from ipfs_accelerate_py.assurance.canonical_byte_cid_vectors import (
    PINNED_CONTRACT_VECTORS,
    PINNED_VECTOR_DOCUMENT_CID as PCPR_041_VECTOR_DOCUMENT_CID,
)
from ipfs_accelerate_py.assurance.cross_repository_compatibility import (
    DATASETS_OWNED_CONTRACTS,
    INCOMPATIBLE_CATEGORIES,
    KIT_OWNED_CONTRACTS,
    PINNED_CATALOG_CID,
    PINNED_COMPATIBILITY_DOCUMENT_CID,
    REQUIRED_REPOSITORIES,
    SUPPORTED_COMBINATION_ID,
    SUPPORTED_LANGUAGES,
    PINNED_042_NEGATIVE_DOCUMENT_CID,
)
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
from ipfs_accelerate_py.assurance.python_compatibility import (
    PYTHON_FLOOR,
    REQUIRES_PYTHON,
)
from ipfs_accelerate_py.assurance.sbom_and_provenance import (
    render_build_provenance,
    render_declared_sbom,
)
from ipfs_accelerate_py.assurance.shared_contracts import (
    CONTRACT_AUTHORITIES,
    CONTRACT_SCHEMA_IDS,
    REQUIRED_CONTRACT_NAMES,
)
from ipfs_accelerate_py.assurance.signed_tags_and_artifacts import (
    INTENDED_TAG_NAME,
    observe_signing_tooling,
    render_declared_checksums,
    render_declared_tag_policy,
)

INTERFACE: Final = "AcceleratePortfolioCompatibilityLock@1"
SCHEMA: Final = "ipfs_accelerate_py/assurance/portfolio-compatibility-lock@1"
LOCK_SCHEMA: Final = (
    "ipfs_accelerate_py/assurance/declared-portfolio-compatibility-lock@1"
)
VERDICT_SCHEMA: Final = (
    "ipfs_accelerate_py/assurance/portfolio-compatibility-lock-verdict@1"
)
CATALOG_SCHEMA: Final = (
    "ipfs_accelerate_py/assurance/platform-portfolio-compatibility-lock-catalog@1"
)
CATALOG_INTERFACE: Final = "PlatformPortfolioCompatibilityLock@1"
PCPR_056_TASK_ID: Final = "PCPR-056"
PCPR_056_GOAL_ID: Final = "PCPR-G600"
PCPR_055_TASK_ID: Final = "PCPR-055"
PCPR_043_TASK_ID: Final = "PCPR-043"
PCPR_002_TASK_ID: Final = "PCPR-002"
PCPR_001_TASK_ID: Final = "PCPR-001"
PCPR_057_TASK_ID: Final = "PCPR-057"
PCPR_PROGRAM_ID: Final = "proof-carrying-platform-qualification-and-release-v1"
PCPR_BOARD_NAMESPACE: Final = (
    "proof-carrying-platform-qualification-and-release-v1"
)
EVIDENCE_ID: Final = "pcpr/portfolio-compatibility-lock@1"

CANONICAL_PYTHON_REQUIRES: Final = ">=3.12"
LOCK_KIND: Final = "declared_portfolio_compatibility_lock"
PORTFOLIO_ID: Final = "portfolio:pcpr-v1"
PORTFOLIO_VERSION: Final = "proof-carrying-platform-0.1.0"
SOURCE_REPOSITORY: Final = "endomorphosis/ipfs_accelerate_py"
SOURCE_ORIGIN_URL: Final = "https://github.com/endomorphosis/ipfs_accelerate_py"
LOCK_DIR_RELPATH: Final = "packaging/pcpr/compatibility/cpython312"
LOCK_JSON_NAME: Final = "release.lock.json"
LOCK_README_RELPATH: Final = "packaging/pcpr/compatibility/README.md"
CATALOG_RELPATH: Final = "packaging/pcpr/compatibility/platform-catalog.json"
SOURCE_DATE_EPOCH: Final = "0"

if PINNED_CATALOG_CID != (
    "baguqeeraqpptps2gf55wmthv3sm5nzlxqqdjfyg2ggkbl5ovfzuvvvsxap4a"
):
    raise RuntimeError("PCPR-056 catalog CID remints PCPR-043")
if PCPR_041_VECTOR_DOCUMENT_CID != (
    "baguqeera6gwcyi3f7fkw5eufnhkovmqw7eevkeas4qxds5ws7b7rloic63da"
):
    raise RuntimeError("PCPR-056 vector document CID remints PCPR-041")

PINNED_COMPONENTS: Final[Mapping[str, Mapping[str, Any]]] = MappingProxyType(
    {
        "ipfs_accelerate_py": MappingProxyType(
            {
                "package_name": "ipfs_accelerate_py",
                "package_version": "0.0.45",
                "intended_tag_name": "ipfs_accelerate_py-v0.0.45",
                "lock_cid": (
                    "baguqeerayvfxbdt6m6go57s6dw2afwsvpkzpt25rewc6kglm2fhwkeowvuxa"
                ),
                "sbom_cid": (
                    "baguqeeraotw7oyqehxscvlcyo4udmmdnaxq3yg6rfz4pspxcnpcnbwhd5fiq"
                ),
                "provenance_cid": (
                    "baguqeera5m46bqdbtzywuntklfkbc7zxyn5ihyfr4ckazumexgjs3zuzvifa"
                ),
                "tag_policy_cid": (
                    "baguqeera3b3o6eu5zb4avcvwq667jekuqt5bl7ucb5pa4jumseizgbuqdl5a"
                ),
                "checksums_cid": (
                    "baguqeeraqqhpdtv7s33b3ha6prioigypgczlhvbx3ena24zwt4k5kxcsww5q"
                ),
            }
        ),
        "ipfs_datasets_py": MappingProxyType(
            {
                "package_name": "ipfs_datasets_py",
                "package_version": "0.2.0",
                "intended_tag_name": "ipfs_datasets_py-v0.2.0",
                "lock_cid": (
                    "baguqeeraxwt7j3qjwfy3yvicngqwrnx2suobs3yxa4ah6gbgbszxrbsrid7q"
                ),
                "sbom_cid": (
                    "baguqeerao2zimzk7v7femnhzw5zhlvalaaor6foyvkrca3fneuhpe6ieoneq"
                ),
                "provenance_cid": (
                    "baguqeeraoq7sa7f5vjjauqzgwuz5w4lo4cayvmf3lwiijyurtn6dhtgakeia"
                ),
                "tag_policy_cid": (
                    "baguqeeraubun3du77i6xczjl2eufg4fxq6v7dcmapqjtgutukf5fbl6wi33a"
                ),
                "checksums_cid": (
                    "baguqeerax4bcjukmqipr76vndwafm2vomppsobybg7czameuzllcsgcymkwa"
                ),
            }
        ),
        "ipfs_kit_py": MappingProxyType(
            {
                "package_name": "ipfs_kit_py",
                "package_version": "0.3.0",
                "intended_tag_name": "ipfs_kit_py-v0.3.0",
                "lock_cid": (
                    "baguqeerapndvhdlxynq6afrv4mfpsa7ajjregns3nwuth5hnho7rf3xc223a"
                ),
                "sbom_cid": (
                    "baguqeeracvwc27foqiebwn3bc65riwro2ucvtpadwt53fsg4uuuyh5ec7eza"
                ),
                "provenance_cid": (
                    "baguqeeral6zl7kn2fy6rylk7rs74hzbwulnntq2w2fycoe2cbpq52fjqvgwq"
                ),
                "tag_policy_cid": (
                    "baguqeerazr3ew2kjxvi7tz4eouy6eqsdoszfp4c6tdsvctvueycwjipdgpaa"
                ),
                "checksums_cid": (
                    "baguqeeragjykpkrsny4wzmq6vhk47o3u47w7c4rbp42554lwyzlrdoinxk2a"
                ),
            }
        ),
    }
)

BOUND_QUALIFICATION_RECEIPTS: Final[tuple[Mapping[str, str], ...]] = (
    MappingProxyType(
        {
            "task_id": "PCPR-001",
            "role": "live-supervisor-qualification",
            "promotion_status": "rnd_non_promoted",
            "evidence_kind": "unavailable",
        }
    ),
    MappingProxyType(
        {
            "task_id": "PCPR-043",
            "role": "cross-repository-compatibility",
            "promotion_status": "rnd_non_promoted",
            "evidence_kind": "measured",
        }
    ),
    MappingProxyType(
        {
            "task_id": "PCPR-050",
            "role": "datasets-clean-package",
            "promotion_status": "rnd_non_promoted",
            "evidence_kind": "measured",
        }
    ),
    MappingProxyType(
        {
            "task_id": "PCPR-051",
            "role": "kit-clean-package",
            "promotion_status": "rnd_non_promoted",
            "evidence_kind": "measured",
        }
    ),
    MappingProxyType(
        {
            "task_id": "PCPR-052",
            "role": "accelerate-clean-package",
            "promotion_status": "rnd_non_promoted",
            "evidence_kind": "measured",
        }
    ),
    MappingProxyType(
        {
            "task_id": "PCPR-053",
            "role": "dependency-locks",
            "promotion_status": "rnd_non_promoted",
            "evidence_kind": "measured",
        }
    ),
    MappingProxyType(
        {
            "task_id": "PCPR-054",
            "role": "sbom-and-provenance",
            "promotion_status": "rnd_non_promoted",
            "evidence_kind": "measured",
        }
    ),
    MappingProxyType(
        {
            "task_id": "PCPR-055",
            "role": "signed-tags-and-artifacts",
            "promotion_status": "rnd_non_promoted",
            "evidence_kind": "measured",
        }
    ),
)

HERMETIC_CANDIDATE_SUITES: Final[tuple[str, ...]] = (
    "test/api/test_agent_supervisor_portfolio_compatibility_lock.py",
)

BOUND_TEST_SUITES: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "ipfs_accelerate_py": (
            "test/api/test_agent_supervisor_signed_tags_and_artifacts.py",
            "test/api/test_agent_supervisor_portfolio_compatibility_lock.py",
        ),
        "ipfs_datasets_py": (
            "tests/unit/test_pcpr_055_signed_tags_and_artifacts.py",
            "tests/unit/test_pcpr_056_portfolio_compatibility_lock.py",
        ),
        "ipfs_kit_py": (
            "tests/test_pcpr_055_signed_tags_and_artifacts.py",
            "tests/test_pcpr_056_portfolio_compatibility_lock.py",
        ),
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

REQUIRED_GOOD_PROBE_IDS: Final[frozenset[str]] = frozenset(
    {
        "lock_files_match_generator",
        "pyproject_portfolio_compatibility_lock_table",
        "hashes_not_invented",
        "signatures_not_invented",
        "no_mutable_main_reference",
        "lock_kind_is_declared_portfolio_compatibility_lock",
        "supported_combination_is_locked",
        "pcpr_043_identities_not_reminted",
        "accelerate_artifact_cids_match_pins",
        "named_git_extras_are_not_this_lock",
    }
)
FORBIDDEN_PRESENT_PROBE_IDS: Final[frozenset[str]] = frozenset(
    {
        "hashes_invented",
        "signatures_invented",
        "mutable_main_reference",
        "simulated_results_represented_as_live",
        "signed_lock_represented_as_live",
        "gpg_signature_represented_as_live",
        "cosign_signature_represented_as_live",
        "closed_release_represented_as_live",
        "wheel_represented_as_live_release",
        "compatibility_identities_reminted",
    }
)

LOCK_README: Final = """# PCPR-056 declared portfolio compatibility lock

These files are the *one* declared proof-carrying-platform-0.1.0
portfolio compatibility lock. They bind Accelerate, Datasets, and Kit
package versions, intended annotated tags, PCPR-053/054/055 artifact
CIDs, PCPR-040/041/042/043 contract and vector identities, the one
supported combination, hermetic candidate test suites, and named
qualification-receipt identities.

They are not a live GPG/SSH signature, not a freeze (PCPR-002), not
branch protection (PCPR-057), not a published wheel or sdist, and not a
closed PCPR release.

- `cpython312/release.lock.json` is the canonical declared lock.
  `lock` is true. Exact commit and tree are bound by the PCPR-056
  receipt `current_tree_binding` because nested admission rewrites HEAD.
  `origin/main` is not the release identity.
- `platform-catalog.json` binds Datasets and Kit lock-binding documents
  when those sibling trees are present. Sibling source is never required.
- Named extras `libp2p`, `mcp-p2p`, `ipfs-transformers`, and
  `ipfs-model-manager` may carry source-checkout Git pins and are not
  this lock.

Sealed validation PATH is exactly `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin`.
Provider-local `~/.local` tools and generated signing keys are not
sealed-environment authority.
"""


class PortfolioCompatibilityLockError(Exception):
    """Fail-closed PCPR-056 contract error."""


class PortfolioCompatibilityLockAdmissionError(PortfolioCompatibilityLockError):
    """Raised when a lock input is rejected."""


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PortfolioCompatibilityLockError(f"{name} must be a non-empty string")
    return value.strip()


def _kind(value: Any, name: str) -> str:
    kind = _text(value, name)
    if kind not in EVIDENCE_KINDS:
        raise PortfolioCompatibilityLockError(
            f"{name} is not an admitted evidence kind"
        )
    return kind


def _reject_closed_release_value(value: Any, name: str) -> None:
    if isinstance(value, str) and value in CLOSED_RELEASE_OUTCOMES:
        raise PortfolioCompatibilityLockError(
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


def parse_pyproject_lock_table(text: str) -> dict[str, Any]:
    import tomllib

    table = tomllib.loads(_text(text, "pyproject.toml"))
    if not isinstance(table, dict):
        raise PortfolioCompatibilityLockError("pyproject.toml must be a table")
    tool = table.get("tool")
    payload: dict[str, Any] = {}
    if isinstance(tool, dict):
        accelerate = tool.get("ipfs_accelerate_py")
        if isinstance(accelerate, dict):
            raw = accelerate.get("portfolio-compatibility-lock")
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
class PortfolioCompatibilityLockVerdict:
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
    signatures_invented: bool
    live_signed_lock: bool
    live_signed_lock_evidence_kind: str
    mutable_main_reference: bool
    simulated_results_represented_as_live: bool
    this_task_created_competing_authority: bool
    probes: tuple[OutcomeProbe, ...]
    blockers: tuple[str, ...]
    verdict_cid: str
    lock_cid: str
    catalog_cid: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.interface,
            "verdict_cid": self.verdict_cid,
            "lock_cid": self.lock_cid,
            "catalog_cid": self.catalog_cid,
            "promotion_status": self.promotion_status,
            "supervisor_disposition": self.supervisor_disposition,
            "closed_release_outcome": self.closed_release_outcome,
            "release_claim": self.release_claim,
            "completion_authoritative": self.completion_authoritative,
            "contracts_frozen": self.contracts_frozen,
            "duckdb_or_quack_state_written": self.duckdb_or_quack_state_written,
            "sibling_source_required": self.sibling_source_required,
            "hashes_invented": self.hashes_invented,
            "signatures_invented": self.signatures_invented,
            "live_signed_lock": self.live_signed_lock,
            "live_signed_lock_evidence_kind": self.live_signed_lock_evidence_kind,
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


def _component_payload(name: str) -> dict[str, Any]:
    pinned = PINNED_COMPONENTS[name]
    return {
        "package_name": pinned["package_name"],
        "package_version": pinned["package_version"],
        "intended_tag_name": pinned["intended_tag_name"],
        "lock_cid": pinned["lock_cid"],
        "sbom_cid": pinned["sbom_cid"],
        "provenance_cid": pinned["provenance_cid"],
        "tag_policy_cid": pinned["tag_policy_cid"],
        "checksums_cid": pinned["checksums_cid"],
        "commit": {
            "status": "observed_at_evaluation",
            "evidence_kind": "measured",
            "live": False,
            "field": "PCPR-056 receipt current_tree_binding",
            "reason": (
                "Exact commit and tree are bound by the task receipt "
                "because nested admission rewrites HEAD. This document "
                "does not pin origin/main."
            ),
        },
        "wheel": typed_unavailable(
            reason=(
                "No published wheel digest is bound. Hermetic private "
                "builds are not a PCPR release. A wheel hash was not invented."
            )
        ),
        "sdist": typed_unavailable(
            reason=(
                "No published sdist digest is bound. Hermetic private "
                "builds are not a PCPR release. An sdist hash was not invented."
            )
        ),
        "container": typed_unavailable(
            reason="No container digest is bound. A digest was not invented."
        ),
        "live": False,
        "signed": False,
        "sibling_source_required": False,
    }


def _supported_combination() -> dict[str, Any]:
    return {
        "id": SUPPORTED_COMBINATION_ID,
        "repositories": list(REQUIRED_REPOSITORIES),
        "python": PYTHON_FLOOR,
        "requires_python": REQUIRES_PYTHON,
        "languages": list(SUPPORTED_LANGUAGES),
        "typescript_compiler": "unavailable",
        "catalog_cid": PINNED_CATALOG_CID,
        "vector_document_cid": PCPR_041_VECTOR_DOCUMENT_CID,
        "negative_document_cid": PINNED_042_NEGATIVE_DOCUMENT_CID,
        "compatibility_document_cid": PINNED_COMPATIBILITY_DOCUMENT_CID,
        "schema_suffix": "@1",
        "contract_count": len(REQUIRED_CONTRACT_NAMES),
        "lock": True,
        "frozen": False,
        "sibling_import": False,
        "duckdb_or_quack_state_written": False,
    }


def _vector_cids() -> dict[str, str]:
    return {
        name: str(PINNED_CONTRACT_VECTORS[name]["cid"])
        for name in REQUIRED_CONTRACT_NAMES
    }


def render_declared_lock(root: Path | None = None) -> dict[str, Any]:
    package_root = root or discover_accelerate_root()
    if package_root is None:
        raise PortfolioCompatibilityLockError("Accelerate package root was not found")
    _ = package_root
    tooling = observe_signing_tooling()
    document = {
        "schema": LOCK_SCHEMA,
        "interface": INTERFACE,
        "task_id": PCPR_056_TASK_ID,
        "goal_id": PCPR_056_GOAL_ID,
        "program_id": PCPR_PROGRAM_ID,
        "board_namespace": PCPR_BOARD_NAMESPACE,
        "evidence_id": EVIDENCE_ID,
        "portfolio_id": PORTFOLIO_ID,
        "portfolio_version": PORTFOLIO_VERSION,
        "lock_kind": LOCK_KIND,
        "lock": True,
        "frozen": False,
        "freeze_task": PCPR_002_TASK_ID,
        "compatibility_task": PCPR_043_TASK_ID,
        "signed_tags_task": PCPR_055_TASK_ID,
        "governance_task": PCPR_057_TASK_ID,
        "python_requires": CANONICAL_PYTHON_REQUIRES,
        "python": PYTHON_FLOOR,
        "requires_python": REQUIRES_PYTHON,
        "languages": list(SUPPORTED_LANGUAGES),
        "typescript_compiler": "unavailable",
        "catalog_cid": PINNED_CATALOG_CID,
        "vector_document_cid": PCPR_041_VECTOR_DOCUMENT_CID,
        "negative_document_cid": PINNED_042_NEGATIVE_DOCUMENT_CID,
        "compatibility_document_cid": PINNED_COMPATIBILITY_DOCUMENT_CID,
        "supported_combination_id": SUPPORTED_COMBINATION_ID,
        "supported_combination": _supported_combination(),
        "supported_combination_count": 1,
        "incompatible_categories": list(INCOMPATIBLE_CATEGORIES),
        "contract_schema_ids": dict(CONTRACT_SCHEMA_IDS),
        "contract_authorities": dict(CONTRACT_AUTHORITIES),
        "vector_cids": _vector_cids(),
        "datasets_owned_contracts": list(DATASETS_OWNED_CONTRACTS),
        "kit_owned_contracts": list(KIT_OWNED_CONTRACTS),
        "components": {
            name: _component_payload(name) for name in REQUIRED_REPOSITORIES
        },
        "test_suites": {
            name: {
                "paths": list(paths),
                "authority": (
                    "candidate_only; cannot satisfy live supervisor "
                    "promotion or a closed PCPR release"
                ),
                "evidence_kind": "measured",
                "live": False,
            }
            for name, paths in BOUND_TEST_SUITES.items()
        },
        "qualification_receipts": [
            dict(item) for item in BOUND_QUALIFICATION_RECEIPTS
        ],
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
                "field": "PCPR-056 receipt current_tree_binding",
                "reason": (
                    "Exact commit and tree are bound by the task receipt "
                    "because nested admission rewrites HEAD. This document "
                    "does not pin origin/main."
                ),
            },
        },
        "signing": {
            "status": "unavailable",
            "evidence_kind": "unavailable",
            "live": False,
            "signed": False,
            "algorithm": "git-gpg-or-ssh-or-cosign",
            "key_id": "unavailable",
            "signature": "unavailable",
            "invented": False,
            "reason": (
                "gpg, ssh signing keys, and cosign are not admitted live "
                "signers in the sealed environment. A signature was not "
                "invented. This declared lock is not a live signed manifest."
            ),
            "tools": dict(tooling.get("signing_tools") or {}),
        },
        "protection": typed_unavailable(
            reason=(
                "Protected signed tags and branch protection are PCPR-057. "
                "This task does not claim GitHub tag protection."
            )
        ),
        "named_git_extras": list(NAMED_GIT_EXTRAS),
        "named_git_extras_role": NAMED_GIT_EXTRAS_ROLE,
        "capabilities": {
            "supported_combination_id": SUPPORTED_COMBINATION_ID,
            "cpu": "declared_not_live_requalified",
            "cuda": typed_unavailable(
                reason="PCPR-038 CUDA live qualification remains typed unavailable."
            ),
            "solvers": typed_unavailable(
                reason="PCPR-017 live solver qualification remains typed unavailable."
            ),
            "ipfs": typed_unavailable(
                reason="PCPR-021 pinned IPFS live qualification remains typed unavailable."
            ),
            "iroh": typed_unavailable(
                reason="PCPR-022 Iroh live qualification remains typed unavailable."
            ),
        },
        "live": False,
        "published": False,
        "release_claim": False,
        "closed_release_outcome": None,
        "hashes_invented": False,
        "signatures_invented": False,
        "sibling_source_required": False,
        "remint": False,
        "source_date_epoch": SOURCE_DATE_EPOCH,
        "evidence_kind": "measured",
    }
    document["lock_cid"] = content_identity(
        {key: value for key, value in document.items() if key != "lock_cid"}
    )
    return document


def refuse_lock_remint(cid: str) -> str:
    if cid != PINNED_LOCK_CID:
        raise PortfolioCompatibilityLockError(
            f"portfolio compatibility lock CID {cid} remints {PINNED_LOCK_CID}"
        )
    return cid


def artifact_paths(root: Path) -> dict[str, Path]:
    return {
        "lock": root / LOCK_DIR_RELPATH / LOCK_JSON_NAME,
        "readme": root / LOCK_README_RELPATH,
        "catalog": root / CATALOG_RELPATH,
    }


def _sibling_binding(path: Path, relative_path: str) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": relative_path,
            "status": "unavailable",
            "evidence_kind": "unavailable",
            "reason": "Sibling lock-binding file is not present beside this checkout.",
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": relative_path,
        "status": "observed",
        "evidence_kind": "measured",
        "package_name": payload.get("package_name"),
        "package_version": payload.get("package_version"),
        "lock_kind": payload.get("lock_kind"),
        "lock_cid": payload.get("lock_cid"),
        "binding_cid": payload.get("binding_cid"),
        "live": False,
        "signed": False,
        "sha256": sha256_bytes(path.read_bytes()),
    }


def platform_compatibility_catalog(start: Path | None = None) -> dict[str, Any]:
    root = discover_accelerate_root(start)
    if root is None:
        raise PortfolioCompatibilityLockError("Accelerate package root was not found")
    parent = root.parent
    lock = render_declared_lock(root)
    datasets_binding = _sibling_binding(
        parent
        / "ipfs_datasets"
        / LOCK_DIR_RELPATH
        / "release.binding.json",
        relative_path=f"../ipfs_datasets/{LOCK_DIR_RELPATH}/release.binding.json",
    )
    kit_binding = _sibling_binding(
        parent / "ipfs_kit" / LOCK_DIR_RELPATH / "release.binding.json",
        relative_path=f"../ipfs_kit/{LOCK_DIR_RELPATH}/release.binding.json",
    )
    catalog = {
        "schema": CATALOG_SCHEMA,
        "interface": CATALOG_INTERFACE,
        "task_id": PCPR_056_TASK_ID,
        "goal_id": PCPR_056_GOAL_ID,
        "lock_kind": LOCK_KIND,
        "portfolio_id": PORTFOLIO_ID,
        "portfolio_version": PORTFOLIO_VERSION,
        "lock_cid": lock["lock_cid"],
        "components": {
            "ipfs_accelerate_py": {
                "lock_path": f"{LOCK_DIR_RELPATH}/{LOCK_JSON_NAME}",
                "status": "observed",
                "evidence_kind": "measured",
                "package_name": PACKAGE_NAME,
                "package_version": PACKAGE_VERSION,
                "intended_tag_name": INTENDED_TAG_NAME,
                "lock_kind": LOCK_KIND,
                "lock_cid": lock["lock_cid"],
                "hashes_invented": False,
                "signatures_invented": False,
                "live": False,
                "signed": False,
            },
            "ipfs_datasets_py": {"binding": datasets_binding},
            "ipfs_kit_py": {"binding": kit_binding},
        },
        "mutable_main_reference": False,
        "hashes_invented": False,
        "signatures_invented": False,
        "live_signed_lock": False,
        "closed_release_outcome": None,
        "release_claim": False,
        "sibling_source_required": False,
        "evidence_kind": "measured",
    }
    catalog["catalog_cid"] = content_identity(
        {key: value for key, value in catalog.items() if key != "catalog_cid"}
    )
    return catalog


def write_portfolio_compatibility_lock_files(
    start: Path | None = None,
) -> dict[str, Any]:
    root = discover_accelerate_root(start)
    if root is None:
        raise PortfolioCompatibilityLockError("Accelerate package root was not found")
    lock = render_declared_lock(root)
    paths = artifact_paths(root)
    _atomic_write(paths["lock"], pretty_json(lock))
    readme = LOCK_README if LOCK_README.endswith("\n") else LOCK_README + "\n"
    _atomic_write(paths["readme"], readme)
    catalog = platform_compatibility_catalog(start)
    _atomic_write(paths["catalog"], pretty_json(catalog))
    return {
        "lock": lock,
        "catalog": catalog,
        "paths": {name: str(path) for name, path in paths.items()},
    }


def verify_portfolio_compatibility_lock_files(
    start: Path | None = None,
) -> dict[str, Any]:
    root = discover_accelerate_root(start)
    if root is None:
        raise PortfolioCompatibilityLockError("Accelerate package root was not found")
    lock = render_declared_lock(root)
    paths = artifact_paths(root)
    missing: list[str] = []
    lock_ok = False
    readme_ok = False
    expected_readme = (
        LOCK_README if LOCK_README.endswith("\n") else LOCK_README + "\n"
    )
    for name, path in paths.items():
        if name == "catalog":
            continue
        if not path.is_file():
            missing.append(name)
            continue
        if name == "lock":
            lock_ok = json.loads(path.read_text(encoding="utf-8")) == lock
        elif name == "readme":
            readme_ok = path.read_text(encoding="utf-8") == expected_readme
    return {
        "ok": not missing and lock_ok and readme_ok,
        "missing": missing,
        "lock_ok": lock_ok,
        "readme_ok": readme_ok,
        "lock_cid": lock["lock_cid"],
        "lock_sha256": (
            sha256_bytes(paths["lock"].read_bytes())
            if paths["lock"].is_file()
            else "unavailable"
        ),
    }


def _accelerate_generated_cids(root: Path) -> dict[str, str]:
    lock = render_release_lock(root)
    sbom = render_declared_sbom(root)
    provenance = render_build_provenance(root, sbom=sbom)
    tag_policy = render_declared_tag_policy(root)
    checksums = render_declared_checksums(root, tag_policy=tag_policy)
    return {
        "lock_cid": str(lock["lock_cid"]),
        "sbom_cid": str(sbom["sbom_cid"]),
        "provenance_cid": str(provenance["provenance_cid"]),
        "tag_policy_cid": str(tag_policy["tag_policy_cid"]),
        "checksums_cid": str(checksums["checksums_cid"]),
    }


def current_head_static_probes(
    start: Path | None = None,
) -> tuple[OutcomeProbe, ...]:
    root = discover_accelerate_root(start)
    if root is None:
        raise PortfolioCompatibilityLockError("Accelerate package root was not found")
    lock = render_declared_lock(root)
    verified = verify_portfolio_compatibility_lock_files(root)
    table = parse_pyproject_lock_table(
        (root / "pyproject.toml").read_text(encoding="utf-8")
    )
    tooling = observe_signing_tooling()
    generated = _accelerate_generated_cids(root)
    pinned_acc = PINNED_COMPONENTS["ipfs_accelerate_py"]
    accelerate_match = all(
        generated[key] == pinned_acc[key]
        for key in (
            "lock_cid",
            "sbom_cid",
            "provenance_cid",
            "tag_policy_cid",
            "checksums_cid",
        )
    )
    remint = lock["lock_cid"] != PINNED_LOCK_CID
    mutable_main = bool(lock["source"]["mutable_main_reference"])
    probes = [
        _probe(
            "lock_files_match_generator",
            verified.get("ok") is True and verified.get("lock_ok") is True,
            reason=(
                "Committed portfolio compatibility lock matches the generator."
                if verified.get("lock_ok") is True
                else "Committed portfolio compatibility lock is missing or drifts."
            ),
            details=verified,
        ),
        _probe(
            "pyproject_portfolio_compatibility_lock_table",
            table.get("interface") == INTERFACE
            and table.get("schema") == SCHEMA
            and table.get("task-id") == PCPR_056_TASK_ID
            and table.get("lock-kind") == LOCK_KIND,
            reason=(
                "pyproject.toml declares AcceleratePortfolioCompatibilityLock@1."
                if table.get("interface") == INTERFACE
                else "pyproject.toml does not declare AcceleratePortfolioCompatibilityLock@1."
            ),
            details={"table": table},
        ),
        _probe(
            "hashes_not_invented",
            lock["hashes_invented"] is False,
            reason="Hashes were not invented; unpublished artifact hashes stay typed unavailable.",
        ),
        _probe(
            "hashes_invented",
            False,
            reason="Hashes must not be invented for unavailable wheel/sdist identities.",
        ),
        _probe(
            "signatures_not_invented",
            lock["signatures_invented"] is False
            and lock["signing"]["invented"] is False,
            reason="Signatures were not invented; live signing stays typed unavailable.",
        ),
        _probe(
            "signatures_invented",
            False,
            reason="A signature must not be invented for an unavailable signer.",
        ),
        _probe(
            "no_mutable_main_reference",
            mutable_main is False,
            reason="Portfolio lock does not pin origin/main as the release identity.",
        ),
        _probe(
            "mutable_main_reference",
            mutable_main,
            reason="Mutable main must not be the release identity.",
        ),
        _probe(
            "lock_kind_is_declared_portfolio_compatibility_lock",
            lock["lock_kind"] == LOCK_KIND and lock["lock"] is True,
            reason="Lock kind is declared_portfolio_compatibility_lock and lock is true.",
        ),
        _probe(
            "supported_combination_is_locked",
            lock["supported_combination"]["lock"] is True
            and lock["supported_combination_id"] == SUPPORTED_COMBINATION_ID,
            reason="The PCPR-043 supported combination is locked by this document.",
        ),
        _probe(
            "pcpr_043_identities_not_reminted",
            lock["catalog_cid"] == PINNED_CATALOG_CID
            and lock["vector_document_cid"] == PCPR_041_VECTOR_DOCUMENT_CID
            and lock["negative_document_cid"] == PINNED_042_NEGATIVE_DOCUMENT_CID
            and lock["compatibility_document_cid"]
            == PINNED_COMPATIBILITY_DOCUMENT_CID,
            reason="PCPR-040/041/042/043 identities are bound and not reminted.",
        ),
        _probe(
            "compatibility_identities_reminted",
            remint,
            reason="A reminted lock CID is forbidden.",
            details={"lock_cid": lock["lock_cid"], "pinned": PINNED_LOCK_CID},
        ),
        _probe(
            "accelerate_artifact_cids_match_pins",
            accelerate_match,
            reason=(
                "Generated Accelerate lock/SBOM/provenance/tag/checksum CIDs "
                "match the PCPR-053/054/055 pins."
                if accelerate_match
                else "Generated Accelerate artifact CIDs remint PCPR-053/054/055."
            ),
            details={"generated": generated, "pinned": dict(pinned_acc)},
        ),
        _probe(
            "named_git_extras_are_not_this_lock",
            True,
            reason=(
                "Named libp2p, mcp-p2p, ipfs-transformers, and ipfs-model-manager "
                "extras may carry source-checkout Git pins; they are not this lock."
            ),
            details={
                "named_git_extras": list(NAMED_GIT_EXTRAS),
                "named_git_extras_role": NAMED_GIT_EXTRAS_ROLE,
            },
        ),
        _probe(
            "simulated_results_represented_as_live",
            False,
            reason="Simulated results are not represented as live.",
        ),
        _probe(
            "signed_lock_represented_as_live",
            False,
            reason="No live signed compatibility lock is represented as live.",
        ),
        _probe(
            "gpg_signature_represented_as_live",
            False,
            reason="No GPG signature is represented as live.",
        ),
        _probe(
            "cosign_signature_represented_as_live",
            False,
            reason="No cosign signature is represented as live.",
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
            "live_signed_lock",
            None,
            evidence_kind="unavailable",
            reason=(
                "gpg/SSH signing keys and cosign are absent as admitted live "
                "signers. A signed lock was not invented."
            ),
            details={"tools": tooling.get("signing_tools")},
        ),
        _probe(
            "published_wheel",
            None,
            evidence_kind="unavailable",
            reason="No published wheel identity is bound by this declared lock.",
        ),
        _probe(
            "published_sdist",
            None,
            evidence_kind="unavailable",
            reason="No published sdist identity is bound by this declared lock.",
        ),
        _probe(
            "container_digest",
            None,
            evidence_kind="unavailable",
            reason="No container digest is bound by this declared lock.",
        ),
        _probe(
            "live_supervisor_qualification",
            None,
            evidence_kind="unavailable",
            reason="PCPR-001 live qualification remains typed unavailable.",
        ),
    ]
    return tuple(probes)


def qualify_portfolio_compatibility_lock(
    probes: Sequence[OutcomeProbe],
    *,
    lock_cid: str,
    catalog_cid: str,
) -> PortfolioCompatibilityLockVerdict:
    if not probes:
        raise PortfolioCompatibilityLockError("at least one probe is required")
    normalized: list[OutcomeProbe] = []
    blockers: list[str] = []
    for probe in probes:
        kind = _kind(probe.evidence_kind, "evidence_kind")
        if probe.live and kind != "measured_live":
            raise PortfolioCompatibilityLockError(
                "live claims require measured_live evidence"
            )
        if probe.simulated_represented_as_live:
            raise PortfolioCompatibilityLockError(
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
        "task_id": PCPR_056_TASK_ID,
        "goal_id": PCPR_056_GOAL_ID,
        "promotion_status": promotion_status,
        "supervisor_disposition": "supervisor_non_promoted",
        "closed_release_outcome": None,
        "release_claim": False,
        "completion_authoritative": False,
        "contracts_frozen": False,
        "duckdb_or_quack_state_written": False,
        "sibling_source_required": False,
        "hashes_invented": False,
        "signatures_invented": False,
        "live_signed_lock": False,
        "live_signed_lock_evidence_kind": "unavailable",
        "mutable_main_reference": mutable,
        "simulated_results_represented_as_live": False,
        "this_task_created_competing_authority": False,
        "probes": [item.to_mapping() for item in normalized],
        "blockers": list(dict.fromkeys(blockers)),
        "lock_cid": lock_cid,
        "catalog_cid": catalog_cid,
    }
    return PortfolioCompatibilityLockVerdict(
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
        signatures_invented=False,
        live_signed_lock=False,
        live_signed_lock_evidence_kind="unavailable",
        mutable_main_reference=mutable,
        simulated_results_represented_as_live=False,
        this_task_created_competing_authority=False,
        probes=tuple(normalized),
        blockers=tuple(dict.fromkeys(blockers)),
        verdict_cid=content_identity(payload),
        lock_cid=lock_cid,
        catalog_cid=catalog_cid,
    )


def qualify_current_head_portfolio_compatibility_lock(
    start: Path | None = None,
) -> PortfolioCompatibilityLockVerdict:
    root = discover_accelerate_root(start)
    lock = render_declared_lock(root)
    catalog = platform_compatibility_catalog(start)
    return qualify_portfolio_compatibility_lock(
        current_head_static_probes(start),
        lock_cid=str(lock["lock_cid"]),
        catalog_cid=str(catalog["catalog_cid"]),
    )


def pcpr_056_receipt_promotion(
    verdict: PortfolioCompatibilityLockVerdict,
) -> dict[str, Any]:
    if verdict.closed_release_outcome is not None:
        raise PortfolioCompatibilityLockError(
            "portfolio compatibility lock must not mint a closed release outcome"
        )
    if verdict.release_claim:
        raise PortfolioCompatibilityLockError(
            "portfolio compatibility lock must not claim a PCPR release"
        )
    if verdict.completion_authoritative:
        raise PortfolioCompatibilityLockError(
            "portfolio-compatibility-lock completion is not authoritative"
        )
    if verdict.duckdb_or_quack_state_written:
        raise PortfolioCompatibilityLockError(
            "portfolio compatibility lock must not write DuckDB or Quack state"
        )
    if verdict.promotion_status in CLOSED_RELEASE_OUTCOMES:
        raise PortfolioCompatibilityLockError(
            "promotion_status must not be a closed release outcome"
        )
    return verdict.to_mapping()


# Pinned after the lock encoder is measured. Drift is a remint.
PINNED_LOCK_CID: Final = (
    "baguqeerawsekbbbt5ccctjt4tydzeahfkahsatb6q5x7k6cy4inrii5lhqmq"
)

# Pinned identity of the ordinary current-head static verdict. Drift means
# the default payload changed and the outer receipt must be regenerated.
CURRENT_HEAD_NON_PROMOTION_VERDICT_CID: Final = (
    "baguqeerarhdm3xw7rqs3ijd43s3f3w4ckl526b3i5l4wy265ga4nvxhkshqq"
)


__all__ = [
    "BOUND_QUALIFICATION_RECEIPTS",
    "CATALOG_INTERFACE",
    "CLOSED_RELEASE_OUTCOMES",
    "CURRENT_HEAD_NON_PROMOTION_VERDICT_CID",
    "HERMETIC_CANDIDATE_SUITES",
    "INTERFACE",
    "LOCK_KIND",
    "OutcomeProbe",
    "PCPR_056_GOAL_ID",
    "PCPR_056_TASK_ID",
    "PINNED_COMPONENTS",
    "PINNED_LOCK_CID",
    "PORTFOLIO_ID",
    "PORTFOLIO_VERSION",
    "PortfolioCompatibilityLockError",
    "PortfolioCompatibilityLockVerdict",
    "SCHEMA",
    "SEALED_PATH",
    "SEALED_PYTHON",
    "SUPPORTED_COMBINATION_ID",
    "current_head_static_probes",
    "pcpr_056_receipt_promotion",
    "platform_compatibility_catalog",
    "qualify_current_head_portfolio_compatibility_lock",
    "qualify_portfolio_compatibility_lock",
    "refuse_lock_remint",
    "render_declared_lock",
    "verify_portfolio_compatibility_lock_files",
    "write_portfolio_compatibility_lock_files",
]
