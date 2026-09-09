"""Fail-closed PCPR-057 Accelerate branch and release gates.

Produce a declared default-branch protection policy and a declared
release gate for the proof-carrying-platform-0.1.0 train. The documents
require current-head and release checks, protect intended signed-tag
patterns, and prohibit publishing a release after a partial required-build
failure.

Live GitHub branch protection, tag protection, required-status
enforcement, and repository-admin application stay typed unavailable.
gh CLI presence, curl presence, or a provider GITHUB_TOKEN is not live
governance. This module never applies GitHub settings, never writes
DuckDB or Quack state, and never emits a closed PCPR release outcome.

Missing repository-admin permission produces an explicit operator-blocking
task. The governance gate is not complete. Simulated results are not live.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from ipfs_accelerate_py.assurance.cross_repository_compatibility import (
    PINNED_CATALOG_CID as PCPR_043_CATALOG_CID,
    PINNED_COMPATIBILITY_DOCUMENT_CID,
    REQUIRED_REPOSITORIES,
    SUPPORTED_COMBINATION_ID,
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
    sha256_bytes,
    typed_unavailable,
)
from ipfs_accelerate_py.assurance.portfolio_compatibility_lock import (
    PINNED_LOCK_CID as PCPR_056_LOCK_CID,
    PORTFOLIO_ID,
    PORTFOLIO_VERSION,
)
from ipfs_accelerate_py.assurance.python_compatibility import (
    PYTHON_FLOOR,
    REQUIRES_PYTHON,
)
from ipfs_accelerate_py.assurance.signed_tags_and_artifacts import (
    INTENDED_TAG_NAME,
    observe_signing_tooling,
)

INTERFACE: Final = "AccelerateBranchAndReleaseGates@1"
SCHEMA: Final = "ipfs_accelerate_py/assurance/branch-and-release-gates@1"
BRANCH_POLICY_SCHEMA: Final = (
    "ipfs_accelerate_py/assurance/declared-branch-protection-policy@1"
)
RELEASE_GATE_SCHEMA: Final = (
    "ipfs_accelerate_py/assurance/declared-release-gate@1"
)
VERDICT_SCHEMA: Final = (
    "ipfs_accelerate_py/assurance/branch-and-release-gates-verdict@1"
)
CATALOG_SCHEMA: Final = (
    "ipfs_accelerate_py/assurance/platform-branch-and-release-gates-catalog@1"
)
CATALOG_INTERFACE: Final = "PlatformBranchAndReleaseGates@1"
PCPR_057_TASK_ID: Final = "PCPR-057"
PCPR_057_GOAL_ID: Final = "PCPR-G600"
PCPR_056_TASK_ID: Final = "PCPR-056"
PCPR_055_TASK_ID: Final = "PCPR-055"
PCPR_093_TASK_ID: Final = "PCPR-093"
PCPR_094_TASK_ID: Final = "PCPR-094"
PCPR_002_TASK_ID: Final = "PCPR-002"
PCPR_001_TASK_ID: Final = "PCPR-001"
PCPR_PROGRAM_ID: Final = "proof-carrying-platform-qualification-and-release-v1"
PCPR_BOARD_NAMESPACE: Final = (
    "proof-carrying-platform-qualification-and-release-v1"
)
EVIDENCE_ID: Final = "pcpr/branch-and-release-gates@1"

CANONICAL_PYTHON_REQUIRES: Final = ">=3.12"
SOURCE_REPOSITORY: Final = "endomorphosis/ipfs_accelerate_py"
SOURCE_ORIGIN_URL: Final = "https://github.com/endomorphosis/ipfs_accelerate_py"
BRANCH_POLICY_KIND: Final = "declared_branch_protection_policy"
RELEASE_GATE_KIND: Final = "declared_release_gate"
DEFAULT_BRANCH: Final = "main"
INTENDED_TAG_PATTERN: Final = f"{PACKAGE_NAME}-v*"
GATES_DIR_RELPATH: Final = "packaging/pcpr/gates/cpython312"
BRANCH_POLICY_JSON_NAME: Final = "release.branch-protection.json"
RELEASE_GATE_JSON_NAME: Final = "release.gate.json"
GATES_README_RELPATH: Final = "packaging/pcpr/gates/README.md"
CATALOG_RELPATH: Final = "packaging/pcpr/gates/platform-catalog.json"
SOURCE_DATE_EPOCH: Final = "0"
OPERATOR_BLOCKING_TASK_ID: Final = "pcpr-057-operator-github-governance"

if PCPR_056_LOCK_CID != (
    "baguqeerawsekbbbt5ccctjt4tydzeahfkahsatb6q5x7k6cy4inrii5lhqmq"
):
    raise RuntimeError("PCPR-057 lock CID remints PCPR-056")
if PCPR_043_CATALOG_CID != (
    "baguqeeraqpptps2gf55wmthv3sm5nzlxqqdjfyg2ggkbl5ovfzuvvvsxap4a"
):
    raise RuntimeError("PCPR-057 catalog CID remints PCPR-043")

REQUIRED_STATUS_CHECKS: Final[tuple[str, ...]] = (
    "pcpr.current-head-qualification",
    "pcpr.hermetic-candidate-suites",
    "pcpr.clean-package",
    "pcpr.dependency-locks",
    "pcpr.sbom-and-provenance",
    "pcpr.signed-tags-and-artifacts",
    "pcpr.portfolio-compatibility-lock",
    "pcpr.branch-and-release-gates",
)

PROTECTED_REPOSITORIES: Final[Mapping[str, Mapping[str, str]]] = MappingProxyType(
    {
        "ipfs_accelerate_py": MappingProxyType(
            {
                "github_repository": "endomorphosis/ipfs_accelerate_py",
                "default_branch": "main",
                "intended_tag_name": "ipfs_accelerate_py-v0.0.45",
                "intended_tag_pattern": "ipfs_accelerate_py-v*",
                "package_version": "0.0.45",
            }
        ),
        "ipfs_datasets_py": MappingProxyType(
            {
                "github_repository": "endomorphosis/ipfs_datasets_py",
                "default_branch": "main",
                "intended_tag_name": "ipfs_datasets_py-v0.2.0",
                "intended_tag_pattern": "ipfs_datasets_py-v*",
                "package_version": "0.2.0",
            }
        ),
        "ipfs_kit_py": MappingProxyType(
            {
                "github_repository": "endomorphosis/ipfs_kit_py",
                "default_branch": "main",
                "intended_tag_name": "ipfs_kit_py-v0.3.0",
                "intended_tag_pattern": "ipfs_kit_py-v*",
                "package_version": "0.3.0",
            }
        ),
    }
)

HERMETIC_CANDIDATE_SUITES: Final[tuple[str, ...]] = (
    "test/api/test_agent_supervisor_branch_and_release_gates.py",
)

GOVERNANCE_TOOLS: Final[tuple[str, ...]] = (
    "gh",
    "curl",
    "git",
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
        "branch_policy_files_match_generator",
        "release_gate_files_match_generator",
        "pyproject_branch_and_release_gates_table",
        "required_status_checks_declared",
        "partial_required_build_failure_prohibits_release",
        "no_mutable_main_reference",
        "branch_policy_kind_is_declared",
        "release_gate_kind_is_declared",
        "pcpr_056_lock_not_reminted",
        "governance_gate_is_not_complete",
        "operator_blocking_task_emitted",
        "named_git_extras_are_not_this_gate",
    }
)
FORBIDDEN_PRESENT_PROBE_IDS: Final[frozenset[str]] = frozenset(
    {
        "mutable_main_reference",
        "simulated_results_represented_as_live",
        "live_branch_protection_represented_as_live",
        "live_tag_protection_represented_as_live",
        "live_github_admin_represented_as_live",
        "closed_release_represented_as_live",
        "partial_release_after_required_build_failure",
        "governance_gate_represented_as_complete",
        "compatibility_identities_reminted",
    }
)

GATES_README: Final = """# PCPR-057 declared branch and release gates

These files are the Accelerate *declared* default-branch protection
policy and release gate for proof-carrying-platform-0.1.0. They are not
live GitHub branch protection, not live tag protection, not a live
required-status configuration, not a freeze (PCPR-002), not a closed
PCPR release (PCPR-093/094), and not a published wheel or sdist.

- `cpython312/release.branch-protection.json` requires protecting
  `main`, pull-request reviews, current-head and release status checks,
  no force-push, no deletion, administrator enforcement, and signed
  commits. Exact commit and tree are bound by the PCPR-057 receipt
  `current_tree_binding` because nested admission rewrites HEAD.
  `origin/main` is not the release identity.
- `cpython312/release.gate.json` requires every named status check to
  pass and prohibits publishing a release after a partial required-build
  failure. Live GitHub enforcement stays typed unavailable.
- `platform-catalog.json` binds Datasets and Kit gate-binding documents
  when those sibling trees are present. Sibling source is never required.
- Missing repository-admin permission is an explicit operator-blocking
  task. The governance gate is not complete.

Sealed validation PATH is exactly `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin`.
Provider-local `~/.local` tools and provider `GITHUB_TOKEN` values are
not sealed-environment authority.
"""


class BranchAndReleaseGatesError(Exception):
    """Fail-closed PCPR-057 contract error."""


class BranchAndReleaseGatesAdmissionError(BranchAndReleaseGatesError):
    """Raised when a branch or release-gate input is rejected."""


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BranchAndReleaseGatesError(f"{name} must be a non-empty string")
    return value.strip()


def _kind(value: Any, name: str) -> str:
    kind = _text(value, name)
    if kind not in EVIDENCE_KINDS:
        raise BranchAndReleaseGatesError(
            f"{name} is not an admitted evidence kind"
        )
    return kind


def _reject_closed_release_value(value: Any, name: str) -> None:
    if isinstance(value, str) and value in CLOSED_RELEASE_OUTCOMES:
        raise BranchAndReleaseGatesError(
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


def observe_github_governance_tooling() -> dict[str, Any]:
    """Measure sealed-PATH GitHub tools. Missing stay unavailable.

    This probe never contacts github.com, never reads token values, and
    never applies branch protection.
    """

    env = observe_sealed_validation_environment()
    tools = {name.replace("-", "_"): _which_sealed(name) for name in GOVERNANCE_TOOLS}
    token_present = bool(
        os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    )
    return {
        **env,
        "governance_tools": tools,
        "gh": tools.get("gh", "unavailable"),
        "curl": tools.get("curl", "unavailable"),
        "git": tools.get("git", "unavailable"),
        "github_token_env_present": token_present,
        "github_token_is_not_sealed_admin_authority": True,
        "github_api_contacted": False,
        "branch_protection_applied": False,
        "tag_protection_applied": False,
        "required_status_checks_applied": False,
        "any_live_github_admin": False,
        "live_branch_protection": typed_unavailable(
            reason=(
                "Sealed PATH has no admitted GitHub admin session. Presence "
                "of gh or curl is not live branch protection. A GitHub API "
                "call was not made."
            )
        ),
        "live_tag_protection": typed_unavailable(
            reason=(
                "Protected signed tags were not observed on GitHub. Tag "
                "protection was not invented and was not applied."
            )
        ),
        "evidence_kind": "measured",
    }


def parse_pyproject_gates_table(text: str) -> dict[str, Any]:
    import tomllib

    table = tomllib.loads(_text(text, "pyproject.toml"))
    if not isinstance(table, dict):
        raise BranchAndReleaseGatesError("pyproject.toml must be a table")
    tool = table.get("tool")
    payload: dict[str, Any] = {}
    if isinstance(tool, dict):
        accelerate = tool.get("ipfs_accelerate_py")
        if isinstance(accelerate, dict):
            raw = accelerate.get("branch-and-release-gates")
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
class BranchAndReleaseGatesVerdict:
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
    live_branch_protection: bool
    live_branch_protection_evidence_kind: str
    live_tag_protection: bool
    live_tag_protection_evidence_kind: str
    live_github_admin: bool
    governance_gate_complete: bool
    operator_blocking_task: str
    mutable_main_reference: bool
    simulated_results_represented_as_live: bool
    this_task_created_competing_authority: bool
    probes: tuple[OutcomeProbe, ...]
    blockers: tuple[str, ...]
    verdict_cid: str
    branch_policy_cid: str
    release_gate_cid: str
    catalog_cid: str
    lock_cid: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.interface,
            "verdict_cid": self.verdict_cid,
            "branch_policy_cid": self.branch_policy_cid,
            "release_gate_cid": self.release_gate_cid,
            "catalog_cid": self.catalog_cid,
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
            "signatures_invented": self.signatures_invented,
            "live_branch_protection": self.live_branch_protection,
            "live_branch_protection_evidence_kind": (
                self.live_branch_protection_evidence_kind
            ),
            "live_tag_protection": self.live_tag_protection,
            "live_tag_protection_evidence_kind": (
                self.live_tag_protection_evidence_kind
            ),
            "live_github_admin": self.live_github_admin,
            "governance_gate_complete": self.governance_gate_complete,
            "operator_blocking_task": self.operator_blocking_task,
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


def _operator_blocking_task() -> dict[str, Any]:
    return {
        "task_id": OPERATOR_BLOCKING_TASK_ID,
        "status": "typed_blocked",
        "evidence_kind": "unavailable",
        "live": False,
        "applied": False,
        "governance_gate_complete": False,
        "requires": (
            "repository-admin permission on endomorphosis/ipfs_accelerate_py, "
            "endomorphosis/ipfs_datasets_py, and endomorphosis/ipfs_kit_py"
        ),
        "action": (
            "Apply GitHub branch protection on default branch main, protect "
            "signed-tag patterns, and require the declared current-head and "
            "release status checks. Do not publish a release after a partial "
            "required-build failure."
        ),
        "reason": (
            "Sealed validation has no admitted GitHub admin identity. gh CLI "
            "and live branch-protection API stay typed unavailable. The "
            "declared policy is not live GitHub protection. This is the "
            "explicit operator-blocking task required when permissions do "
            "not allow applying governance."
        ),
    }


def _source_binding() -> dict[str, Any]:
    return {
        "kind": "git",
        "repository": SOURCE_REPOSITORY,
        "origin_url": SOURCE_ORIGIN_URL,
        "binding": "current_head_not_mutable_main",
        "mutable_main_reference": False,
        "commit": {
            "status": "observed_at_evaluation",
            "evidence_kind": "measured",
            "live": False,
            "field": "PCPR-057 receipt current_tree_binding",
            "reason": (
                "Exact commit and tree are bound by the task receipt "
                "because nested admission rewrites HEAD. This document "
                "does not pin origin/main."
            ),
        },
    }


def render_declared_branch_protection_policy(
    root: Path | None = None,
) -> dict[str, Any]:
    package_root = root or discover_accelerate_root()
    if package_root is None:
        raise BranchAndReleaseGatesError("Accelerate package root was not found")
    _ = package_root
    tooling = observe_github_governance_tooling()
    document = {
        "schema": BRANCH_POLICY_SCHEMA,
        "interface": INTERFACE,
        "task_id": PCPR_057_TASK_ID,
        "goal_id": PCPR_057_GOAL_ID,
        "program_id": PCPR_PROGRAM_ID,
        "board_namespace": PCPR_BOARD_NAMESPACE,
        "evidence_id": EVIDENCE_ID,
        "portfolio_id": PORTFOLIO_ID,
        "portfolio_version": PORTFOLIO_VERSION,
        "policy_kind": BRANCH_POLICY_KIND,
        "package_name": PACKAGE_NAME,
        "package_version": PACKAGE_VERSION,
        "python_requires": CANONICAL_PYTHON_REQUIRES,
        "python": PYTHON_FLOOR,
        "requires_python": REQUIRES_PYTHON,
        "github_repository": SOURCE_REPOSITORY,
        "default_branch": DEFAULT_BRANCH,
        "intended_tag_name": INTENDED_TAG_NAME,
        "intended_tag_pattern": INTENDED_TAG_PATTERN,
        "repositories": {
            name: dict(payload) for name, payload in PROTECTED_REPOSITORIES.items()
        },
        "required_status_checks": {
            "strict": True,
            "contexts": list(REQUIRED_STATUS_CHECKS),
            "applied": False,
            "live": False,
            "evidence_kind": "unavailable",
        },
        "pull_request_reviews": {
            "required": True,
            "required_approving_review_count": 1,
            "dismiss_stale_reviews": True,
            "require_code_owner_reviews": False,
            "require_last_push_approval": False,
            "applied": False,
            "live": False,
        },
        "restrictions": {
            "enforce_admins": True,
            "allow_force_pushes": False,
            "allow_deletions": False,
            "allow_fork_syncing": False,
            "block_creations": False,
            "require_conversation_resolution": True,
            "require_linear_history": False,
            "lock_branch": False,
            "applied": False,
            "live": False,
        },
        "signed_commits": {
            "required": True,
            "applied": False,
            "live": False,
            "evidence_kind": "unavailable",
            "reason": (
                "Signed commits are required by this declared policy. Live "
                "GitHub enforcement of signed commits is typed unavailable."
            ),
        },
        "tag_protection": {
            "patterns": [
                payload["intended_tag_pattern"]
                for payload in PROTECTED_REPOSITORIES.values()
            ],
            "allow_deletions": False,
            "applied": False,
            "live": False,
            "evidence_kind": "unavailable",
            "reason": (
                "Protected signed tags are required by this declared policy. "
                "Live GitHub tag protection is typed unavailable."
            ),
        },
        "source": _source_binding(),
        "operator_blocking_task": _operator_blocking_task(),
        "governance_tools": {
            "gh": tooling.get("gh"),
            "curl": tooling.get("curl"),
            "git": tooling.get("git"),
            "github_token_is_not_sealed_admin_authority": True,
            "github_api_contacted": False,
            "any_live_github_admin": False,
            "reason": (
                "Sealed-PATH tool presence is recorded. A provider "
                "GITHUB_TOKEN is not sealed admin authority and is not "
                "bound by this document."
            ),
        },
        "live": False,
        "applied": False,
        "published": False,
        "release_claim": False,
        "closed_release_outcome": None,
        "governance_gate_complete": False,
        "hashes_invented": False,
        "signatures_invented": False,
        "sibling_source_required": False,
        "source_date_epoch": SOURCE_DATE_EPOCH,
        "evidence_kind": "measured",
    }
    document["branch_policy_cid"] = content_identity(
        {key: value for key, value in document.items() if key != "branch_policy_cid"}
    )
    return document


def render_declared_release_gate(root: Path | None = None) -> dict[str, Any]:
    package_root = root or discover_accelerate_root()
    if package_root is None:
        raise BranchAndReleaseGatesError("Accelerate package root was not found")
    _ = package_root
    signing = observe_signing_tooling()
    document = {
        "schema": RELEASE_GATE_SCHEMA,
        "interface": INTERFACE,
        "task_id": PCPR_057_TASK_ID,
        "goal_id": PCPR_057_GOAL_ID,
        "program_id": PCPR_PROGRAM_ID,
        "board_namespace": PCPR_BOARD_NAMESPACE,
        "evidence_id": EVIDENCE_ID,
        "portfolio_id": PORTFOLIO_ID,
        "portfolio_version": PORTFOLIO_VERSION,
        "gate_kind": RELEASE_GATE_KIND,
        "package_name": PACKAGE_NAME,
        "package_version": PACKAGE_VERSION,
        "python_requires": CANONICAL_PYTHON_REQUIRES,
        "supported_combination_id": SUPPORTED_COMBINATION_ID,
        "lock_cid": PCPR_056_LOCK_CID,
        "compatibility_document_cid": PINNED_COMPATIBILITY_DOCUMENT_CID,
        "catalog_contract_cid": PCPR_043_CATALOG_CID,
        "signed_tags_task": PCPR_055_TASK_ID,
        "compatibility_lock_task": PCPR_056_TASK_ID,
        "closed_decision_tasks": [PCPR_093_TASK_ID, PCPR_094_TASK_ID],
        "freeze_task": PCPR_002_TASK_ID,
        "required_status_checks": list(REQUIRED_STATUS_CHECKS),
        "required_check_count": len(REQUIRED_STATUS_CHECKS),
        "partial_required_build_failure_prohibits_release": True,
        "all_required_checks_must_pass": True,
        "simulated_as_live_prohibits_release": True,
        "unpublished_artifact_is_not_a_release": True,
        "mutable_main_is_not_release_identity": True,
        "hermetic_pass_cannot_qualify_live": True,
        "required_repositories": list(REQUIRED_REPOSITORIES),
        "repositories": {
            name: dict(payload) for name, payload in PROTECTED_REPOSITORIES.items()
        },
        "source": _source_binding(),
        "signing": {
            "status": "unavailable",
            "evidence_kind": "unavailable",
            "live": False,
            "signed": False,
            "invented": False,
            "reason": (
                "A live GPG/SSH/cosign signer was not admitted. Signatures "
                "were not invented. This gate does not create git tags."
            ),
            "tools": dict(signing.get("signing_tools") or {}),
        },
        "operator_blocking_task": _operator_blocking_task(),
        "live": False,
        "applied": False,
        "published": False,
        "release_claim": False,
        "closed_release_outcome": None,
        "governance_gate_complete": False,
        "hashes_invented": False,
        "signatures_invented": False,
        "named_git_extras": list(NAMED_GIT_EXTRAS),
        "named_git_extras_role": NAMED_GIT_EXTRAS_ROLE,
        "sibling_source_required": False,
        "source_date_epoch": SOURCE_DATE_EPOCH,
        "evidence_kind": "measured",
    }
    document["release_gate_cid"] = content_identity(
        {key: value for key, value in document.items() if key != "release_gate_cid"}
    )
    return document


def refuse_policy_remint(cid: str) -> str:
    if cid != PINNED_BRANCH_POLICY_CID:
        raise BranchAndReleaseGatesError(
            f"branch protection policy CID {cid} remints {PINNED_BRANCH_POLICY_CID}"
        )
    return cid


def refuse_gate_remint(cid: str) -> str:
    if cid != PINNED_RELEASE_GATE_CID:
        raise BranchAndReleaseGatesError(
            f"release gate CID {cid} remints {PINNED_RELEASE_GATE_CID}"
        )
    return cid


def artifact_paths(root: Path) -> dict[str, Path]:
    return {
        "branch_policy": root / GATES_DIR_RELPATH / BRANCH_POLICY_JSON_NAME,
        "release_gate": root / GATES_DIR_RELPATH / RELEASE_GATE_JSON_NAME,
        "readme": root / GATES_README_RELPATH,
        "catalog": root / CATALOG_RELPATH,
    }


def _sibling_binding(path: Path, relative_path: str) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": relative_path,
            "status": "unavailable",
            "evidence_kind": "unavailable",
            "reason": "Sibling gate-binding file is not present beside this checkout.",
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": relative_path,
        "status": "observed",
        "evidence_kind": "measured",
        "package_name": payload.get("package_name"),
        "package_version": payload.get("package_version"),
        "policy_kind": payload.get("policy_kind"),
        "gate_kind": payload.get("gate_kind"),
        "branch_policy_cid": payload.get("branch_policy_cid"),
        "release_gate_cid": payload.get("release_gate_cid"),
        "binding_cid": payload.get("binding_cid"),
        "live": False,
        "applied": False,
        "sha256": sha256_bytes(path.read_bytes()),
    }


def platform_gates_catalog(start: Path | None = None) -> dict[str, Any]:
    root = discover_accelerate_root(start)
    if root is None:
        raise BranchAndReleaseGatesError("Accelerate package root was not found")
    parent = root.parent
    branch_policy = render_declared_branch_protection_policy(root)
    release_gate = render_declared_release_gate(root)
    datasets_binding = _sibling_binding(
        parent / "ipfs_datasets" / GATES_DIR_RELPATH / "release.binding.json",
        relative_path=f"../ipfs_datasets/{GATES_DIR_RELPATH}/release.binding.json",
    )
    kit_binding = _sibling_binding(
        parent / "ipfs_kit" / GATES_DIR_RELPATH / "release.binding.json",
        relative_path=f"../ipfs_kit/{GATES_DIR_RELPATH}/release.binding.json",
    )
    catalog = {
        "schema": CATALOG_SCHEMA,
        "interface": CATALOG_INTERFACE,
        "task_id": PCPR_057_TASK_ID,
        "goal_id": PCPR_057_GOAL_ID,
        "policy_kind": BRANCH_POLICY_KIND,
        "gate_kind": RELEASE_GATE_KIND,
        "portfolio_id": PORTFOLIO_ID,
        "portfolio_version": PORTFOLIO_VERSION,
        "branch_policy_cid": branch_policy["branch_policy_cid"],
        "release_gate_cid": release_gate["release_gate_cid"],
        "lock_cid": PCPR_056_LOCK_CID,
        "components": {
            "ipfs_accelerate_py": {
                "branch_policy_path": (
                    f"{GATES_DIR_RELPATH}/{BRANCH_POLICY_JSON_NAME}"
                ),
                "release_gate_path": f"{GATES_DIR_RELPATH}/{RELEASE_GATE_JSON_NAME}",
                "status": "observed",
                "evidence_kind": "measured",
                "package_name": PACKAGE_NAME,
                "package_version": PACKAGE_VERSION,
                "intended_tag_name": INTENDED_TAG_NAME,
                "policy_kind": BRANCH_POLICY_KIND,
                "gate_kind": RELEASE_GATE_KIND,
                "branch_policy_cid": branch_policy["branch_policy_cid"],
                "release_gate_cid": release_gate["release_gate_cid"],
                "live": False,
                "applied": False,
            },
            "ipfs_datasets_py": {"binding": datasets_binding},
            "ipfs_kit_py": {"binding": kit_binding},
        },
        "mutable_main_reference": False,
        "live_branch_protection": False,
        "live_tag_protection": False,
        "governance_gate_complete": False,
        "closed_release_outcome": None,
        "release_claim": False,
        "sibling_source_required": False,
        "evidence_kind": "measured",
    }
    catalog["catalog_cid"] = content_identity(
        {key: value for key, value in catalog.items() if key != "catalog_cid"}
    )
    return catalog


def write_branch_and_release_gate_files(
    start: Path | None = None,
) -> dict[str, Any]:
    root = discover_accelerate_root(start)
    if root is None:
        raise BranchAndReleaseGatesError("Accelerate package root was not found")
    branch_policy = render_declared_branch_protection_policy(root)
    release_gate = render_declared_release_gate(root)
    paths = artifact_paths(root)
    _atomic_write(paths["branch_policy"], pretty_json(branch_policy))
    _atomic_write(paths["release_gate"], pretty_json(release_gate))
    readme = GATES_README if GATES_README.endswith("\n") else GATES_README + "\n"
    _atomic_write(paths["readme"], readme)
    catalog = platform_gates_catalog(start)
    _atomic_write(paths["catalog"], pretty_json(catalog))
    return {
        "branch_policy": branch_policy,
        "release_gate": release_gate,
        "catalog": catalog,
        "paths": {name: str(path) for name, path in paths.items()},
    }


def verify_branch_and_release_gate_files(
    start: Path | None = None,
) -> dict[str, Any]:
    root = discover_accelerate_root(start)
    if root is None:
        raise BranchAndReleaseGatesError("Accelerate package root was not found")
    branch_policy = render_declared_branch_protection_policy(root)
    release_gate = render_declared_release_gate(root)
    paths = artifact_paths(root)
    missing: list[str] = []
    branch_ok = False
    gate_ok = False
    readme_ok = False
    expected_readme = (
        GATES_README if GATES_README.endswith("\n") else GATES_README + "\n"
    )
    for name, path in paths.items():
        if name == "catalog":
            continue
        if not path.is_file():
            missing.append(name)
            continue
        if name == "branch_policy":
            branch_ok = json.loads(path.read_text(encoding="utf-8")) == branch_policy
        elif name == "release_gate":
            gate_ok = json.loads(path.read_text(encoding="utf-8")) == release_gate
        elif name == "readme":
            readme_ok = path.read_text(encoding="utf-8") == expected_readme
    return {
        "ok": not missing and branch_ok and gate_ok and readme_ok,
        "missing": missing,
        "branch_ok": branch_ok,
        "gate_ok": gate_ok,
        "readme_ok": readme_ok,
        "branch_policy_cid": branch_policy["branch_policy_cid"],
        "release_gate_cid": release_gate["release_gate_cid"],
        "branch_policy_sha256": (
            sha256_bytes(paths["branch_policy"].read_bytes())
            if paths["branch_policy"].is_file()
            else "unavailable"
        ),
        "release_gate_sha256": (
            sha256_bytes(paths["release_gate"].read_bytes())
            if paths["release_gate"].is_file()
            else "unavailable"
        ),
    }


def current_head_static_probes(
    start: Path | None = None,
) -> tuple[OutcomeProbe, ...]:
    root = discover_accelerate_root(start)
    if root is None:
        raise BranchAndReleaseGatesError("Accelerate package root was not found")
    branch_policy = render_declared_branch_protection_policy(root)
    release_gate = render_declared_release_gate(root)
    verified = verify_branch_and_release_gate_files(root)
    table = parse_pyproject_gates_table(
        (root / "pyproject.toml").read_text(encoding="utf-8")
    )
    tooling = observe_github_governance_tooling()
    remint = (
        branch_policy["branch_policy_cid"] != PINNED_BRANCH_POLICY_CID
        or release_gate["release_gate_cid"] != PINNED_RELEASE_GATE_CID
        or release_gate["lock_cid"] != PCPR_056_LOCK_CID
    )
    mutable_main = bool(branch_policy["source"]["mutable_main_reference"])
    required_checks = list(release_gate["required_status_checks"])
    probes = [
        _probe(
            "branch_policy_files_match_generator",
            verified.get("ok") is True and verified.get("branch_ok") is True,
            reason=(
                "Committed branch-protection policy matches the generator."
                if verified.get("branch_ok") is True
                else "Committed branch-protection policy is missing or drifts."
            ),
            details=verified,
        ),
        _probe(
            "release_gate_files_match_generator",
            verified.get("ok") is True and verified.get("gate_ok") is True,
            reason=(
                "Committed release gate matches the generator."
                if verified.get("gate_ok") is True
                else "Committed release gate is missing or drifts."
            ),
            details=verified,
        ),
        _probe(
            "pyproject_branch_and_release_gates_table",
            table.get("interface") == INTERFACE
            and table.get("schema") == SCHEMA
            and table.get("task-id") == PCPR_057_TASK_ID
            and table.get("policy-kind") == BRANCH_POLICY_KIND
            and table.get("gate-kind") == RELEASE_GATE_KIND,
            reason=(
                "pyproject.toml declares AccelerateBranchAndReleaseGates@1."
                if table.get("interface") == INTERFACE
                else "pyproject.toml does not declare AccelerateBranchAndReleaseGates@1."
            ),
            details={"table": table},
        ),
        _probe(
            "required_status_checks_declared",
            required_checks == list(REQUIRED_STATUS_CHECKS)
            and release_gate["all_required_checks_must_pass"] is True,
            reason="Declared release gate requires the named current-head and release checks.",
            details={"required_status_checks": required_checks},
        ),
        _probe(
            "partial_required_build_failure_prohibits_release",
            release_gate["partial_required_build_failure_prohibits_release"] is True,
            reason="A partial required-build failure prohibits release publication.",
        ),
        _probe(
            "partial_release_after_required_build_failure",
            False,
            reason="This task does not publish a release after a required-build failure.",
        ),
        _probe(
            "no_mutable_main_reference",
            mutable_main is False,
            reason="Branch and release gates do not pin origin/main as the release identity.",
        ),
        _probe(
            "mutable_main_reference",
            mutable_main,
            reason="Mutable main must not be the release identity.",
        ),
        _probe(
            "branch_policy_kind_is_declared",
            branch_policy["policy_kind"] == BRANCH_POLICY_KIND
            and branch_policy["applied"] is False,
            reason="Policy kind is declared_branch_protection_policy and is not applied.",
        ),
        _probe(
            "release_gate_kind_is_declared",
            release_gate["gate_kind"] == RELEASE_GATE_KIND
            and release_gate["applied"] is False,
            reason="Gate kind is declared_release_gate and is not applied.",
        ),
        _probe(
            "pcpr_056_lock_not_reminted",
            release_gate["lock_cid"] == PCPR_056_LOCK_CID,
            reason="PCPR-056 lock CID is bound and not reminted.",
        ),
        _probe(
            "compatibility_identities_reminted",
            remint,
            reason="A reminted policy, gate, or lock CID is forbidden.",
            details={
                "branch_policy_cid": branch_policy["branch_policy_cid"],
                "release_gate_cid": release_gate["release_gate_cid"],
                "lock_cid": release_gate["lock_cid"],
            },
        ),
        _probe(
            "governance_gate_is_not_complete",
            branch_policy["governance_gate_complete"] is False
            and release_gate["governance_gate_complete"] is False,
            reason="The governance gate is not complete without live GitHub protection.",
        ),
        _probe(
            "governance_gate_represented_as_complete",
            False,
            reason="The governance gate must not be represented as complete.",
        ),
        _probe(
            "operator_blocking_task_emitted",
            branch_policy["operator_blocking_task"]["task_id"]
            == OPERATOR_BLOCKING_TASK_ID
            and branch_policy["operator_blocking_task"]["status"] == "typed_blocked",
            reason=(
                "Missing repository-admin permission emits explicit operator-blocking "
                "task pcpr-057-operator-github-governance."
            ),
        ),
        _probe(
            "named_git_extras_are_not_this_gate",
            True,
            reason=(
                "Named libp2p, mcp-p2p, ipfs-transformers, and ipfs-model-manager "
                "extras may carry source-checkout Git pins; they are not this gate."
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
            "live_branch_protection_represented_as_live",
            False,
            reason="No live GitHub branch protection is represented as live.",
        ),
        _probe(
            "live_tag_protection_represented_as_live",
            False,
            reason="No live GitHub tag protection is represented as live.",
        ),
        _probe(
            "live_github_admin_represented_as_live",
            False,
            reason="No GitHub admin session is represented as live.",
        ),
        _probe(
            "closed_release_represented_as_live",
            False,
            reason="This task does not publish a PCPR release.",
        ),
        _probe(
            "live_branch_protection",
            None,
            evidence_kind="unavailable",
            reason=(
                "gh CLI and a sealed GitHub admin identity are absent. Live "
                "branch protection was not applied and was not invented."
            ),
            details={"tools": tooling.get("governance_tools")},
        ),
        _probe(
            "live_tag_protection",
            None,
            evidence_kind="unavailable",
            reason="Live GitHub tag protection was not observed and was not invented.",
        ),
        _probe(
            "live_github_admin",
            None,
            evidence_kind="unavailable",
            reason="A sealed GitHub admin identity was not admitted.",
        ),
        _probe(
            "live_supervisor_qualification",
            None,
            evidence_kind="unavailable",
            reason="PCPR-001 live qualification remains typed unavailable.",
        ),
    ]
    return tuple(probes)


def qualify_branch_and_release_gates(
    probes: Sequence[OutcomeProbe],
    *,
    branch_policy_cid: str,
    release_gate_cid: str,
    catalog_cid: str,
    lock_cid: str,
) -> BranchAndReleaseGatesVerdict:
    if not probes:
        raise BranchAndReleaseGatesError("at least one probe is required")
    normalized: list[OutcomeProbe] = []
    blockers: list[str] = []
    for probe in probes:
        kind = _kind(probe.evidence_kind, "evidence_kind")
        if probe.live and kind != "measured_live":
            raise BranchAndReleaseGatesError(
                "live claims require measured_live evidence"
            )
        if probe.simulated_represented_as_live:
            raise BranchAndReleaseGatesError(
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
        "task_id": PCPR_057_TASK_ID,
        "goal_id": PCPR_057_GOAL_ID,
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
        "live_branch_protection": False,
        "live_branch_protection_evidence_kind": "unavailable",
        "live_tag_protection": False,
        "live_tag_protection_evidence_kind": "unavailable",
        "live_github_admin": False,
        "governance_gate_complete": False,
        "operator_blocking_task": OPERATOR_BLOCKING_TASK_ID,
        "mutable_main_reference": mutable,
        "simulated_results_represented_as_live": False,
        "this_task_created_competing_authority": False,
        "probes": [item.to_mapping() for item in normalized],
        "blockers": list(dict.fromkeys(blockers)),
        "branch_policy_cid": branch_policy_cid,
        "release_gate_cid": release_gate_cid,
        "catalog_cid": catalog_cid,
        "lock_cid": lock_cid,
    }
    return BranchAndReleaseGatesVerdict(
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
        live_branch_protection=False,
        live_branch_protection_evidence_kind="unavailable",
        live_tag_protection=False,
        live_tag_protection_evidence_kind="unavailable",
        live_github_admin=False,
        governance_gate_complete=False,
        operator_blocking_task=OPERATOR_BLOCKING_TASK_ID,
        mutable_main_reference=mutable,
        simulated_results_represented_as_live=False,
        this_task_created_competing_authority=False,
        probes=tuple(normalized),
        blockers=tuple(dict.fromkeys(blockers)),
        verdict_cid=content_identity(payload),
        branch_policy_cid=branch_policy_cid,
        release_gate_cid=release_gate_cid,
        catalog_cid=catalog_cid,
        lock_cid=lock_cid,
    )


def qualify_current_head_branch_and_release_gates(
    start: Path | None = None,
) -> BranchAndReleaseGatesVerdict:
    root = discover_accelerate_root(start)
    branch_policy = render_declared_branch_protection_policy(root)
    release_gate = render_declared_release_gate(root)
    catalog = platform_gates_catalog(start)
    return qualify_branch_and_release_gates(
        current_head_static_probes(start),
        branch_policy_cid=str(branch_policy["branch_policy_cid"]),
        release_gate_cid=str(release_gate["release_gate_cid"]),
        catalog_cid=str(catalog["catalog_cid"]),
        lock_cid=str(release_gate["lock_cid"]),
    )


def pcpr_057_receipt_promotion(
    verdict: BranchAndReleaseGatesVerdict,
) -> dict[str, Any]:
    if verdict.closed_release_outcome is not None:
        raise BranchAndReleaseGatesError(
            "branch and release gates must not mint a closed release outcome"
        )
    if verdict.release_claim:
        raise BranchAndReleaseGatesError(
            "branch and release gates must not claim a PCPR release"
        )
    if verdict.completion_authoritative:
        raise BranchAndReleaseGatesError(
            "branch-and-release-gates completion is not authoritative"
        )
    if verdict.duckdb_or_quack_state_written:
        raise BranchAndReleaseGatesError(
            "branch and release gates must not write DuckDB or Quack state"
        )
    if verdict.promotion_status in CLOSED_RELEASE_OUTCOMES:
        raise BranchAndReleaseGatesError(
            "promotion_status must not be a closed release outcome"
        )
    if verdict.governance_gate_complete:
        raise BranchAndReleaseGatesError(
            "governance gate must not be represented as complete without live evidence"
        )
    if verdict.live_branch_protection or verdict.live_tag_protection:
        raise BranchAndReleaseGatesError(
            "live GitHub protection requires measured_live evidence"
        )
    return verdict.to_mapping()


# Pinned after the policy encoder is measured. Drift is a remint.
PINNED_BRANCH_POLICY_CID: Final = (
    "baguqeeram3uegwfx6c4i76eyj5nuzlgsqw6yl5vb5c6drkuyacn5l77ksnna"
)
PINNED_RELEASE_GATE_CID: Final = (
    "baguqeerajahrusqwd33fqbuw46zgqcsw4xsuswdqq6td7c3nyddeeg5qofca"
)
CURRENT_HEAD_NON_PROMOTION_VERDICT_CID: Final = (
    "baguqeera7wftrxix6ocid4gaqw7kx6bnxsbybypnyplvx2ytzyo3k6ttkzca"
)


__all__ = [
    "BRANCH_POLICY_KIND",
    "BranchAndReleaseGatesError",
    "BranchAndReleaseGatesVerdict",
    "CATALOG_INTERFACE",
    "CLOSED_RELEASE_OUTCOMES",
    "CURRENT_HEAD_NON_PROMOTION_VERDICT_CID",
    "HERMETIC_CANDIDATE_SUITES",
    "INTERFACE",
    "OPERATOR_BLOCKING_TASK_ID",
    "OutcomeProbe",
    "PCPR_057_GOAL_ID",
    "PCPR_057_TASK_ID",
    "PINNED_BRANCH_POLICY_CID",
    "PINNED_RELEASE_GATE_CID",
    "PORTFOLIO_VERSION",
    "RELEASE_GATE_KIND",
    "REQUIRED_STATUS_CHECKS",
    "SCHEMA",
    "SEALED_PATH",
    "SEALED_PYTHON",
    "current_head_static_probes",
    "observe_github_governance_tooling",
    "pcpr_057_receipt_promotion",
    "platform_gates_catalog",
    "qualify_branch_and_release_gates",
    "qualify_current_head_branch_and_release_gates",
    "refuse_gate_remint",
    "refuse_policy_remint",
    "render_declared_branch_protection_policy",
    "render_declared_release_gate",
    "verify_branch_and_release_gate_files",
    "write_branch_and_release_gate_files",
]
