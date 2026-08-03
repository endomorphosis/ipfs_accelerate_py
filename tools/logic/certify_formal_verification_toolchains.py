#!/usr/bin/env python3
"""Hermetic multi-prover certification for formal verification toolchains.

``FormalVerificationToolchainCertificate@1`` / FVT-G060 (FVT-030).

Runs property-specific offline-pinned live lanes for the real multi-prover
matrix. Available tools must pass positive / negative / mutation / replay
checks with exact identities. Absent or mismatched lanes are explicit
``unavailable`` / ``blocked`` results that only prevent *their own*
promotion. PATH presence alone is never usability. Certification never
installs, downloads, or opens the network, and cross-provider disagreement
is quarantined rather than treated as success.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
import platform
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

try:  # pragma: no cover - script/package import paths vary by worktree
    from tools.logic.certification.public_evidence import (
        public_evidence_audit,
        public_evidence_projection,
    )
except ModuleNotFoundError:  # pragma: no cover
    from certification.public_evidence import (  # type: ignore
        public_evidence_audit,
        public_evidence_projection,
    )

INTERFACE: Final = "FormalVerificationToolchainCertificate@1"
SCHEMA_VERSION: Final = "formal-verification-toolchain-certificate/v1"
GOAL_ID: Final = "FVT-G060"
TASK_ID: Final = "FVT-030"
PROGRAM: Final = "formal-verification-tactician/readiness"
LOCK_INTERFACE: Final = "OfflineToolchainLock@1"
LOCK_SCHEMA: Final = "offline-toolchain-lock/v1"

# Role-aware reissue (FVT-G200 / FVT-053). Elevation is opt-in so the FVT-G060
# hermetic matrix can still distinguish identity-only usability from full
# semantic production certification; the role-aware receipt applies elevation.
# FVT-083 is the objective validation repair that re-proves FVT-G200 and binds
# the synthetic discovery term ``objective validation repair``.
ROLE_AWARE_INTERFACE: Final = "RoleAwareFormalVerificationRelease@1"
ROLE_AWARE_GOAL_ID: Final = "FVT-G200"
ROLE_AWARE_TASK_ID: Final = "FVT-053"
ROLE_AWARE_REPAIR_TASK_ID: Final = "FVT-083"
ROLE_AWARE_OBJECTIVE_VALIDATION_EVIDENCE: Final = "objective validation repair"
ROLE_AWARE_OBJECTIVE_VALIDATION_COMMAND: Final = (
    "python -m pytest "
    "test/integration/test_formal_verification_real_tool_matrix.py "
    "test/integration/test_formal_verification_role_aware_completion.py "
    "test/api/test_formal_verification_tactician_readiness_completion.py -q"
)
RUNTIME_MTL_TS_PACKAGE_RELATIVE: Final = Path(
    "ipfs_datasets_py/typescript/logic-runtime-mtl"
)
RUNTIME_MTL_SEALED_ROOT_ENV: Final = "IPFS_DATASETS_PY_EXTERNAL_PROVER_ROOT"
RUNTIME_MTL_VENDOR_RECEIPT_RELATIVE: Final = Path(
    "docs/architecture/formal_verification_runtime_mtl_external_install_receipt.json"
)
AUTHORIZATION_VENDOR_RECEIPT_RELATIVE: Final = Path(
    "docs/architecture/formal_verification_authorization_vendor_install_receipt.json"
)
HYPERPROPERTY_VENDOR_RECEIPT_RELATIVE: Final = Path(
    "docs/architecture/formal_verification_hyperproperty_vendor_install_receipt.json"
)
RUNTIME_MTL_VENDOR_TOOL_ID: Final = "runtime-mtl-external"
RUNTIME_MTL_VENDOR_PACKAGE_IDENTITY: Final = (
    "@ipfs-datasets/logic-runtime-mtl"
)
RUNTIME_MTL_VENDOR_VERSION: Final = "1.0.0-reviewed"
RUNTIME_MTL_NODE_PROBE_TIMEOUT_SECONDS: Final = 5.0
RUNTIME_MTL_PARITY_TIMEOUT_SECONDS: Final = 10.0
CHECKED_VENDOR_FANIN_SCHEMA: Final = (
    "formal-verification-checked-vendor-semantic-fanin/v1"
)
CHECKED_VENDOR_CAPABILITY_READINESS_SCHEMA: Final = (
    "formal-verification-checked-vendor-capability-readiness/v1"
)
CHECKED_HYPER_VENDOR_FANIN_EVIDENCE_CLASS: Final = (
    "checked_hyperproperty_vendor_bounded_authority"
)

# Pre-merge release candidate fan-in (FVT-G213 / FVT-066). The certificate
# remains the bound matrix; the candidate artifact is assembled by the
# receipt builder and never claims merge or deployment.
RELEASE_CANDIDATE_INTERFACE: Final = (
    "RoleAwareFormalVerificationReleaseCandidate@1"
)
RELEASE_CANDIDATE_GOAL_ID: Final = "FVT-G213"
RELEASE_CANDIDATE_TASK_ID: Final = "FVT-066"
RELEASE_CANDIDATE_MAX_STAGE: Final = "release_candidate"

# Production-semantic elevation fan-in (FVT-G213 / FVT-081). The certificate
# records the durable evidence hooks; the receipt builder performs the
# independent reconstruction and release-candidate fan-in.
PRODUCTION_ELEVATION_FANIN_INTERFACE: Final = (
    "ProductionSemanticElevationFanIn@1"
)
PRODUCTION_ELEVATION_FANIN_GOAL_ID: Final = "FVT-G213"
PRODUCTION_ELEVATION_FANIN_TASK_ID: Final = "FVT-081"
DEFAULT_PRODUCTION_ELEVATION_FANIN_RECEIPT_RELATIVE: Final = Path(
    "docs/architecture/formal_verification_production_elevation_fanin_receipt.json"
)
DEFAULT_PRODUCTION_ELEVATION_FANIN_TEST_RELATIVE: Final = Path(
    "test/integration/toolchains/"
    "test_formal_verification_production_elevation_fanin.py"
)

# Lossless specialized receipt aggregation (FVT-G203 / FVT-065).
# FVT-079 re-proves acceptance when path evidence already exists (objective
# validation repair).
SPECIALIZED_AGGREGATION_INTERFACE: Final = (
    "FormalVerificationSpecializedReceiptAggregation@1"
)
SPECIALIZED_AGGREGATION_SCHEMA: Final = (
    "formal-verification-specialized-receipt-aggregation/v1"
)
SPECIALIZED_AGGREGATION_GOAL_ID: Final = "FVT-G203"
SPECIALIZED_AGGREGATION_TASK_ID: Final = "FVT-065"
SPECIALIZED_AGGREGATION_REPAIR_TASK_ID: Final = "FVT-079"
SPECIALIZED_AGGREGATION_OBJECTIVE_VALIDATION_EVIDENCE: Final = (
    "objective validation repair"
)
SPECIALIZED_AGGREGATION_OBJECTIVE_VALIDATION_COMMAND: Final = (
    "PYTHONPATH=ipfs_datasets_py python -m pytest "
    "test/integration/toolchains/test_formal_verification_specialized_receipt_aggregation.py "
    "test/integration/test_formal_verification_real_tool_matrix.py -q"
)

DEFAULT_LOCK_RELATIVE: Final = Path("config/formal_verification_toolchains.lock.json")
DEFAULT_CERTIFICATE_RELATIVE: Final = Path(
    "docs/architecture/formal_verification_toolchain_certificate.json"
)
DEFAULT_ROLE_AWARE_RECEIPT_RELATIVE: Final = Path(
    "docs/architecture/formal_verification_role_aware_deployment_receipt.json"
)
DEFAULT_RELEASE_CANDIDATE_RELATIVE: Final = Path(
    "docs/architecture/formal_verification_role_aware_release_candidate.json"
)

# Durable, live specialized receipts produced by the focused prover lanes.
# These are optional inputs: absence, stale self-digests, source-surface
# mismatches, failed semantic cases, or missing current artifacts all keep the
# older lane receipt below production authority.  The mapping is deliberately
# lane-specific so a receipt can never be relabelled as evidence for a sibling
# prover family.
LIVE_SPECIALIZED_RECEIPT_SPECS: Final[Mapping[str, Mapping[str, Any]]] = {
    "kernel": {
        "path": Path(
            "docs/architecture/formal_verification_kernel_live_certificate.json"
        ),
        "schema_version": "kernel-live-semantic-fanin/v1",
        "interface": "KernelLiveSemanticFanIn@1",
        "goal_id": "FVT-G206",
        "task_id": "FVT-057",
        "source_modules": (
            Path("tools/logic/certification/lean.py"),
            Path("tools/logic/certification/rocq.py"),
            Path("tools/logic/certification/isabelle.py"),
        ),
        "family": "kernel",
    },
    "kernel_rocq": {
        "path": Path(
            "docs/architecture/formal_verification_kernel_live_certificate.json"
        ),
        "schema_version": "kernel-live-semantic-fanin/v1",
        "interface": "KernelLiveSemanticFanIn@1",
        "goal_id": "FVT-G206",
        "task_id": "FVT-057",
        "source_modules": (
            Path("tools/logic/certification/lean.py"),
            Path("tools/logic/certification/rocq.py"),
            Path("tools/logic/certification/isabelle.py"),
        ),
        "family": "kernel",
    },
    "kernel_isabelle": {
        "path": Path(
            "docs/architecture/formal_verification_kernel_live_certificate.json"
        ),
        "schema_version": "kernel-live-semantic-fanin/v1",
        "interface": "KernelLiveSemanticFanIn@1",
        "goal_id": "FVT-G206",
        "task_id": "FVT-057",
        "source_modules": (
            Path("tools/logic/certification/lean.py"),
            Path("tools/logic/certification/rocq.py"),
            Path("tools/logic/certification/isabelle.py"),
        ),
        "family": "kernel",
    },
    "state_model": {
        "path": Path(
            "docs/architecture/formal_verification_state_model_live_certificate.json"
        ),
        "schema_version": "state-model-live-semantic-certification/v1",
        "interface": "StateModelLiveSemanticCertification@1",
        "goal_id": "FVT-G204",
        "task_id": "FVT-060",
        "source_modules": (
            Path("tools/logic/certification/state_model.py"),
        ),
        "family": "state_model",
    },
    "protocol_tamarin": {
        "path": Path(
            "docs/architecture/formal_verification_protocol_live_certificate.json"
        ),
        "schema_version": "protocol-live-semantic-certification/v1",
        "interface": "ProtocolLiveSemanticCertification@1",
        "goal_id": "FVT-G205",
        "task_id": "FVT-058",
        "source_modules": (
            Path("tools/logic/certification/tamarin.py"),
            Path("tools/logic/certification/proverif.py"),
        ),
        "family": "protocol",
    },
    "protocol_proverif": {
        "path": Path(
            "docs/architecture/formal_verification_protocol_live_certificate.json"
        ),
        "schema_version": "protocol-live-semantic-certification/v1",
        "interface": "ProtocolLiveSemanticCertification@1",
        "goal_id": "FVT-G205",
        "task_id": "FVT-058",
        "source_modules": (
            Path("tools/logic/certification/tamarin.py"),
            Path("tools/logic/certification/proverif.py"),
        ),
        "family": "protocol",
    },
    "atp": {
        "path": Path(
            "docs/architecture/formal_verification_atp_live_certificate.json"
        ),
        "schema_version": "atp-live-semantic-certification/v1",
        "interface": "ATPLiveSemanticCertification@1",
        "goal_id": "FVT-G207",
        "task_id": "FVT-054",
        "source_modules": (Path("tools/logic/certification/atp.py"),),
        "family": "atp",
    },
    "attestation": {
        "path": Path(
            "docs/architecture/formal_verification_zkp_live_deployment_receipt.json"
        ),
        "schema_version": "zkp-live-verifier-deployment/v1",
        "interface": "ZKPLiveVerifierDeployment@1",
        "goal_id": "FVT-G211",
        "task_id": "FVT-059",
        "source_modules": (Path("tools/logic/certification/zkp.py"),),
        "family": "zkp",
    },
}

PROBE_TIMEOUT_SECONDS: Final = 5.0
CHECK_TIMEOUT_SECONDS: Final = 8.0

# Property lanes from FVT-G060. Each lane owns a closed set of tools; absence
# of one tool never conceals or fails unrelated lanes.
PROPERTY_LANES: Final[tuple[dict[str, Any], ...]] = (
    {
        "lane_id": "smt",
        "property_class": "smt_software_verification",
        "description": "SMT solvers for software-verification VCs",
        "tool_ids": ("z3", "cvc5"),
        "authority_tool_ids": ("z3", "cvc5"),
        "check_kind": "smtlib",
    },
    {
        "lane_id": "tla",
        "property_class": "tla_state_model",
        "description": "TLA+/TLC/Apalache state-model checking",
        "tool_ids": ("apalache", "tlc", "java"),
        "authority_tool_ids": ("apalache", "tlc"),
        "check_kind": "identity_only",
    },
    {
        "lane_id": "datalog_secpal",
        "property_class": "authorization_datalog_secpal",
        "description": "Datalog/SecPAL authorization reasoning",
        "tool_ids": (
            "datalog-authorization",
            "secpal-authorization",
            "souffle",
            "secpal",
        ),
        "authority_tool_ids": (
            "datalog-authorization",
            "secpal-authorization",
            "souffle",
            "secpal",
        ),
        "check_kind": "identity_or_in_process",
    },
    {
        "lane_id": "protocol",
        "property_class": "protocol_verification",
        "description": "Tamarin/ProVerif protocol verification",
        "tool_ids": ("tamarin", "proverif", "maude"),
        "authority_tool_ids": ("tamarin", "proverif"),
        "check_kind": "identity_only",
    },
    {
        "lane_id": "hyperltl",
        "property_class": "hyperproperty",
        "description": "HyperLTL / hyperproperty tools",
        "tool_ids": ("hyperltl", "autohyper", "mchyper"),
        "authority_tool_ids": ("hyperltl", "autohyper", "mchyper"),
        "check_kind": "identity_only",
    },
    {
        "lane_id": "atp",
        "property_class": "automated_theorem_proving",
        "description": "First-order ATP portfolio",
        "tool_ids": ("vampire", "eprover"),
        "authority_tool_ids": ("vampire", "eprover"),
        "check_kind": "identity_only",
    },
    {
        "lane_id": "hammer",
        "property_class": "hammer_advisor",
        "description": "Hammer / advisor bridges (non-kernel authority)",
        "tool_ids": ("symbolicai", "ergoai"),
        "authority_tool_ids": ("symbolicai", "ergoai"),
        "check_kind": "identity_or_in_process",
    },
    {
        "lane_id": "kernel",
        "property_class": "interactive_proof_kernel",
        "description": "Lean / Rocq / Isabelle kernels",
        "tool_ids": ("lean", "coq", "isabelle"),
        "authority_tool_ids": ("lean", "coq", "isabelle"),
        "check_kind": "identity_only",
    },
    {
        "lane_id": "runtime_mtl",
        "property_class": "runtime_mtl_monitoring",
        "description": "Runtime MTL monitors",
        "tool_ids": ("runtime-mtl", "runtime-mtl-external"),
        "authority_tool_ids": ("runtime-mtl", "runtime-mtl-external"),
        "check_kind": "identity_or_in_process",
    },
    {
        "lane_id": "attestation",
        "property_class": "attestation_zkp",
        "description": "Attestation / ZKP circuit bindings",
        "tool_ids": ("zkp-circuit",),
        "authority_tool_ids": ("zkp-circuit",),
        "check_kind": "identity_or_in_process",
    },
)

SMT_POSITIVE: Final = """\
(set-logic QF_LIA)
(declare-const x Int)
(assert (and (> x 0) (< x 0)))
(check-sat)
"""

SMT_NEGATIVE: Final = """\
(set-logic QF_LIA)
(declare-const x Int)
(assert (> x 0))
(check-sat)
"""

# Mutation of the positive script: drop one conjunct so the formula becomes sat.
SMT_MUTATED: Final = """\
(set-logic QF_LIA)
(declare-const x Int)
(assert (> x 0))
(check-sat)
"""

# Prefer multi-component versions (4.16.0) over lone digits glued to product
# names (the trailing "3" in "Z3").
_VERSION_TOKEN = re.compile(r"\d+(?:\.\d+)+")
_LONE_VERSION_TOKEN = re.compile(r"\b\d+\b")
SHA256_RE: Final = re.compile(r"[0-9a-f]{64}")
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_JAVA_VERSION_RE = re.compile(
    r'(?im)^\s*(?:openjdk|java)\s+version\s+"'
    r'(?P<version>\d+(?:[._+\-][^"]*)?)"'
)
_SOUFFLE_VERSION_LINE_RE = re.compile(
    r"^Version: (?P<version>\d+(?:\.\d+)+)$"
)
JAVA_OPTION_ENV_VARS: Final = (
    "_JAVA_OPTIONS",
    "JAVA_TOOL_OPTIONS",
    "JDK_JAVA_OPTIONS",
)
APPROVED_IMMUTABLE_DEPLOYMENT_ROOTS: Final[tuple[Path, ...]] = (
    Path("/opt"),
)


# ---------------------------------------------------------------------------
# Offline environment / process helpers
# ---------------------------------------------------------------------------


def repo_root_from(start: Path | None = None) -> Path:
    """Locate the repository root containing the offline toolchain lock."""

    here = (start or Path(__file__).resolve()).resolve()
    candidates = [here] if here.is_dir() else [here.parent]
    candidates.extend(here.parents if not here.is_dir() else here.parents)
    for candidate in candidates:
        if (candidate / DEFAULT_LOCK_RELATIVE).is_file():
            return candidate
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "config"
        ).is_dir():
            return candidate
    return Path.cwd().resolve()


def offline_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Build an environment that blocks opportunistic installs and fetches."""

    env = dict(base if base is not None else os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    env["PIP_NO_INDEX"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["NPM_CONFIG_OFFLINE"] = "true"
    env["npm_config_offline"] = "true"
    env["ELAN_NO_AUTO_INSTALL"] = "1"
    env.setdefault("ELAN_IO_THREADS", "1")
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_OPTIONAL_LOCKS"] = "0"
    env.setdefault("NO_PROXY", "*")
    env.setdefault("no_proxy", "*")
    # Prevent curl|sh installers from being "helpful".
    env["FORMAL_VERIFICATION_CERTIFY_OFFLINE"] = "1"
    env["FORMAL_VERIFICATION_FORBID_INSTALL"] = "1"
    env["FORMAL_VERIFICATION_FORBID_NETWORK"] = "1"
    return env


def first_nonempty_line(text: str) -> str:
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def parse_souffle_version_banner(banner: str | None) -> str | None:
    """Return the sole canonical version from Soufflé's framed banner."""

    versions = [
        match.group("version")
        for line in (banner or "").splitlines()
        if (match := _SOUFFLE_VERSION_LINE_RE.fullmatch(line.strip()))
    ]
    return versions[0] if len(versions) == 1 else None


def parse_java_version_banner(banner: str | None) -> str | None:
    """Return only the quoted java/openjdk identity token, never arbitrary text."""

    match = _JAVA_VERSION_RE.search(banner or "")
    if match is None:
        return None
    return match.group("version")


def java_major_version(banner: str | None) -> int | None:
    """Parse major version from a quoted java/openjdk identity banner only."""

    token = parse_java_version_banner(banner)
    if token is None:
        return None
    components = re.findall(r"\d+", token)
    if not components:
        return None
    if components[0] == "1" and len(components) > 1:
        return int(components[1])
    return int(components[0])


def bounded_run(
    argv: Sequence[str],
    *,
    timeout: float = PROBE_TIMEOUT_SECONDS,
    env: Mapping[str, str] | None = None,
    stdin: str | None = None,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str] | None:
    """Run argv with hard bounds; never shell=True; never raise on timeout."""

    if not argv:
        return None
    if isinstance(argv, (str, bytes, bytearray)):
        raise TypeError("argv must be a sequence of arguments, not a shell string")
    try:
        return subprocess.run(
            list(argv),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=dict(env) if env is not None else offline_env(),
            cwd=str(cwd) if cwd is not None else None,
            input=stdin,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def content_digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _project_semantic_lane_result(
    result: Mapping[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Create a self-consistent public projection of one semantic receipt."""

    projected = public_evidence_projection(result, repo_root=repo_root)
    if not isinstance(projected, dict):
        return {}
    source_digest = str(result.get("digest_sha256") or "")
    receipt = projected.get("receipt")
    if isinstance(receipt, dict):
        for field_name in (
            "receipt_digest_sha256",
            "certificate_digest_sha256",
            "digest_sha256",
        ):
            if field_name in receipt:
                receipt[field_name] = content_digest(
                    {
                        key: value
                        for key, value in receipt.items()
                        if key != field_name
                    }
                )
        projected["digest_sha256"] = content_digest(receipt)
    per_tool = projected.get("per_tool")
    if isinstance(per_tool, Mapping):
        for tool_result in per_tool.values():
            if isinstance(tool_result, dict):
                checks = tool_result.get("checks")
                if isinstance(checks, list):
                    tool_result["check_set_digest_sha256"] = content_digest(checks)
    projected["public_projection"] = {
        "source_receipt_digest_sha256": source_digest or None,
        "portable_paths": True,
        "raw_process_output_retained": False,
        "raw_secret_or_witness_retained": False,
    }
    return projected


def _compact_semantic_tool_projection(
    tool_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a per-tool projection without duplicating its canonical receipt."""

    identity = (
        tool_result.get("identity")
        if isinstance(tool_result.get("identity"), Mapping)
        else {}
    )
    artifacts = [
        dict(item)
        for item in (identity.get("artifacts") or [])
        if isinstance(item, Mapping)
    ]
    artifact_validation = (
        tool_result.get("artifact_validation")
        if isinstance(tool_result.get("artifact_validation"), Mapping)
        else {}
    )
    checks = [
        item
        for item in (tool_result.get("checks") or [])
        if isinstance(item, Mapping)
    ]
    check_status_counts = {
        status: sum(1 for check in checks if check.get("status") == status)
        for status in ("passed", "failed", "skipped", "unavailable")
    }
    check_kinds_present = sorted(
        {
            str(check.get("kind"))
            for check in checks
            if str(check.get("kind") or "")
        }
    )
    return {
        "certified": bool(tool_result.get("certified")),
        "block_reasons": list(tool_result.get("block_reasons") or []),
        "check_kinds_present": check_kinds_present,
        "checks_retained_without_kind_collapse": bool(
            tool_result.get("checks_retained_without_kind_collapse")
        ),
        "checks_passed": check_status_counts["passed"],
        "checks_total": len(checks),
        "check_set_digest_sha256": str(
            tool_result.get("check_set_digest_sha256")
            or content_digest(checks)
        ),
        "check_status_counts": check_status_counts,
        "identity": {
            "executable_path": identity.get("executable_path"),
            "version_string": identity.get("version_string"),
            "identity_probed": bool(identity.get("identity_probed")),
            "artifacts": artifacts,
            "artifacts_digest_sha256": content_digest(artifacts),
        },
        "artifact_validation": {
            "valid": artifact_validation.get("valid") is True,
            "failures": list(artifact_validation.get("failures") or []),
            "has_production_binding": bool(
                artifact_validation.get("has_production_binding")
            ),
            "validated_digest_sha256": content_digest(
                artifact_validation.get("validated") or []
            ),
            "production_bindings_digest_sha256": content_digest(
                artifact_validation.get("production_bindings") or []
            ),
        },
        "handler_key": tool_result.get("handler_key"),
    }


def _compact_semantic_lane_projection(
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Retain one full lane receipt and digest-only derived tool projections."""

    compact = {
        key: value
        for key, value in result.items()
        if key not in {"per_tool"}
    }
    per_tool = result.get("per_tool")
    compact["per_tool"] = {
        str(tool_id): _compact_semantic_tool_projection(tool_result)
        for tool_id, tool_result in (
            per_tool.items() if isinstance(per_tool, Mapping) else ()
        )
        if isinstance(tool_result, Mapping)
    }
    compact["projection_policy"] = {
        "canonical_full_receipt_retained_once": isinstance(
            result.get("receipt"), Mapping
        ),
        "per_tool_checks_bound_by_digest": True,
        "per_tool_artifact_validation_bound_by_digest": True,
    }
    return compact


def _compact_specialized_receipt_aggregation(
    aggregation: Mapping[str, Any],
) -> dict[str, Any]:
    """Project specialized fan-in as identities and content digests only."""

    handlers = aggregation.get("specialized_by_handler")
    handler_projection: dict[str, Any] = {}
    if isinstance(handlers, Mapping):
        for handler_key, raw in sorted(handlers.items()):
            if not isinstance(raw, Mapping):
                continue
            projected_handler = {
                "handler_key": raw.get("handler_key") or handler_key,
                "tool_id": raw.get("tool_id"),
                "property_lane_id": raw.get("property_lane_id"),
                "semantic_lane_id": raw.get("semantic_lane_id"),
                "certifier_family": raw.get("certifier_family"),
                "certified": bool(raw.get("certified")),
                "authority_ceiling": raw.get("authority_ceiling"),
                "receipt_digest_sha256": raw.get("raw_receipt_digest")
                or raw.get("receipt_digest_sha256"),
                "check_set_digest_sha256": raw.get("check_set_digest_sha256")
                or content_digest(raw.get("checks") or []),
                "identity_digest_sha256": content_digest(
                    raw.get("identity") or {}
                ),
                "artifacts_digest_sha256": content_digest(
                    raw.get("artifacts") or []
                ),
                "bindings_digest_sha256": content_digest(
                    raw.get("bindings") or []
                ),
                "cases_digest_sha256": content_digest(raw.get("cases") or []),
                "dependencies_digest_sha256": content_digest(
                    raw.get("dependencies") or []
                ),
                "sources_digest_sha256": content_digest(raw.get("sources") or []),
                "source_tool_evidence_digest_sha256": raw.get(
                    "tool_evidence_digest_sha256"
                ),
            }
            projected_handler["tool_evidence_digest_sha256"] = content_digest(
                projected_handler
            )
            handler_projection[str(handler_key)] = projected_handler

    composites = aggregation.get("composite_lanes")
    composite_projection: dict[str, Any] = {}
    if isinstance(composites, Mapping):
        for lane_id, raw in sorted(composites.items()):
            if not isinstance(raw, Mapping):
                continue
            composite_projection[str(lane_id)] = {
                "property_lane_id": raw.get("property_lane_id") or lane_id,
                "tool_ids": list(raw.get("tool_ids") or []),
                "handler_keys": list(
                    raw.get("specialized_handler_keys") or []
                ),
                "digest_sha256": content_digest(raw),
            }

    policy = (
        aggregation.get("policy")
        if isinstance(aggregation.get("policy"), Mapping)
        else {}
    )
    repair_ok = bool(aggregation.get("objective_validation_repair"))
    projected = {
        "schema_version": aggregation.get("schema_version"),
        "interface": aggregation.get("interface"),
        "goal_id": aggregation.get("goal_id"),
        "task_id": aggregation.get("task_id"),
        "repair_task_id": aggregation.get("repair_task_id")
        or SPECIALIZED_AGGREGATION_REPAIR_TASK_ID,
        "objective_validation_evidence": aggregation.get(
            "objective_validation_evidence"
        )
        or SPECIALIZED_AGGREGATION_OBJECTIVE_VALIDATION_EVIDENCE,
        "objective_validation_repair": repair_ok,
        "objective_validation_command": aggregation.get(
            "objective_validation_command"
        )
        or SPECIALIZED_AGGREGATION_OBJECTIVE_VALIDATION_COMMAND,
        "enabled": bool(
            aggregation.get("interface")
            == SPECIALIZED_AGGREGATION_INTERFACE
        ),
        "lossless": bool(policy.get("lossless")),
        "certifier_families_required": list(
            aggregation.get("certifier_families_required") or []
        ),
        "certifier_families_represented": list(
            aggregation.get("certifier_families_represented") or []
        ),
        "missing_certifier_families": list(
            aggregation.get("missing_certifier_families") or []
        ),
        "all_required_certifiers_represented": bool(
            aggregation.get("all_required_certifiers_represented")
        ),
        "kernel_retained_tool_ids": list(
            aggregation.get("kernel_retained_tool_ids") or []
        ),
        "protocol_retained_tool_ids": list(
            aggregation.get("protocol_retained_tool_ids") or []
        ),
        "composite_lanes": composite_projection,
        "specialized_by_handler": handler_projection,
        "source_aggregation_digest_sha256": aggregation.get(
            "aggregation_digest_sha256"
        ),
        "projection_policy": {
            "canonical_full_receipts_live_in_semantic_lane_results": True,
            "derived_specialized_rows_are_digest_only": True,
            "source_and_projection_digests_are_distinct": True,
        },
        "acceptance": {
            "objective_validation_repair": repair_ok,
            "objective_validation_evidence": (
                SPECIALIZED_AGGREGATION_OBJECTIVE_VALIDATION_EVIDENCE
            ),
            "repair_task_id": SPECIALIZED_AGGREGATION_REPAIR_TASK_ID,
            "goal_id": SPECIALIZED_AGGREGATION_GOAL_ID,
            "task_id": SPECIALIZED_AGGREGATION_TASK_ID,
        },
    }
    projected["aggregation_digest_sha256"] = content_digest(projected)
    return projected


def _compact_elevation_projection(
    elevation: Mapping[str, Any],
) -> dict[str, Any]:
    """Keep an elevation decision while binding, not copying, its checks."""

    checks = elevation.get("checks")
    checks = list(checks) if isinstance(checks, list) else []
    return {
        key: value
        for key, value in elevation.items()
        if key != "checks"
    } | {
        "checks_digest_sha256": elevation.get("checks_digest_sha256")
        or content_digest(checks),
        "checks_count": len(checks),
    }


def _compact_tool_certificate(tool: ToolCertification) -> dict[str, Any]:
    """Keep operational tool fields and shallow checks; bind full evidence."""

    payload = tool.to_dict()
    full_checks = [check.to_dict() for check in tool.checks]
    payload["checks"] = [check.to_public_dict() for check in tool.checks]
    payload["checks_digest_sha256"] = content_digest(full_checks)
    payload["checks_projection"] = "shallow_rows_full_evidence_digest"
    return payload


def file_digest(path: str | Path | None) -> str | None:
    """Return a canonical SHA-256 identity for a regular file."""

    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_file():
        return None
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def classify_executable_artifact(path: str | Path | None) -> str:
    """Classify executable evidence without treating generated shims as provers."""

    if not path:
        return "none"
    candidate = Path(path)
    if not candidate.is_file():
        return "missing"
    try:
        prefix = candidate.read_bytes()[:8192].decode("utf-8", errors="ignore").lower()
    except OSError:
        return "unreadable"
    shim_markers = (
        "hermetic",
        "generated by ",
        "proposal-only",
        "proposal only",
        "shadow shim",
        "parity engine",
    )
    if prefix.startswith("#!") and any(marker in prefix for marker in shim_markers):
        return "generated_hermetic_shim"
    if prefix.startswith("#!"):
        # A launcher script is not the prover artifact.  It can be useful for
        # discovery, but production certification must additionally bind the
        # executable/archive that the launcher dispatches to.
        return "launcher_script"
    return "native_or_managed_binary"


def _canonical_launcher_exec_target(path: Path) -> dict[str, Any]:
    """Resolve a narrowly structured managed launcher without executing it.

    Only the small wrapper grammar emitted by the managed installers is
    accepted.  In particular, there must be exactly one ``exec`` statement,
    its command must be an absolute path, and conditionals, functions, command
    substitutions (apart from the installer-owned ``launcher_dir`` line), and
    arbitrary shell statements fail closed.
    """

    failures: list[str] = []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return {
            "valid": False,
            "failures": ["launcher_unreadable"],
            "target_path": None,
        }
    if len(text.encode("utf-8")) > 64 * 1024:
        return {
            "valid": False,
            "failures": ["launcher_too_large"],
            "target_path": None,
        }

    exec_lines: list[str] = []
    canonical_launcher_dir = (
        'launcher_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"'
    )
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line in {
            "set -eu",
            "set -e",
            "set -euo pipefail",
            "set -eu -o pipefail",
            canonical_launcher_dir,
        }:
            continue
        if line.startswith("unset "):
            if not re.fullmatch(r"unset(?: [A-Za-z_][A-Za-z0-9_]*)+", line):
                failures.append("launcher_unreviewed_unset")
            continue
        if line.startswith("export "):
            # Managed wrappers use literal paths plus ${VAR:+...} suffixes.
            # Backticks/command substitutions/control operators are never
            # required and would make static dispatch analysis unsound.
            if any(token in line for token in ("`", "$(", ";", "&&", "||")):
                failures.append("launcher_dynamic_export")
            elif not re.fullmatch(
                r"export [A-Za-z_][A-Za-z0-9_]*=[^\r\n]+",
                line,
            ):
                failures.append("launcher_unreviewed_export")
            continue
        if line.startswith("exec "):
            exec_lines.append(line)
            continue
        failures.append("launcher_unreviewed_statement")

    if len(exec_lines) != 1:
        failures.append("launcher_exec_count_not_one")
        return {
            "valid": False,
            "failures": sorted(set(failures)),
            "target_path": None,
        }

    try:
        argv = shlex.split(exec_lines[0], posix=True)
    except ValueError:
        argv = []
        failures.append("launcher_exec_parse_failed")
    if len(argv) < 2 or argv[0] != "exec":
        failures.append("launcher_exec_parse_failed")
        target = None
    else:
        raw_target = argv[1]
        if (
            not os.path.isabs(raw_target)
            or "$" in raw_target
            or "`" in raw_target
        ):
            failures.append("launcher_target_not_literal_absolute_path")
            target = None
        else:
            try:
                target = Path(raw_target).resolve()
            except OSError:
                target = Path(raw_target).absolute()

    return {
        "valid": not failures and target is not None,
        "failures": sorted(set(failures)),
        "target_path": str(target) if target is not None else None,
        "exec_argv_prefix": argv[1:2] if len(argv) >= 2 else [],
    }


def _managed_root_for_launcher(path: Path) -> Path:
    """Return the narrow installation root containing a managed launcher."""

    resolved = path.resolve()
    return resolved.parent.parent if resolved.parent.name == "bin" else resolved.parent


def _bare_file_digest(path: Path) -> str:
    return str(file_digest(path) or "").removeprefix("sha256:")


def _runtime_mtl_source_tree_digest(root: Path) -> str:
    """Match the vendor installer's stable digest for a TypeScript source tree."""

    digest = hashlib.sha256()
    if not root.is_dir():
        return digest.hexdigest()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in {"node_modules", "dist", ".git"} for part in relative.parts):
            continue
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_bare_file_digest(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _runtime_mtl_node_env(node: Path) -> dict[str, str]:
    """Return the entire environment for identity probes and parity execution."""

    return {
        "PATH": str(node.parent),
        "HOME": "/nonexistent",
        "LANG": "C",
        "LC_ALL": "C",
        "TZ": "UTC",
        "NO_COLOR": "1",
        "NODE_DISABLE_COLORS": "1",
        "NODE_PATH": "",
        "FORMAL_VERIFICATION_CERTIFY_OFFLINE": "1",
        "FORMAL_VERIFICATION_FORBID_INSTALL": "1",
        "FORMAL_VERIFICATION_FORBID_NETWORK": "1",
        "NPM_CONFIG_OFFLINE": "true",
        "npm_config_offline": "true",
        "NO_PROXY": "*",
        "no_proxy": "*",
    }


def _runtime_mtl_sealed_path_failures(
    root: Path,
    path: Path,
    *,
    expected_directory: bool,
    executable: bool = False,
) -> list[str]:
    """Validate containment, type, ownership, modes, and symlink absence."""

    failures: list[str] = []
    try:
        resolved = path.resolve(strict=True)
        relative = resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError):
        return ["path_not_contained_in_sealed_root"]
    if resolved != path:
        failures.append("path_resolution_changed")

    try:
        root_owner = root.stat().st_uid
    except OSError:
        return ["sealed_root_unreadable"]
    current = root
    for part in relative.parts:
        current = current / part
        try:
            item_stat = current.lstat()
        except OSError:
            failures.append("sealed_path_component_unreadable")
            continue
        if stat.S_ISLNK(item_stat.st_mode):
            failures.append("sealed_path_symlink")
        if item_stat.st_uid != root_owner:
            failures.append("sealed_path_owner_mismatch")
        if stat.S_IMODE(item_stat.st_mode) & 0o222:
            failures.append("sealed_path_writable")

    try:
        final_stat = resolved.stat()
    except OSError:
        failures.append("sealed_path_unreadable")
        return sorted(set(failures))
    if expected_directory:
        if not stat.S_ISDIR(final_stat.st_mode):
            failures.append("sealed_path_not_directory")
    elif not stat.S_ISREG(final_stat.st_mode):
        failures.append("sealed_path_not_regular_file")
    if executable and not (stat.S_IMODE(final_stat.st_mode) & 0o111):
        failures.append("sealed_path_not_executable")
    return sorted(set(failures))


def _parse_runtime_mtl_public_launcher(path: Path) -> dict[str, Any]:
    """Parse only the deterministic vendor launcher grammar; never execute it."""

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return {"valid": False, "failures": ["launcher_unreadable"]}
    if len(text.encode("utf-8")) > 32 * 1024 or "\0" in text:
        return {"valid": False, "failures": ["launcher_invalid_size_or_nul"]}
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    failures: list[str] = []
    if len(lines) != 14:
        return {"valid": False, "failures": ["launcher_statement_count_invalid"]}
    fixed = {
        0: "set -euo pipefail",
        5: 'if [[ ! -x "$NODE" && ! -f "$NODE" ]]; then',
        6: 'echo "runtime-mtl-external: node runtime missing: $NODE" >&2',
        7: "exit 127",
        8: "fi",
        9: 'if [[ ! -f "$CLI" ]]; then',
        10: 'echo "runtime-mtl-external: vendor CLI missing: $CLI" >&2',
        11: "exit 127",
        12: "fi",
        13: 'exec "$NODE" "$CLI" "$@"',
    }
    for index, expected in fixed.items():
        if lines[index] != expected:
            failures.append(f"launcher_statement_{index}_invalid")

    version_match = re.fullmatch(
        r"export RUNTIME_MTL_EXTERNAL_VERSION='([^'\r\n]+)'",
        lines[1],
    )
    identity_match = re.fullmatch(
        r"export RUNTIME_MTL_EXTERNAL_IDENTITY_FILE="
        r"\$\{RUNTIME_MTL_EXTERNAL_IDENTITY_FILE:-'([^'\r\n]+)'\}",
        lines[2],
    )
    node_match = re.fullmatch(r"NODE='([^'\r\n]+)'", lines[3])
    cli_match = re.fullmatch(r"CLI='([^'\r\n]+)'", lines[4])
    for name, match in (
        ("version", version_match),
        ("identity", identity_match),
        ("node", node_match),
        ("cli", cli_match),
    ):
        if match is None:
            failures.append(f"launcher_{name}_assignment_invalid")
    return {
        "valid": not failures,
        "failures": sorted(set(failures)),
        "version": version_match.group(1) if version_match else None,
        "identity_path": identity_match.group(1) if identity_match else None,
        "node_path": node_match.group(1) if node_match else None,
        "cli_path": cli_match.group(1) if cli_match else None,
    }


def _runtime_mtl_identity_relocation_failures(
    identity: Mapping[str, Any],
    *,
    version: str,
) -> list[str]:
    """Require all installer-recorded absolute paths to share one old root."""

    failures: list[str] = []
    old_root_raw = str(identity.get("install_root") or "")
    old_root = Path(old_root_raw)
    if not old_root.is_absolute():
        return ["identity_install_root_not_absolute"]
    suffixes = {
        "executable": (
            "runtime-mtl-vendor",
            RUNTIME_MTL_VENDOR_TOOL_ID,
            version,
            "bin",
            "runtime-mtl-external",
        ),
        "package_dir": (
            "runtime-mtl-vendor",
            RUNTIME_MTL_VENDOR_TOOL_ID,
            version,
            "package",
        ),
        "cli_path": (
            "runtime-mtl-vendor",
            RUNTIME_MTL_VENDOR_TOOL_ID,
            version,
            "package",
            "dist",
            "src",
            "cli.js",
        ),
    }
    for field_name, suffix in suffixes.items():
        expected = old_root.joinpath(*suffix)
        if Path(str(identity.get(field_name) or "")) != expected:
            failures.append(f"identity_{field_name}_relationship_invalid")
    managed_launcher = str(identity.get("managed_launcher") or "")
    if managed_launcher and Path(managed_launcher) != old_root / "bin" / "runtime-mtl":
        failures.append("identity_managed_launcher_relationship_invalid")
    return failures


def _runtime_mtl_managed_prebuilt_binding(
    repo_root: Path,
    *,
    env: Mapping[str, str],
    receipt_path: Path | None = None,
) -> dict[str, Any]:
    """Authenticate a sealed vendor prebuilt without discovery or mutation.

    The sole root authority is ``IPFS_DATASETS_PY_EXTERNAL_PROVER_ROOT`` in the
    caller-supplied offline environment. PATH is deliberately ignored. Every
    path is derived from the exact reviewed version and must be root-owned,
    immutable, non-symlinked, and contained below that approved root.
    """

    checked_receipt = receipt_path or (
        repo_root / RUNTIME_MTL_VENDOR_RECEIPT_RELATIVE
    )
    public: dict[str, Any] = {
        "package_path": RUNTIME_MTL_TS_PACKAGE_RELATIVE.as_posix(),
        "bound": False,
        "authenticated": False,
        "reason": "managed_prebuilt_unavailable",
        "failures": [],
        "certification_builds_or_installs": False,
        "checkout_mutated": False,
        "ambient_path_used": False,
        "sealed_root_environment": RUNTIME_MTL_SEALED_ROOT_ENV,
        "source": None,
        "receipt_path": (
            RUNTIME_MTL_VENDOR_RECEIPT_RELATIVE.as_posix()
            if receipt_path is None
            else "<explicit-test-receipt>"
        ),
        "process_timeout_seconds": RUNTIME_MTL_PARITY_TIMEOUT_SECONDS,
    }
    failures: list[str] = []
    for key in (
        "FORMAL_VERIFICATION_CERTIFY_OFFLINE",
        "FORMAL_VERIFICATION_FORBID_INSTALL",
        "FORMAL_VERIFICATION_FORBID_NETWORK",
    ):
        if str(env.get(key) or "") != "1":
            failures.append(f"offline_environment_missing:{key}")

    raw_root = str(env.get(RUNTIME_MTL_SEALED_ROOT_ENV) or "").strip()
    if not raw_root:
        failures.append("sealed_root_environment_missing")
        root = None
    else:
        candidate_root = Path(raw_root)
        if not candidate_root.is_absolute():
            failures.append("sealed_root_not_absolute")
            root = None
        else:
            try:
                root = candidate_root.resolve(strict=True)
            except (OSError, RuntimeError):
                failures.append("sealed_root_unreadable")
                root = None

    root_stat = None
    if root is not None:
        try:
            root_stat = root.lstat()
        except OSError:
            failures.append("sealed_root_unreadable")
        if root != Path(raw_root):
            failures.append("sealed_root_resolution_changed")
        if root.is_symlink() or root_stat is None or not stat.S_ISDIR(root_stat.st_mode):
            failures.append("sealed_root_not_regular_directory")
        if root_stat is not None and root_stat.st_uid != 0:
            failures.append("sealed_root_not_root_owned")
        if root_stat is not None and stat.S_IMODE(root_stat.st_mode) & 0o222:
            failures.append("sealed_root_writable")
        if not _path_under_approved_immutable_root(root):
            failures.append("sealed_root_not_approved")

    try:
        receipt = json.loads(checked_receipt.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        receipt = None
        failures.append("checked_vendor_receipt_unreadable")
    if not isinstance(receipt, Mapping):
        receipt = {}
        failures.append("checked_vendor_receipt_not_mapping")

    expected_receipt_fields = {
        "schema_version": (
            "formal-verification-runtime-mtl-external-install-receipt/v1"
        ),
        "interface": "ExternalRuntimeMTLVendorCertification@1",
        "goal_id": "FVT-G210",
        "task_id": "FVT-056",
        "repair_task_id": "FVT-072",
        "lane_id": "runtime_mtl_external_vendor",
        "handler_id": "external_runtime_mtl_vendor_certification@1",
        "authority_ceiling": "finite_trace",
    }
    for field_name, expected in expected_receipt_fields.items():
        if receipt.get(field_name) != expected:
            failures.append(f"checked_vendor_receipt_{field_name}_mismatch")
    receipt_digest = content_digest(
        {
            key: value
            for key, value in receipt.items()
            if key != "receipt_digest_sha256"
        }
    )
    if not _digest_matches(receipt.get("receipt_digest_sha256"), receipt_digest):
        failures.append("checked_vendor_receipt_self_digest_mismatch")
    summary = receipt.get("summary")
    if not isinstance(summary, Mapping):
        failures.append("checked_vendor_receipt_summary_missing")
        summary = {}
    try:
        checks_passed = int(summary.get("checks_passed"))
        checks_total = int(summary.get("checks_total"))
    except (TypeError, ValueError):
        checks_passed = 0
        checks_total = -1
    if (
        receipt.get("certified") is not True
        or summary.get("vendor_certified") is not True
        or checks_passed <= 0
        or checks_passed != checks_total
        or list(summary.get("block_reasons") or [])
    ):
        failures.append("checked_vendor_receipt_not_fully_certified")

    engine = receipt.get("runtime_mtl_external")
    if not isinstance(engine, Mapping):
        failures.append("checked_vendor_engine_missing")
        engine = {}
    expected_engine_fields = {
        "tool_id": RUNTIME_MTL_VENDOR_TOOL_ID,
        "version": RUNTIME_MTL_VENDOR_VERSION,
        "package_identity": RUNTIME_MTL_VENDOR_PACKAGE_IDENTITY,
        "usable": True,
        "certified": True,
        "is_vendor_build": True,
        "is_hermetic_parity_engine": False,
        "role": "authority",
        "authority_ceiling": "finite_trace",
        "never_grants_theorem_authority": True,
        "finite_trace_authority_only": True,
        "no_python_reference_dispatch": True,
    }
    for field_name, expected in expected_engine_fields.items():
        if engine.get(field_name) != expected:
            failures.append(f"checked_vendor_engine_{field_name}_mismatch")
    digest_fields = (
        "artifact_sha256",
        "executable_digest_sha256",
        "launcher_digest_sha256",
        "launcher_target_digest_sha256",
        "lockfile_digest_sha256",
        "package_digest_sha256",
        "runtime_digest_sha256",
        "source_digest_sha256",
    )
    for field_name in digest_fields:
        if not SHA256_RE.fullmatch(str(engine.get(field_name) or "")):
            failures.append(f"checked_vendor_engine_{field_name}_invalid")

    version = str(engine.get("version") or RUNTIME_MTL_VENDOR_VERSION)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,127}", version):
        failures.append("checked_vendor_version_invalid")
    paths: dict[str, Path] = {}
    if root is not None:
        version_root = (
            root
            / "runtime-mtl-vendor"
            / RUNTIME_MTL_VENDOR_TOOL_ID
            / version
        )
        package = version_root / "package"
        paths = {
            "version_root": version_root,
            "package": package,
            "identity": version_root / "identity.json",
            "vendor_launcher": version_root / "bin" / "runtime-mtl-external",
            "public_launcher": root / "bin" / "runtime-mtl",
            "package_json": package / "package.json",
            "package_lock": package / "package-lock.json",
            "source": package / "src",
            "source_index": package / "src" / "index.ts",
            "dist": package / "dist",
            "dist_index": package / "dist" / "src" / "index.js",
            "dist_cli": package / "dist" / "src" / "cli.js",
        }
        directory_names = {"version_root", "package", "source", "dist"}
        executable_names = {"vendor_launcher", "public_launcher", "dist_cli"}
        for name, path in paths.items():
            for failure in _runtime_mtl_sealed_path_failures(
                root,
                path,
                expected_directory=name in directory_names,
                executable=name in executable_names,
            ):
                failures.append(f"{name}:{failure}")
        for source_path in sorted(paths["source"].rglob("*")):
            for failure in _runtime_mtl_sealed_path_failures(
                root,
                source_path,
                expected_directory=source_path.is_dir(),
            ):
                failures.append(f"source_tree:{failure}")

    identity: Mapping[str, Any] = {}
    if paths:
        try:
            loaded_identity = json.loads(paths["identity"].read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            loaded_identity = None
        if isinstance(loaded_identity, Mapping):
            identity = loaded_identity
        else:
            failures.append("vendor_identity_unreadable")
    identity_expected = {
        "schema_version": "runtime-mtl-external-vendor-install-receipt/v1",
        "interface": "ExternalRuntimeMTLVendorInstaller@1",
        "goal_id": "FVT-G210",
        "task_id": "FVT-056",
        "tool_id": RUNTIME_MTL_VENDOR_TOOL_ID,
        "version": version,
        "package_identity": RUNTIME_MTL_VENDOR_PACKAGE_IDENTITY,
        "is_vendor_build": True,
        "is_hermetic_parity_engine": False,
        "role": "authority",
        "authority_ceiling": "finite_trace",
        "never_grants_theorem_authority": True,
        "finite_trace_authority_only": True,
        "no_python_reference_dispatch": True,
    }
    for field_name, expected in identity_expected.items():
        if identity.get(field_name) != expected:
            failures.append(f"vendor_identity_{field_name}_mismatch")
    failures.extend(
        _runtime_mtl_identity_relocation_failures(identity, version=version)
    )
    for field_name in digest_fields:
        identity_value = identity.get(field_name)
        if field_name == "launcher_target_digest_sha256":
            identity_value = identity_value or identity.get("cli_artifact_sha256")
        elif field_name == "launcher_digest_sha256":
            identity_value = identity_value or identity.get(
                "executable_digest_sha256"
            )
        if identity_value != engine.get(field_name):
            failures.append(f"vendor_identity_{field_name}_mismatch")

    launcher_binding: Mapping[str, Any] = {}
    if paths:
        launcher_binding = _parse_runtime_mtl_public_launcher(
            paths["public_launcher"]
        )
        if launcher_binding.get("valid") is not True:
            failures.extend(
                str(item) for item in launcher_binding.get("failures") or []
            )
        expected_launcher_paths = {
            "version": version,
            "identity_path": str(paths["identity"]),
            "cli_path": str(paths["dist_cli"]),
        }
        for field_name, expected in expected_launcher_paths.items():
            if launcher_binding.get(field_name) != expected:
                failures.append(f"public_launcher_{field_name}_mismatch")

    node = None
    node_raw = str(launcher_binding.get("node_path") or "")
    if node_raw:
        candidate_node = Path(node_raw)
        if not candidate_node.is_absolute():
            failures.append("node_path_not_absolute")
        else:
            try:
                node = candidate_node.resolve(strict=True)
            except (OSError, RuntimeError):
                failures.append("node_path_unreadable")
            if node is not None:
                try:
                    node_stat = node.lstat()
                except OSError:
                    node_stat = None
                    failures.append("node_path_unreadable")
                if node != candidate_node or candidate_node.is_symlink():
                    failures.append("node_path_resolution_changed")
                if (
                    node_stat is None
                    or not stat.S_ISREG(node_stat.st_mode)
                    or not (stat.S_IMODE(node_stat.st_mode) & 0o111)
                ):
                    failures.append("node_not_regular_executable")
                if node_stat is not None and (
                    node_stat.st_uid != 0
                    or stat.S_IMODE(node_stat.st_mode) & 0o022
                ):
                    failures.append("node_ownership_or_mode_invalid")
    if identity.get("node_executable") != node_raw:
        failures.append("vendor_identity_node_path_mismatch")

    node_banner = ""
    if node is not None and not any(
        failure.startswith("node_") for failure in failures
    ):
        try:
            completed = subprocess.run(
                [str(node), "--version"],
                capture_output=True,
                text=True,
                check=False,
                timeout=RUNTIME_MTL_NODE_PROBE_TIMEOUT_SECONDS,
                env=_runtime_mtl_node_env(node),
            )
        except (OSError, subprocess.TimeoutExpired):
            completed = None
            failures.append("node_version_probe_failed")
        if completed is not None:
            node_banner = (
                (completed.stdout or completed.stderr or "").strip()
            )
            if completed.returncode != 0:
                failures.append("node_version_probe_failed")
    expected_node_version = str(engine.get("node_version") or "")
    if node_banner != f"v{expected_node_version}":
        failures.append("node_version_mismatch")
    if node is not None:
        runtime_digest = hashlib.sha256(
            f"node:{node_banner}:{node}".encode()
        ).hexdigest()
        if runtime_digest != engine.get("runtime_digest_sha256"):
            failures.append("node_runtime_digest_mismatch")

    if paths:
        current_digests = {
            "package_digest_sha256": _bare_file_digest(paths["package_json"]),
            "lockfile_digest_sha256": _bare_file_digest(paths["package_lock"]),
            "source_digest_sha256": _runtime_mtl_source_tree_digest(paths["source"]),
            "executable_digest_sha256": _bare_file_digest(
                paths["vendor_launcher"]
            ),
            "launcher_digest_sha256": _bare_file_digest(
                paths["vendor_launcher"]
            ),
            "launcher_target_digest_sha256": _bare_file_digest(
                paths["dist_cli"]
            ),
        }
        cli_digest = current_digests["launcher_target_digest_sha256"]
        index_digest = _bare_file_digest(paths["dist_index"])
        current_digests["artifact_sha256"] = hashlib.sha256(
            f"{cli_digest}:{index_digest}".encode()
        ).hexdigest()
        for field_name, observed in current_digests.items():
            if observed != engine.get(field_name):
                failures.append(f"sealed_artifact_{field_name}_mismatch")

        repo_package = repo_root / RUNTIME_MTL_TS_PACKAGE_RELATIVE
        repository_digests = {
            "package_digest_sha256": _bare_file_digest(
                repo_package / "package.json"
            ),
            "lockfile_digest_sha256": _bare_file_digest(
                repo_package / "package-lock.json"
            ),
            "source_digest_sha256": _runtime_mtl_source_tree_digest(
                repo_package / "src"
            ),
        }
        for field_name, observed in repository_digests.items():
            if observed != engine.get(field_name):
                failures.append(f"repository_{field_name}_mismatch")

        try:
            package_payload = json.loads(
                paths["package_json"].read_text(encoding="utf-8")
            )
            lock_payload = json.loads(
                paths["package_lock"].read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            package_payload = {}
            lock_payload = {}
            failures.append("sealed_package_metadata_unreadable")
        for payload_name, payload in (
            ("package", package_payload),
            ("lock", lock_payload),
        ):
            if (
                not isinstance(payload, Mapping)
                or payload.get("name") != RUNTIME_MTL_VENDOR_PACKAGE_IDENTITY
                or payload.get("version") != version
            ):
                failures.append(f"sealed_{payload_name}_identity_mismatch")

    failures = sorted(set(failures))
    public.update(
        {
            "bound": not failures,
            "authenticated": not failures,
            "reason": (
                "sealed_vendor_prebuilt_authenticated"
                if not failures
                else "sealed_vendor_prebuilt_rejected"
            ),
            "failures": failures,
            "source": "sealed_managed_vendor_prebuilt" if not failures else None,
            "source_relative": (
                "runtime-mtl-vendor/runtime-mtl-external/"
                f"{version}/package/dist"
            ),
            "receipt_file_sha256": file_digest(checked_receipt),
            "receipt_digest_sha256": receipt.get("receipt_digest_sha256"),
            "identity_sha256": (
                file_digest(paths.get("identity")) if paths else None
            ),
            "package_json_sha256": (
                file_digest(paths.get("package_json")) if paths else None
            ),
            "package_lock_sha256": (
                file_digest(paths.get("package_lock")) if paths else None
            ),
            "source_tree_sha256": (
                str(engine.get("source_digest_sha256") or "") if paths else None
            ),
            "dist_index_sha256": (
                file_digest(paths.get("dist_index")) if paths else None
            ),
            "dist_cli_sha256": (
                file_digest(paths.get("dist_cli")) if paths else None
            ),
            "public_launcher_sha256": (
                file_digest(paths.get("public_launcher")) if paths else None
            ),
            "vendor_launcher_sha256": (
                file_digest(paths.get("vendor_launcher")) if paths else None
            ),
            "node_executable_sha256": (
                file_digest(node) if node is not None else None
            ),
            "node_version": expected_node_version or None,
            "root_owned": bool(root_stat is not None and root_stat.st_uid == 0),
            "immutable": bool(
                root_stat is not None
                and not (stat.S_IMODE(root_stat.st_mode) & 0o222)
            ),
            "containment_verified": not any(
                "contain" in failure or "resolution" in failure
                for failure in failures
            ),
            "process_environment_keys": (
                sorted(_runtime_mtl_node_env(node)) if node is not None else []
            ),
        }
    )
    invocation = None
    if not failures and root is not None:
        invocation = {
            "sealed_root": str(root),
            "timeout_seconds": RUNTIME_MTL_PARITY_TIMEOUT_SECONDS,
        }
    return {"public": public, "invocation": invocation}


def _matching_managed_release_archives(
    entry: Mapping[str, Any],
    *,
    launcher_path: Path,
) -> list[dict[str, Any]]:
    """Bind locally retained archives whose digests match a reviewed pin."""

    managed_root = _managed_root_for_launcher(launcher_path)
    downloads = managed_root / "downloads"
    if not downloads.is_dir():
        return []
    pinned = {
        str(pin.get("sha256") or "").lower()
        for pin in (entry.get("pins") or ())
        if isinstance(pin, Mapping) and str(pin.get("sha256") or "").strip()
    }
    if not pinned:
        return []
    tool_tokens = {
        str(entry.get("tool_id") or "").lower().replace("-", ""),
        str(_pin_version(entry) or "").lower().lstrip("v"),
    }
    tool_tokens.discard("")
    matches: list[dict[str, Any]] = []
    for candidate in sorted(downloads.iterdir()):
        if not candidate.is_file():
            continue
        normalized_name = candidate.name.lower().replace("-", "")
        if tool_tokens and not any(
            token.replace("-", "") in normalized_name
            for token in tool_tokens
        ):
            continue
        digest = file_digest(candidate)
        bare = str(digest or "").removeprefix("sha256:").lower()
        if bare not in pinned:
            continue
        matches.append(
            {
                "kind": "managed_release_archive",
                "path": str(candidate.resolve()),
                "sha256": digest,
                "declared_digest": f"sha256:{bare}",
                "artifact_class": "managed_release_archive",
            }
        )
    return matches


def _managed_state_launcher_binding(
    *,
    executable: Path,
    identity: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Convert a validated state-model manifest into launcher target evidence."""

    managed = identity.get("managed_identity")
    if not isinstance(managed, Mapping):
        return None
    artifacts: list[dict[str, Any]] = []
    failures: list[str] = []
    launcher_rows = managed.get("launchers")
    launcher_rows = launcher_rows if isinstance(launcher_rows, Mapping) else {}
    selected = next(
        (
            row
            for row in launcher_rows.values()
            if isinstance(row, Mapping)
            and row.get("path")
            and Path(str(row["path"])).resolve() == executable.resolve()
        ),
        None,
    )
    if not isinstance(selected, Mapping) or selected.get("structural_match") is not True:
        failures.append("managed_launcher_structural_binding_invalid")

    payload_path = managed.get("payload_path")
    payload_digest = str(managed.get("payload_sha256") or "")
    payload_class = classify_executable_artifact(payload_path)
    if (
        not payload_path
        or managed.get("payload_digest_verified") is not True
        or file_digest(payload_path)
        != (f"sha256:{payload_digest}" if payload_digest else None)
    ):
        failures.append("managed_launcher_payload_digest_invalid")
    else:
        artifacts.append(
            {
                "kind": "launcher_target",
                "path": str(Path(str(payload_path)).resolve()),
                "sha256": f"sha256:{payload_digest}",
                "artifact_class": payload_class,
            }
        )

    artifact_path = managed.get("artifact_path")
    artifact_digest = str(managed.get("artifact_sha256") or "")
    if (
        not artifact_path
        or managed.get("artifact_digest_verified") is not True
        or file_digest(artifact_path)
        != (f"sha256:{artifact_digest}" if artifact_digest else None)
    ):
        failures.append("managed_release_artifact_digest_invalid")
    else:
        artifacts.append(
            {
                "kind": "managed_release_archive",
                "path": str(Path(str(artifact_path)).resolve()),
                "sha256": f"sha256:{artifact_digest}",
                "declared_digest": f"sha256:{artifact_digest}",
                "artifact_class": "managed_release_archive",
            }
        )

    java_path = managed.get("java_executable")
    java_digest = file_digest(java_path)
    if (
        not java_path
        or managed.get("java_executable_present") is not True
        or classify_executable_artifact(java_path) != "native_or_managed_binary"
        or not java_digest
    ):
        failures.append("managed_java_runtime_identity_invalid")
    else:
        artifacts.append(
            {
                "kind": "launcher_runtime",
                "path": str(Path(str(java_path)).resolve()),
                "sha256": java_digest,
                "artifact_class": "native_or_managed_binary",
            }
        )

    if managed.get("launchers_structurally_valid") is not True:
        failures.append("managed_launcher_set_invalid")
    relocation = managed.get("manifest_relocation_binding")
    relocation = relocation if isinstance(relocation, Mapping) else {}
    manifest_bound = bool(
        managed.get("manifest_valid") is True
        or (
            managed.get("manifest_relocation_valid") is True
            and relocation.get("valid") is True
        )
    )
    manifest_path = managed.get("manifest_path")
    manifest_digest = file_digest(manifest_path)
    if manifest_bound and manifest_path and manifest_digest:
        artifacts.append(
            {
                "kind": "managed_runtime_manifest",
                "path": str(Path(str(manifest_path)).resolve()),
                "sha256": manifest_digest,
                "artifact_class": "managed_runtime_manifest",
            }
        )
    else:
        failures.append("managed_runtime_manifest_identity_invalid")
    if not manifest_bound or managed.get("usable") is not True:
        failures.append("managed_runtime_manifest_invalid")
    return {
        "valid": not failures,
        "binding_kind": (
            "managed_state_model_relocated_manifest"
            if managed.get("manifest_relocation_valid") is True
            else "managed_state_model_manifest"
        ),
        "launcher_path": str(executable.resolve()),
        "launcher_sha256": file_digest(executable),
        "target_path": str(payload_path) if payload_path else None,
        "target_sha256": (
            f"sha256:{payload_digest}" if payload_digest else None
        ),
        "target_artifact_class": payload_class,
        "manifest_path": str(manifest_path) if manifest_path else None,
        "manifest_sha256": manifest_digest,
        "manifest_relocation": dict(relocation),
        "artifacts": artifacts,
        "failures": sorted(set(failures)),
    }


def bind_launcher_target_identity(
    entry: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a launcher to its current native/managed target, fail closed."""

    executable_raw = identity.get("executable_path")
    if not executable_raw:
        return {"valid": False, "artifacts": [], "failures": ["launcher_missing"]}
    executable = Path(str(executable_raw)).resolve()
    if classify_executable_artifact(executable) != "launcher_script":
        return {
            "valid": False,
            "artifacts": [],
            "failures": ["artifact_is_not_launcher"],
        }

    if str(entry.get("runtime") or "") == "jvm":
        managed = _managed_state_launcher_binding(
            executable=executable,
            identity=identity,
        )
        if managed is not None:
            return managed
        return {
            "valid": False,
            "artifacts": [],
            "failures": ["managed_state_identity_missing"],
        }

    parsed = _canonical_launcher_exec_target(executable)
    target_raw = parsed.get("target_path")
    artifacts: list[dict[str, Any]] = []
    failures = list(parsed.get("failures") or [])
    target = Path(str(target_raw)).resolve() if target_raw else None
    managed_root = _managed_root_for_launcher(executable)
    if target is None or not target.is_file() or not os.access(target, os.X_OK):
        failures.append("launcher_target_missing_or_not_executable")
    elif target == executable:
        failures.append("launcher_target_self_reference")
    else:
        try:
            target.relative_to(managed_root)
        except ValueError:
            failures.append("launcher_target_outside_managed_root")

    target_class = classify_executable_artifact(target)
    target_digest = file_digest(target)
    if target is not None and target_digest:
        artifacts.append(
            {
                "kind": "launcher_target",
                "path": str(target),
                "sha256": target_digest,
                "artifact_class": target_class,
            }
        )
    archives = _matching_managed_release_archives(
        entry,
        launcher_path=executable,
    )
    artifacts.extend(archives)
    production_target = target_class == "native_or_managed_binary"
    production_archive = bool(archives)
    if not production_target and not production_archive:
        failures.append("launcher_target_has_no_native_or_managed_binding")

    return {
        "valid": not failures,
        "binding_kind": "static_single_exec",
        "launcher_path": str(executable),
        "launcher_sha256": file_digest(executable),
        "target_path": str(target) if target is not None else None,
        "target_sha256": target_digest,
        "target_artifact_class": target_class,
        "managed_archive_digests": [
            item["sha256"] for item in archives
        ],
        "artifacts": artifacts,
        "failures": sorted(set(failures)),
    }


def observed_platform_id() -> str:
    """Return the lock's normalized host platform identifier."""

    system = platform.system().lower()
    machine = platform.machine().lower()
    system_name = {"linux": "linux", "darwin": "darwin"}.get(system, system)
    machine_name = {
        "amd64": "x86_64",
        "x86_64": "x86_64",
        "arm64": "aarch64" if system_name == "linux" else "arm64",
        "aarch64": "aarch64" if system_name == "linux" else "arm64",
    }.get(machine, machine)
    return f"{system_name}-{machine_name}"


# ---------------------------------------------------------------------------
# Pin / shim detectors (shared semantics with packaging gate)
# ---------------------------------------------------------------------------


def detect_lean_shim_toolchain_mismatch(
    selected_toolchain: str | None,
    installed_toolchains: Sequence[str],
) -> bool:
    """True when the selected Lean toolchain is not offline-installed."""

    if not selected_toolchain or not str(selected_toolchain).strip():
        return False
    installed = {
        item.strip() for item in installed_toolchains if item and str(item).strip()
    }
    return selected_toolchain.strip() not in installed


def list_elan_installed_toolchains(
    env: Mapping[str, str] | None = None,
) -> list[str]:
    """Read offline-installed Lean toolchains from the local elan directory."""

    source_env = env if env is not None else os.environ
    neutral_home = Path(str(source_env.get("HOME") or Path.home()))
    elan_home = Path(
        str(source_env.get("ELAN_HOME") or (neutral_home / ".elan"))
    )
    toolchains_dir = elan_home / "toolchains"
    if not toolchains_dir.is_dir():
        return []
    installed: list[str] = []
    for entry in sorted(toolchains_dir.iterdir()):
        if not entry.is_dir():
            continue
        name = entry.name
        if name.startswith("leanprover--lean4---"):
            version = name.split("---", 1)[-1]
            installed.append(f"leanprover/lean4:{version}")
        else:
            installed.append(name.replace("--", "/").replace("---", ":"))
    return installed


def _path_under_approved_immutable_root(path: str | Path | None) -> bool:
    """Return whether a resolved path is beneath a sealed deployment root."""

    if not path:
        return False
    try:
        candidate = Path(path).resolve(strict=True)
    except (OSError, RuntimeError):
        return False
    for raw_root in APPROVED_IMMUTABLE_DEPLOYMENT_ROOTS:
        try:
            candidate.relative_to(Path(raw_root).resolve(strict=True))
        except (OSError, RuntimeError, ValueError):
            continue
        return True
    return False


def _direct_native_lean_identity(
    executable: str | Path | None,
    *,
    banner: str,
    locked_toolchain: str,
) -> dict[str, Any]:
    """Validate a neutral-HOME Lean binary without relying on elan inventory.

    This path is deliberately narrower than the elan-shim path: the executable
    must be a regular native artifact under an approved immutable deployment
    root and its live banner must match the exact locked toolchain version.
    The binary digest is still bound separately by ``certify_tool`` and by the
    live kernel receipt before it can satisfy production authority.
    """

    failures: list[str] = []
    if not executable:
        failures.append("direct_native_lean_executable_missing")
        return {"valid": False, "failures": failures}
    path = Path(executable)
    if (
        not path.is_file()
        or path.is_symlink()
        or not os.access(path, os.X_OK)
    ):
        failures.append("direct_native_lean_not_regular_executable")
    if classify_executable_artifact(path) != "native_or_managed_binary":
        failures.append("direct_native_lean_artifact_class_invalid")
    if not _path_under_approved_immutable_root(path):
        failures.append("direct_native_lean_root_not_approved")
    locked_version = str(locked_toolchain or "").rsplit(":", 1)[-1]
    if (
        not locked_toolchain
        or not banner
        or detect_locked_version_mismatch(locked_version, banner)
    ):
        failures.append("direct_native_lean_locked_version_mismatch")
    return {
        "valid": not failures,
        "path": str(path.resolve()) if path.exists() else str(path),
        "sha256": file_digest(path),
        "artifact_class": classify_executable_artifact(path),
        "locked_toolchain": locked_toolchain or None,
        "failures": sorted(set(failures)),
    }


def _parse_version_tuple(token: str) -> tuple[int, ...]:
    parts = [int(part) for part in token.split(".") if part.isdigit() or part.isdecimal()]
    # Handle pure digit segments only.
    cleaned: list[int] = []
    for segment in token.split("."):
        match = re.match(r"^(\d+)", segment)
        if not match:
            break
        cleaned.append(int(match.group(1)))
    return tuple(cleaned) if cleaned else tuple(parts)


def _extract_version_tokens(text: str) -> list[str]:
    """Extract version-like tokens, preferring dotted multi-component forms."""

    dotted = _VERSION_TOKEN.findall(text or "")
    if dotted:
        return dotted
    return _LONE_VERSION_TOKEN.findall(text or "")


def detect_locked_version_mismatch(
    locked_version: str,
    observed_version_string: str,
) -> bool:
    """True when the locked pin is not reflected in the observed banner.

    Supports exact pins (``1.3.3``, ``v4.31.0``) and simple range pins of the
    form ``>=X,<Y`` (used by the Python Z3 package pin).
    """

    locked = (locked_version or "").strip()
    observed = (observed_version_string or "").strip()
    if not locked:
        return False
    if not observed:
        return True

    # Range pin: >=A,<B or >=A,<=B
    if locked.startswith(">") or locked.startswith("<") or "," in locked:
        return not _range_pin_satisfied(locked, observed)

    candidates = {locked, locked.lstrip("vV")}
    if any(candidate and candidate in observed for candidate in candidates):
        return False
    # Fallback: compare leading numeric tokens.
    locked_tokens = _extract_version_tokens(locked)
    observed_tokens = _extract_version_tokens(observed)
    if locked_tokens and any(token in observed_tokens for token in locked_tokens):
        # Prefer the primary (first) locked token.
        primary = locked_tokens[0]
        return primary not in observed
    return True


def _range_pin_satisfied(range_spec: str, observed: str) -> bool:
    tokens = _extract_version_tokens(observed)
    if not tokens:
        return False
    observed_tuple = _parse_version_tuple(tokens[0])
    if not observed_tuple:
        return False

    lower: tuple[int, ...] | None = None
    upper: tuple[int, ...] | None = None
    upper_inclusive = False
    for clause in range_spec.split(","):
        clause = clause.strip()
        if clause.startswith(">="):
            lower = _parse_version_tuple(clause[2:].strip().lstrip("vV"))
        elif clause.startswith(">"):
            # Exclusive lower — treat as next micro by requiring strictly greater.
            lower = _parse_version_tuple(clause[1:].strip().lstrip("vV"))
            # Approximate exclusive lower as not-equal handling below.
            if observed_tuple == lower:
                return False
        elif clause.startswith("<="):
            upper = _parse_version_tuple(clause[2:].strip().lstrip("vV"))
            upper_inclusive = True
        elif clause.startswith("<"):
            upper = _parse_version_tuple(clause[1:].strip().lstrip("vV"))
            upper_inclusive = False
    if lower is not None and observed_tuple < lower:
        return False
    if upper is not None:
        if upper_inclusive:
            if observed_tuple > upper:
                return False
        elif observed_tuple >= upper:
            return False
    return True


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    check_id: str
    kind: str  # positive | negative | mutation | replay
    status: str  # passed | failed | skipped | unavailable
    expected: str
    observed: str
    detail: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_public_dict(self) -> dict[str, Any]:
        """Public projection without nested evidence that re-emits the row."""

        payload = {
            "check_id": self.check_id,
            "kind": self.kind,
            "status": self.status,
            "expected": self.expected,
            "observed": self.observed,
            "detail": self.detail,
        }
        if self.evidence:
            # Bind residual evidence by digest only; full tool/lane surfaces
            # retain complete check bodies where required.
            residual_keys = sorted(
                key
                for key in self.evidence.keys()
                if key
                not in {
                    "check_id",
                    "kind",
                    "status",
                    "expected",
                    "observed",
                    "detail",
                    "reason_codes",
                    "tool_id",
                }
            )
            payload["evidence"] = {
                "compacted": True,
                "digest_sha256": content_digest(self.evidence),
                "residual_keys": residual_keys,
            }
        return payload


@dataclass
class ToolCertification:
    tool_id: str
    lane_ids: list[str] = field(default_factory=list)
    families: list[str] = field(default_factory=list)
    availability_declared: str = ""
    executable_path: str | None = None
    executable_sha256: str | None = None
    executable_artifact_class: str = "none"
    version_string: str | None = None
    locked_version: str = ""
    locked_version_mismatch: bool = False
    shim_toolchain_mismatch: bool = False
    path_present: bool = False
    identity_probed: bool = False
    installed: bool = False
    usable: bool = False
    production_certified: bool = False
    unavailable: bool = False
    promotion_blocked: bool = True
    block_reasons: list[str] = field(default_factory=list)
    checks: list[CheckResult] = field(default_factory=list)
    artifact_identities: list[dict[str, Any]] = field(default_factory=list)
    launcher_binding: dict[str, Any] = field(default_factory=dict)
    semantic_receipt_digests: list[str] = field(default_factory=list)
    evidence_class: str = "unavailable"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["checks"] = [check.to_dict() for check in self.checks]
        return payload


@dataclass
class DisagreementQuarantine:
    quarantine_id: str
    lane_id: str
    property_class: str
    tool_ids: list[str]
    outcomes: dict[str, str]
    status: str = "quarantined"
    reason: str = "cross_provider_disagreement"
    promotion_blocked_tool_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LaneCertification:
    lane_id: str
    property_class: str
    description: str
    tool_ids: list[str]
    authority_tool_ids: list[str]
    certified_tool_ids: list[str] = field(default_factory=list)
    certified_authority_tool_ids: list[str] = field(default_factory=list)
    unavailable_tool_ids: list[str] = field(default_factory=list)
    blocked_tool_ids: list[str] = field(default_factory=list)
    disagreement_quarantine_ids: list[str] = field(default_factory=list)
    promotion_ready: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Lock loading
# ---------------------------------------------------------------------------


def load_lock(lock_path: Path) -> dict[str, Any]:
    if not lock_path.is_file():
        raise FileNotFoundError(f"offline toolchain lock missing: {lock_path}")
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("toolchain lock must be a JSON object")
    return payload


def lock_tools_by_id(lock: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    tools = lock.get("tools") or []
    if not isinstance(tools, list):
        raise ValueError("lock.tools must be a list")
    return {str(entry["tool_id"]): entry for entry in tools if "tool_id" in entry}


def tool_platform_support(
    entry: Mapping[str, Any],
    *,
    host_platform: str,
    global_supported_platforms: Sequence[str],
) -> dict[str, Any]:
    """Derive per-tool support from the reviewed deployment lock.

    Deployment-contract platforms are authoritative when present. Otherwise,
    exact pins, ``any`` pins, and reviewed source pins are considered. Missing
    platform metadata fails closed for managed tools.
    """

    globally_supported = host_platform in set(global_supported_platforms)
    contract = entry.get("deployment_contract") or {}
    contract_platforms = [
        str(item) for item in (contract.get("supported_platforms") or [])
    ]
    pin_platforms = [
        str(pin.get("platform") or "")
        for pin in (entry.get("pins") or [])
        if isinstance(pin, Mapping)
    ]
    availability = str(entry.get("availability") or "")
    managed = availability == "managed_pin"

    contract_platforms = [item for item in contract_platforms if item]
    pin_platforms = [item for item in pin_platforms if item]
    declared = sorted(set(contract_platforms) | set(pin_platforms))
    if not managed:
        status = "supported_here" if globally_supported else "ambiguous"
        basis = "global_platform_policy"
    else:
        observations: list[tuple[str, bool]] = []
        if contract_platforms:
            observations.append(
                (
                    "deployment_contract.supported_platforms",
                    bool(
                        "any" in contract_platforms
                        or host_platform in contract_platforms
                    ),
                )
            )
        if pin_platforms:
            observations.append(
                (
                    "tool.pins.platform",
                    bool(
                        {"any", "source"} & set(pin_platforms)
                        or host_platform in pin_platforms
                    ),
                )
            )
        if not observations:
            status = "ambiguous"
            basis = "managed_tool_platform_metadata_missing"
        else:
            contract_support = (
                observations[0][1] if contract_platforms else None
            )
            pin_support = (
                next(
                    (
                        value
                        for source, value in observations
                        if source == "tool.pins.platform"
                    ),
                    None,
                )
            )
            # The reviewed deployment contract is the support ceiling.  Pins
            # may narrow a claimed-supported contract when no artifact can run
            # here, but a generic/source pin cannot broaden an explicit host
            # exclusion (notably external SecPAL on linux-aarch64).
            if contract_support is False:
                tool_supported = False
                basis = "deployment_contract.supported_platforms"
            elif contract_support is True and pin_support is False:
                status = "ambiguous"
                basis = "supported_contract_without_host_artifact_pin"
                tool_supported = None
            elif contract_support is True:
                tool_supported = True
                basis = "deployment_contract.supported_platforms"
            else:
                tool_supported = bool(pin_support)
                basis = "tool.pins.platform"

            if tool_supported is None:
                pass
            elif tool_supported and not globally_supported:
                status = "ambiguous"
                basis = "tool_and_global_platform_policy_contradict"
            elif tool_supported:
                status = "supported_here"
            else:
                status = "unsupported_here"

    supported = status == "supported_here"

    return {
        "tool_id": str(entry.get("tool_id") or ""),
        "host_platform": host_platform,
        "availability": availability,
        "managed": managed,
        "supported": bool(supported),
        "classification": status,
        "ambiguous": status == "ambiguous",
        "exception_eligible": status == "unsupported_here",
        "basis": basis,
        "declared_platforms": declared,
        "globally_supported": globally_supported,
    }


# ---------------------------------------------------------------------------
# Identity probing
# ---------------------------------------------------------------------------


def resolve_executable(
    candidates: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
) -> str | None:
    """Resolve candidates against the caller's environment.

    Certification passes an explicit offline environment.  Its ``PATH`` is
    therefore the complete bare-name search authority; falling back to the
    certifier process's ambient ``PATH`` could select an unreviewed host tool.
    Callers that omit ``env`` retain the conventional ambient-PATH behavior.
    """

    search_path = (
        os.environ.get("PATH", "")
        if env is None
        else str(env.get("PATH") or "")
    )
    for name in candidates:
        if not name:
            continue
        # Bare names are PATH lookups.  Treating a same-name file in the
        # certifier's cwd as an executable lets an unreviewed checkout file
        # shadow the locked tool.
        if os.path.isabs(name) or os.sep in name:
            path = Path(name)
            if path.is_file() and os.access(path, os.X_OK):
                return str(path.resolve())
            continue
        if not search_path:
            continue
        found = shutil.which(name, path=search_path)
        if found:
            return found
    return None


def _bare_sha256(value: Any) -> str:
    return str(value or "").strip().lower().removeprefix("sha256:")


def _relocated_state_manifest_binding(
    *,
    root: Path,
    managed: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate an installer manifest after a sealed whole-tree relocation.

    State-model manifests intentionally record absolute publication paths.
    Copying an otherwise immutable prover tree therefore makes the installer's
    exact manifest comparison fail.  This adapter does not rewrite or bless
    arbitrary paths: it accepts only a current root under the approved
    immutable deployment prefix, requires every old artifact/payload/launcher
    path to share one prior root and the exact current relative suffix, and
    independently rehashes the current archive, payload, JVM, and structurally
    regenerated launchers.
    """

    datasets_root = repo_root_from() / "ipfs_datasets_py"
    datasets_text = str(datasets_root)
    if datasets_text not in sys.path:
        sys.path.insert(0, datasets_text)
    from ipfs_datasets_py.logic.backends.installers import state_model

    return state_model.validate_relocated_managed_manifest(
        root,
        managed_identity=managed,
        approved_root_prefixes=APPROVED_IMMUTABLE_DEPLOYMENT_ROOTS,
    )

    failures: list[str] = []
    current_root = root.resolve()
    if not _path_under_approved_immutable_root(current_root):
        failures.append("relocated_state_root_not_approved")

    tool_id = str(managed.get("tool_id") or "")
    version = str(managed.get("version") or "")
    manifest_path = current_root / "manifests" / f"{tool_id}.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        manifest = None
    if (
        manifest_path.is_symlink()
        or not isinstance(manifest, Mapping)
    ):
        failures.append("relocated_state_manifest_unreadable")
        manifest = {}

    common_keys = {
        "schema_version",
        "tool_id",
        "version",
        "artifact_path",
        "artifact_sha256",
        "payload_path",
        "payload_sha256",
        "java_executable",
        "launchers",
    }
    optional_keys = (
        {"release_tag", "revision"}
        if tool_id == "tlc"
        else {"distribution_tree_sha256"}
        if tool_id == "apalache"
        else set()
    )
    if set(manifest) != common_keys | optional_keys:
        failures.append("relocated_state_manifest_field_population_invalid")
    if (
        manifest.get("schema_version") != "state-model-managed-runtime/v1"
        or manifest.get("tool_id") != tool_id
        or manifest.get("version") != version
        or tool_id not in {"tlc", "apalache"}
    ):
        failures.append("relocated_state_manifest_identity_mismatch")

    previous_roots: set[str] = set()
    bound_paths: list[dict[str, Any]] = []

    def bind_suffix(
        *,
        label: str,
        old_path_value: Any,
        current_path_value: Any,
    ) -> Path | None:
        try:
            old_path = Path(str(old_path_value))
            current_path = Path(str(current_path_value)).resolve()
            relative = current_path.relative_to(current_root)
        except (OSError, RuntimeError, ValueError):
            failures.append(f"relocated_state_{label}_current_path_invalid")
            return None
        if not old_path.is_absolute() or any(
            part in {"", ".", ".."} for part in old_path.parts[1:]
        ):
            failures.append(f"relocated_state_{label}_old_path_invalid")
            return None
        relative_parts = relative.parts
        if (
            not relative_parts
            or len(old_path.parts) <= len(relative_parts)
            or old_path.parts[-len(relative_parts):] != relative_parts
        ):
            failures.append(f"relocated_state_{label}_suffix_mismatch")
            return None
        previous_root = Path(*old_path.parts[:-len(relative_parts)])
        previous_roots.add(str(previous_root))
        bound_paths.append(
            {
                "kind": label,
                "previous_relative_suffix": relative.as_posix(),
                "current_path": str(current_path),
                "current_sha256": file_digest(current_path),
            }
        )
        return current_path

    artifact_path = bind_suffix(
        label="artifact",
        old_path_value=manifest.get("artifact_path"),
        current_path_value=managed.get("artifact_path"),
    )
    payload_path = bind_suffix(
        label="payload",
        old_path_value=manifest.get("payload_path"),
        current_path_value=managed.get("payload_path"),
    )
    artifact_digest = _bare_sha256(file_digest(artifact_path))
    payload_digest = _bare_sha256(file_digest(payload_path))
    expected_artifact_digest = _bare_sha256(managed.get("artifact_sha256"))
    expected_payload_digest = _bare_sha256(managed.get("payload_sha256"))
    if (
        not artifact_digest
        or artifact_digest != expected_artifact_digest
        or _bare_sha256(manifest.get("artifact_sha256"))
        != expected_artifact_digest
        or managed.get("artifact_digest_verified") is not True
    ):
        failures.append("relocated_state_artifact_digest_mismatch")
    if (
        not payload_digest
        or payload_digest != expected_payload_digest
        or _bare_sha256(manifest.get("payload_sha256"))
        != expected_payload_digest
        or managed.get("payload_digest_verified") is not True
    ):
        failures.append("relocated_state_payload_digest_mismatch")

    if tool_id == "tlc":
        if (
            manifest.get("release_tag") != managed.get("release_tag")
            or manifest.get("revision") != managed.get("revision")
            or managed.get("jar_manifest_valid") is not True
        ):
            failures.append("relocated_state_tlc_release_identity_mismatch")
    elif (
        manifest.get("distribution_tree_sha256")
        != managed.get("expected_distribution_tree_sha256")
        or managed.get("expected_distribution_tree_sha256")
        != managed.get("observed_distribution_tree_sha256")
        or managed.get("distribution_tree_verified") is not True
        or managed.get("payload_executable") is not True
    ):
        failures.append("relocated_state_apalache_tree_identity_mismatch")

    manifest_launchers = manifest.get("launchers")
    managed_launchers = managed.get("launchers")
    manifest_launchers = (
        manifest_launchers
        if isinstance(manifest_launchers, Mapping)
        else {}
    )
    managed_launchers = (
        managed_launchers
        if isinstance(managed_launchers, Mapping)
        else {}
    )
    if set(manifest_launchers) != set(managed_launchers) or not managed_launchers:
        failures.append("relocated_state_launcher_population_mismatch")
    launcher_digests: dict[str, str] = {}
    for launcher_name, raw_current in sorted(managed_launchers.items()):
        current = raw_current if isinstance(raw_current, Mapping) else {}
        prior = manifest_launchers.get(launcher_name)
        prior = prior if isinstance(prior, Mapping) else {}
        if set(prior) != {"path", "sha256"}:
            failures.append(
                f"relocated_state_launcher_manifest_invalid:{launcher_name}"
            )
        current_path = bind_suffix(
            label=f"launcher:{launcher_name}",
            old_path_value=prior.get("path"),
            current_path_value=current.get("path"),
        )
        current_digest = _bare_sha256(file_digest(current_path))
        launcher_digests[launcher_name] = current_digest
        if (
            current.get("structural_match") is not True
            or current.get("present") is not True
            or current.get("executable") is not True
            or not SHA256_RE.fullmatch(str(prior.get("sha256") or ""))
            or not current_digest
            or current_digest
            != _bare_sha256(current.get("observed_sha256"))
            or current_digest
            != _bare_sha256(current.get("expected_sha256"))
        ):
            failures.append(
                f"relocated_state_launcher_identity_invalid:{launcher_name}"
            )
    if managed.get("launchers_structurally_valid") is not True:
        failures.append("relocated_state_launcher_set_invalid")

    java_path = Path(str(managed.get("java_executable") or ""))
    java_digest = file_digest(java_path)
    if (
        managed.get("java_executable_present") is not True
        or classify_executable_artifact(java_path)
        != "native_or_managed_binary"
        or not java_digest
        or Path(str(manifest.get("java_executable") or "")).name != "java"
    ):
        failures.append("relocated_state_java_identity_invalid")

    if len(previous_roots) != 1 or str(current_root) in previous_roots:
        failures.append("relocated_state_previous_root_population_invalid")

    return {
        "valid": not failures,
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_digest(manifest_path),
        "previous_root": (
            next(iter(previous_roots)) if len(previous_roots) == 1 else None
        ),
        "current_root": str(current_root),
        "bound_paths": bound_paths,
        "java_executable": str(java_path.resolve())
        if java_path.is_file()
        else str(java_path),
        "java_sha256": java_digest,
        "launcher_sha256": launcher_digests,
        "failures": sorted(set(failures)),
    }


def _managed_state_model_identity(
    tool_id: str,
    executable: str,
    *,
    env: Mapping[str, str],
) -> dict[str, Any] | None:
    """Validate the complete managed TLC/Apalache payload and launcher bundle."""

    if tool_id not in {"tlc", "apalache"}:
        return None
    datasets_root = repo_root_from() / "ipfs_datasets_py"
    datasets_text = str(datasets_root)
    if datasets_text not in sys.path:
        sys.path.insert(0, datasets_text)
    from ipfs_datasets_py.logic.backends.installers import state_model

    explicit_java = str(env.get(state_model.JAVA_EXECUTABLE_ENV) or "").strip()
    if not explicit_java:
        java_home = str(env.get("JAVA_HOME") or "").strip()
        if java_home:
            explicit_java = str(Path(java_home) / "bin" / "java")
    if not explicit_java:
        explicit_java = shutil.which("java", path=env.get("PATH")) or ""
    minimum = (
        state_model.TLC_MIN_JAVA_MAJOR
        if tool_id == "tlc"
        else state_model.APALACHE_MIN_JAVA_MAJOR
    )
    java = state_model.probe_java_runtime(
        java_executable=explicit_java or None,
        minimum_major=minimum,
    )
    path = Path(executable)
    if not java.usable or java.executable is None or path.parent.name != "bin":
        return {
            "usable": False,
            "reason": "validated_java_or_managed_root_missing",
            "java_runtime": java.to_dict(),
        }
    root = path.parent.parent
    identity = (
        state_model.managed_tlc_identity(
            root,
            java_executable=java.executable,
        )
        if tool_id == "tlc"
        else state_model.managed_apalache_identity(
            root,
            java_executable=java.executable,
        )
    )
    identity["java_runtime"] = java.to_dict()
    if identity.get("manifest_valid") is not True:
        relocation = _relocated_state_manifest_binding(
            root=root,
            managed=identity,
        )
        identity["manifest_relocation_binding"] = relocation
        identity["manifest_relocation_valid"] = (
            relocation.get("valid") is True
        )
        if relocation.get("valid") is True:
            identity["usable"] = True
    return identity


def _first_pin_sha256(entry: Mapping[str, Any]) -> str:
    pins = entry.get("pins") or ()
    if not pins or not isinstance(pins[0], Mapping):
        return ""
    return str(pins[0].get("sha256") or "").strip().lower()


def probe_tool_identity(
    entry: Mapping[str, Any],
    *,
    env: Mapping[str, str],
) -> dict[str, Any]:
    """Bounded offline identity probe. PATH alone is never usability."""

    tool_id = str(entry.get("tool_id") or "")
    candidates = list(entry.get("executable_candidates") or [])
    availability = str(entry.get("availability") or "")
    probe = dict(entry.get("offline_probe") or {})
    timeout = float(probe.get("timeout_seconds") or PROBE_TIMEOUT_SECONDS)
    argv_suffix = list(probe.get("argv") or ["--version"])

    result: dict[str, Any] = {
        "tool_id": tool_id,
        "path_present": False,
        "executable_path": None,
        "version_string": None,
        "identity_probed": False,
        "installed": False,
        "shim_toolchain_mismatch": False,
        "selected_toolchain": None,
        "installed_toolchains": [],
        "lean_identity_mode": None,
        "direct_native_lean_identity": {},
        "in_process": False,
        "probe_error": None,
    }

    # In-process / declared-gap tools have no executable requirement.
    if availability in {"in_process", "declared_gap", "advisor_only"} and not candidates:
        result["in_process"] = True
        # Still treat declared_gap as not installed unless a module probe succeeds.
        if availability == "declared_gap":
            result["installed"] = False
            result["probe_error"] = "declared_gap"
            return result
        if availability == "in_process":
            module_ok, version = _probe_in_process_module(tool_id, env=env)
            result["identity_probed"] = module_ok
            result["installed"] = module_ok
            result["version_string"] = version
            return result
        if availability == "advisor_only":
            # Advisors may be optional Python packages — probe lightly.
            module_ok, version = _probe_in_process_module(tool_id, env=env)
            result["identity_probed"] = module_ok
            result["installed"] = module_ok
            result["version_string"] = version
            return result

    executable = resolve_executable(candidates or [tool_id], env=env)
    if executable is None:
        result["probe_error"] = "executable_not_on_path"
        return result

    result["path_present"] = True
    result["executable_path"] = executable

    probe_env = dict(env)
    if tool_id == "java":
        # Hostile Java option variables can replace the identity banner or
        # force an otherwise valid runtime to fail before the probe starts.
        for key in JAVA_OPTION_ENV_VARS:
            probe_env.pop(key, None)
    selected_lean_toolchain: str | None = None
    if tool_id == "lean":
        installed_toolchains = list_elan_installed_toolchains(probe_env)
        locked_toolchain = str(probe.get("locked_toolchain") or "").strip()
        result["installed_toolchains"] = installed_toolchains
        if locked_toolchain:
            # A direct immutable Lean binary ignores ELAN_TOOLCHAIN.  An elan
            # proxy must select this exact already-installed pin and is unable
            # to auto-install it under the offline environment.
            probe_env["ELAN_TOOLCHAIN"] = locked_toolchain
        if locked_toolchain and locked_toolchain in installed_toolchains:
            selected_lean_toolchain = locked_toolchain

    completed = bounded_run(
        [executable, *argv_suffix],
        timeout=timeout,
        env=probe_env,
    )
    if completed is None:
        result["probe_error"] = "probe_timeout_or_spawn_failure"
        # PATH presence without a successful identity probe is not installed.
        return result
    combined = "\n".join(
        part for part in (completed.stdout, completed.stderr) if part
    ).strip()
    if tool_id == "tlc":
        cleaned = _ANSI_ESCAPE_RE.sub("", combined)
        accepted_returncodes = {
            int(value) for value in probe.get("accepted_returncodes") or (0, 1)
        }
        required_markers = tuple(
            str(value) for value in probe.get("required_markers") or ()
        )
        semantic_help = bool(required_markers) and all(
            marker in cleaned for marker in required_markers
        )
        managed_identity = _managed_state_model_identity(
            tool_id,
            executable,
            env=probe_env,
        )
        result["managed_identity"] = managed_identity
        managed_digest = str(
            (managed_identity or {}).get("artifact_sha256") or ""
        ).lower()
        managed_revision = str(
            (managed_identity or {}).get("revision") or ""
        ).strip()
        managed_release_tag = str(
            (managed_identity or {}).get("release_tag") or ""
        ).strip()
        expected_revision = str(probe.get("revision") or "").strip()
        expected_release_tag = str(probe.get("release_tag") or "").strip()
        digest_bound = bool(
            managed_digest
            and _first_pin_sha256(entry) == managed_digest
            and str(probe.get("artifact_sha256") or "").lower()
            == managed_digest
        )
        # Lock metadata states what is expected; it cannot substitute for
        # observations from the managed installation.  Both identity fields
        # must be present on the managed payload and match the lock exactly.
        revision_bound = bool(
            managed_revision
            and expected_revision
            and managed_revision == expected_revision
            and managed_release_tag
            and expected_release_tag
            and managed_release_tag == expected_release_tag
        )
        if (
            completed.returncode not in accepted_returncodes
            or not semantic_help
            or not managed_identity
            or not managed_identity.get("usable")
            or not digest_bound
            or not revision_bound
        ):
            result["probe_error"] = (
                "tlc_help_or_managed_digest_identity_failed"
            )
            return result
        result["version_string"] = (
            f"TLC managed release {_pin_version(entry)} "
            f"({managed_release_tag}@{managed_revision}); "
            f"artifact sha256:{managed_identity['artifact_sha256']}"
        )
        result["identity_probed"] = True
        result["installed"] = True
        return result

    if completed.returncode != 0:
        result["probe_error"] = f"identity_probe_nonzero:{completed.returncode}"
        return result

    if tool_id == "apalache":
        managed_identity = _managed_state_model_identity(
            tool_id,
            executable,
            env=probe_env,
        )
        result["managed_identity"] = managed_identity
        managed_digest = str(
            (managed_identity or {}).get("artifact_sha256") or ""
        ).lower()
        digest_bound = bool(
            managed_digest
            and _first_pin_sha256(entry) == managed_digest
            and str(probe.get("artifact_sha256") or "").lower()
            == managed_digest
        )
        if (
            not managed_identity
            or not managed_identity.get("usable")
            or not digest_bound
        ):
            result["probe_error"] = "managed_digest_identity_failed"
            return result

    banner = first_nonempty_line(completed.stdout) or first_nonempty_line(
        completed.stderr
    )
    # Some Java runtimes write the version to stderr; that is valid only when
    # the process itself succeeds.
    if tool_id == "java":
        java_text = combined
        if not parse_java_version_banner(java_text):
            # Retry with -version which java accepts.
            completed = bounded_run(
                [executable, "-version"], timeout=timeout, env=probe_env
            )
            if completed is None:
                result["probe_error"] = "probe_timeout_or_spawn_failure"
                return result
            if completed.returncode != 0:
                result["probe_error"] = (
                    f"identity_probe_nonzero:{completed.returncode}"
                )
                return result
            java_text = "\n".join(
                part
                for part in (completed.stdout, completed.stderr)
                if part
            ).strip()
        quoted = parse_java_version_banner(java_text)
        if not quoted:
            result["probe_error"] = "java_version_banner_unreadable"
            return result
        major = java_major_version(java_text)
        result["version_string"] = (
            f'java version "{quoted}"'
            if major is None
            else f'java version "{quoted}" (major {major})'
        )
        result["java_major"] = major
        result["identity_probed"] = True
        result["installed"] = True
        return result

    if tool_id == "souffle":
        souffle_version = parse_souffle_version_banner(combined)
        if souffle_version is None:
            result["probe_error"] = "souffle_version_banner_unreadable"
            return result
        result["version_string"] = f"Souffle Version: {souffle_version}"
        result["identity_probed"] = True
        result["installed"] = True
        return result

    if not banner:
        result["probe_error"] = "empty_version_banner"
        return result

    result["version_string"] = banner
    result["identity_probed"] = True
    result["installed"] = True

    if tool_id == "lean":
        installed = list(result["installed_toolchains"])
        result["installed_toolchains"] = installed
        locked_toolchain = str(probe.get("locked_toolchain") or "").strip()
        match = re.search(r"version\s+(\d+\.\d+\.\d+)", banner, re.IGNORECASE)
        selected = (
            selected_lean_toolchain
            or (
                f"leanprover/lean4:v{match.group(1)}"
                if match
                else probe.get("locked_toolchain")
            )
        )
        result["selected_toolchain"] = selected
        direct_native = _direct_native_lean_identity(
            executable,
            banner=banner,
            locked_toolchain=locked_toolchain,
        )
        result["direct_native_lean_identity"] = direct_native
        if direct_native.get("valid") is True:
            result["lean_identity_mode"] = "direct_native_immutable_pin"
            result["shim_toolchain_mismatch"] = False
        else:
            result["lean_identity_mode"] = "elan_toolchain_inventory"
            result["shim_toolchain_mismatch"] = (
                detect_lean_shim_toolchain_mismatch(selected, installed)
            )

    return result


def _probe_in_process_module(
    tool_id: str,
    *,
    env: Mapping[str, str],
) -> tuple[bool, str | None]:
    """Bounded isolated import probe for in-process tools. Never installs."""

    if tool_id == "symbolicai":
        # The PyPI distribution is ``symbolicai`` but its import package is
        # ``symai``.  Importing symai is not a harmless availability probe: it
        # may initialize user configuration.  Bind both identities without
        # executing package code.
        try:
            distribution_version = importlib.metadata.version("symbolicai")
            module_spec = importlib.util.find_spec("symai")
        except (
            ImportError,
            ModuleNotFoundError,
            importlib.metadata.PackageNotFoundError,
            ValueError,
        ):
            return False, None
        if module_spec is None or not distribution_version:
            return False, None
        return (
            True,
            "python-distribution:"
            f"symbolicai=={distribution_version};module:symai",
        )

    module_map = {
        "runtime-mtl": "ipfs_datasets_py.logic.software_verification.monitoring.runtime_mtl",
        "datalog-authorization": "ipfs_datasets_py.logic.backends.datalog",
        "secpal-authorization": "ipfs_datasets_py.logic.backends.datalog",
        "ergoai": "ergoai",
        "zkp-circuit": None,  # declared gap — never pretend installed
    }
    module_name = module_map.get(tool_id)
    if not module_name:
        return False, None
    completed = bounded_run(
        (
            sys.executable,
            "-c",
            (
                "import importlib,sys; "
                "module=importlib.import_module(sys.argv[1]); "
                "print(getattr(module, '__version__', '') or "
                "getattr(module, '__file__', '') or 'imported')"
            ),
            module_name,
        ),
        timeout=PROBE_TIMEOUT_SECONDS,
        env=env,
    )
    if completed is None or completed.returncode != 0:
        return False, None
    identity = first_nonempty_line(completed.stdout)
    return bool(identity), f"python-module:{module_name}"


# ---------------------------------------------------------------------------
# Live checks
# ---------------------------------------------------------------------------


def _smt_argv(tool_id: str, executable: str) -> list[str]:
    if tool_id == "z3":
        return [executable, "-in"]
    if tool_id == "cvc5":
        return [executable, "--lang", "smt2", "-"]
    return [executable]


def _normalize_smt_verdict(stdout: str, stderr: str) -> str:
    text = f"{stdout}\n{stderr}".lower()
    # Prefer the last sat/unsat token for solvers that echo scripts.
    verdicts = re.findall(r"\b(unsat|sat|unknown)\b", text)
    if not verdicts:
        return "unknown"
    return verdicts[-1]


def run_smt_checks(
    tool_id: str,
    executable: str,
    *,
    env: Mapping[str, str],
) -> list[CheckResult]:
    """Positive / negative / mutation / replay checks for an SMT solver."""

    checks: list[CheckResult] = []

    def invoke(script: str) -> str:
        completed = bounded_run(
            _smt_argv(tool_id, executable),
            timeout=CHECK_TIMEOUT_SECONDS,
            env=env,
            stdin=script,
        )
        if completed is None:
            return "timeout"
        return _normalize_smt_verdict(completed.stdout, completed.stderr)

    positive = invoke(SMT_POSITIVE)
    checks.append(
        CheckResult(
            check_id=f"{tool_id}.positive",
            kind="positive",
            status="passed" if positive == "unsat" else "failed",
            expected="unsat",
            observed=positive,
            detail="unsatisfiable conjunction (proof-side)",
        )
    )

    negative = invoke(SMT_NEGATIVE)
    checks.append(
        CheckResult(
            check_id=f"{tool_id}.negative",
            kind="negative",
            status="passed" if negative == "sat" else "failed",
            expected="sat",
            observed=negative,
            detail="satisfiable constraint (model/counterexample-side)",
        )
    )

    mutated = invoke(SMT_MUTATED)
    # Mutation of the positive case must change the outcome (unsat → sat).
    mutation_ok = positive == "unsat" and mutated == "sat"
    checks.append(
        CheckResult(
            check_id=f"{tool_id}.mutation",
            kind="mutation",
            status="passed" if mutation_ok else "failed",
            expected="sat (changed from unsat)",
            observed=mutated,
            detail="dropped conjunct on positive formula; outcome must change",
        )
    )

    replay = invoke(SMT_POSITIVE)
    replay_ok = replay == positive == "unsat"
    checks.append(
        CheckResult(
            check_id=f"{tool_id}.replay",
            kind="replay",
            status="passed" if replay_ok else "failed",
            expected=positive,
            observed=replay,
            detail="re-run positive script; outcome must be stable",
        )
    )
    return checks


def run_identity_alive_checks(
    tool_id: str,
    identity: Mapping[str, Any],
) -> list[CheckResult]:
    """For non-SMT tools: identity stability as positive/replay; negative/mutation skip.

    Full semantic positive/negative fixtures for every external prover require
    tool-specific corpora owned by later lanes. Hermetic certification still
    requires an exact identity probe (positive) and a stable re-probe (replay).
    Negative and mutation remain explicit skips so they never silently pass.
    """

    version = str(identity.get("version_string") or "")
    probed = bool(identity.get("identity_probed"))
    checks = [
        CheckResult(
            check_id=f"{tool_id}.positive",
            kind="positive",
            status="passed" if probed and version else "failed",
            expected="non-empty exact identity",
            observed=version or "missing",
            detail="bounded offline identity probe",
        ),
        CheckResult(
            check_id=f"{tool_id}.negative",
            kind="negative",
            status="skipped",
            expected="tool-specific unsat/counterexample fixture",
            observed="skipped",
            detail=(
                "property-specific negative fixture not required for identity "
                "lane; absence does not invent success"
            ),
        ),
        CheckResult(
            check_id=f"{tool_id}.mutation",
            kind="mutation",
            status="skipped",
            expected="tool-specific mutation fixture",
            observed="skipped",
            detail="mutation suite deferred to property corpora; not synthetic pass",
        ),
        CheckResult(
            check_id=f"{tool_id}.replay",
            kind="replay",
            status="passed" if probed and version else "failed",
            expected=version,
            observed=version or "missing",
            detail="identity banner retained as replay witness for this probe",
        ),
    ]
    return checks


def run_in_process_checks(tool_id: str, identity: Mapping[str, Any]) -> list[CheckResult]:
    """Minimal positive/replay for in-process modules (runtime MTL, etc.)."""

    ok = bool(identity.get("identity_probed"))
    version = str(identity.get("version_string") or "")
    return [
        CheckResult(
            check_id=f"{tool_id}.positive",
            kind="positive",
            status="passed" if ok else "failed",
            expected="importable module identity",
            observed=version or "missing",
            detail="in-process module import probe (no install)",
        ),
        CheckResult(
            check_id=f"{tool_id}.negative",
            kind="negative",
            status="skipped",
            expected="module-specific negative fixture",
            observed="skipped",
            detail="no synthetic negative invented",
        ),
        CheckResult(
            check_id=f"{tool_id}.mutation",
            kind="mutation",
            status="skipped",
            expected="module-specific mutation fixture",
            observed="skipped",
            detail="no synthetic mutation invented",
        ),
        CheckResult(
            check_id=f"{tool_id}.replay",
            kind="replay",
            status="passed" if ok else "failed",
            expected=version,
            observed=version or "missing",
            detail="repeat import identity",
        ),
    ]


# ---------------------------------------------------------------------------
# Certification orchestration
# ---------------------------------------------------------------------------


def _pin_version(entry: Mapping[str, Any]) -> str:
    pins = entry.get("pins") or []
    if not pins:
        return ""
    return str(pins[0].get("version") or "")


def certify_tool(
    entry: Mapping[str, Any],
    *,
    lane_ids: Sequence[str],
    check_kind: str,
    env: Mapping[str, str],
) -> ToolCertification:
    tool_id = str(entry["tool_id"])
    families = list(entry.get("families") or [])
    locked_version = _pin_version(entry)
    availability = str(entry.get("availability") or "")

    cert = ToolCertification(
        tool_id=tool_id,
        lane_ids=list(lane_ids),
        families=families,
        availability_declared=availability,
        locked_version=locked_version,
    )

    # Explicit declared gaps are unavailable and never production-certified.
    if availability == "declared_gap":
        cert.unavailable = True
        cert.promotion_blocked = True
        cert.block_reasons.append("declared_gap")
        cert.evidence_class = "unavailable"
        cert.notes = "Declared install/capability gap; blocks only this tool's promotion."
        cert.checks = [
            CheckResult(
                check_id=f"{tool_id}.{kind}",
                kind=kind,
                status="unavailable",
                expected="n/a",
                observed="declared_gap",
                detail="gap blocks promotion only for this tool",
            )
            for kind in ("positive", "negative", "mutation", "replay")
        ]
        return cert

    identity = probe_tool_identity(entry, env=env)
    cert.path_present = bool(identity.get("path_present"))
    cert.executable_path = identity.get("executable_path")
    cert.executable_sha256 = file_digest(cert.executable_path)
    cert.executable_artifact_class = classify_executable_artifact(
        cert.executable_path
    )
    if cert.executable_path:
        cert.artifact_identities.append(
            {
                "kind": "executable",
                "path": cert.executable_path,
                "sha256": cert.executable_sha256,
                "artifact_class": cert.executable_artifact_class,
            }
        )
    if cert.executable_artifact_class == "launcher_script":
        cert.launcher_binding = bind_launcher_target_identity(entry, identity)
        cert.artifact_identities.extend(
            dict(item)
            for item in (cert.launcher_binding.get("artifacts") or ())
            if isinstance(item, Mapping)
        )
    cert.version_string = identity.get("version_string")
    cert.identity_probed = bool(identity.get("identity_probed"))
    cert.installed = bool(identity.get("installed"))
    cert.shim_toolchain_mismatch = bool(identity.get("shim_toolchain_mismatch"))

    if locked_version and cert.version_string:
        cert.locked_version_mismatch = detect_locked_version_mismatch(
            locked_version, cert.version_string
        )
    elif locked_version and not cert.version_string:
        cert.locked_version_mismatch = True

    # PATH presence is not usability / not installed without identity.
    if cert.path_present and not cert.identity_probed:
        cert.block_reasons.append("path_presence_without_identity_probe")
        cert.unavailable = True
        cert.promotion_blocked = True
        cert.evidence_class = "path_shim"
        cert.notes = (
            "Executable on PATH but identity probe failed; PATH shims are not usability."
        )
        cert.checks = [
            CheckResult(
                check_id=f"{tool_id}.{kind}",
                kind=kind,
                status="unavailable",
                expected="exact identity",
                observed="path_only",
                detail="PATH presence is not usability",
            )
            for kind in ("positive", "negative", "mutation", "replay")
        ]
        return cert

    if not cert.installed:
        cert.unavailable = True
        cert.promotion_blocked = True
        cert.block_reasons.append("unavailable")
        cert.evidence_class = "unavailable"
        cert.notes = (
            f"Tool not installed or not probeable ({identity.get('probe_error')}); "
            "blocks only this tool's promotion."
        )
        cert.checks = [
            CheckResult(
                check_id=f"{tool_id}.{kind}",
                kind=kind,
                status="unavailable",
                expected="installed+probed",
                observed="unavailable",
                detail=str(identity.get("probe_error") or "unavailable"),
            )
            for kind in ("positive", "negative", "mutation", "replay")
        ]
        return cert

    if cert.shim_toolchain_mismatch:
        cert.block_reasons.append("shim_toolchain_mismatch")
        cert.usable = False
        cert.promotion_blocked = True
        cert.evidence_class = "shim_mismatch"
        cert.notes = (
            "Selected toolchain not offline-installed; fail closed without download."
        )
    elif cert.locked_version_mismatch:
        cert.block_reasons.append("locked_version_mismatch")
        cert.usable = False
        cert.promotion_blocked = True
        cert.evidence_class = "version_mismatch"
        cert.notes = (
            "Observed version does not match offline lock pin; "
            "production certification blocked without upgrade/download."
        )
    else:
        cert.usable = True
        cert.evidence_class = "live"

    # Live checks — only meaningful when identity is present.
    if check_kind == "smtlib" and cert.executable_path:
        cert.checks = run_smt_checks(tool_id, cert.executable_path, env=env)
    elif identity.get("in_process") or check_kind == "identity_or_in_process":
        if cert.executable_path and not identity.get("in_process"):
            cert.checks = run_identity_alive_checks(tool_id, identity)
        else:
            cert.checks = run_in_process_checks(tool_id, identity)
    else:
        cert.checks = run_identity_alive_checks(tool_id, identity)

    required_kinds = {"positive", "replay"}
    required_passed = all(
        check.status == "passed"
        for check in cert.checks
        if check.kind in required_kinds
    )
    # Production promotion requires semantic positive, negative, mutation, and
    # replay evidence. Identity-only and import-only checks remain useful
    # availability evidence, but explicit skips must never become proof
    # certification merely because the identity is stable.
    all_live_passed = required_passed and all(
        check.status == "passed" for check in cert.checks
    )
    executable_identity_complete = bool(
        cert.executable_artifact_class == "native_or_managed_binary"
        or (
            cert.executable_artifact_class == "launcher_script"
            and cert.launcher_binding.get("valid") is True
        )
    )
    if (
        cert.executable_artifact_class == "launcher_script"
        and not executable_identity_complete
    ):
        cert.block_reasons.append("launcher_target_artifact_unbound")

    if not all_live_passed:
        cert.block_reasons.append("live_checks_incomplete_or_failed")
        cert.promotion_blocked = True
        if cert.usable:
            cert.notes = (cert.notes + " " if cert.notes else "") + (
                "Live checks incomplete or failed; not production-certified."
            ).strip()

    cert.production_certified = bool(
        cert.usable
        and all_live_passed
        and not cert.locked_version_mismatch
        and not cert.shim_toolchain_mismatch
        and not cert.unavailable
        and executable_identity_complete
    )
    if cert.production_certified:
        cert.promotion_blocked = False
        cert.block_reasons = []
        cert.evidence_class = "production_certified"
        cert.notes = (
            "Hermetic offline certification passed with exact identity and live checks."
        )
    elif cert.usable and not cert.production_certified:
        cert.promotion_blocked = True

    return cert


def quarantine_smt_disagreement(
    tool_certs: Mapping[str, ToolCertification],
    *,
    lane_id: str = "smt",
    property_class: str = "smt_software_verification",
) -> DisagreementQuarantine | None:
    """Compare SMT positive outcomes; quarantine hard disagreement."""

    outcomes: dict[str, str] = {}
    for tool_id in ("z3", "cvc5"):
        cert = tool_certs.get(tool_id)
        if cert is None or cert.unavailable or not cert.usable:
            continue
        for check in cert.checks:
            if check.kind == "positive" and check.status in {"passed", "failed"}:
                outcomes[tool_id] = check.observed
                break

    if len(outcomes) < 2:
        return None

    values = set(outcomes.values())
    # unknown/timeout is not hard disagreement — only sat vs unsat is.
    hard = {value for value in values if value in {"sat", "unsat"}}
    if len(hard) <= 1:
        return None

    tool_ids = sorted(outcomes)
    payload = {
        "lane_id": lane_id,
        "property_class": property_class,
        "outcomes": outcomes,
    }
    quarantine_id = f"eq-quarantine:{content_digest(payload)[:16]}"
    return DisagreementQuarantine(
        quarantine_id=quarantine_id,
        lane_id=lane_id,
        property_class=property_class,
        tool_ids=tool_ids,
        outcomes=outcomes,
        promotion_blocked_tool_ids=tool_ids,
    )


def certify_property_lanes(
    tool_certs: Mapping[str, ToolCertification],
    disagreements: Sequence[DisagreementQuarantine],
    *,
    property_lanes: Sequence[Mapping[str, Any]] = PROPERTY_LANES,
) -> list[LaneCertification]:
    """Project lane readiness from certified proof authorities, not runtimes."""

    lanes: list[LaneCertification] = []
    for lane in property_lanes:
        lane_id = str(lane["lane_id"])
        tool_ids = list(lane["tool_ids"])
        authority_tool_ids = list(lane.get("authority_tool_ids") or ())
        if not authority_tool_ids:
            raise ValueError(f"lane {lane_id!r} must declare authority_tool_ids")
        if not set(authority_tool_ids).issubset(tool_ids):
            raise ValueError(
                f"lane {lane_id!r} authority_tool_ids must be a subset of tool_ids"
            )
        certified = [
            tid
            for tid in tool_ids
            if tool_certs.get(tid) and tool_certs[tid].production_certified
        ]
        certified_authorities = [
            tid for tid in authority_tool_ids if tid in certified
        ]
        unavailable = [
            tid
            for tid in tool_ids
            if tool_certs.get(tid) and tool_certs[tid].unavailable
        ]
        blocked = [
            tid
            for tid in tool_ids
            if tool_certs.get(tid) and tool_certs[tid].promotion_blocked
        ]
        q_ids = [
            item.quarantine_id
            for item in disagreements
            if item.lane_id == lane_id
        ]
        lanes.append(
            LaneCertification(
                lane_id=lane_id,
                property_class=str(lane["property_class"]),
                description=str(lane["description"]),
                tool_ids=tool_ids,
                authority_tool_ids=authority_tool_ids,
                certified_tool_ids=certified,
                certified_authority_tool_ids=certified_authorities,
                unavailable_tool_ids=unavailable,
                blocked_tool_ids=blocked,
                disagreement_quarantine_ids=q_ids,
                promotion_ready=bool(certified_authorities) and not q_ids,
                notes=(
                    "Absent tools block only their own promotion."
                    if unavailable
                    else ""
                ),
            )
        )
    return lanes


# ---------------------------------------------------------------------------
# Role-aware semantic elevation (FVT-G200)
# ---------------------------------------------------------------------------


# Semantic certifiers owned by later installation/certification tasks. Each
# entry maps offline lock tool ids to a focused certifier module. Elevation
# never installs, downloads, or conceals unavailable tools.
#
# ``property_lane_id`` groups specialized certifier runs into composite property
# lanes for lossless aggregation (FVT-G203). ``certifier_family`` is the
# acceptance-facing family name used by specialized receipt aggregation.
SEMANTIC_CERTIFIER_SPECS: Final[tuple[dict[str, Any], ...]] = (
    {
        "lane_id": "kernel",
        "property_lane_id": "kernel",
        "certifier_family": "kernel",
        "module_relative": Path("tools/logic/certification/lean.py"),
        "callable_name": "build_certification_receipt",
        "tool_ids": ("lean",),
        "certified_key": "production_certified",
        "interface": "LeanSemanticCertification@1",
        "selector": "root",
        "evidence_class": "usable_pending_kernel_live_fanin",
        "production_elevation_allowed": False,
        "identity_from_receipt": True,
    },
    {
        "lane_id": "kernel_rocq",
        "property_lane_id": "kernel",
        "certifier_family": "kernel",
        "module_relative": Path("tools/logic/certification/rocq.py"),
        "callable_name": "build_certification_receipt",
        "tool_ids": ("coq",),
        "certified_key": "production_certified",
        "interface": "RocqToolchainCertification@1",
        "selector": "root",
        "evidence_class": "external_prover_installation_pending",
        "production_elevation_allowed": False,
        "usable_elevation_allowed": False,
        "identity_from_receipt": True,
    },
    {
        "lane_id": "kernel_isabelle",
        "property_lane_id": "kernel",
        "certifier_family": "kernel",
        "module_relative": Path("tools/logic/certification/isabelle.py"),
        "callable_name": "build_certification_receipt",
        "tool_ids": ("isabelle",),
        "certified_key": "production_certified",
        "interface": "IsabelleToolchainCertification@1",
        "selector": "root",
        "evidence_class": "external_prover_installation_pending",
        "production_elevation_allowed": False,
        "usable_elevation_allowed": False,
        "identity_from_receipt": True,
    },
    {
        "lane_id": "runtime_mtl",
        "property_lane_id": "runtime_mtl",
        "certifier_family": "runtime_mtl_in_process",
        "module_relative": Path("tools/logic/certification/runtime_mtl.py"),
        "callable_name": "certify_runtime_mtl_semantics",
        "tool_ids": ("runtime-mtl",),
        "certified_key": "certified",
        "interface": "RuntimeMTLSemanticCertification@1",
        "selector": "root",
        "evidence_class": "usable_pending_external_runtime_mtl",
        "production_elevation_allowed": False,
    },
    {
        "lane_id": "datalog_secpal",
        "property_lane_id": "datalog_secpal",
        "certifier_family": "authorization_in_process",
        "module_relative": Path("tools/logic/certification/authorization.py"),
        "callable_name": "certify_authorization_semantics",
        "tool_ids": ("datalog-authorization", "secpal-authorization"),
        "certified_key": "certified",
        "interface": "AuthorizationSemanticCertification@1",
        "selector": "engine",
        "evidence_class": "usable_pending_authorization_vendor_fanin",
        "production_elevation_allowed": False,
    },
    {
        "lane_id": "state_model",
        "property_lane_id": "tla",
        "certifier_family": "state",
        "module_relative": Path("tools/logic/certification/state_model.py"),
        "callable_name": "build_certification_receipt",
        "tool_ids": ("apalache", "tlc"),
        "certified_key": "production_certified",
        "interface": "StateModelToolchainCertification@1",
        "selector": "root",
        "evidence_class": "identity_plus_fixture_parser",
        "production_elevation_allowed": False,
    },
    {
        "lane_id": "protocol_tamarin",
        "property_lane_id": "protocol",
        "certifier_family": "protocol",
        "module_relative": Path("tools/logic/certification/tamarin.py"),
        "callable_name": "build_certification_receipt",
        "tool_ids": ("tamarin",),
        "certified_key": "production_certified",
        "interface": "TamarinToolchainCertification@1",
        "selector": "root",
        "evidence_class": "identity_plus_fixture_parser",
        "production_elevation_allowed": False,
    },
    {
        "lane_id": "protocol_proverif",
        "property_lane_id": "protocol",
        "certifier_family": "protocol",
        "module_relative": Path("tools/logic/certification/proverif.py"),
        "callable_name": "build_certification_receipt",
        "tool_ids": ("proverif",),
        "certified_key": "production_certified",
        "interface": "ProVerifToolchainCertification@1",
        "selector": "root",
        "evidence_class": "identity_plus_fixture_parser",
        "production_elevation_allowed": False,
    },
    {
        "lane_id": "atp",
        "property_lane_id": "atp",
        "certifier_family": "atp",
        "module_relative": Path("tools/logic/certification/atp.py"),
        "callable_name": "build_certification_receipt",
        "tool_ids": ("vampire", "eprover"),
        "certified_key": "production_certified",
        "interface": "ATPToolchainCertification@1",
        "selector": "root",
        "evidence_class": "identity_plus_fixture_parser",
        "production_elevation_allowed": False,
    },
    {
        "lane_id": "hyperltl",
        "property_lane_id": "hyperltl",
        "certifier_family": "hyperproperty",
        "module_relative": Path("tools/logic/certification/hyperproperty.py"),
        "callable_name": "certify_hyperproperty_toolchains",
        "tool_ids": ("hyperltl", "autohyper", "mchyper"),
        "certified_key": "certified",
        "interface": "HyperpropertyToolchainCertification@1",
        "selector": "engine",
        "evidence_class": "hermetic_adapter_shim",
        "production_elevation_allowed": False,
        "identity_from_receipt": True,
    },
    {
        "lane_id": "runtime_mtl_external",
        "property_lane_id": "runtime_mtl",
        "certifier_family": "runtime_mtl_external",
        "module_relative": Path("tools/logic/certification/runtime_mtl_external.py"),
        "callable_name": "certify_external_runtime_mtl",
        "tool_ids": ("runtime-mtl-external",),
        "certified_key": "certified",
        "interface": "ExternalRuntimeMTLCertification@1",
        "selector": "engine",
        "evidence_class": "hermetic_adapter_shim",
        "production_elevation_allowed": False,
    },
    {
        "lane_id": "authorization_external",
        "property_lane_id": "datalog_secpal",
        "certifier_family": "authorization_external",
        "module_relative": Path("tools/logic/certification/authorization_external.py"),
        "callable_name": "certify_external_authorization_shadows",
        "tool_ids": ("souffle", "secpal"),
        "certified_key": "certified",
        "interface": "ExternalAuthorizationShadowCertification@1",
        "selector": "engine",
        "evidence_class": "hermetic_shadow_shim",
        "production_elevation_allowed": False,
    },
    {
        "lane_id": "attestation",
        "property_lane_id": "attestation",
        "certifier_family": "zkp",
        "module_relative": Path("tools/logic/certification/zkp.py"),
        "callable_name": "build_certification_receipt",
        "tool_ids": ("zkp-circuit",),
        "certified_key": "production_certified",
        "interface": "ZKPDeploymentCertification@1",
        "selector": "root",
        "evidence_class": "public_artifact_attestation",
        "production_elevation_allowed": True,
        "identity_from_receipt": True,
    },
    {
        "lane_id": "advisors",
        "property_lane_id": "hammer",
        "certifier_family": "advisor",
        "module_relative": Path("tools/logic/certification/advisors.py"),
        "callable_name": "build_certification_receipt",
        "tool_ids": ("symbolicai", "ergoai"),
        "certified_key": "production_certified",
        "interface": "AdvisorRoleCertification@1",
        "selector": "root",
        "evidence_class": "proposal_only_semantics",
        "production_elevation_allowed": False,
    },
)

# A production elevation for the in-process Runtime MTL and authorization
# authorities is a two-sided join.  Their focused reference receipts remain
# the authority-bearing evidence; these independently installed vendor tools
# are differential witnesses only.  The live vendor certifier must re-run
# offline against the one explicit sealed root and reproduce the exact checked
# install receipt before it can unlock one of the listed in-process targets.
CHECKED_VENDOR_FANIN_SPECS: Final[Mapping[str, Mapping[str, Any]]] = {
    "runtime_mtl": {
        "module_relative": Path(
            "tools/logic/certification/runtime_mtl_external.py"
        ),
        "callable_name": "certify_external_runtime_mtl_vendor",
        "checked_receipt_relative": RUNTIME_MTL_VENDOR_RECEIPT_RELATIVE,
        "checked_receipt_schema": (
            "formal-verification-runtime-mtl-external-install-receipt/v1"
        ),
        "live_schema": "external-runtime-mtl-vendor-certification/v1",
        "interface": "ExternalRuntimeMTLVendorCertification@1",
        "goal_id": "FVT-G210",
        "task_id": "FVT-056",
        "repair_task_id": "FVT-072",
        "vendor_section": "runtime_mtl_external",
        "vendor_tool_id": "runtime-mtl-external",
        "summary_certified_key": "vendor_certified",
        "expected_vendor_checks": 37,
        "expected_reference_checks": {"runtime-mtl": "closed_manifest"},
        "install_root_relative": Path("."),
        "dependency_prefix_relative": None,
        "outer_digest_uses_public_projection": False,
        "evidence_class": "reference_plus_checked_runtime_mtl_vendor",
        "managed_readiness_tool_id": "runtime-mtl-external",
        "managed_readiness_role": "differential_witness",
        "managed_readiness_scope": "differential_witness_only",
        "declared_authority_role": "authority",
        "declared_authority_ceiling": "finite_trace",
        "declared_role_can_satisfy_certified_authority": True,
        "managed_readiness_evidence_class": (
            "checked_runtime_mtl_vendor_differential_witness"
        ),
    },
    "datalog_secpal": {
        "module_relative": Path(
            "tools/logic/certification/authorization_external.py"
        ),
        "callable_name": "certify_external_authorization_vendor",
        "checked_receipt_relative": AUTHORIZATION_VENDOR_RECEIPT_RELATIVE,
        "checked_receipt_schema": (
            "formal-verification-authorization-vendor-install-receipt/v1"
        ),
        "live_schema": "external-authorization-vendor-certification/v1",
        "interface": "ExternalAuthorizationVendorCertification@1",
        "goal_id": "FVT-G209",
        "task_id": "FVT-055",
        "repair_task_id": "FVT-073",
        "vendor_section": "souffle",
        "vendor_tool_id": "souffle",
        "summary_certified_key": "souffle_certified",
        "expected_vendor_checks": 32,
        "expected_reference_checks": {
            "datalog-authorization": 24,
            "secpal-authorization": 24,
        },
        "install_root_relative": Path("souffle-vendor"),
        "dependency_prefix_relative": Path(
            "build-dependencies/souffle/ubuntu-noble-arm64/root"
        ),
        "outer_digest_uses_public_projection": True,
        "evidence_class": "reference_plus_checked_souffle_vendor",
        "managed_readiness_tool_id": "souffle",
        "managed_readiness_role": "shadow_checker",
        "managed_readiness_scope": "shadow_checker_only",
        "declared_authority_role": "shadow",
        "declared_authority_ceiling": "none",
        "declared_role_can_satisfy_certified_authority": False,
        "managed_readiness_evidence_class": (
            "checked_native_souffle_vendor_shadow"
        ),
    },
}

# Hyperproperty vendor certification is intentionally separate from the
# Runtime/Auth differential join above.  These three reviewed engines are the
# bounded authority-bearing tools themselves, so no independent reference is
# claimed.  Their complete 3 x 22 live vendor corpus and exact checked receipt
# form a multi-engine adapter to the existing hyperproperty lane.
CHECKED_HYPER_VENDOR_FANIN_SPEC: Final[Mapping[str, Any]] = {
    "lane_id": "hyperltl",
    "module_relative": Path("tools/logic/certification/hyperproperty.py"),
    "callable_name": "certify_hyperproperty_vendor_toolchains",
    "checked_receipt_relative": HYPERPROPERTY_VENDOR_RECEIPT_RELATIVE,
    "checked_receipt_schema": (
        "formal-verification-hyperproperty-vendor-install-receipt/v1"
    ),
    "live_schema": "hyperproperty-vendor-toolchain-certification/v1",
    "interface": "HyperpropertyVendorToolchainCertification@1",
    "goal_id": "FVT-G208",
    "task_id": "FVT-061",
    "repair_task_id": "FVT-077",
    "targets": ("hyperltl", "autohyper", "mchyper"),
    "checks_per_target": 22,
    "expected_vendor_checks": 66,
    "evidence_class": CHECKED_HYPER_VENDOR_FANIN_EVIDENCE_CLASS,
    "vendor_authority": "bounded_hyperproperty_authority",
}

# Acceptance-required specialized certifier families (FVT-G203).
REQUIRED_SPECIALIZED_CERTIFIER_FAMILIES: Final[frozenset[str]] = frozenset(
    {
        "state",
        "protocol",
        "kernel",
        "atp",
        "hyperproperty",
        "advisor",
        "authorization_in_process",
        "authorization_external",
        "runtime_mtl_in_process",
        "runtime_mtl_external",
        "zkp",
    }
)

# Composite property lanes that must retain multi-tool specialized evidence.
COMPOSITE_PROPERTY_LANE_REQUIRED_TOOLS: Final[Mapping[str, tuple[str, ...]]] = {
    "kernel": ("lean", "coq", "isabelle"),
    "protocol": ("tamarin", "proverif"),
}


def _load_module_from_path(module_path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _normalize_semantic_checks(
    tool_id: str,
    raw_checks: Sequence[Mapping[str, Any]] | None,
) -> list[CheckResult]:
    """Retain the complete semantic check set without collapsing by kind.

    FVT-G203: every check participates in the retained evidence. A second
    failed check of an already-present kind is kept (never discarded) so
    promotion can fail closed.
    """

    checks: list[CheckResult] = []
    kinds_present: set[str] = set()
    for raw in raw_checks or ():
        if not isinstance(raw, Mapping):
            continue
        kind = str(raw.get("kind") or "").strip().lower()
        status = str(raw.get("status") or "failed").lower()
        if status not in {"passed", "failed", "skipped", "unavailable"}:
            status = "failed"
        kinds_present.add(kind)
        checks.append(
            CheckResult(
                check_id=str(raw.get("check_id") or f"{tool_id}.{kind}"),
                kind=kind or "unclassified",
                status=status,
                expected=str(raw.get("expected") or "semantic_pass"),
                observed=str(raw.get("observed") or status),
                detail=str(raw.get("detail") or "role-aware semantic elevation"),
                evidence=dict(raw),
            )
        )

    for kind in ("positive", "negative", "mutation", "replay"):
        if kind in kinds_present:
            continue
        checks.append(
            CheckResult(
                check_id=f"{tool_id}.{kind}",
                kind=kind,
                status="failed",
                expected="semantic suite coverage",
                observed="missing_kind",
                detail=(
                    "Role-aware elevation requires positive/negative/mutation/"
                    "replay evidence; missing kinds fail closed."
                ),
                evidence={"missing_required_kind": kind},
            )
        )
    return checks


def second_failed_check_blocks_promotion(
    checks: Sequence[CheckResult | Mapping[str, Any]],
) -> tuple[bool, list[str]]:
    """Return whether a second failed check of an already-present kind blocks.

    Checks are never collapsed by kind. When a kind already has at least one
    retained check and a later check of that same kind fails, promotion is
    blocked with an explicit reason code.
    """

    reasons: list[str] = []
    seen_kinds: set[str] = set()
    for raw in checks:
        if isinstance(raw, CheckResult):
            kind = raw.kind
            status = raw.status
        elif isinstance(raw, Mapping):
            kind = str(raw.get("kind") or "unclassified")
            status = str(raw.get("status") or "failed").lower()
        else:
            continue
        kind = kind.strip().lower() or "unclassified"
        if status == "failed" and kind in seen_kinds:
            reasons.append(f"second_failed_check_of_already_present_kind:{kind}")
        seen_kinds.add(kind)
        if status == "failed" and f"failed_check_kind:{kind}" not in reasons:
            # Any failure blocks; second-failed is an additional explicit signal.
            if f"second_failed_check_of_already_present_kind:{kind}" not in reasons:
                reasons.append(f"failed_check_kind:{kind}")
    # Prefer the second-failed reason as the primary block signal when present.
    ordered = [
        reason
        for reason in reasons
        if reason.startswith("second_failed_check_of_already_present_kind:")
    ] + [
        reason
        for reason in reasons
        if not reason.startswith("second_failed_check_of_already_present_kind:")
    ]
    return bool(ordered), ordered


def _spec_by_lane_id(lane_id: str) -> dict[str, Any] | None:
    for spec in SEMANTIC_CERTIFIER_SPECS:
        if str(spec["lane_id"]) == lane_id:
            return dict(spec)
    return None


def _authority_ceiling_for_tool(
    tool_id: str,
    authority_roles: Mapping[str, Any] | None,
) -> str | None:
    tools = (authority_roles or {}).get("tools") or {}
    if not isinstance(tools, Mapping):
        return None
    meta = tools.get(tool_id)
    if not isinstance(meta, Mapping):
        return None
    ceiling = meta.get("authority_ceiling")
    return str(ceiling) if ceiling is not None else None


def _extract_specialized_tool_record(
    *,
    tool_id: str,
    semantic_result: Mapping[str, Any],
    spec: Mapping[str, Any],
    authority_roles: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project one tool's specialized evidence without discarding identity."""

    per_tool = semantic_result.get("per_tool")
    per_tool = per_tool if isinstance(per_tool, Mapping) else {}
    tool_payload = per_tool.get(tool_id)
    tool_payload = tool_payload if isinstance(tool_payload, Mapping) else {}

    receipt = semantic_result.get("receipt")
    receipt = receipt if isinstance(receipt, Mapping) else {}
    raw_receipt_digest = (
        str(semantic_result.get("digest_sha256") or "")
        or str(receipt.get("receipt_digest_sha256") or "")
        or str(receipt.get("certificate_digest_sha256") or "")
        or str(receipt.get("digest_sha256") or "")
    )

    checks = tool_payload.get("checks")
    if not isinstance(checks, list):
        checks = []
    identity = tool_payload.get("identity")
    identity = identity if isinstance(identity, Mapping) else {}
    artifacts = identity.get("artifacts")
    artifacts = (
        [item for item in artifacts if isinstance(item, Mapping)]
        if isinstance(artifacts, Sequence)
        and not isinstance(artifacts, (str, bytes, bytearray))
        else []
    )

    bindings: list[dict[str, Any]] = []
    for field_name in (
        "lock_path",
        "lock_digest",
        "goal_id",
        "task_id",
        "interface",
        "schema_version",
    ):
        if field_name in receipt:
            bindings.append(
                {
                    "kind": "receipt_binding",
                    "field": field_name,
                    "value": receipt.get(field_name),
                }
            )
    for artifact in artifacts:
        bindings.append(
            {
                "kind": "artifact_binding",
                "artifact": dict(artifact),
            }
        )

    cases = [
        {
            "check_id": (
                check.get("check_id")
                if isinstance(check, Mapping)
                else getattr(check, "check_id", None)
            ),
            "kind": (
                check.get("kind")
                if isinstance(check, Mapping)
                else getattr(check, "kind", None)
            ),
            "status": (
                check.get("status")
                if isinstance(check, Mapping)
                else getattr(check, "status", None)
            ),
        }
        for check in checks
    ]

    dependencies = []
    for key in ("dependencies", "dependency_digests", "pins"):
        value = receipt.get(key)
        if value:
            dependencies.append({"field": key, "value": value})

    sources = [
        {
            "kind": "semantic_certifier_module",
            "path": str(spec.get("module") or semantic_result.get("module") or ""),
            "sha256": semantic_result.get("certifier_module_sha256"),
        },
        {
            "kind": "semantic_lane_result",
            "lane_id": semantic_result.get("lane_id"),
            "digest_sha256": semantic_result.get("digest_sha256"),
        },
    ]

    executable = {
        "path": identity.get("executable_path"),
        "version_string": identity.get("version_string"),
        "identity_probed": identity.get("identity_probed"),
    }

    check_models = [
        CheckResult(
            check_id=str(
                (check.get("check_id") if isinstance(check, Mapping) else "")
                or f"{tool_id}.check"
            ),
            kind=str(
                (check.get("kind") if isinstance(check, Mapping) else "")
                or "unclassified"
            ),
            status=str(
                (check.get("status") if isinstance(check, Mapping) else "failed")
                or "failed"
            ),
            expected=str(
                (check.get("expected") if isinstance(check, Mapping) else "")
                or ""
            ),
            observed=str(
                (check.get("observed") if isinstance(check, Mapping) else "")
                or ""
            ),
            detail=str(
                (check.get("detail") if isinstance(check, Mapping) else "")
                or ""
            ),
            evidence=dict(check) if isinstance(check, Mapping) else {},
        )
        for check in checks
        if isinstance(check, Mapping)
    ]
    blocks, block_reasons = second_failed_check_blocks_promotion(check_models)
    certified = bool(tool_payload.get("certified")) and not blocks
    if blocks:
        block_reasons = list(block_reasons) + list(
            tool_payload.get("block_reasons") or []
        )
    else:
        block_reasons = list(tool_payload.get("block_reasons") or [])

    record = {
        "handler_key": f"{spec.get('property_lane_id') or semantic_result.get('lane_id')}::{tool_id}",
        "semantic_lane_id": semantic_result.get("lane_id"),
        "property_lane_id": spec.get("property_lane_id")
        or semantic_result.get("lane_id"),
        "certifier_family": spec.get("certifier_family"),
        "tool_id": tool_id,
        "certified": certified,
        "promotion_blocked": blocks or not certified,
        "block_reasons": block_reasons,
        "checks": checks,
        "cases": cases,
        "bindings": bindings,
        "executable": executable,
        "artifacts": artifacts,
        "dependencies": dependencies,
        "sources": sources,
        "authority_ceiling": _authority_ceiling_for_tool(tool_id, authority_roles),
        "raw_receipt_digest": raw_receipt_digest,
        "check_set_digest_sha256": tool_payload.get("check_set_digest_sha256")
        or content_digest(checks),
        "identity": dict(identity),
        "artifact_validation": tool_payload.get("artifact_validation"),
        "interface": semantic_result.get("interface") or spec.get("interface"),
        "module": semantic_result.get("module") or spec.get("module"),
    }
    record["tool_evidence_digest_sha256"] = content_digest(record)
    return record


def aggregate_specialized_receipts(
    semantic_results: Sequence[Mapping[str, Any]],
    *,
    authority_roles: Mapping[str, Any] | None = None,
    property_lanes: Sequence[Mapping[str, Any]] = PROPERTY_LANES,
) -> dict[str, Any]:
    """Losslessly aggregate specialized semantic receipts by composite lane.

    ``FormalVerificationSpecializedReceiptAggregation@1`` / FVT-G203.

    * Handlers / evidence are keyed by ``(property_lane_id, tool_id)``.
    * Kernel retains Lean, Rocq (coq), and Isabelle; protocol retains Tamarin
      and ProVerif.
    * Every check, case, binding, executable, artifact, dependency, source,
      authority ceiling, and raw receipt digest participates in the digest.
    * A second failed check of an already-present kind blocks promotion.
    * Sibling tools never overwrite each other; checks are never collapsed by
      kind; installers are never run.
    """

    specs_by_lane = {
        str(spec["lane_id"]): dict(spec) for spec in SEMANTIC_CERTIFIER_SPECS
    }
    families_present: set[str] = set()
    specialized_by_handler: dict[str, dict[str, Any]] = {}
    composite_lanes: dict[str, dict[str, Any]] = {}

    for result in semantic_results:
        lane_id = str(result.get("lane_id") or "")
        spec = specs_by_lane.get(lane_id) or _spec_by_lane_id(lane_id) or {
            "lane_id": lane_id,
            "property_lane_id": lane_id,
            "certifier_family": lane_id,
            "tool_ids": tuple(result.get("tool_ids") or ()),
            "module": result.get("module"),
            "interface": result.get("interface"),
        }
        property_lane_id = str(
            spec.get("property_lane_id") or result.get("lane_id") or lane_id
        )
        family = str(spec.get("certifier_family") or property_lane_id)
        families_present.add(family)

        lane_entry = composite_lanes.setdefault(
            property_lane_id,
            {
                "property_lane_id": property_lane_id,
                "semantic_lane_ids": [],
                "certifier_families": [],
                "tool_ids": [],
                "per_tool": {},
                "raw_receipt_digests": [],
                "specialized_handler_keys": [],
            },
        )
        if lane_id and lane_id not in lane_entry["semantic_lane_ids"]:
            lane_entry["semantic_lane_ids"].append(lane_id)
        if family not in lane_entry["certifier_families"]:
            lane_entry["certifier_families"].append(family)

        raw_digest = str(result.get("digest_sha256") or "")
        if raw_digest and raw_digest not in lane_entry["raw_receipt_digests"]:
            lane_entry["raw_receipt_digests"].append(raw_digest)

        tool_ids = [
            str(tool_id)
            for tool_id in (result.get("tool_ids") or spec.get("tool_ids") or ())
        ]
        for tool_id in tool_ids:
            record = _extract_specialized_tool_record(
                tool_id=tool_id,
                semantic_result=result,
                spec=spec,
                authority_roles=authority_roles,
            )
            handler_key = str(record["handler_key"])
            # Never let a later sibling overwrite an earlier specialized receipt.
            if handler_key in specialized_by_handler:
                continue
            specialized_by_handler[handler_key] = record
            if tool_id not in lane_entry["tool_ids"]:
                lane_entry["tool_ids"].append(tool_id)
            if tool_id not in lane_entry["per_tool"]:
                lane_entry["per_tool"][tool_id] = record
            if handler_key not in lane_entry["specialized_handler_keys"]:
                lane_entry["specialized_handler_keys"].append(handler_key)

    # Ensure required multi-tool composite lanes expose every expected tool slot
    # even when a specialized certifier is absent (explicit gap, not overwrite).
    for property_lane_id, required_tools in COMPOSITE_PROPERTY_LANE_REQUIRED_TOOLS.items():
        lane_entry = composite_lanes.setdefault(
            property_lane_id,
            {
                "property_lane_id": property_lane_id,
                "semantic_lane_ids": [],
                "certifier_families": [],
                "tool_ids": [],
                "per_tool": {},
                "raw_receipt_digests": [],
                "specialized_handler_keys": [],
            },
        )
        for tool_id in required_tools:
            if tool_id in lane_entry["per_tool"]:
                continue
            handler_key = f"{property_lane_id}::{tool_id}"
            gap = {
                "handler_key": handler_key,
                "semantic_lane_id": None,
                "property_lane_id": property_lane_id,
                "certifier_family": (
                    "kernel" if property_lane_id == "kernel" else "protocol"
                ),
                "tool_id": tool_id,
                "certified": False,
                "promotion_blocked": True,
                "block_reasons": ["specialized_receipt_missing"],
                "checks": [],
                "cases": [],
                "bindings": [],
                "executable": {},
                "artifacts": [],
                "dependencies": [],
                "sources": [],
                "authority_ceiling": _authority_ceiling_for_tool(
                    tool_id, authority_roles
                ),
                "raw_receipt_digest": "",
                "check_set_digest_sha256": content_digest([]),
                "identity": {},
                "artifact_validation": {"valid": False, "failures": ["missing"]},
                "interface": None,
                "module": None,
            }
            gap["tool_evidence_digest_sha256"] = content_digest(gap)
            lane_entry["per_tool"][tool_id] = gap
            if tool_id not in lane_entry["tool_ids"]:
                lane_entry["tool_ids"].append(tool_id)
            if handler_key not in lane_entry["specialized_handler_keys"]:
                lane_entry["specialized_handler_keys"].append(handler_key)
            specialized_by_handler.setdefault(handler_key, gap)

    for lane_entry in composite_lanes.values():
        per_tool = lane_entry["per_tool"]
        blocked_tools = sorted(
            tool_id
            for tool_id, record in per_tool.items()
            if record.get("promotion_blocked")
        )
        certified_tools = sorted(
            tool_id
            for tool_id, record in per_tool.items()
            if record.get("certified")
        )
        lane_entry["blocked_tool_ids"] = blocked_tools
        lane_entry["certified_tool_ids"] = certified_tools
        lane_entry["promotion_ready"] = bool(certified_tools) and not blocked_tools
        lane_entry["lane_digest_sha256"] = content_digest(
            {
                "property_lane_id": lane_entry["property_lane_id"],
                "per_tool": per_tool,
                "raw_receipt_digests": lane_entry["raw_receipt_digests"],
                "specialized_handler_keys": lane_entry["specialized_handler_keys"],
            }
        )

    property_lane_index = {
        str(lane["lane_id"]): dict(lane) for lane in property_lanes
    }
    for lane_id, lane_meta in property_lane_index.items():
        if lane_id not in composite_lanes:
            continue
        composite_lanes[lane_id]["property_class"] = lane_meta.get("property_class")
        composite_lanes[lane_id]["description"] = lane_meta.get("description")
        composite_lanes[lane_id]["authority_tool_ids"] = list(
            lane_meta.get("authority_tool_ids") or ()
        )

    missing_families = sorted(
        REQUIRED_SPECIALIZED_CERTIFIER_FAMILIES - families_present
    )
    represented_families = sorted(families_present)

    # Digest material is the lossless composite + handler projection. Do not
    # also re-embed it under ``digest_components`` (that doubled the durable
    # certificate for no additional binding power).
    digest_components = {
        "composite_lanes": {
            key: composite_lanes[key] for key in sorted(composite_lanes)
        },
        "specialized_by_handler": {
            key: specialized_by_handler[key]
            for key in sorted(specialized_by_handler)
        },
        "certifier_families_represented": represented_families,
        "missing_certifier_families": missing_families,
    }
    kernel_tools = list(
        (composite_lanes.get("kernel") or {}).get("tool_ids") or []
    )
    protocol_tools = list(
        (composite_lanes.get("protocol") or {}).get("tool_ids") or []
    )
    kernel_complete = set(COMPOSITE_PROPERTY_LANE_REQUIRED_TOOLS["kernel"]).issubset(
        kernel_tools
    )
    protocol_complete = set(
        COMPOSITE_PROPERTY_LANE_REQUIRED_TOOLS["protocol"]
    ).issubset(protocol_tools)
    # FVT-079 objective validation repair: true only when every required
    # certifier family is present and composite kernel/protocol tool sets
    # retain their sibling specialized evidence.
    objective_validation_repair = (
        not missing_families and kernel_complete and protocol_complete
    )
    aggregation: dict[str, Any] = {
        "schema_version": SPECIALIZED_AGGREGATION_SCHEMA,
        "interface": SPECIALIZED_AGGREGATION_INTERFACE,
        "goal_id": SPECIALIZED_AGGREGATION_GOAL_ID,
        "task_id": SPECIALIZED_AGGREGATION_TASK_ID,
        "repair_task_id": SPECIALIZED_AGGREGATION_REPAIR_TASK_ID,
        "program": PROGRAM,
        # FVT-079 objective validation repair: re-prove FVT-G203 acceptance.
        "objective_validation_evidence": (
            SPECIALIZED_AGGREGATION_OBJECTIVE_VALIDATION_EVIDENCE
        ),
        "objective_validation_repair": objective_validation_repair,
        "objective_validation_command": (
            SPECIALIZED_AGGREGATION_OBJECTIVE_VALIDATION_COMMAND
        ),
        "policy": {
            "handlers_keyed_by_lane_and_tool": True,
            "lossless": True,
            "collapse_by_check_kind": False,
            "sibling_overwrite_forbidden": True,
            "installers_never_run": True,
            "raw_receipt_identity_retained": True,
            "second_failed_check_of_already_present_kind_blocks_promotion": True,
            "every_check_case_binding_executable_artifact_dependency_source_ceiling_and_raw_digest_in_digest": True,
        },
        "certifier_families_required": sorted(REQUIRED_SPECIALIZED_CERTIFIER_FAMILIES),
        "certifier_families_represented": represented_families,
        "missing_certifier_families": missing_families,
        "all_required_certifiers_represented": not missing_families,
        "composite_lanes": {
            key: composite_lanes[key] for key in sorted(composite_lanes)
        },
        "specialized_by_handler": {
            key: specialized_by_handler[key]
            for key in sorted(specialized_by_handler)
        },
        "kernel_retained_tool_ids": kernel_tools,
        "protocol_retained_tool_ids": protocol_tools,
        "acceptance": {
            "objective_validation_repair": objective_validation_repair,
            "objective_validation_evidence": (
                SPECIALIZED_AGGREGATION_OBJECTIVE_VALIDATION_EVIDENCE
            ),
            "repair_task_id": SPECIALIZED_AGGREGATION_REPAIR_TASK_ID,
            "goal_id": SPECIALIZED_AGGREGATION_GOAL_ID,
            "task_id": SPECIALIZED_AGGREGATION_TASK_ID,
            "handlers_keyed_by_lane_and_tool": True,
            "lossless": True,
            "collapse_by_check_kind": False,
            "sibling_overwrite_forbidden": True,
            "installers_never_run": True,
            "kernel_retains_lean_rocq_isabelle": kernel_complete,
            "protocol_retains_tamarin_proverif": protocol_complete,
            "all_required_certifiers_represented": not missing_families,
        },
        "aggregation_digest_sha256": content_digest(digest_components),
    }
    return aggregation


def _tool_certified_from_semantic_receipt(
    tool_id: str,
    receipt: Mapping[str, Any],
    *,
    certified_key: str,
    selector: str,
) -> tuple[bool, list[Mapping[str, Any]], list[str]]:
    """Return (certified, checks, block_reasons) for one tool from a receipt."""

    if selector == "engine":
        for engine in receipt.get("engines") or []:
            if not isinstance(engine, Mapping):
                continue
            if str(engine.get("engine_id") or "") != tool_id:
                continue
            checks = list(engine.get("checks") or [])
            block_reasons = list(engine.get("block_reasons") or [])
            certified = bool(engine.get("certified"))
            return certified, checks, [str(r) for r in block_reasons]
        return False, [], ["engine_missing_from_semantic_receipt"]

    certified = bool(receipt.get(certified_key) or receipt.get("certified"))
    checks = [
        check
        for check in (receipt.get("checks") or [])
        if not isinstance(check, Mapping)
        or not check.get("tool_id")
        or str(check.get("tool_id")) == tool_id
    ]
    block_reasons = list(receipt.get("block_reasons") or [])
    return certified, checks, [str(r) for r in block_reasons]


def recompute_semantic_tool_check_binding(
    semantic_result: Mapping[str, Any],
    tool_id: str,
) -> dict[str, Any]:
    """Rebuild a compact per-tool check commitment from its full lane receipt."""

    lane_id = str(semantic_result.get("lane_id") or "")
    spec = next(
        (
            item
            for item in SEMANTIC_CERTIFIER_SPECS
            if str(item.get("lane_id") or "") == lane_id
        ),
        None,
    )
    receipt = semantic_result.get("receipt")
    if not isinstance(spec, Mapping) or not isinstance(receipt, Mapping):
        return {
            "valid": False,
            "failure": "semantic_spec_or_canonical_receipt_missing",
            "checks": [],
        }
    expected_tool_ids = {
        str(value) for value in (spec.get("tool_ids") or ())
    }
    if tool_id not in expected_tool_ids:
        return {
            "valid": False,
            "failure": "tool_not_owned_by_semantic_lane",
            "checks": [],
        }
    _, raw_checks, _ = _tool_certified_from_semantic_receipt(
        tool_id,
        receipt,
        certified_key=str(spec["certified_key"]),
        selector=str(spec.get("selector") or "root"),
    )
    normalized = _normalize_semantic_checks(tool_id, raw_checks)
    checks = [check.to_dict() for check in normalized]
    status_counts = {
        status: sum(
            1 for check in checks if str(check.get("status") or "") == status
        )
        for status in ("passed", "failed", "skipped", "unavailable")
    }
    return {
        "valid": bool(checks),
        "failure": None if checks else "normalized_check_set_empty",
        "checks": checks,
        "check_set_digest_sha256": content_digest(checks),
        "checks_total": len(checks),
        "checks_passed": status_counts["passed"],
        "check_kinds_present": sorted(
            {
                str(check.get("kind"))
                for check in checks
                if str(check.get("kind") or "")
            }
        ),
        "check_status_counts": status_counts,
    }


def _semantic_tool_identity(
    tool_id: str,
    receipt: Mapping[str, Any],
    *,
    selector: str,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Extract exact executable/artifact identity from a semantic receipt."""

    source: Mapping[str, Any] = receipt
    if selector == "engine":
        source = next(
            (
                engine
                for engine in (receipt.get("engines") or [])
                if isinstance(engine, Mapping)
                and str(engine.get("engine_id") or "") == tool_id
            ),
            {},
        )

    executable = (
        source.get("executable_path")
        or source.get("executable")
        or receipt.get(f"{tool_id.replace('-', '_')}_executable")
    )
    version = (
        source.get("version_string")
        or source.get("version")
        or receipt.get(f"{tool_id.replace('-', '_')}_version_string")
    )
    artifact_path = source.get("lock_path") or receipt.get("lock_path")
    explicit_artifacts = (
        source.get("artifact_identities")
        or receipt.get(f"{tool_id.replace('-', '_')}_artifact_identities")
        or ()
    )

    def portable_path(value: Any) -> str:
        candidate = Path(str(value))
        if repo_root is not None:
            try:
                return candidate.resolve().relative_to(repo_root.resolve()).as_posix()
            except (OSError, ValueError):
                pass
        return str(candidate)

    artifacts: list[dict[str, Any]] = []
    if executable:
        artifacts.append(
            {
                "kind": "semantic_executable",
                "path": portable_path(executable),
                "sha256": file_digest(str(executable)),
                "artifact_class": classify_executable_artifact(str(executable)),
            }
        )
    if artifact_path:
        artifacts.append(
            {
                "kind": "deployment_artifact",
                "path": portable_path(artifact_path),
                "sha256": file_digest(str(artifact_path)),
                "declared_digest": receipt.get("lock_digest"),
                "artifact_class": "public_deployment_binding",
            }
        )
    if (
        isinstance(explicit_artifacts, Sequence)
        and not isinstance(explicit_artifacts, (str, bytes, bytearray))
    ):
        for raw_artifact in explicit_artifacts:
            if not isinstance(raw_artifact, Mapping):
                continue
            artifact = dict(raw_artifact)
            if artifact.get("path"):
                artifact["path"] = portable_path(artifact["path"])
            if artifact not in artifacts:
                artifacts.append(artifact)
    return {
        "executable_path": str(executable) if executable else None,
        "version_string": str(version) if version else None,
        "identity_probed": bool(
            source.get("identity_probed")
            or source.get("usable")
            or receipt.get("identity_probed")
            or receipt.get(f"{tool_id.replace('-', '_')}_identity_probed")
        ),
        "artifacts": artifacts,
    }


def _digest_matches(stored: Any, computed: str) -> bool:
    value = str(stored or "")
    return bool(value) and value in {computed, f"sha256:{computed}"}


def _validate_semantic_receipt_integrity(
    receipt: Mapping[str, Any],
    *,
    spec: Mapping[str, Any],
    module: Any,
) -> dict[str, Any]:
    """Validate a focused receipt's schema identity and declared self-digest."""

    failures: list[str] = []
    expected_fields = {
        "interface": str(spec["interface"]),
        "schema_version": str(getattr(module, "SCHEMA_VERSION", "") or ""),
        "goal_id": str(getattr(module, "GOAL_ID", "") or ""),
        "task_id": str(getattr(module, "TASK_ID", "") or ""),
    }
    for field_name, expected in expected_fields.items():
        observed = str(receipt.get(field_name) or "")
        if not expected or observed != expected:
            failures.append(f"{field_name}_mismatch")

    digest_fields = [
        field_name
        for field_name in (
            "receipt_digest_sha256",
            "certificate_digest_sha256",
            "digest_sha256",
        )
        if field_name in receipt
    ]
    if not digest_fields:
        failures.append("declared_receipt_digest_missing")
    for field_name in digest_fields:
        body = {key: value for key, value in receipt.items() if key != field_name}
        if not _digest_matches(receipt.get(field_name), content_digest(body)):
            failures.append(f"{field_name}_mismatch")

    return {
        "valid": not failures,
        "failures": failures,
        "declared_digest_fields": digest_fields,
        "interface": receipt.get("interface"),
        "schema_version": receipt.get("schema_version"),
        "goal_id": receipt.get("goal_id"),
        "task_id": receipt.get("task_id"),
    }


def _resolve_artifact_path(path: Any, *, repo_root: Path) -> Path | None:
    if not path:
        return None
    candidate = Path(str(path))
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    try:
        return candidate.resolve()
    except OSError:
        return candidate.absolute()


def _validate_artifact_identities(
    artifacts: Sequence[Mapping[str, Any]],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Recompute every artifact hash and identify production-capable bindings."""

    failures: list[str] = []
    validated: list[dict[str, Any]] = []
    production_bindings: list[dict[str, Any]] = []
    root = repo_root.resolve()
    for index, item in enumerate(artifacts):
        if not isinstance(item, Mapping):
            failures.append(f"artifact_{index}_not_mapping")
            continue
        artifact = dict(item)
        path = _resolve_artifact_path(artifact.get("path"), repo_root=root)
        actual = file_digest(path)
        if path is None or actual is None:
            failures.append(f"artifact_{index}_missing")
            continue
        if str(artifact.get("sha256") or "") != actual:
            failures.append(f"artifact_{index}_sha256_mismatch")
            continue
        artifact_class = str(artifact.get("artifact_class") or "")
        if artifact_class == "repository_source":
            try:
                path.relative_to(root)
            except ValueError:
                failures.append(f"artifact_{index}_repository_source_outside_root")
                continue
        if artifact_class == "public_deployment_binding":
            declared = str(artifact.get("declared_digest") or "")
            try:
                parsed = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                parsed = None
            if not declared or parsed is None or not _digest_matches(
                declared, content_digest(parsed)
            ):
                failures.append(f"artifact_{index}_declared_digest_mismatch")
                continue
        if artifact_class == "managed_release_archive":
            declared = str(artifact.get("declared_digest") or "")
            if not declared or declared != actual:
                failures.append(f"artifact_{index}_declared_digest_mismatch")
                continue
        try:
            artifact["resolved_path"] = path.relative_to(root).as_posix()
        except ValueError:
            artifact["resolved_path"] = str(path)
        validated.append(artifact)
        if artifact_class in {
            "native_or_managed_binary",
            "managed_release_archive",
            "public_deployment_binding",
        }:
            production_bindings.append(artifact)

    return {
        "valid": not failures,
        "failures": failures,
        "validated": validated,
        "production_bindings": production_bindings,
        "has_production_binding": bool(production_bindings),
    }


def _invoke_semantic_certifier(
    certifier: Any,
    *,
    repo_root: Path,
    env: Mapping[str, str],
    extra_kwargs: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Call one certifier with only supported, explicitly offline arguments."""

    parameters = inspect.signature(certifier).parameters
    kwargs: dict[str, Any] = {}
    if "repo_root" in parameters:
        kwargs["repo_root"] = repo_root
    if "env" in parameters:
        kwargs["env"] = dict(env)
    if "skip_install" in parameters:
        kwargs["skip_install"] = True
    if "force_install" in parameters:
        kwargs["force_install"] = False
    for key, value in (extra_kwargs or {}).items():
        if key in parameters:
            kwargs[key] = value
    receipt = certifier(**kwargs)
    if not isinstance(receipt, Mapping):
        raise TypeError("semantic certifier returned non-mapping receipt")
    return receipt


def _checked_vendor_outer_digest(
    certificate: Mapping[str, Any],
    *,
    repo_root: Path,
    uses_public_projection: bool,
) -> str:
    """Recompute the live vendor certificate digest before its receipt wrapper."""

    basis = {
        key: value
        for key, value in certificate.items()
        if key
        not in {
            "certificate_digest_sha256",
            "install_receipt",
            "receipt_path",
        }
    }
    if uses_public_projection:
        basis = public_evidence_projection(basis, repo_root=repo_root)
    return content_digest(basis)


def _checked_vendor_sealed_root_failures(
    sealed_root: Path,
    *,
    install_root: Path,
    dependency_prefix: Path | None,
) -> list[str]:
    """Revalidate the explicit immutable root and fixed vendor subroots."""

    failures: list[str] = []
    try:
        root = sealed_root.resolve(strict=True)
        root_stat = sealed_root.lstat()
    except (OSError, RuntimeError):
        return ["sealed_vendor_root_unreadable"]
    if root != sealed_root or sealed_root.is_symlink():
        failures.append("sealed_vendor_root_resolution_changed")
    if not stat.S_ISDIR(root_stat.st_mode):
        failures.append("sealed_vendor_root_not_directory")
    if root_stat.st_uid != 0:
        failures.append("sealed_vendor_root_not_root_owned")
    if stat.S_IMODE(root_stat.st_mode) & 0o222:
        failures.append("sealed_vendor_root_writable")
    if not _path_under_approved_immutable_root(root):
        failures.append("sealed_vendor_root_not_approved")

    for label, path in (
        ("install_root", install_root),
        ("dependency_prefix", dependency_prefix),
    ):
        if path is None:
            continue
        for failure in _runtime_mtl_sealed_path_failures(
            root,
            path,
            expected_directory=True,
        ):
            failures.append(f"{label}:{failure}")
    return sorted(set(failures))


def _checked_vendor_reference_bindings(
    *,
    repo_root: Path,
    semantic_spec: Mapping[str, Any],
    semantic_module: Any,
    receipt: Mapping[str, Any],
    expected_counts: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """Independently validate each authority-bearing focused reference target."""

    integrity = _validate_semantic_receipt_integrity(
        receipt,
        spec=semantic_spec,
        module=semantic_module,
    )
    offline = _offline_observation(
        receipt,
        production_elevation_allowed=True,
    )
    bindings: dict[str, Any] = {}
    failures_by_tool: dict[str, list[str]] = {}
    for raw_tool_id, raw_expected_count in expected_counts.items():
        tool_id = str(raw_tool_id)
        failures: list[str] = []
        expected_count = -1
        manifest_binding: dict[str, Any] | None = None
        if raw_expected_count == "closed_manifest":
            manifest_meta = receipt.get("manifest")
            manifest_meta = (
                manifest_meta if isinstance(manifest_meta, Mapping) else {}
            )
            raw_manifest_path = str(manifest_meta.get("path") or "")
            if raw_manifest_path.startswith("<repo-root>/"):
                manifest_path = repo_root / raw_manifest_path.removeprefix(
                    "<repo-root>/"
                )
            else:
                manifest_path = Path(raw_manifest_path)
            if (
                not manifest_path.is_absolute()
                and not raw_manifest_path.startswith("<repo-root>/")
            ):
                manifest_path = repo_root / manifest_path
            try:
                resolved_manifest = manifest_path.resolve(strict=True)
                manifest_relative = resolved_manifest.relative_to(
                    repo_root.resolve()
                )
            except (OSError, RuntimeError, ValueError):
                resolved_manifest = None
                manifest_relative = None
                failures.append("reference_closed_manifest_path_invalid")
            try:
                manifest_payload = (
                    json.loads(resolved_manifest.read_text(encoding="utf-8"))
                    if resolved_manifest is not None
                    else None
                )
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                manifest_payload = None
            if not isinstance(manifest_payload, Mapping):
                manifest_payload = {}
                failures.append("reference_closed_manifest_unreadable")
            recipes = manifest_payload.get("case_recipes")
            recipes = (
                list(recipes)
                if isinstance(recipes, Sequence)
                and not isinstance(recipes, (str, bytes, bytearray))
                else []
            )
            try:
                expected_count = int(manifest_meta.get("case_count"))
            except (TypeError, ValueError):
                expected_count = -1
            if (
                expected_count <= 0
                or len(recipes) != expected_count
                or manifest_payload.get("schema_version")
                != manifest_meta.get("schema_version")
                or manifest_payload.get("interface")
                != manifest_meta.get("interface")
            ):
                failures.append("reference_closed_manifest_population_invalid")
            manifest_sha = (
                _bare_file_digest(resolved_manifest)
                if resolved_manifest is not None
                else ""
            )
            source_tree = receipt.get("source_tree")
            source_tree = (
                source_tree if isinstance(source_tree, Mapping) else {}
            )
            source_rows = [
                row
                for row in source_tree.get("files") or ()
                if isinstance(row, Mapping)
            ]
            source_row = next(
                (
                    row
                    for row in source_rows
                    if manifest_relative is not None
                    and str(row.get("path") or "")
                    == manifest_relative.as_posix()
                ),
                None,
            )
            if (
                source_row is None
                or source_row.get("exists") is not True
                or source_row.get("content_sha256") != manifest_sha
            ):
                failures.append("reference_closed_manifest_digest_unbound")
            manifest_binding = {
                "path": (
                    manifest_relative.as_posix()
                    if manifest_relative is not None
                    else None
                ),
                "sha256": manifest_sha or None,
                "schema_version": manifest_payload.get("schema_version"),
                "interface": manifest_payload.get("interface"),
                "case_count": expected_count,
            }
        else:
            try:
                expected_count = int(raw_expected_count)
            except (TypeError, ValueError):
                expected_count = -1
                failures.append("reference_expected_check_count_invalid")
        certified, raw_checks, block_reasons = (
            _tool_certified_from_semantic_receipt(
                tool_id,
                receipt,
                certified_key=str(semantic_spec["certified_key"]),
                selector=str(semantic_spec.get("selector") or "root"),
            )
        )
        normalized = _normalize_semantic_checks(tool_id, raw_checks)
        normalized_payload = [check.to_dict() for check in normalized]
        passed = sum(check.status == "passed" for check in normalized)
        kinds = sorted({check.kind for check in normalized})
        second_failed, second_failures = second_failed_check_blocks_promotion(
            normalized
        )
        if integrity.get("valid") is not True:
            failures.append("reference_receipt_integrity_invalid")
        if offline.get("satisfied") is not True:
            failures.append("reference_offline_observation_invalid")
        if not certified or block_reasons:
            failures.append("reference_target_not_certified")
        if (
            expected_count <= 0
            or len(raw_checks) != expected_count
            or len(normalized) != expected_count
            or passed != expected_count
        ):
            failures.append("reference_check_count_or_status_mismatch")
        if not {"positive", "negative", "mutation", "replay"} <= set(kinds):
            failures.append("reference_pnmr_coverage_incomplete")
        if second_failed:
            failures.extend(second_failures)
        summary = receipt.get("summary")
        summary = summary if isinstance(summary, Mapping) else {}
        if raw_expected_count == "closed_manifest" and (
            summary.get("checks_passed") != expected_count
            or summary.get("checks_total") != expected_count
            or summary.get("checks_failed") not in (None, 0)
            or summary.get("checks_skipped") not in (None, 0)
        ):
            failures.append("reference_closed_manifest_summary_mismatch")

        bindings[tool_id] = {
            "certified": not failures,
            "checks_passed": passed,
            "checks_total": len(normalized),
            "expected_checks_total": expected_count,
            "check_kinds_present": kinds,
            "check_set_digest_sha256": content_digest(normalized_payload),
            "receipt_integrity_valid": integrity.get("valid") is True,
            "offline_observation_satisfied": offline.get("satisfied") is True,
            "closed_manifest": manifest_binding,
        }
        failures_by_tool[tool_id] = sorted(set(failures))
    return bindings, failures_by_tool


def _checked_vendor_receipt_artifact(
    *,
    repo_root: Path,
    receipt_relative: Path,
    checked_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Create a production-capable binding to the authenticated public receipt."""

    return {
        "kind": "checked_vendor_install_receipt",
        "path": receipt_relative.as_posix(),
        "sha256": file_digest(repo_root / receipt_relative),
        "declared_digest": content_digest(checked_receipt),
        "artifact_class": "public_deployment_binding",
    }


def _refresh_semantic_receipt_self_digests(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a JSON-deep copy with every focused self-digest refreshed."""

    refreshed = json.loads(json.dumps(receipt))
    digest_fields = [
        field_name
        for field_name in (
            "receipt_digest_sha256",
            "certificate_digest_sha256",
            "digest_sha256",
        )
        if field_name in refreshed
    ]
    if len(digest_fields) != 1:
        raise ValueError("focused semantic receipt must declare one self-digest")
    field_name = digest_fields[0]
    refreshed[field_name] = content_digest(
        {
            key: value
            for key, value in refreshed.items()
            if key != field_name
        }
    )
    return refreshed


def _bind_checked_vendor_fanin_to_receipt(
    receipt: Mapping[str, Any],
    *,
    semantic_spec: Mapping[str, Any],
    fanin: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind fan-in metadata and its checked receipt artifact into the authority."""

    bound = json.loads(json.dumps(receipt))
    bound["checked_vendor_fanin"] = dict(fanin)
    eligible = {
        str(tool_id) for tool_id in fanin.get("eligible_tool_ids") or ()
    }
    artifact = fanin.get("checked_install_receipt_artifact")
    artifact = dict(artifact) if isinstance(artifact, Mapping) else None
    selector = str(semantic_spec.get("selector") or "root")
    if artifact is not None:
        if selector == "engine":
            for engine in bound.get("engines") or []:
                if not isinstance(engine, dict):
                    continue
                tool_id = str(engine.get("engine_id") or "")
                if tool_id not in eligible:
                    continue
                identities = [
                    dict(item)
                    for item in engine.get("artifact_identities") or ()
                    if isinstance(item, Mapping)
                ]
                if artifact not in identities:
                    identities.append(dict(artifact))
                engine["artifact_identities"] = identities
        else:
            for raw_tool_id in semantic_spec.get("tool_ids") or ():
                tool_id = str(raw_tool_id)
                if tool_id not in eligible:
                    continue
                field_name = (
                    f"{tool_id.replace('-', '_')}_artifact_identities"
                )
                identities = [
                    dict(item)
                    for item in bound.get(field_name) or ()
                    if isinstance(item, Mapping)
                ]
                if artifact not in identities:
                    identities.append(dict(artifact))
                bound[field_name] = identities
    return _refresh_semantic_receipt_self_digests(bound)


def _adapt_hyper_vendor_checks(
    tool_id: str,
    raw_checks: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Retain every vendor check while projecting violation to PNMR negative."""

    adapted: list[dict[str, Any]] = []
    for index, raw_check in enumerate(raw_checks):
        check = dict(raw_check)
        source_kind = str(check.get("kind") or "").lower()
        check["check_id"] = str(
            check.get("check_id") or f"{tool_id}.vendor.{index}"
        )
        check["tool_id"] = tool_id
        check["kind"] = (
            "negative" if source_kind == "violation" else source_kind
        )
        check["source_vendor_kind"] = source_kind
        check["source_check_digest_sha256"] = content_digest(raw_check)
        adapted.append(check)
    return adapted


def _build_checked_hyper_vendor_adapter(
    *,
    repo_root: Path,
    sealed_root: Path | None,
    semantic_spec: Mapping[str, Any],
    semantic_module: Any,
    reference_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Authenticate the complete 3 x 22 bounded Hyper vendor certificate."""

    spec = CHECKED_HYPER_VENDOR_FANIN_SPEC
    targets = tuple(str(item) for item in spec["targets"])
    checked_relative = Path(spec["checked_receipt_relative"])
    checked_path = repo_root / checked_relative
    module_relative = Path(spec["module_relative"])
    module_path = repo_root / module_relative
    failures: list[str] = []
    host_platform = observed_platform_id()

    try:
        checked_receipt = json.loads(checked_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        checked_receipt = None
        failures.append("checked_hyper_vendor_receipt_unreadable")
    if not isinstance(checked_receipt, Mapping):
        checked_receipt = {}
        failures.append("checked_hyper_vendor_receipt_not_mapping")
    for field_name, expected in {
        "schema_version": spec["checked_receipt_schema"],
        "interface": spec["interface"],
        "goal_id": spec["goal_id"],
        "task_id": spec["task_id"],
        "repair_task_id": spec["repair_task_id"],
        "host_platform": host_platform,
        "authority_ceiling": "bounded",
        "certified": True,
    }.items():
        if checked_receipt.get(field_name) != expected:
            failures.append(
                f"checked_hyper_vendor_receipt_{field_name}_mismatch"
            )
    checked_self_digest = content_digest(
        {
            key: value
            for key, value in checked_receipt.items()
            if key != "receipt_digest_sha256"
        }
    )
    if not _digest_matches(
        checked_receipt.get("receipt_digest_sha256"),
        checked_self_digest,
    ):
        failures.append("checked_hyper_vendor_receipt_self_digest_mismatch")
    checked_summary = checked_receipt.get("summary")
    checked_summary = (
        checked_summary if isinstance(checked_summary, Mapping) else {}
    )
    if (
        checked_summary.get("vendor_certified") is not True
        or checked_summary.get("checks_passed")
        != spec["expected_vendor_checks"]
        or checked_summary.get("checks_total")
        != spec["expected_vendor_checks"]
        or list(checked_summary.get("block_reasons") or ())
    ):
        failures.append("checked_hyper_vendor_summary_invalid")
    checked_policy = checked_receipt.get("policy")
    checked_policy = (
        checked_policy if isinstance(checked_policy, Mapping) else {}
    )
    if (
        checked_policy.get("never_grants_theorem_authority") is not True
        or checked_policy.get("never_authorizes_universal_proof") is not True
        or checked_policy.get("grants_theorem_authority") is not False
        or checked_policy.get("authorizes_universal_proof") is not False
        or checked_policy.get("authority_ceiling") != "bounded"
    ):
        failures.append("checked_hyper_vendor_authority_policy_invalid")
    for tool_id in targets:
        engine = checked_receipt.get(tool_id)
        engine = engine if isinstance(engine, Mapping) else {}
        if (
            engine.get("tool_id") != tool_id
            or engine.get("certified") is not True
            or engine.get("usable") is not True
            or engine.get("is_vendor_build") is not True
            or engine.get("is_hermetic_engine") is not False
            or engine.get("role") != "authority"
            or engine.get("authority_ceiling") != "bounded"
            or engine.get("never_authorizes_universal_proof") is not True
            or engine.get("never_grants_theorem_authority") is not True
            or not SHA256_RE.fullmatch(
                str(engine.get("artifact_sha256") or "")
            )
        ):
            failures.append(f"checked_hyper_vendor_{tool_id}_identity_invalid")

    if sealed_root is None:
        failures.append("sealed_hyper_vendor_root_unavailable")
    else:
        failures.extend(
            _checked_vendor_sealed_root_failures(
                sealed_root,
                install_root=sealed_root,
                dependency_prefix=None,
            )
        )
    if checked_receipt.get("host_platform") != host_platform:
        failures.append("checked_hyper_vendor_host_platform_mismatch")

    live_certificate: Mapping[str, Any] = {}
    if not module_path.is_file():
        failures.append("checked_hyper_vendor_certifier_missing")
    elif sealed_root is not None and not failures:
        try:
            vendor_module = _load_module_from_path(
                module_path,
                "fvt_checked_hyper_vendor_fanin",
            )
            callable_ = getattr(
                vendor_module,
                str(spec["callable_name"]),
                None,
            )
            if not callable(callable_):
                raise AttributeError("Hyper vendor certifier not callable")
            observed = callable_(
                install_root=sealed_root,
                engines=targets,
                force_install=False,
                skip_install=True,
                platform_id=host_platform,
                repo_root=repo_root,
                lock_path=repo_root / DEFAULT_LOCK_RELATIVE,
                dependency_roots=None,
                write_receipt_path=None,
            )
            if not isinstance(observed, Mapping):
                raise TypeError("Hyper vendor certifier returned non-mapping")
            live_certificate = observed
        except Exception as exc:  # noqa: BLE001 — fail closed without paths
            failures.append(
                f"checked_hyper_vendor_certifier_error:{type(exc).__name__}"
            )

    live_by_target: dict[str, Mapping[str, Any]] = {}
    raw_checks_by_target: dict[str, list[dict[str, Any]]] = {}
    adapted_checks_by_target: dict[str, list[dict[str, Any]]] = {}
    cases_by_target: dict[str, list[dict[str, Any]]] = {}
    artifacts_by_target: dict[str, list[dict[str, Any]]] = {}
    if live_certificate:
        for field_name, expected in {
            "schema_version": spec["live_schema"],
            "interface": spec["interface"],
            "goal_id": spec["goal_id"],
            "task_id": spec["task_id"],
            "repair_task_id": spec["repair_task_id"],
            "host_platform": host_platform,
            "authority_ceiling": "bounded",
            "certified": True,
        }.items():
            if live_certificate.get(field_name) != expected:
                failures.append(f"live_hyper_vendor_{field_name}_mismatch")
        repair = live_certificate.get("objective_validation_repair")
        repair = repair if isinstance(repair, Mapping) else {}
        if (
            repair.get("status") != "satisfied"
            or repair.get("vendor_certified") is not True
            or live_certificate.get("forbids_theorem_authority") is not True
            or live_certificate.get(
                "forbids_universal_claims_beyond_bounds"
            )
            is not True
            or live_certificate.get("install") is not None
        ):
            failures.append("live_hyper_vendor_authority_or_repair_invalid")
        outer_digest = _checked_vendor_outer_digest(
            live_certificate,
            repo_root=repo_root,
            uses_public_projection=True,
        )
        if not _digest_matches(
            live_certificate.get("certificate_digest_sha256"),
            outer_digest,
        ):
            failures.append("live_hyper_vendor_self_digest_mismatch")
        if live_certificate.get("install_receipt") != checked_receipt:
            failures.append("live_hyper_vendor_nested_receipt_mismatch")
        live_summary = live_certificate.get("summary")
        live_summary = (
            live_summary if isinstance(live_summary, Mapping) else {}
        )
        if (
            live_summary.get("vendor_certified") is not True
            or live_summary.get("checks_passed")
            != spec["expected_vendor_checks"]
            or live_summary.get("checks_total")
            != spec["expected_vendor_checks"]
            or list(live_summary.get("block_reasons") or ())
        ):
            failures.append("live_hyper_vendor_summary_invalid")

        engine_list = [
            engine
            for engine in live_certificate.get("engines") or ()
            if isinstance(engine, Mapping)
        ]
        engine_ids = [str(engine.get("engine_id") or "") for engine in engine_list]
        if engine_ids != list(targets) or len(set(engine_ids)) != len(targets):
            failures.append("live_hyper_vendor_engine_population_invalid")
        listed = {str(engine.get("engine_id") or ""): engine for engine in engine_list}
        for tool_id in targets:
            engine = live_certificate.get(tool_id)
            engine = engine if isinstance(engine, Mapping) else {}
            live_by_target[tool_id] = engine
            if engine != listed.get(tool_id):
                failures.append(
                    f"live_hyper_vendor_{tool_id}_section_list_mismatch"
                )
            raw_checks = [
                dict(check)
                for check in engine.get("checks") or ()
                if isinstance(check, Mapping)
            ]
            cases = [
                dict(case)
                for case in engine.get("case_results") or ()
                if isinstance(case, Mapping)
            ]
            raw_checks_by_target[tool_id] = raw_checks
            cases_by_target[tool_id] = cases
            adapted_checks_by_target[tool_id] = _adapt_hyper_vendor_checks(
                tool_id,
                raw_checks,
            )
            check_ids = [
                str(check.get("check_id") or "") for check in raw_checks
            ]
            if (
                engine.get("engine_id") != tool_id
                or engine.get("certified") is not True
                or engine.get("usable") is not True
                or engine.get("role") != "authority"
                or engine.get("authority_ceiling") != "bounded"
                or engine.get("is_vendor_build") is not True
                or engine.get("is_hermetic_engine") is not False
                or engine.get("authorizes_universal_proof") is not False
                or list(engine.get("block_reasons") or ())
                or len(raw_checks) != spec["checks_per_target"]
                or len(check_ids) != spec["checks_per_target"]
                or any(not check_id for check_id in check_ids)
                or len(set(check_ids)) != spec["checks_per_target"]
                or any(check.get("status") != "passed" for check in raw_checks)
            ):
                failures.append(
                    f"live_hyper_vendor_{tool_id}_full_check_set_invalid"
                )
            executable_raw = str(engine.get("executable") or "")
            if sealed_root is None or not executable_raw:
                failures.append(
                    f"live_hyper_vendor_{tool_id}_executable_missing"
                )
            else:
                executable_path = Path(executable_raw)
                for failure in _runtime_mtl_sealed_path_failures(
                    sealed_root,
                    executable_path,
                    expected_directory=False,
                    executable=True,
                ):
                    failures.append(
                        f"live_hyper_vendor_{tool_id}_executable:{failure}"
                    )
                executable_sha = file_digest(executable_path)
                executable_bare_sha = _bare_file_digest(executable_path)
                executable_class = classify_executable_artifact(executable_path)
                expected_class = (
                    "launcher_script"
                    if tool_id == "mchyper"
                    else "native_or_managed_binary"
                )
                if (
                    not _digest_matches(
                        engine.get("artifact_sha256"),
                        executable_bare_sha,
                    )
                    or executable_class != expected_class
                ):
                    failures.append(
                        f"live_hyper_vendor_{tool_id}_executable_identity_invalid"
                    )
                engine_artifacts = [
                    {
                        "kind": "vendor_engine_executable",
                        "path": executable_raw,
                        "sha256": executable_sha,
                        "artifact_class": executable_class,
                    }
                ]
                if tool_id == "mchyper":
                    version_root = executable_path.parent.parent
                    dependency_paths = (
                        (
                            "launcher_runtime",
                            sealed_root
                            / "build-dependencies"
                            / "mchyper"
                            / "python-2.7.18"
                            / "bin"
                            / "python2.7",
                        ),
                        (
                            "launcher_target",
                            executable_path.parent / "src" / "Main",
                        ),
                        ("runtime_dependency_abc", version_root / "abc" / "abc"),
                        (
                            "runtime_dependency_aigtoaig",
                            version_root / "aiger" / "aigtoaig",
                        ),
                    )
                    for artifact_kind, dependency_path in dependency_paths:
                        dependency_failures = _runtime_mtl_sealed_path_failures(
                            sealed_root,
                            dependency_path,
                            expected_directory=False,
                            executable=True,
                        )
                        dependency_class = classify_executable_artifact(
                            dependency_path
                        )
                        dependency_sha = file_digest(dependency_path)
                        if (
                            dependency_failures
                            or dependency_class != "native_or_managed_binary"
                            or not dependency_sha
                        ):
                            failures.append(
                                "live_hyper_vendor_mchyper_"
                                f"{artifact_kind}_identity_invalid"
                            )
                        engine_artifacts.append(
                            {
                                "kind": artifact_kind,
                                "path": str(dependency_path),
                                "sha256": dependency_sha,
                                "artifact_class": dependency_class,
                            }
                        )
                artifacts_by_target[tool_id] = engine_artifacts

    checked_artifact = _checked_vendor_receipt_artifact(
        repo_root=repo_root,
        receipt_relative=checked_relative,
        checked_receipt=checked_receipt,
    )
    vendor_failures = sorted(set(failures))
    vendor_valid = not vendor_failures

    base_engines: list[dict[str, Any]] = []
    for tool_id in targets:
        live_engine = live_by_target.get(tool_id, {})
        base_engines.append(
            {
                "engine_id": tool_id,
                "version": live_engine.get("version"),
                "usable": vendor_valid,
                "identity_probed": vendor_valid,
                "certified": vendor_valid,
                "role": "authority",
                "authority_ceiling": "bounded",
                "authorizes_universal_proof": False,
                "is_theorem_authority": False,
                "checks": adapted_checks_by_target.get(tool_id, []),
                "case_results": cases_by_target.get(tool_id, []),
                "block_reasons": (
                    []
                    if vendor_valid
                    else [f"checked_hyper_vendor_fanin_invalid:{tool_id}"]
                ),
                "executable": live_engine.get("executable"),
                "executable_sha256": next(
                    (
                        artifact.get("sha256")
                        for artifact in artifacts_by_target.get(tool_id, [])
                        if artifact.get("kind") == "vendor_engine_executable"
                    ),
                    None,
                ),
                "executable_artifact_class": classify_executable_artifact(
                    live_engine.get("executable")
                ),
                "artifact_identities": [
                    *[
                        dict(artifact)
                        for artifact in artifacts_by_target.get(tool_id, [])
                    ],
                    dict(checked_artifact),
                ],
                "source_vendor_certificate_digest_sha256": (
                    live_certificate.get("certificate_digest_sha256")
                ),
                "independent_reference_available": False,
                "authority_basis": "complete_live_vendor_bounded_corpus",
            }
        )
    adapter: dict[str, Any] = {
        "schema_version": str(getattr(semantic_module, "SCHEMA_VERSION", "")),
        "interface": str(semantic_spec["interface"]),
        "goal_id": str(getattr(semantic_module, "GOAL_ID", "")),
        "task_id": str(getattr(semantic_module, "TASK_ID", "")),
        "program": "formal-verification-tactician/hyperproperty-vendor-adapter",
        "lane_id": str(semantic_spec["lane_id"]),
        "handler_id": "checked_hyperproperty_vendor_adapter@1",
        "host_platform": host_platform,
        "authority_ceiling": "bounded",
        "forbids_theorem_authority": True,
        "forbids_universal_claims_beyond_bounds": True,
        "certified": vendor_valid,
        "engines": base_engines,
        "engine_ids": list(targets),
        "install_attempted": False,
        "download_attempted": False,
        "network_used": False,
        "policy": {
            "no_install": True,
            "no_download": True,
            "no_network": True,
            "checked_vendor_receipt_required": True,
            "complete_vendor_corpus_required": True,
            "independent_reference_claimed": False,
            "authority_ceiling": "bounded",
            "never_grants_theorem_authority": True,
            "never_authorizes_universal_proof": True,
        },
        "summary": {
            "engines_certified": len(targets) if vendor_valid else 0,
            "engines_total": len(targets),
            "checks_passed": sum(
                check.get("status") == "passed"
                for checks in adapted_checks_by_target.values()
                for check in checks
            ),
            "checks_total": sum(
                len(checks) for checks in adapted_checks_by_target.values()
            ),
            "block_reasons": vendor_failures,
        },
        "independent_reference": {
            "available": False,
            "claimed": False,
            "reason": (
                "The reviewed vendor engines are the bounded capability "
                "targets; no independent peer is relabelled or fabricated."
            ),
        },
        "digest_sha256": "",
    }
    adapter["digest_sha256"] = content_digest(
        {key: value for key, value in adapter.items() if key != "digest_sha256"}
    )
    binding_receipt = (
        reference_receipt
        if isinstance(reference_receipt, Mapping) and reference_receipt
        else adapter
    )
    receipt_integrity = _validate_semantic_receipt_integrity(
        binding_receipt,
        spec=semantic_spec,
        module=semantic_module,
    )
    target_bindings: dict[str, Any] = {}
    target_failures: dict[str, list[str]] = {}
    for tool_id in targets:
        tool_failures: list[str] = []
        certified, raw_checks, block_reasons = (
            _tool_certified_from_semantic_receipt(
                tool_id,
                binding_receipt,
                certified_key=str(semantic_spec["certified_key"]),
                selector=str(semantic_spec.get("selector") or "engine"),
            )
        )
        normalized = _normalize_semantic_checks(tool_id, raw_checks)
        normalized_payload = [check.to_dict() for check in normalized]
        source_digests = [
            str(check.get("source_check_digest_sha256") or "")
            for check in raw_checks
            if isinstance(check, Mapping)
        ]
        expected_source_digests = [
            content_digest(check)
            for check in raw_checks_by_target.get(tool_id, [])
        ]
        kinds = sorted({check.kind for check in normalized})
        if receipt_integrity.get("valid") is not True:
            tool_failures.append("hyper_adapter_receipt_integrity_invalid")
        if not certified or block_reasons:
            tool_failures.append("hyper_adapter_target_not_certified")
        if (
            len(raw_checks) != spec["checks_per_target"]
            or len(normalized) != spec["checks_per_target"]
            or any(check.status != "passed" for check in normalized)
            or not {"positive", "negative", "mutation", "replay"} <= set(kinds)
        ):
            tool_failures.append("hyper_adapter_pnmr_projection_invalid")
        if source_digests != expected_source_digests:
            tool_failures.append("hyper_adapter_source_check_binding_invalid")
        target_bindings[tool_id] = {
            "certified": not tool_failures,
            "checks_passed": sum(
                check.status == "passed" for check in normalized
            ),
            "checks_total": len(normalized),
            "expected_checks_total": spec["checks_per_target"],
            "check_kinds_present": kinds,
            "check_set_digest_sha256": content_digest(normalized_payload),
            "source_vendor_check_set_digest_sha256": content_digest(
                raw_checks_by_target.get(tool_id, [])
            ),
            "source_check_digests_sha256": expected_source_digests,
            "independent_reference_available": False,
            "vendor_pnmr_projection": True,
            "authority_ceiling": "bounded",
        }
        target_failures[tool_id] = sorted(set(tool_failures))

    eligible = [
        tool_id
        for tool_id in targets
        if vendor_valid and not target_failures.get(tool_id)
    ]
    flattened_checks = [
        check
        for tool_id in targets
        for check in raw_checks_by_target.get(tool_id, [])
    ]
    flattened_cases = [
        case
        for tool_id in targets
        for case in cases_by_target.get(tool_id, [])
    ]
    fanin: dict[str, Any] = {
        "schema_version": CHECKED_VENDOR_FANIN_SCHEMA,
        "configured": True,
        "lane_id": str(spec["lane_id"]),
        "vendor_tool_id": "hyperproperty-vendor",
        "vendor_authority": str(spec["vendor_authority"]),
        "reference_authority_retained_by": list(targets),
        "evidence_class": str(spec["evidence_class"]),
        "vendor_valid": vendor_valid,
        "complete": len(eligible) == len(targets),
        "eligible_tool_ids": eligible,
        "failures": vendor_failures,
        "per_tool_failures": target_failures,
        "reference_bindings": target_bindings,
        "checked_install_receipt": {
            "path": checked_relative.as_posix(),
            "file_sha256": file_digest(checked_path),
            "self_digest_sha256": checked_receipt.get(
                "receipt_digest_sha256"
            ),
            "content_digest_sha256": content_digest(checked_receipt),
            "exact_live_nested_match": bool(
                live_certificate
                and live_certificate.get("install_receipt")
                == checked_receipt
            ),
        },
        "checked_install_receipt_artifact": checked_artifact,
        "live_certificate": {
            "schema_version": live_certificate.get("schema_version"),
            "interface": live_certificate.get("interface"),
            "goal_id": live_certificate.get("goal_id"),
            "task_id": live_certificate.get("task_id"),
            "repair_task_id": live_certificate.get("repair_task_id"),
            "host_platform": live_certificate.get("host_platform"),
            "certificate_digest_sha256": live_certificate.get(
                "certificate_digest_sha256"
            ),
            "certified": live_certificate.get("certified") is True,
            "checks_passed": sum(
                check.get("status") == "passed" for check in flattened_checks
            ),
            "checks_total": len(flattened_checks),
            "check_ids": [
                str(check.get("check_id") or "")
                for check in flattened_checks
            ],
            "check_set_digest_sha256": content_digest(flattened_checks),
            "case_results_total": len(flattened_cases),
            "case_result_set_digest_sha256": content_digest(flattened_cases),
            "nested_install_receipt_digest_sha256": (
                live_certificate.get("install_receipt") or {}
            ).get("receipt_digest_sha256")
            if isinstance(live_certificate.get("install_receipt"), Mapping)
            else None,
            "per_engine": {
                tool_id: {
                    "checks_passed": sum(
                        check.get("status") == "passed"
                        for check in raw_checks_by_target.get(tool_id, [])
                    ),
                    "checks_total": len(
                        raw_checks_by_target.get(tool_id, [])
                    ),
                    "check_set_digest_sha256": content_digest(
                        raw_checks_by_target.get(tool_id, [])
                    ),
                    "case_results_total": len(
                        cases_by_target.get(tool_id, [])
                    ),
                    "case_result_set_digest_sha256": content_digest(
                        cases_by_target.get(tool_id, [])
                    ),
                    "artifact_sha256": live_by_target.get(tool_id, {}).get(
                        "artifact_sha256"
                    ),
                    "artifact_identities": [
                        dict(artifact)
                        for artifact in artifacts_by_target.get(tool_id, [])
                    ],
                }
                for tool_id in targets
            },
        },
        "source_module": {
            "path": module_relative.as_posix(),
            "sha256": file_digest(module_path),
        },
        "sealed_root_binding": {
            "environment": RUNTIME_MTL_SEALED_ROOT_ENV,
            "explicit": sealed_root is not None,
            "install_root_relative": ".",
            "dependency_prefix_relative": None,
            "ambient_path_discovery": False,
            "skip_install": True,
            "force_install": False,
            "platform_id": host_platform,
        },
        "policy": {
            "complete_three_engine_vendor_corpus_required": True,
            "exact_nested_install_receipt_required": True,
            "section_list_identity_required": True,
            "independent_reference_claimed": False,
            "vendor_pnmr_projection_retains_every_check": True,
            "bounded_authority_only": True,
            "never_grants_theorem_authority": True,
            "never_authorizes_universal_proof": True,
        },
    }
    fanin["digest_sha256"] = content_digest(fanin)
    bound_adapter = _bind_checked_vendor_fanin_to_receipt(
        adapter,
        semantic_spec=semantic_spec,
        fanin=fanin,
    )
    return {"fanin": fanin, "adapter_receipt": bound_adapter}


def _build_checked_vendor_fanin(
    *,
    repo_root: Path,
    sealed_root: Path | None,
    semantic_spec: Mapping[str, Any],
    semantic_module: Any,
    reference_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-run, authenticate, and join one checked vendor differential witness."""

    lane_id = str(semantic_spec.get("lane_id") or "")
    raw_vendor_spec = CHECKED_VENDOR_FANIN_SPECS.get(lane_id)
    if not isinstance(raw_vendor_spec, Mapping):
        return {
            "configured": False,
            "vendor_valid": False,
            "complete": False,
            "eligible_tool_ids": [],
            "failures": ["checked_vendor_fanin_not_configured"],
        }
    vendor_spec = dict(raw_vendor_spec)
    checked_relative = Path(vendor_spec["checked_receipt_relative"])
    checked_path = repo_root / checked_relative
    module_relative = Path(vendor_spec["module_relative"])
    module_path = repo_root / module_relative
    failures: list[str] = []

    try:
        checked_receipt = json.loads(checked_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        checked_receipt = None
        failures.append("checked_vendor_install_receipt_unreadable")
    if not isinstance(checked_receipt, Mapping):
        checked_receipt = {}
        failures.append("checked_vendor_install_receipt_not_mapping")

    expected_checked_fields = {
        "schema_version": vendor_spec["checked_receipt_schema"],
        "interface": vendor_spec["interface"],
        "goal_id": vendor_spec["goal_id"],
        "task_id": vendor_spec["task_id"],
        "repair_task_id": vendor_spec["repair_task_id"],
        "certified": True,
    }
    for field_name, expected in expected_checked_fields.items():
        if checked_receipt.get(field_name) != expected:
            failures.append(
                f"checked_vendor_install_receipt_{field_name}_mismatch"
            )
    checked_self_digest = content_digest(
        {
            key: value
            for key, value in checked_receipt.items()
            if key != "receipt_digest_sha256"
        }
    )
    if not _digest_matches(
        checked_receipt.get("receipt_digest_sha256"),
        checked_self_digest,
    ):
        failures.append("checked_vendor_install_receipt_self_digest_mismatch")

    checked_summary = checked_receipt.get("summary")
    checked_summary = (
        checked_summary if isinstance(checked_summary, Mapping) else {}
    )
    expected_vendor_checks = int(vendor_spec["expected_vendor_checks"])
    if (
        checked_summary.get(vendor_spec["summary_certified_key"]) is not True
        or checked_summary.get("checks_passed") != expected_vendor_checks
        or checked_summary.get("checks_total") != expected_vendor_checks
        or list(checked_summary.get("block_reasons") or ())
    ):
        failures.append("checked_vendor_install_receipt_not_fully_certified")
    checked_vendor_section = checked_receipt.get(
        vendor_spec["vendor_section"]
    )
    checked_vendor_section = (
        checked_vendor_section
        if isinstance(checked_vendor_section, Mapping)
        else {}
    )
    checked_policy = checked_receipt.get("policy")
    checked_policy = (
        checked_policy if isinstance(checked_policy, Mapping) else {}
    )
    if lane_id == "runtime_mtl" and (
        checked_vendor_section.get("tool_id")
        != vendor_spec["vendor_tool_id"]
        or checked_vendor_section.get("certified") is not True
        or checked_vendor_section.get("usable") is not True
        or checked_vendor_section.get("is_vendor_build") is not True
        or checked_vendor_section.get("is_hermetic_parity_engine") is not False
        or checked_vendor_section.get("no_python_reference_dispatch") is not True
        or checked_vendor_section.get("finite_trace_authority_only") is not True
        or checked_policy.get("never_grants_theorem_authority") is not True
        or checked_policy.get("grants_theorem_authority") is not False
        or checked_policy.get("grants_global_correctness") is not False
    ):
        failures.append("checked_runtime_mtl_vendor_authority_policy_invalid")
    if lane_id == "datalog_secpal":
        checked_exception = checked_receipt.get("secpal_platform_exception")
        checked_exception = (
            checked_exception
            if isinstance(checked_exception, Mapping)
            else {}
        )
        if (
            checked_vendor_section.get("tool_id")
            != vendor_spec["vendor_tool_id"]
            or checked_vendor_section.get("certified") is not True
            or checked_vendor_section.get("usable") is not True
            or checked_vendor_section.get("is_vendor_build") is not True
            or checked_vendor_section.get("is_hermetic_shadow") is not False
            or checked_policy.get(
                "never_grants_authorization_authority_to_shadows"
            )
            is not True
            or checked_policy.get("grants_authorization_decision_authority")
            is not False
            or checked_policy.get("grants_theorem_authority") is not False
            or checked_exception.get("exception") is not True
            or checked_exception.get("narrow_scope") is not True
            or checked_exception.get("classification")
            != "unsupported_here"
            or any(
                checked_exception.get(field_name) is not False
                for field_name in (
                    "installed",
                    "complete",
                    "authoritative",
                    "production_certified",
                )
            )
        ):
            failures.append("checked_souffle_vendor_authority_policy_invalid")

    reference_bindings, reference_failures = (
        _checked_vendor_reference_bindings(
            repo_root=repo_root,
            semantic_spec=semantic_spec,
            semantic_module=semantic_module,
            receipt=reference_receipt,
            expected_counts=vendor_spec["expected_reference_checks"],
        )
    )

    install_root = None
    dependency_prefix = None
    if sealed_root is None:
        failures.append("sealed_vendor_root_unavailable")
    else:
        install_root = (
            sealed_root / Path(vendor_spec["install_root_relative"])
        )
        raw_dependency = vendor_spec.get("dependency_prefix_relative")
        dependency_prefix = (
            None
            if raw_dependency is None
            else (sealed_root / Path(raw_dependency))
        )
        failures.extend(
            _checked_vendor_sealed_root_failures(
                sealed_root,
                install_root=install_root,
                dependency_prefix=dependency_prefix,
            )
        )

    live_certificate: Mapping[str, Any] = {}
    vendor_module = None
    if not module_path.is_file():
        failures.append("checked_vendor_certifier_module_missing")
    elif install_root is not None and not failures:
        try:
            vendor_module = _load_module_from_path(
                module_path,
                f"fvt_checked_vendor_fanin_{lane_id}",
            )
            certifier = getattr(
                vendor_module,
                str(vendor_spec["callable_name"]),
                None,
            )
            if not callable(certifier):
                raise AttributeError("checked vendor certifier not callable")
            kwargs: dict[str, Any] = {
                "install_root": install_root,
                "skip_install": True,
                "force_install": False,
                "repo_root": repo_root,
                "lock_path": repo_root / DEFAULT_LOCK_RELATIVE,
                "write_receipt_path": None,
            }
            if dependency_prefix is not None:
                kwargs["dependency_prefix"] = dependency_prefix
                expected_platform = str(
                    checked_receipt.get("host_platform") or ""
                )
                if expected_platform != observed_platform_id():
                    raise RuntimeError(
                        "checked vendor platform does not match current host"
                    )
                kwargs["platform_id"] = expected_platform
            observed = certifier(**kwargs)
            if not isinstance(observed, Mapping):
                raise TypeError("checked vendor certifier returned non-mapping")
            live_certificate = observed
        except Exception as exc:  # noqa: BLE001 — vendor join fails closed
            failures.append(
                f"checked_vendor_certifier_error:{type(exc).__name__}"
            )

    if live_certificate:
        expected_live_fields = {
            "schema_version": vendor_spec["live_schema"],
            "interface": vendor_spec["interface"],
            "goal_id": vendor_spec["goal_id"],
            "task_id": vendor_spec["task_id"],
            "repair_task_id": vendor_spec["repair_task_id"],
            "certified": True,
            "objective_validation_repair": True,
        }
        for field_name, expected in expected_live_fields.items():
            if live_certificate.get(field_name) != expected:
                failures.append(f"live_vendor_{field_name}_mismatch")
        if live_certificate.get("install") is not None:
            failures.append("live_vendor_install_was_attempted")
        outer_digest = _checked_vendor_outer_digest(
            live_certificate,
            repo_root=repo_root,
            uses_public_projection=bool(
                vendor_spec["outer_digest_uses_public_projection"]
            ),
        )
        if not _digest_matches(
            live_certificate.get("certificate_digest_sha256"),
            outer_digest,
        ):
            failures.append("live_vendor_certificate_self_digest_mismatch")
        if live_certificate.get("install_receipt") != checked_receipt:
            failures.append("live_vendor_nested_install_receipt_mismatch")

        live_summary = live_certificate.get("summary")
        live_summary = (
            live_summary if isinstance(live_summary, Mapping) else {}
        )
        if (
            live_summary.get(vendor_spec["summary_certified_key"]) is not True
            or live_summary.get("checks_passed") != expected_vendor_checks
            or live_summary.get("checks_total") != expected_vendor_checks
            or list(live_summary.get("block_reasons") or ())
        ):
            failures.append("live_vendor_summary_not_fully_certified")

        vendor_section = live_certificate.get(vendor_spec["vendor_section"])
        vendor_section = (
            vendor_section if isinstance(vendor_section, Mapping) else {}
        )
        vendor_checks = vendor_section.get("checks")
        vendor_checks = (
            list(vendor_checks)
            if isinstance(vendor_checks, Sequence)
            and not isinstance(vendor_checks, (str, bytes, bytearray))
            else []
        )
        check_ids = [
            str(check.get("check_id") or "")
            for check in vendor_checks
            if isinstance(check, Mapping)
        ]
        if (
            vendor_section.get("certified") is not True
            or vendor_section.get("usable") is not True
            or list(vendor_section.get("block_reasons") or ())
            or len(vendor_checks) != expected_vendor_checks
            or len(check_ids) != expected_vendor_checks
            or any(not check_id for check_id in check_ids)
            or len(set(check_ids)) != expected_vendor_checks
            or any(
                not isinstance(check, Mapping)
                or check.get("status") != "passed"
                for check in vendor_checks
            )
        ):
            failures.append("live_vendor_full_check_set_invalid")
        executable_raw = str(vendor_section.get("executable") or "")
        if sealed_root is not None and executable_raw:
            executable = Path(executable_raw)
            for failure in _runtime_mtl_sealed_path_failures(
                sealed_root,
                executable,
                expected_directory=False,
                executable=True,
            ):
                failures.append(f"live_vendor_executable:{failure}")
        else:
            failures.append("live_vendor_executable_missing")

        if lane_id == "runtime_mtl":
            if (
                vendor_section.get("engine_id")
                != vendor_spec["vendor_tool_id"]
                or vendor_section.get("is_vendor_build") is not True
                or vendor_section.get("is_hermetic_parity_engine") is not False
                or vendor_section.get("no_python_reference_dispatch") is not True
                or vendor_section.get("finite_trace_authority_only") is not True
                or live_certificate.get("forbids_theorem_authority") is not True
                or live_certificate.get("forbids_global_correctness_claim")
                is not True
            ):
                failures.append("live_runtime_mtl_vendor_authority_policy_invalid")
        elif lane_id == "datalog_secpal":
            secpal_exception = live_certificate.get(
                "secpal_platform_exception"
            )
            secpal_exception = (
                secpal_exception
                if isinstance(secpal_exception, Mapping)
                else {}
            )
            if (
                vendor_section.get("engine_id")
                != vendor_spec["vendor_tool_id"]
                or vendor_section.get("is_vendor_build") is not True
                or vendor_section.get("is_hermetic_shadow") is not False
                or live_certificate.get(
                    "forbids_authorization_authority_on_shadows"
                )
                is not True
                or live_certificate.get("forbids_theorem_authority") is not True
                or secpal_exception.get("exception") is not True
                or secpal_exception.get("narrow_scope") is not True
                or secpal_exception.get("classification")
                != "unsupported_here"
                or any(
                    secpal_exception.get(field_name) is not False
                    for field_name in (
                        "installed",
                        "complete",
                        "authoritative",
                        "production_certified",
                    )
                )
            ):
                failures.append("live_souffle_vendor_authority_policy_invalid")

    vendor_section = live_certificate.get(vendor_spec["vendor_section"])
    vendor_section = (
        vendor_section if isinstance(vendor_section, Mapping) else {}
    )
    vendor_checks = [
        dict(check)
        for check in vendor_section.get("checks") or ()
        if isinstance(check, Mapping)
    ]
    vendor_cases = [
        dict(case)
        for case in vendor_section.get("case_results") or ()
        if isinstance(case, Mapping)
    ]
    vendor_failures = sorted(set(failures))
    vendor_valid = not vendor_failures
    eligible_tool_ids = [
        tool_id
        for tool_id in vendor_spec["expected_reference_checks"]
        if vendor_valid and not reference_failures.get(str(tool_id))
    ]
    per_tool_failures = {
        str(tool_id): sorted(
            set(reference_failures.get(str(tool_id)) or ())
            | (
                {"checked_vendor_fanin_invalid"}
                if not vendor_valid
                else set()
            )
        )
        for tool_id in vendor_spec["expected_reference_checks"]
    }
    checked_artifact = _checked_vendor_receipt_artifact(
        repo_root=repo_root,
        receipt_relative=checked_relative,
        checked_receipt=checked_receipt,
    )
    fanin: dict[str, Any] = {
        "schema_version": CHECKED_VENDOR_FANIN_SCHEMA,
        "configured": True,
        "lane_id": lane_id,
        "vendor_tool_id": vendor_spec["vendor_tool_id"],
        "vendor_authority": "differential_witness_only",
        "reference_authority_retained_by": list(
            vendor_spec["expected_reference_checks"]
        ),
        "vendor_valid": vendor_valid,
        "complete": bool(
            vendor_valid
            and len(eligible_tool_ids)
            == len(vendor_spec["expected_reference_checks"])
        ),
        "eligible_tool_ids": eligible_tool_ids,
        "failures": vendor_failures,
        "per_tool_failures": per_tool_failures,
        "reference_bindings": reference_bindings,
        "checked_install_receipt": {
            "path": checked_relative.as_posix(),
            "file_sha256": file_digest(checked_path),
            "self_digest_sha256": checked_receipt.get(
                "receipt_digest_sha256"
            ),
            "content_digest_sha256": content_digest(checked_receipt),
            "exact_live_nested_match": bool(
                live_certificate
                and live_certificate.get("install_receipt")
                == checked_receipt
            ),
        },
        "checked_install_receipt_artifact": checked_artifact,
        "live_certificate": {
            "schema_version": live_certificate.get("schema_version"),
            "interface": live_certificate.get("interface"),
            "goal_id": live_certificate.get("goal_id"),
            "task_id": live_certificate.get("task_id"),
            "repair_task_id": live_certificate.get("repair_task_id"),
            "certificate_digest_sha256": live_certificate.get(
                "certificate_digest_sha256"
            ),
            "certified": live_certificate.get("certified") is True,
            "checks_passed": sum(
                check.get("status") == "passed" for check in vendor_checks
            ),
            "checks_total": len(vendor_checks),
            "check_ids": [
                str(check.get("check_id") or "") for check in vendor_checks
            ],
            "check_set_digest_sha256": content_digest(vendor_checks),
            "case_results_total": len(vendor_cases),
            "case_result_set_digest_sha256": content_digest(vendor_cases),
            "nested_install_receipt_digest_sha256": (
                live_certificate.get("install_receipt") or {}
            ).get("receipt_digest_sha256")
            if isinstance(live_certificate.get("install_receipt"), Mapping)
            else None,
        },
        "source_module": {
            "path": module_relative.as_posix(),
            "sha256": file_digest(module_path),
        },
        "sealed_root_binding": {
            "environment": RUNTIME_MTL_SEALED_ROOT_ENV,
            "explicit": sealed_root is not None,
            "install_root_relative": Path(
                vendor_spec["install_root_relative"]
            ).as_posix(),
            "dependency_prefix_relative": (
                None
                if vendor_spec.get("dependency_prefix_relative") is None
                else Path(
                    vendor_spec["dependency_prefix_relative"]
                ).as_posix()
            ),
            "ambient_path_discovery": False,
            "skip_install": True,
            "force_install": False,
        },
        "policy": {
            "reference_and_vendor_both_required_per_target": True,
            "exact_nested_install_receipt_required": True,
            "all_live_vendor_checks_required": True,
            "vendor_never_inherits_reference_authority": True,
            "external_tool_ids_never_elevated_by_fanin": True,
        },
    }
    fanin["digest_sha256"] = content_digest(fanin)
    return fanin


def _recover_hyper_vendor_raw_check(
    adapted_check: Mapping[str, Any],
) -> dict[str, Any]:
    """Invert the narrow Hyper PNMR projection for digest verification."""

    recovered = dict(adapted_check)
    source_kind = str(recovered.pop("source_vendor_kind", "") or "")
    recovered.pop("source_check_digest_sha256", None)
    recovered.pop("tool_id", None)
    recovered["kind"] = source_kind
    return recovered


def _recorded_checked_hyper_vendor_fanin_eligibility(
    *,
    repo_root: Path,
    semantic_spec: Mapping[str, Any],
    result_fanin: Any,
    receipt_fanin: Any,
    semantic_receipt: Any = None,
) -> set[str]:
    """Reconstruct the bounded Hyper fan-in from its digest-bound adapter."""

    spec = CHECKED_HYPER_VENDOR_FANIN_SPEC
    targets = tuple(str(item) for item in spec["targets"])
    expected_targets = set(targets)
    if (
        str(semantic_spec.get("lane_id") or "") != spec["lane_id"]
        or tuple(str(item) for item in semantic_spec.get("tool_ids") or ())
        != targets
        or not isinstance(result_fanin, Mapping)
        or not isinstance(receipt_fanin, Mapping)
        or not isinstance(semantic_receipt, Mapping)
        or dict(result_fanin) != dict(receipt_fanin)
        or semantic_receipt.get("checked_vendor_fanin") != dict(result_fanin)
    ):
        return set()
    fanin = dict(result_fanin)
    if str(fanin.get("digest_sha256") or "") != content_digest(
        {key: value for key, value in fanin.items() if key != "digest_sha256"}
    ):
        return set()

    checked_relative = Path(spec["checked_receipt_relative"])
    checked_path = repo_root / checked_relative
    try:
        checked_payload = json.loads(checked_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return set()
    if not isinstance(checked_payload, Mapping):
        return set()
    checked_self_digest = content_digest(
        {
            key: value
            for key, value in checked_payload.items()
            if key != "receipt_digest_sha256"
        }
    )
    checked = fanin.get("checked_install_receipt")
    checked = checked if isinstance(checked, Mapping) else {}
    live = fanin.get("live_certificate")
    live = live if isinstance(live, Mapping) else {}
    bindings = fanin.get("reference_bindings")
    bindings = bindings if isinstance(bindings, Mapping) else {}
    per_tool_failures = fanin.get("per_tool_failures")
    per_tool_failures = (
        per_tool_failures if isinstance(per_tool_failures, Mapping) else {}
    )
    per_engine = live.get("per_engine")
    per_engine = per_engine if isinstance(per_engine, Mapping) else {}
    eligible = {str(item) for item in fanin.get("eligible_tool_ids") or ()}
    expected_checked_artifact = _checked_vendor_receipt_artifact(
        repo_root=repo_root,
        receipt_relative=checked_relative,
        checked_receipt=checked_payload,
    )
    artifact = fanin.get("checked_install_receipt_artifact")
    if (
        checked_payload.get("receipt_digest_sha256") != checked_self_digest
        or fanin.get("schema_version") != CHECKED_VENDOR_FANIN_SCHEMA
        or fanin.get("configured") is not True
        or fanin.get("lane_id") != spec["lane_id"]
        or fanin.get("vendor_tool_id") != "hyperproperty-vendor"
        or fanin.get("vendor_authority") != spec["vendor_authority"]
        or fanin.get("evidence_class") != spec["evidence_class"]
        or set(fanin.get("reference_authority_retained_by") or ())
        != expected_targets
        or fanin.get("vendor_valid") is not True
        or fanin.get("complete") is not True
        or eligible != expected_targets
        or list(fanin.get("failures") or ())
        or set(per_tool_failures) != expected_targets
        or any(list(per_tool_failures.get(tool_id) or ()) for tool_id in targets)
        or checked.get("path") != checked_relative.as_posix()
        or checked.get("file_sha256") != file_digest(checked_path)
        or checked.get("content_digest_sha256")
        != content_digest(checked_payload)
        or checked.get("self_digest_sha256") != checked_self_digest
        or checked.get("exact_live_nested_match") is not True
        or artifact != expected_checked_artifact
        or _validate_artifact_identities(
            [artifact] if isinstance(artifact, Mapping) else [],
            repo_root=repo_root,
        ).get("has_production_binding")
        is not True
        or live.get("schema_version") != spec["live_schema"]
        or live.get("interface") != spec["interface"]
        or live.get("goal_id") != spec["goal_id"]
        or live.get("task_id") != spec["task_id"]
        or live.get("repair_task_id") != spec["repair_task_id"]
        or live.get("host_platform") != observed_platform_id()
        or checked_payload.get("host_platform") != observed_platform_id()
        or live.get("certified") is not True
        or live.get("checks_passed") != spec["expected_vendor_checks"]
        or live.get("checks_total") != spec["expected_vendor_checks"]
        or len(live.get("check_ids") or ()) != spec["expected_vendor_checks"]
        or any(not str(item) for item in live.get("check_ids") or ())
        or len(set(live.get("check_ids") or ()))
        != spec["expected_vendor_checks"]
        or live.get("nested_install_receipt_digest_sha256")
        != checked_self_digest
        or set(per_engine) != expected_targets
        or fanin.get("source_module")
        != {
            "path": Path(spec["module_relative"]).as_posix(),
            "sha256": file_digest(
                repo_root / Path(spec["module_relative"])
            ),
        }
        or fanin.get("sealed_root_binding")
        != {
            "environment": RUNTIME_MTL_SEALED_ROOT_ENV,
            "explicit": True,
            "install_root_relative": ".",
            "dependency_prefix_relative": None,
            "ambient_path_discovery": False,
            "skip_install": True,
            "force_install": False,
            "platform_id": observed_platform_id(),
        }
    ):
        return set()
    policy = fanin.get("policy")
    policy = policy if isinstance(policy, Mapping) else {}
    if (
        policy.get("complete_three_engine_vendor_corpus_required") is not True
        or policy.get("exact_nested_install_receipt_required") is not True
        or policy.get("section_list_identity_required") is not True
        or policy.get("independent_reference_claimed") is not False
        or policy.get("vendor_pnmr_projection_retains_every_check") is not True
        or policy.get("bounded_authority_only") is not True
        or policy.get("never_grants_theorem_authority") is not True
        or policy.get("never_authorizes_universal_proof") is not True
    ):
        return set()

    try:
        semantic_module = _load_module_from_path(
            repo_root / Path(semantic_spec["module_relative"]),
            "fvt_recorded_checked_hyper_vendor_fanin",
        )
    except Exception:  # noqa: BLE001 — fail closed on current source mismatch
        return set()
    if (
        _validate_semantic_receipt_integrity(
            semantic_receipt,
            spec=semantic_spec,
            module=semantic_module,
        ).get("valid")
        is not True
        or semantic_receipt.get("certified") is not True
        or semantic_receipt.get("host_platform") != observed_platform_id()
        or semantic_receipt.get("authority_ceiling") != "bounded"
        or semantic_receipt.get("forbids_theorem_authority") is not True
        or semantic_receipt.get("forbids_universal_claims_beyond_bounds")
        is not True
        or semantic_receipt.get("independent_reference")
        != {
            "available": False,
            "claimed": False,
            "reason": (
                "The reviewed vendor engines are the bounded capability "
                "targets; no independent peer is relabelled or fabricated."
            ),
        }
    ):
        return set()
    receipt_policy = semantic_receipt.get("policy")
    receipt_policy = (
        receipt_policy if isinstance(receipt_policy, Mapping) else {}
    )
    if (
        receipt_policy.get("no_install") is not True
        or receipt_policy.get("no_download") is not True
        or receipt_policy.get("no_network") is not True
        or receipt_policy.get("checked_vendor_receipt_required") is not True
        or receipt_policy.get("complete_vendor_corpus_required") is not True
        or receipt_policy.get("independent_reference_claimed") is not False
        or receipt_policy.get("authority_ceiling") != "bounded"
        or receipt_policy.get("never_grants_theorem_authority") is not True
        or receipt_policy.get("never_authorizes_universal_proof") is not True
    ):
        return set()
    receipt_engines = [
        engine
        for engine in semantic_receipt.get("engines") or ()
        if isinstance(engine, Mapping)
    ]
    if (
        [str(engine.get("engine_id") or "") for engine in receipt_engines]
        != list(targets)
        or len(receipt_engines) != len(targets)
    ):
        return set()
    receipt_by_target = {
        str(engine.get("engine_id") or ""): engine for engine in receipt_engines
    }

    derived: set[str] = set()
    recovered_flattened_checks: list[dict[str, Any]] = []
    flattened_cases: list[dict[str, Any]] = []
    for tool_id in targets:
        binding = bindings.get(tool_id)
        binding = binding if isinstance(binding, Mapping) else {}
        engine = per_engine.get(tool_id)
        engine = engine if isinstance(engine, Mapping) else {}
        receipt_engine = receipt_by_target.get(tool_id)
        receipt_engine = (
            receipt_engine if isinstance(receipt_engine, Mapping) else {}
        )
        adapted_checks = [
            dict(check)
            for check in receipt_engine.get("checks") or ()
            if isinstance(check, Mapping)
        ]
        recovered_checks = [
            _recover_hyper_vendor_raw_check(check)
            for check in adapted_checks
        ]
        cases = [
            dict(case)
            for case in receipt_engine.get("case_results") or ()
            if isinstance(case, Mapping)
        ]
        normalized = _normalize_semantic_checks(tool_id, adapted_checks)
        normalized_payload = [check.to_dict() for check in normalized]
        engine_artifacts = [
            dict(artifact)
            for artifact in engine.get("artifact_identities") or ()
            if isinstance(artifact, Mapping)
        ]
        receipt_artifacts = [
            dict(artifact)
            for artifact in receipt_engine.get("artifact_identities") or ()
            if isinstance(artifact, Mapping)
        ]
        expected_artifact_kinds = (
            {
                "vendor_engine_executable",
                "launcher_runtime",
                "launcher_target",
                "runtime_dependency_abc",
                "runtime_dependency_aigtoaig",
            }
            if tool_id == "mchyper"
            else {"vendor_engine_executable"}
        )
        artifact_validation = _validate_artifact_identities(
            receipt_artifacts,
            repo_root=repo_root,
        )
        source_digests = list(
            binding.get("source_check_digests_sha256") or ()
        )
        if (
            binding.get("certified") is not True
            or binding.get("checks_passed") != spec["checks_per_target"]
            or binding.get("checks_total") != spec["checks_per_target"]
            or binding.get("expected_checks_total")
            != spec["checks_per_target"]
            or not {"positive", "negative", "mutation", "replay"}
            <= set(binding.get("check_kinds_present") or ())
            or binding.get("independent_reference_available") is not False
            or binding.get("vendor_pnmr_projection") is not True
            or binding.get("authority_ceiling") != "bounded"
            or len(source_digests) != spec["checks_per_target"]
            or any(not SHA256_RE.fullmatch(str(item)) for item in source_digests)
            or engine.get("checks_passed") != spec["checks_per_target"]
            or engine.get("checks_total") != spec["checks_per_target"]
            or engine.get("check_set_digest_sha256")
            != binding.get("source_vendor_check_set_digest_sha256")
            or receipt_engine.get("engine_id") != tool_id
            or receipt_engine.get("usable") is not True
            or receipt_engine.get("identity_probed") is not True
            or receipt_engine.get("certified") is not True
            or receipt_engine.get("role") != "authority"
            or receipt_engine.get("authority_ceiling") != "bounded"
            or receipt_engine.get("authorizes_universal_proof") is not False
            or receipt_engine.get("is_theorem_authority") is not False
            or receipt_engine.get("independent_reference_available")
            is not False
            or receipt_engine.get("authority_basis")
            != "complete_live_vendor_bounded_corpus"
            or receipt_engine.get(
                "source_vendor_certificate_digest_sha256"
            )
            != live.get("certificate_digest_sha256")
            or list(receipt_engine.get("block_reasons") or ())
            or receipt_artifacts != [*engine_artifacts, expected_checked_artifact]
            or {str(item.get("kind") or "") for item in engine_artifacts}
            != expected_artifact_kinds
            or artifact_validation.get("valid") is not True
            or artifact_validation.get("has_production_binding") is not True
            or receipt_engine.get("executable")
            != engine_artifacts[0].get("path")
            or receipt_engine.get("executable_sha256")
            != engine_artifacts[0].get("sha256")
            or receipt_engine.get("executable_artifact_class")
            != engine_artifacts[0].get("artifact_class")
            or not _digest_matches(
                engine.get("artifact_sha256"),
                str(receipt_engine.get("executable_sha256") or "").removeprefix(
                    "sha256:"
                ),
            )
            or len(adapted_checks) != spec["checks_per_target"]
            or len(normalized) != spec["checks_per_target"]
            or any(check.status != "passed" for check in normalized)
            or content_digest(normalized_payload)
            != binding.get("check_set_digest_sha256")
            or [
                str(check.get("source_check_digest_sha256") or "")
                for check in adapted_checks
            ]
            != source_digests
            or [content_digest(check) for check in recovered_checks]
            != source_digests
            or content_digest(recovered_checks)
            != binding.get("source_vendor_check_set_digest_sha256")
            or engine.get("case_results_total") != len(cases)
            or engine.get("case_result_set_digest_sha256")
            != content_digest(cases)
            or not SHA256_RE.fullmatch(
                str(engine.get("artifact_sha256") or "")
            )
        ):
            continue
        recovered_flattened_checks.extend(recovered_checks)
        flattened_cases.extend(cases)
        derived.add(tool_id)
    if derived != expected_targets:
        return set()
    receipt_summary = semantic_receipt.get("summary")
    receipt_summary = (
        receipt_summary if isinstance(receipt_summary, Mapping) else {}
    )
    if (
        sum(
            int((per_engine.get(tool_id) or {}).get("checks_total") or 0)
            for tool_id in targets
        )
        != spec["expected_vendor_checks"]
        or [str(check.get("check_id") or "") for check in recovered_flattened_checks]
        != list(live.get("check_ids") or ())
        or content_digest(recovered_flattened_checks)
        != live.get("check_set_digest_sha256")
        or len(flattened_cases) != live.get("case_results_total")
        or content_digest(flattened_cases)
        != live.get("case_result_set_digest_sha256")
        or receipt_summary.get("engines_certified") != len(targets)
        or receipt_summary.get("engines_total") != len(targets)
        or receipt_summary.get("checks_passed")
        != spec["expected_vendor_checks"]
        or receipt_summary.get("checks_total")
        != spec["expected_vendor_checks"]
        or list(receipt_summary.get("block_reasons") or ())
    ):
        return set()
    return derived


def _recorded_checked_vendor_fanin_eligibility(
    *,
    repo_root: Path,
    semantic_spec: Mapping[str, Any],
    result_fanin: Any,
    receipt_fanin: Any,
    semantic_receipt: Any = None,
) -> set[str]:
    """Fail closed unless the two digest-bound fan-in projections agree."""

    lane_id = str(semantic_spec.get("lane_id") or "")
    if lane_id == CHECKED_HYPER_VENDOR_FANIN_SPEC["lane_id"]:
        return _recorded_checked_hyper_vendor_fanin_eligibility(
            repo_root=repo_root,
            semantic_spec=semantic_spec,
            result_fanin=result_fanin,
            receipt_fanin=receipt_fanin,
            semantic_receipt=semantic_receipt,
        )
    vendor_spec = CHECKED_VENDOR_FANIN_SPECS.get(lane_id)
    if not isinstance(vendor_spec, Mapping):
        return set()
    if not isinstance(result_fanin, Mapping) or not isinstance(
        receipt_fanin, Mapping
    ):
        return set()
    fanin = dict(result_fanin)
    if fanin != dict(receipt_fanin):
        return set()
    declared_digest = str(fanin.get("digest_sha256") or "")
    if declared_digest != content_digest(
        {key: value for key, value in fanin.items() if key != "digest_sha256"}
    ):
        return set()
    expected_targets = {
        str(tool_id) for tool_id in vendor_spec["expected_reference_checks"]
    }
    eligible = {str(item) for item in fanin.get("eligible_tool_ids") or ()}
    checked = fanin.get("checked_install_receipt")
    checked = checked if isinstance(checked, Mapping) else {}
    checked_relative = Path(vendor_spec["checked_receipt_relative"])
    checked_path = repo_root / checked_relative
    try:
        checked_payload = json.loads(checked_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return set()
    live = fanin.get("live_certificate")
    live = live if isinstance(live, Mapping) else {}
    reference = fanin.get("reference_bindings")
    reference = reference if isinstance(reference, Mapping) else {}
    per_tool_failures = fanin.get("per_tool_failures")
    per_tool_failures = (
        per_tool_failures if isinstance(per_tool_failures, Mapping) else {}
    )
    expected_vendor_checks = int(vendor_spec["expected_vendor_checks"])
    if (
        fanin.get("schema_version") != CHECKED_VENDOR_FANIN_SCHEMA
        or fanin.get("lane_id") != lane_id
        or fanin.get("vendor_tool_id") != vendor_spec["vendor_tool_id"]
        or fanin.get("vendor_authority") != "differential_witness_only"
        or set(fanin.get("reference_authority_retained_by") or ())
        != expected_targets
        or fanin.get("vendor_valid") is not True
        or list(fanin.get("failures") or ())
        or not eligible <= expected_targets
        or checked.get("path") != checked_relative.as_posix()
        or checked.get("file_sha256") != file_digest(checked_path)
        or checked.get("content_digest_sha256")
        != content_digest(checked_payload)
        or checked.get("self_digest_sha256")
        != checked_payload.get("receipt_digest_sha256")
        or checked.get("exact_live_nested_match") is not True
        or live.get("interface") != vendor_spec["interface"]
        or live.get("schema_version") != vendor_spec["live_schema"]
        or live.get("goal_id") != vendor_spec["goal_id"]
        or live.get("task_id") != vendor_spec["task_id"]
        or live.get("repair_task_id") != vendor_spec["repair_task_id"]
        or live.get("certified") is not True
        or live.get("checks_passed") != expected_vendor_checks
        or live.get("checks_total") != expected_vendor_checks
        or len(live.get("check_ids") or ()) != expected_vendor_checks
        or any(not str(check_id) for check_id in live.get("check_ids") or ())
        or len(set(live.get("check_ids") or ())) != expected_vendor_checks
        or live.get("nested_install_receipt_digest_sha256")
        != checked_payload.get("receipt_digest_sha256")
        or fanin.get("source_module")
        != {
            "path": Path(vendor_spec["module_relative"]).as_posix(),
            "sha256": file_digest(
                repo_root / Path(vendor_spec["module_relative"])
            ),
        }
    ):
        return set()
    artifact = fanin.get("checked_install_receipt_artifact")
    if (
        not isinstance(artifact, Mapping)
        or _validate_artifact_identities(
            [artifact],
            repo_root=repo_root,
        ).get("has_production_binding")
        is not True
    ):
        return set()

    derived: set[str] = set()
    for tool_id in expected_targets:
        binding = reference.get(tool_id)
        binding = binding if isinstance(binding, Mapping) else {}
        expected = vendor_spec["expected_reference_checks"][tool_id]
        try:
            expected_count = int(binding.get("expected_checks_total"))
        except (TypeError, ValueError):
            expected_count = -1
        if expected != "closed_manifest":
            try:
                if expected_count != int(expected):
                    continue
            except (TypeError, ValueError):
                continue
        if (
            binding.get("certified") is True
            and binding.get("receipt_integrity_valid") is True
            and binding.get("offline_observation_satisfied") is True
            and expected_count > 0
            and binding.get("checks_passed") == expected_count
            and binding.get("checks_total") == expected_count
            and {"positive", "negative", "mutation", "replay"}
            <= set(binding.get("check_kinds_present") or ())
            and not list(per_tool_failures.get(tool_id) or ())
        ):
            derived.add(tool_id)
    if eligible != derived:
        return set()
    if bool(fanin.get("complete")) != (derived == expected_targets):
        return set()
    return derived


def build_checked_vendor_capability_readiness_projection(
    *,
    repo_root: Path,
    semantic_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Project exact vendor fan-ins into non-authoritative capability readiness.

    The projection intentionally answers only whether the managed external
    witness/shadow is installed and has completed its closed semantic suite.
    It never changes a tool certificate, never grants authority, and never
    makes the external tool eligible for production elevation.  The in-process
    reference tools retain every authority granted by the joined fan-in.
    """

    results_by_lane: dict[str, Mapping[str, Any]] = {}
    duplicate_lanes: set[str] = set()
    for raw_result in semantic_results:
        if not isinstance(raw_result, Mapping):
            continue
        lane_id = str(raw_result.get("lane_id") or "")
        if not lane_id:
            continue
        if lane_id in results_by_lane:
            duplicate_lanes.add(lane_id)
        else:
            results_by_lane[lane_id] = raw_result

    authority_role_rows = load_authority_roles(repo_root).get("tools")
    authority_role_rows = (
        authority_role_rows
        if isinstance(authority_role_rows, Mapping)
        else {}
    )
    tools: dict[str, dict[str, Any]] = {}
    for lane_id, raw_vendor_spec in CHECKED_VENDOR_FANIN_SPECS.items():
        vendor_spec = dict(raw_vendor_spec)
        external_tool_id = str(
            vendor_spec.get("managed_readiness_tool_id") or ""
        )
        if not external_tool_id:
            continue
        declared_role = authority_role_rows.get(external_tool_id)
        declared_role = (
            declared_role if isinstance(declared_role, Mapping) else {}
        )
        spec = next(
            (
                item
                for item in SEMANTIC_CERTIFIER_SPECS
                if str(item.get("lane_id") or "") == lane_id
            ),
            None,
        )
        result = results_by_lane.get(lane_id)
        failures: list[str] = []
        if lane_id in duplicate_lanes:
            failures.append("semantic_lane_population_not_unique")
        if not isinstance(spec, Mapping):
            failures.append("semantic_spec_missing")
            spec = {}
        if not isinstance(result, Mapping):
            failures.append("semantic_lane_result_missing")
            result = {}
        if result.get("status") != "ran":
            failures.append("semantic_lane_not_run")
        expected_role_binding = {
            "role": vendor_spec["declared_authority_role"],
            "authority_ceiling": vendor_spec[
                "declared_authority_ceiling"
            ],
            "can_satisfy_certified_authority": vendor_spec[
                "declared_role_can_satisfy_certified_authority"
            ],
        }
        if any(
            declared_role.get(field_name) != expected
            for field_name, expected in expected_role_binding.items()
        ):
            failures.append("declared_authority_role_binding_invalid")

        fanin = result.get("checked_vendor_fanin")
        fanin = fanin if isinstance(fanin, Mapping) else {}
        receipt = result.get("receipt")
        receipt = receipt if isinstance(receipt, Mapping) else {}
        receipt_fanin = receipt.get("checked_vendor_fanin")
        receipt_fanin = (
            receipt_fanin if isinstance(receipt_fanin, Mapping) else {}
        )
        expected_reference_ids = {
            str(tool_id)
            for tool_id in vendor_spec["expected_reference_checks"]
        }
        eligible_reference_ids = _recorded_checked_vendor_fanin_eligibility(
            repo_root=repo_root,
            semantic_spec=spec,
            result_fanin=fanin,
            receipt_fanin=receipt_fanin,
        )
        if eligible_reference_ids != expected_reference_ids:
            failures.append("checked_vendor_fanin_not_complete")

        live = fanin.get("live_certificate")
        live = live if isinstance(live, Mapping) else {}
        expected_checks = int(vendor_spec["expected_vendor_checks"])
        try:
            observed_checks_passed = int(live.get("checks_passed"))
            observed_checks_total = int(live.get("checks_total"))
        except (TypeError, ValueError):
            observed_checks_passed = -1
            observed_checks_total = -1
        check_ids = [
            str(check_id) for check_id in live.get("check_ids") or ()
        ]
        if (
            fanin.get("vendor_tool_id") != external_tool_id
            or fanin.get("vendor_authority")
            != "differential_witness_only"
            or fanin.get("vendor_valid") is not True
            or fanin.get("complete") is not True
            or list(fanin.get("failures") or ())
            or live.get("certified") is not True
            or observed_checks_passed != expected_checks
            or observed_checks_total != expected_checks
            or len(check_ids) != expected_checks
            or len(set(check_ids)) != expected_checks
            or any(not check_id for check_id in check_ids)
        ):
            failures.append("checked_vendor_closed_check_set_invalid")

        fanin_policy = fanin.get("policy")
        fanin_policy = (
            fanin_policy if isinstance(fanin_policy, Mapping) else {}
        )
        if any(
            fanin_policy.get(key) is not True
            for key in (
                "reference_and_vendor_both_required_per_target",
                "exact_nested_install_receipt_required",
                "all_live_vendor_checks_required",
                "vendor_never_inherits_reference_authority",
                "external_tool_ids_never_elevated_by_fanin",
            )
        ):
            failures.append("checked_vendor_authority_policy_invalid")

        sealed_binding = fanin.get("sealed_root_binding")
        sealed_binding = (
            sealed_binding
            if isinstance(sealed_binding, Mapping)
            else {}
        )
        if (
            sealed_binding.get("explicit") is not True
            or sealed_binding.get("ambient_path_discovery") is not False
            or sealed_binding.get("skip_install") is not True
            or sealed_binding.get("force_install") is not False
        ):
            failures.append("checked_vendor_sealed_root_binding_invalid")

        artifact = fanin.get("checked_install_receipt_artifact")
        artifact = dict(artifact) if isinstance(artifact, Mapping) else {}
        artifact_validation = _validate_artifact_identities(
            [artifact] if artifact else [],
            repo_root=repo_root,
        )
        if (
            artifact_validation.get("valid") is not True
            or artifact_validation.get("has_production_binding") is not True
            or artifact.get("kind") != "checked_vendor_install_receipt"
            or artifact.get("artifact_class") != "public_deployment_binding"
        ):
            failures.append("checked_vendor_install_artifact_invalid")

        checked = fanin.get("checked_install_receipt")
        checked = checked if isinstance(checked, Mapping) else {}
        if checked.get("exact_live_nested_match") is not True:
            failures.append("checked_vendor_live_nested_receipt_mismatch")

        ready = not failures
        entry: dict[str, Any] = {
            "tool_id": external_tool_id,
            "lane_id": lane_id,
            "role": str(vendor_spec["managed_readiness_role"]),
            "readiness_scope": str(
                vendor_spec["managed_readiness_scope"]
            ),
            "evidence_class": str(
                vendor_spec["managed_readiness_evidence_class"]
            ),
            "declared_authority_role": declared_role.get("role"),
            "declared_authority_ceiling": declared_role.get(
                "authority_ceiling"
            ),
            "declared_role_can_satisfy_certified_authority": (
                declared_role.get(
                    "can_satisfy_certified_authority"
                )
            ),
            "authority_requirement_satisfied": False,
            "ready": ready,
            "installation_ready": ready,
            "semantic_certification_ready": ready,
            "vendor_checks_passed": observed_checks_passed,
            "vendor_checks_total": observed_checks_total,
            "expected_vendor_checks": expected_checks,
            "checked_vendor_fanin_digest_sha256": fanin.get(
                "digest_sha256"
            ),
            "checked_install_receipt_artifact": artifact,
            "failures": sorted(set(failures)),
            "production_certified": False,
            "production_elevation_allowed": False,
            "authority_granted": False,
            "grants_theorem_authority": False,
            "grants_global_correctness": False,
            "grants_authorization_decision_authority": False,
            "reference_authority_retained_by": sorted(
                expected_reference_ids
            ),
        }
        entry["digest_sha256"] = content_digest(entry)
        tools[external_tool_id] = entry

    projection: dict[str, Any] = {
        "schema_version": CHECKED_VENDOR_CAPABILITY_READINESS_SCHEMA,
        "policy": {
            "installation_and_semantic_readiness_is_not_authority": True,
            "external_witnesses_and_shadows_never_production_elevated": True,
            "external_witnesses_and_shadows_never_grant_theorem_authority": True,
            "external_witnesses_and_shadows_never_grant_global_correctness": True,
            "external_shadows_never_grant_authorization_authority": True,
            "exact_checked_vendor_fanin_required": True,
            "external_secpal_platform_exception_not_counted_complete": True,
        },
        "tools": tools,
    }
    projection["digest_sha256"] = content_digest(projection)
    return projection


def _offline_observation(
    receipt: Mapping[str, Any],
    *,
    production_elevation_allowed: bool,
) -> dict[str, Any]:
    """Derive no-install/network claims from each focused receipt."""

    install_attempted = bool(receipt.get("install_attempted"))
    download_attempted = bool(receipt.get("download_attempted"))
    network_used = bool(receipt.get("network_used"))
    install_section = receipt.get("install")
    if isinstance(install_section, Mapping):
        install_attempted = install_attempted or bool(
            install_section.get("attempted")
            or install_section.get("installed")
            or install_section.get("downloaded")
        )
        download_attempted = download_attempted or bool(
            install_section.get("downloaded")
        )
        network_used = network_used or bool(install_section.get("network_used"))
    policy = receipt.get("policy")
    policy = policy if isinstance(policy, Mapping) else {}
    explicit_offline_policy = all(
        policy.get(key) is True for key in ("no_install", "no_download", "no_network")
    )
    in_process_only = policy.get("in_process_only") is True
    declaration_sufficient = bool(
        explicit_offline_policy
        or in_process_only
        or not production_elevation_allowed
    )
    return {
        "install_attempted": install_attempted,
        "download_attempted": download_attempted,
        "network_used": network_used,
        "explicit_offline_policy": explicit_offline_policy,
        "in_process_only": in_process_only,
        "declaration_sufficient": declaration_sufficient,
        "satisfied": declaration_sufficient
        and not (install_attempted or download_attempted or network_used),
    }


_LIVE_SEMANTIC_KIND_ALIASES: Final[Mapping[str, frozenset[str]]] = {
    "positive": frozenset({"positive", "invariant_holds", "secure", "theorem"}),
    "negative": frozenset(
        {
            "negative",
            "violation_trace",
            "attack",
            "counter_satisfiable",
            "corrupted_proof",
            "corrupted_key",
            "corrupted_public_input",
            "circuit_mismatch",
            "revoked",
            "stale",
        }
    ),
    "mutation": frozenset({"mutation"}),
    "replay": frozenset({"replay"}),
}


def _live_receipt_surface_names(receipt: Mapping[str, Any]) -> set[str]:
    """Collect declared certifier surfaces from a specialized live receipt."""

    surfaces: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if key in {"certification_surface", "owner_module"} and isinstance(
                    item, str
                ):
                    surfaces.add(item)
                elif key == "surfaces" and isinstance(item, Sequence) and not isinstance(
                    item, (str, bytes, bytearray)
                ):
                    surfaces.update(str(entry) for entry in item)
                visit(item)
        elif isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            for item in value:
                visit(item)

    visit(receipt)
    return surfaces


def _live_tool_payload(
    receipt: Mapping[str, Any],
    *,
    family: str,
    tool_id: str,
) -> Mapping[str, Any]:
    if family == "kernel":
        kernel_id = "rocq" if tool_id == "coq" else tool_id
        kernels = receipt.get("kernels")
        return (
            kernels.get(kernel_id, {})
            if isinstance(kernels, Mapping)
            else {}
        )
    if family == "protocol":
        tools = receipt.get("tools")
        return tools.get(tool_id, {}) if isinstance(tools, Mapping) else {}
    return receipt


def _live_tool_checks(
    receipt: Mapping[str, Any],
    tool_payload: Mapping[str, Any],
    *,
    family: str,
    tool_id: str,
) -> list[Mapping[str, Any]]:
    raw = tool_payload.get("checks")
    checks = [
        item
        for item in (
            raw
            if isinstance(raw, Sequence)
            and not isinstance(raw, (str, bytes, bytearray))
            else ()
        )
        if isinstance(item, Mapping)
    ]
    if family not in {"state_model", "atp"}:
        return checks

    # State/ATP receipts retain both engines in a single top-level check set.
    # Keep shared policy/binding rows and only this tool's semantic rows.
    selected: list[Mapping[str, Any]] = []
    other_ids = (
        {"apalache", "tlc"} - {tool_id}
        if family == "state_model"
        else {"vampire", "eprover"} - {tool_id}
    )
    for check in checks:
        check_id = str(check.get("check_id") or "").lower()
        mentions_other = any(
            re.search(rf"(^|[._:-]){re.escape(other)}([._:-]|$)", check_id)
            for other in other_ids
        )
        mentions_tool = bool(
            re.search(
                rf"(^|[._:-]){re.escape(tool_id)}([._:-]|$)",
                check_id,
            )
        )
        if mentions_tool or not mentions_other:
            selected.append(check)
    return selected


def _live_receipt_digest_validation(
    receipt: Mapping[str, Any],
) -> tuple[bool, str | None, list[str]]:
    fields = [
        field_name
        for field_name in (
            "receipt_digest_sha256",
            "certificate_digest_sha256",
            "digest_sha256",
        )
        if field_name in receipt
    ]
    failures: list[str] = []
    if len(fields) != 1:
        failures.append("live_receipt_digest_field_population_invalid")
        return False, None, failures
    field_name = fields[0]
    computed = content_digest(
        {key: value for key, value in receipt.items() if key != field_name}
    )
    if not _digest_matches(receipt.get(field_name), computed):
        failures.append("live_receipt_self_digest_mismatch")
    return not failures, computed, failures


def _live_receipt_version_string(
    receipt: Mapping[str, Any],
    tool_payload: Mapping[str, Any],
    *,
    family: str,
    tool_id: str,
) -> str:
    if family == "kernel":
        return str(tool_payload.get("version_string") or "")
    if family == "protocol":
        return str(
            tool_payload.get(f"{tool_id}_version_string")
            or tool_payload.get("version_string")
            or ""
        )
    return str(receipt.get(f"{tool_id.replace('-', '_')}_version_string") or "")


def _live_receipt_binary_digest(
    receipt: Mapping[str, Any],
    tool_payload: Mapping[str, Any],
    *,
    tool_id: str,
) -> str:
    field_name = f"{tool_id.replace('-', '_')}_binary_digest"
    return str(
        tool_payload.get(field_name)
        or receipt.get(field_name)
        or ""
    ).removeprefix("sha256:")


def _live_tool_claims_production(
    receipt: Mapping[str, Any],
    tool_payload: Mapping[str, Any],
    *,
    family: str,
    tool_id: str,
) -> bool:
    if not (
        receipt.get("production_certified") is True
        and receipt.get("promotion_blocked") is not True
        and not list(receipt.get("block_reasons") or [])
    ):
        return False
    if family == "kernel":
        return bool(
            receipt.get("all_kernels_passed") is True
            and tool_payload.get("fanin_passed") is True
            and tool_payload.get("live_executed") is True
            and tool_payload.get("identity_probed") is True
            and tool_payload.get("usable") is True
            and not list(tool_payload.get("block_reasons") or [])
        )
    if family == "protocol":
        usable_field = f"{tool_id}_usable"
        return bool(
            receipt.get("live_execution") is True
            and receipt.get("live_semantic_certified") is True
            and tool_payload.get("production_certified") is True
            and tool_payload.get("live_execution") is True
            and tool_payload.get("live_semantic_certified") is True
            and tool_payload.get(usable_field) is True
            and not list(tool_payload.get("block_reasons") or [])
        )
    if family == "state_model":
        prefix = tool_id.replace("-", "_")
        return bool(
            receipt.get("live_execution") is True
            and receipt.get("live_semantic_corpus_passed") is True
            and receipt.get(f"{prefix}_identity_probed") is True
            and receipt.get(f"{prefix}_usable") is True
        )
    if family == "atp":
        prefix = tool_id.replace("-", "_")
        return bool(
            receipt.get("live_execution") is True
            and receipt.get("live_semantic_corpus_passed") is True
            and receipt.get(f"{prefix}_identity_probed") is True
            and receipt.get(f"{prefix}_usable") is True
        )
    if family == "zkp":
        return bool(
            receipt.get("certified") is True
            and receipt.get("live_execution") is True
            and receipt.get("live_verifier_executed") is True
            and receipt.get("live_corpus_passed") is True
        )
    return False


def _specialized_artifact_binding(
    *,
    repo_root: Path,
    cert: ToolCertification,
    receipt: Mapping[str, Any],
    tool_payload: Mapping[str, Any],
    family: str,
    tool_id: str,
) -> tuple[bool, list[dict[str, Any]], list[str]]:
    """Recompute the current tool/deployment artifacts used by a live receipt."""

    failures: list[str] = []
    artifacts = [
        dict(item)
        for item in cert.artifact_identities
        if isinstance(item, Mapping)
        and (
            item.get("kind") != "executable"
            or cert.executable_artifact_class == "native_or_managed_binary"
        )
    ]

    if family == "zkp":
        lock_path = _resolve_artifact_path(
            receipt.get("lock_path"),
            repo_root=repo_root,
        )
        try:
            lock_payload = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path else None
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            lock_payload = None
        declared = str(receipt.get("lock_digest") or "")
        if (
            lock_path is None
            or lock_payload is None
            or not _digest_matches(declared, content_digest(lock_payload))
        ):
            failures.append("live_zkp_deployment_lock_binding_invalid")
        else:
            artifacts.append(
                {
                    "kind": "deployment_artifact",
                    "path": str(lock_path),
                    "sha256": file_digest(lock_path),
                    "declared_digest": declared,
                    "artifact_class": "public_deployment_binding",
                }
            )
    else:
        current_validation = _validate_artifact_identities(
            [
                item
                for item in cert.artifact_identities
                if isinstance(item, Mapping)
            ],
            repo_root=repo_root,
        )
        if (
            cert.usable is not True
            or cert.identity_probed is not True
            or cert.locked_version_mismatch
            or current_validation.get("valid") is not True
            or current_validation.get("has_production_binding") is not True
        ):
            failures.append("current_tool_artifact_binding_invalid")
        if (
            cert.executable_artifact_class == "launcher_script"
            and cert.launcher_binding.get("valid") is not True
        ):
            failures.append("current_launcher_target_binding_invalid")

        declared_binary = _live_receipt_binary_digest(
            receipt,
            tool_payload,
            tool_id=tool_id,
        )
        current_launcher = str(cert.executable_sha256 or "").removeprefix(
            "sha256:"
        )
        if declared_binary and declared_binary != current_launcher:
            failures.append("live_receipt_launcher_digest_mismatch")

        declared_version = _live_receipt_version_string(
            receipt,
            tool_payload,
            family=family,
            tool_id=tool_id,
        )
        locked_field = (
            tool_payload.get(f"locked_{tool_id.replace('-', '_')}_version")
            or receipt.get(f"locked_{tool_id.replace('-', '_')}_version")
        )
        if cert.locked_version and (
            (
                not declared_version
                or detect_locked_version_mismatch(
                    cert.locked_version,
                    declared_version,
                )
            )
            and str(locked_field or "") != cert.locked_version
        ):
            failures.append("live_receipt_version_mismatch")

    validation = _validate_artifact_identities(
        artifacts,
        repo_root=repo_root,
    )
    if validation.get("valid") is not True:
        failures.extend(
            str(item) for item in validation.get("failures") or []
        )
    if validation.get("has_production_binding") is not True:
        failures.append("live_receipt_production_artifact_binding_missing")
    return not failures, artifacts, sorted(set(failures))


def _build_live_specialized_adapter(
    *,
    repo_root: Path,
    spec: Mapping[str, Any],
    module: Any,
    tool_certs: Mapping[str, ToolCertification],
) -> dict[str, Any]:
    """Validate and adapt one durable live receipt to the lane interface."""

    lane_id = str(spec.get("lane_id") or "")
    live_spec = LIVE_SPECIALIZED_RECEIPT_SPECS.get(lane_id)
    if not isinstance(live_spec, Mapping):
        return {
            "available": False,
            "valid": False,
            "eligible_tool_ids": [],
            "failures": ["live_specialized_receipt_not_configured"],
        }
    receipt_relative = Path(live_spec["path"])
    receipt_path = repo_root / receipt_relative
    if not receipt_path.is_file():
        return {
            "available": False,
            "valid": False,
            "path": receipt_relative.as_posix(),
            "eligible_tool_ids": [],
            "failures": ["live_specialized_receipt_missing"],
        }
    try:
        live_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {
            "available": True,
            "valid": False,
            "path": receipt_relative.as_posix(),
            "eligible_tool_ids": [],
            "failures": ["live_specialized_receipt_unreadable"],
        }
    if not isinstance(live_receipt, Mapping):
        return {
            "available": True,
            "valid": False,
            "path": receipt_relative.as_posix(),
            "eligible_tool_ids": [],
            "failures": ["live_specialized_receipt_not_mapping"],
        }

    failures: list[str] = []
    for field_name in ("schema_version", "interface", "goal_id", "task_id"):
        if str(live_receipt.get(field_name) or "") != str(
            live_spec.get(field_name) or ""
        ):
            failures.append(f"live_receipt_{field_name}_mismatch")
    digest_valid, self_digest, digest_failures = (
        _live_receipt_digest_validation(live_receipt)
    )
    failures.extend(digest_failures)
    public_audit = public_evidence_audit(
        live_receipt,
        repo_root=repo_root,
    )
    if public_audit.get("satisfied") is not True:
        failures.append("live_receipt_public_evidence_invalid")

    source_artifacts: list[dict[str, Any]] = []
    declared_surfaces = _live_receipt_surface_names(live_receipt)
    for source_relative in live_spec.get("source_modules") or ():
        source_relative = Path(source_relative)
        source_path = repo_root / source_relative
        source_digest = file_digest(source_path)
        dotted = source_relative.with_suffix("").as_posix().replace("/", ".")
        if not source_digest:
            failures.append(f"live_source_missing:{source_relative.as_posix()}")
            continue
        if dotted not in declared_surfaces:
            failures.append(
                f"live_source_surface_unbound:{source_relative.as_posix()}"
            )
        source_artifacts.append(
            {
                "kind": "live_semantic_certifier_module",
                "path": source_relative.as_posix(),
                "sha256": source_digest,
                "artifact_class": "repository_source",
            }
        )
    source_artifacts.append(
        {
            "kind": "live_specialized_receipt",
            "path": receipt_relative.as_posix(),
            "sha256": file_digest(receipt_path),
            "artifact_class": "repository_source",
        }
    )
    source_validation = _validate_artifact_identities(
        source_artifacts,
        repo_root=repo_root,
    )
    if source_validation.get("valid") is not True:
        failures.extend(
            f"live_source:{item}"
            for item in source_validation.get("failures") or []
        )

    family = str(live_spec.get("family") or "")
    per_tool_checks: dict[str, list[dict[str, Any]]] = {}
    per_tool_artifacts: dict[str, list[dict[str, Any]]] = {}
    eligible_tool_ids: list[str] = []
    per_tool_failures: dict[str, list[str]] = {}
    for raw_tool_id in spec.get("tool_ids") or ():
        tool_id = str(raw_tool_id)
        tool_failures: list[str] = []
        cert = tool_certs.get(tool_id)
        tool_payload = _live_tool_payload(
            live_receipt,
            family=family,
            tool_id=tool_id,
        )
        nested_digest_field = (
            "contribution_digest_sha256"
            if family == "kernel"
            else "receipt_digest_sha256"
            if family == "protocol"
            else None
        )
        if nested_digest_field:
            nested_computed = content_digest(
                {
                    key: value
                    for key, value in tool_payload.items()
                    if key != nested_digest_field
                }
            )
            if not _digest_matches(
                tool_payload.get(nested_digest_field),
                nested_computed,
            ):
                tool_failures.append(
                    "live_tool_nested_receipt_digest_mismatch"
                )
        checks = _live_tool_checks(
            live_receipt,
            tool_payload,
            family=family,
            tool_id=tool_id,
        )
        if not _live_tool_claims_production(
            live_receipt,
            tool_payload,
            family=family,
            tool_id=tool_id,
        ):
            tool_failures.append("live_tool_production_claim_invalid")
        if not checks or any(
            str(check.get("status") or "") != "passed" for check in checks
        ):
            tool_failures.append("live_tool_checks_incomplete_or_failed")

        canonical_sources: dict[str, Mapping[str, Any]] = {}
        for canonical_kind, aliases in _LIVE_SEMANTIC_KIND_ALIASES.items():
            source_check = next(
                (
                    check
                    for check in checks
                    if str(check.get("status") or "") == "passed"
                    and str(check.get("kind") or "").lower() in aliases
                ),
                None,
            )
            if source_check is None:
                tool_failures.append(
                    f"live_semantic_kind_missing:{canonical_kind}"
                )
            else:
                canonical_sources[canonical_kind] = source_check

        bound = False
        artifacts: list[dict[str, Any]] = []
        if cert is None:
            tool_failures.append("tool_certificate_missing")
        else:
            bound, artifacts, artifact_failures = _specialized_artifact_binding(
                repo_root=repo_root,
                cert=cert,
                receipt=live_receipt,
                tool_payload=tool_payload,
                family=family,
                tool_id=tool_id,
            )
            tool_failures.extend(artifact_failures)
        artifacts.extend(dict(item) for item in source_artifacts)
        per_tool_artifacts[tool_id] = artifacts

        adapted_checks: list[dict[str, Any]] = []
        for index, check in enumerate(checks):
            source_check = dict(check)
            adapted_checks.append(
                {
                    "check_id": str(
                        source_check.get("check_id")
                        or f"{tool_id}.live_specialized.{index}"
                    ),
                    "tool_id": tool_id,
                    "kind": str(
                        source_check.get("kind") or "unclassified"
                    ).lower(),
                    "status": str(
                        source_check.get("status") or "failed"
                    ).lower(),
                    "expected": str(
                        source_check.get("expected") or "semantic_pass"
                    ),
                    "observed": str(
                        source_check.get("observed")
                        or source_check.get("status")
                        or "failed"
                    ),
                    "detail": str(
                        source_check.get("detail")
                        or "live specialized semantic receipt"
                    ),
                    "source_check_digest_sha256": content_digest(source_check),
                    "source_live_receipt_digest_sha256": self_digest,
                }
            )
        for canonical_kind, source_check in canonical_sources.items():
            adapted_checks.append(
                {
                    "check_id": (
                        f"{tool_id}.live_specialized.canonical.{canonical_kind}"
                    ),
                    "tool_id": tool_id,
                    "kind": canonical_kind,
                    "status": "passed",
                    "expected": "passed live semantic evidence",
                    "observed": str(source_check.get("kind") or canonical_kind),
                    "detail": (
                        "Canonical PNMR projection of a passed family-specific "
                        "live semantic check."
                    ),
                    "source_check_id": source_check.get("check_id"),
                    "source_check_digest_sha256": content_digest(source_check),
                    "source_live_receipt_digest_sha256": self_digest,
                }
            )
        binding_ok = bool(bound and not tool_failures and not failures)
        adapted_checks.append(
            {
                "check_id": f"{tool_id}.live_specialized.current_binding",
                "tool_id": tool_id,
                "kind": "binding",
                "status": "passed" if binding_ok else "failed",
                "expected": "current launcher/target and source hashes bound",
                "observed": "bound" if binding_ok else "unbound",
                "detail": ";".join(sorted(set(tool_failures))) or "bound",
                "source_live_receipt_digest_sha256": self_digest,
                "current_artifact_set_digest_sha256": content_digest(artifacts),
            }
        )
        per_tool_checks[tool_id] = adapted_checks
        per_tool_failures[tool_id] = sorted(set(tool_failures))
        if binding_ok:
            eligible_tool_ids.append(tool_id)

    adapter_checks = [
        check
        for tool_id in (str(item) for item in spec.get("tool_ids") or ())
        for check in per_tool_checks.get(tool_id, [])
    ]
    adapter: dict[str, Any] = {
        "schema_version": str(getattr(module, "SCHEMA_VERSION", "") or ""),
        "interface": str(spec["interface"]),
        "goal_id": str(getattr(module, "GOAL_ID", "") or ""),
        "task_id": str(getattr(module, "TASK_ID", "") or ""),
        str(spec["certified_key"]): bool(digest_valid and not failures),
        "certified": bool(digest_valid and not failures),
        "production_certified": bool(digest_valid and not failures),
        "promotion_blocked": bool(failures),
        "block_reasons": list(failures),
        "checks": adapter_checks,
        "policy": {
            "no_install": True,
            "no_download": True,
            "no_network": True,
            "live_specialized_receipt_required": True,
            "launcher_target_binding_required": True,
            "source_hash_binding_required": True,
        },
        "install_attempted": False,
        "download_attempted": False,
        "network_used": False,
        "live_specialized_receipt": {
            "path": receipt_relative.as_posix(),
            "file_sha256": file_digest(receipt_path),
            "self_digest_sha256": self_digest,
            "interface": live_receipt.get("interface"),
            "schema_version": live_receipt.get("schema_version"),
            "goal_id": live_receipt.get("goal_id"),
            "task_id": live_receipt.get("task_id"),
            "source_set_digest_sha256": content_digest(source_artifacts),
            "eligible_tool_ids": eligible_tool_ids,
        },
    }
    for raw_tool_id in spec.get("tool_ids") or ():
        tool_id = str(raw_tool_id)
        cert = tool_certs.get(tool_id)
        prefix = tool_id.replace("-", "_")
        adapter[f"{prefix}_executable"] = (
            cert.executable_path if cert is not None else None
        )
        adapter[f"{prefix}_version_string"] = (
            cert.version_string if cert is not None else None
        )
        adapter[f"{prefix}_identity_probed"] = bool(
            cert is not None and cert.identity_probed
        )
        adapter[f"{prefix}_artifact_identities"] = per_tool_artifacts.get(
            tool_id,
            [],
        )
    if family == "zkp":
        adapter["lock_path"] = live_receipt.get("lock_path")
        adapter["lock_digest"] = live_receipt.get("lock_digest")
        adapter["identity_probed"] = "zkp-circuit" in eligible_tool_ids
    adapter["receipt_digest_sha256"] = content_digest(adapter)

    return {
        "available": True,
        "valid": bool(digest_valid and not failures),
        "path": receipt_relative.as_posix(),
        "file_sha256": file_digest(receipt_path),
        "self_digest_sha256": self_digest,
        "source_artifacts": source_artifacts,
        "source_set_digest_sha256": content_digest(source_artifacts),
        "eligible_tool_ids": eligible_tool_ids,
        "per_tool_failures": per_tool_failures,
        "failures": sorted(set(failures)),
        "adapter_receipt": adapter,
    }


def run_semantic_lane_certifiers(
    *,
    repo_root: Path,
    env: Mapping[str, str],
    tool_certs: Mapping[str, ToolCertification],
) -> list[dict[str, Any]]:
    """Invoke focused semantic certifiers offline; never install or fetch."""

    prebuilt = _runtime_mtl_managed_prebuilt_binding(repo_root, env=env)
    return _run_semantic_lane_certifiers_with_prebuilt(
        repo_root=repo_root,
        env=env,
        tool_certs=tool_certs,
        runtime_mtl_prebuilt_bind=prebuilt["public"],
        runtime_mtl_prebuilt_invocation=prebuilt.get("invocation"),
    )


def _run_semantic_lane_certifiers_with_prebuilt(
    *,
    repo_root: Path,
    env: Mapping[str, str],
    tool_certs: Mapping[str, ToolCertification],
    runtime_mtl_prebuilt_bind: Mapping[str, Any],
    runtime_mtl_prebuilt_invocation: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Inner semantic certifier loop after read-only prebuilt authentication."""

    results: list[dict[str, Any]] = []
    sealed_vendor_root = None
    if (
        runtime_mtl_prebuilt_bind.get("authenticated") is True
        and runtime_mtl_prebuilt_bind.get("bound") is True
        and isinstance(runtime_mtl_prebuilt_invocation, Mapping)
        and runtime_mtl_prebuilt_invocation.get("sealed_root")
    ):
        sealed_vendor_root = Path(
            str(runtime_mtl_prebuilt_invocation["sealed_root"])
        )
    for spec in SEMANTIC_CERTIFIER_SPECS:
        module_path = repo_root / Path(spec["module_relative"])
        lane_id = str(spec["lane_id"])
        tool_ids = tuple(spec["tool_ids"])
        entry: dict[str, Any] = {
            "lane_id": lane_id,
            "property_lane_id": str(spec.get("property_lane_id") or lane_id),
            "certifier_family": str(spec.get("certifier_family") or lane_id),
            "interface": str(spec["interface"]),
            "module": Path(spec["module_relative"]).as_posix(),
            "tool_ids": list(tool_ids),
            "status": "not_run",
            "certified": False,
            "semantically_usable_tool_ids": [],
            "elevated_tool_ids": [],
            "block_reasons": [],
            "digest_sha256": None,
            "notes": "",
            "evidence_class": str(spec["evidence_class"]),
            "production_elevation_allowed": bool(
                spec["production_elevation_allowed"]
            ),
            "usable_elevation_allowed": bool(
                spec.get("usable_elevation_allowed", True)
            ),
            "certifier_module_sha256": file_digest(module_path),
        }
        if lane_id == "runtime_mtl":
            entry["managed_typescript_prebuilt_bind"] = dict(
                runtime_mtl_prebuilt_bind
            )

        if not module_path.is_file():
            entry["status"] = "certifier_missing"
            entry["block_reasons"] = ["semantic_certifier_module_missing"]
            entry["notes"] = f"Missing certifier module: {module_path.as_posix()}"
            results.append(entry)
            continue

        try:
            module = _load_module_from_path(
                module_path,
                f"fvt_role_aware_{lane_id.replace('-', '_')}",
            )
            callable_name = str(spec["callable_name"])
            certifier = getattr(module, callable_name, None)
            if not callable(certifier):
                raise AttributeError(f"{callable_name} not callable on {module_path}")

            precomputed_vendor_fanin: Mapping[str, Any] | None = None
            if lane_id == CHECKED_HYPER_VENDOR_FANIN_SPEC["lane_id"]:
                hyper_adapter = _build_checked_hyper_vendor_adapter(
                    repo_root=repo_root,
                    sealed_root=sealed_vendor_root,
                    semantic_spec=spec,
                    semantic_module=module,
                )
                receipt = hyper_adapter["adapter_receipt"]
                precomputed_vendor_fanin = hyper_adapter["fanin"]
            else:
                semantic_extra_kwargs: dict[str, Any] = {}
                if lane_id == "runtime_mtl" and runtime_mtl_prebuilt_invocation:
                    semantic_extra_kwargs = {
                        "typescript_prebuilt_root": (
                            runtime_mtl_prebuilt_invocation["sealed_root"]
                        ),
                        "typescript_prebuilt_timeout_seconds": (
                            runtime_mtl_prebuilt_invocation["timeout_seconds"]
                        ),
                    }
                receipt = _invoke_semantic_certifier(
                    certifier,
                    repo_root=repo_root,
                    env=env,
                    extra_kwargs=semantic_extra_kwargs,
                )
        except Exception as exc:  # noqa: BLE001 — fail closed per lane
            entry["status"] = "certifier_error"
            entry["block_reasons"] = [f"{type(exc).__name__}:{exc}"]
            entry["notes"] = (
                "Semantic certifier failed closed; tools stay non-elevated."
            )
            results.append(entry)
            continue

        live_specialized = _build_live_specialized_adapter(
            repo_root=repo_root,
            spec=spec,
            module=module,
            tool_certs=tool_certs,
        )
        adapter_receipt = live_specialized.get("adapter_receipt")
        if (
            live_specialized.get("valid") is True
            and isinstance(adapter_receipt, Mapping)
        ):
            receipt = adapter_receipt
        eligible_live_tool_ids = {
            str(item)
            for item in live_specialized.get("eligible_tool_ids") or ()
        }
        checked_vendor_fanin = (
            dict(precomputed_vendor_fanin)
            if isinstance(precomputed_vendor_fanin, Mapping)
            else _build_checked_vendor_fanin(
                repo_root=repo_root,
                sealed_root=sealed_vendor_root,
                semantic_spec=spec,
                semantic_module=module,
                reference_receipt=receipt,
            )
        )
        configured_vendor_fanin = (
            checked_vendor_fanin.get("configured") is True
        )
        eligible_vendor_tool_ids = {
            str(item)
            for item in checked_vendor_fanin.get("eligible_tool_ids") or ()
        }
        if configured_vendor_fanin:
            receipt = _bind_checked_vendor_fanin_to_receipt(
                receipt,
                semantic_spec=spec,
                fanin=checked_vendor_fanin,
            )
        effective_production_allowed = bool(
            spec["production_elevation_allowed"]
            or eligible_live_tool_ids
            or eligible_vendor_tool_ids
        )
        entry["status"] = "ran"
        entry["receipt"] = dict(receipt)
        entry["digest_sha256"] = content_digest(receipt)
        entry["production_elevation_allowed"] = effective_production_allowed
        entry["live_specialized_receipt"] = {
            key: value
            for key, value in live_specialized.items()
            if key != "adapter_receipt"
        }
        if configured_vendor_fanin:
            entry["checked_vendor_fanin"] = dict(checked_vendor_fanin)
        if eligible_live_tool_ids:
            entry["evidence_class"] = "live_specialized_semantic_receipt"
        elif eligible_vendor_tool_ids:
            differential_spec = CHECKED_VENDOR_FANIN_SPECS.get(lane_id, {})
            entry["evidence_class"] = str(
                checked_vendor_fanin.get("evidence_class")
                or differential_spec.get("evidence_class")
                or spec["evidence_class"]
            )
        entry["receipt_integrity"] = _validate_semantic_receipt_integrity(
            receipt,
            spec=spec,
            module=module,
        )
        entry["offline_observation"] = _offline_observation(
            receipt,
            production_elevation_allowed=effective_production_allowed,
        )
        entry["receipt_goal_id"] = receipt.get("goal_id")
        entry["receipt_task_id"] = receipt.get("task_id")
        entry["certified"] = bool(
            receipt.get(str(spec["certified_key"])) or receipt.get("certified")
        ) and bool(entry["receipt_integrity"]["valid"])
        if not entry["receipt_integrity"]["valid"]:
            entry["block_reasons"].extend(
                entry["receipt_integrity"]["failures"]
            )
        if not entry["offline_observation"]["satisfied"]:
            entry["block_reasons"].append("offline_observation_failed")
        entry["per_tool"] = {}
        for tool_id in tool_ids:
            certified, raw_checks, block_reasons = _tool_certified_from_semantic_receipt(
                tool_id,
                receipt,
                certified_key=str(spec["certified_key"]),
                selector=str(spec.get("selector") or "root"),
            )
            normalized_checks = _normalize_semantic_checks(tool_id, raw_checks)
            identity = _semantic_tool_identity(
                tool_id,
                receipt,
                selector=str(spec.get("selector") or "root"),
                repo_root=repo_root,
            )
            identity["artifacts"].append(
                {
                    "kind": "semantic_certifier_module",
                    "path": Path(spec["module_relative"]).as_posix(),
                    "sha256": file_digest(module_path),
                    "artifact_class": "repository_source",
                }
            )
            artifact_validation = _validate_artifact_identities(
                identity["artifacts"],
                repo_root=repo_root,
            )
            checks_complete = bool(normalized_checks) and all(
                check.status == "passed" for check in normalized_checks
            )
            second_failed_blocks, second_failed_reasons = (
                second_failed_check_blocks_promotion(normalized_checks)
            )
            certified = bool(
                certified
                and entry["receipt_integrity"]["valid"]
                and entry["offline_observation"]["satisfied"]
                and checks_complete
                and artifact_validation["valid"]
                and not second_failed_blocks
            )
            tool_block_reasons = list(block_reasons) + list(
                artifact_validation["failures"]
            )
            if second_failed_blocks:
                tool_block_reasons.extend(second_failed_reasons)
            entry["per_tool"][tool_id] = {
                "certified": certified,
                "block_reasons": tool_block_reasons,
                "check_kinds_present": sorted(
                    {
                        str(c.get("kind"))
                        for c in raw_checks
                        if isinstance(c, Mapping) and c.get("kind")
                    }
                ),
                # Lossless: retain every check, including duplicate kinds.
                "checks_retained_without_kind_collapse": True,
                "checks_passed": sum(
                    1
                    for c in raw_checks
                    if isinstance(c, Mapping) and str(c.get("status")) == "passed"
                ),
                "checks_total": len(raw_checks),
                "checks": [check.to_dict() for check in normalized_checks],
                "check_set_digest_sha256": content_digest(
                    [check.to_dict() for check in normalized_checks]
                ),
                "identity": identity,
                "artifact_validation": artifact_validation,
                "handler_key": f"{entry['property_lane_id']}::{tool_id}",
            }
            if certified:
                entry["semantically_usable_tool_ids"].append(tool_id)
            if certified and (
                bool(spec["production_elevation_allowed"])
                or tool_id in eligible_live_tool_ids
                or tool_id in eligible_vendor_tool_ids
            ):
                entry["elevated_tool_ids"].append(tool_id)
        results.append(entry)
    return results


def apply_semantic_elevations(
    tool_certs: dict[str, ToolCertification],
    semantic_results: Sequence[Mapping[str, Any]],
    *,
    repo_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Elevate usable tools whose focused semantic certifiers fully pass.

    Uses the receipts already collected by ``run_semantic_lane_certifiers`` so
    each offline suite runs once. Synthetic passes are forbidden.
    """

    elevations: list[dict[str, Any]] = []
    root = (repo_root or repo_root_from()).resolve()
    for result in semantic_results:
        lane_id = str(result.get("lane_id") or "")
        spec = next(
            (item for item in SEMANTIC_CERTIFIER_SPECS if item["lane_id"] == lane_id),
            None,
        )
        if spec is None:
            continue
        usable_elevation_allowed = bool(
            spec.get("usable_elevation_allowed", True)
        )
        live_specialized = result.get("live_specialized_receipt")
        live_specialized = (
            live_specialized
            if isinstance(live_specialized, Mapping)
            else {}
        )
        live_eligible_tool_ids = {
            str(item)
            for item in live_specialized.get("eligible_tool_ids") or ()
        }
        receipt = result.get("receipt")
        receipt_fanin = (
            receipt.get("checked_vendor_fanin")
            if isinstance(receipt, Mapping)
            else None
        )
        vendor_eligible_tool_ids = (
            _recorded_checked_vendor_fanin_eligibility(
                repo_root=root,
                semantic_spec=spec,
                result_fanin=result.get("checked_vendor_fanin"),
                receipt_fanin=receipt_fanin,
                semantic_receipt=receipt,
            )
        )
        if live_eligible_tool_ids or vendor_eligible_tool_ids:
            usable_elevation_allowed = True
        if not usable_elevation_allowed:
            for pending_tool_id in spec["tool_ids"]:
                pending_cert = tool_certs.get(str(pending_tool_id))
                if pending_cert is None:
                    continue
                pending_cert.usable = False
                pending_cert.unavailable = True
                pending_cert.production_certified = False
                pending_cert.promotion_blocked = True
                pending_cert.evidence_class = str(spec["evidence_class"])
                pending_reason = (
                    "external_prover_installation_and_live_fanin_pending"
                )
                if pending_reason not in pending_cert.block_reasons:
                    pending_cert.block_reasons.append(pending_reason)
        if not isinstance(receipt, Mapping):
            if result.get("elevated_tool_ids"):
                elevations.append(
                    {
                        "lane_id": lane_id,
                        "status": "elevation_aborted",
                        "error": "semantic_receipt_missing",
                    }
                )
            continue
        result_digest = str(result.get("digest_sha256") or "")
        integrity = result.get("receipt_integrity")
        offline = result.get("offline_observation")
        declared_digest_fields = [
            field_name
            for field_name in (
                "receipt_digest_sha256",
                "certificate_digest_sha256",
                "digest_sha256",
            )
            if field_name in receipt
        ]
        declared_digests_valid = bool(declared_digest_fields) and all(
            _digest_matches(
                receipt.get(field_name),
                content_digest(
                    {
                        key: value
                        for key, value in receipt.items()
                        if key != field_name
                    }
                ),
            )
            for field_name in declared_digest_fields
        )
        result_integrity_valid = bool(
            result.get("status") == "ran"
            and result_digest
            and result_digest == content_digest(receipt)
            and str(receipt.get("interface") or "") == str(spec["interface"])
            and declared_digests_valid
            and isinstance(integrity, Mapping)
            and integrity.get("valid") is True
            and isinstance(offline, Mapping)
            and offline.get("satisfied") is True
        )
        if not result_integrity_valid:
            for tool_id in result.get("tool_ids") or spec["tool_ids"]:
                elevations.append(
                    {
                        "tool_id": tool_id,
                        "lane_id": lane_id,
                        "elevated": False,
                        "reason": "semantic_receipt_integrity_failed",
                    }
                )
            continue

        candidate_ids = list(result.get("tool_ids") or spec["tool_ids"])
        for tool_id in candidate_ids:
            cert = tool_certs.get(str(tool_id))
            if cert is None:
                elevations.append(
                    {
                        "tool_id": tool_id,
                        "lane_id": lane_id,
                        "elevated": False,
                        "reason": "target_not_usable",
                    }
                )
                continue
            certified, raw_checks, block_reasons = _tool_certified_from_semantic_receipt(
                str(tool_id),
                receipt,
                certified_key=str(spec["certified_key"]),
                selector=str(spec.get("selector") or "root"),
            )
            if not certified:
                elevations.append(
                    {
                        "tool_id": tool_id,
                        "lane_id": lane_id,
                        "elevated": False,
                        "reason": "semantic_receipt_not_certified",
                        "block_reasons": block_reasons,
                    }
                )
                continue

            projected = _normalize_semantic_checks(str(tool_id), raw_checks)
            all_passed = all(check.status == "passed" for check in projected)
            if not all_passed:
                elevations.append(
                    {
                        "tool_id": tool_id,
                        "lane_id": lane_id,
                        "elevated": False,
                        "reason": "required_check_kinds_incomplete",
                        "checks": [check.to_dict() for check in projected],
                    }
                )
                continue

            per_tool_results = result.get("per_tool")
            per_tool_result = (
                per_tool_results.get(str(tool_id), {})
                if isinstance(per_tool_results, Mapping)
                else {}
            )
            projected_payload = [check.to_dict() for check in projected]
            if (
                not isinstance(per_tool_result, Mapping)
                or per_tool_result.get("certified") is not True
                or str(per_tool_result.get("check_set_digest_sha256") or "")
                != content_digest(projected_payload)
            ):
                elevations.append(
                    {
                        "tool_id": tool_id,
                        "lane_id": lane_id,
                        "elevated": False,
                        "reason": "per_tool_semantic_binding_invalid",
                    }
                )
                continue
            semantic_identity = (
                per_tool_result.get("identity")
                if isinstance(per_tool_result, Mapping)
                else None
            )
            if not isinstance(semantic_identity, Mapping):
                semantic_identity = _semantic_tool_identity(
                    str(tool_id),
                    receipt,
                    selector=str(spec.get("selector") or "root"),
                    repo_root=root,
                )
            artifacts = semantic_identity.get("artifacts")
            artifacts = (
                [item for item in artifacts if isinstance(item, Mapping)]
                if isinstance(artifacts, Sequence)
                and not isinstance(artifacts, (str, bytes, bytearray))
                else []
            )
            artifact_validation = _validate_artifact_identities(
                artifacts,
                repo_root=root,
            )
            recorded_artifact_validation = per_tool_result.get(
                "artifact_validation"
            )
            if (
                not artifact_validation["valid"]
                or not isinstance(recorded_artifact_validation, Mapping)
                or recorded_artifact_validation.get("valid") is not True
            ):
                elevations.append(
                    {
                        "tool_id": tool_id,
                        "lane_id": lane_id,
                        "elevated": False,
                        "reason": "semantic_artifact_identity_invalid",
                        "failures": artifact_validation["failures"],
                    }
                )
                continue
            cert.artifact_identities.extend(
                item
                for item in artifacts
                if item not in cert.artifact_identities
            )
            receipt_digest = result_digest
            if receipt_digest and receipt_digest not in cert.semantic_receipt_digests:
                cert.semantic_receipt_digests.append(receipt_digest)
            cert.checks = projected

            if spec.get("identity_from_receipt"):
                executable = semantic_identity.get("executable_path")
                if executable:
                    resolved_executable = _resolve_artifact_path(
                        executable,
                        repo_root=root,
                    )
                    cert.executable_path = (
                        str(resolved_executable)
                        if resolved_executable is not None
                        else None
                    )
                    cert.executable_sha256 = file_digest(resolved_executable)
                    cert.executable_artifact_class = classify_executable_artifact(
                        resolved_executable
                    )
                if semantic_identity.get("version_string"):
                    cert.version_string = str(semantic_identity["version_string"])
                version_exact = True
                if cert.locked_version and cert.version_string:
                    version_exact = not detect_locked_version_mismatch(
                        cert.locked_version,
                        cert.version_string,
                    )
                usable_artifact_bound = bool(
                    artifact_validation["validated"]
                    and (
                        executable
                        or any(
                            item.get("artifact_class")
                            == "public_deployment_binding"
                            for item in artifact_validation["validated"]
                        )
                    )
                )
                semantic_identity_usable = bool(
                    semantic_identity.get("identity_probed")
                    and usable_artifact_bound
                    and version_exact
                )
                if semantic_identity_usable and usable_elevation_allowed:
                    cert.identity_probed = True
                    cert.locked_version_mismatch = False
                    cert.installed = True
                    cert.usable = True
                    cert.unavailable = False

            tool_production_allowed = bool(
                spec.get("production_elevation_allowed")
                or str(tool_id) in live_eligible_tool_ids
                or str(tool_id) in vendor_eligible_tool_ids
            )
            if not tool_production_allowed:
                cert.production_certified = False
                cert.promotion_blocked = True
                if not usable_elevation_allowed:
                    cert.usable = False
                    cert.unavailable = True
                reason = "evidence_class_cannot_satisfy_production_authority"
                if reason not in cert.block_reasons:
                    cert.block_reasons.append(reason)
                cert.evidence_class = str(
                    result.get("evidence_class") or spec["evidence_class"]
                )
                cert.notes = (
                    f"{spec['interface']} evidence is fully bound but classified "
                    f"as {spec['evidence_class']}; it cannot be relabeled as "
                    "production proof authority."
                )
                elevations.append(
                    {
                        "tool_id": tool_id,
                        "lane_id": lane_id,
                        "elevated": False,
                        "reason": reason,
                        "interface": str(spec["interface"]),
                        "evidence_class": cert.evidence_class,
                        "semantic_receipt_digest_sha256": receipt_digest,
                        "checks": [check.to_public_dict() for check in projected],
                        "checks_digest_sha256": content_digest(
                            [check.to_dict() for check in projected]
                        ),
                    }
                )
                continue

            if spec.get("identity_from_receipt"):
                executable = semantic_identity.get("executable_path")
                exact_artifact = bool(
                    artifact_validation["has_production_binding"]
                )
                if executable:
                    resolved_executable = _resolve_artifact_path(
                        executable,
                        repo_root=root,
                    )
                    cert.executable_path = (
                        str(resolved_executable)
                        if resolved_executable is not None
                        else None
                    )
                    cert.executable_sha256 = file_digest(resolved_executable)
                    cert.executable_artifact_class = classify_executable_artifact(
                        resolved_executable
                    )
                if semantic_identity.get("version_string"):
                    cert.version_string = str(semantic_identity["version_string"])
                version_exact = True
                if cert.locked_version and cert.version_string:
                    version_exact = not detect_locked_version_mismatch(
                        cert.locked_version,
                        cert.version_string,
                    )
                elif cert.locked_version and not any(
                    item.get("artifact_class") == "public_deployment_binding"
                    for item in artifact_validation["production_bindings"]
                ):
                    version_exact = False
                cert.identity_probed = bool(
                    semantic_identity.get("identity_probed")
                    and exact_artifact
                    and version_exact
                    and (
                        cert.executable_artifact_class
                        == "native_or_managed_binary"
                        or (
                            cert.executable_artifact_class
                            == "launcher_script"
                            and cert.launcher_binding.get("valid") is True
                        )
                        or any(
                            item.get("artifact_class")
                            == "public_deployment_binding"
                            for item in artifact_validation[
                                "production_bindings"
                            ]
                        )
                    )
                )
                cert.locked_version_mismatch = not version_exact
                cert.installed = cert.identity_probed
                cert.usable = cert.identity_probed
                cert.unavailable = not cert.identity_probed
                if not cert.identity_probed:
                    elevations.append(
                        {
                            "tool_id": tool_id,
                            "lane_id": lane_id,
                            "elevated": False,
                            "reason": "semantic_identity_not_exactly_bound",
                            "interface": str(spec["interface"]),
                            "evidence_class": str(spec["evidence_class"]),
                            "checks": [
                                check.to_public_dict() for check in projected
                            ],
                            "checks_digest_sha256": content_digest(
                                [check.to_dict() for check in projected]
                            ),
                        }
                    )
                    continue

            if cert.unavailable or not cert.usable:
                elevations.append(
                    {
                        "tool_id": tool_id,
                        "lane_id": lane_id,
                        "elevated": False,
                        "reason": "target_not_usable",
                        "interface": str(spec["interface"]),
                    }
                )
                continue

            cert.production_certified = True
            cert.promotion_blocked = False
            cert.block_reasons = []
            cert.evidence_class = str(
                result.get("evidence_class") or spec["evidence_class"]
            )
            cert.notes = (
                f"Role-aware elevation from {spec['interface']}: full "
                "positive/negative/mutation/replay semantic evidence bound."
            )
            elevations.append(
                {
                    "tool_id": tool_id,
                    "lane_id": lane_id,
                    "elevated": True,
                    "interface": str(spec["interface"]),
                    "evidence_class": cert.evidence_class,
                    "semantic_receipt_digest_sha256": receipt_digest,
                    "checks": [check.to_public_dict() for check in projected],
                    "checks_digest_sha256": content_digest(
                        [check.to_dict() for check in projected]
                    ),
                }
            )
    return elevations


def load_authority_roles(repo_root: Path) -> dict[str, Any]:
    """Bind role / ceiling matrix when the roles surface is present."""

    roles_path = repo_root / "tools" / "logic" / "certification" / "roles.py"
    if not roles_path.is_file():
        return {
            "present": False,
            "reason": "roles_module_missing",
            "tools": {},
            "policy": {},
        }
    try:
        module = _load_module_from_path(roles_path, "fvt_role_aware_roles")
        boundary = module.authority_boundary_report()
        policy = module.build_role_aware_policy()
        tools = {
            item.tool_id: {
                "role": item.role.value,
                "authority_ceiling": item.authority_ceiling.value,
                "can_satisfy_certified_authority": bool(
                    item.can_satisfy_certified_authority
                ),
                "lane_ids": list(getattr(item, "lane_ids", ()) or ()),
            }
            for item in module.list_tool_roles()
        }
        return {
            "present": True,
            "interface": getattr(module, "PROMOTION_INTERFACE", None),
            "role_interface": getattr(module, "INTERFACE", None),
            "tools": tools,
            "boundary": {
                key: boundary.get(key)
                for key in (
                    "support_only",
                    "advisor_or_candidate_only",
                    "shadow_checkers",
                    "kernel_authority",
                    "authorization_only",
                    "finite_trace_authority",
                    "attestation_authority_only",
                    "policy",
                )
                if key in boundary
            },
            "lane_handlers": boundary.get("lane_handlers") or {},
            "policy_digest_sha256": content_digest(policy.to_dict()),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "present": False,
            "reason": f"{type(exc).__name__}:{exc}",
            "tools": {},
            "policy": {},
        }


def apply_role_aware_demotions(
    tool_certs: dict[str, ToolCertification],
    authority_roles: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Support / advisor / shadow presence never yields production certification."""

    demotions: list[dict[str, Any]] = []
    tools = authority_roles.get("tools") or {}
    if not isinstance(tools, Mapping):
        return demotions
    for tool_id, cert in tool_certs.items():
        meta = tools.get(tool_id)
        if not isinstance(meta, Mapping):
            continue
        certifying = bool(meta.get("can_satisfy_certified_authority"))
        role = str(meta.get("role") or "")
        if certifying:
            continue
        if cert.production_certified or cert.usable:
            if cert.production_certified:
                demotions.append(
                    {
                        "tool_id": tool_id,
                        "role": role,
                        "reason": "non_certifying_role_cannot_promote",
                        "authority_ceiling": meta.get("authority_ceiling"),
                    }
                )
            cert.production_certified = False
            cert.promotion_blocked = True
            if "non_certifying_role" not in cert.block_reasons:
                cert.block_reasons.append("non_certifying_role")
            if cert.evidence_class == "production_certified":
                cert.evidence_class = "role_demoted"
            if not cert.notes:
                cert.notes = (
                    f"Role {role!r} cannot satisfy certified authority; "
                    "presence is not production certification."
                )
    return demotions


def build_managed_deployment_readiness(
    *,
    lock: Mapping[str, Any],
    tools_index: Mapping[str, Mapping[str, Any]],
    tool_certs: Mapping[str, ToolCertification],
    authority_roles: Mapping[str, Any],
    repo_root: Path | None = None,
    semantic_results: Sequence[Mapping[str, Any]] | None = None,
    checked_vendor_capability_readiness: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive deployment blockers for every host-supported managed tool."""

    host_platform = observed_platform_id()
    root = (repo_root or repo_root_from()).resolve()
    global_platforms = [
        str(item)
        for item in (
            (lock.get("platform_policy") or {}).get("supported_platforms") or []
        )
    ]
    role_rows = authority_roles.get("tools") or {}
    platform_rows: list[dict[str, Any]] = []
    capabilities: list[str] = []
    dependencies: list[str] = []
    exceptions: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    expected_vendor_readiness: dict[str, Any] | None = None
    vendor_readiness_binding_valid = False
    vendor_readiness_tools: Mapping[str, Any] = {}
    if semantic_results is not None:
        expected_vendor_readiness = (
            build_checked_vendor_capability_readiness_projection(
                repo_root=root,
                semantic_results=semantic_results,
            )
        )
        supplied_vendor_readiness = (
            dict(checked_vendor_capability_readiness)
            if isinstance(checked_vendor_capability_readiness, Mapping)
            else expected_vendor_readiness
        )
        vendor_readiness_binding_valid = bool(
            supplied_vendor_readiness == expected_vendor_readiness
        )
        if vendor_readiness_binding_valid:
            raw_vendor_tools = expected_vendor_readiness.get("tools")
            if isinstance(raw_vendor_tools, Mapping):
                vendor_readiness_tools = raw_vendor_tools

    for tool_id in sorted(tools_index):
        entry = tools_index[tool_id]
        platform_row = tool_platform_support(
            entry,
            host_platform=host_platform,
            global_supported_platforms=global_platforms,
        )
        platform_rows.append(platform_row)
        if not platform_row["managed"]:
            continue

        role_meta = (
            role_rows.get(tool_id)
            if isinstance(role_rows, Mapping)
            else None
        )
        role = (
            str(role_meta.get("role") or "unclassified")
            if isinstance(role_meta, Mapping)
            else "unclassified"
        )
        category = (
            "dependency"
            if role == "support" or tool_id in {"opam", "stack", "maude"}
            else "capability"
        )
        if not platform_row["supported"]:
            if platform_row["exception_eligible"]:
                exceptions.append(
                    {
                        "tool_id": tool_id,
                        "host_platform": host_platform,
                        "declared_platforms": platform_row["declared_platforms"],
                        "basis": platform_row["basis"],
                        "classification": platform_row["classification"],
                        "category": category,
                        "narrow_scope": True,
                        "complete": False,
                        "production_certified": False,
                    }
                )
            else:
                blockers.append(
                    {
                        "tool_id": tool_id,
                        "category": category,
                        "role": role,
                        "evidence_class": "platform_ambiguous",
                        "artifact_classes": [],
                        "reasons": ["platform_support_ambiguous_or_contradictory"],
                    }
                )
            continue

        if category == "dependency":
            dependencies.append(tool_id)
        else:
            capabilities.append(tool_id)

        cert = tool_certs.get(tool_id)
        if cert is None:
            blockers.append(
                {
                    "tool_id": tool_id,
                    "category": category,
                    "reasons": ["certificate_entry_missing"],
                }
            )
            continue

        artifact_classes = {
            str(item.get("artifact_class") or "")
            for item in cert.artifact_identities
            if isinstance(item, Mapping)
        }
        artifact_validation = _validate_artifact_identities(
            [
                item
                for item in cert.artifact_identities
                if isinstance(item, Mapping)
            ],
            repo_root=root,
        )
        exact_artifact = bool(artifact_validation["has_production_binding"])
        public_binding = any(
            item.get("artifact_class") == "public_deployment_binding"
            for item in artifact_validation["production_bindings"]
        )
        launcher_target_bound = bool(
            cert.launcher_binding.get("valid") is True
            and artifact_validation["has_production_binding"]
        )
        genuinely_installed = bool(
            exact_artifact
            and artifact_validation["valid"]
            and (
                (cert.installed and cert.identity_probed)
                or public_binding
            )
        )
        vendor_readiness = vendor_readiness_tools.get(tool_id)
        vendor_readiness = (
            vendor_readiness
            if isinstance(vendor_readiness, Mapping)
            else {}
        )
        vendor_readiness_spec = next(
            (
                raw_spec
                for raw_spec in CHECKED_VENDOR_FANIN_SPECS.values()
                if str(
                    raw_spec.get("managed_readiness_tool_id") or ""
                )
                == tool_id
            ),
            {},
        )
        vendor_capability_ready = bool(
            vendor_readiness_binding_valid
            and vendor_readiness.get("ready") is True
            and vendor_readiness.get("installation_ready") is True
            and vendor_readiness.get("semantic_certification_ready") is True
            and vendor_readiness.get("production_certified") is False
            and vendor_readiness.get("production_elevation_allowed") is False
            and vendor_readiness.get("authority_granted") is False
            and vendor_readiness.get("grants_theorem_authority") is False
            and vendor_readiness.get("grants_global_correctness") is False
            and vendor_readiness.get(
                "grants_authorization_decision_authority"
            )
            is False
            and vendor_readiness.get(
                "authority_requirement_satisfied"
            )
            is False
            and vendor_readiness.get("readiness_scope")
            == vendor_readiness_spec.get("managed_readiness_scope")
            and vendor_readiness.get("role")
            == vendor_readiness_spec.get("managed_readiness_role")
            and vendor_readiness.get("declared_authority_role") == role
            and vendor_readiness.get("declared_authority_ceiling")
            == (
                role_meta.get("authority_ceiling")
                if isinstance(role_meta, Mapping)
                else None
            )
            and vendor_readiness.get(
                "declared_role_can_satisfy_certified_authority"
            )
            == (
                role_meta.get("can_satisfy_certified_authority")
                if isinstance(role_meta, Mapping)
                else None
            )
            and not list(vendor_readiness.get("failures") or ())
        )
        vendor_artifact = vendor_readiness.get(
            "checked_install_receipt_artifact"
        )
        vendor_artifact = (
            vendor_artifact if isinstance(vendor_artifact, Mapping) else {}
        )
        if vendor_capability_ready:
            artifact_classes.add(
                str(
                    vendor_artifact.get("artifact_class")
                    or "public_deployment_binding"
                )
            )

        reasons: list[str] = []
        if not (genuinely_installed or vendor_capability_ready):
            reasons.append("supported_managed_installation_missing_or_shim_only")
        if not artifact_validation["valid"] and not vendor_capability_ready:
            reasons.append("artifact_identity_invalid")
        if (
            "launcher_script" in artifact_classes
            and not (
                public_binding
                or launcher_target_bound
                or vendor_capability_ready
            )
        ):
            reasons.append("launcher_target_artifact_unbound")
        if cert.locked_version_mismatch and not vendor_capability_ready:
            reasons.append("locked_version_mismatch")

        if category == "capability":
            required_kinds = {"positive", "negative", "mutation", "replay"}
            present_kinds = {check.kind for check in cert.checks}
            checks_complete = bool(cert.checks) and all(
                check.status == "passed" for check in cert.checks
            ) and required_kinds <= present_kinds
            checks_complete = bool(
                checks_complete or vendor_capability_ready
            )
            certifying_role = bool(
                isinstance(role_meta, Mapping)
                and role_meta.get("can_satisfy_certified_authority")
            )
            if (
                certifying_role
                and not cert.production_certified
                and not vendor_capability_ready
            ):
                reasons.append("semantic_evidence_below_authority_ceiling")
            if not checks_complete:
                reasons.append("full_semantic_check_set_missing_or_failed")
            if certifying_role and not cert.semantic_receipt_digests and not (
                cert.evidence_class == "production_certified"
                and cert.tool_id in {"z3", "cvc5"}
            ) and not vendor_capability_ready:
                reasons.append("semantic_receipt_not_bound")
            if not certifying_role and cert.production_certified:
                reasons.append("non_certifying_role_incorrectly_promoted")
            if vendor_readiness and cert.production_certified:
                reasons.append(
                    "external_witness_or_shadow_incorrectly_promoted"
                )

        if reasons:
            blockers.append(
                {
                    "tool_id": tool_id,
                    "category": category,
                    "role": role,
                    "authority_ceiling": (
                        role_meta.get("authority_ceiling")
                        if isinstance(role_meta, Mapping)
                        else None
                    ),
                    "evidence_class": cert.evidence_class,
                    "artifact_classes": sorted(artifact_classes),
                    "reasons": sorted(set(reasons)),
                }
            )

    capability_blockers = [
        row for row in blockers if row.get("category") == "capability"
    ]
    dependency_blockers = [
        row for row in blockers if row.get("category") == "dependency"
    ]
    readiness = {
        "host_platform": host_platform,
        "global_supported_platforms": global_platforms,
        "host_globally_supported": host_platform in set(global_platforms),
        "platform_rows": platform_rows,
        "supported_managed_capability_tool_ids": capabilities,
        "supported_managed_dependency_tool_ids": dependencies,
        "platform_exceptions": exceptions,
        "capability_blockers": capability_blockers,
        "dependency_blockers": dependency_blockers,
        "all_blockers": blockers,
        "ready": not blockers,
        "status": (
            "all_supported_managed_capabilities_ready"
            if not blockers
            else "supported_managed_capabilities_blocked"
        ),
    }
    if expected_vendor_readiness is not None:
        readiness["checked_vendor_capability_readiness"] = (
            expected_vendor_readiness
        )
        readiness["checked_vendor_capability_readiness_binding_valid"] = (
            vendor_readiness_binding_valid
        )
        readiness["ready_via_checked_vendor_capability_tool_ids"] = sorted(
            tool_id
            for tool_id, raw_entry in vendor_readiness_tools.items()
            if isinstance(raw_entry, Mapping)
            and raw_entry.get("ready") is True
        )
    return readiness


def build_certificate(
    *,
    repo_root: Path | None = None,
    lock_path: Path | None = None,
    env: Mapping[str, str] | None = None,
    observed_at: str | None = None,
    role_aware: bool = False,
    full_evidence_out: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the full hermetic multi-prover certification and return the certificate.

    When ``role_aware`` is true (FVT-G200 reissue), focused semantic receipts
    are aggregated and checked against their role ceiling. A prover is
    elevated only when its current native/managed artifacts and a durable live
    specialized receipt both verify; otherwise it remains usable-but-pending
    or unavailable at its existing ceiling. Default (FVT-G060) certification
    keeps identity-only lanes non-elevated so usability and semantic
    certification remain distinct.
    """

    root = repo_root or repo_root_from()
    lock_file = lock_path or (root / DEFAULT_LOCK_RELATIVE)
    lock = load_lock(lock_file)
    tools_index = lock_tools_by_id(lock)
    run_env = offline_env(env)
    repository_python_paths = (
        str(root),
        str(root / "ipfs_datasets_py"),
    )
    existing_python_path = str(run_env.get("PYTHONPATH") or "").strip()
    run_env["PYTHONPATH"] = os.pathsep.join(
        (
            *repository_python_paths,
            *((existing_python_path,) if existing_python_path else ()),
        )
    )

    # Map tool → lanes and preferred check kind (smtlib wins when present).
    tool_lane_map: dict[str, list[str]] = {}
    tool_check_kind: dict[str, str] = {}
    for lane in PROPERTY_LANES:
        for tool_id in lane["tool_ids"]:
            tool_lane_map.setdefault(tool_id, []).append(lane["lane_id"])
            prior = tool_check_kind.get(tool_id)
            kind = str(lane["check_kind"])
            if prior is None or kind == "smtlib":
                tool_check_kind[tool_id] = kind

    # Certify every lock tool that appears in a lane; also surface lock-only
    # tools so nothing is concealed.
    lane_tool_ids = {tool_id for lane in PROPERTY_LANES for tool_id in lane["tool_ids"]}
    certify_ids = sorted(set(tools_index) | lane_tool_ids)

    tool_certs: dict[str, ToolCertification] = {}
    for tool_id in certify_ids:
        entry = tools_index.get(tool_id)
        if entry is None:
            # Lane references a tool not in the lock — explicit unavailable.
            cert = ToolCertification(
                tool_id=tool_id,
                lane_ids=tool_lane_map.get(tool_id, []),
                unavailable=True,
                promotion_blocked=True,
                block_reasons=["missing_from_lock"],
                evidence_class="unavailable",
                notes="Referenced by a property lane but absent from the offline lock.",
                checks=[
                    CheckResult(
                        check_id=f"{tool_id}.{kind}",
                        kind=kind,
                        status="unavailable",
                        expected="lock entry",
                        observed="missing_from_lock",
                    )
                    for kind in ("positive", "negative", "mutation", "replay")
                ],
            )
            tool_certs[tool_id] = cert
            continue

        cert = certify_tool(
            entry,
            lane_ids=tool_lane_map.get(tool_id, []),
            check_kind=tool_check_kind.get(tool_id, "identity_only"),
            env=run_env,
        )
        tool_certs[tool_id] = cert

    disagreements: list[DisagreementQuarantine] = []
    smt_quarantine = quarantine_smt_disagreement(tool_certs)
    if smt_quarantine is not None:
        disagreements.append(smt_quarantine)
        for tool_id in smt_quarantine.promotion_blocked_tool_ids:
            cert = tool_certs[tool_id]
            cert.production_certified = False
            cert.promotion_blocked = True
            if "cross_provider_disagreement" not in cert.block_reasons:
                cert.block_reasons.append("cross_provider_disagreement")
            cert.evidence_class = "quarantined_disagreement"
            cert.notes = (
                "Cross-provider SMT disagreement quarantined; "
                "disagreement cannot raise authority or promote."
            )

    # Role matrix binding is cheap; semantic elevation is opt-in and expensive
    # (Lean kernel suite). Default FVT-G060 keeps identity-only usability.
    authority_roles = load_authority_roles(root)
    semantic_results: list[dict[str, Any]] = []
    elevations: list[dict[str, Any]] = []
    demotions: list[dict[str, Any]] = []
    if role_aware:
        semantic_results = run_semantic_lane_certifiers(
            repo_root=root,
            env=run_env,
            tool_certs=tool_certs,
        )
        elevations = apply_semantic_elevations(
            tool_certs,
            semantic_results,
            repo_root=root,
        )
        demotions = apply_role_aware_demotions(tool_certs, authority_roles)

    checked_vendor_capability_readiness = (
        build_checked_vendor_capability_readiness_projection(
            repo_root=root,
            semantic_results=semantic_results,
        )
        if role_aware
        else None
    )
    deployment_readiness = build_managed_deployment_readiness(
        lock=lock,
        tools_index=tools_index,
        tool_certs=tool_certs,
        authority_roles=authority_roles,
        repo_root=root,
        semantic_results=semantic_results if role_aware else None,
        checked_vendor_capability_readiness=(
            checked_vendor_capability_readiness
        ),
    )

    lanes = certify_property_lanes(tool_certs, disagreements)

    production_certified_ids = sorted(
        tid for tid, cert in tool_certs.items() if cert.production_certified
    )
    unavailable_ids = sorted(
        tid for tid, cert in tool_certs.items() if cert.unavailable
    )
    blocked_map = {
        tid: list(cert.block_reasons)
        for tid, cert in tool_certs.items()
        if cert.promotion_blocked
    }

    # Tools that remain merely usable (identity-only) after optional elevation.
    merely_usable_ids = sorted(
        tid
        for tid, cert in tool_certs.items()
        if cert.usable and not cert.production_certified and not cert.unavailable
    )

    offline_policy = dict(lock.get("offline_verification_policy") or {})
    offline_observations = [
        result.get("offline_observation")
        for result in semantic_results
        if isinstance(result.get("offline_observation"), Mapping)
    ]
    required_offline_policy_keys = (
        "forbid_install",
        "forbid_download",
        "forbid_network",
        "forbid_curl_pipe_shell",
        "forbid_system_package_mutation",
    )
    offline_policy_satisfied = all(
        bool(offline_policy.get(key)) for key in required_offline_policy_keys
    ) and all(bool(item.get("satisfied")) for item in offline_observations)
    full_public_semantic_results = [
        _project_semantic_lane_result(result, repo_root=root)
        for result in semantic_results
    ]
    public_semantic_results = [
        _compact_semantic_lane_projection(result)
        for result in full_public_semantic_results
    ]
    public_semantic_by_lane = {
        str(result.get("lane_id") or ""): result
        for result in public_semantic_results
        if str(result.get("lane_id") or "")
    }
    public_elevations: list[dict[str, Any]] = []
    for raw_elevation in elevations:
        if not isinstance(raw_elevation, Mapping):
            continue
        elevation = _compact_elevation_projection(raw_elevation)
        lane = public_semantic_by_lane.get(
            str(elevation.get("lane_id") or "")
        )
        tool_id = str(elevation.get("tool_id") or "")
        if isinstance(lane, Mapping):
            elevation["semantic_receipt_digest_sha256"] = lane.get(
                "digest_sha256"
            )
            per_tool = lane.get("per_tool")
            tool = (
                per_tool.get(tool_id, {})
                if isinstance(per_tool, Mapping)
                else {}
            )
            if isinstance(tool, Mapping):
                elevation["checks_digest_sha256"] = tool.get(
                    "check_set_digest_sha256"
                )
                elevation["checks_count"] = int(
                    tool.get("checks_total") or 0
                )
        public_elevations.append(elevation)
    full_specialized_aggregation = (
        aggregate_specialized_receipts(
            full_public_semantic_results,
            authority_roles=authority_roles,
            property_lanes=PROPERTY_LANES,
        )
        if role_aware
        else {
            "schema_version": SPECIALIZED_AGGREGATION_SCHEMA,
            "interface": SPECIALIZED_AGGREGATION_INTERFACE,
            "goal_id": SPECIALIZED_AGGREGATION_GOAL_ID,
            "task_id": SPECIALIZED_AGGREGATION_TASK_ID,
            "repair_task_id": SPECIALIZED_AGGREGATION_REPAIR_TASK_ID,
            "objective_validation_evidence": (
                SPECIALIZED_AGGREGATION_OBJECTIVE_VALIDATION_EVIDENCE
            ),
            "objective_validation_repair": False,
            "objective_validation_command": (
                SPECIALIZED_AGGREGATION_OBJECTIVE_VALIDATION_COMMAND
            ),
            "enabled": False,
            "reason": "role_aware_elevation_disabled",
            "acceptance": {
                "objective_validation_repair": False,
                "objective_validation_evidence": (
                    SPECIALIZED_AGGREGATION_OBJECTIVE_VALIDATION_EVIDENCE
                ),
                "repair_task_id": SPECIALIZED_AGGREGATION_REPAIR_TASK_ID,
                "goal_id": SPECIALIZED_AGGREGATION_GOAL_ID,
                "task_id": SPECIALIZED_AGGREGATION_TASK_ID,
            },
        }
    )
    specialized_aggregation = (
        _compact_specialized_receipt_aggregation(
            full_specialized_aggregation
        )
        if role_aware
        else full_specialized_aggregation
    )
    if full_evidence_out is not None:
        # Private in-memory hand-off for an independent downstream audit.
        # This evidence is intentionally not embedded in the durable compact
        # certificate; callers must explicitly request and consume it during
        # the same certification run.
        full_evidence_out["semantic_lane_results"] = (
            full_public_semantic_results
        )
        full_evidence_out["specialized_receipt_aggregation"] = (
            full_specialized_aggregation
        )
    certificate: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "interface": INTERFACE,
        "goal_id": GOAL_ID,
        "task_id": TASK_ID,
        "program": PROGRAM,
        "description": (
            "Hermetic offline multi-prover certification receipt. Available tools "
            "pass live positive/negative/mutation/replay checks with exact "
            "identities; absent/mismatched lanes are explicit skips/unavailable "
            "and block only their promotion; PATH shims are not usability; "
            "certification performs no download/network/install and quarantines "
            "disagreement."
            + (
                " Role-aware reissue consumes only hash- and semantics-verified "
                "live specialized receipts; tools without that current binding "
                "remain usable-but-pending or unavailable."
                if role_aware
                else ""
            )
        ),
        "observed_at": observed_at
        or datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "binding_mode": (
            "offline_pinned_live_lanes_role_aware"
            if role_aware
            else "offline_pinned_live_lanes"
        ),
        "lock": {
            "path": str(DEFAULT_LOCK_RELATIVE).replace("\\", "/"),
            "interface": lock.get("interface"),
            "schema_version": lock.get("schema_version"),
            "goal_id": lock.get("goal_id"),
            "task_id": lock.get("task_id"),
            "digest_sha256": content_digest(lock),
            "host_platform": deployment_readiness["host_platform"],
        },
        "certification_policy": {
            "forbid_install": bool(offline_policy.get("forbid_install")),
            "forbid_download": bool(offline_policy.get("forbid_download")),
            "forbid_network": bool(offline_policy.get("forbid_network")),
            "forbid_curl_pipe_shell": bool(
                offline_policy.get("forbid_curl_pipe_shell")
            ),
            "path_presence_is_not_usability": bool(
                offline_policy.get("path_presence_is_not_usability")
            ),
            "require_exact_pin_match_for_production_certification": bool(
                offline_policy.get(
                    "require_exact_pin_match_for_production_certification"
                )
            ),
            "shim_toolchain_mismatch_fails_closed": bool(
                offline_policy.get("shim_toolchain_mismatch_fails_closed")
            ),
            "absent_lanes_block_only_own_promotion": True,
            "optional_tools_not_mandatory_for_unrelated_properties": True,
            "quarantine_disagreement": True,
            "synthetic_evidence_cannot_certify_production": True,
            "role_aware_semantic_elevation": bool(role_aware),
            "non_certifying_roles_cannot_promote": True,
            "offline_policy_satisfied": offline_policy_satisfied,
            "offline_observations": offline_observations,
            "lock_offline_verification_policy": {
                key: bool(offline_policy.get(key))
                for key in (
                    "forbid_install",
                    "forbid_download",
                    "forbid_network",
                    "forbid_curl_pipe_shell",
                    "forbid_system_package_mutation",
                    "require_exact_pin_match_for_production_certification",
                    "path_presence_is_not_usability",
                    "shim_toolchain_mismatch_fails_closed",
                )
            },
        },
        "detection_rules": {
            "lean_shim_toolchain_mismatch": {
                "id": "lean_shim_toolchain_mismatch",
                "effect": {
                    "usable": False,
                    "production_certified": False,
                    "offline_verification": "fail_closed_without_install_or_fetch",
                },
            },
            "locked_version_mismatch": {
                "id": "locked_version_mismatch",
                "effect": {
                    "production_certified": False,
                    "offline_verification": "fail_closed_without_upgrade_or_download",
                },
            },
            "path_presence_is_not_usability": {
                "id": "path_presence_is_not_usability",
                "effect": {
                    "installed": False,
                    "usable": False,
                    "production_certified": False,
                },
            },
            "cross_provider_disagreement_quarantine": {
                "id": "cross_provider_disagreement_quarantine",
                "effect": {
                    "production_certified": False,
                    "status": "quarantined",
                },
            },
            "non_certifying_role_demotion": {
                "id": "non_certifying_role_demotion",
                "effect": {
                    "production_certified": False,
                    "status": "role_demoted",
                },
            },
        },
        "property_lanes": [lane.to_dict() for lane in lanes],
        "tools": [
            _compact_tool_certificate(tool_certs[tid])
            for tid in sorted(tool_certs)
        ],
        "disagreement_quarantines": [item.to_dict() for item in disagreements],
        "promotion": {
            "production_certified_tool_ids": production_certified_ids,
            "unavailable_tool_ids": unavailable_ids,
            "blocked_tool_ids": blocked_map,
            "merely_usable_tool_ids": merely_usable_ids,
            "lane_promotion_ready": {
                lane.lane_id: lane.promotion_ready for lane in lanes
            },
        },
        "authority_roles": authority_roles,
        # Retain every check in a portable public projection. Raw process
        # output and secret/witness values are retained only by digest.
        "semantic_lane_results": public_semantic_results,
        # FVT-G203: lossless per-tool specialized evidence aggregation.
        "specialized_receipt_aggregation": specialized_aggregation,
        "managed_deployment_readiness": deployment_readiness,
        "role_aware": {
            "enabled": bool(role_aware),
            "goal_id": ROLE_AWARE_GOAL_ID,
            "task_id": ROLE_AWARE_TASK_ID,
            "repair_task_id": ROLE_AWARE_REPAIR_TASK_ID,
            "interface": ROLE_AWARE_INTERFACE,
            # FVT-083 objective validation repair: re-prove FVT-G200 acceptance.
            "objective_validation_evidence": (
                ROLE_AWARE_OBJECTIVE_VALIDATION_EVIDENCE
            ),
            "objective_validation_repair": bool(role_aware),
            "objective_validation_command": (
                ROLE_AWARE_OBJECTIVE_VALIDATION_COMMAND
            ),
            "elevations": public_elevations,
            "demotions": demotions,
            "elevated_tool_ids": sorted(
                {
                    str(item.get("tool_id"))
                    for item in elevations
                    if item.get("elevated")
                }
            ),
            "required_baseline_elevations": [
                "lean",
                "runtime-mtl",
                "datalog-authorization",
                "secpal-authorization",
                "coq",
                "isabelle",
            ],
            "release_candidate": {
                "interface": RELEASE_CANDIDATE_INTERFACE,
                "goal_id": RELEASE_CANDIDATE_GOAL_ID,
                "task_id": RELEASE_CANDIDATE_TASK_ID,
                "max_stage": RELEASE_CANDIDATE_MAX_STAGE,
                "path": str(DEFAULT_RELEASE_CANDIDATE_RELATIVE).replace(
                    "\\", "/"
                ),
                "claims_merge": False,
                "claims_deployment": False,
            },
            "production_semantic_elevation_fanin": {
                "interface": PRODUCTION_ELEVATION_FANIN_INTERFACE,
                "goal_id": PRODUCTION_ELEVATION_FANIN_GOAL_ID,
                "task_id": PRODUCTION_ELEVATION_FANIN_TASK_ID,
                "path": str(
                    DEFAULT_PRODUCTION_ELEVATION_FANIN_RECEIPT_RELATIVE
                ).replace("\\", "/"),
                "integration_test": str(
                    DEFAULT_PRODUCTION_ELEVATION_FANIN_TEST_RELATIVE
                ).replace("\\", "/"),
                "claims_merge": False,
                "claims_deployment": False,
            },
            "acceptance": {
                "objective_validation_repair": bool(role_aware),
                "objective_validation_evidence": (
                    ROLE_AWARE_OBJECTIVE_VALIDATION_EVIDENCE
                ),
                "repair_task_id": ROLE_AWARE_REPAIR_TASK_ID,
                "goal_id": ROLE_AWARE_GOAL_ID,
                "task_id": ROLE_AWARE_TASK_ID,
                "role_aware_matrix_executed": bool(role_aware),
            },
        },
        "check_kinds_required": ["positive", "negative", "mutation", "replay"],
        "evidence": {
            "certifier": "tools/logic/certify_formal_verification_toolchains.py",
            "integration_test": (
                "test/integration/test_formal_verification_real_tool_matrix.py"
            ),
            "role_aware_integration_test": (
                "test/integration/test_formal_verification_role_aware_completion.py"
            ),
            "release_candidate_integration_test": (
                "test/integration/"
                "test_formal_verification_role_aware_release_candidate.py"
            ),
            "release_candidate": str(DEFAULT_RELEASE_CANDIDATE_RELATIVE).replace(
                "\\", "/"
            ),
            "production_elevation_fanin_integration_test": str(
                DEFAULT_PRODUCTION_ELEVATION_FANIN_TEST_RELATIVE
            ).replace("\\", "/"),
            "production_elevation_fanin_receipt": str(
                DEFAULT_PRODUCTION_ELEVATION_FANIN_RECEIPT_RELATIVE
            ).replace("\\", "/"),
            "lock": str(DEFAULT_LOCK_RELATIVE).replace("\\", "/"),
        },
        "certificate_digest_sha256": "",  # filled below
    }
    certificate = public_evidence_projection(certificate, repo_root=root)
    public_evidence_policy = public_evidence_audit(certificate, repo_root=root)
    certificate["public_evidence_policy"] = public_evidence_policy
    if not public_evidence_policy["satisfied"]:
        managed = certificate.get("managed_deployment_readiness")
        if isinstance(managed, dict):
            managed["ready"] = False
            blockers = managed.setdefault("dependency_blockers", [])
            if "public_evidence_redaction_failed" not in blockers:
                blockers.append("public_evidence_redaction_failed")
    certificate["certificate_digest_sha256"] = content_digest(
        {key: value for key, value in certificate.items() if key != "certificate_digest_sha256"}
    )
    return certificate


def write_certificate(
    certificate: Mapping[str, Any],
    destination: Path,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(certificate, indent=2, sort_keys=False) + "\n"
    # Atomic replace.
    fd, tmp_name = tempfile.mkstemp(
        prefix=destination.name + ".",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, destination)
    finally:
        if os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Certify the real multi-prover formal-verification matrix in "
            "hermetic offline lanes (FormalVerificationToolchainCertificate@1)."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root (default: auto-detect from this file)",
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=None,
        help="Path to offline toolchain lock JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Certificate output path (default: docs/architecture/...certificate.json)",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print certificate JSON to stdout instead of writing a file",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress human-readable summary",
    )
    parser.add_argument(
        "--role-aware",
        action="store_true",
        help=(
            "Apply FVT-G200 role-aware semantic elevation (Lean / Runtime MTL / "
            "Datalog-SecPAL) and bind authority roles into the certificate"
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = (args.repo_root or repo_root_from()).resolve()
    certificate = build_certificate(
        repo_root=root,
        lock_path=args.lock.resolve() if args.lock else None,
        role_aware=bool(args.role_aware),
    )

    if args.stdout:
        json.dump(certificate, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        output = (
            args.output.resolve()
            if args.output
            else (root / DEFAULT_CERTIFICATE_RELATIVE)
        )
        write_certificate(certificate, output)
        if not args.quiet:
            print(f"wrote {output}", file=sys.stderr)

    if not args.quiet:
        promotion = certificate["promotion"]
        print(
            "production_certified="
            f"{promotion['production_certified_tool_ids']}",
            file=sys.stderr,
        )
        print(
            f"unavailable={promotion['unavailable_tool_ids']}",
            file=sys.stderr,
        )
        print(
            f"quarantines={len(certificate['disagreement_quarantines'])}",
            file=sys.stderr,
        )

    # Exit 0 even when some tools are unavailable — absence is not a certifier
    # failure. Hard failures are schema/lock errors (already raised).
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
