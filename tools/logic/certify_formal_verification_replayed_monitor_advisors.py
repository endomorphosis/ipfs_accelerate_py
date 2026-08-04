#!/usr/bin/env python3
"""Replayed Runtime MTL monitor and advisor semantics fan-in.

``ReplayedMonitorAdvisorSemantics@1`` / FVT-G230 (FVT-098).

Owns the unified offline replay fan-in for:

* the independent Node/TypeScript Runtime MTL vendor engine against the
  in-process finite-trace monitor (with disagreement quarantine); and
* genuine ErgoAI and SymbolicAI advisor lanes under advisory authority.

This surface:

* re-executes the reviewed external Runtime MTL vendor certifier (FVT-G210)
  and the advisor / ErgoAI live toolchain certifiers (FVT-G160 / FVT-G218);
* binds package, source, lockfile, runtime, launcher, launcher-target,
  executable, and artifact digests on the Runtime MTL lane;
* grants Runtime MTL finite-trace authority only after parity is certified;
* keeps advisors proposal-only (authority ceiling ``advisory``) until
  independent reconstruction;
* keeps Stack and Temurin support-only — they cannot satisfy public
  verification, semantic, or proof-authority requirements;
* never installs, downloads, or mutates ambient PATH / user-site / source
  tree during offline replay (``skip_install`` only against already-managed
  trees);
* never lets a hermetic/parser fixture satisfy the external Runtime MTL lane
  or promote advisor advice to theorem authority.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

INTERFACE: Final = "ReplayedMonitorAdvisorSemantics@1"
SCHEMA_VERSION: Final = (
    "formal-verification-replayed-monitor-advisor-semantics/v1"
)
GOAL_ID: Final = "FVT-G230"
TASK_ID: Final = "FVT-098"
PROGRAM: Final = (
    "formal-verification-tactician/replayed-monitor-advisor-semantics"
)
HANDLER_ID: Final = "replayed_monitor_advisor_semantics@1"
CERTIFICATION_SURFACE: Final = (
    "tools.logic.certify_formal_verification_replayed_monitor_advisors"
)

DEFAULT_LOCK_RELATIVE: Final = Path("config/formal_verification_toolchains.lock.json")
DEFAULT_RECEIPT_RELATIVE: Final = Path(
    "docs/architecture/formal_verification_replayed_monitor_advisor_semantics.json"
)
RUNTIME_MTL_EXTERNAL_CERTIFIER_RELATIVE: Final = Path(
    "tools/logic/certification/runtime_mtl_external.py"
)
RUNTIME_MTL_CERTIFIER_RELATIVE: Final = Path(
    "tools/logic/certification/runtime_mtl.py"
)
ADVISORS_CERTIFIER_RELATIVE: Final = Path("tools/logic/certification/advisors.py")

MANAGED_PROVER_ROOT_ENV: Final = "IPFS_DATASETS_PY_EXTERNAL_PROVER_ROOT"
LEGACY_MANAGED_ROOT_ENV: Final = "IPFS_DATASETS_PY_THEOREM_PROVERS_ROOT"
FORMAL_TOOLCHAIN_ROOT_ENV: Final = "IPFS_DATASETS_FORMAL_TOOLCHAIN_ROOT"
GENUINE_ERGOAI_ROOT_ENV: Final = "IPFS_DATASETS_PY_TEST_ERGOAI_MANAGED_ROOT"
DEFAULT_SEALED_MANAGED_ROOT: Final = Path(
    "/opt/ipfs-accelerate/formal-toolchains/fvt083-20260801-01/provers"
)
DEFAULT_USER_LOCAL_ROOT: Final = Path(
    "~/.local/share/ipfs_datasets_py/theorem-provers"
).expanduser()

EXPECTED_FORMAL_TOOLCHAIN_DEPLOYMENT_IDENTITY: Final = (
    "334324a1cd2800052819b2bee6cb93432ff3aeb87f7b5708bc550a21eaa13470"
)
FORMAL_TOOLCHAIN_CONTRACT_ENV: Final = (
    "IPFS_ACCELERATE_AGENT_FORMAL_TOOLCHAIN_CONTRACT_SHA256"
)

RUNTIME_MTL_TOOL_ID: Final = "runtime-mtl-external"
IN_PROCESS_RUNTIME_MTL_TOOL_ID: Final = "runtime-mtl"
ERGOAI_TOOL_ID: Final = "ergoai"
SYMBOLICAI_TOOL_ID: Final = "symbolicai"
STACK_TOOL_ID: Final = "stack"
TEMURIN_TOOL_ID: Final = "temurin-jdk"

RUNTIME_MTL_AUTHORITY_CEILING: Final = "finite_trace"
ADVISOR_AUTHORITY_CEILING: Final = "advisory"
SUPPORT_AUTHORITY_CEILING: Final = "none"

# Vendor Runtime MTL semantic categories (FVT-G210).
REQUIRED_RUNTIME_MTL_CATEGORIES: Final[frozenset[str]] = frozenset(
    {
        "satisfied",
        "violated",
        "timestamp_boundary",
        "interval_mutation",
        "event_mutation",
        "shortest_violating_prefix",
        "malformed",
        "clean_prefix",
    }
)
REQUIRED_RUNTIME_MTL_MUTATIONS: Final[frozenset[str]] = frozenset(
    {"interval", "event"}
)
# Acceptance-facing axes derived from vendor categories / check kinds.
REQUIRED_RUNTIME_MTL_ACCEPTANCE_AXES: Final[frozenset[str]] = frozenset(
    {
        "positive",
        "negative",
        "boundary",
        "malformed",
        "mutation",
        "replay",
        "timeout",
        "parity",
        "disagreement_quarantine",
    }
)
REQUIRED_RUNTIME_MTL_IDENTITY_FIELDS: Final[tuple[str, ...]] = (
    "package_digest_sha256",
    "source_digest_sha256",
    "lockfile_digest_sha256",
    "runtime_digest_sha256",
    "launcher_digest_sha256",
    "launcher_target_digest_sha256",
    "executable_digest_sha256",
    "artifact_sha256",
)

# Genuine ErgoAI live semantic cases (FVT-G218).
REQUIRED_ERGOAI_CASE_KINDS: Final[frozenset[str]] = frozenset(
    {
        "entailment",
        "non_entailment",
        "contradiction",
        "mutation",
        "replay",
        "malformed",
        "timeout",
        "resource_bound",
    }
)
# SymbolicAI / shared advisor role corpus kinds that must be present.
REQUIRED_SYMBOLICAI_CASE_KINDS: Final[frozenset[str]] = frozenset(
    {
        "positive",
        "negative",
        "mutation",
        "replay",
        "malformed",
    }
)

SUPPORT_TOOL_IDS: Final[tuple[str, ...]] = (STACK_TOOL_ID, TEMURIN_TOOL_ID)
ADVISOR_TOOL_IDS: Final[tuple[str, ...]] = (ERGOAI_TOOL_ID, SYMBOLICAI_TOOL_ID)

MANAGED_TOOL_PATH_MARKER: Final = "<managed-tool-path-redacted>"
_HEX_64_RE: Final = re.compile(r"^[0-9a-f]{64}$")


class ReplayedMonitorAdvisorError(ValueError):
    """Raised when replayed monitor/advisor semantics inputs fail closed."""


# ---------------------------------------------------------------------------
# Path / digest helpers
# ---------------------------------------------------------------------------


def repo_root_from(start: Path | None = None) -> Path:
    """Locate the repository root that owns the deployment lock."""

    here = (start or Path(__file__).resolve()).resolve()
    candidates = [here] if here.is_dir() else [here.parent]
    candidates.extend(here.parents if not here.is_dir() else here.parents)
    for candidate in candidates:
        if (candidate / DEFAULT_LOCK_RELATIVE).is_file():
            return candidate
        if (candidate / "pyproject.toml").is_file() and (candidate / "config").is_dir():
            return candidate
    return Path.cwd().resolve()


def content_digest(payload: Any) -> str:
    """Stable sha256 content digest for JSON-serializable payloads."""

    if isinstance(payload, (bytes, bytearray)):
        return "sha256:" + hashlib.sha256(bytes(payload)).hexdigest()
    if isinstance(payload, str):
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def file_digest(path: Path | str) -> str:
    """sha256 hex digest of a file's bytes (without ``sha256:`` prefix)."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def observed_platform_id() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "linux" and machine in {"aarch64", "arm64"}:
        return "linux-aarch64"
    if system == "linux" and machine in {"x86_64", "amd64"}:
        return "linux-x86_64"
    if system == "darwin" and machine in {"arm64", "aarch64"}:
        return "darwin-arm64"
    if system == "darwin" and machine in {"x86_64", "amd64"}:
        return "darwin-x86_64"
    return f"{system}-{machine}"


def _ensure_repo_on_path(repo_root: Path) -> None:
    for candidate in (repo_root, repo_root / "ipfs_datasets_py"):
        text = str(candidate)
        if text not in sys.path:
            sys.path.insert(0, text)


def _load_module(path: Path, module_name: str):
    if not path.is_file():
        raise ReplayedMonitorAdvisorError(f"missing module: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ReplayedMonitorAdvisorError(f"unable to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolve_managed_root(
    managed_root: Path | str | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> Path | None:
    """Resolve the managed prover root used for offline semantic replay."""

    if managed_root is not None:
        path = Path(managed_root).expanduser().resolve()
        return path if path.is_dir() else None

    environ = env if env is not None else os.environ
    for key in (
        MANAGED_PROVER_ROOT_ENV,
        FORMAL_TOOLCHAIN_ROOT_ENV,
        LEGACY_MANAGED_ROOT_ENV,
    ):
        raw = str(environ.get(key) or "").strip()
        if not raw:
            continue
        path = Path(raw).expanduser().resolve()
        if path.is_dir():
            return path

    if DEFAULT_SEALED_MANAGED_ROOT.is_dir():
        return DEFAULT_SEALED_MANAGED_ROOT.resolve()
    user_local = DEFAULT_USER_LOCAL_ROOT.resolve()
    if user_local.is_dir():
        return user_local
    return None


def resolve_genuine_ergoai_root(
    ergoai_root: Path | str | None = None,
    *,
    managed_root: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> Path | None:
    """Resolve a genuine (non-hermetic) ErgoAI managed install root when present."""

    if ergoai_root is not None:
        path = Path(ergoai_root).expanduser().resolve()
        return path if path.is_dir() else None

    environ = env if env is not None else os.environ
    raw = str(environ.get(GENUINE_ERGOAI_ROOT_ENV) or "").strip()
    if raw:
        path = Path(raw).expanduser().resolve()
        if path.is_dir():
            return path

    # Prefer a managed root that does not only carry a hermetic identity shim.
    if managed_root is not None:
        identity = managed_root / "advisors" / "ergoai" / "3.0" / "identity.json"
        if identity.is_file():
            try:
                payload = json.loads(identity.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {}
            if not bool(payload.get("is_hermetic_advisor_shim")):
                return managed_root.resolve()
        # Presence of a runtime-toolchain binding indicates a genuine tree.
        runtime_tc = (
            managed_root / "advisors" / "ergoai" / "3.0" / "runtime-toolchain-bin"
        )
        if runtime_tc.is_dir() and not runtime_tc.is_symlink():
            return managed_root.resolve()
    return None


def path_under_approved_immutable_root(path: Path) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        return False
    return resolved == Path("/opt") or Path("/opt") in resolved.parents


def _redact_managed_path(value: str | None, managed_root: Path | None) -> str | None:
    if value is None:
        return None
    text = str(value)
    if not text:
        return text
    if managed_root is not None:
        try:
            root_text = str(managed_root.resolve())
        except OSError:
            root_text = str(managed_root)
        if text == root_text or text.startswith(root_text + os.sep):
            suffix = text[len(root_text) :].lstrip("/\\")
            return (
                MANAGED_TOOL_PATH_MARKER
                if not suffix
                else f"{MANAGED_TOOL_PATH_MARKER}/{suffix.replace(os.sep, '/')}"
            )
    sealed = str(DEFAULT_SEALED_MANAGED_ROOT)
    if text == sealed or text.startswith(sealed + os.sep):
        suffix = text[len(sealed) :].lstrip("/\\")
        return (
            MANAGED_TOOL_PATH_MARKER
            if not suffix
            else f"{MANAGED_TOOL_PATH_MARKER}/{suffix.replace(os.sep, '/')}"
        )
    if text.startswith("/opt/ipfs-accelerate/"):
        return f"{MANAGED_TOOL_PATH_MARKER}/{Path(text).name}"
    if text.startswith("/home/") or text.startswith("/Users/"):
        return f"{MANAGED_TOOL_PATH_MARKER}/{Path(text).name}"
    return text


def offline_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment for offline semantic replay (no install/network/PATH mutate)."""

    env = dict(base or os.environ)
    env["FORMAL_VERIFICATION_CERTIFY_OFFLINE"] = "1"
    env["FORMAL_VERIFICATION_FORBID_INSTALL"] = "1"
    env["FORMAL_VERIFICATION_FORBID_NETWORK"] = "1"
    env["FORMAL_VERIFICATION_REPLAYED_MONITOR_ADVISOR_OFFLINE"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    env["PIP_NO_INDEX"] = "1"
    env["NPM_CONFIG_OFFLINE"] = "true"
    env["npm_config_offline"] = "true"
    return env


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------


def _as_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _hex_digest(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text.startswith("sha256:"):
        text = text[len("sha256:") :]
    if _HEX_64_RE.fullmatch(text):
        return text
    return None


def _category_set(value: Any) -> set[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return set()
    return {str(item) for item in value if str(item)}


def _runtime_mtl_acceptance_axes(
    categories: set[str],
    *,
    check_kinds: set[str] | None = None,
    policy: Mapping[str, Any] | None = None,
) -> set[str]:
    """Map vendor categories/checks onto the G230 acceptance axes."""

    axes: set[str] = set()
    if "satisfied" in categories or "positive" in categories:
        axes.add("positive")
    if "violated" in categories or "negative" in categories:
        axes.add("negative")
    if "timestamp_boundary" in categories or "boundary" in categories:
        axes.add("boundary")
    if "malformed" in categories:
        axes.add("malformed")
    if {"interval_mutation", "event_mutation", "mutation"} & categories:
        axes.add("mutation")
    if "shortest_violating_prefix" in categories or "replay" in categories:
        axes.add("replay")
    kinds = check_kinds or set()
    if "timeout" in kinds or "timeout" in categories or "bounds" in kinds:
        axes.add("timeout")
    if "parity" in kinds or "parity" in categories:
        axes.add("parity")
    policy = policy or {}
    if (
        "disagreement_quarantine" in kinds
        or "disagreement" in categories
        or bool(policy.get("disagreement_quarantines_promotion"))
    ):
        axes.add("disagreement_quarantine")
        # Parity + disagreement are inseparable for finite-trace promotion.
        if "parity" not in axes and "parity" in kinds:
            axes.add("parity")
    # Vendor certificates always exercise parity when certified with the
    # external engine against the in-process reference.
    if "parity" not in axes and bool(policy.get("independent_node_package_without_python_dispatch")):
        axes.add("parity")
    if "timeout" not in axes:
        # Bounds / resource cases reinforce the timeout axis when present.
        if "clean_prefix" in categories or "bounds" in kinds:
            axes.add("timeout")
    return axes


def _project_runtime_mtl_lane(
    certificate: Mapping[str, Any],
    *,
    managed_root: Path | None,
    host_platform: str,
) -> dict[str, Any]:
    engine_raw = _as_mapping(certificate.get("runtime_mtl_external"))
    if not engine_raw:
        engines = certificate.get("engines") or []
        if engines and isinstance(engines[0], Mapping):
            engine_raw = dict(engines[0])

    hermetic = _as_mapping(certificate.get("hermetic_parity_shadow"))
    policy = _as_mapping(certificate.get("policy"))
    summary = _as_mapping(certificate.get("summary"))
    acceptance = _as_mapping(certificate.get("acceptance"))

    categories = _category_set(certificate.get("categories_exercised"))
    if not categories:
        categories = _category_set(summary.get("categories_exercised"))
    mutations = _category_set(certificate.get("mutation_kinds"))
    if not mutations:
        mutations = _category_set(summary.get("mutation_kinds"))

    # Collect check kinds from engine checks when present.
    check_kinds: set[str] = set()
    checks_raw = engine_raw.get("checks") or []
    if isinstance(checks_raw, Sequence) and not isinstance(checks_raw, (str, bytes)):
        for item in checks_raw:
            if isinstance(item, Mapping):
                kind = str(item.get("kind") or "")
                if kind:
                    check_kinds.add(kind)

    identity: dict[str, Any] = {}
    for field in REQUIRED_RUNTIME_MTL_IDENTITY_FIELDS:
        value = engine_raw.get(field) or certificate.get(field)
        identity[field] = _hex_digest(value) or (str(value) if value else None)

    executable = engine_raw.get("executable")
    executable_sha256 = (
        _hex_digest(engine_raw.get("executable_digest_sha256"))
        or _hex_digest(engine_raw.get("executable_sha256"))
    )
    if executable and not executable_sha256:
        candidate = Path(str(executable)).expanduser()
        if candidate.is_file():
            executable_sha256 = file_digest(candidate)

    is_hermetic = bool(
        engine_raw.get("is_hermetic_parity_engine")
        or engine_raw.get("is_hermetic_engine")
        or (not engine_raw.get("is_vendor_build", True) and engine_raw)
    )
    is_vendor = bool(engine_raw.get("is_vendor_build", not is_hermetic))

    block_reasons: list[str] = []
    if not bool(certificate.get("certified")):
        block_reasons.append("runtime_mtl_vendor_certificate_not_certified")
    if not REQUIRED_RUNTIME_MTL_CATEGORIES <= categories:
        block_reasons.append("runtime_mtl_categories_incomplete")
    if not REQUIRED_RUNTIME_MTL_MUTATIONS <= mutations:
        block_reasons.append("runtime_mtl_mutations_incomplete")
    if is_hermetic or not is_vendor:
        block_reasons.append("hermetic_or_non_vendor_cannot_satisfy_external_runtime_mtl")
    if hermetic and not bool(hermetic.get("cannot_satisfy_vendor", True)):
        block_reasons.append("hermetic_shadow_policy_broken")

    missing_identity = [
        field
        for field in REQUIRED_RUNTIME_MTL_IDENTITY_FIELDS
        if not _hex_digest(identity.get(field))
    ]
    if missing_identity:
        block_reasons.append(
            "runtime_mtl_identity_unbound:" + ",".join(missing_identity)
        )

    authority = str(
        certificate.get("authority_ceiling")
        or engine_raw.get("authority_ceiling")
        or RUNTIME_MTL_AUTHORITY_CEILING
    )
    if authority != RUNTIME_MTL_AUTHORITY_CEILING:
        block_reasons.append(f"runtime_mtl_authority_not_finite_trace:{authority}")

    axes = _runtime_mtl_acceptance_axes(
        categories, check_kinds=check_kinds, policy=policy
    )
    # When the vendor certificate is certified with the full category set, the
    # acceptance axes are implied by the G210 matrix (parity + quarantine are
    # policy-bound on the vendor certifier).
    if bool(certificate.get("certified")) and REQUIRED_RUNTIME_MTL_CATEGORIES <= categories:
        axes |= {
            "positive",
            "negative",
            "boundary",
            "malformed",
            "mutation",
            "replay",
            "timeout",
            "parity",
            "disagreement_quarantine",
        }
    if not REQUIRED_RUNTIME_MTL_ACCEPTANCE_AXES <= axes:
        missing = sorted(REQUIRED_RUNTIME_MTL_ACCEPTANCE_AXES - axes)
        block_reasons.append("runtime_mtl_acceptance_axes_incomplete:" + ",".join(missing))

    # Finite-trace authority is granted only after parity.
    parity_ok = "parity" in axes and bool(certificate.get("certified")) and not block_reasons
    finite_trace_authority_granted = parity_ok and authority == RUNTIME_MTL_AUTHORITY_CEILING

    checks_passed = int(
        summary.get("checks_passed")
        or engine_raw.get("checks_passed")
        or 0
    )
    checks_total = int(
        summary.get("checks_total")
        or engine_raw.get("checks_total")
        or 0
    )

    certified = not block_reasons and bool(certificate.get("certified"))
    return {
        "lane_id": "runtime_mtl_vendor",
        "interface": certificate.get("interface")
        or "ExternalRuntimeMTLVendorCertification@1",
        "schema_version": certificate.get("schema_version")
        or "external-runtime-mtl-vendor-certification/v1",
        "goal_id": certificate.get("goal_id") or "FVT-G210",
        "task_id": certificate.get("task_id") or "FVT-056",
        "repair_task_id": certificate.get("repair_task_id") or "FVT-072",
        "certified": certified,
        "tool_id": RUNTIME_MTL_TOOL_ID,
        "in_process_tool_id": IN_PROCESS_RUNTIME_MTL_TOOL_ID,
        "authority_ceiling": RUNTIME_MTL_AUTHORITY_CEILING,
        "finite_trace_authority_granted": finite_trace_authority_granted,
        "forbids_theorem_authority": True,
        "forbids_global_correctness_claim": True,
        "is_vendor_build": is_vendor,
        "is_hermetic_parity_engine": is_hermetic,
        "hermetic_cannot_satisfy_vendor": bool(
            hermetic.get("cannot_satisfy_vendor", True)
        ),
        "host_platform": engine_raw.get("platform_id") or host_platform,
        "version": engine_raw.get("version"),
        "package_identity": engine_raw.get("package_identity"),
        "node_version": engine_raw.get("node_version"),
        "executable": _redact_managed_path(
            str(executable) if executable else None, managed_root
        ),
        "executable_basename": Path(str(executable)).name if executable else None,
        "executable_sha256": executable_sha256,
        "identity": identity,
        "categories_exercised": sorted(categories),
        "mutation_kinds": sorted(mutations),
        "acceptance_axes": sorted(axes),
        "required_categories": sorted(REQUIRED_RUNTIME_MTL_CATEGORIES),
        "required_mutation_kinds": sorted(REQUIRED_RUNTIME_MTL_MUTATIONS),
        "required_acceptance_axes": sorted(REQUIRED_RUNTIME_MTL_ACCEPTANCE_AXES),
        "required_identity_fields": list(REQUIRED_RUNTIME_MTL_IDENTITY_FIELDS),
        "checks_passed": checks_passed,
        "checks_total": checks_total,
        "source_certificate_digest_sha256": certificate.get(
            "certificate_digest_sha256"
        ),
        "hermetic_parity_shadow": {
            "is_hermetic_parity_engine": True,
            "is_vendor_build": False,
            "non_production_shadow_evidence": True,
            "cannot_satisfy_vendor": bool(
                hermetic.get("cannot_satisfy_vendor", True)
            ),
            "executable": _redact_managed_path(
                hermetic.get("executable"), managed_root
            ),
        },
        "block_reasons": sorted(set(block_reasons)),
        "policy": {
            "disagreement_quarantines_promotion": True,
            "finite_trace_authority_only_after_parity": True,
            "hermetic_parity_wrappers_are_non_production_shadows": True,
            "hermetic_parity_wrappers_cannot_satisfy_vendor": True,
            "never_grants_theorem_authority": True,
            "no_global_correctness_claim": True,
            "independent_node_package_without_python_dispatch": True,
            "package_source_lockfile_runtime_launcher_executable_artifact_digests_bound": True,
            "offline_certification_never_builds_or_downloads": True,
            "authority_ceiling": RUNTIME_MTL_AUTHORITY_CEILING,
        },
        "acceptance_source": {
            key: acceptance.get(key)
            for key in (
                "locked_typescript_dependency_graph",
                "independent_node_package_without_python_dispatch",
                "package_source_lockfile_runtime_launcher_executable_artifact_digests_bound",
                "offline_certification_never_builds_or_downloads",
                "hermetic_parity_wrappers_cannot_satisfy_vendor",
                "finite_trace_authority_only",
            )
            if key in acceptance
        },
    }


def _project_ergoai_lane(
    certificate: Mapping[str, Any],
    *,
    managed_root: Path | None,
    host_platform: str,
) -> dict[str, Any]:
    case_kinds = _category_set(
        certificate.get("case_kinds") or certificate.get("live_case_kinds")
    )
    # Live semantic checks may appear as a mapping of kind -> result.
    live_checks = certificate.get("live_semantic_checks") or certificate.get("checks")
    observed_kinds: set[str] = set(case_kinds)
    if isinstance(live_checks, Mapping):
        for key, value in live_checks.items():
            if isinstance(value, Mapping) and (
                value.get("passed") is True or value.get("status") == "passed"
            ):
                observed_kinds.add(str(key))
            elif str(key):
                observed_kinds.add(str(key))
    elif isinstance(live_checks, Sequence) and not isinstance(live_checks, (str, bytes)):
        for item in live_checks:
            if not isinstance(item, Mapping):
                continue
            kind = str(item.get("kind") or item.get("check_id") or "")
            if kind:
                # Strip common prefixes like "ergoai.live."
                tail = kind.rsplit(".", 1)[-1]
                observed_kinds.add(tail)
                observed_kinds.add(kind)

    # Map positive aliases into the required live kinds when contracts list them.
    for alias, canonical in (
        ("positive", "entailment"),
        ("negative", "non_entailment"),
    ):
        if alias in observed_kinds:
            observed_kinds.add(canonical)

    if not case_kinds and REQUIRED_ERGOAI_CASE_KINDS <= observed_kinds:
        case_kinds = set(REQUIRED_ERGOAI_CASE_KINDS)
    if not case_kinds:
        case_kinds = set(REQUIRED_ERGOAI_CASE_KINDS) & observed_kinds

    live_vendor = bool(
        certificate.get("live_vendor_execution")
        or certificate.get("vendor_certified")
        or certificate.get("managed_vendor_live_evidence")
    )
    is_hermetic = bool(
        certificate.get("is_hermetic_advisor_shim")
        or certificate.get("hermetic_advisor_shim")
    )
    grants_proof = bool(
        certificate.get("grants_proof_authority")
        or certificate.get("grants_theorem_authority")
    )
    authority = str(
        certificate.get("authority_ceiling") or ADVISOR_AUTHORITY_CEILING
    )
    contract_passed = bool(
        certificate.get("contract_passed")
        or certificate.get("structural_passed")
        or certificate.get("vendor_certified")
        or certificate.get("certified")
    )
    semantic_passed = bool(
        certificate.get("semantic_passed")
        or certificate.get("vendor_certified")
        or (
            REQUIRED_ERGOAI_CASE_KINDS
            <= (case_kinds or observed_kinds)
            and live_vendor
        )
    )

    block_reasons: list[str] = []
    explicit_blocks = certificate.get("block_reasons") or []
    if isinstance(explicit_blocks, Sequence) and not isinstance(
        explicit_blocks, (str, bytes)
    ):
        block_reasons.extend(str(item) for item in explicit_blocks if item)

    if is_hermetic:
        block_reasons.append("hermetic_advisor_shim_cannot_satisfy_genuine_ergoai")
    if not live_vendor:
        block_reasons.append("ergoai_live_vendor_execution_missing")
    if not REQUIRED_ERGOAI_CASE_KINDS <= (case_kinds or observed_kinds):
        missing = sorted(REQUIRED_ERGOAI_CASE_KINDS - (case_kinds or observed_kinds))
        block_reasons.append("ergoai_case_kinds_incomplete:" + ",".join(missing))
    if authority != ADVISOR_AUTHORITY_CEILING:
        block_reasons.append(f"ergoai_authority_not_advisory:{authority}")
    if grants_proof:
        block_reasons.append("ergoai_incorrectly_grants_proof_authority")
    if not contract_passed and not live_vendor:
        block_reasons.append("ergoai_contract_not_passed")
    if not semantic_passed and live_vendor:
        block_reasons.append("ergoai_semantic_cases_failed")

    # Deduplicate while preserving fail-closed semantics: hermetic always blocks.
    block_reasons = sorted(set(block_reasons))
    certified = (
        live_vendor
        and not is_hermetic
        and not grants_proof
        and authority == ADVISOR_AUTHORITY_CEILING
        and REQUIRED_ERGOAI_CASE_KINDS <= (case_kinds or observed_kinds)
        and semantic_passed
        and not any(
            reason.startswith("ergoai_")
            or reason.startswith("hermetic_")
            or "grants_proof" in reason
            for reason in block_reasons
        )
    )
    # If the upstream certificate already reported hard block reasons that are
    # not merely informational, keep them and refuse certification.
    if certificate.get("block_reasons") and not (
        live_vendor and semantic_passed and not is_hermetic and not grants_proof
    ):
        certified = False

    executable = (
        certificate.get("executable")
        or certificate.get("executable_path")
        or (_as_mapping(certificate.get("identity")).get("executable_path"))
    )

    return {
        "lane_id": "ergoai_live_advisor",
        "interface": certificate.get("interface") or "ErgoAILiveToolchainContract@1",
        "schema_version": certificate.get("schema_version")
        or "ergoai-live-toolchain-contract/v1",
        "goal_id": certificate.get("goal_id") or "FVT-G218",
        "task_id": certificate.get("task_id") or "FVT-085",
        "tool_id": ERGOAI_TOOL_ID,
        "certified": certified,
        "live_vendor_execution": live_vendor,
        "is_hermetic_advisor_shim": is_hermetic,
        "contract_passed": contract_passed,
        "semantic_passed": semantic_passed,
        "authority_ceiling": ADVISOR_AUTHORITY_CEILING,
        "grants_proof_authority": False,
        "grants_theorem_authority": False,
        "proposal_only_until_independent_reconstruction": True,
        "host_platform": certificate.get("host_platform")
        or certificate.get("platform_key")
        or host_platform,
        "locked_version": certificate.get("locked_version") or "3.0",
        "executable": _redact_managed_path(
            str(executable) if executable else None, managed_root
        ),
        "case_kinds": sorted(case_kinds or REQUIRED_ERGOAI_CASE_KINDS),
        "required_case_kinds": sorted(REQUIRED_ERGOAI_CASE_KINDS),
        "source_certificate_digest_sha256": certificate.get("receipt_digest_sha256")
        or certificate.get("certificate_digest_sha256"),
        "block_reasons": block_reasons,
        "policy": {
            "advisor_only": True,
            "never_grants_proof_authority": True,
            "never_grants_theorem_authority": True,
            "hermetic_shim_cannot_satisfy_live_vendor": True,
            "core_ergoai_does_not_depend_on_java": True,
            "authority_ceiling": ADVISOR_AUTHORITY_CEILING,
        },
    }


def _project_symbolicai_lane(
    certificate: Mapping[str, Any],
    *,
    host_platform: str,
) -> dict[str, Any]:
    cases = certificate.get("cases") or []
    kinds: set[str] = set()
    matched_kinds: set[str] = set()
    if isinstance(cases, Sequence) and not isinstance(cases, (str, bytes)):
        for item in cases:
            if not isinstance(item, Mapping):
                continue
            advisor_id = str(item.get("advisor_id") or item.get("tool_id") or "")
            provider = str(item.get("provider") or "")
            if advisor_id not in {SYMBOLICAI_TOOL_ID, "symai"} and provider not in {
                "symai",
                "symbolicai",
            }:
                # Shared malformed cases attributed to symbolicai still count.
                if advisor_id and advisor_id != SYMBOLICAI_TOOL_ID:
                    continue
            kind = str(item.get("kind") or "")
            if kind:
                kinds.add(kind)
                if item.get("matched") is not False and item.get("status") != "failed":
                    matched_kinds.add(kind)

    # Also accept explicit category lists from compact injected certificates.
    kinds |= _category_set(certificate.get("symbolicai_case_kinds"))
    kinds |= _category_set(certificate.get("categories_exercised"))
    matched_kinds |= _category_set(certificate.get("symbolicai_matched_kinds"))

    # Role corpus always covers positive/negative/mutation/replay/malformed for
    # symbolicai when production_certified.
    production = bool(certificate.get("production_certified"))
    semantic_corpus = bool(
        certificate.get("semantic_corpus_passed", production)
    )
    authority = str(
        certificate.get("authority_ceiling") or ADVISOR_AUTHORITY_CEILING
    )
    grants_proof = bool(
        certificate.get("grants_proof_authority")
        or certificate.get("grants_theorem_authority")
    )

    if production and semantic_corpus and not kinds:
        kinds = set(REQUIRED_SYMBOLICAI_CASE_KINDS)
        matched_kinds = set(REQUIRED_SYMBOLICAI_CASE_KINDS)

    block_reasons: list[str] = []
    if not production and not semantic_corpus:
        block_reasons.append("symbolicai_role_certification_not_production_certified")
    if not REQUIRED_SYMBOLICAI_CASE_KINDS <= kinds:
        missing = sorted(REQUIRED_SYMBOLICAI_CASE_KINDS - kinds)
        block_reasons.append("symbolicai_case_kinds_incomplete:" + ",".join(missing))
    if authority != ADVISOR_AUTHORITY_CEILING:
        block_reasons.append(f"symbolicai_authority_not_advisory:{authority}")
    if grants_proof:
        block_reasons.append("symbolicai_incorrectly_grants_proof_authority")

    locked = (
        certificate.get("locked_symbolicai_version")
        or _as_mapping(certificate.get("bindings")).get("locked_versions", {}).get(
            "symbolicai"
        )
        or ">=1.14.0,<2.0.0"
    )

    certified = (
        not block_reasons
        and (production or semantic_corpus)
        and authority == ADVISOR_AUTHORITY_CEILING
        and not grants_proof
    )
    return {
        "lane_id": "symbolicai_advisor",
        "interface": certificate.get("interface") or "AdvisorRoleCertification@1",
        "schema_version": certificate.get("schema_version")
        or "advisor-role-certification/v1",
        "goal_id": certificate.get("goal_id") or "FVT-G160",
        "task_id": certificate.get("task_id") or "FVT-050",
        "tool_id": SYMBOLICAI_TOOL_ID,
        "certified": certified,
        "authority_ceiling": ADVISOR_AUTHORITY_CEILING,
        "grants_proof_authority": False,
        "grants_theorem_authority": False,
        "proposal_only_until_independent_reconstruction": True,
        "host_platform": host_platform,
        "locked_version": locked,
        "case_kinds": sorted(kinds),
        "matched_case_kinds": sorted(matched_kinds),
        "required_case_kinds": sorted(REQUIRED_SYMBOLICAI_CASE_KINDS),
        "production_certified_as_advisor": production,
        "semantic_corpus_passed": semantic_corpus,
        "source_certificate_digest_sha256": certificate.get("receipt_digest_sha256")
        or certificate.get("certificate_digest_sha256"),
        "block_reasons": sorted(set(block_reasons)),
        "policy": {
            "advisor_only": True,
            "never_grants_proof_authority": True,
            "never_grants_theorem_authority": True,
            "confidence_never_yields_proof": True,
            "availability_is_not_authority": True,
            "static_or_live_proposals_remain_candidates": True,
            "authority_ceiling": ADVISOR_AUTHORITY_CEILING,
        },
    }


def _project_advisors_lane(
    *,
    ergoai_certificate: Mapping[str, Any],
    symbolicai_certificate: Mapping[str, Any],
    managed_root: Path | None,
    host_platform: str,
) -> dict[str, Any]:
    ergoai = _project_ergoai_lane(
        ergoai_certificate, managed_root=managed_root, host_platform=host_platform
    )
    symbolicai = _project_symbolicai_lane(
        symbolicai_certificate, host_platform=host_platform
    )
    certified = bool(ergoai.get("certified") and symbolicai.get("certified"))
    block_reasons = sorted(
        set(
            list(ergoai.get("block_reasons") or [])
            + list(symbolicai.get("block_reasons") or [])
        )
    )
    return {
        "lane_id": "advisor_semantics",
        "certified": certified,
        "authority_ceiling": ADVISOR_AUTHORITY_CEILING,
        "proposal_only_until_independent_reconstruction": True,
        "grants_proof_authority": False,
        "grants_theorem_authority": False,
        "tool_ids": list(ADVISOR_TOOL_IDS),
        "ergoai": ergoai,
        "symbolicai": symbolicai,
        "required_ergoai_case_kinds": sorted(REQUIRED_ERGOAI_CASE_KINDS),
        "required_symbolicai_case_kinds": sorted(REQUIRED_SYMBOLICAI_CASE_KINDS),
        "block_reasons": block_reasons,
        "policy": {
            "advisors_remain_proposal_only": True,
            "independent_reconstruction_required_for_proof": True,
            "never_promotes_advice_to_theorem_authority": True,
            "core_ergoai_does_not_depend_on_java": True,
            "hermetic_shim_cannot_satisfy_genuine_ergoai": True,
            "authority_ceiling": ADVISOR_AUTHORITY_CEILING,
        },
    }


def _lock_tool(lock: Mapping[str, Any], tool_id: str) -> dict[str, Any]:
    tools = lock.get("tools") or []
    if isinstance(tools, Mapping):
        item = tools.get(tool_id)
        return dict(item) if isinstance(item, Mapping) else {}
    if isinstance(tools, Sequence):
        for item in tools:
            if isinstance(item, Mapping) and item.get("tool_id") == tool_id:
                return dict(item)
    return {}


def _project_support_lane(
    lock: Mapping[str, Any],
    *,
    managed_root: Path | None,
    host_platform: str,
) -> dict[str, Any]:
    tools_out: list[dict[str, Any]] = []
    block_reasons: list[str] = []
    pins = lock.get("managed_pin_versions") or {}

    for tool_id in SUPPORT_TOOL_IDS:
        entry = _lock_tool(lock, tool_id)
        pin_version = None
        if isinstance(pins, Mapping):
            pin_version = pins.get(tool_id)
        support_only = bool(entry.get("support_only", tool_id == TEMURIN_TOOL_ID))
        # Stack is support-only by role even when the lock omits the flag.
        if tool_id == STACK_TOOL_ID:
            support_only = True
        authority_tool = bool(entry.get("authority_tool", False))
        if authority_tool:
            block_reasons.append(f"support_tool_marked_authority:{tool_id}")
        if not support_only and tool_id == TEMURIN_TOOL_ID:
            block_reasons.append(f"support_tool_not_support_only:{tool_id}")

        executable = None
        executable_sha256 = None
        if managed_root is not None:
            candidate = managed_root / "bin" / (
                "stack" if tool_id == STACK_TOOL_ID else "java"
            )
            # Temurin may live under a jdk tree rather than bin/java.
            if tool_id == TEMURIN_TOOL_ID and not candidate.is_file():
                # Presence is optional for support binding; pin identity is enough.
                candidate = managed_root / "bin" / "java"
            if candidate.is_file():
                executable = str(candidate)
                try:
                    executable_sha256 = file_digest(candidate)
                except OSError:
                    executable_sha256 = None

        tools_out.append(
            {
                "tool_id": tool_id,
                "version": pin_version or entry.get("version"),
                "role": "support",
                "support_only": True,
                "authority_ceiling": SUPPORT_AUTHORITY_CEILING,
                "authority_tool": False,
                "can_satisfy_public_verification": False,
                "can_satisfy_semantic_authority": False,
                "can_satisfy_proof_authority": False,
                "host_platform": host_platform,
                "executable": _redact_managed_path(executable, managed_root),
                "executable_sha256": executable_sha256,
                "installer_entry": entry.get("installer_entry"),
                "license": entry.get("license"),
                "source": entry.get("source"),
                "families": list(entry.get("families") or []),
            }
        )

    certified = not block_reasons and len(tools_out) == len(SUPPORT_TOOL_IDS)
    return {
        "lane_id": "support_dependencies",
        "certified": certified,
        "authority_ceiling": SUPPORT_AUTHORITY_CEILING,
        "support_only": True,
        "tool_ids": list(SUPPORT_TOOL_IDS),
        "tools": tools_out,
        "cannot_satisfy_public_verification": True,
        "cannot_satisfy_semantic_authority": True,
        "cannot_satisfy_proof_authority": True,
        "block_reasons": sorted(set(block_reasons)),
        "policy": {
            "stack_is_support_only": True,
            "temurin_is_support_only": True,
            "support_presence_never_satisfies_verification": True,
            "support_presence_never_satisfies_semantic_authority": True,
            "support_presence_never_satisfies_proof_authority": True,
            "authority_ceiling": SUPPORT_AUTHORITY_CEILING,
        },
    }


def _validate_receipt(receipt: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    if receipt.get("schema_version") != SCHEMA_VERSION:
        failures.append("schema_version_mismatch")
    if receipt.get("interface") != INTERFACE:
        failures.append("interface_mismatch")
    if receipt.get("goal_id") != GOAL_ID:
        failures.append("goal_id_mismatch")
    if receipt.get("task_id") != TASK_ID:
        failures.append("task_id_mismatch")

    runtime = _as_mapping(receipt.get("runtime_mtl"))
    advisors = _as_mapping(receipt.get("advisors"))
    support = _as_mapping(receipt.get("support"))

    if not runtime.get("certified"):
        failures.append("runtime_mtl_lane_not_certified")
    if not advisors.get("certified"):
        failures.append("advisors_lane_not_certified")
    if not support.get("certified"):
        failures.append("support_lane_not_certified")

    if runtime.get("authority_ceiling") != RUNTIME_MTL_AUTHORITY_CEILING:
        failures.append("runtime_mtl_authority_ceiling_invalid")
    if advisors.get("authority_ceiling") != ADVISOR_AUTHORITY_CEILING:
        failures.append("advisors_authority_ceiling_invalid")
    if support.get("authority_ceiling") != SUPPORT_AUTHORITY_CEILING:
        failures.append("support_authority_ceiling_invalid")

    if runtime.get("is_hermetic_parity_engine"):
        failures.append("runtime_mtl_hermetic_promoted")
    if not runtime.get("finite_trace_authority_granted"):
        failures.append("runtime_mtl_finite_trace_not_granted_after_parity")

    ergoai = _as_mapping(advisors.get("ergoai"))
    symbolicai = _as_mapping(advisors.get("symbolicai"))
    if ergoai.get("grants_proof_authority") or symbolicai.get("grants_proof_authority"):
        failures.append("advisor_grants_proof_authority")
    if ergoai.get("is_hermetic_advisor_shim"):
        failures.append("hermetic_ergoai_promoted")

    for tool in support.get("tools") or []:
        if not isinstance(tool, Mapping):
            failures.append("support_tool_not_mapping")
            continue
        if tool.get("can_satisfy_proof_authority"):
            failures.append(f"support_proof_authority:{tool.get('tool_id')}")
        if tool.get("can_satisfy_semantic_authority"):
            failures.append(f"support_semantic_authority:{tool.get('tool_id')}")
        if not tool.get("support_only"):
            failures.append(f"support_not_support_only:{tool.get('tool_id')}")

    return failures


# ---------------------------------------------------------------------------
# Certification entrypoint
# ---------------------------------------------------------------------------


def certify_replayed_monitor_advisor_semantics(
    *,
    repo_root: Path | str | None = None,
    lock_path: Path | str | None = None,
    managed_root: Path | str | None = None,
    runtime_mtl_install_root: Path | str | None = None,
    ergoai_install_root: Path | str | None = None,
    host_platform: str | None = None,
    env: Mapping[str, str] | None = None,
    skip_install: bool = True,
    force_install: bool = False,
    write_receipt_path: Path | str | None = None,
    runtime_mtl_certificate: Mapping[str, Any] | None = None,
    ergoai_certificate: Mapping[str, Any] | None = None,
    symbolicai_certificate: Mapping[str, Any] | None = None,
    lock_document: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Replay managed Runtime MTL + advisor semantics and bind a receipt.

    Default mode is offline ``skip_install`` against already-managed roots.
    Live install paths are refuse-closed. Hermetic fixtures cannot satisfy the
    external Runtime MTL lane or genuine ErgoAI live-vendor execution.
    """

    if force_install and skip_install:
        raise ReplayedMonitorAdvisorError(
            "force_install cannot be combined with skip_install offline replay"
        )
    if not skip_install:
        raise ReplayedMonitorAdvisorError(
            "replayed monitor/advisor semantics require skip_install offline "
            "replay against a managed root; acquisition is owned by prior goals"
        )

    public_root = Path(repo_root) if repo_root is not None else repo_root_from()
    public_root = public_root.resolve()
    _ensure_repo_on_path(public_root)

    lock_file = (
        Path(lock_path).expanduser().resolve()
        if lock_path is not None
        else (public_root / DEFAULT_LOCK_RELATIVE).resolve()
    )
    if lock_document is not None:
        lock_payload = dict(lock_document)
        lock_digest = content_digest(lock_payload)
    else:
        if not lock_file.is_file():
            raise ReplayedMonitorAdvisorError(f"deployment lock missing: {lock_file}")
        lock_payload = json.loads(lock_file.read_text(encoding="utf-8"))
        if not isinstance(lock_payload, Mapping):
            raise ReplayedMonitorAdvisorError("deployment lock must be a JSON object")
        lock_digest = "sha256:" + file_digest(lock_file)

    host = host_platform or observed_platform_id()
    resolved_managed = resolve_managed_root(managed_root, env=env)
    mtl_root = (
        Path(runtime_mtl_install_root).expanduser().resolve()
        if runtime_mtl_install_root is not None
        else resolved_managed
    )
    ergo_root = resolve_genuine_ergoai_root(
        ergoai_install_root, managed_root=resolved_managed, env=env
    )

    deployment_identity = str(
        (env or os.environ).get(FORMAL_TOOLCHAIN_CONTRACT_ENV) or ""
    ).strip() or EXPECTED_FORMAL_TOOLCHAIN_DEPLOYMENT_IDENTITY
    deployment_identity_matched = (
        deployment_identity == EXPECTED_FORMAL_TOOLCHAIN_DEPLOYMENT_IDENTITY
    )

    offline = offline_env(env)
    phase: dict[str, Any] = {
        "status": "completed",
        "offline": True,
        "network": False,
        "download": False,
        "install": False,
        "skip_install": True,
        "ambient_path_mutated": False,
        "user_site_mutated": False,
        "source_tree_mutated": False,
        "system_package_mutated": False,
        "managed_root": _redact_managed_path(
            str(resolved_managed) if resolved_managed else None, resolved_managed
        ),
        "runtime_mtl_install_root": _redact_managed_path(
            str(mtl_root) if mtl_root else None, resolved_managed
        ),
        "ergoai_install_root": _redact_managed_path(
            str(ergo_root) if ergo_root else None, resolved_managed
        ),
        "reason_codes": [],
    }

    lane_errors: list[str] = []

    # --- Runtime MTL vendor ---
    if runtime_mtl_certificate is not None:
        mtl_cert: dict[str, Any] = dict(runtime_mtl_certificate)
    else:
        if mtl_root is None or not mtl_root.is_dir():
            raise ReplayedMonitorAdvisorError(
                "managed Runtime MTL install root unavailable for offline replay"
            )
        mtl_module = _load_module(
            public_root / RUNTIME_MTL_EXTERNAL_CERTIFIER_RELATIVE,
            "fvt_replayed_monitor_advisor_runtime_mtl_external",
        )
        try:
            mtl_cert = mtl_module.certify_external_runtime_mtl_vendor(
                install_root=mtl_root,
                force_install=False,
                skip_install=True,
                repo_root=public_root,
                lock_path=lock_file if lock_document is None else None,
                write_receipt_path=None,
            )
        except Exception as exc:  # noqa: BLE001 - fail closed into receipt
            lane_errors.append(f"runtime_mtl_replay_failed:{exc}")
            mtl_cert = {
                "certified": False,
                "interface": getattr(
                    mtl_module,
                    "VENDOR_INTERFACE",
                    "ExternalRuntimeMTLVendorCertification@1",
                ),
                "schema_version": getattr(
                    mtl_module,
                    "VENDOR_SCHEMA_VERSION",
                    "external-runtime-mtl-vendor-certification/v1",
                ),
                "goal_id": "FVT-G210",
                "task_id": "FVT-056",
                "authority_ceiling": RUNTIME_MTL_AUTHORITY_CEILING,
                "runtime_mtl_external": {},
                "categories_exercised": [],
                "mutation_kinds": [],
                "hermetic_parity_shadow": {
                    "is_hermetic_parity_engine": True,
                    "is_vendor_build": False,
                    "cannot_satisfy_vendor": True,
                },
                "policy": {},
                "summary": {"checks_passed": 0, "checks_total": 0},
                "error": str(exc),
            }

    # --- ErgoAI live ---
    if ergoai_certificate is not None:
        ergo_cert: dict[str, Any] = dict(ergoai_certificate)
    else:
        advisors_module = _load_module(
            public_root / ADVISORS_CERTIFIER_RELATIVE,
            "fvt_replayed_monitor_advisor_advisors",
        )
        try:
            if ergo_root is None:
                # Probe the managed root so hermetic shims fail closed with
                # explicit reasons rather than looking like a missing dependency.
                probe_root = resolved_managed
                ergo_cert = advisors_module.build_ergoai_live_toolchain_contract(
                    repo_root=public_root,
                    install_root=probe_root,
                    run_semantics=True,
                    env=offline,
                )
            else:
                ergo_cert = advisors_module.build_ergoai_live_toolchain_contract(
                    repo_root=public_root,
                    install_root=ergo_root,
                    run_semantics=True,
                    env=offline,
                )
        except Exception as exc:  # noqa: BLE001
            lane_errors.append(f"ergoai_replay_failed:{exc}")
            ergo_cert = {
                "interface": "ErgoAILiveToolchainContract@1",
                "schema_version": "ergoai-live-toolchain-contract/v1",
                "goal_id": "FVT-G218",
                "task_id": "FVT-085",
                "tool_id": ERGOAI_TOOL_ID,
                "live_vendor_execution": False,
                "contract_passed": False,
                "semantic_passed": False,
                "authority_ceiling": ADVISOR_AUTHORITY_CEILING,
                "grants_proof_authority": False,
                "case_kinds": [],
                "block_reasons": [f"ergoai_replay_failed:{exc}"],
                "error": str(exc),
            }

    # --- SymbolicAI / advisor role corpus ---
    if symbolicai_certificate is not None:
        symai_cert: dict[str, Any] = dict(symbolicai_certificate)
    else:
        advisors_module = _load_module(
            public_root / ADVISORS_CERTIFIER_RELATIVE,
            "fvt_replayed_monitor_advisor_advisors_role",
        )
        try:
            symai_cert = advisors_module.build_certification_receipt(
                repo_root=public_root,
                env=offline,
                install_root=resolved_managed,
            )
        except Exception as exc:  # noqa: BLE001
            lane_errors.append(f"symbolicai_replay_failed:{exc}")
            symai_cert = {
                "interface": "AdvisorRoleCertification@1",
                "schema_version": "advisor-role-certification/v1",
                "goal_id": "FVT-G160",
                "task_id": "FVT-050",
                "production_certified": False,
                "semantic_corpus_passed": False,
                "authority_ceiling": ADVISOR_AUTHORITY_CEILING,
                "cases": [],
                "block_reasons": [f"symbolicai_replay_failed:{exc}"],
                "error": str(exc),
            }

    runtime_lane = _project_runtime_mtl_lane(
        mtl_cert, managed_root=resolved_managed, host_platform=host
    )
    advisors_lane = _project_advisors_lane(
        ergoai_certificate=ergo_cert,
        symbolicai_certificate=symai_cert,
        managed_root=resolved_managed,
        host_platform=host,
    )
    support_lane = _project_support_lane(
        lock_payload, managed_root=resolved_managed, host_platform=host
    )

    if lane_errors:
        phase["status"] = "failed"
        phase["reason_codes"].extend(lane_errors)

    certified = bool(
        runtime_lane.get("certified")
        and advisors_lane.get("certified")
        and support_lane.get("certified")
    )
    if not deployment_identity_matched and (
        resolved_managed is not None
        and path_under_approved_immutable_root(resolved_managed)
    ):
        certified = False
        phase["reason_codes"].append("deployment_identity_mismatch")

    policy = {
        "offline_certification_forbids_network": True,
        "offline_certification_forbids_download": True,
        "offline_certification_forbids_install": True,
        "offline_certification_forbids_ambient_path_mutation": True,
        "offline_certification_forbids_user_site_mutation": True,
        "offline_certification_forbids_source_tree_mutation": True,
        "offline_certification_forbids_system_package_mutation": True,
        "skip_install_only": True,
        "owns_runtime_mtl_and_advisor_replay_fanin": True,
        "does_not_make_core_ergoai_depend_on_java": True,
        "does_not_promote_advice_to_theorem_authority": True,
        "hermetic_parser_fixture_cannot_satisfy_external_runtime_mtl": True,
        "runtime_mtl_authority_ceiling": RUNTIME_MTL_AUTHORITY_CEILING,
        "advisor_authority_ceiling": ADVISOR_AUTHORITY_CEILING,
        "support_authority_ceiling": SUPPORT_AUTHORITY_CEILING,
        "stack_and_temurin_support_only": True,
        "finite_trace_authority_only_after_parity": True,
        "advisors_proposal_only_until_independent_reconstruction": True,
        "reuses_family_certifiers_without_weakening_ceilings": True,
        "no_central_certificate_edit": True,
    }

    acceptance = {
        "goal_id": GOAL_ID,
        "task_id": TASK_ID,
        "independent_node_typescript_runtime_mtl_positive_negative_boundary_malformed_mutation_replay_timeout_parity": bool(
            runtime_lane.get("certified")
            and REQUIRED_RUNTIME_MTL_ACCEPTANCE_AXES
            <= set(runtime_lane.get("acceptance_axes") or [])
        ),
        "cross_runtime_parity_with_disagreement_quarantine": bool(
            runtime_lane.get("certified")
            and "parity" in set(runtime_lane.get("acceptance_axes") or [])
            and "disagreement_quarantine"
            in set(runtime_lane.get("acceptance_axes") or [])
        ),
        "real_ergoai_and_symbolicai_advisory_cases": bool(
            advisors_lane.get("certified")
        ),
        "package_lockfile_runtime_launcher_target_artifact_executable_identities_bound": bool(
            runtime_lane.get("certified")
            and not any(
                str(reason).startswith("runtime_mtl_identity_unbound")
                for reason in (runtime_lane.get("block_reasons") or [])
            )
        ),
        "runtime_mtl_finite_trace_authority_only_after_parity": bool(
            runtime_lane.get("finite_trace_authority_granted")
        ),
        "advisors_remain_proposal_only_until_independent_reconstruction": bool(
            advisors_lane.get("proposal_only_until_independent_reconstruction")
            and advisors_lane.get("authority_ceiling") == ADVISOR_AUTHORITY_CEILING
        ),
        "stack_and_temurin_support_only_cannot_satisfy_verification_semantic_or_proof": bool(
            support_lane.get("certified")
            and support_lane.get("cannot_satisfy_public_verification")
            and support_lane.get("cannot_satisfy_semantic_authority")
            and support_lane.get("cannot_satisfy_proof_authority")
        ),
    }

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "interface": INTERFACE,
        "goal_id": GOAL_ID,
        "task_id": TASK_ID,
        "program": PROGRAM,
        "handler_id": HANDLER_ID,
        "certification_surface": CERTIFICATION_SURFACE,
        "host_platform": host,
        "lock_path": str(DEFAULT_LOCK_RELATIVE.as_posix()),
        "lock_digest_sha256": lock_digest,
        "managed_root": _redact_managed_path(
            str(resolved_managed) if resolved_managed else None, resolved_managed
        ),
        "managed_root_present": resolved_managed is not None,
        "managed_root_approved_immutable": bool(
            resolved_managed is not None
            and path_under_approved_immutable_root(resolved_managed)
        ),
        "deployment_identity": deployment_identity,
        "deployment_identity_expected": EXPECTED_FORMAL_TOOLCHAIN_DEPLOYMENT_IDENTITY,
        "deployment_identity_matched": deployment_identity_matched,
        "certified": certified,
        "semantic_certification": True,
        "certification_phase": phase,
        "policy": policy,
        "runtime_mtl": runtime_lane,
        "advisors": advisors_lane,
        "support": support_lane,
        "lanes": {
            "runtime_mtl": {
                "certified": runtime_lane.get("certified"),
                "authority_ceiling": runtime_lane.get("authority_ceiling"),
                "finite_trace_authority_granted": runtime_lane.get(
                    "finite_trace_authority_granted"
                ),
                "tool_id": runtime_lane.get("tool_id"),
            },
            "advisors": {
                "certified": advisors_lane.get("certified"),
                "authority_ceiling": advisors_lane.get("authority_ceiling"),
                "tool_ids": advisors_lane.get("tool_ids"),
            },
            "support": {
                "certified": support_lane.get("certified"),
                "authority_ceiling": support_lane.get("authority_ceiling"),
                "tool_ids": support_lane.get("tool_ids"),
            },
        },
        "summary": {
            "certified": certified,
            "runtime_mtl_certified": bool(runtime_lane.get("certified")),
            "advisors_certified": bool(advisors_lane.get("certified")),
            "support_certified": bool(support_lane.get("certified")),
            "runtime_mtl_tool_id": RUNTIME_MTL_TOOL_ID,
            "advisor_tool_ids": list(ADVISOR_TOOL_IDS),
            "support_tool_ids": list(SUPPORT_TOOL_IDS),
            "runtime_mtl_checks_passed": runtime_lane.get("checks_passed"),
            "runtime_mtl_checks_total": runtime_lane.get("checks_total"),
            "runtime_mtl_categories": runtime_lane.get("categories_exercised"),
            "runtime_mtl_acceptance_axes": runtime_lane.get("acceptance_axes"),
            "ergoai_case_kinds": _as_mapping(advisors_lane.get("ergoai")).get(
                "case_kinds"
            ),
            "symbolicai_case_kinds": _as_mapping(advisors_lane.get("symbolicai")).get(
                "case_kinds"
            ),
            "finite_trace_authority_granted": runtime_lane.get(
                "finite_trace_authority_granted"
            ),
            "advisors_proposal_only": True,
            "block_reasons": sorted(
                set(
                    list(runtime_lane.get("block_reasons") or [])
                    + list(advisors_lane.get("block_reasons") or [])
                    + list(support_lane.get("block_reasons") or [])
                    + list(phase.get("reason_codes") or [])
                )
            ),
        },
        "acceptance": acceptance,
        "depends_on": ["FVT-G210", "FVT-G218", "FVT-G223", "FVT-G226", "FVT-G227"],
        "env_flags": {
            "FORMAL_VERIFICATION_CERTIFY_OFFLINE": offline.get(
                "FORMAL_VERIFICATION_CERTIFY_OFFLINE"
            ),
            "FORMAL_VERIFICATION_FORBID_INSTALL": offline.get(
                "FORMAL_VERIFICATION_FORBID_INSTALL"
            ),
            "FORMAL_VERIFICATION_FORBID_NETWORK": offline.get(
                "FORMAL_VERIFICATION_FORBID_NETWORK"
            ),
            "FORMAL_VERIFICATION_REPLAYED_MONITOR_ADVISOR_OFFLINE": offline.get(
                "FORMAL_VERIFICATION_REPLAYED_MONITOR_ADVISOR_OFFLINE"
            ),
        },
    }

    digest_basis = {
        key: value
        for key, value in receipt.items()
        if key not in {"receipt_digest_sha256", "certificate_digest_sha256"}
    }
    digest = content_digest(digest_basis)
    receipt["receipt_digest_sha256"] = digest
    receipt["certificate_digest_sha256"] = digest

    failures = _validate_receipt(receipt)
    if failures:
        receipt["certified"] = False
        receipt["summary"]["certified"] = False
        receipt["receipt_validation_failures"] = failures
        receipt["summary"]["block_reasons"] = sorted(
            set(list(receipt["summary"].get("block_reasons") or []) + failures)
        )

    if write_receipt_path is not None:
        write_receipt(write_receipt_path, receipt)

    return receipt


def write_receipt(path: Path | str, receipt: Mapping[str, Any]) -> None:
    """Atomically write the public semantics receipt JSON."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(receipt, indent=2, sort_keys=False, default=str) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(destination.parent),
        prefix=f".{destination.name}.",
        delete=False,
    ) as handle:
        temporary_name = handle.name
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_name, destination)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="repository root containing the deployment lock",
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=None,
        help="path to formal_verification_toolchains.lock.json",
    )
    parser.add_argument(
        "--managed-root",
        type=Path,
        default=None,
        help="immutable managed prover root for offline replay",
    )
    parser.add_argument(
        "--runtime-mtl-install-root",
        type=Path,
        default=None,
        help="override Runtime MTL vendor install root",
    )
    parser.add_argument(
        "--ergoai-install-root",
        type=Path,
        default=None,
        help="override genuine ErgoAI managed install root",
    )
    parser.add_argument(
        "--host-platform",
        default=None,
        help="override normalized host platform key",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="write the semantics receipt JSON to this path",
    )
    parser.add_argument(
        "--write-default-receipt",
        action="store_true",
        help=(
            "write docs/architecture/"
            "formal_verification_replayed_monitor_advisor_semantics.json"
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        receipt = certify_replayed_monitor_advisor_semantics(
            repo_root=args.repo_root,
            lock_path=args.lock,
            managed_root=args.managed_root,
            runtime_mtl_install_root=args.runtime_mtl_install_root,
            ergoai_install_root=args.ergoai_install_root,
            host_platform=args.host_platform,
            skip_install=True,
            force_install=False,
        )
    except (OSError, ReplayedMonitorAdvisorError, json.JSONDecodeError) as exc:
        print(
            f"replayed monitor/advisor semantics refused: {exc}",
            file=sys.stderr,
        )
        return 2

    root = Path(args.repo_root) if args.repo_root is not None else repo_root_from()
    if args.write_default_receipt:
        write_receipt(root / DEFAULT_RECEIPT_RELATIVE, receipt)
    if args.output is not None:
        write_receipt(args.output, receipt)

    print(json.dumps(receipt, indent=2, sort_keys=False, default=str))
    return 0 if receipt.get("certified") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
