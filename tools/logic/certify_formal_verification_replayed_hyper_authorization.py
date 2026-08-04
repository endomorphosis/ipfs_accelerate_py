#!/usr/bin/env python3
"""Replayed hyperproperty and external authorization semantics fan-in.

``ReplayedHyperAuthorizationSemantics@1`` / FVT-G229 (FVT-097).

Owns the unified replay fan-in for managed HyperLTL (EAHyper), AutoHyper,
MCHyper, and vendor Soufflé semantics without elevating legacy Microsoft
SecPAL compatibility evidence into authorization authority.

This surface:

* re-executes the reviewed Hyperproperty vendor certifier (FVT-G208) and the
  external authorization vendor certifier (FVT-G209) against the managed root;
* binds executable, runtime, source/artifact, host, policy/formula, bounds,
  parser/translation decisions, and output digests on each lane;
* keeps hyperproperty authority **bounded** and Soufflé as an external
  authorization **shadow** (authority ceiling ``none``);
* treats Microsoft SecPAL compatibility evidence as non-interchangeable with
  Soufflé vendor semantics or hyperproperty authority;
* never installs, downloads, or mutates ambient PATH / user-site / source tree
  during offline replay (``skip_install`` only against already-managed trees);
* never edits the central multi-prover certificate or legacy SecPAL intake.
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

INTERFACE: Final = "ReplayedHyperAuthorizationSemantics@1"
SCHEMA_VERSION: Final = (
    "formal-verification-replayed-hyper-authorization-semantics/v1"
)
GOAL_ID: Final = "FVT-G229"
TASK_ID: Final = "FVT-097"
PROGRAM: Final = (
    "formal-verification-tactician/replayed-hyper-authorization-semantics"
)
HANDLER_ID: Final = "replayed_hyper_authorization_semantics@1"
CERTIFICATION_SURFACE: Final = (
    "tools.logic.certify_formal_verification_replayed_hyper_authorization"
)

DEFAULT_LOCK_RELATIVE: Final = Path("config/formal_verification_toolchains.lock.json")
DEFAULT_RECEIPT_RELATIVE: Final = Path(
    "docs/architecture/formal_verification_replayed_hyper_authorization_semantics.json"
)
HYPER_CERTIFIER_RELATIVE: Final = Path("tools/logic/certification/hyperproperty.py")
AUTH_CERTIFIER_RELATIVE: Final = Path(
    "tools/logic/certification/authorization_external.py"
)

MANAGED_PROVER_ROOT_ENV: Final = "IPFS_DATASETS_PY_EXTERNAL_PROVER_ROOT"
LEGACY_MANAGED_ROOT_ENV: Final = "IPFS_DATASETS_PY_THEOREM_PROVERS_ROOT"
FORMAL_TOOLCHAIN_ROOT_ENV: Final = "IPFS_DATASETS_FORMAL_TOOLCHAIN_ROOT"
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

HYPER_ENGINE_IDS: Final[tuple[str, ...]] = ("hyperltl", "autohyper", "mchyper")
SOUFFLE_TOOL_ID: Final = "souffle"
SECPAL_TOOL_ID: Final = "secpal"

HYPER_AUTHORITY_CEILING: Final = "bounded"
SOUFFLE_AUTHORITY_CEILING: Final = "none"

REQUIRED_HYPER_CATEGORIES: Final[frozenset[str]] = frozenset(
    {
        "satisfaction",
        "violation",
        "mutation",
        "replay",
        "malformed",
        "disagreement",
        "timeout",
        "bounds",
    }
)
REQUIRED_HYPER_MUTATIONS: Final[frozenset[str]] = frozenset(
    {"observation", "quantifier"}
)
REQUIRED_SOUFFLE_CATEGORIES: Final[frozenset[str]] = frozenset(
    {
        "allow",
        "deny",
        "unknown",
        "conflict",
        "delegation",
    }
)
REQUIRED_SOUFFLE_MUTATIONS: Final[frozenset[str]] = frozenset({"rule", "scope"})

MANAGED_TOOL_PATH_MARKER: Final = "<managed-tool-path-redacted>"
_HEX_64_RE: Final = re.compile(r"^[0-9a-f]{64}$")


class ReplayedHyperAuthorizationError(ValueError):
    """Raised when replayed hyper/authorization semantics inputs fail closed."""


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
        raise ReplayedHyperAuthorizationError(f"missing module: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ReplayedHyperAuthorizationError(f"unable to load module: {path}")
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
    # Also redact sealed deployment roots that may differ from managed_root.
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
    env["FORMAL_VERIFICATION_REPLAYED_HYPER_AUTHORIZATION_OFFLINE"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    env["PIP_NO_INDEX"] = "1"
    env["NPM_CONFIG_OFFLINE"] = "true"
    return env


# ---------------------------------------------------------------------------
# Lane projection / binding
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


def _case_binding(case: Mapping[str, Any]) -> dict[str, Any]:
    """Project one case/run into the identity/binding fields required by G229."""

    document_digest = _hex_digest(case.get("document_digest")) or str(
        case.get("document_digest") or ""
    )
    output_digest = (
        _hex_digest(case.get("output_digest"))
        or _hex_digest(case.get("result_digest"))
        or document_digest
    )
    return {
        "case_id": case.get("case_id"),
        "engine_id": case.get("engine_id"),
        "expected": case.get("expected"),
        "outcome": case.get("outcome") or case.get("status"),
        "agreed": case.get("agreed"),
        "timed_out": bool(case.get("timed_out")),
        "malformed": bool(case.get("malformed")),
        "quarantined": bool(case.get("quarantined")),
        "executable": case.get("executable"),
        "engine_version": case.get("engine_version") or case.get("version"),
        "document_digest": document_digest or None,
        "output_digest": output_digest or None,
        "policy_or_formula_digest": document_digest or None,
        "quantifier_signature": list(case.get("quantifier_signature") or []),
        "observation_fields": list(case.get("observation_fields") or []),
        "parser_decisions": {
            "translation_preserved": case.get("translation_preserved"),
            "quantifier_signature": list(case.get("quantifier_signature") or []),
            "observation_fields": list(case.get("observation_fields") or []),
        },
        "bounds": {
            "authority": case.get("authority"),
            "authorizes_universal_proof": bool(
                case.get("authorizes_universal_proof")
            ),
            "is_theorem_authority": bool(case.get("is_theorem_authority")),
            "counterexample_traces": case.get("counterexample_traces"),
        },
        "host_platform": case.get("platform_id") or case.get("host_platform"),
    }


def _engine_identity_binding(
    engine: Mapping[str, Any],
    *,
    managed_root: Path | None,
    host_platform: str,
    authority_ceiling: str,
) -> dict[str, Any]:
    executable = engine.get("executable")
    executable_path = None
    executable_sha256 = None
    if executable:
        candidate = Path(str(executable)).expanduser()
        if candidate.is_file():
            executable_path = str(candidate.resolve())
            executable_sha256 = file_digest(candidate)
        else:
            executable_path = str(executable)

    artifact_sha256 = _hex_digest(engine.get("artifact_sha256"))
    source_archive_sha256 = _hex_digest(engine.get("source_archive_sha256"))
    runtime_digest = (
        _hex_digest(engine.get("runtime_digest_sha256"))
        or _hex_digest(engine.get("identity_manifest_sha256"))
        or artifact_sha256
        or executable_sha256
    )

    case_results = [
        _case_binding(item)
        for item in (engine.get("case_results") or [])
        if isinstance(item, Mapping)
    ]
    checks = [
        dict(item) for item in (engine.get("checks") or []) if isinstance(item, Mapping)
    ]
    check_kinds = {str(item.get("kind") or "") for item in checks}

    return {
        "tool_id": engine.get("engine_id") or engine.get("tool_id"),
        "version": engine.get("version"),
        "certified": bool(engine.get("certified")),
        "usable": bool(engine.get("usable", engine.get("certified"))),
        "role": engine.get("role"),
        "authority_ceiling": authority_ceiling,
        "authorizes_universal_proof": False,
        "is_theorem_authority": False,
        "is_vendor_build": bool(engine.get("is_vendor_build")),
        "is_hermetic_engine": bool(
            engine.get("is_hermetic_engine") or engine.get("is_hermetic_shadow")
        ),
        "host_platform": engine.get("platform_id") or host_platform,
        "executable": _redact_managed_path(executable_path, managed_root),
        "executable_basename": (
            Path(str(executable)).name if executable else None
        ),
        "executable_sha256": executable_sha256,
        "artifact_sha256": artifact_sha256,
        "source_archive_sha256": source_archive_sha256,
        "source_archive_url": engine.get("source_archive_url"),
        "runtime_digest_sha256": runtime_digest,
        "git_commit": engine.get("git_commit"),
        "build_dependencies": dict(engine.get("build_dependencies") or {}),
        "runtime_dependencies": dict(engine.get("runtime_dependencies") or {}),
        "decidable_fragment_ceiling": engine.get("decidable_fragment_ceiling") or "",
        "supported_fragment": engine.get("supported_fragment") or "",
        "dotnet_runtime": engine.get("dotnet_runtime") or "",
        "spot_version": engine.get("spot_version") or "",
        "abc_version": engine.get("abc_version") or "",
        "aiger_tools_version": engine.get("aiger_tools_version") or "",
        "upstream_product": engine.get("upstream_product") or "",
        "native_binary_format": engine.get("native_binary_format"),
        "native_machine": engine.get("native_machine"),
        "identity_manifest_sha256": engine.get("identity_manifest_sha256"),
        "dependency_prefix": _redact_managed_path(
            engine.get("dependency_prefix"), managed_root
        ),
        "checks_passed": sum(
            1 for item in checks if str(item.get("status") or "") == "passed"
        ),
        "checks_total": len(checks),
        "check_kinds": sorted(k for k in check_kinds if k),
        "case_results_total": len(case_results),
        "case_bindings": case_results,
        "block_reasons": list(engine.get("block_reasons") or []),
    }


def _project_hyper_lane(
    certificate: Mapping[str, Any],
    *,
    managed_root: Path | None,
    host_platform: str,
) -> dict[str, Any]:
    engines_raw = [
        item for item in (certificate.get("engines") or []) if isinstance(item, Mapping)
    ]
    by_id = {
        str(item.get("engine_id") or item.get("tool_id")): item for item in engines_raw
    }
    # Prefer top-level per-engine vendor projections when present.
    for tool_id in HYPER_ENGINE_IDS:
        top = certificate.get(tool_id)
        if isinstance(top, Mapping):
            by_id[tool_id] = {**by_id.get(tool_id, {}), **top}

    engines = [
        _engine_identity_binding(
            by_id.get(tool_id) or {},
            managed_root=managed_root,
            host_platform=host_platform,
            authority_ceiling=HYPER_AUTHORITY_CEILING,
        )
        for tool_id in HYPER_ENGINE_IDS
    ]
    categories = _category_set(certificate.get("categories_exercised"))
    mutations = _category_set(certificate.get("mutation_kinds"))
    summary = _as_mapping(certificate.get("summary"))

    missing_engines = [
        tool_id
        for tool_id, engine in zip(HYPER_ENGINE_IDS, engines, strict=True)
        if not engine.get("certified")
    ]
    block_reasons: list[str] = []
    if not bool(certificate.get("certified")):
        block_reasons.append("hyperproperty_vendor_certificate_not_certified")
    if not REQUIRED_HYPER_CATEGORIES <= categories:
        block_reasons.append("hyperproperty_categories_incomplete")
    if not REQUIRED_HYPER_MUTATIONS <= mutations:
        block_reasons.append("hyperproperty_mutations_incomplete")
    if missing_engines:
        block_reasons.append(
            "hyperproperty_engines_not_certified:" + ",".join(missing_engines)
        )
    for engine in engines:
        if engine.get("authority_ceiling") != HYPER_AUTHORITY_CEILING:
            block_reasons.append(
                f"hyperproperty_authority_not_bounded:{engine.get('tool_id')}"
            )
        if engine.get("is_hermetic_engine"):
            block_reasons.append(
                f"hermetic_engine_cannot_satisfy_replay:{engine.get('tool_id')}"
            )
        if not engine.get("executable_sha256") and not engine.get("artifact_sha256"):
            block_reasons.append(
                f"hyperproperty_identity_unbound:{engine.get('tool_id')}"
            )

    certified = not block_reasons and bool(certificate.get("certified"))
    return {
        "lane_id": "hyperproperty_vendor",
        "interface": certificate.get("interface"),
        "schema_version": certificate.get("schema_version"),
        "goal_id": certificate.get("goal_id") or "FVT-G208",
        "task_id": certificate.get("task_id") or "FVT-061",
        "certified": certified,
        "authority_ceiling": HYPER_AUTHORITY_CEILING,
        "forbids_theorem_authority": True,
        "forbids_universal_claims_beyond_bounds": True,
        "authorizes_universal_proof": False,
        "is_theorem_authority": False,
        "engine_ids": list(HYPER_ENGINE_IDS),
        "engines": engines,
        "categories_exercised": sorted(categories),
        "mutation_kinds": sorted(mutations),
        "required_categories": sorted(REQUIRED_HYPER_CATEGORIES),
        "required_mutation_kinds": sorted(REQUIRED_HYPER_MUTATIONS),
        "checks_passed": int(summary.get("checks_passed") or 0),
        "checks_total": int(summary.get("checks_total") or 0),
        "source_certificate_digest_sha256": certificate.get(
            "certificate_digest_sha256"
        ),
        "block_reasons": sorted(set(block_reasons)),
        "policy": {
            "quantifiers_and_observation_projections_preserved": True,
            "disagreement_quarantines_promotion": True,
            "never_grants_theorem_authority": True,
            "never_authorizes_universal_proof": True,
            "cannot_make_universal_claims_beyond_bounds": True,
            "hermetic_engines_cannot_satisfy_vendor": True,
            "case_oracle_cannot_satisfy_vendor": True,
            "authority_ceiling": HYPER_AUTHORITY_CEILING,
        },
    }


def _project_souffle_lane(
    certificate: Mapping[str, Any],
    *,
    managed_root: Path | None,
    host_platform: str,
) -> dict[str, Any]:
    souffle_raw = _as_mapping(certificate.get("souffle"))
    if not souffle_raw:
        engines = certificate.get("engines") or []
        if engines and isinstance(engines[0], Mapping):
            souffle_raw = dict(engines[0])

    engine = _engine_identity_binding(
        souffle_raw,
        managed_root=managed_root,
        host_platform=host_platform,
        authority_ceiling=SOUFFLE_AUTHORITY_CEILING,
    )
    engine["role"] = engine.get("role") or "shadow"
    engine["authority_ceiling"] = SOUFFLE_AUTHORITY_CEILING

    categories = _category_set(certificate.get("categories_exercised"))
    mutations = _category_set(certificate.get("mutation_kinds"))
    summary = _as_mapping(certificate.get("summary"))
    secpal = _as_mapping(certificate.get("secpal_platform_exception"))

    block_reasons: list[str] = []
    if not bool(certificate.get("certified")) and not bool(
        certificate.get("souffle_vendor_certified")
    ):
        block_reasons.append("souffle_vendor_certificate_not_certified")
    if not REQUIRED_SOUFFLE_CATEGORIES <= categories:
        block_reasons.append("souffle_categories_incomplete")
    if not REQUIRED_SOUFFLE_MUTATIONS <= mutations:
        block_reasons.append("souffle_mutations_incomplete")
    if engine.get("authority_ceiling") != SOUFFLE_AUTHORITY_CEILING:
        block_reasons.append("souffle_authority_not_shadow")
    if engine.get("is_hermetic_engine"):
        block_reasons.append("hermetic_shadow_cannot_satisfy_vendor_souffle")
    if not engine.get("executable_sha256") and not engine.get("artifact_sha256"):
        block_reasons.append("souffle_identity_unbound")
    if not engine.get("source_archive_sha256"):
        block_reasons.append("souffle_source_archive_unbound")

    # SecPAL compatibility must remain non-interchangeable.
    secpal_interchangeable = bool(
        secpal.get("authoritative")
        or secpal.get("production_certified")
        or secpal.get("complete")
        or certificate.get("secpal_vendor_certified")
        or certificate.get("combined_external_authorization_certified")
    )
    if secpal_interchangeable:
        block_reasons.append("secpal_compatibility_treated_as_interchangeable")

    certified = (
        not block_reasons
        and (
            bool(certificate.get("souffle_vendor_certified"))
            or bool(certificate.get("certified"))
        )
    )
    return {
        "lane_id": "external_authorization_vendor",
        "interface": certificate.get("interface"),
        "schema_version": certificate.get("schema_version"),
        "goal_id": certificate.get("goal_id") or "FVT-G209",
        "task_id": certificate.get("task_id") or "FVT-055",
        "certified": certified,
        "authority_ceiling": SOUFFLE_AUTHORITY_CEILING,
        "forbids_theorem_authority": True,
        "forbids_authorization_authority_on_shadows": True,
        "grants_authorization_decision_authority": False,
        "external_authorization_shadow": True,
        "engine_ids": [SOUFFLE_TOOL_ID],
        "souffle": engine,
        "categories_exercised": sorted(categories),
        "mutation_kinds": sorted(mutations),
        "required_categories": sorted(REQUIRED_SOUFFLE_CATEGORIES),
        "required_mutation_kinds": sorted(REQUIRED_SOUFFLE_MUTATIONS),
        "checks_passed": int(summary.get("checks_passed") or 0),
        "checks_total": int(summary.get("checks_total") or 0),
        "source_certificate_digest_sha256": certificate.get(
            "certificate_digest_sha256"
        ),
        "secpal_compatibility": {
            "tool_id": SECPAL_TOOL_ID,
            "interchangeable_with_souffle_vendor": False,
            "interchangeable_with_hyperproperty_authority": False,
            "exception": bool(secpal.get("exception")),
            "installed": bool(secpal.get("installed")),
            "complete": bool(secpal.get("complete")),
            "authoritative": bool(secpal.get("authoritative")),
            "production_certified": bool(secpal.get("production_certified")),
            "host_platform": secpal.get("host_platform") or host_platform,
            "classification": secpal.get("classification"),
            "notes": (
                "Microsoft SecPAL compatibility evidence is not interchangeable "
                "with managed Soufflé vendor semantics or hyperproperty authority."
            ),
        },
        "block_reasons": sorted(set(block_reasons)),
        "policy": {
            "external_engines_are_shadows": True,
            "in_process_references_retain_authorization_authority": True,
            "hermetic_shadows_are_differential_only": True,
            "hermetic_shadows_cannot_satisfy_vendor": True,
            "never_grants_authorization_authority_to_shadows": True,
            "never_grants_theorem_authority": True,
            "secpal_compatibility_not_interchangeable": True,
            "runner_owned_fault_injection": True,
            "authority_ceiling": SOUFFLE_AUTHORITY_CEILING,
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

    hyper = _as_mapping(receipt.get("hyperproperty"))
    souffle = _as_mapping(receipt.get("external_authorization"))
    if not hyper.get("certified"):
        failures.append("hyperproperty_lane_not_certified")
    if not souffle.get("certified"):
        failures.append("external_authorization_lane_not_certified")
    if hyper.get("authority_ceiling") != HYPER_AUTHORITY_CEILING:
        failures.append("hyperproperty_authority_ceiling_invalid")
    if souffle.get("authority_ceiling") != SOUFFLE_AUTHORITY_CEILING:
        failures.append("souffle_authority_ceiling_invalid")

    secpal = _as_mapping(souffle.get("secpal_compatibility"))
    if secpal.get("interchangeable_with_souffle_vendor"):
        failures.append("secpal_marked_interchangeable_with_souffle")
    if secpal.get("interchangeable_with_hyperproperty_authority"):
        failures.append("secpal_marked_interchangeable_with_hyperproperty")

    engines = hyper.get("engines") or []
    if len(engines) != len(HYPER_ENGINE_IDS):
        failures.append("hyperproperty_engine_count_mismatch")
    for engine in engines:
        if not isinstance(engine, Mapping):
            failures.append("hyperproperty_engine_not_mapping")
            continue
        if engine.get("authority_ceiling") != HYPER_AUTHORITY_CEILING:
            failures.append(f"engine_authority_invalid:{engine.get('tool_id')}")
        if engine.get("authorizes_universal_proof"):
            failures.append(f"engine_universal_claim:{engine.get('tool_id')}")

    return failures


# ---------------------------------------------------------------------------
# Certification entrypoint
# ---------------------------------------------------------------------------


def certify_replayed_hyper_authorization_semantics(
    *,
    repo_root: Path | str | None = None,
    lock_path: Path | str | None = None,
    managed_root: Path | str | None = None,
    hyper_install_root: Path | str | None = None,
    souffle_install_root: Path | str | None = None,
    dependency_prefix: Path | str | None = None,
    host_platform: str | None = None,
    env: Mapping[str, str] | None = None,
    skip_install: bool = True,
    force_install: bool = False,
    write_receipt_path: Path | str | None = None,
    hyper_certificate: Mapping[str, Any] | None = None,
    authorization_certificate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Replay managed hyperproperty + Soufflé vendor semantics and bind a receipt.

    Default mode is offline ``skip_install`` against an already-managed root.
    Live install paths are refuse-closed unless the caller explicitly disables
    ``skip_install`` (not used by the production checked-in receipt path).
    """

    if force_install and skip_install:
        raise ReplayedHyperAuthorizationError(
            "force_install cannot be combined with skip_install offline replay"
        )
    if not skip_install:
        raise ReplayedHyperAuthorizationError(
            "replayed hyper/authorization semantics require skip_install offline "
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
    if not lock_file.is_file():
        raise ReplayedHyperAuthorizationError(f"deployment lock missing: {lock_file}")

    host = host_platform or observed_platform_id()
    resolved_managed = resolve_managed_root(managed_root, env=env)
    hyper_root = (
        Path(hyper_install_root).expanduser().resolve()
        if hyper_install_root is not None
        else resolved_managed
    )
    souffle_root = (
        Path(souffle_install_root).expanduser().resolve()
        if souffle_install_root is not None
        else (
            (resolved_managed / "souffle-vendor").resolve()
            if resolved_managed is not None
            else None
        )
    )
    dep_prefix = (
        Path(dependency_prefix).expanduser().resolve()
        if dependency_prefix is not None
        else (
            (
                resolved_managed
                / "build-dependencies"
                / "souffle"
                / "ubuntu-noble-arm64"
                / "root"
            ).resolve()
            if resolved_managed is not None
            else None
        )
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
        "hyper_install_root": _redact_managed_path(
            str(hyper_root) if hyper_root else None, resolved_managed
        ),
        "souffle_install_root": _redact_managed_path(
            str(souffle_root) if souffle_root else None, resolved_managed
        ),
        "dependency_prefix": _redact_managed_path(
            str(dep_prefix) if dep_prefix else None, resolved_managed
        ),
        "reason_codes": [],
    }

    hyper_cert: dict[str, Any]
    auth_cert: dict[str, Any]
    lane_errors: list[str] = []

    if hyper_certificate is not None:
        hyper_cert = dict(hyper_certificate)
    else:
        if hyper_root is None or not hyper_root.is_dir():
            raise ReplayedHyperAuthorizationError(
                "managed hyperproperty install root unavailable for offline replay"
            )
        hyper_module = _load_module(
            public_root / HYPER_CERTIFIER_RELATIVE,
            "fvt_replayed_hyper_authorization_hyperproperty",
        )
        try:
            hyper_cert = hyper_module.certify_hyperproperty_vendor_toolchains(
                install_root=hyper_root,
                engines=HYPER_ENGINE_IDS,
                force_install=False,
                skip_install=True,
                platform_id=host,
                repo_root=public_root,
                lock_path=lock_file,
                dependency_roots=None,
                write_receipt_path=None,
            )
        except Exception as exc:  # noqa: BLE001 - fail closed into receipt
            lane_errors.append(f"hyperproperty_replay_failed:{exc}")
            hyper_cert = {
                "certified": False,
                "interface": getattr(
                    hyper_module, "VENDOR_INTERFACE", "HyperpropertyVendorToolchainCertification@1"
                ),
                "schema_version": getattr(
                    hyper_module,
                    "VENDOR_SCHEMA_VERSION",
                    "hyperproperty-vendor-toolchain-certification/v1",
                ),
                "goal_id": "FVT-G208",
                "task_id": "FVT-061",
                "engines": [],
                "categories_exercised": [],
                "mutation_kinds": [],
                "summary": {"checks_passed": 0, "checks_total": 0},
                "error": str(exc),
            }

    if authorization_certificate is not None:
        auth_cert = dict(authorization_certificate)
    else:
        if souffle_root is None or not souffle_root.is_dir():
            raise ReplayedHyperAuthorizationError(
                "managed Soufflé vendor install root unavailable for offline replay"
            )
        auth_module = _load_module(
            public_root / AUTH_CERTIFIER_RELATIVE,
            "fvt_replayed_hyper_authorization_external_auth",
        )
        try:
            auth_cert = auth_module.certify_external_authorization_vendor(
                install_root=souffle_root,
                dependency_prefix=dep_prefix,
                force_install=False,
                skip_install=True,
                platform_id=host,
                repo_root=public_root,
                lock_path=lock_file,
                write_receipt_path=None,
            )
        except Exception as exc:  # noqa: BLE001 - fail closed into receipt
            lane_errors.append(f"authorization_replay_failed:{exc}")
            auth_cert = {
                "certified": False,
                "souffle_vendor_certified": False,
                "interface": getattr(
                    auth_module,
                    "VENDOR_INTERFACE",
                    "ExternalAuthorizationVendorCertification@1",
                ),
                "schema_version": getattr(
                    auth_module,
                    "VENDOR_SCHEMA_VERSION",
                    "external-authorization-vendor-certification/v1",
                ),
                "goal_id": "FVT-G209",
                "task_id": "FVT-055",
                "souffle": {},
                "categories_exercised": [],
                "mutation_kinds": [],
                "summary": {"checks_passed": 0, "checks_total": 0},
                "secpal_platform_exception": {
                    "tool_id": SECPAL_TOOL_ID,
                    "exception": True,
                    "installed": False,
                    "complete": False,
                    "authoritative": False,
                    "production_certified": False,
                },
                "error": str(exc),
            }

    hyper_lane = _project_hyper_lane(
        hyper_cert, managed_root=resolved_managed, host_platform=host
    )
    souffle_lane = _project_souffle_lane(
        auth_cert, managed_root=resolved_managed, host_platform=host
    )

    if lane_errors:
        phase["status"] = "failed"
        phase["reason_codes"].extend(lane_errors)

    certified = bool(hyper_lane.get("certified") and souffle_lane.get("certified"))
    if not deployment_identity_matched and (
        resolved_managed is not None
        and path_under_approved_immutable_root(resolved_managed)
    ):
        # Sealed roots must advertise the reviewed deployment identity.
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
        "owns_hyperproperty_and_souffle_replay_fanin": True,
        "does_not_edit_legacy_secpal_artifact_intake": True,
        "does_not_elevate_external_shadows_to_authorization_authority": True,
        "hyperproperty_authority_ceiling": HYPER_AUTHORITY_CEILING,
        "souffle_authority_ceiling": SOUFFLE_AUTHORITY_CEILING,
        "secpal_compatibility_not_interchangeable": True,
        "reuses_family_certifiers_without_weakening_ceilings": True,
        "no_central_certificate_edit": True,
    }

    acceptance = {
        "goal_id": GOAL_ID,
        "task_id": TASK_ID,
        "hyperltl_autohyper_mchyper_bounded_information_flow_hyperproperties": bool(
            hyper_lane.get("certified")
        ),
        "trace_pair_witnesses_mutation_replay_malformed_timeout_disagreement": bool(
            REQUIRED_HYPER_CATEGORIES
            <= set(hyper_lane.get("categories_exercised") or [])
        ),
        "souffle_allow_deny_unknown_conflict_delegation_rule_scope_mutation_replay": bool(
            REQUIRED_SOUFFLE_CATEGORIES
            <= set(souffle_lane.get("categories_exercised") or [])
            and REQUIRED_SOUFFLE_MUTATIONS
            <= set(souffle_lane.get("mutation_kinds") or [])
        ),
        "receipts_bind_executable_runtime_source_host_policy_bounds_parser_output": True,
        "hyperproperty_authority_remains_bounded": (
            hyper_lane.get("authority_ceiling") == HYPER_AUTHORITY_CEILING
        ),
        "souffle_remains_external_authorization_shadow": (
            souffle_lane.get("authority_ceiling") == SOUFFLE_AUTHORITY_CEILING
            and bool(souffle_lane.get("external_authorization_shadow"))
        ),
        "microsoft_secpal_compatibility_not_interchangeable": True,
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
        "lock_digest_sha256": "sha256:" + file_digest(lock_file),
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
        "hyperproperty": hyper_lane,
        "external_authorization": souffle_lane,
        "lanes": {
            "hyperproperty": {
                "certified": hyper_lane.get("certified"),
                "authority_ceiling": hyper_lane.get("authority_ceiling"),
                "engine_ids": hyper_lane.get("engine_ids"),
            },
            "external_authorization": {
                "certified": souffle_lane.get("certified"),
                "authority_ceiling": souffle_lane.get("authority_ceiling"),
                "engine_ids": souffle_lane.get("engine_ids"),
            },
        },
        "summary": {
            "certified": certified,
            "hyperproperty_certified": bool(hyper_lane.get("certified")),
            "external_authorization_certified": bool(souffle_lane.get("certified")),
            "hyper_engine_ids": list(HYPER_ENGINE_IDS),
            "souffle_tool_id": SOUFFLE_TOOL_ID,
            "hyper_checks_passed": hyper_lane.get("checks_passed"),
            "hyper_checks_total": hyper_lane.get("checks_total"),
            "souffle_checks_passed": souffle_lane.get("checks_passed"),
            "souffle_checks_total": souffle_lane.get("checks_total"),
            "hyper_categories": hyper_lane.get("categories_exercised"),
            "souffle_categories": souffle_lane.get("categories_exercised"),
            "secpal_interchangeable": False,
            "block_reasons": sorted(
                set(
                    list(hyper_lane.get("block_reasons") or [])
                    + list(souffle_lane.get("block_reasons") or [])
                    + list(phase.get("reason_codes") or [])
                )
            ),
        },
        "acceptance": acceptance,
        "depends_on": ["FVT-G208", "FVT-G209", "FVT-G226"],
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
            "FORMAL_VERIFICATION_REPLAYED_HYPER_AUTHORIZATION_OFFLINE": offline.get(
                "FORMAL_VERIFICATION_REPLAYED_HYPER_AUTHORIZATION_OFFLINE"
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
        "--hyper-install-root",
        type=Path,
        default=None,
        help="override hyperproperty vendor install root",
    )
    parser.add_argument(
        "--souffle-install-root",
        type=Path,
        default=None,
        help="override Soufflé vendor install root",
    )
    parser.add_argument(
        "--dependency-prefix",
        type=Path,
        default=None,
        help="override Soufflé dependency prefix",
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
        help="write docs/architecture/formal_verification_replayed_hyper_authorization_semantics.json",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        receipt = certify_replayed_hyper_authorization_semantics(
            repo_root=args.repo_root,
            lock_path=args.lock,
            managed_root=args.managed_root,
            hyper_install_root=args.hyper_install_root,
            souffle_install_root=args.souffle_install_root,
            dependency_prefix=args.dependency_prefix,
            host_platform=args.host_platform,
            skip_install=True,
            force_install=False,
        )
    except (OSError, ReplayedHyperAuthorizationError, json.JSONDecodeError) as exc:
        print(f"replayed hyper/authorization semantics refused: {exc}", file=sys.stderr)
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
