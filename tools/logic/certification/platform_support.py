#!/usr/bin/env python3
"""Derive exact host support and platform exceptions from the deployment lock.

``FormalVerificationPlatformSupport@1`` / FVT-G201 (FVT-064).

Owns platform normalization and per-tool classification only. This surface:

* derives a normalized host key from the running OS and architecture;
* classifies every locked tool as ``supported_here``, ``unsupported_here``,
  or ``ambiguous`` from its own pins and deployment contract;
* honors ``any`` (and reviewed ``source``) support declarations;
* treats absent, contradictory, or ambiguous metadata as blockers;
* allows only an explicit host exclusion to produce a narrow platform
  exception (never converts mere unavailability into unsupported status);
* never probes PATH, installs tools, or opens the network.

linux-aarch64 under the current reviewed lock classifies HyperLTL, AutoHyper,
MCHyper, Souffle, and external Runtime MTL as supported, external SecPAL as
unsupported (narrow platform exception), and ZKP as a platform-independent
deployment binding. A lock mutation that adds or removes ``linux-aarch64``
from a tool's support declaration changes the classification and the final
report digest.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final, Iterable, Mapping, Sequence

INTERFACE: Final = "FormalVerificationPlatformSupport@1"
SCHEMA_VERSION: Final = "formal-verification-platform-support/v1"
GOAL_ID: Final = "FVT-G201"
TASK_ID: Final = "FVT-064"
PROGRAM: Final = "formal-verification-tactician/platform-support-classifier"
CERTIFICATION_SURFACE: Final = "tools.logic.certification.platform_support"

DEFAULT_LOCK_RELATIVE: Final = Path("config/formal_verification_toolchains.lock.json")
DEFAULT_ZKP_LOCK_RELATIVE: Final = Path(
    "config/formal_verification_zkp_deployment.lock.json"
)

CLASSIFICATION_SUPPORTED: Final = "supported_here"
CLASSIFICATION_UNSUPPORTED: Final = "unsupported_here"
CLASSIFICATION_AMBIGUOUS: Final = "ambiguous"

VALID_CLASSIFICATIONS: Final = frozenset(
    {
        CLASSIFICATION_SUPPORTED,
        CLASSIFICATION_UNSUPPORTED,
        CLASSIFICATION_AMBIGUOUS,
    }
)

# Reviewed wildcard pin / contract tokens that mean "not host-tied".
UNIVERSAL_PLATFORM_TOKENS: Final = frozenset({"any"})
# Reviewed source pins are buildable on the host; they do not claim a binary
# artifact for every host, but they do not fail closed as unsupported either.
SOURCE_PLATFORM_TOKENS: Final = frozenset({"source"})
PIN_SUPPORT_WILDCARDS: Final = UNIVERSAL_PLATFORM_TOKENS | SOURCE_PLATFORM_TOKENS

# Managed tools whose support is a platform-independent deployment binding
# rather than a host-tied binary matrix entry.
PLATFORM_INDEPENDENT_DEPLOYMENT_TOOL_IDS: Final = frozenset({"zkp-circuit"})

# Availability values that participate in the managed pin matrix. Everything
# else falls back to the global platform policy (never inferred from PATH).
MANAGED_AVAILABILITIES: Final = frozenset({"managed_pin"})


class PlatformSupportError(ValueError):
    """Raised when platform classification inputs are invalid or contradictory."""


# ---------------------------------------------------------------------------
# Path / digest helpers
# ---------------------------------------------------------------------------


def repo_root_from(start: Path | None = None) -> Path:
    """Locate the repository root that owns the toolchain deployment lock."""

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
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Host normalization
# ---------------------------------------------------------------------------


def normalize_host_platform(
    system: str | None = None,
    machine: str | None = None,
) -> str:
    """Normalize OS + architecture into the lock's host platform key.

    Mapping matches the reviewed deployment lock vocabulary:

    * linux + x86_64/amd64 → ``linux-x86_64``
    * linux + aarch64/arm64 → ``linux-aarch64``
    * darwin + x86_64/amd64 → ``darwin-x86_64``
    * darwin + arm64/aarch64 → ``darwin-arm64``
    """

    system_raw = (system if system is not None else platform.system()).lower().strip()
    machine_raw = (machine if machine is not None else platform.machine()).lower().strip()
    system_name = {"linux": "linux", "darwin": "darwin"}.get(system_raw, system_raw)
    if system_name == "linux":
        machine_name = {
            "amd64": "x86_64",
            "x86_64": "x86_64",
            "arm64": "aarch64",
            "aarch64": "aarch64",
        }.get(machine_raw, machine_raw)
    elif system_name == "darwin":
        machine_name = {
            "amd64": "x86_64",
            "x86_64": "x86_64",
            "arm64": "arm64",
            "aarch64": "arm64",
        }.get(machine_raw, machine_raw)
    else:
        machine_name = {
            "amd64": "x86_64",
            "x86_64": "x86_64",
            "arm64": "aarch64",
            "aarch64": "aarch64",
        }.get(machine_raw, machine_raw)
    if not system_name or not machine_name:
        raise PlatformSupportError(
            f"unable to normalize host platform from system={system_raw!r} "
            f"machine={machine_raw!r}"
        )
    return f"{system_name}-{machine_name}"


def observed_host_platform() -> str:
    """Return the normalized host platform key for the running machine."""

    return normalize_host_platform()


# ---------------------------------------------------------------------------
# Lock loading
# ---------------------------------------------------------------------------


def load_toolchain_lock(lock_path: Path | str) -> dict[str, Any]:
    path = Path(lock_path)
    if not path.is_file():
        raise FileNotFoundError(f"offline toolchain lock missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PlatformSupportError("toolchain lock must be a JSON object")
    return payload


def load_zkp_deployment_lock(lock_path: Path | str | None = None) -> dict[str, Any] | None:
    """Load the optional ZKP deployment lock when present (read-only)."""

    if lock_path is None:
        return None
    path = Path(lock_path)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PlatformSupportError("zkp deployment lock must be a JSON object")
    return payload


def lock_tools_by_id(lock: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    tools = lock.get("tools") or []
    if not isinstance(tools, list):
        raise PlatformSupportError("lock.tools must be a list")
    index: dict[str, dict[str, Any]] = {}
    for entry in tools:
        if not isinstance(entry, Mapping):
            continue
        tool_id = entry.get("tool_id")
        if not tool_id:
            continue
        index[str(tool_id)] = dict(entry)
    return index


def global_supported_platforms(lock: Mapping[str, Any]) -> list[str]:
    policy = lock.get("platform_policy") or {}
    if not isinstance(policy, Mapping):
        return []
    raw = policy.get("supported_platforms") or []
    if not isinstance(raw, list):
        raise PlatformSupportError("platform_policy.supported_platforms must be a list")
    return [str(item) for item in raw if item]


def _as_platform_list(values: Iterable[Any] | None) -> list[str]:
    if not values:
        return []
    out: list[str] = []
    for item in values:
        text = str(item or "").strip()
        if text:
            out.append(text)
    return out


def _pin_platforms(entry: Mapping[str, Any]) -> list[str]:
    pins = entry.get("pins") or []
    if not isinstance(pins, list):
        return []
    platforms: list[str] = []
    for pin in pins:
        if not isinstance(pin, Mapping):
            continue
        plat = str(pin.get("platform") or "").strip()
        if plat:
            platforms.append(plat)
    return platforms


def _contract_platforms(entry: Mapping[str, Any]) -> list[str]:
    contract = entry.get("deployment_contract") or {}
    if not isinstance(contract, Mapping):
        return []
    return _as_platform_list(contract.get("supported_platforms"))


def _platform_set_supports(platforms: Sequence[str], host_platform: str) -> bool:
    tokens = set(platforms)
    if tokens & UNIVERSAL_PLATFORM_TOKENS:
        return True
    return host_platform in tokens


def _pin_set_supports(platforms: Sequence[str], host_platform: str) -> bool:
    tokens = set(platforms)
    if tokens & PIN_SUPPORT_WILDCARDS:
        return True
    return host_platform in tokens


# ---------------------------------------------------------------------------
# Classification result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ToolPlatformClassification:
    """Auditable host-platform classification for one locked tool."""

    tool_id: str
    host_platform: str
    availability: str
    managed: bool
    classification: str
    supported: bool
    ambiguous: bool
    exception_eligible: bool
    basis: str
    declared_platforms: list[str] = field(default_factory=list)
    contract_platforms: list[str] = field(default_factory=list)
    pin_platforms: list[str] = field(default_factory=list)
    globally_supported: bool = False
    platform_independent_deployment_binding: bool = False
    blocker: bool = False
    blocker_reasons: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.classification not in VALID_CLASSIFICATIONS:
            raise PlatformSupportError(
                f"invalid classification {self.classification!r} for {self.tool_id!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PlatformException:
    """Narrow unsupported-platform exception for an explicitly excluded host."""

    tool_id: str
    host_platform: str
    declared_platforms: list[str]
    basis: str
    classification: str = CLASSIFICATION_UNSUPPORTED
    narrow_scope: bool = True
    complete: bool = False
    production_certified: bool = False
    installed: bool = False
    authoritative: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Core classifier
# ---------------------------------------------------------------------------


def classify_tool_platform_support(
    entry: Mapping[str, Any],
    *,
    host_platform: str,
    global_supported: Sequence[str],
    zkp_deployment_lock: Mapping[str, Any] | None = None,
) -> ToolPlatformClassification:
    """Derive per-tool support from the reviewed deployment lock.

    Deployment-contract platforms are the support ceiling when present.
    Otherwise exact pins, ``any`` pins, and reviewed ``source`` pins are
    considered. Missing platform metadata fails closed for managed tools.

    Rules (fail-closed):

    * non-managed tools: follow global platform policy only (never PATH);
    * managed + contract excludes host → ``unsupported_here`` (exception);
    * managed + contract includes host but pins lack a runnable artifact for
      this host → ``ambiguous`` (blocker, not an exception);
    * managed + no metadata → ``ambiguous`` (blocker);
    * managed + tool claims support while host is outside global policy →
      ``ambiguous`` (contradiction);
    * ``any`` is always honored when present on the authoritative source;
    * unavailability is not observed here and never becomes unsupported.
    """

    if not host_platform or not isinstance(host_platform, str):
        raise PlatformSupportError("host_platform is required")

    tool_id = str(entry.get("tool_id") or "").strip()
    if not tool_id:
        raise PlatformSupportError("tool entry missing tool_id")

    availability = str(entry.get("availability") or "").strip()
    managed = availability in MANAGED_AVAILABILITIES
    global_set = {str(item) for item in global_supported}
    host_globally_supported = host_platform in global_set

    contract_platforms = _contract_platforms(entry)
    pin_platforms = _pin_platforms(entry)
    declared = sorted(set(contract_platforms) | set(pin_platforms))

    notes: list[str] = []
    blocker_reasons: list[str] = []
    platform_independent = False
    basis: str
    classification: str

    if not managed:
        # In-process, advisor, python package, and external_optional tools are
        # not host binary matrix entries. They track the global policy only.
        if host_globally_supported:
            classification = CLASSIFICATION_SUPPORTED
            basis = "global_platform_policy"
        else:
            classification = CLASSIFICATION_AMBIGUOUS
            basis = "global_platform_policy_host_not_listed"
            blocker_reasons.append("host_outside_global_platform_policy")
            notes.append(
                "non-managed tool cannot be classified unsupported without an "
                "explicit host exclusion; host is outside global policy"
            )
    else:
        observations: list[tuple[str, bool]] = []
        if contract_platforms:
            observations.append(
                (
                    "deployment_contract.supported_platforms",
                    _platform_set_supports(contract_platforms, host_platform),
                )
            )
        if pin_platforms:
            observations.append(
                (
                    "tool.pins.platform",
                    _pin_set_supports(pin_platforms, host_platform),
                )
            )

        if not observations:
            classification = CLASSIFICATION_AMBIGUOUS
            basis = "managed_tool_platform_metadata_missing"
            blocker_reasons.append("managed_tool_platform_metadata_missing")
            notes.append(
                "managed tool has neither deployment_contract.supported_platforms "
                "nor pin platforms; missing metadata is a blocker, not an exception"
            )
        else:
            contract_support: bool | None = (
                observations[0][1] if contract_platforms else None
            )
            pin_support: bool | None = None
            for source, value in observations:
                if source == "tool.pins.platform":
                    pin_support = value
                    break

            # The reviewed deployment contract is the support ceiling. Pins may
            # narrow a claimed-supported contract when no artifact can run here,
            # but a generic/source pin cannot broaden an explicit host exclusion
            # (notably external SecPAL on linux-aarch64).
            tool_supported: bool | None
            if contract_support is False:
                tool_supported = False
                basis = "deployment_contract.supported_platforms"
                notes.append(
                    "explicit deployment-contract host exclusion; pin wildcards "
                    "cannot broaden the contract"
                )
            elif contract_support is True and pin_support is False:
                tool_supported = None
                classification = CLASSIFICATION_AMBIGUOUS
                basis = "supported_contract_without_host_artifact_pin"
                blocker_reasons.append("supported_contract_without_host_artifact_pin")
                notes.append(
                    "deployment contract claims host support but pins provide no "
                    "host/any/source artifact for this platform"
                )
            elif contract_support is True:
                tool_supported = True
                basis = "deployment_contract.supported_platforms"
            else:
                # No contract platforms — pins alone decide.
                tool_supported = bool(pin_support)
                basis = "tool.pins.platform"

            if tool_supported is None:
                # classification already set to ambiguous above
                pass
            elif tool_supported and not host_globally_supported:
                classification = CLASSIFICATION_AMBIGUOUS
                basis = "tool_and_global_platform_policy_contradict"
                blocker_reasons.append("tool_and_global_platform_policy_contradict")
                notes.append(
                    "tool claims support on a host outside platform_policy."
                    "supported_platforms; contradiction is a blocker"
                )
            elif tool_supported:
                classification = CLASSIFICATION_SUPPORTED
            else:
                classification = CLASSIFICATION_UNSUPPORTED

        # ZKP is a platform-independent deployment binding: support is declared
        # via pin ``any`` / deployment lock and is not a host binary exception.
        if tool_id in PLATFORM_INDEPENDENT_DEPLOYMENT_TOOL_IDS:
            pin_any = bool(set(pin_platforms) & UNIVERSAL_PLATFORM_TOKENS)
            contract_any = bool(set(contract_platforms) & UNIVERSAL_PLATFORM_TOKENS)
            has_zkp_lock = bool(zkp_deployment_lock)
            if pin_any or contract_any or has_zkp_lock:
                platform_independent = True
                notes.append(
                    "zkp-circuit is a platform-independent deployment binding; "
                    "host binary exclusion is not applicable"
                )
                if classification == CLASSIFICATION_SUPPORTED:
                    basis = (
                        "platform_independent_deployment_binding"
                        if has_zkp_lock or pin_any or contract_any
                        else basis
                    )

    supported = classification == CLASSIFICATION_SUPPORTED
    ambiguous = classification == CLASSIFICATION_AMBIGUOUS
    exception_eligible = classification == CLASSIFICATION_UNSUPPORTED
    # Ambiguous managed (and global-policy) cases are blockers. Unsupported is
    # a narrow exception, not a missing-capability relabel.
    blocker = bool(blocker_reasons) or ambiguous

    if exception_eligible:
        notes.append(
            "narrow platform exception only: never counts as installed, complete, "
            "authoritative, or production-certified"
        )

    return ToolPlatformClassification(
        tool_id=tool_id,
        host_platform=host_platform,
        availability=availability,
        managed=managed,
        classification=classification,
        supported=supported,
        ambiguous=ambiguous,
        exception_eligible=exception_eligible,
        basis=basis,
        declared_platforms=declared,
        contract_platforms=sorted(set(contract_platforms)),
        pin_platforms=sorted(set(pin_platforms)),
        globally_supported=host_globally_supported,
        platform_independent_deployment_binding=platform_independent,
        blocker=blocker,
        blocker_reasons=sorted(set(blocker_reasons)),
        notes=notes,
    )


def build_platform_exceptions(
    rows: Sequence[ToolPlatformClassification],
) -> list[PlatformException]:
    """Collect narrow platform exceptions from unsupported classifications only."""

    exceptions: list[PlatformException] = []
    for row in rows:
        if not row.exception_eligible:
            continue
        exceptions.append(
            PlatformException(
                tool_id=row.tool_id,
                host_platform=row.host_platform,
                declared_platforms=list(row.declared_platforms),
                basis=row.basis,
                classification=row.classification,
            )
        )
    return exceptions


def classification_digest(
    rows: Sequence[ToolPlatformClassification | Mapping[str, Any]],
    *,
    host_platform: str,
    global_supported: Sequence[str],
) -> str:
    """Final digest over host key + ordered classification rows.

    Includes only the fields that define host support so a lock mutation that
    adds or removes a host platform from a tool's declaration changes the
    digest even when unrelated lock metadata is identical.
    """

    serialized_rows: list[dict[str, Any]] = []
    for row in rows:
        payload = row.to_dict() if isinstance(row, ToolPlatformClassification) else dict(row)
        serialized_rows.append(
            {
                "tool_id": payload.get("tool_id"),
                "host_platform": payload.get("host_platform"),
                "classification": payload.get("classification"),
                "supported": payload.get("supported"),
                "ambiguous": payload.get("ambiguous"),
                "exception_eligible": payload.get("exception_eligible"),
                "basis": payload.get("basis"),
                "declared_platforms": list(payload.get("declared_platforms") or []),
                "contract_platforms": list(payload.get("contract_platforms") or []),
                "pin_platforms": list(payload.get("pin_platforms") or []),
                "platform_independent_deployment_binding": bool(
                    payload.get("platform_independent_deployment_binding")
                ),
                "blocker": bool(payload.get("blocker")),
                "blocker_reasons": list(payload.get("blocker_reasons") or []),
            }
        )
    serialized_rows.sort(key=lambda item: str(item.get("tool_id") or ""))
    return content_digest(
        {
            "interface": INTERFACE,
            "schema_version": SCHEMA_VERSION,
            "host_platform": host_platform,
            "global_supported_platforms": list(global_supported),
            "rows": serialized_rows,
        }
    )


def build_platform_support_report(
    lock: Mapping[str, Any],
    *,
    host_platform: str | None = None,
    zkp_deployment_lock: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify every locked tool for the given (or observed) host platform."""

    host = host_platform or observed_host_platform()
    globals_ = global_supported_platforms(lock)
    tools = lock_tools_by_id(lock)
    if not tools:
        raise PlatformSupportError("lock.tools is empty; nothing to classify")

    rows = [
        classify_tool_platform_support(
            tools[tool_id],
            host_platform=host,
            global_supported=globals_,
            zkp_deployment_lock=zkp_deployment_lock,
        )
        for tool_id in sorted(tools)
    ]
    exceptions = build_platform_exceptions(rows)
    blockers = [row for row in rows if row.blocker]
    supported_ids = [row.tool_id for row in rows if row.supported]
    unsupported_ids = [row.tool_id for row in rows if row.exception_eligible]
    ambiguous_ids = [row.tool_id for row in rows if row.ambiguous]
    platform_independent_ids = [
        row.tool_id for row in rows if row.platform_independent_deployment_binding
    ]
    digest = classification_digest(
        rows,
        host_platform=host,
        global_supported=globals_,
    )

    # Policy invariants for the report surface.
    policy = {
        "never_probe_path": True,
        "never_install": True,
        "never_infer_support_from_path": True,
        "never_convert_unavailability_to_unsupported": True,
        "absent_or_ambiguous_metadata_is_blocker": True,
        "only_explicit_host_exclusion_is_platform_exception": True,
        "any_support_honored": True,
        "source_pin_honored_for_managed_buildability": True,
        "deployment_contract_is_support_ceiling": True,
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "interface": INTERFACE,
        "goal_id": GOAL_ID,
        "task_id": TASK_ID,
        "program": PROGRAM,
        "certification_surface": CERTIFICATION_SURFACE,
        "host_platform": host,
        "host_globally_supported": host in set(globals_),
        "global_supported_platforms": list(globals_),
        "lock_interface": str(lock.get("interface") or ""),
        "lock_schema_version": str(lock.get("schema_version") or ""),
        "tool_count": len(rows),
        "classifications": [row.to_dict() for row in rows],
        "by_tool_id": {row.tool_id: row.to_dict() for row in rows},
        "supported_here_tool_ids": supported_ids,
        "unsupported_here_tool_ids": unsupported_ids,
        "ambiguous_tool_ids": ambiguous_ids,
        "platform_independent_deployment_binding_tool_ids": platform_independent_ids,
        "platform_exceptions": [item.to_dict() for item in exceptions],
        "blockers": [
            {
                "tool_id": row.tool_id,
                "classification": row.classification,
                "basis": row.basis,
                "reasons": list(row.blocker_reasons),
            }
            for row in blockers
        ],
        "blocker_count": len(blockers),
        "exception_count": len(exceptions),
        "policy": policy,
        "classification_digest": digest,
        "final_digest": digest,
        "ok": len(blockers) == 0,
        "status": (
            "platform_support_classified"
            if not blockers
            else "platform_support_blockers_present"
        ),
    }


def classify_repository(
    *,
    repo_root: Path | str | None = None,
    lock_path: Path | str | None = None,
    zkp_lock_path: Path | str | None = None,
    host_platform: str | None = None,
) -> dict[str, Any]:
    """Load the reviewed locks from a repository checkout and classify."""

    root = Path(repo_root) if repo_root is not None else repo_root_from()
    lock_file = Path(lock_path) if lock_path is not None else root / DEFAULT_LOCK_RELATIVE
    zkp_file = (
        Path(zkp_lock_path)
        if zkp_lock_path is not None
        else root / DEFAULT_ZKP_LOCK_RELATIVE
    )
    lock = load_toolchain_lock(lock_file)
    zkp_lock = load_zkp_deployment_lock(zkp_file)
    report = build_platform_support_report(
        lock,
        host_platform=host_platform,
        zkp_deployment_lock=zkp_lock,
    )
    report["lock_path"] = str(lock_file)
    report["zkp_lock_path"] = str(zkp_file) if zkp_file.is_file() else None
    report["zkp_deployment_lock_bound"] = zkp_lock is not None
    report["repo_root"] = str(root.resolve())
    return report


def mutate_tool_supported_platforms(
    lock: Mapping[str, Any],
    tool_id: str,
    *,
    add: Sequence[str] | None = None,
    remove: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return a deep-copied lock with deployment-contract platforms mutated.

    Used by validation to prove that adding or removing ``linux-aarch64``
    changes classification and the final digest. Does not write to disk.
    """

    mutated = copy.deepcopy(dict(lock))
    tools = mutated.get("tools")
    if not isinstance(tools, list):
        raise PlatformSupportError("lock.tools must be a list")
    found = False
    for entry in tools:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("tool_id") or "") != tool_id:
            continue
        found = True
        contract = entry.get("deployment_contract")
        if not isinstance(contract, dict):
            contract = {}
            entry["deployment_contract"] = contract
        platforms = [
            str(item)
            for item in (contract.get("supported_platforms") or [])
            if item
        ]
        remove_set = {str(item) for item in (remove or [])}
        platforms = [item for item in platforms if item not in remove_set]
        for item in add or []:
            text = str(item)
            if text and text not in platforms:
                platforms.append(text)
        contract["supported_platforms"] = platforms
        break
    if not found:
        raise PlatformSupportError(f"tool_id {tool_id!r} not found in lock.tools")
    return mutated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Classify host-platform support for every formal-verification "
            f"toolchain lock entry ({INTERFACE})."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root (defaults to auto-detect)",
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=None,
        help="Path to formal_verification_toolchains.lock.json",
    )
    parser.add_argument(
        "--zkp-lock",
        type=Path,
        default=None,
        help="Path to formal_verification_zkp_deployment.lock.json",
    )
    parser.add_argument(
        "--host-platform",
        type=str,
        default=None,
        help="Override normalized host platform (e.g. linux-aarch64)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the full classification report as JSON",
    )
    parser.add_argument(
        "--tool",
        action="append",
        default=[],
        help="Restrict output to one or more tool_id values (repeatable)",
    )
    args = parser.parse_args(argv)

    report = classify_repository(
        repo_root=args.repo_root,
        lock_path=args.lock,
        zkp_lock_path=args.zkp_lock,
        host_platform=args.host_platform,
    )

    if args.tool:
        wanted = set(args.tool)
        report = dict(report)
        report["classifications"] = [
            row
            for row in report["classifications"]
            if row.get("tool_id") in wanted
        ]
        report["by_tool_id"] = {
            key: value
            for key, value in report["by_tool_id"].items()
            if key in wanted
        }

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    print(f"{INTERFACE} host={report['host_platform']} tools={report['tool_count']}")
    print(f"final_digest={report['final_digest']}")
    print(
        f"supported={len(report['supported_here_tool_ids'])} "
        f"unsupported={len(report['unsupported_here_tool_ids'])} "
        f"ambiguous={len(report['ambiguous_tool_ids'])} "
        f"exceptions={report['exception_count']} blockers={report['blocker_count']}"
    )
    for row in report["classifications"]:
        flags = []
        if row.get("exception_eligible"):
            flags.append("exception")
        if row.get("blocker"):
            flags.append("blocker")
        if row.get("platform_independent_deployment_binding"):
            flags.append("platform-independent")
        flag_text = f" [{','.join(flags)}]" if flags else ""
        print(
            f"  {row['tool_id']:24} {row['classification']:18} "
            f"basis={row['basis']}{flag_text}"
        )
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "INTERFACE",
    "SCHEMA_VERSION",
    "GOAL_ID",
    "TASK_ID",
    "PROGRAM",
    "CERTIFICATION_SURFACE",
    "DEFAULT_LOCK_RELATIVE",
    "DEFAULT_ZKP_LOCK_RELATIVE",
    "CLASSIFICATION_SUPPORTED",
    "CLASSIFICATION_UNSUPPORTED",
    "CLASSIFICATION_AMBIGUOUS",
    "VALID_CLASSIFICATIONS",
    "PLATFORM_INDEPENDENT_DEPLOYMENT_TOOL_IDS",
    "PlatformSupportError",
    "ToolPlatformClassification",
    "PlatformException",
    "repo_root_from",
    "content_digest",
    "normalize_host_platform",
    "observed_host_platform",
    "load_toolchain_lock",
    "load_zkp_deployment_lock",
    "lock_tools_by_id",
    "global_supported_platforms",
    "classify_tool_platform_support",
    "build_platform_exceptions",
    "classification_digest",
    "build_platform_support_report",
    "classify_repository",
    "mutate_tool_supported_platforms",
    "main",
]
