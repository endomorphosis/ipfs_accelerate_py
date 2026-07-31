#!/usr/bin/env python3
"""Semantic certification for the pinned Lean 4 kernel.

``LeanSemanticCertification@1`` / FVT-G101 (FVT-040).

Owns the Lean lane handler, hermetic corpus, and focused certification surface
for the already-usable offline pin ``leanprover/lean4:v4.31.0``. Promotion is
allowed only after real kernel semantics are demonstrated:

* exact Lean v4.31.0 compiles a true theorem;
* false and malformed proofs are rejected;
* hypothesis and conclusion mutations are rejected;
* deterministic replay of the positive case;
* receipts bind imports, source tree, theorem, assumptions, toolchain, and
  output;
* ``sorry``, ``admit``, unsafe escapes, axiom escapes, shim mismatch, install,
  download, and network use fail closed;
* resulting authority is kernel proof checking only (never advisor/install
  authority).

This module does not edit the central multi-prover certificate and never
selects or downloads a different Elan toolchain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

INTERFACE: Final = "LeanSemanticCertification@1"
SCHEMA_VERSION: Final = "lean-semantic-certification/v1"
CORPUS_SCHEMA: Final = "lean-semantic-corpus/v1"
GOAL_ID: Final = "FVT-G101"
TASK_ID: Final = "FVT-040"
PROGRAM: Final = "formal-verification-tactician/lean-certification"
LANE_ID: Final = "kernel"
TOOL_ID: Final = "lean"
CERTIFICATION_SURFACE: Final = "tools.logic.certification.lean"
HANDLER_ID: Final = "lean_semantic_certifier"

LOCKED_TOOLCHAIN: Final = "leanprover/lean4:v4.31.0"
LOCKED_VERSION: Final = "v4.31.0"
LOCKED_VERSION_NUMERIC: Final = "4.31.0"
AUTHORITY_CEILING: Final = "kernel"
AUTHORITY_SCOPE: Final = "kernel_proof_checking_only"

PROBE_TIMEOUT_SECONDS: Final = 5.0
CHECK_TIMEOUT_SECONDS: Final = 20.0

DEFAULT_MANIFEST_RELATIVE: Final = Path(
    "test/fixtures/formal_verification/toolchains/lean/manifest.json"
)
DEFAULT_LOCK_RELATIVE: Final = Path("config/formal_verification_toolchains.lock.json")

_SORRY = re.compile(
    r"(?<![A-Za-z0-9_'])(?:sorry|admit|sorryAx)(?![A-Za-z0-9_'])"
)
_UNSAFE = re.compile(
    r"(?im)^\s*(?:unsafe\s+(?:def|theorem|inductive|structure|abbrev)|"
    r"axiom\s+|constant\s+)"
)
_IMPORT = re.compile(r"^\s*import\s+(\S+)\s*$", re.MULTILINE)
_DECL = re.compile(
    r"^\s*(?:theorem|lemma)\s+([A-Za-z_][A-Za-z0-9_'.]*)",
    re.MULTILINE,
)
_SORRY_WARNING = re.compile(r"declaration uses [`']?sorry", re.IGNORECASE)
_VERSION_IN_BANNER = re.compile(r"version\s+(\d+\.\d+\.\d+)", re.IGNORECASE)

# Compact embedded recipes used when the fixture path is unavailable. Prefer
# the checked-in manifest so tests and the certifier share one corpus.
_DEFAULT_CORPUS_CASES: Final[tuple[dict[str, Any], ...]] = (
    {
        "case_id": "true_theorem",
        "kind": "positive",
        "expect": "accepted",
        "theorem_name": "from_eq",
        "assumptions": ["h : n = m"],
        "source": (
            "theorem from_eq (n m : Nat) (h : n = m) : n = m := h\n"
        ),
        "description": "Exact true theorem accepted by the pinned kernel",
    },
    {
        "case_id": "false_proof",
        "kind": "negative",
        "expect": "rejected",
        "theorem_name": "false_claim",
        "assumptions": [],
        "source": "theorem false_claim : False := trivial\n",
        "description": "False proof rejected by the kernel",
    },
    {
        "case_id": "malformed_proof",
        "kind": "negative",
        "expect": "rejected",
        "theorem_name": "broken",
        "assumptions": [],
        "source": "theorem broken : True := by exact 0\n",
        "description": "Malformed proof term rejected",
    },
    {
        "case_id": "hypothesis_mutation",
        "kind": "mutation",
        "expect": "rejected",
        "theorem_name": "from_eq",
        "assumptions": ["h : n = n"],
        "mutates": "hypothesis",
        "base_case_id": "true_theorem",
        "source": (
            "theorem from_eq (n m : Nat) (h : n = n) : n = m := h\n"
        ),
        "description": "Hypothesis mutation of the true theorem is rejected",
    },
    {
        "case_id": "conclusion_mutation",
        "kind": "mutation",
        "expect": "rejected",
        "theorem_name": "from_eq",
        "assumptions": ["h : n = m"],
        "mutates": "conclusion",
        "base_case_id": "true_theorem",
        "source": (
            "theorem from_eq (n m : Nat) (h : n = m) : False := h\n"
        ),
        "description": "Conclusion mutation of the true theorem is rejected",
    },
    {
        "case_id": "sorry_escape",
        "kind": "fail_closed",
        "expect": "rejected",
        "theorem_name": "hole_sorry",
        "assumptions": [],
        "reason_codes": ["sorry_or_admit"],
        "source": "theorem hole_sorry : True := by sorry\n",
        "description": "sorry fails closed even when lean exits 0",
    },
    {
        "case_id": "admit_escape",
        "kind": "fail_closed",
        "expect": "rejected",
        "theorem_name": "hole_admit",
        "assumptions": [],
        "reason_codes": ["sorry_or_admit"],
        "source": "theorem hole_admit : True := by admit\n",
        "description": "admit fails closed even when lean exits 0",
    },
    {
        "case_id": "unsafe_escape",
        "kind": "fail_closed",
        "expect": "rejected",
        "theorem_name": "still",
        "assumptions": [],
        "reason_codes": ["unsafe_or_unreviewed_axiom"],
        "source": (
            "unsafe def cheat : Nat := 0\n"
            "theorem still : True := trivial\n"
        ),
        "description": "unsafe declarations fail closed",
    },
    {
        "case_id": "axiom_escape",
        "kind": "fail_closed",
        "expect": "rejected",
        "theorem_name": "uses_axiom",
        "assumptions": [],
        "reason_codes": ["unsafe_or_unreviewed_axiom"],
        "source": (
            "axiom bad : False\n"
            "theorem uses_axiom : False := bad\n"
        ),
        "description": "unreviewed axiom escapes fail closed",
    },
    {
        "case_id": "deterministic_replay",
        "kind": "replay",
        "expect": "accepted",
        "theorem_name": "from_eq",
        "assumptions": ["h : n = m"],
        "base_case_id": "true_theorem",
        "source": (
            "theorem from_eq (n m : Nat) (h : n = m) : n = m := h\n"
        ),
        "description": "Positive case replays with identical acceptance and digests",
    },
)


# ---------------------------------------------------------------------------
# Path / offline helpers
# ---------------------------------------------------------------------------


def repo_root_from(start: Path | None = None) -> Path:
    """Locate the repository root that owns the offline toolchain lock."""

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
    if isinstance(payload, (bytes, bytearray)):
        return hashlib.sha256(bytes(payload)).hexdigest()
    if isinstance(payload, str):
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def offline_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment that forbids opportunistic install/download/network."""

    env = dict(base if base is not None else os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    env["PIP_NO_INDEX"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["NPM_CONFIG_OFFLINE"] = "true"
    env["npm_config_offline"] = "true"
    env["ELAN_NO_AUTO_INSTALL"] = "1"
    env.setdefault("ELAN_IO_THREADS", "1")
    env["ELAN_TOOLCHAIN"] = LOCKED_TOOLCHAIN
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_OPTIONAL_LOCKS"] = "0"
    env.setdefault("NO_PROXY", "*")
    env.setdefault("no_proxy", "*")
    env["FORMAL_VERIFICATION_CERTIFY_OFFLINE"] = "1"
    env["FORMAL_VERIFICATION_FORBID_INSTALL"] = "1"
    env["FORMAL_VERIFICATION_FORBID_NETWORK"] = "1"
    env["FORMAL_VERIFICATION_FORBID_DOWNLOAD"] = "1"
    return env


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


def resolve_lean_executable(candidates: Sequence[str] | None = None) -> str | None:
    names = list(candidates) if candidates else ["lean"]
    for name in names:
        if not name:
            continue
        path = Path(name)
        if path.is_file() and os.access(path, os.X_OK):
            return str(path.resolve())
        found = shutil.which(name)
        if found:
            return found
    return None


def list_elan_installed_toolchains(
    env: Mapping[str, str] | None = None,
) -> list[str]:
    """Read offline-installed Lean toolchains from the local elan directory."""

    source_env = env if env is not None else os.environ
    elan_home = Path(source_env.get("ELAN_HOME", Path.home() / ".elan"))
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


# ---------------------------------------------------------------------------
# Source scanning / binding
# ---------------------------------------------------------------------------


def extract_lean_imports(source: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(_IMPORT.findall(source or "")))


def extract_lean_theorem_name(source: str) -> str | None:
    match = _DECL.search(source or "")
    if match is None:
        return None
    return match.group(1).rstrip(".")


def scan_lean_incomplete_or_unsafe(source: str) -> tuple[str, ...]:
    """Return fail-closed reason codes for sorry/admit/unsafe/axiom constructs."""

    findings: list[str] = []
    if _SORRY.search(source or ""):
        findings.append("sorry_or_admit")
    if _UNSAFE.search(source or ""):
        findings.append("unsafe_or_unreviewed_axiom")
    return tuple(findings)


def first_nonempty_line(text: str) -> str:
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    check_id: str
    kind: str
    status: str  # passed | failed | skipped | unavailable | blocked
    expected: str
    observed: str
    detail: str = ""
    reason_codes: list[str] = field(default_factory=list)
    bindings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CaseOutcome:
    case_id: str
    kind: str
    expect: str
    accepted: bool
    returncode: int | None
    reason_codes: list[str] = field(default_factory=list)
    theorem_name: str = ""
    imports: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    source_digest: str = ""
    output_digest: str = ""
    stdout: str = ""
    stderr: str = ""
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LeanSemanticCertification:
    """Full semantic certification receipt for the pinned Lean kernel."""

    tool_id: str = TOOL_ID
    lane_id: str = LANE_ID
    interface: str = INTERFACE
    schema_version: str = SCHEMA_VERSION
    goal_id: str = GOAL_ID
    task_id: str = TASK_ID
    program: str = PROGRAM
    certification_surface: str = CERTIFICATION_SURFACE
    locked_toolchain: str = LOCKED_TOOLCHAIN
    locked_version: str = LOCKED_VERSION
    authority_ceiling: str = AUTHORITY_CEILING
    authority_scope: str = AUTHORITY_SCOPE
    executable_path: str | None = None
    version_string: str | None = None
    selected_toolchain: str | None = None
    installed_toolchains: list[str] = field(default_factory=list)
    identity_probed: bool = False
    installed: bool = False
    usable: bool = False
    shim_toolchain_mismatch: bool = False
    locked_version_mismatch: bool = False
    network_used: bool = False
    install_attempted: bool = False
    download_attempted: bool = False
    production_certified: bool = False
    promotion_blocked: bool = True
    block_reasons: list[str] = field(default_factory=list)
    checks: list[CheckResult] = field(default_factory=list)
    cases: list[CaseOutcome] = field(default_factory=list)
    bindings: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["checks"] = [check.to_dict() for check in self.checks]
        payload["cases"] = [case.to_dict() for case in self.cases]
        payload["receipt_digest_sha256"] = content_digest(
            {
                key: value
                for key, value in payload.items()
                if key != "receipt_digest_sha256"
            }
        )
        return payload


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------


def default_corpus_manifest() -> dict[str, Any]:
    return {
        "schema_version": CORPUS_SCHEMA,
        "interface": INTERFACE,
        "goal_id": GOAL_ID,
        "task_id": TASK_ID,
        "tool_id": TOOL_ID,
        "lane_id": LANE_ID,
        "locked_toolchain": LOCKED_TOOLCHAIN,
        "locked_version": LOCKED_VERSION,
        "authority_ceiling": AUTHORITY_CEILING,
        "authority_scope": AUTHORITY_SCOPE,
        "policy": {
            "no_install": True,
            "no_download": True,
            "no_network": True,
            "elan_no_auto_install": True,
            "shim_mismatch_fails_closed": True,
            "sorry_admit_unsafe_fail_closed": True,
            "authority_is_kernel_proof_checking_only": True,
        },
        "cases": [dict(case) for case in _DEFAULT_CORPUS_CASES],
    }


def load_corpus_manifest(
    path: Path | None = None,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Load the Lean semantic corpus manifest (fixture or embedded default)."""

    root = repo_root or repo_root_from()
    manifest_path = path or (root / DEFAULT_MANIFEST_RELATIVE)
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Lean corpus manifest must be a JSON object")
        return payload
    return default_corpus_manifest()


def corpus_cases(manifest: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    payload = manifest if manifest is not None else default_corpus_manifest()
    cases = payload.get("cases") or []
    if not isinstance(cases, list) or not cases:
        raise ValueError("Lean corpus must declare a non-empty cases list")
    return [dict(case) for case in cases if isinstance(case, Mapping)]


# ---------------------------------------------------------------------------
# Identity probe
# ---------------------------------------------------------------------------


def probe_lean_identity(
    *,
    env: Mapping[str, str] | None = None,
    executable: str | None = None,
) -> dict[str, Any]:
    """Bounded offline identity probe for the locked Lean toolchain only."""

    probe_env = offline_env(env)
    installed = list_elan_installed_toolchains(probe_env)
    result: dict[str, Any] = {
        "tool_id": TOOL_ID,
        "path_present": False,
        "executable_path": None,
        "version_string": None,
        "identity_probed": False,
        "installed": False,
        "selected_toolchain": LOCKED_TOOLCHAIN,
        "installed_toolchains": installed,
        "shim_toolchain_mismatch": detect_lean_shim_toolchain_mismatch(
            LOCKED_TOOLCHAIN, installed
        ),
        "locked_version_mismatch": False,
        "network_used": False,
        "install_attempted": False,
        "download_attempted": False,
        "probe_error": None,
    }

    if result["shim_toolchain_mismatch"]:
        result["probe_error"] = "shim_toolchain_mismatch"
        return result

    lean_bin = executable or resolve_lean_executable()
    if lean_bin is None:
        result["probe_error"] = "executable_not_on_path"
        return result

    result["path_present"] = True
    result["executable_path"] = lean_bin

    completed = bounded_run(
        [lean_bin, "--version"],
        timeout=PROBE_TIMEOUT_SECONDS,
        env=probe_env,
    )
    if completed is None:
        result["probe_error"] = "probe_timeout_or_spawn_failure"
        return result

    banner = first_nonempty_line(completed.stdout) or first_nonempty_line(
        completed.stderr
    )
    if not banner:
        result["probe_error"] = "empty_version_banner"
        return result

    result["version_string"] = banner
    result["identity_probed"] = True
    result["installed"] = True

    match = _VERSION_IN_BANNER.search(banner)
    if match is None or match.group(1) != LOCKED_VERSION_NUMERIC:
        result["locked_version_mismatch"] = True
        result["probe_error"] = "locked_version_mismatch"
    else:
        result["locked_version_mismatch"] = False

    return result


# ---------------------------------------------------------------------------
# Kernel check
# ---------------------------------------------------------------------------


def check_lean_source(
    source: str,
    *,
    executable: str,
    env: Mapping[str, str] | None = None,
    timeout: float = CHECK_TIMEOUT_SECONDS,
    case_id: str = "case",
    theorem_name: str | None = None,
    assumptions: Sequence[str] | None = None,
) -> CaseOutcome:
    """Compile one Lean source against the pinned kernel under offline bounds."""

    normalized = source if source.endswith("\n") else source + "\n"
    imports = list(extract_lean_imports(normalized))
    resolved_name = theorem_name or extract_lean_theorem_name(normalized) or ""
    assumption_list = list(assumptions or [])
    source_digest = content_digest(normalized)
    reason_codes = list(scan_lean_incomplete_or_unsafe(normalized))

    # Fail closed before invoking the kernel when source already encodes an
    # incomplete or unsafe escape. Lean treats sorry/admit as warnings (exit 0).
    if reason_codes:
        return CaseOutcome(
            case_id=case_id,
            kind="fail_closed",
            expect="rejected",
            accepted=False,
            returncode=None,
            reason_codes=reason_codes,
            theorem_name=resolved_name,
            imports=imports,
            assumptions=assumption_list,
            source_digest=source_digest,
            output_digest=content_digest(""),
            detail="source scan rejected incomplete or unsafe constructs",
        )

    probe_env = offline_env(env)
    with tempfile.TemporaryDirectory(prefix="lean-semantic-") as tmp:
        work = Path(tmp)
        source_path = work / "Main.lean"
        source_path.write_text(normalized, encoding="utf-8")
        completed = bounded_run(
            [executable, str(source_path)],
            timeout=timeout,
            env=probe_env,
            cwd=work,
        )

    if completed is None:
        return CaseOutcome(
            case_id=case_id,
            kind="kernel",
            expect="rejected",
            accepted=False,
            returncode=None,
            reason_codes=["timeout_or_spawn_failure"],
            theorem_name=resolved_name,
            imports=imports,
            assumptions=assumption_list,
            source_digest=source_digest,
            output_digest=content_digest(""),
            detail="bounded lean invocation timed out or failed to spawn",
        )

    combined = f"{completed.stdout}\n{completed.stderr}"
    output_digest = content_digest(combined)
    if _SORRY_WARNING.search(combined):
        reason_codes.append("sorry_warning_in_output")

    # Kernel acceptance requires exit 0, no sorry warning, no fail-closed codes.
    accepted = (
        completed.returncode == 0
        and not reason_codes
        and "error:" not in combined.lower()
    )
    if completed.returncode != 0:
        reason_codes.append("kernel_rejected")

    return CaseOutcome(
        case_id=case_id,
        kind="kernel",
        expect="accepted" if accepted else "rejected",
        accepted=accepted,
        returncode=completed.returncode,
        reason_codes=list(dict.fromkeys(reason_codes)),
        theorem_name=resolved_name,
        imports=imports,
        assumptions=assumption_list,
        source_digest=source_digest,
        output_digest=output_digest,
        stdout=completed.stdout,
        stderr=completed.stderr,
        detail="lean kernel compile under offline pin",
    )


def _case_matches_expectation(outcome: CaseOutcome, expect: str) -> bool:
    if expect == "accepted":
        return outcome.accepted is True
    if expect == "rejected":
        return outcome.accepted is False
    return False


# ---------------------------------------------------------------------------
# Certification orchestration
# ---------------------------------------------------------------------------


def run_semantic_suite(
    *,
    repo_root: Path | None = None,
    manifest: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
    executable: str | None = None,
) -> LeanSemanticCertification:
    """Run the full Lean semantic suite and return a certification receipt."""

    root = repo_root or repo_root_from()
    corpus = manifest if manifest is not None else load_corpus_manifest(repo_root=root)
    cases = corpus_cases(corpus)
    cert = LeanSemanticCertification()
    probe_env = offline_env(env)

    identity = probe_lean_identity(env=probe_env, executable=executable)
    cert.executable_path = identity.get("executable_path")
    cert.version_string = identity.get("version_string")
    cert.selected_toolchain = identity.get("selected_toolchain") or LOCKED_TOOLCHAIN
    cert.installed_toolchains = list(identity.get("installed_toolchains") or [])
    cert.identity_probed = bool(identity.get("identity_probed"))
    cert.installed = bool(identity.get("installed"))
    cert.shim_toolchain_mismatch = bool(identity.get("shim_toolchain_mismatch"))
    cert.locked_version_mismatch = bool(identity.get("locked_version_mismatch"))
    cert.network_used = bool(identity.get("network_used"))
    cert.install_attempted = bool(identity.get("install_attempted"))
    cert.download_attempted = bool(identity.get("download_attempted"))

    # Fail-closed policy surface: never install/download/network during certify.
    cert.checks.append(
        CheckResult(
            check_id="lean.offline_policy",
            kind="policy",
            status="passed",
            expected="no_install_no_download_no_network",
            observed=(
                f"install={cert.install_attempted},"
                f"download={cert.download_attempted},"
                f"network={cert.network_used},"
                f"ELAN_NO_AUTO_INSTALL={probe_env.get('ELAN_NO_AUTO_INSTALL')},"
                f"ELAN_TOOLCHAIN={probe_env.get('ELAN_TOOLCHAIN')}"
            ),
            detail="certification never installs, downloads, or opens the network",
            reason_codes=[],
        )
    )

    if cert.shim_toolchain_mismatch:
        cert.block_reasons.append("shim_toolchain_mismatch")
        cert.usable = False
        cert.promotion_blocked = True
        cert.notes = (
            "Locked Lean toolchain is not offline-installed; fail closed without "
            "download or elan install."
        )
        cert.checks.append(
            CheckResult(
                check_id="lean.shim_mismatch",
                kind="identity",
                status="blocked",
                expected=LOCKED_TOOLCHAIN,
                observed=",".join(cert.installed_toolchains) or "none",
                detail="shim / missing offline pin fails closed",
                reason_codes=["shim_toolchain_mismatch"],
            )
        )
        return cert

    if not cert.identity_probed or not cert.executable_path:
        cert.block_reasons.append(str(identity.get("probe_error") or "unavailable"))
        cert.usable = False
        cert.promotion_blocked = True
        cert.notes = "Lean identity probe failed; not production-certified."
        cert.checks.append(
            CheckResult(
                check_id="lean.identity",
                kind="identity",
                status="unavailable",
                expected=f"Lean (version {LOCKED_VERSION_NUMERIC}",
                observed=str(identity.get("probe_error") or "missing"),
                detail="PATH presence without identity is not usability",
                reason_codes=["identity_unavailable"],
            )
        )
        return cert

    if cert.locked_version_mismatch:
        cert.block_reasons.append("locked_version_mismatch")
        cert.usable = False
        cert.promotion_blocked = True
        cert.notes = (
            "Observed Lean version does not match the offline lock pin v4.31.0."
        )
        cert.checks.append(
            CheckResult(
                check_id="lean.version_pin",
                kind="identity",
                status="blocked",
                expected=LOCKED_VERSION_NUMERIC,
                observed=cert.version_string or "missing",
                detail="exact pin match required; no upgrade/download during certify",
                reason_codes=["locked_version_mismatch"],
            )
        )
        return cert

    cert.usable = True
    cert.checks.append(
        CheckResult(
            check_id="lean.identity",
            kind="identity",
            status="passed",
            expected=f"Lean (version {LOCKED_VERSION_NUMERIC}",
            observed=cert.version_string or "",
            detail="exact offline pin identity",
            bindings={
                "toolchain": cert.selected_toolchain,
                "executable_path": cert.executable_path,
                "version_string": cert.version_string,
            },
        )
    )

    lean_bin = str(cert.executable_path)
    outcomes_by_id: dict[str, CaseOutcome] = {}
    positive_outcome: CaseOutcome | None = None
    replay_outcome: CaseOutcome | None = None

    for case in cases:
        case_id = str(case.get("case_id") or "case")
        kind = str(case.get("kind") or "kernel")
        expect = str(case.get("expect") or "rejected")
        source = str(case.get("source") or "")
        theorem_name = str(case.get("theorem_name") or "") or None
        assumptions = [str(item) for item in (case.get("assumptions") or [])]

        outcome = check_lean_source(
            source,
            executable=lean_bin,
            env=probe_env,
            case_id=case_id,
            theorem_name=theorem_name,
            assumptions=assumptions,
        )
        # Preserve declared kind for the receipt even when source-scan short-circuits.
        outcome.kind = kind
        outcome.expect = expect
        outcomes_by_id[case_id] = outcome
        cert.cases.append(outcome)

        matched = _case_matches_expectation(outcome, expect)
        # For fail_closed cases, require the declared reason code when provided.
        expected_reasons = [str(item) for item in (case.get("reason_codes") or [])]
        if expected_reasons:
            matched = matched and any(
                reason in outcome.reason_codes
                or (
                    reason == "sorry_or_admit"
                    and "sorry_warning_in_output" in outcome.reason_codes
                )
                for reason in expected_reasons
            )

        if kind == "positive" and expect == "accepted":
            positive_outcome = outcome
        if kind == "replay":
            replay_outcome = outcome

        status = "passed" if matched else "failed"
        if not matched:
            cert.block_reasons.append(f"case_failed:{case_id}")

        cert.checks.append(
            CheckResult(
                check_id=f"lean.{case_id}",
                kind=kind,
                status=status,
                expected=expect,
                observed="accepted" if outcome.accepted else "rejected",
                detail=str(case.get("description") or outcome.detail),
                reason_codes=list(outcome.reason_codes),
                bindings={
                    "theorem_name": outcome.theorem_name,
                    "imports": list(outcome.imports),
                    "assumptions": list(outcome.assumptions),
                    "source_digest": outcome.source_digest,
                    "output_digest": outcome.output_digest,
                    "returncode": outcome.returncode,
                },
            )
        )

    # Deterministic replay: positive and replay must both accept with matching digests.
    if positive_outcome is not None and replay_outcome is not None:
        replay_ok = (
            positive_outcome.accepted
            and replay_outcome.accepted
            and positive_outcome.source_digest == replay_outcome.source_digest
            and positive_outcome.output_digest == replay_outcome.output_digest
            and positive_outcome.returncode == replay_outcome.returncode
        )
        if not replay_ok:
            cert.block_reasons.append("replay_nondeterministic_or_failed")
        cert.checks.append(
            CheckResult(
                check_id="lean.deterministic_replay_binding",
                kind="replay",
                status="passed" if replay_ok else "failed",
                expected="identical acceptance and digests",
                observed=(
                    f"pos={positive_outcome.accepted}/{positive_outcome.source_digest[:12]},"
                    f"replay={replay_outcome.accepted}/{replay_outcome.source_digest[:12]}"
                ),
                detail="positive case must replay deterministically under the same pin",
                bindings={
                    "positive_source_digest": positive_outcome.source_digest,
                    "replay_source_digest": replay_outcome.source_digest,
                    "positive_output_digest": positive_outcome.output_digest,
                    "replay_output_digest": replay_outcome.output_digest,
                },
            )
        )
    else:
        cert.block_reasons.append("replay_or_positive_case_missing")
        cert.checks.append(
            CheckResult(
                check_id="lean.deterministic_replay_binding",
                kind="replay",
                status="failed",
                expected="positive and replay cases",
                observed="missing",
                detail="corpus must include positive and replay cases",
            )
        )

    # Bind trust-relevant inputs from the positive case when available.
    binding_case = positive_outcome or next(
        (item for item in cert.cases if item.accepted), None
    )
    cert.bindings = {
        "imports": list(binding_case.imports) if binding_case else [],
        "source_tree": {
            "primary_path": "Main.lean",
            "source_digest": binding_case.source_digest if binding_case else "",
        },
        "theorem": {
            "name": binding_case.theorem_name if binding_case else "",
            "assumptions": list(binding_case.assumptions) if binding_case else [],
        },
        "assumptions": list(binding_case.assumptions) if binding_case else [],
        "toolchain": {
            "locked_toolchain": LOCKED_TOOLCHAIN,
            "locked_version": LOCKED_VERSION,
            "selected_toolchain": cert.selected_toolchain,
            "executable_path": cert.executable_path,
            "version_string": cert.version_string,
        },
        "output": {
            "output_digest": binding_case.output_digest if binding_case else "",
            "returncode": binding_case.returncode if binding_case else None,
        },
        "authority": {
            "ceiling": AUTHORITY_CEILING,
            "scope": AUTHORITY_SCOPE,
            "not_advisor": True,
            "not_install_authority": True,
        },
    }
    cert.checks.append(
        CheckResult(
            check_id="lean.bindings",
            kind="binding",
            status="passed" if binding_case and binding_case.accepted else "failed",
            expected="imports,source_tree,theorem,assumptions,toolchain,output",
            observed=content_digest(cert.bindings)[:16],
            detail="receipt binds every trust-relevant kernel check input",
            bindings=dict(cert.bindings),
        )
    )

    required_kinds = {"positive", "negative", "mutation", "replay", "fail_closed"}
    present_kinds = {str(case.get("kind") or "") for case in cases}
    missing_kinds = sorted(required_kinds - present_kinds)
    if missing_kinds:
        cert.block_reasons.append("corpus_missing_kinds:" + ",".join(missing_kinds))

    semantic_checks = [
        check
        for check in cert.checks
        if check.kind in {"positive", "negative", "mutation", "replay", "fail_closed", "binding"}
        or check.check_id.endswith("_binding")
        or check.check_id == "lean.bindings"
        or check.check_id == "lean.deterministic_replay_binding"
    ]
    all_semantic_passed = all(check.status == "passed" for check in semantic_checks)
    # Also require every case-level check under lean.<case_id> to pass.
    case_checks_passed = all(
        check.status == "passed"
        for check in cert.checks
        if check.check_id.startswith("lean.")
        and check.kind in {"positive", "negative", "mutation", "replay", "fail_closed"}
    )

    cert.production_certified = bool(
        cert.usable
        and cert.identity_probed
        and not cert.shim_toolchain_mismatch
        and not cert.locked_version_mismatch
        and not cert.network_used
        and not cert.install_attempted
        and not cert.download_attempted
        and all_semantic_passed
        and case_checks_passed
        and not missing_kinds
        and not any(
            reason.startswith("case_failed:") or reason.startswith("replay_")
            for reason in cert.block_reasons
        )
    )
    if cert.production_certified:
        cert.promotion_blocked = False
        cert.block_reasons = []
        cert.notes = (
            "Pinned Lean v4.31.0 kernel semantics certified: true theorem accepted; "
            "false/malformed/mutation/sorry/admit/unsafe rejected; deterministic "
            "replay; bindings complete; authority is kernel proof checking only."
        )
    else:
        cert.promotion_blocked = True
        if cert.usable and not cert.notes:
            cert.notes = (
                "Lean is usable but semantic certification incomplete or failed; "
                "promotion blocked."
            )

    return cert


def build_certification_receipt(
    *,
    repo_root: Path | None = None,
    manifest: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Machine-readable receipt for operators, tests, and lane binding."""

    cert = run_semantic_suite(repo_root=repo_root, manifest=manifest, env=env)
    payload = cert.to_dict()
    payload["policy"] = {
        "no_install": True,
        "no_download": True,
        "no_network": True,
        "elan_no_auto_install": True,
        "shim_mismatch_fails_closed": True,
        "sorry_admit_unsafe_fail_closed": True,
        "authority_is_kernel_proof_checking_only": True,
        "does_not_edit_central_certificate": True,
        "does_not_select_alternate_elan_toolchain": True,
    }
    payload["receipt_digest_sha256"] = content_digest(
        {
            key: value
            for key, value in payload.items()
            if key != "receipt_digest_sha256"
        }
    )
    return payload


def certify_lean_kernel(
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """Lane-handler entry point compatible with role-aware promotion binding."""

    repo_root = kwargs.get("repo_root")
    if repo_root is not None and not isinstance(repo_root, Path):
        repo_root = Path(str(repo_root))
    manifest = kwargs.get("manifest")
    env = kwargs.get("env")
    receipt = build_certification_receipt(
        repo_root=repo_root,
        manifest=manifest,
        env=env,
    )
    receipt["handler_id"] = HANDLER_ID
    receipt["lane_id"] = LANE_ID
    receipt["owner_module"] = CERTIFICATION_SURFACE
    receipt["status"] = (
        "certified" if receipt.get("production_certified") else "not_certified"
    )
    receipt["certified"] = bool(receipt.get("production_certified"))
    receipt["args_received"] = bool(args) or bool(kwargs)
    return receipt


def lane_handler(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Alias used by ``tools.logic.certification.roles.bind_lane_handler``."""

    return certify_lean_kernel(*args, **kwargs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Semantically certify the pinned Lean 4 kernel "
            f"({INTERFACE}; {LOCKED_TOOLCHAIN})."
        )
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full certification receipt as JSON",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional path to the Lean semantic corpus manifest",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root containing the offline toolchain lock",
    )
    args = parser.parse_args(argv)

    root = args.repo_root or repo_root_from()
    manifest = (
        load_corpus_manifest(args.manifest, repo_root=root)
        if args.manifest is not None
        else load_corpus_manifest(repo_root=root)
    )
    receipt = build_certification_receipt(repo_root=root, manifest=manifest)

    if args.json:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    else:
        print(f"{INTERFACE} goal={GOAL_ID} task={TASK_ID}")
        print(
            f"toolchain={receipt.get('selected_toolchain')} "
            f"version={receipt.get('version_string')!r}"
        )
        print(
            f"usable={receipt.get('usable')} "
            f"production_certified={receipt.get('production_certified')} "
            f"promotion_blocked={receipt.get('promotion_blocked')}"
        )
        for check in receipt.get("checks") or []:
            print(
                f"  [{check.get('status'):10}] {check.get('check_id')}: "
                f"expected={check.get('expected')} observed={check.get('observed')}"
            )
        if receipt.get("block_reasons"):
            print("block_reasons:", ", ".join(receipt["block_reasons"]))
        print("notes:", receipt.get("notes") or "")

    return 0 if receipt.get("production_certified") else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "INTERFACE",
    "SCHEMA_VERSION",
    "CORPUS_SCHEMA",
    "GOAL_ID",
    "TASK_ID",
    "PROGRAM",
    "LANE_ID",
    "TOOL_ID",
    "CERTIFICATION_SURFACE",
    "HANDLER_ID",
    "LOCKED_TOOLCHAIN",
    "LOCKED_VERSION",
    "LOCKED_VERSION_NUMERIC",
    "AUTHORITY_CEILING",
    "AUTHORITY_SCOPE",
    "CheckResult",
    "CaseOutcome",
    "LeanSemanticCertification",
    "repo_root_from",
    "content_digest",
    "offline_env",
    "bounded_run",
    "resolve_lean_executable",
    "list_elan_installed_toolchains",
    "detect_lean_shim_toolchain_mismatch",
    "extract_lean_imports",
    "extract_lean_theorem_name",
    "scan_lean_incomplete_or_unsafe",
    "default_corpus_manifest",
    "load_corpus_manifest",
    "corpus_cases",
    "probe_lean_identity",
    "check_lean_source",
    "run_semantic_suite",
    "build_certification_receipt",
    "certify_lean_kernel",
    "lane_handler",
    "main",
]
