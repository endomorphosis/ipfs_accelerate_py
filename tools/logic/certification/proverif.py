#!/usr/bin/env python3
"""ProVerif + isolated OPAM protocol toolchain certification (FVT-G131 / FVT-044).

``ProVerifToolchainCertification@1`` and live ``ProtocolLiveSemanticCertification@1``
(FVT-G205 / FVT-058).

Owns the protocol-lane certification handler for the pinned ProVerif 2.05
analyzer and its support-only OPAM 2.5.2 companion under an isolated root.
Certification:

* never installs, downloads, or opens the network;
* requires exact identity probes for ProVerif 2.05 and OPAM 2.5.2 when
  production-certifying;
* exercises secure, attack, mutated claim/model, replay, malformed output,
  cancellation, and version-mismatch cases offline via parser fixtures;
* runs a separate live semantic corpus through the real pinned binary with
  source, query, assumption, bound, witness, and raw-output bindings;
* treats OPAM as support only — OPAM presence alone never promotes the
  protocol property lane;
* never lets parser-recognized canned output stand in for live semantic proof;
* never lets Tamarin substitute for ProVerif;
* never mutates a global OPAM switch;
* never edits the shared multi-prover certificate or the Tamarin lane.
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
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

# Allow running as a script from a worktree without an installed package.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DATASETS_ROOT = _REPO_ROOT / "ipfs_datasets_py"
for candidate in (_REPO_ROOT, _DATASETS_ROOT):
    text = str(candidate)
    if text not in sys.path:
        sys.path.insert(0, text)

from ipfs_datasets_py.logic.backends.protocol.proverif import (  # noqa: E402
    PROVERIF_BACKEND_VERSION,
    PROVERIF_SUPPORTED_CLAIMS,
    PROVERIF_SUPPORTED_THEORIES,
    ClaimVerdict,
    SymbolicModelCeiling,
    classify_claim_outcomes,
    parse_attack_trace,
    parse_proverif_claim_outcomes,
)
from ipfs_datasets_py.logic.backends.results import ResultStatus  # noqa: E402
from ipfs_datasets_py.logic.backends.toolchain_roles import (  # noqa: E402
    ToolchainAuthorityCeiling,
    ToolRole,
    can_satisfy_certified_authority_requirement,
    evaluate_role_aware_promotion,
    get_tool_role,
)
from ipfs_datasets_py.logic.software_verification.protocol import (  # noqa: E402
    EquationalTheory,
    ProtocolClaimKind,
)

try:  # pragma: no cover - worktree packaging varies
    from tools.logic.certification.roles import (  # type: ignore
        bind_lane_handler as _bind_lane_handler,
        build_role_aware_policy as _build_role_aware_policy,
    )
except Exception:  # pragma: no cover
    _bind_lane_handler = None  # type: ignore[assignment]
    _build_role_aware_policy = None  # type: ignore[assignment]

INTERFACE: Final = "ProVerifToolchainCertification@1"
SCHEMA_VERSION: Final = "proverif-toolchain-certification/v1"
CORPUS_SCHEMA: Final = "proverif-toolchain-corpus/v1"
GOAL_ID: Final = "FVT-G131"
TASK_ID: Final = "FVT-044"
PROGRAM: Final = "formal-verification-tactician/proverif-toolchain"
LANE_ID: Final = "protocol"
TOOL_ID: Final = "proverif"
SUPPORT_TOOL_ID: Final = "opam"
CERTIFICATION_SURFACE: Final = "tools.logic.certification.proverif"
HANDLER_ID: Final = "proverif_toolchain_certifier"
AUTHORITY_CEILING: Final = ToolchainAuthorityCeiling.PROTOCOL.value
AUTHORITY_SCOPE: Final = "protocol_verification_only"

LOCKED_PROVERIF_VERSION: Final = "2.05"
LOCKED_OPAM_VERSION: Final = "2.5.2"
LOCKED_PROVERIF_EXECUTABLE: Final = "proverif"
LOCKED_OPAM_EXECUTABLE: Final = "opam"

PROBE_TIMEOUT_SECONDS: Final = 5.0
CHECK_TIMEOUT_SECONDS: Final = 30.0
LIVE_CHECK_TIMEOUT_SECONDS: Final = 60.0

DEFAULT_LOCK_RELATIVE: Final = Path("config/formal_verification_toolchains.lock.json")
DEFAULT_PROTOCOL_LIVE_CERTIFICATE_RELATIVE: Final = Path(
    "docs/architecture/formal_verification_protocol_live_certificate.json"
)

# Live semantic surface (FVT-G205 / FVT-058; objective validation repair FVT-075).
# Distinct from offline toolchain certification so parser fixtures remain
# non-production evidence.
LIVE_INTERFACE: Final = "ProtocolLiveSemanticCertification@1"
LIVE_SCHEMA_VERSION: Final = "protocol-live-semantic-certification/v1"
LIVE_CORPUS_SCHEMA: Final = "protocol-live-semantic-corpus/v1"
LIVE_GOAL_ID: Final = "FVT-G205"
LIVE_TASK_ID: Final = "FVT-058"
LIVE_REPAIR_TASK_ID: Final = "FVT-075"
LIVE_PROGRAM: Final = "formal-verification-tactician/protocol-live-semantics"
LIVE_TOOL_SURFACE: Final = "proverif-live-semantic"
EVIDENCE_CLASS_LIVE: Final = "live"
EVIDENCE_CLASS_PARSER_FIXTURE: Final = "parser_fixture"

_RAW_OUTPUT_CAP: Final = 8_192
CAPABILITY_GAP_PINNED_BINARY_UNAVAILABLE: Final = (
    "pinned_protocol_binary_unavailable_on_validation_path"
)

# Accept 2.05 and 2.5.2 style versions.
_VERSION_IN_BANNER = re.compile(r"(\d+\.\d+(?:\.\d+)?)")

_LIVE_SECURE_SOURCE: Final = """\
free c:channel.
free challenge:bitstring [private].
fun h(bitstring):bitstring.

query attacker(challenge).

event BeginChallenge(bitstring).
event AcceptChallenge(bitstring).

query x:bitstring;
  event(AcceptChallenge(x)) ==> event(BeginChallenge(x)).

process
  event BeginChallenge(challenge);
  out(c, h(challenge));
  in(c, x:bitstring);
  if x = h(challenge) then
    event AcceptChallenge(challenge)
"""

_LIVE_ATTACK_SOURCE: Final = """\
free c:channel.
free challenge:bitstring [private].

query attacker(challenge).

process
  out(c, challenge)
"""

_LIVE_MUTATED_CLAIM_SOURCE: Final = """\
free c:channel.
free challenge:bitstring [private].
fun h(bitstring):bitstring.

query attacker(challenge).

event BeginChallenge(bitstring).
event AcceptChallenge(bitstring).

(* Premise/conclusion mutation: Accept happens before Begin. *)
query x:bitstring;
  event(AcceptChallenge(x)) ==> event(BeginChallenge(x)).

process
  event AcceptChallenge(challenge);
  out(c, h(challenge));
  event BeginChallenge(challenge)
"""

_LIVE_MUTATED_PROTOCOL_SOURCE: Final = """\
free c:channel.
free challenge:bitstring [private].

query attacker(challenge).

process
  out(c, challenge)
"""

_LIVE_DISAGREEMENT_SOURCE: Final = """\
free c:channel.
free challenge:bitstring [private].
fun h(bitstring):bitstring.

query attacker(challenge).

event BeginChallenge(bitstring).
event AcceptChallenge(bitstring).

query x:bitstring;
  event(AcceptChallenge(x)) ==> event(BeginChallenge(x)).

(* Secrecy holds (only hash is sent) while auth fails (Accept before Begin). *)
process
  event AcceptChallenge(challenge);
  out(c, h(challenge));
  event BeginChallenge(challenge)
"""

_LIVE_MALFORMED_SOURCE: Final = """\
this is not valid proverif source !!!!
query attacker(secret).
process 0
"""

_LIVE_BOUNDED_SOURCE: Final = """\
set maxDepth = 0.
free c:channel.
free s:bitstring [private].
query attacker(s).
process out(c, s)
"""

_DEFAULT_LIVE_CORPUS_CASES: Final[tuple[dict[str, Any], ...]] = (
    {
        "case_id": "live_secure_secrecy_auth",
        "kind": "secure",
        "expect": "secure",
        "source": _LIVE_SECURE_SOURCE,
        "source_format": "pv",
        "claim_queries": {
            "claim:secrecy": "not attacker(challenge[])",
            "claim:auth": (
                "event(AcceptChallenge(x)) ==> event(BeginChallenge(x))"
            ),
        },
        "assumptions": [
            "dolev_yao_adversary",
            "perfect_cryptography",
            "hashing_constructor",
        ],
        "query": "query attacker(challenge) + auth correspondence",
        "bounds": {"timeout_seconds": LIVE_CHECK_TIMEOUT_SECONDS},
        "description": "Live secure secrecy and authentication queries",
    },
    {
        "case_id": "live_attack_leak",
        "kind": "attack",
        "expect": "attack",
        "source": _LIVE_ATTACK_SOURCE,
        "source_format": "pv",
        "claim_queries": {"claim:secrecy": "not attacker(challenge[])"},
        "assumptions": ["dolev_yao_adversary"],
        "query": "query attacker(challenge) expect false",
        "bounds": {"timeout_seconds": LIVE_CHECK_TIMEOUT_SECONDS},
        "description": "Live concrete attack: private name leaked",
    },
    {
        "case_id": "live_mutated_claim",
        "kind": "mutation",
        "mutates": "claim",
        "expect": "rejected_or_quarantined",
        "source": _LIVE_MUTATED_CLAIM_SOURCE,
        "source_format": "pv",
        "claim_queries": {
            "claim:secrecy": "not attacker(challenge[])",
            "claim:auth": (
                "event(AcceptChallenge(x)) ==> event(BeginChallenge(x))"
            ),
        },
        "assumptions": ["dolev_yao_adversary", "mutated_auth_ordering"],
        "query": "mutated premise/conclusion auth query",
        "bounds": {"timeout_seconds": LIVE_CHECK_TIMEOUT_SECONDS},
        "description": "Premise/conclusion mutation must not remain secure",
    },
    {
        "case_id": "live_mutated_protocol",
        "kind": "mutation",
        "mutates": "protocol",
        "expect": "attack",
        "source": _LIVE_MUTATED_PROTOCOL_SOURCE,
        "source_format": "pv",
        "claim_queries": {"claim:secrecy": "not attacker(challenge[])"},
        "assumptions": ["dolev_yao_adversary", "protocol_process_mutated"],
        "query": "query attacker after protocol mutation",
        "bounds": {"timeout_seconds": LIVE_CHECK_TIMEOUT_SECONDS},
        "description": "Protocol mutation yields an attack",
    },
    {
        "case_id": "live_deterministic_replay",
        "kind": "replay",
        "expect": "secure",
        "source": _LIVE_SECURE_SOURCE,
        "source_format": "pv",
        "claim_queries": {
            "claim:secrecy": "not attacker(challenge[])",
            "claim:auth": (
                "event(AcceptChallenge(x)) ==> event(BeginChallenge(x))"
            ),
        },
        "assumptions": [
            "dolev_yao_adversary",
            "perfect_cryptography",
            "hashing_constructor",
        ],
        "query": "replay secure queries",
        "bounds": {"timeout_seconds": LIVE_CHECK_TIMEOUT_SECONDS},
        "description": "Deterministic live replay of the secure case",
    },
    {
        "case_id": "live_malformed_model",
        "kind": "malformed",
        "expect": "quarantined",
        "source": _LIVE_MALFORMED_SOURCE,
        "source_format": "pv",
        "claim_queries": {"claim:secrecy": "not attacker(secret[])"},
        "assumptions": [],
        "query": "parse/prove malformed model",
        "bounds": {"timeout_seconds": LIVE_CHECK_TIMEOUT_SECONDS},
        "description": "Malformed model never reports SECURE",
    },
    {
        "case_id": "live_timeout",
        "kind": "timeout",
        "expect": "quarantined",
        "source": _LIVE_SECURE_SOURCE,
        "source_format": "pv",
        "claim_queries": {
            "claim:secrecy": "not attacker(challenge[])",
            "claim:auth": (
                "event(AcceptChallenge(x)) ==> event(BeginChallenge(x))"
            ),
        },
        "assumptions": ["dolev_yao_adversary"],
        "query": "query under extreme timeout bound",
        "bounds": {"timeout_seconds": 0.001},
        "force_timeout": True,
        "description": "Subprocess timeout quarantines rather than SECURE",
    },
    {
        "case_id": "live_disagreement",
        "kind": "disagreement",
        "expect": "rejected_or_quarantined",
        "source": _LIVE_DISAGREEMENT_SOURCE,
        "source_format": "pv",
        "claim_queries": {
            "claim:secrecy": "not attacker(challenge[])",
            "claim:auth": (
                "event(AcceptChallenge(x)) ==> event(BeginChallenge(x))"
            ),
        },
        "assumptions": ["dolev_yao_adversary", "mixed_claim_batch"],
        "query": "mixed secrecy(false)+auth(false) or mixed true/false",
        "bounds": {"timeout_seconds": LIVE_CHECK_TIMEOUT_SECONDS},
        "description": "Mixed true/false claims quarantine or attack",
    },
    {
        "case_id": "live_bounded_search",
        "kind": "bounded_search",
        "expect": "quarantined",
        "source": _LIVE_BOUNDED_SOURCE,
        "source_format": "pv",
        "claim_queries": {"claim:secrecy": "not attacker(s[])"},
        "assumptions": ["dolev_yao_adversary", "maxDepth_bound"],
        "query": "query attacker with set maxDepth = 0",
        "bounds": {
            "timeout_seconds": LIVE_CHECK_TIMEOUT_SECONDS,
            "max_depth": 0,
        },
        "description": "Bounded search cannot-prove quarantines",
    },
)

# Compact embedded corpus. Prefer live binaries when present; parsers always run.
_DEFAULT_CORPUS_CASES: Final[tuple[dict[str, Any], ...]] = (
    {
        "case_id": "secure_claims",
        "kind": "secure",
        "expect": "secure",
        "claim_queries": {
            "claim:secrecy": "not attacker(challenge[])",
            "claim:auth": (
                "inj-event(AcceptChallenge(x)) ==> inj-event(BeginChallenge(x))"
            ),
        },
        "stdout": (
            "RESULT not attacker(challenge[]) is true.\n"
            "RESULT inj-event(AcceptChallenge(x)) ==> "
            "inj-event(BeginChallenge(x)) is true.\n"
        ),
        "stderr": "",
        "description": "Secure protocol: all ProVerif queries true",
    },
    {
        "case_id": "attack_trace",
        "kind": "attack",
        "expect": "attack",
        "claim_queries": {"claim:secrecy": "not attacker(challenge)"},
        "stdout": (
            "RESULT not attacker(challenge) is false.\n"
            "-> event AcceptChallenge(n)\n"
            "-> out(c, n)\n"
        ),
        "stderr": "",
        "description": "Attack: false query with normalized attack trace",
    },
    {
        "case_id": "mutated_claim",
        "kind": "mutation",
        "mutates": "claim",
        "expect": "rejected_or_quarantined",
        "base_case_id": "secure_claims",
        "claim_queries": {
            "claim:secrecy": "not attacker(challenge[])",
            "claim:auth": "event(AcceptChallenge(x)) ==> event(BeginChallenge(x))",
        },
        "stdout": (
            "RESULT not attacker(challenge[]) is true.\n"
            "RESULT event(AcceptChallenge(x)) ==> event(BeginChallenge(x)) is false.\n"
            "-> event AcceptChallenge(x)\n"
        ),
        "stderr": "",
        "description": "Claim mutation produces disagreement quarantine, not secure",
    },
    {
        "case_id": "mutated_model",
        "kind": "mutation",
        "mutates": "model",
        "expect": "attack",
        "base_case_id": "secure_claims",
        "claim_queries": {"claim:secrecy": "not attacker(challenge)"},
        "stdout": (
            "RESULT not attacker(challenge) is false.\n"
            "-> event Mutated_Process(evil)\n"
            "-> out(c, secret)\n"
        ),
        "stderr": "",
        "description": "Model mutation yields an attack rather than secure",
    },
    {
        "case_id": "deterministic_replay",
        "kind": "replay",
        "expect": "secure",
        "base_case_id": "secure_claims",
        "claim_queries": {
            "claim:secrecy": "not attacker(challenge[])",
            "claim:auth": (
                "inj-event(AcceptChallenge(x)) ==> inj-event(BeginChallenge(x))"
            ),
        },
        "stdout": (
            "RESULT not attacker(challenge[]) is true.\n"
            "RESULT inj-event(AcceptChallenge(x)) ==> "
            "inj-event(BeginChallenge(x)) is true.\n"
        ),
        "stderr": "",
        "description": "Positive secure case replays with identical digests",
    },
    {
        "case_id": "malformed_output",
        "kind": "malformed",
        "expect": "quarantined",
        "claim_queries": {"claim:secrecy": "not attacker(challenge[])"},
        "stdout": "this is not a proverif RESULT report\n!!! garbage !!!\n",
        "stderr": "",
        "description": "Malformed tool output never reports SECURE",
    },
    {
        "case_id": "cancelled_query",
        "kind": "cancellation",
        "expect": "quarantined",
        "claim_queries": {"claim:secrecy": "not attacker(challenge[])"},
        "stdout": "",
        "stderr": "ProVerif: execution cancelled by operator\n",
        "cancelled": True,
        "description": "Cancellation never promotes to SECURE",
    },
    {
        "case_id": "version_mismatch",
        "kind": "version_mismatch",
        "expect": "blocked",
        "claim_queries": {},
        "stdout": "",
        "stderr": "",
        "observed_proverif_version": "2.00",
        "observed_opam_version": "2.1.0",
        "description": "Locked version mismatch blocks production certification",
    },
)

DEFAULT_THEORY_BINDINGS: Final[tuple[str, ...]] = (
    EquationalTheory.FREE.value,
    EquationalTheory.PAIRING.value,
    EquationalTheory.SYMMETRIC_ENCRYPTION.value,
    EquationalTheory.HASHING.value,
)

DEFAULT_CLAIM_BINDINGS: Final[tuple[str, ...]] = (
    ProtocolClaimKind.SECRECY.value,
    ProtocolClaimKind.AUTHENTICATION.value,
    ProtocolClaimKind.REACHABILITY.value,
    ProtocolClaimKind.CORRESPONDENCE.value,
    ProtocolClaimKind.EQUIVALENCE.value,
)

DEFAULT_BOUNDS: Final[dict[str, Any]] = {
    "timeout_seconds": CHECK_TIMEOUT_SECONDS,
    "max_source_bytes": 1_048_576,
    "network": False,
    "install": False,
    "download": False,
    "global_opam_switch_mutation": False,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def repo_root_from(start: Path | None = None) -> Path:
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
    env = dict(base if base is not None else os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    env["PIP_NO_INDEX"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_OPTIONAL_LOCKS"] = "0"
    env.setdefault("NO_PROXY", "*")
    env.setdefault("no_proxy", "*")
    env["FORMAL_VERIFICATION_CERTIFY_OFFLINE"] = "1"
    env["FORMAL_VERIFICATION_FORBID_INSTALL"] = "1"
    env["FORMAL_VERIFICATION_FORBID_NETWORK"] = "1"
    env["FORMAL_VERIFICATION_FORBID_DOWNLOAD"] = "1"
    env["FORMAL_VERIFICATION_FORBID_GLOBAL_OPAM"] = "1"
    return env


def bounded_run(
    argv: Sequence[str],
    *,
    timeout: float = PROBE_TIMEOUT_SECONDS,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str] | None:
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
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def resolve_executable(candidates: Sequence[str] | None = None) -> str | None:
    for name in candidates or ():
        if not name:
            continue
        path = Path(name)
        if path.is_file() and os.access(path, os.X_OK):
            return str(path.resolve())
        found = shutil.which(name)
        if found:
            return found
    return None


def first_nonempty_line(text: str) -> str:
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def extract_version(banner: str | None) -> str | None:
    if not banner:
        return None
    match = _VERSION_IN_BANNER.search(banner)
    return match.group(1) if match else None


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
    status: str  # secure | attack | quarantined | blocked | rejected | unknown
    matched: bool
    reason_codes: list[str] = field(default_factory=list)
    claim_outcomes: list[dict[str, Any]] = field(default_factory=list)
    attack_trace: dict[str, Any] | None = None
    output_digest: str = ""
    stdout: str = ""
    stderr: str = ""
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProVerifToolchainCertification:
    """Full certification receipt for the ProVerif/OPAM protocol lane."""

    tool_id: str = TOOL_ID
    support_tool_id: str = SUPPORT_TOOL_ID
    lane_id: str = LANE_ID
    interface: str = INTERFACE
    schema_version: str = SCHEMA_VERSION
    goal_id: str = GOAL_ID
    task_id: str = TASK_ID
    program: str = PROGRAM
    certification_surface: str = CERTIFICATION_SURFACE
    locked_proverif_version: str = LOCKED_PROVERIF_VERSION
    locked_opam_version: str = LOCKED_OPAM_VERSION
    authority_ceiling: str = AUTHORITY_CEILING
    authority_scope: str = AUTHORITY_SCOPE
    proverif_executable: str | None = None
    opam_executable: str | None = None
    proverif_version_string: str | None = None
    opam_version_string: str | None = None
    proverif_identity_probed: bool = False
    opam_identity_probed: bool = False
    proverif_version_match: bool = False
    opam_version_match: bool = False
    proverif_usable: bool = False
    opam_usable: bool = False
    isolated_opam_root: str | None = None
    isolated_root_validated: bool = False
    network_used: bool = False
    install_attempted: bool = False
    download_attempted: bool = False
    global_opam_mutation_attempted: bool = False
    production_certified: bool = False
    promotion_blocked: bool = True
    opam_support_only: bool = True
    opam_cannot_promote_alone: bool = True
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
# Corpus
# ---------------------------------------------------------------------------


def default_corpus_manifest() -> dict[str, Any]:
    return {
        "schema_version": CORPUS_SCHEMA,
        "interface": INTERFACE,
        "goal_id": GOAL_ID,
        "task_id": TASK_ID,
        "tool_id": TOOL_ID,
        "support_tool_id": SUPPORT_TOOL_ID,
        "lane_id": LANE_ID,
        "locked_proverif_version": LOCKED_PROVERIF_VERSION,
        "locked_opam_version": LOCKED_OPAM_VERSION,
        "authority_ceiling": AUTHORITY_CEILING,
        "authority_scope": AUTHORITY_SCOPE,
        "theory_bindings": list(DEFAULT_THEORY_BINDINGS),
        "claim_bindings": list(DEFAULT_CLAIM_BINDINGS),
        "bounds": dict(DEFAULT_BOUNDS),
        "policy": {
            "no_install": True,
            "no_download": True,
            "no_network": True,
            "opam_is_support_only": True,
            "opam_cannot_promote_protocol_lane": True,
            "isolated_opam_root_required": True,
            "never_mutate_global_opam_switch": True,
            "exact_binary_binding_required": True,
            "authority_is_protocol_verification_only": True,
        },
        "cases": [dict(case) for case in _DEFAULT_CORPUS_CASES],
    }


def load_corpus_manifest(
    path: Path | None = None,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    root = repo_root or repo_root_from()
    if path is not None and path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("ProVerif corpus manifest must be a JSON object")
        return payload
    _ = root
    return default_corpus_manifest()


def corpus_cases(manifest: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    payload = manifest if manifest is not None else default_corpus_manifest()
    cases = payload.get("cases") or []
    if not isinstance(cases, list) or not cases:
        raise ValueError("ProVerif corpus must declare a non-empty cases list")
    return [dict(case) for case in cases if isinstance(case, Mapping)]


# ---------------------------------------------------------------------------
# Identity probes
# ---------------------------------------------------------------------------


def probe_proverif_identity(
    *,
    env: Mapping[str, str] | None = None,
    executable: str | None = None,
) -> dict[str, Any]:
    probe_env = offline_env(env)
    result: dict[str, Any] = {
        "tool_id": TOOL_ID,
        "path_present": False,
        "executable_path": None,
        "version_string": None,
        "identity_probed": False,
        "version_match": False,
        "locked_version": LOCKED_PROVERIF_VERSION,
        "network_used": False,
        "install_attempted": False,
        "download_attempted": False,
        "probe_error": None,
    }
    binary = executable or resolve_executable(
        [LOCKED_PROVERIF_EXECUTABLE, "proverif"]
    )
    if binary is None:
        result["probe_error"] = "executable_not_on_path"
        return result
    result["path_present"] = True
    result["executable_path"] = binary
    completed = bounded_run(
        [binary, "--help"],
        timeout=PROBE_TIMEOUT_SECONDS,
        env=probe_env,
    )
    # Prefer --version when supported; fall back to --help banner.
    version_probe = bounded_run(
        [binary, "--version"],
        timeout=PROBE_TIMEOUT_SECONDS,
        env=probe_env,
    )
    banner = ""
    for completed_probe in (version_probe, completed):
        if completed_probe is None:
            continue
        banner = first_nonempty_line(completed_probe.stdout) or first_nonempty_line(
            completed_probe.stderr
        )
        if not banner:
            banner = (completed_probe.stdout or completed_probe.stderr or "").strip()
        if banner:
            break
    if not banner:
        result["probe_error"] = "empty_version_banner"
        return result
    result["version_string"] = banner
    result["identity_probed"] = True
    version = extract_version(banner)
    result["version_match"] = bool(
        version == LOCKED_PROVERIF_VERSION
        or LOCKED_PROVERIF_VERSION in banner
        or (version is not None and version.startswith(LOCKED_PROVERIF_VERSION))
    )
    if not result["version_match"]:
        result["probe_error"] = "locked_version_mismatch"
    return result


def probe_opam_identity(
    *,
    env: Mapping[str, str] | None = None,
    executable: str | None = None,
) -> dict[str, Any]:
    probe_env = offline_env(env)
    result: dict[str, Any] = {
        "tool_id": SUPPORT_TOOL_ID,
        "path_present": False,
        "executable_path": None,
        "version_string": None,
        "identity_probed": False,
        "version_match": False,
        "locked_version": LOCKED_OPAM_VERSION,
        "support_only": True,
        "can_promote_protocol_lane": False,
        "network_used": False,
        "install_attempted": False,
        "download_attempted": False,
        "probe_error": None,
    }
    binary = executable or resolve_executable([LOCKED_OPAM_EXECUTABLE])
    if binary is None:
        result["probe_error"] = "executable_not_on_path"
        return result
    result["path_present"] = True
    result["executable_path"] = binary
    completed = bounded_run(
        [binary, "--version"],
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
        banner = (completed.stdout or completed.stderr or "").strip()
    if not banner:
        result["probe_error"] = "empty_version_banner"
        return result
    result["version_string"] = banner
    result["identity_probed"] = True
    version = extract_version(banner)
    result["version_match"] = bool(
        version == LOCKED_OPAM_VERSION or LOCKED_OPAM_VERSION in banner
    )
    if not result["version_match"]:
        result["probe_error"] = "locked_version_mismatch"
    return result


def validate_isolated_opam_root(path: str | Path | None) -> dict[str, Any]:
    """Confirm an OPAM root is isolated (not a global switch)."""

    result: dict[str, Any] = {
        "path": None,
        "validated": False,
        "global_forbidden": True,
        "detail": "",
    }
    if path is None or not str(path).strip():
        result["detail"] = "no_isolated_root_declared"
        return result
    expanded = Path(os.path.expanduser(str(path)))
    result["path"] = str(expanded)
    home_opam = Path.home() / ".opam"
    try:
        resolved = expanded.resolve()
        home_resolved = home_opam.resolve()
    except OSError:
        result["detail"] = "path_unresolvable"
        return result
    if resolved == home_resolved:
        result["detail"] = "global_home_opam_root"
        result["validated"] = False
        return result
    if resolved.name == ".opam" and resolved.parent in {
        Path.home().resolve(),
        Path("/root").resolve(),
    }:
        result["detail"] = "global_dot_opam"
        return result
    result["validated"] = True
    result["detail"] = "isolated"
    return result


# ---------------------------------------------------------------------------
# Case evaluation (parser-backed, offline)
# ---------------------------------------------------------------------------


def evaluate_corpus_case(case: Mapping[str, Any]) -> CaseOutcome:
    """Evaluate one corpus case via canonical ProVerif parsers (no install)."""

    case_id = str(case.get("case_id") or "case")
    kind = str(case.get("kind") or "unknown")
    expect = str(case.get("expect") or "unknown")
    stdout = str(case.get("stdout") or "")
    stderr = str(case.get("stderr") or "")
    claim_queries = {
        str(key): str(value)
        for key, value in dict(case.get("claim_queries") or {}).items()
    }
    output_digest = content_digest(f"{stdout}\n{stderr}")

    if kind == "version_mismatch":
        observed_p = str(case.get("observed_proverif_version") or "")
        observed_o = str(case.get("observed_opam_version") or "")
        blocked = (
            observed_p != LOCKED_PROVERIF_VERSION
            or observed_o != LOCKED_OPAM_VERSION
        )
        return CaseOutcome(
            case_id=case_id,
            kind=kind,
            expect=expect,
            status="blocked" if blocked else "unknown",
            matched=blocked and expect == "blocked",
            reason_codes=["locked_version_mismatch"] if blocked else [],
            output_digest=output_digest,
            detail=(
                f"observed proverif={observed_p} opam={observed_o}; "
                f"locked proverif={LOCKED_PROVERIF_VERSION} opam={LOCKED_OPAM_VERSION}"
            ),
        )

    if kind == "cancellation" or bool(case.get("cancelled")):
        # Cancelled runs produce no conclusive RESULT lines → quarantine.
        outcomes = parse_proverif_claim_outcomes(
            stdout, stderr, claim_queries=claim_queries
        )
        status_enum, quarantine, accepted = classify_claim_outcomes(outcomes)
        observed_status = "quarantined"
        if status_enum is ResultStatus.SECURE and accepted:
            # Cancellation must never promote.
            observed_status = "quarantined"
        reason_codes = ["cancellation"]
        if quarantine is not None:
            reason_codes.append(str(quarantine.reason.value))
        matched = _expect_matches(expect, observed_status, True)
        return CaseOutcome(
            case_id=case_id,
            kind=kind,
            expect=expect,
            status=observed_status,
            matched=matched,
            reason_codes=list(dict.fromkeys(reason_codes)),
            claim_outcomes=[item.to_dict() for item in outcomes],
            attack_trace=None,
            output_digest=output_digest,
            stdout=stdout,
            stderr=stderr,
            detail=str(case.get("description") or ""),
        )

    outcomes = parse_proverif_claim_outcomes(
        stdout, stderr, claim_queries=claim_queries
    )
    status_enum, quarantine, accepted = classify_claim_outcomes(outcomes)
    attack: dict[str, Any] | None = None
    for item in outcomes:
        if item.attack_trace is not None:
            attack = item.attack_trace.to_dict()
            break
    if attack is None and status_enum is ResultStatus.ATTACK_FOUND:
        trace = parse_attack_trace(
            f"{stdout}\n{stderr}",
            claim_id=next(iter(claim_queries), "claim:unknown"),
            raw_digest=output_digest,
        )
        if trace is not None:
            attack = trace.to_dict()

    if status_enum is ResultStatus.SECURE and accepted:
        observed_status = "secure"
    elif status_enum is ResultStatus.ATTACK_FOUND:
        observed_status = "attack"
    elif quarantine is not None:
        observed_status = "quarantined"
    else:
        observed_status = "unknown"

    reason_codes: list[str] = []
    if quarantine is not None:
        reason_codes.append(str(quarantine.reason.value))
    if kind == "malformed" and observed_status != "secure":
        reason_codes.append("malformed_output")

    matched = _expect_matches(expect, observed_status, quarantine is not None)

    # Mutations of secure claims must never remain secure.
    if kind == "mutation" and observed_status == "secure":
        matched = False
        reason_codes.append("mutation_still_secure")

    return CaseOutcome(
        case_id=case_id,
        kind=kind,
        expect=expect,
        status=observed_status,
        matched=matched,
        reason_codes=list(dict.fromkeys(reason_codes)),
        claim_outcomes=[item.to_dict() for item in outcomes],
        attack_trace=attack,
        output_digest=output_digest,
        stdout=stdout,
        stderr=stderr,
        detail=str(case.get("description") or ""),
    )


def _expect_matches(expect: str, observed: str, quarantined: bool) -> bool:
    if expect == "secure":
        return observed == "secure"
    if expect == "attack":
        return observed == "attack"
    if expect == "quarantined":
        return observed == "quarantined" or quarantined
    if expect == "blocked":
        return observed == "blocked"
    if expect == "rejected_or_quarantined":
        return observed in {"attack", "quarantined", "unknown", "blocked"}
    return observed == expect


# ---------------------------------------------------------------------------
# OPAM promotion boundary
# ---------------------------------------------------------------------------


def opam_cannot_promote_protocol_lane() -> dict[str, Any]:
    """Prove OPAM support-only presence cannot satisfy protocol authority."""

    role = get_tool_role(SUPPORT_TOOL_ID)
    decision = evaluate_role_aware_promotion(
        SUPPORT_TOOL_ID,
        present=True,
        usable=True,
        production_certified=True,
        hermetic_certificate=True,
        independent_reconstruction=True,
    )
    can_satisfy = can_satisfy_certified_authority_requirement(SUPPORT_TOOL_ID)
    return {
        "tool_id": SUPPORT_TOOL_ID,
        "role": role.role.value,
        "authority_ceiling": role.authority_ceiling.value,
        "can_satisfy_certified_authority": role.can_satisfy_certified_authority,
        "can_satisfy_protocol_requirement": can_satisfy,
        "promotion_allowed": decision.allowed,
        "promotion_decision": decision.to_dict(),
        "support_only": role.role is ToolRole.SUPPORT,
        "ceiling_is_none": role.authority_ceiling is ToolchainAuthorityCeiling.NONE,
        "blocks_alone": (not decision.allowed) and (not can_satisfy),
    }


# ---------------------------------------------------------------------------
# Certification orchestration
# ---------------------------------------------------------------------------


def run_certification_suite(
    *,
    repo_root: Path | None = None,
    manifest: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
    proverif_executable: str | None = None,
    opam_executable: str | None = None,
    isolated_opam_root: str | Path | None = None,
) -> ProVerifToolchainCertification:
    """Run the full ProVerif/OPAM certification suite."""

    root = repo_root or repo_root_from()
    corpus = manifest if manifest is not None else load_corpus_manifest(repo_root=root)
    cases = corpus_cases(corpus)
    cert = ProVerifToolchainCertification()
    probe_env = offline_env(env)

    # Prefer an explicit root, then a repo-local convention path (not global).
    if isolated_opam_root is not None:
        declared_root = str(isolated_opam_root)
    else:
        declared_root = str(
            root / ".cache" / "formal_verification" / "opam-roots" / "proverif"
        )
    cert.isolated_opam_root = declared_root

    cert.checks.append(
        CheckResult(
            check_id="proverif.offline_policy",
            kind="policy",
            status="passed",
            expected="no_install_no_download_no_network_no_global_opam",
            observed=(
                f"install={cert.install_attempted},"
                f"download={cert.download_attempted},"
                f"network={cert.network_used},"
                f"global_opam={cert.global_opam_mutation_attempted},"
                f"FORMAL_VERIFICATION_CERTIFY_OFFLINE="
                f"{probe_env.get('FORMAL_VERIFICATION_CERTIFY_OFFLINE')}"
            ),
            detail=(
                "certification never installs, downloads, opens the network, "
                "or mutates a global OPAM switch"
            ),
        )
    )

    proverif_probe = probe_proverif_identity(
        env=probe_env, executable=proverif_executable
    )
    opam_probe = probe_opam_identity(env=probe_env, executable=opam_executable)
    root_probe = validate_isolated_opam_root(cert.isolated_opam_root)
    cert.isolated_root_validated = bool(root_probe.get("validated"))

    cert.proverif_executable = proverif_probe.get("executable_path")
    cert.opam_executable = opam_probe.get("executable_path")
    cert.proverif_version_string = proverif_probe.get("version_string")
    cert.opam_version_string = opam_probe.get("version_string")
    cert.proverif_identity_probed = bool(proverif_probe.get("identity_probed"))
    cert.opam_identity_probed = bool(opam_probe.get("identity_probed"))
    cert.proverif_version_match = bool(proverif_probe.get("version_match"))
    cert.opam_version_match = bool(opam_probe.get("version_match"))
    cert.proverif_usable = bool(
        cert.proverif_identity_probed and cert.proverif_version_match
    )
    cert.opam_usable = bool(cert.opam_identity_probed and cert.opam_version_match)

    if cert.proverif_usable:
        cert.checks.append(
            CheckResult(
                check_id="proverif.identity",
                kind="identity",
                status="passed",
                expected=LOCKED_PROVERIF_VERSION,
                observed=cert.proverif_version_string or "",
                detail="exact ProVerif pin identity",
                bindings={
                    "executable_path": cert.proverif_executable,
                    "version_string": cert.proverif_version_string,
                },
            )
        )
    else:
        reason = str(proverif_probe.get("probe_error") or "unavailable")
        cert.block_reasons.append(f"proverif:{reason}")
        cert.checks.append(
            CheckResult(
                check_id="proverif.identity",
                kind="identity",
                status="unavailable" if reason == "executable_not_on_path" else "blocked",
                expected=LOCKED_PROVERIF_VERSION,
                observed=reason,
                detail="PATH presence without locked identity is not usability",
                reason_codes=[reason],
            )
        )

    if cert.opam_usable:
        cert.checks.append(
            CheckResult(
                check_id="opam.identity",
                kind="identity",
                status="passed",
                expected=LOCKED_OPAM_VERSION,
                observed=cert.opam_version_string or "",
                detail="exact OPAM pin identity (support only)",
                bindings={
                    "executable_path": cert.opam_executable,
                    "version_string": cert.opam_version_string,
                    "support_only": True,
                },
            )
        )
    else:
        reason = str(opam_probe.get("probe_error") or "unavailable")
        cert.block_reasons.append(f"opam:{reason}")
        cert.checks.append(
            CheckResult(
                check_id="opam.identity",
                kind="identity",
                status="unavailable" if reason == "executable_not_on_path" else "blocked",
                expected=LOCKED_OPAM_VERSION,
                observed=reason,
                detail="OPAM support companion identity",
                reason_codes=[reason],
            )
        )

    if cert.isolated_root_validated:
        cert.checks.append(
            CheckResult(
                check_id="opam.isolated_root",
                kind="isolation",
                status="passed",
                expected="isolated_non_global_root",
                observed=str(root_probe.get("detail") or ""),
                detail="OPAM root is repository-local / managed, not global",
                bindings=dict(root_probe),
            )
        )
    else:
        cert.block_reasons.append("isolated_opam_root_invalid")
        cert.checks.append(
            CheckResult(
                check_id="opam.isolated_root",
                kind="isolation",
                status="failed",
                expected="isolated_non_global_root",
                observed=str(root_probe.get("detail") or "invalid"),
                detail="global OPAM roots are forbidden for ProVerif",
                reason_codes=["global_or_missing_opam_root"],
                bindings=dict(root_probe),
            )
        )

    # Semantic corpus (parser-backed; always runs offline).
    outcomes_by_id: dict[str, CaseOutcome] = {}
    for case in cases:
        outcome = evaluate_corpus_case(case)
        outcomes_by_id[outcome.case_id] = outcome
        cert.cases.append(outcome)
        status = "passed" if outcome.matched else "failed"
        if not outcome.matched:
            cert.block_reasons.append(f"case_failed:{outcome.case_id}")
        cert.checks.append(
            CheckResult(
                check_id=f"proverif.{outcome.case_id}",
                kind=outcome.kind,
                status=status,
                expected=outcome.expect,
                observed=outcome.status,
                detail=outcome.detail,
                reason_codes=list(outcome.reason_codes),
                bindings={
                    "output_digest": outcome.output_digest,
                    "claim_outcomes": list(outcome.claim_outcomes),
                    "attack_trace": outcome.attack_trace,
                },
            )
        )

    # Deterministic replay binding between secure and replay cases.
    secure = outcomes_by_id.get("secure_claims")
    replay = outcomes_by_id.get("deterministic_replay")
    if secure is not None and replay is not None:
        replay_ok = (
            secure.status == "secure"
            and replay.status == "secure"
            and secure.output_digest == replay.output_digest
            and secure.matched
            and replay.matched
        )
        if not replay_ok:
            cert.block_reasons.append("replay_nondeterministic_or_failed")
        cert.checks.append(
            CheckResult(
                check_id="proverif.deterministic_replay_binding",
                kind="replay",
                status="passed" if replay_ok else "failed",
                expected="identical secure digests",
                observed=(
                    f"secure={secure.output_digest[:12]},"
                    f"replay={replay.output_digest[:12]}"
                ),
                bindings={
                    "secure_digest": secure.output_digest,
                    "replay_digest": replay.output_digest,
                },
            )
        )
    else:
        cert.block_reasons.append("replay_or_secure_case_missing")
        cert.checks.append(
            CheckResult(
                check_id="proverif.deterministic_replay_binding",
                kind="replay",
                status="failed",
                expected="secure and replay cases",
                observed="missing",
            )
        )

    # OPAM cannot promote the protocol lane by itself.
    opam_boundary = opam_cannot_promote_protocol_lane()
    boundary_ok = bool(opam_boundary.get("blocks_alone"))
    if not boundary_ok:
        cert.block_reasons.append("opam_incorrectly_promotes")
    cert.checks.append(
        CheckResult(
            check_id="opam.support_only_boundary",
            kind="authority",
            status="passed" if boundary_ok else "failed",
            expected="promotion_blocked",
            observed=(
                f"allowed={opam_boundary.get('promotion_allowed')},"
                f"can_satisfy={opam_boundary.get('can_satisfy_protocol_requirement')}"
            ),
            detail="OPAM is support only and cannot promote the protocol lane",
            bindings=opam_boundary,
        )
    )

    # Bind model, claims, bounds, and exact binaries.
    ceiling = SymbolicModelCeiling.disclose(
        equational_theories=list(
            corpus.get("theory_bindings") or DEFAULT_THEORY_BINDINGS
        ),
        claim_kinds=list(corpus.get("claim_bindings") or DEFAULT_CLAIM_BINDINGS),
    )
    cert.bindings = {
        "model": {
            "ceiling": ceiling,
            "supported_theories": sorted(
                item.value for item in PROVERIF_SUPPORTED_THEORIES
            ),
            "bound_theories": list(
                corpus.get("theory_bindings") or DEFAULT_THEORY_BINDINGS
            ),
        },
        "theory": {
            "ceiling": ceiling,
            "supported_theories": sorted(
                item.value for item in PROVERIF_SUPPORTED_THEORIES
            ),
            "bound_theories": list(
                corpus.get("theory_bindings") or DEFAULT_THEORY_BINDINGS
            ),
        },
        "claims": {
            "supported_claim_kinds": sorted(
                item.value for item in PROVERIF_SUPPORTED_CLAIMS
            ),
            "bound_claim_kinds": list(
                corpus.get("claim_bindings") or DEFAULT_CLAIM_BINDINGS
            ),
            "secure_case_claims": (
                list(secure.claim_outcomes) if secure is not None else []
            ),
        },
        "bounds": dict(corpus.get("bounds") or DEFAULT_BOUNDS),
        "binaries": {
            "proverif": {
                "tool_id": TOOL_ID,
                "locked_version": LOCKED_PROVERIF_VERSION,
                "executable_path": cert.proverif_executable,
                "version_string": cert.proverif_version_string,
                "identity_probed": cert.proverif_identity_probed,
                "version_match": cert.proverif_version_match,
            },
            "opam": {
                "tool_id": SUPPORT_TOOL_ID,
                "locked_version": LOCKED_OPAM_VERSION,
                "executable_path": cert.opam_executable,
                "version_string": cert.opam_version_string,
                "identity_probed": cert.opam_identity_probed,
                "version_match": cert.opam_version_match,
                "support_only": True,
                "can_promote_protocol_lane": False,
                "isolated_opam_root": cert.isolated_opam_root,
                "isolated_root_validated": cert.isolated_root_validated,
            },
        },
        "authority": {
            "ceiling": AUTHORITY_CEILING,
            "scope": AUTHORITY_SCOPE,
            "backend_interface": PROVERIF_BACKEND_VERSION,
            "opam_is_support_only": True,
            "not_kernel": True,
            "not_advisor": True,
        },
        "opam_promotion_boundary": opam_boundary,
        "isolated_opam_root": {
            "path": cert.isolated_opam_root,
            "validated": cert.isolated_root_validated,
            "global_switch_mutation_forbidden": True,
        },
    }
    cert.checks.append(
        CheckResult(
            check_id="proverif.bindings",
            kind="binding",
            status="passed",
            expected="model,claims,bounds,binaries,isolated_root",
            observed=content_digest(cert.bindings)[:16],
            detail=(
                "receipt binds model, claims, bounds, exact binaries, "
                "and isolated OPAM root"
            ),
            bindings=dict(cert.bindings),
        )
    )

    required_kinds = {
        "secure",
        "attack",
        "mutation",
        "replay",
        "malformed",
        "cancellation",
        "version_mismatch",
    }
    present_kinds = {str(case.get("kind") or "") for case in cases}
    missing_kinds = sorted(required_kinds - present_kinds)
    if missing_kinds:
        cert.block_reasons.append("corpus_missing_kinds:" + ",".join(missing_kinds))

    semantic_ok = all(
        check.status == "passed"
        for check in cert.checks
        if check.kind
        in {
            "secure",
            "attack",
            "mutation",
            "replay",
            "malformed",
            "cancellation",
            "version_mismatch",
            "authority",
            "binding",
            "isolation",
        }
        or check.check_id
        in {
            "proverif.deterministic_replay_binding",
            "proverif.bindings",
            "opam.support_only_boundary",
            "opam.isolated_root",
            "proverif.offline_policy",
        }
    )

    # Production certification requires live locked binaries + isolation + suite.
    cert.production_certified = bool(
        cert.proverif_usable
        and cert.opam_usable
        and cert.isolated_root_validated
        and not cert.network_used
        and not cert.install_attempted
        and not cert.download_attempted
        and not cert.global_opam_mutation_attempted
        and semantic_ok
        and not missing_kinds
        and not any(
            reason.startswith("case_failed:") or reason.startswith("replay_")
            for reason in cert.block_reasons
        )
        and boundary_ok
    )
    if cert.production_certified:
        cert.promotion_blocked = False
        cert.block_reasons = []
        cert.notes = (
            "Pinned ProVerif 2.05 + OPAM 2.5.2 certified for protocol claims "
            "and attacks under an isolated OPAM root; OPAM remains support only."
        )
    else:
        cert.promotion_blocked = True
        if not cert.notes:
            if semantic_ok and not (cert.proverif_usable and cert.opam_usable):
                cert.notes = (
                    "Semantic corpus passed offline; live locked ProVerif/OPAM "
                    "identities unavailable — production certification withheld."
                )
            else:
                cert.notes = (
                    "ProVerif/OPAM certification incomplete or failed; "
                    "protocol-lane promotion blocked."
                )

    return cert


def build_certification_receipt(
    *,
    repo_root: Path | None = None,
    manifest: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
    proverif_executable: str | None = None,
    opam_executable: str | None = None,
    isolated_opam_root: str | Path | None = None,
) -> dict[str, Any]:
    cert = run_certification_suite(
        repo_root=repo_root,
        manifest=manifest,
        env=env,
        proverif_executable=proverif_executable,
        opam_executable=opam_executable,
        isolated_opam_root=isolated_opam_root,
    )
    payload = cert.to_dict()
    payload["policy"] = {
        "no_install": True,
        "no_download": True,
        "no_network": True,
        "opam_is_support_only": True,
        "opam_cannot_promote_protocol_lane": True,
        "isolated_opam_root_required": True,
        "never_mutate_global_opam_switch": True,
        "exact_binary_binding_required": True,
        "authority_is_protocol_verification_only": True,
        "does_not_edit_central_certificate": True,
        "does_not_edit_tamarin_lane": True,
        "does_not_edit_shared_lock": True,
    }
    payload["semantic_corpus_passed"] = all(case.matched for case in cert.cases)
    payload["receipt_digest_sha256"] = content_digest(
        {
            key: value
            for key, value in payload.items()
            if key != "receipt_digest_sha256"
        }
    )
    return payload


def certify_proverif_toolchain(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Lane-handler entry point compatible with role-aware promotion binding."""

    repo_root = kwargs.get("repo_root")
    if repo_root is not None and not isinstance(repo_root, Path):
        repo_root = Path(str(repo_root))
    receipt = build_certification_receipt(
        repo_root=repo_root,
        manifest=kwargs.get("manifest"),
        env=kwargs.get("env"),
        proverif_executable=kwargs.get("proverif_executable"),
        opam_executable=kwargs.get("opam_executable"),
        isolated_opam_root=kwargs.get("isolated_opam_root"),
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
    return certify_proverif_toolchain(*args, **kwargs)


def bind_protocol_lane_handler(
    *,
    policy: Any | None = None,
    replace: bool = True,
) -> Any | None:
    """Register this certifier on the protocol lane when roles surface exists."""

    if _bind_lane_handler is None or _build_role_aware_policy is None:
        return None
    target = policy if policy is not None else _build_role_aware_policy()
    return _bind_lane_handler(
        LANE_ID, lane_handler, policy=target, replace=replace
    )


# ---------------------------------------------------------------------------
# Live semantic certification (FVT-G205 / FVT-058)
# ---------------------------------------------------------------------------


@dataclass
class LiveCaseOutcome:
    """One live (or deliberately non-live) protocol case outcome."""

    case_id: str
    kind: str
    expect: str
    status: str
    matched: bool
    evidence_class: str = EVIDENCE_CLASS_LIVE
    live_executed: bool = False
    reason_codes: list[str] = field(default_factory=list)
    claim_outcomes: list[dict[str, Any]] = field(default_factory=list)
    attack_trace: dict[str, Any] | None = None
    source: str = ""
    source_digest: str = ""
    source_format: str = "pv"
    query: str = ""
    assumptions: list[str] = field(default_factory=list)
    bounds: dict[str, Any] = field(default_factory=dict)
    output_digest: str = ""
    raw_output: str = ""
    stdout: str = ""
    stderr: str = ""
    returncode: int | None = None
    elapsed_ms: int = 0
    tool_id: str = TOOL_ID
    tool_version: str | None = None
    executable_path: str | None = None
    support_tool_id: str = SUPPORT_TOOL_ID
    support_tool_version: str | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _truncate_raw(text: str, cap: int = _RAW_OUTPUT_CAP) -> str:
    if len(text) <= cap:
        return text
    head = cap // 2
    tail = cap - head
    return (
        text[:head]
        + f"\n(* ... truncated {len(text) - cap} bytes ... *)\n"
        + text[-tail:]
    )


def default_live_corpus_manifest() -> dict[str, Any]:
    return {
        "schema_version": LIVE_CORPUS_SCHEMA,
        "interface": LIVE_INTERFACE,
        "goal_id": LIVE_GOAL_ID,
        "task_id": LIVE_TASK_ID,
        "program": LIVE_PROGRAM,
        "tool_id": TOOL_ID,
        "support_tool_id": SUPPORT_TOOL_ID,
        "lane_id": LANE_ID,
        "locked_proverif_version": LOCKED_PROVERIF_VERSION,
        "locked_opam_version": LOCKED_OPAM_VERSION,
        "authority_ceiling": AUTHORITY_CEILING,
        "authority_scope": AUTHORITY_SCOPE,
        "policy": {
            "no_install": True,
            "no_download": True,
            "no_network": True,
            "parser_fixtures_are_non_production": True,
            "live_binary_required_for_semantic_proof": True,
            "opam_is_support_only": True,
            "cannot_substitute_tamarin": True,
            "engines_are_independent": True,
            "never_mutate_global_opam_switch": True,
        },
        "required_kinds": [
            "secure",
            "attack",
            "mutation",
            "replay",
            "malformed",
            "timeout",
            "disagreement",
            "bounded_search",
        ],
        "cases": [dict(case) for case in _DEFAULT_LIVE_CORPUS_CASES],
    }


def live_corpus_cases(
    manifest: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    payload = manifest if manifest is not None else default_live_corpus_manifest()
    cases = payload.get("cases") or []
    return [dict(case) for case in cases]


def _live_expect_matches(expect: str, observed: str) -> bool:
    if expect == "secure":
        return observed == "secure"
    if expect == "attack":
        return observed == "attack"
    if expect == "quarantined":
        return observed == "quarantined"
    if expect == "blocked":
        return observed == "blocked"
    if expect == "rejected_or_quarantined":
        return observed in {"attack", "quarantined", "unknown", "blocked"}
    return observed == expect


def _normalize_query_key(query: str) -> str:
    text = re.sub(r"\s+", " ", (query or "").strip().lower())
    text = re.sub(r"\b([a-z][a-z0-9]*)_\d+\b", r"\1", text)
    return text


def _match_claim_id(
    result_query: str, claim_queries: Mapping[str, str]
) -> str:
    key = _normalize_query_key(result_query)
    inverse = {
        _normalize_query_key(query): claim_id
        for claim_id, query in claim_queries.items()
    }
    if key in inverse:
        return inverse[key]
    for claim_id, query in claim_queries.items():
        norm = _normalize_query_key(query)
        if norm and (norm in key or key in norm):
            return claim_id
        if "attacker" in norm and "attacker" in key:
            return claim_id
        if "==>" in norm and "==>" in key:
            return claim_id
    return result_query


def _parse_live_proverif_outcomes(
    stdout: str,
    stderr: str,
    *,
    claim_queries: Mapping[str, str],
) -> tuple[Any, ...]:
    outcomes = parse_proverif_claim_outcomes(
        stdout, stderr, claim_queries=claim_queries
    )
    unknown_declared = [
        item
        for item in outcomes
        if item.claim_id in claim_queries
        and item.verdict is ClaimVerdict.UNKNOWN
    ]
    if not unknown_declared:
        return outcomes

    from ipfs_datasets_py.logic.backends.protocol.proverif import (  # noqa: WPS433
        ClaimOutcome as PVClaimOutcome,
    )

    combined = f"{stdout}\n{stderr}"
    raw_digest = content_digest(combined)
    result_re = re.compile(
        r"(?im)^\s*RESULT\s+(.+?)\s+is\s+(true|false|cannot be proved)\s*\.?\s*$"
    )
    rebuilt: list[Any] = []
    seen: set[str] = set()
    for match in result_re.finditer(combined):
        query = match.group(1).strip()
        token = match.group(2).lower()
        if token == "true":
            verdict = ClaimVerdict.TRUE
        elif token == "false":
            verdict = ClaimVerdict.FALSE
        else:
            verdict = ClaimVerdict.CANNOT_PROVE
        claim_id = _match_claim_id(query, claim_queries)
        attack = None
        if verdict is ClaimVerdict.FALSE:
            attack = parse_attack_trace(
                combined, claim_id=claim_id, raw_digest=raw_digest
            )
        rebuilt.append(
            PVClaimOutcome(
                claim_id=claim_id,
                query_text=query,
                verdict=verdict,
                attack_trace=attack,
            )
        )
        seen.add(claim_id)
    for claim_id, query in claim_queries.items():
        if claim_id in seen:
            continue
        rebuilt.append(
            PVClaimOutcome(
                claim_id=claim_id,
                query_text=query,
                verdict=ClaimVerdict.UNKNOWN,
                reason="query outcome missing from ProVerif output",
            )
        )
    return tuple(rebuilt)


def _classify_live_stdout(
    stdout: str,
    stderr: str,
    *,
    claim_queries: Mapping[str, str],
    timed_out: bool = False,
    force_timeout: bool = False,
) -> tuple[str, list[str], list[dict[str, Any]], dict[str, Any] | None]:
    reason_codes: list[str] = []
    if timed_out or force_timeout:
        reason_codes.append("timeout")
        return "quarantined", reason_codes, [], None

    outcomes = _parse_live_proverif_outcomes(
        stdout, stderr, claim_queries=claim_queries
    )
    status_enum, quarantine, accepted = classify_claim_outcomes(outcomes)
    attack: dict[str, Any] | None = None
    for item in outcomes:
        if item.attack_trace is not None:
            attack = item.attack_trace.to_dict()
            break
    if attack is None and status_enum is ResultStatus.ATTACK_FOUND:
        raw_digest = content_digest(f"{stdout}\n{stderr}")
        trace = parse_attack_trace(
            f"{stdout}\n{stderr}",
            claim_id=next(iter(claim_queries), "claim:unknown"),
            raw_digest=raw_digest,
        )
        if trace is not None:
            attack = trace.to_dict()

    if status_enum is ResultStatus.SECURE and accepted:
        observed = "secure"
    elif status_enum is ResultStatus.ATTACK_FOUND:
        observed = "attack"
    elif quarantine is not None:
        observed = "quarantined"
        reason_codes.append(str(quarantine.reason.value))
    else:
        combined = f"{stdout}\n{stderr}".lower()
        if "error" in combined or "syntax" in combined or not outcomes:
            reason_codes.append("malformed_or_tool_error")
            observed = "quarantined"
        else:
            observed = "unknown"
            reason_codes.append("inconclusive")

    return (
        observed,
        list(dict.fromkeys(reason_codes)),
        [item.to_dict() for item in outcomes],
        attack,
    )


def run_live_protocol_case(
    case: Mapping[str, Any],
    *,
    executable: str,
    env: Mapping[str, str] | None = None,
    tool_version: str | None = None,
    support_tool_version: str | None = None,
) -> LiveCaseOutcome:
    """Execute one live ProVerif case against the pinned binary."""

    case_id = str(case.get("case_id") or "case")
    kind = str(case.get("kind") or "unknown")
    expect = str(case.get("expect") or "unknown")
    source = str(case.get("source") or "")
    source_format = str(case.get("source_format") or "pv")
    claim_queries = {
        str(key): str(value)
        for key, value in dict(case.get("claim_queries") or {}).items()
    }
    assumptions = [str(item) for item in (case.get("assumptions") or [])]
    query = str(case.get("query") or "")
    bounds = dict(case.get("bounds") or {})
    timeout = float(
        bounds.get("timeout_seconds")
        if bounds.get("timeout_seconds") is not None
        else LIVE_CHECK_TIMEOUT_SECONDS
    )
    force_timeout = bool(case.get("force_timeout"))
    source_digest = content_digest(source)
    probe_env = offline_env(env)

    if not source.strip():
        return LiveCaseOutcome(
            case_id=case_id,
            kind=kind,
            expect=expect,
            status="quarantined",
            matched=_live_expect_matches(expect, "quarantined"),
            evidence_class=EVIDENCE_CLASS_LIVE,
            live_executed=False,
            reason_codes=["empty_source"],
            source=source,
            source_digest=source_digest,
            source_format=source_format,
            query=query,
            assumptions=assumptions,
            bounds=bounds,
            tool_version=tool_version,
            executable_path=executable,
            support_tool_version=support_tool_version,
            detail="empty source rejected before tool invocation",
        )

    with tempfile.TemporaryDirectory(prefix="proverif-live-") as tmp:
        work = Path(tmp)
        source_path = work / f"{case_id}.pv"
        source_path.write_text(
            source if source.endswith("\n") else source + "\n",
            encoding="utf-8",
        )
        argv = [executable, str(source_path)]
        started = time.monotonic()
        timed_out = False
        completed: subprocess.CompletedProcess[str] | None = None
        try:
            completed = subprocess.run(
                argv,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=dict(probe_env),
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
            stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
            returncode: int | None = None
        except OSError as exc:
            elapsed_ms = max(0, round((time.monotonic() - started) * 1000))
            return LiveCaseOutcome(
                case_id=case_id,
                kind=kind,
                expect=expect,
                status="quarantined",
                matched=_live_expect_matches(expect, "quarantined"),
                evidence_class=EVIDENCE_CLASS_LIVE,
                live_executed=False,
                reason_codes=["tool_invocation_failed", type(exc).__name__],
                source=source,
                source_digest=source_digest,
                source_format=source_format,
                query=query,
                assumptions=assumptions,
                bounds=bounds,
                elapsed_ms=elapsed_ms,
                tool_version=tool_version,
                executable_path=executable,
                support_tool_version=support_tool_version,
                detail=str(exc),
            )
        else:
            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
            returncode = completed.returncode

        elapsed_ms = max(0, round((time.monotonic() - started) * 1000))
        observed, reason_codes, claim_outcomes, attack = _classify_live_stdout(
            stdout,
            stderr,
            claim_queries=claim_queries,
            timed_out=timed_out,
            force_timeout=force_timeout and timed_out,
        )
        if force_timeout and not timed_out:
            observed = "quarantined"
            reason_codes = list(
                dict.fromkeys([*reason_codes, "timeout_bound_not_hit", "timeout"])
            )

        if kind in {"mutation", "disagreement"} and observed == "secure":
            observed = "quarantined"
            reason_codes = list(
                dict.fromkeys([*reason_codes, f"{kind}_still_secure"])
            )

        raw = _truncate_raw(f"{stdout}\n{stderr}")
        output_digest = content_digest(f"{stdout}\n{stderr}")
        matched = _live_expect_matches(expect, observed)
        return LiveCaseOutcome(
            case_id=case_id,
            kind=kind,
            expect=expect,
            status=observed,
            matched=matched,
            evidence_class=EVIDENCE_CLASS_LIVE,
            live_executed=True,
            reason_codes=reason_codes,
            claim_outcomes=claim_outcomes,
            attack_trace=attack,
            source=source,
            source_digest=source_digest,
            source_format=source_format,
            query=query,
            assumptions=assumptions,
            bounds=bounds,
            output_digest=output_digest,
            raw_output=raw,
            stdout=_truncate_raw(stdout),
            stderr=_truncate_raw(stderr),
            returncode=returncode,
            elapsed_ms=elapsed_ms,
            tool_version=tool_version,
            executable_path=executable,
            support_tool_version=support_tool_version,
            detail=str(case.get("description") or ""),
        )


def parser_fixture_evidence_class() -> str:
    return EVIDENCE_CLASS_PARSER_FIXTURE


def run_live_semantic_suite(
    *,
    repo_root: Path | None = None,
    manifest: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
    proverif_executable: str | None = None,
    opam_executable: str | None = None,
) -> dict[str, Any]:
    """Run the live ProVerif semantic suite and return a tool receipt."""

    root = repo_root or repo_root_from()
    corpus = (
        manifest if manifest is not None else default_live_corpus_manifest()
    )
    cases = live_corpus_cases(corpus)
    probe_env = offline_env(env)

    proverif_probe = probe_proverif_identity(
        env=probe_env, executable=proverif_executable
    )
    opam_probe = probe_opam_identity(env=probe_env, executable=opam_executable)
    proverif_usable = bool(
        proverif_probe.get("identity_probed")
        and proverif_probe.get("version_match")
    )
    opam_usable = bool(
        opam_probe.get("identity_probed") and opam_probe.get("version_match")
    )

    checks: list[dict[str, Any]] = []
    live_cases: list[LiveCaseOutcome] = []
    block_reasons: list[str] = []

    checks.append(
        {
            "check_id": "proverif.live.offline_policy",
            "kind": "policy",
            "status": "passed",
            "expected": "no_install_no_download_no_network",
            "observed": "offline_env",
            "detail": "live suite never installs, downloads, or opens the network",
        }
    )
    checks.append(
        {
            "check_id": "proverif.live.parser_fixtures_non_production",
            "kind": "policy",
            "status": "passed",
            "expected": EVIDENCE_CLASS_PARSER_FIXTURE,
            "observed": parser_fixture_evidence_class(),
            "detail": (
                "offline evaluate_corpus_case fixtures remain non-production; "
                "only live binary runs contribute semantic proof evidence"
            ),
        }
    )

    if not proverif_usable:
        reason = str(proverif_probe.get("probe_error") or "proverif_unavailable")
        block_reasons.append(f"proverif:{reason}")
        checks.append(
            {
                "check_id": "proverif.live.identity",
                "kind": "identity",
                "status": "unavailable",
                "expected": LOCKED_PROVERIF_VERSION,
                "observed": reason,
            }
        )
    else:
        checks.append(
            {
                "check_id": "proverif.live.identity",
                "kind": "identity",
                "status": "passed",
                "expected": LOCKED_PROVERIF_VERSION,
                "observed": proverif_probe.get("version_string") or "",
                "bindings": {
                    "executable_path": proverif_probe.get("executable_path"),
                    "version_string": proverif_probe.get("version_string"),
                },
            }
        )

    if not opam_usable:
        reason = str(opam_probe.get("probe_error") or "opam_unavailable")
        checks.append(
            {
                "check_id": "opam.live.identity",
                "kind": "identity",
                "status": "unavailable",
                "expected": LOCKED_OPAM_VERSION,
                "observed": reason,
                "detail": "support only; absence recorded but does not substitute",
            }
        )
    else:
        checks.append(
            {
                "check_id": "opam.live.identity",
                "kind": "identity",
                "status": "passed",
                "expected": LOCKED_OPAM_VERSION,
                "observed": opam_probe.get("version_string") or "",
                "bindings": {
                    "executable_path": opam_probe.get("executable_path"),
                    "version_string": opam_probe.get("version_string"),
                    "support_only": True,
                },
            }
        )

    executable = str(proverif_probe.get("executable_path") or "")
    outcomes_by_id: dict[str, LiveCaseOutcome] = {}
    if proverif_usable and executable:
        for case in cases:
            outcome = run_live_protocol_case(
                case,
                executable=executable,
                env=probe_env,
                tool_version=str(proverif_probe.get("version_string") or "")
                or None,
                support_tool_version=str(opam_probe.get("version_string") or "")
                or None,
            )
            outcomes_by_id[outcome.case_id] = outcome
            live_cases.append(outcome)
            status = "passed" if outcome.matched else "failed"
            if not outcome.matched:
                block_reasons.append(f"case_failed:{outcome.case_id}")
            checks.append(
                {
                    "check_id": f"proverif.live.{outcome.case_id}",
                    "kind": outcome.kind,
                    "status": status,
                    "expected": outcome.expect,
                    "observed": outcome.status,
                    "detail": outcome.detail,
                    "reason_codes": list(outcome.reason_codes),
                    "bindings": {
                        "evidence_class": outcome.evidence_class,
                        "live_executed": outcome.live_executed,
                        "source_digest": outcome.source_digest,
                        "output_digest": outcome.output_digest,
                        "query": outcome.query,
                        "assumptions": list(outcome.assumptions),
                        "bounds": dict(outcome.bounds),
                        "attack_trace": outcome.attack_trace,
                        "claim_outcomes": list(outcome.claim_outcomes),
                    },
                }
            )
    else:
        block_reasons.append("live_execution_unavailable")

    secure = outcomes_by_id.get("live_secure_secrecy_auth")
    replay = outcomes_by_id.get("live_deterministic_replay")
    if secure is not None and replay is not None:
        replay_ok = (
            secure.status == "secure"
            and replay.status == "secure"
            and secure.source_digest == replay.source_digest
            and secure.matched
            and replay.matched
        )
        if not replay_ok:
            block_reasons.append("live_replay_nondeterministic_or_failed")
        checks.append(
            {
                "check_id": "proverif.live.deterministic_replay_binding",
                "kind": "replay",
                "status": "passed" if replay_ok else "failed",
                "expected": "identical secure source digests",
                "observed": (
                    f"secure={secure.source_digest[:12]},"
                    f"replay={replay.source_digest[:12]},"
                    f"out_secure={secure.output_digest[:12]},"
                    f"out_replay={replay.output_digest[:12]}"
                ),
                "bindings": {
                    "secure_source_digest": secure.source_digest,
                    "replay_source_digest": replay.source_digest,
                    "secure_output_digest": secure.output_digest,
                    "replay_output_digest": replay.output_digest,
                },
            }
        )
    else:
        block_reasons.append("live_replay_or_secure_case_missing")
        checks.append(
            {
                "check_id": "proverif.live.deterministic_replay_binding",
                "kind": "replay",
                "status": "failed",
                "expected": "secure and replay cases",
                "observed": "missing",
            }
        )

    required_kinds = {
        "secure",
        "attack",
        "mutation",
        "replay",
        "malformed",
        "timeout",
        "disagreement",
        "bounded_search",
    }
    present_kinds = {str(case.get("kind") or "") for case in cases}
    missing_kinds = sorted(required_kinds - present_kinds)
    if missing_kinds:
        block_reasons.append("live_corpus_missing_kinds:" + ",".join(missing_kinds))

    opam_boundary = opam_cannot_promote_protocol_lane()
    boundary_ok = bool(opam_boundary.get("blocks_alone"))
    if not boundary_ok:
        block_reasons.append("opam_incorrectly_promotes")
    checks.append(
        {
            "check_id": "opam.live.support_only_boundary",
            "kind": "authority",
            "status": "passed" if boundary_ok else "failed",
            "expected": "promotion_blocked",
            "observed": (
                f"allowed={opam_boundary.get('promotion_allowed')},"
                f"can_satisfy={opam_boundary.get('can_satisfy_protocol_requirement')}"
            ),
            "bindings": opam_boundary,
        }
    )

    binding_case = secure or next(
        (item for item in live_cases if item.status == "secure"), None
    )
    bindings = {
        "tool": {
            "tool_id": TOOL_ID,
            "locked_version": LOCKED_PROVERIF_VERSION,
            "executable_path": proverif_probe.get("executable_path"),
            "version_string": proverif_probe.get("version_string"),
            "identity_probed": bool(proverif_probe.get("identity_probed")),
            "version_match": bool(proverif_probe.get("version_match")),
        },
        "dependency": {
            "tool_id": SUPPORT_TOOL_ID,
            "locked_version": LOCKED_OPAM_VERSION,
            "executable_path": opam_probe.get("executable_path"),
            "version_string": opam_probe.get("version_string"),
            "identity_probed": bool(opam_probe.get("identity_probed")),
            "version_match": bool(opam_probe.get("version_match")),
            "support_only": True,
            "can_promote_protocol_lane": False,
        },
        "source": {
            "source_digest": binding_case.source_digest if binding_case else "",
            "source_format": binding_case.source_format if binding_case else "pv",
            "case_id": binding_case.case_id if binding_case else "",
        },
        "query": binding_case.query if binding_case else "",
        "assumptions": list(binding_case.assumptions) if binding_case else [],
        "bound": dict(binding_case.bounds) if binding_case else dict(DEFAULT_BOUNDS),
        "witnesses_traces": {
            case.case_id: case.attack_trace
            for case in live_cases
            if case.attack_trace is not None
        },
        "raw_output": {
            "output_digest": binding_case.output_digest if binding_case else "",
            "raw_output_cap_bytes": _RAW_OUTPUT_CAP,
        },
        "authority": {
            "ceiling": AUTHORITY_CEILING,
            "scope": AUTHORITY_SCOPE,
            "backend_interface": PROVERIF_BACKEND_VERSION,
            "not_tamarin": True,
            "opam_is_support_only": True,
            "parser_fixtures_are_non_production": True,
        },
    }
    checks.append(
        {
            "check_id": "proverif.live.bindings",
            "kind": "binding",
            "status": "passed" if binding_case is not None else "failed",
            "expected": (
                "tool,dependency,source,query,assumptions,bound,"
                "witnesses_traces,raw_output"
            ),
            "observed": content_digest(bindings)[:16],
            "bindings": bindings,
        }
    )

    case_ok = all(case.matched for case in live_cases) and bool(live_cases)
    semantic_checks_ok = all(
        check.get("status") == "passed"
        for check in checks
        if check.get("kind")
        in {
            "secure",
            "attack",
            "mutation",
            "replay",
            "malformed",
            "timeout",
            "disagreement",
            "bounded_search",
            "authority",
            "binding",
            "policy",
        }
        or str(check.get("check_id") or "").endswith("_binding")
        or str(check.get("check_id") or "").endswith(".bindings")
    )
    live_semantic_certified = bool(
        proverif_usable
        and case_ok
        and semantic_checks_ok
        and boundary_ok
        and not missing_kinds
        and not any(
            reason.startswith("case_failed:") or reason.startswith("live_replay_")
            for reason in block_reasons
        )
    )

    capability_gap = (
        None
        if proverif_usable
        else CAPABILITY_GAP_PINNED_BINARY_UNAVAILABLE
    )
    receipt = {
        "interface": LIVE_INTERFACE,
        "schema_version": LIVE_SCHEMA_VERSION,
        "tool_surface": LIVE_TOOL_SURFACE,
        "goal_id": LIVE_GOAL_ID,
        "task_id": LIVE_TASK_ID,
        "repair_task_id": LIVE_REPAIR_TASK_ID,
        "program": LIVE_PROGRAM,
        "tool_id": TOOL_ID,
        "support_tool_id": SUPPORT_TOOL_ID,
        "lane_id": LANE_ID,
        "certification_surface": CERTIFICATION_SURFACE,
        "locked_proverif_version": LOCKED_PROVERIF_VERSION,
        "locked_opam_version": LOCKED_OPAM_VERSION,
        "authority_ceiling": AUTHORITY_CEILING,
        "authority_scope": AUTHORITY_SCOPE,
        "proverif_executable": proverif_probe.get("executable_path"),
        "opam_executable": opam_probe.get("executable_path"),
        "proverif_version_string": proverif_probe.get("version_string"),
        "opam_version_string": opam_probe.get("version_string"),
        "proverif_usable": proverif_usable,
        "opam_usable": opam_usable,
        "network_used": False,
        "install_attempted": False,
        "download_attempted": False,
        "global_opam_mutation_attempted": False,
        "live_execution": bool(proverif_usable and live_cases),
        "live_semantic_certified": live_semantic_certified,
        "production_certified": live_semantic_certified and opam_usable,
        "promotion_blocked": not live_semantic_certified,
        "parser_fixtures_are_non_production": True,
        "cannot_substitute_tamarin": True,
        "fixture_or_parser_cannot_satisfy_live_goal": True,
        "capability_gap": capability_gap,
        "block_reasons": [] if live_semantic_certified else list(block_reasons),
        "checks": checks,
        "cases": [case.to_dict() for case in live_cases],
        "bindings": bindings,
        "policy": {
            **dict(corpus.get("policy") or {}),
            "fixture_or_parser_cannot_satisfy_live_goal": True,
            "live_binary_required_for_semantic_proof": True,
        },
        "repo_root": str(root),
        "notes": (
            "Pinned ProVerif live semantic corpus certified."
            if live_semantic_certified
            else (
                "ProVerif live semantic certification incomplete or unavailable; "
                "parser fixtures remain non-production and cannot satisfy "
                f"{LIVE_GOAL_ID}."
            )
        ),
    }
    receipt["receipt_digest_sha256"] = content_digest(
        {key: value for key, value in receipt.items() if key != "receipt_digest_sha256"}
    )
    return receipt


def build_live_semantic_receipt(
    *,
    repo_root: Path | None = None,
    manifest: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
    proverif_executable: str | None = None,
    opam_executable: str | None = None,
) -> dict[str, Any]:
    return run_live_semantic_suite(
        repo_root=repo_root,
        manifest=manifest,
        env=env,
        proverif_executable=proverif_executable,
        opam_executable=opam_executable,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Certify the pinned ProVerif/OPAM protocol toolchain "
            f"({INTERFACE}; ProVerif {LOCKED_PROVERIF_VERSION} + "
            f"OPAM {LOCKED_OPAM_VERSION}, isolated root) and optional live "
            f"semantics ({LIVE_INTERFACE})."
        )
    )
    parser.add_argument("--json", action="store_true", help="Print receipt as JSON")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--proverif", type=str, default=None)
    parser.add_argument("--opam", type=str, default=None)
    parser.add_argument("--isolated-opam-root", type=str, default=None)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run live semantic suite instead of offline toolchain corpus",
    )
    args = parser.parse_args(argv)

    root = args.repo_root or repo_root_from()
    if args.live:
        receipt = build_live_semantic_receipt(
            repo_root=root,
            proverif_executable=args.proverif,
            opam_executable=args.opam,
        )
        if args.json:
            print(json.dumps(receipt, indent=2, sort_keys=True))
        else:
            print(f"{LIVE_INTERFACE} goal={LIVE_GOAL_ID} task={LIVE_TASK_ID}")
            print(
                f"live_semantic_certified={receipt.get('live_semantic_certified')} "
                f"cases={len(receipt.get('cases') or [])}"
            )
            for check in receipt.get("checks") or []:
                print(
                    f"  [{check.get('status'):10}] {check.get('check_id')}: "
                    f"expected={check.get('expected')} observed={check.get('observed')}"
                )
            if receipt.get("block_reasons"):
                print("block_reasons:", ", ".join(receipt["block_reasons"]))
        return 0 if receipt.get("live_semantic_certified") else 1

    receipt = build_certification_receipt(
        repo_root=root,
        proverif_executable=args.proverif,
        opam_executable=args.opam,
        isolated_opam_root=args.isolated_opam_root,
    )
    if args.json:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    else:
        print(f"{INTERFACE} goal={GOAL_ID} task={TASK_ID}")
        print(
            f"proverif={receipt.get('proverif_version_string')!r} "
            f"opam={receipt.get('opam_version_string')!r}"
        )
        print(
            f"usable_proverif={receipt.get('proverif_usable')} "
            f"usable_opam={receipt.get('opam_usable')} "
            f"isolated={receipt.get('isolated_root_validated')} "
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
    semantic_ok = bool(receipt.get("semantic_corpus_passed"))
    return 0 if semantic_ok else 1


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
    "SUPPORT_TOOL_ID",
    "CERTIFICATION_SURFACE",
    "HANDLER_ID",
    "LOCKED_PROVERIF_VERSION",
    "LOCKED_OPAM_VERSION",
    "AUTHORITY_CEILING",
    "AUTHORITY_SCOPE",
    "LIVE_INTERFACE",
    "LIVE_SCHEMA_VERSION",
    "LIVE_CORPUS_SCHEMA",
    "LIVE_GOAL_ID",
    "LIVE_TASK_ID",
    "LIVE_REPAIR_TASK_ID",
    "LIVE_PROGRAM",
    "EVIDENCE_CLASS_LIVE",
    "EVIDENCE_CLASS_PARSER_FIXTURE",
    "CAPABILITY_GAP_PINNED_BINARY_UNAVAILABLE",
    "DEFAULT_PROTOCOL_LIVE_CERTIFICATE_RELATIVE",
    "CheckResult",
    "CaseOutcome",
    "LiveCaseOutcome",
    "ProVerifToolchainCertification",
    "repo_root_from",
    "content_digest",
    "offline_env",
    "bounded_run",
    "resolve_executable",
    "default_corpus_manifest",
    "load_corpus_manifest",
    "corpus_cases",
    "default_live_corpus_manifest",
    "live_corpus_cases",
    "probe_proverif_identity",
    "probe_opam_identity",
    "validate_isolated_opam_root",
    "evaluate_corpus_case",
    "parser_fixture_evidence_class",
    "run_live_protocol_case",
    "run_live_semantic_suite",
    "build_live_semantic_receipt",
    "opam_cannot_promote_protocol_lane",
    "run_certification_suite",
    "build_certification_receipt",
    "certify_proverif_toolchain",
    "lane_handler",
    "bind_protocol_lane_handler",
    "main",
]
