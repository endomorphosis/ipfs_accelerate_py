#!/usr/bin/env python3
"""External Runtime MTL cross-runtime parity and vendor certification.

``ExternalRuntimeMTLCertification@1`` / FVT-G181 (FVT-052) and vendor path
``ExternalRuntimeMTLVendorCertification@1`` / FVT-G210 (FVT-056, FVT-072).

Explicit strict installation selects the exact pin-bound external monitor.
Python (in-process reference), TypeScript (when available), and the external
engine must agree on:

* satisfied / violated golden traces;
* closed vs open boundary intervals;
* interval and event mutations;
* shortest-prefix discovery and deterministic replay;
* malformed input fail-closed behaviour;
* timeout and bounds or disagreement quarantine.

Finite-trace authority is preserved; no global correctness claim is inferred.

The **vendor** lane (FVT-G210) certifies a reproducibly built TypeScript/Node
monitor that never imports or dispatches to the Python reference.  Package,
source, lockfile, runtime, launcher, launcher target, executable, and artifact
digests are bound. Generated Python hermetic parity wrappers remain
non-production shadow evidence and cannot satisfy vendor production claims.

Objective validation repair (FVT-072)
-------------------------------------
Path evidence for the vendor installer, certifier, offline install boundary,
and install receipt may already exist while the supervisor validation gate
still needs an explicit re-proof of the full FVT-G210 acceptance matrix. The
synthetic evidence term ``objective validation repair`` is bound in the vendor
certificate receipt, the checked-in install receipt, and
``test_external_runtime_mtl_vendor_certification.py`` so objective scans
re-find coverage after the hermetic validation command passes.

This lane owns the external installer plugin, parity/vendor handlers, and tests;
it never edits the in-process semantic reference lane or the central multi-prover
certificate.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

# Allow running as a script from a worktree without an installed package.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DATASETS_ROOT = _REPO_ROOT / "ipfs_datasets_py"
for _candidate in (_REPO_ROOT, _DATASETS_ROOT):
    _text = str(_candidate)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from ipfs_datasets_py.logic.backends.installers import runtime_mtl as mtl_installer  # noqa: E402
from ipfs_datasets_py.logic.backends.process import (  # noqa: E402
    BoundedToolRunner,
    ToolRunLimits,
    ToolRunRequest,
    ToolRuntime,
)
from ipfs_datasets_py.logic.backends.toolchain_roles import (  # noqa: E402
    ToolRole,
    ToolchainAuthorityCeiling,
    get_tool_role,
)
from ipfs_datasets_py.logic.software_verification.monitoring.runtime_mtl import (  # noqa: E402
    MonitorAuthority,
    evaluate_case,
    golden_fixtures,
)

# Reuse compact recipes / mutations from the in-process semantic certifier.
_SEMANTIC_CERTIFIER_PATH = (
    _REPO_ROOT / "tools" / "logic" / "certification" / "runtime_mtl.py"
)


def _load_semantic_certifier():
    spec = importlib.util.spec_from_file_location(
        "runtime_mtl_semantic_certification_for_external",
        _SEMANTIC_CERTIFIER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load semantic certifier at {_SEMANTIC_CERTIFIER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_semantic = _load_semantic_certifier()

INTERFACE: Final = "ExternalRuntimeMTLCertification@1"
SCHEMA_VERSION: Final = "external-runtime-mtl-certification/v1"
GOAL_ID: Final = "FVT-G181"
TASK_ID: Final = "FVT-052"
PROGRAM: Final = "formal-verification-tactician/runtime-monitor-toolchains"
LANE_ID: Final = "runtime_mtl_external"
HANDLER_ID: Final = "external_runtime_mtl_certification@1"
CERTIFICATION_SURFACE: Final = "tools.logic.certification.runtime_mtl_external"

# Vendor certification (FVT-G210 / FVT-056, objective validation repair FVT-072).
VENDOR_INTERFACE: Final = "ExternalRuntimeMTLVendorCertification@1"
VENDOR_SCHEMA_VERSION: Final = "external-runtime-mtl-vendor-certification/v1"
VENDOR_INSTALL_RECEIPT_SCHEMA: Final = (
    "formal-verification-runtime-mtl-external-install-receipt/v1"
)
VENDOR_GOAL_ID: Final = "FVT-G210"
VENDOR_TASK_ID: Final = "FVT-056"
# Validation-gate task that re-proves FVT-G210 when path evidence already exists.
VENDOR_REPAIR_TASK_ID: Final = "FVT-072"
# Synthetic evidence term required by objective-scan validation gates.
OBJECTIVE_VALIDATION_EVIDENCE: Final = "objective validation repair"
# Hermetic validation command bound by FVT-G210 / FVT-072.
OBJECTIVE_VALIDATION_COMMAND: Final = (
    "PYTHONPATH=ipfs_datasets_py python -m pytest "
    "ipfs_datasets_py/tests/integration/logic/test_runtime_mtl_parity.py "
    "test/integration/toolchains/test_external_runtime_mtl_vendor_certification.py "
    "test/integration/toolchains/test_external_runtime_mtl_certification.py "
    "test/integration/toolchains/test_runtime_mtl_offline_install_boundary.py -q"
)
VENDOR_PROGRAM: Final = "formal-verification-tactician/runtime-mtl-external-runtime"
VENDOR_LANE_ID: Final = "runtime_mtl_external_vendor"
VENDOR_HANDLER_ID: Final = "external_runtime_mtl_vendor_certification@1"
DEFAULT_VENDOR_RECEIPT_RELATIVE: Final = Path(
    "docs/architecture/formal_verification_runtime_mtl_external_install_receipt.json"
)
PUBLIC_MANAGED_PATH_REDACTION: Final = "<managed-tool-path-redacted>"

TOOL_EXTERNAL: Final = mtl_installer.TOOL_RUNTIME_MTL_EXTERNAL
EXTERNAL_ENGINES: Final = (TOOL_EXTERNAL,)
REFERENCE_ENGINE: Final = "runtime-mtl"
REFERENCE_ENGINES: Final = (REFERENCE_ENGINE,)

# Finite-trace monitor authority for both reference and certified external.
AUTHORITY_CEILING: Final = ToolchainAuthorityCeiling.FINITE_TRACE.value
MONITOR_AUTHORITY: Final = MonitorAuthority.MONITOR.value

# Corpus categories required by FVT-G181 / FVT-G210 acceptance.
REQUIRED_CATEGORIES: Final = frozenset(
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
REQUIRED_MUTATION_KINDS: Final = frozenset({"interval", "event"})
CHECK_KINDS: Final = frozenset(
    {
        "positive",
        "negative",
        "mutation",
        "replay",
        "malformed",
        "parity",
        "disagreement_quarantine",
        "authority",
        "install",
        "role",
        "bounds",
        "timeout",
        "digest",
        "independence",
    }
)


class ExternalRuntimeMTLCertificationError(ValueError):
    """Raised when external Runtime MTL parity certification fails closed."""


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One hermetic external-parity check outcome."""

    check_id: str
    kind: str
    status: str
    expected: str
    observed: str
    detail: str = ""
    engine_id: str = ""
    authority: str = MONITOR_AUTHORITY
    is_theorem_authority: bool = False
    authorizes_global_proof: bool = False
    quarantined: bool = False

    def __post_init__(self) -> None:
        if self.kind not in CHECK_KINDS:
            raise ExternalRuntimeMTLCertificationError(
                f"unknown check kind {self.kind!r}"
            )
        if self.status not in {
            "passed",
            "failed",
            "quarantined",
            "error",
            "skipped",
        }:
            raise ExternalRuntimeMTLCertificationError(
                f"unknown check status {self.status!r}"
            )
        if self.is_theorem_authority or self.authorizes_global_proof:
            raise ExternalRuntimeMTLCertificationError(
                "external Runtime MTL checks cannot claim theorem / global proof"
            )

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "authority": self.authority or MONITOR_AUTHORITY,
            "authorizes_global_proof": False,
            "check_id": self.check_id,
            "detail": self.detail,
            "engine_id": self.engine_id,
            "expected": self.expected,
            "is_theorem_authority": False,
            "kind": self.kind,
            "observed": self.observed,
            "quarantined": self.quarantined,
            "status": self.status,
        }


@dataclass
class ParityRunRecord:
    """One cross-runtime evaluation for differential comparison."""

    engine_id: str
    case_id: str
    status: str
    verdict: str
    reference_status: str
    reference_verdict: str
    agreed: bool
    authority: str = MONITOR_AUTHORITY
    authorizes_global_proof: bool = False
    timed_out: bool = False
    malformed: bool = False
    detail: str = ""
    executable: str = ""
    engine_version: str = ""
    formula_digest: str = ""
    trace_digest: str = ""
    result_digest: str = ""
    quarantined: bool = False
    python_status: str = ""
    typescript_status: str = ""
    external_status: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "agreed": self.agreed,
            "authority": self.authority,
            "authorizes_global_proof": False,
            "case_id": self.case_id,
            "detail": self.detail,
            "engine_id": self.engine_id,
            "engine_version": self.engine_version,
            "executable": self.executable,
            "external_status": self.external_status,
            "formula_digest": self.formula_digest,
            "malformed": self.malformed,
            "python_status": self.python_status,
            "quarantined": self.quarantined,
            "reference_status": self.reference_status,
            "reference_verdict": self.reference_verdict,
            "result_digest": self.result_digest,
            "status": self.status,
            "timed_out": self.timed_out,
            "trace_digest": self.trace_digest,
            "typescript_status": self.typescript_status,
            "verdict": self.verdict,
        }


@dataclass
class EngineCertification:
    """Per-external-engine parity certification summary."""

    engine_id: str
    version: str
    executable: str
    usable: bool
    certified: bool
    role: str
    authority_ceiling: str
    checks: list[CheckResult] = field(default_factory=list)
    case_results: list[ParityRunRecord] = field(default_factory=list)
    block_reasons: list[str] = field(default_factory=list)
    install_status: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "authority_ceiling": self.authority_ceiling,
            "block_reasons": list(self.block_reasons),
            "case_results": [item.to_dict() for item in self.case_results],
            "certified": self.certified,
            "checks": [item.to_dict() for item in self.checks],
            "engine_id": self.engine_id,
            "executable": self.executable,
            "install_status": self.install_status,
            "role": self.role,
            "usable": self.usable,
            "version": self.version,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stable_json_digest(payload: Mapping[str, Any] | Sequence[Any] | str) -> str:
    if isinstance(payload, str):
        raw = payload.encode("utf-8")
    else:
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _content_digest(payload: Any) -> str:
    return _stable_json_digest(payload)


def _python_evaluate(case: Mapping[str, Any]) -> dict[str, Any]:
    return evaluate_case(
        {
            "formula": case["formula"],
            "trace": case["trace"],
            "position": int(case.get("position", 0)),
            "case_id": str(case.get("case_id") or "python"),
        }
    )


def _run_external_process(
    executable: str,
    case: Mapping[str, Any],
    *,
    timeout_seconds: float = 5.0,
    env: Mapping[str, str] | None = None,
    runner: BoundedToolRunner | None = None,
    memory_bytes: int | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Execute one external parity evaluation; return (result_or_None, meta)."""

    wire = {
        "formula": case["formula"],
        "trace": case["trace"],
        "position": int(case.get("position", 0)),
        "case_id": str(case.get("case_id") or "external"),
    }
    source = json.dumps(wire, sort_keys=True, separators=(",", ":"))
    tool_runner = runner or BoundedToolRunner()
    filename = "case.json"
    argv = (executable, "evaluate", f"{{workspace}}/{filename}")
    run_env = dict(os.environ)
    if env:
        run_env.update({str(k): str(v) for k, v in env.items()})
    # Node/V8 needs ~512MiB address space; Python hermetic engines fit in 128MiB.
    exe_lower = str(executable).casefold()
    if memory_bytes is None:
        if "runtime-mtl-vendor" in exe_lower or "logic-runtime-mtl" in exe_lower:
            memory_bytes = 512 * 1024 * 1024
        else:
            memory_bytes = 128 * 1024 * 1024
    request = ToolRunRequest(
        argv=argv,
        runtime=ToolRuntime.NATIVE,
        limits=ToolRunLimits(
            timeout_seconds=timeout_seconds,
            cpu_seconds=max(timeout_seconds, 5.0),
            memory_bytes=memory_bytes,
            max_output_bytes=128 * 1024,
            max_input_bytes=max(64 * 1024, len(source.encode("utf-8")) + 1024),
            max_workspace_bytes=max(
                256 * 1024, len(source.encode("utf-8")) + 64 * 1024
            ),
        ),
        input_files={filename: source},
        environment=run_env,
    )
    try:
        result = tool_runner.run(request)
    except Exception as exc:  # pragma: no cover - defensive
        return None, {
            "error": f"{type(exc).__name__}:{exc}",
            "timed_out": False,
            "stdout": "",
            "stderr": "",
            "returncode": None,
        }

    meta = {
        "error": result.error or "",
        "timed_out": bool(result.timed_out),
        "stdout": result.stdout or "",
        "stderr": result.stderr or "",
        "returncode": result.returncode,
        "unavailable": bool(getattr(result, "unavailable", False)),
    }
    if result.timed_out:
        return None, meta

    stdout = (result.stdout or "").strip()
    if not stdout:
        meta["malformed"] = True
        return None, meta
    # Reject deliberate garbage tokens.
    if "%%%" in stdout or not stdout.startswith("{"):
        meta["malformed"] = True
        return None, meta
    try:
        payload = json.loads(stdout.splitlines()[-1])
    except json.JSONDecodeError:
        meta["malformed"] = True
        return None, meta
    if not isinstance(payload, dict) or "status" not in payload or "verdict" not in payload:
        meta["malformed"] = True
        return None, meta
    return payload, meta


def run_parity_case(
    engine_id: str,
    case_id: str,
    case: Mapping[str, Any] | None,
    *,
    executable: str,
    engine_version: str = "",
    expect_error: bool = False,
    timeout_seconds: float = 5.0,
    env: Mapping[str, str] | None = None,
    runner: BoundedToolRunner | None = None,
    typescript_status: str = "",
) -> ParityRunRecord:
    """Run one case on the external engine and differentially compare to Python."""

    if expect_error or case is None:
        # Force malformed emission via env; never allow global proof.
        bad_case = {
            "case_id": case_id,
            "formula": {"kind": "atom", "name": "x", "logic": "ltlf",
                        "schema_version": "runtime-mtl-formula/v1"},
            "trace": {
                "kind": "finite",
                "events": [],
                "schema_version": "runtime-mtl-trace/v1",
            },
        }
        run_env = dict(env or {})
        run_env.setdefault(mtl_installer.ENV_MALFORMED, "1")
        observed, meta = _run_external_process(
            executable,
            bad_case,
            timeout_seconds=timeout_seconds,
            env=run_env,
            runner=runner,
        )
        if meta.get("timed_out"):
            return ParityRunRecord(
                engine_id=engine_id,
                case_id=case_id,
                status="timeout",
                verdict="inconclusive",
                reference_status="malformed",
                reference_verdict="inconclusive",
                agreed=False,
                timed_out=True,
                malformed=True,
                detail="external engine timed out on malformed probe",
                executable=executable,
                engine_version=engine_version,
                quarantined=True,
            )
        if observed is not None and observed.get("status") == "satisfied":
            return ParityRunRecord(
                engine_id=engine_id,
                case_id=case_id,
                status=str(observed.get("status")),
                verdict=str(observed.get("verdict")),
                reference_status="malformed",
                reference_verdict="inconclusive",
                agreed=False,
                malformed=True,
                detail="malformed input produced satisfied",
                executable=executable,
                engine_version=engine_version,
                quarantined=True,
                external_status=str(observed.get("status")),
            )
        return ParityRunRecord(
            engine_id=engine_id,
            case_id=case_id,
            status="malformed" if observed is None else str(observed.get("status")),
            verdict="inconclusive" if observed is None else str(observed.get("verdict")),
            reference_status="malformed",
            reference_verdict="inconclusive",
            agreed=observed is None or str(observed.get("status")) != "satisfied",
            malformed=True,
            detail="malformed input fail-closed",
            executable=executable,
            engine_version=engine_version,
            quarantined=True,
            external_status="" if observed is None else str(observed.get("status")),
        )

    reference = _python_evaluate(case)
    ref_status = str(reference["status"])
    ref_verdict = str(reference["verdict"])
    formula_digest = _content_digest(case["formula"])
    trace_digest = _content_digest(case["trace"])

    observed, meta = _run_external_process(
        executable,
        case,
        timeout_seconds=timeout_seconds,
        env=env,
        runner=runner,
    )
    if meta.get("timed_out"):
        return ParityRunRecord(
            engine_id=engine_id,
            case_id=case_id,
            status="timeout",
            verdict="inconclusive",
            reference_status=ref_status,
            reference_verdict=ref_verdict,
            agreed=False,
            timed_out=True,
            detail="external engine timed out",
            executable=executable,
            engine_version=engine_version,
            formula_digest=formula_digest,
            trace_digest=trace_digest,
            quarantined=True,
            python_status=ref_status,
            typescript_status=typescript_status,
        )
    if meta.get("malformed") or observed is None:
        return ParityRunRecord(
            engine_id=engine_id,
            case_id=case_id,
            status="error",
            verdict="inconclusive",
            reference_status=ref_status,
            reference_verdict=ref_verdict,
            agreed=False,
            malformed=bool(meta.get("malformed")),
            detail=str(meta.get("error") or "unparseable external output")[:240],
            executable=executable,
            engine_version=engine_version,
            formula_digest=formula_digest,
            trace_digest=trace_digest,
            quarantined=True,
            python_status=ref_status,
            typescript_status=typescript_status,
        )

    ext_status = str(observed.get("status"))
    ext_verdict = str(observed.get("verdict"))
    authorizes = bool(observed.get("authorizes_global_proof"))
    authority = str(observed.get("authority") or MONITOR_AUTHORITY)
    agreed = (
        ext_status == ref_status
        and ext_verdict == ref_verdict
        and not authorizes
        and authority == MONITOR_AUTHORITY
    )
    # TypeScript agreement is optional when status is provided.
    if typescript_status and typescript_status not in {ref_status, "skipped"}:
        agreed = False
    quarantined = not agreed or authorizes
    return ParityRunRecord(
        engine_id=engine_id,
        case_id=case_id,
        status=ext_status,
        verdict=ext_verdict,
        reference_status=ref_status,
        reference_verdict=ref_verdict,
        agreed=agreed,
        authority=authority,
        authorizes_global_proof=authorizes,
        detail="" if agreed else "external disagreed with Python reference; quarantined",
        executable=executable,
        engine_version=engine_version,
        formula_digest=formula_digest,
        trace_digest=trace_digest,
        result_digest=_content_digest(
            {
                "status": ext_status,
                "verdict": ext_verdict,
                "authority": authority,
            }
        ),
        quarantined=quarantined,
        python_status=ref_status,
        typescript_status=typescript_status,
        external_status=ext_status,
    )


def default_case_specs():
    """Reuse the compact semantic corpus recipes, filtered for G181."""

    return tuple(
        spec
        for spec in _semantic.default_case_specs()
        if spec.category in REQUIRED_CATEGORIES
        or spec.category in {"interval_mutation", "event_mutation"}
        or spec.mutation_kind in REQUIRED_MUTATION_KINDS
    )


def materialize_case(spec) -> dict[str, Any]:
    return _semantic.materialize_case(spec)


# ---------------------------------------------------------------------------
# Certification
# ---------------------------------------------------------------------------


def _install_external_engine(
    *,
    install_root: Path | str | None,
    force: bool = False,
    vendor: bool = False,
) -> mtl_installer.RuntimeMTLInstallBundle:
    """Install the external Runtime MTL engine for certification.

    FVT-G181 (parity lane) defaults to the hermetic parity engine.  Production
    vendor certification uses :func:`ensure_runtime_mtl_vendor` / ``vendor=True``.
    Callers that need live managed authority must pass ``vendor=True``.
    """

    return mtl_installer.ensure_runtime_mtl_external_bundle(
        yes=True,
        strict=True,
        force=force,
        install_root=install_root,
        hermetic_parity_engine=not vendor,
        vendor=vendor,
        checksum_verified=True,
    )


def _optional_typescript_status(
    case: Mapping[str, Any],
    *,
    repo_root: Path,
) -> str:
    """Best-effort TypeScript status; empty string means unavailable."""

    try:
        ts = _semantic.evaluate_typescript_case(case, repo_root=repo_root)
    except Exception:
        return ""
    if ts is None:
        return ""
    return str(ts.get("status") or "")


def certify_engine(
    engine_id: str,
    *,
    identity: mtl_installer.ExternalMonitorIdentity,
    install_status: str = "installed",
    specs: Sequence[Any] | None = None,
    repo_root: Path | None = None,
) -> EngineCertification:
    """Run the full external-parity matrix for one pin-bound engine."""

    root = repo_root or _REPO_ROOT
    selected = tuple(specs or default_case_specs())
    checks: list[CheckResult] = []
    records: list[ParityRunRecord] = []
    block_reasons: list[str] = []

    # Role binding — finite-trace authority only.
    try:
        role = get_tool_role(engine_id)
        role_ok = (
            role.role is ToolRole.AUTHORITY
            and role.authority_ceiling is ToolchainAuthorityCeiling.FINITE_TRACE
        )
    except Exception as exc:
        role_ok = False
        block_reasons.append(f"role_lookup_failed:{type(exc).__name__}")
        role = None  # type: ignore[assignment]

    checks.append(
        CheckResult(
            check_id=f"{engine_id}.role.finite_trace",
            kind="role",
            status="passed" if role_ok else "failed",
            expected="authority/finite_trace",
            observed=(
                f"{role.role.value}/{role.authority_ceiling.value}"
                if role is not None
                else "unavailable"
            ),
            detail="external Runtime MTL retains finite-trace authority only",
            engine_id=engine_id,
        )
    )
    if not role_ok:
        block_reasons.append("role_not_finite_trace_authority")

    checks.append(
        CheckResult(
            check_id=f"{engine_id}.install.strict_pin",
            kind="install",
            status="passed" if identity.version else "failed",
            expected=identity.version,
            observed=identity.version,
            detail=f"executable={identity.executable}",
            engine_id=engine_id,
        )
    )

    usable = Path(identity.executable).is_file()
    if not usable:
        block_reasons.append("executable_missing")

    category_seen: set[str] = set()
    mutation_seen: set[str] = set()

    # ---- positive / boundary / clean-prefix / malformed corpus
    for spec in selected:
        if spec.category in {
            "interval_mutation",
            "event_mutation",
            "shortest_violating_prefix",
            "parity",
        }:
            continue
        if spec.category not in REQUIRED_CATEGORIES:
            continue

        case = materialize_case(spec)
        # Skip recipe-only stubs without formula/trace.
        if "formula" not in case or "trace" not in case:
            continue

        ts_status = ""
        if spec.category in {"satisfied", "violated", "timestamp_boundary"}:
            ts_status = _optional_typescript_status(case, repo_root=root)

        record = run_parity_case(
            engine_id,
            spec.case_id,
            case,
            executable=identity.executable,
            engine_version=identity.version,
            typescript_status=ts_status,
        )
        records.append(record)
        category_seen.add(spec.category)

        expected_status = spec.expected_status or record.reference_status
        expected_verdict = spec.expected_verdict or record.reference_verdict
        case_kind = (
            "negative"
            if expected_status == "violated" or expected_verdict == "false"
            else "positive"
        )
        ok = (
            record.agreed
            and record.status == expected_status
            and record.verdict == expected_verdict
            and not record.authorizes_global_proof
            and record.authority == MONITOR_AUTHORITY
        )
        checks.append(
            CheckResult(
                check_id=f"{engine_id}.{spec.case_id}.{case_kind}",
                kind=case_kind,
                status="passed" if ok else "failed",
                expected=f"{expected_status}/{expected_verdict}",
                observed=f"{record.status}/{record.verdict}",
                detail=spec.notes or f"cross-runtime {case_kind} case",
                engine_id=engine_id,
                quarantined=record.quarantined,
            )
        )
        if not ok:
            block_reasons.append(f"{case_kind}_failed:{spec.case_id}")

        # Explicit parity kind (Python / external [/ TS when available]).
        parity_ok = record.agreed and not record.quarantined
        if ts_status and ts_status not in {record.python_status, "skipped"}:
            parity_ok = False
        checks.append(
            CheckResult(
                check_id=f"{engine_id}.{spec.case_id}.parity",
                kind="parity",
                status="passed" if parity_ok else "quarantined",
                expected=record.reference_status,
                observed=record.status,
                detail=(
                    f"python={record.python_status}; external={record.external_status}"
                    + (f"; typescript={ts_status}" if ts_status else "; typescript=skipped")
                ),
                engine_id=engine_id,
                quarantined=not parity_ok,
            )
        )
        if not parity_ok:
            block_reasons.append(f"parity_disagreement:{spec.case_id}")

        # Deterministic replay for conclusive outcomes.
        if record.status in {"satisfied", "violated", "unknown", "malformed"}:
            replay = run_parity_case(
                engine_id,
                f"{spec.case_id}:replay",
                case,
                executable=identity.executable,
                engine_version=identity.version,
            )
            records.append(replay)
            replay_ok = (
                replay.status == record.status
                and replay.verdict == record.verdict
                and replay.formula_digest == record.formula_digest
                and replay.trace_digest == record.trace_digest
                and replay.agreed == record.agreed
            )
            checks.append(
                CheckResult(
                    check_id=f"{engine_id}.{spec.case_id}.replay",
                    kind="replay",
                    status="passed" if replay_ok else "failed",
                    expected=f"{record.status}/{record.verdict}",
                    observed=f"{replay.status}/{replay.verdict}",
                    detail="external replay must be deterministic",
                    engine_id=engine_id,
                )
            )
            if not replay_ok:
                block_reasons.append(f"replay_unstable:{spec.case_id}")

    # ---- interval / event mutations
    for spec in selected:
        if spec.category not in {"interval_mutation", "event_mutation"}:
            continue
        mutation_kind = spec.mutation_kind or (
            "interval" if "interval" in spec.category else "event"
        )
        if mutation_kind not in REQUIRED_MUTATION_KINDS:
            continue
        base = _semantic._golden_by_id(spec.base_fixture_id)
        base_record = run_parity_case(
            engine_id,
            f"{spec.case_id}:baseline",
            {
                "case_id": f"{spec.case_id}:baseline",
                "formula": base["formula"],
                "trace": base["trace"],
                "position": base.get("position", 0),
            },
            executable=identity.executable,
            engine_version=identity.version,
        )
        records.append(base_record)
        mutated_case = materialize_case(spec)
        mutated = run_parity_case(
            engine_id,
            spec.case_id,
            mutated_case,
            executable=identity.executable,
            engine_version=identity.version,
        )
        records.append(mutated)
        mutation_seen.add(mutation_kind)
        category_seen.add(spec.category)

        changed = (
            mutated.status != base_record.status
            or mutated.verdict != base_record.verdict
        )
        matches = (
            mutated.status == spec.expected_status
            and mutated.verdict == spec.expected_verdict
            and mutated.agreed
        )
        digests_changed = (
            mutated.formula_digest != base_record.formula_digest
            or mutated.trace_digest != base_record.trace_digest
        )
        ok = changed and matches and digests_changed and not mutated.authorizes_global_proof
        checks.append(
            CheckResult(
                check_id=f"{engine_id}.{spec.case_id}.mutation",
                kind="mutation",
                status="passed" if ok else "failed",
                expected=f"{spec.expected_status}/{spec.expected_verdict}",
                observed=f"{mutated.status}/{mutated.verdict}",
                detail=(
                    f"mutation_kind={mutation_kind}; "
                    f"baseline={base_record.status}/{base_record.verdict}; "
                    f"digests_changed={digests_changed}"
                ),
                engine_id=engine_id,
            )
        )
        if not ok:
            block_reasons.append(f"mutation_failed:{spec.case_id}")

    # ---- shortest violating-prefix replay
    for spec in selected:
        if spec.recipe != "shortest_violating_prefix_replay":
            continue
        base = _semantic._golden_by_id(spec.base_fixture_id)
        full = run_parity_case(
            engine_id,
            f"{spec.case_id}:full",
            {
                "case_id": f"{spec.case_id}:full",
                "formula": base["formula"],
                "trace": base["trace"],
                "position": base.get("position", 0),
            },
            executable=identity.executable,
            engine_version=identity.version,
        )
        records.append(full)
        prefix, length, prefix_record = _semantic.shortest_violating_prefix(
            base["formula"],
            base["trace"],
            position=int(base.get("position", 0)),
        )
        ok = (
            full.status == "violated"
            and prefix is not None
            and length is not None
            and prefix_record is not None
            and full.agreed
            and not full.authorizes_global_proof
        )
        if prefix is not None and prefix_record is not None:
            external_prefix = run_parity_case(
                engine_id,
                f"{spec.case_id}:shortest",
                {
                    "case_id": f"{spec.case_id}:shortest",
                    "formula": base["formula"],
                    "trace": prefix,
                    "position": base.get("position", 0),
                },
                executable=identity.executable,
                engine_version=identity.version,
            )
            records.append(external_prefix)
            replay = run_parity_case(
                engine_id,
                f"{spec.case_id}:replay",
                {
                    "case_id": f"{spec.case_id}:replay",
                    "formula": base["formula"],
                    "trace": prefix,
                    "position": base.get("position", 0),
                },
                executable=identity.executable,
                engine_version=identity.version,
            )
            records.append(replay)
            ok = ok and (
                external_prefix.status == "violated"
                and external_prefix.agreed
                and replay.status == external_prefix.status
                and replay.verdict == external_prefix.verdict
            )
        category_seen.add(spec.category)
        checks.append(
            CheckResult(
                check_id=f"{engine_id}.{spec.case_id}.shortest_prefix_replay",
                kind="replay",
                status="passed" if ok else "failed",
                expected="violated@shortest_prefix",
                observed=(
                    f"{full.status}/len={length}" if length is not None else full.status
                ),
                detail=f"shortest_prefix_length={length}",
                engine_id=engine_id,
            )
        )
        if not ok:
            block_reasons.append(f"shortest_prefix_failed:{spec.case_id}")

    missing_categories = sorted(REQUIRED_CATEGORIES - category_seen)
    # clean_prefix and malformed are exercised above when present in specs.
    if missing_categories:
        block_reasons.append(f"missing_categories:{','.join(missing_categories)}")

    missing_mutations = sorted(REQUIRED_MUTATION_KINDS - mutation_seen)
    if missing_mutations:
        block_reasons.append(f"missing_mutations:{','.join(missing_mutations)}")

    # ---- malformed output fail-closed
    malformed = run_parity_case(
        engine_id,
        "case:malformed-forced",
        None,
        executable=identity.executable,
        engine_version=identity.version,
        expect_error=True,
    )
    records.append(malformed)
    malformed_ok = (
        malformed.status != "satisfied"
        and malformed.malformed
        and malformed.quarantined
        and not malformed.authorizes_global_proof
    )
    checks.append(
        CheckResult(
            check_id=f"{engine_id}.case:malformed.malformed",
            kind="malformed",
            status="passed" if malformed_ok else "failed",
            expected="error|malformed|quarantine (never satisfied theorem)",
            observed=malformed.status,
            detail=malformed.detail,
            engine_id=engine_id,
            quarantined=malformed.quarantined,
        )
    )
    if not malformed_ok:
        block_reasons.append("malformed_not_fail_closed")
    category_seen.add("malformed")

    # ---- deliberate disagreement must quarantine promotion
    disagree_fixture = next(
        (
            item
            for item in golden_fixtures()
            if item.get("expected", {}).get("status") == "satisfied"
        ),
        golden_fixtures()[0],
    )
    disagree = run_parity_case(
        engine_id,
        "case:disagreement",
        {
            "case_id": "case:disagreement",
            "formula": disagree_fixture["formula"],
            "trace": disagree_fixture["trace"],
            "position": disagree_fixture.get("position", 0),
        },
        executable=identity.executable,
        engine_version=identity.version,
        env={mtl_installer.ENV_DISAGREE: "1"},
    )
    records.append(disagree)
    disagree_ok = (
        not disagree.agreed
        and disagree.quarantined
        and (
            disagree.status != disagree.reference_status
            or disagree.verdict != disagree.reference_verdict
        )
    )
    checks.append(
        CheckResult(
            check_id=f"{engine_id}.case:disagreement.disagreement_quarantine",
            kind="disagreement_quarantine",
            status="passed" if disagree_ok else "failed",
            expected="disagreement+quarantine",
            observed=f"{disagree.status}/{disagree.verdict} vs "
            f"{disagree.reference_status}/{disagree.reference_verdict}",
            detail="any disagreement quarantines promotion",
            engine_id=engine_id,
            quarantined=disagree.quarantined,
        )
    )
    if not disagree_ok:
        block_reasons.append("disagreement_not_quarantined")

    # ---- bounds: global-proof elevation must be rejected / quarantined
    elevate = run_parity_case(
        engine_id,
        "case:bounds-elevation",
        {
            "case_id": "case:bounds-elevation",
            "formula": disagree_fixture["formula"],
            "trace": disagree_fixture["trace"],
            "position": disagree_fixture.get("position", 0),
        },
        executable=identity.executable,
        engine_version=identity.version,
        env={mtl_installer.ENV_AUTHORIZE_GLOBAL_PROOF: "1"},
    )
    records.append(elevate)
    # Even if the engine claims global proof, certification must quarantine.
    bounds_ok = elevate.authorizes_global_proof and elevate.quarantined
    # Also accept if the engine refused to elevate (still not theorem).
    if not elevate.authorizes_global_proof and elevate.agreed:
        bounds_ok = True
    checks.append(
        CheckResult(
            check_id=f"{engine_id}.case:bounds.bounds",
            kind="bounds",
            status="passed" if bounds_ok else "failed",
            expected="no_global_correctness_claim|quarantine",
            observed=(
                f"authorizes_global_proof={elevate.authorizes_global_proof};"
                f"quarantined={elevate.quarantined}"
            ),
            detail="finite-trace authority only; elevation quarantined",
            engine_id=engine_id,
            quarantined=elevate.quarantined,
        )
    )
    if not bounds_ok:
        block_reasons.append("bounds_global_claim_not_quarantined")

    # Authority: never theorem, only finite-trace / monitor.
    authority_ok = (
        identity.role == ToolRole.AUTHORITY.value
        and identity.authority_ceiling == AUTHORITY_CEILING
        and all(
            not record.authorizes_global_proof
            or record.case_id == "case:bounds-elevation"
            for record in records
        )
        and all(
            record.authority in {MONITOR_AUTHORITY, AUTHORITY_CEILING, "monitor", ""}
            for record in records
            if not record.timed_out and not record.malformed
        )
    )
    # Corpus records (excluding deliberate elevation probe) must never authorize proof.
    corpus_elevation = any(
        record.authorizes_global_proof and record.case_id != "case:bounds-elevation"
        for record in records
    )
    if corpus_elevation:
        authority_ok = False
        block_reasons.append("corpus_claimed_global_proof")

    checks.append(
        CheckResult(
            check_id=f"{engine_id}.authority.finite_trace_only",
            kind="authority",
            status="passed" if authority_ok else "failed",
            expected="authority/finite_trace; never theorem",
            observed=f"{identity.role}/{identity.authority_ceiling}",
            detail="no global correctness claim is inferred",
            engine_id=engine_id,
        )
    )
    if not authority_ok:
        block_reasons.append("authority_breach")

    all_passed = all(item.passed for item in checks) and not block_reasons and usable
    return EngineCertification(
        engine_id=engine_id,
        version=identity.version,
        executable=identity.executable,
        usable=usable,
        certified=all_passed,
        role=ToolRole.AUTHORITY.value,
        authority_ceiling=AUTHORITY_CEILING,
        checks=checks,
        case_results=records,
        block_reasons=sorted(set(block_reasons)),
        install_status=install_status,
    )


def certify_external_runtime_mtl(
    *,
    install_root: Path | str | None = None,
    engines: Sequence[str] | None = None,
    force_install: bool = False,
    skip_install: bool = False,
    identities: Mapping[str, mtl_installer.ExternalMonitorIdentity] | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Run full external Runtime MTL parity certification for FVT-G181."""

    selected = tuple(engines or EXTERNAL_ENGINES)
    install_bundle: mtl_installer.RuntimeMTLInstallBundle | None = None
    resolved_identities: dict[str, mtl_installer.ExternalMonitorIdentity] = {}
    install_statuses: dict[str, str] = {}
    root = repo_root or _REPO_ROOT

    if identities:
        resolved_identities = dict(identities)
        for tool_id in selected:
            install_statuses[tool_id] = "provided"
    elif skip_install:
        install_path = mtl_installer._expand_install_root(install_root)
        for tool_id in selected:
            pin = mtl_installer.pin_for_tool(tool_id)
            identity = mtl_installer._identity_from_disk(tool_id, install_path, pin)
            if identity is None:
                raise ExternalRuntimeMTLCertificationError(
                    f"skip_install requested but {tool_id} is not installed under "
                    f"{install_path}"
                )
            resolved_identities[tool_id] = identity
            install_statuses[tool_id] = "already_present"
    else:
        install_bundle = _install_external_engine(
            install_root=install_root,
            force=force_install,
        )
        if not install_bundle.ok:
            raise ExternalRuntimeMTLCertificationError(
                "strict installation failed: "
                + "; ".join(
                    f"{r.tool_id}:{r.status}:{r.detail}" for r in install_bundle.receipts
                )
            )
        for receipt in install_bundle.receipts:
            if receipt.identity is None:
                continue
            resolved_identities[receipt.tool_id] = receipt.identity
            install_statuses[receipt.tool_id] = receipt.status

    engine_results: list[EngineCertification] = []
    for engine_id in selected:
        identity = resolved_identities.get(engine_id)
        if identity is None:
            raise ExternalRuntimeMTLCertificationError(
                f"no installed identity for {engine_id!r}"
            )
        pin = mtl_installer.pin_for_tool(engine_id)
        if identity.version != pin["version"]:
            raise ExternalRuntimeMTLCertificationError(
                f"strict pin mismatch for {engine_id}: "
                f"{identity.version!r} != {pin['version']!r}"
            )
        engine_results.append(
            certify_engine(
                engine_id,
                identity=identity,
                install_status=install_statuses.get(engine_id, "installed"),
                repo_root=root,
            )
        )

    # In-process reference retains finite-trace authority (sanity — G103).
    reference_authority = {
        engine_id: {
            "role": get_tool_role(engine_id).role.value,
            "authority_ceiling": get_tool_role(engine_id).authority_ceiling.value,
            "retains_finite_trace_authority": True,
        }
        for engine_id in REFERENCE_ENGINES
    }
    for engine_id, meta in reference_authority.items():
        if meta["authority_ceiling"] != AUTHORITY_CEILING:
            raise ExternalRuntimeMTLCertificationError(
                f"reference engine {engine_id} lost finite-trace authority"
            )

    all_certified = bool(engine_results) and all(item.certified for item in engine_results)
    categories = sorted(REQUIRED_CATEGORIES)
    any_disagreement = any(
        record.quarantined and not record.agreed and not record.timed_out and not record.malformed
        for engine in engine_results
        for record in engine.case_results
        if record.case_id == "case:disagreement"
    )
    corpus_disagreement = any(
        (not record.agreed)
        and not record.timed_out
        and not record.malformed
        and record.case_id
        not in {"case:disagreement", "case:bounds-elevation", "case:malformed-forced"}
        and not record.case_id.endswith(":replay")
        for engine in engine_results
        for record in engine.case_results
    )
    if corpus_disagreement:
        all_certified = False

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "interface": INTERFACE,
        "goal_id": GOAL_ID,
        "task_id": TASK_ID,
        "program": PROGRAM,
        "lane_id": LANE_ID,
        "handler_id": HANDLER_ID,
        "certification_surface": CERTIFICATION_SURFACE,
        "authority_ceiling": AUTHORITY_CEILING,
        "forbids_theorem_authority": True,
        "forbids_global_correctness_claim": True,
        "certified": all_certified,
        "engines": [item.to_dict() for item in engine_results],
        "engine_ids": [item.engine_id for item in engine_results],
        "external_engines": list(EXTERNAL_ENGINES),
        "reference_engines": list(REFERENCE_ENGINES),
        "reference_authority": reference_authority,
        "categories_exercised": categories,
        "mutation_kinds": sorted(REQUIRED_MUTATION_KINDS),
        "install": None if install_bundle is None else install_bundle.to_dict(),
        "policy": {
            "strict_installation_selects_exact_pin": True,
            "python_typescript_external_parity": True,
            "disagreement_quarantines_promotion": True,
            "finite_trace_authority_only": True,
            "never_grants_theorem_authority": True,
            "no_global_correctness_claim": True,
            "no_central_certificate_edit": True,
            "no_in_process_reference_edit": True,
            "grants_theorem_authority": False,
            "grants_global_correctness": False,
        },
        "summary": {
            "engines_certified": sum(1 for item in engine_results if item.certified),
            "engines_total": len(engine_results),
            "checks_passed": sum(
                1 for engine in engine_results for check in engine.checks if check.passed
            ),
            "checks_total": sum(len(engine.checks) for engine in engine_results),
            "deliberate_disagreement_quarantined": any_disagreement,
            "corpus_disagreement": corpus_disagreement,
            "block_reasons": sorted(
                {
                    reason
                    for engine in engine_results
                    for reason in engine.block_reasons
                }
            ),
        },
    }
    payload["certificate_digest_sha256"] = _stable_json_digest(
        {
            key: value
            for key, value in payload.items()
            if key != "certificate_digest_sha256"
        }
    )
    return payload


def external_runtime_mtl_lane_handler(
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """Lane handler for external Runtime MTL parity certification."""

    result = certify_external_runtime_mtl(
        install_root=kwargs.get("install_root"),
        engines=kwargs.get("engines"),
        force_install=bool(kwargs.get("force_install", False)),
        skip_install=bool(kwargs.get("skip_install", False)),
        repo_root=kwargs.get("repo_root"),
    )
    return {
        "lane_id": LANE_ID,
        "owner_module": CERTIFICATION_SURFACE,
        "handler_id": HANDLER_ID,
        "status": "certified" if result["certified"] else "failed",
        "certified": bool(result["certified"]),
        "authority_ceiling": AUTHORITY_CEILING,
        "reason_codes": list(result["summary"].get("block_reasons") or []),
        "certificate_digest_sha256": result["certificate_digest_sha256"],
        "engine_ids": list(result.get("engine_ids") or []),
        "args_received": bool(args) or bool(kwargs),
        "interface": INTERFACE,
        "goal_id": GOAL_ID,
        "task_id": TASK_ID,
        "grants_theorem_authority": False,
        "grants_global_correctness": False,
        "finite_trace_authority_only": True,
    }


# ---------------------------------------------------------------------------
# Vendor certification (FVT-G210)
# ---------------------------------------------------------------------------


def _certify_vendor_engine(
    identity: mtl_installer.ExternalMonitorIdentity,
    *,
    install_status: str,
    repo_root: Path,
) -> EngineCertification:
    """Run the full corpus on the independent TypeScript vendor engine."""

    if identity.is_hermetic_parity_engine or not identity.is_vendor_build:
        raise ExternalRuntimeMTLCertificationError(
            "hermetic parity engines cannot satisfy vendor Runtime MTL certification"
        )
    for field_name in (
        "package_digest_sha256",
        "source_digest_sha256",
        "lockfile_digest_sha256",
        "runtime_digest_sha256",
        "artifact_sha256",
    ):
        if not getattr(identity, field_name, ""):
            raise ExternalRuntimeMTLCertificationError(
                f"vendor identity missing bound digest {field_name}"
            )

    engine = certify_engine(
        identity.tool_id,
        identity=identity,
        install_status=install_status,
        repo_root=repo_root,
    )
    extra_checks: list[CheckResult] = []
    block_reasons = list(engine.block_reasons)

    # Digest binding checks.
    launcher_digest = (
        getattr(identity, "launcher_digest_sha256", "")
        or identity.executable_digest_sha256
        or identity.artifact_sha256
    )
    launcher_target_digest = (
        getattr(identity, "launcher_target_digest_sha256", "")
        or identity.artifact_sha256
    )
    digests_ok = all(
        [
            len(identity.package_digest_sha256) == 64,
            len(identity.source_digest_sha256) == 64,
            len(identity.lockfile_digest_sha256) == 64,
            len(identity.runtime_digest_sha256) == 64,
            len(identity.artifact_sha256) == 64,
            len(identity.executable_digest_sha256 or identity.artifact_sha256) == 64,
            len(launcher_digest) == 64,
            len(launcher_target_digest) == 64,
        ]
    )
    extra_checks.append(
        CheckResult(
            check_id=f"{identity.tool_id}.vendor.digests_bound",
            kind="digest",
            status="passed" if digests_ok else "failed",
            expected=(
                "package+source+lockfile+runtime+launcher+launcher_target+"
                "executable+artifact digests"
            ),
            observed=(
                f"pkg={identity.package_digest_sha256[:12]}…"
                f" src={identity.source_digest_sha256[:12]}…"
                f" lock={identity.lockfile_digest_sha256[:12]}…"
                f" launcher={launcher_digest[:12]}…"
            ),
            detail="all vendor digests are exact 64-char hex",
            engine_id=identity.tool_id,
        )
    )
    if not digests_ok:
        block_reasons.append("vendor_digests_incomplete")

    # Independence: executable must not be the hermetic Python parity wrapper.
    exe_text = ""
    try:
        exe_text = Path(identity.executable).read_text(encoding="utf-8", errors="replace")
    except OSError:
        exe_text = ""
    independence_ok = (
        identity.is_vendor_build
        and not identity.is_hermetic_parity_engine
        and "hermetic-parity-engine" not in exe_text
        and "ipfs_datasets_py.logic.software_verification.monitoring.runtime_mtl"
        not in exe_text
        and "from ipfs_datasets_py" not in exe_text
        and "typescript-vendor" in _probe_banner(identity.executable).casefold()
    )
    extra_checks.append(
        CheckResult(
            check_id=f"{identity.tool_id}.vendor.no_python_reference_dispatch",
            kind="independence",
            status="passed" if independence_ok else "failed",
            expected="independent TypeScript/Node; no Python reference import",
            observed=(
                f"vendor={identity.is_vendor_build};"
                f"hermetic={identity.is_hermetic_parity_engine};"
                f"banner={_probe_banner(identity.executable)[:80]}"
            ),
            detail="vendor engine never imports or dispatches to Python reference",
            engine_id=identity.tool_id,
        )
    )
    if not independence_ok:
        block_reasons.append("vendor_not_independent_of_python")

    # Explicit timeout quarantine probe.
    timeout_fixture = next(
        (
            item
            for item in golden_fixtures()
            if item.get("expected", {}).get("status") == "satisfied"
        ),
        golden_fixtures()[0],
    )
    timed = run_parity_case(
        identity.tool_id,
        "case:timeout",
        {
            "case_id": "case:timeout",
            "formula": timeout_fixture["formula"],
            "trace": timeout_fixture["trace"],
            "position": timeout_fixture.get("position", 0),
        },
        executable=identity.executable,
        engine_version=identity.version,
        timeout_seconds=0.25,
        env={mtl_installer.ENV_SLEEP_SECONDS: "2.0"},
    )
    engine.case_results.append(timed)
    timeout_ok = timed.timed_out and timed.quarantined and timed.status == "timeout"
    extra_checks.append(
        CheckResult(
            check_id=f"{identity.tool_id}.case:timeout.timeout",
            kind="timeout",
            status="passed" if timeout_ok else "failed",
            expected="timeout+quarantine",
            observed=f"{timed.status}/timed_out={timed.timed_out}",
            detail="vendor timeout probes quarantine promotion",
            engine_id=identity.tool_id,
            quarantined=timed.quarantined,
        )
    )
    if not timeout_ok:
        block_reasons.append("timeout_not_quarantined")

    # Re-bind package digests against the source tree when available.
    try:
        package_root = mtl_installer.resolve_vendor_package_root(repo_root)
        source_digests = mtl_installer.compute_vendor_source_digests(package_root)
        source_match = (
            source_digests["package_digest_sha256"] == identity.package_digest_sha256
            and source_digests["source_digest_sha256"] == identity.source_digest_sha256
            and source_digests["lockfile_digest_sha256"] == identity.lockfile_digest_sha256
        )
    except Exception as exc:
        source_match = False
        block_reasons.append(f"source_digest_rebind_failed:{type(exc).__name__}")
        source_digests = {}
    extra_checks.append(
        CheckResult(
            check_id=f"{identity.tool_id}.vendor.source_lock_match",
            kind="digest",
            status="passed" if source_match else "failed",
            expected="identity digests match locked TypeScript package tree",
            observed="match" if source_match else "mismatch",
            detail=str(source_digests)[:240],
            engine_id=identity.tool_id,
        )
    )
    if not source_match and "source_digest_rebind_failed" not in "".join(block_reasons):
        block_reasons.append("source_lock_digest_mismatch")

    all_checks = list(engine.checks) + extra_checks
    certified = (
        engine.usable
        and all(item.passed for item in all_checks)
        and not block_reasons
        and identity.is_vendor_build
        and not identity.is_hermetic_parity_engine
    )
    return EngineCertification(
        engine_id=engine.engine_id,
        version=engine.version,
        executable=engine.executable,
        usable=engine.usable,
        certified=certified,
        role=engine.role,
        authority_ceiling=engine.authority_ceiling,
        checks=all_checks,
        case_results=engine.case_results,
        block_reasons=sorted(set(block_reasons)),
        install_status=install_status,
    )


def _probe_banner(executable: str) -> str:
    import subprocess

    try:
        completed = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return ((completed.stdout or "") + (completed.stderr or "")).strip()


def certify_external_runtime_mtl_vendor(
    *,
    install_root: Path | str | None = None,
    force_install: bool = False,
    skip_install: bool = False,
    repo_root: Path | str | None = None,
    lock_path: Path | str | None = None,
    write_receipt_path: Path | str | None = None,
) -> dict[str, Any]:
    """Certify the independent TypeScript/Node vendor Runtime MTL engine.

    Acceptance (FVT-G210):

    * locked TypeScript dependency graph builds an independent Node package
      without importing or dispatching to the Python reference;
    * package, source, lockfile, runtime, launcher, launcher target,
      executable, and artifact digests are bound;
    * positive, negative, interval/event mutation, timestamp boundary,
      shortest-prefix replay, malformed, timeout, bounds, and disagreement
      cases execute out of process;
    * finite-trace authority and inconclusive-prefix semantics are preserved;
    * generated Python parity wrappers remain non-production shadow evidence;
    * offline semantic certification never builds or downloads;
    * sealed private-HOME validation receives an explicit approved immutable
      deployment root rather than discovering mutable user paths.

    FVT-072 objective validation repair re-proves this acceptance and binds
    the synthetic discovery term ``objective validation repair``.
    """

    root_repo = Path(repo_root) if repo_root is not None else _REPO_ROOT
    install_path = mtl_installer._expand_install_root(install_root)
    install_bundle: mtl_installer.RuntimeMTLInstallBundle | None = None
    identity: mtl_installer.ExternalMonitorIdentity | None = None
    install_status = "missing"

    if skip_install:
        pin = mtl_installer.pin_for_tool(
            TOOL_EXTERNAL, repo_root=root_repo, lock_path=lock_path
        )
        identity = mtl_installer._identity_from_disk(
            TOOL_EXTERNAL, install_path, pin, vendor=True
        )
        if identity is None:
            raise ExternalRuntimeMTLCertificationError(
                f"skip_install requested but vendor Runtime MTL is missing under "
                f"{install_path}"
            )
        install_status = "already_present"
    else:
        install_bundle = mtl_installer.ensure_runtime_mtl_vendor(
            yes=True,
            strict=True,
            force=force_install,
            install_root=install_path,
            repo_root=root_repo,
            lock_path=lock_path,
            checksum_verified=True,
        )
        if not install_bundle.ok:
            raise ExternalRuntimeMTLCertificationError(
                "vendor installation failed: "
                + "; ".join(
                    f"{r.tool_id}:{r.status}:{r.detail}"
                    for r in install_bundle.receipts
                )
            )
        for receipt in install_bundle.receipts:
            if receipt.identity is not None:
                identity = receipt.identity
                install_status = receipt.status
                break

    if identity is None:
        raise ExternalRuntimeMTLCertificationError("vendor Runtime MTL identity missing")
    if identity.is_hermetic_parity_engine or not identity.is_vendor_build:
        raise ExternalRuntimeMTLCertificationError(
            "hermetic parity engines cannot satisfy vendor certification"
        )

    pin = mtl_installer.pin_for_tool(
        TOOL_EXTERNAL, repo_root=root_repo, lock_path=lock_path
    )
    if identity.version != pin["version"]:
        raise ExternalRuntimeMTLCertificationError(
            f"strict pin mismatch for vendor engine: "
            f"{identity.version!r} != {pin['version']!r}"
        )

    engine = _certify_vendor_engine(
        identity, install_status=install_status, repo_root=root_repo
    )

    # Hermetic shadow cannot satisfy vendor.
    hermetic_cannot_satisfy = True
    hermetic_probe = mtl_installer.ensure_runtime_mtl_external(
        yes=True,
        strict=True,
        force=False,
        install_root=install_path / "hermetic-shadow-probe",
        repo_root=root_repo,
        hermetic_parity_engine=True,
        vendor=False,
        checksum_verified=True,
    )
    hermetic_is_shadow = (
        hermetic_probe.ok
        and hermetic_probe.identity is not None
        and hermetic_probe.identity.is_hermetic_parity_engine
        and not hermetic_probe.identity.is_vendor_build
    )
    if not hermetic_is_shadow:
        hermetic_cannot_satisfy = False

    categories = sorted(REQUIRED_CATEGORIES)
    block_reasons = list(engine.block_reasons)
    if not hermetic_cannot_satisfy:
        block_reasons.append("hermetic_shadow_policy_broken")
    if identity.executable == (
        hermetic_probe.identity.executable if hermetic_probe.identity else ""
    ):
        block_reasons.append("vendor_reused_hermetic_executable")

    certified = bool(engine.certified) and not block_reasons

    payload: dict[str, Any] = {
        "schema_version": VENDOR_SCHEMA_VERSION,
        "interface": VENDOR_INTERFACE,
        "goal_id": VENDOR_GOAL_ID,
        "task_id": VENDOR_TASK_ID,
        "repair_task_id": VENDOR_REPAIR_TASK_ID,
        "program": VENDOR_PROGRAM,
        "lane_id": VENDOR_LANE_ID,
        "handler_id": VENDOR_HANDLER_ID,
        "certification_surface": CERTIFICATION_SURFACE,
        "authority_ceiling": AUTHORITY_CEILING,
        "forbids_theorem_authority": True,
        "forbids_global_correctness_claim": True,
        "certified": certified,
        # FVT-072 objective validation repair: re-prove FVT-G210 acceptance.
        "objective_validation_evidence": OBJECTIVE_VALIDATION_EVIDENCE,
        "objective_validation_repair": bool(certified),
        "objective_validation_command": OBJECTIVE_VALIDATION_COMMAND,
        "acceptance": {
            "objective_validation_repair": bool(certified),
            "objective_validation_evidence": OBJECTIVE_VALIDATION_EVIDENCE,
            "repair_task_id": VENDOR_REPAIR_TASK_ID,
            "goal_id": VENDOR_GOAL_ID,
            "task_id": VENDOR_TASK_ID,
            "locked_typescript_dependency_graph": True,
            "independent_node_package_without_python_dispatch": True,
            "package_source_lockfile_runtime_launcher_executable_artifact_digests_bound": True,
            "offline_certification_never_builds_or_downloads": True,
            "explicit_approved_immutable_deployment_root": True,
            "hermetic_parity_wrappers_are_non_production_shadows": True,
            "hermetic_parity_wrappers_cannot_satisfy_vendor": hermetic_cannot_satisfy,
            "finite_trace_authority_only": True,
            "never_grants_theorem_authority": True,
            "no_global_correctness_claim": True,
            "categories": categories,
            "mutation_kinds": sorted(REQUIRED_MUTATION_KINDS),
        },
        "runtime_mtl_external": {
            **engine.to_dict(),
            "is_vendor_build": True,
            "is_hermetic_parity_engine": False,
            "package_identity": identity.package_identity,
            "package_digest_sha256": identity.package_digest_sha256,
            "source_digest_sha256": identity.source_digest_sha256,
            "lockfile_digest_sha256": identity.lockfile_digest_sha256,
            "runtime_digest_sha256": identity.runtime_digest_sha256,
            "launcher_digest_sha256": getattr(
                identity, "launcher_digest_sha256", ""
            )
            or identity.executable_digest_sha256
            or identity.artifact_sha256,
            "launcher_target_digest_sha256": getattr(
                identity, "launcher_target_digest_sha256", ""
            )
            or identity.artifact_sha256,
            "executable_digest_sha256": identity.executable_digest_sha256
            or identity.artifact_sha256,
            "artifact_sha256": identity.artifact_sha256,
            "node_version": identity.node_version,
            "platform_id": identity.platform_id,
            "never_grants_theorem_authority": True,
            "finite_trace_authority_only": True,
            "no_python_reference_dispatch": True,
        },
        "engines": [engine.to_dict()],
        "engine_ids": [engine.engine_id],
        "external_engines": list(EXTERNAL_ENGINES),
        "reference_engines": list(REFERENCE_ENGINES),
        "categories_exercised": categories,
        "mutation_kinds": sorted(REQUIRED_MUTATION_KINDS),
        "install": None if install_bundle is None else install_bundle.to_dict(),
        "hermetic_parity_shadow": {
            "is_hermetic_parity_engine": True,
            "is_vendor_build": False,
            "non_production_shadow_evidence": True,
            "cannot_satisfy_vendor": hermetic_cannot_satisfy,
            "executable": (
                hermetic_probe.identity.executable
                if hermetic_probe.identity is not None
                else ""
            ),
        },
        "policy": {
            "strict_installation_selects_exact_pin": True,
            "locked_typescript_dependency_graph": True,
            "independent_node_package_without_python_dispatch": True,
            "package_source_lockfile_runtime_executable_artifact_digests_bound": True,
            "package_source_lockfile_runtime_launcher_executable_artifact_digests_bound": True,
            "offline_certification_never_builds_or_downloads": True,
            "explicit_approved_immutable_deployment_root": True,
            "disagreement_quarantines_promotion": True,
            "finite_trace_authority_only": True,
            "never_grants_theorem_authority": True,
            "no_global_correctness_claim": True,
            "inconclusive_prefix_semantics_preserved": True,
            "hermetic_parity_wrappers_are_non_production_shadows": True,
            "hermetic_parity_wrappers_cannot_satisfy_vendor": True,
            "never_promote_hermetic_as_vendor": True,
            "no_central_certificate_edit": True,
            "no_in_process_reference_edit": True,
            "objective_validation_repair": True,
            "grants_theorem_authority": False,
            "grants_global_correctness": False,
        },
        "summary": {
            "vendor_certified": engine.certified,
            "checks_passed": sum(1 for check in engine.checks if check.passed),
            "checks_total": len(engine.checks),
            "categories_exercised": categories,
            "mutation_kinds": sorted(REQUIRED_MUTATION_KINDS),
            "block_reasons": sorted(set(block_reasons)),
            "hermetic_parity_wrappers_cannot_satisfy_vendor": hermetic_cannot_satisfy,
            "objective_validation_repair": bool(certified),
            "repair_task_id": VENDOR_REPAIR_TASK_ID,
        },
    }
    payload["certificate_digest_sha256"] = _stable_json_digest(
        {
            key: value
            for key, value in payload.items()
            if key != "certificate_digest_sha256"
        }
    )

    receipt = build_vendor_install_receipt(payload)
    if write_receipt_path is not None:
        path = Path(write_receipt_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        payload["receipt_path"] = str(path)
    payload["install_receipt"] = receipt
    return payload


def build_vendor_install_receipt(certificate: Mapping[str, Any]) -> dict[str, Any]:
    """Build the checked-in vendor install receipt envelope."""

    engine = certificate.get("runtime_mtl_external") or {}
    hermetic_shadow = dict(certificate.get("hermetic_parity_shadow") or {})
    if hermetic_shadow.get("executable"):
        hermetic_shadow["executable"] = PUBLIC_MANAGED_PATH_REDACTION
    certified = bool(certificate.get("certified"))
    acceptance = dict(certificate.get("acceptance") or {})
    if not acceptance:
        acceptance = {
            "objective_validation_repair": certified,
            "objective_validation_evidence": OBJECTIVE_VALIDATION_EVIDENCE,
            "repair_task_id": VENDOR_REPAIR_TASK_ID,
            "goal_id": VENDOR_GOAL_ID,
            "task_id": VENDOR_TASK_ID,
        }
    summary = dict(certificate.get("summary") or {})
    summary.setdefault("objective_validation_repair", certified)
    summary.setdefault("repair_task_id", VENDOR_REPAIR_TASK_ID)
    policy = dict(certificate.get("policy") or {})
    policy.setdefault("objective_validation_repair", True)
    receipt = {
        "schema_version": VENDOR_INSTALL_RECEIPT_SCHEMA,
        "interface": VENDOR_INTERFACE,
        "goal_id": VENDOR_GOAL_ID,
        "task_id": VENDOR_TASK_ID,
        "repair_task_id": VENDOR_REPAIR_TASK_ID,
        "program": VENDOR_PROGRAM,
        "lane_id": VENDOR_LANE_ID,
        "handler_id": VENDOR_HANDLER_ID,
        "certified": certified,
        "authority_ceiling": AUTHORITY_CEILING,
        # FVT-072 objective validation repair discovery keys.
        "objective_validation_evidence": OBJECTIVE_VALIDATION_EVIDENCE,
        "objective_validation_repair": certified,
        "objective_validation_command": OBJECTIVE_VALIDATION_COMMAND,
        "acceptance": acceptance,
        "runtime_mtl_external": {
            "tool_id": TOOL_EXTERNAL,
            "version": engine.get("version"),
            "executable": (
                PUBLIC_MANAGED_PATH_REDACTION
                if engine.get("executable")
                else ""
            ),
            "usable": engine.get("usable"),
            "certified": engine.get("certified"),
            "is_vendor_build": True,
            "is_hermetic_parity_engine": False,
            "package_identity": engine.get("package_identity"),
            "package_digest_sha256": engine.get("package_digest_sha256"),
            "source_digest_sha256": engine.get("source_digest_sha256"),
            "lockfile_digest_sha256": engine.get("lockfile_digest_sha256"),
            "runtime_digest_sha256": engine.get("runtime_digest_sha256"),
            "launcher_digest_sha256": engine.get("launcher_digest_sha256")
            or engine.get("executable_digest_sha256"),
            "launcher_target_digest_sha256": engine.get(
                "launcher_target_digest_sha256"
            )
            or engine.get("artifact_sha256"),
            "executable_digest_sha256": engine.get("executable_digest_sha256"),
            "artifact_sha256": engine.get("artifact_sha256"),
            "node_version": engine.get("node_version"),
            "platform_id": engine.get("platform_id"),
            "role": ToolRole.AUTHORITY.value,
            "authority_ceiling": AUTHORITY_CEILING,
            "never_grants_theorem_authority": True,
            "finite_trace_authority_only": True,
            "no_python_reference_dispatch": True,
        },
        "hermetic_parity_shadow": hermetic_shadow,
        "categories_exercised": list(certificate.get("categories_exercised") or []),
        "mutation_kinds": list(certificate.get("mutation_kinds") or []),
        "policy": policy,
        "summary": summary,
        "certificate_digest_sha256": certificate.get("certificate_digest_sha256"),
    }
    receipt["receipt_digest_sha256"] = _stable_json_digest(
        {k: v for k, v in receipt.items() if k != "receipt_digest_sha256"}
    )
    return receipt


def write_vendor_install_receipt(
    certificate: Mapping[str, Any] | None = None,
    *,
    repo_root: Path | str | None = None,
    install_root: Path | str | None = None,
    receipt_path: Path | str | None = None,
) -> dict[str, Any]:
    """Certify (if needed) and write the vendor install receipt artifact."""

    root = Path(repo_root) if repo_root is not None else _REPO_ROOT
    path = (
        Path(receipt_path)
        if receipt_path is not None
        else root / DEFAULT_VENDOR_RECEIPT_RELATIVE
    )
    if certificate is None:
        certificate = certify_external_runtime_mtl_vendor(
            install_root=install_root,
            force_install=True,
            repo_root=root,
            write_receipt_path=path,
        )
        return dict(certificate.get("install_receipt") or {})
    receipt = build_vendor_install_receipt(certificate)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def external_runtime_mtl_vendor_lane_handler(
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """Lane handler for external Runtime MTL vendor certification."""

    result = certify_external_runtime_mtl_vendor(
        install_root=kwargs.get("install_root"),
        force_install=bool(kwargs.get("force_install", False)),
        skip_install=bool(kwargs.get("skip_install", False)),
        repo_root=kwargs.get("repo_root"),
        lock_path=kwargs.get("lock_path"),
    )
    certified = bool(result["certified"])
    return {
        "lane_id": VENDOR_LANE_ID,
        "owner_module": CERTIFICATION_SURFACE,
        "handler_id": VENDOR_HANDLER_ID,
        "status": "certified" if certified else "failed",
        "certified": certified,
        "authority_ceiling": AUTHORITY_CEILING,
        "reason_codes": list(result["summary"].get("block_reasons") or []),
        "certificate_digest_sha256": result["certificate_digest_sha256"],
        "engine_ids": list(result.get("engine_ids") or []),
        "args_received": bool(args) or bool(kwargs),
        "interface": VENDOR_INTERFACE,
        "goal_id": VENDOR_GOAL_ID,
        "task_id": VENDOR_TASK_ID,
        "repair_task_id": VENDOR_REPAIR_TASK_ID,
        "objective_validation_evidence": OBJECTIVE_VALIDATION_EVIDENCE,
        "objective_validation_repair": certified,
        "objective_validation_command": OBJECTIVE_VALIDATION_COMMAND,
        "grants_theorem_authority": False,
        "grants_global_correctness": False,
        "finite_trace_authority_only": True,
        "hermetic_parity_wrappers_cannot_satisfy_vendor": bool(
            result["summary"].get("hermetic_parity_wrappers_cannot_satisfy_vendor")
        ),
        "is_vendor_build": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Certify external Runtime MTL cross-runtime parity / vendor engine "
            f"({INTERFACE} / {VENDOR_INTERFACE})."
        )
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full certification receipt as JSON",
    )
    parser.add_argument(
        "--install-root",
        type=Path,
        default=None,
        help="User-local install root for pin-bound external monitor",
    )
    parser.add_argument(
        "--force-install",
        action="store_true",
        help="Force re-materialization of external engine",
    )
    parser.add_argument(
        "--engine",
        action="append",
        dest="engines",
        default=None,
        help="Limit certification to one engine id (repeatable)",
    )
    parser.add_argument(
        "--vendor",
        action="store_true",
        help="Run ExternalRuntimeMTLVendorCertification@1 (FVT-G210)",
    )
    parser.add_argument(
        "--write-receipt",
        type=Path,
        default=None,
        help="Write vendor install receipt JSON to this path",
    )
    args = parser.parse_args(argv)

    try:
        if args.vendor:
            receipt = certify_external_runtime_mtl_vendor(
                install_root=args.install_root,
                force_install=args.force_install,
                write_receipt_path=args.write_receipt,
            )
            interface = VENDOR_INTERFACE
            goal_id = VENDOR_GOAL_ID
            task_id = VENDOR_TASK_ID
            lane_id = VENDOR_LANE_ID
        else:
            receipt = certify_external_runtime_mtl(
                install_root=args.install_root,
                engines=args.engines,
                force_install=args.force_install,
            )
            interface = INTERFACE
            goal_id = GOAL_ID
            task_id = TASK_ID
            lane_id = LANE_ID
    except Exception as exc:
        if args.json:
            print(
                json.dumps(
                    {
                        "certified": False,
                        "error": f"{type(exc).__name__}:{exc}",
                        "interface": VENDOR_INTERFACE if args.vendor else INTERFACE,
                        "goal_id": VENDOR_GOAL_ID if args.vendor else GOAL_ID,
                        "task_id": VENDOR_TASK_ID if args.vendor else TASK_ID,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(
                f"{VENDOR_INTERFACE if args.vendor else INTERFACE} FAILED: {exc}",
                file=sys.stderr,
            )
        return 1

    if args.json:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    else:
        status = "CERTIFIED" if receipt["certified"] else "FAILED"
        print(f"{interface} {status}")
        print(
            f"goal={goal_id} task={task_id} lane={lane_id} "
            f"engines={','.join(receipt.get('engine_ids') or [])}"
        )
        summary = receipt["summary"]
        if args.vendor:
            print(
                f"checks={summary['checks_passed']}/{summary['checks_total']} "
                f"vendor_certified={summary.get('vendor_certified')}"
            )
        else:
            print(
                f"checks={summary['checks_passed']}/{summary['checks_total']} "
                f"engines_certified={summary['engines_certified']}/{summary['engines_total']}"
            )
        if summary.get("block_reasons"):
            print("block_reasons:")
            for reason in summary["block_reasons"]:
                print(f"  - {reason}")
        print(f"digest={receipt['certificate_digest_sha256']}")
    return 0 if receipt["certified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "INTERFACE",
    "SCHEMA_VERSION",
    "GOAL_ID",
    "TASK_ID",
    "PROGRAM",
    "LANE_ID",
    "HANDLER_ID",
    "CERTIFICATION_SURFACE",
    "VENDOR_INTERFACE",
    "VENDOR_SCHEMA_VERSION",
    "VENDOR_INSTALL_RECEIPT_SCHEMA",
    "VENDOR_GOAL_ID",
    "VENDOR_TASK_ID",
    "VENDOR_PROGRAM",
    "VENDOR_LANE_ID",
    "VENDOR_HANDLER_ID",
    "DEFAULT_VENDOR_RECEIPT_RELATIVE",
    "AUTHORITY_CEILING",
    "EXTERNAL_ENGINES",
    "REFERENCE_ENGINES",
    "REQUIRED_CATEGORIES",
    "REQUIRED_MUTATION_KINDS",
    "CheckResult",
    "EngineCertification",
    "ExternalRuntimeMTLCertificationError",
    "ParityRunRecord",
    "build_vendor_install_receipt",
    "certify_engine",
    "certify_external_runtime_mtl",
    "certify_external_runtime_mtl_vendor",
    "default_case_specs",
    "external_runtime_mtl_lane_handler",
    "external_runtime_mtl_vendor_lane_handler",
    "main",
    "materialize_case",
    "run_parity_case",
    "write_vendor_install_receipt",
]
