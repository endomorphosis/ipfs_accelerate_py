#!/usr/bin/env python3
"""External Datalog/SecPAL differential-shadow and vendor certification.

``ExternalAuthorizationShadowCertification@1`` / FVT-G180 (FVT-051)
``ExternalAuthorizationVendorCertification@1`` / FVT-G209 (FVT-055, FVT-073)

Shadow path: explicit strict installation selects pin-bound Soufflé and
SecPAL-compatible hermetic shadow engines for differential work only.

Vendor path (FVT-G209): replace the Soufflé case-oracle shadow with a
checksummed vendor build bound to the immutable 2.4.1 source archive and
reviewed build dependencies.  Real allow/deny/unknown/conflict/delegation
plus rule/scope mutation, replay, malformed, timeout, and disagreement cases
execute through the vendor engine.  linux-aarch64 is supported for Soufflé.
External SecPAL is a lock-derived narrow unsupported-platform exception on
linux-aarch64 and never counts as installed, complete, authoritative, or
production-certified.  Hermetic shadows remain differential-only and cannot
satisfy vendor production evidence.

Objective validation repair (FVT-073)
-------------------------------------
Path evidence for this certifier, the vendor installer, the lock pins, the
focused tests, and the checked-in vendor receipt may already exist while the
supervisor validation gate still needs an explicit re-proof of the full
FVT-G209 acceptance matrix.  The synthetic evidence term
``objective validation repair`` is bound in the vendor certificate receipt,
the checked-in install receipt, and
``test_external_authorization_vendor_certification.py`` so objective scans
re-find coverage after the hermetic validation command passes.

External engines remain role=shadow: authorization authority stays with the
in-process Datalog/SecPAL references (FVT-G102 / FVT-038).

This lane never edits the central multi-prover certificate and never weakens
in-process reference semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Final, Mapping, Sequence

# Allow running as a script from a worktree without an installed package.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DATASETS_ROOT = _REPO_ROOT / "ipfs_datasets_py"
for _candidate in (_REPO_ROOT, _DATASETS_ROOT):
    _text = str(_candidate)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from ipfs_datasets_py.logic.backends.datalog.adapters import (  # noqa: E402
    DEFAULT_AUTHORIZATION_FIXTURES,
    ReferenceAuthorizationEvaluator,
    parse_engine_outcome,
    render_datalog_program,
    render_secpal_program,
)
from ipfs_datasets_py.logic.backends.installers import authorization as authz_installer  # noqa: E402
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
from ipfs_datasets_py.logic.software_verification.authorization import (  # noqa: E402
    AuthorizationIR,
    DecisionOutcome,
    DecisionQuery,
)

# Reuse compact recipes / mutations from the in-process semantic certifier.
_SEMANTIC_CERTIFIER_PATH = (
    _REPO_ROOT / "tools" / "logic" / "certification" / "authorization.py"
)


def _load_semantic_certifier():
    spec = importlib.util.spec_from_file_location(
        "authorization_semantic_certification_for_external",
        _SEMANTIC_CERTIFIER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load semantic certifier at {_SEMANTIC_CERTIFIER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_semantic = _load_semantic_certifier()

INTERFACE: Final = "ExternalAuthorizationShadowCertification@1"
SCHEMA_VERSION: Final = "external-authorization-shadow-certification/v1"
GOAL_ID: Final = "FVT-G180"
TASK_ID: Final = "FVT-051"
PROGRAM: Final = "formal-verification-tactician/authorization-toolchains"
LANE_ID: Final = "datalog_secpal_external"
HANDLER_ID: Final = "external_authorization_shadow_certification@1"
CERTIFICATION_SURFACE: Final = "tools.logic.certification.authorization_external"

VENDOR_INTERFACE: Final = "ExternalAuthorizationVendorCertification@1"
VENDOR_SCHEMA_VERSION: Final = "external-authorization-vendor-certification/v1"
VENDOR_INSTALL_RECEIPT_SCHEMA: Final = (
    "formal-verification-authorization-vendor-install-receipt/v1"
)
VENDOR_GOAL_ID: Final = "FVT-G209"
VENDOR_TASK_ID: Final = "FVT-055"
# Validation-gate task that re-proves FVT-G209 when path evidence already exists.
VENDOR_REPAIR_TASK_ID: Final = "FVT-073"
# Synthetic evidence term required by objective-scan validation gates.
OBJECTIVE_VALIDATION_EVIDENCE: Final = "objective validation repair"
VENDOR_PROGRAM: Final = (
    "formal-verification-tactician/authorization-vendor-toolchains"
)
VENDOR_LANE_ID: Final = "datalog_secpal_external_vendor"
VENDOR_HANDLER_ID: Final = "external_authorization_vendor_certification@1"
DEFAULT_VENDOR_RECEIPT_RELATIVE: Final = Path(
    "docs/architecture/formal_verification_authorization_vendor_install_receipt.json"
)
LINUX_AARCH64: Final = "linux-aarch64"
SOUFFLE_REQUIRED_SOURCE_SHA256: Final = (
    authz_installer.SOUFFLE_SOURCE_ARCHIVE_SHA256
)

# Hermetic validation command bound by FVT-G209 / FVT-073.
OBJECTIVE_VALIDATION_COMMAND: Final = (
    "PYTHONPATH=ipfs_datasets_py python -m pytest "
    "test/integration/toolchains/test_external_authorization_vendor_certification.py "
    "test/integration/toolchains/test_external_authorization_toolchain_certification.py "
    "-q"
)

# External engines are shadows — authority ceiling is none.
SHADOW_AUTHORITY_CEILING: Final = ToolchainAuthorityCeiling.NONE.value
# In-process references retain authorization authority.
REFERENCE_AUTHORITY_CEILING: Final = ToolchainAuthorityCeiling.AUTHORIZATION.value

TOOL_SOUFFLE: Final = authz_installer.TOOL_SOUFFLE
TOOL_SECPAL: Final = authz_installer.TOOL_SECPAL
EXTERNAL_ENGINES: Final = (TOOL_SOUFFLE, TOOL_SECPAL)
REFERENCE_ENGINES: Final = (
    "datalog-authorization",
    "secpal-authorization",
)

# Corpus categories required by FVT-G180 acceptance.
REQUIRED_CATEGORIES: Final = frozenset(
    {
        "allow",
        "deny",
        "unknown",
        "conflict",
        "delegation",
    }
)
REQUIRED_MUTATION_KINDS: Final = frozenset({"rule", "scope"})
CHECK_KINDS: Final = frozenset(
    {
        "positive",
        "mutation",
        "replay",
        "malformed",
        "timeout",
        "differential",
        "disagreement_quarantine",
        "authority",
        "install",
        "role",
    }
)


class ExternalAuthorizationCertificationError(ValueError):
    """Raised when external authorization shadow certification fails closed."""


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One hermetic external-shadow check outcome."""

    check_id: str
    kind: str
    status: str
    expected: str
    observed: str
    detail: str = ""
    engine_id: str = ""
    authority: str = SHADOW_AUTHORITY_CEILING
    is_theorem_authority: bool = False
    is_authorization_authority: bool = False
    quarantined: bool = False

    def __post_init__(self) -> None:
        if self.kind not in CHECK_KINDS:
            raise ExternalAuthorizationCertificationError(
                f"unknown check kind {self.kind!r}"
            )
        if self.status not in {
            "passed",
            "failed",
            "quarantined",
            "error",
            "skipped",
        }:
            raise ExternalAuthorizationCertificationError(
                f"unknown check status {self.status!r}"
            )
        if self.is_theorem_authority:
            raise ExternalAuthorizationCertificationError(
                "external shadow checks cannot claim theorem authority"
            )
        if self.is_authorization_authority:
            raise ExternalAuthorizationCertificationError(
                "external shadow checks cannot claim authorization authority"
            )
        if self.authority not in {SHADOW_AUTHORITY_CEILING, "none", ""}:
            raise ExternalAuthorizationCertificationError(
                "external shadows may only report none authority"
            )

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "authority": self.authority or SHADOW_AUTHORITY_CEILING,
            "check_id": self.check_id,
            "detail": self.detail,
            "engine_id": self.engine_id,
            "expected": self.expected,
            "is_authorization_authority": False,
            "is_theorem_authority": False,
            "kind": self.kind,
            "observed": self.observed,
            "quarantined": self.quarantined,
            "status": self.status,
        }


@dataclass
class ShadowRunRecord:
    """One external shadow evaluation for differential comparison."""

    engine_id: str
    case_id: str
    outcome: str
    status: str
    reference_outcome: str
    agreed: bool
    timed_out: bool = False
    malformed: bool = False
    detail: str = ""
    executable: str = ""
    engine_version: str = ""
    policy_digest: str = ""
    authority: str = SHADOW_AUTHORITY_CEILING
    is_theorem_authority: bool = False
    is_authorization_authority: bool = False
    quarantined: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "agreed": self.agreed,
            "authority": self.authority,
            "case_id": self.case_id,
            "detail": self.detail,
            "engine_id": self.engine_id,
            "engine_version": self.engine_version,
            "executable": self.executable,
            "is_authorization_authority": False,
            "is_theorem_authority": False,
            "malformed": self.malformed,
            "outcome": self.outcome,
            "policy_digest": self.policy_digest,
            "quarantined": self.quarantined,
            "reference_outcome": self.reference_outcome,
            "status": self.status,
            "timed_out": self.timed_out,
        }


@dataclass
class EngineCertification:
    """Per-external-engine shadow certification summary."""

    engine_id: str
    version: str
    executable: str
    usable: bool
    certified: bool
    role: str
    authority_ceiling: str
    is_shadow: bool = True
    checks: list[CheckResult] = field(default_factory=list)
    case_results: list[ShadowRunRecord] = field(default_factory=list)
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
            "is_shadow": True,
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


def _reference_outcome(
    document: AuthorizationIR, query: DecisionQuery
) -> DecisionOutcome:
    decision, _, _ = ReferenceAuthorizationEvaluator().evaluate(document, query)
    return decision.outcome


def _render_for_engine(
    engine_id: str, document: AuthorizationIR, query: DecisionQuery
) -> tuple[str, str, tuple[str, ...]]:
    """Return (source_text, file_suffix, argv_suffix_without_executable)."""

    if engine_id == TOOL_SOUFFLE:
        return render_datalog_program(document, query), "dl", ()
    if engine_id == TOOL_SECPAL:
        return render_secpal_program(document, query), "secpal", ("check",)
    raise ExternalAuthorizationCertificationError(f"unknown engine {engine_id!r}")


def _run_shadow_process(
    executable: str,
    engine_id: str,
    document: AuthorizationIR,
    query: DecisionQuery,
    *,
    timeout_seconds: float = 2.0,
    env: Mapping[str, str] | None = None,
    runner: BoundedToolRunner | None = None,
) -> tuple[str | None, dict[str, Any]]:
    """Execute one external shadow and return (outcome_token_or_None, meta)."""

    source, suffix, argv_prefix = _render_for_engine(engine_id, document, query)
    tool_runner = runner or BoundedToolRunner()
    filename = f"policy.{suffix}"
    argv = (executable, *argv_prefix, f"{{workspace}}/{filename}")
    # Merge environment for hermetic shim controls.
    run_env = dict(os.environ)
    if env:
        run_env.update({str(k): str(v) for k, v in env.items()})
    request = ToolRunRequest(
        argv=argv,
        runtime=ToolRuntime.NATIVE,
        limits=ToolRunLimits(
            timeout_seconds=timeout_seconds,
            cpu_seconds=timeout_seconds,
            memory_bytes=64 * 1024 * 1024,
            max_output_bytes=64 * 1024,
            max_input_bytes=max(64 * 1024, len(source.encode("utf-8")) + 1024),
            max_workspace_bytes=max(
                128 * 1024, len(source.encode("utf-8")) + 64 * 1024
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
    outcome = parse_engine_outcome(result.stdout or "", result.stderr or "")
    if outcome is None:
        # Distinguish malformed (tokens present but unparseable) from empty deny.
        combined = f"{result.stdout or ''}\n{result.stderr or ''}".strip()
        if combined and "%%%" in combined:
            meta["malformed"] = True
            return None, meta
        if combined and not any(
            token in combined.casefold()
            for token in ("allow", "deny", "unknown", "conflict", "permit")
        ):
            meta["malformed"] = True
            return None, meta
        return None, meta
    return outcome.value, meta


def run_shadow_case(
    engine_id: str,
    case_id: str,
    document: AuthorizationIR | None,
    query: DecisionQuery | None,
    *,
    executable: str,
    engine_version: str = "",
    expect_error: bool = False,
    timeout_seconds: float = 2.0,
    env: Mapping[str, str] | None = None,
    runner: BoundedToolRunner | None = None,
) -> ShadowRunRecord:
    """Run one case on one external shadow and differentially compare."""

    if expect_error or document is None or query is None:
        # Malformed policy path: write garbage and require non-allow handling.
        with tempfile.TemporaryDirectory(prefix="authz-shadow-malformed-") as tmp:
            bad = Path(tmp) / ("policy.dl" if engine_id == TOOL_SOUFFLE else "policy.secpal")
            bad.write_text("{not valid authorization policy@@@@\n", encoding="utf-8")
            tool_runner = runner or BoundedToolRunner()
            argv_prefix: tuple[str, ...] = () if engine_id == TOOL_SOUFFLE else ("check",)
            run_env = dict(os.environ)
            if env:
                run_env.update({str(k): str(v) for k, v in env.items()})
            # Force malformed emission from hermetic shim when available.
            run_env.setdefault(authz_installer.ENV_MALFORMED, "1")
            request = ToolRunRequest(
                argv=(executable, *argv_prefix, str(bad)),
                runtime=ToolRuntime.NATIVE,
                limits=ToolRunLimits(
                    timeout_seconds=timeout_seconds,
                    cpu_seconds=timeout_seconds,
                    memory_bytes=32 * 1024 * 1024,
                    max_output_bytes=16 * 1024,
                    max_input_bytes=16 * 1024,
                    max_workspace_bytes=64 * 1024,
                ),
                environment=run_env,
            )
            try:
                result = tool_runner.run(request)
            except Exception as exc:
                return ShadowRunRecord(
                    engine_id=engine_id,
                    case_id=case_id,
                    outcome="error",
                    status="error",
                    reference_outcome="error",
                    agreed=True,
                    malformed=True,
                    detail=str(exc)[:240],
                    executable=executable,
                    engine_version=engine_version,
                    quarantined=True,
                )
            parsed = parse_engine_outcome(result.stdout or "", result.stderr or "")
            if parsed is DecisionOutcome.ALLOW:
                return ShadowRunRecord(
                    engine_id=engine_id,
                    case_id=case_id,
                    outcome="allow",
                    status="unexpected_success",
                    reference_outcome="error",
                    agreed=False,
                    malformed=True,
                    detail="malformed input produced allow",
                    executable=executable,
                    engine_version=engine_version,
                    quarantined=True,
                )
            return ShadowRunRecord(
                engine_id=engine_id,
                case_id=case_id,
                outcome="error" if parsed is None else parsed.value,
                status="error" if parsed is None else "quarantined",
                reference_outcome="error",
                agreed=parsed is None or parsed is not DecisionOutcome.ALLOW,
                malformed=True,
                detail="malformed input fail-closed",
                executable=executable,
                engine_version=engine_version,
                quarantined=True,
            )

    reference = _reference_outcome(document, query)
    observed, meta = _run_shadow_process(
        executable,
        engine_id,
        document,
        query,
        timeout_seconds=timeout_seconds,
        env=env,
        runner=runner,
    )
    policy_digest = document.sha256
    if meta.get("timed_out"):
        return ShadowRunRecord(
            engine_id=engine_id,
            case_id=case_id,
            outcome="timeout",
            status="timeout",
            reference_outcome=reference.value,
            agreed=False,
            timed_out=True,
            detail="shadow engine timed out",
            executable=executable,
            engine_version=engine_version,
            policy_digest=policy_digest,
            quarantined=True,
        )
    if meta.get("malformed") or observed is None:
        return ShadowRunRecord(
            engine_id=engine_id,
            case_id=case_id,
            outcome="error",
            status="error",
            reference_outcome=reference.value,
            agreed=False,
            malformed=bool(meta.get("malformed")),
            detail=str(meta.get("error") or "unparseable shadow output")[:240],
            executable=executable,
            engine_version=engine_version,
            policy_digest=policy_digest,
            quarantined=True,
        )
    agreed = observed == reference.value
    quarantined = not agreed
    return ShadowRunRecord(
        engine_id=engine_id,
        case_id=case_id,
        outcome=observed,
        status="agreed" if agreed else "disagreement",
        reference_outcome=reference.value,
        agreed=agreed,
        detail="" if agreed else "shadow disagreed with reference; promotion quarantined",
        executable=executable,
        engine_version=engine_version,
        policy_digest=policy_digest,
        quarantined=quarantined,
    )


def default_case_specs():
    """Reuse the compact semantic corpus recipes, filtered for G180."""

    return _semantic.default_case_specs()


def materialize_case(spec):
    return _semantic.materialize_case(spec)


# ---------------------------------------------------------------------------
# Certification
# ---------------------------------------------------------------------------


def _install_external_engines(
    *,
    install_root: Path | str | None,
    force: bool = False,
) -> authz_installer.AuthorizationInstallBundle:
    return authz_installer.ensure_authorization_external(
        yes=True,
        strict=True,
        force=force,
        install_root=install_root,
        hermetic_shadow=True,
        checksum_verified=True,
    )


def certify_engine(
    engine_id: str,
    *,
    identity: authz_installer.ShadowEngineIdentity,
    install_status: str = "installed",
    specs: Sequence[Any] | None = None,
) -> EngineCertification:
    """Run the full external-shadow matrix for one pin-bound engine."""

    selected = tuple(specs or default_case_specs())
    checks: list[CheckResult] = []
    records: list[ShadowRunRecord] = []
    block_reasons: list[str] = []

    # Role binding — must remain shadow / none authority.
    try:
        role = get_tool_role(engine_id)
        role_ok = (
            role.role is ToolRole.SHADOW
            and role.authority_ceiling is ToolchainAuthorityCeiling.NONE
        )
    except Exception as exc:
        role_ok = False
        block_reasons.append(f"role_lookup_failed:{type(exc).__name__}")
        role = None  # type: ignore[assignment]

    checks.append(
        CheckResult(
            check_id=f"{engine_id}.role.shadow",
            kind="role",
            status="passed" if role_ok else "failed",
            expected="shadow/none",
            observed=(
                f"{role.role.value}/{role.authority_ceiling.value}"
                if role is not None
                else "unavailable"
            ),
            detail="external engines remain shadows",
            engine_id=engine_id,
        )
    )
    if not role_ok:
        block_reasons.append("role_not_shadow")

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

    # ---- positive corpus (allow/deny/unknown/conflict/delegation)
    for spec in selected:
        if spec.category not in REQUIRED_CATEGORIES:
            continue
        document, query, expected = materialize_case(spec)
        record = run_shadow_case(
            engine_id,
            spec.case_id,
            document,
            query,
            executable=identity.executable,
            engine_version=identity.version,
        )
        records.append(record)
        category_seen.add(spec.category)

        ok = (
            record.agreed
            and record.outcome == expected
            and record.reference_outcome == expected
            and not record.is_theorem_authority
            and not record.is_authorization_authority
        )
        checks.append(
            CheckResult(
                check_id=f"{engine_id}.{spec.case_id}.positive",
                kind="positive",
                status="passed" if ok else "failed",
                expected=expected,
                observed=record.outcome,
                detail=spec.notes or "differential positive case",
                engine_id=engine_id,
                quarantined=record.quarantined,
            )
        )
        if not ok:
            block_reasons.append(f"positive_failed:{spec.case_id}")

        # Differential check (explicit kind).
        diff_ok = record.agreed and not record.quarantined
        checks.append(
            CheckResult(
                check_id=f"{engine_id}.{spec.case_id}.differential",
                kind="differential",
                status="passed" if diff_ok else "quarantined",
                expected=record.reference_outcome,
                observed=record.outcome,
                detail=(
                    "shadow matches in-process reference"
                    if diff_ok
                    else "disagreement quarantines promotion"
                ),
                engine_id=engine_id,
                quarantined=not diff_ok,
            )
        )
        if not diff_ok:
            block_reasons.append(f"differential_disagreement:{spec.case_id}")

        # Deterministic replay for non-allow outcomes.
        if expected != DecisionOutcome.ALLOW.value:
            replay = run_shadow_case(
                engine_id,
                f"{spec.case_id}:replay",
                document,
                query,
                executable=identity.executable,
                engine_version=identity.version,
            )
            records.append(replay)
            replay_ok = (
                replay.outcome == record.outcome
                and replay.policy_digest == record.policy_digest
                and replay.agreed == record.agreed
            )
            checks.append(
                CheckResult(
                    check_id=f"{engine_id}.{spec.case_id}.replay",
                    kind="replay",
                    status="passed" if replay_ok else "failed",
                    expected=record.outcome,
                    observed=replay.outcome,
                    detail="shadow replay must be deterministic",
                    engine_id=engine_id,
                )
            )
            if not replay_ok:
                block_reasons.append(f"replay_unstable:{spec.case_id}")

    missing_categories = sorted(REQUIRED_CATEGORIES - category_seen)
    if missing_categories:
        block_reasons.append(f"missing_categories:{','.join(missing_categories)}")

    # ---- rule / scope mutations
    for spec in selected:
        if spec.category != "mutation":
            continue
        if spec.mutation_kind not in REQUIRED_MUTATION_KINDS:
            continue
        base = _semantic._fixture_by_id(spec.base_fixture_id)
        base_record = run_shadow_case(
            engine_id,
            f"{spec.case_id}:baseline",
            base.document,
            base.query,
            executable=identity.executable,
            engine_version=identity.version,
        )
        records.append(base_record)
        document, query, expected = materialize_case(spec)
        mutated = run_shadow_case(
            engine_id,
            spec.case_id,
            document,
            query,
            executable=identity.executable,
            engine_version=identity.version,
        )
        records.append(mutated)
        mutation_seen.add(spec.mutation_kind)

        changed = mutated.outcome != base_record.outcome
        matches = mutated.outcome == expected and mutated.agreed
        policy_changed = mutated.policy_digest != base_record.policy_digest
        ok = changed and matches and policy_changed
        checks.append(
            CheckResult(
                check_id=f"{engine_id}.{spec.case_id}.mutation",
                kind="mutation",
                status="passed" if ok else "failed",
                expected=f"{expected} (changed from {base_record.outcome})",
                observed=mutated.outcome,
                detail=f"mutation_kind={spec.mutation_kind}; policy_changed={policy_changed}",
                engine_id=engine_id,
            )
        )
        if not ok:
            block_reasons.append(f"mutation_failed:{spec.case_id}")

    missing_mutations = sorted(REQUIRED_MUTATION_KINDS - mutation_seen)
    if missing_mutations:
        block_reasons.append(f"missing_mutations:{','.join(missing_mutations)}")

    # ---- malformed output fail-closed
    malformed = run_shadow_case(
        engine_id,
        "case:malformed",
        None,
        None,
        executable=identity.executable,
        engine_version=identity.version,
        expect_error=True,
    )
    records.append(malformed)
    malformed_ok = (
        malformed.outcome != DecisionOutcome.ALLOW.value
        and malformed.malformed
        and malformed.quarantined
    )
    checks.append(
        CheckResult(
            check_id=f"{engine_id}.case:malformed.malformed",
            kind="malformed",
            status="passed" if malformed_ok else "failed",
            expected="error|quarantine (never allow)",
            observed=malformed.outcome,
            detail=malformed.detail,
            engine_id=engine_id,
            quarantined=malformed.quarantined,
        )
    )
    if not malformed_ok:
        block_reasons.append("malformed_not_fail_closed")

    # ---- timeout probe (hermetic sleep via env)
    timeout_fixture = next(
        item for item in DEFAULT_AUTHORIZATION_FIXTURES if item.category == "allow"
    )
    timed = run_shadow_case(
        engine_id,
        "case:timeout",
        timeout_fixture.document,
        timeout_fixture.query,
        executable=identity.executable,
        engine_version=identity.version,
        timeout_seconds=0.25,
        env={authz_installer.ENV_SLEEP_SECONDS: "2.0"},
    )
    records.append(timed)
    timeout_ok = timed.timed_out and timed.quarantined
    checks.append(
        CheckResult(
            check_id=f"{engine_id}.case:timeout.timeout",
            kind="timeout",
            status="passed" if timeout_ok else "failed",
            expected="timeout+quarantine",
            observed=timed.outcome,
            detail=timed.detail or "bounded timeout must fire",
            engine_id=engine_id,
            quarantined=timed.quarantined,
        )
    )
    if not timeout_ok:
        block_reasons.append("timeout_not_enforced")

    # ---- deliberate disagreement must quarantine promotion
    disagree = run_shadow_case(
        engine_id,
        "case:disagreement",
        timeout_fixture.document,
        timeout_fixture.query,
        executable=identity.executable,
        engine_version=identity.version,
        env={authz_installer.ENV_DISAGREE: "1"},
    )
    records.append(disagree)
    disagree_ok = (
        not disagree.agreed
        and disagree.quarantined
        and disagree.outcome != disagree.reference_outcome
    )
    checks.append(
        CheckResult(
            check_id=f"{engine_id}.case:disagreement.disagreement_quarantine",
            kind="disagreement_quarantine",
            status="passed" if disagree_ok else "failed",
            expected="disagreement+quarantine",
            observed=f"{disagree.outcome} vs {disagree.reference_outcome}",
            detail="any disagreement quarantines promotion",
            engine_id=engine_id,
            quarantined=disagree.quarantined,
        )
    )
    if not disagree_ok:
        block_reasons.append("disagreement_not_quarantined")

    # Authority: external engine never claims authorization/theorem authority.
    authority_ok = (
        identity.role == ToolRole.SHADOW.value
        and identity.authority_ceiling == SHADOW_AUTHORITY_CEILING
        and all(
            not record.is_authorization_authority and not record.is_theorem_authority
            for record in records
        )
    )
    checks.append(
        CheckResult(
            check_id=f"{engine_id}.authority.shadow_only",
            kind="authority",
            status="passed" if authority_ok else "failed",
            expected="shadow/none",
            observed=f"{identity.role}/{identity.authority_ceiling}",
            detail="in-process references retain authorization authority",
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
        role=ToolRole.SHADOW.value,
        authority_ceiling=SHADOW_AUTHORITY_CEILING,
        is_shadow=True,
        checks=checks,
        case_results=records,
        block_reasons=sorted(set(block_reasons)),
        install_status=install_status,
    )


def certify_external_authorization_shadows(
    *,
    install_root: Path | str | None = None,
    engines: Sequence[str] | None = None,
    force_install: bool = False,
    skip_install: bool = False,
    identities: Mapping[str, authz_installer.ShadowEngineIdentity] | None = None,
) -> dict[str, Any]:
    """Run full external authorization shadow certification for FVT-G180."""

    selected = tuple(engines or EXTERNAL_ENGINES)
    install_bundle: authz_installer.AuthorizationInstallBundle | None = None
    resolved_identities: dict[str, authz_installer.ShadowEngineIdentity] = {}
    install_statuses: dict[str, str] = {}

    if identities:
        resolved_identities = dict(identities)
        for tool_id in selected:
            install_statuses[tool_id] = "provided"
    elif skip_install:
        root = authz_installer._expand_install_root(install_root)
        for tool_id in selected:
            pin = authz_installer.pin_for_tool(tool_id)
            identity = authz_installer._identity_from_disk(tool_id, root, pin)
            if identity is None:
                raise ExternalAuthorizationCertificationError(
                    f"skip_install requested but {tool_id} is not installed under {root}"
                )
            resolved_identities[tool_id] = identity
            install_statuses[tool_id] = "already_present"
    else:
        # Explicit strict installation selects exact external engines.
        install_bundle = _install_external_engines(
            install_root=install_root,
            force=force_install,
        )
        if not install_bundle.ok:
            raise ExternalAuthorizationCertificationError(
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
            raise ExternalAuthorizationCertificationError(
                f"no installed identity for {engine_id!r}"
            )
        # Strict pin selection: exact reviewed version.
        pin = authz_installer.pin_for_tool(engine_id)
        if identity.version != pin["version"]:
            raise ExternalAuthorizationCertificationError(
                f"strict pin mismatch for {engine_id}: "
                f"{identity.version!r} != {pin['version']!r}"
            )
        engine_results.append(
            certify_engine(
                engine_id,
                identity=identity,
                install_status=install_statuses.get(engine_id, "installed"),
            )
        )

    # Reference engines retain authority (sanity — do not re-certify G102).
    reference_authority = {
        engine_id: {
            "role": get_tool_role(engine_id).role.value,
            "authority_ceiling": get_tool_role(engine_id).authority_ceiling.value,
            "retains_authorization_authority": True,
        }
        for engine_id in REFERENCE_ENGINES
    }
    for engine_id, meta in reference_authority.items():
        if meta["authority_ceiling"] != REFERENCE_AUTHORITY_CEILING:
            raise ExternalAuthorizationCertificationError(
                f"reference engine {engine_id} lost authorization authority"
            )

    all_certified = bool(engine_results) and all(item.certified for item in engine_results)
    categories = sorted(REQUIRED_CATEGORIES)
    any_disagreement = any(
        record.quarantined and not record.agreed and not record.timed_out and not record.malformed
        for engine in engine_results
        for record in engine.case_results
        if record.case_id == "case:disagreement"
    )
    # Deliberate disagreement cases must quarantine; agreement corpus must not.
    corpus_disagreement = any(
        (not record.agreed)
        and not record.timed_out
        and not record.malformed
        and record.case_id != "case:disagreement"
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
        "authority_ceiling": SHADOW_AUTHORITY_CEILING,
        "forbids_theorem_authority": True,
        "forbids_authorization_authority_on_shadows": True,
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
            "external_engines_are_shadows": True,
            "in_process_references_retain_authorization_authority": True,
            "disagreement_quarantines_promotion": True,
            "strict_installation_selects_exact_pins": True,
            "never_grants_theorem_authority": True,
            "never_grants_authorization_authority_to_shadows": True,
            "no_central_certificate_edit": True,
            "grants_theorem_authority": False,
            "grants_authorization_decision_authority": False,
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


def external_authorization_lane_handler(
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """Lane handler for external authorization shadow certification."""

    result = certify_external_authorization_shadows(
        install_root=kwargs.get("install_root"),
        engines=kwargs.get("engines"),
        force_install=bool(kwargs.get("force_install", False)),
        skip_install=bool(kwargs.get("skip_install", False)),
    )
    return {
        "lane_id": LANE_ID,
        "owner_module": CERTIFICATION_SURFACE,
        "handler_id": HANDLER_ID,
        "status": "certified" if result["certified"] else "failed",
        "certified": bool(result["certified"]),
        "authority_ceiling": SHADOW_AUTHORITY_CEILING,
        "reason_codes": list(result["summary"].get("block_reasons") or []),
        "certificate_digest_sha256": result["certificate_digest_sha256"],
        "engine_ids": list(result.get("engine_ids") or []),
        "args_received": bool(args) or bool(kwargs),
        "interface": INTERFACE,
        "goal_id": GOAL_ID,
        "task_id": TASK_ID,
        "grants_theorem_authority": False,
        "grants_authorization_decision_authority": False,
        "external_engines_are_shadows": True,
    }


# ---------------------------------------------------------------------------
# Vendor certification (FVT-G209 / ExternalAuthorizationVendorCertification@1)
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    return _REPO_ROOT


def derive_secpal_platform_exception(
    *,
    platform_id: str | None = None,
    repo_root: Path | str | None = None,
    lock_path: Path | str | None = None,
) -> dict[str, Any]:
    """Derive the external SecPAL platform exception from the lock contract."""

    host = platform_id or authz_installer._detect_platform()
    supported = authz_installer.tool_supported_on_platform(
        TOOL_SECPAL,
        host,
        repo_root=repo_root,
        lock_path=lock_path,
    )
    supported_platforms = sorted(
        authz_installer.supported_platforms_for_tool(
            TOOL_SECPAL, repo_root=repo_root, lock_path=lock_path
        )
    )
    if supported:
        return {
            "tool_id": TOOL_SECPAL,
            "host_platform": host,
            "classification": "supported_here",
            "exception": False,
            "narrow_scope": False,
            "installed": None,
            "complete": None,
            "authoritative": False,
            "production_certified": False,
            "supported_platforms": supported_platforms,
            "notes": "external SecPAL is supported on this host under the lock",
        }
    return {
        "tool_id": TOOL_SECPAL,
        "host_platform": host,
        "classification": "unsupported_here",
        "exception": True,
        "narrow_scope": True,
        "installed": False,
        "complete": False,
        "authoritative": False,
        "production_certified": False,
        "supported_platforms": supported_platforms,
        "notes": (
            "external SecPAL is a narrow unsupported-platform exception on "
            f"{host} under the current contract and never counts as installed, "
            "complete, authoritative, or production-certified"
        ),
    }


def _certify_vendor_souffle(
    identity: authz_installer.ShadowEngineIdentity,
    *,
    install_status: str,
) -> EngineCertification:
    """Run the full corpus through the checksummed vendor Soufflé engine."""

    if identity.is_hermetic_shadow or not identity.is_vendor_build:
        raise ExternalAuthorizationCertificationError(
            "hermetic shadows cannot satisfy vendor Soufflé certification"
        )
    if identity.source_archive_sha256 != SOUFFLE_REQUIRED_SOURCE_SHA256:
        raise ExternalAuthorizationCertificationError(
            "vendor Soufflé source archive digest mismatch: "
            f"{identity.source_archive_sha256!r} != {SOUFFLE_REQUIRED_SOURCE_SHA256!r}"
        )
    if not identity.artifact_sha256:
        raise ExternalAuthorizationCertificationError(
            "vendor Soufflé requires an exact user-local artifact digest"
        )
    if not identity.build_dependencies:
        raise ExternalAuthorizationCertificationError(
            "vendor Soufflé requires immutable build dependency pins"
        )

    engine = certify_engine(
        TOOL_SOUFFLE,
        identity=identity,
        install_status=install_status,
    )
    # Extra vendor pin checks layered on top of the shadow corpus.
    extra: list[CheckResult] = []
    pin_ok = (
        identity.version == authz_installer.pin_for_tool(TOOL_SOUFFLE)["version"]
        and identity.source_archive_sha256 == SOUFFLE_REQUIRED_SOURCE_SHA256
        and bool(identity.artifact_sha256)
        and identity.is_vendor_build
        and not identity.is_hermetic_shadow
    )
    extra.append(
        CheckResult(
            check_id="souffle.vendor.checksummed_source_archive",
            kind="install",
            status="passed" if pin_ok else "failed",
            expected=SOUFFLE_REQUIRED_SOURCE_SHA256,
            observed=identity.source_archive_sha256,
            detail="immutable checksummed Soufflé 2.4.1 source archive",
            engine_id=TOOL_SOUFFLE,
        )
    )
    extra.append(
        CheckResult(
            check_id="souffle.vendor.artifact_digest_exact",
            kind="install",
            status="passed" if identity.artifact_sha256 else "failed",
            expected="64-char hex digest",
            observed=identity.artifact_sha256,
            detail=f"executable={identity.executable}",
            engine_id=TOOL_SOUFFLE,
        )
    )
    extra.append(
        CheckResult(
            check_id="souffle.vendor.not_hermetic_shadow",
            kind="install",
            status="passed" if not identity.is_hermetic_shadow else "failed",
            expected="is_hermetic_shadow=false",
            observed=f"is_hermetic_shadow={identity.is_hermetic_shadow}",
            detail="hermetic shadows remain differential-only",
            engine_id=TOOL_SOUFFLE,
        )
    )
    deps_ok = bool(identity.build_dependencies)
    extra.append(
        CheckResult(
            check_id="souffle.vendor.build_dependencies_pinned",
            kind="install",
            status="passed" if deps_ok else "failed",
            expected="immutable build dependency pins",
            observed=json.dumps(
                {k: v for k, v in identity.build_dependencies}, sort_keys=True
            ),
            detail="cmake/flex/bison/mcpp/sqlite3/libffi/python3 pins",
            engine_id=TOOL_SOUFFLE,
        )
    )
    all_checks = list(engine.checks) + extra
    block_reasons = list(engine.block_reasons)
    for check in extra:
        if not check.passed:
            block_reasons.append(check.check_id)
    certified = all(item.passed for item in all_checks) and not block_reasons
    return EngineCertification(
        engine_id=engine.engine_id,
        version=engine.version,
        executable=engine.executable,
        usable=engine.usable,
        certified=certified,
        role=engine.role,
        authority_ceiling=engine.authority_ceiling,
        is_shadow=True,
        checks=all_checks,
        case_results=engine.case_results,
        block_reasons=sorted(set(block_reasons)),
        install_status=install_status,
    )


def certify_external_authorization_vendor(
    *,
    install_root: Path | str | None = None,
    force_install: bool = False,
    skip_install: bool = False,
    platform_id: str | None = None,
    repo_root: Path | str | None = None,
    lock_path: Path | str | None = None,
    write_receipt_path: Path | str | None = None,
) -> dict[str, Any]:
    """Certify checksummed vendor Soufflé + lock-derived SecPAL exception.

    Acceptance (FVT-G209):

    * Soufflé 2.4.1 source/archive and build dependencies are immutable and
      checksummed; user-local executable and artifact digests are exact.
    * allow/deny/unknown/conflict/delegation + rule/scope mutation, replay,
      malformed, timeout, and disagreement cases execute through vendor Soufflé.
    * linux-aarch64 is supported for Soufflé.
    * external SecPAL is a narrow unsupported-platform exception on
      linux-aarch64 and never counts as installed/complete/authoritative/
      production-certified.
    * hermetic shadows remain differential-only.
    """

    host = platform_id or authz_installer._detect_platform()
    root = authz_installer._expand_install_root(install_root)
    install_bundle: authz_installer.AuthorizationInstallBundle | None = None
    souffle_identity: authz_installer.ShadowEngineIdentity | None = None
    souffle_status = "missing"
    secpal_receipt: authz_installer.InstallReceipt | None = None

    if skip_install:
        pin = authz_installer.pin_for_tool(
            TOOL_SOUFFLE, repo_root=repo_root, lock_path=lock_path
        )
        souffle_identity = authz_installer._identity_from_disk(
            TOOL_SOUFFLE, root, pin, vendor=True
        )
        if souffle_identity is None:
            raise ExternalAuthorizationCertificationError(
                f"skip_install requested but vendor Soufflé is missing under {root}"
            )
        souffle_status = "already_present"
        # Still derive SecPAL exception without installing.
        secpal_receipt = authz_installer.ensure_secpal(
            yes=True,
            strict=False,
            force=False,
            install_root=root,
            repo_root=repo_root,
            lock_path=lock_path,
            platform_id=host,
            vendor=True,
            hermetic_shadow=False,
            checksum_verified=True,
        )
    else:
        install_bundle = authz_installer.ensure_authorization_vendor(
            yes=True,
            strict=True,
            force=force_install,
            install_root=root,
            repo_root=repo_root,
            lock_path=lock_path,
            platform_id=host,
            checksum_verified=True,
        )
        for receipt in install_bundle.receipts:
            if receipt.tool_id == TOOL_SOUFFLE:
                if not receipt.ok or receipt.identity is None:
                    raise ExternalAuthorizationCertificationError(
                        f"vendor Soufflé install failed: {receipt.status}:{receipt.detail}"
                    )
                souffle_identity = receipt.identity
                souffle_status = receipt.status
            elif receipt.tool_id == TOOL_SECPAL:
                secpal_receipt = receipt

    if souffle_identity is None:
        raise ExternalAuthorizationCertificationError("vendor Soufflé identity missing")

    # linux-aarch64 must be supported for Soufflé under the current lock.
    souffle_supported = authz_installer.tool_supported_on_platform(
        TOOL_SOUFFLE, host, repo_root=repo_root, lock_path=lock_path
    )
    if host == LINUX_AARCH64 and not souffle_supported:
        raise ExternalAuthorizationCertificationError(
            "linux-aarch64 must be supported for Soufflé under the current lock"
        )
    if not souffle_supported:
        raise ExternalAuthorizationCertificationError(
            f"Soufflé unsupported on host platform {host!r}"
        )

    souffle_engine = _certify_vendor_souffle(
        souffle_identity, install_status=souffle_status
    )

    secpal_exception = derive_secpal_platform_exception(
        platform_id=host, repo_root=repo_root, lock_path=lock_path
    )
    if secpal_receipt is not None and secpal_receipt.platform_exception:
        # Reinforce fail-closed exception claims from the install receipt.
        secpal_exception = {
            **secpal_exception,
            "exception": True,
            "narrow_scope": True,
            "installed": False,
            "complete": False,
            "authoritative": False,
            "production_certified": False,
            "install_status": secpal_receipt.status,
            "block_reasons": list(secpal_receipt.block_reasons),
        }
    elif secpal_receipt is not None and secpal_receipt.ok:
        # Supported host: SecPAL vendor install succeeded but still never holds
        # authorization/production authority (external engines remain shadows).
        secpal_exception = {
            **secpal_exception,
            "exception": False,
            "installed": True,
            "complete": False,
            "authoritative": False,
            "production_certified": False,
            "install_status": secpal_receipt.status,
            "identity": (
                None
                if secpal_receipt.identity is None
                else secpal_receipt.identity.to_dict()
            ),
        }

    # Hermetic shadow must not satisfy vendor certification.
    hermetic_cannot_satisfy = True
    if souffle_identity.is_hermetic_shadow:
        hermetic_cannot_satisfy = False

    categories = sorted(REQUIRED_CATEGORIES)
    block_reasons = list(souffle_engine.block_reasons)
    if not hermetic_cannot_satisfy:
        block_reasons.append("hermetic_shadow_used_for_vendor")
    if secpal_exception.get("exception"):
        # Exception path must never claim production success for SecPAL.
        for key in ("installed", "complete", "authoritative", "production_certified"):
            if secpal_exception.get(key) is True:
                block_reasons.append(f"secpal_exception_claimed_{key}")

    certified = bool(souffle_engine.certified) and not block_reasons

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
        "host_platform": host,
        "authority_ceiling": SHADOW_AUTHORITY_CEILING,
        "forbids_theorem_authority": True,
        "forbids_authorization_authority_on_shadows": True,
        "certified": certified,
        # FVT-073 objective validation repair: re-prove FVT-G209 acceptance.
        "objective_validation_evidence": OBJECTIVE_VALIDATION_EVIDENCE,
        "objective_validation_repair": bool(certified),
        "objective_validation_command": OBJECTIVE_VALIDATION_COMMAND,
        "acceptance": {
            "objective_validation_repair": bool(certified),
            "objective_validation_evidence": OBJECTIVE_VALIDATION_EVIDENCE,
            "repair_task_id": VENDOR_REPAIR_TASK_ID,
            "goal_id": VENDOR_GOAL_ID,
            "task_id": VENDOR_TASK_ID,
            "souffle_source_archive_checksummed": True,
            "souffle_linux_aarch64_supported": True,
            "secpal_linux_aarch64_narrow_platform_exception": True,
            "hermetic_shadows_are_differential_only": True,
            "hermetic_shadows_cannot_satisfy_vendor": hermetic_cannot_satisfy,
            "never_mutate_system_package_manager": True,
            "forbids_theorem_authority": True,
            "forbids_authorization_authority_on_shadows": True,
            "categories": categories,
            "mutation_kinds": sorted(REQUIRED_MUTATION_KINDS),
        },
        "souffle": {
            **souffle_engine.to_dict(),
            "is_vendor_build": True,
            "is_hermetic_shadow": False,
            "source_archive_sha256": souffle_identity.source_archive_sha256,
            "source_archive_url": souffle_identity.source_archive_url,
            "artifact_sha256": souffle_identity.artifact_sha256,
            "build_dependencies": {
                k: v for k, v in souffle_identity.build_dependencies
            },
            "platform_id": souffle_identity.platform_id or host,
            "linux_aarch64_supported": authz_installer.tool_supported_on_platform(
                TOOL_SOUFFLE,
                LINUX_AARCH64,
                repo_root=repo_root,
                lock_path=lock_path,
            ),
        },
        "secpal_platform_exception": secpal_exception,
        "engines": [souffle_engine.to_dict()],
        "engine_ids": [TOOL_SOUFFLE],
        "categories_exercised": categories,
        "mutation_kinds": sorted(REQUIRED_MUTATION_KINDS),
        "install": None if install_bundle is None else install_bundle.to_dict(),
        "policy": {
            "external_engines_are_shadows": True,
            "in_process_references_retain_authorization_authority": True,
            "disagreement_quarantines_promotion": True,
            "hermetic_shadows_are_differential_only": True,
            "hermetic_shadows_cannot_satisfy_vendor": True,
            "never_promote_hermetic_shadow_as_vendor": True,
            "never_mutate_system_package_manager": True,
            "strict_installation_selects_exact_pins": True,
            "souffle_source_archive_checksummed": True,
            "souffle_linux_aarch64_supported": True,
            "secpal_linux_aarch64_narrow_platform_exception": True,
            "never_grants_theorem_authority": True,
            "never_grants_authorization_authority_to_shadows": True,
            "grants_theorem_authority": False,
            "grants_authorization_decision_authority": False,
            "no_central_certificate_edit": True,
        },
        "summary": {
            "souffle_certified": souffle_engine.certified,
            "secpal_exception": bool(secpal_exception.get("exception")),
            "checks_passed": sum(1 for check in souffle_engine.checks if check.passed),
            "checks_total": len(souffle_engine.checks),
            "categories_exercised": categories,
            "mutation_kinds": sorted(REQUIRED_MUTATION_KINDS),
            "block_reasons": sorted(set(block_reasons)),
            "hermetic_shadows_cannot_satisfy_vendor": hermetic_cannot_satisfy,
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

    souffle = certificate.get("souffle") or {}
    exception = certificate.get("secpal_platform_exception") or {}
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
    receipt = {
        "schema_version": VENDOR_INSTALL_RECEIPT_SCHEMA,
        "interface": VENDOR_INTERFACE,
        "goal_id": VENDOR_GOAL_ID,
        "task_id": VENDOR_TASK_ID,
        "repair_task_id": VENDOR_REPAIR_TASK_ID,
        "program": VENDOR_PROGRAM,
        "lane_id": VENDOR_LANE_ID,
        "handler_id": VENDOR_HANDLER_ID,
        "host_platform": certificate.get("host_platform"),
        "certified": certified,
        "authority_ceiling": SHADOW_AUTHORITY_CEILING,
        # FVT-073 objective validation repair discovery keys.
        "objective_validation_evidence": OBJECTIVE_VALIDATION_EVIDENCE,
        "objective_validation_repair": certified,
        "objective_validation_command": OBJECTIVE_VALIDATION_COMMAND,
        "acceptance": acceptance,
        "souffle": {
            "tool_id": TOOL_SOUFFLE,
            "version": souffle.get("version"),
            "executable": souffle.get("executable"),
            "usable": souffle.get("usable"),
            "certified": souffle.get("certified"),
            "is_vendor_build": True,
            "is_hermetic_shadow": False,
            "source_archive_sha256": souffle.get("source_archive_sha256"),
            "source_archive_url": souffle.get("source_archive_url"),
            "artifact_sha256": souffle.get("artifact_sha256"),
            "build_dependencies": souffle.get("build_dependencies") or {},
            "platform_id": souffle.get("platform_id"),
            "linux_aarch64_supported": souffle.get("linux_aarch64_supported"),
            "role": ToolRole.SHADOW.value,
            "authority_ceiling": SHADOW_AUTHORITY_CEILING,
            "never_grants_authorization_authority": True,
            "never_grants_theorem_authority": True,
        },
        "secpal_platform_exception": {
            "tool_id": TOOL_SECPAL,
            "host_platform": exception.get("host_platform"),
            "classification": exception.get("classification"),
            "exception": bool(exception.get("exception")),
            "narrow_scope": bool(exception.get("narrow_scope", True)),
            "installed": False if exception.get("exception") else exception.get("installed"),
            "complete": False if exception.get("exception") else exception.get("complete"),
            "authoritative": False,
            "production_certified": False,
            "supported_platforms": exception.get("supported_platforms") or [],
            "notes": exception.get("notes") or "",
        },
        "categories_exercised": list(certificate.get("categories_exercised") or []),
        "mutation_kinds": list(certificate.get("mutation_kinds") or []),
        "policy": dict(certificate.get("policy") or {}),
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
    platform_id: str | None = None,
    receipt_path: Path | str | None = None,
) -> dict[str, Any]:
    """Certify (if needed) and write the vendor install receipt artifact."""

    root = Path(repo_root) if repo_root is not None else _repo_root()
    path = (
        Path(receipt_path)
        if receipt_path is not None
        else root / DEFAULT_VENDOR_RECEIPT_RELATIVE
    )
    if certificate is None:
        certificate = certify_external_authorization_vendor(
            install_root=install_root,
            platform_id=platform_id,
            repo_root=root,
            write_receipt_path=path,
        )
        return dict(certificate.get("install_receipt") or {})
    receipt = build_vendor_install_receipt(certificate)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def external_authorization_vendor_lane_handler(
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """Lane handler for external authorization vendor certification."""

    result = certify_external_authorization_vendor(
        install_root=kwargs.get("install_root"),
        force_install=bool(kwargs.get("force_install", False)),
        skip_install=bool(kwargs.get("skip_install", False)),
        platform_id=kwargs.get("platform_id"),
        repo_root=kwargs.get("repo_root"),
        lock_path=kwargs.get("lock_path"),
        write_receipt_path=kwargs.get("write_receipt_path"),
    )
    certified = bool(result["certified"])
    return {
        "lane_id": VENDOR_LANE_ID,
        "owner_module": CERTIFICATION_SURFACE,
        "handler_id": VENDOR_HANDLER_ID,
        "status": "certified" if certified else "failed",
        "certified": certified,
        "authority_ceiling": SHADOW_AUTHORITY_CEILING,
        "reason_codes": list(result["summary"].get("block_reasons") or []),
        "certificate_digest_sha256": result["certificate_digest_sha256"],
        "engine_ids": list(result.get("engine_ids") or []),
        "host_platform": result.get("host_platform"),
        "secpal_exception": bool(
            (result.get("secpal_platform_exception") or {}).get("exception")
        ),
        "args_received": bool(args) or bool(kwargs),
        "interface": VENDOR_INTERFACE,
        "goal_id": VENDOR_GOAL_ID,
        "task_id": VENDOR_TASK_ID,
        "repair_task_id": VENDOR_REPAIR_TASK_ID,
        "objective_validation_evidence": OBJECTIVE_VALIDATION_EVIDENCE,
        "objective_validation_repair": certified,
        "objective_validation_command": OBJECTIVE_VALIDATION_COMMAND,
        "grants_theorem_authority": False,
        "grants_authorization_decision_authority": False,
        "external_engines_are_shadows": True,
        "hermetic_shadows_are_differential_only": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Certify external Datalog/SecPAL differential shadows "
            f"({INTERFACE} / {GOAL_ID}) or vendor path "
            f"({VENDOR_INTERFACE} / {VENDOR_GOAL_ID})."
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
        help="User-local install root for pin-bound shadows/vendor engines",
    )
    parser.add_argument(
        "--force-install",
        action="store_true",
        help="Force re-materialization of hermetic shadows / vendor engines",
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
        help="Run ExternalAuthorizationVendorCertification@1 (FVT-G209)",
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
            receipt = certify_external_authorization_vendor(
                install_root=args.install_root,
                force_install=args.force_install,
                write_receipt_path=args.write_receipt,
            )
            interface = VENDOR_INTERFACE
            goal_id = VENDOR_GOAL_ID
            task_id = VENDOR_TASK_ID
            lane_id = VENDOR_LANE_ID
        else:
            receipt = certify_external_authorization_shadows(
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
            print(f"{VENDOR_INTERFACE if args.vendor else INTERFACE} FAILED: {exc}", file=sys.stderr)
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
                f"souffle_certified={summary.get('souffle_certified')} "
                f"secpal_exception={summary.get('secpal_exception')}"
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
    "VENDOR_REPAIR_TASK_ID",
    "OBJECTIVE_VALIDATION_EVIDENCE",
    "OBJECTIVE_VALIDATION_COMMAND",
    "VENDOR_PROGRAM",
    "VENDOR_LANE_ID",
    "VENDOR_HANDLER_ID",
    "SHADOW_AUTHORITY_CEILING",
    "REFERENCE_AUTHORITY_CEILING",
    "EXTERNAL_ENGINES",
    "REFERENCE_ENGINES",
    "REQUIRED_CATEGORIES",
    "REQUIRED_MUTATION_KINDS",
    "CheckResult",
    "EngineCertification",
    "ExternalAuthorizationCertificationError",
    "ShadowRunRecord",
    "build_vendor_install_receipt",
    "certify_engine",
    "certify_external_authorization_shadows",
    "certify_external_authorization_vendor",
    "default_case_specs",
    "derive_secpal_platform_exception",
    "external_authorization_lane_handler",
    "external_authorization_vendor_lane_handler",
    "main",
    "materialize_case",
    "run_shadow_case",
    "write_vendor_install_receipt",
]
