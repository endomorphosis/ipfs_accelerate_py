"""Explicit reviewed spectral candidate; native validation retains authority.

Only the original signed public instruction and eigen.py preimage select this
operator. Target-interpreter finite checks and conditional ideal-algebra SMT
lemmas are distinct evidence; neither is a LAPACK/IEEE or performance proof.
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

from ..proof.formal_verification_contracts import content_identity
from . import local_planning_admission as local
from .doctor_contract_candidate_runner import SCHEMA, publish_doctor_contract_candidate
from .doctor_scoped_analysis import build_scoped_doctor_analysis
from .terminal_source_partition import _read
from .terminal_task_profile import INSTRUCTION

PROFILE = "dominant-eigenpair-f64-small@1"
BACKEND = "scipy_dgeev"
INSTRUCTION_SHA256 = "6fa938389f99648508e5a9772e0d5e95d1102738c68d1ff61bd65469d76c52ab"
PREIMAGE_SHA256 = "bbd5be97fb11e9ed46fd5f1fb9d407efb19b265b549d3c36e2cfba923efe3d9d"
WORKFLOW_SCHEMA = "supervisor-spectral-contract-workflow@1"
CHECK_SCOPE = (
    "Conditional ideal-algebra SMT adapter lemmas and fixed finite numerical "
    "checks of the reviewed spectral operator in the explicitly selected target "
    "interpreter. Not a kernel theorem, LAPACK/IEEE correctness proof, "
    "all-matrix guarantee, prose-alignment proof or performance proof."
)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _snapshot(value):
    from ipfs_accelerate_py.cli_runtime.grok_structured_output import _bounded_json
    if type(value) is not dict:
        raise ValueError("bounded spectral qualification mapping required")
    value = json.loads(_bounded_json(value, 65_536))
    for key in ("proof_authority", "execution_authority", "publication_authority",
                "completion_authority", "kernel_proved", "whole_program_proved", "performance_proved"):
        if key in value and value[key] is not False:
            raise ValueError("spectral qualification carries unsupported authority")
    return value


def _prove(source_sha256):
    from ..analysis.spectral_kernel_proof import prove_spectral_adapter_lemmas
    return prove_spectral_adapter_lemmas(kernel_source_sha256=source_sha256, timeout_ms=2000)


def _qualify_target(interpreter):
    from .spectral_target_qualification import qualify_spectral_target_runtime
    return qualify_spectral_target_runtime(interpreter=interpreter, timeout_seconds=30)


def _require_target_binding(report, *, interpreter, executable, source_sha256):
    from .spectral_target_qualification import expected_probe_sha256
    if (report.get("schema") != "spectral-target-numeric-qualification@1"
            or report.get("source_sha256") != source_sha256
            or report.get("interpreter") != str(interpreter) or report.get("executable") != str(executable)
            or report.get("executable_sha256") != _sha(executable.read_bytes())
            or report.get("target_probe_sha256") != expected_probe_sha256()
            or type(report.get("case_count")) is not int or report["case_count"] != 22
            or any(type(report.get(key)) is not str or not 1 <= len(report[key]) <= 128
                   for key in ("python_version", "numpy_version", "scipy_version"))
            or report.get("backend") != BACKEND or type(report.get("provider_calls")) is not int
            or report["provider_calls"] != 0
            or any(report.get(key) is not False for key in (
                "proof_authority", "publication_authority", "completion_authority", "kernel_proved", "timing_qualified"))):
        raise ValueError("affirmative target qualification binding differs")


def _require_proof_binding(report, *, source_sha256):
    from ..analysis import spectral_kernel_proof
    checks = report.get("checks")
    names = spectral_kernel_proof.EXPECTED_LEMMA_NAMES
    denials = ("python_program_mechanically_verified", "lapack_implementation_proved",
        "ieee_floating_point_accuracy_proved", "timing_proved", "kernel_proved", "whole_program_proved",
        "proof_authority", "publication_authority", "completion_authority")
    if (report.get("schema") != "spectral-adapter-smt-lemmas@1"
            or report.get("status") != "verified_scoped_lemmas"
            or report.get("scope") != "conditional ideal complex-vector algebra and bounded computed-magnitude selection"
            or report.get("assumptions") != [
                "The trusted solver supplies a complete numerical spectrum.",
                "Packed real columns satisfy the declared ideal backend eigenvector equations.",
                "Selector inputs are finite nonnegative computed magnitudes."]
            or report.get("kernel_source_sha256") != source_sha256
            or report.get("analysis_source_sha256") != _sha(Path(spectral_kernel_proof.__file__).read_bytes())
            or type(report.get("provider_calls")) is not int or report["provider_calls"] != 0
            or any(report.get(key) is not False for key in denials)
            or type(checks) is not list or len(checks) != len(names)
            or any(type(row) is not dict for row in checks)
            or [row.get("name") for row in checks] != list(names)):
        raise ValueError("conditional algebra source or check population differs")
    for row in checks:
        digest = row.get("smtlib_sha256")
        if (row.get("verdict") != "unsat" or row.get("verified") is not True
                or row.get("unknown_reason") is not None or type(row.get("smtlib_bytes")) is not int
                or not 1 <= row["smtlib_bytes"] <= 65_536 or type(digest) is not str or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)):
            raise ValueError("conditional algebra check is not verified UNSAT")


def prepare_spectral_kernel_candidate(*, repository: Path, admission: dict, intent,
        task_cid: str, state: Path, profile: str | None = PROFILE,
        target_interpreter: Path | None = None) -> dict:
    """Propose one checked eigen.py modification without changing task/source.

    Unsupported profile, missing backend and non-affirmative solver results
    stay residual with no model fallback. Admission/source/task drift or forged
    affirmative evidence refuses the handoff. No task module is imported.
    """
    repository, state = Path(repository).absolute(), Path(state).absolute()
    if profile is not None and type(profile) is not str:
        raise ValueError("explicit reviewed spectral profile required")
    if state.resolve() != state or state.is_relative_to(repository) or state.exists():
        raise ValueError("spectral workflow requires a new external state directory")
    verified = local.verify_local_benchmark_admission(admission, initial=True)
    manifest = verified["manifest"]
    if (repository.resolve(strict=True) != repository or str(repository) != manifest["repository"]
            or [task.task_cid for task in verified["graph"].tasks] != [task_cid]):
        raise ValueError("spectral workflow requires the exact single admitted task")
    task = intent.get_task(task_cid)
    if task is None or task["status"] not in {"ready", "in_progress"}:
        raise ValueError("spectral workflow requires the current active task")
    contract, _, _, _ = local._contract(task["body"], task_cid)
    spec = contract["task_spec"]
    if (contract["manifest_cid"] != content_identity(admission["manifest"])
            or manifest["tasks"] != [spec]):
        raise ValueError("spectral task differs from independently signed admission")
    scoped = build_scoped_doctor_analysis(repository=repository, admission=admission, task_cid=task_cid)

    def assert_current():
        scoped.assert_current()
        if intent.get_task(task_cid) != task:
            raise ValueError("task revision changed during spectral qualification")

    result = {"schema": WORKFLOW_SCHEMA, "status": "residual", "repository": str(repository),
        "task_cid": task_cid, "task_revision": task["revision"], "profile": profile,
        "analysis": scoped.report, "provider_calls": 0, "canonical_source_edits": 0,
        "proof_authority": False, "publication_authority": False, "completion_authority": False,
        "kernel_proved": False, "whole_program_proved": False, "performance_proved": False,
        "performance_qualified": False,
        "evidence_kind": "conditional_ideal_algebra_and_finite_numeric_check",
        "check_scope": CHECK_SCOPE, "target_dependency_qualified": False,
        "reason_codes": []}
    state.mkdir(parents=True, mode=0o700)

    def finish(reason=None):
        if reason is not None:
            result["reason_codes"] = [reason]
        assert_current()
        with (state / "result.json").open("x") as stream:
            stream.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
        return result

    if profile != PROFILE:
        return finish("spectral_profile_not_selected")
    if (spec["outputs"] != [{"path": "eigen.py", "effect": "modify", "media_type": "text/x-python"}]
            or "eigen.py" not in scoped.sources):
        return finish("spectral_output_scope_unsupported")
    sources = manifest["sources"]
    if (sources.get(INSTRUCTION, {}).get("sha256") != INSTRUCTION_SHA256
            or sources.get("eigen.py", {}).get("sha256") != PREIMAGE_SHA256):
        return finish("spectral_public_source_profile_unreviewed")
    instruction = _read(repository, INSTRUCTION, sources, 32_768)
    before = _read(repository, "eigen.py", sources, 65_536)
    if _sha(instruction) != INSTRUCTION_SHA256 or before != scoped.sources["eigen.py"]:
        raise ValueError("spectral public input differs from signed scope")
    if target_interpreter is None:
        return finish("target_runtime_unqualified")
    selected = Path(target_interpreter)
    if not selected.is_absolute() or ".." in selected.parts:
        raise ValueError("explicit absolute target interpreter required")
    try:
        resolved_interpreter = selected.resolve(strict=True)
    except (OSError, RuntimeError):
        return finish("target_interpreter_unavailable")
    if not resolved_interpreter.is_file():
        return finish("target_interpreter_unavailable")
    from . import spectral_eigen_kernel as kernel
    source = kernel.candidate_source(backend=BACKEND)
    if type(source) is not str or not source or len(source.encode()) > 65_536:
        raise ValueError("bounded trusted spectral candidate source required")
    after = source.encode("utf-8")
    source_sha256 = _sha(after)
    result.update(operator="reviewed-dominant-eigenpair-scipy-dgeev@1", backend=BACKEND,
                  kernel_source_sha256=source_sha256, target_interpreter=str(selected),
                  resolved_target_interpreter=str(resolved_interpreter))
    target = _snapshot(_qualify_target(selected))
    result["target_qualification"] = target
    assert_current()
    if target.get("accepted") is not True:
        return finish("target_runtime_unqualified")
    _require_target_binding(target, interpreter=selected, executable=resolved_interpreter, source_sha256=source_sha256)
    result["target_dependency_qualified"] = True
    proof = _snapshot(_prove(source_sha256))
    result["conditional_algebra"] = proof
    assert_current()
    if proof.get("all_lemmas_verified") is not True:
        return finish("spectral_conditional_algebra_unverified")
    _require_proof_binding(proof, source_sha256=source_sha256)
    if kernel.candidate_source(backend=BACKEND).encode("utf-8") != after:
        raise ValueError("trusted spectral candidate changed during qualification")
    binding = {"schema": "supervisor-spectral-check-binding@1",
        "manifest_cid": scoped.report["manifest_cid"], "analysis_cid": scoped.report["analysis_cid"],
        "task_cid": task_cid, "task_revision": task["revision"],
        "task_spec_cid": content_identity(spec), "profile": PROFILE,
        "instruction_sha256": INSTRUCTION_SHA256, "before_sha256": PREIMAGE_SHA256,
        "kernel_source_sha256": source_sha256, "target_interpreter": str(selected),
        "target_qualification_sha256": _sha(json.dumps(target, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()),
        "conditional_algebra_sha256": _sha(json.dumps(proof, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()),
        "workflow_sha256": _sha(Path(__file__).read_bytes()),
        "proof_authority": False, "publication_authority": False, "completion_authority": False}
    binding["check_receipt_id"] = content_identity(binding)
    result["check"] = binding
    if _read(repository, "eigen.py", sources, 65_536) != before:
        raise ValueError("spectral preimage changed before candidate handoff")
    _require_target_binding(target, interpreter=selected, executable=selected.resolve(strict=True), source_sha256=source_sha256)
    _require_proof_binding(proof, source_sha256=source_sha256)
    assert_current()
    edit = {"path": "eigen.py", "effect": "modify", "before_sha256": PREIMAGE_SHA256,
            "after_sha256": source_sha256, "after_bytes_base64": base64.b64encode(after).decode()}
    payload = {"schema": SCHEMA, "repository": str(repository),
        "baseline_commit": manifest["baseline_commit"], "task_cid": task_cid,
        "task_id": task["task_alias"], "task_revision": task["revision"],
        "manifest_cid": scoped.report["manifest_cid"], "proof_receipt_id": binding["check_receipt_id"],
        "proof_scope": CHECK_SCOPE, "analysis_cid": scoped.report["analysis_cid"],
        "edits": [edit], "permitted_outputs": spec["outputs"], "provider_calls": 0,
        "publication_authority": False, "completion_authority": False}
    artifact, digest, cid = publish_doctor_contract_candidate(repository, payload)
    result.update(status="candidate_ready", route="doctor_contract_candidate", artifact=str(artifact),
                  sha256=digest, artifact_cid=cid)
    return finish()
