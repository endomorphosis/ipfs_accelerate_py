"""Scoped local-contract repair through the ordinary admitted worker boundary.

The closed AST producer constructs a candidate; real native Tactician and
Hammer verify its local contract. A pinned, inert artifact delegates those
exact candidate bytes to the already-authorized task worker. Validation and
publication still belong to the supervisor. The whole-program Doctor impact
gate is not weakened or represented as closed by this separate local route.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat

from ..analysis.doctor_header_contracts import (
    WsgiHeaderProtocolContract, analyze_http_header_contracts, verify_header_candidate,
)
from ..proof.formal_verification_contracts import content_identity
from . import local_planning_admission as local
from .doctor_contract_candidate_runner import SCHEMA
from .doctor_contract_proof import PROOF_SCOPE, persist_contract_world, prove_header_contract
from .doctor_scoped_analysis import build_scoped_doctor_analysis
from .doctor_security_ir import compile_header_security_ir


@dataclass(frozen=True)
class CweReportOutput:
    """An explicit task-output projection, separate from the security theorem."""

    path: str
    logical_root: str

    def __post_init__(self):
        path, root = PurePosixPath(self.path), PurePosixPath(self.logical_root)
        if (not self.path or path.is_absolute() or str(path) != self.path
                or any(part in {"..", ".git", ".runtime"} for part in path.parts)
                or not root.is_absolute() or ".." in root.parts or str(root) != self.logical_root):
            raise ValueError("exact relative report path and absolute logical source root required")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _publish(repository, payload):
    """Publish only exact public candidate bytes to an owner-controlled locator."""
    current = repository
    for name in (".runtime", "doctor-contract-candidates"):
        current = current / name
        current.mkdir(mode=0o755, exist_ok=True)
        info = current.lstat()
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) & 0o022 or stat.S_IMODE(info.st_mode) & 0o005 != 0o005):
            raise ValueError("candidate artifact directory is not owner-controlled and worker-readable")
    payload = {**payload, "artifact_cid": content_identity(payload)}
    raw = json.dumps(payload, sort_keys=True, indent=2).encode() + b"\n"
    digest = _sha(raw)
    artifact = current / (digest + ".json")
    with artifact.open("xb") as stream:
        stream.write(raw)
        os.fchmod(stream.fileno(), 0o444)
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(current, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return artifact, digest, payload["artifact_cid"]


def prepare_header_contract_repair(*, repository: Path, admission: dict, intent,
                                  task_cid: str, state: Path,
                                  protocol: WsgiHeaderProtocolContract,
                                  lean: Path, z3: Path,
                                  report_output: CweReportOutput | None = None) -> dict:
    """Select from every admitted Python modification; abstain on ambiguity.

    Report rendering is available only with an explicit caller-owned output
    declaration matching the independently admitted create output. It does not
    interpret a natural-language prompt or manufacture acceptance evidence.
    """
    if type(protocol) is not WsgiHeaderProtocolContract:
        raise ValueError("explicit reviewed protocol required")
    repository, state = Path(repository).absolute(), Path(state).absolute()
    if state.resolve() != state or state.is_relative_to(repository) or state.exists():
        raise ValueError("header workflow state must be a new external directory")
    scoped = build_scoped_doctor_analysis(repository=repository, admission=admission, task_cid=task_cid)
    verified = local.verify_local_benchmark_admission(admission, initial=True)
    task = intent.get_task(task_cid)
    if task is None or task["status"] not in {"ready", "in_progress"}:
        raise ValueError("header repair requires the current active admitted task")
    contract, _, _, _ = local._contract(task["body"], task_cid)
    spec = contract["task_spec"]
    if (contract["manifest_cid"] != scoped.report["manifest_cid"]
            or content_identity(spec) != scoped.report["task_spec_cid"]):
        raise ValueError("header repair task differs from independent admission")

    def assert_current():
        scoped.assert_current()
        if intent.get_task(task_cid) != task:
            raise ValueError("task revision changed during local contract repair")

    analyses, candidates = [], []
    for output in spec["outputs"]:
        name = output["path"]
        if output["effect"] != "modify" or not name.endswith(".py"):
            continue
        if name not in scoped.sources:
            raise ValueError("declared Python output is absent from admitted analysis")
        analysis = analyze_http_header_contracts(scoped.sources[name].decode("utf-8"), protocol=protocol)
        # No source or replacement bodies in world memory or routing receipts.
        analyses.append({"path": name, "status": analysis.status, "reason_codes": list(analysis.reason_codes),
            "source_sha256": analysis.source_sha256, "contracts": list(analysis.contracts),
            "open_frontiers": list(analysis.open_frontiers)})
        if analysis.candidate is not None:
            candidates.append((name, analysis.candidate))
    state.mkdir(parents=True, mode=0o700)
    formalization = compile_header_security_ir(scoped=scoped, analyses=analyses, output=state / "security-ir")
    result = {"schema": "supervisor-header-contract-workflow@1", "status": "residual",
        "repository": str(repository),
        "task_cid": task_cid, "task_revision": task["revision"], "analysis": scoped.report,
        "analyses": analyses, "provider_calls": 0, "canonical_source_edits": 0,
        "publication_authority": False, "completion_authority": False,
        "whole_program_proved": False, "benchmark_informed_development": True,
        "proof_scope": PROOF_SCOPE, "security_ir": formalization.report}
    proof = None
    proof_report = {"status": "unsupported", "reason_codes": [], "provider_calls": 0}
    edits = []
    if len(candidates) != 1:
        proof_report["reason_codes"] = ["ambiguous_header_candidates" if candidates else "no_supported_header_candidate"]
    else:
        path, candidate = candidates[0]
        if not verify_header_candidate(scoped.sources[path].decode("utf-8"), candidate):
            raise ValueError("header candidate failed independent exact AST reconstruction")
        candidate_cid = content_identity(candidate.to_dict())
        edits = [{"path": path, "effect": "modify", "before_sha256": candidate.before_sha256,
            "after_sha256": candidate.after_sha256,
            "after_bytes_base64": base64.b64encode(candidate.source.encode()).decode()}]
        if report_output is not None:
            if type(report_output) is not CweReportOutput:
                raise ValueError("typed explicit report projection required")
            expected_output = {"path": report_output.path, "effect": "create", "media_type": "text/plain"}
            if expected_output not in spec["outputs"]:
                raise ValueError("report projection is not an independently declared output")
            raw = (json.dumps({"file_path": str(PurePosixPath(report_output.logical_root) / path),
                              "cwe_id": ["cwe-93"]}, sort_keys=True) + "\n").encode()
            edits.append({"path": report_output.path, "effect": "create", "before_sha256": None,
                "after_sha256": _sha(raw), "after_bytes_base64": base64.b64encode(raw).decode()})
        if {(edit["path"], edit["effect"]) for edit in edits} != {
                (output["path"], output["effect"]) for output in spec["outputs"]}:
            proof_report["reason_codes"] = ["local_operator_does_not_cover_declared_outputs"]
        else:
            assert_current()
            expected_contract = {**candidate.contract,
                "security_ir_declaration_cid": formalization.report["declaration_cid"],
                "security_ir_formalization_cid": formalization.report["artifact_cid"]}
            proof_report, proof = prove_header_contract(scoped=scoped, contract=expected_contract,
                candidate_cid=candidate_cid, state=state / "proof", lean=lean, z3=z3)
            assert_current()
            if not verify_header_candidate(scoped.sources[path].decode("utf-8"), candidate):
                raise ValueError("candidate changed after proof execution")
            result["candidate_cid"] = candidate_cid
            result["synthesis"] = {"status": "exact_ast_replayed", "operator": candidate.operator_id,
                "edit_count": len(candidate.edits), "path": path, "before_sha256": candidate.before_sha256,
                "after_sha256": candidate.after_sha256, "provider_calls": 0,
                "global_impact_closure": False, "task_validation_required": True}
    result["proof"] = proof_report
    result["reason_codes"] = proof_report["reason_codes"]
    result["contract_index"] = persist_contract_world(scoped=scoped, analyses=analyses,
        proof_report=proof_report, proof=proof, state=state / "index", task_id=task["task_alias"],
        formalization=formalization.report)
    assert_current()
    if proof is not None:
        # Index metadata cannot substitute for fresh, sealed native proof.
        if not proof.mutation_capable:
            raise ValueError("local contract proof lost its sealed binding")
        payload = {"schema": SCHEMA, "repository": str(repository),
            "baseline_commit": verified["manifest"]["baseline_commit"],
            "task_cid": task_cid, "task_id": task["task_alias"], "task_revision": task["revision"],
            "manifest_cid": scoped.report["manifest_cid"], "proof_receipt_id": proof.content_id,
            "proof_scope": PROOF_SCOPE, "analysis_cid": scoped.report["analysis_cid"],
            "edits": edits, "permitted_outputs": spec["outputs"], "provider_calls": 0,
            "publication_authority": False, "completion_authority": False}
        artifact, digest, cid = _publish(repository, payload)
        result.update(status="candidate_ready", route="doctor_contract_candidate", artifact=str(artifact),
                      sha256=digest, artifact_cid=cid)
    (state / "result.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return result
