#!/usr/bin/env python3
"""Materialize compact PGIR-050 proof-attempt recipes into sealed traces."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    ResourceBudget,
    canonical_json,
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_provider import (
    J2_PROOF_ATTEMPT_FIELDS,
    AttemptTraceFailureKind,
    ProofAttemptTraceAdmissionError,
    ProofAttemptTraceStore,
    ProviderRequest,
    admit_proof_attempt_trace,
)
from ipfs_accelerate_py.agent_supervisor.proof.kernel_verification import (
    KernelVerificationBindings,
)
from ipfs_accelerate_py.agent_supervisor.proof.leanstral_proof_provider import (
    LEANSTRAL_DRAFT_SCHEMA_VERSION,
    LeanstralProofDraft,
    LeanstralProofProvider,
    verify_leanstral_draft,
)
from ipfs_accelerate_py.agent_supervisor.proof.proof_context import FixedTheoremIdentity


ROOT = Path(__file__).resolve().parent
RECIPES = ROOT / "recipes"
TRACES = ROOT / "traces"
RECEIPTS = ROOT / "receipts"
NATIVE_SOURCE = "theorem Fixed.identity (h : P) : P := sorry\n#print axioms Fixed.identity\n"


def _load_recipe(name: str) -> dict[str, Any]:
    return json.loads((RECIPES / name).read_text(encoding="utf-8"))


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")


def _prove_request(recipe: dict[str, Any]) -> ProviderRequest:
    return ProviderRequest(
        request_id=str(recipe["request_id"]),
        operation="prove",
        payload={
            "prompt": "Prove the fixed theorem using only premise_1.",
            "obligation_ids": [recipe["obligation_id"]],
            "canonical_source_digest": recipe["canonical_source_digest"],
            "resource_class": "model",
        },
        resource_budget=ResourceBudget(
            wall_time_ms=12_000,
            model_token_limit=384,
            max_output_bytes=64 * 1024,
        ),
    )


def _draft(recipe: dict[str, Any]) -> LeanstralProofDraft:
    proof_text = str(recipe["draft_text"])
    output_sha256 = __import__("hashlib").sha256(proof_text.encode("utf-8")).hexdigest()
    theorem = _theorem()
    identity = {
        "schema_version": LEANSTRAL_DRAFT_SCHEMA_VERSION,
        "llm_provider": "leanstral_local",
        "model": "Leanstral",
        "obligation_ids": [recipe["obligation_id"]],
        "canonical_source_digest": recipe["canonical_source_digest"],
        "theorem_id": recipe.get("theorem_id", theorem.theorem_id),
        "theorem_equivalence_key": theorem.equivalence_key,
        "context_capsule_id": "capsule-1",
        "proposal_kind": "proof",
        "prompt_sha256": "a" * 64,
        "output_sha256": output_sha256,
    }
    artifact_id = (
        "leanstral-draft-"
        + __import__("hashlib")
        .sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        .hexdigest()
    )
    return LeanstralProofDraft(
        artifact_id=artifact_id,
        draft_text=proof_text,
        request_id=str(recipe["request_id"]),
        llm_provider="leanstral_local",
        model="Leanstral",
        obligation_ids=(recipe["obligation_id"],),
        canonical_source_digest=recipe["canonical_source_digest"],
        prompt_sha256="a" * 64,
        output_sha256=output_sha256,
        timeout_ms=1_000,
        token_budget=128,
        theorem_id=str(recipe.get("theorem_id", theorem.theorem_id)),
        theorem_equivalence_key=theorem.equivalence_key,
        context_capsule_id="capsule-1",
        proposal_kind="proof",
    )


def _theorem() -> FixedTheoremIdentity:
    return FixedTheoremIdentity(
        theorem_id="Fixed.identity",
        obligation_id="obligation-1",
        declaration_name="Fixed.identity",
        assumptions=("P",),
        conclusion="P",
        template_id="identity",
        template_version="1",
        source_scope=("Fixed.identity",),
        canonical_source_digest="sha256:canonical",
    )


def _bindings(request_id: str) -> KernelVerificationBindings:
    return KernelVerificationBindings(
        obligation_id="obligation-1",
        request_id=request_id,
        candidate_id="candidate-1",
        kernel_id="kernel:lean@test",
        toolchain_id="toolchain:lean@test",
    )


def build() -> dict[str, Any]:
    identity = _load_recipe("identity_candidate.json")
    timeout = _load_recipe("timeout.json")
    rejected = _load_recipe("kernel_rejected.json")
    failures = _load_recipe("admission_failures.json")

    provider = LeanstralProofProvider(
        llm_generate=lambda *_args, **_kwargs: identity["draft_text"]
    )
    prove_result = provider.prove(_prove_request(identity))
    prove_trace = admit_proof_attempt_trace(prove_result["proof_attempt_trace"])
    _write(TRACES / "identity_candidate.json", prove_trace)

    def boom(*_args, **_kwargs):
        raise TimeoutError("bounded timeout")

    try:
        LeanstralProofProvider(llm_generate=boom).prove(_prove_request(timeout))
    except Exception as exc:
        timeout_trace = admit_proof_attempt_trace(exc.failure.details["proof_attempt_trace"])
    else:
        raise RuntimeError("timeout recipe did not time out")
    _write(TRACES / "timeout.json", timeout_trace)

    gate = verify_leanstral_draft(
        _draft(rejected),
        _theorem(),
        native_source=NATIVE_SOURCE,
        bindings=_bindings(rejected["request_id"]),
        kernel_runner=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("rejected proof must not reach the kernel")
        ),
    )
    gate_trace = admit_proof_attempt_trace(gate.to_dict()["proof_attempt_trace"])
    _write(TRACES / "kernel_rejected.json", gate_trace)

    store = ProofAttemptTraceStore()
    admit_proof_attempt_trace(prove_trace, store=store)
    failure_receipts: list[dict[str, Any]] = []
    for case in failures["cases"]:
        kind = case["kind"]
        try:
            if kind == "malformed":
                payload = {**prove_trace, **case["mutate"]}
                payload.pop("attempt_id", None)
                payload.pop("content_id", None)
                admit_proof_attempt_trace(payload)
            elif kind == "stale":
                admit_proof_attempt_trace(prove_trace, expected=case["expected"])
            elif kind == "wrong_statement":
                admit_proof_attempt_trace(prove_trace, expected=case["expected"])
            elif kind == "replayed":
                admit_proof_attempt_trace(prove_trace, store=store)
            else:
                raise RuntimeError(f"unknown admission case {kind}")
        except ProofAttemptTraceAdmissionError as exc:
            failure_receipts.append(
                {
                    "kind": kind,
                    "failure_kind": exc.kind.value,
                    "message": exc.message,
                    "authoritative": False,
                    "candidate_authority": False,
                }
            )
        else:
            raise RuntimeError(f"admission case {kind} unexpectedly passed")

    receipts = {
        "schema": "ipfs_accelerate_py/agent-supervisor/proof-attempt-non-authority-receipt@1",
        "candidate_authority": False,
        "timeout_is_falsehood": False,
        "j2_fields": list(J2_PROOF_ATTEMPT_FIELDS),
        "admitted_traces": {
            "identity_candidate": prove_trace["attempt_id"],
            "timeout": timeout_trace["attempt_id"],
            "kernel_rejected": gate_trace["attempt_id"],
        },
        "admission_failures": failure_receipts,
        "authoritative": False,
    }
    _write(RECEIPTS / "non_authority.json", receipts)

    manifest = {
        "schema": "ipfs_accelerate_py/agent-supervisor/proof-attempt-trace-manifest@1",
        "task_id": "PGIR-050",
        "j2_fields": list(J2_PROOF_ATTEMPT_FIELDS),
        "candidate_authority": False,
        "traces": {
            "identity_candidate": prove_trace["attempt_id"],
            "timeout": timeout_trace["attempt_id"],
            "kernel_rejected": gate_trace["attempt_id"],
        },
        "receipts": {
            "non_authority": content_identity(receipts),
        },
    }
    manifest["manifest_cid"] = content_identity(
        {key: value for key, value in manifest.items() if key != "manifest_cid"}
    )
    _write(ROOT / "manifest.json", manifest)

    result = {
        "schema": "pgir-task-result@1",
        "task_id": "PGIR-050",
        "result_identity": "RESULT(PGIR-050)",
        "completion_authoritative": False,
        "candidate_authority": False,
        "j2_fields_bound": list(J2_PROOF_ATTEMPT_FIELDS),
        "manifest_cid": manifest["manifest_cid"],
        "admitted_trace_ids": [
            prove_trace["attempt_id"],
            timeout_trace["attempt_id"],
            gate_trace["attempt_id"],
        ],
        "admission_failure_kinds": [item["failure_kind"] for item in failure_receipts],
    }
    result["result_cid"] = content_identity(
        {key: value for key, value in result.items() if key != "result_cid"}
    )
    _write(ROOT / "result.json", result)
    return result


def main() -> int:
    result = build()
    print(canonical_json({"ok": True, "result_cid": result["result_cid"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
