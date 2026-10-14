"""Historical exact-byte reconstruction; native retention, no provider calls."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_context_audit as audit
from benchmarks.agent_supervisor.container_coding import terminal_candidate_input_audit as candidate
from ipfs_accelerate_py.agent_supervisor.runtime import candidate_dispatch_context as context
from ipfs_accelerate_py.agent_supervisor.runtime import candidate_reply_contract as replies
from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as router
from test.api.test_candidate_dispatch_context import dispatch_case, source_service, native  # noqa: F401


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def case(dispatch_case):
    prepared = dispatch_case.prepared
    observed = context.validate_candidate_dispatch_context(**dispatch_case.request)
    prompt = prepared.prompt.provider_prompt.decode()
    guidance = router._candidate_round_guidance(observed)
    model, advisory = router.render_model_prompt(prompt=prompt+guidance, purpose="coding",
        workspace=prepared.source_broker.workspace, candidate_transport=True)
    model, reply = replies.apply_candidate_reply_contract(model_prompt=model, binding=observed)
    row = dict(schema="router-implementation-invocation@1", phase="coding", purpose="coding",
        invocation_id="authored-candidate-receipt", workspace=str(prepared.source_broker.workspace),
        semantic_translation=None, public_instruction=None, doctor_residual_context=None,
        candidate_round_dispatch=observed, candidate_reply_contract=reply,
        candidate_retrieval_guidance=dict(sha256=sha(guidance.encode()), bytes=len(guidance.encode()),
            requests_observed=False, proof_authority=False, completion_authority=False))
    for name, value in (("prompt", prompt), ("native_prompt", dispatch_case.original.prompt),
                        ("router_prompt", prompt+guidance), ("model_prompt", model), ("workspace_advisory", advisory)):
        row[name+"_sha256"] = sha(value.encode())
        row[name+"_bytes"] = len(value.encode())
    return dispatch_case, row, model


def replay(case):
    prepared, row, _ = case
    receipt = audit._receipt(row, prepared.original.workspace.parent)
    return audit._model_projection(rendered=prepared.original.prompt, receipt=receipt,
        repository=prepared.original.repository, candidate_parent=prepared.prepared.audit.manifest_path.parent)


def test_roundtrip_after_handoff_cleanup_and_postimage_edits(case):
    prepared, row, expected = case
    prepared.prepared.close()
    (prepared.original.workspace / "mod.py").write_text("# source has changed again\n")
    model, _, checks = replay(case)
    assert model == expected and all(checks.values())
    assert row["prompt_sha256"] != row["native_prompt_sha256"]
    assert prepared.prepared.audit.manifest_path.exists()


def test_tampered_retained_bytes_never_verify(case):
    prepared, _, _ = case
    manifest = json.loads(prepared.prepared.audit.manifest_bytes)
    for field in ("summary", "capsules", "formal", "candidate_input", "context", "handoff",
            "descriptor", "native_prompt", "original_requirements", "original_public_instruction", "prompt_observation"):
        path = prepared.prepared.audit.manifest_path.parent / manifest["artifacts"][field]["path"]
        original = path.read_bytes()
        try:
            path.chmod(0o600); path.write_bytes(b"{}"); path.chmod(0o400)
            with pytest.raises(ValueError):
                replay(case)
        finally:
            path.chmod(0o600); path.write_bytes(original); path.chmod(0o400)


def test_missing_or_unprotected_retention_is_not_rebuilt_from_postimage(case, tmp_path):
    prepared, row, _ = case
    path = prepared.prepared.audit.manifest_path
    original = path.read_bytes()
    for kind in ("missing", "symlink", "mode", "manifest", "wrong_root"):
        parent = path.parent
        try:
            if kind == "missing": path.unlink()
            elif kind == "symlink":
                target = tmp_path / "untrusted.json"
                target.write_bytes(original)
                path.unlink(); path.symlink_to(target)
            elif kind == "mode": path.chmod(0o644)
            elif kind == "manifest":
                path.chmod(0o600); path.write_bytes(b"{}"); path.chmod(0o400)
            else: parent = None
            with pytest.raises((ValueError, KeyError, OSError)):
                candidate.replay(rendered=prepared.original.prompt, receipt=row, parent=parent)
        finally:
            path.unlink(missing_ok=True)
            path.write_bytes(original); path.chmod(0o400)


def test_candidate_receipt_cannot_masquerade_as_baseline_or_other_generation(case):
    prepared, original, _ = case
    for change in ("native", "generation", "baseline", "reply", "guidance", "preflight", "partial"):
        row = copy.deepcopy(original)
        if change == "native": row["native_prompt_sha256"] = "f"*64
        elif change == "generation": row["candidate_round_dispatch"]["workspace_generation"] = "f"*64
        elif change == "baseline": row["source_view_retrieval"] = {}
        elif change == "reply": row["candidate_reply_contract"]["round_number"] = True
        elif change == "guidance": row["candidate_retrieval_guidance"]["bytes"] += 1
        elif change == "preflight": row["candidate_round_preflight"] = {}
        else: del row["candidate_round_dispatch"]
        with pytest.raises((ValueError, KeyError)):
            audit._receipt(row, prepared.original.workspace.parent)


def test_mismatching_provider_input_stays_unverified(case):
    case[1]["router_prompt_sha256"] = "f"*64
    _, _, checks = replay(case)
    assert checks["router_prompt_sha256"] is False


def test_corrupt_source_archive_is_not_a_complete_retained_context(case):
    prepared, _, _ = case
    manifest = json.loads(prepared.prepared.audit.manifest_bytes)
    row = next(iter(manifest["candidate_sources"].values()))
    path = prepared.prepared.audit.manifest_path.parent / row["path"]
    path.chmod(0o600); path.write_bytes(b"replaced"); path.chmod(0o400)
    with pytest.raises(ValueError):
        replay(case)


from benchmarks.agent_supervisor.container_coding.test_terminal_context_audit import compiled  # noqa: E402,F401


def test_failed_projection_does_not_hide_an_independent_receipt(compiled, monkeypatch):
    receipt = compiled["receipts"][0]
    first = dict(receipt, invocation_id="failed-projection")
    compiled["receipts"].insert(0, first)
    original = audit._model_projection
    def project(**kwargs):
        if kwargs["receipt"]["invocation_id"] == first["invocation_id"]:
            raise ValueError("authored unavailable retained input")
        return original(**kwargs)
    monkeypatch.setattr(audit, "_model_projection", project)
    result = audit.audit_terminal_context(state=compiled["state"], receipts=compiled["receipts"],
        workspace_root=compiled["workspace_root"])
    assert result["status"] == "verified_some_observed_model_inputs"
    assert result["all_observed_coding_inputs_verified"] is False
    assert result["invocations"][0]["model_input_verified"] is False
    assert result["invocations"][1]["model_input_verified"] is True
    assert result["read_errors"][0]["kind"] == "model_projection"


def test_rehashed_candidate_cannot_replace_the_input_pinned_by_the_handoff(case):
    prepared, original, _ = case
    row = copy.deepcopy(original)
    archive = prepared.prepared.audit
    manifest = json.loads(archive.manifest_bytes)
    payload = json.loads(prepared.prepared.prompt.provider_prompt)
    payload["instructions"] += " altered retained instruction"
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ref = manifest["artifacts"]["candidate_input"]
    path = archive.manifest_path.parent / ref["path"]
    path.chmod(0o600); path.write_bytes(raw); path.chmod(0o400)
    ref.update(sha256=sha(raw), bytes=len(raw))
    manifest["binding"]["candidate_input_sha256"] = sha(raw)
    archive.manifest_path.chmod(0o600)
    archive.manifest_path.write_text(json.dumps(manifest))
    archive.manifest_path.chmod(0o400)
    row["candidate_round_dispatch"].update(provider_input_sha256=sha(raw), provider_input_bytes=len(raw))
    guidance = router._candidate_round_guidance(row["candidate_round_dispatch"])
    model, advisory = router.render_model_prompt(prompt=raw.decode()+guidance, purpose="coding",
        workspace=Path(row["workspace"]), candidate_transport=True)
    model, row["candidate_reply_contract"] = replies.apply_candidate_reply_contract(
        model_prompt=model, binding=row["candidate_round_dispatch"])
    for name, value in (("prompt", raw.decode()), ("router_prompt", raw.decode()+guidance),
                        ("model_prompt", model), ("workspace_advisory", advisory)):
        row[name+"_sha256"] = sha(value.encode())
        row[name+"_bytes"] = len(value.encode())
    # All public hashes agree with each other; the independently retained
    # dispatched handoff still binds the original input and must win.
    audit._receipt(row, prepared.original.workspace.parent)
    with pytest.raises(ValueError):
        candidate.replay(rendered=prepared.original.prompt, receipt=row, parent=archive.manifest_path.parent)
