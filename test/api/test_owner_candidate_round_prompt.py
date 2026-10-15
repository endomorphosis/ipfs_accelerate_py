"""Initial native requirements -> changed candidate -> bounded owner-only prompt.

Real Git roots, signed public instruction admission, native semantic production,
failed child wait, source generation and prompt revalidation run without models.
Container UID/path admission is authored as in the source-broker tests.
"""
from dataclasses import FrozenInstanceError
import hashlib
import json
import os
from pathlib import Path
import tempfile

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import owner_candidate_round_prompt as prompts
from ipfs_accelerate_py.agent_supervisor.runtime import worker_source_broker as brokers
from ipfs_accelerate_py.agent_supervisor.runtime import source_formalization_context as guard
from ipfs_accelerate_py.agent_supervisor.runtime import router_public_instruction as public
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import prepare_semantic_context
from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
    ContextCompiler, build_text_context_references, render_context_capsule)
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextBudget
from test.api.test_owner_candidate_round_context import source_service, native, waited, prepare, CHANGED  # noqa: F401
from test.api.test_router_public_instruction import selected, scenario  # noqa: F401


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _candidate(service):
    waited(service)
    path = service.workspace / ("answer.py" if (service.workspace / "answer.py").exists() else "mod.py")
    path.write_bytes(CHANGED)
    return prepare(service)


@pytest.fixture
def public_service(selected, monkeypatch):
    root, _, source, _, request = selected
    output = root / ".runtime/semantic-original-requirements"
    prepare_semantic_context(repository=root, paths=["answer.py"],
        program_paths=["answer.py"], required_raw_paths=["answer.py"],
        objective="Repair the answer", task_id="LOCAL-TASK", output=output)
    artifact = output / "worker-context.json"
    refs = build_text_context_references(artifact.read_text(), reference_prefix="semantic-context",
        kind="semantic-context", path=artifact.relative_to(root).as_posix(),
        repository_id="repo:test", tree_id="tree:test", required=True, chunk_bytes=1201)
    compiled = ContextCompiler(ContextBudget(max_input_tokens=32768, max_items=128,
        max_item_bytes=16384, max_serialized_bytes=262144)).compile(
        repository_id="repo:test", tree_id="tree:test", objective_id="LOCAL-TASK",
        objective_revision="sha256:task", policy_id="policy:test", policy_revision="sha256:policy",
        caller="supervisor:test", stage="implementation", goal={"id": "LOCAL-TASK"},
        authority={"mode": "candidate_only", "completion_authority": False},
        scope={"allowed_paths": ["answer.py"]},
        acceptance={"criteria": ["The public answer check must pass; validation remains pending."]}, evidence=refs)
    prompt = render_context_capsule(compiled.capsule) + "\nKeep complete literal guidance: s0 and $semantic_ref.\n"
    with tempfile.TemporaryDirectory(prefix="req-") as folder:
        boundary = Path(folder) / "boundary.json"
        boundary.write_text("authored boundary")
        (boundary.parent / "worker-observations").mkdir(mode=0o750)
        monkeypatch.setattr(guard, "_owner_boundary", lambda path: (
            dict(owner_uid=os.getuid(), worker_uid=os.getuid() + 1), "b" * 64))
        monkeypatch.setattr(guard, "_protected_directory", lambda *a, **kw: None)
        service = brokers.WorkerSourceBroker(repository=root, workspace=request["workspace"],
            boundary=boundary, prompt=prompt, timeout_seconds=90)
        try:
            yield service, dict(artifact=request["artifact"], expected_sha256=request["expected_sha256"],
                task_cid=request["task_cid"]), source
        finally:
            service.close()


def test_native_only_capture_and_prompt_preserve_exact_task_suffix_and_candidate_bindings(source_service):
    original = prompts.capture_native_only_requirements(source_broker=source_service)
    snapshot = _candidate(source_service)
    prepared = prompts.prepare_candidate_round_prompt(snapshot=snapshot, requirements=original)
    wire = json.loads(prepared.provider_prompt)
    metadata = prepared.revalidate()
    assert wire["native_prompt"].encode() == snapshot.native_prompt == original.native_prompt
    assert wire["original_public_instruction_block"] == ""
    assert wire["original_requirements_binding"]["selection"] == "native-only"
    assert wire["baseline_binding"] == snapshot.metadata["baseline_binding"]
    assert wire["candidate_binding"]["semantic_root_cid"] == snapshot.metadata["semantic_root_cid"]
    assert wire["candidate_binding"]["source_hashes"]["mod.py"] == sha(CHANGED)
    assert wire["candidate_binding"]["semantic_root_cid"] != wire["baseline_binding"]["semantic_root_cid"]
    assert wire["candidate_summary"]["pending_obligation_count"] > 0
    formal = json.loads(snapshot.artifacts["formal"])
    assert wire["candidate_summary"]["logic_profile"] == formal["profile"]
    assert wire["candidate_summary"]["unsupported_frontiers"] == formal["frontiers"]
    assert wire["candidate_artifact_references"] == snapshot.metadata["artifacts"]
    assert metadata["provider_prompt_sha256"] == sha(prepared.provider_prompt)
    assert metadata["provider_prompt_bytes"] == len(prepared.provider_prompt)
    assert metadata["required_text_truncated"] is prepared.dispatch_usable is False
    assert all(metadata[key] == expected and wire[key] == expected for key, expected in prompts.FLAGS.items())
    assert CHANGED not in prepared.provider_prompt and snapshot.native_prompt not in prepared.metadata_bytes
    with pytest.raises(FrozenInstanceError):
        prepared.provider_prompt = b"changed"
    metadata["proof_authority"] = True
    assert prepared.metadata["proof_authority"] is False


def test_real_signed_public_requirement_block_is_retained_after_implementation_edits(public_service):
    service, selected_args, source = public_service
    expected, receipt = public.load_public_instruction(**selected_args, prompt=service.prompt,
        repository=service.repository, workspace=service.workspace)
    original = prompts.capture_public_requirements(source_broker=service, **selected_args)
    assert original.public_instruction_block == expected.encode()
    assert original.metadata["public_instruction_receipt"] == receipt
    assert source.decode() in expected and source.decode() not in service.prompt
    service.original_requirements = original
    snapshot = _candidate(service)
    with pytest.raises(ValueError, match="stale"):
        public.load_public_instruction(**selected_args, prompt=service.prompt,
            repository=service.repository, workspace=service.workspace)
    prepared = prompts.prepare_candidate_round_prompt(snapshot=snapshot)
    wire = json.loads(prepared.provider_prompt)
    assert wire["native_prompt"] == service.prompt
    assert wire["original_public_instruction_block"] == expected
    assert wire["original_requirements_binding"]["selection"] == "public-instruction"
    assert wire["original_requirements_binding"]["public_instruction_block_sha256"] == sha(expected.encode())
    metadata = prepared.revalidate()
    assert metadata["original_public_block_preserved"] is metadata["native_prompt_preserved"] is True
    assert metadata["dispatch_usable"] is metadata["baseline_evidence_applicable"] is False
    assert source not in prepared.metadata_bytes
    assert original.metadata["public_instruction_receipt"]["source_freshness_verified"] is True
    assert original.metadata["source_current_at_candidate"] is False
    assert original.metadata["public_instruction_receipt"]["historical_replay"] is False


@pytest.mark.parametrize("mutation", ["source", "artifact", "head", "native_prompt", "candidate"])
def test_prepared_public_prompt_refuses_canonical_requirement_or_candidate_drift(public_service, mutation):
    import subprocess
    service, args, _ = public_service
    original = prompts.capture_public_requirements(source_broker=service, **args)
    snapshot = _candidate(service)
    prepared = prompts.prepare_candidate_round_prompt(snapshot=snapshot, requirements=original)
    if mutation == "source":
        (service.repository / "test_answer.py").write_text("changed public requirement")
    elif mutation == "artifact":
        args["artifact"].chmod(0o600); args["artifact"].write_bytes(args["artifact"].read_bytes() + b" ")
        args["artifact"].chmod(0o444)
    elif mutation == "head":
        subprocess.run(["git", "-C", str(service.repository), "-c", "user.name=Fixture", "-c",
            "user.email=fixture@example.invalid", "commit", "--allow-empty", "-qm", "changed head"], check=True)
    elif mutation == "native_prompt":
        service.prompt += "\nchanged native constraints\n"
    else:
        path = service.workspace / "answer.py"; path.write_bytes(path.read_bytes() + b"\n# changed generation\n")
    with pytest.raises(ValueError):
        prepared.revalidate()


@pytest.mark.parametrize("mode", ["native", "public"])
def test_capture_after_source_edits_refused(public_service, mode):
    service, args, _ = public_service
    (service.workspace / "answer.py").write_bytes(CHANGED)
    with pytest.raises(ValueError):
        (prompts.capture_native_only_requirements(source_broker=service) if mode == "native"
            else prompts.capture_public_requirements(source_broker=service, **args))


def test_capture_after_child_registration_refused(source_service):
    waited(source_service)
    with pytest.raises(ValueError, match="before child registration"):
        prompts.capture_native_only_requirements(source_broker=source_service)


@pytest.mark.parametrize("bad", [None, {}, {"selection": "native-only"}, "serialized-original-requirements"])
def test_missing_or_untyped_capture_cannot_masquerade_as_verified_original(source_service, bad):
    snapshot = _candidate(source_service)
    with pytest.raises(ValueError, match="typed owner original"):
        prompts.prepare_candidate_round_prompt(snapshot=snapshot, requirements=bad)


@pytest.mark.parametrize("budget", [True, 0, 1.5, prompts.MAX_PROMPT_BYTES + 1, None])
def test_invalid_prompt_budget_refuses_before_snapshot_replay(source_service, monkeypatch, budget):
    original = prompts.capture_native_only_requirements(source_broker=source_service)
    snapshot = _candidate(source_service)
    monkeypatch.setattr(prompts.OriginalRequirements, "revalidate", lambda *a, **kw: pytest.fail("invalid budget reached capture"))
    with pytest.raises(ValueError, match="byte budget"):
        prompts.prepare_candidate_round_prompt(snapshot=snapshot, requirements=original, max_prompt_bytes=budget)


def test_required_prompt_bytes_are_never_truncated_to_fit(public_service):
    service, args, _ = public_service
    original = prompts.capture_public_requirements(source_broker=service, **args)
    snapshot = _candidate(service)
    prepared = prompts.prepare_candidate_round_prompt(snapshot=snapshot, requirements=original)
    with pytest.raises(ValueError, match="cannot be truncated"):
        prompts.prepare_candidate_round_prompt(snapshot=snapshot, requirements=original,
            max_prompt_bytes=len(prepared.provider_prompt) - 1)
    assert prompts.prepare_candidate_round_prompt(snapshot=snapshot, requirements=original,
        max_prompt_bytes=len(prepared.provider_prompt)).provider_prompt == prepared.provider_prompt


def test_complete_candidate_summary_over_bound_is_refused_not_summarized_again(source_service, monkeypatch):
    original = prompts.capture_native_only_requirements(source_broker=source_service)
    snapshot = _candidate(source_service)
    monkeypatch.setattr(prompts, "MAX_SUMMARY_BYTES", 1)
    with pytest.raises(ValueError, match="complete candidate summary"):
        prompts.prepare_candidate_round_prompt(snapshot=snapshot, requirements=original)


def test_native_capture_is_bound_to_one_exact_broker_snapshot(source_service, monkeypatch):
    original = prompts.capture_native_only_requirements(source_broker=source_service)
    snapshot = _candidate(source_service)
    object.__setattr__(original, "_source_broker", object())
    with pytest.raises(ValueError, match="another owner snapshot"):
        prompts.prepare_candidate_round_prompt(snapshot=snapshot, requirements=original)


def test_public_capture_uses_native_current_source_loader_and_rejects_bad_nomination(public_service):
    service, args, _ = public_service
    for changes in ({"expected_sha256": "a" * 64}, {"task_cid": "foreign-task"}):
        with pytest.raises(ValueError):
            prompts.capture_public_requirements(source_broker=service, **(args | changes))
    assert service._process is None and not service._registered.is_set()


def test_metadata_copy_cannot_modify_retained_requirements(source_service):
    original = prompts.capture_native_only_requirements(source_broker=source_service)
    original.metadata["task_id"] = "foreign"
    with pytest.raises(FrozenInstanceError):
        original.public_instruction_block = b"unverified replacement"
    snapshot = _candidate(source_service)
    assert prompts.prepare_candidate_round_prompt(snapshot=snapshot, requirements=original).metadata["task_id"] == "TASK-1"
