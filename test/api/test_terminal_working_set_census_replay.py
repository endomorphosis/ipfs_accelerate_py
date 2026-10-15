"""Replay a real isolated census and stopped broker, with authored UID admission."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from copy import deepcopy

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_working_set_census_observation as replay
from benchmarks.agent_supervisor.container_coding import terminal_live_census_provenance as provenance
from benchmarks.agent_supervisor.container_coding.benchmark_provider_profile import GROK_PROFILE, resolve_provider_profile
from benchmarks.agent_supervisor.container_coding import test_terminal_public_acceptance_profile as authored
from ipfs_accelerate_py.agent_supervisor.runtime import owner_candidate_dispatch as owner
from ipfs_accelerate_py.agent_supervisor.runtime import candidate_round_policy as policy
from ipfs_accelerate_py.agent_supervisor.runtime import worker_source_client as source
from test.api import public_validation_fixture as fixture
from test.api import test_public_validation_continuation as continuation
from test.api.test_doctor_candidate_dispatch import _config, _peer_seam


def _child():
    text = continuation.PUBLIC_CHILD
    marker = "try:\n    output,receipt=router.run(**kwargs)"
    assert text.count(marker) == 1
    text = text.split(marker)[0]
    return text + '''
def forbidden(*args,**kwargs):
    raise AssertionError('census reached a provider operation')
intelligence_index.discover_available_providers=forbidden
intelligence_index.select_efficient_route=forbidden
llm_router.get_llm_provider=forbidden
llm_router.generate_text=forbidden
output,receipt=router.run(**dict(kwargs,semantic_working_set='capsule-references@1',working_set_census_only=True,
    working_set_census_schema=config['census_schema']))
assert not calls
print(output)
'''


@pytest.mark.parametrize("expanded_module", [False, True])
@pytest.mark.parametrize("census_schema", ["router-semantic-working-set-census@1", "router-semantic-working-set-census@2"])
def test_owner_replays_exact_census_and_closed_source_accounting(tmp_path, monkeypatch, expanded_module, census_schema):
    if census_schema.endswith("@2"):
        # The public fixture normally supplies only semantic references. Add
        # authored required artifacts through its real compiler before custody
        # capture. This tests nonempty inline objects, not learned inference.
        build = fixture.build_text_context_references
        def required_artifacts(*args, **kwargs):
            references = list(build(*args, **kwargs))
            for kind, prefix, body in (
                    ("intent-world-context", "intent-world", {"schema": "authored-world@1",
                        "advice": 'café 世界 \\ "\n' * 700, "proof_authority": False}),
                    ("code-retrieval-context", "code-retrieval", {"schema": "authored-retrieval@1",
                        "hits": [{"score": .125, "description": "Preserve public eigenpair requirements"}]})):
                text = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                references.extend(build(text, reference_prefix=prefix, kind=kind,
                    path=".runtime/" + prefix + "-authored.json", required=True, chunk_bytes=6144,
                    repository_id=kwargs["repository_id"], tree_id=kwargs["tree_id"]))
            return tuple(references)
        monkeypatch.setattr(fixture, "build_text_context_references", required_artifacts)
    if expanded_module:
        # Additional real numerical helpers, admitted and indexed before capture.
        monkeypatch.setattr(authored, "REFERENCE_SOURCE", authored.REFERENCE_SOURCE + '''
def rayleigh_quotient(matrix, vector):
    return np.vdot(vector, matrix @ vector) / np.vdot(vector, vector)

def residual_norm(matrix, value, vector):
    return np.linalg.norm(matrix @ vector - value * vector)

def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("zero vector")
    return vector / norm
''')
    capture = fixture._capture_broker
    monkeypatch.setattr(fixture, "_capture_broker", lambda service, extra, timeout_seconds:
        capture(service, [*extra, "--semantic-working-set", "capsule-references@1"], timeout_seconds))
    with fixture.public_doctor_service(tmp_path, monkeypatch) as native:
        broker = native.original
        continuation._runner_seams(monkeypatch, broker)
        argv = ["--provider", "grok_cli", "--model", "grok-4.7", "--timeout", "300",
            "--max-output-tokens", "128", "--grok-coding-max-turns", "8",
            "--semantic-repository", str(broker.repository), "--source-view-profile", broker._binding["profile"],
            "--semantic-working-set", "capsule-references@1", *native.extra_argv]
        launch = owner.capture_original_coding_launch(source_broker=broker, argv=argv,
            absolute_deadline=time.monotonic() + 299, doctor_coding_policy=policy.PUBLIC_VALIDATION_POLICY)
        imports = [str(Path(owner.__file__).resolve().parents[3])]
        imports += [str(Path(importlib.util.find_spec(name).origin).resolve().parents[1])
            for name in ("ipfs_datasets_py", "ipfs_kit_py")]
        _peer_seam(monkeypatch, broker, source)
        config = _config(broker, launch, imports, 1)
        config.update(census_schema=census_schema, profile=broker._binding["profile"], public_artifact=str(native.public_request["artifact"]),
            public_sha256=native.public_request["expected_sha256"])
        launch.reserve_first_attempt([*owner._PREFIX, *argv, *broker.extra_argv])
        process = subprocess.Popen([sys.executable, "-I", "-c", _child(), json.dumps(config)],
            cwd=broker.workspace, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        broker.note_spawned(process)
        try:
            stdout, stderr = process.communicate(broker.prompt.encode(), timeout=launch.remaining_seconds)
            assert process.returncode == 0, stderr.decode()[-6000:]
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            owner.retire_coding_worker(launch=launch, process=process, source_broker=broker)
        observed = json.loads(stdout)
        assert observed["schema"] == census_schema
        if census_schema.endswith("@2"):
            assert observed["inline_evidence_census"]["dispatch_enabled"] is False
            assert observed["inline_evidence_census"]["selection"]["view_receipt"]["artifact_count"] == 2
            assert observed["inline_evidence_census"]["selection"]["selected_mode"] == "inline-evidence-objects@1"
            assert observed["semantic_pack_census"]["raw_content_exported"] is False
        assert observed["census"]["selected_mode"] == ("capsule-references@1" if expanded_module else "legacy")
        prompt = tmp_path / "owner-prompt.txt"
        prompt.write_text(broker.prompt)
        options = {"coding_reply_mode": "legacy"}
        for prefix, request in (("doctor_residual", native.doctor_request), ("public_instruction", native.public_request)):
            options.update({prefix + "_artifact": request["artifact"], prefix + "_sha256": request["expected_sha256"],
                prefix + "_task_cid": request["task_cid"]})
        prepared = dict(repository=broker.repository, workspace=broker.workspace, prompt_artifact=prompt,
            runner_kwargs=options, metadata=dict(native_prompt_sha256=hashlib.sha256(broker.prompt.encode()).hexdigest()))
        boundary = dict(manifest_sha256="b" * 64, owner_uid=broker.owner_uid, worker_uid=broker.worker_uid)
        monkeypatch.setattr(provenance, "_worker_boundary", lambda *args: boundary)
        deployment = {"qualification": {"permission_evidence": {"boundary": boundary}}}
        kwargs = dict(prepared=prepared, observed=observed, deployment=deployment,
            provider_selection=resolve_provider_profile(GROK_PROFILE), source_view_profile=broker._binding["profile"],
            source_view_policy=None, source_audit_paths=[broker.audit_path], census_schema=census_schema)
        result = replay.replay_working_set_census(**kwargs)
        assert result["complete_input_replayed"] and result["closed_observation_replayed"]
        assert result["source_session_requests"] == int(expanded_module)
        assert result["provider_input_bytes"] == result["provider_calls"] == 0
        assert result["source_session_stopped"] and result["source_session_accounting_complete"]
        assert result["provider_tokens"] is None
        if census_schema.endswith("@2"):
            for section in ("inline_evidence_census", "semantic_pack_census"):
                changed = deepcopy(observed)
                changed[section]["private_extra"] = "not-closed"
                with pytest.raises(ValueError):
                    replay.replay_working_set_census(**(kwargs | {"observed": changed}))
            with pytest.raises(ValueError):
                replay.replay_working_set_census(**(kwargs | {"census_schema": "router-semantic-working-set-census@1"}))
        for path, value in [(("census", "native_complete_bytes"), 1),
                (("census", "selected_complete_bytes"), 1), (("provider_calls",), 1),
                (("native_prompt_sha256",), "a" * 64), (("private_extra",), "private-owner-content")]:
            changed = deepcopy(observed)
            target = changed
            for field in path[:-1]:
                target = target[field]
            target[path[-1]] = value
            with pytest.raises(ValueError):
                replay.replay_working_set_census(**(kwargs | {"observed": changed}))
        with pytest.raises(ValueError):
            replay.replay_working_set_census(**(kwargs | {"source_audit_paths": []}))
        source_path = broker.repository / "eigen.py"
        source_path.write_bytes(source_path.read_bytes() + b"\n# stale owner source\n")
        with pytest.raises(ValueError):
            replay.replay_working_set_census(**kwargs)
