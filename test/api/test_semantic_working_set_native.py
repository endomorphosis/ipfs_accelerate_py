"""Measure authenticated capsule references in the indexed public/Doctor flow.

The signed public fixture performs real indexing, declared-contract planning,
Doctor residual preparation, sealed requirement capture, and a registered router
child. Only host UID admission and provider output are authored. No benchmark
verifier, live model, or private run artifact is used.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from benchmarks.agent_supervisor.container_coding import terminal_context_audit as audit
from benchmarks.agent_supervisor.container_coding.terminal_source_view_observation import _session
from ipfs_accelerate_py.agent_supervisor.runtime import candidate_round_policy as policy
from ipfs_accelerate_py.agent_supervisor.runtime import owner_candidate_dispatch as owner
from ipfs_accelerate_py.agent_supervisor.runtime import worker_source_client as source
from test.api import public_validation_fixture as fixture
from test.api import test_public_validation_continuation as continuation
from test.api.test_doctor_candidate_dispatch import _config, _peer_seam


UNKNOWN_REQUIREMENT = {
    "required": True,
    "description": "Authored owner extension remains literal through capsule projection.",
    "empty_values": [None, False, 0, "", [], {}],
}


PROVIDER_ASSERTIONS = r'''
import base64
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_router_translation as codec
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_working_set as working
from ipfs_accelerate_py.agent_supervisor.runtime import worker_semantic_evidence_client as evidence
encoded=codec.encode_semantic_router_prompt(prompt=prompt,repository=Path(config['repository']))
loaded=source.load_descriptor(descriptor,digest,workspace=Path.cwd())
view=working.project_semantic_working_set(encoded,loaded)
original_wire=json.loads(encoded.provider_prompt)
projected_wire=json.loads(view.provider_prompt)
measurements=[]
expanded=[]
def resolve(reference):
    cursor=None; chunks=[]
    for _ in range(64):
        response=evidence.request_page(descriptor,digest,workspace=Path.cwd(),
            capsule_index=reference['capsule_index'],cursor=cursor,max_bytes=4096,
            expected_capsule_sha256=reference['capsule_sha256'])
        assert response['ok'],response
        page=response['page']
        assert page['source_obligations_discharged']==0
        assert all(page[key] is False for key in evidence.AUTHORITY_FLAGS)
        chunks.append(page['content'].encode() if page['encoding']=='utf-8'
            else base64.b64decode(page['content'],validate=True))
        cursor=page['next_cursor']
        if cursor is None:break
    else:raise AssertionError('native capsule pagination did not terminate')
    raw=b''.join(chunks)
    assert hashlib.sha256(raw).hexdigest()==reference['capsule_sha256']
    expanded.append(len(raw))
    return raw
native_generate=generate
def generate(model_prompt,**kwargs):
    assert not second
    candidate_selected=view.provider_prompt in model_prompt
    if candidate_selected:
        assert model_prompt.count(view.provider_prompt)==1
        legacy=model_prompt.replace(view.provider_prompt,encoded.provider_prompt,1)
        candidate=model_prompt
    else:
        assert model_prompt.count(encoded.provider_prompt)==1
        legacy=model_prompt
        candidate=model_prompt.replace(encoded.provider_prompt,view.provider_prompt,1)
    selected=working.select_working_set(view,legacy,candidate)
    assert selected.provider_prompt==model_prompt
    assert selected.receipt['selected'] is (len(candidate.encode())<len(legacy.encode()))
    assert projected_wire['native_context']==original_wire['native_context']
    assert projected_wire['native_suffix']==original_wire['native_suffix']
    assert projected_wire['native_context']['qualification_required_extension']==config['unknown_requirement']
    for key in original_wire['translated_semantic']:
        if key!='capsules':
            assert projected_wire['translated_semantic'][key]==original_wire['translated_semantic'][key]
    for before,after in zip(original_wire['translated_semantic']['capsules'],
            projected_wire['translated_semantic']['capsules']):
        assert after=={key:value for key,value in before.items() if key not in working.DEFERRED_FIELDS}
    assert config['public_instruction_block'] in model_prompt
    assert config['doctor_advisory'] in model_prompt
    assert working.restore_semantic_working_set(provider_prompt=view.provider_prompt,
        receipt=view.receipt,encoded=encoded,descriptor=loaded,resolver=resolve)==encoded.provider_prompt
    assert codec.restore_semantic_router_prompt(provider_prompt=encoded.provider_prompt,
        table=encoded.table,repository=Path(config['repository']))==prompt
    result=native_generate(model_prompt,**kwargs)
    reference=projected_wire['working_set']['capsule_references'][0]
    stale=evidence.request_page(descriptor,digest,workspace=Path.cwd(),
        capsule_index=reference['capsule_index'],max_bytes=4096)
    assert stale['ok'] is False and stale['page'] is None
    assert stale['reason_code']=='stale_context'
    measurements.append(dict(selection=selected.receipt,
        exact_transport_inverse=True,exact_native_inverse=True,
        public_instruction_preserved=True,doctor_advisory_preserved=True,
        unknown_required_extension_preserved=True,baseline_expansion_refused_after_edit=True,
        expanded_capsules=len(expanded),expanded_capsule_bytes=sum(expanded)))
    return result
'''


RECEIPT_ASSERTIONS = r'''
assert len(measurements)==1
measurement=measurements[0]
assert receipt['semantic_working_set']==measurement.pop('selection')
selected=working.validate_working_set_selection_receipt(receipt['semantic_working_set'])
assert selected['selected_complete_bytes']==receipt['model_prompt_bytes']
assert selected['selected_complete_sha256']==receipt['model_prompt_sha256']
assert selected['provider_token_usage'] is None
assert selected['provider_token_usage_verified'] is selected['complete_run_token_savings_claimed'] is False
for key in ('native_complete','candidate_complete','selected_complete'):
    for suffix in ('_bytes','_sha256'):
        assert census['census'][key+suffix]==selected[key+suffix]
assert census['census']['selected_mode']==selected['selected_mode']
assert census['census']['fallback_reason']==selected['fallback_reason']
measurement['census_matches_actual_provider_input']=True
measurement['census_provider_calls']=0
if selected['selected_mode']==working.PROFILE:
    assert receipt['semantic_working_set_preflight']['requests']==1
    assert receipt['semantic_working_set_preflight']['provider_input_bytes']==0
else:
    assert selected['fallback_reason']=='complete_input_not_strictly_smaller'
    assert 'semantic_working_set_preflight' not in receipt
print(json.dumps(dict(schema='authored-native-working-set-measurement@1',**measurement),sort_keys=True))
'''


CENSUS_ASSERTIONS = r'''
saved_discover=intelligence_index.discover_available_providers
saved_select=intelligence_index.select_efficient_route
saved_get=llm_router.get_llm_provider
saved_generate=llm_router.generate_text
def forbidden_provider(*args,**kwargs):
    raise AssertionError('working-set census reached provider discovery or generation')
intelligence_index.discover_available_providers=forbidden_provider
intelligence_index.select_efficient_route=forbidden_provider
llm_router.get_llm_provider=forbidden_provider
llm_router.generate_text=forbidden_provider
try:
    census_output,census=router.run(**dict(kwargs,working_set_census_only=True))
finally:
    intelligence_index.discover_available_providers=saved_discover
    intelligence_index.select_efficient_route=saved_select
    llm_router.get_llm_provider=saved_get
    llm_router.generate_text=saved_generate
assert not calls and not measurements
assert census['schema']=='router-semantic-working-set-census@1'
assert census['router_calls']==0
assert census['raw_content_exported'] is False
'''


def _child():
    child = continuation.PUBLIC_CHILD
    provider_marker = "llm_router.generate_text=generate"
    receipt_marker = "    print(json.dumps(dict(status='first_returned'"
    assert child.count(provider_marker) == child.count(receipt_marker) == 1
    child = child.replace(provider_marker, PROVIDER_ASSERTIONS + "\n" + provider_marker)
    child = child.replace("source_view_profile=config['profile'],",
        "semantic_working_set='capsule-references@1',source_view_profile=config['profile'],")
    run_marker = "try:\n    output,receipt=router.run(**kwargs)"
    assert child.count(run_marker) == 1
    child = child.replace(run_marker, CENSUS_ASSERTIONS + "\n" + run_marker)
    return child.replace(receipt_marker,
        "\n".join("    " + line if line else "" for line in RECEIPT_ASSERTIONS.splitlines())
        + "\n" + receipt_marker)


def test_indexed_first_input_measures_complete_working_set_and_authenticated_expansion(
        tmp_path, monkeypatch, record_property):
    capture = fixture._capture_broker

    def selected_capture(service, extra, timeout_seconds):
        context, end = json.JSONDecoder().raw_decode(service.prompt)
        context["qualification_required_extension"] = UNKNOWN_REQUIREMENT
        prompt = source.canonical_json(context).decode() + service.prompt[end:]
        selected_service = SimpleNamespace(prompt=prompt, repository=service.repository,
            boundary=service.boundary, workspace=service.workspace)
        return capture(selected_service, [*extra, "--semantic-working-set", "capsule-references@1"],
            timeout_seconds)

    monkeypatch.setattr(fixture, "_capture_broker", selected_capture)
    with fixture.public_doctor_service(tmp_path, monkeypatch) as native:
        broker = native.original
        continuation._runner_seams(monkeypatch, broker)
        argv = ["--provider", "grok_cli", "--model", "grok-4.7", "--timeout", "600",
            "--max-output-tokens", "128", "--grok-coding-max-turns", "8",
            "--semantic-repository", str(broker.repository), "--source-view-profile", broker._binding["profile"],
            "--semantic-working-set", "capsule-references@1", *native.extra_argv]
        launch = owner.capture_original_coding_launch(source_broker=broker, argv=argv,
            absolute_deadline=time.monotonic() + 599, doctor_coding_policy=policy.PUBLIC_VALIDATION_POLICY)
        imports = [str(Path(owner.__file__).resolve().parents[3])]
        imports.extend(str(Path(importlib.util.find_spec(name).origin).resolve().parents[1])
            for name in ("ipfs_datasets_py", "ipfs_kit_py"))
        _peer_seam(monkeypatch, broker, source)
        config = _config(broker, launch, imports, 1)
        config.update(profile=broker._binding["profile"], public_artifact=str(native.public_request["artifact"]),
            public_sha256=native.public_request["expected_sha256"], unknown_requirement=UNKNOWN_REQUIREMENT,
            public_instruction_block=broker.original_requirements.public_instruction_block.decode())
        launch.reserve_first_attempt([*owner._PREFIX, *argv, *broker.extra_argv])
        process = subprocess.Popen([sys.executable, "-I", "-c", _child(), json.dumps(config)],
            cwd=broker.workspace, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        broker.note_spawned(process)
        try:
            output, error = process.communicate(broker.prompt.encode(), timeout=launch.remaining_seconds)
            assert process.returncode == 0, error.decode()[-12000:]
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            owner.retire_coding_worker(launch=launch, process=process, source_broker=broker)
        rows = [json.loads(line) for line in output.decode().splitlines()]
        receipt = next(row for row in rows if row.get("schema") == "router-implementation-invocation@1")
        measurement = next(row for row in rows if row.get("schema") == "authored-native-working-set-measurement@1")
        assert receipt["router_calls"] == 1 and receipt["status"] == "provider_returned"
        assert native.planning["provider_calls"] == native.dispatch["provider_calls"] == 0
        selected = receipt["semantic_working_set"]
        retained = audit._receipt(dict(receipt, phase="coding"), broker.workspace.parent)
        model, _, checks = audit._model_projection(rendered=broker.prompt, receipt=retained,
            repository=broker.repository)
        assert all(checks.values()), checks
        assert hashlib.sha256(model.encode()).hexdigest() == receipt["model_prompt_sha256"]
        counts, _, _ = _session(broker.audit_path.read_bytes(), dict(receipt, phase="coding"),
            {"task_alias": broker.descriptor["task_id"]}, expected_semantic_working_set="capsule-references@1")
        assert counts["semantic_evidence_counts"]["delivered_bytes"] >= measurement["expanded_capsule_bytes"]
        record_property("semantic_working_set_native", json.dumps(dict(
            **measurement, selected_mode=selected["selected_mode"],
            native_complete_bytes=selected["native_complete_bytes"],
            candidate_complete_bytes=selected["candidate_complete_bytes"],
            selected_complete_bytes=selected["selected_complete_bytes"],
            representation_bytes_removed=selected["native_complete_bytes"]-selected["selected_complete_bytes"],
            retained_model_input_replayed=True, indexed_preparation_native=True,
            planning_provider_calls=0, doctor_provider_calls=0, captured_provider_calls=1,
            live_provider_calls=0, provider_token_usage_verified=False,
            complete_run_token_savings_claimed=False, raw_content_exported=False), sort_keys=True))
